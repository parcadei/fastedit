"""Production benchmark: deterministic text_match accuracy by edit pattern.

Generates 100+ test cases from real code patterns, categorized by the
type of edit. Use this to measure improvements to text_match.py.

Edit patterns (what Claude/agents actually send via MCP):
  P1:  add_guard          — early return before existing code
  P2:  add_line           — insert line between context anchors
  P3:  change_line        — replace one line, keep surrounding context
  P4:  wrap_block         — try/except or if/else around existing (uses marker)
  P5:  add_at_end         — append code after last line + marker
  P6:  replace_block      — drop lines between anchors, insert new
  P7:  change_signature   — modify function def line, keep body via marker
  P8:  multi_insert       — insert multiple lines between anchors
  P9:  class_method       — modify one method in a class (markers for other methods)
  P10: two_markers        — change middle section, preserve top and bottom via markers
  P11: delete_lines       — remove lines between context anchors
  P12: full_replace       — replace entire function body (only sig preserved)
  P13: add_branch         — add elif/else/case to existing conditional
  P14: extend_literal     — add items to list/dict/set literal
  P15: add_decorator      — prepend decorator above function
  P16: rename_variable    — rename variable throughout function
  P17: reorder_statements — swap order of statements
  P18: extract_variable   — extract expression into named variable
  P19: inline_variable    — inline variable value, remove assignment
  P20: change_expression  — modify part of a line (arg, operator, string)
  P21: add_parameter      — add param to signature AND update body
  P22: remove_parameter   — remove param from signature AND its usage

Run with: pytest tests/test_deterministic_benchmark.py -v -s
"""

from __future__ import annotations

import pytest
from fastedit.inference.text_match import deterministic_edit


def _n(code: str) -> str:
    return "\n".join(line.rstrip() for line in code.splitlines()).strip()


# ── Helpers to generate test cases from templates ──

def _case(name, pattern, original, snippet, expected):
    return (name, pattern, original, snippet, expected)


# ════════════════════════════════════════════════════════════════
# Pattern 1: Add guard clause (early return before existing code)
# ════════════════════════════════════════════════════════════════

P1_CASES = [
    _case("p1_empty_list_guard", "add_guard",
        """\
def average(nums):
    total = sum(nums)
    return total / len(nums)""",
        """\
def average(nums):
    if not nums:
        return 0.0
    total = sum(nums)
    # ... existing code ...""",
        """\
def average(nums):
    if not nums:
        return 0.0
    total = sum(nums)
    return total / len(nums)"""),

    _case("p1_none_check", "add_guard",
        """\
def get_name(user):
    return user.first_name + " " + user.last_name""",
        """\
def get_name(user):
    if user is None:
        return "Anonymous"
    return user.first_name + " " + user.last_name""",
        """\
def get_name(user):
    if user is None:
        return "Anonymous"
    return user.first_name + " " + user.last_name"""),

    _case("p1_auth_guard", "add_guard",
        """\
def delete_item(item_id, user):
    item = db.get(item_id)
    db.delete(item)
    return {"deleted": item_id}""",
        """\
def delete_item(item_id, user):
    if not user.is_admin:
        raise PermissionError("Admin required")
    item = db.get(item_id)
    # ... existing code ...""",
        """\
def delete_item(item_id, user):
    if not user.is_admin:
        raise PermissionError("Admin required")
    item = db.get(item_id)
    db.delete(item)
    return {"deleted": item_id}"""),

    _case("p1_bounds_check", "add_guard",
        """\
def get_element(arr, index):
    return arr[index]""",
        """\
def get_element(arr, index):
    if index < 0 or index >= len(arr):
        raise IndexError(f"Index {index} out of range")
    return arr[index]""",
        """\
def get_element(arr, index):
    if index < 0 or index >= len(arr):
        raise IndexError(f"Index {index} out of range")
    return arr[index]"""),

    _case("p1_js_guard", "add_guard",
        """\
function divide(a, b) {
    return a / b;
}""",
        """\
function divide(a, b) {
    if (b === 0) {
        throw new Error("Division by zero");
    }
    return a / b;
}""",
        """\
function divide(a, b) {
    if (b === 0) {
        throw new Error("Division by zero");
    }
    return a / b;
}"""),
]

# ════════════════════════════════════════════════════════════════
# Pattern 2: Add a single line between context anchors
# ════════════════════════════════════════════════════════════════

P2_CASES = [
    _case("p2_add_logging", "add_line",
        """\
def save_record(record):
    validated = validate(record)
    db.insert(validated)
    return validated.id""",
        """\
def save_record(record):
    logger.info(f"Saving record: {record}")
    validated = validate(record)
    # ... existing code ...""",
        """\
def save_record(record):
    logger.info(f"Saving record: {record}")
    validated = validate(record)
    db.insert(validated)
    return validated.id"""),

    _case("p2_add_timing", "add_line",
        """\
def process(data):
    result = transform(data)
    output = format_result(result)
    return output""",
        """\
def process(data):
    start = time.time()
    result = transform(data)
    # ... existing code ...
    logger.debug(f"Took {time.time() - start:.3f}s")
    return output""",
        """\
def process(data):
    start = time.time()
    result = transform(data)
    output = format_result(result)
    logger.debug(f"Took {time.time() - start:.3f}s")
    return output"""),

    _case("p2_add_counter", "add_line",
        """\
def batch_process(items):
    results = []
    for item in items:
        results.append(handle(item))
    return results""",
        """\
def batch_process(items):
    results = []
    processed = 0
    for item in items:
        results.append(handle(item))
        processed += 1
    return results""",
        """\
def batch_process(items):
    results = []
    processed = 0
    for item in items:
        results.append(handle(item))
        processed += 1
    return results"""),

    _case("p2_js_add_log", "add_line",
        """\
function handleClick(event) {
    const target = event.target;
    const value = target.value;
    updateState(value);
}""",
        """\
function handleClick(event) {
    event.preventDefault();
    const target = event.target;
    const value = target.value;
    updateState(value);
}""",
        """\
function handleClick(event) {
    event.preventDefault();
    const target = event.target;
    const value = target.value;
    updateState(value);
}"""),

    _case("p2_add_cache_check", "add_line",
        """\
def get_user(user_id):
    user = db.users.find_one({"_id": user_id})
    if user is None:
        raise NotFound(user_id)
    return User(**user)""",
        """\
def get_user(user_id):
    cached = cache.get(f"user:{user_id}")
    if cached:
        return cached
    user = db.users.find_one({"_id": user_id})
    # ... existing code ...""",
        """\
def get_user(user_id):
    cached = cache.get(f"user:{user_id}")
    if cached:
        return cached
    user = db.users.find_one({"_id": user_id})
    if user is None:
        raise NotFound(user_id)
    return User(**user)"""),
]

# ════════════════════════════════════════════════════════════════
# Pattern 3: Change a single line (keep surrounding context)
# ════════════════════════════════════════════════════════════════

P3_CASES = [
    _case("p3_change_threshold", "change_line",
        """\
def classify(score):
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    else:
        return "C" """,
        """\
def classify(score):
    if score >= 85:
        return "A"
    elif score >= 70:
        return "B"
    else:
        return "C" """,
        """\
def classify(score):
    if score >= 85:
        return "A"
    elif score >= 70:
        return "B"
    else:
        return "C" """),

    _case("p3_change_default", "change_line",
        """\
def create_config(name, debug=False):
    return {"name": name, "debug": debug, "version": "1.0"}""",
        """\
def create_config(name, debug=False):
    return {"name": name, "debug": debug, "version": "2.0"}""",
        """\
def create_config(name, debug=False):
    return {"name": name, "debug": debug, "version": "2.0"}"""),

    _case("p3_change_query", "change_line",
        """\
def find_active_users():
    query = "SELECT * FROM users WHERE active = 1"
    results = db.execute(query)
    return results.fetchall()""",
        """\
def find_active_users():
    query = "SELECT * FROM users WHERE active = 1 AND verified = 1"
    results = db.execute(query)
    return results.fetchall()""",
        """\
def find_active_users():
    query = "SELECT * FROM users WHERE active = 1 AND verified = 1"
    results = db.execute(query)
    return results.fetchall()"""),

    _case("p3_change_timeout", "change_line",
        """\
def make_request(url):
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    return response.json()""",
        """\
def make_request(url):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()""",
        """\
def make_request(url):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()"""),

    _case("p3_js_change_selector", "change_line",
        """\
function getElement() {
    const el = document.querySelector('.old-class');
    el.classList.add('active');
    return el;
}""",
        """\
function getElement() {
    const el = document.querySelector('#main-content');
    el.classList.add('active');
    return el;
}""",
        """\
function getElement() {
    const el = document.querySelector('#main-content');
    el.classList.add('active');
    return el;
}"""),
]

# ════════════════════════════════════════════════════════════════
# Pattern 4: Wrap block with try/except or if/else (uses marker)
# ════════════════════════════════════════════════════════════════

P4_CASES = [
    _case("p4_try_except_basic", "wrap_block",
        """\
def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data""",
        """\
def load_json(path):
    try:
        # ... existing code ...
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load {path}: {e}")
        return {}""",
        """\
def load_json(path):
    try:
        with open(path) as f:
            data = json.load(f)
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load {path}: {e}")
        return {}"""),

    _case("p4_try_except_network", "wrap_block",
        """\
def send_webhook(url, payload):
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.status_code""",
        """\
def send_webhook(url, payload):
    try:
        # ... existing code ...
    except requests.RequestException:
        return None""",
        """\
def send_webhook(url, payload):
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.status_code
    except requests.RequestException:
        return None"""),

    _case("p4_with_lock", "wrap_block",
        """\
def increment_counter(name):
    current = counters.get(name, 0)
    counters[name] = current + 1
    return counters[name]""",
        """\
def increment_counter(name):
    with lock:
        # ... existing code ...""",
        """\
def increment_counter(name):
    with lock:
        current = counters.get(name, 0)
        counters[name] = current + 1
        return counters[name]"""),

    _case("p4_conditional_wrap", "wrap_block",
        """\
def notify(user, message):
    email = compose_email(user.email, message)
    send_email(email)
    log_notification(user.id, message)""",
        """\
def notify(user, message):
    if user.notifications_enabled:
        # ... existing code ...""",
        """\
def notify(user, message):
    if user.notifications_enabled:
        email = compose_email(user.email, message)
        send_email(email)
        log_notification(user.id, message)"""),
]

# ════════════════════════════════════════════════════════════════
# Pattern 5: Add code at end (after last line + marker)
# ════════════════════════════════════════════════════════════════

P5_CASES = [
    _case("p5_add_return_log", "add_at_end",
        """\
def create_user(name, email):
    user = User(name=name, email=email)
    db.add(user)
    db.commit()
    return user""",
        """\
def create_user(name, email):
    # ... existing code ...
    logger.info(f"Created user {user.id}: {name}")
    return user""",
        """\
def create_user(name, email):
    user = User(name=name, email=email)
    db.add(user)
    db.commit()
    logger.info(f"Created user {user.id}: {name}")
    return user"""),

    _case("p5_add_cleanup", "add_at_end",
        """\
def process_upload(file):
    path = save_temp(file)
    result = analyze(path)
    return result""",
        """\
def process_upload(file):
    # ... existing code ...
    os.remove(path)
    return result""",
        """\
def process_upload(file):
    path = save_temp(file)
    result = analyze(path)
    os.remove(path)
    return result"""),

    _case("p5_add_metrics", "add_at_end",
        """\
def handle_request(request):
    data = parse_body(request)
    response = process(data)
    return response""",
        """\
def handle_request(request):
    # ... existing code ...
    metrics.increment("requests_processed")
    return response""",
        """\
def handle_request(request):
    data = parse_body(request)
    response = process(data)
    metrics.increment("requests_processed")
    return response"""),
]

# ════════════════════════════════════════════════════════════════
# Pattern 6: Replace a block (drop lines between anchors)
# ════════════════════════════════════════════════════════════════

P6_CASES = [
    _case("p6_replace_loop", "replace_block",
        """\
def sum_positive(numbers):
    total = 0
    for n in numbers:
        if n > 0:
            total += n
    return total""",
        """\
def sum_positive(numbers):
    total = sum(n for n in numbers if n > 0)
    return total""",
        """\
def sum_positive(numbers):
    total = sum(n for n in numbers if n > 0)
    return total"""),

    _case("p6_replace_formatting", "replace_block",
        """\
def format_date(dt):
    year = dt.year
    month = str(dt.month).zfill(2)
    day = str(dt.day).zfill(2)
    return f"{year}-{month}-{day}" """,
        """\
def format_date(dt):
    return dt.strftime("%Y-%m-%d")""",
        """\
def format_date(dt):
    return dt.strftime("%Y-%m-%d")"""),

    _case("p6_replace_search", "replace_block",
        """\
def find_item(items, target):
    for item in items:
        if item.name == target:
            return item
    return None""",
        """\
def find_item(items, target):
    return next((item for item in items if item.name == target), None)""",
        """\
def find_item(items, target):
    return next((item for item in items if item.name == target), None)"""),

    _case("p6_replace_validation", "replace_block",
        """\
def validate_email(email):
    if "@" not in email:
        return False
    if "." not in email:
        return False
    return True""",
        """\
def validate_email(email):
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))""",
        """\
def validate_email(email):
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))"""),
]

# ════════════════════════════════════════════════════════════════
# Pattern 7: Change function signature, keep body via marker
# ════════════════════════════════════════════════════════════════

P7_CASES = [
    _case("p7_add_param", "change_signature",
        """\
def connect(host, port):
    socket = create_socket(host, port)
    socket.settimeout(10)
    return socket""",
        """\
def connect(host, port, timeout=10):
    socket = create_socket(host, port)
    socket.settimeout(timeout)
    return socket""",
        """\
def connect(host, port, timeout=10):
    socket = create_socket(host, port)
    socket.settimeout(timeout)
    return socket"""),

    _case("p7_add_return_type", "change_signature",
        """\
def parse_int(value):
    return int(value.strip())""",
        """\
def parse_int(value: str) -> int:
    return int(value.strip())""",
        """\
def parse_int(value: str) -> int:
    return int(value.strip())"""),

    _case("p7_make_async", "change_signature",
        """\
def fetch_all(urls):
    results = []
    for url in urls:
        results.append(fetch(url))
    return results""",
        """\
async def fetch_all(urls):
    results = []
    for url in urls:
        results.append(await fetch(url))
    return results""",
        """\
async def fetch_all(urls):
    results = []
    for url in urls:
        results.append(await fetch(url))
    return results"""),

    _case("p7_rename_param", "change_signature",
        """\
def send_message(msg, recipient):
    formatted = format_message(msg)
    deliver(formatted, recipient)
    return True""",
        """\
def send_message(content, recipient):
    formatted = format_message(content)
    deliver(formatted, recipient)
    return True""",
        """\
def send_message(content, recipient):
    formatted = format_message(content)
    deliver(formatted, recipient)
    return True"""),
]

# ════════════════════════════════════════════════════════════════
# Pattern 8: Multi-line insertion between anchors
# ════════════════════════════════════════════════════════════════

P8_CASES = [
    _case("p8_add_middleware", "multi_insert",
        """\
def create_app():
    app = Flask(__name__)
    app.config.from_env()
    register_routes(app)
    return app""",
        """\
def create_app():
    app = Flask(__name__)
    app.config.from_env()
    CORS(app)
    limiter = Limiter(app, default_limits=["100/hour"])
    compress = Compress(app)
    register_routes(app)
    return app""",
        """\
def create_app():
    app = Flask(__name__)
    app.config.from_env()
    CORS(app)
    limiter = Limiter(app, default_limits=["100/hour"])
    compress = Compress(app)
    register_routes(app)
    return app"""),

    _case("p8_add_fields", "multi_insert",
        """\
def build_response(data):
    response = {
        "status": "ok",
        "data": data,
    }
    return response""",
        """\
def build_response(data):
    response = {
        "status": "ok",
        "data": data,
        "timestamp": time.time(),
        "version": "2.0",
        "request_id": generate_id(),
    }
    return response""",
        """\
def build_response(data):
    response = {
        "status": "ok",
        "data": data,
        "timestamp": time.time(),
        "version": "2.0",
        "request_id": generate_id(),
    }
    return response"""),

    _case("p8_add_steps", "multi_insert",
        """\
def deploy(config):
    validate_config(config)
    build_artifact(config)
    upload(config)""",
        """\
def deploy(config):
    validate_config(config)
    run_tests(config)
    lint_check(config)
    build_artifact(config)
    upload(config)""",
        """\
def deploy(config):
    validate_config(config)
    run_tests(config)
    lint_check(config)
    build_artifact(config)
    upload(config)"""),
]

# ════════════════════════════════════════════════════════════════
# Pattern 9: Modify one method in a class (markers for other methods)
# ════════════════════════════════════════════════════════════════

P9_CASES = [
    _case("p9_modify_init", "class_method",
        """\
class Cache:
    def __init__(self, max_size=100):
        self.store = {}
        self.max_size = max_size

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value):
        self.store[key] = value""",
        """\
class Cache:
    def __init__(self, max_size=100, ttl=300):
        self.store = {}
        self.max_size = max_size
        self.ttl = ttl
        self.timestamps = {}

    def get(self, key):
        # ... existing code ...

    def set(self, key, value):
        # ... existing code ...""",
        """\
class Cache:
    def __init__(self, max_size=100, ttl=300):
        self.store = {}
        self.max_size = max_size
        self.ttl = ttl
        self.timestamps = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value):
        self.store[key] = value"""),

    _case("p9_modify_middle_method", "class_method",
        """\
class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[-1]""",
        """\
class Stack:
    # ... existing code ...

    def pop(self):
        if not self.items:
            raise IndexError("Stack is empty")
        return self.items.pop()

    def peek(self):
        # ... existing code ...""",
        """\
class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.items:
            raise IndexError("Stack is empty")
        return self.items.pop()

    def peek(self):
        return self.items[-1]"""),
]

# ════════════════════════════════════════════════════════════════
# Pattern 10: Two markers — change middle, preserve top and bottom
# ════════════════════════════════════════════════════════════════

P10_CASES = [
    _case("p10_change_middle", "two_markers",
        """\
def etl_pipeline(source):
    raw = extract(source)
    cleaned = clean(raw)
    transformed = apply_rules(cleaned)
    enriched = enrich(transformed)
    loaded = load(enriched)
    return loaded""",
        """\
def etl_pipeline(source):
    # ... existing code ...
    transformed = apply_rules_v2(cleaned)
    validated = validate(transformed)
    enriched = enrich(validated)
    # ... existing code ...""",
        """\
def etl_pipeline(source):
    raw = extract(source)
    cleaned = clean(raw)
    transformed = apply_rules_v2(cleaned)
    validated = validate(transformed)
    enriched = enrich(validated)
    loaded = load(enriched)
    return loaded"""),

    _case("p10_change_processing", "two_markers",
        """\
def render_page(request):
    user = authenticate(request)
    data = fetch_data(user)
    filtered = filter_data(data, user.permissions)
    html = template.render(data=filtered)
    response = Response(html)
    add_headers(response)
    return response""",
        """\
def render_page(request):
    # ... existing code ...
    filtered = filter_data(data, user.permissions)
    cached_html = cache.get(f"page:{user.id}")
    if cached_html:
        html = cached_html
    else:
        html = template.render(data=filtered)
        cache.set(f"page:{user.id}", html, ttl=60)
    response = Response(html)
    # ... existing code ...""",
        """\
def render_page(request):
    user = authenticate(request)
    data = fetch_data(user)
    filtered = filter_data(data, user.permissions)
    cached_html = cache.get(f"page:{user.id}")
    if cached_html:
        html = cached_html
    else:
        html = template.render(data=filtered)
        cache.set(f"page:{user.id}", html, ttl=60)
    response = Response(html)
    add_headers(response)
    return response"""),

    _case("p10_swap_step", "two_markers",
        """\
def build(sources):
    parsed = parse_all(sources)
    checked = typecheck(parsed)
    optimized = optimize(checked)
    emitted = emit_code(optimized)
    linked = link(emitted)
    return linked""",
        """\
def build(sources):
    # ... existing code ...
    optimized = optimize(checked)
    optimized = dead_code_elim(optimized)
    emitted = emit_code(optimized)
    # ... existing code ...""",
        """\
def build(sources):
    parsed = parse_all(sources)
    checked = typecheck(parsed)
    optimized = optimize(checked)
    optimized = dead_code_elim(optimized)
    emitted = emit_code(optimized)
    linked = link(emitted)
    return linked"""),
]

# ════════════════════════════════════════════════════════════════
# Pattern 11: Delete lines — remove lines between context anchors
# ════════════════════════════════════════════════════════════════

P11_CASES = [
    _case("p11_remove_debug", "delete_lines",
        """\
def process(data):
    logger.debug("entering process")
    logger.debug(f"data={data}")
    result = transform(data)
    logger.debug(f"result={result}")
    return result""",
        """\
def process(data):
    result = transform(data)
    return result""",
        """\
def process(data):
    result = transform(data)
    return result"""),

    _case("p11_remove_deprecated", "delete_lines",
        """\
def send(payload):
    validate(payload)
    old_format = convert_legacy(payload)
    log_legacy(old_format)
    result = transmit(payload)
    return result""",
        """\
def send(payload):
    validate(payload)
    result = transmit(payload)
    return result""",
        """\
def send(payload):
    validate(payload)
    result = transmit(payload)
    return result"""),

    _case("p11_remove_comments", "delete_lines",
        """\
def calculate(x, y):
    # TODO: optimize this
    # See ticket JIRA-1234
    total = x + y
    return total""",
        """\
def calculate(x, y):
    total = x + y
    return total""",
        """\
def calculate(x, y):
    total = x + y
    return total"""),
]

# ════════════════════════════════════════════════════════════════
# Pattern 12: Full body replace — replace entire function body
# ════════════════════════════════════════════════════════════════

P12_CASES = [
    _case("p12_rewrite_short", "full_replace",
        """\
def greet(name):
    msg = "Hello " + name
    print(msg)
    return msg""",
        """\
def greet(name):
    return f"Hello {name}"  """,
        """\
def greet(name):
    return f"Hello {name}"  """),

    _case("p12_rewrite_implementation", "full_replace",
        """\
def find_max(items):
    best = items[0]
    for item in items[1:]:
        if item > best:
            best = item
    return best""",
        """\
def find_max(items):
    if not items:
        raise ValueError("empty list")
    return max(items)""",
        """\
def find_max(items):
    if not items:
        raise ValueError("empty list")
    return max(items)"""),

    _case("p12_simplify", "full_replace",
        """\
def is_valid(value):
    if value is None:
        return False
    if not isinstance(value, str):
        return False
    if len(value) == 0:
        return False
    return True""",
        """\
def is_valid(value):
    return isinstance(value, str) and len(value) > 0""",
        """\
def is_valid(value):
    return isinstance(value, str) and len(value) > 0"""),
]

# ════════════════════════════════════════════════════════════════
# Pattern 13: Add branch — add elif/else/case to existing conditional
# ════════════════════════════════════════════════════════════════

P13_CASES = [
    _case("p13_add_elif", "add_branch",
        """\
def classify(score):
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    else:
        return "F"  """,
        """\
def classify(score):
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"  """,
        """\
def classify(score):
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"  """),

    _case("p13_add_else", "add_branch",
        """\
def handle(event):
    if event.type == "click":
        process_click(event)
    elif event.type == "hover":
        process_hover(event)""",
        """\
def handle(event):
    if event.type == "click":
        process_click(event)
    elif event.type == "hover":
        process_hover(event)
    else:
        logger.warning(f"Unknown event: {event.type}")""",
        """\
def handle(event):
    if event.type == "click":
        process_click(event)
    elif event.type == "hover":
        process_hover(event)
    else:
        logger.warning(f"Unknown event: {event.type}")"""),

    _case("p13_add_case", "add_branch",
        """\
def get_color(status):
    if status == "success":
        return "green"
    elif status == "error":
        return "red"
    return "gray"  """,
        """\
def get_color(status):
    if status == "success":
        return "green"
    elif status == "error":
        return "red"
    elif status == "warning":
        return "yellow"
    elif status == "info":
        return "blue"
    return "gray"  """,
        """\
def get_color(status):
    if status == "success":
        return "green"
    elif status == "error":
        return "red"
    elif status == "warning":
        return "yellow"
    elif status == "info":
        return "blue"
    return "gray"  """),
]

# ════════════════════════════════════════════════════════════════
# Pattern 14: Extend literal — add items to list/dict/set
# ════════════════════════════════════════════════════════════════

P14_CASES = [
    _case("p14_extend_list", "extend_literal",
        """\
def get_extensions():
    return [
        ".py",
        ".js",
        ".ts",
    ]""",
        """\
def get_extensions():
    return [
        ".py",
        ".js",
        ".ts",
        ".rs",
        ".go",
        ".rb",
    ]""",
        """\
def get_extensions():
    return [
        ".py",
        ".js",
        ".ts",
        ".rs",
        ".go",
        ".rb",
    ]"""),

    _case("p14_extend_dict", "extend_literal",
        """\
def get_defaults():
    return {
        "timeout": 30,
        "retries": 3,
    }""",
        """\
def get_defaults():
    return {
        "timeout": 30,
        "retries": 3,
        "backoff": 2.0,
        "max_delay": 120,
    }""",
        """\
def get_defaults():
    return {
        "timeout": 30,
        "retries": 3,
        "backoff": 2.0,
        "max_delay": 120,
    }"""),

    _case("p14_extend_tuple", "extend_literal",
        """\
def allowed_methods():
    methods = (
        "GET",
        "POST",
    )
    return methods""",
        """\
def allowed_methods():
    methods = (
        "GET",
        "POST",
        "PUT",
        "DELETE",
        "PATCH",
    )
    return methods""",
        """\
def allowed_methods():
    methods = (
        "GET",
        "POST",
        "PUT",
        "DELETE",
        "PATCH",
    )
    return methods"""),
]

# ════════════════════════════════════════════════════════════════
# Pattern 15: Add decorator — prepend decorator above function
# ════════════════════════════════════════════════════════════════

P15_CASES = [
    _case("p15_add_cache", "add_decorator",
        """\
def compute_fib(n):
    if n <= 1:
        return n
    return compute_fib(n - 1) + compute_fib(n - 2)""",
        """\
@lru_cache(maxsize=128)
def compute_fib(n):
    if n <= 1:
        return n
    # ... existing code ...""",
        """\
@lru_cache(maxsize=128)
def compute_fib(n):
    if n <= 1:
        return n
    return compute_fib(n - 1) + compute_fib(n - 2)"""),

    _case("p15_add_route", "add_decorator",
        """\
def health():
    return {"status": "ok"}""",
        """\
@app.get("/health")
def health():
    return {"status": "ok"}""",
        """\
@app.get("/health")
def health():
    return {"status": "ok"}"""),

    _case("p15_add_multiple", "add_decorator",
        """\
def admin_panel(request):
    users = get_all_users()
    return render(users)""",
        """\
@login_required
@permission("admin")
def admin_panel(request):
    users = get_all_users()
    # ... existing code ...""",
        """\
@login_required
@permission("admin")
def admin_panel(request):
    users = get_all_users()
    return render(users)"""),
]

# ════════════════════════════════════════════════════════════════
# Pattern 16: Rename variable throughout function
# ════════════════════════════════════════════════════════════════

P16_CASES = [
    _case("p16_rename_var", "rename_variable",
        """\
def transform(data):
    tmp = normalize(data)
    tmp = filter_nulls(tmp)
    tmp = deduplicate(tmp)
    return tmp""",
        """\
def transform(data):
    result = normalize(data)
    result = filter_nulls(result)
    result = deduplicate(result)
    return result""",
        """\
def transform(data):
    result = normalize(data)
    result = filter_nulls(result)
    result = deduplicate(result)
    return result"""),

    _case("p16_rename_loop_var", "rename_variable",
        """\
def process_items(lst):
    output = []
    for x in lst:
        val = compute(x)
        output.append(val)
    return output""",
        """\
def process_items(lst):
    output = []
    for item in lst:
        val = compute(item)
        output.append(val)
    return output""",
        """\
def process_items(lst):
    output = []
    for item in lst:
        val = compute(item)
        output.append(val)
    return output"""),

    _case("p16_rename_param_usage", "rename_variable",
        """\
def format_output(s):
    cleaned = s.strip()
    upper = cleaned.upper()
    return f"[{upper}]"  """,
        """\
def format_output(text):
    cleaned = text.strip()
    upper = cleaned.upper()
    return f"[{upper}]"  """,
        """\
def format_output(text):
    cleaned = text.strip()
    upper = cleaned.upper()
    return f"[{upper}]"  """),
]

# ════════════════════════════════════════════════════════════════
# Pattern 17: Reorder statements — swap order of lines
# ════════════════════════════════════════════════════════════════

P17_CASES = [
    _case("p17_swap_two", "reorder_statements",
        """\
def setup(config):
    db = connect_db(config.db_url)
    cache = connect_cache(config.cache_url)
    queue = connect_queue(config.queue_url)
    return db, cache, queue""",
        """\
def setup(config):
    cache = connect_cache(config.cache_url)
    db = connect_db(config.db_url)
    queue = connect_queue(config.queue_url)
    return db, cache, queue""",
        """\
def setup(config):
    cache = connect_cache(config.cache_url)
    db = connect_db(config.db_url)
    queue = connect_queue(config.queue_url)
    return db, cache, queue"""),

    _case("p17_move_validation_first", "reorder_statements",
        """\
def process(raw):
    parsed = parse(raw)
    validated = validate(parsed)
    enriched = enrich(validated)
    return enriched""",
        """\
def process(raw):
    validated = validate(raw)
    parsed = parse(validated)
    enriched = enrich(parsed)
    return enriched""",
        """\
def process(raw):
    validated = validate(raw)
    parsed = parse(validated)
    enriched = enrich(parsed)
    return enriched"""),
]

# ════════════════════════════════════════════════════════════════
# Pattern 18: Extract variable — pull expression into named variable
# ════════════════════════════════════════════════════════════════

P18_CASES = [
    _case("p18_extract_condition", "extract_variable",
        """\
def check_access(user, resource):
    if user.role == "admin" or (user.role == "editor" and resource.owner == user.id):
        grant(user, resource)
    return False""",
        """\
def check_access(user, resource):
    is_admin = user.role == "admin"
    is_owner_editor = user.role == "editor" and resource.owner == user.id
    if is_admin or is_owner_editor:
        grant(user, resource)
    return False""",
        """\
def check_access(user, resource):
    is_admin = user.role == "admin"
    is_owner_editor = user.role == "editor" and resource.owner == user.id
    if is_admin or is_owner_editor:
        grant(user, resource)
    return False"""),

    _case("p18_extract_computation", "extract_variable",
        """\
def price(qty, unit_price, tax_rate):
    return round(qty * unit_price * (1 + tax_rate), 2)""",
        """\
def price(qty, unit_price, tax_rate):
    subtotal = qty * unit_price
    total = subtotal * (1 + tax_rate)
    return round(total, 2)""",
        """\
def price(qty, unit_price, tax_rate):
    subtotal = qty * unit_price
    total = subtotal * (1 + tax_rate)
    return round(total, 2)"""),

    _case("p18_extract_url", "extract_variable",
        """\
def fetch_user(user_id):
    resp = requests.get(f"https://api.example.com/v2/users/{user_id}?fields=all")
    return resp.json()""",
        """\
def fetch_user(user_id):
    base_url = "https://api.example.com/v2"
    url = f"{base_url}/users/{user_id}?fields=all"
    resp = requests.get(url)
    return resp.json()""",
        """\
def fetch_user(user_id):
    base_url = "https://api.example.com/v2"
    url = f"{base_url}/users/{user_id}?fields=all"
    resp = requests.get(url)
    return resp.json()"""),
]

# ════════════════════════════════════════════════════════════════
# Pattern 19: Inline variable — remove assignment, use value directly
# ════════════════════════════════════════════════════════════════

P19_CASES = [
    _case("p19_inline_simple", "inline_variable",
        """\
def get_name(user):
    full_name = f"{user.first} {user.last}"
    return full_name""",
        """\
def get_name(user):
    return f"{user.first} {user.last}"  """,
        """\
def get_name(user):
    return f"{user.first} {user.last}"  """),

    _case("p19_inline_flag", "inline_variable",
        """\
def should_retry(response):
    is_server_error = response.status >= 500
    is_timeout = response.status == 408
    return is_server_error or is_timeout""",
        """\
def should_retry(response):
    return response.status >= 500 or response.status == 408""",
        """\
def should_retry(response):
    return response.status >= 500 or response.status == 408"""),

    _case("p19_inline_temp", "inline_variable",
        """\
def read_config(path):
    f = open(path)
    text = f.read()
    f.close()
    data = json.loads(text)
    return data""",
        """\
def read_config(path):
    with open(path) as f:
        return json.loads(f.read())""",
        """\
def read_config(path):
    with open(path) as f:
        return json.loads(f.read())"""),
]

# ════════════════════════════════════════════════════════════════
# Pattern 20: Change expression — modify part of a line
# ════════════════════════════════════════════════════════════════

P20_CASES = [
    _case("p20_change_operator", "change_expression",
        """\
def merge(a, b):
    combined = a + b
    unique = list(set(combined))
    return sorted(unique)""",
        """\
def merge(a, b):
    combined = [*a, *b]
    unique = list(set(combined))
    return sorted(unique)""",
        """\
def merge(a, b):
    combined = [*a, *b]
    unique = list(set(combined))
    return sorted(unique)"""),

    _case("p20_change_method_call", "change_expression",
        """\
def fetch_data(url):
    response = requests.get(url)
    data = response.json()
    return data""",
        """\
def fetch_data(url):
    response = requests.get(url, timeout=30)
    data = response.json()
    return data""",
        """\
def fetch_data(url):
    response = requests.get(url, timeout=30)
    data = response.json()
    return data"""),

    _case("p20_change_string", "change_expression",
        """\
def log_event(event):
    msg = f"Event: {event.name}"
    logger.info(msg)
    return msg""",
        """\
def log_event(event):
    msg = f"[{event.timestamp}] Event: {event.name} (id={event.id})"
    logger.info(msg)
    return msg""",
        """\
def log_event(event):
    msg = f"[{event.timestamp}] Event: {event.name} (id={event.id})"
    logger.info(msg)
    return msg"""),
]

# ════════════════════════════════════════════════════════════════
# Pattern 21: Add parameter — add to signature AND update body usage
# ════════════════════════════════════════════════════════════════

P21_CASES = [
    _case("p21_add_verbose", "add_parameter",
        """\
def run_task(name):
    result = execute(name)
    return result""",
        """\
def run_task(name, verbose=False):
    if verbose:
        print(f"Running: {name}")
    result = execute(name)
    return result""",
        """\
def run_task(name, verbose=False):
    if verbose:
        print(f"Running: {name}")
    result = execute(name)
    return result"""),

    _case("p21_add_retry", "add_parameter",
        """\
def fetch(url):
    response = http.get(url)
    response.raise_for_status()
    return response.data""",
        """\
def fetch(url, retries=3):
    response = http.get(url, retries=retries)
    response.raise_for_status()
    return response.data""",
        """\
def fetch(url, retries=3):
    response = http.get(url, retries=retries)
    response.raise_for_status()
    return response.data"""),

    _case("p21_add_format_param", "add_parameter",
        """\
def export_data(records):
    output = serialize(records)
    write_file(output)
    return len(records)""",
        """\
def export_data(records, fmt="json"):
    output = serialize(records, format=fmt)
    write_file(output)
    return len(records)""",
        """\
def export_data(records, fmt="json"):
    output = serialize(records, format=fmt)
    write_file(output)
    return len(records)"""),
]

# ════════════════════════════════════════════════════════════════
# Pattern 22: Remove parameter — remove from signature AND its usage
# ════════════════════════════════════════════════════════════════

P22_CASES = [
    _case("p22_remove_debug", "remove_parameter",
        """\
def parse(text, debug=False):
    if debug:
        print(f"Parsing: {text[:50]}")
    tokens = tokenize(text)
    ast = build_tree(tokens)
    return ast""",
        """\
def parse(text):
    tokens = tokenize(text)
    ast = build_tree(tokens)
    return ast""",
        """\
def parse(text):
    tokens = tokenize(text)
    ast = build_tree(tokens)
    return ast"""),

    _case("p22_remove_unused", "remove_parameter",
        """\
def save(record, conn, logger):
    conn.execute("INSERT INTO t VALUES (?)", (record,))
    conn.commit()
    return True""",
        """\
def save(record, conn):
    conn.execute("INSERT INTO t VALUES (?)", (record,))
    conn.commit()
    return True""",
        """\
def save(record, conn):
    conn.execute("INSERT INTO t VALUES (?)", (record,))
    conn.commit()
    return True"""),

    _case("p22_remove_and_simplify", "remove_parameter",
        """\
def format_name(first, middle, last):
    if middle:
        return f"{first} {middle} {last}"
    return f"{first} {last}"  """,
        """\
def format_name(first, last):
    return f"{first} {last}"  """,
        """\
def format_name(first, last):
    return f"{first} {last}"  """),
]


# ── Collect all cases ──

ALL_CASES = (
    P1_CASES + P2_CASES + P3_CASES + P4_CASES + P5_CASES +
    P6_CASES + P7_CASES + P8_CASES + P9_CASES + P10_CASES +
    P11_CASES + P12_CASES + P13_CASES + P14_CASES + P15_CASES +
    P16_CASES + P17_CASES + P18_CASES + P19_CASES + P20_CASES +
    P21_CASES + P22_CASES
)


class TestDeterministicBenchmark:
    """Run all production-realistic cases and report by pattern."""

    @pytest.mark.parametrize(
        "name,pattern,original,snippet,expected",
        ALL_CASES,
        ids=[c[0] for c in ALL_CASES],
    )
    def test_edit(self, name, pattern, original, snippet, expected):
        result = deterministic_edit(original, snippet)
        if result is None:
            pytest.skip(f"deterministic returned None (needs model) [{pattern}]")
        assert _n(result) == _n(expected), (
            f"[{pattern}] {name}: wrong result.\n"
            f"Expected:\n{expected}\n\nGot:\n{result}"
        )

    def test_summary_report(self):
        """Aggregate results by pattern — the key metric for text_match improvements."""
        by_pattern: dict[str, dict[str, int]] = {}

        for name, pattern, original, snippet, expected in ALL_CASES:
            if pattern not in by_pattern:
                by_pattern[pattern] = {"pass": 0, "fail": 0, "skip": 0, "total": 0}

            by_pattern[pattern]["total"] += 1
            result = deterministic_edit(original, snippet)

            if result is None:
                by_pattern[pattern]["skip"] += 1
            elif _n(result) == _n(expected):
                by_pattern[pattern]["pass"] += 1
            else:
                by_pattern[pattern]["fail"] += 1

        total_pass = sum(p["pass"] for p in by_pattern.values())
        total_fail = sum(p["fail"] for p in by_pattern.values())
        total_skip = sum(p["skip"] for p in by_pattern.values())
        total = total_pass + total_fail + total_skip
        tried = total_pass + total_fail

        print(f"\n{'='*70}")
        print("DETERMINISTIC TEXT-MATCH BENCHMARK — BY EDIT PATTERN")
        print(f"{'='*70}")
        print(f"{'Pattern':<20s} {'Pass':>5s} {'Fail':>5s} {'Skip':>5s} {'Total':>5s} {'Acc':>8s}")
        print("-" * 70)

        for pattern in [
            "add_guard", "add_line", "change_line", "wrap_block",
            "add_at_end", "replace_block", "change_signature",
            "multi_insert", "class_method", "two_markers",
            "delete_lines", "full_replace", "add_branch",
            "extend_literal", "add_decorator", "rename_variable",
            "reorder_statements", "extract_variable", "inline_variable",
            "change_expression", "add_parameter", "remove_parameter",
        ]:
            if pattern not in by_pattern:
                continue
            p = by_pattern[pattern]
            t = p["pass"] + p["fail"]
            acc = f"{p['pass']/t*100:.0f}%" if t else "n/a"
            print(f"{pattern:<20s} {p['pass']:>5d} {p['fail']:>5d} {p['skip']:>5d} {p['total']:>5d} {acc:>8s}")

        print("-" * 70)
        acc_total = f"{total_pass/tried*100:.1f}%" if tried else "n/a"
        print(f"{'TOTAL':<20s} {total_pass:>5d} {total_fail:>5d} {total_skip:>5d} {total:>5d} {acc_total:>8s}")

        print(f"\n{'='*70}")
        print(f"  Deterministic: {total_pass}/{tried} = {acc_total} accuracy when tried")
        print(f"  Needs model:   {total_skip}/{total} = {total_skip/total*100:.1f}% skipped to model")
        print(f"  Wrong answers: {total_fail}/{total} = {total_fail/total*100:.1f}% (should be 0)")
        print(f"{'='*70}")
