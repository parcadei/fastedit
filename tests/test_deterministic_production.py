"""Benchmark: deterministic text_match on production-realistic edits.

Tests deterministic_edit() on function-scoped snippets — exactly how
replace=symbol_name works in CLI/MCP. Each example has:
- original_func: a single function (~10-30 lines, as AST would extract)
- snippet: edit snippet with context anchors from THAT function only
- expected: the correct merged result

Edit types tested (from real MCP usage patterns):
1. Add a line inside a function (context anchors + new line)
2. Change a condition/expression
3. Wrap with error handling (try/except + marker)
4. Add early return / guard clause
5. Modify return value
6. Multi-line insertion between anchors
7. Replace a block (no marker = drop original between anchors)
8. Add logging/debug line
"""

from __future__ import annotations

import pytest
from fastedit.inference.text_match import deterministic_edit


def _normalize(code: str) -> str:
    return "\n".join(line.rstrip() for line in code.splitlines()).strip()


# ── Test cases: (name, original_func, snippet, expected) ──


CASES = [
    # ────────────────────────────────────────────────
    # 1. Add a line inside a function
    # ────────────────────────────────────────────────
    (
        "python_add_validation_line",
        # original
        """\
def process_data(items):
    results = []
    for item in items:
        results.append(item * 2)
    return results""",
        # snippet
        """\
def process_data(items):
    if not items:
        return []
    results = []
    # ... existing code ...""",
        # expected
        """\
def process_data(items):
    if not items:
        return []
    results = []
    for item in items:
        results.append(item * 2)
    return results""",
    ),

    # ────────────────────────────────────────────────
    # 2. Change a condition
    # ────────────────────────────────────────────────
    (
        "python_change_condition",
        # original
        """\
def is_valid_user(user):
    if user.age >= 18:
        return True
    return False""",
        # snippet
        """\
def is_valid_user(user):
    if user.age >= 18 and user.is_active:
        return True
    return False""",
        # expected
        """\
def is_valid_user(user):
    if user.age >= 18 and user.is_active:
        return True
    return False""",
    ),

    # ────────────────────────────────────────────────
    # 3. Wrap with try/except using marker
    # ────────────────────────────────────────────────
    (
        "python_add_try_except",
        # original
        """\
def fetch_data(url):
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    return data""",
        # snippet
        """\
def fetch_data(url):
    try:
        # ... existing code ...
    except requests.RequestException as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return None""",
        # expected
        """\
def fetch_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.RequestException as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return None""",
    ),

    # ────────────────────────────────────────────────
    # 4. Add early return / guard clause
    # ────────────────────────────────────────────────
    (
        "python_add_guard",
        # original
        """\
def calculate_average(numbers):
    total = sum(numbers)
    count = len(numbers)
    return total / count""",
        # snippet
        """\
def calculate_average(numbers):
    if not numbers:
        return 0.0
    total = sum(numbers)
    # ... existing code ...""",
        # expected
        """\
def calculate_average(numbers):
    if not numbers:
        return 0.0
    total = sum(numbers)
    count = len(numbers)
    return total / count""",
    ),

    # ────────────────────────────────────────────────
    # 5. Modify return value
    # ────────────────────────────────────────────────
    (
        "python_modify_return",
        # original
        """\
def get_user_info(user_id):
    user = db.get(user_id)
    if user is None:
        return None
    return {"name": user.name, "email": user.email}""",
        # snippet
        """\
def get_user_info(user_id):
    # ... existing code ...
    return {"name": user.name, "email": user.email, "role": user.role}""",
        # expected
        """\
def get_user_info(user_id):
    user = db.get(user_id)
    if user is None:
        return None
    return {"name": user.name, "email": user.email, "role": user.role}""",
    ),

    # ────────────────────────────────────────────────
    # 6. Multi-line insertion between anchors
    # ────────────────────────────────────────────────
    (
        "python_multiline_insert",
        # original
        """\
def setup_app(config):
    app = Flask(__name__)
    app.config.from_object(config)
    db.init_app(app)
    return app""",
        # snippet
        """\
def setup_app(config):
    app = Flask(__name__)
    app.config.from_object(config)
    CORS(app)
    limiter = Limiter(app)
    db.init_app(app)
    return app""",
        # expected
        """\
def setup_app(config):
    app = Flask(__name__)
    app.config.from_object(config)
    CORS(app)
    limiter = Limiter(app)
    db.init_app(app)
    return app""",
    ),

    # ────────────────────────────────────────────────
    # 7. Replace a block (drop lines between anchors)
    # ────────────────────────────────────────────────
    (
        "python_replace_block",
        # original
        """\
def format_output(data):
    header = "=== Report ==="
    body = ""
    for key, value in data.items():
        body += f"{key}: {value}\\n"
    footer = "=== End ==="
    return header + "\\n" + body + footer""",
        # snippet
        """\
def format_output(data):
    header = "=== Report ==="
    body = json.dumps(data, indent=2)
    footer = "=== End ==="
    return header + "\\n" + body + footer""",
        # expected
        """\
def format_output(data):
    header = "=== Report ==="
    body = json.dumps(data, indent=2)
    footer = "=== End ==="
    return header + "\\n" + body + footer""",
    ),

    # ────────────────────────────────────────────────
    # 8. Add logging line
    # ────────────────────────────────────────────────
    (
        "python_add_logging",
        # original
        """\
def delete_record(record_id):
    record = db.find(record_id)
    if record is None:
        raise NotFoundError(f"Record {record_id} not found")
    db.delete(record)
    return True""",
        # snippet
        """\
def delete_record(record_id):
    logger.info(f"Deleting record {record_id}")
    record = db.find(record_id)
    # ... existing code ...""",
        # expected
        """\
def delete_record(record_id):
    logger.info(f"Deleting record {record_id}")
    record = db.find(record_id)
    if record is None:
        raise NotFoundError(f"Record {record_id} not found")
    db.delete(record)
    return True""",
    ),

    # ────────────────────────────────────────────────
    # 9. JavaScript: add parameter + modify body
    # ────────────────────────────────────────────────
    (
        "js_add_param_and_modify",
        # original
        """\
function fetchUsers(page) {
    const offset = (page - 1) * 20;
    const users = db.query('SELECT * FROM users LIMIT 20 OFFSET ?', [offset]);
    return users;
}""",
        # snippet
        """\
function fetchUsers(page, pageSize = 20) {
    const offset = (page - 1) * pageSize;
    const users = db.query('SELECT * FROM users LIMIT ? OFFSET ?', [pageSize, offset]);
    return users;
}""",
        # expected
        """\
function fetchUsers(page, pageSize = 20) {
    const offset = (page - 1) * pageSize;
    const users = db.query('SELECT * FROM users LIMIT ? OFFSET ?', [pageSize, offset]);
    return users;
}""",
    ),

    # ────────────────────────────────────────────────
    # 10. TypeScript: add type guard + marker
    # ────────────────────────────────────────────────
    (
        "ts_add_type_guard",
        # original
        """\
function processEvent(event: Event): Result {
    const handler = handlers.get(event.type);
    const result = handler(event.payload);
    return { success: true, data: result };
}""",
        # snippet
        """\
function processEvent(event: Event): Result {
    const handler = handlers.get(event.type);
    if (!handler) {
        return { success: false, error: 'Unknown event type' };
    }
    const result = handler(event.payload);
    // ... existing code ...""",
        # expected
        """\
function processEvent(event: Event): Result {
    const handler = handlers.get(event.type);
    if (!handler) {
        return { success: false, error: 'Unknown event type' };
    }
    const result = handler(event.payload);
    return { success: true, data: result };
}""",
    ),

    # ────────────────────────────────────────────────
    # 11. Rust: add error handling with marker
    # ────────────────────────────────────────────────
    (
        "rust_add_error_handling",
        # original
        """\
fn read_config(path: &str) -> Config {
    let contents = fs::read_to_string(path).unwrap();
    let config: Config = serde_json::from_str(&contents).unwrap();
    config
}""",
        # snippet
        """\
fn read_config(path: &str) -> Result<Config, Box<dyn Error>> {
    let contents = fs::read_to_string(path)?;
    let config: Config = serde_json::from_str(&contents)?;
    Ok(config)
}""",
        # expected
        """\
fn read_config(path: &str) -> Result<Config, Box<dyn Error>> {
    let contents = fs::read_to_string(path)?;
    let config: Config = serde_json::from_str(&contents)?;
    Ok(config)
}""",
    ),

    # ────────────────────────────────────────────────
    # 12. Go: add context parameter
    # ────────────────────────────────────────────────
    (
        "go_add_context_param",
        # original
        """\
func GetUser(id string) (*User, error) {
    row := db.QueryRow("SELECT * FROM users WHERE id = ?", id)
    var user User
    err := row.Scan(&user.ID, &user.Name, &user.Email)
    if err != nil {
        return nil, err
    }
    return &user, nil
}""",
        # snippet
        """\
func GetUser(ctx context.Context, id string) (*User, error) {
    row := db.QueryRowContext(ctx, "SELECT * FROM users WHERE id = ?", id)
    var user User
    // ... existing code ...""",
        # expected
        """\
func GetUser(ctx context.Context, id string) (*User, error) {
    row := db.QueryRowContext(ctx, "SELECT * FROM users WHERE id = ?", id)
    var user User
    err := row.Scan(&user.ID, &user.Name, &user.Email)
    if err != nil {
        return nil, err
    }
    return &user, nil
}""",
    ),

    # ────────────────────────────────────────────────
    # 13. Python: change only the middle of a long function
    # ────────────────────────────────────────────────
    (
        "python_change_middle",
        # original
        """\
def process_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path) as f:
        content = f.read()
    lines = content.splitlines()
    result = []
    for line in lines:
        if line.strip():
            result.append(line.upper())
    return "\\n".join(result)""",
        # snippet
        """\
def process_file(path):
    # ... existing code ...
    for line in lines:
        if line.strip():
            result.append(line.strip().upper())
    return "\\n".join(result)""",
        # expected
        """\
def process_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path) as f:
        content = f.read()
    lines = content.splitlines()
    result = []
    for line in lines:
        if line.strip():
            result.append(line.strip().upper())
    return "\\n".join(result)""",
    ),

    # ────────────────────────────────────────────────
    # 14. Python: add decorator-style wrapping
    # ────────────────────────────────────────────────
    (
        "python_add_caching",
        # original
        """\
def get_price(product_id):
    product = db.products.find_one(product_id)
    if product is None:
        return None
    return product["price"]""",
        # snippet
        """\
def get_price(product_id):
    cached = cache.get(f"price:{product_id}")
    if cached is not None:
        return cached
    product = db.products.find_one(product_id)
    # ... existing code ...
    cache.set(f"price:{product_id}", product["price"], ttl=300)
    return product["price"]""",
        # expected
        """\
def get_price(product_id):
    cached = cache.get(f"price:{product_id}")
    if cached is not None:
        return cached
    product = db.products.find_one(product_id)
    if product is None:
        return None
    cache.set(f"price:{product_id}", product["price"], ttl=300)
    return product["price"]""",
    ),

    # ────────────────────────────────────────────────
    # 15. JavaScript: simple one-line change
    # ────────────────────────────────────────────────
    (
        "js_simple_change",
        # original
        """\
function greet(name) {
    const greeting = `Hello, ${name}!`;
    console.log(greeting);
    return greeting;
}""",
        # snippet
        """\
function greet(name) {
    const greeting = `Hello, ${name}! Welcome back.`;
    console.log(greeting);
    return greeting;
}""",
        # expected
        """\
function greet(name) {
    const greeting = `Hello, ${name}! Welcome back.`;
    console.log(greeting);
    return greeting;
}""",
    ),

    # ────────────────────────────────────────────────
    # 16. Python: add method to class (scoped to class)
    # ────────────────────────────────────────────────
    (
        "python_modify_class_method",
        # original
        """\
class UserService:
    def __init__(self, db):
        self.db = db

    def get_user(self, user_id):
        return self.db.find(user_id)

    def delete_user(self, user_id):
        self.db.remove(user_id)""",
        # snippet
        """\
class UserService:
    # ... existing code ...

    def get_user(self, user_id):
        user = self.db.find(user_id)
        if user is None:
            raise ValueError(f"User {user_id} not found")
        return user

    def delete_user(self, user_id):
        # ... existing code ...""",
        # expected
        """\
class UserService:
    def __init__(self, db):
        self.db = db

    def get_user(self, user_id):
        user = self.db.find(user_id)
        if user is None:
            raise ValueError(f"User {user_id} not found")
        return user

    def delete_user(self, user_id):
        self.db.remove(user_id)""",
    ),

    # ────────────────────────────────────────────────
    # 17. Python: marker at end preserves trailing code
    # ────────────────────────────────────────────────
    (
        "python_marker_at_start",
        # original
        """\
def transform(data):
    validated = validate(data)
    cleaned = clean(validated)
    normalized = normalize(cleaned)
    return normalized""",
        # snippet
        """\
def transform(data):
    logger.debug(f"Transforming {len(data)} items")
    validated = validate(data)
    # ... existing code ...""",
        # expected
        """\
def transform(data):
    logger.debug(f"Transforming {len(data)} items")
    validated = validate(data)
    cleaned = clean(validated)
    normalized = normalize(cleaned)
    return normalized""",
    ),

    # ────────────────────────────────────────────────
    # 18. TypeScript: replace implementation keeping signature
    # ────────────────────────────────────────────────
    (
        "ts_replace_implementation",
        # original
        """\
function sortItems(items: Item[]): Item[] {
    return items.sort((a, b) => a.name.localeCompare(b.name));
}""",
        # snippet
        """\
function sortItems(items: Item[]): Item[] {
    return [...items].sort((a, b) => {
        if (a.priority !== b.priority) {
            return b.priority - a.priority;
        }
        return a.name.localeCompare(b.name);
    });
}""",
        # expected
        """\
function sortItems(items: Item[]): Item[] {
    return [...items].sort((a, b) => {
        if (a.priority !== b.priority) {
            return b.priority - a.priority;
        }
        return a.name.localeCompare(b.name);
    });
}""",
    ),

    # ────────────────────────────────────────────────
    # 19. Python: two markers preserving two gaps
    # ────────────────────────────────────────────────
    (
        "python_two_markers",
        # original
        """\
def pipeline(raw_data):
    step1 = parse(raw_data)
    step2 = validate(step1)
    step3 = transform(step2)
    step4 = enrich(step3)
    step5 = format_output(step4)
    return step5""",
        # snippet
        """\
def pipeline(raw_data):
    # ... existing code ...
    step3 = transform(step2)
    step3 = deduplicate(step3)
    step4 = enrich(step3)
    # ... existing code ...""",
        # expected
        """\
def pipeline(raw_data):
    step1 = parse(raw_data)
    step2 = validate(step1)
    step3 = transform(step2)
    step3 = deduplicate(step3)
    step4 = enrich(step3)
    step5 = format_output(step4)
    return step5""",
    ),

    # ────────────────────────────────────────────────
    # 20. Ruby: add error handling
    # ────────────────────────────────────────────────
    (
        "ruby_add_rescue",
        # original
        """\
def fetch_record(id)
  record = Record.find(id)
  record.to_json
end""",
        # snippet
        """\
def fetch_record(id)
  record = Record.find(id)
  record.to_json
rescue ActiveRecord::RecordNotFound
  { error: "Record not found" }.to_json
end""",
        # expected
        """\
def fetch_record(id)
  record = Record.find(id)
  record.to_json
rescue ActiveRecord::RecordNotFound
  { error: "Record not found" }.to_json
end""",
    ),
]


class TestDeterministicProduction:
    """Test deterministic_edit on production-realistic function-scoped edits."""

    @pytest.mark.parametrize("name,original,snippet,expected", CASES, ids=[c[0] for c in CASES])
    def test_edit(self, name, original, snippet, expected):
        result = deterministic_edit(original, snippet)
        if result is None:
            pytest.skip("deterministic returned None (would need model)")
        assert _normalize(result) == _normalize(expected), (
            f"Deterministic produced wrong result for {name}.\n"
            f"Expected:\n{expected}\n\nGot:\n{result}"
        )

    def test_summary(self):
        """Run all cases and print a summary report."""
        passed = 0
        failed = 0
        skipped = 0
        details = []

        for name, original, snippet, expected in CASES:
            result = deterministic_edit(original, snippet)
            if result is None:
                skipped += 1
                details.append((name, "SKIP"))
            elif _normalize(result) == _normalize(expected):
                passed += 1
                details.append((name, "PASS"))
            else:
                failed += 1
                details.append((name, "FAIL"))

        total = passed + failed + skipped
        tried = passed + failed

        print(f"\n{'='*60}")
        print("PRODUCTION-REALISTIC BENCHMARK (function-scoped edits)")
        print(f"{'='*60}")
        print(f"Total: {total}")
        print(f"  PASS:  {passed} ({passed/total*100:.1f}%)")
        print(f"  FAIL:  {failed} ({failed/total*100:.1f}%)")
        print(f"  SKIP:  {skipped} ({skipped/total*100:.1f}%)")
        if tried:
            print(f"  Accuracy when tried: {passed}/{tried} = {passed/tried*100:.1f}%")

        print(f"\n--- Details ---")
        for name, status in details:
            marker = {"PASS": "+", "FAIL": "X", "SKIP": "-"}[status]
            print(f"  [{marker}] {name}")

        # This is the real question: on function-scoped edits with
        # proper context anchors, does deterministic_edit work?
        if tried > 0:
            accuracy = passed / tried * 100
            if accuracy >= 80:
                print(f"\nDeterministic handles {accuracy:.0f}% of scoped edits — model is fallback only")
            elif accuracy >= 50:
                print(f"\nDeterministic handles {accuracy:.0f}% — model needed for ~half")
            else:
                print(f"\nDeterministic handles {accuracy:.0f}% — model does most of the work")
