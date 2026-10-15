"""Symbol resolution & AST freshness regressions (Step 16: B25, B26, B35, B36).

B25 — ``_resolve_symbol`` returned the FIRST node whose bare name matched, so
two classes each defining ``save()`` made every ``replace=/after=/delete/move``
silently edit the wrong symbol. A bare name that matches more than one distinct
node must raise ``ValueError`` listing every match's qualified name
(``A.save``, ``B.save``, ...); a dotted path (``B.save``, ``Outer.Inner.save``)
must resolve to exactly that node.

B26 — ``locate_chunks`` built its AST map from the DISK file via
``get_ast_map``, whose tldr-daemon cache can hold pre-write line numbers. Other
paths already parse ``original_code`` in-memory (``get_ast_map_from_source``);
locate_chunks must too, or chunks get spliced into the wrong region.

B36 — ``_get_ast_via_extract`` derived every ``line_end`` as
``next_entry.line_start - 1``: a class was truncated to the line above its
first method (the method is nested INSIDE the class), and a method's end
swallowed following lines tldr extract never reported (module constants,
comments, the next symbol's header). Ends must use real AST end positions
where available, with an order-aware clamp (an entry may not cross the start
of a subsequent entry that is not nested inside it) as fallback.

B35 — ``move_symbol`` still used the stale disk-based ``get_ast_map`` while
``delete_symbol`` uses the in-memory map; a move spliced from stale
coordinates corrupts both the moved span and its neighbours.
"""

from __future__ import annotations

import itertools
import json

import pytest

from fastedit.inference.ast_utils import (
    _get_ast_via_extract,
    _resolve_symbol,
    get_ast_map_from_source,
)
from fastedit.inference.chunk_locator import locate_chunks
from fastedit.inference.symbols import delete_symbol, move_symbol

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


DUP_SAVE_SRC = '''\
class A:
    def save(self):
        return "A"

class B:
    def save(self):
        return "B"

def alpha():
    return 1

def beta():
    return 2
'''


NESTED_AND_TOP_SRC = '''\
def save():
    return "top"

class Outer:
    class Inner:
        def save(self):
            return "outer-inner"

class Other:
    class Inner:
        def save(self):
            return "other-inner"
'''


UNIQUE_NAMES_SRC = '''\
class A:
    def save(self):
        return 1

def standalone():
    return 2
'''


# Three top-level functions, one blank line between them.
THREE_FUNCS_OLD = (
    "def alpha():\n"
    "    return 1\n"
    "\n"
    "def beta():\n"
    "    return 2\n"
    "\n"
    "def gamma():\n"
    "    return 3\n"
)

# The same file after an (external) edit inserted two comment lines above
# beta: every symbol below the insertion moved down by two lines.
THREE_FUNCS_NEW = (
    "def alpha():\n"
    "    return 1\n"
    "\n"
    "# external edit landed here A\n"
    "# external edit landed here B\n"
    "def beta():\n"
    "    return 2\n"
    "\n"
    "def gamma():\n"
    "    return 3\n"
)


def _assert_non_overlapping(nodes) -> None:
    """Every pair of nodes must be nested or disjoint — never partial overlap."""
    spans = [(n.name, n.line_start, n.line_end) for n in nodes]
    for (n1, s1, e1), (n2, s2, e2) in itertools.combinations(spans, 2):
        contains_12 = s1 <= s2 and e1 >= e2
        contains_21 = s2 <= s1 and e2 >= e1
        if contains_12 or contains_21:
            continue
        assert e1 < s2 or e2 < s1, (
            f"overlapping line ranges: {n1}={s1}-{e1} and {n2}={s2}-{e2}"
        )


# ---------------------------------------------------------------------------
# B25 — _resolve_symbol: ambiguity + qualified-name grammar
# ---------------------------------------------------------------------------


def test_bare_name_matching_two_methods_raises_listing_both_qualified():
    """B25: bare `save` with two definitions must fail loud, not first-match."""
    nodes = get_ast_map_from_source(DUP_SAVE_SRC, "dup.py")
    assert nodes, "precondition: in-memory AST map must resolve this source"

    with pytest.raises(ValueError) as excinfo:
        _resolve_symbol("save", nodes)

    message = str(excinfo.value)
    assert "A.save" in message, message
    assert "B.save" in message, message


def test_qualified_name_resolves_exactly_that_class_method():
    """`B.save` must resolve to B's method, never A's."""
    nodes = get_ast_map_from_source(DUP_SAVE_SRC, "dup.py")

    node = _resolve_symbol("B.save", nodes)
    assert node is not None
    assert node.name == "save"
    assert node.parent == "B"
    b_class = next(n for n in nodes if n.name == "B")
    a_class = next(n for n in nodes if n.name == "A")
    assert b_class.line_start <= node.line_start <= node.line_end <= b_class.line_end
    assert not (a_class.line_start <= node.line_start <= node.line_end <= a_class.line_end)

    node_a = _resolve_symbol("A.save", nodes)
    assert node_a is not None
    assert node_a.parent == "A"


def test_unique_bare_name_resolves_as_today():
    """Control: a bare name with a single definition keeps resolving."""
    nodes = get_ast_map_from_source(UNIQUE_NAMES_SRC, "unique.py")

    method = _resolve_symbol("save", nodes)
    assert method is not None
    assert method.parent == "A"
    assert method.kind == "method"

    top_level = _resolve_symbol("standalone", nodes)
    assert top_level is not None
    assert top_level.parent is None
    assert top_level.kind == "function"

    cls = _resolve_symbol("A", nodes)
    assert cls is not None
    assert cls.kind == "class"


def test_same_name_nested_and_top_level_is_still_ambiguous():
    """B25: one top-level `save` and one nested `Outer.Inner.save` — both match
    the bare name, so it must stay ambiguous (fail loud), with the nested one
    listed by its FULL dotted path."""
    nodes = get_ast_map_from_source(NESTED_AND_TOP_SRC, "nested.py")
    assert nodes, "precondition: in-memory AST map must resolve this source"

    with pytest.raises(ValueError) as excinfo:
        _resolve_symbol("save", nodes)

    message = str(excinfo.value)
    assert "Outer.Inner.save" in message, message


def test_deep_qualified_path_resolves_the_nested_method():
    """`Outer.Inner.save` resolves through the containment chain, and same-named
    Inner classes in different scopes are told apart."""
    nodes = get_ast_map_from_source(NESTED_AND_TOP_SRC, "nested.py")

    node = _resolve_symbol("Outer.Inner.save", nodes)
    assert node is not None
    assert node.name == "save"
    assert node.parent == "Inner"
    outer = next(n for n in nodes if n.name == "Outer")
    other = next(n for n in nodes if n.name == "Other")
    assert outer.line_start <= node.line_start <= node.line_end <= outer.line_end
    assert not (other.line_start <= node.line_start <= node.line_end <= other.line_end)


def test_dotted_input_that_still_matches_two_nodes_is_ambiguous():
    """`Inner.save` matches the same-named nested classes of both Outer and
    Other — dotted input disambiguates only when it is actually unique."""
    nodes = get_ast_map_from_source(NESTED_AND_TOP_SRC, "nested.py")

    with pytest.raises(ValueError) as excinfo:
        _resolve_symbol("Inner.save", nodes)

    message = str(excinfo.value)
    assert "Outer.Inner.save" in message, message
    assert "Other.Inner.save" in message, message


def test_qualified_input_for_missing_symbol_still_returns_none():
    """A dotted path whose chain does not exist is simply not found — callers
    keep their existing not-found refusal behaviour."""
    nodes = get_ast_map_from_source(DUP_SAVE_SRC, "dup.py")
    assert _resolve_symbol("C.save", nodes) is None
    assert _resolve_symbol("A.missing", nodes) is None
    assert _resolve_symbol("missing", nodes) is None


JAVA_CLASS_WITH_CTOR_SRC = """\
public class Store {
    private java.util.List<String> items;

    public Store() {
        this.items = new java.util.ArrayList<>();
    }
}
"""


def test_bare_name_matching_class_and_its_own_constructor_resolves_to_class():
    """A constructor sharing its class's name is a MEMBER of the class, not a
    competitor: the bare name resolves to the class (the outermost match, as
    first-match behaviour did), and the constructor stays reachable via its
    qualified path ``Store.Store``."""
    nodes = get_ast_map_from_source(JAVA_CLASS_WITH_CTOR_SRC, "store.java")
    assert nodes, "precondition: in-memory AST map must resolve this source"

    cls = _resolve_symbol("Store", nodes)
    assert cls is not None
    assert cls.kind == "class"

    ctor = _resolve_symbol("Store.Store", nodes)
    assert ctor is not None
    assert ctor.kind == "method"
    assert cls.line_start <= ctor.line_start <= ctor.line_end <= cls.line_end


# ---------------------------------------------------------------------------
# B25 — callers surface the ambiguity as a clean refusal
# ---------------------------------------------------------------------------


def test_delete_symbol_ambiguous_bare_name_refuses_cleanly(tmp_path):
    """delete_symbol refuses with both qualified names — no wrong deletion."""
    f = tmp_path / "dup.py"
    f.write_text(DUP_SAVE_SRC)

    with pytest.raises(ValueError) as excinfo:
        delete_symbol(str(f), "save")

    message = str(excinfo.value)
    assert "A.save" in message, message
    assert "B.save" in message, message
    # Nothing was written.
    assert f.read_text() == DUP_SAVE_SRC


def test_move_symbol_ambiguous_bare_name_refuses_cleanly(tmp_path):
    """move_symbol refuses with both qualified names — no wrong move."""
    f = tmp_path / "dup.py"
    f.write_text(DUP_SAVE_SRC)

    with pytest.raises(ValueError) as excinfo:
        move_symbol(str(f), "save", after="alpha")

    message = str(excinfo.value)
    assert "A.save" in message, message
    assert "B.save" in message, message
    assert f.read_text() == DUP_SAVE_SRC


def test_chunked_merge_ambiguous_replace_refuses_cleanly(tmp_path):
    """chunked_merge's replace= fast path refuses an ambiguous bare symbol."""
    from fastedit.inference.chunked_merge import chunked_merge

    def _no_model(*_args, **_kwargs):
        raise AssertionError("merge_fn must not be called on a refused edit")

    f = tmp_path / "dup.py"
    f.write_text(DUP_SAVE_SRC)

    with pytest.raises(ValueError) as excinfo:
        chunked_merge(
            original_code=DUP_SAVE_SRC,
            snippet='def save(self):\n    return "A"\n',
            file_path=str(f),
            merge_fn=_no_model,
            language="python",
            replace="save",
        )

    message = str(excinfo.value)
    assert "A.save" in message, message
    assert "B.save" in message, message


def test_delete_symbol_qualified_input_deletes_exactly_that_method(tmp_path):
    """Control: qualified `A.save` deletes A's method and leaves B's alone."""
    f = tmp_path / "dup.py"
    f.write_text(DUP_SAVE_SRC)

    result = delete_symbol(str(f), "A.save")

    assert result.deleted_symbol == "save"
    assert result.deleted_kind == "method"
    assert result.deleted_lines == (2, 3), (
        f"expected A.save's span (2, 3), got {result.deleted_lines}"
    )
    # delete_symbol returns the spliced result; persistence is the writer's
    # job (tools do _atomic_write) — so assert on merged_code.
    after = result.merged_code
    assert 'return "A"' not in after
    assert 'return "B"' in after, "qualified delete touched the wrong class"
    assert "def beta():" in after


# ---------------------------------------------------------------------------
# B26 — locate_chunks parses the in-memory source, not the disk-cached map
# ---------------------------------------------------------------------------


# The on-disk view (what a stale tldr-daemon cache would still report). This
# mirrors the harness of test_chained_edits_stale_ast.py: monkeypatch the
# caller's `get_ast_map` to return the PRE-edit line numbers.
STALE_DISK_SRC = (
    "def retry_batch(client, records, max_attempts=3):\n"
    "    attempt = 0\n"
    "    while attempt < max_attempts:\n"
    "        try:\n"
    "            return client.send(records)\n"
    "        except TransientError:\n"
    "            attempt += 1\n"
    "    raise RuntimeError('exhausted retries')\n"
    "\n"
    "def fetch_user(db, user_id):\n"
    '    row = db.query("SELECT * FROM users WHERE id = ?", user_id)\n'
    "    if row is None:\n"
    "        return None\n"
    "    return User(id=row[0], name=row[1], email=row[2])\n"
)

# In-memory content after recent edits grew the file: fetch_user moved down.
FRESH_SRC = (
    "# injected prelude comment for line-shift test\n" * 8
) + STALE_DISK_SRC

REPLACE_SNIPPET = (
    "def fetch_user(db, user_id):\n"
    '    row = db.query("SELECT * FROM users WHERE id = ? AND active = 1", user_id)\n'
)


def test_locate_chunks_aligns_to_in_memory_content_after_modification(
    tmp_path, monkeypatch,
):
    """B26: the file was just rewritten (disk still holds the old bytes a
    daemon cache would serve); locate_chunks must derive chunk coordinates
    from the `original_code` argument, so the region starts exactly on the
    symbol's NEW header line."""
    f = tmp_path / "svc.py"
    f.write_text(STALE_DISK_SRC)  # disk = pre-edit view; NOT rewritten below

    stale_nodes = get_ast_map_from_source(STALE_DISK_SRC, str(f))
    assert stale_nodes, "precondition: in-memory AST map must resolve this source"

    # Pin the disk path shut. Before the fix, chunk_locator imported
    # `get_ast_map` and called it — the stub raises on that call. After the
    # fix the module no longer even holds a reference to the disk map, so
    # the stale view is structurally unreachable.
    import fastedit.inference.chunk_locator as chunk_locator_module

    if hasattr(chunk_locator_module, "get_ast_map"):
        monkeypatch.setattr(
            chunk_locator_module, "get_ast_map",
            lambda *_a, **_kw: stale_nodes,
        )

    chunks = locate_chunks(
        REPLACE_SNIPPET,
        FRESH_SRC,
        str(f),
        language="python",
        replace="fetch_user",
    )

    assert len(chunks) == 1, chunks
    region = chunks[0]
    lines = FRESH_SRC.splitlines()
    assert lines[region.start_line - 1].startswith("def fetch_user"), (
        f"chunk starts at line {region.start_line} which is "
        f"{lines[region.start_line - 1]!r}, not the symbol's new header"
    )
    region_text = "\n".join(lines[region.start_line - 1:region.end_line])
    assert "row = db.query" in region_text, region_text
    assert "injected prelude comment" not in region_text, region_text


# ---------------------------------------------------------------------------
# B36 — _get_ast_via_extract end-line derivation
# ---------------------------------------------------------------------------


class _FakeCompleted:
    """Stand-in for subprocess.CompletedProcess with canned tldr JSON."""

    def __init__(self, payload):
        self.returncode = 0
        self.stdout = json.dumps(payload)
        self.stderr = ""


def _patch_extract(monkeypatch, payload):
    monkeypatch.setattr(
        "fastedit.inference.ast_utils.subprocess.run",
        lambda *_a, **_kw: _FakeCompleted(payload),
    )


def test_extract_fallback_method_end_does_not_swallow_following_lines(
    tmp_path, monkeypatch,
):
    """B36: two methods in a class, the second shorter — the first method's
    span must end before its sibling, the sibling must not swallow the module
    constant that follows the class, and the class must span its members."""
    src = (
        "class Calc:\n"                  # 1
        "    def long_method(self):\n"   # 2
        "        total = 1\n"            # 3
        "        total += 2\n"           # 4
        "        total += 3\n"           # 5
        "    def short(self):\n"         # 6
        "        return total\n"         # 7
        "TOP_CONSTANT = 'keep-me'\n"     # 8
        "\n"                             # 9
        "def after():\n"                 # 10
        "    return 2\n"                 # 11
    )
    p = tmp_path / "calc.py"
    p.write_text(src)
    _patch_extract(monkeypatch, {
        "functions": [{"name": "after", "line_number": 10}],
        "classes": [{
            "name": "Calc",
            "line_number": 1,
            "methods": [
                {"name": "long_method", "line_number": 2},
                {"name": "short", "line_number": 6},
            ],
        }],
    })

    nodes = _get_ast_via_extract(str(p), 0)
    by_name = {n.name: n for n in nodes}
    assert {"Calc", "long_method", "short", "after"} <= set(by_name)

    calc = by_name["Calc"]
    long_method = by_name["long_method"]
    short = by_name["short"]

    # The class spans its own members (old rule: end = first member start - 1).
    assert calc.line_end >= short.line_end, (
        f"class Calc ends at {calc.line_end}, before its last method "
        f"({short.line_end}) — truncated span"
    )
    # The shorter sibling's end must not swallow TOP_CONSTANT on line 8.
    assert short.line_end < 8, (
        f"short ends at {short.line_end}, swallowing line 8"
    )
    # Sibling methods never overlap.
    assert long_method.line_end < short.line_start

    _assert_non_overlapping(nodes)


def test_extract_fallback_interleaved_classes_do_not_cross_boundaries(
    tmp_path, monkeypatch,
):
    """B36: class entries interleaved with method entries — each class keeps
    its body, each method stays inside its own class."""
    src = (
        "class First:\n"         # 1
        "    def one(self):\n"   # 2
        "        return 1\n"     # 3
        "class Second:\n"        # 4
        "    def two(self):\n"   # 5
        "        return 2\n"     # 6
    )
    p = tmp_path / "two_classes.py"
    p.write_text(src)
    _patch_extract(monkeypatch, {
        "functions": [],
        "classes": [
            {
                "name": "First",
                "line_number": 1,
                "methods": [{"name": "one", "line_number": 2}],
            },
            {
                "name": "Second",
                "line_number": 4,
                "methods": [{"name": "two", "line_number": 5}],
            },
        ],
    })

    nodes = _get_ast_via_extract(str(p), 0)
    by_name = {n.name: n for n in nodes}
    assert {"First", "one", "Second", "two"} <= set(by_name)

    first, one = by_name["First"], by_name["one"]
    second, two = by_name["Second"], by_name["two"]

    assert first.line_end >= one.line_end, (
        f"First ends at {first.line_end}, before its own method ({one.line_end})"
    )
    assert second.line_end >= two.line_end, (
        f"Second ends at {second.line_end}, before its own method ({two.line_end})"
    )
    # A method's span must not reach into the next class's header.
    assert one.line_end < second.line_start

    _assert_non_overlapping(nodes)


# ---------------------------------------------------------------------------
# B35 — move_symbol uses the in-memory map like delete_symbol
# ---------------------------------------------------------------------------


def test_move_symbol_does_not_consult_disk_ast_map(tmp_path, monkeypatch):
    """B35: move_symbol must resolve coordinates in-memory (as delete_symbol
    already does); consulting the stale disk map is a defect, not a fallback."""
    f = tmp_path / "three.py"
    f.write_text(THREE_FUNCS_OLD)

    def _fail(*_a, **_kw):
        raise AssertionError("move_symbol consulted the disk-based get_ast_map")

    monkeypatch.setattr("fastedit.inference.symbols.get_ast_map", _fail)

    result = move_symbol(str(f), "alpha", after="gamma")

    assert result.moved_symbol == "alpha"
    code = result.merged_code
    assert code.index("def beta():") < code.index("def gamma():") < code.index(
        "def alpha():"
    )


def test_move_symbol_ignores_stale_disk_map_after_external_edit(tmp_path, monkeypatch):
    """B35 end-to-end: the file was externally edited on disk after the AST
    view was built. move_symbol must splice using the CURRENT file content —
    a stale map moves the wrong lines and strands beta without its body."""
    f = tmp_path / "three.py"
    f.write_text(THREE_FUNCS_OLD)  # what the stale disk map still sees

    stale_nodes = get_ast_map_from_source(THREE_FUNCS_OLD, str(f))
    assert stale_nodes, "precondition: in-memory AST map must resolve this source"

    f.write_text(THREE_FUNCS_NEW)  # the external edit lands on disk

    monkeypatch.setattr(
        "fastedit.inference.symbols.get_ast_map",
        lambda *_a, **_kw: stale_nodes,
    )

    result = move_symbol(str(f), "gamma", after="alpha", language="python")

    code = result.merged_code
    assert result.parse_valid, f"moved output does not parse:\n{code}"
    assert code.count("def alpha():") == 1
    assert code.count("def beta():") == 1
    assert code.count("def gamma():") == 1
    # beta keeps its own body — the stale map would have moved `return 2`
    # away as if it were gamma's.
    assert "def beta():\n    return 2" in code, code
    # gamma moves after alpha, beta stays last.
    assert (
        code.index("def alpha():")
        < code.index("def gamma():")
        < code.index("def beta():")
    ), code
    # The external edit's lines survive untouched.
    assert "# external edit landed here A" in code
    assert "# external edit landed here B" in code
