"""Regression test for class-method partial-replace corruption (v0.2.3).

The bug: when a snippet targeting a method (or any function) uses
``# ... existing code ...`` markers with a "new" line whose LHS matches
a line in the preserved gap, the deterministic fast path emitted BOTH
the old and the new line, silently corrupting the output.

Example:

  Snippet::

      def __init__(self, ttl=3600):
          self.ttl = ttl
          self._data = OrderedDict()
      # ... existing code ...
          self.max_size = 1000

  Applied to::

      def __init__(self, ttl=3600):
          self.ttl = ttl
          self._last_access = {}
          self._data = {}
          self.max_size = 1000

  Before fix — both ``self._data = {}`` and ``self._data = OrderedDict()``
  ended up in the output, producing a double-assignment. After fix the
  old line is dropped.

The fix lives in ``fastedit.inference.text_match.deterministic_edit``:
when a ``new`` line in the same section as a marker shares its LHS AND
indent with exactly one gap line, that gap line is skipped. The
exactly-one guard prevents this from misfiring on pipelines that rebind
the same name repeatedly (``data = clean(data)`` / ``data = validate(data)``
etc.) and the indent guard prevents misfire on wrap_block-style rewrites
where the same name is bound in a different scope.
"""

from __future__ import annotations

import ast

from fastedit.inference.chunked_merge import chunked_merge


def _no_model(*_args, **_kwargs):
    """Merge fn that fails loudly if the fast path falls through to the model."""
    raise AssertionError(
        "merge_fn must NOT be called — this edit should hit the "
        "deterministic text-match fast path"
    )


# ---------------------------------------------------------------------------
# Python: class method with a marker-preserved gap that contains a sibling
# line whose LHS matches the new assignment. This is the exact corruption
# path reported for v0.2.3.
# ---------------------------------------------------------------------------

PY_CLASS = """\
class SessionStore:
    def __init__(self, ttl=3600):
        self.ttl = ttl
        self._last_access = {}
        self._data = {}
        self.max_size = 1000

    def get(self, key):
        return self._data.get(key)

    def set(self, key, value):
        self._data[key] = value

    def clear(self):
        self._data.clear()
"""

# Partial snippet: change one line in the middle of __init__, preserve
# the rest via marker. Pre-fix this duplicated ``self._data``. The
# marker is placed at body indent so the preserved-gap indent logic
# leaves the ``_last_access`` line alone.
PY_CLASS_METHOD_SNIPPET = (
    "    def __init__(self, ttl=3600):\n"
    "        self.ttl = ttl\n"
    "        self._data = OrderedDict()\n"
    "        # ... existing code ...\n"
    "        self.max_size = 1000\n"
)


def test_class_method_replace_with_body_change(tmp_path):
    """BUG REPRO: replace= on a class method with a marker-preserved gap
    must NOT duplicate the replaced line alongside the new one."""
    file_path = tmp_path / "store.py"
    file_path.write_text(PY_CLASS)

    result = chunked_merge(
        original_code=file_path.read_text(),
        snippet=PY_CLASS_METHOD_SNIPPET,
        file_path=str(file_path),
        merge_fn=_no_model,
        language="python",
        replace="__init__",
    )
    merged = result.merged_code

    # Primary assertion: the old assignment must be gone.
    assert "self._data = {}" not in merged, (
        f"old assignment retained alongside new — corruption!\n{merged}"
    )
    # The new assignment must be present.
    assert "self._data = OrderedDict()" in merged, (
        f"new assignment missing:\n{merged}"
    )
    # Appears exactly once.
    assert merged.count("self._data = OrderedDict()") == 1

    # Output must still be valid Python.
    ast.parse(merged)

    # Deterministic fast path.
    assert result.model_tokens == 0


def test_class_method_replace_preserves_siblings(tmp_path):
    """Sibling methods (get/set/clear) must be untouched after edit to __init__."""
    file_path = tmp_path / "store.py"
    file_path.write_text(PY_CLASS)

    result = chunked_merge(
        original_code=file_path.read_text(),
        snippet=PY_CLASS_METHOD_SNIPPET,
        file_path=str(file_path),
        merge_fn=_no_model,
        language="python",
        replace="__init__",
    )
    merged = result.merged_code

    # Unchanged sibling methods — verbatim.
    assert "    def get(self, key):\n        return self._data.get(key)" in merged
    assert "    def set(self, key, value):\n        self._data[key] = value" in merged
    assert "    def clear(self):\n        self._data.clear()" in merged

    # And the preserved __init__ gap line (not the one we replaced) survives.
    assert "        self._last_access = {}" in merged
    assert "        self.max_size = 1000" in merged

    # Class shell intact.
    assert "class SessionStore:" in merged


PY_TOP_LEVEL = """\
def setup(ttl=3600):
    global_ttl = ttl
    last_access = {}
    data = {}
    max_size = 1000
"""

PY_TOP_LEVEL_SNIPPET = (
    "def setup(ttl=3600):\n"
    "    global_ttl = ttl\n"
    "    data = OrderedDict()\n"
    "    # ... existing code ...\n"
    "    max_size = 1000\n"
)
# Note: marker at indent 4 here matches function-body indent (since there
# is no outer class) — preserved gap keeps its 4-space indent unchanged.


def test_top_level_function_replace_still_works(tmp_path):
    """Top-level function — same scenario must also pass. No regression on
    the behavior 0.2.2 delivered for plain functions."""
    file_path = tmp_path / "setup.py"
    file_path.write_text(PY_TOP_LEVEL)

    result = chunked_merge(
        original_code=file_path.read_text(),
        snippet=PY_TOP_LEVEL_SNIPPET,
        file_path=str(file_path),
        merge_fn=_no_model,
        language="python",
        replace="setup",
    )
    merged = result.merged_code

    assert "data = {}" not in merged
    assert "data = OrderedDict()" in merged
    assert merged.count("data = OrderedDict()") == 1
    assert "last_access = {}" in merged
    assert "max_size = 1000" in merged
    ast.parse(merged)
    assert result.model_tokens == 0


PY_MULTI_LINE = """\
class Cache:
    def __init__(self, capacity=128):
        self.capacity = capacity
        self.ttl = 60
        self.storage = {}
        self.stats = {"hits": 0, "misses": 0}
        self.logger = None
"""

# Multi-line change: replace two assignments in the middle of __init__.
PY_MULTI_LINE_SNIPPET = (
    "    def __init__(self, capacity=128):\n"
    "        self.capacity = capacity\n"
    "        self.ttl = 120\n"
    "        self.storage = OrderedDict()\n"
    "        # ... existing code ...\n"
    "        self.logger = None\n"
)


def test_class_method_with_multi_line_change(tmp_path):
    """Multiple new lines before the marker, each replacing a gap line."""
    file_path = tmp_path / "cache.py"
    file_path.write_text(PY_MULTI_LINE)

    result = chunked_merge(
        original_code=file_path.read_text(),
        snippet=PY_MULTI_LINE_SNIPPET,
        file_path=str(file_path),
        merge_fn=_no_model,
        language="python",
        replace="__init__",
    )
    merged = result.merged_code

    # Both replacements applied.
    assert "self.ttl = 120" in merged
    assert "self.ttl = 60" not in merged
    assert "self.storage = OrderedDict()" in merged
    assert "self.storage = {}" not in merged

    # Non-replaced gap lines preserved.
    assert 'self.stats = {"hits": 0, "misses": 0}' in merged
    # Anchor preserved.
    assert "self.logger = None" in merged

    # Exactly once each.
    assert merged.count("self.ttl = ") == 1
    assert merged.count("self.storage = ") == 1

    ast.parse(merged)
    assert result.model_tokens == 0


# ---------------------------------------------------------------------------
# Cross-language: Rust impl method. Same shape of corruption applies to any
# language with ``name = value`` binding lines in a marker-preserved gap.
# ---------------------------------------------------------------------------

RUST_ORIGINAL = """\
pub struct SessionStore {
    ttl: u64,
    last_access: HashMap<String, Instant>,
    data: HashMap<String, String>,
    max_size: usize,
}

impl SessionStore {
    pub fn new(ttl: u64) -> Self {
        let ttl = ttl;
        let last_access = HashMap::new();
        let data = HashMap::new();
        let max_size = 1000;
        Self { ttl, last_access, data, max_size }
    }
}
"""

RUST_METHOD_SNIPPET = (
    "    pub fn new(ttl: u64) -> Self {\n"
    "        let ttl = ttl;\n"
    "        let data = BTreeMap::new();\n"
    "        // ... existing code ...\n"
    "        Self { ttl, last_access, data, max_size }\n"
    "    }\n"
)


def test_rust_method_partial_replace(tmp_path):
    """Rust equivalent: ``let data = ...`` replaced inside an impl method.

    Before fix: both ``let data = HashMap::new();`` and
    ``let data = BTreeMap::new();`` ended up in the output.
    """
    file_path = tmp_path / "session.rs"
    file_path.write_text(RUST_ORIGINAL)

    result = chunked_merge(
        original_code=file_path.read_text(),
        snippet=RUST_METHOD_SNIPPET,
        file_path=str(file_path),
        merge_fn=_no_model,
        language="rust",
        replace="new",
    )
    merged = result.merged_code

    assert "let data = HashMap::new();" not in merged, (
        f"old binding retained alongside new — corruption!\n{merged}"
    )
    assert "let data = BTreeMap::new();" in merged
    assert merged.count("let data = ") == 1

    # Non-replaced gap lines preserved.
    assert "let last_access = HashMap::new();" in merged
    assert "let max_size = 1000;" in merged
    # Anchors intact.
    assert "let ttl = ttl;" in merged
    assert "Self { ttl, last_access, data, max_size }" in merged

    assert result.model_tokens == 0
