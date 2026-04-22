"""Tests for `preserve_siblings=True` on chunked_merge's `replace=` path.

When a caller wants to edit a subset of a class's members (change one field,
add one method, etc.) without enumerating the siblings they want to keep,
`preserve_siblings=True` lets the snippet contain only the class shell plus
the members being changed/added. Any named sibling present in the original
class but missing from the snippet is carried over verbatim.

Empirically verified byte-exact with zero model tokens on 4 fasteditBench
class_method fixtures (java/09, kotlin/09, swift/06, typescript/09) in the
M10 prototype. This module locks in that behavior.
"""

from __future__ import annotations

import pytest

from fastedit.inference.ast_utils import BatchEdit
from fastedit.inference.chunked_merge import chunked_merge
from fastedit.inference.symbols import batch_chunked_merge


def _no_model(*_args, **_kwargs):
    """Merge fn that fails loudly if called — preserve_siblings must be pure AST."""
    raise AssertionError(
        "merge_fn must NOT be called — preserve_siblings is a zero-token path"
    )


# ---------------------------------------------------------------------------
# 1. Kotlin: field-type change, 3 methods preserved
# ---------------------------------------------------------------------------

KT_ORIGINAL = """\
class Store {
    private val items: MutableList<String> = mutableListOf()

    fun add(item: String) {
        items.add(item)
    }

    fun get(index: Int): String {
        return items[index]
    }

    fun size(): Int {
        return items.size
    }
}
"""

KT_SNIPPET = """\
class Store {
    private val items: MutableMap<Int, String> = mutableMapOf()

    // ... existing code ...
}
"""

KT_EXPECTED = """\
class Store {
    private val items: MutableMap<Int, String> = mutableMapOf()

    fun add(item: String) {
        items.add(item)
    }

    fun get(index: Int): String {
        return items[index]
    }

    fun size(): Int {
        return items.size
    }
}
"""


def test_preserve_siblings_kotlin_field_change(tmp_path):
    """Kotlin data-class-shaped edit: snippet lists only the class shell and
    the changed field; `add`/`get`/`size` are preserved verbatim."""
    file_path = tmp_path / "Store.kt"
    file_path.write_text(KT_ORIGINAL)

    result = chunked_merge(
        original_code=KT_ORIGINAL,
        snippet=KT_SNIPPET,
        file_path=str(file_path),
        merge_fn=_no_model,
        language="kotlin",
        replace="Store",
        preserve_siblings=True,
    )

    assert result.model_tokens == 0
    assert result.chunks_used == 0
    assert result.merged_code == KT_EXPECTED


# ---------------------------------------------------------------------------
# 2. Java: new field + new constructor body, two methods preserved
# ---------------------------------------------------------------------------

JAVA_ORIGINAL = """\
public class Store {
    private java.util.List<String> items;

    public Store() {
        this.items = new java.util.ArrayList<>();
    }

    public String get(int k) {
        return this.items.get(k);
    }

    public void set(int k, String v) {
        this.items.set(k, v);
    }
}
"""

JAVA_SNIPPET = """\
public class Store {
    private java.util.Map<Integer, String> items;

    public Store() {
        this.items = new java.util.HashMap<>();
    }
    // ... existing code ...
}
"""

JAVA_EXPECTED = """\
public class Store {
    private java.util.Map<Integer, String> items;

    public Store() {
        this.items = new java.util.HashMap<>();
    }

    public String get(int k) {
        return this.items.get(k);
    }

    public void set(int k, String v) {
        this.items.set(k, v);
    }
}
"""


def test_preserve_siblings_java_field_and_ctor(tmp_path):
    """Java: change field type + constructor body, preserve get/set methods."""
    file_path = tmp_path / "Store.java"
    file_path.write_text(JAVA_ORIGINAL)

    result = chunked_merge(
        original_code=JAVA_ORIGINAL,
        snippet=JAVA_SNIPPET,
        file_path=str(file_path),
        merge_fn=_no_model,
        language="java",
        replace="Store",
        preserve_siblings=True,
    )

    assert result.model_tokens == 0
    assert result.chunks_used == 0
    assert result.merged_code == JAVA_EXPECTED


# ---------------------------------------------------------------------------
# 3. Swift: stored-property + init body changed, other members preserved
# ---------------------------------------------------------------------------

SWIFT_ORIGINAL = """\
class Store {
    private var items: [Int]

    init() {
        self.items = []
    }

    func get(_ k: Int) -> Int {
        return items[k]
    }

    func set(_ k: Int, _ v: Int) {
        items[k] = v
    }
}
"""

SWIFT_SNIPPET = """\
class Store {
    private var items: [String: Int]

    init() {
        self.items = [:]
    }
    // ... existing code ...
}
"""

SWIFT_EXPECTED = """\
class Store {
    private var items: [String: Int]

    init() {
        self.items = [:]
    }

    func get(_ k: Int) -> Int {
        return items[k]
    }

    func set(_ k: Int, _ v: Int) {
        items[k] = v
    }
}
"""


def test_preserve_siblings_swift_stored_prop_and_init(tmp_path):
    file_path = tmp_path / "Store.swift"
    file_path.write_text(SWIFT_ORIGINAL)

    result = chunked_merge(
        original_code=SWIFT_ORIGINAL,
        snippet=SWIFT_SNIPPET,
        file_path=str(file_path),
        merge_fn=_no_model,
        language="swift",
        replace="Store",
        preserve_siblings=True,
    )

    assert result.model_tokens == 0
    assert result.chunks_used == 0
    assert result.merged_code == SWIFT_EXPECTED


# ---------------------------------------------------------------------------
# 4. TypeScript: generic class field initializer change, methods preserved
# ---------------------------------------------------------------------------

TS_ORIGINAL = """\
export class Store<K, V> {
  private items: K[] = [];

  get(k: number): K {
    return this.items[k];
  }

  set(k: number, v: K): void {
    this.items[k] = v;
  }
}
"""

TS_SNIPPET = """\
export class Store<K, V> {
  private items: Map<K, V> = new Map();

  // ... existing code ...
}
"""

TS_EXPECTED = """\
export class Store<K, V> {
  private items: Map<K, V> = new Map();

  get(k: number): K {
    return this.items[k];
  }

  set(k: number, v: K): void {
    this.items[k] = v;
  }
}
"""


def test_preserve_siblings_typescript_generic(tmp_path):
    file_path = tmp_path / "store.ts"
    file_path.write_text(TS_ORIGINAL)

    result = chunked_merge(
        original_code=TS_ORIGINAL,
        snippet=TS_SNIPPET,
        file_path=str(file_path),
        merge_fn=_no_model,
        language="typescript",
        replace="Store",
        preserve_siblings=True,
    )

    assert result.model_tokens == 0
    assert result.chunks_used == 0
    assert result.merged_code == TS_EXPECTED


# ---------------------------------------------------------------------------
# 5. Regression: preserve_siblings=False (default) still enforces extras check
# ---------------------------------------------------------------------------

def test_extras_check_fires_without_preserve_siblings(tmp_path):
    """Without preserve_siblings=True, a snippet that names additional
    top-level symbols under `replace=Store` must still raise ValueError."""
    file_path = tmp_path / "Store.java"
    file_path.write_text(JAVA_ORIGINAL)

    # Snippet that enumerates the whole class — names `get`/`set` in
    # addition to `Store`. Without the flag, this is the silent-deletion
    # case we guard against.
    snippet_with_extras = """\
public class Store {
    private java.util.Map<Integer, String> items;

    public Store() {
        this.items = new java.util.HashMap<>();
    }

    public String get(int k) {
        return this.items.get(k);
    }

    public void set(int k, String v) {
        this.items.set(k, v);
    }
}
"""

    with pytest.raises(ValueError, match=r"additional symbol"):
        chunked_merge(
            original_code=JAVA_ORIGINAL,
            snippet=snippet_with_extras,
            file_path=str(file_path),
            merge_fn=_no_model,
            language="java",
            replace="Store",
        )


# ---------------------------------------------------------------------------
# 6. preserve_siblings=True with a snippet that lists every sibling
#    behaves like a full replace (no extras carried over because none
#    are missing from the snippet). Merged output must match snippet verbatim.
# ---------------------------------------------------------------------------

def test_preserve_siblings_when_snippet_names_everything(tmp_path):
    """If the snippet includes every original sibling, preserve_siblings
    has no siblings to add — merged output is the snippet's class verbatim,
    still zero model tokens."""
    file_path = tmp_path / "Store.java"
    file_path.write_text(JAVA_ORIGINAL)

    complete_snippet = """\
public class Store {
    private java.util.Map<Integer, String> items;

    public Store() {
        this.items = new java.util.HashMap<>();
    }

    public String get(int k) {
        return this.items.get(k);
    }

    public void set(int k, String v) {
        this.items.set(k, v);
    }
}
"""

    result = chunked_merge(
        original_code=JAVA_ORIGINAL,
        snippet=complete_snippet,
        file_path=str(file_path),
        merge_fn=_no_model,
        language="java",
        replace="Store",
        preserve_siblings=True,
    )

    assert result.model_tokens == 0
    assert result.chunks_used == 0
    assert result.merged_code == complete_snippet


# ---------------------------------------------------------------------------
# 7. Edge: preserve_siblings=True without replace= → ValueError
# ---------------------------------------------------------------------------

def test_preserve_siblings_without_replace_errors(tmp_path):
    """`preserve_siblings` is only meaningful with `replace=`. Using it
    alone (or with `after=`) should raise a clear error."""
    file_path = tmp_path / "Store.java"
    file_path.write_text(JAVA_ORIGINAL)

    with pytest.raises(ValueError, match=r"preserve_siblings.*replace"):
        chunked_merge(
            original_code=JAVA_ORIGINAL,
            snippet=JAVA_SNIPPET,
            file_path=str(file_path),
            merge_fn=_no_model,
            language="java",
            preserve_siblings=True,
        )

    with pytest.raises(ValueError, match=r"preserve_siblings.*replace"):
        chunked_merge(
            original_code=JAVA_ORIGINAL,
            snippet="public void foo() {}\n",
            file_path=str(file_path),
            merge_fn=_no_model,
            language="java",
            after="Store",
            preserve_siblings=True,
        )


# ---------------------------------------------------------------------------
# 8. BatchEdit: preserve_siblings=True on one edit, False on another
# ---------------------------------------------------------------------------

BATCH_ORIGINAL = """\
public class Store {
    private java.util.List<String> items;

    public Store() {
        this.items = new java.util.ArrayList<>();
    }

    public String get(int k) {
        return this.items.get(k);
    }

    public void set(int k, String v) {
        this.items.set(k, v);
    }
}

public class Cache {
    private int size = 10;
}
"""


def test_batch_edit_mixed_preserve_siblings(tmp_path):
    """First edit: `replace=Store` with preserve_siblings=True — narrow edit
    to Store's field + ctor, preserving get/set. Second edit: plain
    `replace=Cache` full replacement (no preserve_siblings needed)."""
    file_path = tmp_path / "Multi.java"
    file_path.write_text(BATCH_ORIGINAL)

    edit1 = BatchEdit(
        snippet=JAVA_SNIPPET,
        replace="Store",
        preserve_siblings=True,
    )
    edit2 = BatchEdit(
        snippet="public class Cache {\n    private int size = 100;\n}\n",
        replace="Cache",
    )

    result = batch_chunked_merge(
        original_code=BATCH_ORIGINAL,
        edits=[edit1, edit2],
        file_path=str(file_path),
        merge_fn=_no_model,
        language="java",
    )

    assert result.model_tokens == 0
    expected = """\
public class Store {
    private java.util.Map<Integer, String> items;

    public Store() {
        this.items = new java.util.HashMap<>();
    }

    public String get(int k) {
        return this.items.get(k);
    }

    public void set(int k, String v) {
        this.items.set(k, v);
    }
}

public class Cache {
    private int size = 100;
}
"""
    assert result.merged_code == expected


# ---------------------------------------------------------------------------
# 9. Parse-valid guard: result.parse_valid reflects actual parse state.
#     Under normal preserve_siblings use the output is parse-valid;
#     this test just locks in that the field is populated (not always True
#     by default) when language is passed.
# ---------------------------------------------------------------------------

def test_preserve_siblings_reports_parse_valid(tmp_path):
    file_path = tmp_path / "Store.ts"
    file_path.write_text(TS_ORIGINAL)

    result = chunked_merge(
        original_code=TS_ORIGINAL,
        snippet=TS_SNIPPET,
        file_path=str(file_path),
        merge_fn=_no_model,
        language="typescript",
        replace="Store",
        preserve_siblings=True,
    )
    assert result.parse_valid is True


# ---------------------------------------------------------------------------
# 10. Whitespace: blank lines between preserved siblings in the original
#     are retained in the merged output.
# ---------------------------------------------------------------------------

def test_preserve_siblings_retains_blank_lines_between_siblings(tmp_path):
    """Original has blank lines between each method. Snippet changes only
    the field. Preserved siblings must include the blank-line separators
    so get/set aren't jammed together."""
    file_path = tmp_path / "Store.kt"
    file_path.write_text(KT_ORIGINAL)

    result = chunked_merge(
        original_code=KT_ORIGINAL,
        snippet=KT_SNIPPET,
        file_path=str(file_path),
        merge_fn=_no_model,
        language="kotlin",
        replace="Store",
        preserve_siblings=True,
    )

    # Between `items.add(item)\n    }` and `fun get(...)` there must be
    # a blank line; same between `get` and `size`.
    out = result.merged_code
    # There must be exactly 4 blank lines inside the class:
    # after field, after add, after get, before close brace? Count carefully.
    # Expected structure matches KT_EXPECTED.
    assert out == KT_EXPECTED
    # Redundant explicit checks on blank-line presence:
    assert "    }\n\n    fun get(" in out
    assert "    }\n\n    fun size(" in out
