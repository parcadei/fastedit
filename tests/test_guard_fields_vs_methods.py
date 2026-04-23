"""Guard invariant: the `replace=X` extras check distinguishes methods from fields.

When `replace="ClassName"` is used with a snippet that contains the full class
body, the guard at chunked_merge.py must:

  - PASS silently when the snippet contains only fields/variables (no silent-
    deletion risk — a full-class direct swap is the intended operation).
  - FIRE (ValueError) when the snippet contains methods/functions that are
    NOT the target class itself — those are a silent-deletion risk unless the
    caller uses `preserve_siblings=True`.

These tests pin both halves of the invariant across Python, Java, Rust, and
TypeScript so future changes to the guard can't regress one half in favor of
the other.
"""

from __future__ import annotations

import pytest

from fastedit.inference.chunked_merge import chunked_merge


def _no_model(*_args, **_kwargs):
    raise AssertionError("merge_fn must not be called on the field-only path")


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------

PY_ORIGINAL = '''\
class Cache:
    size = 10
'''

PY_FIELD_ONLY_SNIPPET = '''\
class Cache:
    size = 100
'''

PY_METHODS_SNIPPET = '''\
class Cache:
    def get(self, k):
        return None

    def set(self, k, v):
        pass
'''


def test_python_field_only_does_not_fire_guard(tmp_path):
    """Python: class with only a field value change must not raise."""
    file_path = tmp_path / "cache.py"
    file_path.write_text(PY_ORIGINAL)
    # Should not raise — this is a direct swap case.
    chunked_merge(
        original_code=PY_ORIGINAL,
        snippet=PY_FIELD_ONLY_SNIPPET,
        file_path=str(file_path),
        merge_fn=_no_model,
        language="python",
        replace="Cache",
    )


def test_python_methods_fires_guard(tmp_path):
    """Python: class with nested methods must raise ValueError."""
    file_path = tmp_path / "cache.py"
    file_path.write_text(PY_ORIGINAL)
    with pytest.raises(ValueError, match=r"additional symbol"):
        chunked_merge(
            original_code=PY_ORIGINAL,
            snippet=PY_METHODS_SNIPPET,
            file_path=str(file_path),
            merge_fn=_no_model,
            language="python",
            replace="Cache",
        )


# ---------------------------------------------------------------------------
# Java
# ---------------------------------------------------------------------------

JAVA_ORIGINAL = """\
public class Cache {
    private int size = 10;
}
"""

JAVA_FIELD_ONLY_SNIPPET = """\
public class Cache {
    private int size = 100;
}
"""

JAVA_METHODS_SNIPPET = """\
public class Cache {
    public int get() { return 0; }

    public void set(int v) {}
}
"""


def test_java_field_only_does_not_fire_guard(tmp_path):
    file_path = tmp_path / "Cache.java"
    file_path.write_text(JAVA_ORIGINAL)
    chunked_merge(
        original_code=JAVA_ORIGINAL,
        snippet=JAVA_FIELD_ONLY_SNIPPET,
        file_path=str(file_path),
        merge_fn=_no_model,
        language="java",
        replace="Cache",
    )


def test_java_methods_fires_guard(tmp_path):
    file_path = tmp_path / "Cache.java"
    file_path.write_text(JAVA_ORIGINAL)
    with pytest.raises(ValueError, match=r"additional symbol"):
        chunked_merge(
            original_code=JAVA_ORIGINAL,
            snippet=JAVA_METHODS_SNIPPET,
            file_path=str(file_path),
            merge_fn=_no_model,
            language="java",
            replace="Cache",
        )


# ---------------------------------------------------------------------------
# Rust
# ---------------------------------------------------------------------------

RUST_ORIGINAL = """\
struct Cache {
    size: i32,
}
"""

RUST_FIELD_ONLY_SNIPPET = """\
struct Cache {
    size: u64,
}
"""

RUST_METHODS_SNIPPET = """\
impl Cache {
    fn get(&self) -> i32 { 0 }
    fn set(&mut self, v: i32) {}
}
"""


def test_rust_field_only_does_not_fire_guard(tmp_path):
    file_path = tmp_path / "cache.rs"
    file_path.write_text(RUST_ORIGINAL)
    chunked_merge(
        original_code=RUST_ORIGINAL,
        snippet=RUST_FIELD_ONLY_SNIPPET,
        file_path=str(file_path),
        merge_fn=_no_model,
        language="rust",
        replace="Cache",
    )


# ---------------------------------------------------------------------------
# TypeScript
# ---------------------------------------------------------------------------

TS_ORIGINAL = """\
class Cache {
    size: number = 10;
}
"""

TS_FIELD_ONLY_SNIPPET = """\
class Cache {
    size: number = 100;
}
"""

TS_METHODS_SNIPPET = """\
class Cache {
    get(k: string): string { return ""; }

    set(k: string, v: string): void {}
}
"""


def test_typescript_field_only_does_not_fire_guard(tmp_path):
    file_path = tmp_path / "cache.ts"
    file_path.write_text(TS_ORIGINAL)
    chunked_merge(
        original_code=TS_ORIGINAL,
        snippet=TS_FIELD_ONLY_SNIPPET,
        file_path=str(file_path),
        merge_fn=_no_model,
        language="typescript",
        replace="Cache",
    )


def test_typescript_methods_fires_guard(tmp_path):
    file_path = tmp_path / "cache.ts"
    file_path.write_text(TS_ORIGINAL)
    with pytest.raises(ValueError, match=r"additional symbol"):
        chunked_merge(
            original_code=TS_ORIGINAL,
            snippet=TS_METHODS_SNIPPET,
            file_path=str(file_path),
            merge_fn=_no_model,
            language="typescript",
            replace="Cache",
        )
