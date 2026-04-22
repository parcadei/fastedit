"""Regression test for stale-AST corruption on chained fast_edit calls.

The bug: when `fast_edit` is called multiple times in sequence on the same
file, the `chunked_merge` replace=/after= fast paths call `get_ast_map()`,
which shells out to `tldr structure`. `tldr` runs a persistent daemon with
a salsa cache. After the (N-1)th edit writes to disk, if the Nth call runs
before the daemon's file-watcher invalidates, tldr returns STALE line
numbers from before the last edit. FastEdit then splices the new edit at
wrong coordinates -> leftover lines from the old function body.

The fix: parse `original_code` in-memory with tree-sitter directly. The
in-memory source is always authoritative; no daemon, no cache staleness.

This test replays the exact 3-call sequence that was observed to corrupt
a Python file in the wild and asserts the output is clean.
"""

from __future__ import annotations

import ast
import textwrap

from fastedit.inference.chunked_merge import chunked_merge


def _no_model(*_args, **_kwargs):
    """Merge fn that fails loudly if the fast path falls through to the model."""
    raise AssertionError(
        "merge_fn must NOT be called — these edits should hit the "
        "deterministic AST fast path (replace=/after= with no markers)"
    )


PY_STARTING = """\
def retry_batch(client, records, max_attempts=3):
    attempt = 0
    while attempt < max_attempts:
        try:
            return client.send(records)
        except TransientError:
            attempt += 1
    raise RuntimeError("exhausted retries")


def fetch_user(db, user_id):
    row = db.query("SELECT * FROM users WHERE id = ?", user_id)
    if row is None:
        return None
    return User(id=row[0], name=row[1], email=row[2])
"""


# Step 1: modify retry_batch body — add a guard. Replace via deterministic
# text-match with context markers.
PY_STEP1_SNIPPET = """\
def retry_batch(client, records, max_attempts=3):
    if not records:
        return []
    attempt = 0
    # ... existing code ...
"""

# Step 2: insert a new helper after retry_batch — pure AST splice.
PY_STEP2_SNIPPET = """\
def log_attempt(attempt, max_attempts):
    print(f"attempt {attempt}/{max_attempts}")
"""

# Step 3: replace fetch_user entirely — this is where the stale AST bug
# would splice at shifted-but-stale coordinates and leave leftover lines.
PY_STEP3_SNIPPET = """\
def fetch_user(db, user_id):
    row = db.query("SELECT * FROM users WHERE id = ? AND active = 1", user_id)
    if row is None:
        return None
    return User(id=row[0], name=row[1], email=row[2], active=True)
"""


def test_chained_python_edits_no_stale_ast_corruption(tmp_path):
    """Three chained edits on the same Python file must produce clean output.

    After all three steps, the file must contain exactly one `fetch_user`,
    with the new body (`AND active = 1`), no leftover lines from the old
    body, and must be syntactically valid Python.
    """
    file_path = tmp_path / "file3.py"
    file_path.write_text(PY_STARTING)

    # Step 1: replace retry_batch with a guard addition (marker-driven
    # deterministic text-match).
    result1 = chunked_merge(
        original_code=file_path.read_text(),
        snippet=PY_STEP1_SNIPPET,
        file_path=str(file_path),
        merge_fn=_no_model,
        language="python",
        replace="retry_batch",
    )
    assert result1.parse_valid, "step 1 produced unparseable Python"
    file_path.write_text(result1.merged_code)

    # Step 2: insert log_attempt after retry_batch (pure insertion).
    result2 = chunked_merge(
        original_code=file_path.read_text(),
        snippet=PY_STEP2_SNIPPET,
        file_path=str(file_path),
        merge_fn=_no_model,
        language="python",
        after="retry_batch",
    )
    assert result2.parse_valid, "step 2 produced unparseable Python"
    file_path.write_text(result2.merged_code)

    # Step 3: replace fetch_user entirely (direct-swap fast path). This is
    # the step that corrupts the file if the AST lookup returns stale
    # pre-step-1/2 line numbers.
    result3 = chunked_merge(
        original_code=file_path.read_text(),
        snippet=PY_STEP3_SNIPPET,
        file_path=str(file_path),
        merge_fn=_no_model,
        language="python",
        replace="fetch_user",
    )
    assert result3.parse_valid, "step 3 produced unparseable Python"
    file_path.write_text(result3.merged_code)

    final = file_path.read_text()

    # 1. Must parse as valid Python.
    ast.parse(final)

    # 2. `fetch_user` appears exactly once.
    assert final.count("def fetch_user") == 1, (
        f"fetch_user appears {final.count('def fetch_user')} times:\n{final}"
    )

    # 3. The old body (without `active=True`) must not appear.
    assert "email=row[2])" not in final, (
        f"leftover old body detected:\n{final}"
    )
    # New body must be present.
    assert "AND active = 1" in final, f"new body missing:\n{final}"
    assert "active=True" in final, f"new body missing:\n{final}"

    # 4. log_attempt helper from step 2 must be there.
    assert "def log_attempt" in final, f"log_attempt missing:\n{final}"

    # 5. retry_batch guard from step 1 must be there.
    assert "if not records:" in final, f"step-1 guard missing:\n{final}"

    # 6. Line count in expected window — guards against runaway leftover
    # lines. Exactly 20 lines of content + trailing newline.
    line_count = final.count("\n")
    assert 18 <= line_count <= 24, (
        f"unexpected final line count {line_count}:\n{final}"
    )


RS_STARTING = """\
fn retry_batch(client: &Client, records: Vec<Record>, max_attempts: u32) -> Vec<Response> {
    let mut attempt = 0;
    while attempt < max_attempts {
        match client.send(&records) {
            Ok(r) => return r,
            Err(_) => attempt += 1,
        }
    }
    panic!("exhausted retries")
}


fn fetch_user(db: &Db, user_id: u64) -> Option<User> {
    let row = db.query("SELECT * FROM users WHERE id = ?", user_id)?;
    Some(User { id: row[0], name: row[1].clone(), email: row[2].clone() })
}
"""

RS_STEP1_SNIPPET = """\
fn retry_batch(client: &Client, records: Vec<Record>, max_attempts: u32) -> Vec<Response> {
    if records.is_empty() {
        return Vec::new();
    }
    let mut attempt = 0;
    // ... existing code ...
}
"""

RS_STEP2_SNIPPET = """\
fn log_attempt(attempt: u32, max_attempts: u32) {
    println!("attempt {}/{}", attempt, max_attempts);
}
"""

RS_STEP3_SNIPPET = """\
fn fetch_user(db: &Db, user_id: u64) -> Option<User> {
    let row = db.query("SELECT * FROM users WHERE id = ? AND active = 1", user_id)?;
    Some(User { id: row[0], name: row[1].clone(), email: row[2].clone(), active: true })
}
"""


def test_chained_rust_edits_no_stale_ast_corruption(tmp_path):
    """Same chained-edit scenario in Rust — exercises the tree-sitter
    walker on a different grammar (function_item vs function_definition)."""
    file_path = tmp_path / "file3.rs"
    file_path.write_text(RS_STARTING)

    result1 = chunked_merge(
        original_code=file_path.read_text(),
        snippet=RS_STEP1_SNIPPET,
        file_path=str(file_path),
        merge_fn=_no_model,
        language="rust",
        replace="retry_batch",
    )
    assert result1.parse_valid, "rust step 1 produced unparseable output"
    file_path.write_text(result1.merged_code)

    result2 = chunked_merge(
        original_code=file_path.read_text(),
        snippet=RS_STEP2_SNIPPET,
        file_path=str(file_path),
        merge_fn=_no_model,
        language="rust",
        after="retry_batch",
    )
    assert result2.parse_valid, "rust step 2 produced unparseable output"
    file_path.write_text(result2.merged_code)

    result3 = chunked_merge(
        original_code=file_path.read_text(),
        snippet=RS_STEP3_SNIPPET,
        file_path=str(file_path),
        merge_fn=_no_model,
        language="rust",
        replace="fetch_user",
    )
    assert result3.parse_valid, "rust step 3 produced unparseable output"
    file_path.write_text(result3.merged_code)

    final = file_path.read_text()

    # Exactly one fetch_user definition.
    assert final.count("fn fetch_user") == 1, (
        f"fn fetch_user count: {final.count('fn fetch_user')}\n{final}"
    )
    # Old body gone.
    assert "email: row[2].clone() }" not in final, (
        f"leftover old Rust body:\n{final}"
    )
    # New body present.
    assert "AND active = 1" in final, f"new body missing:\n{final}"
    assert "active: true" in final, f"new body missing:\n{final}"
    # Step-2 insertion present.
    assert "fn log_attempt" in final, f"log_attempt missing:\n{final}"
    # Step-1 guard present.
    assert "records.is_empty()" in final, f"step-1 guard missing:\n{final}"


# Silence unused-import lint (textwrap is kept as a hint for future fixture helpers).
_ = textwrap


# ---------------------------------------------------------------------------
# Simulation test: force a stale AST lookup, prove in-memory parsing wins
# ---------------------------------------------------------------------------
#
# The real-world bug depends on a race between the tldr daemon's
# file-watcher and the next `fast_edit` call. That race is hard to
# reproduce deterministically on all machines. This test simulates the
# race by monkeypatching `get_ast_map` (the disk-reading path) to return
# the PRE-edit line numbers when called after the file has been modified.
#
# On broken code: `chunked_merge` calls `get_ast_map` -> gets stale coords
# -> splices at wrong range -> corruption.
#
# On fixed code: `chunked_merge` parses `original_code` in-memory via the
# new `get_ast_map_from_source` -> coords are always correct.

def test_replace_fast_path_ignores_stale_disk_ast(tmp_path, monkeypatch):
    """Simulate the tldr daemon race: stale disk AST, fresh in-memory source.

    After applying the fix, the `replace=` fast path must not consult the
    disk AST at all for the edit coordinates — it must derive them from
    the `original_code` argument.
    """
    file_path = tmp_path / "user.py"
    original = PY_STARTING
    file_path.write_text(original)

    # Capture the real AST (before any mutation).
    from fastedit.inference import ast_utils as au
    real_nodes = au.get_ast_map(str(file_path), len(original.splitlines()))
    assert real_nodes, "precondition: tldr must return nodes here"

    # Simulate prior edits having grown the file. We pass the GROWN source
    # as `original_code`, but leave the on-disk file untouched so the
    # "stale" disk AST points at the original coordinates.
    injection = "# injected prelude comment for line-shift test\n" * 8
    grown = injection + original
    # IMPORTANT: do NOT write `grown` to disk. The broken code path reads
    # disk; we want disk to return stale pre-injection coordinates.

    # Monkeypatch get_ast_map to return the PRE-injection nodes — this
    # emulates the tldr daemon's stale cache exactly.
    monkeypatch.setattr(
        "fastedit.inference.chunked_merge.get_ast_map",
        lambda *_a, **_kw: real_nodes,
    )

    # Now issue a replace= on fetch_user against the GROWN source.
    snippet = PY_STEP3_SNIPPET
    result = chunked_merge(
        original_code=grown,
        snippet=snippet,
        file_path=str(file_path),
        merge_fn=_no_model,
        language="python",
        replace="fetch_user",
    )

    # Must parse cleanly.
    ast.parse(result.merged_code)

    # Must still contain exactly one fetch_user — not two, not a corrupted
    # half-spliced mess.
    assert result.merged_code.count("def fetch_user") == 1

    # New body must be present, old body must be absent.
    assert "AND active = 1" in result.merged_code
    assert "active=True" in result.merged_code
    # The bare "email=row[2])" pattern (no active=True) must not appear as
    # leftover from the old body.
    leftover = [
        ln for ln in result.merged_code.splitlines()
        if ln.strip() == "return User(id=row[0], name=row[1], email=row[2])"
    ]
    assert not leftover, (
        f"leftover old-body line detected: {leftover}\n"
        f"full output:\n{result.merged_code}"
    )

    # Injection lines at the top must be preserved.
    assert result.merged_code.startswith(injection), (
        "prelude injection lost — edit spliced at wrong absolute offset"
    )
