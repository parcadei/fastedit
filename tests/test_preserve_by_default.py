"""Preserve-by-default regression suite (RC1/RC2/RC3/RC5 — RED first).

These tests define "fixed" for the destructive deterministic-editor semantics
documented in ``implementation_plan.md`` §1 (bugs B1, B2, B5, B9, B15, B32)
and §3 Step 1. Per CLAUDE.md (tests before implementation) every test here
asserts the CORRECT post-fix behavior and is expected to FAIL until the
preserve-by-default fixes land (Steps 2-5).

All tests drive the PUBLIC entry point :func:`chunked_merge` on its
deterministic path only (inputs with >=2 matching context anchor lines;
``merge_fn`` fails loudly if the model is ever invoked). No network, no
model download, no disk reads outside ``tmp_path``.

Governing principle (implementation_plan.md §2): a snippet declares an edit;
it does not license deletion. Unmentioned original lines survive.
"""

from __future__ import annotations

import pytest

from fastedit.inference.chunked_merge import ChunkedMergeResult, chunked_merge


def _no_model(*_args, **_kwargs):
    """Merge fn that fails loudly if the edit falls through to the model."""
    raise AssertionError(
        "merge_fn must NOT be called — this edit should hit the "
        "deterministic text-match fast path"
    )


def _indent_of(line: str) -> str:
    """Return the leading-whitespace prefix of a line, literally."""
    return line[: len(line) - len(line.lstrip())]


def _stripped_lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines()]


def _merge(tmp_path, name, original_code, snippet, replace, language="python"):
    """Run chunked_merge deterministically against a tmp file."""
    file_path = tmp_path / name
    file_path.write_text(original_code)
    return chunked_merge(
        original_code=original_code,
        snippet=snippet,
        file_path=str(file_path),
        merge_fn=_no_model,
        language=language,
        replace=replace,
    )


# ---------------------------------------------------------------------------
# B1 — trailing-suffix deletion
# ---------------------------------------------------------------------------


def test_replace_trailing_new_lines_preserve_original_suffix(tmp_path):
    """Snippet with trailing NEW lines and no keep-marker must not delete
    the original suffix after the last context anchor (B1,
    ``text_match.py`` trailing-section handling).

    The snippet adds ``log(a)`` after ``a = 1`` and says nothing about the
    rest of the body — under preserve-by-default the unmentioned lines
    (``b = 2``, ``c = 3``, ``d = 4``, ``return d``) must survive.
    """
    original = (
        "def foo():\n"
        "    a = 1\n"
        "    b = 2\n"
        "    c = 3\n"
        "    d = 4\n"
        "    return d\n"
    )
    snippet = (
        "def foo():\n"
        "    a = 1\n"
        "    log(a)\n"
    )

    result = _merge(tmp_path, "foo.py", original, snippet, replace="foo")

    assert result.model_tokens == 0
    lines = _stripped_lines(result.merged_code)
    # The new line was added...
    assert "log(a)" in lines
    # ...and the unmentioned original suffix survived.
    assert "b = 2" in lines
    assert "c = 3" in lines
    assert "d = 4" in lines
    assert "return d" in lines


# ---------------------------------------------------------------------------
# B2 — mid-gap / sibling deletion (documented "edit in the MIDDLE" pattern:
# replace='func', snippet='<anchor>\n<new_lines>\n#...\n', tools_edit.py:46)
# ---------------------------------------------------------------------------


def test_replace_mid_gap_lines_preserved(tmp_path):
    """The documented middle-edit snippet shape must preserve sibling lines.

    ``data = audit(data)`` is added after the ``validate`` anchor and a
    keep-marker at column 0 says "keep the rest". The unmentioned sibling
    ``data = clean(data)`` must survive, and ``return data`` must keep its
    4-space body indent.
    """
    original = (
        "def process(data):\n"
        "    data = validate(data)\n"
        "    data = clean(data)\n"
        "    return data\n"
    )
    snippet = (
        "def process(data):\n"
        "    data = validate(data)\n"
        "    data = audit(data)\n"
        "# ...\n"
    )

    result = _merge(tmp_path, "process.py", original, snippet, replace="process")

    assert result.model_tokens == 0
    lines = result.merged_code.splitlines()
    # The sibling pipeline step survives...
    assert "    data = clean(data)" in lines
    # ...the tail survives at its original indent...
    assert "    return data" in lines
    # ...and the new step was added.
    assert "    data = audit(data)" in lines


# ---------------------------------------------------------------------------
# B5 — marker at column 0 computes a negative indent delta and dedents
# preserved tail lines out of their block
# ---------------------------------------------------------------------------


def test_marker_at_column_zero_does_not_dedent_preserved_lines(tmp_path):
    """A keep-marker at column 0 must not dedent preserved original lines.

    Same middle-edit shape as the sibling test but with an insertion line
    that shares no assignment LHS with the body, so every preserved
    original body line is unambiguous. Each preserved line must keep its
    exact original leading whitespace (compared literally).
    """
    original = (
        "def handler(payload):\n"
        "    payload = parse(payload)\n"
        "    payload = enrich(payload)\n"
        "    return payload\n"
    )
    snippet = (
        "def handler(payload):\n"
        "    payload = parse(payload)\n"
        "    log(payload)\n"
        "# ...\n"
    )

    result = _merge(tmp_path, "handler.py", original, snippet, replace="handler")

    assert result.model_tokens == 0
    merged_lines = result.merged_code.splitlines()
    original_body = [ln for ln in original.splitlines() if ln.strip()]

    for orig_line in original_body:
        content = orig_line.strip()
        orig_indent = _indent_of(orig_line)
        matches = [ln for ln in merged_lines if ln.strip() == content]
        assert matches, f"original line dropped from output: {orig_line!r}"
        for out_line in matches:
            assert _indent_of(out_line) == orig_indent, (
                f"preserved line {content!r} changed indent "
                f"{orig_indent!r} -> {_indent_of(out_line)!r}"
            )

    # The new line was added at body indent.
    assert "    log(payload)" in merged_lines


# ---------------------------------------------------------------------------
# B15 — `_is_marker` substring matching swallows real comments
# ---------------------------------------------------------------------------


def test_comment_containing_marker_phrase_is_not_a_marker(tmp_path):
    """A code line whose COMMENT merely contains the marker phrase is real
    content, not a keep-marker (B15, ``_is_marker`` substring matching).

    ``x = compute(a)  # ... see docs for details`` must be inserted verbatim
    and must not suppress/alter the handling of the surrounding lines.
    """
    original = (
        "def run(a):\n"
        "    a = compute(a)\n"
        "    a = log_value(a)\n"
        "    return a\n"
    )
    comment_line = "    x = compute(a)  # ... see docs for details"
    snippet = (
        "def run(a):\n"
        "    a = compute(a)\n"
        f"{comment_line}\n"
    )

    result = _merge(tmp_path, "run.py", original, snippet, replace="run")

    assert result.model_tokens == 0
    lines = result.merged_code.splitlines()
    # The comment line appears verbatim — it was not swallowed as a marker.
    assert comment_line in lines
    # And no original line was dropped as a result of the misclassification.
    assert "    a = compute(a)" in lines
    assert "    a = log_value(a)" in lines
    assert "    return a" in lines


# ---------------------------------------------------------------------------
# B9 — auto-prepended multi-line signatures duplicate fragments
# ---------------------------------------------------------------------------


def test_multiline_signature_replace_no_duplicate_close_paren(tmp_path):
    """Replacing into a multi-line signature must not duplicate closers.

    ``chunked_merge`` auto-prepends the target's signature when the snippet
    omits it (B9). The signature continuation lines (``data,``,
    ``options=None,``) and the ``):`` closer must appear exactly once each,
    and the merged result must parse.

    Two snippet shapes are covered:

    * no marker — the whole body restated with a guard added;
    * the documented middle-edit shape (``<anchor>\n<new>\n#...\n``) —
      today this shape emits the auto-prepended ``):`` a second time
      (fragment duplication) and reports a parse failure.
    """
    original = (
        "def process(\n"
        "    data,\n"
        "    options=None,\n"
        "):\n"
        "    data = clean(data)\n"
        "    return data\n"
    )
    snippets = {
        "no-marker": (
            "    if not data:\n"
            "        return None\n"
            "    data = clean(data)\n"
            "    return data\n"
        ),
        "middle-edit": (
            "    if not data:\n"
            "        return None\n"
            "    # ...\n"
            "    return data\n"
        ),
    }

    for name, snippet in snippets.items():
        result = _merge(
            tmp_path, f"process_{name}.py", original, snippet, replace="process"
        )

        assert result.model_tokens == 0, name
        stripped = _stripped_lines(result.merged_code)
        # Exactly one closer — no duplicated signature fragments.
        assert stripped.count("):") == 1, (
            f"[{name}] expected exactly one '):' line, got "
            f"{stripped.count('):')}: {result.merged_code!r}"
        )
        # Signature continuation lines appear exactly once.
        assert stripped.count("data,") == 1, name
        assert stripped.count("options=None,") == 1, name
        # The body edit is present and the output parses as Python.
        assert "if not data:" in stripped, name
        assert "data = clean(data)" in stripped, name
        assert "return data" in stripped, name
        assert result.parse_valid is True, (
            f"[{name}] merged output is parse-invalid:\n{result.merged_code}"
        )


# ---------------------------------------------------------------------------
# B9 (Step 5) — brace-on-signature grammars: the auto-prepended signature
# span must include the body-opening brace, and the prepended lines must be
# pinned as fixed context so they are never re-emitted as "new" content
# ---------------------------------------------------------------------------


def test_braceless_signature_replace_deterministic(tmp_path):
    """Rust-style replace with a body-only snippet must be deterministic.

    ``pub fn process(\\n    data: u32,\\n) -> u32 {`` places the body
    opener on the signature's own line. The auto-prepended signature span
    must therefore end AFTER the ``{`` (a brace-less prepend yields a
    brace-less splice the structural-balance gate must decline), and the
    prepended continuation lines must be pinned context — never
    re-emitted at column 0 as "new" lines.

    A body-only snippet with a keep-marker must merge with 0 model
    tokens, the signature emitted exactly once including ``-> u32 {``,
    exactly one closing brace, and a valid parse.
    """
    original = (
        "pub fn process(\n"
        "    data: u32,\n"
        ") -> u32 {\n"
        "    let total = data + 1;\n"
        "    total\n"
        "}\n"
    )
    snippet = (
        "    let doubled = data * 2;\n"
        "    // ...\n"
    )

    result = _merge(
        tmp_path, "process.rs", original, snippet,
        replace="process", language="rust",
    )

    assert result.model_tokens == 0, result.merged_code
    lines = _stripped_lines(result.merged_code)
    # Signature emitted exactly once — opener brace included.
    assert lines.count("pub fn process(") == 1, result.merged_code
    assert lines.count("data: u32,") == 1, result.merged_code
    assert lines.count(") -> u32 {") == 1, result.merged_code
    # Exactly one closing brace — no duplicated closers.
    assert lines.count("}") == 1, result.merged_code
    # The edit landed and the marker-preserved body survived.
    assert "let doubled = data * 2;" in lines, result.merged_code
    assert "let total = data + 1;" in lines, result.merged_code
    assert result.parse_valid is True, (
        f"merged output is parse-invalid:\n{result.merged_code}"
    )


# ---------------------------------------------------------------------------
# Result surface — callers can refuse to write parse-invalid output
# ---------------------------------------------------------------------------


def test_result_surface_reports_parse_valid(tmp_path):
    """``ChunkedMergeResult`` must expose ``parse_valid`` so callers can
    refuse to persist parse-invalid merges (surface contract for B10).

    True case: a deterministic merge whose output is valid Python reports
    ``parse_valid=True``. False case (contrived): a result carrying
    ``parse_valid=False`` must be refusable by a caller honoring the field.
    """
    original = (
        "def process(\n"
        "    data,\n"
        "    options=None,\n"
        "):\n"
        "    data = clean(data)\n"
        "    return data\n"
    )
    snippet = (
        "    if not data:\n"
        "        return None\n"
        "    data = clean(data)\n"
        "    return data\n"
    )

    result = _merge(tmp_path, "process.py", original, snippet, replace="process")

    assert result.model_tokens == 0
    # True case: the real merge reports parse validity.
    assert result.parse_valid is True

    def caller_may_write(merge_result: ChunkedMergeResult) -> bool:
        """The write-guard every caller builds on the public result fields."""
        return bool(merge_result.parse_valid)

    # False case: a parse-invalid result is refusable via the same field.
    rejected = ChunkedMergeResult(
        merged_code="def broken(:\n",
        parse_valid=False,
        chunks_used=0,
        chunk_regions=[],
        model_tokens=0,
        latency_ms=0.0,
    )
    assert caller_may_write(result) is True
    assert caller_may_write(rejected) is False


# ---------------------------------------------------------------------------
# B32 — whitespace-only original lines vanish
# ---------------------------------------------------------------------------


def test_whitespace_only_original_lines_preserved(tmp_path):
    """A whitespace-only original line must survive byte-exact.

    The body contains a blank line with trailing spaces. The snippet edits
    elsewhere (adds ``log(a)``) and never mentions that line — preserving
    it verbatim (not as an empty string) is required (B32).
    """
    whitespace_only_line = "    "
    original = (
        "def foo():\n"
        "    a = 1\n"
        f"{whitespace_only_line}\n"
        "    b = 2\n"
        "    return b\n"
    )
    snippet = (
        "def foo():\n"
        "    a = 1\n"
        "    log(a)\n"
        "    b = 2\n"
        "    return b\n"
    )

    result = _merge(tmp_path, "foo.py", original, snippet, replace="foo")

    assert result.model_tokens == 0
    raw_lines = result.merged_code.split("\n")
    # Byte-exact survival: the whitespace-only line is still four spaces,
    # not an empty string, and not duplicated.
    assert raw_lines.count(whitespace_only_line) == 1, (
        f"whitespace-only line {whitespace_only_line!r} not preserved "
        f"byte-exact: {result.merged_code!r}"
    )
    lines = _stripped_lines(result.merged_code)
    assert "log(a)" in lines
    assert "b = 2" in lines
    assert "return b" in lines


# ---------------------------------------------------------------------------
# B6 — `_reindent_new_lines` bases the group indent on the MINIMUM indent
# across the new lines, so one low-indent line over-indents the siblings
# ---------------------------------------------------------------------------


def test_flush_left_new_line_does_not_overindent_siblings(tmp_path):
    """A flush-left line among the new lines must not drag the group's
    base to column 0 and over-indent the sibling code lines (B6,
    ``_reindent_new_lines``).

    The snippet inserts two body-indent calls around a flush-left
    comment (the string-content/comment shape users paste verbatim).
    The reference anchor is the signature (marker-position insertion),
    and the code lines must land at exactly the anchor's body indent —
    not double-indented — with the merged output parsing as Python.
    """
    original = (
        "def process(data):\n"
        "    data = clean(data)\n"
        "    return data\n"
    )
    snippet = (
        "def process(data):\n"
        "    data = scrub(data)\n"
        "# NOTE: ordering matters here\n"
        "    data = stamp(data)\n"
        "# ...\n"
    )

    result = _merge(tmp_path, "process.py", original, snippet, replace="process")

    assert result.model_tokens == 0
    lines = result.merged_code.splitlines()
    # The code lines land at exactly the body indent (4 spaces), not
    # pushed to 8 by the flush-left comment dragging the base to 0.
    assert "    data = scrub(data)" in lines, result.merged_code
    assert "    data = stamp(data)" in lines, result.merged_code
    # The flush-left comment is preserved as declared (never negative).
    assert "# NOTE: ordering matters here" in lines, result.merged_code
    # The structure is valid Python.
    assert result.parse_valid is True, (
        f"merged output is parse-invalid:\n{result.merged_code}"
    )


# ---------------------------------------------------------------------------
# B8 — `_adjust_indent` picks the indent character from the first character
# only and applies space-style column counts in tab-indented files
# ---------------------------------------------------------------------------


def test_tab_indented_file_gets_tab_indented_insertions(tmp_path):
    """Insertions into a tab-indented file must use the file's dominant
    indent character — tabs, one per level — even when the snippet's new
    lines are written with 4-space indent (B8, ``_adjust_indent``).

    The context lines are copied from the file (tabs); the new lines are
    typed with spaces. Every inserted line's leading whitespace must be
    tabs only, at the file's own level width, with the relative nesting
    of the inserted lines preserved.
    """
    original = (
        "def fetch(self):\n"
        "\tself.data = load()\n"
        '\tself.log("fetched")\n'
        "\treturn self.data\n"
    )
    snippet = (
        "def fetch(self):\n"
        "\tself.data = load()\n"
        "    if not self.data:\n"
        "        return None\n"
        '\tself.log("fetched")\n'
        "\treturn self.data\n"
    )

    result = _merge(tmp_path, "fetch.py", original, snippet, replace="fetch")

    assert result.model_tokens == 0
    lines = result.merged_code.splitlines()
    inserted = [
        ln for ln in lines
        if ln.strip() in ("if not self.data:", "return None")
    ]
    assert len(inserted) == 2, (
        f"expected both inserted lines, got {inserted!r}:\n{result.merged_code}"
    )
    for ln in inserted:
        leading = ln[: len(ln) - len(ln.lstrip())]
        assert leading and " " not in leading, (
            f"inserted line {ln!r} has leading whitespace {leading!r} — "
            f"spaces in a tab-indented file"
        )
    # Relative nesting preserved at the file's own level width: the
    # guard body sits exactly one tab deeper than the guard.
    assert "\tif not self.data:" in lines, result.merged_code
    assert "\t\treturn None" in lines, result.merged_code
    # Original lines are untouched (tabs preserved byte-exact).
    assert "\tself.data = load()" in lines
    assert '\tself.log("fetched")' in lines
    assert "\treturn self.data" in lines
    assert result.parse_valid is True, (
        f"merged output is parse-invalid:\n{result.merged_code}"
    )


# ---------------------------------------------------------------------------
# B6/B8 — a new line dedented below column 0 relative to the anchor is
# floored at 0; content is preserved, never stripped
# ---------------------------------------------------------------------------


def test_new_line_indent_clamps_at_zero_without_negative(tmp_path):
    """A new line whose anchor-relative offset would take it below
    column 0 must be floored at 0 with its content intact (clamp on the
    anchor-relative re-indent introduced by the B6 fix).

    The snippet pastes the whole edit one level too deep (the signature
    anchor sits at 8 spaces). The body line re-bases to the body indent;
    the comment written below the anchor's snippet indent would take a
    negative indent — it must land at column 0, unstripped, and the
    merged output must stay parse-valid.
    """
    original = (
        "def process(data):\n"
        "    data = clean(data)\n"
        "    return data\n"
    )
    snippet = (
        "        def process(data):\n"
        "            data = stamp(data)\n"
        "    # pasted from a shallower context\n"
        "        # ...\n"
    )

    result = _merge(tmp_path, "process.py", original, snippet, replace="process")

    assert result.model_tokens == 0
    lines = result.merged_code.splitlines()
    # Content preserved verbatim — no negative slice stripped characters.
    assert "# pasted from a shallower context" in lines, result.merged_code
    # ...floored at column 0 (no leading whitespace, nothing mangled).
    assert lines.count("# pasted from a shallower context") == 1, (
        f"clamped line not floored at column 0 exactly once:\n"
        f"{result.merged_code!r}"
    )
    # The sibling code line still re-bases to the body indent.
    assert "    data = stamp(data)" in lines, result.merged_code
    assert result.parse_valid is True, (
        f"merged output is parse-invalid:\n{result.merged_code}"
    )


# ---------------------------------------------------------------------------
# B3 — the deterministic splice must pass the content-faithfulness check
# (Step 6). The splice site used to persist ``deterministic_edit`` output
# with ONLY a parse check; a parse cannot see a dropped original line, a
# leaked keep-marker or a selectively re-indented survivor. The helper
# under test gates the splice on the shared content validator
# (``_check_hallucinations``, Step 7 semantics): the deterministic path
# must never bypass it.
#
# The helper is imported INSIDE the tests on purpose: it is introduced by
# this very step, and a module-level import would break collection of the
# whole file during the RED phase instead of failing exactly the new tests.
# ---------------------------------------------------------------------------

_MID_GAP_ORIGINAL = (
    "def process(data):\n"
    "    data = validate(data)\n"
    "    data = clean(data)\n"
    "    return data\n"
)

_MID_GAP_SNIPPET = (
    "def process(data):\n"
    "    data = validate(data)\n"
    "    data = audit(data)\n"
    "# ...\n"
)


def _deterministic_result_is_faithful(
    original_func: str, edited: str, snippet: str
) -> bool:
    """Module-level name resolved lazily (see the section note above)."""
    from fastedit.inference.chunked_merge import (
        _deterministic_result_is_faithful as _impl,
    )

    return _impl(original_func, edited, snippet)


def test_faithful_deterministic_result_passes_the_faithfulness_check():
    """(a) A real ``deterministic_edit`` output scores faithful (B3).

    The B3 invariant: EVERY output the deterministic path emits must
    satisfy the shared content validator. This is the documented
    middle-edit idiom — the new line is inserted and every unmentioned
    original survives — so the gate must let it through.
    """
    from fastedit.inference.text_match import deterministic_edit

    edited = deterministic_edit(_MID_GAP_ORIGINAL, _MID_GAP_SNIPPET)
    assert edited is not None, (
        "deterministic_edit declined the canonical middle-edit shape — "
        "that is a Step 2-5 behavior change to investigate FIRST; this "
        "test guards the validator, not the editor"
    )
    # The gate accepts the editor's own output.
    assert _deterministic_result_is_faithful(
        _MID_GAP_ORIGINAL, edited, _MID_GAP_SNIPPET
    ) is True


# (label, original_func, snippet, corrupted) — the corruption is exactly
# what the pre-fix editor emitted for each shape (parse-valid, lossy).
_DROP_CORRUPTION_CASES = [
    (
        # The exact trailing-suffix truncation the pre-Step-2 editor
        # emitted for the B1 shape: everything after the last anchor
        # silently deleted.
        "B1-trailing-suffix-drop",
        (
            "def foo():\n"
            "    a = 1\n"
            "    b = 2\n"
            "    c = 3\n"
            "    d = 4\n"
            "    return d\n"
        ),
        (
            "def foo():\n"
            "    a = 1\n"
            "    log(a)\n"
        ),
        (
            "def foo():\n"
            "    a = 1\n"
            "    log(a)\n"
        ),
    ),
    (
        # The B2 mid-gap class: sibling statements between the last
        # anchor and the marker vanish from the output.
        "B2-mid-gap-sibling-drop",
        (
            "def process(data):\n"
            "    data = validate(data)\n"
            "    data = clean(data)\n"
            "    data = stamp(data)\n"
            "    return data\n"
        ),
        (
            "def process(data):\n"
            "    data = validate(data)\n"
            "    data = audit(data)\n"
            "# ...\n"
        ),
        (
            "def process(data):\n"
            "    data = validate(data)\n"
            "    data = audit(data)\n"
            "    return data\n"
        ),
    ),
]


@pytest.mark.parametrize(
    ("label", "original_func", "snippet", "corrupted"),
    _DROP_CORRUPTION_CASES,
    ids=[case[0] for case in _DROP_CORRUPTION_CASES],
)
def test_deterministic_result_dropping_unmentioned_lines_is_not_faithful(
    label, original_func, snippet, corrupted,
):
    """(b) An output that silently drops an unmentioned original line —
    the B1/B2 corruption class — must be rejected by the gate, even
    though it parses cleanly. This is the regression the splice-site
    parse check alone could never catch."""
    assert _deterministic_result_is_faithful(
        original_func, corrupted, snippet
    ) is False, f"[{label}] dropped-line corruption passed the gate"


def test_deterministic_result_leaking_keep_marker_is_not_faithful():
    """(c) An output that echoes a keep-marker line into the file must be
    rejected (the marker is a directive, never content)."""
    corrupted = (
        "def process(data):\n"
        "    data = validate(data)\n"
        "    data = audit(data)\n"
        "# ... existing code ...\n"
        "    data = clean(data)\n"
        "    return data\n"
    )
    assert _deterministic_result_is_faithful(
        _MID_GAP_ORIGINAL, corrupted, _MID_GAP_SNIPPET
    ) is False


def test_deterministic_result_selectively_reindenting_survivors_is_not_faithful():
    """(d) A merge that selectively re-indents a surviving original line
    (flattening it out of its block) must be rejected: every line's
    content and multiplicity is intact, so only the indent-faithfulness
    rule (B14) can catch it — a parse check cannot."""
    original_func = (
        "def process(data):\n"
        "    data = validate(data)\n"
        "    if data:\n"
        "        data = clean(data)\n"
        "        data = stamp(data)\n"
        "    return data\n"
    )
    corrupted = (
        "def process(data):\n"
        "    data = validate(data)\n"
        "    data = audit(data)\n"
        "    if data:\n"
        "    data = clean(data)\n"
        "        data = stamp(data)\n"
        "    return data\n"
    )
    assert _deterministic_result_is_faithful(
        original_func, corrupted, _MID_GAP_SNIPPET
    ) is False


def test_unfaithful_deterministic_result_is_discarded_to_the_model_path(
    tmp_path, monkeypatch,
):
    """The splice site consults the gate with the exact span-local inputs
    and DISCARDS an unfaithful deterministic result: it must neither
    splice nor return — the edit falls through to the chunk/model
    pipeline instead (B3, Step 6).

    ``deterministic_edit`` never emits an unfaithful merge today (Steps
    2-5), so the gate is forced to decline via monkeypatching to make the
    wiring observable; the fake ``locate_chunks`` records the fall-through
    without consulting tldr.
    """
    seen: list[tuple] = []

    def _gate(*args):
        seen.append(args)
        return False

    locate_calls: list[tuple] = []

    def _fake_locate_chunks(*args, **kwargs):
        locate_calls.append(args)
        return []

    monkeypatch.setattr(
        "fastedit.inference.chunked_merge._deterministic_result_is_faithful",
        _gate,
    )
    monkeypatch.setattr(
        "fastedit.inference.chunked_merge.locate_chunks",
        _fake_locate_chunks,
    )

    file_path = tmp_path / "process.py"
    file_path.write_text(_MID_GAP_ORIGINAL)
    result = chunked_merge(
        original_code=_MID_GAP_ORIGINAL,
        snippet=_MID_GAP_SNIPPET,
        file_path=str(file_path),
        merge_fn=_no_model,
        language="python",
        replace="process",
    )

    # The gate was consulted exactly once, on the span-local triple.
    assert len(seen) == 1, f"gate consulted {len(seen)} times"
    gate_args = seen[0]
    gate_original, gate_edited, gate_snippet = gate_args[:3]
    assert gate_original == _MID_GAP_ORIGINAL
    assert "data = audit(data)" in gate_edited
    # No signature prepend happened for this shape.
    assert len(gate_args) == 4 and gate_args[3] == 0
    # The gate sees the SAME snippet the editor consumed — post marker
    # normalization (chunked_merge normalizes short forms up front).
    from fastedit.inference.markers import normalize_markers
    assert gate_snippet == normalize_markers(_MID_GAP_SNIPPET)

    # The deterministic output was discarded: the edit fell through to
    # the chunk pipeline and nothing was spliced into the file.
    assert locate_calls, (
        "unfaithful deterministic result must fall through to the "
        "chunk/model pipeline, not be silently dropped"
    )
    assert result.merged_code == _MID_GAP_ORIGINAL
    assert "data = audit(data)" not in result.merged_code
