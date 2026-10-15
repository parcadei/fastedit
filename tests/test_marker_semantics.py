"""Marker semantics across the merge pipeline (B16 + the B20 splice half).

Section layout (later steps append their own section to this file, per the
implementation plan):

  Section 1 — Step 12: the ``after=`` fast path in ``chunked_merge``

  Section 2 — Step 13 (placeholder): markers/strings/indent semantics
  (B17 remainder, B18, B7, B30, B33 land here later).

===========================================================================
Section 1 — Step 12: the ``after=`` fast path
===========================================================================

The ``after=`` path splices the snippet VERBATIM after an anchor symbol
with zero model tokens. Two defects are covered here:

* **B16 — marker leakage.** A preservation marker inside the snippet was
  written into the file as a literal comment line. It parses as a comment,
  so nothing downstream ever objects — silent corruption. Markers are
  merge directives, never content: the splice must drop them (logging the
  drop), and a snippet made ONLY of markers must fail loudly instead of
  silently reporting a successful no-op (CLAUDE.md: fail loudly, never
  silently write garbage).

* **B20 (splice half) — bare-LF pieces.** The splice hardcoded ``"\n"``
  separators and never normalized the snippet's own terminators,
  introducing bare-LF pieces into CRLF files (mixed-ending seams). The
  inserted piece must carry the ORIGINAL's line-ending convention. The
  central EOL normalizer over every return path is Step 14; this section
  only stops the ``after=`` path from INTRODUCING bare-LF pieces.

All tests drive the PUBLIC entry point :func:`chunked_merge` with a
``merge_fn`` that fails loudly if the model is ever invoked (the after=
path is zero-token). No network, no model download, no disk reads outside
``tmp_path``.
"""

from __future__ import annotations

import logging

import pytest

from fastedit.inference import snippet_analysis
from fastedit.inference.ast_utils import ASTNode
from fastedit.inference.chunked_merge import chunked_merge
from fastedit.inference.indent import _escape_tags, _new_tag_nonce, _unescape_tags
from fastedit.inference.markers import normalize_markers


def _no_model(*_args, **_kwargs):
    """Merge fn that fails loudly if the edit falls through to the model."""
    raise AssertionError(
        "merge_fn must NOT be called — the after= path is a zero-token splice"
    )


def _merge_after(tmp_path, name, original_code, snippet, after, language="python"):
    """Run chunked_merge on the after= fast path against a tmp file."""
    file_path = tmp_path / name
    file_path.write_text(original_code)
    return chunked_merge(
        original_code=original_code,
        snippet=snippet,
        file_path=str(file_path),
        merge_fn=_no_model,
        language=language,
        after=after,
    )


def _bare_lf_count(text: str) -> int:
    """Count ``\\n`` bytes not preceded by ``\\r`` — the mixed-ending seam metric."""
    data = text.encode("utf-8")
    return data.count(b"\n") - data.count(b"\r\n")


# ---------------------------------------------------------------------------
# Fixtures: a two-function LF original and the exact splice result the
# after= path must produce for the snippet "def gamma():\n    return 3\n"
# inserted after alpha (blank separator line before, none after — the
# original already has blank lines following the anchor).
# ---------------------------------------------------------------------------

_ORIGINAL_LF = (
    "def alpha():\n"
    "    return 1\n"
    "\n"
    "\n"
    "def beta():\n"
    "    return 2\n"
)

_EXPECTED_GAMMA_LF = (
    "def alpha():\n"
    "    return 1\n"
    "\n"
    "def gamma():\n"
    "    return 3\n"
    "\n"
    "\n"
    "def beta():\n"
    "    return 2\n"
)

_CRLF_ORIGINAL = (
    "def alpha():\r\n"
    "    return 1\r\n"
    "\r\n"
    "\r\n"
    "def beta():\r\n"
    "    return 2\r\n"
)

_CRLF_EXPECTED_GAMMA = (
    "def alpha():\r\n"
    "    return 1\r\n"
    "\r\n"
    "def gamma():\r\n"
    "    return 3\r\n"
    "\r\n"
    "\r\n"
    "def beta():\r\n"
    "    return 2\r\n"
)


# ---------------------------------------------------------------------------
# B16 (a) — a canonical long-form marker in the snippet must never reach
# the file; the edit otherwise applies exactly as if the marker line had
# never been sent.
# ---------------------------------------------------------------------------


def test_after_path_drops_canonical_long_form_marker(tmp_path):
    """``# ... existing code ...`` in an after= snippet is a directive, not
    content: the merged file must be byte-identical to the marker-free
    splice. Pre-fix, the marker line landed in the file as a literal
    comment."""
    snippet = (
        "def gamma():\n"
        "# ... existing code ...\n"
        "    return 3\n"
    )

    result = _merge_after(
        tmp_path, "marker_long.py", _ORIGINAL_LF, snippet, after="alpha"
    )

    assert result.model_tokens == 0
    assert result.parse_valid is True, result.merged_code
    assert result.merged_code == _EXPECTED_GAMMA_LF, (
        f"marker line leaked into the splice:\n{result.merged_code!r}"
    )


def test_after_path_drops_indented_canonical_marker(tmp_path):
    """A marker carrying snippet indentation is still a marker line — the
    predicate is line-anchored on stripped content — and must be dropped."""
    snippet = (
        "def gamma():\n"
        "    # ... existing code ...\n"
        "    return 3\n"
    )

    result = _merge_after(
        tmp_path, "marker_indented.py", _ORIGINAL_LF, snippet, after="alpha"
    )

    assert result.model_tokens == 0
    assert result.merged_code == _EXPECTED_GAMMA_LF, (
        f"indented marker line leaked into the splice:\n{result.merged_code!r}"
    )


# ---------------------------------------------------------------------------
# B16 (b) — a short-form marker must be normalized to the canonical long
# form and then dropped, exactly like the long form.
# ---------------------------------------------------------------------------


def test_after_path_normalizes_short_form_marker_then_drops_it(tmp_path):
    """``#...`` (short form) is rewritten to the canonical long form by
    ``normalize_markers`` and must then be dropped like any other marker.
    Pre-fix, the rewritten marker line was spliced into the file."""
    snippet = (
        "def gamma():\n"
        "#...\n"
        "    return 3\n"
    )

    result = _merge_after(
        tmp_path, "marker_short.py", _ORIGINAL_LF, snippet, after="alpha"
    )

    assert result.model_tokens == 0
    assert result.parse_valid is True, result.merged_code
    assert result.merged_code == _EXPECTED_GAMMA_LF, (
        f"short-form marker leaked into the splice:\n{result.merged_code!r}"
    )


def test_after_path_drops_marker_first_snippet_without_skewing_alignment(tmp_path):
    """A snippet that OPENS with a marker must still splice its content at
    the anchor's indent level: markers are dropped before the indent
    arithmetic, so a leading marker line cannot become the alignment base."""
    snippet = (
        "# ... existing code ...\n"
        "def gamma():\n"
        "    return 3\n"
    )

    result = _merge_after(
        tmp_path, "marker_first.py", _ORIGINAL_LF, snippet, after="alpha"
    )

    assert result.model_tokens == 0
    assert result.merged_code == _EXPECTED_GAMMA_LF, (
        f"leading marker corrupted the splice:\n{result.merged_code!r}"
    )


# ---------------------------------------------------------------------------
# B16 (c) — control: non-marker content lines splice exactly as before.
# The marker-drop tests above must produce byte-identical output to this
# marker-free splice.
# ---------------------------------------------------------------------------


def test_after_path_splices_non_marker_content_lines_intact(tmp_path):
    """A marker-free snippet splices verbatim after the anchor — the
    baseline the marker-dropping behavior must reproduce exactly."""
    snippet = (
        "def gamma():\n"
        "    return 3\n"
    )

    result = _merge_after(
        tmp_path, "content_only.py", _ORIGINAL_LF, snippet, after="alpha"
    )

    assert result.model_tokens == 0
    assert result.parse_valid is True, result.merged_code
    assert result.merged_code == _EXPECTED_GAMMA_LF, result.merged_code


# ---------------------------------------------------------------------------
# B20 (splice half) — the inserted piece must carry the ORIGINAL's
# line-ending convention. No bare-LF bytes may be introduced into a CRLF
# file, neither by the hardcoded separators nor by an unnormalized
# snippet. (Full central EOL normalization is Step 14; this only stops
# this path from INTRODUCING bare-LF pieces.)
# ---------------------------------------------------------------------------


def test_after_path_on_crlf_file_emits_crlf_terminated_snippet_lines(tmp_path):
    """A bare-LF snippet inserted into a CRLF file must land with CRLF
    terminators — byte-exact expected output and zero bare LF bytes."""
    snippet = (
        "def gamma():\n"
        "    return 3\n"
    )

    result = _merge_after(
        tmp_path, "crlf.py", _CRLF_ORIGINAL, snippet, after="alpha"
    )

    assert result.model_tokens == 0
    assert result.parse_valid is True, result.merged_code
    assert result.merged_code == _CRLF_EXPECTED_GAMMA, (
        f"splice did not adopt the file's CRLF convention:\n{result.merged_code!r}"
    )
    assert _bare_lf_count(result.merged_code) == 0, (
        f"bare LF bytes introduced into a CRLF file:\n{result.merged_code!r}"
    )


def test_after_path_on_crlf_file_drops_marker_and_normalizes_to_crlf(tmp_path):
    """Combined B16+B20 case: a short-form marker inside a bare-LF snippet
    is dropped AND the surviving content is spliced with CRLF endings."""
    snippet = (
        "def gamma():\n"
        "#...\n"
        "    return 3\n"
    )

    result = _merge_after(
        tmp_path, "crlf_marker.py", _CRLF_ORIGINAL, snippet, after="alpha"
    )

    assert result.model_tokens == 0
    assert result.parse_valid is True, result.merged_code
    assert result.merged_code == _CRLF_EXPECTED_GAMMA, (
        f"marker/EOL splice wrong:\n{result.merged_code!r}"
    )
    assert _bare_lf_count(result.merged_code) == 0, (
        f"bare LF bytes introduced into a CRLF file:\n{result.merged_code!r}"
    )


# ---------------------------------------------------------------------------
# B16 (e) — a snippet made only of markers declares no insertable code.
# Chosen behavior: FAIL LOUDLY with a clear ValueError. A silent no-op
# would return a parse-valid, zero-token result indistinguishable from a
# successful insert (the MCP/CLI gates would report success while nothing
# happened); every other caller error on this path (unknown anchor,
# preserve_siblings without replace, multi-symbol replace) already raises
# ValueError, so rejection is the repo-consistent choice.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "snippet"),
    [
        ("canonical", "# ... existing code ...\n"),
        ("short_form", "#...\n"),
        ("unicode_ellipsis", "…\n"),
        ("markers_and_blank_lines", "#...\n\n\n"),
    ],
    ids=["canonical", "short-form", "unicode-ellipsis", "markers-and-blanks"],
)
def test_after_path_marker_only_snippet_is_rejected_loudly(
    tmp_path, label, snippet
):
    """A marker-only snippet (every line a marker or blank) must raise a
    clear ValueError — never splice empty separators and report success."""
    with pytest.raises(ValueError, match="no insertable code"):
        _merge_after(
            tmp_path, f"reject_{label}.py", _ORIGINAL_LF, snippet, after="alpha"
        )


# ---------------------------------------------------------------------------
# B16 — the drop must be observable: a WARNING naming the count is logged
# so a model that hedged with markers is visible in the edit's trail.
# ---------------------------------------------------------------------------


def test_after_path_marker_drop_is_logged_with_count(tmp_path, caplog):
    """Dropping marker lines from an after= snippet logs a WARNING that
    includes how many lines were dropped."""
    snippet = (
        "def gamma():\n"
        "# ... existing code ...\n"
        "    return 3\n"
    )

    with caplog.at_level(logging.WARNING, logger="fastedit.chunked_merge"):
        result = _merge_after(
            tmp_path, "logged.py", _ORIGINAL_LF, snippet, after="alpha"
        )

    assert result.merged_code == _EXPECTED_GAMMA_LF
    marker_warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "marker" in r.getMessage()
    ]
    assert marker_warnings, (
        "expected a WARNING about dropped marker lines, got: "
        f"{[r.getMessage() for r in caplog.records]!r}"
    )
    assert "1" in marker_warnings[0].getMessage(), (
        "the warning must include the dropped-line count: "
        f"{marker_warnings[0].getMessage()!r}"
    )


# ===========================================================================
# Section 2 — Step 13: markers/strings/indent semantics.
#
#   B17 remainder — _find_insertion_region must never treat marker lines
#     (short OR long form) as context candidates.
#   B18 — normalize_markers must never rewrite a marker-LOOKING line that
#     lives inside a string literal (embedded template/shell scripts are
#     user data, not merge directives).
#   B33 — escape-tag placeholders carry a per-call random nonce so user
#     text containing the old literal placeholder round-trips unchanged.
#
# (B7/B30 — indent arithmetic vs string contents — are indent-specific and
# live in tests/test_indent.py.)
# ===========================================================================

# --- B17 remainder: marker lines are never insertion-context candidates ---

# A snippet line that IS a marker must be skipped when hunting for context
# lines, exactly like blank lines. Pre-fix, only the long forms were
# recognized (snippet_analysis._MARKER_RE), so a bare ``#...`` / ``//...`` /
# ``…`` line in the snippet was matched against the original file and, when
# the original happened to contain the same text, steered the insertion
# region toward that decoy instead of the real context.

def _insertion_region_for(monkeypatch, marker, context_line="    setup()"):
    """Drive _find_insertion_region with a snippet holding one REAL context
    line and one marker line; the original contains both texts. The region
    must anchor on the real context line (alpha), never on the marker decoy."""
    original_lines = [
        "import os",
        "",
        "def alpha():",
        "    setup()",
        "",
        marker,  # the decoy: the same marker text exists in the original
        "def omega():",
        "    teardown()",
    ]
    gamma = ASTNode(
        name="gamma", kind="function",
        line_start=2, line_end=3, signature="def gamma():",
    )
    monkeypatch.setattr(
        snippet_analysis, "_get_snippet_definitions",
        lambda _snippet, _language=None: [gamma],
    )
    ast_nodes = [
        ASTNode(name="alpha", kind="function",
                line_start=3, line_end=4, signature="def alpha():"),
        ASTNode(name="omega", kind="function",
                line_start=7, line_end=8, signature="def omega():"),
    ]
    snippet = f"{context_line}\ndef gamma():\n    pass\n{marker}\n"
    return snippet_analysis._find_insertion_region(
        snippet, original_lines, ast_nodes, 8,
    )


@pytest.mark.parametrize(
    "marker",
    ["#...", "//...", "…", "# ... existing code ..."],
    ids=["short-hash", "short-slash", "unicode-ellipsis", "long-form-control"],
)
def test_find_insertion_region_never_treats_marker_lines_as_context(
    monkeypatch, marker,
):
    """Marker snippet lines must not steer the insertion region.

    Anchoring on the real context line (setup(), inside alpha) spans
    alpha..omega = lines 3..8. Anchoring on the marker decoy (line 6)
    would produce omega-only lines 7..8 — the regression this test pins.
    """
    region = _insertion_region_for(monkeypatch, marker)

    assert region is not None
    assert (region.start_line, region.end_line) == (3, 8), (
        f"marker line {marker!r} steered the insertion region: "
        f"{(region.start_line, region.end_line)}"
    )


# --- B18: normalize_markers is string-aware -------------------------------

def test_normalize_markers_leaves_markers_inside_double_quote_triple_string():
    """``#...`` / ``…`` lines INSIDE a triple-quoted string are user data
    (an embedded template or shell script) — never rewritten to keep-markers
    (pre-fix they were rewritten and later dropped as markers: data loss)."""
    snippet = (
        "x = 1\n"
        'template = """\n'
        "#...\n"
        "…\n"
        '"""\n'
    )
    assert normalize_markers(snippet) == snippet


def test_normalize_markers_leaves_markers_inside_single_quote_triple_string():
    """Same protection for the ``'''`` delimiter family."""
    snippet = "script = '''\n#...\n//...\n'''\n"
    assert normalize_markers(snippet) == snippet


def test_normalize_markers_still_rewrites_bare_top_level_markers():
    """Control: genuine top-level short-form markers are still normalized."""
    assert normalize_markers("#...\n") == "# ... existing code ...\n"
    assert normalize_markers("…\n") == "# ... existing code ...\n"
    assert normalize_markers("x = 1\n#...\n") == "x = 1\n# ... existing code ...\n"


def test_normalize_markers_resumes_after_the_string_closes():
    """String-awareness masks only the string's span: a marker line AFTER the
    closing delimiter is a real directive and must still be rewritten."""
    snippet = 'template = """\n#...\n"""\n#...\n'
    assert normalize_markers(snippet) == (
        'template = """\n#...\n"""\n# ... existing code ...\n'
    )


# --- B33: escape-tag placeholders carry a per-call random nonce -----------

_TAG_OPEN = "<updated-code>"
_TAG_CLOSE = "</updated-code>"
_LEGACY_OPEN_SAFE = "__FASTEDIT_TAG_OPEN__"
_LEGACY_CLOSE_SAFE = "__FASTEDIT_TAG_CLOSE__"


def test_new_tag_nonce_is_hex_and_nonempty():
    nonce = _new_tag_nonce()
    assert len(nonce) >= 8
    int(nonce, 16)  # hex — must not raise


def test_escape_tags_with_nonce_uses_nonced_placeholder():
    nonce = _new_tag_nonce()
    escaped = _escape_tags(f"a {_TAG_OPEN} b", nonce)
    assert _TAG_OPEN not in escaped
    assert f"__FASTEDIT_TAG_OPEN_{nonce}__" in escaped


def test_escape_unescape_round_trip_with_nonce():
    """escape → unescape with the SAME nonce restores tags verbatim — and a
    legacy placeholder inside user text survives untouched."""
    nonce = _new_tag_nonce()
    text = (
        f"line {_TAG_OPEN} x\n"
        f"{_TAG_CLOSE}\n"
        f"legacy {_LEGACY_OPEN_SAFE} stays\n"
    )
    assert _unescape_tags(_escape_tags(text, nonce), nonce) == text


def test_user_text_with_legacy_placeholder_round_trips_unchanged():
    """B33: user code containing the OLD literal placeholder
    ``__FASTEDIT_TAG_OPEN__`` must round-trip byte-identically — no tag
    injection on unescape. Pre-fix, that string collided with the escape
    placeholder and came back as a literal <updated-code> tag."""
    nonce = _new_tag_nonce()
    user = "x = '__FASTEDIT_TAG_OPEN__'\ny = '__FASTEDIT_TAG_CLOSE__'\n"
    escaped = _escape_tags(user, nonce)
    assert escaped == user, (
        "escape must not rewrite the legacy placeholder inside user text"
    )
    assert _unescape_tags(escaped, nonce) == user, (
        "unescape injected tags from a user-data placeholder collision"
    )
    assert _LEGACY_OPEN_SAFE in _unescape_tags(escaped, nonce)


def test_unescape_only_maps_its_own_nonce():
    """A placeholder from a DIFFERENT nonce is inert content — only the
    matching nonce's placeholders are restored."""
    text = f"a {_TAG_OPEN} b"
    escaped = _escape_tags(text, "aaaa1111")
    assert _unescape_tags(escaped, "bbbb2222") == escaped
    assert _unescape_tags(escaped, "aaaa1111") == text


# ===========================================================================
# Section 3 — the HTML-comment marker form (issue #1 hole 2).
#
#   ``<!-- ... existing code ... -->`` is the canonical HTML/XML comment
#   idiom and was INVISIBLE to every marker gate: ``_MARKER_PHRASES`` and
#   the short-form regexes only knew ``#`` and ``//`` prefixes, so the
#   marker line was written into files literally (case f of the repro).
#
#   The grammar is LINE-ANCHORED like every other form: a bare (or
#   indented) ``<!-- ... existing code ... -->`` line IS a marker; the
#   same text mid-line — ``<p>text <!-- ... existing code ... --></p>``
#   — is CONTENT and must survive every merge untouched.
# ===========================================================================

from fastedit.inference.markers import is_marker_line

_HTML_MARKER = "<!-- ... existing code ... -->"


@pytest.mark.parametrize(
    "line",
    [
        _HTML_MARKER,
        "  " + _HTML_MARKER,
        "\t" + _HTML_MARKER,
    ],
    ids=["bare", "indented-spaces", "indented-tab"],
)
def test_is_marker_line_true_for_html_comment_form(line):
    """The canonical HTML-comment keep-marker is a marker line, bare or
    indented — line-anchored on stripped content like every other form."""
    assert is_marker_line(line) is True, line


def test_is_marker_line_false_for_mid_line_html_marker():
    """A marker phrase embedded mid-line is CONTENT (the line-anchored
    rule): ``<p>text <!-- ... existing code ... --></p>`` must survive the
    merge and must never be treated as a directive."""
    assert is_marker_line("<p>text <!-- ... existing code ... --></p>") is False


def test_is_marker_line_false_for_partial_html_marker_shapes():
    """Near-miss lines are content: a comment about the marker, an unclosed
    opener, and a different comment payload are not directives."""
    assert is_marker_line("<!-- the marker below is documented -->") is False
    assert is_marker_line("<!-- ... existing code ...") is False
    assert is_marker_line("<!-- do not remove -->") is False


def test_normalize_markers_rewrites_html_short_form():
    """The short HTML form ``<!--...-->`` is rewritten to the canonical
    hash marker like the other short forms."""
    assert normalize_markers("<!--...-->\n") == "# ... existing code ...\n"
    assert normalize_markers("x\n<!--...-->\n") == (
        "x\n# ... existing code ...\n"
    )


def test_normalize_markers_keeps_canonical_html_form_unchanged():
    """The canonical HTML form passes through normalization unchanged (it is
    already canonical) and is recognized as a marker afterward."""
    assert normalize_markers(_HTML_MARKER + "\n") == _HTML_MARKER + "\n"
    assert is_marker_line(_HTML_MARKER) is True


def test_normalize_markers_preserves_html_marker_indent():
    """Indentation on a rewritten HTML short form is preserved, matching the
    other short forms' indent semantics."""
    assert normalize_markers("  <!--...-->\n") == (
        "  # ... existing code ...\n"
    )


def test_normalize_markers_leaves_html_marker_inside_string_alone():
    """B18 string-awareness covers the HTML form too: a marker-looking line
    inside a triple-quoted string is user data, never rewritten."""
    snippet = (
        "x = 1\n"
        'template = """\n'
        + _HTML_MARKER + "\n"
        '"""\n'
    )
    assert normalize_markers(snippet) == snippet


def test_normalize_markers_still_recognizes_html_marker_after_string_closes():
    """A HTML marker line AFTER the closing delimiter is a real directive
    (string masking is span-local): the canonical form passes through
    normalization unchanged and is still recognized as a marker — a short
    form in the same position would be rewritten."""
    snippet = 'template = """\n' + _HTML_MARKER + '\n"""\n' + _HTML_MARKER + "\n"
    normalized = normalize_markers(snippet)
    assert normalized.splitlines()[-1] == _HTML_MARKER
    assert is_marker_line(normalized.splitlines()[-1]) is True
    # The in-string copy stays untouched user data.
    assert normalized.splitlines()[1] == _HTML_MARKER
    rewritten = normalize_markers(
        'template = """\n' + _HTML_MARKER + '\n"""\n<!--...-->\n'
    )
    assert rewritten.splitlines()[-1] == "# ... existing code ..."


# ---------------------------------------------------------------------------
# End-to-end: an after= snippet carrying the HTML marker behaves exactly
# like the # marker — the directive is dropped, never written literally.
# ---------------------------------------------------------------------------


def test_after_path_drops_html_comment_marker(tmp_path):
    """Case f of the repro, at the merge layer: ``<!-- ... existing code
    ... -->`` in an after= snippet is a directive and must be dropped, not
    spliced into the file as a literal comment."""
    snippet = (
        "def gamma():\n"
        + _HTML_MARKER + "\n"
        "    return 3\n"
    )

    result = _merge_after(
        tmp_path, "marker_html.py", _ORIGINAL_LF, snippet, after="alpha"
    )

    assert result.model_tokens == 0
    assert result.parse_valid is True, result.merged_code
    assert result.merged_code == _EXPECTED_GAMMA_LF, (
        f"HTML-comment marker leaked into the splice:\n{result.merged_code!r}"
    )


def test_after_path_marker_only_html_snippet_is_rejected_loudly(tmp_path):
    """A snippet made only of HTML markers declares no insertable code and
    must fail loudly like every other marker-only after= snippet."""
    with pytest.raises(ValueError, match="no insertable code"):
        _merge_after(
            tmp_path, "reject_html.py", _ORIGINAL_LF,
            _HTML_MARKER + "\n", after="alpha",
        )


def test_after_path_normalizes_html_short_form_then_drops_it(tmp_path):
    """The short HTML form ``<!--...-->`` is normalized to the canonical
    marker and then dropped like any other marker line."""
    snippet = (
        "def gamma():\n"
        "<!--...-->\n"
        "    return 3\n"
    )

    result = _merge_after(
        tmp_path, "marker_html_short.py", _ORIGINAL_LF, snippet, after="alpha"
    )

    assert result.model_tokens == 0
    assert result.merged_code == _EXPECTED_GAMMA_LF, (
        f"short HTML marker leaked into the splice:\n{result.merged_code!r}"
    )


# ---------------------------------------------------------------------------
# End-to-end markdown (issue #1 repro case f, whole pipeline): a
# replace=<section> snippet carrying the HTML marker flows through
# chunked_merge exactly like the # marker — the leak-guard/battery sees it,
# and the marker is never written into the file literally.
# ---------------------------------------------------------------------------

_MD_ORIGINAL = (
    "# Guide\n"
    "\n"
    "Intro paragraph.\n"
    "\n"
    "## Usage\n"
    "\n"
    "Usage notes.\n"
    "\n"
    "## Next\n"
    "\n"
    "more text\n"
)


def _merge_md_replace(tmp_path, model_merged_code, snippet, calls):
    """Drive chunked_merge on a markdown replace= with a stubbed model that
    always returns ``model_merged_code`` (recorded per call in ``calls``)."""
    from types import SimpleNamespace

    file_path = tmp_path / "doc.md"
    file_path.write_text(_MD_ORIGINAL)

    def merge_fn(chunk, snippet_text, language):
        calls.append(snippet_text)
        return SimpleNamespace(
            merged_code=model_merged_code, parse_valid=True,
            tokens_generated=7, latency_ms=1.0, truncated=False,
        )

    return chunked_merge(
        original_code=_MD_ORIGINAL,
        snippet=snippet,
        file_path=str(file_path),
        merge_fn=merge_fn,
        language="markdown",
        replace="Usage",
    )


def test_md_replace_html_marker_tail_preserved_never_written(tmp_path):
    """The HTML marker in a markdown replace= snippet is a directive: the
    text-match path consumes it and the merged output preserves the section
    tail with the new line appended — the marker itself never reaches the
    file (repro case f: it used to be written literally)."""
    snippet = (
        "## Usage\n"
        "<!-- ... existing code ... -->\n"
        "Updated usage notes.\n"
    )
    expected = (
        "# Guide\n"
        "\n"
        "Intro paragraph.\n"
        "\n"
        "## Usage\n"
        "\n"
        "Usage notes.\n"
        "Updated usage notes.\n"
        "\n"
        "## Next\n"
        "\n"
        "more text\n"
    )
    calls: list[str] = []

    result = _merge_md_replace(tmp_path, expected, snippet, calls)

    # Zero-token: the deterministic text-match path placed the edit.
    assert calls == [], "the model must not be needed for a placed marker edit"
    assert result.merged_code == expected, result.merged_code
    assert "existing code" not in result.merged_code, result.merged_code
    assert result.model_tokens == 0


def test_md_replace_model_leaked_html_marker_is_rejected_by_battery(tmp_path):
    """The leak-guard sees the HTML form: a model merge that echoes the
    ``<!-- ... existing code ... -->`` line into its output fails the
    content-faithfulness battery on every retry and the chunk is rejected —
    the original markdown is kept and the marker is never persisted."""
    snippet = (
        "## Usage\n"
        "First inserted line.\n"
        "<!-- ... existing code ... -->\n"
        "Second inserted line.\n"
    )
    leaked = (
        "# Guide\n"
        "\n"
        "Intro paragraph.\n"
        "\n"
        "## Usage\n"
        "\n"
        "First inserted line.\n"
        "<!-- ... existing code ... -->\n"
        "Usage notes.\n"
        "Second inserted line.\n"
        "\n"
        "## Next\n"
        "\n"
        "more text\n"
    )
    calls: list[str] = []

    result = _merge_md_replace(tmp_path, leaked, snippet, calls)

    # The model path ran (ambiguous marker position) and was retried.
    assert len(calls) >= 2, calls
    # Rejection convention: original kept, every chunk reported rejected.
    assert result.chunks_rejected >= 1
    assert result.chunks_rejected >= result.chunks_used
    assert result.merged_code == _MD_ORIGINAL, result.merged_code
    assert "existing code" not in result.merged_code
