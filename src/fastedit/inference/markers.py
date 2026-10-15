"""Single source of truth for keep-marker detection (B15 unification).

Every question "is this line a marker?" in the codebase is answered by
:func:`is_marker_line` in this module. Before this module existed, three
definitions coexisted and disagreed:

  * ``text_match._is_marker`` matched marker phrases by SUBSTRING
    containment, so a real comment like
    ``x = compute(a)  # ... see docs for details`` was misclassified as a
    keep-marker: it was swallowed from the merge output and its presence
    suppressed preserved-suffix emission (B15).
  * ``chunked_merge._is_marker_line`` had the same substring defect in
    the validator's view (feeding B14).
  * ``chunk_locator._narrow_large_node`` carried its own private copy.

The rule here is LINE-ANCHORED: a line is a marker iff its stripped
content exactly equals one of the canonical marker phrases, or it is a
bare short-form marker (``#...``, ``//...``, ``…``). A marker phrase
embedded mid-line is content, never an instruction.

:func:`normalize_markers` rewrites short/Unicode marker forms to the
canonical long form before downstream processing. It is a pure string
transform with one guard (B18): the rewrite is decided against a
string-span-masked copy of the snippet (:func:`split_join.mask_string_spans`),
so a marker-LOOKING line inside a multi-line string — an embedded template
or shell script — is user data and is never rewritten.
"""

from __future__ import annotations

import re

from ..split_join import mask_string_spans

# Canonical long-form markers the rest of the pipeline understands.
_CANONICAL_HASH_MARKER = "# ... existing code ..."
_CANONICAL_SLASH_MARKER = "// ... existing code ..."
# Issue #1 (hole 2): the HTML/XML comment idiom is the marker form models
# emit for markup documents (markdown sections, html elements). It was
# invisible to every gate below and was written into files literally.
_CANONICAL_HTML_MARKER = "<!-- ... existing code ... -->"

# Phrases that mark "keep everything here" in snippets. Membership is
# LINE-ANCHORED (exact match on the stripped line) — substring
# containment misclassified real comments as markers (B15).
#
# The canonical long forms MUST be members: under the old substring rule
# ``# ... existing code ...`` matched via its ``# ...`` prefix, so the
# exact-membership rewrite needs them spelled out or the documented
# long-form marker stops being a marker everywhere downstream
# (normalized short forms included).
_MARKER_PHRASES = (
    _CANONICAL_HASH_MARKER,
    _CANONICAL_SLASH_MARKER,
    _CANONICAL_HTML_MARKER,
    # Legacy forms kept for backward compatibility with the pre-B15
    # substring set: a bare "... existing code ..." line, and the
    # ellipsis-only comment stubs.
    "... existing code ...",
    "// ...",
    "# ...",
)

# Short-form markers (v0.2.4): recognized by ``normalize_markers`` and
# rewritten to the canonical long form before any downstream processing.
#
# Detection (stripped line):
#   - Exact: ``#...``, ``//...``, ``<!--...-->``, ``…`` (Unicode ellipsis U+2026)
#   - Legacy long forms match via exact membership in ``_MARKER_PHRASES``.
#   - Generic regex catches spacing variants (e.g. ``# ...``, ``// ..``).
#
# The rule is intentionally permissive on the short side — a lone
# ``#...`` on a line is vanishingly unlikely to be real code.
_SHORT_HASH_RE = re.compile(r"^\s*#\s*\.\.\.\s*$")
_SHORT_SLASH_RE = re.compile(r"^\s*//\s*\.\.\.\s*$")
_UNICODE_ELLIPSIS_RE = re.compile(r"^\s*…\s*$")

# HTML-comment short form and canonical long form. Both are LINE-ANCHORED:
# the regex matches a whole line (allowing surrounding whitespace) so the
# marker text mid-line — ``<p>text <!-- ... existing code ... --></p>`` —
# stays CONTENT, exactly like the ``#``/``//`` line-anchored rule.
_SHORT_HTML_RE = re.compile(r"^<!--\s*\.\.\.\s*-->")
_LONG_HTML_RE = re.compile(r"^<!--\s*\.\.\..*existing.*\.\.\.\s*-->")

# Exact-form set used by ``snippet_has_keep_marker`` (text_match) —
# deliberately NOT the permissive substring set: "# ..." / "// ..." as
# bare substrings misclassify real code.
_EXACT_SHORT_MARKERS = ("#...", "//...", "…", "#…", "//…")

# Long-form marker line regex (moved verbatim from snippet_analysis so
# every marker constant lives in one module; snippet_analysis imports it
# back, keeping ``snippet_analysis._MARKER_RE`` importable).
_MARKER_RE = re.compile(r'^\s*(?:#|//|/\*)\s*\.\.\..*(?:existing|rest).*\.\.\.')

# Hot-path indexes over the phrase table (governor mission, PART 1):
# ``stripped in _MARKER_PHRASES`` scanned a 6-tuple per line and the regex
# chain ran up to five ``.match`` calls per line on EVERY content line of
# every chunk (``_raw_content_lines`` runs per validation attempt). Both
# checks have exact necessary-condition fast paths: every marker phrase and
# every regex form begins with one of five characters, so a line whose
# stripped head is anything else is decided False without touching the
# table or the regexes.
_MARKER_PHRASE_SET = frozenset(_MARKER_PHRASES)
_MARKER_FIRST_CHARS = frozenset("#/.<…")


def is_marker_text(stripped: str) -> bool:
    """True when the ALREADY-STRIPPED ``stripped`` is a marker line.

    The engine behind :func:`is_marker_line` — same verdict, one ``strip``
    cheaper for callers that already stripped the line (the hot
    ``_raw_content_lines`` path in chunked_merge runs this per line).
    """
    if not stripped or stripped[0] not in _MARKER_FIRST_CHARS:
        return False
    return (
        stripped in _MARKER_PHRASE_SET
        or bool(_SHORT_HASH_RE.match(stripped))
        or bool(_SHORT_SLASH_RE.match(stripped))
        or bool(_UNICODE_ELLIPSIS_RE.match(stripped))
        or bool(_LONG_HTML_RE.match(stripped))
        or bool(_SHORT_HTML_RE.match(stripped))
    )


def is_marker_line(line: str) -> bool:
    """True when ``line`` IS a keep-marker line (line-anchored, B15).

    A line is a marker iff its stripped content exactly equals one of
    :data:`_MARKER_PHRASES`, or it is a bare short-form marker
    (``#...``, ``//...``, a lone ``…``). A marker phrase embedded
    mid-line — ``x = compute(a)  # ... see docs for details`` — is real
    content and returns False: such lines must survive the merge, and
    their presence must never suppress preserved-suffix emission.

    This predicate deliberately does NOT rely on ``normalize_markers``
    having run first: short-form markers are recognized directly, so
    callers that bypass normalization (e.g. ``deterministic_edit``
    reached via the CLI's fast deterministic-replace path) classify
    marker lines correctly instead of writing them into the file
    literally (TRDD-CMRMA2YG).
    """
    return is_marker_text(line.strip())


def normalize_markers(snippet: str) -> str:
    """Rewrite short/Unicode marker forms to the canonical long form.

    Accepts (per line, after leading-whitespace strip):

      * ``#...``              → ``# ... existing code ...``
      * ``//...``             → ``// ... existing code ...``
      * ``<!--...-->``        → ``# ... existing code ...`` (HTML short
        form; same canonical hash target as the other short forms)
      * ``…`` (U+2026)    → ``# ... existing code ...`` (hash form;
        ``is_marker_line`` recognizes the canonical form regardless of
        language, so one canonical form suffices)

    Legacy long-form markers (``# ... existing code ...``,
    ``// ... existing code ...``, and the HTML/XML
    ``<!-- ... existing code ... -->``) are passed through unchanged
    (spacing variants of the HTML long form are tightened to the exact
    canonical text).

    Indentation is preserved on the rewritten line so downstream indent
    arithmetic in ``deterministic_edit`` (which uses marker snippet
    indent to infer gap-body indent deltas) continues to match the
    surrounding snippet lines.

    This is a pure string transform — no parsing, no I/O, no side
    effects. Safe to call on any input.

    String awareness (B18): the per-line rewrite decision runs against a
    string-span-masked copy of the snippet (:func:`split_join.mask_string_spans`),
    so a line whose text lives inside a multi-line string (an embedded
    template or shell script) can never match a marker form and is passed
    through untouched. Masking is a same-length, 1:1 substitution that
    keeps every ``\\r``/``\\n`` in place, so the masked text splits into
    exactly the same lines as the original. A match against the masked body
    implies the marker text itself is unmasked — masking cannot forge the
    ``#``/``.``/``…`` characters of a marker — so the indent taken from the
    masked body is genuine when a rewrite fires.

    PERFORMANCE (governor mission, PART 1): masking the whole snippet is
    the expensive half (a same-length scan over every byte). Every rewrite
    form — exact short forms, spacing variants, the HTML forms — begins
    with ``#``, ``/``, ``<`` or ``…`` after leading-whitespace strip (the
    leading whitespace itself is never masked, so ``mask_string_spans``
    preserves it exactly), so a snippet whose lines carry none of those
    heads after their indent returns verbatim WITHOUT paying the mask. One
    cheap pre-scan, byte-identical outcome.
    """
    # Cheap pre-scan (see docstring): any line that could possibly be
    # rewritten must carry a marker head character after its indent. The
    # first non-space/tab character of a masked line equals the first
    # non-space/tab character of the raw line for any input.
    could_rewrite = False
    for raw in snippet.splitlines():
        stripped = raw.lstrip(" \t")
        if stripped[:1] in ("#", "/", "<", "…"):
            could_rewrite = True
            break
    if not could_rewrite:
        return snippet

    # Masked copy: in-string characters become NUL, structure unchanged.
    masked_lines = mask_string_spans(snippet).splitlines(keepends=True)
    out: list[str] = []
    for raw, masked_raw in zip(snippet.splitlines(keepends=True), masked_lines):
        # Separate the line body from its terminator so we can rewrite
        # the body without losing ``\n`` / ``\r\n``.
        if raw.endswith("\r\n"):
            body, term = raw[:-2], "\r\n"
        elif raw.endswith("\n"):
            body, term = raw[:-1], "\n"
        else:
            body, term = raw, ""
        masked_body = masked_raw[: len(body)]

        # Decide on the MASKED body: in-string content is NUL-ed out and
        # can never equal a marker form (B18).
        indent_len = len(masked_body) - len(masked_body.lstrip())
        indent = masked_body[:indent_len]
        stripped = masked_body[indent_len:].rstrip()

        if stripped == "#...":
            out.append(indent + _CANONICAL_HASH_MARKER + term)
        elif stripped == "//...":
            out.append(indent + _CANONICAL_SLASH_MARKER + term)
        elif stripped == "…":
            out.append(indent + _CANONICAL_HASH_MARKER + term)
        elif stripped == "<!--...-->":
            # HTML short form → the canonical hash marker (same rewrite
            # target as the other short forms: one canonical form suffices
            # because is_marker_line recognizes the canonical form
            # regardless of language).
            out.append(indent + _CANONICAL_HASH_MARKER + term)
        elif _SHORT_HASH_RE.match(masked_body) and "existing" not in masked_body:
            # Covers spacing variants like ``# ...`` / ``#  ...`` that
            # aren't full legacy long-form markers. ``is_marker_line``
            # already accepts these, but normalizing here keeps the two
            # code paths consistent and makes the position-semantics
            # check below simpler.
            out.append(indent + _CANONICAL_HASH_MARKER + term)
        elif _SHORT_SLASH_RE.match(masked_body) and "existing" not in masked_body:
            out.append(indent + _CANONICAL_SLASH_MARKER + term)
        elif _UNICODE_ELLIPSIS_RE.match(masked_body):
            out.append(indent + _CANONICAL_HASH_MARKER + term)
        elif _LONG_HTML_RE.match(masked_body.strip()):
            # Spacing variants of the canonical HTML long form are
            # rewritten to the exact canonical HTML text (it is already a
            # member of _MARKER_PHRASES, so this only tightens spacing).
            out.append(indent + _CANONICAL_HTML_MARKER + term)
        else:
            out.append(raw)
    return "".join(out)
