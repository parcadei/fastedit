"""CJK-aware trait heuristics for STRUCTURELESS text (Step D1, req. 6 + 9).

A *structureless* file (``language=None`` — ``.txt``, ``.log``, ...) has no
grammar to parse, so the A2 validation battery cannot run the relative
tree-sitter rule for it. This module is the trait oracle that takes its
place: it records COUNTABLE TRAITS of the input text, computes the traits
the declared op SHOULD produce, and compares them to the actual output
under a documented tolerance policy.

THE TRAIT MODEL (req. 9 — EDIT-NOT-CORRECT). Counts are computed from
INPUT + op, never from a "correct text" ideal: the validator compares
traits, it never corrects. A DEFECTIVE input is a trait like any other —
a duplicated paragraph, an unbalanced quote count or a pre-existing typo's
byte shape must survive an edit that does not target it (preserved garbage
PASSES), while an uncommanded change (a dropped paragraph, a quote that
vanished) FAILS with a reason naming the trait. Exactness of untouched
spelling is the CONTENT validator's job (line identity) plus the pipeline's
byte-exact splices; this oracle's job is everything the line-level view
cannot see — whole-file counts, layout lines and byte volume.

THE TRAITS — exact definitions (``text_traits`` returns one integer per
name, and :data:`TRAIT_NAMES` is the canonical order used for checks):

  ``blank_lines``    lines whose stripped content is empty. These are
                     exactly the lines the battery's content view skips
                     (they carry no comparable content), which is why this
                     trait is checked FIRST — it is the oracle's witness
                     for the one corruption class the line-level validator
                     is blind to.
  ``lines``          ``len(text.splitlines())`` — the SAME line model the
                     pipeline itself uses (``chunked_merge`` slices with
                     ``splitlines(keepends=True)``; a trailing terminator
                     never changes the count).
  ``words``          CJK-aware word count. The text is partitioned into
                     MAXIMAL CONTIGUOUS CJK RUNS and the non-CJK stretches
                     between them; each maximal CJK run counts as exactly
                     ONE word (快速编辑 → 1 word; the ideographic colon in
                     中文注释：快速 breaks the run → 2 words), and each
                     non-CJK stretch is split on whitespace with every
                     resulting token that contains at least one ALPHANUMERIC
                     character counting as one word — so digits count (a
                     bare ``2024`` is a word) and decorative runs of pure
                     punctuation (``---``, a standalone emoji) do not. An
                     emoji glued between letters does not split a token:
                     whitespace is the only splitter.
  ``cjk_chars``      characters in the CJK word ranges: Hiragana/Katakana,
                     CJK Extension A, the Unified Ideographs, Hangul
                     syllables, the Compatibility Ideographs and the
                     astral Extensions B-F (see ``_CJK_WORD_CLASS``).
  ``punct_cjk``      characters in the CJK punctuation classes: the CJK
                     Symbols and Punctuation block (。、《》「」…), the
                     fullwidth ASCII forms (，！？：；（）), and the em
                     dash / horizontal ellipsis / middle dot that Chinese
                     prose uses beyond those blocks.
  ``punct_ascii``    characters in ``.,;:!?`` — the sentence-level ASCII
                     punctuation the requirement names.
  ``quotes_straight`` ASCII ``"`` and ``'`` characters. An UNPAIRED quote
                     shifts the count, which is how a broken quote count in
                     the input is preserved (trait) and a new unbalanced
                     quote in the output is caught (regression).
  ``quotes_curly``   the paired curly forms “ ” ‘ ’ per character.
  ``digits``         ASCII ``0-9`` characters.
  ``bytes``          ``len(text.encode("utf-8"))`` — the byte volume the
                     untouched regions must account for.

THE ORACLE PATTERN (tests/corpus.py's doctrine): trait deltas are computed
on the spans the op touches — ``expected = original - removed + payload``.
The arithmetic is EXACT for line-aligned spans (the pipeline's ops always
are: line-splice semantics), including ``lines``/``blank_lines`` because a
line-aligned span carries its own terminators.

TOLERANCE POLICY (req. 6, "tight by default"):

  * ``lines`` is EXACT (±0) for line-anchored ops — every tolerance preset
    keeps it at zero unless the op DECLARES :attr:`TextOp.prose_rewrap`,
    whose invariant is the content, not the line breaks (documented band:
    25% with a floor of 2). ``blank_lines`` is exact with it.
  * count traits (words/cjk/punct/quotes/digits/bytes) get ``relative``
    (a fraction of the EXPECTED count) with an ``absolute_floor``:
    :data:`TOLERANCE_EXACT` (0/0) for byte-exact ops and deterministic
    callers; :data:`TOLERANCE_MODEL_PROSE` (±2%, floor 2) for merges a
    model produced. Justification (measured on the real fastedit mlx-8bit
    model — tests/test_real_llm_text.py runs the probe): a faithful merge
    echoes preserved lines byte-exactly, so the observed drift is zero;
    the small band exists so a bounded reflow (a layout blank at a merge
    seam, a trimmed run of trailing spaces inside the band) retries
    instead of hard-failing, while any content-sized drift (a dropped
    paragraph is tens of words / hundreds of bytes) stays far outside it.
  * ``layout_slack`` (a caller-supplied widening of ``lines`` AND
    ``blank_lines`` by the same amount) models OP-SIDE uncertainty: the
    battery passes the snippet's own blank-line count, because a snippet
    blank is either echoed as new layout or already present among the
    preserved originals and the classifier cannot tell which.
  * ``removable_traits`` lowers the expected FLOOR (never the ceiling) by
    the per-trait size of the span(s) the op's shape could justify
    removing — the snippet-derived removal capacity the battery computes.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass

__all__ = [
    "TOLERANCE_EXACT",
    "TOLERANCE_MODEL_PROSE",
    "TRAIT_NAMES",
    "TextOp",
    "TraitTolerance",
    "expected_traits_after_op",
    "text_traits",
    "validate_text_output",
]

# ---------------------------------------------------------------------------
# Character classes (documented in the module docstring)
# ---------------------------------------------------------------------------

# Writing systems written WITHOUT spaces — each maximal contiguous run is
# one word for the CJK-aware word count.
_CJK_WORD_CLASS = (
    "\u3040-\u30ff"          # Hiragana + Katakana
    "\u3400-\u4dbf"          # CJK Extension A
    "\u4e00-\u9fff"          # CJK Unified Ideographs
    "\uac00-\ud7af"          # Hangul syllables
    "\uf900-\ufaff"          # CJK Compatibility Ideographs
    "\U00020000-\U0002fa1f"  # Extensions B-F + Compatibility Supplement
)

# CJK punctuation: the CJK Symbols and Punctuation block, the fullwidth
# ASCII forms, and the em dash / ellipsis / middle dot of Chinese prose.
# Deliberately DISJOINT from the quote classes (fullwidth and curly quotes
# are counted by their own traits, never twice).
_CJK_PUNCT_CLASS = (
    "\u3000-\u303f"          # CJK Symbols and Punctuation (。、《》「」…)
    "\uff01-\uff0f"          # fullwidth !"#&'()*+,-./
    "\uff1a-\uff1f"          # fullwidth :;<=>?
    "\uff3b-\uff40"          # fullwidth [\]^_`
    "\uff5b-\uff65"          # fullwidth {|}~ + halfwidth ideographic marks
    "\u2014\u2026\u00b7"     # em dash, horizontal ellipsis, middle dot
)

_CURLY_QUOTE_CLASS = "\u2018\u2019\u201c\u201d"  # ‘ ’ “ ”

# One scan over the text classifies every counted character; the classes
# are pairwise disjoint, so the alternation order cannot double-count.
_CHAR_CLASS_RE = re.compile(
    f"(?P<punct_cjk>[{_CJK_PUNCT_CLASS}])"
    f"|(?P<punct_ascii>[.,;:!?])"
    "|(?P<quotes_straight>[\"'])"
    f"|(?P<quotes_curly>[{_CURLY_QUOTE_CLASS}])"
    "|(?P<digits>[0-9])"
)
_CJK_RUN_RE = re.compile(f"[{_CJK_WORD_CLASS}]+")
# Whitespace-delimited tokens of the NON-CJK stretches (CJK runs are
# blanked out before tokenizing — see text_traits).
_WHITESPACE_TOKEN_RE = re.compile(r"\S+")

TRAIT_NAMES: tuple[str, ...] = (
    "blank_lines", "lines", "words", "cjk_chars",
    "punct_cjk", "punct_ascii", "quotes_straight",
    "quotes_curly", "digits", "bytes",
)
"""The traits, in CHECK ORDER.

``blank_lines`` first: it is the sharpest diagnostic for the content
view's blind spot (layout lines), so its reason pinpoints layout
destruction instead of the blunter line-count headline. ``bytes`` last:
it aggregates every other drift and is only reached when no finer trait
explains the deviation.
"""


def text_traits(text: str) -> dict[str, int]:
    """Compute the trait vector of ``text`` (definitions in the docstring).

    Pure function — no parsing, no grammar, no I/O. Every value is an
    exact integer over the definitions above; the empty text has all-zero
    traits.
    """
    lines = text.splitlines()
    traits: dict[str, int] = {
        "blank_lines": sum(1 for line in lines if not line.strip()),
        "lines": len(lines),
        "words": 0,
        "cjk_chars": 0,
        "punct_cjk": 0,
        "punct_ascii": 0,
        "quotes_straight": 0,
        "quotes_curly": 0,
        "digits": 0,
        "bytes": len(text.encode("utf-8")),
    }

    # CJK-aware word count: each maximal CJK run is ONE word; the runs are
    # blanked out of a residue copy whose whitespace tokens (with at least
    # one alphanumeric character) are the remaining words.
    words = 0
    cjk_chars = 0
    residue_parts: list[str] = []
    last = 0
    for match in _CJK_RUN_RE.finditer(text):
        words += 1
        cjk_chars += match.end() - match.start()
        residue_parts.append(text[last:match.start()])
        residue_parts.append(" ")
        last = match.end()
    residue_parts.append(text[last:])
    for token in _WHITESPACE_TOKEN_RE.finditer("".join(residue_parts)):
        if any(ch.isalnum() for ch in token.group()):
            words += 1
    traits["words"] = words
    traits["cjk_chars"] = cjk_chars

    # Single scan for the per-character classes.
    for match in _CHAR_CLASS_RE.finditer(text):
        traits[match.lastgroup] += 1
    return traits


# ---------------------------------------------------------------------------
# The op spec and the trait oracle
# ---------------------------------------------------------------------------

_OP_KINDS = frozenset({"insert", "replace", "delete", "append", "none"})


@dataclass(frozen=True)
class TextOp:
    """A declared text edit — the op spec the trait oracle transforms.

    Attributes:
        kind: one of ``insert``/``replace``/``delete``/``append``/``none``
            (``append`` is the EOF flavor of ``insert``; the arithmetic is
            identical). Unknown kinds fail loudly.
        payload: the EXACT text the op declares adding. LINE-ALIGNED: the
            pipeline's ops always are (line-splice semantics — each span
            starts at a line start and every line it adds carries its own
            terminator), and the arithmetic below relies on it.
        removed: the EXACT original span the op declares removing (same
            line-alignment contract). An insertion-shaped op leaves it
            empty; its removal capacity is then carried separately by
            ``validate_text_output``'s ``removable_traits``.
        prose_rewrap: declares that the op may re-flow line breaks inside
            the payload (the invariant is the content, not the breaks) —
            the only documented escape from the exact line-count rule.
    """

    kind: str = "none"
    payload: str = ""
    removed: str = ""
    prose_rewrap: bool = False

    def __post_init__(self) -> None:
        if self.kind not in _OP_KINDS:
            raise ValueError(
                f"unknown TextOp kind {self.kind!r}; "
                f"expected one of {sorted(_OP_KINDS)}",
            )


def expected_traits_after_op(original_text: str, op: TextOp) -> dict[str, int]:
    """The traits the output SHOULD have — original, transformed by the op.

    Pure arithmetic over the trait vector (the tests/corpus.py oracle
    pattern: deltas are computed on the spans the op touches)::

        expected[trait] = original[trait] - removed[trait] + payload[trait]

    An op with no ``removed``/``payload`` text contributes zero for that
    side, so a ``none`` op expects the original's own traits and a
    defective input's traits are expected to survive (req. 9): the oracle
    never asks for a "correct" text, only for the declared delta.

    Preconditions: ``op.removed`` and ``op.payload`` are line-aligned
    spans (see :class:`TextOp`) — under that contract the arithmetic is
    exact for every trait, ``lines`` and ``blank_lines`` included.
    """
    original = text_traits(original_text)
    removed = text_traits(op.removed) if op.removed else {}
    payload = text_traits(op.payload) if op.payload else {}
    return {
        trait: (
            original[trait] - removed.get(trait, 0) + payload.get(trait, 0)
        )
        for trait in TRAIT_NAMES
    }


@dataclass(frozen=True)
class TraitTolerance:
    """Per-trait comparison band for :func:`validate_text_output`.

    Attributes:
        relative: the ± fraction of the EXPECTED count granted to the
            count traits (words/cjk/punct/quotes/digits/bytes). ``lines``
            and ``blank_lines`` never get it — they are exact for
            line-anchored ops.
        absolute_floor: minimum absolute slack for the count traits, so
            tiny counts still get a meaningful band.
        layout_floor: extra ± slack for ``lines`` AND ``blank_lines``
            (model seam blanks). Zero in :data:`TOLERANCE_EXACT` — the
            validator-level line rule stays exact.
        rewrap_lines_relative / rewrap_lines_floor: the band ``lines``
            gets when the op declares :attr:`TextOp.prose_rewrap`.
    """

    relative: float = 0.0
    absolute_floor: int = 0
    layout_floor: int = 0
    rewrap_lines_relative: float = 0.25
    rewrap_lines_floor: int = 2

    def slack(
        self,
        trait: str,
        expected: int,
        op: TextOp,
        layout_slack: int = 0,
    ) -> int:
        """The ± slack for one trait under this policy."""
        if trait in ("lines", "blank_lines"):
            if trait == "lines" and op.prose_rewrap:
                return max(
                    self.rewrap_lines_floor,
                    math.ceil(self.rewrap_lines_relative * max(expected, 1)),
                )
            # Line counts are exact for line-anchored ops; the only width
            # is the caller's op-side layout uncertainty plus the policy's
            # seam floor.
            return layout_slack + self.layout_floor
        return max(self.absolute_floor, math.ceil(self.relative * abs(expected)))


TOLERANCE_EXACT = TraitTolerance()
"""Byte-exact ops and deterministic callers: every trait exact."""

TOLERANCE_MODEL_PROSE = TraitTolerance(
    relative=0.02, absolute_floor=2, layout_floor=1,
)
"""Merges a model produced over prose.

±2% of the expected count (floor ±2) for the count traits, ±1 layout
blank for ``lines``/``blank_lines``. Defaults justified against the real
model (see the module docstring's tolerance section and the measured
probe in tests/test_real_llm_text.py): faithful merges show ZERO trait
drift, so the band only has to absorb bounded seam noise — it stays far
below any content-sized corruption.
"""


def validate_text_output(
    original_text: str,
    op: TextOp,
    output_text: str,
    tolerance: TraitTolerance = TOLERANCE_EXACT,
    *,
    layout_slack: int = 0,
    removable_traits: dict[str, int] | None = None,
    trait_names: Iterable[str] = TRAIT_NAMES,
) -> tuple[bool, str]:
    """Compare ``output_text``'s traits against the op-transformed input.

    Args:
        original_text: the input text the op applies to.
        op: the declared op spec (:class:`TextOp`).
        output_text: the actual merge output.
        tolerance: the tolerance policy (default byte-exact).
        layout_slack: op-side uncertainty in LAYOUT lines — the battery
            passes the snippet's own blank-line count (a snippet blank may
            be echoed as new layout or already present among the preserved
            originals); it widens ``lines`` and ``blank_lines`` equally.
        removable_traits: per-trait size of the span(s) the op's shape
            could justify removing. Lowers the expected FLOOR (never the
            ceiling): the output may legitimately be smaller by exactly
            that much, while anything larger is an uncommanded loss.
        trait_names: the traits to check, in order (default
            :data:`TRAIT_NAMES`); the first failing trait's reason wins.

    Returns:
        ``(True, "")`` when every checked trait sits inside its band, else
        ``(False, reason)`` with a human-readable reason naming the trait,
        the observed count and the expected range — the string the retry
        loop feeds back to the model.
    """
    expected = expected_traits_after_op(original_text, op)
    actual = text_traits(output_text)
    removable = removable_traits or {}
    for trait in trait_names:
        exp = expected[trait]
        act = actual[trait]
        slack = tolerance.slack(trait, exp, op, layout_slack)
        floor = exp - slack - removable.get(trait, 0)
        ceiling = exp + slack
        if floor <= act <= ceiling:
            continue
        detail = (
            f"{trait} count {act} is outside the expected range "
            f"{floor}..{ceiling} (expected {exp}, tolerance ±{slack}"
        )
        if removable.get(trait):
            detail += (
                f", minus up to {removable[trait]} the op may remove"
            )
        detail += ")"
        return False, detail
    return True, ""
