"""C2 regression — brace-language model-path wraps were rejected forever.

Defect (found by the 100MB real-LLM stress tier, ``tests/
test_stress_100mb_llm.py``): a marker-bearing ``replace=`` snippet that
wraps a function body in a new brace block (``if x { ... } else { ... }``)
could NEVER converge against the real model — 9/9 attempts rejected — on
every ``{``-body language (rust, typescript, go, ...). Python's ``with``
wrap converged on the first attempt, which is why the small-file suites
(A1/A2) never saw it.

Root cause: the content-faithfulness validator's snippet scan
(:func:`fastedit.inference.chunked_merge._classify_snippet`) binds a
snippet line as a context anchor whenever its stripped content matches an
unconsumed original line. For a wrap snippet whose lines open more
brackets than they close, the trailing closer line (``    }``) matches the
function's own closer and was bound as CONTEXT — i.e. treated as a
restatement of the ORIGINAL closer. But that line is the NEW block's
closer (the snippet's own braces are unbalanced; a faithful merge must
emit it). The binding stole the anchor: the model's newly emitted closer
was paired with the anchor and the original's surviving closer was
orphaned outside every segment — read as an INVENTION — so the merge was
rejected, the corrective note changed nothing, and the budget exhausted.

Fix: when the snippet's own bracket balance is positive (an opened block
the snippet never closes), a pure-closer snippet line classifies NEW even
though its content matches the original closer. The original closer then
survives through the shared survivor LCS exactly as preserve-by-default
intends, and the declared closer accounts for the model's new one.

These tests are HERMETIC (scripted merge_fn — the plan's unit-tier
pattern): the real-model convergence itself is proven by the stress tier;
this file pins the validator contract that makes it possible.
"""

from __future__ import annotations

from types import SimpleNamespace

from fastedit.inference.chunked_merge import (
    _check_hallucinations,
    _classify_snippet,
    _tokens_declare_insertion_only,
    chunked_merge,
)

# A small rust file: alpha is the wrap target, beta guards the file tail so
# the alpha chunk never spans the whole file (the per-chunk loop runs).
RS_SOURCE = (
    "fn alpha(value_a: i32, value_b: i32) -> i32 {\n"
    "    let mut acc_alpha = value_a + value_b + 1; // alpha step 1\n"
    "    acc_alpha += 2; // alpha step 2\n"
    "    acc_alpha\n"
    "}\n"
    "\n"
    "fn beta(value_a: i32, value_b: i32) -> i32 {\n"
    "    value_a + value_b\n"
    "}\n"
)

# The wrap snippet: signature anchor, a new `if` opener, a preservation
# marker, the declared else block, the new block's closer, and a trailing
# preservation marker. Bracket balance: +1 (the `if` opener is never closed
# by the DECLARED lines' own opener... the `    }` closes the if/else — the
# signature's `{` is what stays open, which is exactly the C2 condition).
RS_WRAP_SNIPPET = (
    "fn alpha(value_a: i32, value_b: i32) -> i32 {\n"
    "    if value_a > 0 {\n"
    "        // ... existing code ...\n"
    "    } else {\n"
    "        0\n"
    "    }\n"
    "    // ... existing code ...\n"
)

# What a faithful model merge of the alpha chunk looks like: the body
# (without the original closer) re-indented inside the if, the declared
# else block, and the ORIGINAL closer preserved last.
RS_WRAPPED_CHUNK = (
    "fn alpha(value_a: i32, value_b: i32) -> i32 {\n"
    "    if value_a > 0 {\n"
    "        let mut acc_alpha = value_a + value_b + 1; // alpha step 1\n"
    "        acc_alpha += 2; // alpha step 2\n"
    "        acc_alpha\n"
    "    } else {\n"
    "        0\n"
    "    }\n"
    "}\n"
)

RS_WRAPPED_FILE = RS_SOURCE.replace(
    RS_SOURCE[: RS_SOURCE.index("\n\n")],
    RS_WRAPPED_CHUNK.rstrip("\n"),
)

# A BALANCED snippet (restates the closer as part of a full re-definition):
# its trailing `}` must keep binding as CONTEXT — the C2 rule must not fire.
RS_REDEF_SNIPPET = (
    "fn alpha(value_a: i32, value_b: i32) -> i32 {\n"
    "    let changed_alpha = value_a * 2;\n"
    "    changed_alpha\n"
    "}\n"
)


def _stub_merge(merged_code: str, tokens: int = 9):
    return SimpleNamespace(
        merged_code=merged_code,
        parse_valid=True,
        tokens_generated=tokens,
        latency_ms=1.0,
        truncated=False,
    )


def _content_lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


# ---------------------------------------------------------------------------
# The classifier contract (the fix itself)
# ---------------------------------------------------------------------------


def test_unbalanced_snippet_closer_classifies_new_not_context():
    """C2 rule: a snippet with unbalanced opens owns its trailing closer."""
    tokens = _classify_snippet(RS_WRAP_SNIPPET, _content_lines(RS_SOURCE))
    kinds = [(kind, value) for kind, _idx, value in tokens]
    assert ("new", "if value_a > 0 {") in kinds
    # The `    }` closing the declared if/else must be a DECLARED new line,
    # not a context anchor stealing the original closer's slot.
    assert ("new", "}") in kinds, kinds
    assert ("context", "fn alpha(value_a: i32, value_b: i32) -> i32 {") in kinds


def test_balanced_snippet_closer_still_binds_context():
    """No unbalanced opens → the old contract is untouched: the trailing `}`
    of a full re-definition restates the original closer (context)."""
    tokens = _classify_snippet(RS_REDEF_SNIPPET, _content_lines(RS_SOURCE))
    kinds = [(kind, value) for kind, _idx, value in tokens]
    assert ("context", "}") in kinds, kinds


# ---------------------------------------------------------------------------
# The validator accepts the faithful wrap it used to reject forever
# ---------------------------------------------------------------------------


def test_faithful_brace_wrap_merge_passes_the_battery():
    """The exact merge a faithful model produces scores clean."""
    assert _check_hallucinations(
        RS_SOURCE[: RS_SOURCE.index("\n\n")] + "\n",
        RS_WRAPPED_CHUNK,
        RS_WRAP_SNIPPET,
    ) == 1.0


def test_brace_wrap_lands_through_chunked_merge_with_scripted_model():
    """End to end: the faithful wrap is accepted on the FIRST attempt.

    Pre-fix this exhausted the retry budget (9/9 rejected) and kept the
    original file — the wrap could never land on any brace language.
    """
    calls: list[tuple[str, str, str | None]] = []

    def merge_fn(code, snippet, language):
        calls.append((code, snippet, language))
        return _stub_merge(RS_WRAPPED_CHUNK)

    result = chunked_merge(
        original_code=RS_SOURCE,
        snippet=RS_WRAP_SNIPPET,
        file_path="wrap_probe.rs",
        merge_fn=merge_fn,
        language="rust",
        replace="alpha",
    )
    assert result.model_tokens > 0, "the (scripted) model path must have run"
    assert result.chunks_rejected == 0, (
        f"the faithful wrap was rejected: retries={result.retries} "
        f"merged={result.merged_code!r}"
    )
    assert result.retries == 0, (
        f"the faithful wrap must land on the first attempt, got "
        f"{result.retries} retries"
    )
    assert result.parse_valid
    assert result.merged_code == RS_WRAPPED_FILE, result.merged_code
    assert calls, "merge_fn was never invoked"


# ---------------------------------------------------------------------------
# The rule must not over-accept: undeclared closers stay inventions
# ---------------------------------------------------------------------------


def test_extra_closer_against_balanced_snippet_is_still_rejected():
    """A balanced snippet declares NO new closer — the wrapped merge's extra
    else block is an invention and must keep failing the battery."""
    assert _check_hallucinations(
        RS_SOURCE[: RS_SOURCE.index("\n\n")] + "\n",
        RS_WRAPPED_CHUNK,
        RS_REDEF_SNIPPET,
    ) == 0.0


def test_dropped_original_closer_is_still_rejected():
    """With the C2 rule active, a merge that swallows the original closer
    (emitting only the model's new one) still fails: the deletion has no
    justification — preserve-by-default is not loosened by the fix."""
    dropped_closer = RS_WRAPPED_CHUNK[: RS_WRAPPED_CHUNK.rindex("}\n")] + "\n"
    assert _check_hallucinations(
        RS_SOURCE[: RS_SOURCE.index("\n\n")] + "\n",
        dropped_closer,
        RS_WRAP_SNIPPET,
    ) == 0.0


# ---------------------------------------------------------------------------
# C2 stress defect #2 — a wrap snippet licenses INSERTIONS, never deletions
# ---------------------------------------------------------------------------

# The real-model failure the 100MB stress tier caught: the model wrapped the
# body but DROPPED the target's docstring, and the validator RATIFIED it —
# the snippet's `with audit_lock:` (an identity-free new line declared before
# the marker) counted as front positional-deletion capacity, so the dropped
# docstring read as a "justified front replacement". A wrap declares that the
# WHOLE existing body moves inside the new scope; no deletion is licensed.
PY_CHUNK = (
    "def fn_121265(value_a, value_b):\n"
    '    """fn_121265: deterministic corpus function (plain)."""\n'
    "    acc_fn_121265 = value_a + value_b + 1  # fn_121265 step 1\n"
    "    return acc_fn_121265  # fn_121265 result\n"
)
PY_BODY_LINES = [
    '    """fn_121265: deterministic corpus function (plain)."""\n',
    "    acc_fn_121265 = value_a + value_b + 1  # fn_121265 step 1\n",
    "    return acc_fn_121265  # fn_121265 result\n",
]
PY_WRAP_SNIPPET = (
    "def fn_121265(value_a, value_b):\n"
    "    with audit_lock:\n"
    "        # ... existing code ...\n"
)
PY_WRAPPED_CHUNK = (
    "def fn_121265(value_a, value_b):\n"
    "    with audit_lock:\n"
    # every body line survives, one level deeper (uniform group shift, B14)
    + "".join("    " + line for line in PY_BODY_LINES)
)
PY_WRAPPED_MINUS_DOCSTRING = PY_WRAPPED_CHUNK.replace(
    "    " + PY_BODY_LINES[0], "",
)


def test_genuine_wrap_snippet_is_detected_as_insertion_only():
    """Any snippet whose NEW lines declare a block opener before the marker
    is insertion-only — the scope-introducing op (wrap or add-guard)
    licenses insertions, never deletions. Judged on classified tokens, so a
    restated ``def foo():`` signature (a context anchor ending in a colon)
    never triggers it."""

    def declares(snippet: str, original: str) -> bool:
        return _tokens_declare_insertion_only(
            _classify_snippet(snippet, _content_lines(original)),
        )

    assert declares(PY_WRAP_SNIPPET, PY_CHUNK)
    # An add-guard declares its own opener too: the guard is inserted at the
    # top of the body and every original line must survive beneath it.
    assert declares(
        "def foo():\n"
        "    if not ready:\n"
        "        return\n"
        "    # ... existing code ...\n",
        "def foo():\n    total = 1\n    return total\n",
    )
    # A peer-statement replacement idiom (the new line does NOT open a
    # block) keeps its positional justification — the accepted B4 idiom.
    assert not declares(
        "def foo():\n"
        "    x = 1\n"
        "    return x + 1\n"
        "# ... existing code ...\n",
        "def foo():\n    x = 1\n    return x\n",
    )
    # A restated signature's colon is a CONTEXT anchor, never a new opener.
    assert not declares(
        "def foo():\n"
        "# ... existing code ...\n"
        "    return total + 1\n",
        "def foo():\n    return total\n",
    )


def test_wrap_merge_that_drops_the_docstring_is_rejected():
    """The model's docstring-dropping 'wrap' must fail the battery.

    Pre-fix this scored 1.0: the positional fallback counted the wrap
    opener as front deletion capacity, so the model's unfaithful merge was
    accepted on every attempt and the 100MB stress test could never see a
    byte-exact wrap.
    """
    assert PY_WRAPPED_MINUS_DOCSTRING != PY_WRAPPED_CHUNK
    assert _check_hallucinations(
        PY_CHUNK, PY_WRAPPED_CHUNK, PY_WRAP_SNIPPET,
    ) == 1.0, "the faithful wrap itself must stay accepted"
    assert _check_hallucinations(
        PY_CHUNK, PY_WRAPPED_MINUS_DOCSTRING, PY_WRAP_SNIPPET,
    ) == 0.0, (
        "a wrap that drops the docstring must be rejected — the wrap "
        "snippet licenses insertions only, never deletions"
    )


# ---------------------------------------------------------------------------
# C3 stress defect — a merge that re-indents the CONTEXT ANCHORS themselves
# ---------------------------------------------------------------------------

# The real-model failure the C3 100MB seams stress caught (go, narrowed
# chunk): the model shifted HALF the restated body one tab deeper and the
# battery RATIFIED the merge — B14 checks indent deltas for surviving
# originals WITHIN segments, but the segment machinery splits the span at
# the context anchors, so the anchors' own indents were never checked. A
# selective re-indentation of anchors is the same corruption B14 exists for.
GO_CHUNK = (
    "func Fn000023(valueA int, valueB int) int {\n"
    "\tfor i := 0; i < 2; i++ {\n"
    "\t\tif valueA > 1 {\n"
    "\t\t\tacc += 1 // Fn000023 inner 1\n"
    "\t\t\tacc += 2 // Fn000023 inner 2\n"
    "\t\tacc += 1 // Fn000023 step 01\n"
    "\t\tacc += 2 // Fn000023 step 02\n"
    "\t\tacc += 3 // Fn000023 step 03\n"
    "\t\tacc += 4 // Fn000023 step 04\n"
    "\t\t}\n"
    "\t}\n"
    "}\n"
)
GO_SNIPPET = GO_CHUNK + "\tseamProbe := \"... existing code ...\"\n"
GO_CLEAN_APPEND = GO_CHUNK + "\tseamProbe := \"... existing code ...\"\n"
GO_ANCHOR_REINDENT = (
    "func Fn000023(valueA int, valueB int) int {\n"
    "\tfor i := 0; i < 2; i++ {\n"
    "\t\tif valueA > 1 {\n"
    "\t\t\tacc += 1 // Fn000023 inner 1\n"
    "\t\t\tacc += 2 // Fn000023 inner 2\n"
    "\t\t\tacc += 1 // Fn000023 step 01\n"
    "\t\t\tacc += 2 // Fn000023 step 02\n"
    "\t\t\tacc += 3 // Fn000023 step 03\n"
    "\t\t\tacc += 4 // Fn000023 step 04\n"
    "\t\t\t}\n"
    "\t\t}\n"
    "}\n"
    "\tseamProbe := \"... existing code ...\"\n"
)


def test_anchor_indent_shift_is_rejected():
    """C3: re-indenting the declared context anchors must fail the battery.

    The anchors are surviving originals; the per-segment B14 check never
    sees them (segments run BETWEEN anchors), so the go model's selective
    re-indentation of half the restated body used to score clean and be
    written. The same delta rule now runs across the anchor sequence.
    """
    assert _check_hallucinations(GO_CHUNK, GO_CLEAN_APPEND, GO_SNIPPET) == 1.0, (
        "the clean append itself must stay accepted"
    )
    assert _check_hallucinations(
        GO_CHUNK, GO_ANCHOR_REINDENT, GO_SNIPPET,
    ) == 0.0, (
        "a merge that selectively re-indents the context anchors must be "
        "rejected — the per-segment B14 check never covered them"
    )
