"""Shared per-file write gates for the chunk-rejection convention.

Step 18 (B34) extracted the MCP edit tools' hallucination gate so every
fastedit surface — MCP ``fast_edit``/``fast_batch_edit``/``fast_multi_edit``
and the CLI's ``edit``/``batch-edit``/``multi-edit`` model paths — enforces
the SAME signal with the SAME message: a merge whose chunks were ALL
rejected as hallucinations is refused, the file is never written, and the
refusal names the size/split remedy.

The convention this gate consumes lives in ``chunked_merge``: on
retry-exhaustion the merge returns ``merged_code`` set to the ORIGINAL
file with ``parse_valid=False`` and ``chunks_rejected >= chunks_used`` —
error-shaped, so a parse gate alone cannot refuse it (the original parses
like the original, by definition). Only the chunk accounting can.
"""

from __future__ import annotations


def _all_chunks_rejected(result) -> bool:
    """True when every chunk the merge used was rejected as a hallucination.

    The ``> 0`` guard matters: a zero-model batch (pure ``after=`` /
    ``preserve_siblings=`` splices report ``chunks_used == 0``) must not
    read as "everything rejected" via ``0 >= 0``.
    """
    rejected = getattr(result, "chunks_rejected", 0)
    return rejected > 0 and rejected >= result.chunks_used


def _rejection_refusal(result, metrics: str) -> str:
    """Fail-loud refusal for an all-chunks-rejected merge. Never
    force-overridable: a hallucinated merge has no safe interpretation.

    Issue #12(5): the refusal also says what a follow-up ``fastedit diff``
    will show — nothing — so a diff after the refusal never reads as if the
    edit might have half-applied ("No backup recorded ..."). The clause
    lives HERE, in the shared refusal both the MCP tools and the CLI print
    verbatim, so the CLI/MCP wording stays byte-identical.
    """
    return (
        f"Error: edit rejected — model hallucinated on {result.chunks_rejected} chunk(s). "
        f"File unchanged. The function may be too large ({result.chunks_used} chunk(s)) "
        f"for the 1.7B model. Try a smaller edit or split the function. {metrics}"
        f" — `fastedit diff` will show no changes."
    )
