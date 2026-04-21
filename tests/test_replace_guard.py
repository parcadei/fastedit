"""Tests for the replace= multi-symbol guard in chunked_merge.

A snippet used with replace='X' must define at most X itself — not X plus
additional new symbols. The earlier bug: snippets with two functions under
replace= silently broke speculative decoding and burned minutes of AR.
"""

from __future__ import annotations

import pytest

from fastedit.inference.chunked_merge import chunked_merge


TS_SOURCE = """\
const DAY_MS = 86400000

interface FlyNode { id: string }
interface Bucket { label: string; items: FlyNode[] }

function bucketByTime(items: FlyNode[], dateFor: (n: FlyNode) => string, order: 'newest' | 'oldest'): Bucket[] {
  const sorted = [...items].sort((a, b) => dateFor(a).localeCompare(dateFor(b)))
  return [{ label: 'All', items: sorted }]
}

export function SessionsPage() { return null }
"""

BAD_SNIPPET = """\
function bucketByTime(items: FlyNode[], dateFor: (n: FlyNode) => string, order: 'newest' | 'oldest'): Bucket[] {
  const today: FlyNode[] = []
  return [{ label: 'Today', items: today }]
}

function parseMeta(node: FlyNode): Record<string, unknown> {
  return {}
}
"""

GOOD_SNIPPET = """\
function bucketByTime(items: FlyNode[], dateFor: (n: FlyNode) => string, order: 'newest' | 'oldest'): Bucket[] {
  return [{ label: 'Only', items: [...items] }]
}
"""


def _no_model(*_args, **_kwargs):
    """Merge fn that fails loudly if called — guard must short-circuit."""
    raise AssertionError(
        "merge_fn should NOT be called — guard must short-circuit before the model"
    )


@pytest.fixture
def ts_file(tmp_path):
    path = tmp_path / "SessionsPage.tsx"
    path.write_text(TS_SOURCE)
    return str(path)


def test_replace_rejects_multi_symbol_snippet(ts_file):
    """replace='bucketByTime' with a snippet defining bucketByTime AND parseMeta
    must raise ValueError before reaching the model."""
    with pytest.raises(ValueError) as exc_info:
        chunked_merge(
            original_code=TS_SOURCE,
            snippet=BAD_SNIPPET,
            file_path=ts_file,
            merge_fn=_no_model,
            language="typescript",
            replace="bucketByTime",
        )
    msg = str(exc_info.value)
    assert "parseMeta" in msg, f"error should mention the extra symbol: {msg}"
    assert "fast_batch_edit" in msg, f"error should suggest fast_batch_edit: {msg}"
    assert "bucketByTime" in msg


def test_replace_single_symbol_passes_guard(ts_file):
    """replace='bucketByTime' with a snippet defining only bucketByTime must
    pass the guard. The deterministic text-match path resolves this edit
    without calling the model at all."""
    result = chunked_merge(
        original_code=TS_SOURCE,
        snippet=GOOD_SNIPPET,
        file_path=ts_file,
        merge_fn=_no_model,
        language="typescript",
        replace="bucketByTime",
    )
    assert result.model_tokens == 0, "deterministic path should handle this"
    assert "Only" in result.merged_code, "new content should be present"
    assert "bucketByTime" in result.merged_code


def test_after_path_accepts_multi_symbol_snippet(ts_file):
    """after='X' is pure text splice — any number of definitions in the snippet
    is fine. The multi-symbol guard must NOT apply here.

    Note: language=None skips post-splice parse validation (which would
    require tree_sitter_typescript). We're testing the guard, not the parser.
    """
    result = chunked_merge(
        original_code=TS_SOURCE,
        snippet=BAD_SNIPPET,
        file_path=ts_file,
        merge_fn=_no_model,
        language=None,
        after="bucketByTime",
    )
    assert result.model_tokens == 0, "after= should not invoke the model"
    assert "parseMeta" in result.merged_code
    assert "bucketByTime" in result.merged_code


def test_replace_error_message_is_actionable(ts_file):
    """Error message must tell the user exactly how to fix the call."""
    with pytest.raises(ValueError) as exc_info:
        chunked_merge(
            original_code=TS_SOURCE,
            snippet=BAD_SNIPPET,
            file_path=ts_file,
            merge_fn=_no_model,
            language="typescript",
            replace="bucketByTime",
        )
    msg = str(exc_info.value)
    # Must name the offending symbol
    assert "parseMeta" in msg
    # Must suggest the right tool
    assert "fast_batch_edit" in msg
    # Must show both the replace and after examples
    assert "replace" in msg and "after" in msg
