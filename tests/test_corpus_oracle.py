"""C1 oracle self-test — the corpus generator and its independent oracle.

Default tier (fast, hermetic, no 100MB runs — those are C2/C3 under the
``llm``/``stress`` markers). Covers the C1 self-test list:

(a) determinism — same seed -> byte-identical output twice;
(b) size accuracy — 1MB/5MB targets land within tolerance (actual reported
    in the assertion messages);
(c) oracle round-trip — ``apply_op_oracle`` then ``inverse_op`` regenerates
    the original byte-for-byte for EVERY op kind x {LF, CRLF,
    no-final-EOL} x {ascii, CJK-sprinkled};
(d) cross-check — for the committed MEDIUM goldens (tests/golden/big/), the
    oracle's expected output equals the committed golden file, and the
    committed original is byte-exactly regenerable from the manifest's
    seed recipe (a stale fixture fails loudly instead of pinning wrong
    behavior);
(e) the generator emits <=150-line symbols (verified from the manifest
    against the text);
(f) CJK content survives a no-op round trip byte-exactly.

Plus the strongest check of the oracle contract: the oracle's output is
compared BYTE-EXACTLY against the real zero-token deterministic pipeline
paths (fast-path insert / direct swap / AST delete) across op kinds and
line-ending conventions — B3's matrix doctrine, scaled to corpus ops. No
model is ever loaded; the merge callback raises if the model path were
ever taken.

The corpus module (tests/corpus.py) never imports fastedit; THIS test may
import it to pin the extra guarantee that generated corpora actually parse
clean (C2/C3 run the pipeline against them).
"""

from __future__ import annotations

import json
from pathlib import Path

import corpus
import pytest
from corpus import (
    GOLDEN_BIG_CASES,
    OP_KINDS,
    apply_line_op,
    apply_op_oracle,
    apply_op_sequence,
    build_corpus_case,
    generate_big_source,
    inverse_op,
    line_op_from_manifest,
    op_from_manifest,
    op_to_manifest,
    verify_symbol_spans,
)

GOLDEN_BIG_DIR = Path(__file__).parent / "golden" / "big"

# Bound on how far short of the byte target the generator may stop: one
# declined symbol (max ~135 lines) plus the end-with-a-function fixup.
_SIZE_TOLERANCE_BYTES = 16_384

_EOL_VARIANTS = (
    pytest.param(
        {"id": "lf-python", "eol": "\n", "final_eol": True, "language": "python"},
        id="lf-python",
    ),
    pytest.param(
        {"id": "crlf-python", "eol": "\r\n", "final_eol": True,
         "language": "python"},
        id="crlf-python",
    ),
    pytest.param(
        {"id": "noeol-rust", "eol": "\n", "final_eol": False, "language": "rust"},
        id="noeol-rust",
    ),
)


# ---------------------------------------------------------------------------
# (a) determinism
# ---------------------------------------------------------------------------


def test_generation_is_byte_deterministic():
    first = generate_big_source("python", 300_000, seed=7)
    second = generate_big_source("python", 300_000, seed=7)
    assert isinstance(first, str)
    assert first == second, "same seed must produce byte-identical output"
    assert first.manifest == second.manifest, "the manifest must be stable too"
    other = generate_big_source("python", 300_000, seed=8)
    assert first != other, "different seeds must not coincide (sanity)"


# ---------------------------------------------------------------------------
# (b) size accuracy — actual sizes reported in the failure messages
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("language", "target"),
    [pytest.param("python", 1_000_000, id="1mb-python"),
     pytest.param("rust", 5_000_000, id="5mb-rust")],
)
def test_size_accuracy_within_tolerance(language, target):
    source = generate_big_source(language, target, seed=11)
    actual = len(source.encode("utf-8"))
    assert actual <= target, (
        f"{language} corpus overshot the {target}-byte target: actual={actual}"
    )
    under_run = target - actual
    assert under_run <= _SIZE_TOLERANCE_BYTES, (
        f"{language} corpus under-ran the {target}-byte target by "
        f"{under_run} bytes (tolerance {_SIZE_TOLERANCE_BYTES}); actual={actual}"
    )


def test_size_accuracy_small_target_still_yields_targets():
    # Tiny budgets must still emit enough symbols for op targeting.
    source = generate_big_source("go", 30_000, seed=3)
    assert source.manifest.symbol_count >= 8
    assert len(source.encode("utf-8")) >= 30_000 - _SIZE_TOLERANCE_BYTES


# ---------------------------------------------------------------------------
# (c) oracle round-trip: op kind x EOL convention x CJK sprinkle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", OP_KINDS, ids=lambda k: k)
@pytest.mark.parametrize("sprinkle_cjk", [False, True], ids=["ascii", "cjk"])
@pytest.mark.parametrize("variant", _EOL_VARIANTS)
def test_oracle_round_trip_regenerates_original(kind, sprinkle_cjk, variant):
    case = build_corpus_case(
        variant["language"], 48_000, seed=101, sprinkle_cjk=sprinkle_cjk,
        kinds=(kind,), eol=variant["eol"], final_eol=variant["final_eol"],
    )
    original = str(case.source)
    assert len(case.ops) == 1
    op = case.ops[0]
    assert op.kind == kind

    edited = apply_op_oracle(original, op)
    assert edited != original, f"{kind} must change the bytes"
    # the payload actually landed, in the file's own convention
    if kind in ("insert_after_symbol", "append_at_eof"):
        assert "fn_ins_101_a" in edited or "FnIns101A" in edited \
            or "fn_ins_101_b" in edited or "FnIns101B" in edited
    elif kind == "replace_symbol_body":
        assert "edited step 1" in edited
        assert "fn_ins" not in edited
    else:
        assert op.symbol not in edited, "deleted symbol must be gone"
    if variant["eol"] == "\r\n":
        assert "\r\n" in edited and edited.count("\n") == edited.count("\r\n")
    else:
        assert "\r" not in edited
    if not variant["final_eol"] and kind != "append_at_eof":
        assert not edited.endswith(("\n", "\r")), (
            "a non-EOF edit must preserve the missing final EOL"
        )

    # THE round trip: edited -> inverse -> original, byte for byte.
    restored = apply_line_op(edited, inverse_op(op))
    assert restored == original, (
        f"{kind} ({variant['id']}, cjk={sprinkle_cjk}): inverse_op did not "
        f"regenerate the original byte-exactly"
    )


def test_op_sequence_backward_reconstruction():
    """All four ops applied in order; inverses in reverse restore the bytes."""
    for kwargs in (
        {"language": "python", "eol": "\n", "final_eol": True,
         "sprinkle_cjk": False},
        {"language": "rust", "eol": "\n", "final_eol": False,
         "sprinkle_cjk": False},
        {"language": "go", "eol": "\n", "final_eol": True,
         "sprinkle_cjk": True},
    ):
        case = build_corpus_case(kwargs["language"], 60_000, seed=19, **{
            key: value for key, value in kwargs.items() if key != "language"
        })
        original = str(case.source)
        assert len(case.ops) == len(OP_KINDS)
        edited, inverses = apply_op_sequence(original, case.ops)
        assert edited != original
        for line_op in reversed(inverses):
            edited = apply_line_op(edited, line_op)
        assert edited == original, f"sequence inverse failed for {kwargs}"


# ---------------------------------------------------------------------------
# (d) committed MEDIUM goldens agree with the independent oracle
# ---------------------------------------------------------------------------


def _golden_case_dirs():
    manifests = sorted(GOLDEN_BIG_DIR.glob("*/manifest.json"))
    assert manifests, (
        "tests/golden/big/ carries no committed corpus goldens — "
        "regenerate them: uv run python tests/golden/big/_generate.py"
    )
    return manifests


@pytest.mark.parametrize(
    "manifest_path", _golden_case_dirs(), ids=lambda p: p.parent.name,
)
def test_medium_golden_matches_oracle_and_seed(manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    case_dir = manifest_path.parent
    assert manifest["schema"] == corpus.CORPUS_GOLDEN_SCHEMA

    # The manifest's recipe must be one of the PINNED cases — a hand-edited
    # recipe would quietly detach the fixture from the generator.
    pinned = {entry["case"]: entry for entry in GOLDEN_BIG_CASES}
    entry = pinned[manifest["case"]]
    assert manifest["recipe"] == {
        key: value for key, value in entry.items() if key != "case"
    }

    original_bytes = (case_dir / manifest["filename"]).read_bytes()

    # (d1) the committed original is byte-exactly regenerable from the seed.
    regenerated = build_corpus_case(**manifest["recipe"])
    assert regenerated.source.encode("utf-8") == original_bytes, (
        f"{manifest['case']}: committed original drifted from the seed recipe "
        f"— regenerate the goldens (uv run python "
        f"tests/golden/big/_generate.py)"
    )
    # ...and the committed ops are exactly the regenerated ones.
    for op, op_dict in zip(
        regenerated.ops, manifest["ops"], strict=True,
    ):
        assert op_to_manifest(op, op_dict["expected"]) == op_dict, (
            f"{manifest['case']}: committed op record drifted from the "
            f"generator (op {op['kind']} {op['symbol']})"
        )

    original_text = original_bytes.decode("utf-8")
    assert regenerated.source.manifest.max_symbol_lines <= corpus.MAX_SYMBOL_LINES

    # (d2) every committed expected file equals the oracle's output for its
    # op, and the inverse regenerates the committed original byte-exactly.
    for op_dict in manifest["ops"]:
        op = op_from_manifest(op_dict)
        expected = (case_dir / op_dict["expected"]).read_bytes()
        got = apply_op_oracle(original_text, op).encode("utf-8")
        assert got == expected, (
            f"{manifest['case']}: committed expected file "
            f"{op_dict['expected']} drifted from the oracle — regenerate the "
            f"goldens (uv run python tests/golden/big/_generate.py)"
        )
        restored = apply_line_op(got.decode("utf-8"), inverse_op(op))
        assert restored == original_text, (
            f"{manifest['case']}: inverse of {op_dict['op']} "
            f"{op_dict['symbol']} did not regenerate the committed original"
        )
        # the serialized inverse data matches the recomputed inverse
        assert line_op_from_manifest(op_dict["inverse"]) == inverse_op(op)


# ---------------------------------------------------------------------------
# (e) symbol spans: <= the parent-snap cap, verified from the manifest
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "language", ["python", "rust", "go", "typescript"], ids=str,
)
def test_symbol_spans_within_parent_snap_cap(language):
    source = generate_big_source(language, 120_000, seed=13)
    verify_symbol_spans(str(source), source.manifest)  # raises on any drift
    max_lines = source.manifest.max_symbol_lines
    assert max_lines <= corpus.MAX_SYMBOL_LINES, (
        f"{language}: a symbol spans {max_lines} lines, above the "
        f"{corpus.MAX_SYMBOL_LINES}-line parent-snap cap"
    )


# ---------------------------------------------------------------------------
# (f) CJK content survives a no-op round trip byte-exactly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("language", "indent_note"),
    [pytest.param("python", "4-space"), pytest.param("go", "tab")], ids=str,
)
def test_cjk_survives_noop_round_trip(language, indent_note):
    source = generate_big_source(language, 60_000, seed=17, sprinkle_cjk=True)
    text = str(source)
    # the sprinkle really happened, and no lossy replacement crept in
    assert "中文注释" in text, f"{language}: no CJK comment line found"
    assert "-fastedit-corpus-" not in text
    assert "\ufffd" not in text, "U+FFFD replacement character in generated text"
    # no-op round trip through the line machinery the oracle uses
    assert "".join(text.splitlines(keepends=True)) == text
    # no-op round trip through the byte machinery the write path uses
    assert text.encode("utf-8").decode("utf-8") == text
    verify_symbol_spans(text, source.manifest)


# ---------------------------------------------------------------------------
# Oracle <-> pipeline agreement: the oracle encodes the deterministic paths'
# DOCUMENTED splice semantics — prove it against the real zero-token paths.
# (This is B3's matrix doctrine, scaled to corpus ops; no model is ever
# loaded and the merge callback raises if the model path were ever taken.)
# ---------------------------------------------------------------------------


def _no_model(*_args, **_kwargs):
    raise AssertionError(
        "the model path ran on a deterministic corpus op — the corpus oracle "
        "only encodes the zero-token deterministic paths",
    )


def _pipeline_bytes(original_text: str, op, language: str, ext: str, tmp_path):
    """Run one corpus op through its real deterministic pipeline path."""
    from fastedit.inference.chunked_merge import chunked_merge
    from fastedit.inference.symbols import delete_symbol

    tmp_file = tmp_path / f"corpus.{ext}"
    tmp_file.write_bytes(original_text.encode("utf-8"))
    if op.kind == "delete_symbol":
        return delete_symbol(str(tmp_file), op.symbol, language=language).merged_code
    if op.kind == "replace_symbol_body":
        result = chunked_merge(
            original_code=original_text, snippet=op.new_text,
            file_path=str(tmp_file), merge_fn=_no_model, language=language,
            replace=op.symbol,
        )
    else:  # insert_after_symbol / append_at_eof: the zero-token fast path
        result = chunked_merge(
            original_code=original_text, snippet=op.new_text,
            file_path=str(tmp_file), merge_fn=_no_model, language=language,
            after=op.symbol,
        )
    return result.merged_code


@pytest.mark.parametrize("kind", OP_KINDS, ids=lambda k: k)
def test_oracle_matches_deterministic_pipeline(kind, tmp_path):
    case = build_corpus_case("python", 40_000, seed=77, kinds=(kind,))
    original = str(case.source)
    op = case.ops[0]
    expected = apply_op_oracle(original, op)
    got = _pipeline_bytes(original, op, case.language, case.ext, tmp_path)
    assert got == expected, (
        f"the deterministic pipeline's {kind} output diverges from the "
        f"independent corpus oracle"
    )


@pytest.mark.parametrize(
    "kind", ["insert_after_symbol", "replace_symbol_body", "delete_symbol"],
    ids=str,
)
def test_oracle_matches_pipeline_crlf(kind, tmp_path):
    case = build_corpus_case("python", 40_000, seed=78, eol="\r\n", kinds=(kind,))
    original = str(case.source)
    op = case.ops[0]
    expected = apply_op_oracle(original, op)
    got = _pipeline_bytes(original, op, case.language, case.ext, tmp_path)
    assert got == expected, (
        f"CRLF: the deterministic pipeline's {kind} output diverges from the "
        f"independent corpus oracle"
    )


@pytest.mark.parametrize(
    "kind", ["insert_after_symbol", "replace_symbol_body", "append_at_eof"],
    ids=str,
)
def test_oracle_matches_pipeline_no_final_eol(kind, tmp_path):
    # The risky convention: EOF-span ops meet the EOL funnel's trailing rule.
    case = build_corpus_case("rust", 40_000, seed=79, final_eol=False,
                             kinds=(kind,))
    original = str(case.source)
    op = case.ops[0]
    expected = apply_op_oracle(original, op)
    got = _pipeline_bytes(original, op, case.language, case.ext, tmp_path)
    assert got == expected, (
        f"no-final-EOL: the deterministic pipeline's {kind} output diverges "
        f"from the independent corpus oracle"
    )


def test_oracle_matches_pipeline_go_cjk_direct_swap(tmp_path):
    case = build_corpus_case("go", 40_000, seed=80, sprinkle_cjk=True,
                             kinds=("replace_symbol_body",))
    original = str(case.source)
    op = case.ops[0]
    expected = apply_op_oracle(original, op)
    got = _pipeline_bytes(original, op, case.language, case.ext, tmp_path)
    assert got == expected, (
        "go/CJK: the deterministic direct-swap output diverges from the "
        "independent corpus oracle"
    )


# ---------------------------------------------------------------------------
# Manifest (de)serialization + EOL conventions + parse cleanliness
# ---------------------------------------------------------------------------


def test_op_manifest_json_round_trip():
    case = build_corpus_case("rust", 30_000, seed=23, sprinkle_cjk=True)
    dumped = json.dumps(corpus.case_manifest(case, "roundtrip"),
                        ensure_ascii=False)
    loaded = json.loads(dumped)
    for i, op in enumerate(case.ops):
        assert op_from_manifest(loaded["ops"][i]) == op
        serialized = line_op_from_manifest(loaded["ops"][i]["inverse"])
        assert serialized == inverse_op(op)


@pytest.mark.parametrize(
    "kwargs",
    [pytest.param({"eol": "\n", "final_eol": True}, id="lf-final-eol"),
     pytest.param({"eol": "\r\n", "final_eol": True}, id="crlf"),
     pytest.param({"eol": "\n", "final_eol": False}, id="no-final-eol")],
)
def test_line_ending_conventions(kwargs):
    source = generate_big_source("python", 40_000, seed=29, **kwargs)
    text = str(source)
    if kwargs["eol"] == "\r\n":
        assert text.count("\n") == text.count("\r\n"), "bare LF in a CRLF corpus"
        assert "\r\n" in text
    else:
        assert "\r" not in text, "stray CR in an LF corpus"
    if kwargs["final_eol"]:
        assert text.endswith(kwargs["eol"])
    else:
        assert not text.endswith(("\n", "\r"))


@pytest.mark.parametrize(
    "language", ["python", "rust", "go", "typescript"], ids=str,
)
def test_generated_corpora_parse_clean(language):
    """The corpus is C2/C3 pipeline input — it must parse as its language."""
    from fastedit.data_gen.ast_analyzer import parse_diagnostics

    for extra in (
        {"eol": "\n", "final_eol": True},
        {"eol": "\r\n", "final_eol": True},
        {"eol": "\n", "final_eol": False},
    ):
        source = generate_big_source(language, 120_000, seed=31, **extra)
        diagnostics = parse_diagnostics(str(source), language)
        assert diagnostics.is_valid, (
            f"{language} corpus ({extra}) does not parse: "
            f"{diagnostics.errors[:3]}"
        )


@pytest.mark.parametrize(
    "manifest_path", _golden_case_dirs(), ids=lambda p: p.parent.name,
)
def test_committed_golden_originals_parse_clean(manifest_path):
    from fastedit.data_gen.ast_analyzer import parse_diagnostics

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    original_bytes = (manifest_path.parent / manifest["filename"]).read_bytes()
    diagnostics = parse_diagnostics(original_bytes.decode("utf-8"),
                                    manifest["language"])
    assert diagnostics.is_valid, (
        f"{manifest['case']}: committed corpus original does not parse as "
        f"{manifest['language']}: {diagnostics.errors[:3]}"
    )
