"""B3 golden matrix runner — manifest-driven, zero test code per language.

Consumes every ``tests/golden/<lang>/manifest.json`` produced by the
committed generator (``tests/golden/_generate.py``, whose line-splice
oracle is independent of fastedit) and, for each declared op, asserts:

* **byte-exact output** — the pipeline's merged code equals the committed
  expected file, byte for byte (``read_bytes`` equality; ``first_diff_tag``
  diagnostics on failure);
* **oracle agreement** — the committed expected file is re-derived from the
  manifest's line indices with the generator's oracle, so a stale artifact
  fails loudly instead of pinning wrong behavior;
* **the deterministic path** — ``model_tokens == 0`` and ``chunks_used == 0``
  (the model callback raises if ever called), and the specific zero-token
  path the manifest pins (fast-path insert / direct-swap / text-match) is
  the one that actually ran;
* **the relative parse rule (req. 9)** —
  :func:`merged_is_acceptable` accepts the merged output against the
  original's own diagnostics;
* **GIGO (req. 9)** — for the one defective-original case per format class,
  the pre-existing benign defect in an UNRELATED region survives the edit
  byte-exact.

Adding a language = adding a golden directory; no test code changes.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import subprocess
import sys
from pathlib import Path

import pytest
from llm_fixtures import first_diff_tag

from fastedit.data_gen.ast_analyzer import parse_diagnostics
from fastedit.inference.chunked_merge import (
    chunked_merge,
    delete_symbol,
    merged_is_acceptable,
)

GOLDEN_DIR = Path(__file__).parent / "golden"
CLI_MODULE = [sys.executable, "-m", "fastedit"]

# The committed oracle — imported by file path (tests/golden is not a
# package). It is fastedit-independent by construction; see its docstring.
_spec = importlib.util.spec_from_file_location(
    "golden_oracle", GOLDEN_DIR / "_generate.py",
)
oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oracle)


def _wheel_present(module: str) -> bool:
    spec = importlib.util.find_spec(module)
    return spec is not None


def _manifest_cases():
    cases = []
    for manifest_path in sorted(GOLDEN_DIR.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("census_excluded"):
            continue  # excluded language: no ops — see the census-exclusion test
        lang_dir = manifest_path.parent
        language = manifest["language"]
        skip_marks = []
        wheel = manifest.get("requires_wheel")
        if wheel and not _wheel_present(wheel):
            skip_marks = [
                pytest.mark.skipif(
                    True,
                    reason=(
                        f"{wheel} is an all-grammars extra dependency, not a "
                        f"hard dependency — the {language} golden case is the "
                        f"optional extra's job, never the default install's"
                    ),
                ),
            ]
        for op in manifest["ops"]:
            cases.append(pytest.param(
                lang_dir, manifest, {"op": op, "original": manifest["filename"]},
                id=f"{language}:{op['op']}:{op['symbol']}",
                marks=skip_marks,
            ))
        if "gigo" in manifest:
            gigo = manifest["gigo"]
            cases.append(pytest.param(
                lang_dir, manifest, {"op": gigo["op"], "original": gigo["original"],
                                     "gigo": gigo},
                id=f"{language}:gigo:{gigo['op']['symbol']}",
                marks=skip_marks,
            ))
    return cases


def _no_model(*_args, **_kwargs):
    raise AssertionError(
        "the model path ran on a deterministic golden op — model_tokens must "
        "be 0 and the merge callback must never be called",
    )


def _run_op(case_op, language, original_text, tmp_file):
    kind = case_op["op"]
    if kind == "delete_symbol":
        result = delete_symbol(str(tmp_file), case_op["symbol"], language=language)
        return result
    kwargs = {"after": case_op["symbol"]} if kind == "insert_after" else {
        "replace": case_op["symbol"],
    }
    return chunked_merge(
        original_code=original_text,
        snippet=case_op["snippet"],
        file_path=str(tmp_file),
        merge_fn=_no_model,
        language=language,
        **kwargs,
    )


def _oracle_bytes(manifest, case_op, original_text) -> bytes:
    lines = original_text.splitlines(keepends=True)
    kind = case_op["op"]
    if kind == "insert_after":
        merged = oracle.oracle_insert_after(
            lines, case_op["oracle"]["anchor_end_line"], case_op["snippet"],
        )
    elif kind == "replace_symbol":
        merged = oracle.oracle_replace_span(
            lines,
            case_op["oracle"]["start_line"],
            case_op["oracle"]["end_line"],
            case_op["snippet"],
            case_op.get("prepend_signature_lines", 0),
        )
    else:
        merged = oracle.oracle_delete_span(
            lines,
            case_op["oracle"]["start_line"],
            case_op["oracle"]["end_line"],
        )
    return "".join(merged).encode("utf-8")


@pytest.mark.parametrize(("lang_dir", "manifest", "case"), _manifest_cases())
def test_golden_case(lang_dir, manifest, case, tmp_path, caplog):
    language = manifest["language"]
    ext = manifest["ext"]
    case_op = case["op"]
    is_gigo = "gigo" in case

    original_bytes = (lang_dir / case["original"]).read_bytes()
    original_text = original_bytes.decode("utf-8")
    tmp_file = tmp_path / f"golden.{ext}"
    tmp_file.write_bytes(original_bytes)

    # 1. the committed expected file agrees with the independent oracle
    #    re-derived from the manifest's line indices.
    expected = (lang_dir / case_op["expected"]).read_bytes()
    assert _oracle_bytes(manifest, case_op, original_text) == expected, (
        f"committed expected file {case_op['expected']} drifted from the "
        f"oracle — regenerate the goldens (tests/golden/_generate.py)"
    )

    # 2. run the op through the deterministic path (the model callback raises
    #    if it is ever called).
    caplog.set_level(logging.INFO, logger="fastedit.chunked_merge")
    result = _run_op(case_op, language, original_text, tmp_file)

    # 3. deterministic-path bookkeeping: zero model tokens, zero chunks.
    if case_op["op"] == "delete_symbol":
        span = (
            case_op["oracle"]["start_line"],
            case_op["oracle"]["end_line"],
        )
        assert result.deleted_lines == span
    else:
        assert result.model_tokens == 0, "golden ops must not consume tokens"
        assert result.chunks_used == 0, "golden ops must not spawn chunks"
        pinned = case_op.get("path")
        if pinned == "fast_path":
            assert f"Fast-path insert after '{case_op['symbol']}'" in caplog.text
        elif pinned == "direct_swap":
            assert f"Direct-swap for replace='{case_op['symbol']}'" in caplog.text
        elif pinned == "text_match":
            assert (
                f"Deterministic text-match for replace='{case_op['symbol']}'"
                in caplog.text
            )
        else:
            raise AssertionError(f"unpinned deterministic path: {pinned!r}")

    # 4. byte-exact output vs the committed expected file.
    merged_text = result.merged_code
    got = merged_text.encode("utf-8")
    assert got == expected, (
        f"{language} {case_op['op']} {case_op['symbol']} is not byte-exact: "
        f"{first_diff_tag(merged_text, expected.decode('utf-8'))}"
    )

    # 5. the relative parse rule (req. 9): the merged output is acceptable
    #    against the ORIGINAL's own diagnostics.
    ok, reason = merged_is_acceptable(
        parse_diagnostics(original_text, language),
        parse_diagnostics(merged_text, language),
    )
    assert ok, f"relative parse rule rejected the golden merge: {reason}"

    # 6. GIGO: the pre-existing defect in the unrelated region survives
    #    byte-exact (EDIT-NOT-CORRECT — the edit never touches it).
    if is_gigo:
        gigo = case["gigo"]
        start, end = gigo["defect_lines"]
        original_lines = original_text.splitlines(keepends=True)
        defect_region = "".join(original_lines[start - 1 : end])
        assert defect_region and defect_region in merged_text, (
            f"the pre-existing defect was not preserved byte-exact: "
            f"{defect_region!r} vanished from the merged output"
        )


def test_every_manifest_op_is_supported_or_declared():
    """No silent holes: every manifest covers the three core ops unless it
    declares an explicit ``unsupported_reason`` for the missing one."""
    for manifest_path in sorted(GOLDEN_DIR.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("requires_wheel"):
            continue  # pack-only samples carry a single representative op
        declared = {op["op"] for op in manifest["ops"]}
        declared |= {
            u["op"] for u in manifest.get("unsupported", [])
        }
        missing = {"insert_after", "replace_symbol", "delete_symbol"} - declared
        assert not missing, (
            f"{manifest_path}: ops {sorted(missing)} are neither exercised "
            f"nor declared unsupported"
        )


# ---------------------------------------------------------------------------
# F2 census-batch fixtures: one parse-clean fixture per census language.
#
# These manifests carry no (or few) anchoring ops — the census languages
# mostly have no declarative symbol anchoring row in ast_utils yet, and each
# manifest declares the hole via its ``unsupported`` entries. What EVERY
# census fixture does claim is that its original parses with ZERO error
# traits in its own language (for the parse_degraded-retry languages that
# claim IS the F2 finding: the census's trivial probe line was the problem,
# not the grammar). This test pins that claim so a grammar bump that breaks
# a fixture fails loudly instead of silently rotting the golden corpus.
# ---------------------------------------------------------------------------

def _census_fixture_cases():
    cases = []
    for manifest_path in sorted(GOLDEN_DIR.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("census_fixture"):
            continue  # B3's op-bearing fixtures already parse via the ops
        lang_dir = manifest_path.parent
        language = manifest["language"]
        skip_marks = []
        wheel = manifest.get("requires_wheel")
        if wheel and not _wheel_present(wheel):
            skip_marks = [
                pytest.mark.skipif(
                    True,
                    reason=(
                        f"{wheel} is an all-grammars extra dependency, not a "
                        f"hard dependency — the {language} census fixture "
                        f"describes this venv's pack and cannot parse without "
                        f"it"
                    ),
                ),
            ]
        cases.append(pytest.param(
            lang_dir, manifest,
            id=f"{language}:census-original-parses-clean",
            marks=skip_marks,
        ))
    return cases


@pytest.mark.parametrize(("lang_dir", "manifest"), _census_fixture_cases())
def test_census_fixture_original_parses_clean(lang_dir, manifest):
    language = manifest["language"]
    original = (lang_dir / manifest["filename"]).read_text(encoding="utf-8")
    diags = parse_diagnostics(original, language)
    assert diags.is_valid, (
        f"{language} census fixture {manifest['filename']} does not parse "
        f"clean: {diags.errors[:3]} — the fixture claims a parse-clean "
        f"canonical snippet for this language"
    )


# ---------------------------------------------------------------------------
# F3 census exclusions: census languages whose grammar is genuinely BROKEN
# (recorded in the generator's EXCLUDED_LANGUAGES, written into the manifest
# as ``census_excluded``). The exclusion is fail-loud documentation, never a
# silent shrug: this test re-parses EVERY recorded attempt and pins the
# EXACT parse errors, so an upstream grammar fix fails here — at which point
# the exclusion is lifted (real census fixture in CENSUS_LANGUAGES with a
# census_note + pack_census.json verdict flip), never outlived silently.
# ---------------------------------------------------------------------------

def _census_excluded_cases():
    cases = []
    for manifest_path in sorted(GOLDEN_DIR.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("census_excluded"):
            continue  # ordinary manifests are covered by the tests above
        lang_dir = manifest_path.parent
        language = manifest["language"]
        skip_marks = []
        wheel = manifest.get("requires_wheel")
        if wheel and not _wheel_present(wheel):
            skip_marks = [
                pytest.mark.skipif(
                    True,
                    reason=(
                        f"{wheel} is an all-grammars extra dependency, not a "
                        f"hard dependency — the {language} exclusion record "
                        f"describes this venv's pack and cannot parse without "
                        f"it"
                    ),
                ),
            ]
        cases.append(pytest.param(
            lang_dir, manifest,
            id=f"{language}:census-excluded-grammar-defect",
            marks=skip_marks,
        ))
    return cases


@pytest.mark.parametrize(("lang_dir", "manifest"), _census_excluded_cases())
def test_census_excluded_grammar_defect_is_real(lang_dir, manifest):
    language = manifest["language"]
    excluded = manifest["census_excluded"]
    assert excluded.get("reason"), f"{language}: exclusion needs a reason"
    attempts = excluded.get("attempts")
    assert attempts, f"{language}: exclusion needs recorded attempts"
    for attempt in attempts:
        diags = parse_diagnostics(attempt["snippet"], language)
        # G1a: known grammar artifacts (e.g. `test`'s MISSING-at-EOF
        # emission) are removed from .errors and tagged in
        # .grammar_artifacts — the relative parse rule must never count
        # them. This exclusion pin therefore watches the UNFILTERED
        # emission (real traits + tagged artifacts) so the defect stays
        # fail-loud: an upstream grammar fix empties the view entirely and
        # the exclusion must be lifted loudly, exactly as before G1a.
        unfiltered = [list(err) for err in diags.errors] + [
            list(err) for err in diags.grammar_artifacts
        ]
        assert unfiltered, (
            f"{language} census-excluded attempt "
            f"({attempt['description']}) now parses with no error traits "
            f"at all — the grammar defect is fixed upstream: lift the "
            f"exclusion by authoring a real census fixture "
            f"(tests/golden/_generate.py CENSUS_LANGUAGES) with a "
            f"census_note and flipping the pack_census.json verdict"
        )
        assert (
            unfiltered == [list(err) for err in attempt["parse_errors"]]
        ), (
            f"{language} census-excluded attempt "
            f"({attempt['description']}) parse errors changed to "
            f"{unfiltered} — re-record the exclusion (or lift it if the "
            f"grammar is now healthy)"
        )


# ---------------------------------------------------------------------------
# CLI-level e2e regression: the CLI's deterministic replace path must see the
# in-memory symbol map (B26 rationale) or every new-format `edit --replace`
# would refuse "Symbol not found" from the tldr daemon's empty map.
# ---------------------------------------------------------------------------

def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [*CLI_MODULE, *args], capture_output=True, text=True, check=False,
    )


@pytest.mark.parametrize(
    ("lang", "filename", "flag", "symbol", "snippet", "expected_name"),
    [
        # replace= exercises the CLI's deterministic replace entry point
        # (html is a format the tldr daemon knows nothing about, so the
        # in-memory symbol map is what makes this work at all).
        (
            "html", "original.html", "--replace", "details",
            (
                '  <section id="details" class="wide">\n    <h2>Details</h2>\n'
                "    <p>Expanded detail paragraph.</p>\n  </section>\n"
            ),
            "expected_replace_symbol_details.html",
        ),
        # after= exercises the CLI's zero-model insert path on a format the
        # daemon also cannot resolve.
        ("yaml", "original.yaml", "--after", "version", "license: MIT\n",
         "expected_insert_after_version.yaml"),
    ],
)
def test_cli_deterministic_edit_end_to_end(
    tmp_path, lang, filename, flag, symbol, snippet, expected_name,
):
    target = tmp_path / f"sample.{filename.rsplit('.', 1)[1]}"
    original = (GOLDEN_DIR / lang / filename).read_bytes()
    target.write_bytes(original)
    result = _run_cli(
        "edit", str(target), flag, symbol, "--snippet", snippet,
    )
    assert result.returncode == 0, result.stderr
    expected = (GOLDEN_DIR / lang / expected_name).read_bytes()
    assert target.read_bytes() == expected, (
        f"CLI edit is not byte-exact: "
        f"{first_diff_tag(target.read_text(encoding='utf-8'), expected.decode('utf-8'))}"
    )
