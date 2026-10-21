"""Step G1a — parse watchdog for pathological-recovery grammars + known
grammar-artifact filtering.

Two mechanisms, both born from the Phase F 175-language census:

1. ``PATHOLOGICAL_RECOVERY_GRAMMARS`` (founding member: ``cobol``) — the
   grammar resolves instantly and parses REAL cobol in 0.0s, but its
   error recovery LOOPS FOREVER on unrecoverable input (the census probe
   hung attempt #1; classified unresolvable only because the census runs
   per-language subprocesses). Guarded languages parse under a hard
   subprocess deadline: healthy input returns real diagnostics; a wedge
   raises the typed :class:`ParseWatchdogTimeout` instead of hanging the
   pipeline. ``get_ast_map_from_source`` mirrors the unresolvable-grammar
   contract: timeout verdict → empty AST map.

2. ``_GRAMMAR_ARTIFACT_FILTERS`` (founding member: ``test``) — the
   language-pack grammar emits one zero-width MISSING trait at EOF for
   EVERY complete record (grammar defect, recorded fail-loud with exact
   errors in tests/golden/test/manifest.json). The systematic artifact
   carries no edit information, so ``parse_diagnostics`` removes it from
   the trait lists (tagged in ``.grammar_artifacts``): under
   ``merged_is_acceptable`` it can never be 'new' and can never absorb a
   genuinely new defect into its multiset slot, while a NEW real error in
   a .test file still rejects the merge.

The wedge tests use the REAL hang — the only honest test — bounded by the
subprocess's own timeout (no pytest-timeout involvement).
"""

from __future__ import annotations

import importlib.util
import time

import pytest

from fastedit.data_gen import ast_analyzer
from fastedit.data_gen.ast_analyzer import (
    _GRAMMAR_ARTIFACT_FILTERS,
    PARSE_WATCHDOG_TIMEOUT_SECONDS,
    PATHOLOGICAL_RECOVERY_GRAMMARS,
    ParseDiagnostics,
    ParseWatchdogTimeout,
    _is_known_grammar_artifact,
    bounded_parse_diagnostics,
    parse_diagnostics,
    validate_parse,
)
from fastedit.inference.ast_utils import get_ast_map_from_source
from fastedit.inference.chunked_merge import merged_is_acceptable

_PACK_INSTALLED = (
    importlib.util.find_spec("tree_sitter_language_pack") is not None
)
requires_pack = pytest.mark.skipif(
    not _PACK_INSTALLED,
    reason="tree-sitter-language-pack (all-grammars extra) is not installed",
)

# Measured healthy input: parses in 0.0s with zero error traits.
HEALTHY_COBOL = (
    "       IDENTIFICATION DIVISION.\n"
    "       PROGRAM-ID. HELLO.\n"
    "       PROCEDURE DIVISION.\n"
    '           DISPLAY "HELLO".\n'
    "           STOP RUN.\n"
)
# Measured pathological input: the grammar's error recovery loops forever.
PATHOLOGICAL_COBOL = "x\n"

# The manifest's "complete test record" snippet: one zero-width MISSING
# trait at byte 75 == len(source) == EOF, for EVERY complete record.
TEST_RECORD = (
    "================\nSample test\n================\nprint(\"hi\")\n"
    "================\n"
)
TEST_RECORD_HEADER_ONLY = "================\nSample test\n================\n"


# ---------------------------------------------------------------------------
# Declarative tables — the founding members, pinned
# ---------------------------------------------------------------------------


def test_declarative_tables_pin_the_founding_members():
    assert PATHOLOGICAL_RECOVERY_GRAMMARS == {"cobol"}
    assert PARSE_WATCHDOG_TIMEOUT_SECONDS == 15.0
    assert _GRAMMAR_ARTIFACT_FILTERS == {"test": (("MISSING", "eof"),)}


# ---------------------------------------------------------------------------
# Watchdog routing — healthy input, real diagnostics
# ---------------------------------------------------------------------------


@requires_pack
def test_guarded_grammar_healthy_input_returns_real_diagnostics():
    t0 = time.perf_counter()
    diags = parse_diagnostics(HEALTHY_COBOL, "cobol")
    elapsed = time.perf_counter() - t0
    assert diags.errors == []
    assert diags.is_valid is True
    assert diags.grammar_artifacts == ()
    assert diags.source == HEALTHY_COBOL
    assert elapsed < 15.0, "healthy guarded parse must stay inside the bound"


@requires_pack
def test_validate_parse_sees_healthy_guarded_input_as_valid():
    assert validate_parse(HEALTHY_COBOL, "cobol") is True


def test_guarded_grammar_routes_through_the_watchdog(monkeypatch):
    # Routing control: the listed grammar is dispatched to the bounded
    # helper (canonicalized — "COBOL" must not slip past the guard), and
    # the helper's verdict is returned verbatim.
    sentinel = ParseDiagnostics(errors=[], is_valid=True, source="s")
    seen: dict[str, object] = {}

    def fake_bounded(source, language, timeout_seconds=None):
        seen["args"] = (source, language)
        return sentinel

    monkeypatch.setattr(ast_analyzer, "bounded_parse_diagnostics", fake_bounded)
    diags = parse_diagnostics("whatever", "COBOL")
    assert diags is sentinel
    assert seen["args"] == ("whatever", "cobol")


# ---------------------------------------------------------------------------
# Watchdog routing — the real wedge, bounded
# ---------------------------------------------------------------------------


@requires_pack
def test_guarded_grammar_pathological_input_raises_typed_timeout():
    # The REAL hang (grammar error recovery loops on `x`), bounded by the
    # shipped default deadline — the only honest wedge test.
    t0 = time.perf_counter()
    with pytest.raises(ParseWatchdogTimeout) as excinfo:
        parse_diagnostics(PATHOLOGICAL_COBOL, "cobol")
    elapsed = time.perf_counter() - t0
    assert excinfo.value.language == "cobol"
    assert excinfo.value.timeout_seconds == PARSE_WATCHDOG_TIMEOUT_SECONDS
    assert "cobol" in str(excinfo.value)
    # Killed at the deadline, never unbounded.
    assert elapsed < PARSE_WATCHDOG_TIMEOUT_SECONDS + 5.0


@requires_pack
def test_bounded_helper_honors_a_custom_deadline():
    t0 = time.perf_counter()
    with pytest.raises(ParseWatchdogTimeout) as excinfo:
        bounded_parse_diagnostics(PATHOLOGICAL_COBOL, "cobol", timeout_seconds=2.0)
    elapsed = time.perf_counter() - t0
    assert excinfo.value.timeout_seconds == 2.0
    assert elapsed < 12.0


def test_validate_parse_routes_guarded_grammar_through_the_watchdog(
    monkeypatch,
):
    def wedged(*_args, **_kwargs):
        raise ParseWatchdogTimeout("cobol", 15.0)

    monkeypatch.setattr(ast_analyzer, "bounded_parse_diagnostics", wedged)
    with pytest.raises(ParseWatchdogTimeout):
        validate_parse(PATHOLOGICAL_COBOL, "cobol")


# ---------------------------------------------------------------------------
# Non-listed grammars bypass the watchdog — in-process path unchanged
# ---------------------------------------------------------------------------


def test_non_listed_grammars_bypass_the_watchdog(monkeypatch):
    def must_not_run(*_args, **_kwargs):
        raise AssertionError("watchdog must not run for unlisted grammars")

    monkeypatch.setattr(ast_analyzer, "bounded_parse_diagnostics", must_not_run)
    # python: in-process diagnostics and verdicts, exactly as before G1a.
    assert parse_diagnostics("def f():\n    retrun 1\n", "python").is_valid is False
    assert validate_parse("def f():\n    return 1\n", "python") is True


def test_ast_map_unlisted_grammar_never_probes(monkeypatch):
    def must_not_run(*_args, **_kwargs):
        raise AssertionError("watchdog must not run for unlisted grammars")

    monkeypatch.setattr(ast_analyzer, "bounded_parse_diagnostics", must_not_run)
    nodes = get_ast_map_from_source("def f():\n    return 1\n", "m.py")
    assert [n.name for n in nodes] == ["f"]


# ---------------------------------------------------------------------------
# get_ast_map_from_source — timeout verdict mirrors unresolvable grammar
# ---------------------------------------------------------------------------


def test_ast_map_timeout_verdict_returns_empty_map(monkeypatch):
    # The unresolvable-grammar contract is `except (ValueError, RuntimeError,
    # ImportError) -> []`. ParseWatchdogTimeout is a RuntimeError, raised at
    # the same guarded call site, so the wedge yields the SAME empty map.
    def wedged(*_args, **_kwargs):
        raise ParseWatchdogTimeout("cobol", 15.0)

    monkeypatch.setattr(ast_analyzer, "bounded_parse_diagnostics", wedged)
    assert get_ast_map_from_source(PATHOLOGICAL_COBOL, "bad.cob", "cobol") == []


@requires_pack
def test_ast_map_guarded_healthy_probes_then_parses(monkeypatch):
    probes: list[str] = []
    real_bounded = ast_analyzer.bounded_parse_diagnostics

    def recording_bounded(source, language, timeout_seconds=None):
        probes.append(language)
        return real_bounded(source, language, timeout_seconds)

    monkeypatch.setattr(
        ast_analyzer, "bounded_parse_diagnostics", recording_bounded,
    )
    t0 = time.perf_counter()
    nodes = get_ast_map_from_source(HEALTHY_COBOL, "hello.cob", "cobol")
    elapsed = time.perf_counter() - t0
    assert probes == ["cobol"]  # the guarded probe ran exactly once
    assert isinstance(nodes, list)  # real map path, never a hang
    assert elapsed < 15.0


@requires_pack
def test_ast_map_guarded_pathological_returns_empty_map_within_bound():
    # End-to-end with the REAL wedge: the empty map, never a hang.
    t0 = time.perf_counter()
    nodes = get_ast_map_from_source(PATHOLOGICAL_COBOL, "bad.cob", "cobol")
    elapsed = time.perf_counter() - t0
    assert nodes == []
    assert elapsed < PARSE_WATCHDOG_TIMEOUT_SECONDS + 5.0


# ---------------------------------------------------------------------------
# Known grammar artifacts — the `test` grammar's MISSING-at-EOF emission
# ---------------------------------------------------------------------------


@requires_pack
def test_known_grammar_artifact_is_filtered_and_tagged():
    # Exact errors are the manifest's fail-loud convention: an upstream
    # grammar fix changes these pins and lifts the exclusion loudly.
    diags = parse_diagnostics(TEST_RECORD, "test")
    assert diags.errors == []
    assert diags.is_valid is True
    assert diags.grammar_artifacts == ((75, 75, "MISSING"),)


@requires_pack
def test_header_only_test_file_keeps_its_real_error():
    # The filter is narrow: a header-only file is an outright ERROR node —
    # a genuine defect, never mistaken for the systematic artifact.
    diags = parse_diagnostics(TEST_RECORD_HEADER_ONLY, "test")
    assert diags.errors == [(0, 46, "ERROR")]
    assert diags.is_valid is False
    assert diags.grammar_artifacts == ()


def test_artifact_signature_is_narrow():
    # White box: the "eof" locator matches ONLY a zero-width trait pinned
    # at the very end of the source; other positions or kinds never match.
    filters = (("MISSING", "eof"),)
    assert _is_known_grammar_artifact((3, 3, "MISSING"), 3, filters) is True
    assert _is_known_grammar_artifact((2, 2, "MISSING"), 3, filters) is False
    assert _is_known_grammar_artifact((0, 3, "MISSING"), 3, filters) is False
    assert _is_known_grammar_artifact((3, 3, "ERROR"), 3, filters) is False
    assert _is_known_grammar_artifact((3, 3, "MISSING"), 4, filters) is False


# ---------------------------------------------------------------------------
# The relative rule with filtered artifacts
# ---------------------------------------------------------------------------


@requires_pack
def test_merged_is_acceptable_accepts_inherited_artifact():
    # A faithful edit that CHANGES THE FILE LENGTH moves the artifact's EOF
    # position. Filtered from BOTH sides, it is inherited (never 'new') and
    # the merge lands — no phantom trait can reject a faithful merge.
    original = parse_diagnostics(TEST_RECORD, "test")
    edited = parse_diagnostics(
        TEST_RECORD.replace('print("hi")', 'print("hello world")'), "test",
    )
    assert edited.grammar_artifacts != ()  # still produced, just filtered
    ok, reason = merged_is_acceptable(original, edited)
    assert ok is True, reason


@requires_pack
def test_new_real_error_in_test_file_still_rejects():
    # NEGATIVE control: the artifact filter loosens NOTHING. The merged
    # output's header-only ERROR trait is a real defect the (valid)
    # original does not have → rejected with the standard reason.
    original = parse_diagnostics(TEST_RECORD, "test")
    merged = parse_diagnostics(TEST_RECORD_HEADER_ONLY, "test")
    ok, reason = merged_is_acceptable(original, merged)
    assert ok is False
    assert "parse error" in reason
    assert "ERROR" in reason
