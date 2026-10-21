"""Definitive F1 census: every language fastedit can resolve + parse HERE.

Step F1 of the implementation plan. This module enumerates the installed
``tree_sitter_language_pack``'s language names (plus fastedit's wheel-added
canonical names — the declarative :data:`LANGUAGE_ALIASES` entries the pack
does not list) and probes EVERY name through fastedit's real resolver
(:func:`fastedit.data_gen.ast_analyzer.get_language` → ``get_parser`` →
:func:`parse_diagnostics` on trivial probe snippets).

Classification per language:

* ``ok`` — resolved, and at least one probe snippet parses with zero error
  traits. v2 flavors (optional entry fields): ``guarded_parse`` — a
  G1a-guarded grammar (PATHOLOGICAL_RECOVERY_GRAMMARS) whose REAL snippet
  parsed clean under the subprocess watchdog (healthy-with-watchdog;
  guarded grammars never receive trivial probes, which would wedge);
  ``filtered_artifacts`` — the parse's known-grammar-artifact traits were
  filtered per G1a (the ``test`` grammar's systematic MISSING-at-EOF);
* ``parse_degraded`` — resolved, but no probe snippet parses cleanly (the
  grammar cannot express any of the trivial one-liners, or parsing raised);
* ``unresolvable`` — the resolver could not serve the language, the probe
  crashed, or the probe TIMED OUT.

Hang-proofing (the F1 mission's hard rule, learned from two earlier attempts
that wedged on a grammar load): every language is probed in an ISOLATED
subprocess that (a) runs in its own session (``start_new_session=True``) and
(b) is SIGKILLed as a whole process group the moment its per-language timeout
(default 10 s) fires. One hanging grammar dylib can therefore never wedge the
loop — it costs exactly its timeout and is classified ``unresolvable`` with
reason ``timeout`` and the language's name. No probe ever touches the
network; the pack's 0.x line bundles every grammar in the wheel (the 1.x
line downloads dylibs and is deliberately capped ``<1.0`` in pyproject).

Caching / snapshot contract — the snapshot IS the cache:

* ``tests/golden/pack_census.json`` holds the last full census. ``census()``
  loads it and re-probes only names that are missing (new pack version,
  grown wheel table, bumped :data:`PROBE_VERSION`), so re-runs are fast;
* ``python tests/pack_census.py`` refreshes the snapshot (add ``--force`` to
  re-probe every language) and prints the headline breakdown;
* ``FASTEDIT_REGEN_CENSUS=1 uv run pytest tests/test_pack_census.py``
  regenerates the snapshot from a full re-probe (see the test module);
* the cache is invalidated wholesale when the pack version changes or the
  probe program/schema version (:data:`PROBE_VERSION`) is bumped.

The volatile wall-clock timings (``probe_runtime_seconds`` /
``probe_seconds``) are measured and reported but ``probe_runtime_seconds`` is
stripped from the snapshot so the committed golden file stays deterministic.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib
import importlib.metadata
import json
import os
import re
import signal
import subprocess
import sys
import time
import typing
from pathlib import Path

from fastedit.data_gen.ast_analyzer import LANGUAGE_ALIASES

PROBE_VERSION = 2
"""Bump when the probe program, snippets, or entry schema change — it
invalidates the cached snapshot so every language is re-probed.

v2: real per-language snippets for the two grammars whose verdict the
trivial one-liners cannot express — cobol (G1a-guarded: a trivial line is
unrecoverable input whose parse wedges, so guarded grammars NEVER receive
trivial probes — each wedge would cost the whole census timeout) and test
(G1a artifact filter: the trivial lines are outright ERROR nodes while the
real record parses with the systematic artifact filtered). An ok verdict
may additionally carry ``guarded_parse`` (healthy-with-watchdog) or
``filtered_artifacts`` (the known-grammar-artifact count)."""

PACK_MODULE = "tree_sitter_language_pack"
SNAPSHOT_PATH = Path(__file__).resolve().parent / "golden" / "pack_census.json"
REGEN_ENV_VAR = "FASTEDIT_REGEN_CENSUS"
DEFAULT_PROBE_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_WORKERS = 12

STATUSES = ("ok", "parse_degraded", "unresolvable")
UNRESOLVABLE_REASONS = ("timeout", "grammar_unavailable", "error")


class PackUnavailableError(RuntimeError):
    """The aggregate pack this census describes is not installed here."""


# --------------------------------------------------------------------------
# The probe program — executed as `python -c <_PROBE_PROGRAM> <language>` in
# an isolated subprocess. It ALWAYS prints exactly one JSON verdict line, and
# it must never outlive the parent's per-language timeout (the parent kills
# it). Kept dependency-light: fastedit's resolver + tree-sitter only.
# --------------------------------------------------------------------------
_PROBE_PROGRAM = r'''
import importlib
import json
import sys

name = sys.argv[1]

# Trivial one-line probe snippets spanning the major syntax families. A
# language is "ok" when at least ONE parses with zero error traits; a
# grammar that resolves but parses none of these cleanly is "parse_degraded".
# (No per-language fixture authoring here — that is F2/F3's job.)
SNIPPETS = (
    "x = 1\n",
    "x\n",
    "1\n",
    "# comment\n",
    "int x = 1;\n",
    "class A {}\n",
    "function f() {}\n",
    "fn f() {}\n",
    "func f() {}\n",
    "SELECT 1;\n",
    "{\"a\": 1}\n",
    "<div>x</div>\n",
    "key: value\n",
    "<root/>\n",
)

verdict = {
    "status": "unresolvable",
    "detail": "",
    "served_by": "language_pack",
}


def finish():
    print(json.dumps(verdict))
    sys.stdout.flush()
    sys.exit(0)


try:
    import fastedit.data_gen.ast_analyzer as aa
except Exception as exc:  # noqa: BLE001 — any import failure is a verdict
    verdict["reason"] = "error"
    verdict["detail"] = "fastedit import failed: %s: %s" % (type(exc).__name__, exc)
    finish()


def wheel_only_resolves(canonical):
    """True iff fastedit's DIRECT-wheel resolution path (no pack fallback)
    yields a Language for *canonical* — i.e. the grammar ships as its own
    ``tree_sitter_<x>`` wheel with a working entry point."""
    for module_name in aa._grammar_module_candidates(canonical):
        try:
            module = importlib.import_module(module_name)
        except Exception:  # noqa: BLE001 — absent/broken wheel: not this path
            continue
        for entry in aa._grammar_entry_candidates(canonical):
            try:
                if aa._language_from_module(module, entry) is not None:
                    return True
            except Exception:  # noqa: BLE001
                continue
    return False


try:
    verdict["served_by"] = (
        "direct_wheel" if wheel_only_resolves(name) else "language_pack"
    )
    aa.get_language(name)
    aa.get_parser(name)
except aa.GrammarUnavailableError as exc:
    verdict["status"] = "unresolvable"
    verdict["reason"] = "grammar_unavailable"
    verdict["detail"] = str(exc)
    finish()
except Exception as exc:  # noqa: BLE001 — wheel ABI errors etc. = unresolvable
    verdict["status"] = "unresolvable"
    verdict["reason"] = "error"
    verdict["detail"] = "%s: %s" % (type(exc).__name__, exc)
    finish()

parse_exception = None
parsed_any = False
clean = False
filtered_artifacts = 0
# Real per-language snippets for the languages whose verdict the trivial
# one-liners cannot express (probe v2). Guarded grammars
# (aa.PATHOLOGICAL_RECOVERY_GRAMMARS) receive ONLY these: a trivial line is
# unrecoverable input for such a grammar and the parse wedges until the
# watchdog deadline (15s) — longer than the census's own per-language kill
# (10s) — so probing trivial lines at a guarded grammar can only burn the
# census timeout. `test` is the artifact case: every trivial line is an
# outright ERROR node, while the real record parses with the known
# systematic artifact filtered (G1a) — exactly the ok claim to verify.
REAL_SNIPPETS = {
    "cobol": (
        "       IDENTIFICATION DIVISION.\n"
        "       PROGRAM-ID. HELLO.\n"
        "       PROCEDURE DIVISION.\n"
        "       MAIN-PARA.\n"
        '           DISPLAY "HELLO".\n'
        "           STOP RUN.\n"
    ),
    "test": (
        "================\nSample test\n================\n"
        'print("hi")\n================\n'
    ),
}
guarded = name in aa.PATHOLOGICAL_RECOVERY_GRAMMARS
candidates = tuple(REAL_SNIPPETS.get(name, ()))
if not guarded:
    candidates = candidates + SNIPPETS
for snippet in candidates:
    try:
        diagnostics = aa.parse_diagnostics(snippet, name)
    except Exception as exc:  # noqa: BLE001 — a crashing parse is a verdict
        parse_exception = "%s: %s" % (type(exc).__name__, exc)
        continue
    parsed_any = True
    if diagnostics.is_valid:
        clean = True
        filtered_artifacts = len(diagnostics.grammar_artifacts)
        break

if clean:
    verdict["status"] = "ok"
    verdict["reason"] = None
    if guarded:
        verdict["guarded_parse"] = True
        verdict["detail"] = (
            "resolved; real per-language snippet parsed with zero error "
            "traits under fastedit's subprocess parse watchdog "
            "(PATHOLOGICAL_RECOVERY_GRAMMARS member — healthy-with-watchdog)"
        )
    else:
        verdict["detail"] = "resolved; probe snippet parsed with zero error traits"
        if filtered_artifacts:
            verdict["filtered_artifacts"] = filtered_artifacts
            verdict["detail"] += (
                f" ({filtered_artifacts} known grammar artifact trait(s) "
                f"filtered — see _GRAMMAR_ARTIFACT_FILTERS)"
            )
    finish()
elif parsed_any:
    verdict["status"] = "parse_degraded"
    verdict["reason"] = "no_clean_parse"
    verdict["detail"] = (
        "resolved; no probe snippet parsed without error traits"
        + ("; last parse exception: " + parse_exception if parse_exception else "")
    )
else:
    verdict["status"] = "parse_degraded"
    verdict["reason"] = "parse_raised"
    verdict["detail"] = (
        "resolved; parse raised on every probe snippet; last: "
        + str(parse_exception)
    )
finish()
'''

_HEX_ADDRESS = re.compile(r"0x[0-9a-fA-F]+")


def _sanitize_detail(text: str, limit: int = 400) -> str:
    """Make a probe detail snapshot-stable: strip hex addresses, squash
    whitespace, truncate. Exception text often embeds ``0x7f...`` pointers
    that change between runs."""
    cleaned = _HEX_ADDRESS.sub("0x…", text or "")
    cleaned = " ".join(cleaned.split())
    if len(cleaned) > limit:
        cleaned = cleaned[: limit - 1] + "…"
    return cleaned


def _empty_unresolvable(reason: str = "error", detail: str = "") -> dict:
    return {
        "status": "unresolvable",
        "reason": reason,
        "detail": detail,
        "served_by": "language_pack",
        "probe_seconds": 0.0,
    }


def _kill_process_group(proc: subprocess.Popen) -> None:
    """SIGKILL the probe's whole process group (the child was started in its
    own session, so its pgid is its pid) — no orphaned grandchild can keep
    the output pipes open and wedge the parent."""
    try:
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGKILL)
            return
    except (ProcessLookupError, PermissionError):
        pass
    proc.kill()


def _drain(proc: subprocess.Popen, timeout: float = 5.0) -> tuple[str, str]:
    """Reap a killed child's pipes with a bounded wait (belt and braces)."""
    try:
        return proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            return proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:  # pragma: no cover — SIGKILL reaps
            return "", ""


def _probe_one(name: str, probe_timeout: float) -> dict:
    """Probe ONE language in an isolated subprocess. Never raises, never
    hangs: the child runs in its own session and is SIGKILLed as a whole
    process group when *probe_timeout* fires; the timeout itself is recorded
    as ``unresolvable``/``timeout`` with the language's name (the entry key).
    """
    started = time.monotonic()
    entry = _empty_unresolvable()
    try:
        proc = subprocess.Popen(
            [sys.executable, "-c", _PROBE_PROGRAM, name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            start_new_session=True,
        )
    except OSError as exc:
        entry["detail"] = (
            f"probe subprocess spawn failed: {type(exc).__name__}: {exc}"
        )
        entry["probe_seconds"] = round(time.monotonic() - started, 2)
        return entry

    try:
        out, err = proc.communicate(timeout=probe_timeout)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        _drain(proc)
        entry = _empty_unresolvable(
            "timeout",
            "probe subprocess exceeded the "
            f"{probe_timeout:g}s per-language limit and was killed",
        )
        entry["probe_seconds"] = round(time.monotonic() - started, 2)
        return entry

    entry["probe_seconds"] = round(time.monotonic() - started, 2)
    verdict = None
    for line in reversed((out or "").strip().splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            verdict = json.loads(line)
        except json.JSONDecodeError:
            verdict = None
        else:
            break
    if not isinstance(verdict, dict) or verdict.get("status") not in STATUSES:
        tail = _sanitize_detail(err or "", limit=240)
        entry["detail"] = (
            f"probe subprocess produced no verdict (rc={proc.returncode}); "
            f"stderr tail: {tail}"
        )
        return entry

    entry["status"] = verdict["status"]
    reason = verdict.get("reason")
    if verdict["status"] == "unresolvable" and reason not in UNRESOLVABLE_REASONS:
        reason = "error"
    entry["reason"] = reason
    entry["detail"] = _sanitize_detail(str(verdict.get("detail", "")))
    served_by = verdict.get("served_by")
    entry["served_by"] = (
        served_by if served_by in ("direct_wheel", "language_pack")
        else "language_pack"
    )
    # G1b flavor fields (both optional; plain ok entries carry neither):
    # ``guarded_parse`` — the ok was earned under the subprocess parse
    # watchdog (healthy-with-watchdog); ``filtered_artifacts`` — how many
    # known grammar artifact traits the ok parse filtered (G1a).
    if verdict.get("guarded_parse") is True:
        entry["guarded_parse"] = True
    filtered = verdict.get("filtered_artifacts")
    if isinstance(filtered, int) and filtered > 0:
        entry["filtered_artifacts"] = filtered
    return entry


def _probe_all(
    names: list[str],
    probe_timeout: float,
    max_workers: int,
    progress: bool = True,
) -> dict[str, dict]:
    """Probe every name, parallelized across isolated subprocesses."""
    results: dict[str, dict] = {}
    total = len(names)
    done = 0
    workers = max(1, min(max_workers, total))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_name = {
            pool.submit(_probe_one, name, probe_timeout): name for name in names
        }
        for future in concurrent.futures.as_completed(future_to_name):
            name = future_to_name[future]
            try:
                results[name] = future.result()
            except Exception as exc:  # noqa: BLE001 — pragma: no cover; harness must not raise
                entry = _empty_unresolvable(
                    "error", f"probe harness failure: {type(exc).__name__}: {exc}"
                )
                results[name] = entry
            done += 1
            if progress and (done % 25 == 0 or done == total):
                print(
                    f"[pack_census] probed {done}/{total}",
                    file=sys.stderr,
                    flush=True,
                )
    return results


def pack_language_names() -> tuple[list[str], str]:
    """Enumerate the installed pack's language names introspectively (no
    grammar is loaded — safe and offline) and return ``(sorted names,
    distribution version)``.

    Raises :class:`PackUnavailableError` when the pack is absent or exposes
    no name list. Handles both pack lines: the bundled 0.x
    ``SupportedLanguage`` typing Literal and a 1.x-style
    ``SUPPORTED_LANGUAGES`` sequence.
    """
    try:
        pack = importlib.import_module(PACK_MODULE)
    except ImportError as exc:
        raise PackUnavailableError(
            f"optional dependency {PACK_MODULE} is not installed in this "
            "environment (it ships via fastedit's `all-grammars` extra); the "
            "census cannot describe a pack that is absent"
        ) from exc
    try:
        version = importlib.metadata.version(PACK_MODULE)
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        version = "unknown"
    names = getattr(pack, "SUPPORTED_LANGUAGES", None)
    if not names:
        literal = getattr(pack, "SupportedLanguage", None)
        names = list(typing.get_args(literal)) if literal is not None else []
    cleaned = sorted({str(name) for name in names})
    if not cleaned:
        raise PackUnavailableError(
            f"installed {PACK_MODULE} exposes no introspectable language "
            "name list (neither SUPPORTED_LANGUAGES nor SupportedLanguage)"
        )
    return cleaned, version


def census_universe() -> tuple[list[str], list[str], str]:
    """The census universe: every pack name plus fastedit's wheel-added
    canonical names (declarative LANGUAGE_ALIASES entries the pack does not
    list — e.g. sub-grammars like ``markdown_inline``/``xml_dtd``).

    Returns ``(universe, wheel_added_names, pack_version)``, all sorted.
    """
    pack_names, pack_version = pack_language_names()
    wheel_added = sorted(set(LANGUAGE_ALIASES) - set(pack_names))
    universe = sorted(set(pack_names) | set(wheel_added))
    return universe, wheel_added, pack_version


# --------------------------------------------------------------------------
# Snapshot (= cache) I/O
# --------------------------------------------------------------------------

def load_snapshot(path: Path = SNAPSHOT_PATH) -> dict | None:
    """Load the committed census snapshot, or None when absent."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _snapshot_payload(report: dict) -> dict:
    """The snapshot-stable projection of a report (drops wall-clock totals)."""
    return {key: value for key, value in report.items() if key != "probe_runtime_seconds"}


def write_snapshot(report: dict, path: Path = SNAPSHOT_PATH) -> Path:
    """Write the snapshot (deterministic ordering, trailing newline)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_snapshot_payload(report), indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def _cached_results(pack_version: str) -> dict[str, dict]:
    """Per-language results reused from the snapshot, or {} when the cache is
    absent or stale (pack version changed / probe schema bumped)."""
    snapshot = load_snapshot()
    if snapshot is None:
        return {}
    if snapshot.get("probe_version") != PROBE_VERSION:
        return {}
    if snapshot.get("pack", {}).get("version") != pack_version:
        return {}
    languages = snapshot.get("languages")
    if not isinstance(languages, dict):
        return {}
    return {
        name: entry
        for name, entry in languages.items()
        if isinstance(entry, dict)
        and entry.get("status") in STATUSES
        and "detail" in entry
        and "served_by" in entry
    }


# --------------------------------------------------------------------------
# Census
# --------------------------------------------------------------------------

def _summarize(languages: dict[str, dict]) -> dict:
    """Breakdown with grouped names (deterministically sorted)."""
    by_status: dict[str, list[str]] = {status: [] for status in STATUSES}
    for name, entry in languages.items():
        by_status[entry["status"]].append(name)
    for names in by_status.values():
        names.sort()
    by_reason: dict[str, list[str]] = {}
    for name in by_status["unresolvable"]:
        by_reason.setdefault(languages[name].get("reason") or "error", []).append(
            name
        )
    for names in by_reason.values():
        names.sort()
    return {
        "total_names": len(languages),
        "resolvable": len(by_status["ok"]) + len(by_status["parse_degraded"]),
        "ok": len(by_status["ok"]),
        "parse_degraded": len(by_status["parse_degraded"]),
        "unresolvable": len(by_status["unresolvable"]),
        "ok_names": by_status["ok"],
        "parse_degraded_names": by_status["parse_degraded"],
        "unresolvable_names": by_status["unresolvable"],
        "unresolvable_by_reason": by_reason,
    }


def census(
    force: bool = False,
    probe_timeout: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
    max_workers: int = DEFAULT_MAX_WORKERS,
    progress: bool = True,
) -> dict:
    """Run the census and return the structured report.

    With ``force=False`` the committed snapshot doubles as the cache: only
    names missing from it are probed (a fresh environment re-probes
    everything; a matching one probes nothing). ``force=True`` re-probes the
    whole universe. The report contains:

    * ``probe_version`` / ``pack`` (module, version, name_count);
    * ``wheel_added_names`` — canonical names fastedit serves from its own
      wheels that the pack does not list;
    * ``languages`` — per-name entry: ``status``, ``reason`` (for
      unresolvable/degraded), ``detail``, ``served_by``, ``probe_seconds``;
    * ``summary`` — the headline counts plus grouped name lists;
    * ``probe_runtime_seconds`` — wall-clock of THIS invocation's probing
      (stripped from the snapshot; see :func:`write_snapshot`).
    """
    started = time.monotonic()
    universe, wheel_added, pack_version = census_universe()
    cached = {} if force else _cached_results(pack_version)
    todo = [name for name in universe if name not in cached]
    probed = (
        _probe_all(todo, probe_timeout, max_workers, progress=progress)
        if todo
        else {}
    )
    languages = {
        name: (probed[name] if name in probed else cached[name]) for name in universe
    }
    report = {
        "probe_version": PROBE_VERSION,
        "pack": {
            "module": PACK_MODULE,
            "version": pack_version,
            "name_count": len(universe) - len(wheel_added),
        },
        "wheel_added_names": wheel_added,
        "languages": languages,
        "summary": _summarize(languages),
    }
    report["probe_runtime_seconds"] = round(time.monotonic() - started, 1)
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI: run the census, refresh the snapshot, print the headline."""
    parser = argparse.ArgumentParser(
        description=(
            "Census of every language fastedit can resolve+parse in this venv "
            "(isolated, timeout-bounded per-language probes)."
        )
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-probe every language, ignoring the cached snapshot",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_PROBE_TIMEOUT_SECONDS,
        help="per-language probe subprocess timeout in seconds (default 10)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="parallel probe subprocesses (default 12)",
    )
    args = parser.parse_args(argv)
    try:
        report = census(
            force=args.force,
            probe_timeout=args.timeout,
            max_workers=args.workers,
        )
    except PackUnavailableError as exc:
        print(f"pack_census: unavailable: {exc}", file=sys.stderr)
        return 2
    path = write_snapshot(report)
    summary = report["summary"]
    print(
        f"pack {report['pack']['module']}=={report['pack']['version']}: "
        f"{report['pack']['name_count']} names "
        f"(+{len(report['wheel_added_names'])} wheel-added: "
        f"{', '.join(report['wheel_added_names']) or 'none'})"
    )
    print(
        f"TOTAL resolvable: {summary['resolvable']}/{summary['total_names']} "
        f"(ok={summary['ok']}, parse_degraded={summary['parse_degraded']}, "
        f"unresolvable={summary['unresolvable']})"
    )
    for reason, names in sorted(summary["unresolvable_by_reason"].items()):
        print(f"  unresolvable[{reason}] ({len(names)}): {', '.join(names)}")
    if summary["parse_degraded_names"]:
        print(
            f"  parse_degraded ({len(summary['parse_degraded_names'])}): "
            f"{', '.join(summary['parse_degraded_names'])}"
        )
    print(f"probe loop runtime: {report['probe_runtime_seconds']}s")
    print(f"snapshot written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
