"""Step F1 golden tests — the language-pack census snapshot.

The census (:mod:`pack_census`) probes EVERY language of the installed
``tree_sitter_language_pack`` (plus fastedit's wheel-added canonical names)
through fastedit's real resolver, one isolated subprocess per language with a
hard per-language timeout. These tests pin the snapshot that captures the
result:

* the snapshot matches THIS environment's pack and wheel table (a stale
  snapshot fails with regeneration instructions);
* the summary is internally consistent with the per-language entries;
* the headline floor: the pack puts fastedit's resolvable language count in
  the hundreds;
* every ``EXTENSION_TO_LANGUAGE`` name is resolvable in the census —
  ``detect_language()`` promises those on a default install;
* unresolvable probes (including timeouts) carry a reason and detail, with
  the language's name discoverable (entry key + grouped name list);
* the probe harness itself is live: one real isolated probe resolves a core
  wheel language and classifies a nonsense name as ``grammar_unavailable``.

Regeneration — snapshot == cache, so it must be refreshed explicitly:

* ``FASTEDIT_REGEN_CENSUS=1 uv run pytest tests/test_pack_census.py``
  re-probes the whole universe (bounded per-language, ~1-3 min) and rewrites
  ``tests/golden/pack_census.json`` before asserting;
* ``uv run python tests/pack_census.py [--force]`` does the same from the
  CLI and prints the breakdown (cached re-runs are instant).

The tests skip (with the explicit reason) when the optional pack — shipped
via fastedit's ``all-grammars`` extra — is not installed: the census
describes this venv's pack and cannot run without it.
"""

from __future__ import annotations

import importlib.util
import json
import os
from collections import Counter

import pack_census
import pytest

from fastedit.data_gen.ast_analyzer import EXTENSION_TO_LANGUAGE

pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec(pack_census.PACK_MODULE) is None,
        reason=(
            "optional aggregate pack tree-sitter-language-pack is not "
            "installed (fastedit ships it via the all-grammars extra) — the "
            "F1 census describes THIS venv's pack and cannot run without it"
        ),
    ),
]


@pytest.fixture(scope="module")
def snapshot():
    """The committed census snapshot, regenerated first when
    ``FASTEDIT_REGEN_CENSUS=1`` (a full bounded re-probe)."""
    if os.environ.get(pack_census.REGEN_ENV_VAR) == "1":
        report = pack_census.census(force=True)
        pack_census.write_snapshot(report)
    if not pack_census.SNAPSHOT_PATH.exists():
        pytest.skip(
            "census snapshot tests/golden/pack_census.json does not exist yet "
            "— generate it with `FASTEDIT_REGEN_CENSUS=1 uv run pytest "
            "tests/test_pack_census.py` or "
            "`uv run python tests/pack_census.py --force`"
        )
    return json.loads(pack_census.SNAPSHOT_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Snapshot ↔ environment
# ---------------------------------------------------------------------------

def test_census_snapshot_matches_this_environment(snapshot):
    """The snapshot must describe exactly the pack + wheel table installed
    here — otherwise it is stale and must be regenerated."""
    universe, wheel_added, pack_version = pack_census.census_universe()
    snap_names = sorted(snapshot["languages"])
    missing = sorted(set(universe) - set(snap_names))
    stale = sorted(set(snap_names) - set(universe))
    assert not missing and not stale, (
        "census snapshot is stale for this environment: "
        f"{len(missing)} names missing (e.g. {missing[:5]}), "
        f"{len(stale)} no longer present (e.g. {stale[:5]}). Regenerate with "
        f"`{pack_census.REGEN_ENV_VAR}=1 uv run pytest "
        "tests/test_pack_census.py` or "
        "`uv run python tests/pack_census.py --force`"
    )
    assert snapshot["wheel_added_names"] == wheel_added
    assert snapshot["pack"]["module"] == pack_census.PACK_MODULE
    assert snapshot["pack"]["version"] == pack_version
    assert snapshot["pack"]["name_count"] == len(universe) - len(wheel_added)
    assert snapshot["probe_version"] == pack_census.PROBE_VERSION


def test_snapshot_serialization_is_deterministic(snapshot):
    """The golden file must be exactly the canonical serialization of its
    own data (indent=2, sorted keys, trailing newline) — future regenerations
    stay diff-stable."""
    text = pack_census.SNAPSHOT_PATH.read_text(encoding="utf-8")
    assert text == json.dumps(snapshot, indent=2, sort_keys=True) + "\n"


# ---------------------------------------------------------------------------
# Summary consistency + headline
# ---------------------------------------------------------------------------

def test_census_summary_is_internally_consistent(snapshot):
    """Every count and grouped name list in ``summary`` must be recomputable
    from the per-language entries, and every entry must carry the full
    schema."""
    languages = snapshot["languages"]
    assert languages, "the census must cover a non-empty universe"
    statuses = Counter(entry["status"] for entry in languages.values())
    assert set(statuses) <= set(pack_census.STATUSES)

    summary = snapshot["summary"]
    assert summary["total_names"] == len(languages)
    assert summary["ok"] == statuses["ok"]
    assert summary["parse_degraded"] == statuses["parse_degraded"]
    assert summary["unresolvable"] == statuses["unresolvable"]
    assert summary["resolvable"] == summary["ok"] + summary["parse_degraded"]
    assert summary["resolvable"] + summary["unresolvable"] == summary["total_names"]
    assert summary["ok_names"] == sorted(
        name for name, entry in languages.items() if entry["status"] == "ok"
    )
    assert summary["parse_degraded_names"] == sorted(
        name for name, entry in languages.items()
        if entry["status"] == "parse_degraded"
    )
    assert summary["unresolvable_names"] == sorted(
        name for name, entry in languages.items()
        if entry["status"] == "unresolvable"
    )
    reasons: dict[str, list[str]] = {}
    for name in summary["unresolvable_names"]:
        reasons.setdefault(languages[name]["reason"], []).append(name)
    assert summary["unresolvable_by_reason"] == {
        reason: sorted(names) for reason, names in reasons.items()
    }

    for name, entry in languages.items():
        assert entry["served_by"] in ("direct_wheel", "language_pack"), name
        assert isinstance(entry["detail"], str) and entry["detail"], name
        assert isinstance(entry.get("probe_seconds"), (int, float)), name
        if entry["status"] == "unresolvable":
            assert entry["reason"] in pack_census.UNRESOLVABLE_REASONS, name


def test_census_headline_resolvable_floor(snapshot):
    """THE F1 headline: with the pack installed, fastedit resolves+parses
    HUNDREDS of languages. A degenerate environment (pack broken, every
    probe timing out) must fail here rather than pass vacuously."""
    summary = snapshot["summary"]
    assert summary["resolvable"] >= 100, (
        f"only {summary['resolvable']}/{summary['total_names']} languages "
        f"resolved — expected the ~170-language pack to put fastedit in the "
        f"hundreds; unresolvable by reason: "
        f"{summary['unresolvable_by_reason']}"
    )


def test_extension_wired_languages_are_resolvable_in_census(snapshot):
    """detect_language() promises every EXTENSION_TO_LANGUAGE value on a
    default install — the census must show each one resolvable (``ok`` or
    ``parse_degraded``, never ``unresolvable``)."""
    languages = snapshot["languages"]
    wired = sorted(set(EXTENSION_TO_LANGUAGE.values()))
    assert wired, "the extension table must be non-empty"
    for name in wired:
        entry = languages.get(name)
        assert entry is not None, (
            f"extension-wired language {name!r} is missing from the census "
            "universe"
        )
        assert entry["status"] in ("ok", "parse_degraded"), (
            f"detect_language() promises {name!r}, but the census classified "
            f"it {entry['status']}: {entry['detail']}"
        )


def test_unresolvable_probes_carry_reason_and_name(snapshot):
    """Unresolvable entries — timeouts included — must say WHY, and the
    language's name must stay discoverable: the entry key IS the name and
    every timeout is grouped under reason ``timeout``."""
    languages = snapshot["languages"]
    for name, entry in languages.items():
        if entry["status"] != "unresolvable":
            continue
        assert entry["reason"] in pack_census.UNRESOLVABLE_REASONS, name
        assert entry["detail"], f"unresolvable {name} needs a detail"
    timed_out = [
        name for name, entry in languages.items()
        if entry.get("reason") == "timeout"
    ]
    for name in timed_out:
        assert name in snapshot["summary"]["unresolvable_by_reason"]["timeout"]
        assert "killed" in languages[name]["detail"], name


# ---------------------------------------------------------------------------
# The probe harness itself (one real bounded probe, independent of the cache)
# ---------------------------------------------------------------------------

def test_probe_harness_is_live_isolated_and_bounded(snapshot):
    """One REAL probe through the isolated-subprocess harness (its own
    session, hard per-language kill) proves the harness works independently
    of the cached snapshot: a core wheel language resolves ``ok`` via its
    direct wheel, and a name no wheel and no pack ships is classified
    ``unresolvable``/``grammar_unavailable``."""
    live_python = pack_census._probe_one("python", probe_timeout=30)
    assert live_python["status"] == "ok", live_python["detail"]
    assert live_python["served_by"] == "direct_wheel"
    assert live_python["status"] == snapshot["languages"]["python"]["status"]

    fake = "definitely_not_a_fastedit_language"
    live_fake = pack_census._probe_one(fake, probe_timeout=30)
    assert live_fake["status"] == "unresolvable", live_fake["detail"]
    assert live_fake["reason"] == "grammar_unavailable"
    assert live_fake["served_by"] == "language_pack"
    assert fake in live_fake["detail"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
