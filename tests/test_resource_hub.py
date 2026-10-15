"""GLOBAL RESOURCE GOVERNOR tests + hot-path equivalence tests.

Two sections:

  Section 1 — ``mask_string_spans`` optimization equivalence (PART 1).
  The hot-path optimizer replaces the per-char ``list(text)`` scanner with a
  compiled-regex scanner. These tests pin the contract: byte-identical
  output against a REFERENCE COPY of the pre-optimization implementation on
  adversarial inputs (escaped quotes, CRLF, lone CR, multi-byte astral
  chars, unterminated strings, backtick templates, triple quotes, NUL
  masks, empty text) plus a seeded fuzz sweep. They pass BEFORE the
  optimization (characterization) and must keep passing AFTER it.

  Section 2 — the resource hub (PART 2): ``fastedit.resource_hub``.
  Cross-process slot semaphores under ``~/.fastedit`` (FASTEDIT_HUB_DIR
  sandboxed here): limits (defaults, limits.json, env; fail-loud on bad
  values), flock-based slot exhaustion + queueing + timeout, reentrancy,
  state files written/pruned, dead-pid pruning, heavy-file threshold,
  doctor's hub section, and the install-dev.sh --check hub section.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import threading
import time

import pytest

from fastedit.split_join import mask_string_spans

# ===========================================================================
# Section 1 — mask_string_spans equivalence (PART 1)
# ===========================================================================

# REFERENCE: the pre-optimization implementation, copied VERBATIM from
# src/fastedit/split_join.py before the regex-scanner rewrite (git history
# pins the copy). The optimizer must reproduce it byte-for-byte.


def _reference_mask_string_spans(text: str, mask: str = "\x00") -> str:
    """The ORIGINAL per-char scanner (pre-optimization reference)."""

    def string_end(start: int, delim: str, multi_line: bool) -> int:
        i = start
        while i < len(text):
            c = text[i]
            if c == "\\":
                i += 2  # escaped char (quote, backslash, newline, ...)
                continue
            if text.startswith(delim, i):
                return i + len(delim)
            if not multi_line and c == "\n":
                return i  # unterminated single-line string ends at the break
            if c not in "\r\n":
                chars[i] = mask
            i += 1
        return len(text)  # unterminated multi-line span runs to EOF

    chars = list(text)
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == "\\":
            i += 2  # escaped pair — never a delimiter start
            continue
        if text.startswith(('"""', "'''"), i):
            delim = text[i:i + 3]
            i = string_end(i + 3, delim, multi_line=True)
        elif c in "\"'`":
            i = string_end(i + 1, c, multi_line=c == "`")
        else:
            i += 1
    return "".join(chars)


_ADVERSARIAL_INPUTS = [
    "",
    "plain text, no strings at all",
    'x = "simple"',
    "x = 'single'",
    "x = `template`",
    'x = """triple\nline"""',
    "y = '''triple\nsingle'''",
    'x = "escaped \\" quote inside"',
    "x = 'escaped \\' quote inside'",
    'x = "backslash \\\\ escape"',
    'x = "trailing backslash \\"',
    "x = 'unterminated",
    'x = "unterminated',
    "x = `unterminated template",
    'x = """unterminated triple',
    'crlf = "line1\r\nline2\r\n"',  # CRLF file, quoted CRLF line
    'lone_cr = "a\rb"',  # lone CR inside a single-line string
    "multi = `a\r\nb\rc\nd`",  # terminators inside a multi-line template
    "emoji = \"🚀🎉 unicode 😀 inside\"",  # multi-byte content
    "emoji_bare = 🚀 not in a string",
    "mixed = 'a' + \"b\" + `c` + \"\"\"d\"\"\"",
    "empty = \"\"",
    "empty3 = \"\"\"\"\"\"",
    "adjacent = \"\"\"\"  # quote soup",
    "quote_soup = \"''\"''\"''\"",
    "continuation = \\\n    next_line_code()",
    "x = \"a\"  # trailing \\",  # trailing backslash outside strings
    "x = '\\'\\\\'",
    'tabbed = "\tindent\tkept"',
    "nul_mask = 'default NUL mask on " + chr(0) + " content'",
    "# ... existing code ...",  # a marker phrase (unmasked)
    "x = \"# ... existing code ...\"",  # marker INSIDE a string (masked)
]


@pytest.mark.parametrize("text", _ADVERSARIAL_INPUTS)
def test_mask_string_spans_equivalence_adversarial(text):
    """Optimized masker == reference byte-for-byte on adversarial inputs."""
    assert mask_string_spans(text) == _reference_mask_string_spans(text)


@pytest.mark.parametrize("mask", ["\x00", "#", "·", "AB"])
def test_mask_string_spans_equivalence_custom_mask(mask):
    texts = ['a = "hidden"', "b = 'x\ry'", "c = `t\r\nu`"]
    for text in texts:
        assert mask_string_spans(text, mask) == _reference_mask_string_spans(
            text, mask,
        )


def test_mask_string_spans_equivalence_fuzz():
    """Seeded fuzz: 500 random strings, byte-identical to the reference.

    Alphabet is weighted toward the scanner's decision points: quote
    characters, backslashes, newlines/CRs, triple-quote runs and multi-byte
    characters, so the fuzz explores delimiter/escape/terminator corners
    instead of long runs of plain letters.
    """
    rng = random.Random(0xF00D)
    alphabet = (
        ["a", "b", " ", "=", ";", "#"] * 3
        + ['"', "'", "`", "\\", "\n", "\r"] * 6
        + ['"""', "'''"] * 3
        + ["…", "🚀", "\x00"]
    )
    for case in range(500):
        length = rng.randrange(0, 60)
        text = "".join(rng.choice(alphabet) for _ in range(length))
        got = mask_string_spans(text)
        want = _reference_mask_string_spans(text)
        assert got == want, f"fuzz case {case} diverged: {text!r}"


def test_mask_string_spans_preserved_properties():
    """Structural properties the pipeline relies on (same length, EOLs kept)."""
    for text in _ADVERSARIAL_INPUTS:
        masked = mask_string_spans(text)
        assert len(masked) == len(text)
        # Every \r and \n survives in place: line counts and offsets are
        # identical between text and the masked copy.
        assert [c for c in text if c in "\r\n"] == [
            c for c in masked if c in "\r\n"
        ]
        # Masking is idempotent (the default mask char is neither a
        # delimiter nor an escape char).
        assert mask_string_spans(masked) == masked


def test_mask_string_spans_string_content_masked():
    """The actual SEMANTICS: interior content masked, delimiters/EOLs not."""
    assert mask_string_spans('x = "abc"') == 'x = "\x00\x00\x00"'
    assert mask_string_spans('x = "a" + "b"') == 'x = "\x00" + "\x00"'
    # A real newline TERMINATES a single-line string (content after it is
    # outside); the backtick multi-line form passes terminators through.
    assert mask_string_spans('x = `a\nb`') == 'x = `\x00\n\x00`'
    assert mask_string_spans('x = "a\nb"') == 'x = "\x00\nb"'


# ===========================================================================
# Section 2 — the resource hub (PART 2)
# ===========================================================================

_WINDOWS = sys.platform == "win32"


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """Sandboxed hub root + clean registry + clean limit env for one test."""
    hub_root = tmp_path / "hub"
    monkeypatch.setenv("FASTEDIT_HUB_DIR", str(hub_root))
    for name in (
        "FASTEDIT_MAX_MODEL_INSTANCES",
        "FASTEDIT_MAX_HEAVY_JOBS",
        "FASTEDIT_HEAVY_FILE_BYTES",
        "FASTEDIT_SLOT_WAIT_TIMEOUT_S",
    ):
        monkeypatch.delenv(name, raising=False)
    import fastedit.resource_hub as rh

    rh._REGISTRY.clear()
    yield rh
    # Release anything a test left held so slot flocks never leak between
    # tests sharing the process.
    for record in list(rh._REGISTRY.values()):
        record.release()
    rh._REGISTRY.clear()


class _ForeignHolder:
    """Simulates ANOTHER process holding one slot file.

    flock excludes per open-file-description, so a raw fd taken directly on
    the slot file (bypassing the hub's reentrancy registry) conflicts with
    the hub's own acquisition exactly as a second process would. Optionally
    stamps the pid so the queue line can name it, and can release itself
    from a timer thread.
    """

    def __init__(self, lock_file, stamp_pid: int | None = None):
        import fcntl

        self.lock_file = lock_file
        self.fd = os.open(lock_file, os.O_RDWR | os.O_CREAT, 0o666)
        fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if stamp_pid is not None:
            payload = f"pid={stamp_pid}\nstarted={time.time():.6f}\n".encode()
            os.lseek(self.fd, 0, os.SEEK_SET)
            os.write(self.fd, payload)
            os.ftruncate(self.fd, len(payload))

    def release_later(self, delay: float) -> threading.Timer:
        timer = threading.Timer(delay, self.release)
        timer.daemon = True
        timer.start()
        return timer

    def release(self) -> None:
        import fcntl

        try:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
        finally:
            os.close(self.fd)


# ---------------------------------------------------------------------------
# Limits: defaults, limits.json, env, fail-loud parsing
# ---------------------------------------------------------------------------


class TestLimits:
    def test_defaults(self, hub):
        limits = hub.load_limits()
        assert limits.max_model_instances == 2
        assert limits.max_heavy_jobs == 2
        assert limits.heavy_file_bytes == 10_000_000
        assert limits.slot_wait_timeout_s == 600

    def test_limits_json_overrides_defaults(self, hub):
        hub.limits_file().write_text(json.dumps({
            "max_model_instances": 3,
            "max_heavy_jobs": 1,
            "heavy_file_bytes": 5000,
            "slot_wait_timeout_s": 12.5,
        }))
        limits = hub.load_limits()
        assert limits.max_model_instances == 3
        assert limits.max_heavy_jobs == 1
        assert limits.heavy_file_bytes == 5000
        assert limits.slot_wait_timeout_s == 12.5

    def test_env_overrides_limits_json(self, hub, monkeypatch):
        hub.limits_file().write_text(json.dumps({"max_model_instances": 3}))
        monkeypatch.setenv("FASTEDIT_MAX_MODEL_INSTANCES", "1")
        assert hub.load_limits().max_model_instances == 1

    def test_env_only_overrides_its_own_key(self, hub, monkeypatch):
        monkeypatch.setenv("FASTEDIT_HEAVY_FILE_BYTES", "1")
        limits = hub.load_limits()
        assert limits.heavy_file_bytes == 1
        assert limits.max_model_instances == 2  # default untouched

    def test_bad_json_fails_loud(self, hub):
        hub.limits_file().write_text("{not json")
        with pytest.raises(ValueError, match="limits"):
            hub.load_limits()

    def test_non_object_json_fails_loud(self, hub):
        hub.limits_file().write_text("[1, 2]")
        with pytest.raises(ValueError):
            hub.load_limits()

    def test_unknown_key_fails_loud(self, hub):
        hub.limits_file().write_text(json.dumps({"max_model_instancez": 2}))
        with pytest.raises(ValueError, match="max_model_instancez"):
            hub.load_limits()

    def test_zero_slots_fails_loud(self, hub):
        hub.limits_file().write_text(json.dumps({"max_model_instances": 0}))
        with pytest.raises(ValueError):
            hub.load_limits()

    def test_negative_heavy_bytes_fails_loud(self, hub):
        hub.limits_file().write_text(json.dumps({"heavy_file_bytes": -1}))
        with pytest.raises(ValueError):
            hub.load_limits()

    def test_malformed_env_fails_loud(self, hub, monkeypatch):
        monkeypatch.setenv("FASTEDIT_MAX_MODEL_INSTANCES", "two")
        with pytest.raises(ValueError):
            hub.load_limits()

    def test_hub_dir_override_must_be_absolute(self, monkeypatch):
        monkeypatch.setenv("FASTEDIT_HUB_DIR", "relative/path")
        import fastedit.resource_hub as rh

        with pytest.raises(ValueError, match="absolute"):
            rh.hub_dir()

    def test_hub_dir_created_on_use(self, tmp_path, monkeypatch):
        nested = tmp_path / "deep" / "hub"
        monkeypatch.setenv("FASTEDIT_HUB_DIR", str(nested))
        import fastedit.resource_hub as rh

        assert not nested.exists()  # nothing created yet
        rh.limits_dir()
        assert (nested / "limits").is_dir()
        rh.state_dir()
        assert (nested / "state").is_dir()


# ---------------------------------------------------------------------------
# Slot acquisition: exhaustion, queueing, timeout, reentrancy, crash safety
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_WINDOWS, reason="POSIX flock fd semantics")
class TestSlotAcquisition:
    def test_first_acquire_ok_and_state_written(self, hub, monkeypatch):
        monkeypatch.setenv("FASTEDIT_MAX_MODEL_INSTANCES", "1")
        with hub.acquire_model_slot(file="big.py"):
            assert (hub.limits_dir() / "model.slot.0").exists()
            state_files = list(hub.state_dir().glob("model-*.json"))
            assert len(state_files) == 1
            data = json.loads(state_files[0].read_text())
            assert data["pid"] == os.getpid()
            assert data["file"] == "big.py"
            assert "started_epoch" in data
            assert isinstance(data["argv"], list)
            holders = hub.read_hub_state()
            assert [(h["kind"], h["slot"]) for h in holders] == [("model", 0)]
        # Release prunes the state file; the slot file stays (crash-safe).
        assert not list(hub.state_dir().glob("model-*.json"))
        assert (hub.limits_dir() / "model.slot.0").exists()

    def test_slot_exhaustion_queues_then_succeeds(
        self, hub, monkeypatch, capsys,
    ):
        monkeypatch.setenv("FASTEDIT_MAX_MODEL_INSTANCES", "1")
        monkeypatch.setenv("FASTEDIT_SLOT_WAIT_TIMEOUT_S", "30")
        slot_file = hub.limits_dir() / "model.slot.0"
        holder = _ForeignHolder(slot_file, stamp_pid=os.getpid())
        holder.release_later(1.0)  # stub holder frees the slot after 1s
        t0 = time.monotonic()
        with hub.acquire_model_slot(file="queued.py"):
            waited = time.monotonic() - t0
            assert waited >= 0.9  # actually queued until the release
        err = capsys.readouterr().err
        assert "waiting for a model slot: 1/1 busy" in err
        assert f"pids {os.getpid()}" in err
        assert "held" in err

    def test_timeout_fails_loud_naming_limits_file(self, hub, monkeypatch):
        monkeypatch.setenv("FASTEDIT_MAX_MODEL_INSTANCES", "1")
        monkeypatch.setenv("FASTEDIT_SLOT_WAIT_TIMEOUT_S", "0.5")
        slot_file = hub.limits_dir() / "model.slot.0"
        holder = _ForeignHolder(slot_file)
        try:
            with pytest.raises(
                hub.SlotWaitTimeout, match=r"queued too long.*raise limits",
            ):
                hub.acquire_model_slot().__enter__()
        finally:
            holder.release()

    def test_timeout_message_naming_is_heavy_kind(self, hub, monkeypatch):
        monkeypatch.setenv("FASTEDIT_MAX_HEAVY_JOBS", "1")
        monkeypatch.setenv("FASTEDIT_SLOT_WAIT_TIMEOUT_S", "0.5")
        slot_file = hub.limits_dir() / "heavy.slot.0"
        holder = _ForeignHolder(slot_file)
        try:
            with pytest.raises(
                hub.SlotWaitTimeout, match="heavy slot",
            ), hub.acquire_heavy_slot(file="x.py"):
                pass
        finally:
            holder.release()

    def test_second_slot_available_without_queueing(self, hub, monkeypatch, capsys):
        monkeypatch.setenv("FASTEDIT_MAX_MODEL_INSTANCES", "2")
        holder = _ForeignHolder(hub.limits_dir() / "model.slot.0", stamp_pid=424242)
        try:
            with hub.acquire_model_slot():
                pass  # took model.slot.1 immediately — no queue line
        finally:
            holder.release()
        assert "waiting for a model slot" not in capsys.readouterr().err

    def test_reentrancy_inner_release_keeps_slot(self, hub, monkeypatch):
        monkeypatch.setenv("FASTEDIT_MAX_MODEL_INSTANCES", "1")
        outer = hub.acquire_model_slot()
        outer.__enter__()
        inner = hub.acquire_model_slot()
        inner.__enter__()
        inner.__exit__(None, None, None)
        # Inner release must NOT unlock: a foreign fd still cannot take the
        # only slot while the outer lease is live.
        slot_file = hub.limits_dir() / "model.slot.0"
        with pytest.raises(OSError):
            _ForeignHolder(slot_file)
        outer.__exit__(None, None, None)
        _ForeignHolder(slot_file).release()  # now acquirable

    def test_state_file_removed_only_at_outermost_release(self, hub, monkeypatch):
        monkeypatch.setenv("FASTEDIT_MAX_HEAVY_JOBS", "1")
        outer = hub.acquire_heavy_slot(file="big.py")
        outer.__enter__()
        inner = hub.acquire_heavy_slot(file="big.py")
        inner.__enter__()
        inner.__exit__(None, None, None)
        assert list(hub.state_dir().glob("heavy-*.json"))
        outer.__exit__(None, None, None)
        assert not list(hub.state_dir().glob("heavy-*.json"))

    def test_state_files_are_per_slot(self, hub, monkeypatch):
        monkeypatch.setenv("FASTEDIT_MAX_HEAVY_JOBS", "2")
        with hub.acquire_heavy_slot(file="a.py"), hub.acquire_model_slot(file="b.py"):
            names = sorted(p.name for p in hub.state_dir().glob("*.json"))
            assert names == ["heavy-0.json", "model-0.json"]


# ---------------------------------------------------------------------------
# Hub state observability: dead-pid pruning
# ---------------------------------------------------------------------------


class TestHubState:
    def _dead_pid(self) -> int:
        proc = subprocess.Popen(
            [sys.executable, "-c", "pass"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        proc.wait()
        return proc.pid

    def test_dead_pid_pruned(self, hub):
        state = hub.state_dir()
        dead = self._dead_pid()
        (state / "model-0.json").write_text(json.dumps({
            "pid": dead, "argv": ["fastedit", "edit"], "file": "x.py",
            "started_epoch": time.time(), "kind": "model", "slot": 0,
        }))
        (state / "heavy-1.json").write_text(json.dumps({
            "pid": os.getpid(), "argv": ["fastedit", "edit"], "file": "y.py",
            "started_epoch": time.time(), "kind": "heavy", "slot": 1,
        }))
        holders = hub.read_hub_state()
        assert [(h["kind"], h["slot"], h["pid"]) for h in holders] == [
            ("heavy", 1, os.getpid()),
        ]
        assert not (state / "model-0.json").exists()  # dead holder pruned
        assert (state / "heavy-1.json").exists()  # live holder kept

    def test_corrupt_state_file_pruned(self, hub):
        (hub.state_dir() / "model-0.json").write_text("{garbage")
        assert hub.read_hub_state() == []
        assert not (hub.state_dir() / "model-0.json").exists()

    def test_missing_state_dir_is_empty(self, hub):
        import shutil

        shutil.rmtree(hub.state_dir())
        assert hub.read_hub_state() == []


# ---------------------------------------------------------------------------
# Heavy-file threshold
# ---------------------------------------------------------------------------


class TestHeavyThreshold:
    def test_threshold_boundary(self, hub, tmp_path, monkeypatch):
        monkeypatch.setenv("FASTEDIT_HEAVY_FILE_BYTES", "100")
        small = tmp_path / "small.py"
        small.write_text("x" * 99)
        exact = tmp_path / "exact.py"
        exact.write_text("x" * 100)
        big = tmp_path / "big.py"
        big.write_text("x" * 101)
        assert hub.is_heavy_file(small) is False
        assert hub.is_heavy_file(exact) is True  # >= threshold
        assert hub.is_heavy_file(big) is True
        assert hub.is_heavy_file(tmp_path / "missing.py") is False

    def test_small_file_skips_slots(self, hub, tmp_path):
        small = tmp_path / "small.py"
        small.write_text("x = 1\n")
        with hub.acquire_heavy_slot_for_path(small):
            assert not (hub.limits_dir() / "heavy.slot.0").exists()
            assert not list(hub.state_dir().glob("*.json"))
            assert hub.read_hub_state() == []

    def test_big_file_takes_slot(self, hub, tmp_path, monkeypatch):
        monkeypatch.setenv("FASTEDIT_HEAVY_FILE_BYTES", "10")
        big = tmp_path / "big.py"
        big.write_text("x" * 11)
        with hub.acquire_heavy_slot_for_path(big):
            assert (hub.limits_dir() / "heavy.slot.0").exists()
            holders = hub.read_hub_state()
            assert len(holders) == 1
            assert holders[0]["file"] == str(big)


# ---------------------------------------------------------------------------
# Doctor hub section + install-dev.sh --check hub section
# ---------------------------------------------------------------------------


class TestDoctorHubSection:
    def test_doctor_reports_hub(self, hub, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("FASTEDIT_HEAVY_FILE_BYTES", "10")
        big = tmp_path / "big.py"
        big.write_text("x" * 11)
        with hub.acquire_heavy_slot_for_path(big):
            from fastedit.doctor import run_doctor

            run_doctor()
        out = capsys.readouterr().out
        assert "resource hub" in out
        assert "max_model_instances=2" in out
        assert "max_heavy_jobs=2" in out
        assert "600" in out  # queue-wait budget
        assert "heavy-0" in out  # active holder named

    def test_doctor_survives_bad_limits(self, hub, capsys):
        hub.limits_file().write_text("{bad")
        from fastedit.doctor import run_doctor

        run_doctor()  # must not raise: diagnostics degrade to a WARN row
        out = capsys.readouterr().out
        assert "resource hub" in out


def test_install_dev_check_has_hub_section():
    """--check's read-only report gains the same hub section (spec wiring)."""
    script = os.path.join(
        os.path.dirname(__file__), "..", "scripts", "install-dev.sh",
    )
    with open(script, encoding="utf-8") as fh:
        content = fh.read()
    assert "== resource hub ==" in content
    assert "resource_hub" in content  # reads the hub through the module


# ===========================================================================
# Cross-process integration: two real processes contend for one slot
# ===========================================================================


@pytest.mark.skipif(_WINDOWS, reason="POSIX flock fd semantics")
class TestTwoProcessContention:
    def test_child_process_queues_until_parent_releases(
        self, hub, tmp_path, monkeypatch,
    ):
        """A real second PROCESS queues while this process holds the slot.

        The child re-execs python with FASTEDIT_MAX_MODEL_INSTANCES=1 and
        the same sandboxed FASTEDIT_HUB_DIR, acquires the model slot, and
        completes only after the parent releases. Its stderr carries the
        queue line — the exact observable the two-process demo relies on.
        """
        monkeypatch.setenv("FASTEDIT_MAX_MODEL_INSTANCES", "1")
        monkeypatch.setenv("FASTEDIT_SLOT_WAIT_TIMEOUT_S", "60")
        script = tmp_path / "child.py"
        script.write_text(
            "import sys, time\n"
            "t0 = time.monotonic()\n"
            "from fastedit.resource_hub import acquire_model_slot\n"
            "with acquire_model_slot(file='child.py'):\n"
            "    waited = time.monotonic() - t0\n"
            "    print(f'WAITED {waited:.1f}s', file=sys.stderr)\n"
            "    print('DONE')\n"
        )
        env = dict(os.environ)
        src = os.path.join(os.path.dirname(__file__), "..", "src")
        env["PYTHONPATH"] = os.pathsep.join(
            [src, env.get("PYTHONPATH", "")],
        )
        proc = subprocess.Popen(
            [sys.executable, str(script)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
        )
        try:
            # Hold the only slot long enough that the child MUST queue.
            with hub.acquire_model_slot(file="parent.py"):
                time.sleep(3.0)
            out, err = proc.communicate(timeout=60)
        finally:
            if proc.poll() is None:  # pragma: no cover — failure cleanup
                proc.kill()
                proc.communicate()
        assert "DONE" in out
        assert "waiting for a model slot" in err
        assert "1/1 busy" in err
        waited_line = [ln for ln in err.splitlines() if ln.startswith("WAITED")]
        assert waited_line, err
        waited = float(waited_line[0].split()[1].rstrip("s"))
        assert waited >= 2.0  # actually queued behind the parent's hold
