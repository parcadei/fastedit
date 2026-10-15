"""Contract tests for scripts/install-dev.sh — the one installer.

The ONE install shape is an editable install from a local clone: running the
script inside a fastedit checkout installs THAT working tree; a standalone
copy (curl|bash — no pyproject.toml beside the script) maintains the
installer-managed clone at ~/.fastedit/src (override: FASTEDIT_CLONE_DIR),
cloned on first run and fetched/hard-reset to the tracked branch on every
run after. There is no source menu, and --dev is accepted as a no-op
compatibility flag. Covers the model-cache preflight (VALID kept / STALE
removed), the all-grammars prompt/flag, the agent-skill install (Vercel
skills CLI, sourced from the SAME clone the package comes from), the revert
path (which removes the default managed clone), the dry-run transcript, the
--check read-only state report, and the successful-install footer (the
from-scratch one-liners plus the stale-main branch pin note), all through
the script's real CLI.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "install-dev.sh"
REPO_ROOT = SCRIPT.parent.parent
FORK_URL = "https://github.com/Emasoft/fastedit"
# The fork's INSTALLABLE branch: the GitHub default branch (main) is stale,
# so the raw-script URL, the uvx one-liner and the branch-pin note all pin
# THIS branch, never the default.
INSTALLABLE_BRANCH = "feat/create-file"


def run(*args: str, env_extra: dict | None = None, script: Path = SCRIPT) -> subprocess.CompletedProcess[str]:
    # stdin=DEVNULL pins the installer to a non-TTY stdin so the all-grammars
    # prompt can never block a test waiting on a human, no matter how pytest
    # itself was launched. The non-TTY path is itself under test: all-grammars
    # defaults to yes with a one-line notice, and a stale model cache is
    # removed without asking. `script` lets the managed-clone tests run a
    # standalone copy of the installer (no fastedit checkout beside it).
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [str(script), *args],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        # Generous budget: dry runs are sub-second idle, but the full suite
        # loads the machine and an oversubscribed uv invocation must surface
        # as a test failure, not as a hang.
        timeout=90,
        check=False,
        env=env,
    )


def spec_argv(stdout: str, prefix: str) -> str:
    """Find the "+ <prefix><spec>" dry-run line and return the shell-unquoted spec.

    Asserts the spec is copy-pasteable as ONE argv token (shlex.split
    backslash-unescapes it back to the original string) rather than
    checking for a literal quoting style, since printf %q's escaping
    convention can vary. The spec is the LAST token on the line --
    `--force --editable` always precedes it: every install shape is an
    editable install from a local clone.
    """
    for line in stdout.splitlines():
        if line.startswith(f"+ {prefix}"):
            tokens = shlex.split(line)
            return tokens[-1]
    raise AssertionError(f"no dry-run line starting with +{prefix!r} in:\n{stdout}")


def _npx_lines(stdout: str) -> list[list[str]]:
    """Every "+ npx ..." transcript line, shell-split into tokens.

    The skill add/remove lines are printed through the same quote_argv as the
    uv lines, so the same shlex round-trip applies: the skill SOURCE PATH must
    survive as ONE token (it holds no spaces or quotes, so quote_argv leaves
    it bare).
    """
    lines = []
    for line in stdout.splitlines():
        if line.startswith("+ npx "):
            lines.append(shlex.split(line))
    return lines


def _platform_extras() -> str:
    """The bare platform extras install-dev.sh auto-selects for THIS platform.

    Mirrors detect_backend_extra in the script, which installs EVERY extra this
    platform can actually install (owner directive: a default install must not
    be crippled). "All extras" cannot be literal -- mlx ships no Linux wheels
    and vllm no macOS wheels, so asking for both would fail to resolve on every
    platform. Hence: mlx+mcp on Apple Silicon, vllm+mcp on Linux with an NVIDIA
    driver, mcp alone elsewhere. mcp is unconditional: pure Python, and it is
    the MCP server other tools launch.

    Imports are local and complete on purpose: the Linux branch never executes
    on the machine this is developed on, so a missing module-level import would
    be a NameError that only ever fires on Linux and is invisible here.
    """
    import platform
    import shutil

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mlx,mcp"
    if platform.system() == "Linux" and shutil.which("nvidia-smi"):
        return "vllm,mcp"
    return "mcp"


def _with_all_grammars(extras: str) -> str:
    """Mirror the script's extras_with_all_grammars: append, dedupe, empty -> all-grammars."""
    if extras == "all-grammars" or extras.endswith(",all-grammars"):
        return extras
    return f"{extras},all-grammars" if extras else "all-grammars"


def _checkout_spec(extras: str) -> str:
    """The install spec for a run inside this checkout: `REPO_ROOT[extras]`.

    There is no fork-git-URL install shape any more: the ONE install shape is
    an editable install from a local clone, so from inside this repo every
    spec is this working tree's path, with or without bracketed extras.
    """
    if extras:
        return f"{REPO_ROOT}[{extras}]"
    return str(REPO_ROOT)


def _run_with_pty_stdin(feed: bytes, *args: str) -> subprocess.CompletedProcess[str]:
    """Run the installer with a pty as stdin, write `feed` as the answers.

    These are REAL full-installer runs (no --dry-run) with a hard 30s
    subprocess budget, so they opt out of every axis whose subject is not the
    prompt: --no-skill skips two live npx calls (~7s+ each) and --no-model
    skips `fastedit pull` plus `fastedit doctor`. Neither lever touches the
    grammar prompt under test; the skill axis is covered hermetically by
    TestAgentSkillInstall instead.

    Extra args go through to the script. On a TTY the script asks the
    all-grammars [Y/n] question, so `feed` must carry one line to answer
    (e.g. b"n\\n" = grammars no). The master side stays open until the child
    exits: closing it early makes the slave's read fail instead of delivering
    buffered input, which would silently flip the answer to the Enter default.
    """
    pty = pytest.importorskip("pty")
    master, slave = pty.openpty()
    try:
        proc = subprocess.Popen(
            [str(SCRIPT), "--no-skill", "--no-model", *args],
            stdin=slave,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except BaseException:
        os.close(master)
        os.close(slave)
        raise
    os.close(slave)
    # The pty line discipline buffers this input until the script's read
    # consumes it, so there is no write-before-read race to sleep away.
    os.write(master, feed)
    try:
        # Generous budget: the real install steps take ~10s idle, and the full
        # suite loads the machine — an oversubscribed run must surface as a
        # test failure, not as a hang.
        out, err = proc.communicate(timeout=90)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise
    finally:
        os.close(master)
    return subprocess.CompletedProcess(proc.args, proc.returncode, out, err)


def _stub_toolchain(
    tmp_path: Path, with_npx: bool = True, npx_add_fails: bool = False
) -> dict[str, str]:
    """A fake uv/pipx/pip3/fastedit/npx PATH so a REAL (non-dry-run) installer
    run can be exercised hermetically.

    Every mutating tool is a no-op stub and every reporting tool reports
    nothing installed, so a real run touches nothing outside the HOME the test
    hands it (where the model-cache dirs live). The fastedit stub prints the
    fork subcommand names on --help so the postflight verification passes. The
    npx stub answers `skills add` with success and `skills list -g` with an
    installed fastedit skill, so the agent-skill axis is hermetic too;
    with_npx=False drops it entirely to test the npx-missing path (PATH keeps
    only /usr/bin:/bin beyond the stubs, where no node lives on this machine),
    and npx_add_fails makes `skills add` exit non-zero with `skills list -g`
    reporting nothing, to test the failure-tolerant path. PATH is restricted
    to the stubs plus /usr/bin:/bin so the real uv (and the venv's real
    fastedit) on this machine can never be reached.

    A `git` stub is included so a standalone (managed-clone) run stays
    hermetic: the installer fetches/checks-out/resets the managed clone
    through git, and a real `git fetch origin <ref>` here would hit the
    network. The stub logs its argv to .git-args files under the stub dir
    (appended, newline-delimited, shell-quoted) so tests can assert which git
    steps ran, without the managed clone itself observing a fake git.
    """
    fake = tmp_path / "stub-bin"
    fake.mkdir()
    (fake / "uv").write_text(
        "#!/bin/bash\n"
        '[[ "$1" == "--version" ]] && { echo "uv 0.12.12 (stub)"; exit 0; }\n'
        "exit 0\n"
    )
    (fake / "pipx").write_text("#!/bin/bash\nexit 0\n")
    (fake / "pip3").write_text(
        "#!/bin/bash\n"
        '[[ "$1" == "show" ]] && exit 1\n'
        "exit 0\n"
    )
    (fake / "fastedit").write_text(
        "#!/bin/bash\n"
        'if [[ "$1" == "--help" ]]; then\n'
        '  echo "usage: fastedit [command]"\n'
        '  echo "commands:"\n'
        '  echo "  create"\n'
        '  echo "  duplicate"\n'
        '  echo "  split"\n'
        '  echo "  join"\n'
        "  exit 0\n"
        "fi\n"
        "exit 0\n"
    )
    if with_npx:
        # npx args reach the stub as `npx --yes skills <add|list> ...`, so the
        # subcommand is $3, not $1 -- the same --yes the installer passes.
        if npx_add_fails:
            add_body = "  echo 'error: skills add failed (stub)' >&2\n  exit 1\n"
            list_body = "  exit 0\n"
        else:
            add_body = "  exit 0\n"
            list_body = '  echo "  fastedit   (agent skill, global)"\n  exit 0\n'
        (fake / "npx").write_text(
            "#!/bin/bash\n"
            'if [[ "$2" == "skills" && "$3" == "add" ]]; then\n'
            + add_body +
            "fi\n"
            'if [[ "$2" == "skills" && "$3" == "list" ]]; then\n'
            + list_body +
            "fi\n"
            "exit 0\n"
        )
    # git stub: logs its argv (shell-quoted, one invocation per line) to a
    # .git-args file beside it so tests can assert the managed-clone git
    # steps, while every subcommand itself succeeds as a no-op. Logs under
    # the stub dir, never inside a clone, so the fake tree stays inspectable.
    # FASTEDIT_STUB_GIT_DIRTY=1 makes `status --porcelain` report an uncommitted
    # change, to drive the dirty-override-clone refusal.
    (fake / "git").write_text(
        "#!/bin/bash\n"
        'printf "%s\\n" "$(printf "%q " "$@")" >> "$(dirname "$0")/.git-args"\n'
        'if [[ "$FASTEDIT_STUB_GIT_DIRTY" == "1" ]]; then\n'
        '  for a in "$@"; do\n'
        '    if [[ "$a" == "--porcelain" ]]; then\n'
        '      echo " M dirty-file.txt"\n'
        "      exit 0\n"
        "    fi\n"
        "  done\n"
        "fi\n"
        "exit 0\n"
    )
    for stub in fake.iterdir():
        stub.chmod(0o755)
    return {"PATH": f"{fake}:/usr/bin:/bin"}


def _git_invocations(tmp_path: Path) -> list[list[str]]:
    """The git invocations a stubbed run made, shell-split per line.

    Reads the .git-args log the git stub appended to, so tests can assert the
    managed-clone steps (clone / fetch / checkout / reset --hard) without
    touching a real network. File-descriptor-style flags (e.g. the `2>/dev/null`
    a caller redirects) are never part of argv and so never appear here.
    """
    log = tmp_path / "stub-bin" / ".git-args"
    if not log.exists():
        return []
    return [shlex.split(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]


class TestUvVersionFloor:
    """The preflight warns on a uv too old for bracket extras -- and only then."""

    def _run_with_fake_uv(self, tmp_path: Path, version_line: str):
        """Run the installer with a stub uv that prints `version_line`."""
        fake = tmp_path / "bin"
        fake.mkdir(exist_ok=True)
        stub = fake / "uv"
        stub.write_text(
            "#!/bin/bash\n"
            '[[ "$1" == "--version" ]] && { echo "' + version_line + '"; exit 0; }\n'
            "exit 0\n"
        )
        stub.chmod(0o755)
        env = dict(os.environ, PATH=str(fake) + os.pathsep + os.environ["PATH"])
        return subprocess.run(
            ["bash", str(SCRIPT), "--dry-run"],
            capture_output=True, text=True, env=env, timeout=60, check=False,
            stdin=subprocess.DEVNULL,
        )

    def test_old_uv_warns_about_silently_dropped_extras(self, tmp_path: Path) -> None:
        """A sub-0.5 uv must warn: it drops bracket extras SILENTLY, so nothing else would tell you."""
        r = self._run_with_fake_uv(tmp_path, "uv 0.4.30 (x 2024-01-01)")
        assert "older than 0.5" in r.stdout + r.stderr

    def test_modern_uv_is_silent(self, tmp_path: Path) -> None:
        """A current uv must produce no version warning at all."""
        r = self._run_with_fake_uv(tmp_path, "uv 0.12.12 (Homebrew)")
        assert "older than 0.5" not in r.stdout + r.stderr

    def test_unparseable_version_does_not_cry_wolf(self, tmp_path: Path) -> None:
        """A shifted or absent version field must NOT warn.

        MEASURED regression guard. Before the MAJOR.MINOR shape check, a
        `uv --version` whose second field is not the version -- a wrapper or a
        localized build printing "uv version 0.12.12" -- made the comparison
        operate on the literal word, and bash arithmetic treats an unset name as
        0, so both tests passed and a MODERN uv was warned at. A warning that
        cries wolf is worse than none: it teaches the reader to scroll past the
        one line that would have mattered.
        """
        for line in ("uv version 0.12.12", "uv"):
            r = self._run_with_fake_uv(tmp_path, line)
            assert "older than 0.5" not in r.stdout + r.stderr, line


class TestInstallDevScriptExists:
    def test_script_exists_and_is_executable(self) -> None:
        """The installer ships in the repo and is directly runnable."""
        assert SCRIPT.is_file()
        mode = SCRIPT.stat().st_mode
        assert mode & stat.S_IXUSR, "install-dev.sh must be executable"


class TestInstallDevDryRun:
    def test_dry_run_exits_zero_and_shows_default_install(self) -> None:
        """--dry-run succeeds and prints the uninstall sweep plus a platform-matched editable spec."""
        result = run("--dry-run")
        assert result.returncode == 0, result.stderr
        assert "uv tool uninstall fastedits" in result.stdout
        # stdin is DEVNULL (non-TTY): the grammar prompt defaults to yes.
        assert spec_argv(result.stdout, "uv tool install ") == _checkout_spec(
            _with_all_grammars(_platform_extras())
        )

    def test_dry_run_prints_the_editable_from_this_checkout_line(self) -> None:
        """The in-repo run names its install source: THIS checkout, editable."""
        result = run("--dry-run")
        assert result.returncode == 0, result.stderr
        assert f"installing editable from this checkout: {REPO_ROOT}" in result.stdout

    def test_dry_run_sweeps_every_install_method(self) -> None:
        """Uninstall-first sweeps uv tool, pipx and pip so no leftover install collides."""
        result = run("--dry-run")
        assert result.returncode == 0, result.stderr
        assert "uv tool uninstall fastedits" in result.stdout
        assert "pipx uninstall fastedits" in result.stdout
        assert "uninstall -y fastedits" in result.stdout

    def test_dry_run_shows_model_pull_for_this_platform(self) -> None:
        """The forward install picks a model command for this platform in dry-run too."""
        result = run("--dry-run")
        assert result.returncode == 0, result.stderr
        assert "fastedit pull --model" in result.stdout

    def test_no_model_flag_skips_the_model_pull_line(self) -> None:
        """--no-model omits the pull command entirely."""
        result = run("--no-model", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert "fastedit pull --model" not in result.stdout

    def test_ref_note_in_repo(self) -> None:
        """--ref inside a checkout has no effect on the install, and the run says so."""
        result = run("--ref", "v1.2.3", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert (
            "note: --ref has no effect when running from inside a checkout "
            "(the working tree is the install source)"
        ) in result.stderr

    def test_extras_flag_adds_bracketed_extras_to_package_spec(self) -> None:
        """--extras mlx,mcp installs fastedit-clone[mlx,mcp,...] rather than the bare path."""
        result = run("--extras", "mlx,mcp", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert spec_argv(result.stdout, "uv tool install ") == _checkout_spec(
            _with_all_grammars("mlx,mcp")
        )

    def test_extras_all_grammars_is_not_duplicated(self) -> None:
        """--extras all-grammars plus a yes answer must not yield all-grammars twice."""
        result = run("--extras", "all-grammars", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert spec_argv(result.stdout, "uv tool install ") == _checkout_spec("all-grammars")

    def test_help_lists_the_new_flags(self) -> None:
        """-h documents --dev and --all-grammars."""
        result = run("-h")
        assert result.returncode == 0, result.stderr
        assert "--dev" in result.stdout
        assert "--all-grammars" in result.stdout

    def test_usage_names_the_managed_clone_and_the_dev_noop(self) -> None:
        """-h speaks the managed-clone vocabulary: --ref tracks the managed
        clone, --dev is spelled as a no-op compatibility flag, and the install
        shape is the editable-from-local-clone one."""
        result = run("-h")
        assert result.returncode == 0, result.stderr
        assert "managed clone" in result.stdout
        assert "Accepted for compatibility, no longer needed" in result.stdout
        assert "editable install from a local clone" in result.stdout


class TestInRepoSourceResolution:
    """Run inside a checkout: the working tree IS the install source, and the
    managed-clone machinery (clone/fetch/reset, --ref) stays out of the way."""

    def test_non_tty_dry_run_installs_this_checkout(self) -> None:
        """Non-TTY default: no menu, the spec is THIS tree, editable."""
        result = run("--dry-run")
        assert result.returncode == 0, result.stderr
        assert f"installing editable from this checkout: {REPO_ROOT}" in result.stdout
        # No terminal on stdin and no menu in the script at all: no prompt
        # fragment may appear on either stream.
        assert "Install from:" not in result.stdout + result.stderr
        assert spec_argv(result.stdout, "uv tool install ") == _checkout_spec(
            _with_all_grammars(_platform_extras())
        )

    def test_explicit_ref_has_no_effect_inside_a_checkout(self) -> None:
        """--ref names the branch a MANAGED clone tracks; a checkout run says
        so, never touches git, and keeps the working tree as the spec."""
        result = run("--ref", "v1.2.3", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert (
            "note: --ref has no effect when running from inside a checkout "
            "(the working tree is the install source)"
        ) in result.stderr
        # The ref is swallowed, not installed: no git+ URL, no v1.2.3 pin, no
        # git invocation printed.
        assert "git+" not in result.stdout
        assert "@v1.2.3" not in result.stdout
        assert "git " not in result.stdout
        assert spec_argv(result.stdout, "uv tool install ") == _checkout_spec(
            _with_all_grammars(_platform_extras())
        )

    def test_dev_is_a_no_op_note(self) -> None:
        """--dev is accepted for compatibility and says so; the install is the
        same editable-from-this-checkout shape."""
        result = run("--dev", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert (
            "note: --dev is a no-op: every install is now an editable install "
            f"from a local clone (this checkout, or the managed clone at {Path.home()}/.fastedit/src)"
        ) in result.stderr
        assert f"installing editable from this checkout: {REPO_ROOT}" in result.stdout
        assert spec_argv(result.stdout, "uv tool install ") == _checkout_spec(
            _with_all_grammars(_platform_extras())
        )


class TestManagedCloneStandalone:
    """Standalone runs (no pyproject.toml beside the script): the installer
    maintains its own editable clone at ~/.fastedit/src — cloned on first run,
    fetched/check-out/hard-reset on every run after, deletable only at the
    DEFAULT path, and never hard-reset when FASTEDIT_CLONE_DIR points at a
    tree the user manages. Every real git step is stubbed; HOME is sandboxed;
    the script is COPIED to a neutral dir so no checkout is detected.
    """

    def _copy_script(self, tmp_path: Path) -> Path:
        """A runnable copy of the installer in a neutral dir (no pyproject.toml
        beside it), so the script resolves HAVE_LOCAL_TREE=0. copyfile would
        drop the exec bit, so 0o755 is re-applied."""
        script_copy = tmp_path / "standalone" / "install-dev.sh"
        script_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SCRIPT, script_copy)
        script_copy.chmod(0o755)
        return script_copy

    def _standalone_env(self, tmp_path: Path, home_name: str) -> tuple[dict[str, str], Path]:
        """A sandboxed standalone run: script copy outside any checkout, stub
        toolchain (incl. git) on PATH, private HOME."""
        home = tmp_path / home_name
        home.mkdir()
        env = {
            **_stub_toolchain(tmp_path),
            "HOME": str(home),
        }
        return env, self._copy_script(tmp_path)

    def test_dry_run_prints_the_clone_transcript_without_executing_git(
        self, tmp_path: Path
    ) -> None:
        """First standalone run, dry: the `+ git clone --branch ...` transcript
        line is printed, and no git (or clone) actually runs."""
        env, script_copy = self._standalone_env(tmp_path, "mc-dry-home")
        result = run("--dry-run", "--no-model", env_extra=env, script=script_copy)
        assert result.returncode == 0, result.stderr
        expected = f"{tmp_path}/mc-dry-home/.fastedit/src"
        assert f"no fastedit clone here — cloning into {expected} (branch feat/create-file)..." in result.stdout
        assert f"+ git clone --branch feat/create-file {FORK_URL} {expected}" in result.stdout
        # Dry-run purity: no git ran, nothing was cloned.
        assert _git_invocations(tmp_path) == []
        assert not Path(expected).exists()

    def test_real_run_clones_into_the_default_managed_dir(self, tmp_path: Path) -> None:
        """First standalone run, real: exactly one `git clone --branch` with
        the built-in default ref, into ~/.fastedit/src under the test's HOME."""
        env, script_copy = self._standalone_env(tmp_path, "mc-clone-home")
        result = run("--no-model", env_extra=env, script=script_copy)
        assert result.returncode == 0, result.stderr
        home = tmp_path / "mc-clone-home"
        expected = f"{home}/.fastedit/src"
        assert f"no fastedit clone here — cloning into {expected} (branch feat/create-file)..." in result.stdout
        invocations = _git_invocations(tmp_path)
        assert invocations == [["clone", "--branch", "feat/create-file", FORK_URL, expected]]

    def test_real_run_updates_an_existing_managed_clone(self, tmp_path: Path) -> None:
        """An existing managed clone is fetched, checked out and hard-reset to
        origin/<tracked branch> — and never re-cloned."""
        home = tmp_path / "mc-update-home"
        home.mkdir()
        clone = home / ".fastedit" / "src"
        clone.mkdir(parents=True)
        # The marker the script uses to recognize a fastedit clone.
        (clone / "pyproject.toml").write_text("[project]\nname = 'fastedits'\n", encoding="utf-8")
        script_copy = self._copy_script(tmp_path)
        env = {**_stub_toolchain(tmp_path), "HOME": str(home)}
        result = run("--no-model", env_extra=env, script=script_copy)
        assert result.returncode == 0, result.stderr
        assert "managed clone updated to origin/feat/create-file (" in result.stdout
        assert str(clone) in result.stdout
        # The update steps, in order: a dirtiness probe (status --porcelain),
        # fetch, checkout, reset --hard, then the sha lookup for the "managed
        # clone updated ... (<sha>)" line. `git -C <dir> ...` puts -C and <dir>
        # BEFORE the subcommand, so the subcommand is argv[2]; the reset target
        # is origin/<ref>, never a local ref.
        invocations = _git_invocations(tmp_path)
        assert [inv[2] for inv in invocations] == ["status", "fetch", "checkout", "reset", "rev-parse"]
        assert invocations[1] == ["-C", str(clone), "fetch", "origin", "feat/create-file"]
        assert invocations[2] == ["-C", str(clone), "checkout", "feat/create-file"]
        assert invocations[3][:4] == ["-C", str(clone), "reset", "--hard"]
        assert invocations[3][4] == "origin/feat/create-file"
        assert invocations[4][2] == "rev-parse"

    def test_dirty_override_clone_is_refused_with_exit_1(self, tmp_path: Path) -> None:
        """FASTEDIT_CLONE_DIR pointing at a user-managed tree with uncommitted
        changes is refused (exit 1) — never hard-reset, nothing touched."""
        home = tmp_path / "mc-dirty-home"
        home.mkdir()
        clone = tmp_path / "user-tree"
        clone.mkdir()
        (clone / "pyproject.toml").write_text("[project]\nname = 'fastedits'\n", encoding="utf-8")
        script_copy = self._copy_script(tmp_path)
        env = {
            **_stub_toolchain(tmp_path),
            "HOME": str(home),
            "FASTEDIT_CLONE_DIR": str(clone),
            "FASTEDIT_STUB_GIT_DIRTY": "1",
        }
        result = run("--no-model", env_extra=env, script=script_copy)
        assert result.returncode == 1
        assert (
            f"error: FASTEDIT_CLONE_DIR points at {clone}, which has uncommitted changes — "
            "refusing to reset a tree you manage."
        ) in result.stderr
        # Nothing was touched: only the dirtiness probe ran, no fetch/reset,
        # and the tree keeps its file.
        assert [inv[2] for inv in _git_invocations(tmp_path)] == ["status"]
        assert (clone / "pyproject.toml").is_file()

    def test_revert_removes_the_default_managed_clone(self, tmp_path: Path) -> None:
        """--revert deletes the installer-OWNED default clone: `+ rm -rf` and
        'removed:' — but keeps an override clone the user pointed at."""
        home = tmp_path / "mc-revert-home"
        home.mkdir()
        clone = home / ".fastedit" / "src"
        clone.mkdir(parents=True)
        (clone / "pyproject.toml").write_text("[project]\nname = 'fastedits'\n", encoding="utf-8")
        script_copy = self._copy_script(tmp_path)
        env = {**_stub_toolchain(tmp_path), "HOME": str(home)}
        result = run("--revert", "--no-model", env_extra=env, script=script_copy)
        assert result.returncode == 0, result.stderr
        assert f"+ rm -rf {clone}" in result.stdout
        assert f"removed: installer-managed clone at {clone}" in result.stdout
        assert not clone.exists()

    def test_revert_leaves_an_override_clone_in_place(self, tmp_path: Path) -> None:
        """FASTEDIT_CLONE_DIR outside the default path is the user's: --revert
        reports it and leaves it in place."""
        home = tmp_path / "mc-revert-override-home"
        home.mkdir()
        clone = tmp_path / "user-override-tree"
        clone.mkdir()
        (clone / "pyproject.toml").write_text("[project]\nname = 'fastedits'\n", encoding="utf-8")
        script_copy = self._copy_script(tmp_path)
        env = {
            **_stub_toolchain(tmp_path),
            "HOME": str(home),
            "FASTEDIT_CLONE_DIR": str(clone),
        }
        result = run("--revert", "--no-model", env_extra=env, script=script_copy)
        assert result.returncode == 0, result.stderr
        assert f"note: FASTEDIT_CLONE_DIR points outside ~/.fastedit — leaving {clone} in place" in result.stdout
        assert (clone / "pyproject.toml").is_file()

    def test_check_reports_the_managed_clone_state(self, tmp_path: Path) -> None:
        """Standalone --check reports the managed clone path and whether it is
        absent or present — the read-only answer to 'what does this machine
        have at ~/.fastedit/src?'."""
        env, script_copy = self._standalone_env(tmp_path, "mc-check-home")
        clone = tmp_path / "mc-check-home" / ".fastedit" / "src"
        result = run("--check", env_extra=env, script=script_copy)
        assert result.returncode == 0, result.stderr
        assert f"managed clone: {clone} (absent)" in result.stdout
        # After a clone exists, the same report flips the state word.
        clone.mkdir(parents=True)
        (clone / "pyproject.toml").write_text("[project]\nname = 'fastedits'\n", encoding="utf-8")
        result = run("--check", env_extra=env, script=script_copy)
        assert result.returncode == 0, result.stderr
        assert f"managed clone: {clone} (present)" in result.stdout


class TestAllGrammarsFlag:
    def test_all_grammars_no_omits_the_extra(self) -> None:
        """--all-grammars no installs exactly the platform extras -- no grammar pack, no notice."""
        result = run("--all-grammars", "no", "--dry-run")
        assert result.returncode == 0, result.stderr
        # The choice must not leak into the spec or any notice -- the exact
        # spec equality above already proves the extra is absent.
        assert "non-interactive" not in result.stdout + result.stderr
        assert spec_argv(result.stdout, "uv tool install ") == _checkout_spec(_platform_extras())

    def test_all_grammars_yes_includes_the_extra(self) -> None:
        """--all-grammars yes appends the pack to the platform extras without asking."""
        result = run("--all-grammars", "yes", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert spec_argv(result.stdout, "uv tool install ") == _checkout_spec(
            _with_all_grammars(_platform_extras())
        )
        assert "non-interactive" not in result.stdout + result.stderr

    def test_non_tty_default_enables_all_grammars_with_a_notice(self) -> None:
        """Piped stdin: no prompt is asked; all grammars default ON, with a notice saying so."""
        result = run("--dry-run")
        assert result.returncode == 0, result.stderr
        assert "non-interactive: all grammars enabled by default" in result.stderr
        assert "pass --all-grammars no to skip" in result.stderr
        assert spec_argv(result.stdout, "uv tool install ") == _checkout_spec(
            _with_all_grammars(_platform_extras())
        )

    def test_all_grammars_no_skips_the_model_pull_with_empty_extras(self) -> None:
        """--extras "" --all-grammars no: bare spec, model pull skipped because no backend."""
        result = run("--extras", "", "--all-grammars", "no", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert spec_argv(result.stdout, "uv tool install ") == _checkout_spec("")
        assert "fastedit pull --model" not in result.stdout

    def test_all_grammars_rejects_an_invalid_value(self) -> None:
        """--all-grammars only accepts yes|no (y/n aliases included)."""
        result = run("--all-grammars", "maybe")
        assert result.returncode != 0
        assert "yes" in result.stderr and "no" in result.stderr

    def test_all_grammars_rejects_a_missing_value(self) -> None:
        """A valueless --all-grammars is an argument error, not a silently ignored flag."""
        result = run("--all-grammars")
        assert result.returncode != 0


class TestAllGrammarsPrompt:
    """The interactive [Y/n] prompt — needs a real TTY, so driven through a pty.

    There is no source menu, so the grammars question is the ONLY thing a TTY
    is ever asked: each feed answers exactly that one question.
    """

    def test_enter_defaults_to_yes(self) -> None:
        """A bare Enter at the grammars prompt means yes: the grammar pack joins the install."""
        r = _run_with_pty_stdin(b"\n")  # grammars -> Enter (yes)
        assert r.returncode == 0, r.stderr
        assert "Install all grammars?" in r.stderr
        assert spec_argv(r.stdout, "uv tool install ") == _checkout_spec(
            _with_all_grammars(_platform_extras())
        )

    def test_no_answers_no(self) -> None:
        """Answering n skips the pack."""
        r = _run_with_pty_stdin(b"n\n")  # grammars -> n
        assert r.returncode == 0, r.stderr
        assert "Install all grammars?" in r.stderr
        assert spec_argv(r.stdout, "uv tool install ") == _checkout_spec(_platform_extras())

    def test_explicit_flag_suppresses_the_prompt(self) -> None:
        """--all-grammars yes must never ask, even on a TTY."""
        # The grammars question is pre-answered by the flag, so the script
        # reads nothing from stdin at all.
        r = _run_with_pty_stdin(b"", "--all-grammars", "yes")
        assert r.returncode == 0, r.stderr
        assert "Install all grammars?" not in r.stdout + r.stderr
        assert spec_argv(r.stdout, "uv tool install ") == _checkout_spec(
            _with_all_grammars(_platform_extras())
        )


class TestModelCachePreflight:
    """Preflight cache detection: VALID caches are kept, STALE (partial) ones removed."""

    def _home_with_cache(self, tmp_path: Path, home_name: str, files: dict[str, str] | None) -> Path:
        """A HOME whose model cache holds one model dir with the given files (None = dir only)."""
        home = tmp_path / home_name
        model_dir = home / ".cache" / "fastedit" / "models" / "some-model"
        model_dir.mkdir(parents=True)
        for fname, content in (files or {}).items():
            (model_dir / fname).write_text(content, encoding="utf-8")
        return home

    def test_dry_run_reports_preflight_headers_and_an_absent_cache(self, tmp_path: Path) -> None:
        """The preflight report prints on a dry run too, and says so when nothing is cached."""
        home = tmp_path / "empty-home"
        home.mkdir()
        result = run("--dry-run", env_extra={"HOME": str(home)})
        assert result.returncode == 0, result.stderr
        assert "preflight: currently installed fastedits (uv tool / pipx / pip):" in result.stdout
        assert f"preflight: model caches under {home}/.cache/fastedit/models:" in result.stdout
        assert "(no cache directory — nothing cached yet)" in result.stdout

    def test_dry_run_labels_stale_and_valid_and_never_removes(self, tmp_path: Path) -> None:
        """A partial download is STALE (would remove), a safetensors-bearing one is VALID (kept)."""
        home = tmp_path / "cache-home"
        models = home / ".cache" / "fastedit" / "models"
        stale = models / "stale-model"
        stale.mkdir(parents=True)
        (stale / "config.json").write_text("{}", encoding="utf-8")
        valid = models / "valid-model"
        valid.mkdir(parents=True)
        (valid / "model.safetensors").write_text("weights", encoding="utf-8")

        result = run("--dry-run", env_extra={"HOME": str(home)})
        assert result.returncode == 0, result.stderr
        out = result.stdout
        assert "stale-model" in out and "STALE (partial, no *.safetensors)" in out
        assert "valid-model" in out and "VALID" in out
        assert "would remove stale model cache stale-model" in out
        assert "would remove stale model cache valid-model" not in out
        assert "model cache valid-model: kept — shared cache" in out
        # A dry run removes nothing.
        assert stale.exists() and valid.exists()

    def test_default_run_removes_a_stale_cache_without_asking(self, tmp_path: Path) -> None:
        """Non-TTY default: a partial cache is removed WITHOUT a prompt, so the next pull reinstalls cleanly.

        A REAL run (no --dry-run) against the stubbed toolchain: the install
        steps are no-ops, and the explicit --ref keeps the run off the network
        (it skips branch autodetect and the fork-remote verification).
        """
        home = self._home_with_cache(tmp_path, "stale-home", {"config.json": "{}"})
        env = {**_stub_toolchain(tmp_path), "HOME": str(home)}
        result = run("--ref", "feat/create-file", "--no-model", env_extra=env)
        assert result.returncode == 0, result.stderr
        assert "note: non-interactive: removing stale model cache 'some-model'" in result.stderr
        assert "removed stale model cache 'some-model'" in result.stdout
        assert not (home / ".cache" / "fastedit" / "models" / "some-model").exists()

    def test_default_run_keeps_a_valid_cache(self, tmp_path: Path) -> None:
        """Non-TTY default: a cache holding *.safetensors is kept — it is shared with upstream."""
        home = self._home_with_cache(tmp_path, "valid-home", {"model.safetensors": "weights"})
        env = {**_stub_toolchain(tmp_path), "HOME": str(home)}
        result = run("--ref", "feat/create-file", "--no-model", env_extra=env)
        assert result.returncode == 0, result.stderr
        assert "model cache some-model: kept — shared cache" in result.stdout
        assert "removed stale model cache" not in result.stdout
        kept = home / ".cache" / "fastedit" / "models" / "some-model" / "model.safetensors"
        assert kept.exists()

    def test_revert_keeps_every_cache(self, tmp_path: Path) -> None:
        """--revert never touches model weights — not even stale ones."""
        home = self._home_with_cache(tmp_path, "revert-home", {"config.json": "{}"})
        result = run("--revert", "--dry-run", env_extra={"HOME": str(home)})
        assert result.returncode == 0, result.stderr
        assert (
            "model caches: kept — --revert leaves downloaded weights in place (shared with upstream)"
        ) in result.stdout
        assert "would remove" not in result.stdout
        assert (home / ".cache" / "fastedit" / "models" / "some-model").exists()


class TestDevInstall:
    """--dev is a no-op compatibility flag: the install is the same editable
    one, and the run says so rather than silently swallowing the flag."""

    def test_dev_dry_run_spec_is_the_local_repo_path(self) -> None:
        """The install is editable from the working tree: path[extras], no git URL."""
        result = run("--dev", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert f"installing editable from this checkout: {REPO_ROOT}" in result.stdout
        assert "git+" not in result.stdout
        assert spec_argv(
            result.stdout, "uv tool install --force --editable "
        ) == f"{REPO_ROOT}[{_with_all_grammars(_platform_extras())}]"

    def test_dev_dry_run_uses_force_and_editable(self) -> None:
        """The install line carries --force --editable so re-runs refresh in place."""
        result = run("--dev", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert "+ uv tool install --force --editable " in result.stdout

    def test_dev_dry_run_with_all_grammars_no(self) -> None:
        """--dev --all-grammars no keeps the platform extras only."""
        result = run("--dev", "--all-grammars", "no", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert spec_argv(
            result.stdout, "uv tool install --force --editable "
        ) == f"{REPO_ROOT}[{_platform_extras()}]"

    def test_dev_with_explicit_extras(self) -> None:
        """--dev --extras honours the explicit list exactly."""
        result = run("--dev", "--extras", "mcp", "--all-grammars", "no", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert spec_argv(
            result.stdout, "uv tool install --force --editable "
        ) == f"{REPO_ROOT}[mcp]"

    def test_dev_dry_run_still_sweeps_and_shows_model_pull(self) -> None:
        """--dev keeps the sweep and the model logic — only the note is new."""
        result = run("--dev", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert "uv tool uninstall fastedits" in result.stdout
        assert "pipx uninstall fastedits" in result.stdout
        assert "fastedit pull --model" in result.stdout

    def test_dev_ref_combination_warns_and_uses_the_local_path(self) -> None:
        """--ref has no effect inside a checkout, and the script says so."""
        result = run("--dev", "--ref", "v1.2.3", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert "--ref has no effect" in result.stderr
        assert "git+" not in result.stdout
        assert "@v1.2.3" not in result.stdout

    def test_dev_revert_combination_is_rejected(self) -> None:
        """--dev --revert contradicts itself: editable working tree vs upstream PyPI."""
        result = run("--dev", "--revert")
        assert result.returncode != 0


class TestInstallDevRevert:
    def test_revert_dry_run_shows_uninstall_then_pypi_install(self) -> None:
        """--revert undoes the swap: sweep-uninstall the fork, reinstall plain fastedits."""
        result = run("--revert", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert "uv tool uninstall fastedits" in result.stdout
        assert spec_argv(result.stdout, "uv tool install ") == "fastedits"

    def test_revert_dry_run_never_mentions_a_git_url(self) -> None:
        """--revert must install from PyPI, never reference the fork's git URL."""
        result = run("--revert", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert "git+" not in result.stdout
        assert "git+" not in result.stderr

    def test_revert_dry_run_never_pulls_a_model(self) -> None:
        """--revert never touches model weights — they're shared with the fork."""
        result = run("--revert", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert "fastedit pull" not in result.stdout

    def test_revert_never_asks_and_never_adds_grammars(self) -> None:
        """--revert restores upstream: no prompt, no grammar pack."""
        result = run("--revert", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert "Install all grammars?" not in result.stdout + result.stderr
        assert "all-grammars" not in result.stdout + result.stderr
        assert "non-interactive" not in result.stdout + result.stderr


class TestAgentSkillInstall:
    """The agent-skill axis: installed with the Vercel skills CLI from the
    SAME clone the package comes from (this checkout, or the managed clone on
    a standalone run) — never from a GitHub tree URL or the repo shorthand.
    Both GitHub forms resolve the fork's DEFAULT branch (main), which still
    carries the legacy claude-skill content, and the tree-URL form fails
    outright in non-TTY mode; the local tree is the locally-verified install
    shape. Failure-tolerant exactly like the optional backend install -- a
    skill failure must never leave this machine without fastedit, and the
    postflight reports what actually happened.
    """

    SKILL_SOURCE_DIR = f"{REPO_ROOT}/skills/fastedit"

    def _add_tokens(self, result: subprocess.CompletedProcess[str]) -> list[str]:
        add = [t for t in _npx_lines(result.stdout) if "add" in t]
        assert len(add) == 1, f"expected exactly one skills-add line in:\n{result.stdout}"
        return add[0]

    def test_dry_run_prints_the_add_command_from_the_local_tree(self) -> None:
        """Default dry run: the skill add points at THIS clone's skills/fastedit,
        global and non-interactive with the claude-code target, no --skill filter."""
        result = run("--dry-run")
        assert result.returncode == 0, result.stderr
        tokens = self._add_tokens(result)
        assert tokens[1:6] == ["npx", "--yes", "skills", "add", self.SKILL_SOURCE_DIR]
        assert tokens[6:] == ["-g", "-a", "claude-code", "-y"]
        # The path IS the skill: no --skill filter and no GitHub URL anywhere.
        assert "--skill" not in tokens
        assert not any("github.com" in token for token in tokens)
        # The source is the repo skill the packaged-copy drift guard keeps in sync.
        assert (REPO_ROOT / "skills" / "fastedit" / "SKILL.md").is_file()

    def test_ref_pins_the_package_but_never_the_skill(self) -> None:
        """--ref (from inside a checkout: swallowed with a note) must never
        reach the skill command: the skill always comes from the local tree."""
        result = run("--ref", "v1.2.3", "--dry-run")
        assert result.returncode == 0, result.stderr
        tokens = self._add_tokens(result)
        assert self.SKILL_SOURCE_DIR in tokens
        assert not any("v1.2.3" in token for token in tokens)
        assert spec_argv(result.stdout, "uv tool install ") == _checkout_spec(
            _with_all_grammars(_platform_extras())
        )

    def test_no_skill_flag_omits_every_skill_command(self) -> None:
        """--no-skill skips the whole axis: no npx invocation is printed at all."""
        result = run("--no-skill", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert "npx" not in result.stdout

    def test_forward_run_never_prints_the_remove_command(self) -> None:
        """The remove command belongs to --revert only."""
        result = run("--dry-run")
        assert result.returncode == 0, result.stderr
        assert "skills remove" not in result.stdout + result.stderr

    def test_revert_prints_the_skill_remove_command(self) -> None:
        """--revert additionally removes the globally installed skill, best-effort shape."""
        result = run("--revert", "--dry-run")
        assert result.returncode == 0, result.stderr
        remove = [t for t in _npx_lines(result.stdout) if "remove" in t]
        assert len(remove) == 1, f"expected exactly one skills-remove line in:\n{result.stdout}"
        assert remove[0][1:] == ["npx", "--yes", "skills", "remove", "fastedit", "-g", "-y"]
        assert "skills add" not in result.stdout

    def test_revert_with_no_skill_skips_the_remove(self) -> None:
        """--no-skill skips the whole axis in revert mode too."""
        result = run("--revert", "--no-skill", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert "npx" not in result.stdout

    def test_help_documents_the_local_tree_source(self) -> None:
        """-h documents --no-skill, the automatic skill install, and the LOCAL
        tree source — the GitHub tree-URL form is gone."""
        result = run("-h")
        assert result.returncode == 0, result.stderr
        assert "--no-skill" in result.stdout
        assert "agent skill" in result.stdout
        assert "skills/fastedit" in result.stdout
        assert "/tree/" not in result.stdout

    def test_real_run_installs_and_reports_the_skill(self, tmp_path: Path) -> None:
        """Real run with a stubbed toolchain: the add runs, the postflight reports installed."""
        home = tmp_path / "skill-home"
        home.mkdir()
        env = {**_stub_toolchain(tmp_path), "HOME": str(home)}
        result = run("--ref", "feat/create-file", "--no-model", env_extra=env)
        assert result.returncode == 0, result.stderr
        assert "installed: agent skill" in result.stdout
        assert "agent skill: installed (claude-code, global)" in result.stdout

    def test_postflight_reports_the_targeted_agent_id(self) -> None:
        """The installer targets exactly ONE agent, spelled in the agent-id
        vocabulary `fastedit init --skill-agent` takes (claude-code), and its
        self-report derives from that same SKILL_AGENT variable — not from a
        second, driftable 'Claude Code' display brand: the `-a` value, both
        success lines, and the usage prose cannot disagree with what ran."""
        script = SCRIPT.read_text(encoding="utf-8")
        assert 'SKILL_AGENT="claude-code"' in script
        # The invocation and every report line derive from the variable.
        assert '-a "$SKILL_AGENT"' in script
        assert "agent skill: installed (${SKILL_AGENT}, global)" in script
        assert "installed: agent skill 'fastedit' (${SKILL_AGENT}, global)" in script
        assert "Claude Code" not in script, (
            "the installer reports the targeted agent id (claude-code, the "
            "--skill-agent vocabulary), not a hardcoded display brand"
        )

    def test_npx_missing_skips_the_skill_and_says_so(self, tmp_path: Path) -> None:
        """No npx on PATH: the install prints the manual-install note (the LOCAL
        tree command), the postflight says skipped -- and the run still exits 0
        with the package installed."""
        home = tmp_path / "nonpx-home"
        home.mkdir()
        env = {**_stub_toolchain(tmp_path, with_npx=False), "HOME": str(home)}
        result = run("--ref", "feat/create-file", "--no-model", env_extra=env)
        assert result.returncode == 0, result.stderr
        assert (
            "npx not found — skipping the agent skill; install manually: "
            f"npx --yes skills add {self.SKILL_SOURCE_DIR} -g -a claude-code -y"
        ) in result.stdout
        assert "agent skill: skipped (npx not found)" in result.stdout
        assert "verified: fastedit at" in result.stdout

    def test_failed_skill_install_warns_and_the_install_survives(self, tmp_path: Path) -> None:
        """A failed skill install must never fail the installer, and the postflight
        must report NOT FOUND rather than pretend the skill is there."""
        home = tmp_path / "failskill-home"
        home.mkdir()
        env = {**_stub_toolchain(tmp_path, npx_add_fails=True), "HOME": str(home)}
        result = run("--ref", "feat/create-file", "--no-model", env_extra=env)
        assert result.returncode == 0, result.stderr
        assert "warning: the agent skill could not be installed" in result.stderr
        assert "agent skill: NOT FOUND (see warnings above)" in result.stdout
        # The manual-install warning names the local tree source.
        assert self.SKILL_SOURCE_DIR in result.stderr
        # The package itself still landed: the fork postflight passed.
        assert "verified: fastedit at" in result.stdout


class TestInstallDevArgumentValidation:
    def test_unknown_flag_exits_non_zero(self) -> None:
        """An unrecognized flag is rejected instead of silently ignored."""
        result = run("--this-flag-does-not-exist")
        assert result.returncode != 0


class TestCheckMode:
    """--check: a read-only state report — installs nothing, removes nothing,
    needs no npx, and works when the network is unreachable."""

    def test_check_exits_zero_and_is_explicitly_non_mutating(self) -> None:
        result = run("--check")
        assert result.returncode == 0, result.stderr
        assert "read-only" in result.stdout + result.stderr

    def test_check_does_not_install_or_sweep(self) -> None:
        """The mutating paths stay dark: no sweep, no install, no pull, no skill add."""
        result = run("--check")
        assert result.returncode == 0, result.stderr
        assert "uv tool uninstall" not in result.stdout
        assert "uv tool install" not in result.stdout
        assert "pipx uninstall" not in result.stdout
        assert "fastedit pull" not in result.stdout
        assert "skills add" not in result.stdout
        assert "skills remove" not in result.stdout

    def test_check_needs_no_npx(self, tmp_path: Path) -> None:
        """--check is a no-npx surface: the stub toolchain minus npx must not warn.

        Asserted on the report's actual npx surfaces (warnings, the skills CLI
        invocation, the agent-skill axis) rather than the bare substring: the
        binary's PATH is echoed in the report and test directory names can
        legitimately contain "npx".
        """
        home = tmp_path / "check-home"
        home.mkdir()
        env = {**_stub_toolchain(tmp_path, with_npx=False), "HOME": str(home)}
        result = run("--check", env_extra=env)
        assert result.returncode == 0, result.stderr
        out = result.stdout + result.stderr
        assert "npx not found" not in out
        assert "npx --yes" not in out
        assert "agent skill" not in out
        assert "skills" not in out

    def test_check_survives_a_remote_that_answers_nothing(self, tmp_path: Path) -> None:
        """A fork remote that cannot be probed never fails --check.

        With the stub PATH the script's `git ls-remote --heads` resolves to the
        git stub, which exits 0 but answers nothing — the same shape an
        unreachable-but-not-failing remote produces. The report must say the
        branch state is unresolved and STILL exit 0: --check is read-only and
        the network is not its problem.
        """
        home = tmp_path / "check-home"
        home.mkdir()
        env = {**_stub_toolchain(tmp_path, with_npx=False), "HOME": str(home)}
        result = run("--check", "--ref", "feat/create-file", env_extra=env)
        assert result.returncode == 0, result.stderr
        assert f"branch '{INSTALLABLE_BRANCH}' exists on the fork remote" not in result.stdout
        assert "(MISSING)" in result.stdout

    def test_check_reports_a_fork_install_from_the_stub_binary(self, tmp_path: Path) -> None:
        """A fastedit on PATH that lists the fork verbs reports a fork install."""
        home = tmp_path / "check-home"
        home.mkdir()
        env = {**_stub_toolchain(tmp_path, with_npx=False), "HOME": str(home)}
        result = run("--check", "--ref", "feat/create-file", env_extra=env)
        assert result.returncode == 0, result.stderr
        assert "fork install: YES" in result.stdout

    def test_check_on_this_machine_reports_upstream_or_fork(self) -> None:
        """The real machine: one of the two install states, and never a traceback."""
        result = run("--check")
        assert result.returncode == 0, result.stderr
        out = result.stdout
        assert "fork install: YES" in out or "fork install: NO" in out

    def test_check_reports_absent_binary_and_no_model_cache(self, tmp_path: Path) -> None:
        """An empty HOME with nothing installed: the report says NO, not an error.

        PATH keeps only the stub uv (no fastedit stub) plus the system dirs,
        so the binary axis exercises its "not found" branch.
        """
        home = tmp_path / "empty-home"
        home.mkdir()
        fake = tmp_path / "bin"
        fake.mkdir(exist_ok=True)
        (fake / "uv").write_text(
            "#!/bin/bash\n"
            '[[ "$1" == "--version" ]] && { echo "uv 0.12.12 (stub)"; exit 0; }\n'
            "exit 0\n"
        )
        (fake / "uv").chmod(0o755)
        env = {"PATH": f"{fake}:/usr/bin:/bin", "HOME": str(home)}
        result = run("--check", "--ref", "feat/create-file", env_extra=env)
        assert result.returncode == 0, result.stderr
        out = result.stdout
        assert "fastedit on PATH: not found" in out
        assert "fork install: NO" in out
        assert "(no cache directory — nothing cached yet)" in out

    def test_check_reports_valid_and_stale_model_caches(self, tmp_path: Path) -> None:
        """One VALID and one STALE cache dir are labelled without being touched."""
        home = tmp_path / "cache-home"
        models = home / ".cache" / "fastedit" / "models"
        valid = models / "valid-model"
        valid.mkdir(parents=True)
        (valid / "model.safetensors").write_text("weights", encoding="utf-8")
        stale = models / "stale-model"
        stale.mkdir(parents=True)
        (stale / "config.json").write_text("{}", encoding="utf-8")

        env = {**_stub_toolchain(tmp_path, with_npx=False), "HOME": str(home)}
        result = run("--check", "--ref", "feat/create-file", env_extra=env)
        assert result.returncode == 0, result.stderr
        out = result.stdout
        assert "valid-model" in out and "VALID" in out
        assert "stale-model" in out and "STALE (partial, no *.safetensors)" in out
        assert "would remove" not in out
        assert valid.exists() and stale.exists()

    def test_check_reports_local_clone_state_and_pin_line(self) -> None:
        """A run from inside this clone names the local tree and the installable branch."""
        result = run("--check")
        assert result.returncode == 0, result.stderr
        out = result.stdout
        assert f"local clone: {REPO_ROOT}" in out
        # The stale-main pin note: names the stale default branch AND the
        # installable one, in the same wording the success footer uses.
        assert "stale default branch (main)" in out
        assert f"the installable branch is {INSTALLABLE_BRANCH}" in out

    def test_check_reports_grammars_via_the_tool_python(self, tmp_path: Path) -> None:
        """The grammar axis probes the tool env's own python, exactly like the postflight."""
        home = tmp_path / "check-home"
        home.mkdir()
        env = {**_stub_toolchain(tmp_path, with_npx=False), "HOME": str(home)}
        result = run("--check", "--ref", "feat/create-file", env_extra=env)
        assert result.returncode == 0, result.stderr
        out = result.stdout
        # The report carries a grammars section whose outcome is one of the
        # three real probe shapes.
        assert "== grammars ==" in out
        assert re.search(
            r"present \(verified via the tool's own python\)|"
            r"not importable in the tool environment|"
            r"no tool python found at ",
            out,
        )

    def test_check_reports_version_of_installed_binary(self, tmp_path: Path) -> None:
        """When the binary answers `--version`, the report carries that version line."""
        fake = tmp_path / "ver-bin"
        fake.mkdir()
        (fake / "fastedit").write_text(
            "#!/bin/bash\n"
            'if [[ "$1" == "--version" ]]; then echo "fastedit 0.5.0"; exit 0; fi\n'
            'if [[ "$1" == "--help" ]]; then\n'
            '  echo "usage: fastedit [command]"\n'
            '  echo "commands:"\n'
            '  echo "  create"\n'
            '  echo "  duplicate"\n'
            '  echo "  split"\n'
            '  echo "  join"\n'
            "  exit 0\n"
            "fi\n"
            "exit 0\n"
        )
        (fake / "fastedit").chmod(0o755)
        home = tmp_path / "check-home"
        home.mkdir()
        env = {**_stub_toolchain(tmp_path, with_npx=False), "HOME": str(home)}
        env["PATH"] = f"{fake}:{env['PATH']}"
        result = run("--check", "--ref", "feat/create-file", env_extra=env)
        assert result.returncode == 0, result.stderr
        assert "fastedit --version: fastedit 0.5.0" in result.stdout

    def test_check_reports_a_non_answer_version_probe(self, tmp_path: Path) -> None:
        """A binary without a working --version gets the honest 'did not answer' line."""
        fake = tmp_path / "nover-bin"
        fake.mkdir()
        (fake / "fastedit").write_text("#!/bin/bash\nexit 0\n")
        (fake / "fastedit").chmod(0o755)
        home = tmp_path / "check-home"
        home.mkdir()
        env = {**_stub_toolchain(tmp_path, with_npx=False), "HOME": str(home)}
        env["PATH"] = f"{fake}:{env['PATH']}"
        result = run("--check", "--ref", "feat/create-file", env_extra=env)
        assert result.returncode == 0, result.stderr
        assert "fastedit --version: (did not answer)" in result.stdout

    def test_check_resolves_the_real_fork_state(self) -> None:
        """The one live-integration assertion: this branch IS installable from the fork."""
        result = run("--check")
        assert result.returncode == 0, result.stderr
        out = result.stdout
        assert f"branch '{INSTALLABLE_BRANCH}' exists on the fork remote" in out
        assert f"{FORK_URL}@{INSTALLABLE_BRANCH}" in out

    def test_check_with_dry_run_is_a_clean_error(self) -> None:
        """--check --dry-run is a contradiction; the script says so and exits non-zero."""
        result = run("--check", "--dry-run")
        assert result.returncode != 0
        assert "--dry-run" in result.stderr


class TestSuccessFooter:
    """The footer a SUCCESSFUL real install prints: the from-scratch one-liners
    and the stale-main pin note. Hermetic via the stubbed toolchain."""

    def test_real_run_prints_the_uvx_one_liner(self, tmp_path: Path) -> None:
        home = tmp_path / "footer-home"
        home.mkdir()
        env = {**_stub_toolchain(tmp_path), "HOME": str(home)}
        result = run("--ref", "feat/create-file", "--no-model", env_extra=env)
        assert result.returncode == 0, result.stderr
        assert (
            f"uvx --from 'fastedits[mcp] @ git+{FORK_URL}@{INSTALLABLE_BRANCH}' fastedit --help"
        ) in result.stdout

    def test_real_run_prints_the_pinned_curl_one_liner(self, tmp_path: Path) -> None:
        home = tmp_path / "footer-home"
        home.mkdir()
        env = {**_stub_toolchain(tmp_path), "HOME": str(home)}
        result = run("--ref", "feat/create-file", "--no-model", env_extra=env)
        assert result.returncode == 0, result.stderr
        assert f"curl -fsSL https://raw.githubusercontent.com/Emasoft/fastedit/{INSTALLABLE_BRANCH}/scripts/install-dev.sh | bash" in result.stdout

    def test_real_run_mentions_fastedit_version_for_verification(self, tmp_path: Path) -> None:
        home = tmp_path / "footer-home"
        home.mkdir()
        env = {**_stub_toolchain(tmp_path), "HOME": str(home)}
        result = run("--ref", "feat/create-file", "--no-model", env_extra=env)
        assert result.returncode == 0, result.stderr
        assert "fastedit --version" in result.stdout

    def test_real_run_prints_the_branch_pin_note(self, tmp_path: Path) -> None:
        """The stale-main warning names the default branch AND the installable one."""
        home = tmp_path / "footer-home"
        home.mkdir()
        env = {**_stub_toolchain(tmp_path), "HOME": str(home)}
        result = run("--ref", "feat/create-file", "--no-model", env_extra=env)
        assert result.returncode == 0, result.stderr
        assert "GitHub landing page shows the stale default branch (main)" in result.stdout
        assert f"installable branch is {INSTALLABLE_BRANCH}" in result.stdout

    def test_footer_is_suppressed_in_check_mode(self, tmp_path: Path) -> None:
        """--check shows the one-liners under its own report header, but never
        the SUCCESS footer ("Install from scratch on another machine") — that
        belongs to a real install that just succeeded."""
        home = tmp_path / "check-home"
        home.mkdir()
        env = {**_stub_toolchain(tmp_path, with_npx=False), "HOME": str(home)}
        result = run("--check", "--ref", "feat/create-file", env_extra=env)
        assert result.returncode == 0, result.stderr
        out = result.stdout
        assert "Install from scratch on another machine" not in out
        assert "== install from scratch (no clone needed) ==" in out
        assert f"uvx --from 'fastedits[mcp] @ git+{FORK_URL}@{INSTALLABLE_BRANCH}' fastedit --help" in out

    def test_footer_is_suppressed_in_revert_mode(self, tmp_path: Path) -> None:
        home = tmp_path / "revert-home"
        home.mkdir()
        env = {**_stub_toolchain(tmp_path), "HOME": str(home)}
        result = run("--revert", "--no-model", env_extra=env)
        assert result.returncode == 0, result.stderr
        assert "uvx --from" not in result.stdout
