#!/usr/bin/env bash
# Install the `fastedits` CLI from this fork — or revert to upstream.
#
# ONE installer, two forward modes:
#
#   default (fork)  install from the fork's git repo, pinned to --ref.
#   --dev           install EDITABLE from this repo's working tree, so local
#                   changes take effect without reinstalling. For development.
#
# How the mode is chosen:
#   interactive (terminal on stdin): you are ASKED — "[1] local working tree
#     (editable) [2] remote GitHub fork". A standalone copy of this script
#     (curl|bash, no fastedit clone next to it) has no local tree to offer,
#     so it skips the menu and installs from the fork, saying so in one line.
#   non-interactive: no menu; --dev decides, otherwise the remote fork.
#   --revert never shows the menu.
#
# --check is the READ-ONLY axis: it prints the resolved state of this
# machine (installed fastedits per method, the fastedit binary that wins on
# PATH and what `fastedit --version` reports, VALID/STALE model caches, the
# grammar pack, the local clone) plus the fork's branch state — whether the
# ref exists on the remote, and the stale-default-branch pin note — and
# exits. It installs nothing, removes nothing, and needs neither npx nor a
# working network (an unreachable remote downgrades to a warning).
#
# In fork mode, run from inside a clone, the ref defaults to THAT CLONE's
# current branch (`git rev-parse --abbrev-ref HEAD`) and — on a real run —
# is verified to exist on the fork remote (`git ls-remote --heads`, bounded)
# before anything is touched; a missing branch aborts loudly listing the
# remote's actual heads. An explicit --ref overrides autodetect (and skips
# the check: it may name a tag or a sha, which --heads cannot see).
#
# This fork is deliberately NOT published to PyPI (PyPI stays upstream's
# release channel). Fork mode IS the fork's distribution mechanism: it
# points `uv tool install` at the fork's git repo instead.
#
# The fork and upstream share the same PyPI name (`fastedits`) and the
# same console-script names (fastedit, fastedit-hook, fastedit-mcp), so
# a leftover install from ANY method collides with the one this script
# just installed, and whichever sits first on PATH silently wins. That is
# why the preflight below reports every installed fastedits (uv tool /
# pipx / pip, plus which fastedit binary wins on PATH right now), the
# sweep uninstalls every method, and the postflight step resolves the
# binary that will actually run and confirms it is the fork, not just
# that *a* fastedit exists somewhere.
#
# Model caches under ~/.cache/fastedit/models are reported as VALID
# (holds *.safetensors) or STALE (a partial download). VALID caches are
# kept — they are shared between fork and upstream. STALE caches are
# removed (asked on a terminal [Y/n]; removed by default without one) so
# the next pull reinstalls cleanly. --revert keeps every cache: it never
# touches model weights (existing contract).
#
# After the package is in place, the installer also installs the repo's agent
# skill (skills/fastedit) with the Vercel skills CLI, FROM THIS CLONE'S
# WORKING TREE — the same skill content the package ships. GitHub sources
# (the repo shorthand and tree URLs) would resolve the fork's default branch
# (main), which still carries the legacy claude-skill content, so the local
# tree is the only correct source. The skill is an add-on, not the product: a
# missing npx or a failed skill install only warns (--no-skill skips the axis
# entirely; --revert removes the skill best-effort).
set -euo pipefail

FORK_URL="https://github.com/Emasoft/fastedit"
PACKAGE="fastedits"
REF="feat/create-file"
# The branch people should install and land on, spelled once. The GitHub
# DEFAULT branch (main) is stale — the repo's landing page shows old code —
# so every user-facing one-liner (raw.githubusercontent script URL, uvx
# direct-ref) and every pin note names THIS branch, never "main".
INSTALLABLE_BRANCH="feat/create-file"
RAW_BASE_URL="https://raw.githubusercontent.com/Emasoft/fastedit"
# The installer-managed editable clone for standalone runs (no local tree
# beside this script): the ONE install shape is `uv tool install --force
# --editable <clone>[extras]`, so the curl one-liner produces exactly the
# same result as `--dev` from a checkout. FASTEDIT_CLONE_DIR moves the clone:
# a tree there is the user's, so a dirty one is refused (never hard-reset)
# and --revert leaves it in place — only the default ~/.fastedit/src path is
# installer-owned and deletable.
CLONE_DIR="${FASTEDIT_CLONE_DIR:-${HOME}/.fastedit/src}"
EXTRAS=""
# Whether --extras was PASSED, tracked separately from its value. `--extras ""`
# is a deliberate opt-out and must be distinguishable from never passing the
# flag at all; testing `-z "$EXTRAS"` alone conflates the two and silently
# installs a backend the user explicitly declined (measured: it did exactly
# that before this flag existed).
EXTRAS_SET=0
REVERT=0
DRY_RUN=0
# CHECK_ONLY=1 (--check) turns the whole run into a read-only state report.
# Declared before the flag loop, like every other mode, because the flag loop
# references it and the loop runs before any other statement.
CHECK_ONLY=0
NO_MODEL=0
# NO_SKILL=1 (--no-skill) skips the whole agent-skill axis: nothing is
# installed in the forward modes and --revert does not try to remove it.
NO_SKILL=0
# The ONE agent the skills CLI is pointed at, spelled in the agent-id
# vocabulary `fastedit init --skill-agent` takes. Every `-a` flag and every
# success line derives from this variable, so what ran and what is reported
# cannot disagree.
SKILL_AGENT="claude-code"
# DEV=1 installs editable from this repo's working tree instead of the fork's
# git URL. It is set by --dev, or by picking [1] in the source menu. REPO_ROOT
# is resolved unconditionally (this script's parent directory) because the
# menu needs to know whether a local fastedit tree EXISTS (pyproject.toml
# beside the script) — but it is only VALIDATED when DEV resolves to 1, so a
# downloaded copy of this script run without --dev never depends on where it
# happens to sit.
# DEPRECATED: no-op since the one-install-shape conversion (--dev accepted for compatibility).
DEV=0
REPO_ROOT=""
HAVE_LOCAL_TREE=0
# Whether --ref was passed -- tracked only so --dev can say --ref had no
# effect, rather than silently swallowing a flag the user typed.
REF_SET=0

# The all-grammars decision. ALL_GRAMMARS holds the resolved answer
# ("yes"/"no"); ALL_GRAMMARS_SET records whether the FLAG was passed, so a
# resolved answer can be told apart from an unresolved one -- the same
# flag-vs-empty distinction EXTRAS_SET makes for --extras.
ALL_GRAMMARS=""
ALL_GRAMMARS_SET=0

# Model cache root — mirrors src/fastedit/model_download.py's
# DEFAULT_CACHE_DIR (~/.cache/fastedit/models). fastedit's own pull resolves
# env var -> local dir -> cache dir -> download, in that order, so we don't
# re-implement that lookup here; the script only needs WHERE caches live to
# report and clean them.
MODELS_DIR="${HOME}/.cache/fastedit/models"

# Tolerable "nothing to do" outcomes across uv / pipx / pip. A real
# failure (permissions, corrupt env, network) must NOT match this and
# must still abort the script.
TOLERABLE_UNINSTALL_RE='not installed|nothing to uninstall|externally-managed-environment'

# ONE line stating the platform mapping — the single source used by both
# usage() and the source menu, so the two surfaces cannot drift apart.
platform_line() {
  printf '%s\n' "Platforms: Darwin/arm64 -> mlx (model mlx-8bit); Linux with NVIDIA -> vllm (model bf16); anything else -> mcp only; Windows: run this script under Git Bash/WSL (bash required, no native path)"
}

usage() {
  cat <<'EOF'
Usage: install-dev.sh [--dev] [--ref REF] [--extras LIST] [--all-grammars yes|no]
                      [--no-skill] [--revert] [--no-model] [--dry-run] [--check]

  --check        Read-only state report, then exit: installs nothing, removes
                 nothing. Prints the installed fastedits (uv tool / pipx / pip),
                 the fastedit binary that wins on PATH and what its
                 `fastedit --version` reports, the model caches (VALID/STALE),
                 the grammar pack, and — from the fork remote — whether the
                 installable branch exists and the pin note for the stale
                 default branch. Needs no npx; works offline (an unreachable
                 remote is a warning, not a failure).

  --dev          Accepted for compatibility, no longer needed: EVERY install
                 is now an editable install from a local clone (a checkout
                 you run the script from, or the installer-managed clone at
                 ~/.fastedit/src). Cannot be combined with --revert.
  --ref REF      Branch the MANAGED CLONE tracks (standalone runs; created/
                 updated at ~/.fastedit/src, override with
                 FASTEDIT_CLONE_DIR). No effect when running from inside a
                 checkout (the working tree is the install source). Without
                 --ref the managed clone tracks the built-in default
                 (feat/create-file).
  --extras LIST  Comma-separated extras. DEFAULT: every extra this platform
                 can install (mlx,mcp on Apple Silicon; vllm,mcp on Linux with
                 an NVIDIA driver; mcp elsewhere). Pass --extras "" for none --
                 the model pull is then skipped, since nothing could load it.
  --all-grammars yes|no
                 Install (yes) or skip (no) the offline 173-language grammar
                 pack (the `all-grammars` extra) without being asked. When the
                 flag is not passed the script ASKS (Enter = yes); with no
                 terminal on stdin it defaults to yes and says so.
  --no-skill     Skip the agent skill entirely: nothing is installed by the
                 forward modes, and --revert does not try to remove it.
  --revert       Uninstall the fork and reinstall upstream fastedits
                 from PyPI (undoes the swap; leaves downloaded model
                 weights in place — they're shared with the fork).
                 Also removes the globally installed agent skill
                 (best-effort; skipped when npx is missing or --no-skill
                 was passed). Never asks about grammars and never
                 installs them.
  --no-model     Skip downloading the merge model (~3 GB). The stale
                 model-cache cleanup below still runs. Ignored with
                 --revert, which never touches the model.
  --dry-run      Print the commands that would run; execute nothing
                 (model caches are reported, never removed)
  -h, --help     Show this help
EOF
  platform_line
  cat <<'EOF'

How it works: the ONE install shape is an editable install from a local clone.
  * Run inside a fastedit checkout -> that working tree is the install source.
  * Run standalone (the curl|bash one-liner) -> the installer maintains a
    managed clone at ~/.fastedit/src (override: FASTEDIT_CLONE_DIR): cloned
    on first run, then fetched and hard-reset to the tracked branch on every
    run, so an update is just re-running this script (or `git -C
    ~/.fastedit/src pull`).
The install is always `uv tool install --force --editable`; --dev exists as
a no-op compatibility flag. The agent skill (skills/fastedit) is installed
from the SAME clone the package comes from.

Preflight: before installing, the script reports every installed fastedits
(uv tool, pipx, pip — plus which fastedit binary wins on PATH right now) and
every model cache under ~/.cache/fastedit/models (VALID = holds
*.safetensors). VALID caches are kept — shared between fork and upstream.
STALE (partial) caches are removed so the next pull reinstalls cleanly:
asked on a terminal [Y/n], removed by default without one, printed as
"would remove" by --dry-run. --revert keeps every cache.

Agent skill: after the package install, the script also installs the agent
skill with the Vercel skills CLI, from the SAME clone the package comes from
(the checkout, or the managed clone on a standalone run):
  npx --yes skills add <install_source>/skills/fastedit -g -a claude-code -y
(global, non-interactive, claude-code target; the path IS the skill, and
--ref never affects it — no GitHub source is involved, because the repo
shorthand and tree URLs would resolve the fork's default branch (main),
which still carries the legacy claude-skill content). A missing npx prints a
one-line manual-install note and a failed install only warns — neither ever
fails the installer. The postflight reports the axis truthfully:
 "agent skill: installed (claude-code, global)" / "agent skill: NOT FOUND
(see warnings above)" / "agent skill: skipped (...)". --revert removes the
skill best-effort with "npx --yes skills remove fastedit -g -y".

Installable branch: the fork's GitHub DEFAULT branch (main) is stale — the
repo's landing page shows old code — so the installable branch is
feat/create-file, the branch the managed clone tracks. The uvx one-liner and
the raw.githubusercontent script URL pin it explicitly; a successful install
prints both from-scratch one-liners plus a `fastedit --version` verification
hint in its footer.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dev)
      DEV=1
      shift
      ;;
    --ref)
      REF="$2"
      REF_SET=1
      shift 2
      ;;
    --extras)
      EXTRAS="$2"
      EXTRAS_SET=1
      shift 2
      ;;
    --all-grammars)
      if [[ $# -lt 2 ]]; then
        echo "error: --all-grammars requires a value: yes or no" >&2
        usage >&2
        exit 1
      fi
      case "$2" in
        yes|YES|Yes|y|Y) ALL_GRAMMARS="yes"; ALL_GRAMMARS_SET=1 ;;
        no|NO|No|n|N)    ALL_GRAMMARS="no";  ALL_GRAMMARS_SET=1 ;;
        *)
          echo "error: --all-grammars expects 'yes' or 'no', got: $2" >&2
          usage >&2
          exit 1
          ;;
      esac
      shift 2
      ;;
    --revert)
      REVERT=1
      shift
      ;;
    --no-model)
      NO_MODEL=1
      shift
      ;;
    --no-skill)
      NO_SKILL=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --check)
      CHECK_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "$DEV" -eq 1 && "$REVERT" -eq 1 ]]; then
  echo "error: --dev and --revert contradict each other: --dev installs this working tree," >&2
  echo "error: --revert restores upstream fastedits from PyPI. Pick one." >&2
  exit 1
fi

if [[ "$CHECK_ONLY" -eq 1 && ("$DEV" -eq 1 || "$REVERT" -eq 1 || "$DRY_RUN" -eq 1) ]]; then
  echo "error: --check is read-only and combines with nothing: it installs and mutates nothing," >&2
  echo "error: so --dev, --revert and --dry-run are all meaningless beside it. Drop them." >&2
  exit 1
fi

if [[ -n "$EXTRAS" ]]; then
  PKG_SPEC="${PACKAGE}[${EXTRAS}]"
else
  PKG_SPEC="${PACKAGE}"
fi

# PREFLIGHT — fail before touching anything if uv is missing, so we
# never half-uninstall the current tool.
if ! command -v uv >/dev/null 2>&1; then
  echo "error: 'uv' is not on PATH. Install it first: https://docs.astral.sh/uv/" >&2
  exit 1
fi

# Resolve the script's own repo root unconditionally: the source menu (and the
# branch autodetect below) need to know whether a local fastedit tree EXISTS
# (pyproject.toml beside this script), not just --dev. Validation stays with
# DEV -- a downloaded copy of this script run without --dev must not abort
# just because of where it happens to sit.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${REPO_ROOT}/pyproject.toml" ]]; then
  HAVE_LOCAL_TREE=1
fi

# Extras in a PEP 508 direct reference (`pkg[extra] @ git+...`) are what this
# script installs, and older uv did not support bracket extras on a tool
# install (astral-sh/uv#6296). The failure mode there is the one this whole
# script exists to prevent and it is SILENT: extras quietly dropped, the model
# still downloaded, the backend absent, exit 0. Verified working on 0.12.12;
# a floor of 0.5.0 is well below that and well above the issue. Warn rather
# than abort -- the bound is inferred from an issue, not measured on old
# versions, so refusing to install on it would be a guess with teeth. The
# same bracket shape appears in dev mode as `path[extra]`, so the warning
# covers both modes.
#
# The MAJOR.MINOR shape is validated before comparing, and that guard is not
# decoration -- it was measured. Without it, a `uv --version` whose second field
# is not the version (a wrapper, a shim, a localized build printing
# "uv version 0.12.12") makes the comparison operate on the literal word, and
# bash arithmetic treats an unset name as 0, so BOTH tests pass and a MODERN uv
# gets warned at. A warning that cries wolf is worse than no warning: it teaches
# the reader to scroll past the one line that would have mattered.
_uv_ver="$(uv --version 2>/dev/null | awk '{print $2}')"
if [[ "$_uv_ver" =~ ^[0-9]+\.[0-9]+ ]]; then
  _uv_major="${_uv_ver%%.*}"
  _uv_minor="${_uv_ver#*.}"; _uv_minor="${_uv_minor%%.*}"
  if [[ "$_uv_major" -eq 0 && "$_uv_minor" -lt 5 ]]; then
    echo "warning: uv $_uv_ver is older than 0.5; bracket extras on a tool install may be" >&2
    echo "warning: silently ignored, leaving the merge model installed but unloadable." >&2
    echo "warning: run 'fastedit doctor' afterwards and check the backend line." >&2
  fi
fi

# Bound a command's runtime when a timeout binary exists. macOS ships without
# coreutils' `timeout` (homebrew provides `gtimeout`); there we run unbounded
# rather than fail -- the bound is a safety net for a hung network call, not a
# correctness requirement.
run_bounded() {
  local secs="$1"
  shift
  if command -v timeout >/dev/null 2>&1; then
    timeout "$secs" "$@"
  elif command -v gtimeout >/dev/null 2>&1; then
    gtimeout "$secs" "$@"
  else
    "$@"
  fi
}

# Render argv as a shell-quoted line a user can copy-paste and get the
# same invocation (so a spec like "fastedits @ git+https://..." stays one
# paste-able token). An argument made only of characters that never need
# quoting (word chars, path/URL punctuation) is left bare -- that's the
# common case (uv, tool, uninstall, fastedits) and unquoted reads best.
# Anything else is wrapped in single quotes -- far more readable than
# printf %q's backslash-per-space/bracket escaping -- except when the
# argument itself contains a single quote, where %q is used instead so
# correctness never depends on the nicer-looking branch.
quote_argv() {
  local out="" arg quoted
  for arg in "$@"; do
    if [[ "$arg" =~ ^[A-Za-z0-9_./:+@=-]+$ ]]; then
      quoted="$arg"
    elif [[ "$arg" == *"'"* ]]; then
      printf -v quoted '%q' "$arg"
    else
      quoted="'${arg}'"
    fi
    out+="${out:+ }${quoted}"
  done
  printf '%s' "$out"
}

# ONE install shape: an EDITABLE install from a local clone. Inside a
# fastedit checkout the clone is that working tree; standalone (no
# pyproject.toml beside this script — the curl|bash case) the installer
# manages its own clone at CLONE_DIR: clone if missing, else fetch +
# checkout + `git reset --hard origin/<branch>` to the branch tip (the
# clone is installer-owned, so a hard reset is the correct sync; a
# FASTEDIT_CLONE_DIR override pointing at a user-managed tree with
# UNCOMMITTED changes is refused instead of reset). --dev is accepted as
# a no-op compatibility flag: every install is editable now. --dry-run
# prints the git steps as `+ ` transcript lines and executes none of them:
# a preview must never clone, fetch or reset for real.
resolve_source_mode() {
  if [[ "$REVERT" -eq 1 ]]; then
    return 0
  fi
  if [[ "$HAVE_LOCAL_TREE" -eq 1 ]]; then
    INSTALL_SOURCE="$REPO_ROOT"
    echo "installing editable from this checkout: ${INSTALL_SOURCE}"
    return 0
  fi
  INSTALL_SOURCE="${CLONE_DIR}"
  if [[ -f "${CLONE_DIR}/pyproject.toml" ]]; then
    local porcelain=""
    porcelain="$(run_bounded 30 git -C "$CLONE_DIR" status --porcelain 2>/dev/null || true)"
    if [[ -n "$porcelain" && "${CLONE_DIR}" != "${HOME}/.fastedit/src" ]]; then
      echo "error: FASTEDIT_CLONE_DIR points at ${CLONE_DIR}, which has uncommitted changes — refusing to reset a tree you manage." >&2
      exit 1
    fi
    # The dry-run guard covers the check too: the run is preview-only, so no
    # git subcommand against the managed clone may fire (stdout stays hermetic).
    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "+ git -C ${CLONE_DIR} fetch origin ${REF}"
      echo "+ git -C ${CLONE_DIR} checkout ${REF}"
      echo "+ git -C ${CLONE_DIR} reset --hard origin/${REF}"
      echo "managed clone updated to origin/${REF} (dry-run): ${CLONE_DIR}"
      return 0
    fi
    run_bounded 120 git -C "$CLONE_DIR" fetch origin "$REF"
    run_bounded 30 git -C "$CLONE_DIR" checkout "$REF"
    run_bounded 30 git -C "$CLONE_DIR" reset --hard "origin/${REF}"
    echo "managed clone updated to origin/${REF} ($(run_bounded 10 git -C "$CLONE_DIR" rev-parse --short HEAD 2>/dev/null || echo '?')): ${CLONE_DIR}"
  else
    echo "no fastedit clone here — cloning into ${CLONE_DIR} (branch ${REF})..."
    echo "+ git clone --branch ${REF} ${FORK_URL} ${CLONE_DIR}"
    if [[ "$DRY_RUN" -eq 1 ]]; then
      return 0
    fi
    run_bounded 300 git clone --branch "$REF" "$FORK_URL" "$CLONE_DIR"
  fi
}

# Run one uninstall command, tolerating "there was nothing to remove"
# in whatever shape that tool spells it, without masking a real failure.
run_uninstall_step() {
  local label="$1"
  shift
  echo "+ $(quote_argv "$@")"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    return 0
  fi
  local out status=0
  out=$("$@" 2>&1) || status=$?
  if [[ -n "$out" ]]; then
    echo "$out"
  fi
  if [[ "$status" -ne 0 ]] && ! grep -qiE "$TOLERABLE_UNINSTALL_RE" <<<"$out"; then
    echo "error: ${label} failed" >&2
    exit "$status"
  fi
}

# Sweep every install method that could be shadowing the one we're about
# to put in place. uv is always attempted (it's our own preflight
# dependency); pipx/pip only if actually present on this machine.
sweep_uninstall() {
  local pkg="$1"
  echo "Sweeping other install locations for ${pkg} (uv tool, pipx, pip)..."
  run_uninstall_step "uv tool uninstall" uv tool uninstall "$pkg"

  if command -v pipx >/dev/null 2>&1; then
    run_uninstall_step "pipx uninstall" pipx uninstall "$pkg"
  fi

  local pipbin=""
  if command -v pip3 >/dev/null 2>&1; then
    pipbin="pip3"
  elif command -v pip >/dev/null 2>&1; then
    pipbin="pip"
  fi
  if [[ -n "$pipbin" ]]; then
    run_uninstall_step "${pipbin} uninstall" "$pipbin" uninstall -y "$pkg"
  fi
}

# The package spec for a given extras list: ALWAYS `path[extras]` against
# the resolved local clone (INSTALL_SOURCE — the checkout this script sits
# in, or the managed clone). Empty extras -> bare path. Verified on uv
# 0.12.13: bracket extras on a local path resolve, and --editable tracks
# the tree.
build_spec() {
  local extras="$1"
  if [[ -n "$extras" ]]; then
    printf '%s[%s]' "$INSTALL_SOURCE" "$extras"
  else
    printf '%s' "$INSTALL_SOURCE"
  fi
}

do_install() {
  local spec="$1"
  echo "+ uv tool install --force --editable $(quote_argv "$spec")"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    return 0
  fi
  uv tool install --force --editable "$spec"
}

# Same install, but a failure is REPORTED rather than fatal. This exists for
# exactly one caller: the optional backend-extra attempt below. The sweep
# uninstalls before it installs, so any install step that can fail is a step
# that can leave this machine with NO fastedit at all — and with no fastedit
# there is no sanctioned way to edit source and repair it. A compiled extra
# (mlx) is the only genuinely failure-prone spec this script builds, so it is
# the one install that must never be allowed to abort the run.
do_install_optional() {
  do_install "$1" || return 1
}

# The forward path does NOT pre-verify the branch on the fork remote: the
# managed clone is fetched/reset by git itself, and a missing ref fails
# there, loudly, before the sweep. (The old verify_remote_ref pre-check was
# dropped with the fork-URL install mode; --check still probes the remote.)

# PREFLIGHT DETECTION — report what is installed and what is cached, BEFORE
# anything is uninstalled. Runs even in --dry-run: a dry run should print
# everything the real run would act on, so the transcript is a complete
# preview. Purely informational: never installs, never removes.
detect_installed_fastedits() {
  echo "preflight: currently installed ${PACKAGE} (uv tool / pipx / pip):"
  local found=0 line

  # `uv tool list` prints one tool per block, headed "fastedits v0.1.2".
  local uv_list=""
  uv_list="$(run_bounded 15 uv tool list 2>/dev/null || true)"
  if [[ -n "$uv_list" ]]; then
    while IFS= read -r line; do
      case "$line" in
        "$PACKAGE "*) echo "  uv tool: $line"; found=1 ;;
      esac
    done <<<"$uv_list"
  fi

  # `pipx list --short` prints one "name version" pair per line.
  if command -v pipx >/dev/null 2>&1; then
    local pipx_list=""
    pipx_list="$(run_bounded 15 pipx list --short 2>/dev/null || true)"
    if [[ -n "$pipx_list" ]]; then
      while IFS= read -r line; do
        case "$line" in
          "$PACKAGE "*) echo "  pipx: $line"; found=1 ;;
        esac
      done <<<"$pipx_list"
    fi
  fi

  # Same pip resolution the sweep uses, so the report never names a pip the
  # sweep would not touch. `pip show` exits non-zero when absent.
  local pipbin=""
  if command -v pip3 >/dev/null 2>&1; then
    pipbin="pip3"
  elif command -v pip >/dev/null 2>&1; then
    pipbin="pip"
  fi
  if [[ -n "$pipbin" ]]; then
    local show=""
    if show="$(run_bounded 15 "$pipbin" show "$PACKAGE" 2>/dev/null)"; then
      local ver loc
      ver="$(awk -F': ' '/^Version:/ {print $2; exit}' <<<"$show")"
      loc="$(awk -F': ' '/^Location:/ {print $2; exit}' <<<"$show")"
      echo "  ${pipbin}: ${PACKAGE} ${ver:-?} (${loc:-unknown location})"
      found=1
    fi
  fi

  # The binary that would actually win on PATH right now -- the same
  # resolution the postflight re-checks after the install.
  local bin_path=""
  bin_path="$(command -v fastedit 2>/dev/null || true)"
  if [[ -n "$bin_path" ]]; then
    echo "  binary: ${bin_path} (the fastedit that wins on PATH right now)"
    found=1
  fi

  if [[ "$found" -eq 0 ]]; then
    echo "  (none found — clean slate)"
  fi
}

# VALID or STALE for one model directory: VALID iff it holds at least one
# real weight file (*.safetensors). A dir with only config/tokenizer files
# is a partial download -- pulling into it would resume over a half-written
# state, so it is the thing the cleanup below offers to remove.
model_cache_state() {
  local dir="$1"
  if compgen -G "${dir}/*.safetensors" >/dev/null 2>&1; then
    printf '%s' "VALID"
  else
    printf '%s' "STALE (partial, no *.safetensors)"
  fi
}

detect_model_caches() {
  echo "preflight: model caches under ${MODELS_DIR}:"
  if [[ ! -d "$MODELS_DIR" ]]; then
    echo "  (no cache directory — nothing cached yet)"
    return 0
  fi
  local d name size state any=0
  # Non-glob iteration guard: with no entries the pattern stays literal and
  # the [[ -d ]] test skips it.
  for d in "$MODELS_DIR"/*; do
    [[ -d "$d" ]] || continue
    any=1
    name="$(basename "$d")"
    size="$(run_bounded 15 du -sh "$d" 2>/dev/null | awk '{print $1}' || true)"
    state="$(model_cache_state "$d")"
    echo "  ${name}  ${size:-?}  ${state}"
  done
  if [[ "$any" -eq 0 ]]; then
    echo "  (no model directories)"
  fi
}

# Act on what detect_model_caches reported: keep VALID caches (they are
# shared between fork and upstream -- deleting them would cost a ~3 GB
# re-download for nothing), remove STALE ones so the next pull reinstalls
# cleanly. --revert keeps EVERY cache (existing contract: revert never
# touches model weights). Like every prompt in this script, the question is
# resolved BEFORE the sweep, so nobody sits uninstalled while a prompt waits.
# In --dry-run nothing is removed: the action is printed as "would remove".
handle_model_caches() {
  if [[ "$REVERT" -eq 1 ]]; then
    echo "model caches: kept — --revert leaves downloaded weights in place (shared with upstream)"
    return 0
  fi
  if [[ ! -d "$MODELS_DIR" ]]; then
    return 0
  fi
  local d name answer state
  for d in "$MODELS_DIR"/*; do
    [[ -d "$d" ]] || continue
    name="$(basename "$d")"
    state="$(model_cache_state "$d")"
    if [[ "$state" == "VALID" ]]; then
      echo "model cache ${name}: kept — shared cache"
      continue
    fi
    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "would remove stale model cache ${name} (${d}) — partial download, no *.safetensors; the next pull reinstalls it cleanly"
      continue
    fi
    answer=""
    if [[ -t 0 ]]; then
      printf '%s' "remove stale model cache '${name}' (${d})? [Y/n] " >&2
      read -r answer || answer=""
      answer="${answer//[[:space:]]/}"
      case "$answer" in
        n|N|no|NO|No)
          echo "model cache ${name}: kept (kept at your request)"
          continue
          ;;
      esac
    else
      echo "note: non-interactive: removing stale model cache '${name}' (partial, no *.safetensors); the pull reinstalls it cleanly" >&2
    fi
    run_bounded 60 rm -rf "$d"
    echo "removed stale model cache '${name}' (${d})"
  done
}

# Resolve whether all-grammars goes into the install. Precedence:
# --revert (never asks, never adds), then the --all-grammars flag, then the
# interactive prompt / non-interactive default.
ask_all_grammars() {
  if [[ "$REVERT" -eq 1 ]]; then
    printf '%s' "no"
    return 0
  fi
  if [[ "$ALL_GRAMMARS_SET" -eq 1 ]]; then
    printf '%s' "$ALL_GRAMMARS"
    return 0
  fi
  if [[ -t 0 ]]; then
    local answer=""
    printf '%s' "Install all grammars? (offline 173-language tree-sitter pack — every tree-sitter-supported format works) [Y/n] " >&2
    read -r answer || answer=""
    answer="${answer//[[:space:]]/}"
    case "$answer" in
      n|N|no|NO|No) printf '%s' "no" ;;
      *) printf '%s' "yes" ;;  # Enter (empty), y/yes, and anything unrecognized -> the [Y] default
    esac
  else
    echo "note: non-interactive: all grammars enabled by default; pass --all-grammars no to skip" >&2
    printf '%s' "yes"
  fi
}

# Append all-grammars to an extras list, deduped. Answering yes to the prompt
# IS an instruction, so it outranks an empty --extras opt-out for grammars
# only: "" becomes exactly "all-grammars", and a list that already carries the
# extra is left alone. Answering no changes nothing.
extras_with_all_grammars() {
  local extras="$1"
  if [[ "$GRAMMARS_ANSWER" != "yes" ]]; then
    printf '%s' "$extras"
    return 0
  fi
  if [[ -z "$extras" ]]; then
    printf '%s' "all-grammars"
    return 0
  fi
  case ",${extras}," in
    *,all-grammars,*) printf '%s' "$extras" ;;
    *) printf '%s' "${extras},all-grammars" ;;
  esac
}

# `fastedit pull` (get_model_path) already resolves env var -> local dir
# -> cache dir -> download, in that order, so it's a no-op when the
# model is already cached. We don't re-implement that lookup here —
# duplicating fastedit's own cache path would just be a second, driftable
# copy of the same logic. We only decide WHICH model this platform wants.
detect_model() {
  local os arch
  os="$(uname -s)"
  arch="$(uname -m)"
  if [[ "$os" == "Darwin" && "$arch" == "arm64" ]]; then
    echo "mlx-8bit"
    return 0
  fi
  if [[ "$os" == "Linux" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    echo "bf16"
    return 0
  fi
  return 1
}

# EVERY extra this platform can actually install (owner directive 2026-09-13:
# "it must install with all extras by default"). A default install must not be
# crippled -- an extras-less one cached 1.7 GB at mlx-8bit and then died with
# `ModuleNotFoundError: No module named 'mlx'` on the first model-merge edit.
#
# "All extras" cannot be taken literally, and that is the whole reason this is
# a function rather than a constant: the three declared extras are not
# co-installable. `mlx` ships no Linux wheels; `vllm` ships no macOS wheels.
# Asking for both guarantees a failed resolve on EVERY platform, which would
# turn a crippled install into no install at all. So it means every extra that
# is installable HERE:
#
#   Darwin/arm64      -> mlx,mcp     (mlx is the backend for the mlx-8bit model)
#   Linux + nvidia    -> vllm,mcp    (vllm is the declared CUDA backend extra)
#   anything else     -> mcp         (pure Python, installs anywhere)
#
# `mcp` is unconditional: it is the MCP server, part of the advertised surface,
# and pure Python via fastmcp. The install is non-fatal (see do_install_optional)
# precisely because mlx and vllm are compiled wheels that CAN fail to build --
# a failure falls back to the bare spec rather than leaving no fastedit at all.
detect_backend_extra() {
  local os arch
  os="$(uname -s)"
  arch="$(uname -m)"
  if [[ "$os" == "Darwin" && "$arch" == "arm64" ]]; then
    echo "mlx,mcp"
    return 0
  fi
  if [[ "$os" == "Linux" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    echo "vllm,mcp"
    return 0
  fi
  echo "mcp"
  return 0
}

pull_model() {
  local model="$1"
  echo "+ fastedit pull --model $(quote_argv "$model")"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    return 0
  fi
  fastedit pull --model "$model"
}

# The agent skill ships in the clone this installer resolves (INSTALL_SOURCE)
# under skills/fastedit and is installed with the Vercel skills CLI (`npx
# skills`) FROM THAT LOCAL TREE — never from a GitHub source. The fork's
# default branch (main) still carries the legacy claude-skill/SKILL.md, so the
# repo shorthand installs the WRONG content,
# and the tree-URL form fails outright in non-TTY mode (both measured). The
# installer always runs from a clone, and skills/fastedit is the very skill
# the package is built from, so the local tree needs no branch pin at all
# (--ref stays a package-only concern). The whole axis is optional by
# contract, exactly like the optional backend install: a missing npx, a
# missing local tree, or a failed install only warns, and none of them ever
# aborts the run — a skill must never be the reason this machine ends up
# without fastedit.
skill_source_dir() {
  printf '%s' "${INSTALL_SOURCE}/skills/fastedit"
}

install_agent_skill() {
  if [[ "$NO_SKILL" -eq 1 ]]; then
    return 0
  fi
  local skill_dir
  skill_dir="$(skill_source_dir)"
  if ! command -v npx >/dev/null 2>&1; then
    echo "note: npx not found — skipping the agent skill; install manually: npx --yes skills add ${skill_dir} -g -a ${SKILL_AGENT} -y"
    return 0
  fi
  if [[ ! -f "${skill_dir}/SKILL.md" ]]; then
    echo "note: no agent skill at ${skill_dir}/SKILL.md — skipping; the clone at ${INSTALL_SOURCE} has no skills/fastedit"
    return 0
  fi
  echo "+ npx --yes skills add $(quote_argv "$skill_dir") -g -a ${SKILL_AGENT} -y"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    return 0
  fi
  local out status=0
  out=$(npx --yes skills add "$skill_dir" -g -a "$SKILL_AGENT" -y 2>&1) || status=$?
  if [[ -n "$out" ]]; then
    echo "$out"
  fi
  if [[ "$status" -ne 0 ]]; then
    echo "warning: the agent skill could not be installed (npx skills add exited ${status}) — fastedit itself is unaffected." >&2
    echo "warning: install it manually later: npx --yes skills add ${skill_dir} -g -a ${SKILL_AGENT} -y" >&2
    return 0
  fi
  echo "installed: agent skill 'fastedit' (${SKILL_AGENT}, global)"
}

# --revert's counterpart: undo the global skill install. Tolerated-absent on
# purpose — no npx, or "nothing to remove", is a fine outcome on a machine
# that never had the skill — and best-effort like the install: upstream's
# PyPI package is the thing that matters on a revert.
remove_agent_skill() {
  if [[ "$NO_SKILL" -eq 1 ]]; then
    return 0
  fi
  if ! command -v npx >/dev/null 2>&1; then
    echo "note: npx not found — skipping agent-skill removal (nothing was installed by this run)"
    return 0
  fi
  echo "+ npx --yes skills remove fastedit -g -y"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    return 0
  fi
  local out status=0
  out=$(npx --yes skills remove fastedit -g -y 2>&1) || status=$?
  if [[ -n "$out" ]]; then
    echo "$out"
  fi
  if [[ "$status" -ne 0 ]]; then
    echo "note: agent-skill removal reported nothing to remove (exit ${status}) — continuing"
    return 0
  fi
  echo "removed: agent skill 'fastedit' (global)"
}

# --revert's counterpart for the OTHER thing a standalone install leaves
# behind: the installer-managed editable clone at CLONE_DIR. A local checkout
# is a user's repo and is never touched; a FASTEDIT_CLONE_DIR override points
# somewhere the user chose, so it is reported and left in place too — only
# the default ~/.fastedit/src path is deleted. Tolerated-absent by design:
# nothing to remove is the common case for checkout runs.
remove_managed_clone() {
  if [[ "$HAVE_LOCAL_TREE" -eq 1 ]]; then return 0; fi
  local clone="${CLONE_DIR}"
  if [[ ! -d "$clone" ]]; then return 0; fi
  if [[ "$clone" != "${HOME}/.fastedit/src" ]]; then
    echo "note: FASTEDIT_CLONE_DIR points outside ~/.fastedit — leaving ${clone} in place"
    return 0
  fi
  echo "+ rm -rf ${clone}"
  if [[ "$DRY_RUN" -eq 1 ]]; then return 0; fi
  rm -rf "$clone"
  echo "removed: installer-managed clone at ${clone}"
}

# A failed postflight is only useful if it tells the user how to fix it,
# not just what's wrong -- otherwise it gets shrugged off as noise ("it
# always says that") and the safety check it guards stops being trusted.
# The usual fix is the one `uv tool install` itself hints at moments
# earlier: uv's own bin dir isn't on PATH yet.
print_path_hint() {
  local uv_bin=""
  uv_bin=$(uv tool dir --bin 2>/dev/null || true)
  if [[ -n "$uv_bin" ]]; then
    echo "hint: uv installs tools into ${uv_bin} -- make sure it's on PATH." >&2
  fi
  echo "hint: run 'uv tool update-shell' (then open a new shell) or add uv's tool bin dir to PATH yourself." >&2
}

# POSTFLIGHT — the whole point of this script. Resolve the fastedit that
# will actually run (not just "one exists somewhere") and confirm it is
# the fork. A stale upstream earlier on PATH, or an upstream reinstalled
# by something else, must fail loudly here rather than pass silently.
verify_fork_install() {
  local resolved
  if ! resolved=$(command -v fastedit); then
    echo "error: no 'fastedit' found on PATH after install" >&2
    print_path_hint
    exit 1
  fi

  local all_matches
  all_matches=$(type -a fastedit 2>/dev/null | grep '^fastedit is ' | sed 's/^fastedit is //' || true)
  local match_count
  match_count=$(printf '%s\n' "$all_matches" | grep -c . || true)

  local help_output
  if ! help_output=$(fastedit --help 2>&1); then
    echo "error: 'fastedit --help' failed after install" >&2
    echo "$help_output" >&2
    exit 1
  fi

  local missing=()
  local cmd
  for cmd in create duplicate split join; do
    if ! grep -qw "$cmd" <<<"$help_output"; then
      missing+=("$cmd")
    fi
  done

  if [[ ${#missing[@]} -gt 0 ]]; then
    echo "error: the 'fastedit' on PATH (${resolved}) is missing fork subcommands: ${missing[*]}" >&2
    if [[ "$match_count" -gt 1 ]]; then
      echo "error: multiple 'fastedit' binaries are on PATH — a shadowed upstream may be winning:" >&2
      printf '  %s\n' "$all_matches" >&2
    fi
    echo "error: the install did not actually put the fork in place — aborting" >&2
    print_path_hint
    exit 1
  fi

  if [[ "$match_count" -gt 1 ]]; then
    echo "warning: multiple 'fastedit' binaries found on PATH; the one that wins is ${resolved}:" >&2
    printf '  %s\n' "$all_matches" >&2
  fi
  echo "verified: fastedit at ${resolved} lists create, duplicate, split, join"
}

# The grammar axis, reported honestly. The probe runs ONLY when all-grammars
# was actually selected -- interrogating an install that never asked for the
# pack would just report an expected absence and teach the reader to scroll
# past this block. It uses the tool env's OWN python (the venv uv made for
# the installed tool), not whatever python is first on PATH: the point is to
# test what the installed tool can actually load.
verify_grammars() {
  if [[ "$GRAMMARS_SELECTED" -ne 1 ]]; then
    echo "grammars: all-grammars not selected — extended languages (scala, lua, perl, ...) will not resolve"
    return 0
  fi
  local tool_python=""
  tool_python="$(uv tool dir 2>/dev/null)/${PACKAGE}/bin/python"
  if [[ ! -x "$tool_python" ]]; then
    echo "warning: all-grammars was selected, but no python was found at ${tool_python}; cannot verify the grammar pack." >&2
    return 0
  fi
  if "$tool_python" -c "from tree_sitter_language_pack import get_language" 2>/dev/null; then
    echo "grammars: all-grammars pack present (verified via the tool's own python)"
  else
    echo "warning: all-grammars was selected, but 'tree_sitter_language_pack' is not importable in the tool environment." >&2
    if [[ "$DEV" -eq 1 ]]; then
      echo "warning: extended languages will not resolve. Retry: uv tool install --force --editable '$(build_spec "all-grammars")'" >&2
    else
      echo "warning: extended languages will not resolve. Retry: uv tool install --force '$(build_spec "all-grammars")'" >&2
    fi
  fi
}

# The skill axis of the postflight, reported as honestly as the grammar axis:
# what was actually verified on this machine, never what we hope happened.
# The listing is bounded (npx can be slow cold) and digested to ONE line
# either way — a full `skills list` dump would bury the one fact the reader
# needs. Real runs only: the listing is a live npx call and a dry run must
# stay hermetic.
verify_agent_skill() {
  if [[ "$NO_SKILL" -eq 1 ]]; then
    echo "agent skill: skipped (--no-skill)"
    return 0
  fi
  if ! command -v npx >/dev/null 2>&1; then
    echo "agent skill: skipped (npx not found)"
    return 0
  fi
  local list_out status=0
  list_out=$(run_bounded 60 npx --yes skills list -g 2>&1) || status=$?
  if [[ "$status" -ne 0 ]]; then
    echo "warning: could not list installed agent skills ('npx skills list -g' exited ${status})." >&2
    echo "agent skill: NOT FOUND (see warnings above)"
    return 0
  fi
  if ! grep -qw fastedit <<<"$list_out"; then
    echo "warning: the agent skill install reported success, but 'npx skills list -g' does not list fastedit." >&2
    echo "agent skill: NOT FOUND (see warnings above)"
    return 0
  fi
  echo "agent skill: installed (${SKILL_AGENT}, global)"
}

# ---------------------------------------------------------------------------
# --check: the read-only state report
# ---------------------------------------------------------------------------

# The exact from-scratch one-liners and the stale-branch pin note. ONE
# function, because the footer of a successful install (print_success_footer)
# and the --check report must show the same commands — two copies would drift.
print_from_scratch_one_liners() {
  cat <<EOF
  uvx --from 'fastedits[mcp] @ git+${FORK_URL}@${INSTALLABLE_BRANCH}' fastedit --help
  curl -fsSL ${RAW_BASE_URL}/${INSTALLABLE_BRANCH}/scripts/install-dev.sh | bash
EOF
}

print_branch_pin_note() {
  cat <<EOF
note: the fork's GitHub landing page shows the stale default branch (main);
note: the installable branch is ${INSTALLABLE_BRANCH} — the one-liners above pin it.
note: verify after installing: fastedit --version
EOF
}

# The grammar axis of the --check report: same probe the postflight runs
# (the tool env's OWN python at <tool dir>/fastedits/bin/python), so --check
# answers with what the installed tool can actually load — never with what
# whatever python is first on PATH can.
check_report() {
  echo "install-dev.sh --check (read-only: installs nothing, removes nothing)"
  echo ""

  echo "== installed fastedits =="
  local any=0 line
  local uv_list=""
  uv_list="$(run_bounded 15 uv tool list 2>/dev/null || true)"
  if [[ -n "$uv_list" ]]; then
    while IFS= read -r line; do
      case "$line" in
        "$PACKAGE "*) echo "  uv tool: $line"; any=1 ;;
      esac
    done <<<"$uv_list"
  fi
  if command -v pipx >/dev/null 2>&1; then
    local pipx_list=""
    pipx_list="$(run_bounded 15 pipx list --short 2>/dev/null || true)"
    if [[ -n "$pipx_list" ]]; then
      while IFS= read -r line; do
        case "$line" in
          "$PACKAGE "*) echo "  pipx: $line"; any=1 ;;
        esac
      done <<<"$pipx_list"
    fi
  fi
  local pipbin=""
  if command -v pip3 >/dev/null 2>&1; then
    pipbin="pip3"
  elif command -v pip >/dev/null 2>&1; then
    pipbin="pip"
  fi
  if [[ -n "$pipbin" ]]; then
    local show=""
    if show="$(run_bounded 15 "$pipbin" show "$PACKAGE" 2>/dev/null)"; then
      echo "  ${pipbin}: ${PACKAGE} $(awk -F': ' '/^Version:/ {print $2; exit}' <<<"$show")"
      any=1
    fi
  fi
  if [[ "$any" -eq 0 ]]; then
    echo "  (none found)"
  fi
  echo ""

  echo "== fastedit binary =="
  local bin_path=""
  bin_path="$(command -v fastedit 2>/dev/null || true)"
  if [[ -z "$bin_path" ]]; then
    echo "  fastedit on PATH: not found"
    echo "  fork install: NO"
  else
    echo "  fastedit on PATH: $bin_path"
    local ver_out=""
    ver_out="$(run_bounded 10 fastedit --version 2>/dev/null || true)"
    if [[ -n "$ver_out" ]]; then
      echo "  fastedit --version: $ver_out"
    else
      echo "  fastedit --version: (did not answer)"
    fi
    # The fork verbs are the one signature upstream's binary lacks: the same
    # check verify_fork_install runs after a real install.
    local help_out=""
    help_out="$(run_bounded 10 fastedit --help 2>/dev/null || true)"
    local missing=() cmd
    for cmd in create duplicate split join; do
      grep -qw "$cmd" <<<"$help_out" || missing+=("$cmd")
    done
    if [[ ${#missing[@]} -eq 0 ]]; then
      echo "  fork install: YES (create/duplicate/split/join present)"
    else
      echo "  fork install: NO (missing fork subcommands: ${missing[*]})"
    fi
  fi
  echo ""

  echo "== model caches under ${MODELS_DIR} =="
  if [[ ! -d "$MODELS_DIR" ]]; then
    echo "  (no cache directory — nothing cached yet)"
  else
    local d name size state any_cache=0
    for d in "$MODELS_DIR"/*; do
      [[ -d "$d" ]] || continue
      any_cache=1
      name="$(basename "$d")"
      size="$(run_bounded 15 du -sh "$d" 2>/dev/null | awk '{print $1}' || true)"
      state="$(model_cache_state "$d")"
      echo "  ${name}  ${size:-?}  ${state}"
    done
    if [[ "$any_cache" -eq 0 ]]; then
      echo "  (no model directories)"
    fi
  fi
  echo ""

  echo "== grammars =="
  local tool_python=""
  tool_python="$(uv tool dir 2>/dev/null)/${PACKAGE}/bin/python"
  if [[ ! -x "$tool_python" ]]; then
    echo "  no tool python found at ${tool_python} — nothing installed via uv tool"
  elif "$tool_python" -c "from tree_sitter_language_pack import get_language" 2>/dev/null; then
    echo "  present (verified via the tool's own python)"
  else
    echo "  not importable in the tool environment — extended languages (scala, lua, perl, ...) will not resolve"
  fi
  echo ""

  echo "== resource hub =="
  # The global governor's state (read-only): limits + live slot holders.
  # Read through the fastedit module so the report cannot drift from the
  # acquisition logic; a missing/broken install degrades to a note, never
  # a failure (--check installs nothing and must not fail on diagnostics).
  if [[ -x "$tool_python" ]]; then
    if ! "$tool_python" -c "import fastedit.resource_hub" 2>/dev/null; then
      echo "  fastedit.resource_hub not importable in the tool environment — hub section unavailable"
    else
      "$tool_python" - <<'PYHUB'
from fastedit import resource_hub

try:
    limits = resource_hub.load_limits()
except ValueError as e:
    print(f"  limits: MALFORMED ({e})")
else:
    print(
        f"  limits: max_model_instances={limits.max_model_instances} "
        f"max_heavy_jobs={limits.max_heavy_jobs} "
        f"heavy_file_bytes={limits.heavy_file_bytes} "
        f"slot_wait_timeout_s={limits.slot_wait_timeout_s:g}"
    )
    print(f"  hub dir: {resource_hub.hub_dir()}")
try:
    holders = resource_hub.read_hub_state()
except OSError as e:
    print(f"  hub state: unreadable ({e})")
else:
    if holders:
        for h in holders:
            file_note = f" file={h.get('file')}" if h.get("file") else ""
            print(f"  active: {h['kind']}-{h['slot']} pid={h.get('pid')}{file_note}")
    else:
        print("  active holders: none")
PYHUB
    fi
  else
    echo "  (no tool python — run an install first to see hub state)"
  fi
  echo ""

  echo "== local clone =="
  if [[ "$HAVE_LOCAL_TREE" -eq 1 ]]; then
    echo "  local clone: ${REPO_ROOT}"
    local branch=""
    branch="$(run_bounded 10 git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
    if [[ -n "$branch" && "$branch" != "HEAD" ]]; then
      echo "  current branch: ${branch}"
    else
      echo "  current branch: (detached HEAD)"
    fi
  else
    echo "  local clone: none detected (this script is not sitting in a fastedit checkout)"
    echo "  managed clone: ${CLONE_DIR} ($([[ -f "${CLONE_DIR}/pyproject.toml" ]] && echo present || echo absent))"
  fi
  echo ""

  echo "== fork remote (${FORK_URL}) =="
  echo "  default branch on GitHub: main (stale — the landing page shows old code)"
  local heads status=0
  heads="$(run_bounded 20 git ls-remote --heads "$FORK_URL" 2>/dev/null)" || status=$?
  if [[ "$status" -ne 0 ]]; then
    echo "  warning: could not reach the fork remote (offline?) — branch existence not verified." >&2
    echo "  branch '${INSTALLABLE_BRANCH}' exists on the fork remote: unknown (could not reach ${FORK_URL})"
  elif awk -v want="refs/heads/${INSTALLABLE_BRANCH}" '$2 == want { found=1 } END { exit found ? 0 : 1 }' <<<"$heads"; then
    echo "  branch '${INSTALLABLE_BRANCH}' exists on the fork remote"
    echo "  managed clone tracks: ${INSTALLABLE_BRANCH}"
  else
    echo "  error: branch '${INSTALLABLE_BRANCH}' does NOT exist on the fork remote — the one-liners below would fail." >&2
    echo "  managed clone tracks: ${INSTALLABLE_BRANCH} (MISSING)"
  fi
  echo ""

  echo "== install from scratch (no clone needed) =="
  print_from_scratch_one_liners
  echo ""
  print_branch_pin_note
}

# The footer of a SUCCESSFUL real install: the from-scratch one-liners for
# people without a clone, and the verification hint. --check embeds the same
# two commands in its report (print_from_scratch_one_liners is the one
# source); the revert path prints neither — it restores upstream, and fork
# one-liners have no business on a reverted machine.
print_success_footer() {
  echo ""
  echo "Install from scratch on another machine (no clone needed):"
  print_from_scratch_one_liners
  echo ""
  print_branch_pin_note
}

# ---------------------------------------------------------------------------
# Mode resolution, branch autodetect, preflight detection
# ---------------------------------------------------------------------------

# --check exits before ANY mutating or validating surface: no source menu, no
# remote-ref verification abort, no prompts. The ref matters to the report
# only as the name it verifies, so --check resolves it from the same inputs
# the real run would (explicit --ref, else the local clone's branch, else the
# built-in default) without the forward install's autodetect line.
if [[ "$CHECK_ONLY" -eq 1 ]]; then
  if [[ "$REF_SET" -eq 0 && "$HAVE_LOCAL_TREE" -eq 1 ]]; then
    detected_ref="$(run_bounded 10 git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
    if [[ -n "$detected_ref" && "$detected_ref" != "HEAD" ]]; then
      REF="$detected_ref"
    fi
  fi
  check_report
  exit 0
fi

# --dev is now a no-op compatibility flag (every install is editable from a
# clone), so only --revert skips the source resolution below. --ref names
# the branch a MANAGED clone tracks; from inside a checkout the clone's own
# tree is used and --ref has no effect on the install.
if [[ "$REF_SET" -eq 1 && "$HAVE_LOCAL_TREE" -eq 1 ]]; then
  echo "note: --ref has no effect when running from inside a checkout (the working tree is the install source)" >&2
fi

# --dev is a no-op compatibility flag, and the run must SAY so rather than
# silently swallow the flag a user (or a script) typed out of habit.
if [[ "$DEV" -eq 1 ]]; then
  echo "note: --dev is a no-op: every install is now an editable install from a local clone (this checkout, or the managed clone at ${CLONE_DIR})" >&2
fi

resolve_source_mode

# PREFLIGHT DETECTION — printed for real runs and dry runs alike, before
# anything is uninstalled.
detect_installed_fastedits
detect_model_caches

if [[ "$REVERT" -eq 1 ]]; then
  echo "Reverting to upstream ${PACKAGE} from PyPI..."
  handle_model_caches
  sweep_uninstall "$PACKAGE"
  do_install "$PKG_SPEC"
  remove_agent_skill
  remove_managed_clone
  if [[ "$DRY_RUN" -eq 0 ]]; then
    echo "reverted: ${PACKAGE} is now upstream (no create/duplicate/split/join — that's expected)"
  fi
else
  # Resolve the grammar question BEFORE the sweep: the sweep uninstalls the
  # current fastedit, and nobody should be left uninstalled while a prompt
  # waits for an answer. The stale-cache question follows the same rule.
  GRAMMARS_ANSWER=$(ask_all_grammars)
  if [[ "$GRAMMARS_ANSWER" == "yes" ]]; then
    GRAMMARS_SELECTED=1
  else
    GRAMMARS_SELECTED=0
  fi
  handle_model_caches

  echo "editable install from ${INSTALL_SOURCE} (branch ${REF})"
  sweep_uninstall "$PACKAGE"

  # Two-phase, and the order is the whole point. The model is only useful if
  # the backend that loads it is installed, so try the backend extra FIRST --
  # but never let that attempt decide whether this machine ends up with a
  # fastedit. `mlx` is a compiled wheel; resolution or build CAN fail, and the
  # sweep above has already removed the previous install. So a failure falls
  # back to the bare spec (the near-unfailable pure-Python one) and downgrades
  # to deterministic-edit-only, which still works.
  # BACKEND_READY is tracked rather than probed. It records what we INSTALLED,
  # which is not the same as what is IMPORTABLE: a wheel can install and still
  # fail to load (wrong arch under Rosetta, a broken build). So this closes the
  # common case, not the whole class -- if the import fails anyway, the pull is
  # a wasted download, not a corruption.
  #
  # 0 means "we have positive evidence there is no backend", NOT "we don't
  # know". Absence of a mapping is not evidence of an absent backend: on a
  # platform where we never attempted an install we must NOT infer failure,
  # because the user may already have a working runtime we know nothing about.
  # Downgrading their previously-working model pull on a guess would be the
  # same unverified inference we refused to encode in detect_backend_extra.
  # Tri-state, deliberately. BACKEND_ATTEMPTED distinguishes "we tried to
  # install a backend and it failed" from "we never tried, so we know
  # nothing" -- and ONLY the first is grounds for withholding the model.
  BACKEND_ATTEMPTED=0
  BACKEND_READY=0
  BACKEND_EXTRA=""
  INSTALL_EXTRAS=""
  if [[ "$EXTRAS_SET" -eq 0 ]] && BACKEND_EXTRA=$(detect_backend_extra); then
    # `mcp` is NOT a merge backend -- it is the MCP server. On a platform where
    # the auto-selected set is mcp-only (neither mlx nor vllm is installable),
    # the install succeeding tells us nothing about merge capability, so the
    # model pull must stay withheld. Setting READY on a bare `mcp` success would
    # silently restore the original defect: 1.8 GB downloaded with nothing able
    # to load it.
    INSTALL_EXTRAS=$(extras_with_all_grammars "$BACKEND_EXTRA")
    case ",${INSTALL_EXTRAS}," in
      *,mlx,*|*,vllm,*) BACKEND_ATTEMPTED=1 ;;
    esac
    if do_install_optional "$(build_spec "$INSTALL_EXTRAS")"; then
      case ",${INSTALL_EXTRAS}," in
        *,mlx,*|*,vllm,*) BACKEND_READY=1 ;;
      esac
    else
      echo "warning: installing the '${INSTALL_EXTRAS}' extras failed; falling back to a bare install." >&2
      echo "warning: deterministic edits will work; model-merge edits will not, and the model pull is skipped." >&2
        echo "warning: to retry later: uv tool install --force --editable '$(build_spec "$INSTALL_EXTRAS")'" >&2
      do_install "$(build_spec "")"
    fi
  else
    # An explicit --extras is the user's call: honour it exactly and make no
    # backend guess on top of it (the grammar answer is the one addition it
    # can carry, because the user was asked about it directly).
    INSTALL_EXTRAS=$(extras_with_all_grammars "$EXTRAS")
    do_install "$(build_spec "$INSTALL_EXTRAS")"
    case ",${INSTALL_EXTRAS}," in *,mlx,*|*,vllm,*) BACKEND_READY=1 ;; esac
    # `--extras ""` is not ignorance, it is an instruction. The user just told
    # us not to install a backend, so this install provides none -- that is
    # POSITIVE evidence, and pulling 1.7 GB of weights for the backend they
    # declined would be the same "act on an inference" error as the Linux
    # branch above, with the sign flipped. An unmapped PLATFORM teaches us
    # nothing (attempted stays 0); an explicit empty --extras teaches us
    # something (attempted is 1). They are different states and must not
    # collapse into one flag value.
    if [[ "$EXTRAS_SET" -eq 1 && "$BACKEND_READY" -eq 0 ]]; then
      BACKEND_ATTEMPTED=1
    fi
  fi

  if [[ "$DRY_RUN" -eq 0 ]]; then
    verify_fork_install
    verify_grammars
  fi

  # The footer follows the verifications, so the from-scratch one-liners and
  # the version hint print only against a confirmed fork install. Dry runs
  # skip it: nothing was installed, so a success footer would be a lie. (It is
  # a fork-install footer by definition — the revert path must never print it.)
  if [[ "$DRY_RUN" -eq 0 ]]; then
    print_success_footer
  fi

  if [[ "$NO_MODEL" -eq 0 ]]; then
    if [[ "$BACKEND_ATTEMPTED" -eq 1 && "$BACKEND_READY" -eq 0 ]]; then
      # Skip ONLY on positive evidence: we tried to install the backend on this
      # platform and the install failed. Pulling ~1.7 GB that nothing can load
      # is what this script used to do, and it produced a green install whose
      # first model-merge edit raised ModuleNotFoundError.
      #
      # The condition is deliberately NOT `BACKEND_READY -eq 0` alone. That
      # would also catch the case where we never attempted an install (no
      # extra is mapped for this platform), and withhold the model from a
      # machine that may well have a working runtime already -- a downgrade of
      # previously-working behaviour, inferred from our own ignorance.
      # Two different reasons land here and they must not share a message: a
      # FAILED attempt (BACKEND_EXTRA names what we tried) versus the user
      # DECLINING extras. Telling someone who passed --extras "" that an
      # install "failed" sends them debugging a failure that never happened.
      if [[ -n "$BACKEND_EXTRA" ]]; then
        echo "note: the '${BACKEND_EXTRA}' backend failed to install — skipping the model pull (~1.7 GB nothing could load)."
        echo "note: install a backend first, then run 'fastedit pull' — see the warnings above."
      else
        echo "note: no backend extra was installed (--extras \"\") — skipping the model pull (~1.7 GB nothing could load)."
        echo "note: if you already have a backend, run 'fastedit pull' yourself, or re-run without --extras."
      fi
    elif MODEL=$(detect_model); then
      pull_model "$MODEL"
      if [[ "$DRY_RUN" -eq 0 ]]; then
        fastedit doctor || true
      fi
    else
      echo "note: no known model for this platform ($(uname -s)/$(uname -m)) — skipping model pull."
      echo "note: run 'fastedit pull --model mlx-8bit' (Apple Silicon) or 'fastedit pull --model bf16' (Linux GPU) manually if needed."
    fi
  fi

  # The agent skill goes last: the package is in place and the model question
  # is settled, so a tolerated skill failure is reported against an install
  # that already succeeded. The postflight line for it prints right after —
  # real runs only, since the listing is a live npx call and a dry run must
  # stay hermetic.
  install_agent_skill
  if [[ "$DRY_RUN" -eq 0 ]]; then
    verify_agent_skill
  fi
fi
