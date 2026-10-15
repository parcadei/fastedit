"""Single source of truth for the ``fastedit --help`` guide.

This module holds the guide as structured data (``SECTIONS``, ``COMMANDS``) and
renders it into the ``EPILOG`` string that ``fastedit.cli`` passes to argparse.
Tests in ``tests/test_cli_help.py`` guarantee that every example command parses
against the real CLI parser (no invented flags) and that the troubleshooting
entries quote error strings that really exist in ``src/``.

Keep every rendered line at 100 characters or fewer — the data below is
hand-wrapped, and the tests enforce the width.
"""

from __future__ import annotations

import shlex
from typing import NamedTuple


class Example(NamedTuple):
    """One example command: ``argv`` is the parse-checked truth, ``explanation``
    is rendered below the command line. ``display`` is ``argv`` shell-quoted on
    one line (embedded newlines shown as ``\\n``) so the guide can never drift
    from what the parser actually accepts."""

    argv: tuple[str, ...]
    explanation: tuple[str, ...]

    @property
    def display(self) -> str:
        return shlex.join(list(self.argv)).replace("\n", "\\n")


class Command(NamedTuple):
    name: str
    purpose: str
    examples: tuple[Example, ...]


class Section(NamedTuple):
    title: str
    lines: tuple[str, ...]


# ---------------------------------------------------------------------------
# COMMANDS — one entry per real subcommand; examples use only real flags.
# ---------------------------------------------------------------------------

COMMANDS: tuple[Command, ...] = (
    Command(
        "read",
        "Show file structure: full content (small files) or a symbol map with line ranges (large).",
        (
            Example(
                ("read", "app.py"),
                (
                    "Files <=100 lines print in full; larger files print imports and",
                    "definitions with line ranges (L12-40). Uses tldr for the structure",
                    "pass. No model, no writes.",
                ),
            ),
        ),
    ),
    Command(
        "search",
        "Search the codebase for symbols, references, and regex or hybrid matches.",
        (
            Example(
                ("search", "handle_request", "src/", "--mode", "references", "--top-k", "20"),
                (
                    "AST-verified references via tldr. --mode: search (default), regex,",
                    "hybrid (+ --regex-filter), references. --top-k caps results (default 10).",
                ),
            ),
            Example(
                ("search", "def .*retry", "--mode", "regex"),
                ("The path argument is optional and defaults to '.'.",),
            ),
        ),
    ),
    Command(
        "diff",
        "Unified diff between a pre-edit backup and the file on disk. Base: the NEWEST",
        (
            Example(
                ("diff", "app.py"),
                (
                    "Read-only; safe to run anytime; never consumes the undo history.",
                    "'No backup recorded' just means no fastedit edit ran on that file yet.",
                    "Default base (--base last) = the NEWEST backup, i.e. exactly the last",
                    "edit's change. --base all diffs against the OLDEST backup — every",
                    "change still in the undo history; use it to detect losses across",
                    "multiple edits.",
                ),
            ),
        ),
    ),
    Command(
        "edit",
        "Apply one edit snippet to a file: insert after a symbol or replace one.",
        (
            Example(
                ("edit", "app.py", "--after", "main", "--snippet", 'def health_check(): return "ok"'),
                ("--after inserts the snippet verbatim below the named symbol: 0 tokens, instant.",),
            ),
            Example(
                ("edit", "app.py", "--replace", "process", "--snippet", "def process(data): return transform(data)"),
                (
                    "--replace splices the snippet over the symbol's span, signature",
                    "auto-preserved. Multi-line snippets with '# ... existing code ...' keep",
                    "the rest of the body; '--snippet @f.py' reads it from a file, '-' from",
                    "piped stdin; '--snippet-is-literal' disables that resolution.",
                ),
            ),
            Example(
                ("edit", "app.py", "--snippet", "-", "--backend", "vllm", "--api-base", "http://localhost:8000/v1"),
                (
                    "Backend overrides for a server install: --backend, --model-path,",
                    "--api-base, --api-model (meanings under PARAMETERS).",
                ),
            ),
            Example(
                ("edit", "app.rs", "--snippet", "    audit_trail.record(x);", "--insert-after", "let x = audit(a);"),
                (
                    "Hermetic insert: splices the snippet after the UNIQUE line whose full",
                    "stripped content equals the literal (0 matches / >1 matches refuse, exit",
                    "1). --insert-before mirrors it. --lines FROM:TO --delete removes an",
                    "inclusive 1-indexed line range. All three: 0 tokens, no model.",
                ),
            ),
        ),
    ),
    Command(
        "batch-edit",
        "Apply several sequential edits to one file in a single merge pass.",
        (
            Example(
                ("batch-edit", "app.py", "--edits", '[{"snippet": "import redis", "after": "import json"}]'),
                (
                    'Ordered edits to ONE file; each item is {"snippet", "after" | "replace"}',
                    "and each sees the previous edit's result.",
                ),
            ),
            Example(
                ("batch-edit", "app.py", "--edits", "-"),
                ("'-' reads the JSON list from stdin.",),
            ),
        ),
    ),
    Command(
        "multi-edit",
        "Apply edits across multiple files; all-or-nothing: nothing written on any failure.",
        (
            Example(
                ("multi-edit", "--file-edits", '[{"file_path":"a.py","edits":[{"snippet":"s","after":"m"}]}]'),
                (
                    'Each item: {"file_path", "edits": [{"snippet", "after" | "replace"}, ...]}.',
                    "All-or-nothing: every target is validated and merged before ANY file is",
                    "written; one failure (or a file changed since read) and nothing is written.",
                ),
            ),
            Example(
                ("multi-edit", "--file-edits", "-"),
                ("'-' reads the JSON list from stdin.",),
            ),
        ),
    ),
    Command(
        "delete",
        "Remove a function, method or class by name (AST-scoped, no model).",
        (
            Example(
                ("delete", "app.py", "legacy_handler"),
                (
                    'Removes a function, method ("MyClass.method") or class by AST. Refuses',
                    "while other files still reference the symbol.",
                ),
            ),
            Example(
                ("delete", "app.py", "legacy_handler", "--force"),
                ("--force skips the cross-file caller check; parse validation still applies.",),
            ),
        ),
    ),
    Command(
        "move",
        "Reorder a symbol within a file, right after another symbol.",
        (
            Example(
                ("move", "app.py", "helper_func", "--after", "main"),
                ("Relocates the symbol's lines below the target. Same file only; no model.",),
            ),
        ),
    ),
    Command(
        "rename",
        "Rename every AST-verified code reference in one file; strings/comments untouched.",
        (
            Example(
                ("rename", "app.py", "old_name", "new_name", "--dry-run"),
                (
                    "Matches only real code references (tldr, AST-verified): strings,",
                    "comments and docstrings are untouched. --dry-run previews without writing.",
                ),
            ),
            Example(
                ("rename", "app.py", "old_name", "new_name"),
                ("Prints a unified diff of what changed.",),
            ),
        ),
    ),
    Command(
        "rename-all",
        "Rename a symbol across every supported file in a directory tree.",
        (
            Example(
                ("rename-all", "src/", "old_name", "new_name"),
                (
                    "Walks the directory (vendor/build dirs skipped) and rewrites every file",
                    "that references the symbol.",
                ),
            ),
            Example(
                ("rename-all", "src/", "old_name", "new_name", "--only", "function"),
                ("--only narrows to definition kind: class | function | method | variable.",),
            ),
        ),
    ),
    Command(
        "move-to-file",
        "Move a symbol to another file and rewrite every consumer's imports.",
        (
            Example(
                ("move-to-file", "cache_get", "app.py", "utils.py", "--dry-run"),
                (
                    "Relocates a symbol to another file and rewrites the imports of every",
                    "consumer it finds. The destination must exist and share the language.",
                    "--dry-run prints the plan without writing.",
                ),
            ),
            Example(
                ("move-to-file", "cache_get", "app.py", "utils.py", "--after", "load_cache"),
                ("--after places the symbol below a named symbol in the destination.",),
            ),
        ),
    ),
    Command(
        "create",
        "Create a new text file from --content, --content-file or stdin.",
        (
            Example(
                ("create", "app.py", "--content", "def created(): return True"),
                (
                    "Refuses to overwrite an existing file (--force), refuses missing parents",
                    "(--parents), refuses binary content. With neither content flag, reads stdin.",
                ),
            ),
            Example(
                ("create", "docs/notes.md", "--content-file", "notes.md", "--parents"),
                (
                    "--content-file takes a path, or '-' for piped stdin. '--content @f.py'",
                    "reads content from a file; '--content-is-literal' uses the text verbatim.",
                ),
            ),
        ),
    ),
    Command(
        "duplicate",
        "Copy a file byte-for-byte to a new path (binary-safe).",
        (
            Example(
                ("duplicate", "app.py", "app.py.bak", "--force"),
                (
                    "A raw byte copy — binary files duplicate cleanly. --force overwrites an",
                    "existing destination; --parents creates missing parent directories.",
                ),
            ),
        ),
    ),
    Command(
        "split",
        "Split a file into parts under --out, format-aware where possible.",
        (
            Example(
                ("split", "data.csv", "--out", "parts/", "--rows", "500"),
                (
                    "--rows (csv/tsv; header repeated on every part), --by heading (markdown;",
                    "--level, default 2), --by element (json array / xml / html), --lines N",
                    "(any text file). The modes are mutually exclusive; one is required.",
                    "A manifest is written alongside the parts.",
                ),
            ),
            Example(
                ("split", "README.md", "--out", "parts/", "--by", "heading", "--level", "2"),
                ("XML/HTML element splits are lossy and read-only: there is no join for them.",),
            ),
        ),
    ),
    Command(
        "join",
        "Join split parts back into one file (inverts split).",
        (
            Example(
                ("join", "parts/", "-o", "data.csv"),
                ("Inverts split via the directory's manifest (parts re-joined in split order).",),
            ),
            Example(
                ("join", "parts/part-0000.csv", "parts/part-0001.csv", "-o", "data.csv"),
                ("Or list explicit part files in order. -o/--out is required.",),
            ),
        ),
    ),
    Command(
        "undo",
        "Revert the file's last edit from its backup.",
        (
            Example(
                ("undo", "app.py"),
                (
                    "Pops the newest backup, restores it byte-for-byte and prints the reverted",
                    "diff. Repeat to step back further — bounded by the 5-deep/24h backup window.",
                ),
            ),
        ),
    ),
    Command(
        "init",
        "One-shot setup: install the fastedit agent skill for your coding agent.",
        (
            Example(
                ("init",),
                (
                    "Stages the skill that SHIPS INSIDE this fastedit package into a",
                    "temp dir and hands that directory to the skills CLI (npx) — no",
                    "GitHub source and no branch to resolve, so the skill an agent",
                    "reads always matches the installed fastedit. Prints next-step",
                    "guidance; --skill-agent targets another agent.",
                ),
            ),
        ),
    ),
    Command(
        "pull",
        "Download the merge model from HuggingFace (~3 GB, one-time, cached).",
        (
            Example(
                ("pull", "--model", "mlx-8bit"),
                (
                    "Downloads the ~3 GB merge model into the local cache, one-time. mlx-8bit",
                    "for Apple Silicon (MLX); bf16 for Linux/GPU (vLLM).",
                ),
            ),
        ),
    ),
    Command(
        "doctor",
        "Run self-diagnostics: binaries, extras, model cache, MCP config, tldr.",
        (
            Example(
                ("doctor",),
                (
                    "Checks binaries, Python version, backend extras, model cache state, MCP",
                    "config and tldr. Run this first when anything breaks.",
                ),
            ),
        ),
    ),
    Command(
        "mcp-install",
        "Write the fastedit MCP entry into Claude Code's config.",
        (
            Example(
                ("mcp-install",),
                (
                    "Writes the MCP entry for Claude Code at user scope (~/.claude.json).",
                    "Idempotent; backs up existing config before modifying.",
                ),
            ),
            Example(
                ("mcp-install", "--scope", "project"),
                ("--scope project writes ./.mcp.json instead.",),
            ),
        ),
    ),
)


def _commands_section() -> Section:
    lines: list[str] = []
    for command in COMMANDS:
        lines.append(f"  {command.name} — {command.purpose}")
        for example in command.examples:
            lines.append(f"    {example.display}")
            for expl in example.explanation:
                lines.append(f"    {expl}")
    return Section("COMMANDS", tuple(lines))


# ---------------------------------------------------------------------------
# Prose sections. All hand-wrapped to <=100 characters per rendered line.
# ---------------------------------------------------------------------------

_QUICKSTART = Section(
    "QUICKSTART",
    (
        "  1. Install the fork: scripts/install-dev.sh (venv, extras, model; see its --help).",
        "     Or from PyPI: uv tool install 'fastedits[mlx,mcp]'.",
        "  2. Get the merge model (one-time, ~3 GB, cached):",
        "       fastedit pull --model mlx-8bit     (Apple Silicon)",
        "       fastedit pull --model bf16         (Linux / GPU)",
        "  3. Check the install: fastedit doctor (binaries, extras, model cache, MCP, tldr).",
        "  4. First edit, then verify — and be able to step back:",
        "       fastedit edit app.py --replace process --snippet 'def process(data): return data'",
        "       fastedit diff app.py      (see what changed)",
        "       fastedit undo app.py      (step back if unsure)",
    ),
)

_EDIT_MODES = Section(
    "THE THREE EDIT MODES",
    (
        "  Every edit targets a symbol fastedit finds by name with tree-sitter — you write only",
        "  the change, never the old code.",
        "    --after SYMBOL              Pure text insertion below the symbol. 0 model tokens.",
        "    --replace SYMBOL (determin)",
        "                                Snippet lines matching the original act as context",
        "                                anchors; new lines are spliced between them. 0 tokens.",
        "    --replace SYMBOL (model)    When anchors cannot resolve the edit, a 1.7B merge",
        "                                model rewrites just the ~35-line chunk around the",
        "                                symbol. ~40 tokens, <1s.",
        "  Keep-markers: '# ... existing code ...' (short form '#...'; '// ... existing code ...'",
        "  in C-family) means \"keep the untouched lines that belong here\". A marker is honored",
        "  only when an anchor line in the snippet matches the original body: markers must",
        "  match real lines, and a marker with no matching anchor is refused, never guessed.",
        "  Validation & retries: every merged result is parse-checked relative to the original",
        "  and content-checked byte-exact for untouched lines. A rejected attempt is retried",
        "  with its failure reason appended (FASTEDIT_MAX_RETRIES, default 8); on exhaustion",
        "  the edit is refused and the file is left unchanged.",
    ),
)

_PARAMETERS = Section(
    "PARAMETERS",
    (
        "  --snippet TEXT ('-' = stdin)   The edit itself. Lines matching the original are",
        "                                 context anchors; new lines are the change;",
        "                                 '# ... existing code ...' keeps untouched lines.",
        "  --after SYMBOL                 Insert the snippet below this symbol, verbatim.",
        "                                 0 tokens; cannot fail on snippet shape.",
        "  --replace SYMBOL               Target the symbol's span: signature auto-preserved.",
        "                                 A wholesale swap needs the snippet's own definition",
        "                                 line; a partial rewrite needs matching anchors.",
        "  --insert-after LITERAL         Hermetic insert next to the UNIQUE line whose full",
        "                                 stripped content equals LITERAL (--insert-before",
        "                                 mirrors it). 0 tokens, no model; 0 or >1 matches",
        "                                 refuse with exit 1.",
        "  --lines FROM:TO --delete       Delete the inclusive 1-indexed line range. Pure",
        "                                 deletion, no model; --delete is required.",
        "  --backend {mlx,vllm}           Inference backend; overrides FASTEDIT_BACKEND.",
        "                                 mlx = local Apple Silicon; vllm = OpenAI-compatible",
        "                                 server.",
        "  --model-path DIR               MLX model directory; overrides FASTEDIT_MODEL_PATH.",
        "                                 Defaults to the 'fastedit pull' cache.",
        "  --api-base URL                 vLLM endpoint (env FASTEDIT_VLLM_API_BASE).",
        "  --api-model NAME               vLLM served model name (env FASTEDIT_VLLM_MODEL).",
        "  --force (delete)               Skip the cross-file caller-safety check (on",
        "                                 create/duplicate: overwrite the target). Never",
        "                                 bypasses parse validation or the concurrent-",
        "                                 instance file lock.",
        "  Per command: split's --by/--lines/--rows are mutually exclusive (one required);",
        "  join -o is the output file; search --top-k caps results (default 10); rename-all",
        "  --only takes class | function | method | variable; mcp-install --scope is",
        "  user | project (default user).",
    ),
)

_LIMITS = Section(
    "LIMITS",
    (
        "  Model context is ~40960 tokens, but the model only ever sees a ~35-line chunk",
        "  around the edit; larger symbols are chunked below that limit.",
        "  Files >150 lines without a usable AST refuse whole-file merges: give a snippet",
        "  line that matches the file (an anchor) so a ~40-line window can be located.",
        "  Backups: 5 per file, pruned after 24h, under ~/.fastedit/backups — undo depth is",
        "  bounded by both.",
        "  Latency, realistically: deterministic edits are 0 tokens and sub-millisecond;",
        "  model merges are ~40 tokens (<1s on Apple Silicon/GPU, ~50 tok/s).",
        "  Concurrency: one fastedit process per file — a second exits 1 naming the holder;",
        "  the lock is a kernel flock, so it self-releases if the holder crashes.",
    ),
)

_LIMITATIONS = Section(
    "LIMITATIONS",
    (
        "  Markers must match real lines: '# ... existing code ...' only works where an",
        "  anchor line matches the original; unmatched markers are refused loudly, never",
        "  guessed.",
        "  Garbage in, garbage out: fastedit validates and edits, it never corrects your",
        "  source. Pre-existing defects are preserved byte-exact; an edit that introduces",
        "  new parse errors is refused.",
        "  Unsupported extensions refuse symbol-targeted edits (no grammar = no AST to",
        "  locate the symbol); create, duplicate, split and join accept any text file.",
        "  Plain-text (non-code) files need anchors — snippet lines that match the file —",
        "  for every edit.",
        "  Not a formatter or linter: no style normalization, no auto-fixes, no import",
        "  sorting.",
        "  Not a cross-file transaction: multi-edit validates everything before writing,",
        "  but a crash mid-write can leave earlier targets written.",
    ),
)

_BEST_PRACTICES = Section(
    "BEST PRACTICES",
    (
        "  Write small snippets with one or two real context lines — anchors let the",
        "  deterministic path resolve the edit at 0 tokens.",
        "  Use '# ... existing code ...' for partial rewrites instead of restating the",
        "  whole body.",
        "  Read before writing: 'fastedit read' shows the exact symbol names and ranges.",
        "  Preview first: --dry-run (rename, rename-all, move-to-file), and 'fastedit",
        "  diff' afterwards. Default base is the last edit's change; after SEVERAL",
        "  edits run 'fastedit diff --base all' — it diffs against the oldest backup",
        "  so a loss from an earlier edit stays visible (loss detection).",
        "  Run 'fastedit undo' right after any edit you are unsure about — backups are",
        "  5-deep/24h.",
        "  Treat --force as a deliberate decision: it skips the delete caller check, not",
        "  validation.",
        "  Start with 'fastedit doctor' when anything misbehaves.",
    ),
)

_TROUBLESHOOTING = Section(
    "TROUBLESHOOTING",
    (
        "  \"another fastedit instance (pid N, running Xs) is editing <path>\" — a second",
        "  fastedit holds the file's lock. Wait for it to finish; the lock self-releases",
        "  if that process dies, then re-run.",
        "  \"file changed on disk since it was read\" — another (non-fastedit) writer",
        "  modified the file between read and write. Nothing was written; re-run against",
        "  the current content.",
        "  \"parse errors; refusing to write\" — your edit introduced new syntax errors.",
        "  Fix the snippet's syntax and retry; the file is unchanged.",
        "  \"Symbol 'x' not found\" — the target name does not exist in the file. The",
        "  error lists available symbols; 'fastedit read' shows the same map.",
        "  Marker refusals (\"contains a keep-marker but the edit cannot be",
        "  placed deterministically\", \"has no definition\") — the refusal names",
        "  the fix: include more surrounding unique lines, or pass the full",
        "  replacement including the definition line, or use --after to insert.",
        "  \"literal not found: ...\" / \"literal matches N lines\" — the --insert-",
        "  after/--insert-before anchor must equal one line's full stripped content.",
        "  Verify what an edit changed with 'fastedit diff' (base: the newest backup =",
        "  the last edit); after SEVERAL edits use 'fastedit diff --base all' — every",
        "  change still in the undo history, for loss detection across edits.",
        "  \"Model not found locally\" — run 'fastedit pull --model mlx-8bit' (Apple",
        "  Silicon) or 'fastedit pull --model bf16' (Linux/GPU).",
        "  Anything else: run 'fastedit doctor'.",
    ),
)

_INSTALL_UPDATE = Section(
    "INSTALL & UPDATE",
    (
        "  Install this fork: scripts/install-dev.sh (fork install, repair, revert — see",
        "  its --help). Dev install of your working tree: scripts/install-dev.sh --dev.",
        "  Back to upstream PyPI: scripts/install-dev.sh --revert.",
        "  Install from PyPI (upstream): uv tool install 'fastedits[mlx,mcp]' on Apple",
        "  Silicon, 'fastedits[vllm,mcp]' on Linux GPU, 'fastedits[mcp]' for an external",
        "  server only.",
        "  Model: fastedit pull --model mlx-8bit or fastedit pull --model bf16 (~3 GB,",
        "  one-time).",
        "  Grammars: the installer asks once about the offline 173-language tree-sitter",
        "  pack (Enter = yes; --all-grammars no to skip).",
        "  Agents (MCP): fastedit mcp-install — user scope, or fastedit mcp-install",
        "  --scope project.",
        "  Agent skill: fastedit init installs the agent skill (npx skills add, global,",
        "  claude-code by default; --skill-agent targets another agent).",
        "  Verify or repair: fastedit doctor. Upgrade: uv tool upgrade fastedits, or",
        "  re-run scripts/install-dev.sh.",
    ),
)

SECTIONS: tuple[Section, ...] = (
    _QUICKSTART,
    _EDIT_MODES,
    _commands_section(),
    _PARAMETERS,
    _LIMITS,
    _LIMITATIONS,
    _BEST_PRACTICES,
    _TROUBLESHOOTING,
    _INSTALL_UPDATE,
)

SECTION_HEADERS: tuple[str, ...] = tuple(section.title for section in SECTIONS)


def render_epilog(sections: tuple[Section, ...]) -> str:
    """Render the sections into the stable epilog string (one blank line between
    sections, trailing newline)."""
    return "\n\n".join(section.title + "\n" + "\n".join(section.lines) for section in sections) + "\n"


EPILOG = render_epilog(SECTIONS)
