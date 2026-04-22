# FastEdit

## CRITICAL: No Hardcoding — Universal Solutions Only

**Do NOT hardcode to solve problems.** When something doesn't work for a specific framework, library, or spec:

1. **Find the root cause** — why does it fail? What assumption is broken?
2. **Design a general, declarative solution** — push the variance into the YAML spec, not into `if/elif` branches in Python code.
3. **Never add framework-specific branches** — no `if style == "routable"`, no `if is_async`, no `if pkg == "bottle"`. If you catch yourself writing a condition that checks for a specific framework or style, STOP. The YAML spec should declare the behavior, and the pipeline should consume it generically.

**Examples of what NOT to do:**
- Adding `if is_routable: return "json.dumps(...)"` to TODO composition — instead, add a `response_wrapper` field to the YAML context
- Adding `if ctx.test_async: prefix = "async "` in skeleton.py — instead, let the YAML declare `handler_prefix: "async"` and render it generically
- Adding `not has_routes` guards to exclude entity params — instead, the YAML declares which params the handler receives

**The test:** If adding support for a new framework requires changing Python code (not just writing a new YAML spec), the design is wrong.

## Build & Test Commands

```bash
# Build
uv run python -m app

# Test
uv run pytest

# Lint
uv run ruff check .
```

## Architecture

Language: python

## Conventions

- Follow existing code patterns
- Write tests before implementation
- Run lint before committing
- When the user asks for the "entire corpus of worker-relevant Python" or an equivalent complete curriculum/taxonomy, do not stop after reporting gaps. Keep patching the taxonomy, generators, and validators in the same turn until the major worker-relevant areas are covered well enough to act on, then summarize only the residual low-priority gaps.
- When the user's intent is clear and scoped, answer that scoped question directly.
- Do not broaden a narrow question into a wider or more literal question unless the user explicitly asks for the wider framing.
- If there is a choice between a defensive interpretation and the user's clearly intended interpretation, prefer the user's intended interpretation.
- Do not substitute semantic hedging for answering the actual question.
- If a stricter or broader reading would change the answer, mention it only after answering the user's intended question first.
- When the user asks why one specific script or file behaves differently from another, read those named files first and answer from the code before running experiments.
- Do not substitute benchmarking, timing runs, or broader investigation when the user asked for a scoped file comparison unless the user explicitly asks for measurement.
