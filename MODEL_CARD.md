---
language:
  - en
license: mit
library_name: transformers
base_model: Qwen/Qwen2.5-Coder-1.5B-Instruct
tags:
  - code
  - code-editing
  - merge
  - fastedit
  - qwen2
pipeline_tag: text-generation
---

# FastEdit 1.7B

A fine-tuned **Qwen2.5-Coder-1.5B-Instruct** for merging code edit snippets into source files. Given an original code chunk (~35 lines) and a compact edit snippet with context markers, the model produces the merged result.

This model is designed to be used with the [FastEdit](https://github.com/parcadei/fastedit) toolkit, which handles AST scoping, deterministic edits, and post-processing. **Using the model directly requires the exact prompt format described below.**

## Model variants

All variants are in this repo under subfolders:

| Subfolder | Format | Size | Use case |
|-----------|--------|------|----------|
| `bf16/` | BF16 safetensors | 3.2 GB | Fine-tuning, reference, GPU serving via vLLM/TGI |
| `mlx-8bit/` | MLX 8-bit | 1.7 GB | Apple Silicon (recommended for local use) |
| `gguf/` | GGUF Q8_0 | 1.7 GB | llama.cpp, LM Studio, Ollama |

## Prompt format

The model expects a specific 2-message chat format. **Using a different prompt will produce poor results.**

### System message

```
You are a coding assistant that helps merge code updates, ensuring every modification is fully integrated. /no_think
```

The `/no_think` suffix disables Qwen's thinking mode — without it, the model may emit thousands of reasoning tokens before producing output.

### User message

```
Merge all changes from the <update> snippet into the <code> below.
- Preserve the code's structure, order, comments, and indentation exactly.
- Output only the updated code, enclosed within <updated-code> and </updated-code> tags.
- Do not include any additional text, explanations, placeholders, ellipses, or code fences.

<code>{original_code}</code>

<update>{update_snippet}</update>

Provide the complete updated code.
```

### Expected output

The model outputs the merged code wrapped in `<updated-code>` tags:

```
<updated-code>
def process(data):
    try:
        result = transform(data)
        return result
    except Error as e:
        return {"error": str(e)}
</updated-code>
```

### Complete example

**Original code** (what tree-sitter extracts for the target function):

```python
def process(data):
    result = transform(data)
    return result
```

**Edit snippet** (what the user/agent writes):

```python
def process(data):
    try:
        # ... existing code ...
    except Error as e:
        return {"error": str(e)}
```

**Model output:**

```python
<updated-code>
def process(data):
    try:
        result = transform(data)
        return result
    except Error as e:
        return {"error": str(e)}
</updated-code>
```

The model understands `# ... existing code ...` markers (and language-specific variants like `// ... existing code ...`) as instructions to preserve the original lines in that region.

## How it fits into FastEdit

In production, the model is the **fallback** — not the primary path:

1. **AST scoping** — tree-sitter finds the target function by name (~35 lines), so the model never sees the whole file
2. **Deterministic text-match** �� 74% of edits are resolved by matching context lines and splicing in new lines (0 tokens, <1ms)
3. **Model merge** — the remaining 26% of edits (structural changes like wrapping in try/catch, full rewrites) go to this model

The model only ever processes ~35-line chunks. It was trained on function-scoped edits, not whole files. Feeding it large inputs will degrade quality.

## Using without FastEdit

If you want to use the model directly (without the toolkit), you need to:

1. **Scope the input yourself** — extract only the target function/class, not the whole file
2. **Use the exact prompt format** above — different prompts will produce different (worse) results
3. **Parse the output** — extract text between `<updated-code>` and `</updated-code>` tags
4. **Handle edge cases** — the model may emit `<think>` blocks (strip them), use variant tag names (`<update-code>`, `<updated_code>`), or truncate output on long functions

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# BF16 (GPU / fine-tuning)
model = AutoModelForCausalLM.from_pretrained("continuous-lab/FastEdit", subfolder="bf16", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("continuous-lab/FastEdit", subfolder="bf16")

messages = [
    {"role": "system", "content": "You are a coding assistant that helps merge code updates, ensuring every modification is fully integrated. /no_think"},
    {"role": "user", "content": """Merge all changes from the <update> snippet into the <code> below.
- Preserve the code's structure, order, comments, and indentation exactly.
- Output only the updated code, enclosed within <updated-code> and </updated-code> tags.
- Do not include any additional text, explanations, placeholders, ellipses, or code fences.

<code>def process(data):
    result = transform(data)
    return result</code>

<update>def process(data):
    try:
        # ... existing code ...
    except Error as e:
        return {"error": str(e)}</update>

Provide the complete updated code."""}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0)
result = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
# Parse: extract text between <updated-code> and </updated-code>
```

## Training

- **Base model**: Qwen2.5-Coder-1.5B-Instruct
- **Task**: Code edit merging across 13 languages

## Evaluation

Tested on 22 structurally distinct edit patterns (73 cases) across 13 languages:

| Path | Accuracy | Avg tokens | Avg latency |
|------|----------|------------|-------------|
| Deterministic (74% of edits) | 100% | 0 | <1ms |
| Model (26% of edits) | 92% | ~40 | ~500ms |
| **Combined** | **~98%** | **~10** | **~130ms** |

Per-language model accuracy (156-example benchmark):

| Language | Accuracy |
|----------|----------|
| Python, Java, Kotlin, C, PHP | 92% |
| JavaScript, TypeScript, Rust, Swift | 85% |
| Go, C++, Ruby | 77% |

## Limitations

- Performance degrades on inputs longer than ~100 lines.
- Does not handle whole-file edits well — use the FastEdit toolkit's AST scoping.
- The edit snippet must use `# ... existing code ...` markers (or language-equivalent) for context preservation. Without markers, the model treats the entire snippet as a replacement.
- Languages not in the training set may work but are untested.

## License

MIT
