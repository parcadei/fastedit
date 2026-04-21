"""Inference prompt templates for FastEdit merge model."""

from __future__ import annotations


INFERENCE_SYSTEM_PROMPT = (
    "You are a coding assistant that helps merge code updates, "
    "ensuring every modification is fully integrated. /no_think"
)

INFERENCE_USER_PROMPT = """Merge all changes from the <update> snippet into the <code> below.
- Preserve the code's structure, order, comments, and indentation exactly.
- Output only the updated code, enclosed within <updated-code> and </updated-code> tags.
- Do not include any additional text, explanations, placeholders, ellipses, or code fences.

<code>{original_code}</code>

<update>{update_snippet}</update>

Provide the complete updated code."""
