"""Auto-download models from HuggingFace on first use.

Standard model cache: ~/.cache/fastedit/models/
Override: FASTEDIT_MODEL_PATH environment variable.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("fastedit.model_download")

# HuggingFace repo — single repo, multiple subfolders
HF_REPO = "continuous-lab/FastEdit"

MODELS = {
    "mlx-8bit": "mlx-8bit",
    "bf16": "bf16",
}

DEFAULT_MODEL = "mlx-8bit"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "fastedit" / "models"


def get_model_path(
    model_name: str | None = None,
    cache_dir: str | None = None,
) -> str:
    """Resolve model path: env var → cache dir → auto-download.

    Resolution order:
    1. FASTEDIT_MODEL_PATH env var (explicit override)
    2. Local ./models/<name> directory (development)
    3. Cache dir ~/.cache/fastedit/models/<name> (already downloaded)
    4. Auto-download from HuggingFace → cache dir

    Args:
        model_name: Model name (default: fastedit-1.7b-mlx-8bit)
        cache_dir: Override cache directory

    Returns:
        Absolute path to the model directory.

    Raises:
        RuntimeError: If download fails and model is not available.
    """
    name = model_name or DEFAULT_MODEL

    # 1. Explicit env var
    env_path = os.environ.get("FASTEDIT_MODEL_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path

    # 2. Local ./models/ directory (for development)
    # Check both new subfolder name and legacy full name
    for local_name in [name, f"fastedit-1.7b-{name}"]:
        local = Path("models") / local_name
        if local.is_dir():
            return str(local.resolve())

    # 3. Cache directory (already downloaded)
    cache = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    cached_path = cache / name
    if cached_path.is_dir() and any(cached_path.iterdir()):
        return str(cached_path)

    # 4. Auto-download subfolder from HuggingFace
    subfolder = MODELS.get(name)
    if not subfolder:
        raise RuntimeError(
            f"Unknown model '{name}'. Available: {list(MODELS.keys())}"
        )

    return _download_model(subfolder, cached_path)


def _download_model(subfolder: str, target_dir: Path) -> str:
    """Download a model subfolder from the HuggingFace repo."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise RuntimeError(
            "Model not found locally. To auto-download, install huggingface_hub:\n"
            "  pip install huggingface-hub\n"
            "  fastedit pull\n\n"
            "Or download manually:\n"
            f"  huggingface-cli download {HF_REPO} --include '{subfolder}/*'"
        )

    logger.info("Downloading %s/%s → %s", HF_REPO, subfolder, target_dir)
    print(f"Downloading {HF_REPO} ({subfolder})...", file=sys.stderr)
    print("  (one-time download)", file=sys.stderr)

    target_dir.parent.mkdir(parents=True, exist_ok=True)

    downloaded = snapshot_download(
        repo_id=HF_REPO,
        allow_patterns=[f"{subfolder}/*"],
        local_dir=str(target_dir.parent),
    )

    # snapshot_download puts files in <local_dir>/<subfolder>/
    result = Path(downloaded) / subfolder
    if not result.is_dir():
        raise RuntimeError(f"Download succeeded but {result} not found")

    print(f"  Done: {result}", file=sys.stderr)
    logger.info("Download complete: %s", result)
    return str(result)
