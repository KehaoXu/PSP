#!/usr/bin/env bash
set -euo pipefail

DTTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[setup-dtts-sd15] diffusion-tts dir: ${DTTS_DIR}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[setup-dtts-sd15] ERROR: python3 not found in PATH."
  exit 1
fi

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  echo "[setup-dtts-sd15] Active conda env: ${CONDA_DEFAULT_ENV:-unknown} (${CONDA_PREFIX})"
else
  echo "[setup-dtts-sd15] WARNING: no active conda env detected."
  echo "[setup-dtts-sd15]          Recommended: conda activate diffusion-tts"
fi

export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

echo "[setup-dtts-sd15] Repairing pip/setuptools toolchain for ImageReward build compatibility"
python3 -m pip install --force-reinstall \
  "pip==24.3.1" \
  "setuptools==75.8.0" \
  "wheel==0.45.1" \
  "backports.tarfile"

echo "[setup-dtts-sd15] Installing runtime dependencies (IR-only, no HPS)"
python3 -m pip install --no-build-isolation \
  "torch==2.4.0" \
  "torchvision==0.19.0" \
  "accelerate==1.2.1" \
  "diffusers==0.36.0" \
  "transformers==4.38.2" \
  "tokenizers==0.15.2" \
  "numpy==1.26.3" \
  "tqdm==4.66.4" \
  "packaging" \
  "safetensors==0.5.2" \
  "pillow==11.1.0" \
  "sentencepiece==0.2.0" \
  "typing_extensions" \
  "google-genai" \
  "git+https://github.com/openai/CLIP.git"

echo "[setup-dtts-sd15] Installing ImageReward from source (required)"
python3 -m pip install --no-build-isolation "git+https://github.com/THUDM/ImageReward.git"

echo "[setup-dtts-sd15] Validating imports"
cd "${DTTS_DIR}"
python3 - <<'PY'
import importlib
mods = [
    "torch",
    "torchvision",
    "accelerate",
    "diffusers",
    "transformers",
    "tokenizers",
    "numpy",
    "tqdm",
    "safetensors",
    "ImageReward",
]
for m in mods:
    importlib.import_module(m)

import run_multi_backbone_eps_greedy_geneval as runner  # noqa: F401
print("[setup-dtts-sd15] import check OK")
PY

DTTS_PREFETCH="${DTTS_PREFETCH:-1}"
if [[ "${DTTS_PREFETCH}" == "1" ]]; then
  echo "[setup-dtts-sd15] Prefetching SD1.5 + ImageReward weights"
  python3 - <<'PY'
from huggingface_hub import snapshot_download
import ImageReward as RM

print("[setup-dtts-sd15] downloading runwayml/stable-diffusion-v1-5")
snapshot_download(repo_id="runwayml/stable-diffusion-v1-5", resume_download=True)
print("[setup-dtts-sd15] downloading ImageReward-v1.0")
RM.load("ImageReward-v1.0")
print("[setup-dtts-sd15] prefetch complete")
PY
else
  echo "[setup-dtts-sd15] Skipping prefetch (DTTS_PREFETCH=${DTTS_PREFETCH})"
fi

echo "[setup-dtts-sd15] Done."
