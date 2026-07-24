#!/usr/bin/env bash
set -euo pipefail

DTTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${DTTS_DIR}/../.." && pwd)"
SD35_REQS="${REPO_ROOT}/requirements_sd3.5.txt"

echo "[setup-dtts-sd35] diffusion-tts dir: ${DTTS_DIR}"
echo "[setup-dtts-sd35] repo root: ${REPO_ROOT}"

if [[ ! -f "${SD35_REQS}" ]]; then
  echo "[setup-dtts-sd35] ERROR: ${SD35_REQS} not found."
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "[setup-dtts-sd35] ERROR: python3 not found in PATH."
  exit 1
fi

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  echo "[setup-dtts-sd35] Active conda env: ${CONDA_DEFAULT_ENV:-unknown} (${CONDA_PREFIX})"
else
  echo "[setup-dtts-sd35] WARNING: no active conda env detected."
  echo "[setup-dtts-sd35]          Recommended: conda activate diffusion-tts"
fi

export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

echo "[setup-dtts-sd35] Repairing pip/setuptools toolchain for ImageReward build compatibility"
python3 -m pip install --force-reinstall \
  "pip==24.3.1" \
  "setuptools==75.8.0" \
  "wheel==0.45.1" \
  "backports.tarfile"

echo "[setup-dtts-sd35] Installing runtime dependencies (IR-only, no HPS)"
python3 -m pip install --no-build-isolation \
  "torch==2.4.0" \
  "torchvision==0.19.0" \
  "accelerate==1.2.1" \
  "diffusers==0.36.0" \
  "transformers==4.38.2" \
  "tokenizers==0.15.2" \
  "protobuf==3.20.3" \
  "numpy==1.26.3" \
  "tqdm==4.66.4" \
  "packaging" \
  "safetensors==0.5.2" \
  "pillow==11.1.0" \
  "sentencepiece==0.2.0" \
  "typing_extensions" \
  "google-genai" \
  "git+https://github.com/openai/CLIP.git"

echo "[setup-dtts-sd35] Applying SD3.5 overrides"
python3 -m pip install -r "${SD35_REQS}"

echo "[setup-dtts-sd35] Installing ImageReward from source (required)"
python3 -m pip install --no-build-isolation "git+https://github.com/THUDM/ImageReward.git"

echo "[setup-dtts-sd35] Validating imports and FlowMatch scheduler support"
cd "${DTTS_DIR}"
python3 - <<'PY'
import importlib
import inspect

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

import google.protobuf  # noqa: F401

from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
init_params = set(inspect.signature(FlowMatchEulerDiscreteScheduler.__init__).parameters.keys())
print(f"[setup-dtts-sd35] FlowMatch stochastic_sampling available={'stochastic_sampling' in init_params}")

import run_multi_backbone_eps_greedy_geneval as runner  # noqa: F401
print("[setup-dtts-sd35] import check OK")
PY

DTTS_PREFETCH="${DTTS_PREFETCH:-1}"
if [[ "${DTTS_PREFETCH}" == "1" ]]; then
  echo "[setup-dtts-sd35] Prefetching SD3.5 + ImageReward weights"
  python3 - <<'PY'
from huggingface_hub import snapshot_download
import ImageReward as RM

print("[setup-dtts-sd35] downloading stabilityai/stable-diffusion-3.5-large")
snapshot_download(repo_id="stabilityai/stable-diffusion-3.5-large", resume_download=True)
print("[setup-dtts-sd35] downloading ImageReward-v1.0")
RM.load("ImageReward-v1.0")
print("[setup-dtts-sd35] prefetch complete")
PY
else
  echo "[setup-dtts-sd35] Skipping prefetch (DTTS_PREFETCH=${DTTS_PREFETCH})"
fi

echo "[setup-dtts-sd35] Done."
