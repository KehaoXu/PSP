#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FITS_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BACKBONE="${1:-all}"

echo "[setup] FITS root: ${FITS_ROOT}"
echo "[setup] Target backbone: ${BACKBONE}"

cd "${FITS_ROOT}"

echo "[setup] Installing standard FITS dependencies (README setup)..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/openai/CLIP.git
pip install -r requirements.txt
pip install -e .

echo "[setup] Installing t2v_metrics editable package..."
cd "${FITS_ROOT}/third-party/t2v_metrics"
pip install -e .

cd "${FITS_ROOT}"

echo "[setup] Installing/ensuring runtime deps used in our FITS modifications..."
# These are already pinned in requirements.txt; we keep this explicit for robustness.
pip install omegaconf==2.3.0 image-reward==1.5

echo "[setup] Enforcing NumPy/Torch compatibility (fixes 'RuntimeError: Numpy is not available')..."
# Some dependency resolutions can upgrade NumPy to an incompatible major version for torch==2.1.x.
# Force a known-good NumPy build after all installs.
pip install --force-reinstall "numpy==1.26.4"

echo "[setup] Running compatibility sanity check..."
python - <<'PY'
import numpy as np
import torch
_ = torch.from_numpy(np.arange(8, dtype=np.int64))
print(f"[setup] OK: torch={torch.__version__}, numpy={np.__version__}")
PY

echo "[setup] Prefetching model weights for ${BACKBONE}..."
BACKBONE_ENV="${BACKBONE}" python - <<'PY'
import os
from huggingface_hub import snapshot_download
import ImageReward as RM

backbone = os.environ.get("BACKBONE_ENV", "all").lower()
repos_by_backbone = {
    "sd15": ["runwayml/stable-diffusion-v1-5"],
    "sdxl": ["stabilityai/stable-diffusion-xl-base-1.0"],
    "sd35": ["stabilityai/stable-diffusion-3.5-large"],
}

repos = []
if backbone in repos_by_backbone:
    repos = repos_by_backbone[backbone]
else:
    for values in repos_by_backbone.values():
        repos.extend(values)

for repo_id in repos:
    print(f"[setup] Prefetching {repo_id} ...")
    snapshot_download(repo_id=repo_id)

print("[setup] Prefetching ImageReward-v1.0 ...")
RM.load("ImageReward-v1.0")
print("[setup] Prefetch complete.")
PY

echo "[setup] Completed setup for ${BACKBONE}."
