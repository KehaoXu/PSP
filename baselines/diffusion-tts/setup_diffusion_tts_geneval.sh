#!/usr/bin/env bash
set -euo pipefail

DTTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${DTTS_DIR}/../.." && pwd)"
DSEARCH_REQS="${REPO_ROOT}/baselines/DSearch/requirements.txt"
SD35_REQS="${REPO_ROOT}/requirements_sd3.5.txt"

echo "[setup-dtts] diffusion-tts dir: ${DTTS_DIR}"
echo "[setup-dtts] repo root: ${REPO_ROOT}"

if [[ ! -f "${DSEARCH_REQS}" ]]; then
  echo "[setup-dtts] ERROR: ${DSEARCH_REQS} not found."
  exit 1
fi
if [[ ! -f "${SD35_REQS}" ]]; then
  echo "[setup-dtts] ERROR: ${SD35_REQS} not found."
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "[setup-dtts] ERROR: python3 not found in PATH."
  exit 1
fi

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  echo "[setup-dtts] Active conda env: ${CONDA_DEFAULT_ENV:-unknown} (${CONDA_PREFIX})"
else
  echo "[setup-dtts] WARNING: no active conda env detected."
  echo "[setup-dtts]          Recommended: conda activate diffusion-tts"
fi
echo "[setup-dtts] python3 path: $(command -v python3)"

export PYTHONNOUSERSITE=1
unset PYTHONPATH || true
echo "[setup-dtts] PYTHONNOUSERSITE=${PYTHONNOUSERSITE}"

echo "[setup-dtts] Repairing pip/setuptools toolchain for ImageReward compatibility"
python3 -m pip install --force-reinstall \
  "pip==24.3.1" \
  "setuptools==75.8.0" \
  "wheel==0.45.1" \
  "backports.tarfile"

echo "[setup-dtts] Installing core runtime dependencies (torch/diffusers/transformers/etc)"
python3 -m pip install --no-build-isolation -r "${DSEARCH_REQS}"

echo "[setup-dtts] Applying SD3.5-compatible overrides"
python3 -m pip install -r "${SD35_REQS}"

echo "[setup-dtts] Installing ImageReward from source (no build isolation)"
python3 -m pip install --no-build-isolation "git+https://github.com/THUDM/ImageReward.git"

echo "[setup-dtts] Ensuring hpsv2 tokenizer asset exists (bpe_simple_vocab_16e6.txt.gz)"
python3 - <<'PY'
import importlib.util
import pathlib
import urllib.request

spec = importlib.util.find_spec("hpsv2")
if spec is None or spec.origin is None:
    raise SystemExit("hpsv2 is not installed after requirements install.")

hps_dir = pathlib.Path(spec.origin).resolve().parent
bpe_path = hps_dir / "src" / "open_clip" / "bpe_simple_vocab_16e6.txt.gz"
bpe_path.parent.mkdir(parents=True, exist_ok=True)

if not bpe_path.exists():
    url = "https://openaipublic.blob.core.windows.net/clip/bpe_simple_vocab_16e6.txt.gz"
    print(f"[setup-dtts] Downloading missing BPE asset to {bpe_path}")
    urllib.request.urlretrieve(url, str(bpe_path))
else:
    print(f"[setup-dtts] BPE asset already present: {bpe_path}")
PY

echo "[setup-dtts] Validating runtime imports for diffusion-tts experiments"
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
    "safetensors",
    "numpy",
    "tqdm",
    "clip",
    "hpsv2",
    "ImageReward",
]
for m in mods:
    importlib.import_module(m)

from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

init_params = set(inspect.signature(FlowMatchEulerDiscreteScheduler.__init__).parameters.keys())
print(f"[setup-dtts] FlowMatch stochastic_sampling available={'stochastic_sampling' in init_params}")

import run_multi_backbone_eps_greedy_geneval as runner  # noqa: F401
print("[setup-dtts] import check OK: run_multi_backbone_eps_greedy_geneval")
PY

DTTS_PREFETCH="${DTTS_PREFETCH:-0}"
if [[ "${DTTS_PREFETCH}" == "1" ]]; then
  echo "[setup-dtts] Prefetching SD1.5 + SDXL + SD3.5 + ImageReward weights (can take a while)"
  python3 - <<'PY'
from huggingface_hub import snapshot_download
import ImageReward as RM

for repo_id in [
    "runwayml/stable-diffusion-v1-5",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/stable-diffusion-3.5-large",
]:
    print(f"[setup-dtts] downloading {repo_id}")
    snapshot_download(repo_id=repo_id, resume_download=True)

print("[setup-dtts] downloading ImageReward-v1.0")
RM.load("ImageReward-v1.0")
print("[setup-dtts] prefetch complete")
PY
else
  echo "[setup-dtts] Skipping prefetch (set DTTS_PREFETCH=1 to enable)"
fi

echo
echo "[setup-dtts] Done."
echo "[setup-dtts] Next: run from diffusion-tts/"
echo "  bash exps/sd15_eps_greedy_ir/gpu0.sh"
