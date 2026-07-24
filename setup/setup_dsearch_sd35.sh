#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DSEARCH_REQS="${REPO_ROOT}/baselines/DSearch/requirements.txt"
SD35_REQS="${REPO_ROOT}/requirements_sd3.5.txt"
FK_TEXT_TO_IMAGE_DIR="${REPO_ROOT}/Fk-Diffusion-Steering/text_to_image"

echo "[setup-dsearch-sd35] Repo root: ${REPO_ROOT}"
echo "[setup-dsearch-sd35] DSearch requirements: ${DSEARCH_REQS}"
echo "[setup-dsearch-sd35] SD3.5 overrides: ${SD35_REQS}"

if [[ ! -f "${DSEARCH_REQS}" ]]; then
  echo "[setup-dsearch-sd35] ERROR: ${DSEARCH_REQS} not found."
  exit 1
fi
if [[ ! -f "${SD35_REQS}" ]]; then
  echo "[setup-dsearch-sd35] ERROR: ${SD35_REQS} not found."
  exit 1
fi
if [[ ! -d "${FK_TEXT_TO_IMAGE_DIR}" ]]; then
  echo "[setup-dsearch-sd35] ERROR: ${FK_TEXT_TO_IMAGE_DIR} not found."
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "[setup-dsearch-sd35] ERROR: python3 not found in PATH."
  exit 1
fi

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  echo "[setup-dsearch-sd35] Active conda env: ${CONDA_DEFAULT_ENV:-unknown} (${CONDA_PREFIX})"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
  echo "[setup-dsearch-sd35] Active virtualenv: ${VIRTUAL_ENV}"
else
  echo "[setup-dsearch-sd35] WARNING: no conda/venv detected; using system python."
fi
echo "[setup-dsearch-sd35] python3 path: $(command -v python3)"

# Install zip if apt is available (needed by some downstream tooling).
echo "[setup-dsearch-sd35] Installing system dependency: zip"
if command -v apt-get >/dev/null 2>&1; then
  APT_CMD=""
  if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
    APT_CMD="apt-get"
  elif command -v sudo >/dev/null 2>&1; then
    APT_CMD="sudo apt-get"
  fi

  if [[ -n "${APT_CMD}" ]]; then
    ${APT_CMD} update
    ${APT_CMD} install -y zip
  else
    echo "[setup-dsearch-sd35] WARNING: apt-get available but no root/sudo. Skipping zip install."
  fi
else
  echo "[setup-dsearch-sd35] apt-get not available. Skipping zip install."
fi

# Avoid user-site leakage into this environment.
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

echo "[setup-dsearch-sd35] Repairing pip/setuptools toolchain for ImageReward build compatibility"
python3 -m pip install --force-reinstall \
  "pip==24.3.1" \
  "setuptools==75.8.0" \
  "wheel==0.45.1" \
  "backports.tarfile"
python3 - <<'PY'
import pkg_resources
import setuptools.build_meta
print("[setup-dsearch-sd35] pkg_resources + setuptools.build_meta import OK")
PY

echo "[setup-dsearch-sd35] Installing DSearch runtime requirements"
python3 -m pip install --no-build-isolation -r "${DSEARCH_REQS}"

echo "[setup-dsearch-sd35] Installing ImageReward from source (no build isolation)"
python3 -m pip install --no-build-isolation \
  "git+https://github.com/THUDM/ImageReward.git"

echo "[setup-dsearch-sd35] Applying SD3.5 compatibility overrides"
python3 -m pip uninstall -y diffusers || true
python3 -m pip install -r "${SD35_REQS}"

echo "[setup-dsearch-sd35] Ensuring hpsv2 tokenizer asset exists"
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
    print(f"[setup-dsearch-sd35] Downloading missing BPE asset to {bpe_path}")
    urllib.request.urlretrieve(url, str(bpe_path))
else:
    print(f"[setup-dsearch-sd35] BPE asset already present: {bpe_path}")
PY

echo "[setup-dsearch-sd35] Validating runtime imports and SD3.5 scheduler support"
python3 - <<'PY'
import importlib
import inspect
import pathlib
import sys

repo_root = pathlib.Path.cwd()
text_to_image = repo_root / "Fk-Diffusion-Steering" / "text_to_image"
if str(text_to_image) not in sys.path:
    sys.path.insert(0, str(text_to_image))
baselines_dir = repo_root / "baselines"
if str(baselines_dir) not in sys.path:
    sys.path.insert(0, str(baselines_dir))

mods = [
    "torch",
    "torchvision",
    "accelerate",
    "diffusers",
    "transformers",
    "numpy",
    "tqdm",
    "clip",
    "hpsv2",
    "ImageReward",
    "google.genai",
]
for m in mods:
    importlib.import_module(m)

from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
init_params = set(inspect.signature(FlowMatchEulerDiscreteScheduler.__init__).parameters.keys())
print(
    "[setup-dsearch-sd35] stochastic_sampling available="
    f"{'stochastic_sampling' in init_params}"
)

import DSearch.run_dsearch_geneval as runner  # noqa: F401
print("[setup-dsearch-sd35] import check OK: DSearch.run_dsearch_geneval")
PY

# Set DSEARCH_PREFETCH=0 to skip heavy model downloads.
DSEARCH_PREFETCH="${DSEARCH_PREFETCH:-1}"
if [[ "${DSEARCH_PREFETCH}" == "1" ]]; then
  echo "[setup-dsearch-sd35] Prefetching sd35 + reward weights (this can take a while)"
  python3 - <<'PY'
from huggingface_hub import snapshot_download
import ImageReward as RM

print("[setup-dsearch-sd35] downloading stabilityai/stable-diffusion-3.5-large")
snapshot_download(
    repo_id="stabilityai/stable-diffusion-3.5-large",
    resume_download=True,
)

print("[setup-dsearch-sd35] downloading ImageReward-v1.0")
RM.load("ImageReward-v1.0")
print("[setup-dsearch-sd35] prefetch complete")
PY
else
  echo "[setup-dsearch-sd35] Skipping model prefetch (DSEARCH_PREFETCH=${DSEARCH_PREFETCH})"
fi

echo
echo "[setup-dsearch-sd35] Done."
echo "[setup-dsearch-sd35] NOTE: sd35 is gated. Run 'huggingface-cli login' and"
echo "[setup-dsearch-sd35] accept model terms at https://huggingface.co/stabilityai/stable-diffusion-3.5-large"
