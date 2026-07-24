#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DSEARCH_REQS="${REPO_ROOT}/baselines/DSearch/requirements.txt"
FK_TEXT_TO_IMAGE_DIR="${REPO_ROOT}/Fk-Diffusion-Steering/text_to_image"

echo "[setup-dsearch] Repo root: ${REPO_ROOT}"
echo "[setup-dsearch] Requirements: ${DSEARCH_REQS}"

if [[ ! -f "${DSEARCH_REQS}" ]]; then
  echo "[setup-dsearch] ERROR: ${DSEARCH_REQS} not found."
  exit 1
fi

if [[ ! -d "${FK_TEXT_TO_IMAGE_DIR}" ]]; then
  echo "[setup-dsearch] ERROR: ${FK_TEXT_TO_IMAGE_DIR} not found."
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "[setup-dsearch] ERROR: python3 not found in PATH."
  exit 1
fi

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  echo "[setup-dsearch] Active conda env: ${CONDA_DEFAULT_ENV:-unknown} (${CONDA_PREFIX})"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
  echo "[setup-dsearch] Active virtualenv: ${VIRTUAL_ENV}"
else
  echo "[setup-dsearch] WARNING: no conda/venv detected; using system python."
fi
echo "[setup-dsearch] python3 path: $(command -v python3)"

# Install zip if apt is available (needed by some downstream tooling).
echo "[setup-dsearch] Installing system dependency: zip"
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
    echo "[setup-dsearch] WARNING: apt-get available but no root/sudo. Skipping zip install."
  fi
else
  echo "[setup-dsearch] apt-get not available. Skipping zip install."
fi

# Avoid user-site leakage into this environment.
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

echo "[setup-dsearch] Repairing pip/setuptools toolchain for ImageReward build compatibility"
python3 -m pip install --force-reinstall \
  "pip==24.3.1" \
  "setuptools==75.8.0" \
  "wheel==0.45.1" \
  "backports.tarfile"
python3 - <<'PY'
import pkg_resources
import setuptools.build_meta
print("[setup-dsearch] pkg_resources + setuptools.build_meta import OK")
PY

echo "[setup-dsearch] Installing DSearch runtime requirements"
python3 -m pip install --no-build-isolation -r "${DSEARCH_REQS}"

echo "[setup-dsearch] Installing ImageReward from source (no build isolation)"
python3 -m pip install --no-build-isolation \
  "git+https://github.com/THUDM/ImageReward.git"

echo "[setup-dsearch] Ensuring hpsv2 tokenizer asset exists"
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
    print(f"[setup-dsearch] Downloading missing BPE asset to {bpe_path}")
    urllib.request.urlretrieve(url, str(bpe_path))
else:
    print(f"[setup-dsearch] BPE asset already present: {bpe_path}")
PY

echo "[setup-dsearch] Validating runtime imports for DSearch Geneval runner"
python3 - <<'PY'
import importlib
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

import DSearch.run_dsearch_geneval as runner  # noqa: F401
print("[setup-dsearch] import check OK: DSearch.run_dsearch_geneval")
PY

# Set DSEARCH_PREFETCH=0 to skip heavy model downloads.
DSEARCH_PREFETCH="${DSEARCH_PREFETCH:-1}"
if [[ "${DSEARCH_PREFETCH}" == "1" ]]; then
  echo "[setup-dsearch] Prefetching sd15/sdxl + reward weights (this can take a while)"
  python3 - <<'PY'
from huggingface_hub import snapshot_download
import ImageReward as RM

repos = [
    "runwayml/stable-diffusion-v1-5",
    "stabilityai/stable-diffusion-xl-base-1.0",
]
for repo in repos:
    print(f"[setup-dsearch] downloading {repo}")
    snapshot_download(repo_id=repo, resume_download=True)

print("[setup-dsearch] downloading ImageReward-v1.0")
RM.load("ImageReward-v1.0")
print("[setup-dsearch] prefetch complete")
PY
else
  echo "[setup-dsearch] Skipping model prefetch (DSEARCH_PREFETCH=${DSEARCH_PREFETCH})"
fi

echo
echo "[setup-dsearch] Done."
