#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="${REPO_ROOT}/Fk-Diffusion-Steering"

echo "[setup] Repo root: ${REPO_ROOT}"
echo "[setup] Project dir: ${PROJECT_DIR}"

if [[ ! -d "${PROJECT_DIR}" ]]; then
  echo "[setup] ERROR: ${PROJECT_DIR} not found."
  exit 1
fi

cd "${PROJECT_DIR}"

echo "[setup] Installing system deps (zip + GUI/runtime deps required by hpsv2 on headless VMs)"
if command -v apt-get >/dev/null 2>&1; then
  APT_CMD=""
  if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
    APT_CMD="apt-get"
  elif command -v sudo >/dev/null 2>&1; then
    APT_CMD="sudo apt-get"
  fi

  if [[ -n "${APT_CMD}" ]]; then
    ${APT_CMD} update
    ${APT_CMD} install -y \
      zip \
      libx11-6 libxext6 libxrender1 libxft2 libxinerama1 libxrandr2 tk tcl
  else
    echo "[setup] WARNING: apt-get available but no root/sudo. Skipping system deps install."
  fi
else
  echo "[setup] apt-get not available. Skipping system deps install."
fi

echo "[setup] Upgrading pip + wheel and pinning setuptools for ImageReward build compatibility"
python3 -m pip install --upgrade pip wheel
python3 -m pip install "setuptools==69.5.1"

echo "[setup] Installing project requirements"
python3 -m pip install --no-build-isolation -r requirements.txt

echo "[setup] Ensuring hpsv2 tokenizer asset exists (bpe_simple_vocab_16e6.txt.gz)"
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
    print(f"[setup] Downloading missing BPE asset to {bpe_path}")
    urllib.request.urlretrieve(url, str(bpe_path))
else:
    print(f"[setup] BPE asset already present: {bpe_path}")
PY

echo "[setup] Validating core imports"
python3 -c "import pkg_resources, ImageReward; print('setup OK: pkg_resources + ImageReward import succeeded')"
python3 -c "import hpsv2; from hpsv2.src.open_clip import tokenizer; print('setup OK: hpsv2 tokenizer import succeeded')"

echo
echo "[setup] Done."