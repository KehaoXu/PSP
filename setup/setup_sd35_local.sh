#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="${REPO_ROOT}/Fk-Diffusion-Steering"
SD35_REQS="${REPO_ROOT}/requirements_sd3.5.txt"

echo "[setup-sd35-local] Repo root: ${REPO_ROOT}"
echo "[setup-sd35-local] Project dir: ${PROJECT_DIR}"

if [[ ! -d "${PROJECT_DIR}" ]]; then
  echo "[setup-sd35-local] ERROR: ${PROJECT_DIR} not found."
  exit 1
fi

if [[ ! -f "${SD35_REQS}" ]]; then
  echo "[setup-sd35-local] ERROR: ${SD35_REQS} not found."
  exit 1
fi

cd "${PROJECT_DIR}"

echo "[setup-sd35-local] Skipping system deps install (no apt-get update/install)."

if [[ -z "${CONDA_DEFAULT_ENV:-}" ]]; then
  echo "[setup-sd35-local] WARNING: no active conda env detected."
  echo "[setup-sd35-local] Proceeding anyway, but recommended: conda activate <sd35-env>"
else
  echo "[setup-sd35-local] Active conda env: ${CONDA_DEFAULT_ENV}"
fi

echo "[setup-sd35-local] Isolating from user-site packages"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true
echo "[setup-sd35-local] PYTHONNOUSERSITE=${PYTHONNOUSERSITE}"

echo "[setup-sd35-local] Repairing pip/setuptools toolchain"
python3 -m pip uninstall -y setuptools wheel || true
python3 -m pip install --force-reinstall \
  "pip==24.3.1" \
  "setuptools==75.8.0" \
  "wheel==0.45.1" \
  "backports.tarfile"
python3 - <<'PY'
import setuptools.build_meta
print("[setup-sd35-local] setuptools.build_meta import OK")
PY

echo "[setup-sd35-local] Installing base project requirements"
python3 -m pip install --no-build-isolation -r requirements.txt

echo "[setup-sd35-local] Replacing old editable diffusers with SD3.5-capable build"
python3 -m pip uninstall -y diffusers || true
python3 -m pip install -r "${SD35_REQS}"

echo "[setup-sd35-local] Ensuring hpsv2 tokenizer asset exists (bpe_simple_vocab_16e6.txt.gz)"
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
    print(f"[setup-sd35-local] Downloading missing BPE asset to {bpe_path}")
    urllib.request.urlretrieve(url, str(bpe_path))
else:
    print(f"[setup-sd35-local] BPE asset already present: {bpe_path}")
PY

echo "[setup-sd35-local] Validating imports and versions"
python3 - <<'PY'
import inspect
import diffusers
import tokenizers
import transformers
import ImageReward
from diffusers import StableDiffusion3Pipeline
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)

print(f"[setup-sd35-local] diffusers={diffusers.__version__}")
print(f"[setup-sd35-local] transformers={transformers.__version__}")
print(f"[setup-sd35-local] tokenizers={tokenizers.__version__}")

init_params = set(inspect.signature(FlowMatchEulerDiscreteScheduler.__init__).parameters.keys())
supports_stochastic = "stochastic_sampling" in init_params
print(f"[setup-sd35-local] scheduler={FlowMatchEulerDiscreteScheduler.__module__}.{FlowMatchEulerDiscreteScheduler.__name__}")
print(f"[setup-sd35-local] stochastic_sampling available={supports_stochastic}")

scheduler = FlowMatchEulerDiscreteScheduler()
print(
    "[setup-sd35-local] stochastic_sampling default="
    f"{getattr(scheduler.config, 'stochastic_sampling', 'UNSUPPORTED')}"
)

print("[setup-sd35-local] SD3 pipeline scheduler type: FlowMatchEulerDiscreteScheduler (default)")
print("[setup-sd35-local] import check OK: StableDiffusion3Pipeline + ImageReward")
PY

echo
echo "[setup-sd35-local] Done."
echo "[setup-sd35-local] Next steps:"
echo "  1) huggingface-cli login"
echo "  2) Accept terms at: https://huggingface.co/stabilityai/stable-diffusion-3.5-large"
echo "  3) bash exps/run_launch_eval_multiturn_sdv3.5/gpu0_scheduled_geneval.sh"
