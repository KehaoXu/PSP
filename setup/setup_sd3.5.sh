#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="${REPO_ROOT}/Fk-Diffusion-Steering"
SD35_REQS="${REPO_ROOT}/requirements_sd3.5.txt"

echo "[setup-sd35] Repo root: ${REPO_ROOT}"
echo "[setup-sd35] Project dir: ${PROJECT_DIR}"

if [[ ! -d "${PROJECT_DIR}" ]]; then
  echo "[setup-sd35] ERROR: ${PROJECT_DIR} not found."
  exit 1
fi

if [[ ! -f "${SD35_REQS}" ]]; then
  echo "[setup-sd35] ERROR: ${SD35_REQS} not found."
  exit 1
fi

cd "${PROJECT_DIR}"

echo "[setup-sd35] Installing system deps (zip + headless GUI libs for hpsv2)"
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
    echo "[setup-sd35] WARNING: apt-get available but no root/sudo. Skipping system deps install."
  fi
else
  echo "[setup-sd35] apt-get not available. Skipping system deps install."
fi

echo "[setup-sd35] Upgrading pip tooling"
python3 -m pip install --upgrade pip wheel
python3 -m pip install "setuptools==69.5.1"

echo "[setup-sd35] Installing base project requirements"
python3 -m pip install --no-build-isolation -r requirements.txt

echo "[setup-sd35] Replacing old editable diffusers with SD3.5-capable build"
python3 -m pip uninstall -y diffusers || true
python3 -m pip install -r "${SD35_REQS}"

echo "[setup-sd35] Ensuring hpsv2 tokenizer asset exists (bpe_simple_vocab_16e6.txt.gz)"
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
    print(f"[setup-sd35] Downloading missing BPE asset to {bpe_path}")
    urllib.request.urlretrieve(url, str(bpe_path))
else:
    print(f"[setup-sd35] BPE asset already present: {bpe_path}")
PY

echo "[setup-sd35] Validating imports and versions"
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

print(f"[setup-sd35] diffusers={diffusers.__version__}")
print(f"[setup-sd35] transformers={transformers.__version__}")
print(f"[setup-sd35] tokenizers={tokenizers.__version__}")

# Check whether the installed scheduler implementation supports stochastic sampling.
init_params = set(inspect.signature(FlowMatchEulerDiscreteScheduler.__init__).parameters.keys())
supports_stochastic = "stochastic_sampling" in init_params
print(f"[setup-sd35] scheduler={FlowMatchEulerDiscreteScheduler.__module__}.{FlowMatchEulerDiscreteScheduler.__name__}")
print(f"[setup-sd35] stochastic_sampling available={supports_stochastic}")

# Check what the scheduler default is in this environment.
scheduler = FlowMatchEulerDiscreteScheduler()
print(
    "[setup-sd35] stochastic_sampling default="
    f"{getattr(scheduler.config, 'stochastic_sampling', 'UNSUPPORTED')}"
)

# SD3 pipeline uses FlowMatchEulerDiscreteScheduler by default in diffusers.
print("[setup-sd35] SD3 pipeline scheduler type: FlowMatchEulerDiscreteScheduler (default)")

print("[setup-sd35] import check OK: StableDiffusion3Pipeline + ImageReward")
PY

echo
echo "[setup-sd35] Done."
echo "[setup-sd35] Next steps:"
echo "  1) huggingface-cli login"
echo "  2) Accept terms at: https://huggingface.co/stabilityai/stable-diffusion-3.5-large"
echo "  3) bash exps/run_launch_eval_multiturn_sdv3.5/gpu0_scheduled_geneval.sh"
