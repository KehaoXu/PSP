#!/usr/bin/env bash
# One-time setup on the pod: a Python 3.10 venv on the network volume (survives pod restarts).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"

if command -v apt-get >/dev/null 2>&1 && ! command -v tmux >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq tmux
fi
python3 -m pip install -q uv
[ -x "$VENV/bin/python" ] || uv venv "$VENV" --python 3.10 --seed
PY="$VENV/bin/python"

uv pip install --python "$PY" "setuptools==69.5.1" wheel
uv pip install --python "$PY" -r "$HERE/requirements_cache.txt"
# As in setup/setup.sh and the Colab runs: ImageReward's setup.py needs pkg_resources (setuptools 69.5.1,
# no build isolation) and its declared deps would downgrade transformers / timm, hence --no-deps.
# clip-anytorch only satisfies the `import clip` at the top of PSP's rewards.py.
uv pip install --python "$PY" --no-build-isolation --no-deps \
  "fairscale==0.4.13" \
  "clip-anytorch" \
  "diffusers @ git+https://github.com/huggingface/diffusers@af28ae2d5ba0ef80d99fff7859ebea730e1cf3f8" \
  "image-reward @ git+https://github.com/THUDM/ImageReward.git@2ca71bac4ed86b922fe53ddaec3109fe94d45fd3"

# hpsv2's tokenizer vocabulary is missing from the PyPI package (same fix as setup/setup.sh)
"$PY" - <<'EOF'
import importlib.util, pathlib, urllib.request
bpe = pathlib.Path(importlib.util.find_spec("hpsv2").origin).parent / "src" / "open_clip" / "bpe_simple_vocab_16e6.txt.gz"
if not bpe.exists():
    bpe.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve("https://openaipublic.blob.core.windows.net/clip/bpe_simple_vocab_16e6.txt.gz", bpe)
print("hpsv2 vocab:", bpe)
EOF

cd "$HERE"
"$PY" - <<'EOF'
import cache_lib  # noqa: F401  (sets up the PSP import path)
import torch, diffusers, transformers, ImageReward  # noqa: F401
from fkd_diffusers.fkd_pipeline_sd import FKDStableDiffusion  # noqa: F401
from fkd_diffusers.rewards import _install_turtle_shim_for_hpsv2
_install_turtle_shim_for_hpsv2()
from hpsv2 import img_score  # noqa: F401
from transformers import BlipForImageTextRetrieval  # noqa: F401
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| GPUs", torch.cuda.device_count(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")
print("diffusers", diffusers.__version__, "| transformers", transformers.__version__)
print("environment OK")
EOF
