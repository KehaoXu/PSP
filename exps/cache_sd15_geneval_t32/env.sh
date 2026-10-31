# Paths shared by the scripts in this directory; source it. RunPod mounts the network volume at /workspace.
export WORKSPACE="${WORKSPACE:-/workspace}"
export VENV="${VENV:-$WORKSPACE/venv_psp_cache}"
export HF_HOME="${HF_HOME:-$WORKSPACE/hf_home}"
export CACHE_DIR="${CACHE_DIR:-$WORKSPACE/cache_sd15_geneval_t32}"
export PYTHONUNBUFFERED=1

# ImageReward always downloads to ~/.cache/ImageReward; keep it on the volume
mkdir -p "$WORKSPACE/model_cache/ImageReward" "$HOME/.cache"
if [ ! -e "$HOME/.cache/ImageReward" ]; then
  ln -s "$WORKSPACE/model_cache/ImageReward" "$HOME/.cache/ImageReward"
fi
if [ -x "$VENV/bin/python" ]; then
  source "$VENV/bin/activate"
fi
