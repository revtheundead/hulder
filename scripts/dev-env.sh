# Source this before any uv or vLLM command: `source scripts/dev-env.sh`.
# Keeps the interpreter, model weights and every JIT/compile cache inside the
# repo, so the workspace deletes cleanly (docs/design.md §8.1).

HULDER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HULDER_ROOT

export UV_PYTHON_INSTALL_DIR="$HULDER_ROOT/.cache/uv-python"
export HF_HOME="$HULDER_ROOT/.cache/hf"
export XDG_CACHE_HOME="$HULDER_ROOT/.cache/xdg"           # vLLM compile cache, misc
export VLLM_CACHE_ROOT="$HULDER_ROOT/.cache/vllm"
export TRITON_CACHE_DIR="$HULDER_ROOT/.cache/triton"
export FLASHINFER_WORKSPACE_BASE="$HULDER_ROOT/.cache"     # -> .cache/.cache/flashinfer
