#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# Kaggle already provides a CUDA-compatible PyTorch; expose it to the isolated environment.
uv venv --python "${PYTHON:-python3}" --system-site-packages
if .venv/bin/python -c 'import torch' 2>/dev/null; then
  uv pip install --python .venv/bin/python -e '.[openvino]'
else
  uv pip install --python .venv/bin/python -e '.[openvino,cpu]'
fi
uv pip install --python .venv/bin/python --no-deps -e projects/visionqc
.venv/bin/python -c 'import anomalib, torch; print("anomalib", anomalib.__version__); print("torch", torch.__version__, "CUDA", torch.version.cuda, "GPU", torch.cuda.is_available())'
