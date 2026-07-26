#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# Reuse a working system CUDA stack (Kaggle); otherwise install the repository's matching backend.
if "${PYTHON:-python3}" -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
  uv venv --python "${PYTHON:-python3}" --system-site-packages
  uv pip install --python .venv/bin/python -e '.[openvino]'
elif command -v nvidia-smi >/dev/null; then
  uv venv --python "${PYTHON:-python3}"
  uv pip install --python .venv/bin/python -e '.[openvino,cu126]'
else
  uv venv --python "${PYTHON:-python3}"
  uv pip install --python .venv/bin/python -e '.[openvino,cpu]'
fi
uv pip install --python .venv/bin/python --no-deps -e projects/visionqc
.venv/bin/python -c 'import anomalib, torch; print("anomalib", anomalib.__version__); print("torch", torch.__version__, "CUDA", torch.version.cuda, "GPU", torch.cuda.is_available())'
