#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-linearrag}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if ! command -v conda >/dev/null 2>&1; then
    echo "Conda was not found. Install Miniconda or Anaconda first." >&2
    exit 1
fi

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists; reusing it."
else
    conda create -n "$ENV_NAME" python=3.9 pip -y
fi

conda run -n "$ENV_NAME" python -m pip install \
    torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu121
conda run -n "$ENV_NAME" python -m pip install -r requirements.txt
conda run -n "$ENV_NAME" python -m spacy download en_core_web_trf
conda run -n "$ENV_NAME" python scripts/download_embedding_model.py
conda run -n "$ENV_NAME" python scripts/check_environment.py

echo "Environment is ready. Activate it with: conda activate $ENV_NAME"
