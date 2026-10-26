# Shared job setup: activate the `torch` conda env on the compute node itself
# (not inherited from the login shell) and make sure every dependency imports.
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate torch
echo "python: $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import matplotlib" 2>/dev/null || python -m pip install --quiet matplotlib
python -c "import torch, nibabel, matplotlib; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
