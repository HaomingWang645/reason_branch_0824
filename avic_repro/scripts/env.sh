# Common environment for all AVIC reproduction launches.
export CUDA_DEVICE_ORDER=PCI_BUS_ID          # CUDA index == nvidia-smi index on this box
export AVIC_ROOT=/home/haoming/reason_branch_0824/avic_repro
export AVIC_SRC=$AVIC_ROOT/avic/visual_spatial_reasoning
export WORLD_MODEL_TYPE=svc
export PYTHONPATH=$AVIC_SRC:${PYTHONPATH:-}
export PYTHONUTF8=1
export TOKENIZERS_PARALLELISM=false
# OpenAI key (plain OpenAI backend; see patches/ for the api.py change). Read from the
# user's codex auth file if not already exported.
if [ -z "${OPENAI_API_KEY:-}" ]; then
  export OPENAI_API_KEY=$(python3 -c "import json;print(json.load(open('/home/haoming/.codex/auth.json'))['OPENAI_API_KEY'])")
fi
unset AZURE_OPENAI_ENDPOINT
source ~/miniconda3/etc/profile.d/conda.sh
conda activate avic
cd $AVIC_SRC
