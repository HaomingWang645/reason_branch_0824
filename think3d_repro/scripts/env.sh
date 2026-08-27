# Common environment for all Think3D reproduction launches.
# CUDA device order MUST be PCI_BUS_ID so that CUDA index == nvidia-smi index
# (this box mixes H100 NVL / PCIe, and CUDA's default FASTEST_FIRST order differs).
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export REPRO_ROOT=/home/haoming/reason_branch_0824/think3d_repro
export SPAGENT_ROOT=$REPRO_ROOT/spagent
export LMUData=$SPAGENT_ROOT/dataset/LMUData          # VLMEvalKit data root (BLINK.tsv)
export PYTHONUTF8=1
export TOKENIZERS_PARALLELISM=false
