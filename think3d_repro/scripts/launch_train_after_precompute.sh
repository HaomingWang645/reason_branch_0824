#!/usr/bin/env bash
# Wait for the Pi3X cache precompute (GPU 5) to finish, then start GRPO training on GPU 5.
cd /home/haoming/reason_branch_0824/think3d_repro
while pgrep -f "precompute_pi3x_cach[e]" >/dev/null; do sleep 20; done
echo "[chain] precompute finished at $(date); cache files: $(ls spagent/dataset/pi3x_cache | wc -l)"
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 5)" -lt 500 ]; do sleep 5; done
echo "[chain] launching GRPO training on GPU 5 at $(date)"
bash scripts/train_grpo_4b.sh > spagent/logs/train_grpo_4b.log 2>&1
echo "[chain] training exited with code $? at $(date)"
