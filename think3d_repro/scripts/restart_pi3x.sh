#!/usr/bin/env bash
source /home/haoming/reason_branch_0824/think3d_repro/scripts/env.sh
source /home/haoming/miniconda3/etc/profile.d/conda.sh
conda activate spagent
for p in $(pgrep -f "pi3x_server.py"); do kill $p 2>/dev/null; done; sleep 5
cd $SPAGENT_ROOT/spagent/external_experts/Pi3
CUDA_VISIBLE_DEVICES=6 nohup python pi3x_server.py --checkpoint_path $SPAGENT_ROOT/checkpoints/pi3x/model.safetensors --port 20031 > $SPAGENT_ROOT/logs/pi3x_server.log 2>&1 &
until grep -q "Running on all" $SPAGENT_ROOT/logs/pi3x_server.log 2>/dev/null; do sleep 3; done
echo "pi3x ready: $(curl -s http://localhost:20031/health | head -c 50)"
