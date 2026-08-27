#!/usr/bin/env bash
# Sequentially bring up the GPU-6 serving stack: Pi3X server -> Qwen3-VL-4B vLLM -> SPAgent-4B vLLM
source /home/haoming/reason_branch_0824/think3d_repro/scripts/env.sh
cd $SPAGENT_ROOT
source /home/haoming/miniconda3/etc/profile.d/conda.sh
conda activate spagent
(cd spagent/external_experts/Pi3 && CUDA_VISIBLE_DEVICES=6 nohup python pi3x_server.py --checkpoint_path $SPAGENT_ROOT/checkpoints/pi3x/model.safetensors --port 20031 > $SPAGENT_ROOT/logs/pi3x_server.log 2>&1 &)
until grep -q "Running on all" logs/pi3x_server.log; do grep -q "ERROR" logs/pi3x_server.log && { echo "PI3X FAILED"; exit 1; }; sleep 3; done
echo "pi3x ready: $(curl -s http://localhost:20031/health)"
Q=$(ls -d /home/haoming/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/*)
nohup bash logs/serve_vllm.sh "$Q" Qwen3-VL-4B-Instruct 30058 0.30 6 > logs/vllm_qwen3vl4b.log 2>&1 &
until grep -q "Application startup complete" logs/vllm_qwen3vl4b.log; do grep -q "initialization failed" logs/vllm_qwen3vl4b.log && { echo "QWEN VLLM FAILED"; exit 1; }; sleep 5; done
echo "qwen vllm ready"
S=$(ls -d /home/haoming/.cache/huggingface/hub/models--jialianjie--SPAgent-4B/snapshots/*)
nohup bash logs/serve_vllm.sh "$S" SPAgent-4B 30059 0.30 6 > logs/vllm_spagent4b.log 2>&1 &
until grep -q "Application startup complete" logs/vllm_spagent4b.log; do grep -q "initialization failed" logs/vllm_spagent4b.log && { echo "SPAGENT VLLM FAILED"; exit 1; }; sleep 5; done
echo "spagent vllm ready"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 5,6
