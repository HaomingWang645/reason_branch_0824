#!/usr/bin/env bash
# Re-run the samples whose Pi3X render was blank (race), with the locked server. Resume re-runs only missing samples.
cd /home/haoming/reason_branch_0824/think3d_repro
until curl -s -m 5 http://localhost:20031/health | grep -q model_loaded; do sleep 5; done
if ! curl -s -m 5 http://localhost:30060/v1/models | grep -q px256k; then
  S=$(ls -d /home/haoming/.cache/huggingface/hub/models--jialianjie--SPAgent-4B/snapshots/*)
  nohup bash spagent/logs/serve_vllm.sh "$S" SPAgent-4B-px256k 30060 0.30 6 "--mm-processor-kwargs {\"max_pixels\":262144}" > spagent/logs/vllm_spagent4b_px256k.log 2>&1 &
  until grep -q "Application startup complete" spagent/logs/vllm_spagent4b_px256k.log; do sleep 5; done
fi
echo "[rerun] servers ready $(date +%T)"
python3 - <<'PY' > spagent/logs/rerun_affected_jobs.txt
import json
for key in json.load(open('outputs/affected_blank_render_samples.json')):
    run, tag, ds = key.split('|')
    model, port = ('SPAgent-4B-px256k', 30060) if 'px256k' in tag else ('SPAgent-4B', 30059)
    print(model, port, run.replace('run', ''), ds)
PY
while read model port run ds; do
  echo "[rerun] $model $ds run$run"; bash scripts/run_eval.sh $model $port think3d $run $ds --rl-trained >> outputs/logs/rerun_affected.log 2>&1
done < spagent/logs/rerun_affected_jobs.txt
/home/haoming/miniconda3/envs/spagent/bin/python scripts/score.py --tables > spagent/logs/rerun_tables.log 2>&1
echo "[rerun] DONE $(date +%T)"
