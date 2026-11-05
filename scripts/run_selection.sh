#!/usr/bin/env bash
# Torque-label low-pass selection on VALIDATION flights (docs/final_protocol.md):
# 3 network models x 4 cutoffs x seeds 0-2 = 36 runs, results/sel_<model>_a<cutoff>_s<seed>.
# Finished runs are skipped, so the script can be restarted. Then: uv run python scripts/select_alpha.py
#
#   bash scripts/run_selection.sh            # MAX parallel runs (default 4; WSL with ~6 GB RAM)
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
MAX=${MAX:-4}
LOG=results/logs; mkdir -p "$LOG"
alive() { pgrep -fc "bin/python scripts/run\.py"; }
for s in 0 1 2; do for m in costa ddm ddm_kin; do for a in none 40 20 10; do
  id=sel_${m}_a${a}_s${s}
  [ -f "results/$id/metrics_dev.json" ] && continue
  rm -rf "results/$id"
  while [ "$(alive)" -ge "$MAX" ]; do sleep 15; done
  cut=(); [ "$a" != none ] && cut=(--alpha-cutoff "$a")
  echo "$(date +%T) start $id"
  uv run python scripts/run.py --model "$m" --seed "$s" "${cut[@]}" --roles val --eval-stride 20 \
      --run-id "$id" > "$LOG/$id.log" 2>&1 &
  sleep 20   # let it load its data before counting again
done; done; done
wait
while [ "$(alive)" -gt 0 ]; do sleep 15; done
echo "$(date +%T) all selection runs finished"
