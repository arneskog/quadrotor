#!/usr/bin/env bash
# Final runs (docs/final_protocol.md). Finished runs are skipped, so the script can be restarted.
#
#   bash scripts/run_final.sh train   # stage 1: fit every model, evaluate on train + val flights
#   bash scripts/run_final.sh test    # stage 2: evaluate every frozen final run ONCE on the test flights
#
# Models: pbm and linear_drag (deterministic, one run), costa / ddm / ddm_kin with seeds 0-4.
# Run ids: results/final_<model>_s<seed>. Logs: results/logs/.
set -u
cd "$(dirname "$0")/.."
STAGE=${1:?usage: run_final.sh train|test}
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
LOG=results/logs; mkdir -p "$LOG"
alive() { pgrep -fc "bin/python scripts/run\.py"; }

if [ -n "$(git status --porcelain --untracked-files=no -- src scripts configs)" ]; then
  echo "uncommitted changes to tracked files in src/ scripts/ configs/: commit first (runs record the git hash)" >&2
  exit 1
fi

runs=(final_pbm_s0 final_linear_drag_s0)
for s in 0 1 2 3 4; do for m in costa ddm ddm_kin; do runs+=("final_${m}_s${s}"); done; done

for id in "${runs[@]}"; do
  m=${id#final_}; m=${m%_s*}; s=${id##*_s}
  if [ "$STAGE" = train ]; then
    MAX=${MAX:-4}
    [ -f "results/$id/metrics_dev.json" ] && continue
    rm -rf "results/$id"
    while [ "$(alive)" -ge "$MAX" ]; do sleep 15; done
    echo "$(date +%T) train $id"
    uv run python scripts/run.py --model "$m" --seed "$s" --run-id "$id" > "$LOG/$id.log" 2>&1 &
  elif [ "$STAGE" = test ]; then
    MAX=${MAX:-3}
    [ -f "results/$id/metrics_dev.json" ] || { echo "skip $id: stage 1 not finished" >&2; continue; }
    [ -f "results/$id/metrics_test.json" ] && continue
    while [ "$(alive)" -ge "$MAX" ]; do sleep 15; done
    echo "$(date +%T) test $id"
    uv run python scripts/run.py --from-run "results/$id" --roles test_interp test_near test_extrap \
        > "$LOG/${id}_test.log" 2>&1 &
  else
    echo "unknown stage $STAGE" >&2; exit 2
  fi
  sleep 20   # let it load its data before counting again
done
wait
while [ "$(alive)" -gt 0 ]; do sleep 15; done
echo "$(date +%T) stage $STAGE finished"
