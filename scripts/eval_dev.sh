#!/usr/bin/env bash
set -euo pipefail
SUBMISSION_DIR="${SUBMISSION_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
python "$SUBMISSION_DIR/src/apply_frontier_accel.py" examples/maze_plr.py

for seed in 0 1 2; do
  for run in dr_repro plr_perp_repro accel_repro; do
    python examples/maze_plr.py --mode eval \
      --checkpoint_directory "./checkpoints/${run}/${seed}" \
      --checkpoint_to_eval -1 --seed "$seed" --run_name "$run"
  done
  python examples/maze_frontier_accel.py --mode eval \
    --checkpoint_directory "./checkpoints/frontier_accel/${seed}" \
    --checkpoint_to_eval -1 --seed "$seed" --run_name frontier_accel
done
