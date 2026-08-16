#!/usr/bin/env bash
set -euo pipefail
SUBMISSION_DIR="${SUBMISSION_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
python "$SUBMISSION_DIR/src/apply_frontier_accel.py" examples/maze_plr.py

# Baselines. No PPO/student flags are overridden.
for seed in 0 1 2; do
  python examples/maze_plr.py --seed "$seed" --run_name dr_repro --checkpoint_save_interval 17
  python examples/maze_plr.py --use_accel --seed "$seed" --run_name accel_repro --checkpoint_save_interval 17
done

# Robust PLR / PLR-perp configuration uses the upstream replay code without ACCEL edits.
for seed in 0 1 2; do
  python examples/maze_plr.py --use_plr --seed "$seed" --run_name plr_perp_repro --checkpoint_save_interval 17
done

# Proposed teacher: ACCEL search with Frontier score.
for seed in 0 1 2; do
  python examples/maze_frontier_accel.py \
    --use_accel \
    --score_function frontier \
    --frontier_floor 0.25 \
    --frontier_maxmc_weight 0.15 \
    --seed "$seed" \
    --run_name frontier_accel \
    --checkpoint_save_interval 17
done
