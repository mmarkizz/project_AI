# Frontier-ACCEL — JaxUED Maze UED submission

This repository contains a **teacher-only** curriculum modification for the JaxUED Maze benchmark. The PPO+LSTM student, PPO hyperparameters, architecture, evaluation levels, and full interaction budget are left unchanged.

## Method

**Frontier-ACCEL** keeps ACCEL's replay + mutation search, but replaces the fragile single regret proxy with a frontier score computed from signals already present in the rollout:

- **solvability evidence**: a level must have achieved positive return at least once (`max_return > 0`);
- **negative-surprise difficulty**: trajectories that perform worse than the critic expected receive more priority;
- **learnability**: levels solved sometimes but not always are preferred using `4 p (1-p)`;
- **MaxMC fallback**: a small normalized MaxMC component prevents the curriculum from collapsing when a 256-step rollout contains too few completed attempts to estimate `p` reliably.

The method adds **no extra environment interactions**. This is deliberate: the assignment fixes the student-side environment-step budget at 30,000 updates (~245.76M interactions).

## Install

```bash
git clone https://github.com/DramaCow/jaxued.git
cd jaxued
pip install "jax[cuda12]==0.4.30" flax==0.8.5 chex==0.1.86 optax==0.2.3 \
  distrax==0.1.5 gymnax==0.0.8 orbax-checkpoint==0.5.3 "numpy<2" \
  wandb==0.17.5 pillow imageio
pip install --no-deps -e .
```

Clone this repository next to JaxUED (or set `SUBMISSION_DIR` explicitly), then create the modified training file:

```bash
python ../project_AI/src/apply_frontier_accel.py examples/maze_plr.py
python -m py_compile examples/maze_frontier_accel.py
```

## Smoke test

```bash
python -m pytest ../project_AI/tests -q
```

## Full training

From the JaxUED root:

```bash
SUBMISSION_DIR=../project_AI bash ../project_AI/scripts/run_full.sh
```

This launches 3 seeds of the required baselines and Frontier-ACCEL using the upstream defaults. The scripts do not override PPO/student hyperparameters and save checkpoints with `--checkpoint_save_interval 17`.

## Evaluation

```bash
SUBMISSION_DIR=../project_AI bash ../project_AI/scripts/eval_dev.sh
python ../project_AI/scripts/aggregate_results.py results
```

The eight built-in dev levels are **evaluation only**. They are not referenced by the teacher and must not be used for hyperparameter selection.

## Repository layout

- `src/apply_frontier_accel.py` — deterministic source-to-source patch against upstream `examples/maze_plr.py`.
- `src/frontier_score_reference.py` — framework-independent NumPy reference implementation.
- `tests/` — score invariants and guardrails.
- `scripts/` — full-budget training/evaluation helpers.
- `REPORT.md` — research motivation, method, ablations, and analysis protocol.
- `results/` — organiser reference numbers and, after GPU runs, measured results.
- `checkpoints/` — populated by actual full training; no fabricated weights are committed.

## Reproducibility rule

The patcher fails loudly if the upstream `maze_plr.py` structure differs from the expected JaxUED version. This is intentional: silently patching the wrong student implementation would invalidate the comparison.

## Status

Code and experiment pipeline are prepared. **Measured full-budget results and trained checkpoints must come from actual GPU runs and are never fabricated.**
