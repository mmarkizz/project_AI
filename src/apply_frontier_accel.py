#!/usr/bin/env python3
"""Create examples/maze_frontier_accel.py from upstream JaxUED maze_plr.py.

Only teacher-side level scoring is changed. The original file is never modified.
"""
from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit("usage: apply_frontier_accel.py <path/to/examples/maze_plr.py>")

src = Path(sys.argv[1])
text = src.read_text()

old_score = '''def compute_score(config, dones, values, max_returns, advantages):
    if config['score_function'] == "MaxMC":
        return max_mc(dones, values, max_returns)
    elif config['score_function'] == "pvl":
        return positive_value_loss(dones, advantages)
    else:
        raise ValueError(f"Unknown score function: {config['score_function']}")
'''

new_score = '''def frontier_score(dones, rewards, values, max_returns, advantages, floor=0.25, maxmc_weight=0.15):
    """Budget-neutral teacher score for Frontier-ACCEL.

    This intentionally uses only signals from the rollout that JaxUED already
    collected. It is inspired by SFL's learning-frontier criterion and the
    negative-advantage direction investigated by recent UED work, but is not
    claimed to be the exact MNA estimator.
    """
    solved_ever = (max_returns > 0).astype(jnp.float32)

    # Hard-but-plausibly-learnable: outcome worse than current value expectation.
    negative_surprise = jnp.maximum(-advantages, 0.0).mean(axis=0)

    # AutoReplayWrapper can complete multiple attempts of the same level in one
    # rollout. Positive reward marks a successful Maze completion.
    completed = dones.astype(jnp.float32).sum(axis=0)
    successes = (rewards > 0).astype(jnp.float32).sum(axis=0)
    p = jnp.where(completed > 0, successes / jnp.maximum(completed, 1.0), 0.0)
    p = jnp.clip(p, 0.0, 1.0)
    learnability = 4.0 * p * (1.0 - p)
    frontier_gate = floor + (1.0 - floor) * learnability

    # A small MaxMC fallback retains ranking information when a long maze yields
    # very few terminations inside the rollout. Per-batch normalization keeps it
    # on a comparable scale without changing the student or rollout budget.
    mmc = jnp.maximum(max_mc(dones, values, max_returns), 0.0)
    ns_scale = negative_surprise.mean() + 1e-8
    mmc_scale = mmc.mean() + 1e-8
    mmc_rescaled = mmc * (ns_scale / mmc_scale)

    score = (1.0 - maxmc_weight) * negative_surprise * frontier_gate + maxmc_weight * mmc_rescaled
    return solved_ever * score


def compute_score(config, dones, rewards, values, max_returns, advantages):
    if config['score_function'] == "MaxMC":
        return max_mc(dones, values, max_returns)
    elif config['score_function'] == "pvl":
        return positive_value_loss(dones, advantages)
    elif config['score_function'] == "frontier":
        return frontier_score(
            dones, rewards, values, max_returns, advantages,
            floor=config["frontier_floor"],
            maxmc_weight=config["frontier_maxmc_weight"],
        )
    else:
        raise ValueError(f"Unknown score function: {config['score_function']}")
'''

if old_score not in text:
    raise SystemExit("Expected upstream compute_score block not found. Refusing to patch a different JaxUED revision.")
text = text.replace(old_score, new_score, 1)

old_call = "scores = compute_score(config, dones, values, max_returns, advantages)"
if text.count(old_call) != 3:
    raise SystemExit(f"Expected 3 teacher score calls, found {text.count(old_call)}")
text = text.replace(old_call, "scores = compute_score(config, dones, rewards, values, max_returns, advantages)")

old_arg = 'group.add_argument("--score_function", type=str, default="MaxMC", choices=["MaxMC", "pvl"])'
new_arg = '''group.add_argument("--score_function", type=str, default="MaxMC", choices=["MaxMC", "pvl", "frontier"])
    group.add_argument("--frontier_floor", type=float, default=0.25,
                       help="teacher-only floor on the SFL-style learnability gate")
    group.add_argument("--frontier_maxmc_weight", type=float, default=0.15,
                       help="teacher-only fallback weight for normalized MaxMC")'''
if old_arg not in text:
    raise SystemExit("Expected score_function argparse declaration not found")
text = text.replace(old_arg, new_arg, 1)

out = src.with_name("maze_frontier_accel.py")
out.write_text(text)
print(f"wrote {out}")
