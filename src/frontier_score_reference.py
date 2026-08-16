"""NumPy reference implementation for Frontier-ACCEL's score."""
import numpy as np


def frontier_score(dones, rewards, values, max_returns, advantages, floor=0.25, maxmc_weight=0.15, maxmc=None):
    dones = np.asarray(dones, dtype=float)
    rewards = np.asarray(rewards, dtype=float)
    max_returns = np.asarray(max_returns, dtype=float)
    advantages = np.asarray(advantages, dtype=float)

    solved = (max_returns > 0).astype(float)
    negative_surprise = np.maximum(-advantages, 0.0).mean(axis=0)
    completed = dones.sum(axis=0)
    successes = (rewards > 0).astype(float).sum(axis=0)
    p = np.where(completed > 0, successes / np.maximum(completed, 1.0), 0.0)
    p = np.clip(p, 0.0, 1.0)
    learnability = 4.0 * p * (1.0 - p)
    gate = floor + (1.0 - floor) * learnability

    if maxmc is None:
        maxmc = np.zeros_like(negative_surprise)
    maxmc = np.maximum(np.asarray(maxmc, dtype=float), 0.0)
    ns_scale = negative_surprise.mean() + 1e-8
    mmc_scale = maxmc.mean() + 1e-8
    mmc_rescaled = maxmc * ns_scale / mmc_scale

    return solved * ((1.0 - maxmc_weight) * negative_surprise * gate + maxmc_weight * mmc_rescaled)
