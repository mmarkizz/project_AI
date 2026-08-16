import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from frontier_score_reference import frontier_score


def test_never_solved_is_zero():
    s = frontier_score([[1]], [[0]], [[0]], [0], [[-2]], maxmc=[3])
    assert s[0] == 0


def test_intermediate_success_beats_mastered_without_fallback():
    dones = np.ones((2, 2))
    rewards = np.array([[1, 1], [0, 1]], dtype=float)  # p=.5 vs p=1
    advantages = -np.ones((2, 2))
    s = frontier_score(dones, rewards, np.zeros_like(dones), [1, 1], advantages, maxmc_weight=0)
    assert s[0] > s[1]


def test_positive_advantage_has_no_negative_surprise():
    s = frontier_score([[1]], [[1]], [[0]], [1], [[2]], maxmc_weight=0)
    assert s[0] == 0


def test_scores_are_nonnegative():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(16, 8))
    d = rng.integers(0, 2, size=(16, 8))
    r = rng.integers(0, 2, size=(16, 8))
    s = frontier_score(d, r, np.zeros_like(a), np.ones(8), a, maxmc=np.arange(8))
    assert np.all(s >= 0)
