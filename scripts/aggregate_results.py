#!/usr/bin/env python3
"""Inspect JaxUED result archives without inventing missing metrics."""
from pathlib import Path
import sys
import numpy as np

root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
files = sorted(root.rglob("*.npz"))
if not files:
    raise SystemExit(f"No .npz result files found under {root}")

for path in files:
    z = np.load(path, allow_pickle=True)
    print(f"\n{path}")
    print("keys:", list(z.keys()))
    for key in ("cum_rewards", "eval_returns", "returns"):
        if key in z:
            x = np.asarray(z[key])
            print(key, "shape=", x.shape, "mean=", float(x.mean()), "positive_fraction=", float((x > 0).mean()))
