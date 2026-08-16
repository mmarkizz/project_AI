# Frontier-ACCEL: research report

## 1. Question

Can we improve ACCEL's zero-shot transfer in JaxUED Maze by changing only the teacher-side estimate of which levels are worth replaying and mutating, while keeping the PPO+LSTM student and full interaction budget fixed?

The organiser reference solve rates over three seeds are DR **0.58±0.16**, PLR⊥ **0.44±0.08**, and ACCEL **0.67±0.12**. The real target is therefore ACCEL.

## 2. Literature-derived diagnosis

PAIRED motivates regret rather than raw difficulty: useful environments should be difficult relative to the current agent while remaining feasible. PLR turns this into replay-based curriculum selection and Robust PLR avoids policy updates on uncurated new levels. ACCEL preserves that structure and mutates promising replay levels, exploiting locality in the level space.

The weak point is the score. *No Regrets* (SFL) shows that standard UED regret proxies such as MaxMC/PVL can be poor indicators of learnability, especially in long partially observable mazes. Its central alternative is `p(1-p)`: success probability `p` supplies evidence that useful successful experience can occur, while `1-p` represents remaining room to improve. The score peaks at intermediate success.

Recent work on negative-advantage UED argues that a level on which the policy performs worse than its own expectation is a useful complementary difficulty signal. The implementation here uses a deliberately simple **negative-surprise** statistic from the already-computed PPO GAE; it does **not** claim to reproduce the exact MNA estimator from DEGen.

## 3. Proposed method

Frontier-ACCEL retains ACCEL's generator, replay schedule, buffer, staleness mechanism, mutation operator, and student updates. Only level ranking changes.

For each environment in a rollout:

1. `solved_ever = 1[max_return > 0]`.
2. `negative_surprise = mean_t max(-A_t, 0)` where `A_t` is the unchanged PPO GAE.
3. `p = successes / completed_attempts` from the same AutoReplay rollout.
4. `learnability = 4 p (1-p)`.
5. `frontier_gate = f + (1-f) learnability`, with precommitted `f=0.25`.
6. Compute the existing MaxMC score and rescale it to the batch mean of negative surprise.
7. Final score:

`solved_ever * [(1-w) * negative_surprise * frontier_gate + w * MaxMC_rescaled]`

with precommitted `w=0.15`.

### Why the solvability mask?

A critic can be badly wrong on an impossible/unlucky level. Without evidence that positive return has ever occurred, negative surprise alone cannot distinguish productive difficulty from impossibility.

### Why not pure `p(1-p)`?

A long Maze may complete too few attempts in a 256-step rollout for a reliable one-window success estimate. The floor preserves a difficulty signal even when `p` is noisy.

### Why retain a small MaxMC component?

It is a conservative fallback for sparse termination windows and makes this first implementation less brittle. Its weight is deliberately small so it cannot dominate the frontier signal.

### Why no extra SFL candidate rollouts?

The assignment fixes the environment-step budget. Frontier-ACCEL extracts all statistics from interactions ACCEL already performs, so it does not gain by secretly spending a larger environment budget.

## 4. Hypotheses

H1: Compared with MaxMC-ACCEL, Frontier-ACCEL will spend less replay priority on mastered levels.

H2: The gain should be largest on long-horizon held-out mazes, where value-based regret approximations are most fragile.

H3: Removing the solvability mask will increase attraction to never-solved levels and hurt transfer.

H4: Setting `frontier_floor=1` removes the learnability preference and should reduce the method to solvability-masked negative-surprise ranking plus the MaxMC fallback.

H5: Setting `frontier_maxmc_weight=1` should move behavior back toward the MaxMC baseline.

## 5. Experimental protocol

### Baseline reproduction

Run DR, PLR⊥, and ACCEL for seeds 0,1,2 using upstream student defaults and 30,000 updates. ACCEL reproduction should be checked against the organiser's 0.67±0.12 before interpreting the proposed method.

### Main experiment

Run Frontier-ACCEL for the same seeds and budget. Save checkpoints at interval 17. Evaluate only after training on the eight built-in dev levels.

### No dev leakage

The eight named dev levels are forbidden for training and hyperparameter selection. The values `f=0.25` and `w=0.15` are precommitted engineering choices. If further tuning is necessary, use a separately serialized validation suite sampled from independent generator seeds.

### Ablations

After the main result, the most informative teacher-only ablations are:

- `frontier_floor=1.0` (remove learnability modulation);
- `frontier_maxmc_weight=0.0` (remove fallback);
- no solvability mask;
- pure MaxMC ACCEL.

Short runs may be used only to catch instability or implementation bugs. Final method ordering must be based on the full budget because curriculum methods can cross during training.

## 6. Required result table

Do not replace `PENDING` until the corresponding full GPU run has actually completed.

| Method | Seed 0 | Seed 1 | Seed 2 | Mean ± std |
|---|---:|---:|---:|---:|
| DR (organiser reference) | — | — | — | 0.58 ± 0.16 |
| PLR⊥ (organiser reference) | — | — | — | 0.44 ± 0.08 |
| ACCEL (organiser reference) | — | — | — | 0.67 ± 0.12 |
| DR reproduction | PENDING | PENDING | PENDING | PENDING |
| PLR⊥ reproduction | PENDING | PENDING | PENDING | PENDING |
| ACCEL reproduction | PENDING | PENDING | PENDING | PENDING |
| Frontier-ACCEL | PENDING | PENDING | PENDING | PENDING |

## 7. Per-level analysis

For each of the eight dev levels, report solve rate by method and seed. The analysis should explicitly ask:

- Is an aggregate win broad or driven by one level?
- Does the method help the Labyrinth family more than StandardMaze?
- Are any gains accompanied by higher variance?
- Does the method fail on short/easy mazes because it moves away from mastered levels too aggressively?

## 8. Teacher diagnostics

The next logging patch should record, for training/replay batches only:

- fraction ever solved;
- fraction with zero completed attempts in the rollout;
- empirical `p` and `4p(1-p)`;
- mean negative surprise;
- MaxMC contribution;
- structural statistics already available in JaxUED (wall count / shortest path where available).

These diagnostics provide a mechanistic explanation without touching evaluation levels.

## 9. Failure modes and next iteration

If Frontier-ACCEL fails while reducing mastered-level replay, the likely bottleneck is noisy one-window `p`. The next version should store a Beta-smoothed historical success estimate in `LevelSampler.level_extra`.

If it over-selects long never-ending episodes, increase reliance on solvability/history rather than changing PPO.

If MaxMC fallback dominates, lower `w`; if the score becomes too sparse, raise the frontier floor. These choices must be validated away from the official dev levels.

A transfer-aware Co-Learnability component inspired by TRACED is a promising later extension, but it is intentionally excluded from v1 because naive probe rollouts would consume additional environment interactions and make the budget comparison ambiguous.

## 10. Integrity / current status

The implementation does not alter `ActorCritic`, PPO hyperparameters, or the evaluation level set. The full-run scripts request the required checkpoint cadence. This repository does **not** claim numerical wins or contain fake checkpoints before actual GPU training. Once runs are complete, the measured tables, plots, checkpoint directories, and failure analysis should replace the PENDING section above.
