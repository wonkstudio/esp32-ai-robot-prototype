# Source Data Notes

All numeric figures in this README pack come from evaluation logs produced during the DESKRO development session.

## RL snapshot chart

The three stages are separate saved-policy evaluations:

1. Integrated gaze — FULL GAZE NOMINAL, 100 episodes
2. Yaw-focus — FULL GAZE NOMINAL, 100 episodes
3. Overnight best — OVERNIGHT BEST - NOMINAL, 100 episodes

These values are not a continuous training curve.

## Moving-target settling mean

The real-style controller evaluator reports seven cases, including `center`.

`center` starts already at the requested pose and therefore has a settle time of `0.0 s`.

All-seven mean:
0.889285714 s

Moving-target-only mean:
1.037500000 s

The README uses the moving-target-only mean because it better represents actual head movement.

## Exact data files

- `rl_nominal_evaluation_snapshots.csv`
- `real_gaze_controller_tests.csv`
- `ppo_micro_jitter_diagnostic.csv`
