# DESKRO v0.1.0 — Simulation Baseline

This release marks the first complete simulation and head-control baseline for DESKRO.

## Included

- ESP32 + OLED + dual-SG90 hardware validation
- MuJoCo 2-DOF head simulation
- stabilized head/neck physics
- lightweight training model and visual viewer model
- PPO gaze-tracking experiments
- yaw-focused curriculum
- overnight settling experiments
- policy action diagnostics
- deterministic real-style gaze controller
- ESP32 reference gaze-control firmware

## RL evaluation result

Best overnight checkpoint:

**Nominal, 100 episodes**
- success rate: 0.21
- yaw error: 0.82°
- pitch error: 0.73°
- yaw speed: 7.73°/s
- pitch speed: 3.54°/s

**Robustness, 100 episodes**
- success rate: 0.13
- yaw error: 1.33°
- pitch error: 0.55°
- yaw speed: 10.13°/s
- pitch speed: 4.76°/s

These are evaluation snapshots, not training-curve values.

## Important diagnostic result

A fixed-target diagnostic revealed:

- yaw PPO target peak-to-peak: 120.000°
- pitch PPO target peak-to-peak: 40.495°

The policy achieved accurate tracking while producing large alternating low-level commands.

This motivated a design change:

- deterministic control for low-level servo motion
- learned policies reserved for higher-level attention and behavior

## Real-style gaze controller

Current provisional software limits:

- yaw range: -45° to +45°
- pitch range: -15° to +20°
- yaw speed limit: 60°/s
- pitch speed limit: 45°/s

Simulation test summary:

- 7 / 7 tests settled
- moving-target mean settle time: ~1.04 s
- worst yaw overshoot: 0.0445°
- worst pitch overshoot: 0.0355°

## Next

Physical pan/tilt assembly and sim-to-real calibration.
