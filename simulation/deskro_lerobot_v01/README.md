# DESKRO

**DESKRO** is an experimental desktop Physical AI robot project built around an ESP32, an expressive OLED face, and a 2-DOF pan/tilt head.

The long-term goal is to build a small desktop companion that can **look at people, react, express simple emotions, speak, listen, and gradually gain more natural behavior**.

> Current stage: simulation and low-level head-control validation.  
> The physical body has not been built yet.

---

## Planned V1 hardware

- ESP32 DevKit V1
- SSD1306 OLED
- 2 × SG90 servos for head yaw and pitch
- INMP441 I2S microphone
- MAX98357A I2S amplifier
- 3 W speaker
- 2-axis pan/tilt neck
- Custom desktop enclosure

Already validated on the real ESP32:

- serial communication
- OLED output
- multiple facial expressions
- two SG90 servos
- independent yaw / pitch control
- simultaneous two-servo motion

The INMP441 microphone is temporarily paused because the current module appears to have a header/contact issue. The amplifier and speaker have not yet been integrated.

---

## Simulation stack

DESKRO currently uses **MuJoCo** for physics simulation.

The simulation includes:

- a 2-DOF yaw / pitch neck
- head mass and inertia
- gravity compensation
- servo-like position actuators
- a square retro-TV visual model
- a lightweight training model for parallel environments

The detailed visual model and lightweight training model are kept separate to reduce memory usage during parallel training.

---

## Reinforcement-learning experiment

The first RL experiment asked a deliberately simple question:

> Can a PPO policy learn to move a simulated 2-DOF head toward a target yaw / pitch direction?

The policy observation included joint position, joint velocity, target direction, and simple robot state.

The action represented yaw target, pitch target, and face state.

### Important chart note

The charts below are **not training-loss curves and not per-timestep learning curves**.

They compare **separate nominal evaluation snapshots of different saved policies** at three development stages.

### Nominal evaluation snapshots

| Saved-policy stage | Yaw error | Pitch error | Yaw speed | Pitch speed |
|---|---:|---:|---:|---:|
| Integrated gaze | 13.44° | 0.66° | 13.35°/s | 5.15°/s |
| Yaw-focused policy | 3.65° | 1.43° | 16.99°/s | 6.71°/s |
| Overnight best checkpoint | **0.82°** | **0.73°** | **7.73°/s** | **3.54°/s** |

![Nominal evaluation error snapshots](docs/assets/rl_nominal_evaluation_error_snapshots.png)

![Nominal evaluation residual motion snapshots](docs/assets/rl_nominal_evaluation_speed_snapshots.png)

The best overnight checkpoint was also evaluated under the robustness configuration:

| Metric | Robustness result |
|---|---:|
| Episodes | 100 |
| Success rate | 0.13 |
| Yaw error | 1.33° |
| Pitch error | 0.55° |
| Yaw speed | 10.13°/s |
| Pitch speed | 4.76°/s |

Exact source values used for the figures are stored in:

```text
docs/data/rl_nominal_evaluation_snapshots.csv
```

---

## The important failure: PPO learned bang-bang control

The final tracking error looked excellent, but the simulated head still showed visible micro-jitter.

Instead of adding more filters blindly, I instrumented the control path:

```text
PPO target
    ↓
servo trajectory command
    ↓
MuJoCo joint
    ↓
joint velocity
```

The fixed-target diagnostic reported:

```text
YAW
PPO target peak-to-peak:      120.000°
Servo command peak-to-peak:     3.119°
Joint peak-to-peak:             3.145°
Mean absolute joint speed:      8.455°/s

PITCH
PPO target peak-to-peak:       40.495°
Servo command peak-to-peak:     1.207°
Joint peak-to-peak:             1.258°
Mean absolute joint speed:      2.964°/s
```

The PPO policy was therefore not producing a calm final target command.

It was effectively using large alternating commands — a form of **bang-bang control** — while the trajectory controller suppressed most of the resulting motion.

So the policy achieved high positional accuracy, but with undesirable low-level behavior.

That result changed the control architecture.

---

## Architecture after the RL experiment

Low-level neck motion is now deterministic:

```text
Camera / target estimation
          ↓
Target yaw / pitch
          ↓
Real Gaze Controller
          ↓
Speed + acceleration + braking profile
          ↓
ESP32
          ↓
SG90 yaw + pitch
```

Future learned policies will focus on higher-level behavior instead:

- who to look at
- when to look
- when to return to center
- whether to react to sound or speech
- which expression to show
- how to choose between attention targets

The goal is to separate:

```text
Learned behavior:
"What should I do?"

from

Deterministic motor control:
"How should the motor move?"
```

---

## Real-style gaze controller

The current deterministic controller uses provisional software limits:

```text
Yaw range:   -45° to +45°
Pitch range: -15° to +20°

Yaw speed limit:   60°/s
Pitch speed limit: 45°/s

Yaw acceleration limit:   180°/s²
Pitch acceleration limit: 130°/s²
```

These values are **software behavior targets**, not measured SG90 performance specifications.

They will be calibrated after the physical pan/tilt bracket and head are assembled.

The controller includes:

- short reaction delay
- bounded target movement
- acceleration limiting
- braking before the target
- overshoot prevention

### Deterministic simulation evaluation

Seven target cases were evaluated without PPO or RL.

All **7 / 7** reached the settle criterion.

One case (`center`) starts already at the requested pose, so its reported settle time is `0.0 s`.

For the **six tests that required actual movement**:

```text
Mean settle time: ~1.04 s
```

The original evaluator's mean across all seven tests, including the already-centered case, was:

```text
0.889 s
```

For that reason, the moving-target mean (~1.04 s) is the more useful motion metric.

![Moving-target settle times](docs/assets/real_gaze_moving_target_settle_times.png)

Measured simulation summary:

```text
Tests: 7
Settled: 7 / 7

Moving-target mean settle time: ~1.04 s

Worst yaw overshoot:   0.0445°
Worst pitch overshoot: 0.0355°

Observed max yaw speed:   60.963°/s
Observed max pitch speed: 45.401°/s
```

The near-zero final angle error in MuJoCo should **not** be interpreted as real SG90 accuracy.

Real hardware will introduce effects such as:

- gear backlash
- servo deadband
- power-supply variation
- mounting error
- mechanical collision limits
- head inertia

Exact per-test values are stored in:

```text
docs/data/real_gaze_controller_tests.csv
```

---

## Current reference firmware

The current ESP32 reference controller uses:

```text
Yaw SG90 signal   → GPIO 13
Pitch SG90 signal → GPIO 14
```

Example serial command:

```text
T 25 8
```

Meaning:

```text
Yaw target:   +25°
Pitch target:  +8°
```

Return to center:

```text
C
```

The firmware is still a reference implementation. Servo center offsets, direction signs, safe limits, and collision limits must be measured on the physical assembly.

---

## What the RL experiment taught me

The main result was architectural rather than simply “RL worked” or “RL failed.”

A learned policy can optimize the requested tracking metric while still producing undesirable actuator-level behavior.

For DESKRO, the current design therefore uses:

- deterministic low-level servo motion
- learned policies later for attention and behavior selection

The best RL model is still kept as an experimental artifact:

```text
outputs/deskro_overnight_best.zip
```

---

## Next steps

1. Build the physical SG90 pan/tilt neck
2. Mount the square TV-style head
3. Measure real servo center positions
4. Measure safe yaw / pitch ranges
5. Measure real movement speed and acceleration
6. Match MuJoCo parameters to the real hardware
7. Connect webcam-based face tracking
8. Convert face position into target yaw / pitch
9. Add higher-level learned attention behavior
10. Continue microphone, speaker, and conversational integration

---

## Project status

```text
ESP32 + OLED                    ✅
Two SG90 servos                 ✅
OLED expressions                ✅
MuJoCo 2-DOF neck               ✅
Neck physics stabilization      ✅
PPO tracking experiments        ✅
PPO action diagnostics          ✅
Real-style gaze controller      ✅
ESP32 gaze reference firmware   ✅

Physical pan/tilt assembly      ⏳
Real sim-to-real calibration    ⏳
Camera tracking                 ⏳
Microphone integration          ⏳
Speaker integration             ⏳
Conversation behavior           ⏳
```

---

## Milestone

The project is approaching:

**DESKRO v0.1.0 — Simulation Baseline**

This milestone freezes the first simulation and control baseline before moving into physical pan/tilt calibration and real-world perception.

---

## Why DESKRO?

I wanted to build my own little robot buddy.

The body does not exist yet, but the head-control stack now has a simulation baseline ready for the next step: real hardware.
