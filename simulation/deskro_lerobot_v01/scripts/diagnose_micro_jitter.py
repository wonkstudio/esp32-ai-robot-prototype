from __future__ import annotations

import csv
from pathlib import Path
import sys
import time

import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "envhub"))

from overnight_settling_env import OvernightSettlingGazeEnv

MODEL_PATH = ROOT / "outputs" / "deskro_overnight_best.zip"
CSV_PATH = ROOT / "outputs" / "jitter_diagnostic.csv"

if not MODEL_PATH.exists():
    raise SystemExit(f"Model not found: {MODEL_PATH}")

model = PPO.load(MODEL_PATH)

# Long episode so we can stare at one fixed target for a while.
env = OvernightSettlingGazeEnv(
    episode_length=100000,
    domain_randomization=False,
    model_filename="model.xml",
)

obs, _ = env.reset(seed=132000)

# Force one repeatable, non-trivial target so the trace is easy to compare.
env._raw_target_yaw_deg = 28.0
env._raw_target_pitch_deg = 8.0
env._target_yaw_deg = 0.0
env._target_pitch_deg = 0.0
env._reaction_steps_left = 8
env._update_target_marker()
mujoco.mj_forward(env.model, env.data)
obs = env._observation()

rows = []
started = time.monotonic()
last_print = started
duration_s = 18.0

print("DESKRO MICRO-JITTER DIAGNOSTIC")
print("==============================")
print("No controller filter is active.")
print("Policy: outputs\\deskro_overnight_best.zip")
print("Fixed target: yaw +28 deg, pitch +8 deg")
print()
print("Watch the head. The script will also record:")
print("  PPO target -> servo trajectory command -> actual joint -> joint speed")
print()

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    while viewer.is_running():
        now = time.monotonic()
        elapsed = now - started
        if elapsed >= duration_s:
            break

        action, _ = model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(3)

        obs, _, _, _, info = env.step(action)
        viewer.sync()

        # Deskro policy action mapping.
        policy_yaw_target_deg = float(action[0]) * 60.0
        policy_pitch_target_deg = float(action[1]) * 25.0

        row = {
            "time_s": elapsed,
            "raw_target_yaw_deg": float(info["raw_target_yaw_deg"]),
            "raw_target_pitch_deg": float(info["raw_target_pitch_deg"]),
            "perceived_target_yaw_deg": float(info["perceived_target_yaw_deg"]),
            "perceived_target_pitch_deg": float(info["perceived_target_pitch_deg"]),
            "policy_yaw_target_deg": policy_yaw_target_deg,
            "policy_pitch_target_deg": policy_pitch_target_deg,
            "servo_yaw_command_deg": float(info["servo_yaw_command_deg"]),
            "servo_pitch_command_deg": float(info["servo_pitch_command_deg"]),
            "joint_yaw_deg": float(info["head_yaw_deg"]),
            "joint_pitch_deg": float(info["head_pitch_deg"]),
            "joint_yaw_speed_deg_s": float(info["yaw_velocity_deg_s"]),
            "joint_pitch_speed_deg_s": float(info["pitch_velocity_deg_s"]),
        }
        rows.append(row)

        if now - last_print >= 0.5:
            print(
                f"policy=({policy_yaw_target_deg:+6.2f},"
                f"{policy_pitch_target_deg:+6.2f}) "
                f"servo=({row['servo_yaw_command_deg']:+6.2f},"
                f"{row['servo_pitch_command_deg']:+6.2f}) "
                f"joint=({row['joint_yaw_deg']:+6.2f},"
                f"{row['joint_pitch_deg']:+6.2f}) "
                f"speed=({row['joint_yaw_speed_deg_s']:+6.2f},"
                f"{row['joint_pitch_speed_deg_s']:+6.2f})"
            )
            last_print = now

        time.sleep(env.model.opt.timestep * env.frame_skip)

env.close()

CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

# Analyze only the last 8 seconds, after the target filter and main movement
# have had plenty of time to settle.
arr = rows
if not arr:
    raise SystemExit("No samples collected.")

end_t = arr[-1]["time_s"]
stable = [r for r in arr if r["time_s"] >= max(0.0, end_t - 8.0)]

def stats(key):
    v = np.asarray([r[key] for r in stable], dtype=np.float64)
    return {
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "p2p": float(np.ptp(v)),
        "mean_abs": float(np.mean(np.abs(v))),
    }

metrics = {
    "policy_yaw": stats("policy_yaw_target_deg"),
    "servo_yaw": stats("servo_yaw_command_deg"),
    "joint_yaw": stats("joint_yaw_deg"),
    "yaw_speed": stats("joint_yaw_speed_deg_s"),
    "policy_pitch": stats("policy_pitch_target_deg"),
    "servo_pitch": stats("servo_pitch_command_deg"),
    "joint_pitch": stats("joint_pitch_deg"),
    "pitch_speed": stats("joint_pitch_speed_deg_s"),
}

print()
print("=" * 72)
print("LAST 8s JITTER SUMMARY")
print("=" * 72)

for axis in ("yaw", "pitch"):
    print(axis.upper())
    print(
        f"  PPO target p2p : {metrics['policy_' + axis]['p2p']:.3f} deg"
    )
    print(
        f"  Servo cmd p2p  : {metrics['servo_' + axis]['p2p']:.3f} deg"
    )
    print(
        f"  Joint p2p      : {metrics['joint_' + axis]['p2p']:.3f} deg"
    )
    print(
        f"  Mean |speed|   : {metrics[axis + '_speed']['mean_abs']:.3f} deg/s"
    )

print()
print("AUTOMATIC INTERPRETATION")
print("------------------------")

def interpret(axis):
    p = metrics["policy_" + axis]["p2p"]
    s = metrics["servo_" + axis]["p2p"]
    j = metrics["joint_" + axis]["p2p"]

    if p >= 1.2:
        return (
            f"{axis.upper()}: PPO command itself is still moving materially "
            f"near the target (policy p2p {p:.2f} deg)."
        )
    if p < 1.2 and s >= 0.9:
        return (
            f"{axis.upper()}: PPO is fairly stable, but the low-level servo "
            f"trajectory command still moves (servo p2p {s:.2f} deg)."
        )
    if s < 0.9 and j >= max(0.9, s * 1.8):
        return (
            f"{axis.upper()}: command is stable but the MuJoCo joint oscillates "
            f"more than the command. Actuator/damping is the likely source."
        )
    return (
        f"{axis.upper()}: all command layers are relatively small; the visible "
        f"motion is near the current simulation/measurement floor."
    )

print(interpret("yaw"))
print(interpret("pitch"))

print()
print("CSV saved to:")
print(CSV_PATH)
print()
print("Send the LAST 8s JITTER SUMMARY + AUTOMATIC INTERPRETATION back to ChatGPT.")
