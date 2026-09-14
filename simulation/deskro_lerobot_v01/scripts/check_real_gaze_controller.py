from __future__ import annotations

from pathlib import Path
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "envhub"))
sys.path.insert(0, str(ROOT / "control"))

from deskro_env import DeskroV1Env
from real_gaze_controller import RealGazeController


# These are software behavior targets, not claims about the raw SG90 maximum.
YAW_SERVO_MAX_SPEED = 60.0
PITCH_SERVO_MAX_SPEED = 45.0
YAW_SERVO_MAX_ACCEL = 180.0
PITCH_SERVO_MAX_ACCEL = 130.0


env = DeskroV1Env(
    episode_length=100000,
    frame_skip=5,
    domain_randomization=False,
    model_filename="model.xml",
)

# Make the simulation use the same conservative motion profile that will later
# be implemented on the ESP32.
env._servo_max_speed_deg_s[:] = np.array(
    [YAW_SERVO_MAX_SPEED, PITCH_SERVO_MAX_SPEED],
    dtype=np.float64,
)
env._servo_max_accel_deg_s2[:] = np.array(
    [YAW_SERVO_MAX_ACCEL, PITCH_SERVO_MAX_ACCEL],
    dtype=np.float64,
)

controller = RealGazeController()
controller.reset(0.0, 0.0)

# A set of targets that remain inside the provisional real V1 neck limits.
sequence = [
    ("CENTER",       0.0,   0.0, 3.0),
    ("RIGHT-UP",    30.0,   8.0, 4.0),
    ("LEFT",       -35.0,   4.0, 4.5),
    ("RIGHT-DOWN",  25.0, -10.0, 4.5),
    ("LEFT-UP",    -25.0,  15.0, 4.5),
    ("CENTER",       0.0,   0.0, 4.0),
]

obs, _ = env.reset(seed=4000)

# We control the target marker ourselves for this engineering test.
phase_index = 0
label, raw_yaw, raw_pitch, phase_seconds = sequence[phase_index]
controller.set_target(raw_yaw, raw_pitch, attention_switch=True)

env._target_yaw_deg = raw_yaw
env._target_pitch_deg = raw_pitch
env._update_target_marker()
mujoco.mj_forward(env.model, env.data)

phase_started = time.monotonic()
last_print = 0.0
control_dt = float(env.model.opt.timestep) * env.frame_skip

print("DESKRO REAL-STYLE GAZE CONTROLLER")
print("=================================")
print("NO PPO / NO RL")
print("Temporary V1 software limits:")
print("  yaw   -45 .. +45 deg")
print("  pitch -15 .. +20 deg")
print("Friendly servo profile:")
print("  yaw   60 deg/s, 180 deg/s^2")
print("  pitch 45 deg/s, 130 deg/s^2")
print()
print("Orange sphere = gaze target.")
print()

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    while viewer.is_running():
        now = time.monotonic()

        if now - phase_started >= phase_seconds:
            phase_index = (phase_index + 1) % len(sequence)
            label, raw_yaw, raw_pitch, phase_seconds = sequence[phase_index]

            controller.set_target(
                raw_yaw,
                raw_pitch,
                attention_switch=True,
            )

            env._target_yaw_deg = raw_yaw
            env._target_pitch_deg = raw_pitch
            env._update_target_marker()
            mujoco.mj_forward(env.model, env.data)

            phase_started = now

        cmd_yaw, cmd_pitch = controller.update(control_dt)

        # DeskroV1Env action interface is normalized.
        # We use it only as a transport into the existing servo trajectory layer.
        action = np.array(
            [
                cmd_yaw / 60.0,
                cmd_pitch / 25.0,
                0.0,
            ],
            dtype=np.float32,
        )

        obs, _, _, _, info = env.step(action)
        viewer.sync()

        if now - last_print >= 0.5:
            print(
                f"{label:10s} "
                f"target=({raw_yaw:+5.1f},{raw_pitch:+5.1f}) "
                f"gaze=({cmd_yaw:+5.1f},{cmd_pitch:+5.1f}) "
                f"servo=({info['servo_yaw_command_deg']:+5.1f},"
                f"{info['servo_pitch_command_deg']:+5.1f}) "
                f"joint=({info['head_yaw_deg']:+5.1f},"
                f"{info['head_pitch_deg']:+5.1f}) "
                f"react={int(controller.reacting)}"
            )
            last_print = now

        time.sleep(control_dt)

env.close()
