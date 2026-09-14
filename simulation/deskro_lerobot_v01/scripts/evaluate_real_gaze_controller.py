from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "envhub"))
sys.path.insert(0, str(ROOT / "control"))

from deskro_env import DeskroV1Env
from real_gaze_controller import RealGazeController


YAW_SERVO_MAX_SPEED = 60.0
PITCH_SERVO_MAX_SPEED = 45.0
YAW_SERVO_MAX_ACCEL = 180.0
PITCH_SERVO_MAX_ACCEL = 130.0

TESTS = [
    ("right_up",    30.0,   8.0),
    ("left",       -35.0,   4.0),
    ("right_down",  25.0, -10.0),
    ("left_up",    -25.0,  15.0),
    ("edge_yaw",    45.0,   0.0),
    ("edge_pitch",   0.0,  20.0),
    ("center",       0.0,   0.0),
]


def run_case(name, yaw_target, pitch_target):
    env = DeskroV1Env(
        episode_length=100000,
        frame_skip=5,
        domain_randomization=False,
        model_filename="model_train.xml",
    )

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

    obs, _ = env.reset(seed=5000)

    env._target_yaw_deg = float(yaw_target)
    env._target_pitch_deg = float(pitch_target)
    controller.set_target(
        yaw_target,
        pitch_target,
        attention_switch=True,
    )

    dt = float(env.model.opt.timestep) * env.frame_skip
    elapsed = 0.0
    settled_for = 0.0
    settle_time = None

    max_yaw_speed = 0.0
    max_pitch_speed = 0.0
    max_yaw_overshoot = 0.0
    max_pitch_overshoot = 0.0

    info = {}

    for _ in range(int(7.0 / dt)):
        cmd_yaw, cmd_pitch = controller.update(dt)

        action = np.array(
            [
                cmd_yaw / 60.0,
                cmd_pitch / 25.0,
                0.0,
            ],
            dtype=np.float32,
        )

        obs, _, _, _, info = env.step(action)
        elapsed += dt

        yaw = float(info["head_yaw_deg"])
        pitch = float(info["head_pitch_deg"])

        # DeskroV1Env stores joint velocity in the observation:
        # agent_pos = [yaw, pitch, yaw_vel, pitch_vel, ...]
        agent_pos = np.asarray(obs["agent_pos"], dtype=np.float64)
        yaw_speed_signed = float(agent_pos[2])
        pitch_speed_signed = float(agent_pos[3])

        yaw_speed = abs(yaw_speed_signed)
        pitch_speed = abs(pitch_speed_signed)

        max_yaw_speed = max(max_yaw_speed, yaw_speed)
        max_pitch_speed = max(max_pitch_speed, pitch_speed)

        # Overshoot beyond target in the original direction of travel.
        if yaw_target > 0:
            max_yaw_overshoot = max(
                max_yaw_overshoot,
                yaw - yaw_target,
            )
        elif yaw_target < 0:
            max_yaw_overshoot = max(
                max_yaw_overshoot,
                yaw_target - yaw,
            )

        if pitch_target > 0:
            max_pitch_overshoot = max(
                max_pitch_overshoot,
                pitch - pitch_target,
            )
        elif pitch_target < 0:
            max_pitch_overshoot = max(
                max_pitch_overshoot,
                pitch_target - pitch,
            )

        yaw_error = abs(yaw - yaw_target)
        pitch_error = abs(pitch - pitch_target)

        if (
            yaw_error < 1.0
            and pitch_error < 1.0
            and yaw_speed < 3.0
            and pitch_speed < 3.0
        ):
            settled_for += dt
            if settled_for >= 0.30 and settle_time is None:
                settle_time = elapsed - 0.30
        else:
            settled_for = 0.0

    yaw = float(info["head_yaw_deg"])
    pitch = float(info["head_pitch_deg"])
    env.close()

    return {
        "name": name,
        "target_yaw_deg": yaw_target,
        "target_pitch_deg": pitch_target,
        "final_yaw_error_deg": abs(yaw - yaw_target),
        "final_pitch_error_deg": abs(pitch - pitch_target),
        "settle_time_s": settle_time,
        "max_yaw_speed_deg_s": max_yaw_speed,
        "max_pitch_speed_deg_s": max_pitch_speed,
        "max_yaw_overshoot_deg": max(0.0, max_yaw_overshoot),
        "max_pitch_overshoot_deg": max(0.0, max_pitch_overshoot),
    }


print("DESKRO REAL-STYLE GAZE CONTROLLER EVALUATION")
print("=============================================")
print("No RL model is used.")
print()

results = [run_case(*case) for case in TESTS]

for result in results:
    print(result)

settle_times = [
    result["settle_time_s"]
    for result in results
    if result["settle_time_s"] is not None
]

print()
print("SUMMARY")
print({
    "tests": len(results),
    "settled_tests": len(settle_times),
    "mean_settle_time_s": (
        float(np.mean(settle_times)) if settle_times else None
    ),
    "worst_final_yaw_error_deg": float(
        max(result["final_yaw_error_deg"] for result in results)
    ),
    "worst_final_pitch_error_deg": float(
        max(result["final_pitch_error_deg"] for result in results)
    ),
    "observed_max_yaw_speed_deg_s": float(
        max(result["max_yaw_speed_deg_s"] for result in results)
    ),
    "observed_max_pitch_speed_deg_s": float(
        max(result["max_pitch_speed_deg_s"] for result in results)
    ),
    "worst_yaw_overshoot_deg": float(
        max(result["max_yaw_overshoot_deg"] for result in results)
    ),
    "worst_pitch_overshoot_deg": float(
        max(result["max_pitch_overshoot_deg"] for result in results)
    ),
})
