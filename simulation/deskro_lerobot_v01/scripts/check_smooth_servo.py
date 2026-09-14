from __future__ import annotations

from pathlib import Path
import sys
import time
import numpy as np
import mujoco.viewer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "envhub"))
from deskro_env import DeskroV1Env

env = DeskroV1Env(
    episode_length=100000,
    domain_randomization=False,
)
obs, _ = env.reset(seed=123)

# action[1] maps -1..1 to -25..25 deg.
sequence = [
    ("CENTER", 0.0, 3.0),
    ("UP", 15.0, 5.0),
    ("CENTER", 0.0, 4.0),
    ("DOWN", -15.0, 5.0),
    ("CENTER", 0.0, 4.0),
]

print("Smooth-servo test. NO PPO policy is used.")
print("Expected: gradual acceleration, gradual deceleration, then hold.")

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    index = 0
    phase_start = time.monotonic()
    last_report = 0.0

    while viewer.is_running():
        label, desired_pitch_deg, seconds = sequence[index]
        action = np.array(
            [0.0, desired_pitch_deg / 25.0, -1.0],
            dtype=np.float32,
        )
        obs, _, _, _, info = env.step(action)
        viewer.sync()

        now = time.monotonic()
        if now - last_report >= 0.5:
            print(
                f"{label:6s} "
                f"joint={info['head_pitch_deg']:+6.2f}deg "
                f"joint_speed={obs['agent_pos'][3]:+7.2f}deg/s "
                f"servo_cmd={info['servo_pitch_command_deg']:+6.2f}deg "
                f"cmd_speed={info['servo_pitch_command_speed_deg_s']:+6.2f}deg/s"
            )
            last_report = now

        if now - phase_start >= seconds:
            index = (index + 1) % len(sequence)
            phase_start = now

        time.sleep(env.model.opt.timestep * env.frame_skip)

env.close()
