from __future__ import annotations

import math
from pathlib import Path
import time

import mujoco
import mujoco.viewer

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "envhub" / "model.xml"

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

yaw_act = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_head_yaw"
)
pitch_act = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_head_pitch"
)
pitch_joint = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_JOINT, "joint_head_pitch"
)
pitch_qpos_adr = model.jnt_qposadr[pitch_joint]

# command name, pitch target in degrees, duration in seconds
sequence = [
    ("CENTER HOLD", 0.0, 5.0),
    ("LOOK UP", 12.0, 5.0),
    ("CENTER HOLD", 0.0, 5.0),
    ("LOOK DOWN", -12.0, 5.0),
]

print("Neck stability test")
print("The head must hold center, look up, return center, and look down.")
print("This script uses NO PPO policy.")

with mujoco.viewer.launch_passive(model, data) as viewer:
    index = 0
    phase_started = time.monotonic()
    last_report = 0.0

    while viewer.is_running():
        label, target_deg, duration = sequence[index]
        now = time.monotonic()

        data.ctrl[yaw_act] = 0.0
        data.ctrl[pitch_act] = math.radians(target_deg)

        mujoco.mj_step(model, data)
        viewer.sync()

        if now - last_report >= 1.0:
            actual_deg = math.degrees(float(data.qpos[pitch_qpos_adr]))
            print(
                f"{label:12s} target={target_deg:+5.1f} deg "
                f"actual={actual_deg:+6.2f} deg"
            )
            last_report = now

        if now - phase_started >= duration:
            index = (index + 1) % len(sequence)
            phase_started = now

        time.sleep(model.opt.timestep)
