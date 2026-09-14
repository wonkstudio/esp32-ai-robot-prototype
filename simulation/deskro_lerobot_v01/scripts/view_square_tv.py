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

print("Cute square-TV exterior preview")
print("Physics, joints, actuator names, and learning interface are unchanged.")

with mujoco.viewer.launch_passive(model, data) as viewer:
    started = time.monotonic()

    while viewer.is_running():
        t = time.monotonic() - started
        yaw_deg = 18.0 * math.sin(2.0 * math.pi * t / 14.0)
        pitch_deg = 8.0 * math.sin(2.0 * math.pi * t / 10.0)

        data.ctrl[yaw_act] = math.radians(yaw_deg)
        data.ctrl[pitch_act] = math.radians(pitch_deg)

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
