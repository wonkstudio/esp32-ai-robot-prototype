from pathlib import Path
import sys
import time

import mujoco.viewer
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "envhub"))
from overnight_settling_env import OvernightSettlingGazeEnv

model_path = ROOT / "outputs" / "deskro_overnight_best.zip"
if not model_path.exists():
    raise SystemExit(
        "Best overnight model not found.\n"
        "Run .\\42_eval_overnight_checkpoints.bat first."
    )

model = PPO.load(model_path)

env = OvernightSettlingGazeEnv(
    domain_randomization=False,
    model_filename="model.xml",
)

seed = 70000
obs, _ = env.reset(seed=seed)

print("DESKRO overnight best candidate")
print("Orange sphere = true target.")
print("Friendly software speed caps: yaw 60 deg/s, pitch 50 deg/s.")

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    step = 0
    while viewer.is_running():
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        viewer.sync()

        if step % 10 == 0:
            print(
                f"err=({info['raw_yaw_error_deg']:4.1f},"
                f"{info['raw_pitch_error_deg']:4.1f}) "
                f"speed=({info['yaw_velocity_deg_s']:+5.1f},"
                f"{info['pitch_velocity_deg_s']:+5.1f}) "
                f"hold={int(bool(info.get('overnight_hold_now', False)))} "
                f"streak={int(info.get('gaze_success_streak', 0)):02d}"
            )

        time.sleep(env.model.opt.timestep * env.frame_skip)
        step += 1

        if term or trunc:
            seed += 1
            time.sleep(0.35)
            obs, _ = env.reset(seed=seed)
            step = 0

env.close()
