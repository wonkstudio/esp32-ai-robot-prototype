from pathlib import Path
import sys
import time

import mujoco.viewer
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "envhub"))
from gaze_env import GazeTrackingDeskroEnv

model_path = ROOT / "outputs" / "deskro_yaw_focus_20m.zip"
if not model_path.exists():
    raise SystemExit(f"Model not found: {model_path}")

model = PPO.load(model_path)

# Watch on the FULL integrated target range, not the easier yaw curriculum.
env = GazeTrackingDeskroEnv(
    domain_randomization=False,
    model_filename="model.xml",
)

seed = 3900
obs, _ = env.reset(seed=seed)
print("FULL gaze viewer after yaw-focused curriculum.")
print("Orange sphere = true target.")

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    step = 0
    while viewer.is_running():
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        viewer.sync()

        if step % 10 == 0:
            print(
                f"yaw err={info['raw_yaw_error_deg']:5.1f} "
                f"pitch err={info['raw_pitch_error_deg']:4.1f} "
                f"speed=({info['yaw_velocity_deg_s']:+5.1f},"
                f"{info['pitch_velocity_deg_s']:+5.1f}) "
                f"streak={info['gaze_success_streak']:02d}"
            )

        time.sleep(env.model.opt.timestep * env.frame_skip)
        step += 1

        if term or trunc:
            seed += 1
            time.sleep(0.35)
            obs, _ = env.reset(seed=seed)
            step = 0

env.close()
