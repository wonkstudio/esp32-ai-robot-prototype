from pathlib import Path
import sys, time

import mujoco.viewer
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "envhub"))
from gaze_env import GazeTrackingDeskroEnv

model_path = ROOT / "outputs" / "deskro_gaze_30m.zip"
if not model_path.exists():
    raise SystemExit(f"Model not found: {model_path}")

model = PPO.load(model_path)

# Pretty square-TV model only for one visual environment.
env = GazeTrackingDeskroEnv(
    domain_randomization=False,
    model_filename="model.xml",
)

seed = 1800
obs, info = env.reset(seed=seed)

print("Orange sphere = real target.")
print("Robot first reacts after a short delay, then follows smoothly.")

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    step = 0
    while viewer.is_running():
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        viewer.sync()

        if step % 10 == 0:
            print(
                f"head=({info['head_yaw_deg']:+5.1f},"
                f"{info['head_pitch_deg']:+5.1f}) "
                f"raw=({info['raw_target_yaw_deg']:+5.1f},"
                f"{info['raw_target_pitch_deg']:+5.1f}) "
                f"seen=({info['perceived_target_yaw_deg']:+5.1f},"
                f"{info['perceived_target_pitch_deg']:+5.1f}) "
                f"err=({info['raw_yaw_error_deg']:4.1f},"
                f"{info['raw_pitch_error_deg']:4.1f}) "
                f"speed=({info['yaw_velocity_deg_s']:+5.1f},"
                f"{info['pitch_velocity_deg_s']:+5.1f})"
            )

        time.sleep(env.model.opt.timestep * env.frame_skip)
        step += 1

        if term or trunc:
            seed += 1
            time.sleep(0.5)
            obs, info = env.reset(seed=seed)
            step = 0
