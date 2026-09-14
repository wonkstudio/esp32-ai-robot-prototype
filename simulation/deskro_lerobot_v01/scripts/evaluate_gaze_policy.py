from pathlib import Path
import sys
import numpy as np
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "envhub"))
from gaze_env import GazeTrackingDeskroEnv

model_path = ROOT / "outputs" / "deskro_gaze_30m.zip"
if not model_path.exists():
    raise SystemExit(f"Model not found: {model_path}")

model = PPO.load(model_path)

def run(randomized, episodes=100, seed_base=22000):
    success, yaw_err, pitch_err, yaw_speed, pitch_speed = [], [], [], [], []

    for i in range(episodes):
        # Lightweight model keeps sequential evaluation memory small.
        env = GazeTrackingDeskroEnv(
            domain_randomization=randomized,
            model_filename="model_train.xml",
        )
        obs, _ = env.reset(seed=seed_base+i)
        done = False
        info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc

        env.close()

        success.append(float(info.get("is_success", False)))
        yaw_err.append(float(info["raw_yaw_error_deg"]))
        pitch_err.append(float(info["raw_pitch_error_deg"]))
        yaw_speed.append(abs(float(info["yaw_velocity_deg_s"])))
        pitch_speed.append(abs(float(info["pitch_velocity_deg_s"])))

    return {
        "episodes": episodes,
        "success_rate": float(np.mean(success)),
        "final_yaw_error_mean_deg": float(np.mean(yaw_err)),
        "final_pitch_error_mean_deg": float(np.mean(pitch_err)),
        "final_yaw_speed_mean_deg_s": float(np.mean(yaw_speed)),
        "final_pitch_speed_mean_deg_s": float(np.mean(pitch_speed)),
    }

print("GAZE NOMINAL")
print(run(False, 100, 22000))
print()
print("GAZE ROBUSTNESS")
print(run(True, 100, 32000))
