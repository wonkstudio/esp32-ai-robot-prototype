from pathlib import Path
import sys
import numpy as np
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "envhub"))

from gaze_env import GazeTrackingDeskroEnv
from yaw_focus_env import YawFocusGazeEnv

model_path = ROOT / "outputs" / "deskro_yaw_focus_20m.zip"
if not model_path.exists():
    raise SystemExit(f"Model not found: {model_path}")

model = PPO.load(model_path)


def evaluate(env_cls, randomized, episodes, seed_base):
    env = env_cls(
        domain_randomization=randomized,
        model_filename="model_train.xml",
    )

    rows = []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed_base + i)
        done = False
        info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc

        rows.append({
            "success": float(info.get("is_success", False)),
            "yaw_error": float(info["raw_yaw_error_deg"]),
            "pitch_error": float(info["raw_pitch_error_deg"]),
            "yaw_speed": abs(float(info["yaw_velocity_deg_s"])),
            "pitch_speed": abs(float(info["pitch_velocity_deg_s"])),
        })

    env.close()

    return {
        "episodes": episodes,
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "final_yaw_error_mean_deg": float(np.mean([r["yaw_error"] for r in rows])),
        "final_pitch_error_mean_deg": float(np.mean([r["pitch_error"] for r in rows])),
        "final_yaw_speed_mean_deg_s": float(np.mean([r["yaw_speed"] for r in rows])),
        "final_pitch_speed_mean_deg_s": float(np.mean([r["pitch_speed"] for r in rows])),
    }


print("YAW-FOCUS NOMINAL")
print(evaluate(YawFocusGazeEnv, False, 100, 41000))

print()
print("FULL GAZE NOMINAL")
print(evaluate(GazeTrackingDeskroEnv, False, 100, 42000))

print()
print("FULL GAZE ROBUSTNESS")
print(evaluate(GazeTrackingDeskroEnv, True, 100, 43000))
