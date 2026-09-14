from pathlib import Path
import sys
import numpy as np
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


def run(randomized, episodes, seed_base):
    env = OvernightSettlingGazeEnv(
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
            done = bool(term or trunc)

        rows.append({
            "success": float(info.get("is_success", False)),
            "yaw_error": float(info["raw_yaw_error_deg"]),
            "pitch_error": float(info["raw_pitch_error_deg"]),
            "yaw_speed": abs(float(info["yaw_velocity_deg_s"])),
            "pitch_speed": abs(float(info["pitch_velocity_deg_s"])),
        })

    env.close()

    return {
        "episodes": int(episodes),
        "success_rate": float(np.mean([x["success"] for x in rows])),
        "final_yaw_error_mean_deg": float(
            np.mean([x["yaw_error"] for x in rows])
        ),
        "final_pitch_error_mean_deg": float(
            np.mean([x["pitch_error"] for x in rows])
        ),
        "final_yaw_speed_mean_deg_s": float(
            np.mean([x["yaw_speed"] for x in rows])
        ),
        "final_pitch_speed_mean_deg_s": float(
            np.mean([x["pitch_speed"] for x in rows])
        ),
    }


print("OVERNIGHT BEST - NOMINAL")
print(run(False, 100, 80000))
print()
print("OVERNIGHT BEST - ROBUSTNESS")
print(run(True, 100, 90000))
