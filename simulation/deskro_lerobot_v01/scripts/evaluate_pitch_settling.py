from pathlib import Path
import sys
import numpy as np
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "envhub"))
from pitch_settling_env import PitchSettlingDeskroEnv

model_path = ROOT / "outputs" / "deskro_pitch_settling_15m.zip"
if not model_path.exists():
    raise SystemExit(f"Model not found: {model_path}")

model = PPO.load(model_path)

def run(randomized, episodes=100, seed_base=95000):
    succ, err, speed = [], [], []
    for i in range(episodes):
        # Create only ONE environment, close it, then create the next.
        env = PitchSettlingDeskroEnv(domain_randomization=randomized)
        obs, _ = env.reset(seed=seed_base+i)
        done = False
        info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        env.close()

        succ.append(float(info.get("is_success", False)))
        err.append(float(info["pitch_error_deg"]))
        speed.append(abs(float(info["pitch_velocity_deg_s"])))

    return {
        "episodes": episodes,
        "success_rate": float(np.mean(succ)),
        "final_pitch_error_mean_deg": float(np.mean(err)),
        "final_pitch_speed_mean_deg_s": float(np.mean(speed)),
    }

print("SETTLING NOMINAL")
print(run(False, 100, 95000))
print()
print("SETTLING ROBUSTNESS")
print(run(True, 100, 105000))
