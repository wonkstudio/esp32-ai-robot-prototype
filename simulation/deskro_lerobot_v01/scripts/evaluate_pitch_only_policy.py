from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "envhub"))

from pitch_only_env import PitchOnlyDeskroEnv


def evaluate(model: PPO, episodes: int, randomized: bool, seed_base: int):
    successes = []
    returns = []
    pitch_errors = []
    pitch_velocities = []

    for episode in range(episodes):
        env = PitchOnlyDeskroEnv(
            domain_randomization=randomized,
            curriculum=False,
            full_range_deg=18.0,
        )
        obs, _ = env.reset(seed=seed_base + episode)
        done = False
        total_reward = 0.0
        info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        env.close()
        successes.append(float(info.get("is_success", False)))
        returns.append(total_reward)
        pitch_errors.append(float(info.get("pitch_error_deg", np.nan)))
        pitch_velocities.append(
            abs(float(info.get("pitch_velocity_deg_s", np.nan)))
        )

    return {
        "episodes": episodes,
        "success_rate": float(np.mean(successes)),
        "mean_return": float(np.mean(returns)),
        "final_pitch_error_mean_deg": float(np.nanmean(pitch_errors)),
        "final_pitch_speed_mean_deg_s": float(
            np.nanmean(pitch_velocities)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")

    model = PPO.load(args.model)

    print("PITCH NOMINAL")
    print(evaluate(model, args.episodes, False, 51000))
    print()
    print("PITCH ROBUSTNESS")
    print(evaluate(model, args.episodes, True, 61000))


if __name__ == "__main__":
    main()
