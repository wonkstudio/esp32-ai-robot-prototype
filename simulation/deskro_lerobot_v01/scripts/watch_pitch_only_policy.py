from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import mujoco.viewer
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "envhub"))

from pitch_only_env import PitchOnlyDeskroEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=411)
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")

    model = PPO.load(args.model)
    env = PitchOnlyDeskroEnv(
        episode_length=160,
        domain_randomization=False,
        curriculum=False,
        full_range_deg=18.0,
    )

    seed = args.seed
    obs, _ = env.reset(seed=seed)

    print("Orange target moves only up/down.")
    print("Goal: pitch error < 2.5 deg and settle without oscillation.")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        step = 0
        while viewer.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            viewer.sync()

            if step % 10 == 0:
                print(
                    f"step={step:03d} "
                    f"pitch={info['head_pitch_deg']:+5.1f} "
                    f"target={info['target_pitch_deg']:+5.1f} "
                    f"error={info['pitch_error_deg']:4.1f} "
                    f"velocity={info['pitch_velocity_deg_s']:+6.1f} "
                    f"success={info.get('is_success', False)}"
                )

            time.sleep(env.model.opt.timestep * env.frame_skip)
            step += 1

            if terminated or truncated:
                time.sleep(0.5)
                seed += 1
                obs, _ = env.reset(seed=seed)
                step = 0

    env.close()


if __name__ == "__main__":
    main()
