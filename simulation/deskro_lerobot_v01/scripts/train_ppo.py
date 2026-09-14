"""Train a PPO policy for DESKRO V1.

Training intentionally runs without a 3D viewer so multiple simulations can run
quickly. Checkpoints are saved during training and can be watched separately
with scripts/watch_ppo.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "envhub"))
from deskro_env import DeskroV1Env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "deskro_ppo")
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    args = parser.parse_args()

    if args.steps <= 0:
        raise SystemExit("--steps must be greater than zero")
    if args.n_envs <= 0:
        raise SystemExit("--n-envs must be greater than zero")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output.parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env = make_vec_env(
        lambda: Monitor(DeskroV1Env(domain_randomization=True)),
        n_envs=args.n_envs,
        seed=123,
    )

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        n_steps=1024,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        tensorboard_log=str(args.output.parent / "tensorboard"),
    )

    # SB3 callback frequency counts calls, while each call advances n_envs steps.
    save_freq = max(args.checkpoint_every // args.n_envs, 1)
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(checkpoint_dir),
        name_prefix="deskro_ppo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    print("Training starts headless (no 3D window).")
    print("TensorBoard logs:", args.output.parent / "tensorboard")
    print("Checkpoints:", checkpoint_dir)

    model.learn(total_timesteps=args.steps, callback=checkpoint_callback)
    model.save(str(args.output))
    env.close()

    print("saved:", str(args.output) + ".zip")
    print("watch with: 06_watch_policy.bat")


if __name__ == "__main__":
    main()
