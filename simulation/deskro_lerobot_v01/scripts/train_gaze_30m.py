from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "envhub"))
from gaze_env import GazeTrackingDeskroEnv


def find_source():
    candidates = [
        ROOT / "outputs" / "deskro_pitch_settling_15m.zip",
        ROOT / "outputs" / "best_pitch_settling_15m" / "best_model.zip",
        ROOT / "outputs" / "best_pitch_only_20m" / "best_model.zip",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


class WallClockStop(BaseCallback):
    def __init__(self, seconds):
        super().__init__(verbose=1)
        self.seconds = float(seconds)
        self.started = 0.0
        self.last_report = 0.0

    def _on_training_start(self):
        self.started = time.monotonic()
        self.last_report = self.started

    def _on_step(self):
        now = time.monotonic()
        elapsed = now - self.started
        if now - self.last_report >= 300:
            print(
                f"[wall-clock] elapsed={elapsed/60:.1f}min "
                f"remaining={max(self.seconds-elapsed,0)/60:.1f}min "
                f"timesteps={self.num_timesteps}"
            )
            self.last_report = now
        return elapsed < self.seconds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=30.0)
    parser.add_argument("--n-envs", type=int, default=4)
    args = parser.parse_args()

    source = find_source()
    if not source.exists():
        raise SystemExit(f"Source policy not found: {source}")

    print("Source policy:", source)
    print("Training model: lightweight model_train.xml")
    print("Parallel virtual DESKRO robots:", args.n_envs)
    print("Task: integrated yaw + pitch gaze tracking")
    print("Target reaction delay: 0.15~0.30 sec")

    env = make_vec_env(
        lambda: Monitor(
            GazeTrackingDeskroEnv(
                domain_randomization=False,
                model_filename="model_train.xml",
            )
        ),
        n_envs=args.n_envs,
        seed=1270,
    )

    model = PPO.load(source, env=env)
    lr = 7e-5
    model.learning_rate = lr
    model.lr_schedule = get_schedule_fn(lr)
    for group in model.policy.optimizer.param_groups:
        group["lr"] = lr

    checkpoint_dir = ROOT / "outputs" / "checkpoints_gaze_30m"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        CheckpointCallback(
            save_freq=max(250_000 // args.n_envs, 1),
            save_path=str(checkpoint_dir),
            name_prefix="deskro_gaze",
            verbose=1,
        ),
        WallClockStop(args.minutes * 60.0),
    ]

    model.learn(
        total_timesteps=100_000_000,
        callback=callbacks,
        reset_num_timesteps=False,
        log_interval=1,
    )

    out = ROOT / "outputs" / "deskro_gaze_30m"
    model.save(str(out))
    env.close()

    print()
    print("Gaze fine-tuning finished.")
    print("Saved:", str(out) + ".zip")
    print("Watch with: 36_watch_gaze_policy.bat")
    print("Evaluate with: 37_eval_gaze_policy.bat")


if __name__ == "__main__":
    main()
