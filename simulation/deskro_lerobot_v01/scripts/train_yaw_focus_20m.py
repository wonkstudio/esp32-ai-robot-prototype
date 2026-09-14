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
from yaw_focus_env import YawFocusGazeEnv


def find_source():
    candidates = [
        ROOT / "outputs" / "deskro_gaze_30m.zip",
        ROOT / "outputs" / "deskro_pitch_settling_15m.zip",
        ROOT / "outputs" / "best_pitch_only_20m" / "best_model.zip",
    ]
    for path in candidates:
        if path.exists():
            return path
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
        if now - self.last_report >= 300.0:
            print(
                f"[wall-clock] elapsed={elapsed/60:.1f}min "
                f"remaining={max(self.seconds-elapsed, 0)/60:.1f}min "
                f"timesteps={self.num_timesteps}"
            )
            self.last_report = now
        return elapsed < self.seconds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=20.0)
    parser.add_argument("--n-envs", type=int, default=4)
    args = parser.parse_args()

    source = find_source()
    if not source.exists():
        raise SystemExit(f"Source policy not found: {source}")

    print("Source policy:", source)
    print("Curriculum: YAW FOCUS + pitch retention")
    print("Yaw targets: mostly 12~45 deg left/right")
    print("Pitch targets: -6~+6 deg")
    print("Parallel lightweight environments:", args.n_envs)

    env = make_vec_env(
        lambda: Monitor(
            YawFocusGazeEnv(
                domain_randomization=False,
                model_filename="model_train.xml",
            )
        ),
        n_envs=args.n_envs,
        seed=2800,
    )

    model = PPO.load(source, env=env)

    lr = 5e-5
    model.learning_rate = lr
    model.lr_schedule = get_schedule_fn(lr)
    for group in model.policy.optimizer.param_groups:
        group["lr"] = lr

    checkpoint_dir = ROOT / "outputs" / "checkpoints_yaw_focus_20m"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        CheckpointCallback(
            save_freq=max(250_000 // args.n_envs, 1),
            save_path=str(checkpoint_dir),
            name_prefix="deskro_yaw_focus",
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

    out = ROOT / "outputs" / "deskro_yaw_focus_20m"
    model.save(str(out))
    env.close()

    print()
    print("Yaw-focus curriculum finished.")
    print("Saved:", str(out) + ".zip")
    print("Watch: .\\39_watch_yaw_focus.bat")
    print("Evaluate: .\\40_eval_yaw_focus.bat")


if __name__ == "__main__":
    main()
