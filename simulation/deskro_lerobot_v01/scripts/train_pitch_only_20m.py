"""20-minute pitch-only curriculum fine-tuning for DESKRO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "envhub"))

from pitch_only_env import PitchOnlyDeskroEnv


def find_resume_model() -> Path:
    candidates = [
        ROOT / "outputs" / "best_retro_finetune_1h" / "best_model.zip",
        ROOT / "outputs" / "deskro_retro_finetune_1h.zip",
        ROOT / "outputs" / "best_1h" / "best_model.zip",
        ROOT / "outputs" / "deskro_ppo_1h.zip",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def evaluate_pitch(model: PPO, episodes: int, seed_base: int = 31000):
    success = []
    errors = []
    returns = []

    for episode in range(episodes):
        env = PitchOnlyDeskroEnv(
            domain_randomization=False,
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
        success.append(float(info.get("is_success", False)))
        errors.append(float(info.get("pitch_error_deg", np.nan)))
        returns.append(total_reward)

    return {
        "success_rate": float(np.mean(success)),
        "mean_pitch_error_deg": float(np.nanmean(errors)),
        "mean_return": float(np.mean(returns)),
    }


class PitchMetricCallback(BaseCallback):
    """Save the model with best pitch success, then lowest pitch error."""

    def __init__(
        self,
        eval_every_timesteps: int,
        n_envs: int,
        episodes: int,
        save_dir: Path,
    ) -> None:
        super().__init__(verbose=1)
        self.eval_every_calls = max(eval_every_timesteps // n_envs, 1)
        self.episodes = int(episodes)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_success = -1.0
        self.best_error = float("inf")
        self.history_path = self.save_dir / "pitch_eval_history.jsonl"

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_every_calls != 0:
            return True

        metrics = evaluate_pitch(
            self.model,
            episodes=self.episodes,
            seed_base=31000 + self.n_calls,
        )
        metrics["timesteps"] = int(self.num_timesteps)

        print(
            "[pitch-eval] "
            f"steps={self.num_timesteps} "
            f"success={metrics['success_rate']:.3f} "
            f"pitch_error={metrics['mean_pitch_error_deg']:.3f} deg "
            f"return={metrics['mean_return']:.2f}"
        )

        with self.history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

        success = metrics["success_rate"]
        error = metrics["mean_pitch_error_deg"]
        better = (
            success > self.best_success + 1e-12
            or (
                abs(success - self.best_success) <= 1e-12
                and error < self.best_error
            )
        )

        if better:
            self.best_success = success
            self.best_error = error
            path = self.save_dir / "best_model"
            self.model.save(str(path))
            (self.save_dir / "best_metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print("[pitch-eval] saved new best:", str(path) + ".zip")

        return True


class StopAfterWallClockCallback(BaseCallback):
    def __init__(self, seconds: float, report_every_seconds: float = 300.0):
        super().__init__(verbose=1)
        self.seconds = float(seconds)
        self.report_every_seconds = float(report_every_seconds)
        self.started_at = 0.0
        self.last_report_at = 0.0

    def _on_training_start(self) -> None:
        self.started_at = time.monotonic()
        self.last_report_at = self.started_at

    def _on_step(self) -> bool:
        now = time.monotonic()
        elapsed = now - self.started_at

        if now - self.last_report_at >= self.report_every_seconds:
            remaining = max(self.seconds - elapsed, 0.0)
            print(
                f"[wall-clock] elapsed={elapsed / 60:.1f} min, "
                f"remaining={remaining / 60:.1f} min, "
                f"timesteps={self.num_timesteps}"
            )
            self.last_report_at = now

        if elapsed >= self.seconds:
            print(
                f"[wall-clock] Reached {self.seconds / 60:.1f} minutes. "
                "Stopping after the current environment step."
            )
            return False
        return True


def monitored_env():
    return Monitor(
        PitchOnlyDeskroEnv(
            domain_randomization=False,
            curriculum=True,
            full_range_deg=18.0,
        ),
        info_keywords=(
            "is_success",
            "pitch_error_deg",
            "pitch_curriculum_range_deg",
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=20.0)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--resume", type=Path, default=find_resume_model())
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "deskro_pitch_only_20m",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()

    if args.minutes <= 0:
        raise SystemExit("--minutes must be greater than zero")
    if not args.resume.exists():
        raise SystemExit(f"Existing policy not found: {args.resume}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output.parent / "checkpoints_pitch_only_20m"
    best_dir = args.output.parent / "best_pitch_only_20m"
    tensorboard_dir = args.output.parent / "tensorboard_pitch_only"

    for folder in (checkpoint_dir, best_dir, tensorboard_dir):
        folder.mkdir(parents=True, exist_ok=True)

    train_env = make_vec_env(
        monitored_env,
        n_envs=args.n_envs,
        seed=619,
    )

    print("Loading existing policy:", args.resume)
    model = PPO.load(args.resume, env=train_env)

    model.learning_rate = args.learning_rate
    model.lr_schedule = get_schedule_fn(args.learning_rate)
    for param_group in model.policy.optimizer.param_groups:
        param_group["lr"] = args.learning_rate
    model.tensorboard_log = str(tensorboard_dir)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // args.n_envs, 1),
        save_path=str(checkpoint_dir),
        name_prefix="deskro_pitch_only",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=2,
    )
    metric_callback = PitchMetricCallback(
        eval_every_timesteps=100_000,
        n_envs=args.n_envs,
        episodes=30,
        save_dir=best_dir,
    )
    timer_callback = StopAfterWallClockCallback(
        seconds=args.minutes * 60.0,
        report_every_seconds=300.0,
    )

    print()
    print("=== DESKRO PITCH-ONLY CURRICULUM ===")
    print("This reuses the existing policy; it is not training from zero.")
    print("Temporary curriculum: yaw=0, friendly face fixed, pitch only.")
    print(f"Duration: {args.minutes:.1f} minutes")
    print(f"Parallel environments: {args.n_envs}")
    print("Final checkpoint:", str(args.output) + ".zip")
    print("Best pitch checkpoint:", best_dir / "best_model.zip")
    print()

    model.learn(
        total_timesteps=100_000_000,
        callback=[checkpoint_callback, metric_callback, timer_callback],
        reset_num_timesteps=False,
        tb_log_name="PPO_pitch_only_20m",
        log_interval=1,
    )

    model.save(str(args.output))
    train_env.close()

    print()
    print("Pitch-only curriculum finished.")
    print("Original source model remains unchanged:", args.resume)
    print("Final pitch checkpoint:", str(args.output) + ".zip")
    if (best_dir / "best_model.zip").exists():
        print("Best pitch checkpoint:", best_dir / "best_model.zip")
        print("Watch it with: 24_watch_pitch_only_best.bat")
    print()
    print(
        "This is a curriculum checkpoint. "
        "Do not deploy it as the final 2-axis robot policy."
    )


if __name__ == "__main__":
    main()
