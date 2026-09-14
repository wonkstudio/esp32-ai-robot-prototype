from __future__ import annotations
from pathlib import Path
import sys, time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "envhub"))
from pitch_settling_env import PitchSettlingDeskroEnv


class WallClockStop(BaseCallback):
    def __init__(self, seconds=900):
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
                f"[wall-clock] elapsed={elapsed/60:.1f} min "
                f"remaining={max(self.seconds-elapsed,0)/60:.1f} min "
                f"timesteps={self.num_timesteps}"
            )
            self.last_report = now

        if elapsed >= self.seconds:
            print("[wall-clock] 15 minutes reached.")
            return False
        return True


def main():
    source = ROOT / "outputs" / "best_pitch_only_20m" / "best_model.zip"
    if not source.exists():
        raise SystemExit(f"Missing source model: {source}")

    # Exactly ONE MuJoCo model is kept in memory during training.
    env = make_vec_env(
        lambda: Monitor(PitchSettlingDeskroEnv(domain_randomization=False)),
        n_envs=1,
        seed=821,
    )

    print("Loading:", source)
    print("Ultra memory-safe mode: ONE MuJoCo model only")
    print("No evaluation model will be created during training.")

    model = PPO.load(source, env=env)

    lr = 5e-5
    model.learning_rate = lr
    model.lr_schedule = get_schedule_fn(lr)
    for group in model.policy.optimizer.param_groups:
        group["lr"] = lr

    checkpoint_dir = ROOT / "outputs" / "checkpoints_pitch_settling_15m"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        CheckpointCallback(
            save_freq=50000,
            save_path=str(checkpoint_dir),
            name_prefix="deskro_pitch_settling",
            verbose=1,
        ),
        WallClockStop(seconds=15*60),
    ]

    print()
    print("=== PITCH SETTLING TRAINING ===")
    print("15 minutes")
    print("Goal: keep pitch accuracy, learn to stop/hold")
    print("Training view: OFF")
    print("Mid-training evaluation: OFF (memory saving)")
    print()

    model.learn(
        total_timesteps=100_000_000,
        callback=callbacks,
        reset_num_timesteps=False,
        log_interval=1,
    )

    out = ROOT / "outputs" / "deskro_pitch_settling_15m"
    model.save(str(out))
    env.close()

    print()
    print("Training finished.")
    print("Saved:", str(out) + ".zip")
    print("Now evaluate AFTER training with:")
    print("  32_eval_pitch_settling_best.bat")


if __name__ == "__main__":
    main()
