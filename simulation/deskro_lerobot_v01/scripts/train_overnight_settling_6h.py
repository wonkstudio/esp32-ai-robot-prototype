from __future__ import annotations

import argparse
import json
from collections import deque
from datetime import datetime
from pathlib import Path
import sys
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "envhub"))
from overnight_settling_env import OvernightSettlingGazeEnv


class OvernightCallback(BaseCallback):
    """Wall-clock stop + hourly checkpoints + TensorBoard diagnostics."""

    def __init__(self, duration_seconds, output_dir, checkpoint_seconds=3600):
        super().__init__(verbose=1)
        self.duration_seconds = float(duration_seconds)
        self.checkpoint_seconds = float(checkpoint_seconds)
        self.output_dir = Path(output_dir)

        self.started = 0.0
        self.next_checkpoint = self.checkpoint_seconds
        self.checkpoint_index = 1
        self.last_console_report = 0.0

        self.yaw_error = deque(maxlen=1000)
        self.pitch_error = deque(maxlen=1000)
        self.yaw_speed = deque(maxlen=1000)
        self.pitch_speed = deque(maxlen=1000)
        self.hold = deque(maxlen=1000)

    def _on_training_start(self):
        self.started = time.monotonic()
        self.last_console_report = self.started

    def _save_hour_checkpoint(self):
        path = self.output_dir / (
            f"deskro_overnight_hour_{self.checkpoint_index:02d}"
        )
        self.model.save(str(path))
        print()
        print(
            f"[checkpoint] hour {self.checkpoint_index:02d} saved -> "
            f"{path}.zip"
        )
        self.checkpoint_index += 1
        self.next_checkpoint += self.checkpoint_seconds

    def _record_infos(self):
        infos = self.locals.get("infos") or []
        for info in infos:
            if "raw_yaw_error_deg" in info:
                self.yaw_error.append(float(info["raw_yaw_error_deg"]))
            if "raw_pitch_error_deg" in info:
                self.pitch_error.append(float(info["raw_pitch_error_deg"]))
            if "yaw_velocity_deg_s" in info:
                self.yaw_speed.append(abs(float(info["yaw_velocity_deg_s"])))
            if "pitch_velocity_deg_s" in info:
                self.pitch_speed.append(abs(float(info["pitch_velocity_deg_s"])))
            self.hold.append(
                1.0 if bool(info.get("overnight_hold_now", False)) else 0.0
            )

    def _log_metrics(self):
        if self.yaw_error:
            self.logger.record(
                "deskro/yaw_error_deg_rolling",
                float(np.mean(self.yaw_error)),
            )
            self.logger.record(
                "deskro/pitch_error_deg_rolling",
                float(np.mean(self.pitch_error)),
            )
            self.logger.record(
                "deskro/yaw_speed_deg_s_rolling",
                float(np.mean(self.yaw_speed)),
            )
            self.logger.record(
                "deskro/pitch_speed_deg_s_rolling",
                float(np.mean(self.pitch_speed)),
            )
            self.logger.record(
                "deskro/hold_fraction_rolling",
                float(np.mean(self.hold)),
            )

    def _on_step(self):
        self._record_infos()
        self._log_metrics()

        now = time.monotonic()
        elapsed = now - self.started

        # Time-based checkpoints, independent of FPS/timestep count.
        while (
            elapsed >= self.next_checkpoint
            and self.next_checkpoint <= self.duration_seconds
        ):
            self._save_hour_checkpoint()

        if now - self.last_console_report >= 300.0:
            remaining = max(self.duration_seconds - elapsed, 0.0)
            if self.yaw_error:
                print(
                    f"[overnight] {elapsed/3600:.2f}h elapsed | "
                    f"{remaining/3600:.2f}h left | "
                    f"yaw_err~{np.mean(self.yaw_error):.2f}deg | "
                    f"pitch_err~{np.mean(self.pitch_error):.2f}deg | "
                    f"yaw_speed~{np.mean(self.yaw_speed):.2f}deg/s | "
                    f"pitch_speed~{np.mean(self.pitch_speed):.2f}deg/s"
                )
            else:
                print(
                    f"[overnight] {elapsed/3600:.2f}h elapsed | "
                    f"{remaining/3600:.2f}h left"
                )
            self.last_console_report = now

        return elapsed < self.duration_seconds


def make_env(randomized):
    def _factory():
        return Monitor(
            OvernightSettlingGazeEnv(
                domain_randomization=randomized,
                model_filename="model_train.xml",
            )
        )
    return _factory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=6.0)
    parser.add_argument("--n-envs", type=int, default=4)
    args = parser.parse_args()

    source = ROOT / "outputs" / "deskro_yaw_focus_20m.zip"
    if not source.exists():
        raise SystemExit(
            "Source model not found:\n"
            f"  {source}\n\n"
            "Run 38_train_yaw_focus_20m.bat first."
        )

    output_dir = ROOT / "outputs" / "overnight_settling_6h"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3 nominal robots + 1 randomized robot.
    env_fns = []
    for i in range(args.n_envs):
        randomized = i == args.n_envs - 1
        env_fns.append(make_env(randomized))

    env = DummyVecEnv(env_fns)

    model = PPO.load(source, env=env)

    # Deliberately conservative for a long fine-tune.
    lr = 2.0e-5
    model.learning_rate = lr
    model.lr_schedule = lambda _: lr
    for group in model.policy.optimizer.param_groups:
        group["lr"] = lr

    tensorboard_dir = ROOT / "outputs" / "tensorboard_overnight_6h"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    model.tensorboard_log = str(tensorboard_dir)

    config = {
        "started_local": datetime.now().isoformat(timespec="seconds"),
        "source_model": str(source),
        "hours": float(args.hours),
        "n_envs": int(args.n_envs),
        "env_mix": "3 nominal + 1 domain-randomized" if args.n_envs == 4
                   else "last env randomized, others nominal",
        "learning_rate": lr,
        "mid_training_evaluation": False,
        "checkpoint_interval_hours": 1.0,
        "yaw_software_speed_cap_deg_s": 60.0,
        "pitch_software_speed_cap_deg_s": 50.0,
        "yaw_software_accel_cap_deg_s2": 180.0,
        "pitch_software_accel_cap_deg_s2": 150.0,
        "goal": (
            "preserve target accuracy while reducing residual yaw/pitch "
            "motion and action chatter near target"
        ),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    callback = OvernightCallback(
        duration_seconds=args.hours * 3600.0,
        output_dir=output_dir,
        checkpoint_seconds=3600.0,
    )

    print("DESKRO OVERNIGHT SETTLING")
    print("==========================")
    print("Source:", source)
    print(f"Duration: {args.hours:.1f} hours")
    print("Parallel lightweight MuJoCo robots:", args.n_envs)
    print("Environment mix: 3 nominal + 1 randomized")
    print("Learning rate: 2e-5")
    print("Hourly wall-clock checkpoints: enabled")
    print("Mid-training evaluation: disabled")
    print("TensorBoard diagnostics: enabled")
    print("Friendly servo caps: yaw 60 deg/s, pitch 50 deg/s")
    print()
    print("You can leave this terminal running.")
    print()

    final_path = ROOT / "outputs" / "deskro_overnight_settling_6h"

    try:
        model.learn(
            total_timesteps=1_000_000_000,
            callback=callback,
            reset_num_timesteps=False,
            log_interval=1,
            tb_log_name="yaw_pitch_settling_6h",
        )
        model.save(str(final_path))
        print()
        print("Overnight training finished.")
        print("Final:", str(final_path) + ".zip")
        print("Next: .\\42_eval_overnight_checkpoints.bat")
    except KeyboardInterrupt:
        interrupted = ROOT / "outputs" / "deskro_overnight_interrupted"
        model.save(str(interrupted))
        print()
        print("Training interrupted by user.")
        print("Emergency model saved:", str(interrupted) + ".zip")
        raise
    finally:
        env.close()


if __name__ == "__main__":
    main()
