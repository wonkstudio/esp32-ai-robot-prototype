"""Pitch-only curriculum environment for DESKRO.

This is a temporary curriculum stage, not the final robot policy.

It keeps the SAME observation and action shapes as DeskroV1Env, so an existing
PPO checkpoint can be loaded. During this stage:
- target yaw is fixed at 0 degrees,
- yaw command is forced to 0 degrees,
- wake word is always active,
- face is forced to big_smile,
- reward and success depend mainly on pitch accuracy and settling.

After pitch becomes accurate, the best checkpoint should be fine-tuned again in
the full yaw+pitch environment.
"""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

from deskro_env import DeskroV1Env


class PitchOnlyDeskroEnv(DeskroV1Env):
    task = "pitch_only_curriculum"
    task_description = "Learn accurate up/down head tracking before full 2-axis training."

    def __init__(
        self,
        episode_length: int = 160,
        frame_skip: int = 5,
        domain_randomization: bool = False,
        curriculum: bool = True,
        full_range_deg: float = 18.0,
    ) -> None:
        super().__init__(
            episode_length=episode_length,
            frame_skip=frame_skip,
            domain_randomization=domain_randomization,
        )
        self.curriculum = bool(curriculum)
        self.full_range_deg = float(full_range_deg)
        self._curriculum_resets = 0
        self._pitch_success_streak = 0
        self._previous_pitch_action = 0.0
        self._active_pitch_range_deg = 6.0

    def _sample_pitch_target(self) -> float:
        if self.curriculum:
            # Easy -> medium -> full range. Each vector environment advances
            # independently, which is fine for PPO.
            if self._curriculum_resets < 80:
                max_abs = 6.0
            elif self._curriculum_resets < 200:
                max_abs = 12.0
            else:
                max_abs = self.full_range_deg
        else:
            max_abs = self.full_range_deg

        self._active_pitch_range_deg = max_abs

        # Avoid too many trivial near-zero targets.
        magnitude = float(self.np_random.uniform(2.0, max_abs))
        sign = -1.0 if self.np_random.random() < 0.5 else 1.0
        return sign * magnitude

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        observation, info = super().reset(seed=seed, options=options)

        self._curriculum_resets += 1
        self._pitch_success_streak = 0
        self._previous_pitch_action = 0.0

        self._target_yaw_deg = 0.0
        self._target_pitch_deg = self._sample_pitch_target()
        self._wake_word = 1
        self._face_state = 2  # big_smile

        self._update_target_marker()
        self._update_face_visual()
        mujoco.mj_forward(self.model, self.data)

        info.update(
            {
                "task": self.task,
                "target_yaw_deg": 0.0,
                "target_pitch_deg": self._target_pitch_deg,
                "pitch_curriculum_range_deg": self._active_pitch_range_deg,
            }
        )
        return self._observation(), info

    @staticmethod
    def _pitch_precision(error_deg: float, sigma_deg: float = 3.5) -> float:
        ratio = error_deg / sigma_deg
        return math.exp(-0.5 * ratio * ratio)

    def step(self, action: np.ndarray):
        raw_action = np.asarray(action, dtype=np.float32).reshape(3)
        raw_action = np.clip(raw_action, -1.0, 1.0)

        _, pitch_before, _, _ = self._joint_state_deg()
        error_before = abs(pitch_before - self._target_pitch_deg)

        # Preserve the 3-D action interface, but isolate the pitch problem.
        # 0.0 on face action maps to face_state 2 (big_smile).
        forced_action = np.array([0.0, raw_action[1], 0.0], dtype=np.float32)

        observation, _, _, _, info = super().step(forced_action)

        yaw, pitch, yaw_vel, pitch_vel = self._joint_state_deg()
        pitch_error = abs(pitch - self._target_pitch_deg)
        pitch_progress = error_before - pitch_error

        precision = self._pitch_precision(pitch_error)
        broad_tracking = max(0.0, 1.0 - pitch_error / 25.0)
        settled_bonus = 0.80 if pitch_error < 2.5 and abs(pitch_vel) < 8.0 else 0.0

        velocity_penalty = 0.0018 * abs(pitch_vel)
        yaw_drift_penalty = 0.012 * abs(yaw)
        unused_output_penalty = (
            0.025 * abs(float(raw_action[0]))
            + 0.015 * abs(float(raw_action[2]))
        )
        pitch_action_change_penalty = 0.020 * abs(
            float(raw_action[1]) - self._previous_pitch_action
        )

        reward = (
            1.80 * precision
            + 0.55 * broad_tracking
            + 0.12 * pitch_progress
            + settled_bonus
            - velocity_penalty
            - yaw_drift_penalty
            - unused_output_penalty
            - pitch_action_change_penalty
        )

        self._previous_pitch_action = float(raw_action[1])

        success_now = pitch_error < 2.5 and abs(pitch_vel) < 8.0
        self._pitch_success_streak = (
            self._pitch_success_streak + 1 if success_now else 0
        )
        terminated = self._pitch_success_streak >= 8
        truncated = self._step_count >= self.episode_length

        info.update(
            {
                "task": self.task,
                "is_success": bool(terminated),
                "yaw_error_deg": float(abs(yaw)),
                "pitch_error_deg": float(pitch_error),
                "head_yaw_deg": float(yaw),
                "head_pitch_deg": float(pitch),
                "target_yaw_deg": 0.0,
                "target_pitch_deg": float(self._target_pitch_deg),
                "pitch_velocity_deg_s": float(pitch_vel),
                "pitch_curriculum_range_deg": float(
                    self._active_pitch_range_deg
                ),
            }
        )

        return observation, float(reward), terminated, truncated, info
