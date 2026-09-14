"""Yaw-focused curriculum for DESKRO v0.2.8.

Why this exists:
- Pitch is already accurate/stable.
- Integrated gaze still has ~13 deg yaw error.
- We keep pitch active so the shared PPO network does not simply forget it,
  but make yaw the dominant learning signal for a short curriculum stage.
"""

from __future__ import annotations

import math
import numpy as np
import mujoco

from gaze_env import GazeTrackingDeskroEnv


class YawFocusGazeEnv(GazeTrackingDeskroEnv):
    task = "yaw_focus_gaze"

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        # Most episodes deliberately ask for a meaningful left/right turn.
        # A minority still train near center so the policy can settle there too.
        if float(self.np_random.random()) < 0.80:
            sign = -1.0 if float(self.np_random.random()) < 0.5 else 1.0
            self._raw_target_yaw_deg = sign * float(
                self.np_random.uniform(12.0, 45.0)
            )
        else:
            self._raw_target_yaw_deg = float(
                self.np_random.uniform(-10.0, 10.0)
            )

        # Keep pitch alive, but narrow the range during yaw curriculum.
        self._raw_target_pitch_deg = float(
            self.np_random.uniform(-6.0, 6.0)
        )

        # Perceived target starts neutral and is still subject to the exact same
        # reaction-delay and slew-rate logic as the integrated gaze environment.
        self._target_yaw_deg = 0.0
        self._target_pitch_deg = 0.0
        self._gaze_success_streak = 0
        self._prev_action[:] = 0.0

        self._update_target_marker()
        self._update_face_visual()
        mujoco.mj_forward(self.model, self.data)

        info.update({
            "task": self.task,
            "raw_target_yaw_deg": self._raw_target_yaw_deg,
            "raw_target_pitch_deg": self._raw_target_pitch_deg,
            "perceived_target_yaw_deg": self._target_yaw_deg,
            "perceived_target_pitch_deg": self._target_pitch_deg,
        })
        return self._observation(), info

    @staticmethod
    def _bell(error_deg, sigma_deg):
        ratio = float(error_deg) / float(sigma_deg)
        return math.exp(-0.5 * ratio * ratio)

    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)

        yaw_error = float(info["yaw_error_deg"])
        pitch_error = float(info["pitch_error_deg"])
        yaw_speed = abs(float(info["yaw_velocity_deg_s"]))
        pitch_speed = abs(float(info["pitch_velocity_deg_s"]))

        # Broad yaw term gives learning signal while far away.
        # Fine yaw term teaches precise fixation near the target.
        yaw_coarse = self._bell(yaw_error, 15.0)
        yaw_fine = self._bell(yaw_error, 4.0)
        pitch_keep = self._bell(pitch_error, 3.0)

        motion = yaw_speed + 0.7 * pitch_speed
        stillness = self._bell(motion, 15.0)

        streak = int(info.get("gaze_success_streak", 0))
        settling_bonus = 0.20 * min(streak, 8)

        reward = (
            1.30 * yaw_coarse
            + 1.60 * yaw_fine
            + 0.70 * pitch_keep
            + 0.50 * stillness
            + settling_bonus
        )

        info["task"] = self.task
        info["yaw_focus_reward"] = float(reward)
        return obs, float(reward), terminated, truncated, info
