"""DESKRO overnight settling curriculum v0.2.9.

Starts from the yaw-focus policy. The goal is NOT to relearn target direction.
The goal is to keep current accuracy while reducing residual motion near target.

This environment also applies intentionally gentle software speed caps that are
closer to the motion style desired for the physical DESKRO prototype. These are
NOT claims about the raw SG90 motor maximum speed; they are comfort limits.
"""

from __future__ import annotations

import math
import numpy as np

from gaze_env import GazeTrackingDeskroEnv


class OvernightSettlingGazeEnv(GazeTrackingDeskroEnv):
    task = "overnight_gaze_settling"

    def __init__(
        self,
        episode_length=300,
        frame_skip=5,
        domain_randomization=False,
        model_filename="model_train.xml",
    ):
        super().__init__(
            episode_length=episode_length,
            frame_skip=frame_skip,
            domain_randomization=domain_randomization,
            model_filename=model_filename,
            reaction_delay_min_s=0.15,
            reaction_delay_max_s=0.30,
        )

        # Friendly software motion limits for DESKRO V1.
        # Raw SG90 can move faster; this intentionally keeps the robot readable.
        self._servo_max_speed_deg_s[:] = np.array(
            [60.0, 50.0], dtype=np.float64
        )
        self._servo_max_accel_deg_s2[:] = np.array(
            [180.0, 150.0], dtype=np.float64
        )

        self._overnight_prev_action = np.zeros(2, dtype=np.float32)

    @staticmethod
    def _bell(value, sigma):
        ratio = float(value) / float(sigma)
        return math.exp(-0.5 * ratio * ratio)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._overnight_prev_action[:] = 0.0
        info.update({
            "task": self.task,
            "servo_yaw_speed_cap_deg_s": 60.0,
            "servo_pitch_speed_cap_deg_s": 50.0,
        })
        return obs, info

    def step(self, action):
        raw_action = np.asarray(action, dtype=np.float32).reshape(3)
        raw_action = np.clip(raw_action, -1.0, 1.0)

        action_delta = (
            abs(float(raw_action[0]) - float(self._overnight_prev_action[0]))
            + abs(float(raw_action[1]) - float(self._overnight_prev_action[1]))
        )
        self._overnight_prev_action[:] = raw_action[:2]

        obs, _, terminated, truncated, info = super().step(raw_action)

        yaw_error = float(info["yaw_error_deg"])
        pitch_error = float(info["pitch_error_deg"])
        yaw_speed = abs(float(info["yaw_velocity_deg_s"]))
        pitch_speed = abs(float(info["pitch_velocity_deg_s"]))

        # Coarse terms preserve the already-learned ability to reach targets.
        yaw_coarse = self._bell(yaw_error, 12.0)
        pitch_coarse = self._bell(pitch_error, 7.0)

        # Fine terms protect the accuracy we gained from the yaw curriculum.
        yaw_fine = self._bell(yaw_error, 3.0)
        pitch_fine = self._bell(pitch_error, 2.5)

        # Only strongly care about stopping once the head is already near target.
        near_target = yaw_fine * pitch_fine
        stillness = (
            self._bell(yaw_speed, 8.0)
            * self._bell(pitch_speed, 6.0)
        )

        hold_now = (
            yaw_error < 3.0
            and pitch_error < 2.5
            and yaw_speed < 8.0
            and pitch_speed < 6.0
            and int(info.get("reaction_steps_left", 0)) == 0
        )
        hold_bonus = 2.4 if hold_now else 0.0

        # Near target, residual velocity and policy chatter become expensive.
        near_velocity_penalty = near_target * (
            0.012 * yaw_speed + 0.016 * pitch_speed
        )
        chatter_penalty = 0.075 * action_delta

        reward = (
            0.75 * yaw_coarse
            + 0.55 * pitch_coarse
            + 1.05 * yaw_fine
            + 0.95 * pitch_fine
            + 1.70 * near_target * stillness
            + hold_bonus
            - near_velocity_penalty
            - chatter_penalty
        )

        info.update({
            "task": self.task,
            "overnight_reward": float(reward),
            "near_target_score": float(near_target),
            "stillness_score": float(stillness),
            "action_delta": float(action_delta),
            "overnight_hold_now": bool(hold_now),
        })
        return obs, float(reward), terminated, truncated, info
