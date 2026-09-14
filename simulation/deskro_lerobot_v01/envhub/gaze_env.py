"""Integrated yaw + pitch gaze curriculum.

Raw target = actual virtual user (orange sphere).
Filtered target = what the policy sees after reaction-delay / target smoothing.

The policy keeps the same 8-D observation and 3-D action interface, so the
pitch curriculum checkpoint can be reused.
"""

from __future__ import annotations

import math
import numpy as np
import mujoco

from deskro_env import DeskroV1Env


class GazeTrackingDeskroEnv(DeskroV1Env):
    task = "integrated_gaze_tracking"

    def __init__(
        self,
        episode_length=260,
        frame_skip=5,
        domain_randomization=False,
        model_filename="model_train.xml",
        reaction_delay_min_s=0.15,
        reaction_delay_max_s=0.30,
    ):
        super().__init__(
            episode_length=episode_length,
            frame_skip=frame_skip,
            domain_randomization=domain_randomization,
            model_filename=model_filename,
        )
        self.reaction_delay_min_s = float(reaction_delay_min_s)
        self.reaction_delay_max_s = float(reaction_delay_max_s)

        self._raw_target_yaw_deg = 0.0
        self._raw_target_pitch_deg = 0.0
        self._reaction_steps_left = 0

        # The perceived gaze target itself moves smoothly.
        self._target_filter_yaw_speed_deg_s = 105.0
        self._target_filter_pitch_speed_deg_s = 75.0

        self._gaze_success_streak = 0
        self._prev_action = np.zeros(2, dtype=np.float32)

    def _update_target_marker(self):
        # Show the TRUE user location, not the filtered target shown to policy.
        raw_yaw = getattr(
            self, "_raw_target_yaw_deg", self._target_yaw_deg
        )
        raw_pitch = getattr(
            self, "_raw_target_pitch_deg", self._target_pitch_deg
        )

        radius = 0.34
        head_center_z = 0.24
        yaw = math.radians(raw_yaw)
        pitch = math.radians(raw_pitch)
        horizontal = radius * math.cos(pitch)

        self.data.mocap_pos[self._target_mocap_id] = np.array(
            [
                horizontal * math.sin(yaw),
                -horizontal * math.cos(yaw),
                head_center_z + radius * math.sin(pitch),
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _slew(current, target, max_rate_deg_s, dt):
        max_delta = max_rate_deg_s * dt
        delta = float(np.clip(target - current, -max_delta, max_delta))
        return current + delta

    def _advance_perceived_target(self):
        if self._reaction_steps_left > 0:
            self._reaction_steps_left -= 1
            return

        control_dt = float(self.model.opt.timestep) * self.frame_skip
        self._target_yaw_deg = self._slew(
            self._target_yaw_deg,
            self._raw_target_yaw_deg,
            self._target_filter_yaw_speed_deg_s,
            control_dt,
        )
        self._target_pitch_deg = self._slew(
            self._target_pitch_deg,
            self._raw_target_pitch_deg,
            self._target_filter_pitch_speed_deg_s,
            control_dt,
        )

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        # Preserve the randomly sampled full 2-axis target as the real target.
        self._raw_target_yaw_deg = float(self._target_yaw_deg)
        self._raw_target_pitch_deg = float(self._target_pitch_deg)

        # Every gaze episode starts from a neutral perceived target.
        # This creates a human-like reaction phase instead of instant snapping.
        self._target_yaw_deg = 0.0
        self._target_pitch_deg = 0.0
        self._wake_word = 1
        self._face_state = 2  # friendly face fixed for this curriculum

        control_dt = float(self.model.opt.timestep) * self.frame_skip
        delay_s = float(
            self.np_random.uniform(
                self.reaction_delay_min_s,
                self.reaction_delay_max_s,
            )
        )
        self._reaction_steps_left = max(1, int(round(delay_s / control_dt)))
        self._gaze_success_streak = 0
        self._prev_action[:] = 0.0

        self._update_target_marker()
        self._update_face_visual()
        mujoco.mj_forward(self.model, self.data)

        info.update({
            "raw_target_yaw_deg": self._raw_target_yaw_deg,
            "raw_target_pitch_deg": self._raw_target_pitch_deg,
            "perceived_target_yaw_deg": self._target_yaw_deg,
            "perceived_target_pitch_deg": self._target_pitch_deg,
            "reaction_delay_s": delay_s,
        })
        return self._observation(), info

    @staticmethod
    def _precision(error_deg, sigma_deg):
        ratio = error_deg / sigma_deg
        return math.exp(-0.5 * ratio * ratio)

    def step(self, action):
        self._advance_perceived_target()

        raw_action = np.asarray(action, dtype=np.float32).reshape(3)
        raw_action = np.clip(raw_action, -1.0, 1.0)

        # Integrate yaw + pitch now, but keep face fixed until gaze is stable.
        forced_action = np.array(
            [raw_action[0], raw_action[1], 0.0],
            dtype=np.float32,
        )
        obs, _, _, _, info = super().step(forced_action)

        yaw = float(info["head_yaw_deg"])
        pitch = float(info["head_pitch_deg"])
        yaw_vel = math.degrees(
            float(self.data.qvel[self._yaw_dof_adr])
        )
        pitch_vel = math.degrees(
            float(self.data.qvel[self._pitch_dof_adr])
        )

        yaw_error = abs(yaw - self._target_yaw_deg)
        pitch_error = abs(pitch - self._target_pitch_deg)

        yaw_precision = self._precision(yaw_error, 4.5)
        pitch_precision = self._precision(pitch_error, 3.0)

        total_speed = abs(yaw_vel) + abs(pitch_vel)
        stillness = math.exp(-0.5 * (total_speed / 16.0) ** 2)

        action_delta = (
            abs(float(raw_action[0]) - float(self._prev_action[0]))
            + abs(float(raw_action[1]) - float(self._prev_action[1]))
        )
        self._prev_action[:] = raw_action[:2]

        # Once the perceived target has reached the raw target, reward accurate
        # and stable fixation more strongly.
        filter_yaw_error = abs(
            self._target_yaw_deg - self._raw_target_yaw_deg
        )
        filter_pitch_error = abs(
            self._target_pitch_deg - self._raw_target_pitch_deg
        )
        filter_settled = filter_yaw_error < 1.0 and filter_pitch_error < 1.0

        hold_now = (
            filter_settled
            and yaw_error < 3.0
            and pitch_error < 2.5
            and abs(yaw_vel) < 10.0
            and abs(pitch_vel) < 8.0
        )
        hold_bonus = 1.8 if hold_now else 0.0

        reward = (
            0.95 * yaw_precision
            + 1.15 * pitch_precision
            + 0.55 * stillness
            + hold_bonus
            - 0.045 * action_delta
        )

        self._gaze_success_streak = (
            self._gaze_success_streak + 1 if hold_now else 0
        )
        terminated = self._gaze_success_streak >= 8
        truncated = self._step_count >= self.episode_length

        raw_yaw_error = abs(yaw - self._raw_target_yaw_deg)
        raw_pitch_error = abs(pitch - self._raw_target_pitch_deg)

        info.update({
            "task": self.task,
            "is_success": bool(terminated),
            "yaw_error_deg": yaw_error,
            "pitch_error_deg": pitch_error,
            "raw_yaw_error_deg": raw_yaw_error,
            "raw_pitch_error_deg": raw_pitch_error,
            "yaw_velocity_deg_s": yaw_vel,
            "pitch_velocity_deg_s": pitch_vel,
            "raw_target_yaw_deg": self._raw_target_yaw_deg,
            "raw_target_pitch_deg": self._raw_target_pitch_deg,
            "perceived_target_yaw_deg": self._target_yaw_deg,
            "perceived_target_pitch_deg": self._target_pitch_deg,
            "reaction_steps_left": self._reaction_steps_left,
            "gaze_success_streak": self._gaze_success_streak,
        })

        return obs, float(reward), terminated, truncated, info
