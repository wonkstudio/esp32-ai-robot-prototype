from __future__ import annotations
import math
import numpy as np
from pitch_only_env import PitchOnlyDeskroEnv

class PitchSettlingDeskroEnv(PitchOnlyDeskroEnv):
    task = "pitch_settling_curriculum"

    def __init__(self, episode_length=180, frame_skip=5,
                 domain_randomization=False, full_range_deg=18.0):
        super().__init__(
            episode_length=episode_length,
            frame_skip=frame_skip,
            domain_randomization=domain_randomization,
            curriculum=False,
            full_range_deg=full_range_deg,
        )
        self._settled_streak = 0
        self._prev_pitch_action = 0.0

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._settled_streak = 0
        self._prev_pitch_action = 0.0
        return obs, info

    def step(self, action):
        raw = np.asarray(action, dtype=np.float32).reshape(3)
        raw = np.clip(raw, -1.0, 1.0)

        # Keep yaw and face fixed; fine-tune pitch only.
        forced = np.array([0.0, raw[1], 0.0], dtype=np.float32)
        obs, _, _, _, info = super().step(forced)

        pitch = float(info["head_pitch_deg"])
        target = float(info["target_pitch_deg"])
        speed = abs(float(info["pitch_velocity_deg_s"]))
        error = abs(pitch - target)

        precision = math.exp(-0.5 * (error / 2.0) ** 2)
        stillness = math.exp(-0.5 * (speed / 8.0) ** 2)

        close_speed_penalty = 0.018 * speed if error < 3.0 else 0.0
        velocity_penalty = 0.0035 * speed
        action_change_penalty = 0.07 * abs(float(raw[1]) - self._prev_pitch_action)
        self._prev_pitch_action = float(raw[1])

        hold_now = error < 2.0 and speed < 6.0
        hold_bonus = 2.0 if hold_now else 0.0

        reward = (
            1.7 * precision
            + 1.25 * stillness
            + hold_bonus
            - velocity_penalty
            - close_speed_penalty
            - action_change_penalty
        )

        self._settled_streak = self._settled_streak + 1 if hold_now else 0
        terminated = self._settled_streak >= 12
        truncated = self._step_count >= self.episode_length

        info.update({
            "is_success": bool(terminated),
            "pitch_error_deg": error,
            "pitch_velocity_deg_s": float(info["pitch_velocity_deg_s"]),
            "settled_streak": self._settled_streak,
        })

        return obs, float(reward), terminated, truncated, info
