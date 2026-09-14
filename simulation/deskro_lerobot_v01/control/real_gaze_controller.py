"""DESKRO real-style 2DOF gaze controller.

This is intentionally NOT reinforcement learning.

Input:
    target yaw / pitch in degrees

Output:
    a bounded, delayed, smoothly changing gaze target

The MuJoCo servo trajectory layer (and later the ESP32 firmware) handles
speed/acceleration-limited joint motion.

Conventions:
    yaw   + = look right
    yaw   - = look left
    pitch + = look up
    pitch - = look down
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class GazeLimits:
    # Conservative DESKRO V1 software limits until the real bracket is measured.
    yaw_min_deg: float = -45.0
    yaw_max_deg: float = 45.0
    pitch_min_deg: float = -15.0
    pitch_max_deg: float = 20.0

    # High-level gaze-target slew. The servo itself is slower than this.
    target_yaw_slew_deg_s: float = 100.0
    target_pitch_slew_deg_s: float = 75.0

    # Short "noticed something" reaction delay.
    reaction_delay_s: float = 0.20


class RealGazeController:
    def __init__(self, limits: GazeLimits | None = None):
        self.limits = limits or GazeLimits()

        self._requested_yaw_deg = 0.0
        self._requested_pitch_deg = 0.0

        self._command_yaw_deg = 0.0
        self._command_pitch_deg = 0.0

        self._reaction_remaining_s = 0.0

    def reset(self, yaw_deg: float = 0.0, pitch_deg: float = 0.0) -> None:
        yaw, pitch = self._clamp(yaw_deg, pitch_deg)
        self._requested_yaw_deg = yaw
        self._requested_pitch_deg = pitch
        self._command_yaw_deg = yaw
        self._command_pitch_deg = pitch
        self._reaction_remaining_s = 0.0

    def set_target(
        self,
        yaw_deg: float,
        pitch_deg: float,
        *,
        attention_switch: bool = True,
    ) -> tuple[float, float]:
        """Set a new gaze target.

        attention_switch=True:
            use the short reaction delay, appropriate when changing who/what
            DESKRO is looking at.

        attention_switch=False:
            update continuously without restarting the reaction delay,
            appropriate later for webcam face-tracking updates.
        """
        yaw, pitch = self._clamp(yaw_deg, pitch_deg)
        self._requested_yaw_deg = yaw
        self._requested_pitch_deg = pitch

        if attention_switch:
            self._reaction_remaining_s = self.limits.reaction_delay_s

        return yaw, pitch

    @property
    def requested_target(self) -> tuple[float, float]:
        return self._requested_yaw_deg, self._requested_pitch_deg

    @property
    def command_target(self) -> tuple[float, float]:
        return self._command_yaw_deg, self._command_pitch_deg

    @property
    def reacting(self) -> bool:
        return self._reaction_remaining_s > 0.0

    @staticmethod
    def _slew(current: float, target: float, rate_deg_s: float, dt: float) -> float:
        max_delta = max(0.0, rate_deg_s * dt)
        return current + float(np.clip(target - current, -max_delta, max_delta))

    def _clamp(self, yaw_deg: float, pitch_deg: float) -> tuple[float, float]:
        yaw = float(
            np.clip(
                yaw_deg,
                self.limits.yaw_min_deg,
                self.limits.yaw_max_deg,
            )
        )
        pitch = float(
            np.clip(
                pitch_deg,
                self.limits.pitch_min_deg,
                self.limits.pitch_max_deg,
            )
        )
        return yaw, pitch

    def update(self, dt: float) -> tuple[float, float]:
        dt = max(float(dt), 0.0)

        if self._reaction_remaining_s > 0.0:
            self._reaction_remaining_s = max(
                0.0,
                self._reaction_remaining_s - dt,
            )
            return self.command_target

        self._command_yaw_deg = self._slew(
            self._command_yaw_deg,
            self._requested_yaw_deg,
            self.limits.target_yaw_slew_deg_s,
            dt,
        )
        self._command_pitch_deg = self._slew(
            self._command_pitch_deg,
            self._requested_pitch_deg,
            self.limits.target_pitch_slew_deg_s,
            dt,
        )

        return self.command_target
