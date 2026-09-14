"""MuJoCo/Gymnasium environment for DESKRO V1.

Pitch convention in this version:
  positive pitch = look up
  negative pitch = look down

The reward scores yaw and pitch separately so the larger yaw range cannot hide
pitch errors.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
import math
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


FACE_NAMES = ("idle", "happy", "big_smile", "surprised", "angry")


class DeskroV1Env(gym.Env):
    """Two-DOF desktop robot task: smoothly look toward a virtual user."""

    metadata = {"render_modes": []}
    task = "look_at_target_pitch_fixed"
    task_description = "Look toward the user with corrected up/down pitch semantics."

    def __init__(
        self,
        episode_length: int = 240,
        frame_skip: int = 5,
        domain_randomization: bool = True,
        model_filename: str = "model.xml",
    ) -> None:
        super().__init__()
        model_path = Path(__file__).with_name(model_filename)
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        self.episode_length = int(episode_length)
        self.frame_skip = int(frame_skip)
        self.domain_randomization = bool(domain_randomization)

        self.observation_space = spaces.Dict(
            {
                "agent_pos": spaces.Box(
                    low=np.array([-60, -25, -500, -500, -60, -25, 0, 0], dtype=np.float32),
                    high=np.array([60, 25, 500, 500, 60, 25, 1, 1], dtype=np.float32),
                    dtype=np.float32,
                )
            }
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

        self._yaw_joint = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "joint_head_yaw"
        )
        self._pitch_joint = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "joint_head_pitch"
        )
        self._yaw_qpos_adr = int(self.model.jnt_qposadr[self._yaw_joint])
        self._pitch_qpos_adr = int(self.model.jnt_qposadr[self._pitch_joint])
        self._yaw_dof_adr = int(self.model.jnt_dofadr[self._yaw_joint])
        self._pitch_dof_adr = int(self.model.jnt_dofadr[self._pitch_joint])

        self._yaw_actuator = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_head_yaw"
        )
        self._pitch_actuator = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_head_pitch"
        )

        target_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target_marker"
        )
        self._target_mocap_id = int(self.model.body_mocapid[target_body])
        self._left_eye_geom = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_eye"
        )
        self._right_eye_geom = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_eye"
        )

        self._base_damping = self.model.dof_damping.copy()
        self._step_count = 0
        self._success_streak = 0
        self._target_yaw_deg = 0.0
        self._target_pitch_deg = 0.0
        self._wake_word = 0
        self._face_state = 0
        self._motor_scale = np.ones(2, dtype=np.float64)
        self._motor_bias_deg = np.zeros(2, dtype=np.float64)
        self._action_delay_steps = 0
        self._action_queue: deque[np.ndarray] = deque()

        # Low-level trajectory generator shared conceptually with the real robot.
        # The RL policy still outputs desired joint angles.  These states turn
        # those desired angles into a speed/acceleration limited servo command.
        self._servo_command_deg = np.zeros(2, dtype=np.float64)
        self._servo_command_vel_deg_s = np.zeros(2, dtype=np.float64)

        # Intentionally much slower than the raw SG90 no-load maximum so the
        # desktop robot moves in a friendly, readable way.
        self._servo_max_speed_deg_s = np.array([90.0, 70.0], dtype=np.float64)
        self._servo_max_accel_deg_s2 = np.array([300.0, 240.0], dtype=np.float64)

    def _joint_state_deg(self) -> tuple[float, float, float, float]:
        yaw = math.degrees(float(self.data.qpos[self._yaw_qpos_adr]))
        pitch = math.degrees(float(self.data.qpos[self._pitch_qpos_adr]))
        yaw_vel = math.degrees(float(self.data.qvel[self._yaw_dof_adr]))
        pitch_vel = math.degrees(float(self.data.qvel[self._pitch_dof_adr]))
        return yaw, pitch, yaw_vel, pitch_vel

    def _observation(self) -> dict[str, np.ndarray]:
        yaw, pitch, yaw_vel, pitch_vel = self._joint_state_deg()
        state = np.array(
            [
                yaw,
                pitch,
                yaw_vel,
                pitch_vel,
                self._target_yaw_deg,
                self._target_pitch_deg,
                float(self._wake_word),
                float(self._face_state) / float(len(FACE_NAMES) - 1),
            ],
            dtype=np.float32,
        )
        return {"agent_pos": state}

    def _update_target_marker(self) -> None:
        radius = 0.34
        head_center_z = 0.24
        yaw = math.radians(self._target_yaw_deg)
        pitch = math.radians(self._target_pitch_deg)
        horizontal = radius * math.cos(pitch)
        self.data.mocap_pos[self._target_mocap_id] = np.array(
            [
                horizontal * math.sin(yaw),
                -horizontal * math.cos(yaw),
                head_center_z + radius * math.sin(pitch),
            ],
            dtype=np.float64,
        )

    def _update_face_visual(self) -> None:
        colors = {
            0: (0.20, 0.80, 1.00, 1.0),
            1: (0.30, 1.00, 0.55, 1.0),
            2: (1.00, 0.85, 0.20, 1.0),
            3: (1.00, 0.45, 0.20, 1.0),
            4: (1.00, 0.15, 0.12, 1.0),
        }
        color = np.asarray(colors[self._face_state], dtype=np.float32)
        self.model.geom_rgba[self._left_eye_geom] = color
        self.model.geom_rgba[self._right_eye_geom] = color

    def _decode_action(self, action: np.ndarray) -> tuple[float, float, int]:
        action = np.asarray(action, dtype=np.float32).reshape(3)
        action = np.clip(action, -1.0, 1.0)
        yaw_target_deg = float(action[0]) * 60.0
        pitch_target_deg = float(action[1]) * 25.0
        face01 = (float(action[2]) + 1.0) * 0.5
        face_state = int(
            np.clip(round(face01 * (len(FACE_NAMES) - 1)), 0, len(FACE_NAMES) - 1)
        )
        return yaw_target_deg, pitch_target_deg, face_state

    @staticmethod
    def _advance_axis_profile(
        command_deg: float,
        command_vel_deg_s: float,
        desired_deg: float,
        max_speed_deg_s: float,
        max_accel_deg_s2: float,
        dt: float,
    ) -> tuple[float, float]:
        """One acceleration-limited trajectory-generator step."""
        error = desired_deg - command_deg

        if abs(error) < 1e-6 and abs(command_vel_deg_s) < 1e-4:
            return desired_deg, 0.0

        # Braking-distance profile: automatically decelerates before target.
        braking_speed = math.sqrt(
            max(0.0, 2.0 * max_accel_deg_s2 * abs(error))
        )
        target_speed = math.copysign(
            min(max_speed_deg_s, braking_speed),
            error,
        )

        max_delta_v = max_accel_deg_s2 * dt
        velocity_delta = float(
            np.clip(
                target_speed - command_vel_deg_s,
                -max_delta_v,
                max_delta_v,
            )
        )
        new_vel = command_vel_deg_s + velocity_delta
        new_command = command_deg + new_vel * dt

        # Never overshoot the desired command.
        if (
            (error > 0.0 and new_command >= desired_deg)
            or (error < 0.0 and new_command <= desired_deg)
        ):
            return desired_deg, 0.0

        return new_command, new_vel

    def _advance_servo_profile(
        self,
        desired_yaw_deg: float,
        desired_pitch_deg: float,
    ) -> None:
        dt = float(self.model.opt.timestep)
        desired = np.array(
            [desired_yaw_deg, desired_pitch_deg],
            dtype=np.float64,
        )

        for axis in range(2):
            command, velocity = self._advance_axis_profile(
                command_deg=float(self._servo_command_deg[axis]),
                command_vel_deg_s=float(
                    self._servo_command_vel_deg_s[axis]
                ),
                desired_deg=float(desired[axis]),
                max_speed_deg_s=float(
                    self._servo_max_speed_deg_s[axis]
                ),
                max_accel_deg_s2=float(
                    self._servo_max_accel_deg_s2[axis]
                ),
                dt=dt,
            )
            self._servo_command_deg[axis] = command
            self._servo_command_vel_deg_s[axis] = velocity

        self.data.ctrl[self._yaw_actuator] = math.radians(
            float(self._servo_command_deg[0])
        )
        self.data.ctrl[self._pitch_actuator] = math.radians(
            float(self._servo_command_deg[1])
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0
        self._success_streak = 0
        self._servo_command_deg[:] = 0.0
        self._servo_command_vel_deg_s[:] = 0.0

        self._target_yaw_deg = float(self.np_random.uniform(-45.0, 45.0))
        self._target_pitch_deg = float(self.np_random.uniform(-18.0, 18.0))
        self._wake_word = int(self.np_random.random() < 0.75)
        self._face_state = 0

        if self.domain_randomization:
            self.model.dof_damping[:] = self._base_damping * self.np_random.uniform(
                0.80, 1.25, self.model.nv
            )
            self._motor_scale = np.array(
                [
                    self.np_random.uniform(0.92, 1.08),
                    self.np_random.uniform(0.96, 1.04),
                ],
                dtype=np.float64,
            )
            self._motor_bias_deg = np.array(
                [
                    self.np_random.uniform(-1.5, 1.5),
                    self.np_random.uniform(-0.8, 0.8),
                ],
                dtype=np.float64,
            )
            self._action_delay_steps = int(self.np_random.integers(0, 3))
        else:
            self.model.dof_damping[:] = self._base_damping
            self._motor_scale[:] = 1.0
            self._motor_bias_deg[:] = 0.0
            self._action_delay_steps = 0

        neutral = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self._action_queue = deque(
            [neutral.copy() for _ in range(self._action_delay_steps + 1)],
            maxlen=self._action_delay_steps + 1,
        )

        self._update_target_marker()
        self._update_face_visual()
        mujoco.mj_forward(self.model, self.data)
        return self._observation(), {
            "task": self.task,
            "target_yaw_deg": self._target_yaw_deg,
            "target_pitch_deg": self._target_pitch_deg,
            "wake_word": self._wake_word,
        }

    @staticmethod
    def _axis_score(error_deg: float, sigma_deg: float) -> float:
        """Smooth precision reward: 1 near target and approaches 0 far away."""
        ratio = error_deg / sigma_deg
        return math.exp(-0.5 * ratio * ratio)

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        previous_yaw, previous_pitch, _, _ = self._joint_state_deg()

        self._action_queue.append(np.asarray(action, dtype=np.float32).copy())
        delayed_action = self._action_queue[0]
        yaw_target_deg, pitch_target_deg, self._face_state = self._decode_action(
            delayed_action
        )

        actual_yaw_target = (
            yaw_target_deg * self._motor_scale[0] + self._motor_bias_deg[0]
        )
        actual_pitch_target = (
            pitch_target_deg * self._motor_scale[1] + self._motor_bias_deg[1]
        )
        actual_yaw_target = float(np.clip(actual_yaw_target, -60.0, 60.0))
        actual_pitch_target = float(np.clip(actual_pitch_target, -25.0, 25.0))

        self._update_face_visual()

        # The policy target is held for this control interval, but the actual
        # servo command approaches it gradually with bounded velocity and
        # acceleration.  This same idea will be used by the ESP32 adapter.
        for _ in range(self.frame_skip):
            self._advance_servo_profile(
                desired_yaw_deg=actual_yaw_target,
                desired_pitch_deg=actual_pitch_target,
            )
            mujoco.mj_step(self.model, self.data)

        yaw, pitch, yaw_vel, pitch_vel = self._joint_state_deg()
        movement = abs(yaw - previous_yaw) + abs(pitch - previous_pitch)

        if self._wake_word:
            yaw_error = abs(yaw - self._target_yaw_deg)
            pitch_error = abs(pitch - self._target_pitch_deg)

            # Pitch gets an independent precision score, instead of being hidden
            # inside yaw+pitch total error.
            yaw_score = self._axis_score(yaw_error, sigma_deg=12.0)
            pitch_score = self._axis_score(pitch_error, sigma_deg=5.0)
            tracking_reward = 0.55 * yaw_score + 0.70 * pitch_score

            friendly_reward = 0.25 if self._face_state in (1, 2) else 0.0
            precision_bonus = (
                0.35 if yaw_error < 6.0 and pitch_error < 3.5 else 0.0
            )
            smoothness_penalty = 0.0025 * movement + 0.00015 * (
                abs(yaw_vel) + abs(pitch_vel)
            )
            reward = (
                tracking_reward
                + friendly_reward
                + precision_bonus
                - smoothness_penalty
            )
            success_now = (
                yaw_error < 6.0
                and pitch_error < 3.5
                and self._face_state in (1, 2)
            )
        else:
            yaw_error = abs(yaw)
            pitch_error = abs(pitch)
            idle_reward = 0.65 if self._face_state == 0 else 0.0
            center_score = 0.35 * self._axis_score(
                yaw_error, sigma_deg=12.0
            ) + 0.45 * self._axis_score(pitch_error, sigma_deg=5.0)
            smoothness_penalty = 0.006 * movement + 0.00015 * (
                abs(yaw_vel) + abs(pitch_vel)
            )
            reward = idle_reward + center_score - smoothness_penalty
            success_now = (
                yaw_error < 6.0
                and pitch_error < 3.5
                and self._face_state == 0
            )

        self._success_streak = self._success_streak + 1 if success_now else 0
        terminated = self._success_streak >= 10
        truncated = self._step_count >= self.episode_length

        info = {
            "task": self.task,
            "is_success": bool(terminated),
            "angular_error_deg": float(yaw_error + pitch_error),
            "yaw_error_deg": float(yaw_error),
            "pitch_error_deg": float(pitch_error),
            "head_yaw_deg": float(yaw),
            "head_pitch_deg": float(pitch),
            "target_yaw_deg": self._target_yaw_deg,
            "target_pitch_deg": self._target_pitch_deg,
            "wake_word": self._wake_word,
            "face_state": self._face_state,
            "face_name": FACE_NAMES[self._face_state],
            "action_delay_steps": self._action_delay_steps,
            "servo_yaw_command_deg": float(self._servo_command_deg[0]),
            "servo_pitch_command_deg": float(self._servo_command_deg[1]),
            "servo_yaw_command_speed_deg_s": float(
                self._servo_command_vel_deg_s[0]
            ),
            "servo_pitch_command_speed_deg_s": float(
                self._servo_command_vel_deg_s[1]
            ),
        }
        return self._observation(), float(reward), terminated, truncated, info

    def close(self) -> None:
        return None
