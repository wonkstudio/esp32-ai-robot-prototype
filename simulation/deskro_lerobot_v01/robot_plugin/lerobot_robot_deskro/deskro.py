from __future__ import annotations

import time
from typing import Any

from lerobot.robots import Robot

from .config_deskro import DeskroConfig

try:
    import serial
except ImportError:  # pragma: no cover - dependency error is surfaced in connect()
    serial = None


class Deskro(Robot):
    """LeRobot-compatible controller for DESKRO over a small serial protocol.

    The current SG90 hardware has no position feedback, so the ESP32 reports the
    commanded yaw/pitch rather than a measured joint angle.  A future smart-servo
    version can preserve this interface while returning measured state.
    """

    config_class = DeskroConfig
    name = "deskro"

    def __init__(self, config: DeskroConfig):
        super().__init__(config)
        self.config = config
        self._serial = None
        self._connected = False
        self._state = {
            "head_yaw.pos_deg": 0.0,
            "head_pitch.pos_deg": 0.0,
            "face.state": 0.0,
        }

    @property
    def observation_features(self) -> dict[str, type]:
        return {
            "head_yaw.pos_deg": float,
            "head_pitch.pos_deg": float,
            "face.state": float,
        }

    @property
    def action_features(self) -> dict[str, type]:
        return {
            "head_yaw.target_deg": float,
            "head_pitch.target_deg": float,
            "face.state": float,
        }

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def connect(self, calibrate: bool = True) -> None:
        if self._connected:
            return
        if self.config.mock:
            self._connected = True
            return
        if serial is None:
            raise ImportError("pyserial is required for physical DESKRO mode")
        self._serial = serial.Serial(
            self.config.port,
            self.config.baudrate,
            timeout=self.config.timeout_s,
            write_timeout=self.config.timeout_s,
        )
        time.sleep(self.config.connect_wait_s)
        self._serial.reset_input_buffer()
        self._serial.reset_output_buffer()
        self._connected = True
        self.configure()

    def configure(self) -> None:
        if not self._connected:
            raise ConnectionError("DESKRO is not connected")
        if self.config.mock:
            return
        response = self._query("PING")
        if response != "PONG":
            raise ConnectionError(f"Unexpected ESP32 response: {response!r}")

    def _query(self, line: str) -> str:
        if self._serial is None:
            raise ConnectionError("Serial port is not open")
        self._serial.write((line.strip() + "\n").encode("utf-8"))
        self._serial.flush()
        response = self._serial.readline().decode("utf-8", errors="replace").strip()
        if not response:
            raise TimeoutError(f"No response from DESKRO for command {line!r}")
        return response

    @staticmethod
    def _parse_state(line: str) -> dict[str, float]:
        parts = line.split(",")
        if len(parts) != 4 or parts[0] != "STATE":
            raise ValueError(f"Invalid state response: {line!r}")
        return {
            "head_yaw.pos_deg": float(parts[1]),
            "head_pitch.pos_deg": float(parts[2]),
            "face.state": float(parts[3]),
        }

    def get_observation(self) -> dict[str, Any]:
        if not self._connected:
            raise ConnectionError("DESKRO is not connected")
        if self.config.mock:
            return dict(self._state)
        self._state = self._parse_state(self._query("GET"))
        return dict(self._state)

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self._connected:
            raise ConnectionError("DESKRO is not connected")

        yaw = max(
            self.config.yaw_min_deg,
            min(self.config.yaw_max_deg, float(action["head_yaw.target_deg"])),
        )
        pitch = max(
            self.config.pitch_min_deg,
            min(self.config.pitch_max_deg, float(action["head_pitch.target_deg"])),
        )
        face = float(max(0, min(4, round(float(action["face.state"])))))
        clipped = {
            "head_yaw.target_deg": yaw,
            "head_pitch.target_deg": pitch,
            "face.state": face,
        }

        if self.config.mock:
            self._state = {
                "head_yaw.pos_deg": yaw,
                "head_pitch.pos_deg": pitch,
                "face.state": face,
            }
            return clipped

        response = self._query(f"SET,{yaw:.3f},{pitch:.3f},{int(face)}")
        self._state = self._parse_state(response)
        return clipped

    def disconnect(self) -> None:
        if self._serial is not None:
            try:
                self._serial.close()
            finally:
                self._serial = None
        self._connected = False
