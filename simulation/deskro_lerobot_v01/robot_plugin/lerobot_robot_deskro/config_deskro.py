from dataclasses import dataclass

from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("deskro")
@dataclass
class DeskroConfig(RobotConfig):
    """Configuration for an ESP32 DESKRO robot.

    Set ``mock=True`` to test LeRobot without connecting physical hardware.
    """

    id: str | None = "deskro_v1"
    port: str = "COM6"
    baudrate: int = 115200
    timeout_s: float = 1.0
    connect_wait_s: float = 2.0
    mock: bool = True
    yaw_min_deg: float = -60.0
    yaw_max_deg: float = 60.0
    pitch_min_deg: float = -25.0
    pitch_max_deg: float = 25.0
