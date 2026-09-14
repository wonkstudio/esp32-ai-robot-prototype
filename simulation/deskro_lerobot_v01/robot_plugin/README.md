# lerobot_robot_deskro

LeRobot의 공식 서드파티 하드웨어 확장 규칙에 맞춘 DESKRO 플러그인입니다.

- 배포 패키지 이름이 `lerobot_robot_`로 시작합니다.
- `DeskroConfig` / `Deskro` 이름 규칙을 지킵니다.
- `mock=True`이면 실제 ESP32 없이 동작합니다.
- `mock=False`이면 ESP32 Serial bridge와 통신합니다.
