# DESKRO V1 EnvHub

LeRobot EnvHub 규격의 MuJoCo 가상환경입니다.

- `env.py`: LeRobot가 찾는 진입점
- `deskro_env.py`: DESKRO Gymnasium 환경
- `model.xml`: 2-DOF 가상 몸체

LeRobot에서 사용하는 핵심 상태는 `agent_pos`이며, LeRobot 전처리 단계에서
`observation.state`로 변환됩니다.
