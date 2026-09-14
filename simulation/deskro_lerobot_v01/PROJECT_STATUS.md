# Project status

## 이번 패키지에서 구현됨

- LeRobot EnvHub 규격 `env.py`
- MuJoCo DESKRO V1 2-DOF 몸체
- 여러 병렬 환경 생성
- domain randomization 기초
- 실제 `LeRobotDataset` v3 기록 스크립트
- LeRobot 서드파티 로봇 플러그인 구조
- 실제 하드웨어 없이 확인하는 mock mode
- ESP32 Serial bridge 펌웨어
- PPO baseline 학습/평가 스크립트

## 아직 실물 검증 안 됨

- ESP32 Serial bridge와 PC 플러그인의 실제 연결
- SG90 실제 각도 피드백(현재 SG90은 명령 각도만 알 수 있음)
- INMP441 마이크
- MAX98357A 스피커
- 시뮬 정책을 실물에 자동 배포하는 rollout
- LeRobot 공식 HIL actor/learner에 DESKRO를 직접 연결하는 작업

`PPO baseline`은 현재 Stable-Baselines3를 사용합니다. LeRobot는 EnvHub, 데이터셋,
하드웨어 플러그인 계층에 실제로 사용됩니다.
