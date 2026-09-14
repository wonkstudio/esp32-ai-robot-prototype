# DESKRO × LeRobot v0.1

어렵게 생각할 필요 없이, 이 패키지는 아래 네 가지를 한 폴더에 묶은 첫 버전입니다.

```text
가상 DESKRO 몸체 (MuJoCo)
        ↓
LeRobot가 환경을 읽음 (EnvHub)
        ↓
LeRobot 형식으로 시뮬레이션 데이터를 저장
        ↓
나중에 같은 명령을 실제 ESP32로 전송
```

## 지금 바로 확인하는 순서 — Windows

### 0. 필요한 것

- Python **3.12 이상**
- 인터넷 연결(첫 설치 시 라이브러리 다운로드)
- 실제 로봇은 아직 없어도 됨

### 1. 설치

이 폴더에서 PowerShell을 열고 한 번만 실행합니다.

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
```

### 2. LeRobot가 가상환경을 읽는지 확인

`01_check_env.bat`을 더블클릭합니다.

성공 시:

```text
LeRobot local EnvHub load: OK
observation keys: ['agent_pos']
```

### 3. 가상 로봇 보기

`02_view_env.bat`을 실행합니다.

MuJoCo 창에서 DESKRO가 고개를 좌우/상하로 움직입니다. 주황색 구는
가상 사용자 또는 소리 방향입니다.

### 4. 진짜 LeRobot 형식 데이터 만들기

`03_record_dataset.bat`을 실행합니다.

결과가 다음 폴더에 생성됩니다.

```text
datasets/deskro_sim_v01/
├─ data/        # Parquet
└─ meta/        # info, stats, episode metadata
```

이건 임의 JSON 예제가 아니라 `LeRobotDataset.create()`, `add_frame()`,
`save_episode()`, `finalize()`로 생성되는 LeRobot v3 데이터입니다.

### 5. PPO 학습 시작

`04_train_ppo.bat`을 실행합니다.

학습 결과:

```text
outputs/deskro_ppo.zip
```

현재 PPO trainer만 작은 외부 라이브러리(Stable-Baselines3)를 사용합니다.
가상환경, 데이터 포맷, 향후 실제 로봇 연결은 LeRobot 규격입니다.

### 6. 실제 로봇 연결 코드 미리 확인

`05_test_robot_mock.bat`을 실행합니다. 실제 ESP32 없이도 LeRobot 로봇
플러그인의 관측/행동 인터페이스를 확인합니다.

## 실제 ESP32에 연결할 때

Arduino IDE에서 아래 파일을 업로드합니다.

```text
firmware/deskro_serial_bridge/deskro_serial_bridge.ino
```

그다음 `DeskroConfig(mock=False, port="COM6")`로 바꾸면 PC가 다음 명령을
ESP32로 보냅니다.

```text
SET,30.0,-10.0,2
```

ESP32는 SG90 두 개와 OLED를 움직인 뒤 다음처럼 응답합니다.

```text
STATE,30.000,-10.000,2
```

## 새 로봇 모델을 추가하는 방법

새 모델 저장소에도 아래 하나만 제공하면 LeRobot EnvHub가 읽을 수 있습니다.

```python
def make_env(n_envs=1, use_async_envs=False):
    ...
```

즉 DESKRO V2, 로봇팔, 바퀴 로봇도 각각 EnvHub 저장소로 만들고 LeRobot의
같은 데이터·정책 계층을 재사용할 수 있습니다.

## 현재 정직한 한계

- 실제 ESP32 Serial 연결은 아직 이 패키지에서 실물 검증하지 않았습니다.
- SG90은 실제 각도를 돌려주는 스마트 서보가 아니므로 현재 state는 명령 각도입니다.
- 마이크/스피커/대화 AI는 다음 단계입니다.
- 이 v0.1의 첫 task는 `사용자 방향 바라보기`입니다.

자세한 구현 상태는 `PROJECT_STATUS.md`에 있습니다.
