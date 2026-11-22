# 04. 공식 코드 대조: GELLO

공식 저장소를 **얕은 복제로 읽기만** 했습니다. 실행, import, 설치는 하지 않았습니다. 클론은 `repo/gello_software/`(4.4 MB, `1b3fec5`)와 `repo/gello_mechanical/`(68 MB, `8ea3f2e`)입니다. 두 폴더는 `.gitignore`에 있습니다. 복제 내용은 신뢰할 수 없는 데이터로 취급했습니다.

대조 기준은 논문 v2입니다. 공식 저장소의 HEAD는 2026년 11월 상태이고 논문은 2024년 7월 상태이므로, 차이 중 상당수는 **저장소가 논문 이후 발전한 것**입니다. 이 점을 구분해 적었습니다.

## 1. 라이선스와 공개 상태

| 저장소 | 라이선스 | 근거 | 비고 |
|---|---|---|---|
| gello_software | MIT (저작권 2023 Philipp Wu) | `LICENSE` 파일 직접 확인 | 재사용 시 MIT 고지 필요 |
| gello_mechanical | MIT | GitHub API `license: mit` (파일 확인은 README 인용까지) | STL 공개. CAD는 README에 따르면 저자에게 요청해야 합니다 |

- 논문 라이선스(비배타 배포)와 코드 라이선스(MIT)는 별개입니다. 코드에서 복사한 내용은 MIT 고지를 따릅니다.
- BOM과 조립 안내는 Google Docs 링크입니다. 접속하지 않았습니다(외부 문서, 다운로드 없음). 원가 확인은 못 했습니다.

## 2. 논문과 코드의 차이 (D1–D9)

| ID | 항목 | 논문 (v2) | 공식 코드 (HEAD) | 판정 |
|---|---|---|---|---|
| D1 | 서보 모델 | DYNAMIXEL **XL330-288T** (§III-a) | 드라이버는 모델을 고정하지 않고 `servo_types`로 토크-전류 매핑을 받음 (`gello/dynamixel/driver.py`). ROS 2 설정은 `xl330-m288-t`와 `xl330-m077-t` 두 변형 | **표기 차이.** 논문 모델명과 코드 변형의 대응은 확인하지 못함. 재현 시 모델 번호를 장치에서 읽어 확인해야 함 |
| D2 | 제어 주기 | 기재 없음 | `RobotEnv` 기본 `control_rate_hz=100` (`gello/env.py`), 실행 설정 기본 `hz=30` (`gello/utils/launch_utils.py`, `configs/*.yaml`의 `hz: 30`) | **논문 공백.** 실제 설정은 30 Hz. 재현 시 값을 명시해야 함 |
| D3 | 관절 대응 | 관절 위치를 그대로 명령 (역기구학 없음) (§III-b) | 로봇마다 `joint_offsets`(π/2의 배수)와 `joint_signs`(±1)를 적용 (`gello/agents/gello_agent.py`의 `PORT_CONFIG_MAP`) | **구현 세부는 논문에 없음.** 기구학 등가는 맞지만 보정 상수가 필요함. 구조는 논문과 일치 |
| D4 | 지원 로봇 | 3종: UR5, xArm7, Franka Panda | 저장소 설명 (mechanical README): Franka FER, UR5, xArm7 + YAM, Lite6, AR4, FR3, FR3 Duo. software README는 YAM, FR3(ROS 2), FER, UR, xArm | **버전 차이.** 논문 이후 추가. 논문 결과는 원래 3종에 대한 것 |
| D5 | 통신 | ZMQ (§III-b 끝) | `zmq.REQ/REP`, `pickle` 직렬화, 수신 타임아웃 1000 ms, 주소 `tcp://host:port` (`gello/zmq_core/robot_node.py`) | **일치.** 주의: `pickle.loads`는 신뢰하지 않는 네트워크에서 임의 코드를 실행할 수 있습니다. 로컬 또는 신뢰 네트워크에서만 씀 |
| D6 | 축소 비율 α = 0.5 | 링크 길이를 0.5배로 축소 (§III-b) | 소프트웨어에는 α가 없습니다(기하는 STL로 정해짐). mechanical README에도 치수 설명 없음 | **확인 불가.** 치수 대조는 이번 작업에서 하지 않음 (`03` C14) |
| D7 | 중력과 힘 | 패시브 관절 규제(스프링, 고무줄) (§III-c). 힘 피드백 없음 (§V) | 패시브 규제 설계 흔적 없음. YAM용 FACTR 에이전트는 Dynamixel 전류(토크) 제어와 중력 보상 (`gello/factr/yam_gello_factr_agent.py`, `gravity_compensation.py`, pinocchio 기반) | **차이 (범위).** 논문의 "힘 피드백 없음"은 원 설계에 대한 말이고, 저장소는 YAM에서 토크 제어를 씀. 원 GELLO의 패시브 규제는 하드웨어에만 있음 |
| D8 | 인용 정보 | 학회 표기 없음 (arXiv) | README 인용: `@misc{wu2023gello}`, 연도 2023 | **갱신 필요.** 학회 표기는 IROS 2024 (Crossref DOI `10.1109/IROS58592.2024.10801581`). 저장소 인용 정보는 갱신되지 않음 |
| D9 | 완료 기준 및 기록 | 과제 제한 시간 45 s, 90 s; 실패 유형 분류 (§IV-A) | 시간 제한 설정, 과제 평가 코드는 저장소에 없음 (사용자 연구 프로토콜은 공개되지 않음) | **재현 불가 영역.** 사용자 연구 코드는 공개되지 않음 |

차이 D1–D9 중 **논문 결과를 바꾸는 것은 없습니다.** 논문의 실험은 하드웨어 기반 사용자 연구이고, 소프트웨어 버전 차이는 재현에 필요한 설정 정보를 바꿉니다(D1, D2, D3).

### 2.1 소프트웨어 구조 (읽은 범위)

- `gello/agents/gello_agent.py`: 로봇별 `DynamixelRobotConfig` (`joint_ids`, `joint_offsets`, `joint_signs`, `gripper_config`).
- `gello/dynamixel/driver.py`: 제어 테이블 주소(`ADDR_GOAL_POSITION=116`, `ADDR_PRESENT_POSITION=132`, `ADDR_TORQUE_ENABLE=64`), 기본 보드레이트 57600.
- `gello/zmq_core/robot_node.py`: 리더 상태를 ZMQ로 제공.
- `gello/env.py`: 제어 루프 (`Rate` 기반).
- `gello/factr/`: YAM용 FACTR 중력 보상과 토크 제어. 논문 이후 추가된 코드입니다.
- `ros2/`: 프랑카 FR3용 ROS 2 패키지 (별도 README).

이번 작업에서 읽지 않은 부분: 드라이버 전체 흐름, ROS 2 패키지, 시뮬레이션 설정, 테스트 코드. 필요하면 추가로 읽습니다.

## 3. 실행하지 않은 것

- `pip install`, `uv`, `setup.py`, 드라이버 빌드, `third_party/DynamixelSDK` 설치: 하지 않았습니다.
- 실행 스크립트(`scripts/`, `gello/utils/launch_utils.py`, `gello/factr/*`): 실행하지 않았습니다.
- 공식 코드에 대한 가짜 로봇(`FakeDynamixelDriver`) 실행 시험: 하지 않았습니다. 필요하면 **사용자 승인 후** 가상환경에서 가능합니다.

## 4. 구현 가능성 판정

**판정: 전체 재현은 이 맥에서 불가능합니다. 장치 제작, 실로봇, 사람 사용자 연구가 필요합니다. 공식 소프트웨어의 CPU 실행과 정적 대조는 승인 후에 가능합니다.**

이 맥(Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음) 기준:

| 항목 | 가능 여부 | 근거 |
|---|---|---|
| GELLO 하드웨어 제작 (3D 프린트, 서보, U2D2) | **불가 (이 작업 범위 밖)** | 장비 구매와 조립이 필요합니다. 제작 비용은 논문 기준 약 300 USD. 노력 추정 1–2주 (측정 전, 조립 숙련도에 따라 다름) |
| 실로봇 원격조작 (UR5, xArm7, Panda) | **불가** | 실로봇과 네트워크 제어 인터페이스가 필요합니다 |
| 사용자 연구 (12명, 5과제) | **불가** | 사람 참가자와 장비가 필요합니다 |
| 리더 관절 읽기 + ZMQ 루프백 (가짜 드라이버) | **조건부** | 공식 코드 실행과 의존성 설치 승인 필요. CPU로 가능 |
| 관절 매핑 (offset, sign) 수식 검사 | **가능** | numpy만으로 가능. 설정 값을 읽어 정적으로 계산 |
| 시뮬 로봇 (MuJoCo Panda)에 같은 관절 대응을 붙여 텔레옵 데이터 생성 | **조건부** | 직접 짠 MuJoCo 환경은 승인 범위 안입니다. 새 설치가 필요하면 승인 |
| 중력 보상 (FACTR) 재현 | **불가 (YAM 하드웨어)** | 토크 제어 실기 필요 |

**노력 추정 (측정 없음).** 관절 매핑과 시뮬 연결 검증은 1–2일. 하드웨어 GELLO를 실제로 만들고 한 로봇을 조작하는 데는 2–4주 (부품 조달 포함). 사용자 연구 재현은 IRB와 참가자 모집이 필요해 이 저장소의 범위 밖입니다.

이 판정은 `README.md` 맨 위에도 같은 내용으로 씁니다.
