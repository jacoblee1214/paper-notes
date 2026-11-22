# TacSL 공식 코드 대조 (04)

*범위:* 공식 저장소 `isaac-sim/IsaacGymEnvs`의 `tacsl` 브랜치 커밋 `f769e39`(2024-08-27)를 **얕은 복제해서 정적으로 읽었습니다.** 실행, import, 설치를 하지 않았습니다. 복제본은 `repo/`에 있고 gitignore에 있습니다. 체크포인트, 데이터, 바이너리는 받지 않았습니다.

## 1. 권리 (라이선스)

| 위치 | 라이선스 표시 | 판단 |
|---|---|---|
| 저장소 루트 `LICENSE.txt` | BSD 3-Clause, "Copyright (c) 2018-2023, NVIDIA Corporation" | 저장소 기본 라이선스 |
| `isaacgymenvs/tasks/tacsl/tacsl_base.py`, `tacsl_task_insertion.py` 등 | BSD 3-Clause 머리말 (2024 NVIDIA) | BSD 3 |
| `isaacgymenvs/tacsl_sensors/` 안의 **촉각 센서 모듈 6개 파일** (`tacsl_sensors.py`, `tactile_utils/gelsight_render.py`, `shear_tactile_viz_utils.py`, 예제 3개) | "NVIDIA CORPORATION and its licensors retain all intellectual property … Any use, reproduction, disclosure or distribution … without an express license agreement from NVIDIA CORPORATION is strictly prohibited." | **독점 조건.** BSD 3과 충돌 |
| `isaacgymenvs/tasks/amp/poselib/` 안의 파일 8개 | 같은 독점 문구 | 이 분석의 범위 밖 |
| `pyproject`, `setup.py` 메타데이터 | GitHub API 라이선스 NOASSERTION | 참고만 |

**판단:** 논문의 핵심인 촉각 시뮬레이션 모듈(`tacsl_sensors`)은 **독점 조건이 붙은 파일**입니다. 같은 저장소 BSD 3 표기와 충돌하므로, 이 코드를 복사하거나 공개 저장소에 넣지 않습니다. 읽기와 대조는 이 분석 안에서만 합니다. 다시 구현하려면 논문 식만 보고 새로 짜야 합니다(§4).

**논문 자체의 권리:** arXiv 라이선스 링크는 비배타적 배포 허락입니다. 그래서 논문 그림은 게시본에 넣지 않습니다. 원문 PDF는 저장소에 있지만 공개하지 않습니다.

## 2. 논문과 코드의 차이 (D1–D11)

| 번호 | 논문 | 공식 코드 (정적 확인) | 판정 |
|---|---|---|---|
| **D1** | 온라인 증류(DAgger), 오프라인 BC, **AACD** 2단계 학습 (§IV-A, §IV-B2, 알고리즘 2–3) | 저장소 안에 DAgger나 BC 구현이 없습니다 (`dagger`, `behavior_clon`, `distill`, `aacd`, `pretrained` 검색 0건). AACD 흔적은 `train.py` 216행의 `load_critic_only` 한 줄뿐이고, `freeze_critic`은 문서에만 있습니다. 실제 동작은 외부 `rl_games` 코드에 달려 있어 여기서 확인하지 못했습니다. 문서 `docs/tacsl.md`는 AACD 2단계를 설명합니다 | **미구현 (공개 범위 밖).** 학습 도구의 핵심 두 가지가 공개되지 않음 |
| **D2** | 페널티 힘장의 k_n, k_d, k_t, µ | 값이 코드에 하드코딩됨: k_n = 1.0, k_d = 0.003(사용 안 함), µ = 2.0, k_t = 0.1 (`tacsl_sensors.py` 360–363, 701–704행). 법선 힘 식에서 감쇠 항은 **주석 처리됨** (`#- self.tactile_damping * depth_dot * depth`, 859행). 논문 식 (f_n의 k_d ḋ 항)과 다름 | **차이.** 논문은 값을 안 적었고, 공식 코드는 댐핑을 뺌 |
| **D3** | §V-A와 §V-B: 물리 무작위화와 영상 증강을 학습에 씀 | 설정 기본값이 모두 **꺼짐**: `randomize_compliance: False`, `use_ep_image_aug: False`, `randomize_color_channel: False`, `socket_pos_xyz_obs_noise: [0,0,0]`. 공개 학습 스크립트 3개(`train_state.sh`, `train_tactile.sh`, `train_wrist.sh`)와 문서 예제도 이 플래그를 켜지 않음 | **차이.** 스크립트 그대로 돌리면 §V의 sim-to-real 처방이 빠짐 |
| **D4** | 보정 κ = 200과 300 (부록 B), 학습 범위 150–350 | 설정 기본 `compliance_stiffness: 350.0`, 범위 `[150, 350]`(무작위화를 켤 때만). 문서의 Isaac Lab 예제는 κ = 100, 감쇠 1.0을 씀 | **범위는 일치, 기본값은 다름.** 보정 절차 코드는 없음 |
| **D5** | v1은 "Isaac Gym", v2는 "Isaac Simulator". 참고문헌 [13]은 Isaac Gym 논문 | 저장소는 `IsaacGymEnvs`이며 `tacsl_setup.md`가 "**deprecated Isaac Gym**"이라고 적음. "Isaac Lab에서 공식 코드를 공개할 예정"이고 "이 버전은 더 이상 지원하지 않는다"고 명시 | **공식 공개 상태 차이.** 논문 시뮬레이터명은 공식 코드가 가리키는 엔진과 다름 |
| **D6** | 표 VII 환경 수 128, 표 VIII DAgger 256 롤아웃/반복 | 공개 스크립트는 촉각 영상 학습 64 환경, 상태 학습 512 환경을 씀. horizon 512, 학습률 1e-4, γ 0.99, 클립 0.2, GAE 0.95는 일치 (`TacSLTaskInsertionPPO_LSTM_dict_AAC.yaml`). DAgger 설정은 코드에 없음 | **일부 차이.** 환경 수는 실험별로 달라 보임. DAgger 항목은 검증 불가 |
| **D7** | 표 V "관절 감쇠 잡음 [−1.5, 1.5]" | `ige_dr`의 Franka 관절 감쇠 범위 `[0.00001, 3.0]`, 덧셈 방식 (`TacSLTaskInsertion.yaml`) | **정의 불명.** 표의 "잡음"이 중심값 기준 덧셈인지 확인 불가 |
| **D8** | 표 VI CNN 채널 **[32, 32, 64]**, 커널 [8, 4, 3], 스트라이드 [2, 1, 1] | 공식 학습 설정의 CNN 필터는 **[32, 64, 64]** (`train_tactile.sh`, `docs/tacsl.md`, 같은 커널과 스트라이드) | **불일치.** 두 번째 층 채널이 32 대 64 |
| **D9** | §VI-B1 본문: 막대 회전은 **±20°** 범위. 표 V: peg X축 회전 **[−0.628, 0.628] rad** (±36°) | 공식 스크립트는 `plug_noise_rot_in_gripper=[0.0, 0.628, 0.0]`(y축 0.628 rad = ±36°). 설정 주석은 y축 범위를 [−π/10, π/10] (±18°)이라고 적음 | **세 값이 서로 맞지 않음.** 본문 ±20°, 표 ±36°, 주석 ±18° |
| **D10** | 소형 설명 문서 | `docs/tacsl.md`의 인용 항목이 제목 "IndustReal"과 학회 "ArXiv"로 되어 있음. 이미지 증강 링크가 `isaacgymenvs/task/`로 되어 있으나 실제는 `tasks/` | **문서 오류** (사소) |
| **D11** | 표 IV, 시뮬 학습 입력 크기 | 촉각 영상 80×60×3 (`obsDims`, 스크립트 `[80,60,3]`), 손목 64×64×3 (`[64,64,3]`)은 일치. 행동: 위치 0.01 m, 회전 0.05 rad (`pos_action_scale`, `rot_action_scale`)과 60 Hz 작업 공간 임피던스(dt = 1/60, substeps 2)도 일치 | **일치 (검증 항목)** |

**일치 확인 항목 (D11 외):** 촉각 힘장의 마찰 식 f_t = −(v_t/‖v_t‖) min(k_t‖v_t‖, µ‖f_n‖)는 코드(`ft_world`, 864–869행)와 같습니다. 법선 힘 f_n의 강성 항도 같습니다. 접촉 강성 κ는 PhysX의 `compliance` 속성으로 들어가며, 감쇠 c는 `compliant_damping`으로 들어갑니다(`tacsl_sensors.py` 78–107행).

**코드에 남은 자체 표시 문제:** `query_distance_and_normal_with_physx`의 법선 방향에 `# TODO: check direction of normal` 주석이 있습니다. 부호 규약(거리 = −SDF 값)도 논문 표기와 다릅니다. 논문 식을 그대로 옮길 때 부호를 따로 검사해야 합니다.

## 3. 저장소 정보 (GitHub 표)

| 저장소 | 관계 | 라이선스 | 최근 push | 별 | 상태 |
|---|---|---|---|---|---|
| [isaac-sim/IsaacGymEnvs](https://github.com/isaac-sim/IsaacGymEnvs) 브랜치 `tacsl` | 공식 코드 (정책 학습 도구, 센서 모듈) | BSD 3 (루트) + 독점 표기 (센서 모듈) | 2024-10-26 (main). `tacsl` 브랜치 최신 커밋 2024-08-27 `f769e39` | 2,949 | **저장소 archived 표시** (2026-11 기준). 복제 크기 344 MB (asset 198 MB, docs 20 MB) |
| [iakinola23/tacsl](https://github.com/iakinola23/tacsl) | 프로젝트 페이지 저장소 (README와 index.html) | 없음 | 2025-07-11 | 3 | 코드 아님 |
| [Denys88/rl_games](https://github.com/Denys88/rl_games) | 설치 의존성 (PPO, AAC) | 확인 안 함 | 2026-11-21 | 1,409 | 분석 안 함 |
| Isaac Lab TacSL 센서 (NVIDIA 공식 문서 `docs.nvidia.com/isaaclab`) | **후속 공식 구현** (실험적, `isaaclab_contrib`의 `VisuoTactileSensor`) | 문서만 확인 | 문서 Release 3.0 | — | §4 참조 |

- 프로젝트 페이지의 "Code" 버튼은 `isaac-sim/IsaacGymEnvs`의 `tacsl` 브랜치 설치 문서를 가리킵니다.
- 설치 문서는 Isaac Gym **TacSL 전용 바이너리**를 Google Drive에서 받으라고 합니다. 이 바이너리는 받지 않았습니다.

## 4. 구현 가능성 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**판정: 이 맥에서 공식 코드를 실행하거나 논문 수치를 재현하는 것은 불가능합니다. 논문 식을 따른 독립 재구현(CPU/MPS 소규모)은 가능하고, 일부는 이미 했습니다.**

| 항목 | 필요 자원 | 이 맥 | 판정 |
|---|---|---|---|
| 공식 코드 (Isaac Gym TacSL) 실행 | Linux, NVIDIA GPU와 CUDA, Python 3.8, 전용 바이너리 (Drive), torch 2.2 | macOS, CUDA 없음 | **불가** |
| Isaac Lab TacSL 센서 예제 | Isaac Sim (`uv run --extra isaacsim`), NVIDIA GPU, PhysX 백엔드 | Isaac Lab 2.3 설치 문서가 요구하는 OS는 Ubuntu 22.04 또는 Windows 11입니다. macOS 지원 없음 | **불가** |
| 속도 재현 (표 I, II) | 같은 GPU와 CPU 기준선 | 불가 | **불가** |
| 정책 학습 재현 (표 IV, 512 환경, 1500 반복) | GPU 학습 수십 시간 이상 (추정) | 불가 | **불가** |
| 실로봇 (표 실험, 81회) | Franka, 촉각 센서, 컨트롤러 브리지 | 불가 | **불가** |
| 페널티 힘장 식의 독립 재구현 (구 물체, numpy) | numpy | 가능 | **가능 (미실행, 2–4일 추정)** |
| 접촉 식 (5)–(9) 검산 | numpy | 가능 | **완료** (`analysis/toy_implicit_contact.py`, 잔차 8e−11) |
| AACD 2단계 저차원 토이 (독립 구현) | torch, MPS 또는 CPU | 가능 | **조건부** (추정 1주; 공식 코드 없음이라 기준 대조 불가) |

- **노력 추정 (측정 전, 추정):** 공식 환경을 Linux GPU 서버에 올리는 것은 설치와 디버깅 포함 2–5일입니다. 같은 세팅으로 학습 한 번은 수 시간에서 하루입니다. 논문 표 IV 재현은 세 알고리즘과 일곱 입력 조합을 세 시드씩 돌려야 하므로 GPU 여러 대에서 1–2주입니다. 이는 추정치이며 측정하지 않았습니다.
- **Isaac Lab 경로 (공식 후속):** 같은 센서 상수(k_n = 1.0, µ = 2.0, k_t = 0.1)를 쓰고 감쇠 항은 없습니다(D2와 같음). 따라서 이 맥에서 할 수 있는 것은 문서 대조까지입니다.

## 5. 검증 방법과 한계

- 정적 읽기만 했습니다. 실행, import, 설치, 바이너리 다운로드는 하지 않았습니다.
- 공개 저장소의 `tacsl` 브랜치 커밋만 봤습니다. 논문 제출 이후 바뀐 내용은 확인하지 못했습니다.
- Isaac Lab 문서는 공식 페이지의 마크다운을 한 번 받아 읽었습니다. 저장소 메타데이터(API)는 요청 한도 때문에 Isaac Lab 쪽은 확인하지 못했습니다.
- 학습 쪽 D1(AACD)은 `rl_games` 설치본을 보지 않아 완전히 확인하지 못했습니다.
