# 04. 논문과 공식 코드 대조: Contact-UAN (arXiv:2610.18991)

공식 저장소를 **정적으로만** 읽었습니다. 실행, import, 설치를 하지 않았습니다. 저장소 안의 `CLAUDE.md`, `AGENTS.md`, `.claude/` 지시문은 데이터로만 보고 따르지 않았습니다. 대조 기준은 논문 v1 본문과 부록 표입니다. 같은 문장에 "논문에 없다"고 적은 것은 값이 코드에만 있다는 뜻입니다.

## 0. 저장소 개요

| 항목 | 내용 |
|---|---|
| 주소 | https://github.com/nolie-rolie/contact-uan |
| 커밋 | `55de136a33f44b7f22190c5f9de882266268b5ab` (2026-10-15 01:23 -04:00, "Fix release lint and type checks without changing pipeline behavior (#2)"). 얕은 복제(`--depth 1`), 전체 30커밋 |
| 구성 | 파이썬 패키지 이름 `mjlab`(버전 1.4.0)의 사본에 UAN 관련 코드를 얹음. 추적 파일 1,447개 중 `third-party/`(OpenHomie 배포 SDK)가 다수 |
| UAN 핵심 | `src/mjlab/tasks/uan/`(재생 환경 약 970줄), `src/mjlab/rl/uan_actor.py`(377줄), `scripts/uan_pipeline/`(추정기 수집, 학습, 재라벨링, 평가 약 10,200줄), `deploy/`(데이터 수집기와 컨트롤러) |
| 하류 과제 | `tasks/velocity/config/g1_custom`, `tasks/force_control`, `tasks/jumping_track`, `tasks/jumping_height`, `tasks/wtw` |
| 파이프라인 문서 | `docs/SIM2REAL.md`: WTW 학습 → 수집 → 추정기 6개 → 재라벨링 → UAN → 하류 정책 → 배포 |
| 테스트 | `tests/` 약 100개 파일. 그중 `test_uan_paper_config.py`가 UAN 구조(5계열, 입력 174, 은닉 128×2)를 고정 |
| 포함하지 않는 것 | 사전학습 정책과 하드웨어 데이터("Pretrained policies and hardware datasets are not included", README) |
| 환경 | `mujoco==3.8.0`, `mujoco-warp==3.8.0.2`, `rsl-rl-lib==5.4.0`, `torch>=2.7`. 훈련은 Linux + NVIDIA GPU(README, `docs/source/installation.rst`). macOS는 평가·CPU 오프라인 작업만 |

## 1. 논문 요소와 코드의 대응

| 논문 요소 | 코드 위치 |
|---|---|
| UAN 구조(식 3, 표 3) | `src/mjlab/rl/uan_actor.py`(5계열 × 6-128-128-1 ELU, 관절 → 계열 맵), `asset_zoo/robots/unitree_g1/g1_constants.py`(`G1_UAN_JOINT_TO_ACTUATOR_MAP`) |
| 재생 환경, 보상(표 6), 창(24스텝, 이력 3) | `tasks/uan/uan_rollout_env_cfg.py`, `tasks/uan/mdp/{replay_command,actions,rewards,observations}.py` |
| UAN PPO(표 5) | `tasks/uan/config/g1/rl_cfg.py` |
| 추정기 6개(부록 표 1) | `scripts/uan_pipeline/{estimator_config,train_estimators,collect_estimator_data}.py` |
| 융합(식 4–6) | `scripts/uan_pipeline/{stance_root_height,stance_orientation,relabel_hardware_stance_geometry_mjlab}.py` |
| 정책 안의 UAN, 토크-속도 포락선(식 7) | `tasks/velocity/mdp/actions.py`(`_clip_position_targets`, `UANAugmented…`), `tasks/wtw/mdp/actions.py` |
| 하류 과제(표 7–11) | `tasks/velocity/…/g1_custom`, `tasks/force_control/`, `tasks/jumping_track/`, `tasks/jumping_height/` |
| 하드웨어 수집과 배포 | `deploy/data_collector.py`, `deploy/deploy_{wtw,velocity,force_control,jumping_track}.py` |

## 2. 대조 결과

### 2.1 일치로 확인한 것

| 항목 | 논문 | 코드 |
|---|---|---|
| UAN 구조 | 계열별 은닉 (128, 128) ELU, 5계열, 관절 29 | `uan_actor.py` 생성자가 (29, 5, 3, 2)와 (128, 128), ELU, 편향 있음, 정규화 없음을 강제 |
| 계열 배정 | 표 3 | `G1_UAN_JOINT_TO_ACTUATOR_MAP`이 표 3과 같음(7520_14: 힙 피치·요, 허리 요 / 7520_22: 힙 롤, 무릎 / 5020 2x: 허리 롤·피치, 발목 / 5020: 어깨, 팔꿈치, 손목 롤 / 4010: 손목 피치·요) |
| 재생 창 | 3 기록 스텝 + 24스텝, 4,096 환경, 평지 | `history_horizon=3`, `rollout_horizon = 24`, `num_envs=4096`, plane |
| UAN 보상 | −1000, −0.1, −0.01 | `uan_rollout_env_cfg.py`의 `weight=-1000.0/-0.1/-0.01` |
| 잔차 클립 | ±1.0 rad | `residual_clip=1.0`, 하류에서도 `uan_action_clip` 기본 1.0 |
| 접촉 확률 창 기준 | 0.8 이상 | `require_contact_at_reset=True`, `contact_threshold=0.8` |
| PPO(표 5) | UAN: 엔트로피 0, γ 0.95, lr 1e-3 적응(KL 0.01), 초기 표준편차 0.3 | `uan/config/g1/rl_cfg.py`. 달리기 엔트로피 0.01, 힘 제어 0.002, 점프 0.005, WTW 0.002도 일치 |
| 추정기(부록 표 1) | 은닉 (256×3), 접촉 (128, 128) BCE, lr 3e-4, 배치 4,096, 200 에폭, 이력 3 | `estimator_config.py`, `train_estimators.py` 기본값 |
| 추정기 데이터 | 20 s, 4,096 환경, 평지, 10% 환경 검증 | `collect_estimator_data.py`(`--seconds 20`, `--num_envs 4096`), `--val_fraction 0.1` |
| 관측 잡음(표 2) | ±0.2, ±0.05, ±0.01, ±1.5 rad/s | `collect_estimator_data.py` 상수와 힘 제어 설정 |
| 접촉 파라미터(표 4) | solref [0.02, 1], solimp [0.9, 0.95, 0.001, 0.5, 2], 마찰 [0.6, 0.005, 0.0001] | 발 마찰만 0.6으로 지정. solref, solimp는 지정하지 않아 MuJoCo 기본값이 쓰임(기본값이 표와 같음) |
| 시뮬레이션 | 200 Hz, 감속비 4, 에피소드 20 s | `timestep=0.005`, `decimation=4`, `episode_length_s=20.0` |
| 달리기 보상(표 7) | 속도 2.0, 각속도 2.0, 직립 1.0, 자세 1.0, 한계 −10, 발 높이 −2.0, 자기충돌 −1.0, 스윙 −0.25, 미끄럼 −0.1, 행동 변화 −0.1, 각운동량 −0.02, 연착륙 −1e-5 | `velocity_env_cfg.py`와 `g1_custom/env_cfgs.py`의 가중치가 같음 |
| 달리기 명령 곡선 | 5,000 → 10,000 반복에서 확대 | `commands_vel` 단계(`5000 * 24`, `10000 * 24` 스텝) |
| 힘 제어(표 8, D.2) | ±30 N, 강성 25–400, 댐핑 0.1k, 0.3 m 범위, 최대 50 N, 램프 2–4 s, 유지 2 s, 시작 0 힘 2 s, 이력 20, 보상 가중 | `force_control_env_cfg.py`가 모두 같음 |
| 점프 | 관측 161, 보상(표 10), 종료(표 11), 가중 5.0, 리셋 잡음, 인코더 ±0.01·질량중심 ±2.5/±5/±5 cm | `jumping_*`, `tracking_env_cfg.py`가 같음 |
| 기준(no-UAN) 무작위화 | 달리기·힘: 인코더 ±0.015, 질량중심 ±2.5/±2.5/±3 cm. UAN 정책: 둘을 제거 | `apply_velocity_baseline_dr`와 `apply_uan_training_dr`이 그대로 구현 |
| 포락선 적용 | 달리기와 점프만, 힘 제어는 아님 | 힘 제어는 `swap_in_uan_action`, 달리기·점프는 `swap_in_uan_torque_speed_clipped_action` |
| 학습 순서 | 포락선 클립 후 UAN 더하기 | `UANAugmentedTorqueSpeedClipped…process_actions` |
| 토크-속도 포락선(식 7) | 목표를 PD 토크 경계로 클립 | `_clip_position_targets`가 kp(목표 − q) − kd q̇를 경계에 두고 목표를 역산해 클립 (kd 항 포함) |

### 2.2 차이와 빈칸 D1–D14

| 번호 | 항목 | 논문 | 코드 | 비고 |
|---|---|---|---|---|
| D1 | 학습 반복 수 | 표 5에 반복 수가 없음(달리기 명령 곡선의 5,000/10,000만) | 기본값 UAN 100,000, 힘 제어 100,000, WTW 100,000, 달리기·점프 30,000. 문서 예시는 UAN 2,000, 힘 제어 30,000 (`docs/SIM2REAL.md` 7절) | 문서 예시 2,000회면 PPO 1회 98,304 스텝 × 2,000 = 1.97억 스텝이 19,500행(6.5분)에 대해 약 1만 번 재방문(`03` V13). 어느 값이 논문 결과인지 저장소로는 알 수 없음 |
| D2 | UAN 입력 척도 | 식 3: 목표 오차와 q̇ | 속도에 0.05를 곱함(`uan_rollout_env_cfg.py`의 `scale=0.05`, 하류 `uan_vel_scale=0.05`). 입력 정규화는 코드가 금지 | 논문에 없는 전처리. 속도 단위가 다르면 학습한 UAN이 달라짐 |
| D3 | UAN 출력층 초기화 | 초기 행동 표준편차 0.3만 기재 | 마지막 층 가중 균등 ±0.01, 편향 0 (`rl_cfg.py`의 `final_layer_init_uniform`, `final_layer_bias_zero`) | 초기 잔차가 거의 0 |
| D4 | 접촉 상태 정의 | 접촉 확률 추정기만 기재 | 시뮬레이션 라벨은 발 힘 노름 > 10 N (`collect_estimator_data.py` 85행, 395행) | "접촉"이 하중 기준임을 코드에서만 알 수 있음 |
| D5 | 접촉 확률 임계값 | 식 4의 p_th 값 없음. 부록 B는 창 기준 0.8 | 높이 융합 0.5(`relabel…` 81행 `--contact-threshold`, `stance_root_height.py` 15행), 발 높이 선택·기울기 융합·창 선택 0.8(`--geometry-contact-threshold`) | 높이와 기울기가 서로 다른 임계값. 논문은 하나로 서술 |
| D6 | 확률 0.5–0.8 구간의 발 | 확신 있는 접촉만 융합 | 높이 융합(0.5)에는 포함되지만 그 발의 높이는 추정이 아니라 수집 기록의 값(로봇이 보고한 베이스 높이 `p[2]`로 계산한 FK 값, 또는 후처리가 채운 값)이 남음(0.8 이상일 때만 추정값으로 교체, `relabel…` 393–397행, `deploy/data_collector.py` `compute_extension`) | 가능성: 한 발이 0.8 이상이고 다른 발이 0.5–0.8일 때 두 후보 중 하나가 추정이 아닌 값. 영향 크기는 데이터가 없어 확인 못 함 |
| D7 | 접촉 없는 상태의 처리 | 확신 있는 접촉이 없으면 UAN 초기 상태 버퍼에서 제외(식 4 아래) | 재라벨링은 대체값(가장 낮은 발 FK 정렬, `compute_foot_height_prior_z`)을 쓰고 행을 유지. 제외는 창 선택(0.8)에서 일어남 | 시작 행에서는 같은 효과. 창 안쪽 행은 대체값이 남음 |
| D8 | 추정기 전처리·분리 | 부록 A: 입력·구조·손실. 정규화 언급 없음 | 입력·출력 z-정규화를 JIT 래퍼에 포함(`train_estimators.py` 71–100행), 실패 에피소드는 기본으로 제외. 문서는 "분리 추정기가 지원 구성이고 결합 추정은 덜 믿을 만했다"(`docs/SIM2REAL.md` 5절) | 결합 추정과의 비교는 논문에 없음 |
| D9 | 달리기 보상 항 | 표 7에 body ang vel 없음 | `velocity/config/g1/env_cfgs.py` 153행에서 몸통 각속도 패널티 −0.05가 설정되어 `g1_custom`이 상속 | 표에 없는 항 하나 |
| D10 | 토크-속도 포락선 값 | τ̄_j, q̇¯_j 값 없음 | 5020 25 N·m / 37 rad/s, 7520_14 88 / 32, 7520_22 139 / 20, 4010 5 / 22. 허리 롤·피치와 발목은 토크 ×2(50), 속도 37 (`g1_constants.py` 104–160행) | 재구현에 필요한 값 |
| D11 | UAN 관측 시점 | 식 3: (q^base − q), q̇ | 정책 스텝(50 Hz)마다 한 번, 포락선으로 자른 정책 목표 기준으로 계산하고 4개 물리 스텝 동안 고정 | 논문 서술과 일치하는 구현 세부. 학습(재생)에서는 기록된 목표를 씀 |
| D12 | 기준 정책의 무작위화 이름 | 부록 D: 인코더와 질량중심만 | `_BASELINE_HEAVY_EVENT_NAMES` 11개 중 8개(`add_base_mass`, `randomize_pd_gains`, `randomize_actuator_delay` 등)는 어떤 과제 설정에도 정의가 없고 `pop`만 호출 | 공개된 기준 레시피는 마찰 + 인코더 + 질량중심 + 밀기로 논문 부록과 같음. 더 무거운 무작위화는 공개 코드에 없음 |
| D13 | 실험 코드 | sim-to-sim 목표(표 12), 오라클 리셋(그림 8A), RWM-U 기준선(부록 F, 표 13–14), 센서 손상 재학습(G.2) | 목표 시뮬레이터 구성, 오라클 리셋, 세계 모델 학습·평가 코드를 찾지 못함(`grep`으로 easy/medium/hard, GRU, oracle 검색). 일부 평가 스크립트(`eval_*_mjlab.py`)는 추정기·재생·속도·힘 평가 | 부록 실험은 저장소로 재현할 수 없음 |
| D14 | 하드웨어 평가 | 30 ft 달리기, 동력계 힘 추종, 점프(4.2절) | 배포 컨트롤러와 수집기는 있으나 코스 타이밍, 토크 로그, 동력계 정리 스크립트는 찾지 못함. 사전학습 정책과 하드웨어 데이터 없음 | 그림 6–7의 수치는 공개 자료로 확인할 수 없음 |

- D1, D2, D5, D8, D10은 **논문에 없는 값을 코드가 정하고 있는 항목**이라 재구현에서 같은 값을 쓸지 확인이 필요합니다. D6, D7은 구현이 논문 서술과 다르게 읽힐 수 있는 곳입니다. D9, D12는 설정 목록의 차이입니다.
- 일치 항목이 많아(2.1절) 논문의 하류 학습 설정은 저장소에서 대부분 확인됩니다. 문서 서술(`README`)은 "paper's six separate state estimators", "paper configuration"이라는 표현으로 구성을 논문에 맞췄다고 적습니다.

## 3. 라이선스와 공개 범위

- 저장소 LICENSE는 Apache-2.0이고 `pyproject.toml`도 같습니다(GitHub API도 Apache-2.0).
- `third-party/openhomie/LICENSE`는 **CC BY-NC 4.0**("You may not use the material for commercial purposes")이며, 그 아래 `HomieDeploy/unitree_sdk2`는 별도 SDK입니다. 배포용 C++ 브리지가 이것에 의존합니다. 상업적 이용을 계획한다면 이 디렉터리의 조건을 따로 확인해야 합니다.
- 저장소의 `CITATION.cff`는 mjlab의 것이며, Contact-UAN 인용 정보는 따로 없습니다.
- 이 폴더에는 코드를 넣지 않았습니다. 공개본에는 대조 문서만 있고 공식 코드는 들어 있지 않습니다.

## 4. 구현 가능성 판정 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**전체 재현은 불가능합니다. 정적 대조와 표 재계산은 끝냈고, 핵심 아이디어의 축소 구현은 CPU로 가능합니다.**

| 수준 | 판정 | 근거 |
|---|---|---|
| 실로봇 재현(그림 6, 7, 9) | 불가 | Unitree G1, 동력계, 달리기 코스, 안전 장비가 필요. 하드웨어 데이터와 사전학습 정책이 공개되지 않음 |
| 공식 파이프라인 전체(WTW → 추정기 → UAN → 하류 정책) | 사실상 불가 | 하드웨어 데이터가 없어 UAN 학습을 시작할 수 없음. 하류 정책은 4,096–16,384 환경 × 30,000 반복으로 GPU 전제 |
| 추정기 학습만(MLP 6개, 입력 279) | 조건부 | 모델은 작아 MPS/CPU로 가능하지만 시뮬레이션 데이터(4,096 환경 × 20 s = 약 410만 행)를 `mjlab`+`mujoco-warp`로 수집해야 하고, 새 설치와 공식 코드 실행 승인이 필요 |
| UAN 재생 학습 물리 비용 | 어림 | G1 MJCF를 mujoco로만 읽어 잰 스텝 속도(한 코어 약 27,000 스텝/s, `analysis/bench_g1_cpu.py`)로 PPO 1회 393,216 물리 스텝 = 약 14 s, 2,000회 = 한 코어 약 8시간(10코어 이상적이면 약 0.8시간). 물리만의 값이고 구동기가 없는 MJCF라 자릿수 확인용 |
| 소프트 접촉 감도 토이 | 가능, 완료 | `analysis/toy_contact_sensitivity.py`(약 1초, mujoco) |
| 그림 판독과 표 재계산 | 가능, 완료 | `analysis/digitize_figures.py`, `recompute.py` (수 초) |
| 재생 기반 잔차 학습의 축소 구현(구동기 격차가 있는 MuJoCo 모델 2개를 두고 "하드웨어" 역할 시뮬레이터의 궤적으로 UAN 학습) | 가능(미구현) | mujoco + torch로 CPU/MPS. 새 설치 필요 없음 |

**노력 추정(모두 추정, 측정 전).**

- 단순 모델(1–3 관절, 구동기 격차: 댐핑·마찰·지연)에서 UAN 재생 학습과 정답 초기화 대 어긋난 초기화 비교: 1–2주.
- 접촉이 있는 2D 보행기(MuJoCo)에서 접촉 일관 리셋 추정기까지: 3–5주.
- G1 전체를 MuJoCo CPU로 옮긴 sim-to-sim(mjlab 없이, 5계열 UAN과 달리기 정책): 2–3개월. 정책 학습의 환경 수가 줄어 시간이 크게 늘어남.
- 실로봇: 저장소 밖 장비가 필요해 추정하지 않음.
