# SimToolReal 논문 ↔ 공식 코드 대조 (정적 읽기)

대상: 공식 저장소 `repo/`(tylerlum/simtoolreal, MIT, HEAD `313d5ae`, 2026-08-13, 얕은 clone). **실행, import, 설치는 하지 않았습니다.** 파일을 읽고 값을 대조했습니다. 논문 쪽 값은 v3 PDF의 표 I과 부록 C–D입니다.

## 1. 저장소 구조 요약

- `isaacgymenvs/tasks/simtoolreal/env.py`: 환경 본체(보상, 관측, 리셋, 랜덤화). `cfg/task/SimToolReal.yaml`: 주 설정.
- `isaacsimenvs/`: Isaac Sim (Isaac Lab, Python 3.11) 환경. README는 **권장 경로**로 소개합니다.
- `rl_games/`: SAPG 등 학습기. `deployment/`: 실로봇(Sharpa 노드, 목표 포즈 노드)과 sim-to-sim.
- `dextoolbench/`: 평가 스크립트 (`run_all_evals_isaacgym.py`, `run_all_evals_isaacsim.py`). `download_dextoolbench_data.py`와 `download_pretrained_policy.py`는 외부에서 받습니다(받지 않음).
- `baselines/`: 키네마틱 리타게팅(HaMeR, ICP, DLS)과 PyRoki 궤적 최적화.
- README는 "논문 결과는 Isaac Gym 경로로 학습"이라고 씁니다. 따라서 Isaac Gym 쪽이 논문과 대응합니다.

## 2. 대조표

| 항목 | 논문 (v3) | 코드 (값, 위치) | 판정 |
|---|---|---|---|
| 환경 수 | 24,576 (표 I) | `numEnvs` 기본 8192, README 명령에서 24576으로 덮어씀 | 일치 (재정의) — D6 |
| 블록 크기 | 4096 (표 I) | README: `expl_coef_block_size=4096`, 블록 6개 | 일치 |
| 에피소드 길이 | 600 (표 I) | `episodeLength: 600` | 일치 |
| 제어 주파수 | 60 Hz 정책, 120 Hz 물리 (표 I) | `controlFrequencyInv: 1` (60 Hz) | 일치 (물리 120 Hz는 미확인) |
| 물체 포즈 잡음 | 1 cm, 5° (표 I) | `objectStateXyzNoiseStd 0.01`, `objectStateRotationNoiseDegrees 5.0` | 일치 |
| 물체 포즈 지연 | 최대 10스텝 (표 I) | `objectStateDelayMax: 10` | 일치 |
| 행동·관측 지연 | 3스텝 (표 I) | `obsDelayMax: 3`, `actionDelayMax: 3` | 일치 |
| 관절 속도 잡음 | 0.1 rad/s (표 I) | `jointVelocityObsNoiseStd: 0.1` | 일치 |
| 보상 λ_arm, λ_hand | 0.03, 0.003 (표 I) | `kukaActionsPenaltyScale 0.03`, `handActionsPenaltyScale 0.003` | 일치 |
| λ_approach | 50 (표 I) | `distanceDeltaRewScale: 50.0` (이름이 다름) | 일치 (이름 차이) |
| λ_lift, B_lift | 20, 300 (표 I) | `liftingRewScale 20.0`, `liftingBonus 300.0` | 일치 — 단 D5 |
| λ_goal | 200 (표 I) | `keypointRewScale: 200.0` | 일치 |
| B_succ | 1000 (표 I) | `reachGoalBonus: 1000.0` | 일치 |
| 키포인트 척도 | 보상 0.14/0.03/0.03 m (부록 A) | `keypointScale 1.5`, `objectBaseSize 0.04` | 부분 확인 — D2 |
| 물체 위치 범위 | ±10 cm (표 I) | `resetPositionNoiseX/Y 0.1`, `Z 0.02` | 부분 일치 — Z 값 차이 D8 |
| 성공 허용오차 | 1 cm (표 I), 평가 2 cm (IV) | `successTolerance 0.075`→`targetSuccessTolerance 0.01`, `evalSuccessTolerance` 기본 null | 불일치 — D1 |
| 연속 성공 | 논문 미기재 | `successSteps: 10`, `maxConsecutiveSuccesses: 50` | 코드만 있음 — D11 |
| 외력 | 5 N (표 I) | `forceScale: 20.0` × 물체 질량 | 불일치 — D4 |
| 외토크 | 0.5 Nm (표 I) | `torqueScale: 2.0` × 물체 질량 | 불일치 — D4 |
| 테이블 힘 한계 | 100 N (부록 D) | `TABLE_FORCE_THRESHOLD = 100.0` | 일치 |
| 손 이탈 | 1.5 m (부록 D) | 손끝 최대 거리 > 1.5 | 일치 (대상이 손끝) — D9 |
| 리프트 기준 | 0.73 m 절대 (표 I), 10 cm (부록 E) | `liftingBonusThreshold 0.15` (초기 높이 기준) | 불일치 — D5 |
| 목표 샘플 범위 | x ±0.35, y −0.1…0.2, z 0.15…0.52 (부록 C) | `targetVolumeOrigin [0, 0.05, 0.8]` 및 extent | 좌표계가 달라 미확인 — D7 |
| 액션 매핑 | 팔 Δ, k = 0.025 · EMA α 0.1; 손 절대 (부록 C.2) | EMA 함수는 테이블 힘 평활만 확인. 팔·손 EMA는 찾지 못함 | 미확인 — D3 |
| 목표 교체 | 방향 최대 0.1 m, 90° (부록 C.5) | 미확인 | 미확인 — D10 |

## 3. 불일치 목록 (D1–D11)

- **D1 (성공 허용오차, 중요).** 표 I은 1 cm, 본문 IV는 평가 ε = 2 cm입니다. 코드는 학습에서 0.075 m에서 시작해 0.9배씩 줄여 목표 0.01 m로 가고(curriculum), 평가 기본값은 0.01입니다. `dextoolbench/eval_isaacgym.py`(592행)의 평가 설정에서도 `evalSuccessTolerance: 0.01`입니다. 2 cm는 코드에서 찾지 못했습니다. 허용 키포인트 거리 = 허용오차 × `keypointScale`(1.5)입니다(`env.py` 2606행). 실세계 Task Progress의 2 cm가 이 방식과 같은지 확인이 필요합니다.
- **D2 (키포인트 척도).** 논문 부록 A는 보상 키포인트 오프셋을 (0.14, 0.03, 0.03) m으로 고정합니다. 코드는 `keypointScale × objectBaseSize`로 척도를 만듭니다(`env.py` 2124–2142행). 값이 같은지 계산하지 않았습니다. 방향(긴 축 x)은 맞습니다. 미확인.
- **D3 (행동 EMA).** 논문은 팔·손 모두에 EMA(α = 0.1)를 씁니다. 찾은 EMA 함수는 테이블 힘 평활뿐입니다. 팔 델타 제어(k = 0.025)는 찾지 못했습니다. `deployment/`나 `isaacsimenvs/`에 있을 수 있어 미확인입니다.
- **D4 (외력·외토크 단위).** 논문 표 I은 5 N, 0.5 Nm입니다. 코드는 힘을 `forceScale × 질량`, 토크를 `torqueScale × 질량`으로 씁니다(20과 2). 단위가 N/kg 형태이므로 값을 직접 비교할 수 없습니다. 논문 표 I의 단위 설명이 부족합니다.
- **D5 (리프트 기준).** 표 I의 z_lifted = 0.73 m는 절대 높이입니다. 코드 `liftingBonusThreshold = 0.15`는 초기 높이 대비 상대 거리이고(`env.py` 1386, 2366행), 1386행에는 초기 높이 − 0.05 + 임계값을 쓰는 식이 따로 있으나, 용도는 확인하지 않았습니다. 부록 E의 10 cm 절단 기준(데모 자르기)과도 다른 값입니다. 세 기준이 서로 다른 목적인지 논문은 설명하지 않습니다.
- **D6 (환경 수).** 코드 기본값은 8192이고 논문과 README는 24576입니다. 재정의 값이므로 큰 문제는 아닙니다. 다만 GPU 메모리 한계에서 블록 6개 규칙을 지켜야 합니다(README).
- **D7 (목표 샘플 좌표계).** 논문은 테이블 중심 기준 범위를 씁니다. 코드는 `targetVolumeOrigin [0, 0.05, 0.8]`을 씁니다. 좌표 원점이 다르면 같은 범위가 아닙니다. 미확인.
- **D8 (물체 위치 잡음).** 논문 표 I의 물체 위치 범위는 x, y ±10 cm, 테이블 높이 ±1 cm입니다. 코드는 z 방향 `resetPositionNoiseZ 0.02`(2 cm)입니다. 테이블 높이 무작위화와 같은 것인지 확인되지 않았습니다.
- **D9 (손 이탈 기준).** 논문은 "손과 물체 거리 1.5 m 초과"라고 쓰고, 코드는 손끝의 최대 거리(`curr_fingertip_distances.max`)가 1.5를 넘는지 봅니다. 기준이 다릅니다. 영향의 크기는 확인하지 않았습니다.
- **D10 (목표 교체).** 논문은 다음 목표를 최대 0.1 m, 90° 무작위 이동으로 샘플링합니다. 관련 코드를 찾지 못했습니다. 미확인.
- **D11 (코드만 있는 설정).** `successSteps 10`(허용 안에 10스텝 머물러야 성공), `maxConsecutiveSuccesses 50`은 논문에 없습니다. 성공 판정 규칙이 논문보다 구체적입니다. 논문 부록의 "연속 성공" 종료와 연결됩니다.

## 4. 공개 범위와 실행 가능성 (정적 판단)

- **라이선스:** 코드 MIT, 자산 YCB와 KUKA iiwa 라이선스는 `assets/licenses/`에 따로 있습니다. 게시본에는 코드를 복사하지 않습니다.
- **Isaac Gym 실행:** NVIDIA GPU와 Isaac Gym 바이너리(Python 3.8)가 필요합니다. 맥에서는 불가능합니다.
- **Isaac Sim 실행:** NVIDIA RTX GPU와 Isaac Lab(Python 3.11)이 필요합니다. 맥에서는 불가능합니다.
- **사전학습 정책:** 외부 서버(`download.cs.stanford.edu`)에서 받습니다. 받지 않았습니다.
- **데이터:** DexToolBench 데이터는 외부에서 받습니다. 받지 않았습니다.
- **실로봇 배치:** Sharpa 손, KUKA iiwa, ZED, SAM, FoundationPose 포크가 필요합니다. 맥에서는 불가능합니다.
- **정적으로 확인된 것:** 보상, 관측, 지연, 잡음, 정책 구조(LSTM), 학습 설정은 대부분 논문과 맞습니다. 다만 성공 기준(D1)과 힘·리프트 단위(D4, D5)는 논문 본문과 맞지 않습니다. 재현 시 이 값들을 논문 표가 아니라 코드 값으로 맞춰야 합니다.
