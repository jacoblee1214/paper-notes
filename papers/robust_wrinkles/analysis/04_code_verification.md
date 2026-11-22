# 04. 공식 코드 대조와 구현 가능성

## 0. 대상과 방법

- 대상: `github.com/trustworthyrobotics/CORD-SLS` 커밋 **`77e82b4098e47c3420f7d3f00ba870ce37a485b0`** (2026-08-08, "Update grip position in drag rope task"). 커밋은 두 개뿐입니다(초기 `5a80709` 2026-06-11, 위 커밋). arXiv v1은 2026-06-12에 올라왔으므로 초기 커밋이 논문 시점의 코드입니다. GitHub API로 초기 커밋의 `benchmark/methods/SLS.py`, `MPPI.py`, `PPO.py`, `APG.py`, `planners/gpu_sls/src/gpu_sls/gpu_sqp.py`를 받아 비교했고 **모두 HEAD와 같습니다.** 달라진 것은 `drag_rope_task.py`의 그리퍼 초기 높이 하나(0.01 m → 0.02 m)뿐입니다(D12).
- 방법: **정적 읽기만** 했습니다. 코드를 실행, import, 설치하지 않았고, 데모 `.pkl` 4개(총 약 1.5 MB)는 pickle을 열지 않았습니다(파일 크기만 확인). 아래 "(추정)" 표시가 붙은 것은 실행 없이 코드와 JAX 의미에서 추론한 것입니다.
- 파일 구성(`repo/`): `environments/`(밧줄 `rope_env.py` 505줄, 천 `cloth_env.py` 649줄, 충돌 `collision.py`), `benchmark/`(과제 4개와 방법 4개, `run_benchmark.py` 112줄), `planners/gpu_sls/`(GPU-SLS 사본), `examples/`(노트북 12개). README는 제목 한 줄입니다.

## 1. 설치 요구사항과 라이선스

| 항목 | 내용 |
|---|---|
| `requirements.txt` | `jax[cuda13]`, `viser`, `trajax`(GitHub 설치) 세 줄뿐. 코드가 import하는 `optax`, `orbax`, `flax`, `gymnasium`, `stable_baselines3`, `torch`, `rich`는 적혀 있지 않음 |
| `planners/gpu_sls` | `pyproject.toml`: `jax[cuda12]==0.8.1`, mujoco, mujoco-mjx, gym-quadruped, pycddlib 등. `environment.yaml`은 python 3.11 |
| 정밀도 | `run_benchmark.py`는 JAX float64가 기본(`--jax_x64 True`). PPO만 float32를 쓰는 부분이 있음(D11) |
| 장치 | `torch.cuda.is_available()` 분기가 PPO의 BC에 있고, JAX 장치는 지정하지 않음(GPU 없으면 CPU) |
| 라이선스 | **최상위 LICENSE 없음**(GitHub API license: null). `planners/gpu_sls/LICENSE`만 MIT(저작자 JeffreyFang, 2026). `external/primal_dual_ilqr`에는 라이선스 파일이 없음. 최상위 코드(시뮬레이터, 벤치마크)를 내 저장소에 가져가려면 저자 확인이 필요합니다 |

## 2. 논문 ↔ 코드 대조 (D 항목)

| 번호 | 항목 | 논문 | 코드 (파일:줄) | 판정과 영향 |
|---|---|---|---|---|
| **D8** | **SLS 관(tube) 계산이 벤치마크에서 켜져 있는가** | 표 1의 Ours는 CORD-SLS(견고 MPC), 그림 2b는 "uncertainty tubes" | `benchmark/methods/SLS.py:47`과 예제 노트북 전부(`plan_*.ipynb`)가 `enable_fastsls=False`. `gpu_sqp.py:206–209`: `use_nominal = not(enable_fastsls) or ...`이고 `use_nominal`이면 `run_nominal`(QP만 풀고 `backoffs = 0`, Φ 전부 0)을 호출. `enable_output_feedback`는 저장소 어디서도 True로 설정되지 않음(기본 False, `gpu_sls.py:19`). 상류 `gpu_sls` 저장소도 같은 논리 | **다름(정적 읽기, 중대)**. 저장소의 벤치마크 설정으로는 SLS 단계가 건너뛰어져 "Ours"는 제약이 있는 **공칭 SQP-MPC**(ADMM QP, 1회 SQP, 백오프 0)로 읽힙니다. 관과 출력 피드백이 표 1의 수치에 기여하지 않을 수 있습니다. 논문 실험이 다른 설정으로 돌았을 가능성은 배제 못 하며 실행으로 확인하지 않았습니다 |
| **D9** | 플랜트와 모델의 관계, 외란 | 불확실 동역학 (식 1), "견고성" | 모든 방법이 실행에 `task.step`(같은 `RopeEnv.step`)을 쓰고(`SLS.py:118`, `MPPI.py:119`) 외란을 주입하는 코드가 없음. 계획에 넣는 외란은 `alpha = 0.003 * dt = 6e-5`의 등방 값(`SLS.py:29`) | 표 1의 안전율은 **완전 모델, 무잡음 롤아웃**의 것입니다. 하드웨어 CP 반지름 0.03 m의 500분의 1 규모(`03` V8)라 설령 관이 계산되어도 매우 작습니다. "외란을 고려하지 않아 RL이 위반한다"(§5)는 서술은 이 평가에서 외란이 없으므로 직접 시험된 것이 아닙니다 |
| D1 | 안전율의 정의와 제약 | "obstacle avoidance, goal reaching, task requirements" (§1), 안전율(표 1, 정의 없음) | `task.py:29–40`: 위반 단계 수 `num_constraint_violations_steps`를 셈(단계 비율이 표의 값과 정합, `03` V3). 제약: 들기, 접기, 펴기는 **제어 한계(속도 ±0.2, 그립 ±10)뿐**(`lift_rope_task.py`, `fold_cloth_task.py:65–70`, `flatten_cloth_task.py`), 끌기만 z ≥ −0.05(사실상 항상 만족)와 장애물 반경 0.1 m(`drag_rope_task.py`). 기록되는 제어는 **클립 전 명령**(`SLS.py:117`, `MPPI.py:118`)이고 `step`이 클립함 | 천 과제에서 MPPI의 안전율 64%, 80%는 "클립 전 명령이 제어 한계를 넘은 단계 비율"이며 물리적 안전 위반이 아닙니다. 밧줄 들기에서 PPO의 96.5%도 같은 정의입니다. 장애물이 있는 끌기만 기하 제약이 있습니다 |
| D14 | 천 접기의 높이 제약 | 부록 D: "cloth height를 3 cm로 제한하는 제약" | `fold_cloth_task.py`의 `constrtaints`는 제어 한계뿐 | 논문 서술과 다름. 그림 8의 "3 cm" 제약은 저장소 벤치마크에 없음 |
| D2 | MPPI의 장애물 벌점 | "장애물 벌점을 키우면 밧줄이 초기 위치에 머문다" (§5, 결과 없음) | `MPPI.py:75`: `비용 + 10 · Σ max(c, 0)²`, 장애물 구속 c = r² − d² (m², 최대 0.01). 노드 한 개 한 단계의 최대 벌점 10 × 0.01² = 1e-3. 비용은 `1.0·‖err‖² + 9.0·‖err[:3]‖²`(끌기, 오차 O(0.1–1)). 시드와 무관하게 `PRNGKey(0)` 고정 | **기준선이 거의 구속을 무시하도록 설정됨**(추정, 벌점이 비용의 약 1% 이하). MPPI의 안전율 1.0%는 알고리즘의 한계라기보다 이 가중의 결과일 수 있습니다. 벌점을 키운 실험은 없음 |
| D3 | 그리퍼 부착 평활화 함수 | D_i = D_max exp(−‖c_i‖²/κ) (§4.1) | `rope_env.py:272`, `cloth_env.py:385`: `k *= exp(−sqrt(d² + 1e-12)/κ)`, 곱하기 `sigmoid(15(c_grip − 0.5))`. κ = 1e-3 m. 하드 모델은 `d² ≤ (r_밧줄 + r_그리퍼)²` 표시자(`:270`) | **형태가 다름**: 거리의 제곱이 아니라 거리 자체이고 가우시안이 아니라 지수. 감쇠 길이가 1 mm라 가중치는 5 mm에서 0.7%, 10 mm에서 4.5e-5, 하드 접촉 반경(들기 17 mm)에서 4.1e-8, 들기 초기 높이 50 mm에서 2e-22(`03` V9). 논문의 κ→0 해석과 단위가 다르고, "접촉 전 기울기"는 몇 mm 안에서만 유효합니다 |
| D5 | 접촉 평활화 외 요소: 지면 접촉과 2단계 | 지면 접촉은 Anitescu 볼록 완화(식 5)를 QP(식 7)로 풀고 KKT를 미분(식 9), 완화 λᵀ(…) = κ | `rope_env.py:423`: 지면 접촉은 벌점 힘 `f = Jᵀ smoothrelu(−k φ, κ·0.0)`을 **한 번의 선형 풀이**로 적용(`:424–426`). `smoothrelu`의 평활 계수에 `·0.0`을 곱해 평활화가 꺼져 있음. KKT 미분은 없고 자동 미분을 그대로 씀. 탄성 헤시안의 미분은 `stop_gradient`(`:332`). 구속 잔차에 `0.9 · φ` 계수(`:317`, 논문에 없음) | **다름(방법 수준)**. 마찰 원뿔은 다각형화(`PolygonizeFrictionCone`)되지만 볼록 원뿔 QP가 아니라 벌점 근사입니다. 논문의 식 (5), (7), (9)는 지면 접촉에 구현되지 않았고, 평활화는 그리퍼 부착 가중에만 있습니다. 1단계(그리퍼 구속)의 `custom_vjp`는 식 (8), (10)과 같은 구조(LU 인수분해 재사용, `rope_env.py:339–409`)로 확인됨 |
| D4 | 순방향은 비평활, 기울기는 평활 | §4.1 끝 | `rope_env.py:450–451`: `stop_gradient(true) + smoothed − stop_gradient(smoothed)` | **일치**(straight-through surrogate). 비평활 순방향이 "contact_smoothing=0"의 표시자 모델과 같음 |
| D6 | PPO 예산과 시연 | PPO는 CORD-SLS 시연 10개로 웜스타트, "같은 상호작용 예산"(§5), 표 1 16M, 40M | `PPO.py:310`: `total_timesteps = n_epochs(10,000) × horizon` = 들기 2M, 끌기 5M(표 값의 1/8). 환경은 하나이고 `batch_size = 8`은 미니배치 크기. 시연은 `load_demo(task, k=0)` 한 개(`run_benchmark.py:74`), `.pkl` 크기가 한 궤적(200 또는 500단계 × (상태 + 제어) × 8 B)과 일치(밧줄 들기 140,986 B ≈ 200 × 77 × 8 + 여유) | **다름**: 기본 설정은 PPO 예산이 APG의 1/8이고 시연은 한 궤적입니다. 그림 6의 가로축이 1.6e7, 4e7이므로 논문 실행은 8배 설정이었을 수 있으나 저장소에는 없음. "10 시연"의 정의도 코드로는 확인 안 됨 |
| D7 | APG의 사전학습 | PPO만 웜스타트라고 서술 (§5) | `APG.py:103`: `pretrain_policy`(BC, Adam 1e-2, 10,000회)를 **APG 학습 전에도** 같은 데모로 수행. 신경망 64–64 tanh, 에포크 10,000, 배치 8, lr 1e-4, 초기 상태 잡음 1e-3 | APG도 CORD-SLS 시연으로 시작하므로 "APG는 미분 가능 시뮬레이터 덕분에 탐색 불필요"를 시연 효과와 분리할 수 없음. 비용은 `cost + 1e-3 Σ max(c, 0)²`(`APG.py:35`) |
| D10 | 저장소에 없는 것 | 정합 예측(§4.4), 지각(부록 B), 하드웨어 제어, 정리 1, 그림 3과 9의 출력 피드백 롤아웃, DEFORM 비교, 표 3 | 모두 없음. 출력 피드백은 `plan_rope_weld.ipynb` 한 곳에서만 `output_equation=h(q)=q`로 호출(15세그먼트 + 용접 끝이라 그림 9의 설정과 다름)하되 `enable_output_feedback`는 기본값 False라 실제 경로(`gpu_sqp.py:213`)는 타지 않음 | 논문의 CP, 하드웨어, 출력 피드백 주장은 저장소로 확인할 수 없음 |
| D13 | 병렬 스캔 구현 | 식 (16)–(17)의 결합 연산으로 칼만 재귀 병렬화 | `gpu_sls_of.py:59, 203`: Φ 전파를 `lax.associative_scan`으로 행렬곱 결합. 관측자 이득은 이중(dual) TV-LQR을 k마다 `vmap`으로 풂(`get_observer_gains`, `:105–172`)하며, 풀이기 `tvlqr_gpu`(`primal_tvlqr.py:338–450`)가 `lax.associative_scan`(병렬 LQ 결합)을 씀. 식 (16)과 같은 모양의 결합 규칙이 `controller_pas`(`gpu_sls.py:27`)에 있음 | **일치(구조)**. 칼만 이득을 식 (16)의 (Ã, D̃, P̃) 원소로 직접 푸는 것이 아니라 이중 LQ로 푸는 같은 계열의 구현입니다. 식 (16)–(17) 자체는 우리가 수치로 확인함(`03` T2). 다만 출력 피드백 경로는 D8로 비활성 |
| D15 | MPPI의 표본·horizon과 표 1의 평가 횟수 | 표 1: MPPI 1,024,000(밧줄), 64,000(천) | `MPPI.py:71, 89`: N = 20(밧줄) 또는 10(천), 표본 1024. 표 값을 200단계로 나누면 단계당 5120(= 1024 × **5**), 천 320. 코드 설정이면 20,480과 10,240 (JAX `debug.callback`이 vmap에서 표본마다 호출된다는 가정, 추정). SLS의 N + 7은 코드와 맞음 | **불일치(추정)**: MPPI 횟수는 이 저장소의 설정(N = 20, 10)이 아닌 다른 설정에서 나온 것으로 보입니다. MPPI 쪽 비교(190배)의 근거가 다른 설정일 수 있습니다 |
| D16 | 끌기 비용의 슬라이스 | 끌기 비용: 전 노드 + 첫 노드 가중, 제어 비용 (§5에 식 없음) | `drag_rope_task.py`의 `cost`가 그리퍼 2개 기준 슬라이스를 그대로 씀: `state_err[:-6]`, `control_err[:-2]`. 끌기는 그리퍼 1개(상태 끝 3차원, 제어 4차원)라 **마지막 노드의 좌표(3차원)가 상태 비용에서 빠지고 제어 비용에는 vx, vy만** 남음(vz, 그립 제외) | 사소하지만 끌기의 비용 함수가 의도와 다를 수 있음(추정). 표 1의 끌기 결과 해석에는 영향이 작음 |
| D11 | PPO 행동의 클립과 float32 | 안전율(표 1) | SB3 `predict`는 행동을 Box로 클립. 제약은 float64 `_u_max`와 비교(`task.py:33`), PPO 행동은 float32(`PPO.py:9` x64 해제, 실행 시 `run_benchmark.py:35`가 다시 켬) | 가설: 포화한 행동이 float32 반올림으로 0.2보다 약간 커져(0.20000000298) 위반으로 집계될 수 있음. 들기의 PPO 96.5%(유일한 제약이 제어 한계)가 이 때문인지는 실행 없이 확인 못 함 |
| D12 | 시간 지표와 HEAD 차이 | "Execution Time Per-Step" (표 1) | `SLS.py:98–111`, `MPPI.py:102–113`: 단계 시간의 **합**(첫 단계 제외). RL은 `run_PPO`, `run_APG`가 추론 시간의 합. 체크포인트 없이 돌리면 `execution_time`은 **학습 시간 전체**(`PPO.py:332`) | 표의 "단계당"은 합을 단계 수로 나눈 값이라고 읽힘. 변환 방법은 코드에 없음. 초기 커밋과 HEAD의 차이는 끌기 그리퍼 초기 z 하나(0.01 → 0.02) |

### 2.1 논문 주장의 코드 쪽 확인 (일치한 것)

- 순방향 비평활, 기울기 평활 straight-through (D4), 1단계 암묵 미분(`custom_vjp`, LU 재사용) (D5의 일부).
- 상태 차원: 밧줄 21노드 + 그리퍼(끌기 66, 들기 69), 천 10×10 노드 + 그리퍼 2개(306). 논문의 "66 states", "300 DoF"과 일치(`03` K2).
- 계획 horizon: SLS와 MPPI 모두 밧줄 N = 20, 천 N = 10(`SLS.py:27`, `MPPI.py:71`). SLS의 평가 횟수 단계당 27, 17(= N + 7)이 표 1과 맞음(`03` V4).
- APG의 평가 횟수 16M, 40M = 배치 8 × horizon × 10,000 (`03` V4).
- 평가 지표 `final_state_error`는 들기, 접기, 펴기에서 전 노드 L2 노름, **끌기에서는 첫 노드(`state[:3]`)의 위치 오차만**(`drag_rope_task.py`). 표 1의 "Goal Error"가 이 정의인지 논문에 적혀 있지 않습니다.

### 2.2 논문이 말하지만 코드에서 확인할 수 없는 것

CP 교정(0.03 m, 87%), SAM3 + TAPNext++ 파이프라인, KUKA 제어, 부록 F 표 2·3의 측정(DEFORM, DaXBench 비교 코드 없음), 그림 6의 시드 수와 음영 정의, 그림 3과 9의 롤아웃 설정(D10).

## 3. 안정성과 증명 검토

- **정리 1(부록 A)**: 증명은 q_k = q̄_k − e_k와 ‖e_k‖ ≤ ρ의 결정론적 포함 관계뿐이라 맞습니다. 그러나 보장의 핵심은 전제 "P(q̄_k ∈ T̄ ∀k) ≥ 1−δ"이고, 이는 CP 교정과 견고 SLS 합성이 함께 줘야 합니다. 하드웨어의 교정(n = 100, 87%)은 이 전제를 만족하지 못합니다(`03` V7, T1).
- 교환 가능성 가정(무작위 100개 제어에서 교정, 폐루프에서 사용)은 시험되지 않았고, 부록 D가 "재계획 때문에 실행 궤적이 이전 관 밖으로 나갈 수 있다"고 밝혀 폐루프 보장도 없습니다.
- 관은 **평활화 편향을 포함하지 않습니다**(한계 절은 [17]에 위임). 따라서 "certified"의 범위는 비평활 순방향과 유계 외란 모델에 대한 것입니다(`02` §4).

## 4. 구현 가능성 (이 맥)

이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음. 설치된 것은 저장소 루트 가상환경의 torch 2.15와 mujoco 4.0뿐이고 **JAX는 없습니다.** 새 설치와 공식 코드 실행, 데이터 다운로드는 하지 않았습니다.

**판정: 논문 규모의 재현과 하드웨어 재현은 불가능합니다. 공식 코드의 소규모 실행은 조건부 가능(JAX CPU 설치와 실행 승인 필요, 측정 전)이고, 수치 재계산과 직접 짠 소형 검사는 가능하며 일부 끝냈습니다.**

| 수준 | 판정 | 근거와 노력 |
|---|---|---|
| 하드웨어 실험 (그림 1, CP 교정) | **불가** | KUKA iiwa 두 대, Schunk 그리퍼, Orbbec 카메라 두 대, RTX 두 대. 지각, 제어 코드는 저장소에 없음(D10) |
| 표 1 전체 재현 (4과제 × 4방법) | **불가에 가까움** | MPPI(1024표본 × N = 20 × 매 단계 암묵 풀이)와 APG/PPO 학습(10,000 에포크 × 배치 8 × 200–500단계 미분 롤아웃, 또는 PPO 2M–16M 시뮬 호출)은 GPU 전제. CPU 시간은 측정하지 않음 |
| 공식 코드로 밧줄 과제 한 개의 SLS 또는 MPPI를 JAX CPU로 실행 | **조건부** | JAX CPU 설치(`jax[cuda13]`는 불가, CPU 휠로 교체), `optax`, `viser` 등 설치 승인과 코드 실행 승인. 밧줄은 63차원이라 CPU에서도 가능할 것으로 추정되나 단계당 시간은 측정 전. 천(306차원)은 느릴 것(추정). 앞의 D8 때문에 이 실행은 SLS가 아니라 공칭 SQP-MPC를 돌리게 됨 |
| 직접 짠 소형 준정적 밧줄 시뮬레이터(numpy/torch, 21노드, 암묵 오일러 + 그리퍼 스프링)로 평활화 기울기와 접촉 발견 비교 | **가능** | 1–2주(추정). 접촉 평활화의 효과를 절제로 직접 시험하는 것이 논문에 없는 부분 |
| CP 교정 설정의 궤적 수준 보장 검사, 병렬 스캔 결합 규칙 검사 | **가능, 완료** | `toy_cp_coverage.py`, `toy_scan_riccati.py` (각 1 s 미만) |
| GPU-SLS의 torch 포팅 | **조건부** | MPS에서 float64 불가라 float32 정밀도 문제 가능. 3–6주(추정, 측정 전) |
| 표 1과 2, 3의 수치 재계산 | **가능, 완료** | `recompute.py` |

## 5. 재실행

```bash
python3 analysis/recompute.py
python3 analysis/toy_cp_coverage.py
python3 analysis/toy_scan_riccati.py
```

공식 코드는 `repo/`에 정적으로만 있고, 실행하지 않았습니다. 다시 받으려면 `git clone --depth 1 https://github.com/trustworthyrobotics/CORD-SLS repo`입니다.
