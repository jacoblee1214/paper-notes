# SuperMimic: 논문 ↔ 공식 코드 대조와 구현 가능성

## 0. 대상과 방법

- **저장소:** `HybridRobotics/supermimic` (Apache-2.0), 커밋 `abc124c` (2026-11-11), 프로젝트 페이지(`supermimic.github.io`)가 저장소의 홈페이지로 지정되어 있습니다. arXiv abs 페이지와 PDF에는 코드 링크가 없습니다. 공식성 근거는 (a) 저장소 홈페이지가 프로젝트 페이지, (b) 조직명 HybridRobotics가 공동 저자 Sreenath의 연구실명으로 보인다는 점(외부 확인은 하지 않음), (c) 제3자 재현 저장소의 설명("For the authors' code see HybridRobotics/supermimic")입니다. (a)와 (c)가 가장 직접적인 근거이고, 저자 본인의 확인은 받지 못했습니다.
- **의존 저장소:** `HybridRobotics/mjlab_research` (Apache-2.0), 커밋 `dd51a9f` (2026-11-11). 저장소 `pyproject.toml`이 mjlab을 이 저장소에서 가져오므로 함께 읽었습니다. 종료 임계, 보상 가중치, 마찰 범위는 여기에 있습니다.
- **클론 위치:** `supermimic/repo/`와 `supermimic/repo/mjlab_research/` (`--depth 1`, LFS 건너뜀, `core.hooksPath` 비활성). `.gitignore`에 `repo/`가 있습니다.
- **방법:** 정적 읽기만 했습니다. 실행, import, 설치, 테스트 실행, 가중치와 데이터 다운로드를 하지 않았습니다. `tests/`와 CI 설정은 읽기만 했습니다.
- **읽은 파일:** `README.md`, `src/supermimic/runners/supermimic_runner.py`, `learners/wm_learner.py`, `learners/policy_learner.py`, `learners/learner_utils.py`, `envs/world_model_env.py`, `models.py`, `representations.py`, `sim_context.py`, `tasks/env_cfgs.py`, `tasks/task_spec.py`, `runners/reset_sampling.py`, `scripts/train.py`, `scripts/benchmark.py`, `mjlab_research`의 `tasks/tracking/tracking_env_cfg.py`, `mdp/terminations.py`, `config/g1/env_cfgs.py`.

## 1. 대조 결과 (D1–D20)

판정: **일치** (논문과 같은 값이 코드에 있음), **차이** (값이나 구조가 다름), **미구현** (논문에 있으나 저장소에 없음), **미확인** (정적으로 확인할 수 없음).

| # | 항목 | 논문 | 코드 (파일·위치) | 판정 |
|---|---|---|---|---|
| D1 | world model 손실 | 식 1: 상태, 보상, 관측, BCE의 제곱오차 합 | 상태 블록별 제곱오차에 1/σ² 가중치(Bryson), 평균으로 합산 (`wm_learner.py` L65–68, L183–191). 보상과 종료에 손실 항 없음 | 차이 (형식). 보상·종료가 해석적이면 손실 항이 없다는 §3.2 문장과는 맞음 |
| D2 | 동역학 구조 | f_ω(ŝ, a, ξ)로 상태 예측 | 가속도 수준 증분(Δv_b, Δω_b, Δv_joint)을 예측하고 자세는 해석적으로 적분 (`representations.py` L472–476) | 논문에 없음. 설명 부족 |
| D3 | 보상과 종료 head | §3.2: "학습되거나 해석적으로 계산" 하지만 d_ω는 학습된 것처럼 씀 | 보상은 예측 상태에 BeyondMimic 식을 적용. 종료 확률은 d̂ = sigmoid((max(\|Δz\|/0.25, tilt/0.8) − 1)/0.05) (`world_model_env.py` L163–220, `termination_softness=0.05` `supermimic_runner.py` L59) | 차이 (표기). 둘 다 해석적 |
| D4 | 정책 입력 | §3.3, 부록 A: 잡음 있는 proprio, 최근 10 관측 | 기본값 `actor_obs="rep_state"`, `history_length=1` (`supermimic_runner.py` L53–58). proprio 구성은 Table A.1과 같음 (`env_cfgs.py` `_sensor_actor_group`) | **차이 (기본값)**. 논문 설정이 별도 실행 스크립트에 있는지 확인 불가 |
| D5 | 목표 창 | K = 10 미래 기준 프레임 | 10프레임, 간격 5 스텝 (약 0.9 s 앞) (`representations.py` L95–100) | 일치 (간격은 논문에 없음) |
| D6 | 네트워크 크기 | 동역학 (2048)³, 정책 (2048,1024,512,256), 비평가 (256,128), 잠재 32, tanh | `dynamics_hidden_dims=(2048,2048,2048)`, `actor_hidden_dims=(2048,1024,512,256)`, `critic_hidden_dims=(256,128)`, `latent_dim=32`, `torch.tanh` (`supermimic_runner.py` L60, L74–76; tanh는 `models.py` L199) | 일치 |
| D7 | 다중 GPU | Table A.5: 데이터 병렬, 기울기 평균. 2 GPU 실행(Table A.8) | DDP, all_reduce, world_size 등 없음 | **미구현** (저장소에 없음). 2 GPU 결과 재현 경로 없음 |
| D8 | 학습률 일정 | Table A.5: 250,000 반복까지 코사인 감소, 0.03배까지 | 스케줄러 코드 없음 (cosine, scheduler, LambdaLR 검색 0건) | **미구현** |
| D9 | 반복 수 | 320,000 | 기본값 `max_iterations=1,000,000` (`supermimic_runner.py` L72) | 차이 (기본값). 320,000은 실행 인자일 수 있음 |
| D10 | 학습률, 기울기 노름 | 2 × 10⁻⁴ (1 GPU), 기울기 노름 1.0 | `lr=2e-4` (`wm_learner.py` L26, `policy_learner.py` L22). 노름 1.0, 원소별 0.1 (`policy_learner.py` L30) | 일치 (2 GPU 학습률은 확인 불가) |
| D11 | world model 배치와 창 | 10 배치 × 2048 창, k = 8 | `num_batches=10`, `batch_size=2048`, `horizon=8` (`wm_learner.py` L27–29) | 일치 |
| D12 | 정책 상상 | H = 32, 배치 4096, Adam (0.9, 0.95) | `horizon=32`, `batch_size=4096`, `beta2=0.95` (`policy_learner.py` L23–30) | 일치 |
| D13 | 비평가 | GAE λ = 0.95, γ = 0.99, 5 에폭 × 4 미니배치, 목표 네트워크 없음 | `ActorCriticPolicyLearnerCfg` γ = 0.99, λ = 0.95. `_fit_critic`는 5 × 4 미니배치, 목표 네트워크 없음 (`policy_learner.py` L173–198) | 일치 |
| D14 | 비평가 값 상한 | 논문에 없음 | `clamp(max = max_step_reward / (1 − γ))` (`policy_learner.py` L163, L169, L187) | **논문에 없는 추가 안정화** |
| D15 | 재설정 분포 | 50% 균등, 40% softmax(−25 G_b / G_max), 10% 유지 | `explore=0.5`, `steepness=25`, `keep=0.1` (`reset_sampling.py` L24–32, L149) | 일치 |
| D16 | 버퍼와 갱신 비율 | 환경당 1000 스텝, 갱신 대 데이터 0.4 | `buffer_size=1000`, `replay_ratio=16`. 계산하면 0.4 (`supermimic_runner.py` L53, L73, L664–668) | 일치 |
| D17 | 행동 잡음 | N(0, 0.1²) | `noise=0.1`를 수집 행동에 더함 (`policy_learner.py` L25, `supermimic_runner.py` L615) | 일치 (0.1을 표준편차로 읽을 때) |
| D18 | 추적 보상 | Table A.2: 가중치 1.0 (몸체), 0.5 (루트), std 0.3, 0.4, 1.0, 3.14; 행동 변화 −0.1; 관절 한계 −2 | `tracking_env_cfg.py` L209–240 (std와 가중치 일치), 행동 변화 −0.1 (L240). 관절 한계는 `env_cfgs.py` L194에서 −2로 바꿈. 자기충돌 항 제거 (`env_cfgs.py` L193) | 일치 |
| D19 | 종료 | Appendix A: \|e_z\| > 0.25 m, e_tilt > 0.8, 손끝 종료 없음 | `bad_anchor_pos_z_only` 임계 0.25, `bad_anchor_ori` 임계 0.8 (`tracking_env_cfg.py` L262–271). 손끝 종료 제거 (`env_cfgs.py` L198) | 일치 |
| D20 | 도메인 무작위화와 관측 잡음 | Table A.1, A.3: 마찰 U[0.3, 1.2], 관절 보정 ±0.01, 몸통 질량중심 ±0.025/0.05, 밀치기 없음 | 마찰 (0.3, 1.2) (`mjlab_research` `tracking_env_cfg.py` L199), 관절 보정 ±0.01 (L190), 밀치기 제거 (`env_cfgs.py`의 이벤트 필터). 관측 잡음은 `representations.py` L114–120의 값이 Table A.1과 같음 | 일치 (질량중심 범위는 확인하지 않음) |
| D21 | 상상 학습에서 world model 고정 | ω를 고정하고 θ, ψ만 갱신 | 정책 갱신 때 동역학과 표현을 `requires_grad_(False)`, 비평가는 별도 (`policy_learner.py` L84–90, L168, L190) | 일치 |
| D22 | 속도 명령 범위 (부록 G.1) | v_x ∈ [−1.5, 2], v_y ∈ [−1.32, 1.32], ω_z ∈ [−3.35, 3.35] | 기본값 v_x ∈ [−1.5, 1.5], v_y ∈ [−0.5, 0.5], ω_z ∈ [−3.14, 3.14] (`task_spec.py` L185–187). 명령 잡음 (0.5, 0.5, 0.785) 추가 (L190) | **차이 (기본값)**. 논문 실험용 설정이 따로 있을 수 있음 |
| D23 | 속도 인터페이스 샘플 비율 | p(I) 균등, 한 배치에 두 인터페이스 반반 | 기본 `slot_mix={"motion": 1.0, "velocity": 0.0}` (`supermimic_runner.py` L82–85) | 기본값은 모션 전용. 논문 실험에는 다른 인자가 필요 |
| D24 | 속도 창 | 25프레임 중심 평균 | `window_frames=25` (`task_spec.py` L188) | 일치 |
| D25 | 잠재 정렬 손실 | 가중치 0.1, 운동 인코더 잠재에 정지 기울기 | `latent_bc_weight=1e-1` (`supermimic_runner.py` L77) | 일치 (정지 기울기 세부는 확인 안 함) |
| D26 | 평가 프로토콜 (F.1): 129,514 모션, 도메인 무작위화 없음, 마찰 1.0, 결정론적 물리 | 평가 스크립트 | `scripts/benchmark.py`는 처리 속도만 잼. 실패 개수를 세는 평가 스크립트 없음 | **미확인** (저장소에 없음) |
| D27 | 두 정책 비교에서 SONIC | 공개 체크포인트 평가 | 저장소에 SONIC 평가 코드 없음 (SONIC은 NVlabs/GR00T-WholeBodyControl 계열, 이 분석에서 확인하지 않음) | **미확인** |
| D28 | 가중치와 데이터 | 공개 BONES-SEED G1 부분 | 공개 레지스트리에서 모션 자동 다운로드 (`train.py` `registry_name`), 가중치 없음 | 확인 필요 (다운로드 크기 미확인) |

요약: 학습 알고리즘 본체(상상 롤아웃, BPTT, 비평가, 재설정 분포, 보상·종료, 네트워크 크기)는 논문과 대부분 일치합니다. 어긋나는 곳은 (a) 기본 정책 입력(D4), (b) 다중 GPU와 학습률 일정 같은 대규모 학습 장치(D7, D8), (c) 속도 실험 범위(D22), (d) 실패 개수 평가 스크립트(D26)입니다. 2-GPU 실행과 논문의 주 결과 설정은 공개 저장소만으로 재현할 수 없습니다.

## 2. 주장 검증 (C1–C8)

| # | 논문 주장 | 코드에서 확인한 것 | 판정 |
|---|---|---|---|
| C1 | 정책 기울기가 world model 자코비안을 통한 BPTT다 | `_train_batch`가 `wm_env.step`을 H번 호출하고 상태를 끊지 않고 이어 씀. 손실은 `-ret + aux_loss` (`policy_learner.py` L100–126) | 확인 |
| C2 | 생존 가중치로 종료를 미분 가능하게 씀 | `survival = survival * (1 − term_prob)`, term_prob은 시그모이드 (`policy_learner.py` L113–115, `world_model_env.py` L200–219) | 확인 |
| C3 | 정책은 잡음 있는 proprio와 10 관측을 쓴다 | 기본값은 rep_state, 창 1 (D4) | **기본값과 다름** |
| C4 | world model은 결정론적이고 ξ를 입력으로 받음 | 동역학은 `(rep_state, action, aux_state, sim_ctx)`를 받고 샘플링 없음 (`models.py` L86–92). 센서 잡음은 `add_proprio_noise`(`representations.py` L351)로 별도 적용 | 확인 |
| C5 | 모델 기반 학습에서 PPO 대비 기울기 분산이 작다 | 평가 대상 아님. 토이 S로 확인 (`03` §2) | 토이로만 확인 |
| C6 | world model 오차가 학습 중에도 안정적 | 저장소에 오차 측정 코드 없음. Fig 3c는 논문 값만 있음 | 미확인 |

## 3. 저장소 상태 (재현 준비 관점)

- `README.md`는 `mjlab_research`를 형제 디렉터리로 클론하라고 합니다. 두 저장소는 Python 3.13, `uv`에 의존하고, 기본 실행에 `CUDA_VISIBLE_DEVICES`가 필요합니다.
- `tests/`에 단위 테스트 약 20개와 CI 설정이 있습니다(실행하지 않음).
- 라이선스는 Apache-2.0입니다. 이 분석은 코드를 재배포하지 않고 `repo/`는 `.gitignore`에 있습니다.
- Claude 관련 워크플로(`.github/workflows/claude.yml`)가 있습니다. 실행하지 않았습니다.

## 4. 구현 가능성 판정 (이 맥)

**대상 환경:** Apple M4 Pro, 약 24 GiB 통합 메모리, MPS 있음, CUDA 없음. 설치된 것은 torch 2.15와 mujoco 4.0입니다(승인된 범위).

**판정 한 줄: 논문 결과의 재현은 이 맥에서 불가능합니다. 알고리즘의 핵심(world model, BPTT, 비평가)은 PyTorch로 구현할 수 있고, 소형 규모의 검증까지는 가능합니다.**

| 수준 | 판정 | 이유 |
|---|---|---|
| 공식 저장소 실행 (mjlab + MuJoCo Warp 학습 파이프라인) | **불가** | 저장소가 NVIDIA GPU와 `CUDA_VISIBLE_DEVICES`를 전제합니다. MuJoCo Warp의 GPU 백엔드가 CUDA 기반이라는 점은 설치 전 확인이 필요하지만(추정), 이 맥에서 실행 경로가 없다고 봅니다 |
| 논문 주 결과 (1 H100 3 h, 2 GPU 9.8 h, 129,514 모션) | **불가** | 위와 같은 이유에 더해 2-GPU 코드가 없습니다(D7). 데이터 다운로드도 필요합니다(D28) |
| 논문 SONIC 비교 (공개 체크포인트 평가, 실패 개수 재계산) | **불가** | 평가 스크립트가 없고(D26), SONIC 체크포인트를 이 분석에서 받지 않았습니다 |
| 실물 Unitree G1 | **불가** | 하드웨어와 배포 스택이 없습니다 |
| 핵심 메커니즘 재구현: 동역학 모델 + BPTT 상상 롤아웃 + 비평가 (PyTorch, MPS 또는 CPU) | **가능** | 식 1–3과 Algorithm A.1이 PyTorch로 옮길 만큼 명확합니다. 토이 S(numpy)로 BPTT와 분산 비교는 이미 확인했습니다 |
| 소형 G1 MuJoCo CPU 환경에서 짧은 추적 학습 (수 시간 이내) | **조건부** | MuJoCo CPU는 `mujoco 4.0`으로 가능합니다. G1 모델 파일 다운로드가 필요하며(크기는 확인 필요, 1 GB 미만으로 예상(추정)), 학습 속도가 논문과 비교 가능한 규모인지는 실측 필요 |
| 논문 규모 world model 학습 (데이터 264 h, 8스텝 k, 2048 폭) | **조건부** | 동역학 MLP 자체는 MPS에서 가능할 수 있지만, 데이터 264 h의 시뮬레이션 전이 생성이 CPU MuJoCo에서 수 주 걸릴 것으로 추정됩니다 |

**효과 추정(추정치, 근거 없는 추정):**

- 알고리즘 재구현(PyTorch, 동역학·BPTT·비평가·재설정 분포, 토이 규모 검증까지): 600–900줄, 1–2주.
- 소형 G1 MuJoCo CPU 환경 추가: 2–4주. G1 모델 준비와 CPU 시뮬레이션 처리량 측정이 큰 변수입니다.
- 논문 규모 재현: 이 맥에서 계획하지 않습니다. 시뮬레이션 전이 생성과 학습 시간이 수백 시간 이상 될 것으로 보입니다(추정).

**재현 계획은 사용자 확인 후에만 시작합니다.** 설치와 공식 코드 실행은 승인 없이 하지 않습니다. 이 분석은 정적 검토와 토이까지입니다.

## 5. 공개 저장소와 논문의 차이 요약 (질문 목록)

1. 주 결과 설정은 어떤 인자로 실행했는가? (D4, D9, D22, D23)
2. 2-GPU 실행 코드와 학습률 식은 어디에 있는가? (D7, D10)
3. 학습률 코사인 일정은 공개되지 않았는가? (D8)
4. 129,514 모션 평가 스크립트와 SONIC 실행 방법은 공개할 계획인가? (D26, D27)
