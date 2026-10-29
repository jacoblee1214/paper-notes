# 04. 코드 검증: 논문 ↔ 구현 대조, 수치 재계산, 구현 가능성

- **대상:** `AIGeeksGroup/DexPolicy` @ `98bf668` (2026-10-04, MIT). Hugging Face `AIGeeksGroup/DexPolicy` @ `575645a`는 메타데이터 JSON 6개와 model card만 읽었고, 가중치는 받지 않았습니다. 기반 코드 `zerchen/vividex_mujoco` @ `9790140`은 PPO 설정, 학습 스크립트, 보상·성공 판정 파일만 받았습니다.
- **범위:** GPU 실행과 패키지 설치 없이 코드 읽기, 설정 대조, CPU 재계산과 toy만 했습니다. 공식 저장소는 README에서 "method-core release"라고 밝힙니다. 환경, 메쉬, 궤적, 로봇 드라이버, 실제 로봇 데이터는 없습니다.

## 1. 모듈 매핑

| 논문 | 구현 위치 | 일치 여부 |
|---|---|---|
| 기준선 PPO: 학습되는 log σ, 초기 0.20 (Sec. I, IV-A) | `config/agent/ppo.yaml`: `log_std_init: -1.60`(σ = 0.2019), SB3 차원별 `log_std` | ✅ "nominal 0.20" |
| PPO 고정 설정: lr 1e-5, clip 0.20, 5 epoch, grad norm 0.50 | `ppo.yaml`, preset 공통값 | ✅ ViViDex 원래 설정과도 같음 |
| 선택된 스케줄 Eq. 4 | `phase_schedule_core.py: NO_MID_JUMP_5M` (0.20→0.10, 0.10→0.05, 0.04) | ✅ 정확히 일치 |
| log σ 얼림, 스칼라를 모든 차원에 공유 (Eq. 1) | `PhaseScheduleCallback`: `requires_grad_(False)`, `log_std.data.fill_(log σ)` | ✅ |
| σ를 rollout 수집과 update 내내 고정 (Eq. 2) | `_on_rollout_start`에서만 적용 | ✅ |
| 초기 screen 곡선(2M에서 0.10→0.08)과 최적화기 수축 (Table IV) | `SCREEN_5M`: 2–4M lr 3e-6·clip 0.10·3 epoch, 4M 이후 lr 1e-6·clip 0.05·2 epoch·grad 0.25 | ✅ |
| 모양 후보 (Table V) | `LINEAR_5M`(0.20→0.05), `FRONTLOADED_5M`, `NO_FINAL_DROP_5M`, `NO_MID_JUMP_5M` | ✅ |
| GRPO: critic 없음, value loss·KL 없음, 엔트로피 0.001, 8,192 rollout, batch 256, 5 epoch, clip 0.2, grad 0.5 | `algos/grpo.py: TrajectoryGRPO.train`, `grpo.yaml` | ✅ 다만 advantage는 D1 |
| GRPO advantage (Eq. 5) | `trajectory_group_advantages` | ❌ D1 |
| FPO: 0에서 시작하는 8단계 결정론적 적분 평균 + Gaussian likelihood + PPO clip | `fpo_core.py: flow_act(deterministic=True)` → `x_t = zeros`, `sampling_steps: 8`, `gaussian_ppo_surrogate_loss` | ✅ README대로 덮어쓸 때(D4) |
| FPO σ: 기준선 0.10 고정, DexPolicy 0.20 → 0.05 선형 | `_scaled_action_std`: 0.20 × [0.25 + 0.75(1 − s/5M)] | ✅ |
| 체크포인트 설정 | HF JSON: PPO·FPO 5M step, lr 1e-5, clip 0.20, 5 epoch, 스케줄 문자열 | ✅ PPO·FPO는 논문과 같음. GRPO는 D2 |

## 2. 불일치와 논문에 적히지 않은 구현 요소

| # | 내용 | 영향 |
|---|---|---|
| D1 | **공개 GRPO advantage가 Eq. 5가 아닙니다.** `trajectory_group_advantages`는 완료 궤적마다 `rewards[start:end].sum()`(할인 없음)을 구해 rollout의 모든 완료 궤적끼리 표준화하고, 그 값을 궤적의 모든 step에 넣습니다. 나이별 그룹도, γ = 0.95도 없습니다. README도 "returns across complete episodes"라고 적어 코드와는 일치합니다. 반면 HF 메타데이터의 GRPO 두 모델은 `"credit_module": "grpo_rtg"`라서 실험에는 reward-to-go 모듈을 쓴 것으로 보입니다. | 공개 코드로는 Table I·II의 GRPO 결과를 재현할 수 없습니다. toy에서 두 advantage의 상관은 0.73이고 부호 불일치는 7.6%였습니다(`03` §3). |
| D2 | **GRPO 연속 학습 스케줄이 preset에 없습니다.** 논문은 σ를 0.20으로 다시 키운 뒤 0.20 → 0.10 → 0.03으로 줄인다고 합니다(토마토는 세 번째 단계 0.10 → 0.02). preset의 GRPO 항목(`grpo_hold_010_5m`, `grpo_reexpand_018_5m`, `grpo_contract_release_5m`, `grpo_early_release_5m`)은 모두 0–5M 절대 시계의 다른 곡선입니다. README는 "GRPO plateau experiment"에 `grpo_hold_010_5m`(0.10 유지)를 쓰라고 하는데, 이 실험은 논문에 없습니다. HF 메타데이터의 GRPO 모델은 `source_steps: 507904`, `additional_steps: 250000`, lr 5e-5, σ 0.10 → 0.03입니다. 논문의 "약 5M 초기화 + 0.762M step"과 표현이 다르고, 253,952가 아니라 250,000으로 적혀 있습니다. | 연속 학습의 시작점(source)과 단계 구성을 공개 자료만으로는 확정할 수 없습니다. |
| D3 | **최종 정책 선형 대조(0.20 → 0.04, Sec. V-E)의 preset이 없습니다.** `LINEAR_5M`은 0.20 → 0.05(Table V의 uniform linear)입니다. | 사소하지만 "같은 끝점의 선형 감쇠"를 재현하려면 직접 추가해야 합니다. |
| D4 | **FPO 설정 파일의 기본값이 논문 실험과 다릅니다.** `fpo.yaml` 기본값은 `actor_objective: fpo`(flow matching ratio), `gaussian_action_std: 0.05` 고정, `max_grad_norm: 1.0`, `save_best_model: True`입니다. README대로 `actor_objective=gaussian_ppo`, `trust_region_mode=ppo`, `action_head_mode=flow_residual`과 σ 0.20·감쇠 5M·`min_scale 0.25`를 덮어써야 논문 설정이 됩니다. | `save_best_model`이 켜져 있으면, 최종 체크포인트가 아니라 평가 최고 체크포인트를 쓸 위험이 있습니다. 논문은 "final" 체크포인트를 평가한다고 적었지만, 어느 파일을 평가했는지는 확인할 수 없습니다. |
| D5 | **논문에 보고되지 않은 변형이 코드에 많습니다.** A2C preset 2개, TRPO용 기울기 0 훅, `std_centered`(학습을 유지하면서 평균을 목표 σ로 옮김), 진행도 트리거(사전 파지 성공 95% 이후 시계 시작), GRPO 변형 4개, `selected_10m`, `full_20m`(20M, 다단계 최적화기 수축). | README도 "research ablations"라고 합니다. 여러 갈래를 시험한 뒤 일부만 보고했을 수 있으므로, 개별 비교의 p값을 더 보수적으로 읽어야 합니다. |
| D6 | **Target·Lift 정의가 공개 환경과 다릅니다.** ViViDex 공개 코드의 목표 성공은 `success_goal`: 최종 물체 위치 오차 ≤ `thresh × 5` = 0.01 × 5 = **5 cm**이고, 접촉을 함께 요구합니다(`rewards.py`). 논문 Target은 3 cm입니다. ViViDex에는 높이 5 cm Lift 판정이 없고, 들어 올리기 보너스 기준은 바닥 + 2 cm입니다. | DexPolicy의 평가 코드가 공개되지 않아 3 cm 판정을 확인할 수 없습니다. 공개 환경으로 재현하면 성공률 숫자가 달라집니다. |
| D7 | **레퍼런스 시퀀스 선택이 없습니다.** ViViDex 자산(`zerchen/object_sim`)에는 물체마다 DexYCB 시퀀스가 여럿 있습니다(논문의 8개 물체는 각 18–20개, 예: 머스터드 18개, 바나나 20개). 논문은 "물체마다 레퍼런스 하나"라고만 하고, HF 메타데이터도 물체 이름(`ycb-006_mustard_bottle`)만 적었습니다. | 같은 물체라도 시퀀스에 따라 난이도가 다를 수 있어, 물체별 수치를 그대로 재현하기 어렵습니다. |
| D8 | **기준선 PPO 학습 예산이 ViViDex 원래 설정의 1/20입니다.** ViViDex `config.yaml`의 기본은 `total_timesteps: 100000000`, `n_envs: 32`, 평가 주기 1M step입니다. DexPolicy는 5M step, 평가 250k step마다입니다. ViViDex PPO 설정의 `warm_start_mean: True`(평균을 초기 자세 근처로 시작)는 DexPolicy `ppo.yaml`에 없습니다. 환경 쪽에서 처리하는지는 확인할 수 없습니다. | 5M step에서는 학습된 log σ가 최대 약 0.98만 움직일 수 있어 0.04에 도달할 수 없습니다(`03` §2). 논문의 PPO 비교는 짧은 예산 조건의 결과로 읽어야 합니다. |
| D9 | **엔트로피 보너스가 기준선 σ를 키웁니다.** PPO·GRPO 모두 `ent_coef: 0.001`입니다. 얼린 σ에서는 상수라 영향이 없지만, 학습 σ 쪽에서는 σ를 키우는 방향의 기울기를 더합니다. | 기준선 σ가 거의 줄지 않은 원인 중 하나일 수 있습니다. 논문의 PPO 고정 설정 목록에는 엔트로피 계수가 없습니다. |
| D10 | **연속 학습에서 스케줄 시계가 모호합니다.** callback은 `self.num_timesteps`를 시계로 씁니다. 약 5M step 체크포인트를 불러 계속 학습할 때 `reset_num_timesteps`를 어떻게 했는지에 따라, 0–5M preset이 처음 단계로 시작하거나 마지막 단계(0.04)에 머뭅니다. 연속 학습 스크립트는 공개되지 않았습니다. | D2와 함께, GRPO 결과를 재현하려면 공개되지 않은 학습 스크립트가 필요합니다. |
| D11 | **공개 체크포인트는 6개뿐입니다.** 머스터드 seed 0의 PPO·GRPO·FPO × 기준선·DexPolicy입니다. 논문 평가 모델은 PPO·FPO 96개, GRPO 30개, 하드웨어 18개입니다. 하드웨어 모델과 하드웨어용 시뮬레이터(RM75 + RH56)는 없습니다. | seed 0 머스터드는 스케줄을 고른 바로 그 조건이라, 대표 예시로는 가장 유리한 쌍일 수 있습니다. |

## 3. 수치 재계산 (`analysis/table_check.py`)

| # | 주장 (출처) | 재계산 | 판정 |
|---|---|---|---|
| C1 | Overall = 다섯 물체 평균 (Table I) | 6행 모두 일치(예: FPO+DexPolicy 68.08 → 68.1). 8개 물체 평균은 나머지 세 물체 ≈ 0일 때 PPO 20.0 / 22.3, FPO 30.86 / 42.55로 본문과 맞음 | ✅ |
| C2 | 본문 seed별 값 (Sec. V-B, V-C) | FPO 머스터드 51.67 ± 45.96, 토마토 62.33 ± 54.0, 머그 25.33 ± 34.79 → 31.0 ± 53.69, GRPO 머스터드 30.83 ± 53.4 → 34.5 ± 56.7, seed 1/2 46.25 → 1.75. 모두 Table I과 일치 | ✅ |
| C2b | FPO Overall SD가 공개 seed 값과 양립하는가 | 바나나·설탕을 seed마다 평균으로 두면 기준선 SD 19.2(보고 21.5), DexPolicy 11.2(보고 11.4) | ✅ 양립 |
| C3 | 본문 평균 차이 = Table II 물체별 차이의 평균 | seed 1/2: 21.9 = 37.2 − 15.3. σ 0.03: 26.98 ≈ 27.0. σ 0.10: −7.4 = 8.7 − 16.1 | ✅ |
| C4 | 하드웨어 평균 (Sec. V-D) | Target 8.3 → 43.3, 10.0 → 63.3, 25.0 → 85.0. Lift 38.3 → 58.3, 46.7 → 95.0, 45.0 → 93.3. 모든 행에서 Target ≤ Lift | ✅ |
| C4b | 하드웨어 차이가 시행 잡음보다 큰가 | 합산 Fisher p: PPO 1.7×10⁻⁵, GRPO 1.1×10⁻⁹, FPO 3.1×10⁻¹¹. 물체별로는 PPO 설탕(0/20 vs 0/20)과 GRPO 머스터드(p = 0.08)가 구분되지 않음 | ✅ 시행 단위로만 |
| C5 | "세 설정 모두에서 오른다" (기여 1) | PPO는 물체 2개 증가, 2개 감소, 1개 같음, Welch p ≈ 0.39. GRPO p ≈ 0.04, FPO p ≈ 0.27(비대응 근사) | ⚠️ 평균만 |
| C6 | GRPO 연속 학습 step 수 | 507,904 = 62 × 8,192, 253,952 = 31 × 8,192, 합 761,856 ≈ 0.762M, 토마토 1,015,808 ≈ 1.016M | ✅ |
| C6b | 학습된 σ가 0.04에 갈 수 있는가 | Adam step 97,656 × lr 1e-5 ≈ 0.98 < 필요 1.62 | ❌ 원리상 불가 (`03` §2) |
| C7 | Table V 통과 판정, Table VII 상대 변화, 10M 재평가 | 판정 4개 모두 일치. Table VII 5개 중 2개가 0.1%p 반올림 차이(25.1 vs 25.2, 11.8 vs 11.7). 재평가 +11.2%, 높이 +2.7%(보고 +2.8%) | ✅ (반올림 수준) |
| C8 | 하드웨어 "조건당 모델 1개"의 의미 | 시뮬레이션 seed 값으로 예시: FPO 머그에서 모델을 하나씩 뽑으면 DexPolicy가 높을 확률 33%, 낮을 확률 44%. GRPO 머스터드는 높을 확률 78% | 참고 |

## 4. 구현 가능성 판단

### 4.1 공식 코드를 그대로 쓰는 경우

| 구성 | 공개 여부 | 메모 |
|---|---|---|
| σ 스케줄 preset과 SB3 callback | ✅ | 약 600줄. 어떤 SB3 PPO에도 붙일 수 있습니다 |
| GRPO 업데이트 | △ | 공개된 advantage는 논문 Eq. 5가 아닙니다(D1) |
| flow 평균 trainer | ✅ | trainer·core·buffer 합쳐 4,400줄 남짓. README대로 덮어써야 논문 설정이 됩니다(D4) |
| 시뮬레이션 환경·보상·평가 | ❌ | ViViDex 공개 환경으로 대체할 수 있지만 Target 정의가 다릅니다(D6) |
| 레퍼런스 시퀀스 선택 | ❌ | D7 |
| 하드웨어 시뮬레이터(RM75 + RH56), 드라이버, FoundationPose 파이프라인 | ❌ | 실제 로봇 결과는 재현할 수 없습니다 |
| 체크포인트 | △ | 머스터드 seed 0 6개 |

환경 요구(README): Python 3.10, PyTorch 2.4.1, NumPy 1.24 미만, Gym 0.25.2(legacy API), Stable-Baselines3 1.1.0입니다. 이 버전 조합은 패키지 메타데이터상 서로 맞지 않는데, README는 "기존 호환 환경을 쓰고 SB3를 올리지 말라"고 경고합니다. ViViDex 쪽은 mujoco 2.3.7, dm-control 1.0.14, open3d, chumpy, nlopt 등이 더 필요합니다.

**판단:** 방법 핵심은 공개되어 있지만, 논문 표를 그대로 재현할 수 있는 패키지는 아닙니다. 가장 가까운 경로는 **ViViDex 공개 환경 + DexPolicy callback**으로 PPO 시뮬레이션 비교를 다시 만드는 것입니다. 이때 Target은 ViViDex 정의(5 cm)로 바뀌고, 레퍼런스 시퀀스는 직접 골라야 합니다.

### 4.2 이 맥(Apple M4 Pro, CUDA 없음)에서

| 단계 | 가능 여부 | 근거 |
|---|---|---|
| 표 재계산, 보상·advantage toy | ✅ 완료 | numpy, 각 20초 이내 |
| ViViDex + DexPolicy PPO(머스터드, 5M step) | △ 조건부 | MuJoCo는 CPU로 돌고, SB3 PPO의 작은 MLP([256, 128])도 CPU로 충분합니다. ViViDex README의 `pytorch-cuda` 설치는 visual policy용이라 state policy에는 필요 없어 보입니다. 다만 위의 오래된 버전 조합, open3d·chumpy 등의 macOS arm64 설치는 확인하지 않았습니다. 실행 시간도 측정하지 않았습니다 |
| HF 체크포인트 6개 평가 | △ 조건부 | 체크포인트 6개는 합쳐 약 20 MB입니다. 환경이 서야 의미가 있고, Target 정의 차이(D6)가 남습니다 |
| GRPO 연속 학습 재현 | ❌ | `grpo_rtg` 모듈과 연속 학습 스크립트가 없습니다(D1, D2, D10) |
| 하드웨어 | ❌ | 장비와 하드웨어 모델이 없습니다 |

### 4.3 아이디어만 다시 구현하는 경우

| 구성 | 난이도 | 메모 |
|---|---|---|
| σ 스케줄(log σ 얼림 + rollout마다 덮어쓰기) | 매우 쉬움 | 수십 줄. 어떤 PPO 구현에도 붙습니다 |
| Eq. 5 GRPO advantage | 쉬움 | 나이별 그룹 표준화. `grpo_credit_toy.py`의 `eq5_advantages`가 numpy 구현입니다 |
| 결정론적 평가 프로토콜과 공통 잡음 평가 | 쉬움 | 평가 루프에 σ_eval 인자를 추가하면 됩니다 |
| 공정한 기준선 | 중간 | 저자가 빠뜨린 "잘 조정한 고정 저잡음 PPO", 엔트로피 계수 스케줄, 충분한 예산의 학습 σ를 함께 둬야 결론을 낼 수 있습니다 |

**요약:** 아이디어 구현은 쉽습니다. 다만 "스케줄이 학습 σ보다 낫다"는 결론을 확인하려면, 논문에 없는 대조군(고정 저잡음, 더 긴 예산)이 필요합니다. 이 맥에서 할 수 있는 가장 현실적인 다음 단계는 ViViDex 환경 설치가 되는지 확인하는 것입니다(설치 승인 필요).
