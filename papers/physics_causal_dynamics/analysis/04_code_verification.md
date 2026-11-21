# 04. 공식 코드 대조 (논문 ↔ 코드)와 구현 가능성

*공식 저장소 `jungbbal/PICR`, 커밋 `d22568a8` (2026-11-13), `repo/PICR/`에 얕은 복제*. **정적 읽기만 했습니다.** 설치, import, 실행, 학습, 평가를 하지 않았습니다. 체크포인트 `picr.pt`는 로드하지 않았고 로컬 복제에서 지웠습니다. 라이선스는 없습니다(`02` §2). 분석일 2026-11-21.

## 0. 확인한 파일

| 경로 (`repo/PICR/` 기준) | 내용 |
|---|---|
| `configs/picr.yaml` | 학습 설정 전체 (학습 실행에 쓴 값) |
| `src/cdelan/models/context.py` | 문맥 인코더, soft bound, reparameterize, KL |
| `src/cdelan/models/delan.py` | DeLaNCore, FiLMLayer, FiLMDerivativeMLP, 야코비안 |
| `src/cdelan/models/arm_payload_dynamics.py` | 전체 모델, 가속도, 사다리꼴 적분, 인코더 입력, FK |
| `src/cdelan/training/train.py` | 손실 조합, EMA, 체크포인트 저장, 기본값 |
| `src/cdelan/training/context_losses.py` | InfoNCE(식 3), KL, 역동역학 손실 |
| `src/cdelan/eval/pose_metrics.py`, `paper/*/` | ADD 지표, 논문 표 재생성 스크립트 |

## 1. 대조 결과 요약

| 논문 항목 | 코드 위치 | 판정 |
|---|---|---|
| 식 (1), H = AAᵀ + 10⁻⁴I, A_ii = softplus + 10⁻³ (식 13) | `delan.py` `make_l_and_derivatives`, `inertia` (eps 1e-4, diag_eps 1e-3) | 일치 |
| 망 구조: 입력 q만, 은닉 1024, 3층 중 2층 FiLM, 출력 7/21/1 | `FiLMDerivativeMLP` (trunk, film_layers depth 2, out), `picr.yaml` hidden_dim 1024 | 일치 |
| 식 (10)과 (15): a = 1 + ψ(γ)⊙(Wh+b) + ψ(β), ψ = αT tanh(·/T), α = 0.3, T = 3 | `FiLMLayer.modulation`: scale = 1 + limit·T·tanh(γ/T), shift = limit·T·tanh(β/T). `picr.yaml` `delan_film_modulation_limit 0.3`, `delan_film_temperature 3.0`, β 온도 미지정 → γ와 같음 | 일치 |
| FiLM 맵 0 초기화 (부록 A.2) | `FiLMLayer.__init__` (zeros_) | 일치 |
| 야코비안 식 (16–17) | `FiLMDerivativeMLP.forward_with_jacobian` | 일치 (구현은 식과 같은 체인 규칙) |
| 인코더: 입력 (q−q₀, q̇, τ̄_{k−1}) 252차원, MLP 256×2 SiLU, μ는 10 tanh(·/10), log σ² clip [−10, 5] | `ContextEncoder`, `arm_payload_dynamics.encode_context`, `picr.yaml` `history_len 12`, `history_torque_lag 1`, `context_input_arm_only true`, `latent_dim 4` | 일치 (인코더 입력 옵션 `context_residual_features`, `context_input_standardize`, `context_ls_refine`, `context_drop_q`는 기본값 off) |
| 샘플링 식 (8), 추론 z = μ | `reparameterize`, `encode_context` (`deterministic_context` = not training) | 일치 |
| L_q̈ = MSE(H⁻¹(τ − c − g), q̈_FD), 관절별 스케일 | `train.py` `_qdd_label_loss` (`model.acceleration`, FD 목표), `lambda_qdd_label 1.0` | 일치 |
| L_ID = MSE(Hq̈ + c + g, τ), 관절 1–7 | `context_losses.inverse_dynamics_loss` (arm_dof 7, tau_scale), `lambda_inverse_dynamics 0.3` | 일치 (계수 0.3은 논문 본문에 없음, D3) |
| L_con, 식 (3): µ 정규화, κ = 0.1, w_ij = exp(−Δm²/2σ_m² − ‖Δr‖²/2σ_r²), σ_m = 0.271, σ_r = 1 cm, 분모 k≠i | `context_infonce_loss` (`positives="mass_com_gauss"`, `infonce_temperature 0.1`, `infonce_mass_sigma 0.271`, `infonce_com_sigma 0.01`), `lambda_context_infonce 0.1` | 일치 |
| 인접 창 간격 12 프레임 (두 관점) | `infonce_min_view_gap 12`, `make_dataloaders` 두 관점 배치 | 일치 |
| L_KL, 계수 10⁻⁴ | `context_kl_loss`, `lambda_context_kl 0.0001` | 대체로 일치 (KL 계산 시 µ에 bound가 한 번 더 적용됨, D2) |
| 사다리꼴 적분 (q_{k+1} = q_k + Δt/2 (q̇_k + q̇_{k+1})), 가속도는 시작 상태에서 | `_step_arm`, `picr.yaml` `ode_solver trapezoid` | 일치 |
| 입력 자세 q는 기본 Franka 자세에서의 상대값 | `absolute_q` (`default_joint_pos`), `ArmPayloadWMDynamics` | 일치 (데이터 로더를 실행하지 않아 값은 확인 못 함) |
| 학습 318 / 검증 40 에피소드, 8스텝 교사 강제, 배치 256, 40 에피소드, Adam 3×10⁻⁴, 코사인 | `official_train.json` (train 318, val 40), `picr.yaml` (`batch_size 256`, `epochs 40`, `lr 3e-4`, `lr_schedule cosine`, `rollout_horizon 8`) | 일치 (코사인 하한, EMA, 가중치 감쇠 등은 D4) |
| 평가: h = 15 (300 ms) 개루프, ADD 2000점 | `scripts/eval.sh`, `pose_metrics.py` (`n_points 2000`), `paper/fig_payload_sweep/table_add.py` | 일치 |
| 반사실 gain = ⟨Δq̈_pred, Δq̈_sim⟩/‖Δq̈_sim‖², 중앙값, 12개 평균 | `paper/fig_counterfactual/gain_table.py` | 일치 (두 부호 합침, 깨끗한 창 기준 slip ≤ 1 cm) |
| 표 1 RMSE (g₂, [Hq̈]₂, c₂, 관절 2, 3 kg) | `paper/fig_lagrangian/table1_rmse.py`: **한 에피소드(`demo_2130`), 1 s** | 값은 일치 (1.84 / 1.63 / 1.46 = 시드 0). 표 1 본문 값(2.15 / 2.31 / 1.59)과 다름 (D6) |
| 학습 밖 ADD, 표 1 | `table_add.py`: 저장소 시드 0 체크포인트 기대값 4.58 / 6.92 | 표 1(3 시드 평균)과 다름 (D6) |
| 기준선 구현 (CaDM, RWM, CaDeLaC, DeLaN, Neural-Fly) | 저장소에 없음 ("Baseline implementations are not part of this repository", README) | 검증 불가 (D7) |

## 2. 차이 (D1–D10)

**D1. 평가와 체크포인트에 EMA 가중치를 씁니다 (논문에 없음).**
- `train.py` 1463–1470행: `ema_decay > 0`이면 `AveragedModel`을 감쇠 0.999로 만들고 "validation and checkpoints use the averaged weights"라고 출력합니다. 설정 파일이 0.999로 켭니다.
- 1567행: 검증 모델이 `ema.module`입니다. 1583행: 저장되는 `model_state_dict`도 이 모델입니다.
- 논문은 "final weights"를 평가한다고 적었습니다(부록 B.1). 이 문장은 EMA 가중치와 같은 뜻으로 읽을 수 있지만, EMA의 존재와 감쇠율은 논문에 없습니다.
- 영향: 체크포인트와 논문 수치가 EMA에서 나왔다면, 재학습할 때 EMA를 빼면 결과가 달라질 수 있습니다. 크기는 측정하지 못했습니다.

**D2. KL 계산에서 µ에 soft bound가 두 번 적용됩니다.**
- `context.py`: `ContextEncoder.forward`가 `mu = bound_latent(mu_head(h))`로 이미 바운드된 µ를 돌려줍니다. 그런데 `kl_divergence_standard_normal`이 다시 `mu = bound_latent(mu)`를 합니다.
- 논문 식 (6)은 µ를 10 tanh(·/10)으로 정의합니다. 논문 본문 "L_KL은 soft bound s를 적용하기 전에 계산한다"는 식 (8)의 z에 대한 말로 읽힙니다. 그래서 µ에 한 번 더 bound를 거는 것은 논문과 어긋납니다.
- 영향: λ_KL = 10⁻⁴이라서 작을 것으로 추정합니다. 측정하지 않았습니다.

**D3. 손실 계수 중 논문에 없는 값이 있습니다.**
- 저장소 설정: λ_a(`lambda_qdd_label`) = 1.0, λ_ID(`lambda_inverse_dynamics`) = 0.3, λ_con(`lambda_context_infonce`) = 0.1, λ_KL = 10⁻⁴.
- 논문 본문에는 λ_a와 λ_ID 값이 없습니다. 부록 C.2의 λ_con = 0.1과 λ_KL은 일치합니다.
- 롤아웃 손실은 `lambda_rollout 0.0`이라서 **학습에 쓰지 않고 지표로만 씁니다.** 논문도 학습 목적에 롤아웃 항을 넣지 않아 일치합니다.
- `lambda_context_contrastive 0.0`(라벨 RSA 형태의 옛 대조 항)은 꺼져 있습니다. 논문의 식 (3)은 `infonce` 경로입니다.

**D4. 논문에 없는 최적화 설정이 있습니다.**
- Adam의 weight decay 10⁻⁶(`weight_decay`), 코사인 스케줄의 하한이 0.1 × lr(`train.py` 1472행 부근), 기울기 노름 클리핑 10(`train.py` 1209–1210행, 기본값; 설정에는 없음).
- 논문은 "cosine"만 적었습니다. 하한과 클리핑은 적지 않았습니다.

**D5. 체크포인트 선택과 "ID 테스트" 집합.**
- `best_metric: rollout_loss`이라서 검증 롤아웃 손실이 가장 낮은 `best.pt`가 따로 저장됩니다. 공개된 체크포인트는 epoch 40의 `last.pt`입니다(README). 논문은 "final epoch"을 씁니다. 이 점은 맞습니다.
- `official_train.json`에서 `id_test`와 `val`은 같은 40 에피소드입니다(`demo_101`, `demo_105`, …). 논문 본문에서 ID 평가 집합과 검증 집합이 분리되어 있다고 읽히지만, 실제로는 같은 집합입니다. 표 1의 "seen" 수치는 `ramp8_up` 창에서 나오므로 학습과 검증의 누수는 아닙니다. 하지만 "ID 테스트"라는 말은 정확하지 않습니다.

**D6. 표 1 수치와 저장소의 기대값이 다릅니다 (가장 중요).**
- 논문 표 1은 "mean over three seeds"입니다. 저장소 `paper/fig_payload_sweep/README.md`는 "Expected: seen 4.58, unseen 6.92"라고 적었고, `paper/fig_counterfactual/README.md`는 gain 0.86을, `paper/fig_lagrangian/README.md`는 RMSE 1.84 / 1.63 / 1.46을 적었습니다. README 최상단은 이 값을 "PICR column of Table 1 from the released checkpoint"라고 합니다.
- 즉 저장소가 재현하는 표 1은 **시드 0 값**이고, 논문 표 1은 **3 시드 평균**입니다. 논문 표 3(시드 0)과 표 6(시드 0)은 저장소와 일치합니다.
- RMSE도 저장소는 **한 에피소드 1 s**에서 계산합니다. 논문은 "a fixed-payload carry with a 3 kg payload, first second"라고 적었습니다(부록 C.1). 에피소드 수는 적지 않았습니다.
- 공개된 체크포인트는 시드 0 하나뿐입니다. 3 시드 평균은 이 저장소로 재현할 수 없습니다.

**D7. 기준선 구현이 공개되지 않았습니다.** 저장소 README가 "Baseline implementations are not part of this repository"라고 씁니다. 표 1, 4, 6의 기준선 수치와 부록 B의 구성(크기, 옵티마이저, 학습 예산)을 검증할 수 없습니다. CaDeLaC의 "원시 토크로 전체 시스템 학습" 변형(부록 B.4)도 코드로 확인할 수 없습니다.

**D8. 큐브 자세와 ADD는 Franka 정기구학에 의존합니다.** `ArmPayloadWMDynamics`는 `FrankaForwardKinematics(tool_z_offset=0.1034)`와 상수 `DEFAULT_FRANKA_JOINT_POS`를 씁니다. 동역학 망은 이 정보를 쓰지 않습니다(`acceleration`은 `core(absolute_q(q), …)`만 호출). 그러나 `assemble_state`와 `infer_grasp`는 FK로 큐브 자세를 만들고 ADD가 이를 씁니다. 초록의 "without a hand-specified robot model"은 동역학 예측에 한정해서 맞습니다.

**D9. 주석과 명칭의 불일치 (수치 영향 없음).**
- `ArmPayloadWMDynamicsConfig`의 기본값은 `robot_dof=9`, `torque_dim=9`, `hidden_dim=256`, `ode_solver="euler"`이고, 주석은 "9 coordinates"라고 적었습니다. 실제 PICR 설정은 `robot_dof 7`, `hidden_dim 1024`, `ode_solver trapezoid`입니다.
- `FiLMLayer`의 주석 "T 게이트"와 `context_losses.py`의 "SupCon" 설명은 논문 식 (3)과 (15)를 가리킵니다. 논문 본문에는 이 이름과 출처 인용이 없습니다.

**D10. 조인트 속도 클램프는 기본 꺼짐입니다.** `CDELAN_DQ_CLIP` 환경 변수가 있으면 롤아웃에서 관절 속도를 클램프합니다. `env.sh`, `scripts/*.sh`, `paper/*/run.sh`에는 설정이 없습니다. 논문은 이 클램프를 언급하지 않았습니다. 공개 스크립트에서는 꺼져 있으므로 일치합니다. 하지만 논문 실험을 그 환경 변수로 돌렸는지는 확인할 수 없습니다.

## 3. 데이터, 가중치, 재현 가능성

- **데이터셋은 공개되지 않았습니다.** 프로젝트 페이지는 "The simulation datasets are not publicly available yet", `data/README.md`는 데이터셋 ID(`official_train`, `official_ood_2to4kg`, `ramp8_up`, `official_intervene_jlm7000_h13`, …)와 경로만 적었습니다. 계약 JSON(`data/contracts/*.json`)은 에피소드 목록입니다.
- **가중치**: `checkpoints/picr.pt`(seed 0, epoch 40, EMA). SHA-256 `c9a710751de002061961bc9bdb3d2b5a044a6a8d3de36d91677789328dbc4110`. 로드하지 않았고 로컬 복제에서 삭제했습니다.
- **학습 시간**: README는 40 에피소드에 RTX 4090 한 장으로 약 5.5 시간, 세 실행을 나눠 썼다고 합니다.
- **환경**: `requirements.txt`는 Python 3.8, torch 2.4.1, numpy 1.24.4, h5py 3.11.0, scipy 1.10.1, matplotlib 3.7.5 등입니다. "CUDA 12.1, RTX 4090에서 시험"이라고 적었습니다.
- **Isaac 의존**: `paper/*/render/` 스크립트는 Isaac Lab 2.3.2 / Isaac Sim 5.1.0이 필요합니다. 데이터 생성 파이프라인(`src/cdelan/data`, HDF5)도 Isaac 로그를 전제합니다.

## 4. 구현 가능성 판정

**판정: 전체 재현은 이 맥에서 불가능합니다. 구조 재구현은 가능합니다. 판정 기준은 이 맥(Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)입니다.**

| 작업 | 필요 자원 | 이 맥 | 판정 |
|---|---|---|---|
| 논문 표 1 재현 (3 시드) | 공개되지 않은 데이터 + 3회 학습 + 기준선 코드 | 데이터 생성에 NVIDIA GPU 필요, 기준선 코드 없음 | **불가** |
| 공식 코드 학습 (시드 0) | 데이터 (비공개) + RTX급 GPU 5.5 시간 | CUDA 없음. 데이터도 없음 | **불가** |
| 공식 체크포인트로 추론 (저장소 데이터 없음) | 설치, 공식 코드 실행 (사용자 승인 필요) | README가 CUDA GPU(RTX 4090)를 전제합니다. MPS나 CPU로 옮길 수 있는지는 확인하지 않았습니다 | **조건부**: 승인과 설치 계획 뒤에 가능 (이 분석에서는 실행하지 않음) |
| 시뮬레이터로 데이터 생성 (Isaac Sim) | RTX GPU, Isaac Sim 5.1 | 불가 | **불가** |
| 실로봇 | Franka와 제어 인터페이스 | 불가 | **불가** |
| 구조 재구현: FiLM-Lagrangian, 문맥 인코더, InfoNCE, L_q̈, L_ID (PyTorch) | PyTorch, 자체 시뮬 데이터 (예: MuJoCo 등) | 6.5M 파라미터, MPS 또는 CPU에서 학습 가능한 크기 | **시작 가능** (노력 추정 1–2주, 측정 전) |
| 표 재계산 (3 시드 평균과 검정) | 논문 표 값 | CPU, 수 초 | **완료** (`recompute.py`) |
| 토이 실험 (2-링크 등으로 FiLM과 입력 연결 비교) | numpy 또는 PyTorch, CPU | 가능 | **이 분석에서는 하지 않음** (이유: `03` §0) |

- 노력 추정은 측정하지 않았습니다. 구조 재구현의 가장 큰 위험은 데이터 생성 쪽입니다. 질량과 CoM을 바꾼 정책 궤적과 토크 교란을 만들어야 합니다. 논문은 PPO 정책과 PID 제어 설정을 부록 D에 적었지만 코드가 없습니다.
- 따라서 "이 논문을 맥에서 재현할 수 있는가?"에 대한 답은 "표는 재현할 수 없고, 방법은 구현할 수 있다"입니다.

## 5. 라이선스와 공개 범위

- **코드**: GitHub API가 `licenseInfo: null`이라고 보고합니다. LICENSE 파일이 없습니다. 그래서 복제본은 gitignore하고 공개본에는 코드를 넣지 않습니다. 공개본에는 우리가 쓴 스크립트와 분석만 넣습니다.
- **데이터**: 공개되지 않았습니다.
- **가중치**: 라이선스가 없습니다. 공개본에 넣지 않습니다.
- **논문 그림**: arXiv 라이선스가 CC BY가 아니므로 공개본에 넣지 않습니다. 이 분석의 그래프는 논문 표 값을 우리가 다시 그린 것입니다.
