# 04. 코드 검증: 논문 ↔ 구현 대조, 수치 재계산, 구현 가능성

- **대상:** `nvidia-isaac/video_to_data` @ `9cad120da522d4086b47052207d7a79ae06a6893` (2026-11-02)의 `robotic_grounding/`. 미디어(mp4 등)를 뺀 sparse clone이고 `repo/`에 있습니다(git 제외). 두 학습 경로를 모두 읽었습니다.
  - **PPO 경로:** Isaac Lab + rsl_rl. `source/robotic_grounding/robotic_grounding/tasks/v2d/` (이하 `v2d/`), `scripts/rsl_rl/train.py --task Sharpa-V2D-v0`
  - **FlashCHORD 경로:** Newton + Warp + JAX, PPO와 FlashSAC. `flash_chord/src/flash_chord/` (이하 `fc/`)
- **범위:** GPU 실행 없이 코드 읽기, 설정 대조, CPU 재계산만 했습니다. 공개 저장소에는 1,831개 결과의 rollout·학습 로그·체크포인트가 없어서 Fig. 5나 Table 1을 결과 파일로 재계산할 수는 없습니다.

## 1. 모듈 매핑

| 논문 | 구현 위치 | 일치 여부 |
|---|---|---|
| 마찰원뿔 다면체, 단위 모서리 d개 (Eq. 1) | `v2d/mdp/utils.py: compute_friction_cone_edges` (n + μ(cos θ t₁ + sin θ t₂) 정규화, 법선 추가) | ✅ 모서리 8개 + 법선 1개 = 9개, μ = 0.1 (D4) |
| primitive wrench [f, p × f] (Eq. 1) | `utils.py: compute_wrench_space` | ✅ 토크를 경계 구 반지름 r_c로 나눔. 논문 식에는 r_c 없음 |
| 미리 뽑은 단위 방향 b개 | `utils.py: sample_wrench_space_basis_scaled(512, rc=1.0)` (`hand_object_commands.py:751–752`), `fc/configs/objective/tracking.yaml: wrench_basis_seed 0` | ✅ 6차원 가우시안 정규화 = 단위 구 균일. body마다 다른 기저 |
| 지지 함수 σ = max_col(ℬᵀ𝒲) (Eq. 2) | `utils.py: compute_wrench_space_support_function` | ✅ 0 아래는 잘라냄 |
| 사람 쪽 σ_h | `hand_object_commands.py: _precompute_contact_wrench_support_values` (리타겟 데이터의 접촉 위치·링크 법선, 프레임별 미리 계산) | ✅ |
| 로봇 쪽 σ_r | `utils_jit.py: wrench_preprocess_jit` → `wrench_support_one_body_jit` (시뮬레이터 접촉력 방향을 법선으로) | ✅ 힘 크기는 정규화로 사라짐 (D7) |
| 접촉 wrench-space 보상 (Eq. 3), β, v | `utils_jit.py: contact_wrench_support_reward_jit`, `tolerance 0.1`, `var 0.1`, 가중치 10 (`v2d_hand_env_cfg.py:251–258`) | ⚠️ 식이 다름 (D1) |
| r_unintend, r_miss | `unintended_contact_penalty_jit`(가중치 −10), `missed_contact_penalty_jit`(−1) | ✅ unintended는 body 단위 (D10) |
| r_fc (Sec. 3.2) | `v2d/mdp/rewards.py: force_closure_reward(min_support=0.01)`, 가중치 0 | ⚠️ 손별 계산, 기본 꺼짐 (D8) |
| r_task (물체 자세 exp) | `object_keypoints_tracking_exp` (var 0.1) | ✅ SE(3) ⊖ 대신 keypoint 거리 |
| r_relative (두 물체 상대 자세, App. B) | `rewards.py: relative_object_pose_reward` 등 정의만 있음 | ❌ 기본 보상 설정 어디에도 등록되지 않음 (D11) |
| r_imit | `hand_keypoints_tracking_exp`(0.25), `hand_joint_pos_tracking_exp`(0.25) | ✅ keypoint에 포화 임계값 있음 |
| VOC 커리큘럼 | `v2d/mdp/curriculum.py: FixedTimestepCurriculum`, `fc/lifecycle/curriculum.py: FixedCurriculum` | ✅ 같은 `bisect_right` 의미, FlashCHORD는 10배 빠름 (C7) |
| 리셋: 임의 프레임, 손가락 [0, 0.7] 보간, VOC 20스텝 | `v2d_hand_env_cfg.py:63–67` (`reset_finger_openness 0.7`, `virtual_object_control_decay_steps 20`) | ✅ |
| 행동 배율 0.05 m, 0.15 rad, 0.15 rad, EMA 0.3 | `v2d/mdp/actions/actions_cfg.py:39–57` | ✅ |
| 제어 20 Hz | `decimation = 5`, `sim.dt = 0.01` | ✅ |
| FlashSAC, 환경 2,048개, L40S 약 2시간 | `fc/configs/experiment/sharpa_flash_sac.yaml` (`world_count 4096`, 2.5억 전이) | ⚠️ 환경 수 다름 (D6), 시간은 확인 불가 |
| 성공: 15 cm / 40° 이탈 없이 끝까지, 완료율 > 0.7 | `fc/evaluation/metrics.py: chord_success`, `evaluation/harness.py: evaluation_thresholds` | ⚠️ 임계값 다름, 0.7 규칙 없음 (D3) |
| 전신 확장 | `v2d_whole_body/`, `fc/configs/experiment/g1_recon_body_*.yaml` | 읽기만 함(세부 대조 안 함) |
| GR00T 후속 학습 | `groot_finetune/` (전문가 rollout → LeRobot → GR00T) | 논문에 없는 부분 |

## 2. 불일치와 논문에 적히지 않은 구현 요소

| # | 내용 | 영향 |
|---|---|---|
| D1 | **보상 식이 Eq. 3과 다릅니다.** 코드는 방향별 loss_b를 따로 exp한 뒤, 사람·로봇 지지값이 모두 10⁻³ 초과인 칸만 더해 사람 활성 방향 수로 나눕니다(`utils_jit.py:472–526`). 그 뒤 body 평균, 손 평균입니다. Eq. 3은 512방향 위반 합의 exp입니다. | toy에서 Eq. 3은 법선 10°에서 0.004, 코드는 0.95입니다(`03` §2). 논문 식대로 재구현하면 보상이 거의 0이라 학습이 안 될 수 있습니다. |
| D2 | **공개 순서와 논문 주장의 시점이 어긋납니다.** 첫 학습 코드 v0.2.0(`cbee067`, 2026-08-14)은 PPO 경로만 있습니다. FlashCHORD는 v0.4.0(`33129dd`, 2026-09-23)에 처음 들어왔습니다. 논문 v2(2026-08-14)는 FlashSAC으로 2시간이라고 적었습니다. 82.12%·90.77%와 그림 파일은 v1과 같습니다. FlashCHORD의 `reference_curriculum()`은 "W&B 실행 `sizirwre`에서 복사"이고 값이 PPO 설정과 같습니다. | FDR이 PPO 경로로 CHORD를 돌린 것은 당시 공개 상태와 맞습니다(`02` §3). 1,831개 결과가 어느 경로에서 나왔는지는 밝혀져 있지 않습니다. |
| D3 | **종료·성공 임계값이 세 가지입니다.** 논문: 물체 15 cm / 40°. PPO 경로: 손목 0.2 m, 물체 0.2 m / 0.7 rad(40.1°) (`v2d_hand_env_cfg.py:303–318`). FlashCHORD: 손목 0.15 m, 물체 0.10 m / 0.50 rad(28.6°) (`fc/configs/termination/tracking.yaml`). FlashCHORD 평가는 체크포인트 설정의 종료 임계값을 그대로 성공 판정에 씁니다(`harness.py: evaluation_thresholds`). 완료율 > 0.7을 과제 성공으로 세는 코드는 찾지 못했습니다. | 논문의 40°는 PPO의 0.7 rad, 15 cm는 FlashCHORD의 손목 값과 숫자가 맞습니다. GNR이 "CHORD 관례"로 쓴 HOT3D 150 mm는 논문 숫자입니다. 코드로 같은 판정을 재현하려면 임계값을 직접 맞춰야 합니다. |
| D4 | **μ = 0.1, 모서리 8 + 법선 1, 방향 512개, 토크/r_c는 논문에 없습니다**(`commands_cfg.py:107–113`, `fc/configs/objective/tracking.yaml:62–65`). μ = 0.1은 원뿔 반각 5.7°입니다. | 재구현에 꼭 필요한 값입니다. μ에 대한 보상 민감도는 작습니다(`03` §2.5). |
| D5 | **추적 가중치 일정이 두 경로에서 다릅니다.** PPO: 물체 keypoint 가중치 0 → 0.1 → 0.25 → … → 1 → 20(마지막 단계), 손 keypoint·관절 0.25. FlashCHORD: 물체 keypoint 20을 처음부터. 논문에는 가중치가 없습니다. | PPO 경로는 학습 초반에 물체 추적 보상이 없고 wrench 보상(가중치 10)만으로 접촉을 배웁니다. 논문의 "VOC를 줄이기 전에 wrench 보상이 먼저 수렴해야 한다"(Sec. 4.1)는 이 일정과 맞습니다. |
| D6 | **환경 수:** 논문 App. B는 2,048개, Table 2 캡션은 "시드당 4,096개", 코드는 두 경로 모두 4,096개(`v2d_hand_env_cfg.py:386`, `sharpa_flash_sac.yaml:26`). | 2시간 주장이 어느 설정인지 불분명합니다. |
| D7 | **힘 크기를 보지 않습니다.** 로봇 쪽 법선은 접촉력 방향이고(`wrench_preprocess_jit`), `compute_wrench_space`가 정규화합니다. 모서리도 단위 길이입니다. | 0.1–10배 힘에서 지지값이 같습니다(`03` §3). DexForce와의 핵심 차이입니다(`02` §4). |
| D8 | **r_fc는 손별이고 기본 꺼짐입니다.** body를 amax로 합친 뒤 손 하나의 지지값으로 비율을 계산하고, 참조 접촉 레이블로만 켭니다. 가중치 0이며 "잡음 있는 레퍼런스에서는 켜고 접촉 항 셋을 0으로 하라"는 주석이 있습니다(`v2d_hand_env_cfg.py:276–285`, `rewards.py:627`). | 양손으로만 force closure가 되는 물체는 손별 r_fc가 1에 닿지 못합니다. 512개 표본이라 force closure의 필요조건일 뿐입니다(`03` §4). |
| D9 | **시연을 절반 속도로 재생합니다.** `motion_speed 0.5`(`commands_cfg.py:64`, `v2d_hand_env_cfg.py:63`, `fc/configs/task/default.yaml:4`, `sharpa_flash_sac.yaml:20`). 전신 G1 설정은 1.0입니다. | 논문에 없습니다. Fig. 7의 "40–48초"가 원본 시간인지 알 수 없고, 실물 재생 속도에도 영향을 줍니다. Do as I Do도 실물을 약 0.5배속으로 재생했습니다(`do_as_i_do/01` §2). |
| D10 | **같은 body 위의 추가 접촉은 약하게만 벌합니다.** 보상은 사람 지지값이 0인 방향을 세지 않고, r_unintend는 사람 접촉이 없는 body에만 걸립니다. | toy에서 옆면 추가 접촉은 보상 1.00 → 0.87로만 떨어집니다(`03` §2.4). |
| D11 | **r_relative가 꺼져 있습니다.** 함수는 있지만(`rewards.py:487–560`) 기본 RewardsCfg와 FlashCHORD 목적 함수 어디에도 등록되지 않았습니다. 커리큘럼 인자 기본값도 0입니다(`curriculum.py:132–133`). | 논문이 다물체 과제(삽입·붓기)의 핵심으로 설명한 항입니다. 공개 설정으로는 Table 1의 다물체 19개 결과가 이 항 없이 나온 것인지 알 수 없습니다. |
| D12 | **시드 수:** Table 1 캡션은 3개, App. G는 DexMachina·SPIDER 시드 5개. | 표 오차 막대의 의미가 기준선마다 다를 수 있습니다. |
| D13 | **SP-SR의 위치 오차 정의가 구현마다 다릅니다(2026-11-12 추가).** CHORD 공개 평가 코드 `fc/evaluation/metrics.py: spider_success`는 시간 평균한 원 위치 오차를 10 cm·0.5 rad와 비교하고, docstring에 "원 구현의 시간 평균 빼기는 생략한다"고 적었습니다. 중심화한 오차(`spider_centered_position_error`)는 진단 값(`spider_position_error_centered_m`)으로만 기록합니다. SPIDER 원 스크립트는 양손 모드에서 궤적 평균을 빼고(`flashdexretarget/04` D3), V2D 챌린지 키트 `metric_code/track_2/SP-SR.py`는 첫 프레임을 강체 정렬(`_warp_to_reference_start`)한 뒤 원 오차를 씁니다(`v2d_corl_tutorial/04` D2). 다음 커밋 `5226e34`의 `spider_success`도 같습니다. | Table 1이 어느 정의로 계산됐는지는 논문에 없습니다. 처음 분석에서 SP-SR을 "궤적 평균으로 중심화하는 관대한 지표"라고 한 것은 SPIDER 원 스크립트 기준이었고, CHORD 공개 코드와는 맞지 않아 `01` §3.3, `02` 표와 §3.4, 블로그를 고쳤습니다. SPIDER 대비 동률(t = 0.58)이라는 통계 결론은 그대로입니다. |

## 3. 주장별 재계산

`table_check.py`, `wrench_toy.py` 결과입니다. 자세한 해설은 `03`에 있습니다.

| # | 주장 (출처) | 재계산 | 판정 |
|---|---|---|---|
| C1 | 1,831개 평균 82.12% (Abstract, Fig. 5) | n 합 1,831. 막대 n 가중 평균 0.781, 단순 평균 0.801. 82.12% × 1,831 = 1,503.6(정수 아님), 4,511/5,493 = 82.12%(시드 3개면 맞음). 평가의 70%가 HOT3D | ⚠️ 집계 방식 미기재 |
| C2 | 전신 90.77% (Abstract, Fig. 5 오른쪽, Table 3) | 17개 가중 평균 0.9077, Table 3 전체 = 손만 12개 평균 0.925 | ✅ |
| C3 | 모든 지표에서 기준선보다 낫다 (Table 1) | SPIDER 대비 SP-SR t = 0.58(p = 0.60), RPE t = −1.9(p = 0.17)는 유의하지 않음. AUC·MP-SR·JPE·MPPE는 p < 0.05 | ⚠️ SPIDER 지표에서는 동률 |
| C4 | 접촉 지도가 풍부할수록 좋다 (Table 2) | Position Only 대비 6개 지표 모두 p < 0.03, SP-SR 이득의 41%가 wrench 몫. CHORD 행은 Table 1과 같은 실행 | ✅ |
| C5 | 실물 전이 (Sec. 4.6) | 23/25, Wilson [0.75, 0.98]. 상자 open 6/7 vs closed 7/7, Fisher p = 1 | ✅ (표본 작음) |
| C6 | CWS 보상이 성공과 상관 (Fig. 6) | r² = 0.637 ≈ 2/3 | ✅ (인과 아님) |
| C7 | 과제당 약 2시간, 기준선 4–20시간 (App. B) | FlashCHORD 2.5억 전이 = 2시간이면 34.7k/s. PPO 경로는 19.7억 전이. FDR 측정 PPO 모션당 57–66 GPU-h | ⚠️ PPO 경로에는 해당 안 됨 |
| C8 | 15 cm / 40° 성공 판정 (Sec. 4.1) | 코드 PPO 0.2 m / 0.7 rad, FlashCHORD 0.10 m / 0.50 rad | ❌ 코드와 다름 (D3) |
| C9 | 지지 함수 비교가 위치 보상보다 접촉을 잘 구분 (Sec. 3.1 그림) | 같은 위치, 법선 90° 차이: 위치 보상 1.00, 코드 보상 0.33 | ✅ |
| C10 | r_fc 최대화 = force closure (Sec. 3.2) | 두 점 집기(force closure 아님)에서 r_fc = 1, 표본 ε = 0.053 | ❌ 512개 표본에서는 필요조건일 뿐 |
| C11 | Ferrari–Canny ε = min 지지값 (App. D) | 512개 방향의 덮는 각도 중앙값 22°, 최대 38°. 표본 최솟값은 진짜 ε의 상한 | ⚠️ Fig. 3 ε 분포는 상한 |
| C12 | wrench는 힘·토크로 비교 (Abstract) | 힘 크기 0.1–10배에서 지지값 동일 | ⚠️ 방향만 비교 |
| C13 | v1 ↔ v2 | 그림 11개 바이트 동일, 초록 수치 동일, Table 1·2와 학습 설정 단락은 v2에서 새로 씀 (`01` §5) | 기록 |
| C14 | 프로젝트 페이지 `chord.pdf` = arXiv v2 | 단어 12,075개 중 날짜 도장만 다름 | ✅ |

## 4. 구현할 수 있는지

### 4.1 공식 재현: 이 맥에서는 불가, NVIDIA GPU가 있으면 가능

- **이 맥(Apple M4 Pro, 메모리 약 24 GiB, MPS, CUDA 없음):** 두 경로 모두 실행할 수 없습니다. Isaac Lab 경로는 Docker + NVIDIA Container Toolkit + 드라이버 580(CUDA 13 권장)이 전제입니다(`robotic_grounding/README.md`). FlashCHORD는 Linux, 드라이버 580 이상, 전체 학습에 **48 GiB GPU와 여유 46,000 MiB 이상**을 요구합니다(`flash_chord/README.md`). Newton·Warp에 CPU 장치가 있지만 FlashCHORD는 Docker·NVIDIA 전제로 짜여 있어 맥 CPU 실행은 공식 경로가 아닙니다(시도하지 않음).
- **NVIDIA GPU가 있으면:** 코드(Apache-2.0)와 과제 라이브러리(Hugging Face, CC BY 4.0)가 공개되어 있습니다. `tissue_box_simple` 시퀀스 하나는 리타겟까지 끝난 상태로 들어 있어 바로 학습해 볼 수 있습니다. 원천 데이터셋과 MANO는 각자 라이선스 동의 후 직접 받아야 합니다.
- **비용 견적:** FlashCHORD 논문 기준 과제당 약 2 L40S-h이면 Table 1(60개 × 시드 3) = 360 GPU-h, 1,831개 한 번 = 약 3,700 GPU-h입니다. PPO 경로는 FDR 측정 기준 모션당 57–66 RTX 3090 GPU-h입니다. 24 GiB 소비자 GPU는 FlashCHORD 전체 학습 요구(48 GiB)를 채우지 못하고, PPO 경로는 가능합니다(FDR이 RTX 3090으로 돌림).
- **재현 시 맞춰야 할 것:** 종료·성공 임계값(D3), 완료율 0.7 집계(C1), r_relative 활성화(D11), 환경 수(D6), 시드 수(D12). 이것들이 논문과 코드에서 달라서 "공식 코드를 그대로 돌리면 논문 수치가 나온다"고 기대하기 어렵습니다.

### 4.2 아이디어 재구현: 쉽고, 이미 핵심은 numpy로 옮겼음

- 핵심 보상(마찰원뿔 → wrench 행렬 → 지지 함수 → 보상·벌칙)은 numpy로 약 80줄입니다(`analysis/wrench_toy.py`). torch로 옮기면 어느 시뮬레이터의 접촉점·접촉력 출력에도 붙일 수 있습니다. MuJoCo의 `mj_contactForce`나 MJX가 주는 접촉점·힘 방향이면 충분합니다.
- 재구현 시 권장: Eq. 3이 아니라 코드식(방향별 exp 평균)을 쓰고, μ·기저 개수·r_c 정규화를 명시하고, force closure 판정은 표본 최솟값 대신 LP나 더 많은 방향으로 확인합니다.
- 정책 학습까지 포함한 축소판(한 손 + 상자 뚜껑, MuJoCo CPU)은 이 맥에서도 가능합니다. mujoco 설치가 필요하고, 정책 학습은 numpy CEM이나 torch(MPS)로 할 수 있습니다.

### 4.3 이 맥에서 이미 한 것과 할 수 있는 것

| 작업 | 상태 |
|---|---|
| 표·그림 재계산(C1–C6, C13–C14), v1 ↔ v2 단어·그림 대조 | 완료 |
| 공식 보상 함수 numpy 이식, 상자 뚜껑·위치 이동·힘 크기·μ·force closure toy(W1–W9) | 완료 |
| 학습 예산 산술과 FDR 수치 정합(C7, `02` §3) | 완료 |
| 공개 과제 라이브러리 일부(수 MB)를 받아 접촉 지지값 분포·표본 ε 상한 효과 측정 | 가능(다운로드 승인 필요) |
| MuJoCo 한 손 축소 실험 | 가능(mujoco 설치 승인 필요) |
| 공식 학습 재현 | 불가(NVIDIA GPU 필요) |

## 5. 재현 정보 체크리스트

| 항목 | 논문 | 코드 |
|---|---|---|
| 보상 가중치 | 없음 | 있음(D5) |
| β, v | 기호만 | 0.1, 0.1 |
| μ, 모서리 수, 방향 수 | 없음 | 0.1, 8 + 1, 512 |
| 환경 수 | 2,048 / 4,096 혼재 | 4,096 |
| 총 학습량 | "약 2시간" | PPO 19.7억, FlashCHORD 2.5억 전이 |
| 종료 임계값 | 15 cm / 40° | 0.2 m / 0.7 rad 또는 0.10 m / 0.50 rad |
| 과제 목록(1,831개, 60개) | 없음 | 없음(데이터셋에 과제 전체는 있음) |
| 학습 로그·체크포인트 | 없음 | 없음 |
| 시드 | 3(Table 1) / 5(App. G) | 설정 인자 |
