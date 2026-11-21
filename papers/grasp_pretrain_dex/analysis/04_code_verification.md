# 04. 코드 대조 (논문 ↔ 공식 코드, 차이 D1–Dn)

공식 코드 두 개를 **정적으로만** 읽었습니다. 실행, import, 설치, 데이터 다운로드를 하지 않았습니다. 확인 기준은 `02_prior_work.md` §1의 복제본 커밋입니다(G2D-Policy `80c204a`, DexCraft `1bed0cd`). 두 저장소 모두 라이선스가 없어 복제본은 `repo/` 아래에 두고 게시하지 않습니다.

판정 기호: **일치**, **차이**(논문과 코드가 다름), **확인 불가**(저장소에 없거나 찾지 못함).

## 1. 대조 범위와 결과 요약

- 대조 대상은 논문 본문, 표 5(고수준), 표 6(저수준), 표 8(어댑터), 부록 A.1–A.3(과제·데이터), 부록 B.3(궤적 생성), 부록 C.1(기준선)입니다.
- **차이** 항목은 D2, D3, D4, D5, D6, D11, D14, D20입니다. 대부분 문서 차이이고, 재현 결과에 영향을 줄 수 있는 것은 D3, D4, D6, D14입니다.
- **확인 불가** 항목은 D13, D16, D17, D22입니다. 저장소에 해당 데이터, 설정, 코드가 없거나 찾지 못했습니다.
- 나머지는 일치합니다(D1, D7, D8, D9, D10, D12, D15, D18, D19, D21).

## 2. 대조표

| 번호 | 논문 (위치) | 코드 (위치) | 판정 | 설명 |
|---|---|---|---|---|
| D1 | 고수준: PointNet++ MSG, 6층 SA와 6층 FP, 손 키포인트 16개, 고정 분산 {0.01, 0.05, 0.1, 0.25, 0.5}, 균일 계수 0.1 (§5.1, 표 5) | `policy/high_level.py`(키포인트 16, `fixed_variance`), `config/high_level.yaml`, `model/vision/pointnet2.py:195–276` (SA 6, FP 6) | **일치** | 구조와 손실 계수가 맞습니다. SA 점 수(1024, 512)와 반경(0.025/0.05)도 표 5와 맞습니다. 나머지 층은 확인하지 않았습니다. |
| D2 | 고수준 옵티마이저: AdamW (표 5) | `scripts/train_high_level.py:75`: `torch.optim.Adam(model.parameters(), lr=cfg.lr)` | **차이** | 옵티마이저 종류가 다릅니다. 표 5에는 가중치 감쇠 값이 없어 AdamW 기본값(0.01)인지 0인지 알 수 없습니다. 영향은 작을 것으로 보입니다. |
| D3 | 고수준 학습 세부 (표 5에 없음) | `config/high_level.yaml`: `bn_recalib_batches: 200`(저장 전 BatchNorm 통계 재추정), `val_ratio: 0.02`(검증 2% 분할) | **차이** (논문에 없는 구현 세부) | 논문 설명에 없는 두 가지 단계가 있습니다. 결과에 영향을 줄 수 있으니 재현에서 명시해야 합니다. |
| D4 | 저수준 인코더 출력 차원 64 (표 6) | `config/low_level.yaml`: `encoder_output_dim: 60` | **차이** | 표 6과 코드가 다릅니다. 어느 값이 실제 실험에 쓰였는지 저자 확인이 필요합니다. |
| D5 | 저수준 상태: 관절 위치 22차원, 관절 속도 22차원, 이전 관절 목표 22차원, 말단 포즈 7차원 (§5.2) → 합계 73차원 | `policy/dp3.py:1–10` 주석과 `config/task/spraybottle.yaml`: `agent_pos` 74차원. 목표 블록은 7차원 팔 목표 포즈와 16차원 손 목표로 구성 | **차이** (문서 설명이 코드와 다름) | 논문 나열로는 73인데 코드는 74입니다. 코드의 "이전 목표"는 관절 22개가 아니라 팔 목표 포즈 7개와 손 16개입니다. 논문 설명을 코드 기준으로 고쳐 써야 합니다. |
| D6 | 저수준 학습 epoch 500 (표 6) | `config/pretrain_low_level.yaml`: `num_epochs: 5`, `checkpoint_every_steps: 5000`, `lr_schedule_steps: 1e9`(상수에 가까운 학습률) | **차이 또는 불명확** | 표 6의 500은 하위 과제 미세 조정에 해당하는 값으로 보입니다. 사전학습 일정은 논문에 명시되지 않았습니다. 사전학습의 정확한 스텝 수는 알 수 없습니다. |
| D7 | 사전학습 행동 28차원(6 + Shadow 22), 하위 과제 22차원(6 + LEAP 16) (§4, §5.2) | `config/task/g2d_pretrain.yaml`(action 28), `config/task/spraybottle.yaml`(action 22) | **일치** | |
| D8 | 어댑터: LEAP 16관절을 표 8의 Shadow 슬롯 16개에 매핑, 나머지 6슬롯 0 채움 (부록 B.4, 표 8) | `policy/dp3.py:24–30`: `SHADOW_TO_LEAP_JOINT = [1,5,9,18,0,4,8,17,2,6,10,20,3,7,11,21]`. 표 8의 16개 쌍과 모두 일치. 역매핑 `LEAP_TO_SHADOW_JOINT`도 서로 역함수 관계 | **일치** | LEAP 관절 번호는 **DexCraft 컨트롤러 순서**(`dexcraft/agents/xarm6_leap.py:42–46`의 `hand_joint_names`: leap_1, leap_5, leap_9, leap_12, leap_0, …)입니다. 논문 표 8은 이 정의를 쓰지 않아 URDF 번호로 읽을 수 있습니다. 명세가 부족합니다. |
| D9 | 성공 기준: 목표 위치 0.04 m 이내, 방향 30° 이내 (부록 A.1) | `dexcraft/envs/base.py:79–80`: `goal_thresh = 0.04`, `orient_thresh = 30` | **일치** | |
| D10 | 파지 기준: 두 손가락 이상, 접촉력 0.5 N 이상 (부록 A.1) | `dexcraft/agents/xarm6_leap.py:152`: `is_grasping(min_force=0.5)`. `dexcraft/agents/floating_shadow.py:110`: `n_contacts >= 2` | **일치** | 손가락 접촉 개수는 확인했고, 힘 임계값은 Shadow 쪽 함수에서 확인하지 않았습니다. |
| D11 | 목표 샘플링: 높이 0.1–0.3 m 위, 위치 0.1 m 정사각형, 분무기 z축 회전 ±30° (부록 A.2) | 높이: `dexcraft/envs/base.py:212` 일치. 위치: `spawn_half_size = 0.05`(`base.py:84`) 일치. 회전: `base.py:70`의 기본 `obj_rot_bounds`가 z ∈ [5π/6, 7π/6], 즉 180° ± 30°. 라이터는 `lighter.py:18`에서 z ∈ [-π/6, π/6]로 논문과 같습니다 | **차이 가능** (좌표 규약 차이로 보임) | 분무기 샘플링이 180° 기준인지 논문에 적혀 있지 않습니다. 시뮬레이션 방향 규약 차이입니다. 저자 확인이 필요합니다. |
| D12 | 궤적 생성: 마지막 프레임에서 파지 유지인 궤적만 남김 (§5.2, 부록 B.3) | `DexCraft/scripts/dexonomy/collect_dataset.py:113–116, 130`: 마지막 프레임 `is_grasped` 확인 후 저장 | **일치** | 모션 플래닝과 보간 세부는 확인하지 않았습니다. |
| D13 | G2D-Pretrain 355,677 궤적 (표 7) | 공개 HF 데이터는 DexCraft README 기준 `demos/`와 `generated/`(MimicGen 생성분). G2D-Pretrain 항목은 README에 없습니다 | **확인 불가** (재생성 필요로 보임) | Dexonomy 자산(DexLearn)과 `collect_dataset.py`로 다시 만들어야 할 가능성이 큽니다. HF 내용은 내려받지 않아 확인하지 못했습니다. |
| D14 | 과제마다 MimicGen으로 **100개**로 늘림 (§4) | `DexCraft/README.md` Step 7: "논문에 쓴 생성 데이터는 학습 500 + 검증 200개". `task.dataset.n_demo` 기본값은 100(`spraybottle.yaml`) | **차이** | 생성 풀 크기(500/200)와 논문의 100개가 다릅니다. 학습 부분집합 정의(처음 N개)도 논문에 없습니다. |
| D15 | 평가 500 에피소드 (표 1 캡션) | `config/task/spraybottle.yaml`: `n_episodes: 500`. 에피소드 최대 스텝 800(`max_steps`). 환경 자체의 `max_episode_steps=30000` | **일치** (최대 스텝은 논문에 없음) | 평가 길이가 800스텝인지 논문에는 없습니다. |
| D16 | 제어 주파수 20 Hz (부록 A.2) | `grep`으로 `control_freq`, `sim_freq`, `20 Hz` 등을 찾았으나 명시적 설정을 찾지 못함 | **확인 불가** | ManiSkill 기본값에 기대는 것일 수 있습니다. |
| D17 | 기준선: DP는 RGB 두 대(128px)와 ResNet-18, 1000 epoch (표 9). DP3는 목표 키포인트를 0으로 채움 (C.1) | 저장소에 DP 구현이 없고, DP3 전용 설정(목표 0 채움)도 method 설정에 없음(`config/method/`에 `g2d_policy.yaml`과 `hp_scratch.yaml`만 있음) | **확인 불가** | 표 1의 DP, DP3, DP3-PT, GraspXL-PT, UniDex-VLA, Dex1B 기준선 코드는 저장소에 없습니다. 기준선 재현은 따로 구현해야 합니다. |
| D18 | 저수준 DP3: Act3D형 교차 주의, FiLM 조건, 관측 2, 예측 16, 실행 4, 학습 확산 단계 100, 추론 10 (표 6) | `config/low_level.yaml`: `condition_type: film`, `n_obs_steps: 2`, `horizon: 16`, `n_action_steps: 4`, `num_train_timesteps: 100`, `num_inference_steps: 10` | **일치** | 학습률 1e-4, 배치 128, AdamW(β 0.95, 0.999, 감쇠 1e-6), 코사인, 웜업 500, EMA 0.9999도 맞습니다. |
| D19 | 고수준 샘플링: 혼합 가중치로 점 하나를 뽑고 그 평균을 부목표로 씀 (B.1) | `policy/high_level.py:27`의 `sample_goal`: 앵커 점 샘플링 후 평균 반환 | **일치** | |
| D20 | 부록 C.1 "DexHier demonstrations" | 저장소에 `DexHier` 명칭 없음 | **차이(용어)** | 논문 내부 명칭 잔재로 보입니다. 본문 정의가 없습니다. |
| D21 | 논문 본문: 고수준과 저수준을 정답 부목표로 독립 학습 (§5.1) | `scripts/train_high_level.py`와 `train_low_level.py`는 별도 스크립트. 부목표는 `goal_gripper_pcd`로 제공(`DexCraft/scripts/mimicgen/dataset_states_to_obs.py`의 `subtask_goals`) | **일치** | 부목표 신호는 `grasp`와 `place`(물체 배치와 방향)입니다. 논문의 세 단계 중 정렬과 작동을 이 두 신호로 나눈 것으로 보입니다. |
| D22 | 실로봇 설정 (§6.1, Franka, Azure Kinect 두 대, GELLO, Manus 장갑) | 저장소에 실로봇 코드 없음 | **확인 불가** | |

## 3. 재현 설정에서 걸리는 점

- **디바이스가 CUDA로 고정되어 있습니다.** `config/low_level.yaml`의 `training.device: "cuda:0"`, `config/high_level.yaml`의 `device: cuda`입니다. CPU나 MPS에서 돌리려면 설정을 고쳐야 합니다.
- **환경 설치가 무겁습니다.** DexCraft는 pixi, PyTorch3D, robomimic(MimicGen 요구 버전)을 씁니다(`DexCraft/README.md`). G2D-Policy는 DexCraft 환경 안에서 `pip install -e`로 설치합니다.
- **데이터 경로가 고정되어 있습니다.** `data/dexcraft/<과제>/demo_obs.hdf5`, `data/g2d_pretrain/`을 기대합니다(`G2D-Policy/README.md`).
- **체크포인트 경로.** 사전학습 체크포인트는 `ckpts/g2d_pretrain_low_level.ckpt`로 복사해야 `method=g2d_policy`가 읽습니다.

## 4. 구현 가능성 판정 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**판정: 공식 수치 재현은 이 맥에서 불가능합니다. 대조와 구조 검증은 가능하고 이미 끝냈습니다.**

| 작업 | 맥에서 가능? | 근거 |
|---|---|---|
| 표 재계산과 검정 | **가능, 완료** | `analysis/recompute.py` |
| 두 저장소의 정적 대조 | **가능, 완료** | 본 문서 |
| 어댑터 매핑의 단위 검사 (표 8, 상태 인덱스) | **가능** (numpy/torch CPU) | 순수 인덱스 연산. 설치 없이 `dp3.py`의 상수만 읽어 검사할 수 있으나, import에는 torch와 여러 의존성이 필요합니다 |
| DexCraft 환경을 CPU에서 한 번 실행 | **조건부** | pixi, ManiSkill3, PyTorch3D, robomimic 설치가 필요합니다. 설치 전 승인이 필요합니다 |
| 고수준 또는 저수준 학습 | **불가 (현재 설정)** | 설정이 CUDA로 고정되어 있고, README는 4 GPU를 씁니다. 500 epoch 학습은 GPU 수일 규모로 보입니다(측정 전) |
| G2D-Pretrain 재생성 (355k 궤적) | **조건부** | 수십만 궤적의 시뮬레이션 롤아웃이 필요합니다. CPU 시간을 측정하기 전에는 판단하기 어렵습니다 |
| 실로봇 재현 | **불가** | 실로봇, Azure Kinect 두 대, 장갑 장비가 필요합니다 |

**노력 추정(측정 전, 추정치):**

- 어댑터와 데이터 로더의 CPU 단위 검사: 1–2일.
- 학습 없이 DexCraft 한 과제를 CPU에서 돌리고 관측 형식을 맞추기: 3–5일(설치 포함, 승인 후).
- 전체 파이프라인을 MPS나 CPU용 torch로 옮기고 G2D-Pretrain을 재생성하기: 수 주. 데이터 생성 규모에 따라 크게 달라집니다.

## 5. 이 문서에서 가정하거나 열어 둔 것

- 논문 본문과 코드 중 **어느 쪽이 실험 설정을 정확히 반영하는지** 알 수 없습니다(D4, D5, D6). 저자 확인이 필요합니다.
- HF 데이터셋의 실제 내용은 내려받지 않았습니다(D13).
- 복제본 파일은 `git log`로 커밋을 기록했지만 파일 목록 전체를 읽지는 않았습니다. 서브모듈은 확인하지 않았습니다.
