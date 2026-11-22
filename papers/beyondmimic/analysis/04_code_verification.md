# 04. 공식 코드 대조: BeyondMimic

대상: `HybridRobotics/whole_body_tracking`, 커밋 `9db915838b17785c6fec90c91fe80b95b147e1bd` (2026-10-15, "Mark repository as retired and point to mjlab"). 얕은 클론을 `repo/`에 두지 않고 임시 폴더에서 정적으로만 읽었습니다. 실행, import, 설치는 하지 않았습니다. 이 저장소는 **추종 학습 코드만** 있습니다. 확산 제어, VAE, 실로봇 배포 C++ 코드는 없습니다. 아래 대조는 추종 부분만 다룹니다.

## 1. 저장소 상태와 라이선스

- **아카이브됨.** README 첫머리에 "This repository is retired and no longer maintained"와 mjlab 이전 안내가 있습니다. 마지막 커밋은 2026-10-15 (`9db9158`)입니다. 마지막 기능 커밋은 2025-10-05 (`cd65172`)입니다.
- 라이선스: MIT (`LICENCE`). 첫 줄에 `Copyright (c) 2024, The Isaac Lab Project Developers`가 있습니다. Isaac Lab 템플릿에서 가져온 문구입니다.
- 파일 43개, 얕은 클론 376 KB.
- 실행 요구 사항(README): Isaac Lab 2.1.0, Isaac Sim 4.5.0, Python 3.10, Linux x86-64. CUDA GPU 필수. 로봇 설명 파일은 외부 URL에서 받아야 합니다.

## 2. 논문 값과 코드 값 대조

표의 "일치"는 논문 값과 코드 값이 같다는 뜻입니다.

| 항목 | 논문 | 코드 (파일:위치) | 판정 |
|---|---|---|---|
| 정책 주기, 시뮬 dt | 50 Hz (보충 S4) | decimation 4, sim dt 0.005 s (`tracking_env_cfg.py` 312–315) | 일치 |
| PPO 구조 | [512,256,128] ELU, lr 1e-3, clip 0.2, entropy 0.005, γ 0.99, λ 0.95, desired KL 0.01, 5 epoch, 4 미니배치, 24 스텝/환경, 30k 반복 (표 S5) | `rsl_rl_ppo_cfg.py` 같은 값 | 일치 |
| PPO 추가 설정 | 표 S5에 없음 | `max_grad_norm = 1.0`, `schedule = "adaptive"`, `use_clipped_value_loss = True` | **D5 (논문에 없음)** |
| 작업 보상 σ | p 0.3, R 0.4, v 1.0, ω 3.14 | `rewards` weight 0.5/0.5/1/1/1/1, std 0.3/0.4/1.0/3.14 (`tracking_env_cfg.py` 202–230) | 일치 |
| 정규화 가중치 | 관절 한계 −10, 행동 변화 −0.1, 자기 충돌 −0.1 | `joint_limit` −10.0, `action_rate_l2` −1e-1, `undesired_contacts` −0.1 (임계 1.0 N) | 일치 |
| 관절 한계 소프트 계수 | 0.9 | `soft_joint_pos_limit_factor=0.9` (`robots/g1.py`:61) | 일치 |
| 게인 공식 | k_p = Iω², k_d = 2ζIω, ω = 10 Hz, ζ = 2 | `NATURAL_FREQ = 10 Hz`, `DAMPING_RATIO = 2.0`, 반사 관성 기반 (`robots/g1.py`:12–23) | 일치 |
| 행동 스케일 | 0.25 τ_max/k_p | `G1_ACTION_SCALE = 0.25 * e / s` (`robots/g1.py`:195) | 일치 |
| 마찰 범위 | 정지 0.3–1.6, 동적 0.3–1.2, 반발 0–0.5 | 같은 범위 (`tracking_env_cfg.py` 163–165) | 일치 |
| 토르소 질량 중심 | x ±0.025, y ±0.05, z ±0.05 | 같은 범위, `torso_link` (`tracking_env_cfg.py` 182–185) | 일치 |
| 밀치기 간격과 속도 | 간격 1–3 s, 선속도 x,y ±0.5, z ±0.2, 각속도 ±0.52, ±0.52, ±0.78 | `interval_range_s=(1,3)`, `VELOCITY_RANGE` 같은 값 | 일치 |
| 기본 관절 위치 오프셋 | 대부분 ±0.01 rad, **발목 ±0.1 rad** | 모든 관절 ±0.01 (`pos_distribution_params (-0.01, 0.01)`, `joint_names=[".*"]`) | **D3 (발목 예외 없음)** |
| 리셋 자세 섭동 | x,y ±0.05, z ±0.01, roll·pitch ±0.1, yaw ±0.2 | 같은 값 (`tracking_env_cfg.py` 89–97) | 일치 |
| 리셋 관절 섭동 | 보충 S1에 없음 | `joint_position_range=(-0.1, 0.1)` (`tracking_env_cfg.py` 103), `commands.py` 268에서 리셋 시 더함 | **D2 (논문에 없음)** |
| 관측 잡음 | 보충 S1·표 S2에 관측 잡음 없음. 논문은 DR을 "3개 파라미터"로 한정 | 정책 관측에 Unoise: 앵커 위치 ±0.25, 앵커 자세 ±0.05, 기저 선속도 ±0.5, 각속도 ±0.2, 관절 위치 ±0.01, 관절 속도 ±0.5 (`tracking_env_cfg.py` 119–135) | **D1 (논문에 없음, 큰 차이)** |
| 비대칭 액터-크리틱 | 크리틱에 몸통별 상대 자세 추가 (보충 S1) | 크리틱 관측은 잡음 없음, 같은 관측 (`tracking_env_cfg.py` 138–139) | 부분 일치 (크리틱의 몸통 상대 자세는 확인 못 함) |
| 자세 표현 | Rot6D (관측) | `matrix_from_quat(ori)[..., :2]` (`observations.py` 82–83) | 일치 |
| 종료 조건 | 앵커 |z 오차| > 0.25 m 또는 |자세 오차| > 0.8 rad. 끝단 z 오차 > 0.25 m | 같은 값. 끝단 목록: 발목 2개, **손목 요 링크 2개** (`tracking_env_cfg.py` 255–277) | 값 일치. **D6: 논문은 "손(hands)", 코드는 wrist_yaw 링크** |
| 적응 샘플링 실패율 EMA | 0.999 이전 값 + 0.001 새 값 | `adaptive_alpha=0.001`, 이전 값 가중 `(1 − α)` (`commands.py` 296–297) | 일치 |
| 균등 바닥값 | 0.1/S | `adaptive_uniform_ratio=0.1` (`commands.py` 217) | 일치 |
| 커널 | ρ = 0.8, 조회 u ∈ {0, 1, 2} (K = 3) | `adaptive_lambda=0.8`, **`adaptive_kernel_size=1` (기본값, 설정에서 덮어쓰지 않음)** (`commands.py` 368–369) | **D4: K = 1이면 조회 구간이 없음 (논문과 다름)** |
| 구간 길이 | 1초 | `bin_count = time_step_total / (decimation · dt)` (`commands.py` 80) | 구간은 1/(decimation·dt)=0.02 s 단위 빈으로 쪼갬. 논문 1초와 다름 (D7, 주의) |
| 학습 환경 수 | 논문 미기재 | `num_envs = 4096`, `episode_length_s = 10` (`tracking_env_cfg.py` 298, 313) | 보고되지 않음 (U6) |

### 2.1 D7 해석 주의

`commands.py`:80의 빈 수는 `time_step_total // (decimation · sim.dt)`입니다. decimation·dt = 4 × 0.005 = 0.02 s이므로, 빈 하나는 0.02 s(정책 한 스텝)입니다. 논문의 1초 구간과 다릅니다. 다만 코드 주석이나 설정 이름을 보면 빈 크기 설정 변수가 따로 있는지는 확인하지 못했습니다. 그래서 D7은 "코드 기본값이 1초가 아님"이라고만 적습니다.

## 3. 차이 목록 (D1–Dn)

| 코드 | 내용 | 영향 | 근거 |
|---|---|---|---|
| D1 | 정책 관측에 잡음이 들어감. 논문은 3개 DR만 말함 | 실로봇 전이에 중요할 수 있음. 논문 Results의 "minimal domain randomization" 해석과 맞지 않음 | `tracking_env_cfg.py` 119–135 |
| D2 | 리셋 관절 위치 섭동 ±0.1 rad가 있음 | 동작 초기 상태의 다양성이 논문보다 큼 | `tracking_env_cfg.py` 103, `commands.py` 268 |
| D3 | 기본 관절 위치 오프셋은 발목 예외 없이 ±0.01 rad | 논문 보충 S1의 ±0.1 rad 발목 값과 다름 | `tracking_env_cfg.py` 170–176 |
| D4 | 커널 K = 1 (기본값), 논문은 K = 3 | 실패율 평활이 논문보다 약함 | `commands.py` 368 |
| D5 | PPO에 `max_grad_norm = 1.0`, 적응 학습률 스케줄이 있음 | 표 S5 목록에 없음 | `rsl_rl_ppo_cfg.py` |
| D6 | 끝단 종료 목록에 손목 요 링크가 있고 손은 없음 | 논문 문구와 다름 (값은 같음) | `tracking_env_cfg.py` 273–276 |
| D7 | 빈 크기가 정책 스텝 0.02 s | 논문 1초 구간과 다름, 설정 확인 필요 | `commands.py` 80 |
| D8 | 확산 제어 관련 코드(VAE, 확산, 분류기 안내, 대칭 증강)가 저장소에 없음 | 논문의 주요 기여 2 (확산 제어)를 이 코드로는 재현할 수 없음 | 저장소 파일 목록 (43개) |

## 4. 구현 가능성 판정 (이 맥)

**판정: 추종 학습은 이 맥에서 불가능하고, 확산 제어는 코드 없이 구현은 가능하지만 학습은 GPU가 필요합니다.**

- 이 맥: Apple M4 Pro, 약 24 GiB 메모리, MPS 가능, **CUDA 없음**.
- 추종 학습 코드는 Isaac Sim·Isaac Lab (NVIDIA GPU, Linux)이 필요합니다. mjlab은 MuJoCo-Warp 기반이라 CUDA가 전제된 것으로 보입니다 (README에 "GPU" 요구가 있는지 이 확인에서는 읽지 않음, 미확인 U7).
- 실로봇 배포: Unitree G1과 외부 C++ 컨트롤러, 상태 추정 장치 필요 (보충 S4). 불가.
- **CPU로 가능한 것**: (a) 보고 수치 재계산 (이 폴더의 `recompute.py`), (b) 확산 구조의 장난감 재구현 (2-링크 등 작은 모델은 CPU로 충분하지만 이 저장소의 1차 확인 대상은 아님), (c) 보상 함수의 값 계산 (정적 식 확인).
- **노력 추정 (추정, 측정 없음):** 추종 재현 1–2주 (GPU 1장, 동작 1개 기준). 확산 제어를 논문에서 다시 구현하는 작업은 2–4주 (GPU 필요). 실로봇은 별도 장비가 필요해 판단 불가.

## 5. 확인 못 한 것 (U-항목)

- U1: 확산 제어, VAE, 학습 가중치, 실로봇 C++ 코드의 공개 여부. 이 저장소에는 없음. `motion_tracking_controller` (MIT, 2026-09-10 push)에 배포 코드가 있을 수 있으나 클론하지 않음.
- U6: 학습 환경 수 4096 등 설정은 코드에서 확인했지만 논문 본문에 없음. 시드 수와 학습 곡선 기록은 없음.
- U7: mjlab의 GPU 요구 사항 세부는 이 확인에서 읽지 않음.
