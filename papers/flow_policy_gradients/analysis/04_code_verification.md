# 04 코드 대조, 주장 검증, 구현 가능성 (Flow Policy Gradients, arXiv:2602.02481)

## 0. 대상과 방법

- **공식 저장소:** https://github.com/amazon-far/fpo-control, 클론 커밋 `8f460e4415a2a6debfbeb689093f5e36140bd178` (2026-08-12, 제목 "Fix ASPO objective: keep PPO surrogate per-element before torch.where"). 얕은 복제(`--depth 1`)로 `repo/`에 받았습니다.
- **방법:** 정적 읽기만 했습니다. 코드를 실행하거나 import하거나 설치하거나, 가중치와 데이터를 받지 않았습니다. 저장소 밖 자료(IsaacLab, 베이스 정책 W&B 체크포인트, DexMimicGen 등)도 받지 않았습니다.
- **저장소 구성:** `isaaclab_experiments/`(IsaacLab 확장 `isaaclab_fpo`, 보행과 모션 트래킹, CUDA 12.1+ 및 IsaacSim 4.5 필요), `manipulation_experiments/`(조작 미세조정 `finetune_online_rl.py`, robosuite·LeRobot·DexMimicGen, 개발 환경 L40S 46GB, Ubuntu 24.04).
- **판정 기호:** ✓ 일치, ✗ 불일치, ~ 부분 일치, ? 저장소에서 확인 불가.

## 1. 불일치 목록 (D1–D21)

심각도는 "재현할 때 결과가 달라질 가능성" 기준입니다. 높음은 결과 해석을 바꿀 수 있고, 중간은 설정을 맞춰야 하며, 낮음은 문서화 수준입니다.

| 번호 | 논문 | 저장소 | 심각도 | 근거 |
|---|---|---|---|---|
| **D1** | 식 (5): a^τ = τa + (1−τ)ε (τ=1이 데이터) | `actor_critic.py`: x_t = t·ε + (1−t)·a (t=1이 노이즈), 목표 ε − a, 적분 1→0 | 낮음 | 같은 경로를 시간 방향만 바꾼 규약. 부호를 검산해 적분 방향이 맞음을 확인 |
| **D2** | 식 (10)–(11): ψ_PPO는 Â ≥ 0, ψ_SPO는 Â < 0 | `fpo.py`: `torch.where(advantages_batch > 0, ppo, spo)`. 조작 쪽 `finetune_online_rl.py`도 `> 0` | 낮음 | Â = 0이면 두 항이 모두 0이라 결과 무영향 |
| **D3** | 식 (7): 제곱 L2 노름 ‖v̂ − (a−ε)‖² | `rl_cfg.py` 기본 `cfm_loss_reduction="sqrt"`(합/√d). 모션 설정은 `"mean"`(합/d). 조작은 Huber(D8) | **높음** | `rl_cfg.py` L89, `task_cfgs.py` `G1FlatMotionTracking…`. 비율 exp(ΔL)의 척도가 행동 차원에 따라 달라짐. 토이 2(`03` §5)가 척도 의존을 보임 |
| **D4** | 본문에 없음 (B.2.3에서 Huber와 클램프만 언급) | 시간 t를 역 CDF(Beta(1, β))로 샘플링하고 [0.005, 0.995]로 제한. β=1이면 균등 | 중간 | `fpo.py` L179–183, `rl_cfg.py` `cfm_loss_t_inverse_cdf_beta=1.0` |
| **D5** | B.2.3: 손실 클램프 후 차이 클램프 | 보행 기본 `cfm_loss_clamp=20`, `cfm_diff_clamp_max=10`, 음의 advantage 손실 상한 20. 모션은 3.0과 3.0. 조작은 `clamp_logratio=5`, `clamp_old_cfm_loss=4`(과제별 `clamp_ste`, 기울기 보존) | 중간 | `rl_cfg.py` L241–265, `task_cfgs.py`, `finetune_online_rl.py` L1216–1228, `fpo_experiments_summary.md` |
| **D6** | 본문에 없음 | 액터 EMA 기본 `ema_decay=0.95`, 워밍업 500 스텝, 기본으로 켜짐 | 중간 | `rl_cfg.py` L283–301, `on_policy_runner.py` L298–302 |
| **D7** | 본문에 없음 | 학습 행동 섭동 `action_perturb_std` 기본 0.02, 모션 설정 0.1 | 중간 | `rl_cfg.py` L101, `task_cfgs.py` 모션 |
| **D8** | Fig. 5: FPO++와 FPO를 같은 알고리즘의 두 변형으로 비교 | 조작 FPO++와 vanilla FPO는 `cfm_loss_average_group_size`(1 대 −1)뿐 아니라 Huber 사용(FPO++ 켬, FPO 끔), 클램프(FPO++ 켬, FPO 끔), 과제별 `clip_coef`(0.01–0.03), `max_grad_norm`(1–25), Huber δ가 다름 | **높음** | `fpo_experiments_summary.md` §1.1–1.3 표 |
| **D9** | 표 A.1: 행동당 샘플 16 | Spot, H1, G1 설정은 32, Go2만 기본 16 | 중간 | `task_cfgs.py` |
| **D10** | 표 A.2 (모션): 학습률 3e-4, 에포크 5, 미니배치 4 | `G1FlatMotionTracking…`은 학습률을 덮어쓰지 않아 기본 1e-4, 에포크 기본 16, 미니배치 6 | **높음** | `rl_cfg.py` L116–129, `task_cfgs.py` 모션 설정 |
| **D11** | 표 A.2: 적분 단계 50, 배포 5단계 | 모션 설정에 `sampling_steps` 덮어쓰기가 없어 기본 64. 배포 5단계 설정은 코드에 없음 | 중간 | `rl_cfg.py` L86, `task_cfgs.py`. 0 샘플링은 `act_inference(eval_mode="zero")`에 있음 |
| **D12** | B.2.2: 롤아웃 길이 96 | 모션 설정 `num_steps_per_env=48` | 중간 | `task_cfgs.py` `G1FlatMotionTracking…` |
| **D13** | 본문과 C.3: 적응 학습률은 "가능한 실험 변형"이며 핵심 목적이 아님 | 모션 설정 `schedule="adaptive"`로 KL형 학습률 조정이 켜짐. 지표는 노이즈 예측이 아니라 x1(행동 추정) 예측의 변화 제곱 평균 | **높음** | `task_cfgs.py` 모션 설정(`schedule="adaptive"`), `fpo.py` `update()`의 `schedule == "adaptive"` 블록(`x1_pred`). 논문 C.3는 "노이즈 예측에서 대수적으로 복원"이라고 씀 |
| **D14** | 조작 ASPO 결과 (C.6–C.7) | 커밋 `8f460e4`가 "ASPO 목적을 고침"을 제목으로 씀. 이전 버전 코드가 조작 ASPO 실험에 쓰였는지 알 수 없음 | **높음** | `finetune_online_rl.py` L1302–1320의 주석. 커밋 히스토리는 얕은 복제에 없음 |
| **D15** | 식 (12): 샘플 합 Σ_i | 코드는 손실을 모든 샘플에 대해 `.mean()`. 상수 배 차이 | 낮음 | `fpo.py` ASPO 분기, 상수 배라 Adam에서 대부분 상쇄 |
| **D16** | 본문에 없음 | 정규화된 advantage(기본 켬), advantage 클램프 100(기본) | 낮음 | `rl_cfg.py` L212–228 |
| **D17** | 표 A.1: Adam betas (0.9, 0.95) | 기본 `adam_betas=(0.9, 0.999)`. 모션 설정도 기본값 | 중간 | `rl_cfg.py` L135 |
| **D18** | 표 A.1: 가중치 감쇠 1e-4 | 기본 `weight_decay=1e-4` (AdamW) | ~ | 일치 |
| **D19** | D.1: 기울기 코사인 유사도 (그림 A.12) | 저장소에 해당 지표 코드가 없음 | 중간 | `grep -rIn cosine_sim`에서 일치 없음 |
| **D20** | 부록 B.2.2 "We will release code" (v2) | 저장소가 이미 공개됨. 코드는 있으나 결과 재생성 스크립트와 베이스 정책 체크포인트는 없음 | 낮음 | `README.md`, `manipulation_experiments/docs/reproduce.md` (베이스 정책은 W&B `flow-bc` 프로젝트 ID로 참조) |
| **D21** | 표 I (v1), 그림 A.9 값, C.4 베이스 정책 성공률 | 해당 평가 스크립트나 로그가 없음. `eval_base_policies.sh`는 있으나 원자료는 없음 | 중간 | `scripts/eval_base_policies.sh`, `03` §1 |

## 2. 주장별 검증 (C1–C19)

| 번호 | 논문 주장 | 저장소에서 확인한 것 | 판정 |
|---|---|---|---|
| C1 | 샘플별 비율 (식 (9)) | `fpo.py`에서 `log_ratio = old − new`를 샘플별로 계산하고 평균하지 않음. 조작에서는 `cfm_loss_average_group_size=1` | ✓ |
| C2 | ASPO: Â ≥ 0에 PPO, Â < 0에 SPO (식 (11)) | `fpo.py` ASPO 분기. `torch.where(advantages_batch > 0, ppo, spo)` | ✓ (D2) |
| C3 | SPO 식 (10): ρÂ − |Â|/(2ε)(ρ−1)² | `fpo.py`에 `ratio*A − |A|/(2*clip)*(ratio−1)²` 그대로 | ✓ |
| C4 | 대리 비율 exp(L_old − L_new) (식 (3)) | `log_ratio = old_cfm_loss_batch − cfm_loss_batch`, `ratio = exp(log_ratio)` | ✓ (단, 손실 노름 D3) |
| C5 | 제곱 L2 손실 (식 (7)) | 기본 `sqrt` 축약 | ✗ (D3) |
| C6 | 학습 64단계, 조작 10단계 | `sampling_steps=64` 기본. 조작 `--sampling_steps=10`(`run_main_benchmark.sh`) | ✓ |
| C7 | 0 샘플링은 평가에 사용 (§4.4, B.2.2) | `act_inference(eval_mode="zero")`와 `zero_sampling=True`. 모션 배포 5단계는 확인 못 함 | ~ (D11) |
| C8 | 조작 ASPO 비활성화 | 실행 스크립트 `--trust_region_mode=ppo` | ✓ |
| C9 | 조작 행동 청크 16, 샘플 8, 적분 10 | `--n_action_steps 16`, `--n_action_samples 8`, `--sampling_steps 10` | ✓ |
| C10 | 조작 액터 학습률 1e-5, 비평기 1e-4 (B.3) | `finetune_online_rl.py` `learning_rate_actor=1e-5`, `learning_rate_critic=1e-4` | ✓ |
| C11 | 과제별 γ (B.3) | `fpo_experiments_summary.md`의 과제 표: Can 0.99, Square·Box 0.995, Tray·Threading 0.999 | ✓ |
| C12 | 조작 GAE λ 0.99 (B.3) | `--gae_lambda 0.99` | ✓ |
| C13 | 보행 1500/2000 반복, 4096 환경 | `task_cfgs.py` `max_iterations` 1500(쿼드루페드), 2000(휴머노이드). 환경 수는 README 예시 `--num_envs 4096` | ✓ |
| C14 | 보행 학습률 1e-4, 클리핑 0.05, γ 0.99, λ 0.95, 에포크 16/32, 미니배치 4 (표 A.1) | `rl_cfg.py` 기본과 `task_cfgs.py` 에포크 | ✓ (betas ✗ D17, 샘플 ✗ D9) |
| C15 | 모션 트래킹 설정 (표 A.2) | 클리핑 0.01 ✓, γ·λ ✓, 관측 정규화 ✓, 네트워크 1024-512-256 ✓. 학습률·에포크·미니배치·적분 단계·롤아웃 길이 ✗ | ✗ (D10–D12) |
| C16 | 조작 주 실험 60회 = 5과제 × 4방법 × 3시드 (부록) | `run_main_benchmark.sh` 주석과 `reproduce.md` | ✓ |
| C17 | 가우시안 PPO 비교는 rsl_rl 기본값, 클리핑만 {0.1, 0.15, 0.2, 0.25} 스윕 (B.1.1) | 저장소에 가우시안 PPO 실행 설정이 없음(`isaaclab_fpo`는 flow 정책만). 비교 실험 코드는 공개되지 않음 | ? |
| C18 | 엔트로피 정규화(KL 추정기)는 가우시안 PPO 비교에만 사용 (C.3) | `fpo.py`의 `_compute_knn_entropy`와 `knn_entropy_coef`는 있음. 보행 기본값은 0 | ✓ (단, 기본 꺼짐) |
| C19 | 기울기 분산 감소 (D.1) | 해당 코드 없음 (D19) | ? |

## 3. 재현 경로 (저장소 기준, 실행하지 않음)

1. **보행:** IsaacLab 4.5 + IsaacSim + CUDA 12.1+, `isaaclab_experiments/setup_env.sh`(conda 생성, 서브모듈 초기화). Go2·Spot·H1·G1 학습은 4096 환경과 GPU가 필요합니다. README의 기대 곡선 그림은 저장소에 있습니다.
2. **모션 트래킹:** `Tracking-Flat-G1-v0`, LAFAN 동작 파일은 `whole_body_tracking_reference_data/download_lafan_data.py`로 받습니다(외부 데이터, 실행하지 않음). D10–D13 설정을 논문 표 A.2에 맞춰야 합니다.
3. **조작:** `manipulation_experiments/setup_env.sh`(Python 3.10, robosuite, DexMimicGen 클론), 베이스 정책은 W&B `flow-bc` 프로젝트 체크포인트(공개 여부 미확인). `scripts/run_main_benchmark.sh`에 60개 실행 명령이 있습니다(`DRY_RUN=1`로 명령만 출력 가능).

## 4. 구현 가능성 판정 (이 맥: Apple M4 Pro, 약 24 GiB 통합 메모리, MPS, CUDA 없음)

**판정: 알고리즘은 구현할 수 있습니다. 논문 결과 재현은 이 맥에서 불가능합니다. 구현 검증은 1차원 토이와 소형 MuJoCo 과제 수준까지 가능합니다.**

- **구현 가능 (노력 중간, 추정 3–5일, 350–500줄 PyTorch):** 샘플별 CFM 비율, ASPO, 손실 클램프, Euler 샘플러, 0 샘플링 배포를 MPS 또는 CPU에서 돌아가는 작은 MLP 정책으로 옮길 수 있습니다. 토이 2는 이미 numpy로 같은 식을 검증했습니다(`03` §5).
- **소형 과제 검증 (조건부):** MuJoCo 기반 짧은 과제(예: Hopper, 또는 `gymnasium` 제어 과제)에서 FPO++와 가우시안 PPO를 비교하는 것은 CPU에서 가능합니다. 다만 `mujoco`와 `gymnasium` 설치가 필요하고, 사용자 승인 전에는 시작하지 않습니다.
- **보행 재현 (불가):** Go2, Spot, H1, G1은 IsaacLab과 GPU가 필요합니다. 4096 환경 학습은 맥에서 현실적이지 않습니다. 논문 설정으로는 CPU에서 돌릴 수 없습니다.
- **모션 트래킹과 sim-to-real (불가):** GPU, LAFAN 데이터, 실물 G1·T1이 필요합니다.
- **조작 미세조정 (불가):** robosuite와 DexMimicGen 설치, 베이스 정책 접근, 시각 인코더 학습이 필요하고, 원 실험은 L40S급 GPU 1장에서 과제당 수 시간 이상입니다.

**노력 추정 (추정):** 알고리즘 재구현 3–5일, 소형 MuJoCo 비교 실험 추가 2–4일, 논문 규모 재현 불가.

## 5. 라이선스와 재배포

- **논문:** arXiv abs 페이지에서 확인(2026-11-15). 라이선스 링크는 `creativecommons.org/licenses/by/4.0/`입니다(CC BY 4.0). PDF 본문에는 ©, IEEE, All rights reserved 같은 문구가 없고 arXiv 스탬프만 있습니다.
- **그림 재사용:** CC BY 4.0이면 출처를 밝혀 쓸 수 있습니다. 그래도 이번 판에서는 논문 그림을 넣지 않았습니다. 그림 1과 3은 래스터 이미지(로봇 사진과 렌더 화면이 섞임)이고, 그림 A.2 등 조작 장면도 렌더 화면입니다. 그림 2, 4, 5, 6은 `pdfimages` 목록에 래스터 이미지가 없어 벡터 그래프로 보이지만, 쪽별로 잘라 쓰려면 별도 작업이 필요합니다. 게시본과 리포트는 우리가 그린 그래프만 씁니다.
- **공식 코드 (`amazon-far/fpo-control`):** GitHub `licenseInfo: null`, 최상위 LICENSE 파일 없음. README는 하위 구성요소의 라이선스만 나열합니다. 따라서 **저장소 코드의 재배포 조건이 명시되어 있지 않습니다.** 코드를 복사해 공개하지 않고, 읽은 내용만 요약했습니다.
- **구성요소:** IsaacLab BSD-3(README), rsl_rl BSD-3(README, 적응 코드), LeRobot Apache-2.0(벤더링, `thirdparty/lerobot/LICENSE`), robosuite MIT(벤더링), DPPO MIT(`src/vit.py` 일부), DexMimicGen NVIDIA 소스 라이선스(비상업, 설치 시 클론). 구성요소 라이선스는 README 기준이며 원본 파일은 전부 확인하지 않았습니다.

## 6. 저자에게 물어볼 질문 (보내지 않음)

1. 식 (7)은 제곱 L2인데 기본 설정은 `sqrt`입니다. 본문 실험은 어느 쪽을 썼습니까?
2. 표 A.2(모션 트래킹)의 학습률·에포크·미니배치·적분 단계는 실제 실험과 다릅니까? 실험에 쓴 설정 파일을 공개할 수 있습니까?
3. 적응 학습률(schedule adaptive)은 G1 sim-to-real 정책에 쓰였습니까? 쓰였다면 본문의 FPO++ 설명에 포함되어야 합니다.
4. ASPO 목적 수정(커밋 `8f460e4`) 이전 결과와 이후 결과의 차이는 얼마입니까? 조작 ASPO 실험(C.6–C.7)은 수정 후 코드로 다시 돌렸습니까?
5. 베이스 정책 성공률의 평가 횟수(1,000 대 다른 값)와 그림 A.9 값의 계산 방식은 무엇입니까?
6. 그림 5의 FPO와 FPO++는 Huber, 클램프, 기울기 노름, 클리핑이 다릅니다. 같은 설정에서 손실 형식만 바꾼 비교가 있습니까?
7. 기울기 코사인 유사도(D.1)의 계산 코드를 공개할 수 있습니까?

## 7. 메모

- 공식 저장소에는 vendored `lerobot`과 `robosuite`가 포함되어 있어 추적 파일이 약 1,900개입니다(`git ls-files`). 재배포 시 용량과 라이선스 검토가 필요합니다.
- 저장소 `.gitmodules`는 IsaacLab과 whole_body_tracking을 서브모듈로 가리킵니다. 얕은 복제로는 서브모듈 내용을 받지 않았습니다.
