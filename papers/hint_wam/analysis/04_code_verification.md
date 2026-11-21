# HINT-WAM 공식 코드 대조 (arXiv:2611.24789 v1)

대상 저장소: `github.com/tuandattt/hintwam_code`, 커밋 `6a49a33dc389fb933d62f178a6b1f13e63ff3506`(2026-11-19), 얕은 복제 `repo/`(정적 읽기만). 저장소는 생성 2026-11-16, 별 4개, 포크 없음입니다. 실행, import, 설치는 하지 않았습니다. 이 저장소는 Fast-WAM 공식 코드(MIT)를 기반으로 하고, 그 위에 hindsight 교사, 적응기, 에너지 점수, 학습 설정을 더했습니다.

## 0. 판정 요약

- **코드와 논문의 핵심 수식은 일치합니다.** 에너지 점수(식 8, 비제곱 노름), 구현된 학생 샘플(식 3의 implied action), 교사 ODE 표본, BC 앵커, 게이트 적응기, 힌드사이트 인코더의 구조가 맞습니다.
- **다른 점이 있습니다(D1–D5).** Stage A의 유효 배치(256 대 192), 게이트 초기값 문서화 오류, 누락된 토이 스크립트, 절제에서 감독 점 개수 차이(K=3 대 K=1), 학습 데이터 설명(3,921 대 10% 분리)입니다.
- **재현에 필요한 것 중 저장소에 없는 것이 있습니다.** 평가기(Fast-WAM 저장소 의존), 연속 학습의 재생 설정, RoboTwin 25과제 선정 스크립트, 토이 스크립트입니다.
- **구현 가능성 판정은 §4에 있습니다.** 전체 재현은 GPU가 필요해 이 맥에서는 불가능합니다.

## 1. 라이선스와 출처

| 항목 | 확인 결과 |
|---|---|
| 코드 LICENSE 파일 | MIT, "Copyright (c) 2026 The FastWAM Authors" (저장소 루트 `LICENSE`) |
| GitHub API 라이선스 필드 | `NOASSERTION` (저작권 문구가 표준 MIT와 달라 자동 인식되지 않은 것으로 보임) |
| Hugging Face 모델 `tuandattttt/HINT-WAM` | 태그 `license:mit`, 마지막 수정 2026-11-19 (메타데이터만 확인, 가중치는 받지 않음) |
| 논문 라이선스 | arXiv abs 페이지의 "view license" 링크가 `creativecommons.org/licenses/by/4.0/`를 가리킴 (CC BY 4.0) |
| 원문 PDF 저작권 표시 | ©, copyright, All rights reserved, 출판사·기관 고지 없음 (`pdftotext` 전체 확인) |
| 모델 가중치 | 받지 않음. Hugging Face 페이지에 조회 시점에 36회 다운로드 |
| 프로젝트 페이지 | `tuandattt.github.io/hintwam/`. 학회를 명시하지 않고 BibTeX는 `journal={arXiv preprint}` |

## 2. 일치하는 항목

| 항목 | 논문 | 코드 (파일:설정) | 판정 |
|---|---|---|---|
| 에너지 점수 식 | 식 (8): (1/M) Σ‖â_m − a_T‖₂ − (1/2M²) Σ‖â_m − â_m'‖₂, 청크 전체 평탄화 | `energy_score_loss`: `chunk_norm`(비제곱 L2), `0.5 × diversity` | 일치 (D6 참조) |
| 학생 표본 | 식 (3) implied action, 한 번의 전방 계산 | `implied_action`: `state − sigma × velocity` | 일치 |
| 교사 표본 | 프레시 노이즈에서 교사 ODE 10스텝의 최종 상태 | `teacher_traj.final`, `torch.randn_like` | 일치 |
| 상태 선택 | Δ²로 가중, top-K, 정규화 | `select_sigma_points`(`topk`, `renormalize=True`) | 일치 |
| 샘플링 스케줄 | 10스텝, shift 5 | `distill_inference_steps: 10`, `_phi(u, shift)` | 일치 |
| 학생 표본 수 | M = 4 | `energy_samples: 4` | 일치 |
| 감독 점 개수 (energy) | K = 1 | `num_sigma_points: 1` | 일치 |
| 손실 가중치 | λ_dist = λ_BC = 1 | `lambda_flow: 1.0`, `lambda_bc: 1.0` | 일치 |
| 학습률, 스케줄 | 1e-4 (SFT, B), 3e-4 (A), 코사인, 5% 워밍업 | 설정 파일, `trainer.py`의 `warmup_steps = 5%` | 일치 |
| 옵티마이저 | AdamW, β = (0.9, 0.95), wd 0.01, 기울기 클리핑 1.0 | `trainer.py` `betas=(0.9, 0.95)`, `train.yaml` `max_grad_norm: 1.0`, 설정 `weight_decay: 1e-2` | 일치 |
| 스텝 수 | SFT 5,000 (3,000 사용), A 2,000, B 2,000 | `libero90_sft.yaml`, `libero90_opsd_teacher.yaml`, `libero90_opsd_distill_energy.yaml` | 일치 |
| 게이트 적응기 | tanh(α), α = 1, W_up = 0 | `alpha_init: 1.0`, `zero_init_up: true`, `GatedCrossAttnAdapter` | 일치 (D2 참조) |
| 적응기 위치 | 레이어 {4, 9, 14, 19, 24, 29}, 병목 512 | `adapter_layers: [4, 9, 14, 19, 24, 29]`, `adapter_bottleneck: 512` | 일치 |
| 힌드사이트 인코더 | 첫 잠재 프레임 제외, 공간 평균, 선형 48→512, LN, 학습 쿼리 8개 | `FutureLatentProvider`: `skip_first_frame`, `mean(dim=(3,4))`, `proj`, `queries` (8), 잠재 채널 48 (`z_dim=48`) | 일치 |
| 학생의 입력 | 현재 관측만 (φ 없음) | 학생 `ctx.velocity(..., phi=None)` | 일치 |
| 교사 기준 | 동결된 SFT 백본과 동결된 비디오 전문가 | `frozen_experts: [video]` (Stage B), `[video, action]` (Stage A) | 일치 |
| 셔플 대조 | 다른 에피소드의 미래 | `shuffle_c` (`torch.roll` on latents), 설정 `libero90_opsd_teacher_shuffled.yaml` | 일치 |
| 온폴리시 상태 | 학생의 샘플러 상태 | `state_source: student`, `record_student_trajectory` | 일치 |
| 오프폴리시 대조 | 노이즈 낀 시연 행동 | `state_source: demo` | 일치 |
| 행동 복제 앵커 | 식 (1)을 학생에 적용 | `training_loss(phi=None, action_only=True)` | 일치 |
| 제곱오차 대조 | 교사 표본 하나의 MSE | `loss_type: mse` → `flow_distillation_loss` | 일치 |
| 행동 제곱오차 대조 | 행동 공간 MSE | `loss_type: action_mse` → `action_mse_loss` | 일치 |

## 3. 차이와 누락 (D 항목)

| 번호 | 논문 | 저장소 | 영향 |
|---|---|---|---|
| D1 | Stage A의 유효 배치 192 (표 3 "All runs share ... effective batch size of 192", 본문 4.1절) | `libero90_opsd_teacher.yaml`: batch 8, 누적 4, 8 GPU → 유효 배치 256. 저장소 README 표도 "256 (8 × 8 GPUs × 4)"라고 적음 | 교사 학습 설정이 본문과 다름. 교사 품질에 미치는 영향은 측정하지 않음 |
| D2 | 게이트 α = 1, W_up = 0 (본문 4.1절). "in the style of Flamingo" (식 5 근처) | 설정 α = 1.0, zero_init_up. 그러나 저장소 README(Stage A 확인 항목), `engine.py` `mean_gate` 독스트링, Stage A 설정 주석은 "phi_gate는 0에서 시작"이라고 씀. 실제 초기 `mean_gate` = tanh(1) ≈ 0.76 | 출력은 W_up = 0 때문에 초기에 0이라 기능 문제는 없음. 문서 오류이며, 학습 로그의 게이트 값을 읽는 사람을 혼동시킴. "Flamingo 스타일"이라는 말도 α = 0 원래 설정과 다름 |
| D3 | 보충 J의 토이 실험 | `losses.py` 독스트링이 `scripts/toy_energy_distillation.py`를 가리키지만 저장소에 파일이 없음 | 보충 J의 토이는 이 저장소로 재현할 수 없음. 본 분석은 독자 구현으로 확인함(03 V1) |
| D4 | 그림 4 절제: "same teacher, on-policy states, anchor, M, and training budget"(본문 4.5절) | velocity matching 설정 `libero90_opsd_distill_mse.yaml`은 `num_sigma_points: 3`(K = 3), 에너지 점수는 K = 1. 보충 I는 이를 인정함 | 그림 4의 목적함수 비교에 감독 점 개수가 섞여 있음. 본문 설명에는 이 차이가 없음 (03 K5) |
| D5 | 본문 4.1절 "We post-train on its 3,921 demonstrations" | `configs/data/libero90.yaml`: `val_set_proportion: 0.1`, `is_training_set: true` → 90%만 학습(약 3,528 에피소드). 저장소 README도 "10% ... held out automatically" | 본문 문장이 부정확. 보충 G와 코드는 일치 (03 K3) |
| D6 | 식 (8)에 노름 표기 ‖·‖₂ | `energy_score_loss`: `norm(dim=-1)` (비제곱) | 일치. 제곱 노름이었다면 적합 규칙이 깨짐 (03 §5). 이 점을 확인함 |
| D7 | 힌드사이트 개수, 적응기 파라미터 약 13M (0.2%) | 설정에서 계산: 인코더 + 적응기. 정확한 파라미터 수는 실행하지 않아 확인하지 않음 | 미확인 (세부 수치) |
| D8 | 연속 학습에서 경험 재생 (5 demos per task, 19% 프레임, 모든 단계에 섞음) | 재생 관련 설정, 데이터 로더, 설정 파일이 저장소에 없음 (`grep -ri replay` 결과 없음) | 표 2의 "with replay" 행은 이 저장소만으로 재현할 수 없음 |
| D9 | BC 앵커와 증류 항을 λ = 1로 합침. 보충 I는 "loss weights were not tuned" | 증류 항은 청크 전체 L2 노름(척도 약 √(H·D) × 오차), BC 항은 원소별 평균 MSE. 두 항의 척도가 다름 | 앵커의 상대 세기가 불명확함. 기울기 크기를 측정하지 않아 판정하지 않음 (추정) |
| D10 | 평가 프로토콜 (보충 G, RoboTwin 25과제 선정) | 평가기는 Fast-WAM 저장소 의존(README §5). 25과제 선정 스크립트는 없음 | 평가 수치는 이 저장소만으로 재현할 수 없음 |
| D11 | 학습 설정 전부 논문과 동일하다는 주장 (보충 I) | 시드 42 고정, 단일 실행. 시드 반복 스크립트 없음 | 시드 변동 미측정 (03 K8) |
| D12 | 논문에 없는 기능 | `src/fastwam/privileged/`(RoboTwin 접촉 특징 `c`, 촉각 `univtac`), `probe.py`, `eval_sampling.py` | 본 논문 실험과의 관계는 확인하지 못함. 공개 코드의 확장으로 보임 |

## 4. 구현 가능성 판정

**판정: 전체 재현은 이 맥에서 불가능합니다. 손실과 학습 구조의 축소 구현은 조건부로 가능합니다. 통계, 비용, 토이 검증은 완료했습니다.**

### 4.1 이 맥(Apple M4 Pro, 약 24 GiB 통합 메모리, MPS, CUDA 없음)에서 할 수 있는 것

- **완료:** 표 재계산과 검정, 비용 회계, 샘플러 스케줄 점검, 에너지 점수의 적합성 닫힌 형식 검사, 토이 V1 (CPU, 약 10초).
- **완료:** 공식 코드의 정적 대조 (이 문서 §2, §3).
- **조건부 (설치와 데이터 승인 필요):** 수 M 파라미터의 flow-matching 정책을 MPS에서 학습해 LIBERO 한 스위트에서 SFT, BC, 에너지 점수를 비교. LIBERO 설치, mujoco, 데이터 다운로드가 필요하고, 학습 속도는 측정하지 않았습니다. 노력은 1–2주(추정).
- **불가:** CUDA 전용 설정 그대로 실행. 공식 코드는 `torch==2.7.1+cu128`, CUDA 12.8, 8 × A5000을 전제합니다(README §1).

### 4.2 GPU, 데이터, 시뮬레이터가 필요한 것

| 필요 자원 | 근거 | 이 맥에서 |
|---|---|---|
| 8 × 24 GB GPU, 단계 A+B+SFT 3k | 보충 표 4: SFT 3k 33.5 h, A 4.4 h, B 33.7 h (8 GPU, 합계 71.6 h, 572 GPU-h) | 불가 |
| 5B 비디오 전문가 (동결, 로드) + 1B 행동 전문가 | 보충 A | 가중치 크기 미확인 (수 GB 이상으로 추정). 다운로드와 MPS 포팅 필요 |
| LIBERO-90 LeRobot 데이터 (3,921 에피소드, 569K 프레임) | README §3 | 용량 미확인, 다운로드 승인 필요 |
| RoboTwin 2.0 데이터 (분할 tar) | README §3 | 용량 미확인 |
| Fast-WAM 평가기 (LIBERO, RoboTwin 시뮬레이터) | README §5 | 시뮬레이터 설치 필요 (CUDA 또는 Vulkan 가능성) |
| Fast-WAM 체크포인트 (`libero_uncond_2cam224.pt` 등) | README §2 | 다운로드 승인 필요 |

### 4.3 한 줄 요약

- 재현에 가장 큰 장벽은 **GPU 시간(8 GPU × 약 72 h, 평가 제외)**과 **평가용 시뮬레이터와 체크포인트**입니다.
- 손실의 핵심 아이디어(에너지 점수와 온폴리시 상태)는 축소된 설정에서 검증할 수 있습니다. 토이에서는 이미 재현했습니다.
- 공식 수치를 맞추는 재현은 저장소에 없는 평가 코드(D10)와 재생 설정(D8) 때문에 저장소만으로는 완결되지 않습니다.

## 5. 재현 전 확인할 사항

1. D1: Stage A 배치 256이 실제 실행 설정인지 저자에게 확인 (본문은 192).
2. D4와 K5: velocity matching을 K = 1로 맞춘 대조 실행이 있는지 확인.
3. K1: 과제별 성공 횟수 원자료가 있는지 확인 (기준선 SFT 78.88 등).
4. D8: 재생 설정 공개 여부.
5. D10: 평가기 버전 (Fast-WAM 커밋) 확인.
