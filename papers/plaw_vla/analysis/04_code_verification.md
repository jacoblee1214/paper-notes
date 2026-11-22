# 04. 공식 코드와 논문 대조, 구현 가능성

**대상:** https://github.com/RainyRobo/PLaW-VLA, 커밋 `5fb8f32cba29e47f6838ebf572e7bf036282a68b` (2026-10-13, 커밋 1개, "Initial commit"), Apache-2.0. 얕은 복제를 `repo/`에 두고 **정적으로만** 읽었습니다(실행, import, 설치 없음). 파이썬 파일 약 42,000줄, 7.2 MB. 이슈 1개(저장소 공개 준비 작업의 풀리퀘스트 설명, 닫힘), 브랜치 1개. 그 설명은 "formal weights remain TODO", RoboTwin과 LIBERO-Plus는 선택 벤치마크라고 밝히고, 단위 테스트 188개 통과와 짧은 3단계 실행 점검을 기록합니다(논문 수치 재현이 아님).

**읽은 파일:** `README.md`, `docs/{pretraining,data,benchmarks,licenses}.md`, `src/plawvla/training/config.py`(전체 구성), `models/{pi0_config,vjepa2,gemma}.py`, `models_pytorch/{pi0_pytorch,world_model_pytorch}.py`, `transforms.py`, `scripts/train/*.sh`, `examples/{libero,libero_plus,robotwin}/*`. 다른 파일(데이터 변환기, 서빙, 정책 어댑터)은 구조만 보았습니다.

## 1. 공개 범위

| 항목 | 공개 여부 |
|---|---|
| 모델(π0.5 + 월드 모델 전문가), 3단계 학습 스크립트, 설정 | 있음 (`stage1_world_model_pretraining`, `stage2_pretraining`, `stage3_finetuning_libero`, `stage3_finetuning_robotwin`) |
| LIBERO·RoboTwin·LIBERO-Plus 평가 클라이언트 | 있음 (서브모듈, LIBERO-Plus는 상위 라이선스 미선언) |
| 데이터 변환기 (InternData-A1, AgiBotWorld, RoboTwin, EgoDex, LIBERO) | 있음 |
| **PLaW-VLA 사전학습 체크포인트** | **없음** ("Release planned", 프로젝트 페이지 Model 버튼 비활성) |
| Stage I/II 변환 데이터와 정확한 혼합 목록 | 없음 (README가 "author-verified source manifest에 의존"이라 밝힘) |
| 표 4의 절제 변형 (Cosmos VAE, 예측 손실 제거, 미래 조건화 제거, 가까운/먼 미래 마스킹) | 없음 (코드에 해당 옵션·설정 없음) |
| 실로봇 Stage III 설정, 지연 측정 스크립트 | 없음 |
| README가 말하는 `auto` 경로 | 공개 π0.5로 시작해 **월드 모델을 새로 초기화**하고 Stage III만 돌림. README가 "paper 결과가 아니다"라고 명시 |

## 2. 논문과 일치한 항목 M1–M9

| # | 항목 | 논문 | 코드 |
|---|---|---|---|
| M1 | 3단계 하이퍼파라미터 (표 11) | 배치 512 / 256 / 256, LR 5e-5, 스텝 100K / 100K / 50K, 워밍업 0 / 0 / 10K, AdamW, 클립 1.0 | `_stage_train_config`: 동일. 스케줄은 코사인이지만 peak = decay = 5e-5라 상수와 같음 (Stage III는 `decay_steps = 1_000_000`) |
| M2 | 월드 모델 분기 드롭아웃 α | 0.3, 배치마다, m = 0이면 과거·미래 토큰과 L_wm 제거 | Stage II, III `wm_loss_dropout_alpha=0.3`. `_should_drop_wm_branch`가 `post_training` 단계에서만 뽑고 `_forward_action_only`로 분기 |
| M3 | 시간 분포와 목표 | τ ~ Beta(1.5, 1), 보간 a^τ = τ ε + (1−τ) a, 목표 ε − a | `sample_beta(1.5, 1.0)` × 0.999 + 0.001, `x_t = t·noise + (1−t)·actions`, `u_t = noise − actions` |
| M4 | 예측 손실 | 고정 V-JEPA 2 잠재, 정지 기울기, MSE | `wm_target = fut_target_embeddings.detach()`, `F.mse_loss`, 인코더 `.eval()` 고정 (`VJepa2Adapter.train`) |
| M5 | 학습 대상 | Stage I: 월드 모델만. Stage II–III: 전체 MoT (V-JEPA 제외) | `set_training_stage`: `wm_alignment`는 월드 모델 전문가, 예측 헤드, 슬롯 빌더만, `post_training`은 인코더와 LM 헤드만 얼림 |
| M6 | 월드 모델 전문가 | Gemma 300M, 처음부터 학습 | `world_model_expert_variant = gemma_300m` (311M), 슬롯 임베딩·위치 임베딩·예측 헤드 새로 초기화 |
| M7 | 구조적 인과 어텐션 | P → H → F → A (그림 3) | `make_att_2d_masks`가 블록 하삼각을 만듦 (`analysis/toy_attention_mask.py`로 확인). F 블록은 첫 통과 예측값을 되먹인 토큰 |
| M8 | 10 Hz 리샘플링, 시간 오프셋 | 10 Hz, 6프레임 단위 | `action_time_step_s = 0.1`, 시간 오프셋 −1.0 … +1.2 s, 간격 0.2 s (5 Hz 프레임) |
| M9 | RoboTwin 평가 | 청정(Easy) / 무작위(Hard), 정수 % | 기본 `--test-num 100`, `demo_clean` / `demo_randomized`, 전문가 실패 시드는 건너뜀(`expert_check`) |

미래 목표는 **미래 6프레임만 따로** V-JEPA 2에 넣어 만듭니다(과거 클립과 합쳐 인코딩하지 않음, `build_training_inputs`: `hist = adapter(images_history)`, `fut = adapter(images_future)`). 논문 "frozen encoder로 미래 관측에서 목표를 뽑는다"와 같은 서술이지만, 목표가 과거 문맥 없이 인코딩된다는 점은 논문에 없습니다(비교: V-JEPA 2의 일반 사용은 연속 클립 한 번).

## 3. 차이와 논문에 없는 설정 D1–D17

| # | 항목 | 논문 | 코드 | 영향 |
|---|---|---|---|---|
| D1 | 행동 청크, 재계획 | 청크 길이 H_a 미기재 | `action_horizon = 10`, 평가 `replan_steps = 5` (LIBERO, RoboTwin 모두 5 웨이포인트) | 재현 필수 값. π0.5 기본은 50 |
| D2 | 월드 모델 입력 길이 | H_o 미기재, "관측 이력" | 과거 6프레임(−1.0 … 0.0 s, 5 Hz) = 3 튜블릿 = **768 토큰**. 월드 모델에는 **정면(head) 카메라 한 대**만 들어감. 손목 카메라는 VLM만 | 추론에 1초 분량 프레임 버퍼 필요 |
| D3 | 예측 손실 가중 λ_wm | 식 5에 있으나 값 없음 | `wm_loss_weight = 0.1`, `Pi0Config`에 필드가 없어 **코드에 고정** | 재현 필수 값 |
| D4 | 미래 단위 수 K | K = 2 (가까운, 먼 미래), "6프레임 = 현재 + 미래 2" (B.4) | 미래 6프레임 = **미래 3 튜블릿 = 768 슬롯 토큰** (`wm_slot_max_len = 768`, 주석 "six future frames need 768 slots"). 과거 6프레임에 현재가 포함됨 | 논문 B.4의 "먼 미래/가까운 미래" 정의와 코드 구성이 다름. 어느 쪽이 보고 결과의 구성인지 불명 |
| D5 | 행동 손실이 예측값으로 역전파되는가 | 식 3에서 **목표**만 정지 기울기 | 학습 때 행동 전문가가 읽는 토큰은 `wm_pred`를 **detach 없이** 월드 모델 전문가에 다시 넣은 출력. 행동 손실 기울기가 월드 모델과 예측 헤드로 흐름 (`_forward_action_and_wm`). 월드 모델 전문가는 한 제어 스텝에 **두 번** 통과(슬롯 예측 → [과거, 예측] 재인코딩) | 논문은 이 두 가지를 쓰지 않음. 추론 비용 계산에 영향 |
| D6 | V-JEPA 2 크기 | "V-JEPA 2"만 | 기본 `vitl-256` (ViT-L, `facebook/vjepa2-vitl-fpc64-256`). ViT-H/g는 입력 투영기와 함께 선택 가능 | +0.9B(논문) 대 기본 설정 약 0.62B (`03` V8). ViT-H로 세면 0.95B |
| D7 | 고유수용 상태 입력 | "VLM이 관찰과 지시를 접지"(프로젝트 페이지는 proprioception도 언급) | `discrete_state_input = False`와 π0.5 경로(`not self.pi05`일 때만 상태 투영)로 **정책이 로봇 상태를 입력으로 받지 않음** (openpi의 `pi05_libero`와 같은 설정) | 논문 본문에 없는 설계. 실로봇 이식 시 영향 |
| D8 | Stage I의 입력 | "과거 시각 이력과 정지 기울기 미래 목표만, 행동 라벨 없이" (D.1) | `_forward_world_only`는 VLM 접두부(이미지·언어 프롬프트)를 얼린 PaliGemma에 통과시킨 뒤 월드 모델이 그 접두부도 읽음 (`prompt_from_task=True`) | 월드 모델이 언어 지시도 조건으로 사용 |
| D9 | 분기 드롭아웃의 분산 처리 | 배치마다 Bernoulli | `shared_bernoulli`: 한 번 뽑아 모든 GPU가 같은 분기를 씀 | 논문과 같은 의도. 구현 세부 |
| D10 | 데이터 혼합 가중치 | 프레임 수만 (표 12: 377.7M, 56.9M, 6.1M, 0.6M, 77.1M) | Stage I: InternData-A1 0.20, AgiBotWorld 0.30, RoboTwin 0.15, LIBERO 0.08, EgoDex 0.15. Stage II: 0.20, 0.30, 0.15, 0.10 (합이 1이 아니라 샘플러가 정규화) | 프레임 수와 샘플링 비율이 크게 다름(InternData는 프레임이 가장 많지만 20%) |
| D11 | LIBERO 설정 | 학습 데이터 1,693개 시연 | 공개 변환본 `RainyBot/libero_v3_eef`를 받아 씀(크기·시연 수 확인 안 함) | 논문 수치와의 동일성 미확인 |
| D12 | 평가 시드, 체크포인트 | 3 평가 시드, 30K 체크포인트 보고 | 클라이언트 `--seed 42` 한 번에 한 시드, 체크포인트 5K마다 저장 | 3 시드와 체크포인트 선택은 사용자가 수동으로 함. 선택 기준 없음 |
| D13 | 표 4 절제 | Cosmos VAE, w/o prediction loss, w/o future conditioning, 마스킹 | 코드·설정 없음 | 표 4 재현 불가 |
| D14 | 사전학습 체크포인트와 데이터 | 3단계 사전학습 사용 | 공개 안 됨. 공개 π0.5로 시작하는 `auto` 경로는 결과가 논문과 다르다고 README가 밝힘 | 논문 수치 재현 경로 없음 |
| D15 | LIBERO 실행 방식 | 식 6: 절대 목표를 현재 상태 기준 상대값으로 학습 | 클라이언트는 서버가 돌려준 절대 EEF 목표 웨이포인트를 20 Hz로 보간해 OSC 명령으로 변환 | 논문에 없는 실행 어댑터. LIBERO 결과가 이 어댑터에 의존 |
| D16 | 지연 측정 | 그림 5 "단일 H100 지연" | 측정 스크립트 없음 | 1/19 주장 재현 불가 |
| D17 | 라이선스 | | 코드 Apache-2.0(NOTICE, MIT 일부), Gemma 약관 사본, **LIBERO-Plus 서브모듈은 상위 라이선스 미선언**, InternData-A1·AgiBotWorld CC BY-NC-SA, EgoDex CC BY-NC-ND | 모델·데이터를 쓸 때 비상업 조건 확인 필요 |

## 4. 구현할 수 있는지 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**공식 코드와 논문 수치의 재현은 불가능합니다. 구조의 축소판 구현은 가능합니다.**

| 수준 | 판정 | 근거 |
|---|---|---|
| 공식 코드 설치·실행 | **불가** | README가 Linux x86_64 + NVIDIA GPU(CUDA 12.8), Python 3.12, `uv sync --frozen`을 전제하고, `VJepa2Adapter.__init__`가 CUDA 장치가 아니면 `RuntimeError`를 냄. MPS 대체 경로 없음. 새 설치와 공식 코드 실행은 원칙상 보류 대상 |
| Stage III LIBERO 미세조정 | **불가** | README 예시가 `NUM_GPUS=8`, 배치 256, 50K 스텝. 사전학습 체크포인트가 없어 `auto`는 월드 모델을 새로 학습한 결과(논문 수치 아님) |
| Stage I–II 사전학습 | **불가** | 수억 프레임(InternData-A1 377.7M 등) 데이터, 데이터 수 TB 규모(크기 미측정), GPU 시간 미공개. 데이터 라이선스가 비상업 |
| 공식 체크포인트 추론 | **불가** | 체크포인트가 없음 |
| 논문 표 재계산, 정적 코드 대조 | **가능, 완료** | `analysis/recompute.py`, 이 문서 |
| 어텐션·비용 토이 | **가능, 완료** | `analysis/toy_attention_mask.py`, `cost_accounting.py` (수 초) |
| **소형 구조 재구현 (MoT 세 전문가, 블록 인과, flow matching, 예측 지향 대 재구성 지향 목표 비교)** | **가능** | 작은 인코더 + 합성 또는 소형 동역학 데이터로 MPS/CPU 가능. 노력 추정 2–3주(근거 없는 추정, 측정 전) |
| LIBERO 시뮬 평가를 CPU로 | 조건부 | 새 파이썬 3.8 환경과 MuJoCo/robosuite 설치가 필요. 훈련된 정책이 없어 의미가 작음 |
| 실로봇 | **불가** | 양팔 바퀴 휴머노이드와 시연 400개 × 4과제 필요 |

노력 추정: 소형 재구현 2–3주(추정, 측정 전). 공식 코드로 논문 수치를 재현하려면 8 GPU 이상, 3단계 사전학습 데이터 수집과 변환(수 주), 체크포인트가 공개되기 전에는 경로가 없습니다.

## 5. 정리

- 코드는 논문 구조(3전문가 MoT, 정지 기울기 목표, 분기 드롭아웃, 3단계 하이퍼파라미터)와 일치하고, 비교 가능한 설정은 대체로 맞습니다(M1–M9).
- **재현에 필요하지만 논문에 없는 값**이 코드에 고정돼 있습니다: 행동 청크 10, 재계획 5(D1), λ_wm = 0.1(D3), 과거 6프레임·단일 카메라(D2), V-JEPA 2 ViT-L(D6), 상태 입력 없음(D7), 혼합 가중치(D10).
- **논문과 코드가 다른 지점**은 미래 단위 수(D4), 행동 손실의 역전파 경로와 이중 통과(D5)입니다.
- 사전학습 체크포인트, 표 4 절제 코드, 지연 측정이 없어 **논문의 핵심 수치(표 1–4, 그림 5)는 공개 자료만으로 재현 경로가 없습니다**.
