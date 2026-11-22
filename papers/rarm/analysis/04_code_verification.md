# 04. 논문 ↔ 공식 코드 정적 대조와 구현 가능성

공식 저장소 `pengzhi1998/rarm`을 `repo/`에 얕게 clone했습니다. `main`은 `55ef3fe`(2026-11-06 "Use generic epoch 45 reward checkpoint"), LIBERO 정책 학습이 있는 `policy_training_RARM` 가지는 `c51695a`(같은 날)이고 `repo/_policy_training_RARM/`에 `git archive`로 풀어 읽었습니다. **실행, import, 설치, 가중치와 데이터 다운로드를 하지 않았습니다.** 읽은 파일: `Reference-Anchored_RM/`의 `models/*.py`, `utils/loss.py`, `utils/dataset.py`, `scripts/train.py`, `scripts/reward_model.py`, `scripts/video_clip_comp.py`, 정책 가지의 `reward_model_baseline/Libero/` 전체(`drqv2_runner.py`, `task_training.py`, `task_eval.py`, `RewardModels/*.py`, `envs/libero_env_wrapper.py`, `utils/LIBERODemoRegister.py`, `constants/*.py`, `agents/drqv2/drqv2.py`), 두 가지의 README, `PROVENANCE.md`, `THIRD_PARTY_NOTICES.md`.

## 1. 대조 범위

- **공개된 것:** (1) 보상 모델 구조·대조 학습·유사도 추론(main), (2) LIBERO-10 DrQ-v2 학습과 평가, RARM 보상(진행 게이트 포함)(정책 가지), (3) 체크포인트 `checkpoint_epoch_45_generic.pt`(HF, 233 MB).
- **공개되지 않은 것(D15):** MetaWorld 실험, 모든 기준선(GVL, RoboDopamine, RoboMeter, RoboCLIP, TemporalOT), 실로봇 π0 + DSRL 코드, 표·그림 생성, 학습 영상 목록과 전처리, 정적 영상 필터(IoU).
- **README의 자기 고지:** 정책 가지 README는 "원래 LIBERO 설정은 epoch-20 체크포인트를 썼고 generic epoch-45로는 결과를 다시 확인하지 않았다"고 씁니다. `PROVENANCE.md`는 코드가 2026-01 개발 저장소의 두 커밋에서 잘라 낸 것이고 "원 학습 실행과 설정은 독립적으로 확립되지 않았다"고, 후속 음성 샘플링 변경(정적·떨리는 클립)은 포함하지 않았다고 씁니다.
- **라이선스:** RARM 고유 코드에는 라이선스 파일이 없습니다(PROVENANCE는 "원 트리에 루트 라이선스가 없어 만들지 않았다"). 체크포인트는 DINOv3 라이선스로 표기됩니다. 코드의 재사용 권리는 불명확합니다.

## 2. 대조표

| 논문 항목 | 코드 위치 | 판정 |
|---|---|---|
| 식 (1) 소프트 최근접 이웃 손실, τ = 0.1 | `utils/loss.py` `SoftNearestNeighborsLoss`(logsumexp 형태), `scripts/train.py` `temperature=0.1` | 일치 (D13) |
| 음성 세 가지(같은 영상 떨어진 구간, 시간 역전, 다른 영상) | `utils/dataset.py` `first_source_negatives`, `second_source_negatives`(역방향 정렬 표본), 배치 내 다른 앵커 | 일치. 샘플링 세부 D9–D11 |
| T = 5프레임 클립 | `num_frames=5`(학습은 5 고정) | 일치 |
| DINOv3 + LoRA | `models/vit_dinov3.py`(ViT-B/16, 커밋 고정), `vit_lora.py`(r = 16, α = 32, qkv·proj, dropout 0.1) | 일치. 크기·LoRA 설정은 논문에 없음 (D12) |
| 공간 풀링 + 시간 RoPE + 프레임 간 자기 어텐션 | `models/temporal_processor.py`(24x24 → 8x8 평균 풀링, RoPE, 어텐션) | 일치 |
| 교차 어텐션(X 질의, Y 키·값) + 가벼운 헤드 | `cross_attn.py`(1블록, 잔차 없음, 512로 사영), `comp_head.py`(2층 TransformerEncoder + MLP) | 일치 |
| 식 (2) argmax | `reward_model.py` `step()` 313–317행(`sims.argmax`) | 일치 |
| 식 (4) m_j = max_k g(r_j, r_k) | `_compute_threshold()` `per_clip_max` | 일치 (자기 일치 포함) |
| 식 (5) δ_j = m_j − Δ, Δ = 0.5 | `per_clip_thresholds = [ms - offset]`, `offset = 0.5` | 일치 (D4) |
| 식 (3) 진행값 갱신 + 보상 1{p_t > p_{t-1}} | `BinaryProgressRewardModel.py` `BinaryProgressClipRewardModel.step()` 57–86행 | 일치(고수위선 방식). 같은 파일의 기본 `ClipRewardModel.step()`은 다른 보상(D1) |
| 진행값 π_j ∈ [0, 1] | `demo_progress = (start + mid) / total_ref_frames * 10` (0–10) | 척도만 다름 (D3) |
| 보상 = RARM x 가중치 + 성공 5 − 스텝 0.5(F.1) | `drqv2_runner.py` 227행, `env_constants.py`(`BINARY_REWARD_SCALER = 5`, `STEP_COST = 0.5`) | 일치. 가중치 값은 논문에 없음, 코드 기본 1.0 (D2) |
| 중앙 프레임 귀속, 앞 두 샘플 확장(F.1) | `drqv2_runner.py` 162–167, 217–219행 | 일치. 보상 크기는 D2 |
| DrQ-v2 하이퍼파라미터(표 4) | `drqv2_runner.py` 시그니처 기본값 | 거의 일치 (D14) |
| 신경망(표 5), 관측(표 6) | `agents/drqv2/drqv2.py`, `replay: frame_stack * 6` | 일치 |
| 10^6 스텝, 시드 5, 10k마다 10롤아웃 평가 | `task_training.py` `--total_timesteps 1_000_000`, `eval_freq = 20_000`, `n_eval_episodes = 10` | 평가 주기 불일치 (D8) |
| LIBERO 초기 상태 | `libero_env_wrapper.py` `reset()` | **불일치 (D5)** |
| 데모를 활용한 학습 보조 | `use_demo_init`, 행동 재생 | 논문에 일부만 있음 (D6) |
| 학습 영상 필터(정적 제거) | 코드 없음(프레임 수 30 미만만 제외) | 미공개 (D10) |
| MetaWorld 4과제, 실로봇 4과제, 기준선, 표 1·2, 그림 3–5 | 없음 | 미공개 (D15) |

## 3. 차이와 빈칸

판정 기호: 불일치 = 논문과 코드가 다름, 빈칸 = 논문에 없음, 미공개 = 저장소에 없음. 영향은 결론 수준의 추정입니다.

**D1. 두 가지 보상 구현 (일치, 주의).** 정책 학습(`reward_type=1`)은 `BinaryProgressClipRewardModel`을 쓰고, 이것은 식 (3)과 같습니다: 최고 도달 진행(`max_reached_progress`)보다 높은 진행에 신뢰 일치가 있을 때 +1, 아니면 0(`BinaryProgressRewardModel.py` 57–86행). 반면 같은 클래스의 부모 `ClipRewardModel.step()`(`reward_model.py` 264–328행)은 **진행 변화 x 신뢰도**(`last_raw_delta * (max_sim − threshold) / 0.5`)를 보상으로 주고, 진행값을 최고치가 아니라 **마지막 일치**로 갱신하므로 뒤로 가는 일치에는 음의 보상이 나옵니다. 논문 식 (3)에 해당하는 것은 정책 학습 경로의 이진 변형뿐이고, README의 "policy branch adds the progress-gating logic"은 이 변형입니다. 영향: 이진 변형을 쓰는 한 논문과 일치. 부모 변형은 논문에 없는 별개 보상입니다.

**D2. 보상 크기와 배치 (빈칸, 영향 중간).** 논문 식 (3)은 +1입니다. 코드는 이벤트 하나를 (a) 클립 중앙 프레임으로 옮기고, (b) 앞의 두 샘플에도 1로 채우며(`drqv2_runner.py` 217–219행), (c) 이미지가 4스텝마다 있으므로 각 샘플의 값을 **4 스텝에 걸쳐 유지**해(`searchsorted` 방식 매핑) 스텝에 뿌립니다. 격리된 이벤트의 에피소드 보상 합은 12입니다(`03` §4.2의 재구현). 여기에 성공 5(한 스텝당)와 스텝 비용 −0.5가 더해집니다(`drqv2_runner.py` 227행). 논문 F.1은 "보상 모델 가중치"로 스케일한다고만 쓰고 값이 없으며, 코드 기본값은 `rm_coef = 1.0`입니다. 영향: 진행 보상 합이 성공 보상보다 1–2자릿수 크고(`03` §4.2), 논문의 "+1"과 크기 감각이 다릅니다. DrQ-v2 외 학습 결과에 이 스케일이 미치는 영향은 모릅니다(실행하지 않음).

**D3. 진행 척도 (불일치, 작음).** 논문은 π_j ∈ [0, 1]. 코드는 0–10 척도이고 `(start + mid_offset) / total_ref_frames * 10`이라서 첫 클립도 0이 아니고 마지막 클립도 10 미만입니다. `max_reached_progress ≥ max(demo_progress)`이면 이후 보상이 0으로 고정됩니다(`BinaryProgressRewardModel.py` 65행). 정규화(`/10`)는 로깅용 `image_progress`에서만 합니다. 영향: 거의 없음.

**D4. 게이트 상수와 절제 변형 (일치, 빈칸 하나).** Δ = 0.5와 `m_j = max_k g(r_j, r_k)`(자기 일치 포함)는 논문과 같습니다. `use_cross_attn=False` 절제는 유사도를 `−||x − y||₂`(임베딩 평균 L2 거리)로 바꾸는데 같은 `offset = 0.5`를 그대로 씁니다. 점수 척도가 다르므로(거리 단위) 이 절제에서 Δ의 의미가 달라집니다. 논문 그림 4가 이 코드로 만들었는지 확인할 수 없습니다. 죽은 코드: `_compute_threshold()`의 `max_sim − 1.0`은 반환만 되고 쓰이지 않습니다.

**D5. 평가와 초기 상태 (불일치, 영향 큼).** `libero_env_wrapper.py` `reset()`(254–271행): RSI(데모 중간 상태에서 시작)가 적용되지 않으면 **기준 시연의 첫 상태로 상태를 설정**한 뒤 관절 잡음(표준편차 0.0005 rad)과 물체 XY 잡음(0.0005 m = 0.5 mm)만 더합니다. 코드 주석이 이유를 설명합니다: "Without this, LIBERO's placement sampler places objects at random positions within the region (up to ±2.5cm), which causes RM early-clip mismatch during eval." 평가 환경(`use_eval_rsi=False`가 기본, `drqv2_runner.py`)도 같은 경로입니다. 따라서 이 코드가 내는 LIBERO 성공률은 **LIBERO의 표준 초기 분포가 아니라, 기준 시연의 한 초기 배치(±0.5 mm)**에서의 성공률입니다. 또한 `--seed`는 "RNG 시드이자 기준 시연 번호"입니다(`task_training.py` 도움말, `demo_seed=args.seed`): 시드를 바꾸면 기준 시연도 바뀌므로 표의 시드 변동에는 시연 선택 변동이 섞입니다. 논문 §4.1과 부록 A·F.1은 이 초기 상태 처리를 밝히지 않습니다. 영향: 초기 배치가 고정이면 정책이 한 배치에 과적합해도 성공률이 높게 나옵니다. 이것이 표 2 "개입 없음 95.6%"에 어떻게 반영되는지는 알 수 없습니다(그 실험 코드는 공개되지 않음).

**D6. 데모 상태·행동 사용 (빈칸, 영향 중간).** README의 학습 명령은 `--use_demo_init --use_demo_decay`입니다. 학습 중 에피소드의 `demo_init_prob`(시작 1.0, 환경 생성 시 기본값은 0.95)는 `1 − t/T`로 선형 감쇠하고(`drqv2_runner.py` 451행), RSI 시작 상태는 기준 시연 상태의 1–95% 구간에서 뽑으며 **50% 확률로 이후 70스텝의 데모 행동(+ 잡음 σ = 0.01)을 그대로 재생**합니다(`libero_env_wrapper.py` 158–166행, `env_constants.py`). 재생된 행동이 리플레이에도 저장됩니다(`executed_action`). 논문 F.1은 "중간 상태에서의 초기화"만 쓰고 데모 행동 재생과 확률 감쇠는 쓰지 않습니다. **기준선 구현이 공개되지 않아 기준선에도 같은 보조를 줬는지 알 수 없습니다.** 영향: 처음부터 학습이라는 설명과 달리 시연 정보가 상태·행동 수준에서 학습에 들어갑니다.

**D7. 체크포인트와 학습 설정 (빈칸, 영향 중간).** README: LIBERO 원 설정은 epoch-20 체크포인트, 공개본은 generic epoch-45이고 "결과를 다시 확인하지 않았다". `PROVENANCE.md`: 공개 체크포인트가 공개 학습 스크립트로 학습되었는지 확립되지 않았고, 이후 "static/jittery 음성 샘플링"은 공개 코드에 없습니다. 논문은 에폭, 배치, 학습률, 학습 영상 수를 쓰지 않습니다(코드 기본: 200 에폭, 배치 64, AdamW lr 1e-4, wd 0.01, 체크포인트 5 에폭마다 저장). 영향: 공개 가중치로 논문 표를 재현할 수 있다는 보장이 없습니다.

**D8. 평가 주기와 최고 정책 선택 (불일치, 작음).** 논문은 10k 스텝마다 평가. 코드 기본 `eval_freq = 20_000`(`drqv2_runner.py` 336행)이며 `task_training.py`가 이를 바꾸지 않습니다. 또 `drqv2_best.pt`는 **평가 성공률이 가장 높았던** 체크포인트를 저장하고(`best_success_rate`), `task_eval.py`는 기본으로 이 "best"를 읽습니다. 같은 10 에피소드 평가로 고른 정책을 다시 10–수 에피소드로 평가하면 낙관적입니다. 논문 표·곡선이 어느 경로로 만들어졌는지 알 수 없습니다.

**D9. 학습 클립과 추론 클립의 간격 (빈칸, 영향 불명).** 학습 샘플러(`dataset.py`): 앵커는 8–40프레임 구간에서 5프레임을 "25/50/25 규칙"(첫 프레임은 앞 25%, 가운데 3프레임은 중간 50%에서 무작위 비복원 추출·정렬, 마지막은 뒤 25%)으로 뽑습니다. 즉 프레임 간격이 불규칙하고 클립이 구간 길이만큼 걸쳐 있습니다. 추론(`reward_model.py`)은 **연속 5프레임**(`frame_stride=1`, 렌더 간격 4스텝)을 쓰고, 기준 클립은 `clip_stride=1`로 겹치게 만듭니다(M = 프레임 수 − 4). 학습 영상의 프레임 속도와 환경의 4 스텝 렌더 간격 사이의 대응이 논문에 없어, 학습 분포와 추론 분포의 시간 간격이 맞는지 알 수 없습니다.

**D10. 학습 데이터 처리 (미공개).** 논문 D.2: DAVIS와 MOSE(MOSEv2)에서 마스크 IoU 평균이 높은 정적 영상 제거. 코드에는 이 필터가 없고(프레임 30 미만만 제외, `dataset.py` 77행), 영상 수와 목록도 없으며 기본 경로가 `datasets/dataset_1500/...`인 것만 단서입니다. 영향: 학습 데이터를 재구성할 수 없습니다.

**D11. 증강과 반전 (불일치, 영향 작음 추정).** 논문 §3.1: "같은 영상에서 뽑은 모든 클립은 같은 좌우 반전을 공유". 코드: 앵커와 양성은 같은 반전이지만(`should_flip`), **음성(같은 영상의 다른 구간, 역방향 클립)은 각자 50% 확률로 독립 반전**(`dataset.py` 269, 279행). 그러면 반전이 어긋난 쌍은 항상 음성이라서 반전 일치가 약한 단서가 될 수 있습니다(우리 추정, 실험 없음). 크롭은 "무작위 리사이즈 크롭"이지만 배율이 0.9–1.0의 정사각형뿐이라 약합니다.

**D12. 구조 세부 (빈칸).** 논문에 크기가 없는 것: DINOv3 ViT-B/16, LoRA r = 16, α = 32, 드롭아웃 0.1, 풀링 24x24 → 8x8, 시간 자기 어텐션 헤드 8개, 교차 어텐션 1블록(잔차 없음), 헤드 2층(d = 512, ff 512) + MLP, 전 구성이 bfloat16(`video_clip_comp.py`). 입력은 384x384(부록 B). 영향: 논문 "lightweight"의 크기를 파악할 수 있는 것은 코드뿐이고, 인코더만 8600만 파라미터입니다(`03` §6).

**D13. 손실 (일치).** `SoftNearestNeighborsLoss`는 `logsumexp(pos) − logsumexp(pos ∪ neg)`로 식 (1)과 같고 τ = 0.1을 전달합니다(클래스 기본값 0.07은 쓰이지 않음). 제3 음성은 같은 배치의 다른 앵커 B−1개입니다.

**D14. DrQ-v2 하이퍼파라미터 (거의 일치).** 일치: lr 1e-4, 배치 256, γ 0.99, n-step 3, 목표 EMA 0.01, 2스텝마다 갱신, 시드 프레임 4000, 탐색 2000, σ(t) = linear(1.0, 0.1, 500000), 클립 0.3, feature 50, hidden 1024, frame stack 3(LIBERO 채널 6x3 = 18). 논문에 없음: 리플레이 크기 750,000(`drqv2_runner.py`), `torch.use_deterministic_algorithms(True)`. 에이전트는 DrM(휴면 비율 최소화) 코드에서 이식되어 `cal_dormant_ratio`를 계산하지만 로깅용입니다(`drqv2.py` 272행). 진행 임베딩 입력 옵션(`use_progress`)은 꺼져 있습니다.

**D15. 미공개 범위 (미공개).** MetaWorld 정책·환경 코드, 기준선 다섯 개, 실로봇 π0 + DSRL 코드(보상 질의 스트라이드 4, 기준 시연 수집 포함), 표·그림 스크립트, 학습 영상 목록·전처리. `reward_model_baseline/` 디렉터리 이름과 달리 기준선 구현은 들어 있지 않고 RARM 보상만 있습니다. 영향: 논문의 비교(기준선 대비 이득)를 이 저장소로 재현할 수 없습니다. 공개된 것은 RARM 자체의 LIBERO 경로뿐입니다.

## 4. 구현 가능성 판정 (구현할 수 있는지)

**결론: 논문 수치의 재현은 이 맥(Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)에서 불가능합니다. RARM의 아이디어를 작은 인코더로 옮긴 축소판은 조건부로 구현할 수 있고, 공개된 보상 모델의 유사도 추론은 승인 후 부분적으로 돌려 볼 수 있습니다.**

| 대상 | 이 맥에서 | 필요한 것 | 근거 |
|---|---|---|---|
| 정적 재계산과 토이(이 분석) | **완료** | numpy, scipy, 설치됨 | `03` |
| 질의 비용·학습 비용 측정/추정 | **완료**(무작위 가중치, MPS) | torch 설치됨 | `03` §6 |
| 공개 체크포인트로 유사도 히트맵(`reward_model.py` CLI) | **조건부** | 새 설치(peft, transformers, timm, opencv 등), DINOv3 가중치(별도 신청, 약 350 MB), RARM 체크포인트 233 MB 다운로드와 공식 코드 실행 승인. 코드는 `torch.hub`로 DINOv3 저장소를 GitHub에서 받고 `cuda` 플래그와 `bfloat16`을 전제. MPS 이식은 가능해 보이나 미확인. 질의 한 번 약 0.19 s(추정, `03` §6.2) | `04` D12, README |
| 공개 정책 학습(LIBERO-10, DrQ-v2, 10^6 스텝 x 5 시드) | **불가** | README는 "Linux, CUDA GPU, EGL"을 요구. LIBERO와 mujoco 3.1 계열, 데모 hdf5. 질의가 보상 계산만으로 한 번의 실행에서 약 13 시간(추정). 5 시드 x 5 과제면 약 300 시간 | `03` §6.2 |
| 보상 모델 학습(DAVIS+MOSE, 200 에폭 기본) | **불가** | GPU(ViT-B, 배치 64 x 36클립 x 5프레임, 한 스텝 약 2.5–3.8 PFLOP). 이 맥에서 45 에폭 약 6.5–9.8일(추정) | `03` §6.3 |
| 표 1(실로봇), 표 2, 그림 3–5, MetaWorld, 기준선 | **불가** | 코드·로봇·모델 미공개. AIRBOT Play, π0, DSRL 구현 필요 | `04` D15 |
| **축소판:** 작은 CNN 비교기를 시간 역전·구간 음성으로 SNN 손실 학습 → 자작 MuJoCo 과제(오프스크린 렌더 확인됨)에서 기준 궤적 대비 진행 게이트 보상 → 작은 RL | **조건부(구현 가능)** | mujoco 4.0, torch 2.15 설치됨. 학습 영상은 절차적 생성 또는 자작 시뮬 궤적(다운로드 없음). 렌더 속도는 측정 안 함 | - |
| **게이트의 민감도 실험(토이 확장)** | **시작 가능** | numpy | `03` §4.3 |

**노력 추정:** 축소판(MuJoCo 도달·밀기 과제 한 개, 소형 비교기, 게이트 Δ 스윕, 단순 DQN 또는 SAC 비교) 1–2주. 공식 파이프라인을 CUDA 기기에서 돌린다면 설치와 데이터 준비 1–2일, LIBERO 한 과제 한 시드 학습 1–2일(GPU 시간, 추정), 기준선을 같은 조건으로 다시 만드는 일은 별도로 수 주(기준선 코드가 없음).

**권장 순서:** (1) 저자에게 D5, D6, D7을 질문, (2) 축소판으로 게이트 Δ와 비교기 크기의 민감도 확인, (3) 공개 체크포인트의 유사도 추론은 승인 후 CPU/MPS에서 한 롤아웃만.

## 5. 저자에게 물을 질문

1. **초기 상태(D5):** 표 2와 그림 3의 LIBERO 평가는 기준 시연의 첫 상태에서 시작합니까, LIBERO 표준 초기 분포에서 시작합니까? 평가에서 물체 위치를 바꾸면 성공률이 어떻게 변합니까?
2. **데모 사용(D6):** 데모 행동 재생과 RSI 확률 감쇠는 모든 기준선에도 적용했습니까? 논문 F.1에 쓰지 않은 이유는?
3. **체크포인트(D7):** 논문 표는 epoch-20과 epoch-45 중 무엇으로 만들었고, 학습 영상 수, 에폭, 정적 필터 임계, 후속 음성 샘플링 변경은 무엇입니까?
4. **보상 가중치(D2):** 논문의 "reward-model weight" 값은 얼마이고, 에피소드당 진행 이벤트 수는 대략 몇 개입니까?
5. **표 2:** ± 의 정의, 시드별 값, 과제 범위(5과제인지 10과제인지), 평가 횟수는?
6. **표 1:** 예산 30/60/90의 체크포인트는 한 번의 DSRL 학습 실행에서 나온 것입니까, 독립 실행입니까? v1의 10회 평가는 v5의 30회 평가에 포함됩니까?
7. **게이트(Δ):** Δ = 0.5의 선택 근거와 기준 영상 길이(M)에 대한 민감도는? 점수 분포는?
8. **표 7:** "프레임당" 시간의 정의와 M, 측정 GPU는? 질의마다 5프레임을 다시 인코딩합니까?
9. **행동 격자:** F.2의 "50 Hz 격자"와 "약 20 Hz 실행"의 관계는?
10. **VIP:** 그림 5와 표 7의 VIP는 어떤 설정으로 학습했고 왜 §4.1에 없습니까?
11. **코드 공개 범위:** MetaWorld와 기준선 구현, 실로봇 코드를 공개할 계획이 있습니까? 라이선스는?
