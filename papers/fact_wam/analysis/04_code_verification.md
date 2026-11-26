# 04. 공식 코드 대조

대상: GitHub `Bariona/FACT` (Apache-2.0), **커밋 `5b299a2059f00c043fedc5c9dd17bf43fa14cbec`**(2026-09-15, "docs: update FACT citation to CoRL 2026"). 얕은 복제를 **정적으로 읽기만** 했고 실행, 설치, 가져오기를 하지 않았으며 가중치와 데이터는 받지 않았습니다. 다시 받으려면 `git clone https://github.com/Bariona/FACT repo && git -C repo checkout 5b299a2059f00c043fedc5c9dd17bf43fa14cbec`입니다. 논문은 v2(`01`)입니다. 코드는 Python과 셸을 합쳐 약 1.4만 줄(모델 `world_action_model/models/transformer_wa_casual.py` 2,479줄, 학습기 `world_action_model/trainer/wa_casual_trainer.py` 959줄)입니다.

## 1. 저장소가 담은 것과 담지 않은 것

README는 "RoboTwin 파이프라인(데이터 준비, 학습, 추론, 폐루프 시뮬레이션 평가)"이라 밝히고, 감사 문구에서 WAN2.2, giga-world-policy, GigaTrain, GigaDatasets, RoboTwin을 적습니다.

| 있음 | 위치 |
|---|---|
| 모델(교사 강제 마스크, 행동 어댑터, 2단계 추론) | `world_action_model/models/`, `pipeline/` |
| 학습기와 설정 (RoboTwin) | `world_action_model/trainer/`, `configs/robotwin.py`, `fact_train/` |
| 데이터 변환(값 목표, 행동 손실 마스크), 실패 메타 읽기 | `world_action_model/transformers/wa_transforms_lerobot.py`, `fact_datasets/datasets/lerobot_dataset.py` |
| 추론 서버, best-of-N | `scripts/inference_server.py` |
| RoboTwin 평가 연결 | `evaluation/robotwin/` (패치 하나와 셸 스크립트) |

| 없음 (저장소에서 찾지 못함) | 논문 대응 |
|---|---|
| 실패 롤아웃을 수집하고 실패 시작 시점을 주석하는 코드 | 알고리즘 2 |
| 실패 롤아웃 데이터 약 1.3K (RoboTwin) | 4.1절 |
| 실로봇 학습·평가 코드와 데이터, 기준선 코드(π0, π0.5, Motus, Cosmos) | 4.2절 |
| 절제 설정(인과 마스크 제거, 영상 공동학습 제거, 실패 행동 손실 포함), 실패 비율 p 혼합 | 표 1, 2, 그림 6 |
| 미래 예측 PSNR·SSIM 평가 | 표 4, 8 |

## 2. 논문과 일치하는 것 (K1–K9)

| # | 항목 | 코드 근거 |
|---|---|---|
| K1 | 학습률 백본 2e-5, 로봇 토큰 2e-4 | `configs/robotwin.py` `LR`, `ACTION_LR`. 로봇 토큰 계열(어댑터, 상태·행동·가치 인코더와 디코더) 전체에 2e-4가 적용됨(`wa_casual_trainer.py` 83–110) |
| K2 | H = 48, 미래 영상 오프셋 [0, 12, 24, 36, 48] | `NUM_FRAMES = 48`, `_image_frame_offsets`. 추론은 `NUM_FRAMES: 5`, 단계 20, 안내 0.0 |
| K3 | 토큰 순서와 마스크: A는 P와 A만, G는 P와 G만, V는 P·G·V, I는 P·G·V·I | `_build_teacher_forcing_mask`, flex 버전 `teacher_forcing_mask_mod`. 그림 3과 같습니다. 코드는 V 자리에 미래 상태 토큰도 같은 구간(SV)으로 묶음 (D2) |
| K4 | 실패 에피소드의 행동 손실 마스크, 가치·영상 손실은 유지 | `wa_transforms_lerobot.py` 514, `wa_casual_trainer.py` 835–853 |
| K5 | λ_fail = 1 | `VALUE_PENALTY_SCALE = 1.0` (식은 D3, D4) |
| K6 | WAN2.2-5B 초기화, 공유 백본 하나, 별도 세계 전문가 없음 | 설정 `pretrained=WAN_MODEL_DIR` (WAN2.2-TI2V-5B-Diffusers) |
| K7 | FFN 뒤 로봇 토큰 어댑터 | `WanTransformerBlock` 안 `robot_ffn_adapter` (FFN 출력에 어댑터 출력을 더함) |
| K8 | 2단계 추론: 1단계로 행동, 그 행동을 G에 넣어 2단계 | `pipeline/wa_pipeline.py` (A 자리는 영 텐서, G에 1단계 행동) |
| K9 | 흐름 매칭 손실(속도 목표 노이즈 − 데이터), 보간 (1 − τ)x + τε | `wa_casual_trainer.py` 696–715 (D6, D7 참조) |

## 3. 논문과 다른 것 (D1–D17)

### 손실, 목표, 모델 구성

**D1. 손실 가중치.** 논문 3.3절은 w_a = 20, w_v = w_I = 1입니다. 공개 설정은 `ACTION_LOSS_WEIGHT = 10.0`, `VALUE_LOSS_WEIGHT = 0.4`, `VISUAL_LOSS_WEIGHT = 1.0`입니다(`configs/robotwin.py` 112–115). 비율로 행동 대 영상이 20 대 1(논문)과 10 대 1(코드), 가치 대 영상이 1 대 1(논문)과 0.4 대 1(코드)입니다. 공개 체크포인트의 학습 설정이 이 파일과 같은지는 확인하지 못했습니다.

**D2. 논문에 없는 미래 상태 가지.** 코드의 토큰 순서는 `[state | ref | pred_action | gt_action | future_state | value | future_image]`이고 손실에 `future_state_loss`(가중치 0.4)가 있습니다. 논문 식 (5)와 식 (8)의 순서와 항에는 미래 고유감각 상태 예측이 없습니다.

**D3. 가치 목표의 방향과 형식.** 논문은 진행도 p_{t+H} = (t+H)/T로 클수록 좋고 선택은 argmax(식 10)입니다. 코드는 **시간 잔여** `(L − future_state_index − 1)/(L − 1)`로 작을수록 좋은 값입니다(`wa_transforms_lerobot.py` 500–510). best-of-N은 `torch.argmin`으로 가장 작은 값을 고릅니다(`inference_server.py` 428, `launch_config.yml` "lowest predicted value (time-to-go; smaller = closer to success)"). 성공 구간에서는 1 − p와 같은 값이라 방향이 뒤집혔을 뿐 같은 정보입니다(재계산 상관 1.0). 논문의 식 (10)과 코드는 부호가 반대지만 같은 선택을 합니다.

**D4. 실패 목표.** 논문 식 (6)은 clip(p − λ·1_fail, 0, 1)입니다(λ = 1에서 실패 창의 목표가 0 하나, `03` V9). 코드는 실패가 켜진 창에서 값에 `+1.0`을 더한 뒤 [−1, 2] 범위에서 [−1, 1]로 정규화하고 [−1, 1]로 자릅니다(`wa_transforms_lerobot.py` 508–511, `scripts/compute_norm_stats.py` `_build_default_value_stats`: min −1, max 2). 결과적으로 성공은 정규화 값 [−0.33, 0.33], 실패는 (0.33, 1.0]에 놓이고 **실패 창 안에서도 시간 잔여 순서가 보존됩니다.** 식 (6)을 문자 그대로 쓸 때와 다른 목표입니다. 정규화에 쓰는 min = −1, max = 2는 기본 상수로 고정되어 있습니다.

**D5. 실패 표시 시점과 행동 마스크의 범위.** 논문은 지시자 1_fail(t + H), 즉 실행하는 행동 청크가 끝난 시점까지 실패가 일어났는지를 씁니다. 코드는 창 시작 프레임이 `failure_active_from_frame` 이상인지(`local_frame_index >= active_from`)로 정하고(`lerobot_dataset.py` 641–647), 값이 없으면 0으로 두어 에피소드 전체가 실패 표시가 됩니다. 시작 시점 onset과 H = 48일 때 t + H ≥ onset이지만 t < onset인 창은 최대 48개이고(재계산 예: onset 100), 이 창은 논문 정의에서는 실패, 코드 정의에서는 성공 목표를 받습니다. 행동 손실 마스크는 `failure_episode`로 **실패 에피소드 전체**(실패 시작 이전 프레임 포함)를 끕니다.

**D6. 노이즈 수준 공유.** 코드는 샘플마다 노이즈 수준 σ 하나를 뽑아(균일 난수에 flow shift 3.0 적용, 천 단계 반올림) 행동, 미래 상태, 가치, 영상에 모두 씁니다(`get_timestep_and_sigma`, `wa_casual_trainer.py` 705–715). 논문 식 (7)은 모달리티마다 τ를 쓰는 것처럼 읽히지만 표집 방식과 shift 값을 적지 않았습니다.

**D7. 영상 노이즈의 중심.** 논문 식 (7)은 z1 ~ N(0, I)입니다. 코드의 영상 노이즈는 `visual_latents[:, :, :1] + torch.randn_like(visual_latents)`로 **첫 프레임 잠재가 더해진 가우시안**입니다(`wa_casual_trainer.py` 702). 같은 값이 속도 목표에도 들어갑니다. 영상 손실은 첫 프레임을 제외한 미래 프레임에만 곱해집니다. 이 형식은 논문에 적혀 있지 않습니다.

**D8. 행동 손실 정규화.** 행동 손실은 배치 안 비실패 샘플의 수로 나눈 평균입니다(`_masked_mean`, 849). 실패 비율이 높은 배치에서 성공 샘플당 행동 손실 가중이 커집니다. 논문 알고리즘 1의 ma · wa · La와는 정규화가 다릅니다.

### 학습 설정과 데이터

**D9. 논문에 없는 학습 설정.** 8 GPU, DeepSpeed ZeRO-2, bf16, GPU당 배치 32(전체 256), 최대 150,000 스텝, AdamW β = (0.9, 0.95), 가중치 감쇠 1e-4(1차원·정규화 파라미터는 제외), 워밍업 500스텝 뒤 코사인 감쇠, 해상도 메인 256 × 192와 손목 각 128 × 96을 합친 384 × 192 캔버스, 15 fps, T5 길이 64가 코드에 있고 논문은 적지 않았습니다(3.3절은 AdamW와 두 학습률만 명시).

**D10. 어댑터 크기.** 논문은 "lightweight"라고만 쓰고, 코드는 병목 차원 1024의 MLP입니다(`ROBOT_ADAPTER_RANK = 1024`). 블록 폭 3,072와 30블록을 가정하면 어댑터 파라미터는 약 1.9억(추정, 백본 크기에서 가정한 값이고 설정 파일로 확인하지 못함)입니다.

**D11. 실패 데이터 선택.** 기본 데이터 탐색은 최상위 그룹 `Fail`과 `EvalRollouts`를 **제외**합니다(`robotwin_utils.py` 10, 142). 실패 롤아웃은 과제 폴더 안 `meta/failure_rollouts.jsonl`(에피소드 번호, 실패 여부, 시작 프레임)이 있을 때만 읽히고(`lerobot_dataset.py` 527), 이 파일을 만드는 코드는 없습니다. 공개 데이터 카드의 메타데이터에는 파일이 `robotwin-v2.tar` 하나이고 실패 롤아웃 포함 여부는 확인하지 못했습니다.

**D12. 미니배치 구성.** 알고리즘 1은 B_s ⊂ D_s와 B_f ⊂ D_f를 따로 뽑는 것처럼 쓰지만, 코드의 `EpisodeMixtureSampler`는 과제 폴더별 가중치 × 에피소드 수에 비례해 에피소드를 뽑고 프레임을 균일하게 뽑습니다. 실패 비율은 폴더 구성에서 정해지고 따로 조절하는 인자는 없습니다. 그림 6의 p 혼합을 만드는 코드는 없습니다.

### 추론과 평가

**D13. "행동만" 경로의 기본값.** 논문 부록 E의 380 ms는 "행동만 내는 배포"로 설명됩니다. 코드의 파이프라인은 `action_only=True`여도 2단계에서 미래 상태와 가치를 디노이징하고(`wa_pipeline.py` 주석 "this stage still runs for future_state/value"), 그것을 건너뛰는 옵션 `--skip_future_state_value`는 서버 설정 기본값이 `"0"`(끔)입니다(`launch_config.yml` 32). 따라서 공개 설정을 그대로 쓰면 행동만이 아니라 2단계 가치 계산까지 포함된 지연입니다. 380 ms가 어느 설정의 값인지는 확인하지 못했습니다(`03` V11의 그림 판독 지연과 함께 볼 것).

**D14. 평가 횟수와 지시문.** 논문은 과제·분할당 100회(표 6)입니다. 평가 연결 설정은 `TEST_NUM: 50`이고 전체 스윕 스크립트의 예시도 `eval_all_tasks.sh demo_clean 50`입니다(`launch_config.yml` 50). 클라이언트 스크립트는 환경변수가 없으면 `test_num=1`이고, RoboTwin 원본은 100이 고정이라 저장소가 패치(`robotwin_test_num.patch`)로 덮어쓰기를 허용합니다. 또 `deploy_policy.yml`의 `instruction_type: unseen`이 기본인데, 논문은 지시문 형식을 적지 않았습니다.

**D15. 재계획 간격과 렌더.** 기본 `execute_actions_per_plan: 48`로 계획 하나의 48스텝을 열린 루프로 실행합니다. 논문은 재계획 주기를 적지 않았습니다. `LOW_FREQUENCY_RGB: "1"`이 기본이라 재계획 스텝에서만 RGB를 렌더합니다.

**D16. 점수화 기본값.** 평가 설정에서 `ENABLE_SAMPLE: "0"`, `BEST_OF_N: 4`입니다. 시뮬레이션은 기본으로 점수화를 쓰지 않아 표 1의 설정(점수화 없음)과 맞습니다. 후보 시드는 `time.time_ns()`에서 만들어 실행마다 달라집니다(`inference_server.py` 407).

**D17. 절제와 기준선.** 인과 마스크 제거, 영상 공동학습 제거(영상 손실 0으로 두는 정도로 가능해 보임), 실패 행동 손실 포함(`action_loss_mask`를 끄면 되어 보임)은 설정 플래그로 제공되지 않습니다. Hugging Face 모델 저장소에는 `transformer/`와 `success_only/transformer/`가 함께 있으나, 두 체크포인트의 학습 설정은 확인하지 못했습니다.

## 4. 구현 가능성 판정 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**논문 수치의 재현은 불가능합니다. 방법의 핵심(마스크, 행동 손실 마스킹, 가치 목표)은 소형 모델로 구현해 볼 수 있습니다.**

| 일 | 판정 | 이유와 노력 |
|---|---|---|
| 공식 설정 학습 (WAN2.2-5B, 8 GPU, 150k 스텝, 배치 256) | 불가 | CUDA, DeepSpeed, bf16 전제. 설정 파일이 `GPU_IDS = [0..7]`. 약 5B 파라미터 학습은 이 맥 메모리(24 GiB)로 불가 |
| RoboTwin 시뮬레이션 평가 | 불가 | RoboTwin은 Linux·CUDA 시뮬레이터(SAPIEN) 전제. 설치 스크립트가 Linux용 Miniconda를 받음 |
| 실로봇 (YAM 팔 둘, GELLO) | 불가 | 장비, 데이터, 코드 없음 |
| 공개 체크포인트 추론 | 조건부, 어려움 | 가중치 약 10.4 GB, WAN2.2-5B와 T5 인코더 추가 다운로드, 새 가상환경, 서버 기본 장치 `cuda:0`과 `nvidia-ml-py`, NUMA 설정 의존. MPS 포팅과 메모리 확인이 필요(1–2주 추정, 측정 전). 다운로드와 실행 승인 필요 |
| 표 재계산과 정적 대조 | 가능, 완료 | `analysis/recompute.py` |
| 소형 재구현: 교사 강제 마스크 + 실패 행동 손실 마스킹 + 가치 목표 | 가능 | torch 2.15 CPU/MPS. 작은 트랜스포머와 합성 환경으로 마스크 단위 시험, 손실 마스킹 시험, 식 (6) 대 코드식 목표 비교. 노력 약 3–5일(추정, 측정 전) |
| 소형 시뮬 비교(성공만 / 실패 포함 / 실패 모방) | 조건부 | 작은 접촉 과제 필요. mujoco 4.0 설치됨. 1–2주(추정) |

노력은 모두 추정이고 이 세션에서 측정하지 않았습니다.

## 5. 라이선스와 자산

- 코드: Apache-2.0(LICENSE 파일, `pyproject.toml`의 `license`도 Apache-2.0). 모델 카드 Apache-2.0, 데이터 카드 MIT(HF 메타데이터). 백본 WAN2.2는 Apache-2.0(GitHub API).
- 이 폴더에는 공식 코드를 넣지 않았고, 가중치와 데이터는 받지 않았습니다.
