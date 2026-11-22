# 04. 논문 ↔ 공식 코드 대조: X-VLA

대상: `repo/` (`2toinf/X-VLA`, 얕은 clone, HEAD `6c4ecfb`, 2026-10-20, Apache-2.0). **정적으로만 읽었습니다.** 코드를 실행하거나 import하거나 설치하지 않았습니다. 가중치와 데이터는 받지 않았습니다. 클론한 내용은 신뢰할 수 없는 데이터로 취급했습니다.

읽은 파일: `models/modeling_xvla.py`, `models/transformer.py`, `models/action_hub.py`, `models/configuration_xvla.py`, `train.py`, `peft_train.py`, `datasets/dataset.py`, `datasets/domain_config.py`, `README.md`, `evaluation/*/README.md`, `evaluation/SoftFold-Agilex/readme.md`.

판정 기호: 일치, 부분 일치, 불일치, 빈칸 (논문에 없거나 코드에 없음).

## 1. 일치하는 부분

| 항목 | 논문 | 코드 | 위치 |
|---|---|---|---|
| 원천별 소프트 프롬프트 | §3 (d), §4 | `soft_prompt_hub = nn.Embedding(num_domains, len × hidden)`, 정규분포 0.02 초기화 | `models/transformer.py` 335–337 |
| 원천별 행동 입·출력 선형층 | 그림 10 (0.04% 파라미터) | `action_encoder`, `action_decoder`가 `DomainAwareLinear` (원천별 가중치 행렬) | `transformer.py` 330–333 |
| 공유 시각·언어 투영 | 그림 10 ("Shared Linear Projection") | 기본값 `use_hetero_proj=False`에서 `vlm_proj`, `aux_visual_proj`는 공유 `nn.Linear` | `transformer.py` 318–322, `configuration_xvla.py` |
| 고정 시점은 전체 VLM, 보조 시점은 ViT만 | §4.1-1, 부록 C | 첫 번째 시점은 `language_model.encoder`, 나머지는 `_encode_image`만 | `modeling_xvla.py` 104–146 |
| 고유 자세와 행동, 시간 이어 붙이기 | §4.1-2 | `cat[action_noisy, proprio, time]` 후 `action_encoder` | `transformer.py` 약 370–380 |
| 30개 anchor, 4초 | §4.2.2 | `num_actions=30`. Droid, RoboMind 등 핸들러에서 `qdur=4.0` | `configuration_xvla.py`, `datasets/domain_handler/robomind.py` 52–70 |
| 위치와 6D 회전은 MSE, 그리퍼는 BCE (일반 EE6D) | §4.2.2 | `EE6DActionSpace`에서 MSE 위치·회전, `BCEWithLogitsLoss` 그리퍼 | `models/action_hub.py` 109–170 |
| 원천 가중 샘플링, 원천 간·궤적 간 섞기 | §4.2.2, 표 8 | `DATA_WEIGHTS`가 표 8 가중치와 같음 (0.4, 0.15, 0.15, 0.1, 0.1, 0.07, 0.03) | `datasets/dataset.py` 80–125, `domain_config.py` 19–27 |
| 두 단계 적응 (동결 후 공동 학습) | §4.2.1, 부록 H | `freeze_steps=1000`, `build_optimizer`에서 VLM과 트랜스포머 코어의 lr을 0으로 | `train.py` 116–172 |
| 사전학습 원천 7개, 합 288K | §1, 그림 3, 표 8 | 원천 메타 경로는 인자로만 받음, 원천 목록 파일은 없음 | `train.py` `--train_metas_path` |
| 벤치마크 평가 클라이언트 | §5, 부록 M | `evaluation/{calvin, libero, robotwin-2.0, vlabench}` 클라이언트와 README. Simpler는 README만 | `evaluation/` |
| 20차원 행동 (팔 두 개 × 10) | §4.2.2 (3+6+1) | 인덱스 `POS_IDX_1=(0,1,2)`, `ROT=(3..8)`, `gripper=9`, 두 번째 팔 10–19 | `action_hub.py` 109–125 |

## 2. 불일치 또는 차이

### D1. 소프트 프롬프트 길이 (빈칸)

- 논문은 프롬프트 길이 k를 본문과 부록 C 어디에도 적지 않습니다 (§3 "p_i ∈ R^k"만 있음).
- 코드 기본값은 `len_soft_prompts=32`입니다 (`configuration_xvla.py`).
- 논문의 재현에 필요한 값이 빠져 있으므로 빈칸으로 둡니다.

### D2. 원천 수 기본값 (빈칸 · 작은 불일치)

- 논문은 사전학습 원천 7개 (표 8). 코드 `num_domains` 기본값은 `configuration_xvla.py`에서 30, `SoftPromptedTransformer` 클래스 기본값에서는 20입니다. `DATA_DOMAIN_ID` 매핑은 `domain_config.py`에 있습니다.
- 학습 설정이 어느 값으로 고정됐는지는 공개 파일로 알 수 없습니다.

### D3. 공유 투영 대 원천별 HPT식 투영 (구현 옵션 존재)

- 기본 설정은 공유 투영입니다 (그림 10과 일치).
- 논문 §3 (b) HPT식 투영은 `use_hetero_proj=True`로 켤 수 있습니다 (`transformer.py` 318–322). 논문 어블레이션 (그림 4)이 이 옵션을 썼는지는 알 수 없습니다.

### D4. 그리퍼 손실: AGIBOT 행동 공간은 MSE (불일치)

- 논문 §4.2.2: 그리퍼는 BCE.
- 코드 `EE6DActionSpace` (일반): 그리퍼 BCE (`GRIPPER_SCALE=1.0`).
- 코드 `AGIBOTEE6DActionSpace`: **그리퍼에 MSE** (`action_hub.py` 232–240, `GRIPPER_SCALE=10.0`).
- AGIBOT은 사전학습 데이터의 48.9% (표 8)라서 영향이 큽니다. 논문 본문과 달리 대부분의 학습 신호는 MSE입니다. 어떤 행동 공간이 논문 실험에 쓰였는지 본문에 명시되지 않았습니다.

### D5. 학습 목표: 속도 회귀 대 깨끗한 행동 회귀 (불일치, 중요)

- 논문 §2, 식 (2): flow matching은 속도장 v_θ(A_t, o, t)를 A − A_0에 맞춥니다 (A_0 ∼ N(0, I), A_t = (1−t)A_0 + tA).
- 코드 학습: `x_t = t·noise + (1−t)·gt` (t = 1이 잡음), 네트워크 출력을 **깨끗한 행동 gt와 직접** 비교합니다 (`modeling_xvla.py` 158–178, `compute_loss(pred_action, action)`). 즉 속도가 아니라 x-예측 (clean-action prediction)입니다.
- 샘플링: `action = 0`에서 시작하고 `x_t = x1·t + action·(1−t)` (`x1 ∼ N(0,I)`, `t = i/steps`, i = steps … 1)로 네트워크 출력을 `action`에 덮어씁니다 (`modeling_xvla.py` 196–215). 이는 Euler 속도 적분이 아니라 x-예측의 반복 정제입니다.
- 두 방식은 같은 분포 목표를 다른 매개화로 쓸 수 있지만, 논문 식 (2)와 코드는 다른 목적함수입니다. 논문의 식 (2)가 실험에 쓰인 목적이 아니라는 뜻입니다. 논문 표기와 실제 학습의 차이를 본문에서 설명하지 않았습니다.
- 시간 샘플링: 코드는 배치마다 `(rand(1) + arange(B)/B) mod (1 − 1e-5)`로 **층화 샘플링**합니다 (`modeling_xvla.py` 165–166). 논문은 U(0,1)입니다. 주변 분포는 같지만 배치 안의 상관이 다릅니다.
- 실행 확인 없이는 두 방식의 성능 차이를 말할 수 없습니다. 정적 읽기로 확인된 사실만 적었습니다.

### D6. 학습률 축소 (값 빈칸, 기본값은 축소 없음)

- 논문 §4.2.1: 프롬프트와 VLM의 학습률을 줄입니다. 배율은 적지 않았습니다.
- 코드: `--learning_coef` 기본값 **1.0** (`train.py` 84). 즉 기본 실행은 축소하지 않습니다.
- README 예시 명령은 `--learning_coef 0.1`입니다. 논문 실험 값이 0.1인지는 확인되지 않았습니다.
- 축소 대상은 코드에서 VLM과 소프트 프롬프트가 같은 배율입니다 (`update_group_lrs`). 논문 "프롬프트와 VLM"과 일치합니다.

### D7. weight decay (불일치: 기본값 대 논문)

- 논문 표 7: 0.01.
- 코드 `--weight_decay` 기본값 **0.0** (`train.py` 85). README 예시 명령에 지정이 없으므로 0.0으로 실행됩니다.
- 논문 설정을 재현하려면 명령에 `--weight_decay 0.01`을 넣어야 합니다.

### D8. 배치, 스텝, 워밍업 (불일치)

| 항목 | 논문 | 코드 기본값 | README 예시 |
|---|---|---|---|
| 사전학습 배치 | 1024 (표 7, 부록 G) | 16 (`--batch_size`) | 명시 없음 |
| 사전학습 스텝 | 200K (표 7) | 1,000,000 (`--iters`) | 미세조정 50K |
| 동결 스텝 | 1000 (부록 H) | 1000 | 1000 |
| 워밍업 | **1000 (부록 H)** | **2000** (`--warmup_steps`) | **2000** |
| 그래디언트 클리핑 | 언급 없음 | 1.0 (`--max_grad_norm`) | 명시 없음 |

- 부록 H의 "1,000-iteration warm-up"과 코드 기본값 2000이 다릅니다. README 예시도 2000이라 실제 실행 설정은 코드를 따릅니다.
- 사전학습 실행 스크립트 (노드 수, 데이터 메타 파일)는 공개되지 않았습니다. 빈칸입니다.

### D9. Soft-Fold 데이터 규모와 권리 (불일치)

- 논문: 1,200 궤적 (§5.2, 부록 F, 그림 11).
- HF 데이터셋 카드 (`Facebear/XVLA-Soft-Fold`, MIT): **"1,500 episodes"**, 파일 1,724개, 475.6 GB.
- README 모델 표: "cloth folding with a 100% success rate in 2 hours" (Soft-Fold 체크포인트). 논문은 "33 folds per hour".
- 논문은 "we will release the dataset"이라고 쓰고 실제로 공개되어 있습니다. 규모가 다르므로 어느 버전이 논문 실험인지 저자에게 확인해야 합니다.

### D10. 원천 메타 파일과 사전학습 스크립트 (빈칸)

- 사전학습 원천 메타 JSON은 공개되지 않았습니다. `train.py`는 `--train_metas_path`만 받습니다.
- `datasets/domain_handler/`에는 원천별 핸들러 8개 (agiworld, droid, lerobot_agibot, lerobotv21, real_world, robomind, simulations, x2robot)가 있지만, 288K 에피소드를 어떻게 메타 파일로 묶었는지는 알 수 없습니다.
- 공개된 것은 사전학습 **체크포인트** `X-VLA-Pt`와 미세조정 체크포인트들입니다.

### D11. README 수치와 논문 표 (불일치)

| 항목 | README | 논문 | 위치 |
|---|---|---|---|
| LIBERO LoRA Spatial | 96.2% | 95.4 (표 3), 96.6 (표 6) | `README.md` 90 행 |
| LIBERO LoRA Object | 96% | 96.6 (표 3), 95.4 (표 6) | 91 행 |
| LIBERO LoRA Goal | 94.4% | 96.0 (표 3), 95.0 (표 6) | 92 행 |
| LIBERO LoRA Long | 83.2% | 84.2 (표 3, 6) | 93 행 |
| Simpler Google VM / VA | 83.5% / 76.4% | 80.4 / 75.7 (표 2, 표 12) | 84 행 근처 |
| Simpler WidowX LoRA | 66.7% | 54.2 (표 3), 54% (본문) | 94 행 |
| LIBERO 전체 미세조정 | 98.1% | 98.1 (표 2) | 일치 |
| CALVIN ABC→D | 4.43 | 4.43 (표 2) | 일치 (표 14 합 4.414와는 불일치, `03` V4) |
| RoboTwin2 Easy | 70% | 70.0 (표 2) | 일치 (Hard 39.0와는 `03` V4 불일치) |
| VLABench | 51.1 | 51.1 (표 2) | 일치 |
| WidowX 전체 | 95.8% | 95.8 (표 2) | 일치 |

- README 상단: 공개 HF 형식으로 바꾼 뒤 "약 1%의 성능 하락"이 있다고 씁니다 (미해결 표시 `[ ]`). 공개 체크포인트의 수치는 논문과 약 1% 다를 수 있습니다.
- LoRA 모델 수치의 차이는 체크포인트 버전이나 평가 방식 차이일 가능성이 큽니다.

### D12. LoRA 설정 (논문 빈칸 대 코드)

- 논문: 9M 파라미터 (약 1%), rank 미기재 (§5.2, 표 3).
- 코드 `peft_train.py` 197–201: `LoraConfig(r=8, lora_alpha=16, target_modules="all-linear")`.
- 파라미터 수 9M 검산은 모델을 불러와야 하므로 하지 않았습니다. r=8 all-linear가 0.9B에서 약 9M인지는 실행 확인이 필요합니다.

### D13. 평가 코드 범위 (빈칸)

- `evaluation/`에는 CALVIN, LIBERO, RoboTwin-2.0, VLABench, Simpler(README만), SoftFold-Agilex가 있습니다.
- **NAVSIM 평가 코드가 없습니다.** 논문 표 2와 표 11의 PDMS (87.3)는 이 저장소만으로 재현할 수 없습니다.
- 실로봇 (WidowX, AgileX, AIRBOT) 평가는 `deploy.py` 서버 구조만 있고, 과제 정의나 성공 판정 코드는 없습니다.

### D14. 학회 표기 (일치)

- README 상단: "X-VLA Accepted to ICLR 2026". ICLR 2026 공식 proceedings 목록에서 같은 제목 확인 (`README.md` venue 항목). 일치.

## 3. 논문에 없는 코드 기본값 (주의)

- 손실 가중치 XYZ 500, ROT 10, 그리퍼 1 (EE6D) 또는 10 (AGIBOT). 논문에는 가중치 언급이 없습니다. 이 값들이 논문 결과에 필수인지 불분명합니다.
- `max_grad_norm = 1.0`, `dim_time = 32`, `max_len_seq = 512`, `mlp_ratio = 4.0`는 논문 부록 C와 G에 없습니다.

## 4. 구현 가능성 (이 맥)

| 작업 | 자원 | 판정 |
|---|---|---|
| 추론 확인 (`X-VLA-Pt` 또는 LIBERO 체크포인트, 10 denoise 스텝) | 가중치 3.5 GB, MPS, MuJoCo/LIBERO 설치 | 조건부. 설치와 가중치 다운로드 승인 후 MPS 포팅이 필요할 수 있음. 속도는 측정하지 않음 |
| LoRA 미세조정 (9M, 40K 스텝) | 0.9B 가중치 로드, 배치 128, bf16 | 불가 (노력이 과도함, 측정 전 판단) |
| 사전학습 (64 A100, 4일) | GPU 클러스터 | 불가 |
| Soft-Fold 미세조정 (400K 스텝, 476 GB) | GPU, 대용량 데이터 | 불가 |
| NAVSIM 평가 | 코드 없음, 시뮬레이터 | 불가 (코드 공개 없음) |

- 학습 코드는 공개되어 있지만 CUDA 기반 `accelerate` 실행 설정이고, MPS에서 0.9B를 학습하는 것은 메모리와 시간 모두 맞지 않습니다.

## 5. 저자에게 확인할 질문 (정리)

1. 소프트 프롬프트 길이 k는 얼마인가? (D1)
2. 논문 실험의 그리퍼 손실은 AGIBOT 행동 공간에서도 BCE인가, MSE인가? (D4)
3. 논문 실험의 `learning_coef`, `weight_decay`, 워밍업 값은 무엇인가? (D6–D8)
4. Soft-Fold 데이터는 1,200인가 1,500인가? 논문 실험에 쓴 버전은? (D9)
5. 그림 5의 R² = −0.925는 어떤 적합인가? (`03` V9)
6. 표 3의 LIBERO 스위트 값 (Spatial과 Object)은 어느 쪽이 맞는가? (`03` V2)
7. NAVSIM 평가 코드는 공개할 계획이 있는가? (D13)
