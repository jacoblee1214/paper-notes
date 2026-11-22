# EgoMimic: 논문 ↔ 공식 코드 대조 (04)

**대조 대상:** `SimarKareer/EgoMimic` @ `6d63e9a` (2024-11-10), robomimic 포크 `3d65f03` (2024-12-23). 정적 읽기만 했습니다. 실행, import, 설치는 하지 않았습니다.

## 0. 라이선스 (근거와 판정)

- 논문: `nonexclusive-distrib/1.0` (CC BY 아님). 공개본은 우리 그래프만 사용하고 논문 그림을 넣지 않습니다.
- 코드: MIT (LICENSE 파일 확인).

## 1. 확인한 일치 항목

| 항목 | 논문 | 코드 | 판정 |
|---|---|---|---|
| Z-score 정규화 (손과 로봇 행동, proprio를 각각) | §III-B, 부록 A | `robomimic/utils/obs_utils.py` `normalize_batch`: `(obs − mean)/std`, `normalize_actions=True`. 통계는 데이터셋별 (`pl_model.py` `norm_dicts`) | 일치 |
| 두 데이터셋의 정규화 통계를 따로 씀 | §III-B | `pl_model.py:training_step`에서 `[train_dataset1, train_dataset2]` 통계를 각 배치에 맞춰 적용 | 일치 |
| 사람 데이터 1초 구간 | 부록 A | `aria_to_robomimic.py`: `HORIZON=10`, `STEP=3.0`, 30 Hz 기준 10샘플×3프레임 = 1초 | 일치 (30 Hz 가정) |
| 청크 길이 100 | 부록 A | 설정 `seq_length`, `seq_length_hand` = 100. 사람 행동은 `interpolate_arr(…, 100)`으로 보간 | 일치 |
| 손과 로봇 모두 SAM2로 가림 | 부록 A | `masking/utils.py`가 `sam2`를 import, `sam2_hiera_tiny.pt` 사용 | 일치 (체크포인트 크기 차이 가능) |
| 공유 인코더·트랜스포머, 포즈 헤드와 관절 헤드 | §III-C, Fig. 4 | `egomimic.py` `EgoMimicModel`: `robot_action_head`(관절), `hand_action_head`(보조 포즈 헤드), 나머지 공유 | 일치 |
| 사람 데이터는 포즈 손실만, 로봇은 관절 손실도 | 부록 D | `_forward_training`: 손 모달리티 `l1 * hand_lambda`, 로봇 모달리티 `l1_robot + l1_hand` | 일치 |
| ACT 기반 CVAE 잠재 | §III-C, 부록 D | `ACT`, `StyleEncoder`, `latent_dim` 32 | 일치 |
| 옵티마이저 AdamW, 초기 학습률 5e-5, 선형 스케줄 | Table V | `optim_params`: adamw, 5e-5, `scheduler_type: linear` | 일치 |

## 2. 차이 (D1–D7)

**D1. 손실 함수: MSE vs L1 (논문 내부 불일치 포함)**
- 논문 Alg. 1은 `MSE`를 씁니다 (Alg. 1 줄 6, 11, 12).
- 본문 §III-C와 부록 D는 `L1`을 씁니다.
- 코드: `egomimic.py:_forward_training`에서 `F.l1_loss`, 설정 `"l1_weight": 1.0`, `"l2_weight": 0.0`.
- 판정: 코드는 L1입니다. 논문은 두 곳에서 서로 다릅니다. 재현할 때는 L1을 기준으로 삼습니다.

**D2. KL 가중치: 논문 암묵적 1 vs 코드 20**
- 논문 식 `L = L_robot + L_hand`에는 KL이 있지만 가중치를 쓰지 않습니다 (부록 D: "KL ... as in ACT").
- 코드: `act.py` `_compute_losses`에서 `action_loss = recons + kl_weight * kl`, `kl_weight = 20`(설정 `act.kl_weight`, `act_config.py`에도 20).
- 판정: 가중치 20은 ACT의 기본값으로 보입니다. 논문에는 밝히지 않았습니다.

**D3. 배치와 반복: 논문 128·120k vs 설정 16**
- 논문 Table V와 부록 D: 전역 배치 128, 4대 A40, 120,000 반복, 약 24시간.
- 공식 설정 `egomimic_oboo.json`, `egomimic_groceries.json`, `egomimic_clothes.json`: `batch_size = 16`. `act.json`, `actHand.json`: 32. `base_config.py` 기본값: 100.
- 반복 수는 설정에서 `num_epochs = 10000`과 `epoch_every_n_steps = 100`으로 정하지만 실제 총 스텝은 Lightning 설정에 따라 다릅니다 (`pl_train_utils.py`의 `max_epochs`). 120k 반복과 같은지 확인하지 못했습니다.
- 판정: 논문 학습 설정과 공식 설정 파일이 맞지 않습니다. 논문 결과를 재현하려면 설정 파일을 어떻게 맞췄는지 저자에게 확인이 필요합니다.

**D4. 두 배치의 손실 결합: 합 vs 평균**
- 논문 Alg. 1: 한 번 업데이트에 `L_hand + L_robot + L_robot_q`를 더해 씀 (합).
- 코드 `pl_model.py:training_step`: 로봇 배치와 손 배치를 **각각 forward**한 뒤 `losses[key] = torch.mean(torch.stack(...))`로 평균하고, 그 평균의 `action_loss`를 반환합니다.
- 결과: 두 항의 상대 가중치는 같지만 전체 크기가 절반입니다. AdamW는 손실 크기에 거의 불변이므로 실제 차이는 작을 것으로 예상하지만, 측정하지 않았습니다.

**D5. 손 행동 표현: SE(3) 포즈 vs 3차원 위치**
- 논문 §III-A: `H p ∈ SE(3)×SE(3)`, 로봇 `R p ∈ SE(3)×SE(3)`. 부록 B: 로봇의 `R p_t = FK(q_t) ∈ R^{3×3}` (세 점의 3차원 좌표).
- 코드 `aria_to_robomimic.py`: 사람 행동은 `wrist_and_palm_poses`에서 만든 `actions_xyz`, `ac_dim = 3` (한 손), `6` (양손). 설정은 `actions_xyz_act`를 씀.
- 판정: 실제로 쓰는 행동은 **손목 또는 손바닥의 3차원 위치**입니다. 회전(SE(3))은 예측하지 않습니다. 논문 본문의 SE(3) 표기는 과장되었거나 다른 표현을 가리킵니다.

**D6. 사람 데이터 감속: "4배" 인자를 찾지 못함**
- 논문 부록 A: "4배 느리게 맞춤", Table VII: "0.25 배".
- 코드: 사람 쪽은 `HORIZON × STEP` (10 샘플, 3프레임 간격)으로 1초 구간을 만듭니다. 그러나 명시적 "4배" 또는 "0.25" 인자는 찾지 못했습니다. 로봇 쪽 4초 구간을 확인하지 않았습니다 (`aloha_to_robomimic.py` 분석 생략).
- 판정: 1초 구간은 일치하고, 4배 인자의 실제 위치는 미확인입니다.

**D7. 증강: 논문 색 흔들기만 vs 설정의 자르기와 색 흔들기 다섯 종류**
- 논문 Table V: "Color Jitter" 하나.
- 설정 `egomimic_oboo.json`: `CropRandomizer` (크기 240×320, 크기 비율 0.9–1.0), `brightness`, `contrast`, `saturation` (0.9–1.1), `hue` (±0.05).
- 판정: 논문에 없는 자르기 증강이 있습니다. 결과 차이 가능성이 있습니다.

## 3. 확인하지 않은 항목

- 평가 루프 (`eval_real.py`): 1 Hz 추론, 25 Hz 제어, RTX 4090 설정은 읽지 않았습니다.
- Mimicplay 기준선의 목표 조건 제거와 구현 (`mimicplay.py`): 읽지 않았습니다.
- `robomimic` 포크 전체: 정규화 함수만 확인했습니다.
- 데이터 HDF5의 실제 내용과 키 (용량 243 GB라 받지 않음).

## 4. 구현 가능성 판정 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**판정: 논문 학습 재현은 이 맥에서 불가능합니다. 정규화와 마스킹 코드의 일부를 CPU에서 검사하는 것은 가능합니다.**

| 항목 | 판정 | 근거 |
|---|---|---|
| 공식 학습 (`pl_train.py`) | **불가 (코드 수정 없이)** | `egomimic.py:189`, `act.py:270`에 `model.cuda()`가 고정. 설정의 `"cuda": true`. 이 맥에서 실행하려면 MPS 포팅이 필요 |
| 사람 데이터 처리 (`aria_to_robomimic.py`) | **불가 (코드 수정 없이)** | `:437`에 `torch.cuda.get_device_properties(0)`, `:445`에 `torch.autocast("cuda", …)`. CUDA가 없으면 실패 |
| 샘플 데이터 HDF5 (약 243 GB) | **조건부** | 디스크 용량과 다운로드 승인이 필요. 1 GB 이상이라 승인 필요 |
| 정규화 통계 재계산 (CPU) | 가능 | `normalize_batch`와 평균·표준편차는 numpy로 옮길 수 있음 |
| 정책 구조 재구현 (MPS 또는 CPU, 소형) | 조건부 | 직접 짠 torch 코드는 허용. 원 구조의 축소판만 가능 |
| Eve 실로봇 평가 | 불가 | 하드웨어 필요 (`EgoMimic-Eve`) |

**노력 추정 (측정 전):** 정규화와 행동 표현 검사 CPU 구현 1–2일. 소형 정책 비교 1–2주. 공식 코드를 MPS로 포팅해 학습을 재현하는 일은 대략 1–2주의 엔지니어링과 데이터 다운로드가 필요하며, 논문 규모 학습은 A40 4대 약 24시간 (논문 기준)입니다.
