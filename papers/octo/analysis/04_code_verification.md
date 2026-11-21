# 04. 논문 ↔ 공식 코드 정적 대조 (Octo)

## 0. 방법과 범위

- **대상:** `https://github.com/octo-models/octo`, 커밋 `241fb35`(2024-07-30 작성, 2024-07-31 push). `repo/`에 얕게 클론했습니다(`--depth 1`). 라이선스는 MIT입니다.
- **방법:** 파일을 읽고 `grep`으로 설정값을 찾았습니다. **코드를 실행하거나 import하거나 설치하지 않았습니다.** 체크포인트, 데이터, 외부 패키지는 받지 않았습니다. 저장소에 있는 테스트용 TFRecord(`tests/debug_dataset`)는 열지 않았습니다.
- **한계:** 정적 대조는 설정과 구조가 같은지만 봅니다. 실제로 같은 결과가 나오는지는 확인하지 않았습니다. 공개 설정이 논문의 최종 학습 설정과 같은지는 저자만 답할 수 있습니다.

## 1. 일치하는 항목

| 항목 | 논문 | 코드 (파일:줄) | 판정 |
|---|---|---|---|
| 최대 학습률, 워밍업, 감쇠 | 3e-4, 2000, 역제곱근 (표 IV) | `scripts/configs/config.py:26–30` (peak 3e-4, warmup 2000, rsqrt, timescale 10000) | 일치 |
| 가중치 감쇠, 기울기 클리핑 | 0.1, 1.0 (표 IV) | `config.py:33–34` | 일치 |
| 확산 헤드 구조 | 3층 잔차 MLP, 은닉 256, 층 정규화 (부록 D) | `octo/model/components/action_heads.py:405–411` (`num_blocks=3`, `hidden_dim=256`), `diffusion.py:89, 106` (LayerNorm, 잔차) | 일치 |
| 확산 단계와 스케줄 | 20단계, 코사인 (부록 D) | `action_heads.py:411, 429` (`diffusion_steps=20`, `cosine_beta_schedule`) | 일치 |
| 이미지 해상도와 토큰 수 | 3인칭 256×256 → 256 토큰, 손목 128×128 → 64 토큰, 패치 16 (부록 D) | `octo_pretrain_config.py:99–100` (256, 128), `:36, 42` (`SmallStem16`, 패치 16) | 일치 |
| 언어 인코더 | t5-base, 동결, 16 토큰 (부록 D, 부록 E) | `octo_pretrain_config.py:48–49` (`finetune_encoder=False`), `:112` (`frozen_keys=("*hf_model*",)`), `:138–146` (`max_length 16`) | 일치 |
| 관측 이력 | 2프레임 (§IV-D) | `octo_pretrain_config.py:110` (`window_size=2`) | 일치 |
| 읽기 토큰 수 (행동) | 행동당 읽기 토큰 (그림 2, §III-A) | `octo_pretrain_config.py:53` (`readouts = {"action": 1}`) | 일치 |
| 부록 E: 궤적당 샘플 수 제한 | 한 궤적에서 최대 100 스텝 (부록 E) | `config.py:127` (`subsample_length=100`) | 일치 |
| 부록 E: 셔플 버퍼 | 최대 500k (부록 E) | `octo_pretrain_config.py:133` (`shuffle_buffer_size=500000`) | 일치 |
| 사전학습 스텝 | 300k (부록 D) | `octo_pretrain_config.py:109` (`num_steps=300000`) | 일치 |
| 미세조정 스텝 | 50k (§IV-D) | `scripts/configs/finetune_config.py:50` (`max_steps = 50000`) | 일치 |
| 모델 크기 (ViT-S, ViT-B) | 12층, MLP 1536/3072, 헤드 6/12 (표 V) | `octo/model/components/transformer.py:274–283` | 일치 (은닉 크기 384/768은 확인 생략) |
| 공개 체크포인트 크기 | 27M, 93M (§III-E) | README 표 (`README.md:61–62`) | 일치 (이름은 D2) |

## 2. 차이와 빈칸 (D1–D14)

| 번호 | 논문 | 코드 | 판정 |
|---|---|---|---|
| **D1** | 사전학습 배치 2048 (부록 D, 표 IV) | 공개 사전학습 설정 `batch_size=512` (`octo_pretrain_config.py:132`). 기본 설정도 512 (`config.py:154`). `scripts/train.py:156`이 프로세스 수로 나눔 | **불일치.** 공개 설정은 논문 배치의 1/4입니다. 논문의 2048이 전체 배치인지 기기당인지 설명이 필요합니다(§5 Q1) |
| **D2** | 체크포인트 `hf://rail-berkeley/octo-base`, `octo-small` (부록 B, 표 V) | README는 `octo-base-1.5`, `octo-small-1.5` 사용 (`README.md:18, 51, 61–62`) | **버전 차이.** 논문 결과와 1.5 체크포인트의 결과가 같다는 보장이 없습니다(§5 Q2) |
| **D3** | 행동 청크 크기 언급 없음 (부록 E는 "행동 청크"만 말함) | 사전학습 `action_horizon=4` (`octo_pretrain_config.py:58, 123`). README: "action chunking size of 4" (`README.md:148`). 미세조정 기본 설정은 `action_horizon=1` (`config.py:94, 125`) | **논문 본문에 빠진 숫자.** 본문만 보면 청크 길이를 알 수 없습니다. ALOHA 설정은 청크 64, 12 실행으로 따로 적혀 있습니다(부록 F) |
| **D4** | 확산 헤드 설명 | `action_heads.py:386–` (`DiffusionActionHead`). 확산 헤드의 드롭아웃을 끔 (README `README.md:154`) | 논문에 없는 세부 사항 (드롭아웃 끔) |
| **D5** | 블록 단위 마스크로 관측과 과제 토큰 구분 (§III-A, 그림 2) | `octo/model/octo_module.py`, `components/block_transformer.py`에 구현. 이번 정적 읽기에서는 마스크 규칙을 전부 확인하지 않음 | 확인 생략 (주된 구조는 README와 일치) |
| **D6** | 데이터 증강: 3인칭은 무작위 자르기 후 크기 조정, 색 지터. 손목은 자르기 없이 128로 조정 (부록 D) | `octo_pretrain_config.py:99–100`과 증강 설정 (세부는 이번 읽기에서 생략) | 일치 가능 (세부 생략) |
| **D7** | 학습 과제 문장: 학습 예시마다 언어나 목표 이미지를 0으로 가림 (§III-B) | `octo_pretrain_config.py:110–126`: `task_augment_strategy="delete_and_rephrase"`, `rephrase_prob=0.5`, 패러프레이즈 저장소 `rail-berkeley/OXE_paraphrases` | **논문에 없는 세부.** 말을 바꾸는 패러프레이즈 증강이 쓰입니다 |
| **D8** | 목표 이미지는 같은 궤적의 나중 상태 (hindsight relabeling, §III-B) | `octo/data/utils/goal_relabeling.py` (이번 읽기에서 세부 생략) | 확인 생략 |
| **D9** | 25개 데이터셋, 이미지 있음과 델타 엔드이펙터만 사용 (§III-B, 표 III) | `octo/data/oxe/oxe_dataset_mixes.py`의 `OXE_MAGIC_SOUP`(활성 항목 25개), 공개 설정 `data_mix="oxe_magic_soup"` (`octo_pretrain_config.py:116`) | 일치 (개수 25) |
| **D10** | 표 III의 최종 비율 (Fractal 17.0%, Kuka 17.0%, Bridge 17.0% …) | 공개 코드는 **샘플링 가중치**를 씀 (Fractal 0.54, Kuka 0.83, Bridge 1.0, Taco 2.0, NYU Franka 3.0, UCSD 2.0 등). `balance_weights=True` (`octo_pretrain_config.py:134`). 최종 비율은 데이터 크기 메타가 있어야 계산됨 | **정적으로 재현 불가.** 표현 방식이 다르고, UCSD는 가중치 2.0인데 표 III는 `<0.1%`입니다. 데이터 크기 차이 때문일 수 있음(추정) |
| **D11** | 미세조정: 같은 레시피, 전체 모델 갱신, 학습률 미기재 (§IV-D) | `scripts/configs/finetune_config.py`: 모드 `full`, `head_only`, `head_mlp_only` 선택. 학습률 peak 3e-4 (`:75`) | 논문 방식(full)은 옵션으로 있음. 학습률은 논문에 없는 숫자 |
| **D12** | Octo-Small, Base 크기 (표 V) | `transformer.py:274–283`. 기본 `transformer_size="vit_s"` (`config.py:13`) | 일치 |
| **D13** | Octo-Tiny(10M), 그림 6에만 있음 | `transformer.py`에 `vit_t`(12층, MLP 768, 헤드 3)가 있으나 Tiny 설정, 체크포인트, 학습 설정은 없음 | **공개되지 않음.** 구조 정의만 존재 |
| **D14** | "OXE 데이터셋을 사용" (§III-B) | 사전학습 설정이 저자 버킷의 재처리본을 씀: `gs://rail-orca-central2/resize_256_256` (`octo_pretrain_config.py:117`) | **데이터 접근성.** 공개 OXE 원본과 같은 전처리인지는 코드만으로 알 수 없음(§5 Q3) |

## 3. 기본 설정 파일의 흔적

- `scripts/configs/config.py:87–95`의 기본 모델 설정은 `MSEActionHead`를 씁니다. 사전학습 설정은 이를 `DiffusionActionHead`로 바꿉니다(`octo_pretrain_config.py:54–59`). 공개된 Octo 학습은 확산 헤드를 쓰므로 논문과 맞습니다. 다만 기본 설정이 MSE라서 다른 용도로 쓸 때 헷갈릴 수 있습니다.

## 4. 구현 가능성 판정

**판정: 이 맥(Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)에서 Octo의 추론과 구조 검산은 가능합니다. 미세조정과 사전학습 재현은 불가능합니다. 논문 수치(표 I, 표 VI)의 재현도 불가능합니다.**

| 작업 | 판정 | 근거와 추정 |
|---|---|---|
| 공개 체크포인트 추론 (Small 27M, Base 93M) | **조건부 가능** | 메모리는 충분합니다(93M 파라미터). 다만 공식 코드는 **JAX + TensorFlow** 기반입니다. JAX의 Apple GPU(MPS) 경로는 공식 지원이 아니므로 CPU 백엔드가 현실적입니다(추정, 미확인). 체크포인트 다운로드와 새 가상환경 설치는 사용자 승인이 필요합니다 |
| 구조 검산 (확산 헤드 한 스텝, 토큰화, 블록 마스크) | 가능 | 소형 설정과 임의 입력으로 계산 흐름을 확인할 수 있습니다. 추정 1–2일 |
| 미세조정 (Small 또는 Base, 50k 스텝, 약 100개 시연) | **불가** | 논문은 A5000 한 장에서 Base를 약 5시간 학습했다고 합니다(§IV-D). 맥 CPU로는 현실적이지 않습니다(추정). 시연 데이터도 없습니다 |
| 사전학습 (OXE 80만 에피소드, 300k 스텝, 배치 2048) | **불가** | TPU v4-128에서 14시간이 필요했습니다. 저자 버킷 데이터(D14)에도 접근해야 합니다 |
| 논문 표 I 재현 (실로봇 6개 과제) | **불가** | 로봇, 시연 데이터, 평가 환경이 없습니다 |
| 논문 표 VI 재현 (WidowX 실로봇 평가) | **불가** | 실로봇 평가이고, 공개된 평가 설정도 충분하지 않습니다 |
| ALOHA 시뮬레이션 (`examples/envs/aloha_sim_env.py`) | **조건부** | 환경 코드는 있으나 Bimanual 미세조정 가중치가 공개되지 않아 의미 있는 평가는 어렵습니다(추정) |
| CPU toy (확산 대 MSE 헤드의 다중 모드 차이) | 가능 | 논문의 중심 주장은 아니므로 이번 분석에서는 만들지 않았습니다. 필요하면 1–2시간 정도 |

**노력 추정 (추정):** 추론과 구조 검산까지 1–3일, 공식 코드 실행 환경을 맞추는 데 1–2일입니다. 미세조정 재현은 CUDA 기기가 필요하므로 이 맥의 범위를 벗어납니다.

## 5. 저자에게 물어볼 질문

1. **Q1 (D1):** 사전학습 배치 2048이 전체 배치인지 기기당인지. 공개 설정 512와 어떤 관계인지.
2. **Q2 (D2):** 논문 표 I, VI의 결과가 `octo-small`, `octo-base`(v1 체크포인트)에서 나왔는지, `-1.5` 체크포인트와 같은지.
3. **Q3 (D14):** 사전학습에 쓴 `resize_256_256` 재처리본과 전처리 코드를 공개할 계획이 있는지.
4. **Q4 (D3):** 사전학습 행동 청크 4가 논문 본문에 왜 없는지. 평가 때 실행 개수는 몇 개인지.
5. **Q5 (D10):** 표 III 비율을 만든 샘플링 과정과 데이터 크기 기준.
