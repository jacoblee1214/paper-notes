# 04. 공식 코드 대조: 논문 ↔ 저장소 (D1–D21)

대상: [thu-ml/RoboticsDiffusionTransformer](https://github.com/thu-ml/RoboticsDiffusionTransformer), 커밋 `cd79363`(2026-01-21), 얕은 복제 `repo/`. **정적으로만 읽었습니다.** 실행, 설치, import, 가중치와 데이터 다운로드는 하지 않았습니다. 저장소 내용은 신뢰하지 않는 데이터로 취급합니다.

## 1. 라이선스와 저장소 상태

- `LICENSE`: MIT (TSAIL group, 2024). HF 체크포인트 카드도 MIT입니다. 저장소 설명과 API가 같은 값을 줍니다.
- 논문 PDF의 라이선스는 arXiv 비독점 배포이고 CC BY가 아닙니다(`01` §0). 코드 라이선스는 별개이며 MIT로 허용됩니다. 그래도 논문 그림은 공개본에 넣지 않습니다.
- 체크포인트 `pytorch_model.bin` 약 2.46 GB는 받지 않았습니다(`02` §2).

## 2. 설정과 구조 대조

판정: **일치** = 논문과 코드가 같은 값, **차이** = 값이 다름 또는 문서 사이 모순, **미확인** = 코드에서 확인할 수 없음.

| ID | 항목 | 논문 | 코드 (근거) | 판정 |
|---|---|---|---|---|
| D1 | 파인튜닝 스텝 | 130K 스텝, 약 3일 (§5.1) | `finetune.sh`: `--max_train_steps=200000`. 실제로 몇 스텝을 돌렸는지는 저장소에 없음 | **차이** |
| D2 | 파인튜닝 시작 체크포인트 | 500K 사전학습 체크포인트 (부록 H: "scheduling reasons") | `finetune.sh`: `--pretrained_model_name_or_path="robotics-diffusion-transformer/rdt-1b"`. README는 이 체크포인트를 1M 스텝으로 설명 | **차이** (문서 간 불일치) |
| D3 | 사전학습 최적화 | 1M 스텝, 배치 32×48, lr 1e-4 상수, warm-up 500, AdamW (β 0.9/0.999, wd 1e-2, ε 1e-8), bf16 (표 10) | `pretrain.sh`: `--train_batch_size=32 --max_train_steps=1000000 --learning_rate=1e-4 --lr_scheduler=constant --mixed_precision=bf16`. `main.py` 기본 warm-up 500, Adam 기본값 동일. GPU 수는 `hostfile.txt`가 저장소에 없어 확인 불가 | 일치 (GPU 수 미확인) |
| D4 | `main.py` 기본 lr | 논문에 없음 | `main.py` 기본 `--learning_rate` = 5e-6. 두 스크립트가 1e-4로 덮어씀 | 일치 (스크립트 기준). 기본값은 다름, 주의 |
| D5 | 확산 스케줄 | DDPM 학습 1000 스텝, squaredcos_cap_v2, 추론 DPM-Solver++ 5 스텝 (부록 H) | `configs/base.yaml`: `num_train_timesteps: 1000`, `beta_schedule: squaredcos_cap_v2`, `num_inference_timesteps: 5`, `prediction_type: sample`. `models/rdt_runner.py`: DDPMScheduler와 DPMSolverMultistepScheduler | 일치 |
| D6 | 모델 크기 | 깊이 28, 은닉 2048, 헤드 32, 1.2B (표 9) | `base.yaml`: `depth: 28`, `hidden_size: 2048`, `num_heads: 32`. `models/rdt/blocks.py`의 `RDTBlock`에서 FFN 은닉 = 은닉 크기(비율 1) | 일치. 파라미터 수 약 10d² × 28 ≈ 1.17B, 1.2B와 맞음 (정적 계산, 실행 안 함) |
| D7 | QKNorm과 RMSNorm | QKNorm, LayerNorm 대신 RMSNorm (§4.1) | `blocks.py`: `RmsNorm`, `qk_norm=True` (자기·교차 어텐션) | 일치 |
| D8 | MLP 디코더 | 마지막 선형 디코더를 MLP로 (§4.1) | `blocks.py` `FinalLayer.ffn_final = Mlp(...)` | 일치 |
| D9 | 교대 조건 주입 (ACI) | 한 층은 이미지, 다음 층은 언어 (§4.1) | `models/rdt/model.py`: `c, mask = conds[i%2], masks[i%2]` | 일치. 다만 `model.py`의 주석 "Inject the language condition at the final layer"는 최종 층에 조건을 주지 않는 코드와 어긋남(주석 오류로 보임) |
| D10 | 어댑터와 토큰 차원 | 언어 4096, 이미지 1152, 토큰 2048, 언어·이미지 2층 MLP, 행동 3층 MLP (표 8) | `base.yaml`: `lang_token_dim: 4096`, `img_token_dim: 1152`, `lang_adaptor: mlp2x_gelu`, `img_adaptor: mlp2x_gelu`, `state_adaptor: mlp3x_gelu` | 일치 |
| D11 | 청크 길이, 이미지 이력 | 청크 64, 이미지 이력 2 (부록 H) | `base.yaml`: `action_chunk_size: 64`, `img_history_size: 2`, `num_cameras: 3` | 일치 |
| D12 | 입력 모달리티 마스크 | 학습 중 각 입력 독립 10% 마스크 (§4.1) | `main.py`/`dataset.py`: `cond_mask_prob=0.1`이며 `base.yaml`에는 없음. 마스크 대상은 상태 평균 대체, 제어 주파수 0, 이미지 마스크, 언어 빈 임베딩 | 일치 |
| D13 | 이미지 증강 | 색상 지터, 이미지 손상, 파인튜닝 (부록 F) | `finetune.sh`: `--image_aug`. `train/image_corrupt.py`와 `dataset.py`의 색상·손상 증강 | 일치 (파인튜닝 전용) |
| D14 | proprioception 노이즈 | proprioception에 SNR 40 dB 가우시안 (부록 F) | `finetune.sh`: `--state_noise_snr=40`. `dataset.py`: 노이즈 표준편차 `state_std / sqrt(10^(SNR/10))`. `main.py` 기본값은 None이지만 파인튜닝 스크립트는 설정 | 일치 (파인튜닝 전용) |
| D15 | 데이터 길이 필터 | 32 미만 제외, 2048 초과는 2048로 줄임 (부록 H) | `base.yaml`: `epsd_len_thresh_low: 32`, `epsd_len_thresh_high: 2048` | 일치 |
| D16 | 지수이동평균 (EMA) | 논문 본문에 언급 없음 | `base.yaml`: "We do not use EMA currently" (설정은 있으나 비활성) | 논문과 무관 (정보) |
| D17 | 언어·이미지 인코더 | T5-XXL, SigLIP, 학습 중 고정 (§4.1, 표 8) | `pretrain.sh`·`finetune.sh`: `google/t5-v1_1-xxl`, `google/siglip-so400m-patch14-384`. SigLIP의 `unfreeze_mm_vision_tower` 기본값 False. T5 동결 여부는 옵티마이저 대상에서 확인하지 못함 | 일치 (T5 동결은 미확인) |

## 3. 데이터와 행동 공간

| ID | 항목 | 논문 | 코드 (근거) | 판정 |
|---|---|---|---|---|
| D18 | 통합 행동 공간 128차원 | 128차원, 표 4 구간 (부록 C) | `configs/state_vec.py`: `STATE_VEC_LEN = 128`. 표 4 구간을 모두 대조함. 관절 [0,10), 그리퍼 [10,15), 관절 속도 [15,25), 그리퍼 속도 [25,30), 말단 위치 [30,33), 6D 자세 [33,39), 말단 속도 [39,42), 말단 각속도 [42,45), 왼팔은 [50,...), 베이스 [100,102), 각속도 [102], 예약 구간 비어 있음 | 일치 (정적 대조, 항목별 이름 확인) |
| D19 | 사전학습 데이터 가중치 | 46개 데이터셋, 표 5 샘플 비율 (합 100.01%) | `configs/pretrain_sample_weights.json`: 46개, 합 3012. 비율로 바꾼 값을 표 5 값과 정렬해 비교하면 **불일치 0** (`analysis/recompute.py`). 항목별 이름 1:1 대응은 전수 확인하지 않음. 예: fractal 271/3012 = 9.00% = RT-1 | 일치 (정렬 비교, 이름 대응은 일부) |
| D20 | 제어 주파수 예시 | 부록 E: 3 Hz, 15 Hz, 10 Hz (예시) | `configs/dataset_control_freq.json`: fractal 3, taco_play 15, jaco_play 10 | 일치 (예시 3개만 확인) |
| D21 | 실로봇 추론 배포 | 온보드 RTX 4090 24GB, 6 Hz 청크, 381 Hz 행동 (§5.1) | `scripts/agilex_inference.py`, `inference.sh`. README 주의: RTX 4090 이하에서는 `t5-v1_1-xxl`을 올리기 어려워 **언어 임베딩 사전계산**을 권함 (`scripts/encode_lang.py`). 논문은 온보드 추론에서 언어 인코딩 방식을 쓰지 않음 | **차이 (문서 불일치)**: 논문은 온보드에서 돌렸다고 쓰고, README는 사전계산을 전제함 |

## 4. 구현 가능성 판정 (이 맥 기준, 측정 전)

이 맥은 Apple M4 Pro, 약 24 GiB 통합 메모리, MPS, CUDA 없음입니다(메모리 파일 기준).

| 작업 | 판정 | 근거 |
|---|---|---|
| 1.2B 사전학습 | **불가** | 48 H100 × 1개월 (§5.1). CUDA 전제 (`pretrain.sh`의 NCCL, `cutlass`, `zero2.json`). 1.2B 전체 학습의 가중치, 기울기, Adam 상태만 fp32 기준 약 19 GB이고 활성값은 별도(맥 24 GiB에 빠듯함) |
| 파인튜닝 (전체) | **불가** | 48 H100 × 3일 (§5.1). 같은 이유 |
| 파인튜닝 (LoRA 등 부분 학습) | **조건부 (미측정)** | 코드에 LoRA 경로 없음. 새로 짜야 함. 메모리와 속도 측정 전 |
| 추론 (5 스텝, 청크 64) | **조건부 (미측정)** | 1.2B bf16 가중치 약 2.4 GB, SigLIP 약 0.4B(고정), T5-XXL 인코더는 맥 메모리에 빠듯해서 **언어 임베딩을 미리 계산**해야 함 (D21). MPS에서 bf16 지원과 속도 미확인. 설치와 체크포인트 다운로드(2.46 GB)는 승인 필요 |
| 구조 재구현 (torch, 작은 크기) | **가능 (CPU, 짧은 검사)** | 블록, 조건 교대 주입, RMSNorm, QKNorm은 작은 차원으로 CPU에서 확인 가능 |
| 실로봇 재현 (ALOHA) | **불가** | 하드웨어와 데이터 필요 (D22, D25) |
| 시뮬 재현 (ManiSkill) | **조건부** | 논문 수치 없음. 평가 스크립트만 있음. 설치와 시뮬 자산 필요 |

**노력 추정 (측정 전, 추정).** 추론 경로만 맞추는 데 1–3일(언어 임베딩 사전계산, MPS 점검, 체크포인트 로드). 구조 재구현은 1주. 파인튜닝 재현(부분)은 2주 이상이며 맥에서는 메모리 점검이 먼저입니다.

## 5. 평가와 빠진 것

| ID | 항목 | 논문 | 코드 | 판정 |
|---|---|---|---|---|
| D22 | 실로봇 평가 (표 3) | 과제별 시도 8 또는 25회, 성공률 (§5.1) | 실로봇 평가 자동화 스크립트 없음. 배포 스크립트만 있음 | **미공개** |
| D23 | 베이스라인 실험 설정 | ACT: 8000 에폭 (표 11), OpenVLA: LoRA (표 12), Octo (표 13) | `eval_sim/eval_openvla.py`, `eval_octo.py`, `eval_dp.py`는 ManiSkill 시뮬 평가. 자체 ALOHA 파인튜닝 설정은 저장소에 없음 | **미공개** (논문 베이스라인 재현 불가) |
| D24 | 시뮬 결과 | 논문 본문에 시뮬 성공률 없음 | `eval_sim/eval_rdt_maniskill.py`, README의 ManiSkill 명령. 이 저장소에 시뮬 결과 표는 없음 | 논문과 무관 (정보) |
| D25 | 자체 ALOHA 데이터 (6K+) | 6K+ 에피소드, 과제별 시연 수 (§5.1) | 데이터 없음. 변환 스크립트(`data/preprocess_scripts/agilex.py`)는 있음 | **미공개** (데이터 자체는 없음) |
| D26 | 주석 오류 | — | `models/rdt/model.py` 최종 층 주석 (D9 참고) | 정보 |

## 6. 종합

- **구조와 학습 설정은 대부분 일치합니다.** 깊이, 폭, 헤드, 확산 스케줄, 청크 길이, 마스크 확률, 128차원 행동 공간, 사전학습 데이터 가중치가 논문 표 값과 맞습니다(D3–D12, D15, D18–D19). 1.2B 파라미터 수는 정적 계산으로도 맞습니다(D6).
- **차이는 파인튜닝과 배포에 있습니다.** 파인튜닝 스텝(130K 대 200K 스크립트 값, D1), 시작 체크포인트(500K 대 1M, D2), 온보드 추론 방식(D21)입니다. 재현하려면 이 세 가지를 논문 값으로 맞추거나 명시적으로 고정해야 합니다.
- **논문 결과의 재현 조건이 빠져 있습니다.** 실로봇 평가 코드, 자체 데이터, 베이스라인 학습 설정이 없어 표 3 수치를 저장소만으로 다시 얻을 수 없습니다(D22–D25).
- **구현 난이도.** 구조는 약 1.2B 파라미터의 표준 PyTorch 코드입니다. 학습은 다중 GPU와 DeepSpeed를 전제합니다(`pretrain.sh`의 NCCL, `zero2.json`). 맥의 CUDA 없는 환경에서는 학습 경로가 맞지 않습니다(§4 판정).

## 7. 기록

- 대조에 쓴 파일: `repo/README.md`, `configs/base.yaml`, `configs/state_vec.py`, `configs/dataset_control_freq.json`, `configs/pretrain_sample_weights.json`, `models/rdt/blocks.py`, `models/rdt/model.py`, `models/rdt_runner.py`, `main.py`, `train/dataset.py`, `pretrain.sh`, `finetune.sh`, `scripts/agilex_inference.py`.
- 정적 계산: 블록 파라미터 수 10d² × 28 (`01` §2.3, D6).
- 실행하지 않은 것: 학습, 추론, 설치, 가중치 다운로드, 시뮬 평가.
