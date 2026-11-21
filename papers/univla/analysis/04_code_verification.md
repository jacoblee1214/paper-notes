# 04. 논문 ↔ 공식 코드 정적 대조와 구현 가능성

대상: `github.com/OpenDriveLab/UniVLA`, 커밋 `0ab9e9d` (2025-11-19). `univla/repo/`에 얕게 clone했고, 파일을 읽기만 했습니다. 실행, import, 패키지 설치, 가중치 다운로드는 하지 않았습니다. 코드 줄 번호는 이 커밋 기준입니다. 논문 줄은 `01`의 절 번호로 씁니다.

## 1. 대조 범위

| 영역 | 파일 | 대조 내용 |
|---|---|---|
| 잠재 행동 모델 (LAM) | `latent_action_model/genie/modules/lam.py`, `genie/model.py`, `config/lam-stage-*.yaml`, `genie/dataset.py` | 입력 쌍, DINOv2, 코드북, 지시문, 손실, 2단계 고정 |
| 데이터 혼합과 프레임 간격 | `prismatic/vla/datasets/rlds/oxe/mixtures.py`, `rlds/dataset.py`, `datasets.py` | Table A-I, 1초 간격 |
| 사전학습 설정 | `prismatic/conf/vla.py` | 배치, 학습률, 혼합 |
| 미세조정 (LIBERO) | `vla-scripts/finetune_libero.py` | 디코더, LoRA, 학습률, 스텝, 배치 |
| 평가 (LIBERO) | `experiments/robot/libero/run_libero_eval.py` | 청크 앙상블, 시행 수, seed |
| 디코더 구조 | `prismatic/models/policy/transformer_utils.py` | MAPBlock, SwishGLU, RMSNorm |
| README | `README.md` | 결과 표, 버그 수정 공지, 체크포인트 목록, 하드웨어 |

## 2. 대조표

| 논문 | 코드 | 일치 |
|---|---|---|
| 연속 두 프레임 `{o_t, o_{t+k}}` [Sec. III-A] | 두 프레임 쌍 (`lam.py` 111행, 249행 `T=2`). 라벨링도 두 프레임 (`datasets.py` 112행) | 일치 |
| 프레임 간격 약 1초 [Sec. III-A] | 데이터셋 주파수별 보정 (`rlds/dataset.py` 359–366행) | 일치 |
| DINOv2 패치 특징 예측 [Sec. III-A] | `dinov2_vitb14_reg` (ViT-B/14 with registers), 동결 (43행) | 일치 (레지스터 토큰 포함은 논문에 없음) |
| 1단계 T5 지시문, 인코더와 디코더 모두 [Sec. III-A] | `T5EncoderModel("./t5-base")`, 인코더와 디코더에 투입 (81행, 139–151행) | 일치 |
| 1단계 코드북 |C| = 16 [Sec. III-B] | `num_latents=16` (lam-stage-1 설정 값, `lam.py` 62–64행) | 일치 |
| 행동 토큰 N = 4 [Sec. III-B] | `num_codes = 4` (48행) | 일치 |
| 2단계: 1단계 코드북 고정, 새 코드북 학습, 지시문 없음 [Sec. III-A] | `self.vq.requires_grad_(False)` (240행), `vq_action` 새로 초기화 (231행), 2단계 인코더 호출 `self.encoder(padded_patches)`에 지시문 없음 (257행) | 일치 |
| 2단계 디코더 입력 `[O_t; ã_TI; ã_TC]` [Sec. III-A] | `[action_up(z_q), action_up_uncontrol(z_q_uncontrol), video_patches]` (294–299행) | 일치 |
| 1단계 디코더는 과거 프레임을 받지 않음 [Sec. III-A 문장] | **받음**: `video_patches = patch_up(patches[:, :-1])`가 디코더 입력에 들어감 (146–151행). 식의 `F([O_t; ã; ℓ])`과 같음 | **불일치 (D2)** |
| 정책 학습 라벨 = 과제 중심 코드 [Sec. III-B] | 라벨은 LAM `indices` (`datasets.py` 112행). 2단계 모델에서 `indices`는 `vq_action`(TC) 출력 (283행) | 일치 (코드 흐름으로 추정) |
| 사전학습: 32 A100, 배치 1024, 학습률 2e-5 상수 [App. A-1] | `expected_world_size=32`, `global_batch_size=1024`, `per_device=32`, 기본 학습률 2e-5 상수 (`conf/vla.py` 80–125행) | 일치 |
| 사전학습 데이터 "OpenX + Human" [App. A-1] | 기본 설정 `omni_magic_soup_plus` (Human 없음). Human 포함 설정은 주석 처리 (`conf/vla.py` 120–121행) | **기본값 불일치 (D10)** |
| 디코더 청크 12 [Sec. III-C] | `window_size = 12`, 출력 `7 × 12` | 일치 |
| 디코더 구조: MAP 풀링 두 개, 헤드 8, 512, MLP 비율 4 [Tab. A-VI] | `ActionDecoder`: 2개 MAPBlock, `vis_dim=4096`, `n_heads = 512/64 = 8`, SwishGLU | 일치 |
| 디코더 12.6M [Sec. III-C, Tab. A-VI] / 10.8M [서론] | 정적 계산 12,640,340 (`03` V12) | 본문·표 일치, **서론 불일치 (D3)** |
| LoRA 사용 [Sec. III-C] | `LoraConfig(r=32, lora_alpha=min(r,16), target all-linear, init gaussian)` (213–222행) | 일치 (rank·alpha는 논문에 없음) |
| LIBERO 스텝: 40k (Long), 30k (나머지) [Sec. IV-A1] | `max_steps=30000`, README "Long 40k" 명시 | 일치 |
| LIBERO 전역 배치 128 [Sec. IV-A1] | `batch_size=8 × grad_accum=2 × 8 GPU = 128` (116–120행) | 일치 |
| 학습률 (미세조정) | 3.5e-4, AdamW (wd 1e-3), StepLR 0.1배 at 80% (119행, 239–240행) | 논문에 없음 (D11) |
| 청크 앙상블 | 지수 가중 앙상블, balancing 0.1 (`run_libero_eval.py` 131–157행) | **논문에 없음 (D8)** |
| LIBERO 500회 (suite당 50 × 10), seed 7 [README] | `num_trials_per_task=50`, `seed=7` (61–73행) | 시행 수 일치, "3 seed" 평균은 코드에 없음 (D9) |
| LAM 1단계 설정 (배치, 학습률, epoch) | 학습률 1e-4 AdamW, 배치 64 × 8 GPU, 20 epoch, 16-bit mixed (lam-stage-1.yaml) | 논문에 없음 (D12) |

## 3. 불일치와 빈칸

### D1. SimplerEnv 전체 평균 (본문 42.7 대 표 47.9)

- 공식 README에서 UniVLA 행은 표 A-III의 최신 값(47.9)과 같습니다.
- README에는 "입력 처리 버그를 고쳐 UniVLA 결과가 원 논문 수치보다 높다"는 공지가 있고, 그 바로 아래 표가 SimplerEnv 표입니다.
- 따라서 (a) 본문 42.7은 수정 전 수치, (b) 표 47.9는 수정 후 수치일 가능성이 큽니다. 논문 v3는 표만 고치고 본문은 그대로 두었습니다(`01` §1, `03` V8).
- 질문 (저자에게): 버그의 내용과 수정 전후의 전체 결과를 알고 싶습니다. 수정이 LIBERO와 실로봇 결과에도 영향을 주었는지도 묻습니다.

### D2. 1단계 디코더의 과거 프레임 입력

- 논문 본문: "디코더는 과거 프레임을 받지 않는다" [Sec. III-A].
- 같은 절의 식: `Ô_{t+k} = F([O_t; ã_TI; ℓ])`. 여기에는 `O_t`가 들어 있습니다.
- 코드: `O_t` 패치가 디코더에 들어갑니다 (`lam.py` 146–151행). 식과 같고 본문 문장과는 다릅니다.
- 판정: 논문 내부 모순. 코드는 식을 따릅니다. 본문 문장이 "과거 프레임 전체의 기억을 막는다"는 뜻이었다면, 이는 단일 프레임 조건이라는 뜻입니다.

### D3. 디코더 파라미터 수

- 서론 10.8M, 본문 §III-C와 표 A-VI 12.6M.
- 정적 계산 12.64M (`03` V12). 본문과 표 A-VI 쪽이 맞습니다. 서론 값만 틀렸습니다.

### D4. 2단계 학습 체크포인트 이름

- `lam-stage-2.yaml`은 `stage_one_ckpt: epoch=0-step=100000.ckpt`를 가리킵니다.
- LIBERO 미세조정 설정 `finetune_libero.py` 108행은 `epoch=0-step=200000.ckpt`(2단계)를 씁니다.
- 1단계 체크포인트는 20k마다 저장되므로 100k는 가능합니다. 어느 체크포인트를 논문 결과에 썼는지는 논문에 없습니다. 판정: 설정만 확인, 결과 대응은 불명확.

### D5. 입력 해상도와 `num_frames`

- LAM 설정 `resolution: 224`, `num_frames: 16 # TODO`. 실제 `vq_encode`는 `T=2`로 재배열하므로 16은 쓰이지 않는 것으로 보입니다.
- 판정: 설정 파일의 잔여 항목. 결과에 영향은 없어 보입니다.

### D6. 인코더 개수, 인과 마스크

- 논문: 공간-시간 트랜스포머, 인과 시간 마스크 [Sec. III-A].
- 코드: `SpatioTemporalTransformer(..., causal_temporal=True)` (`lam.py` 53–60행). 일치.

### D7. 이력 잠재 행동 (4 토큰)

- 평가 코드는 직전 4개 잠재 행동 토큰 묶음을 프롬프트에 씁니다 (`run_libero_eval.py` 확인 범위 내, `start_idx` 최대 4). 논문과 일치합니다.

### D8. 청크 앙상블 (논문에 없음)

- 평가 코드의 디코더는 청크 12를 시간 가중 평균으로 합칩니다 (`run_libero_eval.py` 119–160행, 지수 가중치 `exp(-0.1 i)`). 주석은 "temporal aggregation"입니다.
- 논문은 청크를 쓴다고만 하고 앙상블은 언급하지 않습니다. 이 앙상블 유무가 실로봇(10Hz 폐루프)과 LIBERO 결과에 영향을 줄 수 있습니다. 어블레이션이 없습니다.

### D9. 3 seed 평균 (코드 대 논문)

- 논문: "3 seed 평균" [Sec. IV-A1].
- 코드: `seed=7` 고정, `set_seed_everywhere`. 세 seed를 돌리는 스크립트나 집계 코드는 저장소에 없습니다.
- README 표의 ± 값(예: UniVLA 96.5 ± 0.5)의 출처도 설명이 없습니다.

### D10. 사전학습 데이터 혼합

- 논문 Table A-I: 항목 26개 + GNM 3개 + Ego4D, 비율 합 약 100%.
- 코드 `omni_magic_soup_plus_plus`: 항목 33개, 가중치 합 40.2. 정규화 시 Fractal 1.2% (표 13.9%), NYU Franka 7.5% (표 1.0%), Ego4D 4개 합 10.0% (표 3.0%). GNM 합 14.9% (표 14.7%)는 가깝습니다(`03` V11).
- 표 A-I가 `balance_weights`(데이터셋 크기 곱) 방식으로 만들어졌을 가능성이 있어 직접 비교할 수 없습니다.
- 기본 사전학습 설정이 Human 데이터를 뺀 `omni_magic_soup_plus`라는 점도 D10의 일부입니다. 논문 주 결과가 Human 포함 모델이라면, 공개 기본값은 그 설정이 아닙니다.

### D11. 미세조정 학습률과 LoRA alpha

- 3.5e-4, AdamW (weight decay 1e-3), StepLR (80%, 0.1배), LoRA r = 32, alpha = min(r, 16). 논문에 없습니다.
- 학습률 3.5e-4는 LoRA에서 큰 편이라 설정 민감도가 있을 수 있습니다.

### D12. LAM 학습 설정

- 1단계: AdamW 1e-4, weight decay 1e-2, 배치 64 × 8 GPU, 최대 20 epoch, 16-bit mixed, 그래디언트 클리핑 0.1.
- 논문에는 LAM 학습 스텝과 배치가 없습니다.

### D13. README의 SimplerEnv 시행 설명

- README: "Averaged across 3 seeds" (SimplerEnv). 이 표의 ± 값은 있지만 시행 수 설명은 없습니다. 논문 본문의 24회/과제와 맞추면 과제당 72회입니다(`03` §1).

## 4. 구현 가능성 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

### 4.1 막히는 곳과 수치

- **학습 (재현 불가):**
  - 사전학습: 32 A100, 20k 스텝, 960 A100-hours (논문, README 확인). 이 맥에 CUDA가 없어 불가능합니다.
  - LAM: 8 GPU, 20 epoch, RLDS 대용량 데이터(`shuffle_buffer 45000`, "1,600 GB 메모리 기준" 주석). 불가능합니다.
  - 미세조정: 8 GPU, 30k–40k 스텝. 불가능합니다.
- **환경 장벽:** `requirements.txt`는 `tensorflow==2.15.0` (RLDS), `flash-attn==2.5.5` (CUDA 빌드 필요), `torch 2.2.0 + cuda 12.1`을 권장합니다 (README 158–169행).
- **평가 코드의 CUDA 전제:** `run_libero_eval.py` 185행에 `action_decoder.eval().cuda()`가 있고, 파일 안에 `.cuda()` 호출이 있습니다. MPS로 옮기려면 수정이 필요합니다. 7B 모델의 bfloat16 추론이 MPS에서 가능한지는 측정하지 않았습니다.
- **데이터:** LIBERO RLDS(openvla/modified_libero_rlds), 가중치(qwbu/univla-7b 등, 약 15 GB 추정)를 받아야 합니다. 다운로드는 승인 전이라 하지 않았습니다.

### 4.2 세부 판단

- 잠재 행동 모델의 **추론**(라벨링)은 DINOv2와 T5-base, LAM 가중치만 있으면 작은 규모로 가능할 수 있습니다. 측정하지 않았으므로 확인은 필요합니다.
- 정책 **추론**(7B)은 MPS에서 가능성이 있으나 시간이 필요합니다. 측정 전이며, 실행 시간 추정은 근거가 없습니다.
- 논문 수치를 **재현**하려면 GPU 클러스터가 필요합니다.

### 4.3 노력 추정 (근거 없는 추정, 측정 전)

- 공개 체크포인트로 LIBERO 한 suite 추론 확인 (MPS 포팅, 설치, 다운로드 포함): 1–2주.
- LAM 추론 확인과 잠재 행동 할당 시각화 (작은 샘플): 3–5일.
- 전체 재학습: GPU 클러스터 필요, 맥에서는 해당 없음.

## 5. 저자에게 물을 질문

1. SimplerEnv 결과에서 입력 처리 버그를 고친 내용과, 고치기 전후 전체 결과를 알려 주실 수 있나요? 본문 42.7과 표 47.9 중 어느 것이 최종 값인가요?
2. 1단계 디코더에 `O_t`를 넣는 것이 의도된 설계인가요? 본문 문장("과거 프레임을 받지 않는다")은 무엇을 뜻하나요?
3. 평가에 쓴 청크 시간 앙상블(지수 가중치 0.1)이 논문 결과에 포함됐나요? 빼면 결과가 얼마나 달라지나요?
4. LIBERO와 SimplerEnv의 "3 seed"는 학습 seed 세 개를 말하나요, 평가 seed를 말하나요? seed별 값을 공개할 수 있나요?
5. 실로봇 시행 수는 과제당 몇 회인가요? 표 A-V의 UniVLA (OXE) 평균 63.3은 어떤 값을 옮긴 것인가요?
6. Table A-I의 비율은 `balance_weights` 방식으로 계산했나요? Ego4D 비율(3.0%)은 어떻게 나왔나요?
7. 사전학습 기본 설정(`omni_magic_soup_plus`)과 논문 주 결과(Human 포함) 중 어느 쪽이 실제 체크포인트 `univla-7b`에 쓰였나요?
8. MaIL 평균 83.5는 어떤 suite 값으로 계산했나요?

## 6. 요약 판정

- 코드와 논문은 전반적으로 맞습니다. 모델 구조, 코드북 수, 토큰 수, 청크 크기, 2단계 고정, 사전학습 배치와 학습률, 미세조정 스텝과 배치는 일치합니다.
- 불일치는 주로 **본문 문장과 식, 서론 수치, 버그 수정 후 표**에 있습니다(D1–D3).
- 논문에 없는 설정이 결과에 영향을 줄 수 있습니다: 청크 앙상블(D8), 미세조정 학습률과 LoRA alpha(D11), 3 seed 집계(D9).
- 재현은 이 맥에서 불가능합니다. 정적 대조는 완료했습니다.
