# 04. 공식 코드 대조와 구현 가능성

대상: `repo/` (openvla/openvla, 커밋 `c8f03f4`, 2025-03-23, 얕은 복제). 정적 읽기만 했습니다. 설치, 실행, import, 체크포인트 다운로드는 하지 않았습니다. 코드는 `prismatic/`(학습 공통), `vla-scripts/`(학습·미세조정 진입점), `experiments/robot/`(평가)를 읽었습니다.

## 1. 라이선스와 출처

| 대상 | 표기 (출처) | 판정 |
|---|---|---|
| 코드 (openvla/openvla) | MIT (`LICENSE` 파일 확인, GitHub API `mit`) | 확인 |
| 기반 코드 (TRI-ML/prismatic-vlms) | MIT (GitHub API). 복제 안 함 | 확인 (메타데이터) |
| 가중치 (openvla/openvla-7b) | HF 모델 카드 태그 `license:mit` | 불일치 (D14) |
| 가중치의 상위 라이선스 | README: "Llama 2에서 파생되어 Llama Community License의 제약을 받을 수 있음" | 주의 (D14) |
| 논문 | arXiv CC BY 4.0 (abs "view license"). PDF에 저작권 표시 없음 | 확인 |

- **재배포와 상업 이용.** 코드는 MIT라 자유롭게 씁니다. 가중치의 상업 이용은 Llama Community License 조건을 확인해야 합니다(D14). 이 분석은 가중치를 받지 않았습니다.

## 2. 학습 설정 대조 (논문 대 공식 설정)

| 항목 | 논문 | 공식 코드 (파일:위치) | 판정 |
|---|---|---|---|
| 행동 정규화 | 1st–99th 분위수 → 256 빈 (§3.2) | `BOUNDS_Q99`: q01, q99로 [-1, 1] 정규화 (`prismatic/vla/datasets/rlds/utils/data_utils.py:76–83`, `datasets.py:99`). 토크나이저는 고정 [-1, 1] 256 빈 (`action_tokenizer.py`) | 일치 (D1) |
| 행동 토큰 수와 위치 | 256 토큰, 마지막 256 토큰 덮어쓰기 (§3.2) | `action_token_begin_idx = vocab_size − (n_bins + 1)`, 토큰 = `vocab_size − digitize` (`action_tokenizer.py`) | 일치 (단, 실사용 255 구간, D2) |
| 학습 률 | 고정 2e-5, 워밍업 없음 (§3.4) | `learning_rate = 2e-5`, `lr_scheduler_type = "constant"`, `warmup_ratio = 0.0` (`prismatic/conf/vla.py:84–89`) | 일치 (D6) |
| 전역 배치 | 2048 (§3.5) | `global_batch_size = 2048`, `per_device = 32`, `expected_world_size = 64` (`vla.py:126–136`) | 일치 (D6) |
| 시각 인코더 | 미동결 (§3.4) | `freeze_vision_backbone` 기본 False (`vla.py:69`) | 일치 (D7) |
| 해상도 | 224×224 (§3.4) | `224px` 설정 이름 (`vla.py` 전체) | 일치 (D7) |
| 에폭 | 27 (§3.4) | 기본 `epochs = 1000`, `max_steps = None` (`vla.py:78–79`). 최종 설정이 값을 바꾸지 않음 | 불일치: 설정으로 재현 불가 (D3) |
| 데이터 일정 (DROID) | 10% 가중, 마지막 1/3에서 제외 (§3.3, 표 3 각주 6) | 설정 주석: "70% 동안 DROID 포함, 이후 재샘플링" (`vla.py:130`). 실제 `data_mix = "oxe_magic_soup_plus_minus"` (`vla.py:132`), 이 목록에 DROID 없음. 2단계 설정은 구현되어 있지 않음 | 불일치 (D4) |
| 데이터 혼합 가중 | 표 3 비율 (Octo 가중 기반) | `oxe_magic_soup_plus_minus` 가중치(정규화 전 합 31.3) (`mixtures.py:140–`). 샘플링 가중에 프레임 수를 곱함 (`balance_weights=True`, `datasets.py:115`, `dataset.py:518–519`) | 재계산 불가 (D5) |
| Bridge 첫 전이 | 첫 전이 제거 (부록 C) | `bridge_orig_dataset_transform`에서 `[1:]` 슬라이스 (`prismatic/vla/datasets/rlds/oxe/transforms.py:61–75`) | 일치 (D10) |
| 미세조정 LoRA | r=32, 모든 선형층 (§5.3, 표 1) | `lora_rank = 32`, `target_modules = "all-linear"`, `lora_alpha = min(r, 16)` (`vla-scripts/finetune.py:99–101, 174–178`) | 일치 (D9). 알파는 논문에 없음 (D8) |
| 미세조정 학습률 | 논문에 값 없음 | `learning_rate = 5e-4`, `batch_size = 16`, `max_steps = 200,000` (`finetune.py:87–90`) | 논문에 없음 (D8) |
| 추론 정밀도 | bfloat16 기본, int8과 int4 (§5.4) | `torch_dtype=torch.bfloat16`, `load_in_8bit/4bit` 선택 (`experiments/robot/openvla_utils.py:46–48`, `run_bridgev2_eval.py:46–47`) | 일치 (D12) |

## 3. 평가 설정과 행동 복원

| 항목 | 논문 | 공식 코드 | 판정 |
|---|---|---|---|
| Bridge 제어 주파수, 비차단 | 5 Hz 비차단 (§5.1, §5.4) | `control_frequency = 5`, `blocking = False` (`run_bridgev2_eval.py:68, 71`) | 일치 (D11) |
| Bridge 행동 복원 키 | 논문에 명시 없음 | `unnorm_key = "bridge_orig"` (`run_bridgev2_eval.py:87`) | 확인 (D11) |
| 중앙 자르기 | 논문에 없음 | 금지 (`assert not cfg.center_crop`, `run_bridgev2_eval.py:84`) | 확인 (D11) |
| 행동 복원 식 | 분위수 정규화의 역 | `0.5(a+1)(q99−q01)+q01`, 마스크 적용 (`prismatic/models/vlas/openvla.py:95–101`) | 일치 (D1) |

## 4. 구현 가능성 판정 (이 맥 기준)

**결론: 논문의 실험 대부분은 이 맥에서 재현할 수 없습니다. 추론과 시뮬 평가 일부는 조건부이고, 행동 토큰화와 표 재계산은 CPU에서 완료했습니다.**

| 작업 | 필요한 자원 (근거) | 이 맥에서 | 판정 | 노력 (추정, 측정 없음) |
|---|---|---|---|---|
| 표 재계산과 검정 | 논문 표 값 | 가능 (완료) | 완료 | 완료 |
| 행동 토큰화 왕복 검사 | numpy (설치 불필요) | 가능 (완료, D2) | 완료 | 완료 |
| 7.5B 추론 (bf16, 가중치 약 15 GB) | 15 GB 이상 + 활성 메모리 (§3.5). 공식 예시는 `cuda:0`과 flash-attn 선택 | 메모리 여유는 빠듯함. MPS 경로와 bf16 지원, 속도 미측정. int8과 int4(bitsandbytes)는 CUDA 전용 | 조건부 (포팅 후 측정) | 포팅과 측정 1–2일 |
| 추론 속도 6 Hz 주장 확인 | 추론 경로가 돌아야 함 | 측정 전 | 조건부 | 위와 같음 |
| LoRA 미세조정 | README: A100 80 GB 예시, 최소 약 27 GB | 24 GiB 미만이고 CUDA 의존 (bitsandbytes 제외) | 불가 | — |
| 전체 미세조정 | 8 A100 노드, 체크포인트 30 GB (README) | 불가 | 불가 | — |
| 사전학습 재현 | 64 A100 × 14일 (§3.5) | 불가 | 불가 | — |
| OXE 데이터 준비 | 대용량 RLDS (크기 미확인), 다운로드 승인 필요 | 저장 용량과 시간 미확인 | 불가 (현실적으로) | — |
| LIBERO 시뮬 평가 (표 12) | 4 스위트 × 500 × 3 시드 = 6,000 에피소드. LIBERO와 robosuite 설치, 추론 필요 | 추론 횟수는 에피소드당 300 스텝으로 가정하면 약 180만 회. 1 Hz면 수 주, 6 Hz면 수 일 | 불가 (전체), 조건부 (한 스위트 소표본) | 설치 1–2일 + 소표본 수 시간 (추정) |
| 실로봇 (Bridge WidowX, Google, Franka) | 로봇과 실험실 (§5.1, §5.2) | 불가 | 불가 | — |
| 절제 실험 (표 9, 10, 11) | 학습 (GPU), 실로봇 또는 시뮬 | 불가 | 불가 | — |

- **설치 승인.** 어떤 설치나 다운로드도 하지 않았습니다. 추론 측정을 원하면 먼저 설치와 체크포인트 다운로드(약 15 GB)에 대해 사용자 승인이 필요합니다.
- **이 맥에서 바꿔야 할 것.** `.to("cuda:0")`를 MPS로 바꾸고, flash-attn을 끄고, 양자화는 빼야 합니다. 이 변경이 결과를 바꾸는지는 측정해야 알 수 있습니다.

## 5. 대조 항목 (D1–D16)

**D1 (일치, 구현 방식이 논문 설명과 다름).** 논문은 "분위수 구간을 256 빈으로 나눈다"고 씁니다. 코드는 데이터를 q01–q99로 [-1, 1]에 옮긴 뒤(BOUNDS_Q99) 고정 [-1, 1] 빈을 씁니다. 결과는 같습니다. 분위수 바깥 값은 ±1로 잘립니다. 역변환도 같은 분위수를 씁니다.

**D2 (경미, 측정함).** `ActionTokenizer`는 빈 경계 256개로 `np.digitize`를 씁니다. 결과 인덱스는 1–256이고 토큰은 256개입니다. 디코딩은 인덱스 − 1을 255개 빈 중심에 맞추고 254에서 자릅니다. 그래서 정규화 값 +1.0(정확히 경계)은 0.996으로 되돌아가고, 실제로 쓰이는 토큰은 255개입니다. `analysis/toy_action_bins.py`로 확인했습니다: 쓰인 토큰 255개, 디코딩 값 255개, 최대 왕복 오차 0.00392 = 반 빈 너비. 영향은 범위 양 끝의 1/256 정도입니다.

**D3 (불일치, 설정으로 재현 불가).** 논문은 27 에폭을 돌았다고 씁니다(§3.4). 공식 설정은 에폭을 1000으로 두고 `max_steps`를 두지 않으며, 최종 설정이 이를 바꾸지 않습니다. README 예시 체크포인트 이름은 `step-295000-epoch-40-loss=0.2200.pt`입니다(README 학습 예시). 어느 실행인지 밝히지 않아 27 에폭과 연결할 수 없습니다.

**D4 (불일치, 세 곳이 서로 다름).** 논문: DROID 10%, 마지막 1/3에서 제외. 설정 주석: 70% 동안 DROID 포함(`oxe_magic_soup_plus`), 이후 재샘플링. 실제 설정: `oxe_magic_soup_plus_minus`가 전 구간 사용되며 DROID 항목이 없습니다. 변환 스크립트 `vla-scripts/extern/convert_openvla_weights_to_hf.py:42`의 실행 이름은 `…magic-soup-plus+n8+b32+x7`입니다. 실제 학습이 `plus`로 시작해 `plus_minus`로 이어졌을 가능성이 있지만, 공개 설정으로는 확인되지 않습니다.

**D5 (재계산 불가).** 표 3은 비율을 주고, 코드는 가중치 목록을 줍니다. 두 목록은 다릅니다. 예를 들어 `plus_minus`의 정규화 가중치(전체 31.33 기준)는 NYU Franka 9.6%, TACO 6.4%, Fractal 3.2%, Bridge 3.2%입니다. 표 3은 Fractal 12.7%, Bridge 13.3%, NYU Franka 0.8%입니다. 다만 코드는 `balance_weights=True`로 가중치에 데이터셋 프레임 수를 곱합니다(`dataset.py:518`). 그래서 표 3의 값은 프레임 수 없이는 맞출 수 없습니다. 데이터셋 크기를 받지 않았으므로 **재계산 불가**로 둡니다.

**D6 (일치).** 학습 률 2e-5, 워밍업 0, 상수 스케줄, 전역 배치 2048(64 × 32). 논문의 설정과 같습니다.

**D7 (일치).** 시각 인코더 미동결, 224 해상도.

**D8 (논문에 없음).** 미세조정의 학습률(5e-4), LoRA 알파(min(r, 16) = 16), 최대 스텝(200,000)은 논문 본문과 부록에서 찾지 못했습니다. 논문 표 1과 8의 결과를 재현하려면 이 값들이 필요합니다.

**D9 (일치).** LoRA r=32, 모든 선형층. 논문 표 1의 "모든 선형층"과 맞습니다.

**D10 (일치).** Bridge 첫 전이 제거. 부록 C 설명과 같습니다. RT-2-X는 이 전처리가 없으므로 비교 조건이 다릅니다(`03` C5).

**D11 (일치, 논문에 없는 설정 있음).** Bridge 평가는 5 Hz 비차단입니다. 중앙 자르기는 금지되어 있는데, 논문에는 적혀 있지 않습니다. 이는 재현 설정에 영향을 줍니다.

**D12 (일치).** bfloat16이 기본이고, int8과 int4는 bitsandbytes로 불러옵니다. 학습용 QLoRA 4비트(NF4)도 옵션으로 있으나 논문 실험에는 쓰이지 않았습니다(`finetune.py:102, 146–147`).

**D13 (주의, 논문에 명시됨).** §5.3과 §5.4의 표 1, 2는 Octo 혼합과 SigLIP 단일 인코더를 쓴 v0.1 모델입니다(§5.3 각주 4). 공개된 flagship(`openvla-7b`)은 DINO+SigLIP과 Magic Soup++입니다. 표 1과 2의 결과를 flagship에 옮기려면 이 차이를 고려해야 합니다. 논문 본문은 이를 각주에만 씁니다.

**D14 (불일치, 문서 간).** HF 모델 카드는 `license:mit`이고 README는 Llama Community License 적용 가능성을 말합니다. 법적 해석은 하지 않았고, 두 표기가 다르다는 사실만 기록합니다. 재배포 전에 확인해야 합니다.

**D15 (구현 제약).** 설치 요건은 PyTorch 2.2.0, CUDA 12.4(conda), flash-attn 2.5.5(CUDA 빌드)입니다(README 설치 절). 추론 예시는 `.to("cuda:0")`를 씁니다. LoRA 예시는 A100 80 GB 한 장을 쓰며 "최소 약 27 GB"라고 적습니다. 이 맥(Apple M4 Pro, 24 GiB, MPS, CUDA 없음)에서는 설치 자체가 막힙니다(§4 판정).

**D16 (확인 불가).** README의 학습 예시 파일명(epoch-40)이 논문의 27 에폭과 어떻게 맞는지 확인할 수 없습니다. 예시로만 보고 판정에서 쓰지 않습니다.

## 6. 확인하지 못한 항목

- 가중치의 실제 해시, 체크포인트 파일 내용과 학습 설정 일치 여부.
- 표 3 비율과 코드 가중치의 차이 (D5). 데이터셋 프레임 수가 필요합니다.
- 27 에폭과 DROID 일정 (D3, D4)을 실제 학습 로그로 확인하는 것.
- 미세조정 학습률과 LoRA 알파 (D8). 논문 부록 B.3에 없습니다.
- 추론 속도 (6 Hz)와 양자화 속도.
