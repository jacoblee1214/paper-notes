# AnyTouch: 공식 코드 대조 (D1–D11)

공식 저장소 GeWu-Lab/AnyTouch를 **얕은 복제(`--depth 1`)하여 정적으로만 읽었습니다.** 설치, 실행, import, 체크포인트·데이터 다운로드는 하지 않았습니다. 분석 대상 커밋은 `9c43a1a` (2026-01-13, "fix probe resume")입니다. 읽은 파일은 `config.py`, `main_stage1.py`, `main_stage2.py`, `stage1_engine.py`, `stage2_engine.py`, `model/mae_model.py`, `model/multi_model.py`, `model/linear_probe.py`, `model/process_clip.py`, `main_probe.py`, `probe_engine.py`, `config_probe.py`, `dataloader/cross_dataset.py`, `dataloader/stage2_dataset.py`, `dataloader/downstream_dataset.py`, `train_stage*.sh`, `run_probe_*.sh`, `README.md`, `requirements.txt`입니다. 논문 수치와 비교한 대상은 `01_paper_analysis.md`입니다.

## 0. 저장소 메타

| 항목 | 값 (GitHub API, 2026-11-22 조회) |
|---|---|
| 저장소 | github.com/GeWu-Lab/AnyTouch |
| 라이선스 | MIT (`LICENSE`, 2025 GeWu-Lab) |
| 별 / 포크 | 97 / 10 |
| 생성 / 마지막 push | 2025-02-06 / 2026-01-13 |
| 분석 커밋 | `9c43a1a6eb38d904fd767712eb9dcb2d98b8d56b` |
| 크기 | 약 19 MB (`.git` 12 MB, 작업 트리 26 MB) |
| 환경 명세 | Python 3.9, PyTorch 2.1.0, CUDA 11.8, `requirements.txt` (accelerate 0.33, timm 1.0.3, transformers 4.42.3, peft 0.12.0 등) |
| 체크포인트 | Google Drive 링크 (README). 이 분석에서 받지 않음 |
| 데이터 | README가 TAG, VisGel, Cloth, TVL, SSVTP, YCB-Slide, OF Real, Octopi, TacQuad, Touch and Go, ObjectFolder 1.0/2.0, Feel을 요구. 이 분석에서 받지 않음 |

## 1. 재현성 확인 (정적)

- 학습 스크립트 `train_stage1.sh`와 `train_stage2.sh`의 값 중 논문과 같은 것: 마스크 비율 0.75, lr 2e-4, 워밍업 1, 1단계 20 에폭, 2단계 12 에폭, `alpha_vt` 1.0, `cross_alpha` 0.1, `init_temp` 0.07 (논문에 없음), 배치 64 × 4 GPU (논문에 없음), `beta_start` 0.0, `beta_end` 0.75 (논문 A.6의 p_u 선형 증가와 같음), L=5 센서 토큰 (`mae_model.py`: `torch.zeros(10, 5, hidden)`).
- 논문 A.6의 디코더 설정(8층, 폭 512, 헤드 16, 패치 14)은 `main_stage1.py`와 `main_stage2.py`에서 같습니다.
- README 표 (TAG 80.82 / 86.74 / 94.68, Feel 80.53, OF1 49.62, OF2 76.02)는 논문 표 2, 표 3, 표 6의 전체 모델 행과 같습니다. 체크포인트와 README 수치가 같은 모델을 가리킨다고 보는 것은 **가정**입니다. 체크포인트를 받지 않아 확인하지 못했습니다.
- 공식 저장소에는 표 2·3의 기준선(UniTouch, TLV-Link, OmniBind, VIT-LENS-2, T3) 코드가 없습니다. 표 7(교차 센서 생성)과 표 4(실로봇) 코드도 없습니다.

## 2. 논문과 코드의 차이 (D1–D11)

| 번호 | 논문 | 코드 | 위치 | 영향 |
|---|---|---|---|---|
| **D1** | 식 (9): $\mathcal{L}_{stage2} = \mathcal{L}_{align} + \lambda\mathcal{L}_{match}$. MAE 항 없음 | `stage2_engine.py` 비교차 분기: `loss = aligh_loss + mae_loss` (`--no_mae` 미지정 시). 스크립트 `train_stage2.sh`는 `--no_mae`를 주지 않음. `model/multi_model.py`에서 `do_mae=True` | `stage2_engine.py` L196–204, `multi_model.py` L64–68, L425–426 | 2단계에서 MAE 재구성이 계속 학습됩니다. 표 6의 "2단계 제거" 등 절제 해석이 달라질 수 있습니다. 논문은 2단계를 정렬과 매칭만으로 설명합니다. |
| **D2** | 식 (9)는 정렬과 매칭 손실을 **한 스텝에서 합산** | 한 스텝은 `do_cross`로 **둘 중 하나만** 계산합니다. 데이터 인덱스 `data_choice`가 이미지·비디오 데이터셋과 교차 데이터셋을 번갈아 고릅니다 | `stage2_engine.py` L100–145, L169–189 | 같은 배치에서 기울기가 합쳐지지 않습니다. 동일한 λ=0.1이라도 최적화 궤적이 다릅니다. |
| **D3** | 식 (5): 원소별 곱 뒤 **MLP** $m=\mathrm{MLP}(\cdot)$ | `self.cross_sensor_head = nn.Linear(projection_dim, 1)`: **선형층 하나** | `multi_model.py` L87, L279–285 | 매칭 헤드의 표현력이 논문보다 작습니다. 논문의 MLP 크기는 기재되지 않았습니다. |
| **D4** | 식 (4): $\frac{\alpha_{TL}}{2}(\mathcal{L}_{T\to L}+\mathcal{L}_{L\to T})$ (모든 항에 1/2) | `clip_loss`: 터치–텍스트 항이 `self.alpha_lt*(touch_text_loss + text_touch_loss)`로 **1/2 없음**. 비전 항은 `/2.0`이 있음 | `multi_model.py` L121 | α_TL=1.0이면 실효 가중이 논문 식 (4)의 두 배입니다. 결과에 얼마나 영향을 주는지는 측정하지 않았습니다. |
| **D5** | 텍스트: "다른 물체 **또는 위치**의 샘플을 음성으로" | `PretrainDataset_cross*.__getitem__`: 음성은 **다른 물체**인 샘플만 씁니다. 같은 물체의 다른 위치는 음성이 되지 않습니다 | `dataloader/cross_dataset.py` L237–243 (이미지), L523–529 (비디오) | 매칭이 "위치" 구분을 학습하지 않습니다. 논문의 "같은 물체, 다른 위치" 음성 설명은 코드와 다릅니다. |
| **D6** | (해당 없음, 실행 가능성) | `main_stage2.py` L200 `if not args.no_text:`와 `multi_model.py` L65 `args.no_text`가 있으나, `config.py`에 `--no_text` 인자 정의가 **없습니다**. `train_stage2.sh`도 주지 않음 | 전체 저장소 `grep -rn no_text` 결과 3건 (사용처만) | 정적 판독으로는 `main_stage2.py`가 `args.no_text` 참조 시점에 `AttributeError`로 멈춥니다. 실행으로 확인하지 않았습니다. 논문 A.6의 "텍스트 인코더 동결 + 비전 LoRA"를 구현한 줄이 바로 이 조건문 안에 있습니다. |
| **D7** | 추론 때 새 센서는 항상 범용 센서 토큰 $s_u$ 사용 (§4.5) | 학습 코드의 센서 ID는 **0–4**와 **−1**(범용)만 나옵니다. 범용 토큰은 행 번호 −1, 즉 10행 테이블의 마지막 행입니다. 그런데 `run_probe_OF2.sh` 평가는 OF2.0 데이터의 센서 ID를 **5**로 둡니다 (`dataloader/downstream_dataset.py` L104) | `mae_model.py` L49, `downstream_dataset.py` L104, `run_probe_OF2.sh` | 표 3 OF2.0 행(AnyTouch 76.02, 같은 데이터 85.87)은 학습되지 않은 행 5의 토큰으로 평가되었을 가능성이 있습니다. 체크포인트에서 행 5가 어떤 값인지 확인하지 못했습니다. 표 6의 OF2 절제 해석에도 같은 문제가 있습니다. |
| **D8** | 표 3: 선형 탐침 정확도 (프로토콜 세부 없음) | `main_probe.py`는 **에폭마다 테스트 집합**에서 정확도를 계산하고, `max_accuracy`를 갱신하며 "Max accuracy"를 출력합니다. 최적 에폭의 테스트 정확도가 보고될 수 있는 구조입니다 | `main_probe.py` L214–240 | 테스트 집합으로 에폭을 고르면 낙관적 편향이 생깁니다. 논문이 어느 값을 보고했는지는 확인하지 못했습니다. 검증 집합 분리는 없습니다. |
| **D9** | 논문: 선형 탐침, 공통 설정 | 데이터셋마다 다릅니다. TAG: lr 2e-4, wd 0.05, `--pooling cls`, `--use_sensor_token`. Feel: lr 1e-3, wd 0.05, cls. OF1: lr 1e-3, **wd 0.01**, **global** pooling. OF2: lr 1e-3, wd 0.01, cls. 모두 50 에폭 | `run_probe_*.sh` | 데이터셋별로 다른 하이퍼파라미터를 골랐다는 뜻입니다. 고른 기준이 테스트 집합인지 확인하지 못했습니다. 논문에는 이 값이 없습니다. |
| **D10** | (논문에 없음) | `train_stage2.sh`의 `--TAG_times 2`와 `--cross_iter`는 `config.py`에 정의되어 있으나 **다른 곳에서 쓰이지 않습니다** (`grep` 결과 정의와 스크립트뿐). | `config.py` L32–33, `train_stage2.sh` L18 | 실효가 없습니다. TAG 가중 샘플링이 있다는 오해를 피해야 합니다. |
| **D11** | 논문에 없음 또는 불충분: 배치 크기, 가중치 감쇠, AdamW β, 온도 τ, LoRA 랭크 | 코드: 배치 64 (×4 GPU), 가중치 감쇠 **0.05** (기본값; 스크립트는 0.1), β=(0.9, 0.99), 온도는 학습 가능한 `logit_scale`을 `init_temp` 0.07로 초기화, LoRA r=16, α=16, dropout 0 (`config.py`, `process_clip.py` L532–545) | 여러 파일 | 논문 재현에 필요한 값이 대부분 코드에 있습니다. 논문은 이 값을 본문에 적지 않았습니다. |

**확인했고 차이가 없는 항목**: 마스크 비율 0.75, 1단계 손실 구성 (재구성 두 개와 다음 프레임 예측, `mae_model.py` L514–534), 범용 센서 토큰 확률의 선형 증가 (`stage1_engine.py` L71–73, `stage2_engine.py` L161–163), LoRA 대상 (비전 인코더의 q, k, v, out_proj, `process_clip.py` L532–545), OpenCLIP ViT-L 초기화, 프로브가 인코더를 동결하고 헤드만 학습하는 구조 (`probe` 옵티마이저는 `model.head.parameters()`만 받음, `main_probe.py` L200).

## 3. 기타 관찰

- **센서 토큰 개수 10:** 학습 센서는 5종이지만 테이블은 10행입니다. 행 5–8은 학습되지 않았습니다. 행 9는 −1로 간접 참조되는 범용 토큰입니다 (D7과 연결).
- **교차 센서 쌍의 위치 정의:** 양성 샘플은 `object_index_pairs`의 같은 범위(CSV 한 행)에서 고르고 센서 ID가 다른 것을 씁니다. CSV 한 행이 "물체"인지 "접촉 세트"인지 확인하지 못했습니다(CSV 파일은 받지 않음). 논문의 "같은 위치"가 이 구조로 보장되는지는 미확인입니다.
- **README 인용:** `Accepted by: ICLR 2025`, Citation의 booktitle은 "Thirteenth ICLR"입니다. OpenReview 기록(ICLR 2025 Poster)과 맞습니다.
- **오류율:** GPT-4o 텍스트의 검증 절차는 코드에 없습니다(데이터 제작 스크립트 미공개).

## 4. 구현 가능성 판정 (이 맥: Apple M4 Pro, 약 24 GiB 통합 메모리, MPS, CUDA 없음)

**판정: 학습 재현은 이 맥에서 불가능합니다. 프로브 평가 재현은 데이터 다운로드를 승인받은 뒤 조건부로 가능하지만, 그 전에 코드 수정이 필요합니다.**

| 작업 | 판정 | 근거 |
|---|---|---|
| 1단계 사전학습 (ViT-L 인코더 전체, 2.48M 프레임, 20 에폭) | **불가** | A800 4장 기준 설정. 이 맥에서 같은 배치와 에폭을 돌리는 것은 측정 전이라도 수주 이상으로 추정됩니다. 코드 자체가 CUDA 전용 (아래). |
| 2단계 사전학습 (LoRA, 12 에폭) | **불가** | 1단계와 같은 이유. 또한 D6로 시작 단계에서 멈춥니다. |
| 공식 체크포인트 추론 | **조건부** | Drive 체크포인트의 크기는 확인하지 못했습니다. MPS 포팅과 `autocast`/`cuda` 수정이 필요합니다. 표 3 OF2 행의 센서 ID 문제 (D7)를 확인하려면 체크포인트 행 5 값을 봐야 합니다. |
| 선형 탐침 (TAG, Feel, OF1, OF2) | **조건부** | 데이터셋 다운로드와 용량 확인이 필요합니다(Touch and Go, ObjectFolder 1.0/2.0, Feel). 사용자 승인 필요. MPS 포팅 필요. 프로브 학습은 헤드만 학습하므로 계산량은 작습니다. 테스트 최고치 보고 문제 (D8)를 같이 재계산해야 합니다. |
| 표 6 절제 재현 | **불가** | 각 절제마다 2단계 사전학습을 다시 해야 합니다. |
| 실로봇 pouring (표 4) | **불가** | xArm 6, Robotiq 2F-140, GelSight Mini, 키보드 데모 수집이 필요합니다. |
| 공식 코드 정적 대조와 재계산 | **완료** | 이 문서와 `03_analysis.md`. |

**CUDA 전용 경로 (정적 확인):** `torch.cuda.amp.autocast`, `torch.cuda.synchronize`, `--device` 기본값 `cuda`가 `stage1_engine.py`, `stage2_engine.py`, `probe_engine.py`, `main_stage1.py`, `main_stage2.py`, `main_probe.py`, `config.py`, `config_probe.py`, `util/misc.py`에 있습니다. 분산 학습은 `dist_backend = 'nccl'` (`util/misc.py` L242)입니다. MPS 포팅 시 `autocast`를 CPU/MPS용으로 바꾸고, `torch.distributed`를 단일 장치 경로로 바꾸고, `nccl`을 쓰지 않아야 합니다. 상세 노력 추정은 측정 전입니다.

**노력 추정 (측정 전):**
- 코드 수정 (D6 해결, MPS와 단일 장치 경로): 1–2일.
- 프로브 평가 재현 (데이터 다운로드 제외, 체크포인트 추론 포함): 승인 뒤 수일. 데이터 용량에 따라 달라짐.
- 학습 재현 (1·2단계, 이 맥에서): 불가로 판정. 축소 실험(작은 ViT, 소수 센서)은 별도 설계가 필요하며, 논문 주장을 검증하는 수준은 되지 않습니다.

## 5. 라이선스와 재배포

- 코드: MIT. 복사와 수정이 허용됩니다. 저작권 표시와 허가 문구를 유지해야 합니다.
- 논문: CC BY 4.0 (arXiv abs 링크 `creativecommons.org/licenses/by/4.0/`). 원문 PDF에 ©나 All rights reserved가 없습니다.
- 데이터: TacQuad HF 카드 license mit. 다른 데이터셋(TAG, TVL, ObjectFolder 등)의 라이선스는 이 분석에서 확인하지 않았습니다.
- 체크포인트: Google Drive. 라이선스 표기는 확인하지 못했습니다.
