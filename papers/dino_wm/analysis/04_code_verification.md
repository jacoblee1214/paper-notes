# 04. 공식 코드 대조와 구현 가능성

대상: [gaoyuezhou/dino_wm](https://github.com/gaoyuezhou/dino_wm), 얕은 복제 `repo/` (commit `0a9492f`, 2025-03-24, "add checkpoints"). **정적 읽기만 했습니다.** 설치, 실행, import를 하지 않았고, 저장소 안의 스크립트도 돌리지 않았습니다. 외부 코드는 신뢰하지 않는 자료로 취급했습니다.

## 1. 라이선스와 공개 범위

- **MIT License** (`LICENSE`, Copyright (c) 2025 gaoyuezhou). 저장소 내 논문 저자 이름과 같은 개인 계정입니다.
- 논문은 CC BY 4.0이고, 코드 저장소는 MIT입니다. 코드 조각을 게시본에 옮기지 않았습니다. 이 문서는 설정 값과 파일 위치만 인용합니다.
- 데이터(OSF)와 체크포인트(OSF)는 저장소 라이선스에 포함되지 않았고, 확인하지 못했습니다.

## 2. 논문 ↔ 코드 대조 (D1–D16)

"차이"는 논문 본문, 부록, 표와 공식 설정 파일이 **서로 다른 값을 적은 것**입니다. "확인 불가"는 정적 읽기로 값을 찾지 못한 것입니다.

| 번호 | 항목 | 논문 | 공식 저장소 | 판정 |
|---|---|---|---|---|
| D1 | 전이 모델 학습률 | 표 12: predictor lr **5e-5** | `conf/train.yaml`: `predictor_lr: 5e-4` (10배) | **차이** |
| D2 | 인코더 학습률 | 본문: 관측 모델은 동결 (§3.1.1) | `encoder_lr: 1e-6`, `train_encoder: False` | 동결이므로 실질 영향 없음. 설정만 있음 |
| D3 | CEM 샘플과 반복 | 부록 A.8: **100 샘플 × 10 반복** (시간 측정 조건) | `conf/planner/cem.yaml`: 300 샘플, topk 30, opt_steps 30. 환경별 `plan_wall.yaml`, `plan_point_maze.yaml`은 opt_steps 10, `plan_pusht.yaml`은 30 | **차이.** 논문 시간 측정 조건과 공식 기본값이 다름 |
| D4 | 디코더 | §3.1.3: 전치 합성곱, 픽셀 MSE | `conf/train.yaml` 기본 `decoder: vqvae` (VQ 손실 가중치 0.25 포함, `models/visual_world_model.py`). `conf/decoder/transposed_conv.yaml`은 있음 | **차이 (기본값).** 논문 설정을 쓰려면 전환해야 함 |
| D5 | 계획 목적 | 식 4: 마지막 예측 잠재와 목표 잠재의 **시각 MSE** | `planning/objectives.py`: 시각 MSE + **alpha × 고유감각 MSE**, `plan.yaml` alpha 1 | **차이.** 고유감각이 없는 환경에서는 같음. 있는 환경에서는 목적이 다름 |
| D6 | 입력 해상도 | 표 12와 §4.1: **224**. 부록 A.9: **196 × 196**, 패치 14 × 14 | `conf/train.yaml`: `img_size: 224` | **불일치 (논문 내부)**. 코드는 224 (K5) |
| D7 | 인과 마스크 장치 | 장치 언급 없음 | `models/vit.py:58`: `generate_mask_matrix(...).to('cuda')` (하드코딩) | **구현 차단 요인.** CPU와 MPS에서 바로 실행되지 않음 |
| D8 | 환경별 이력 길이 H와 frameskip | 표 11: H 3 (PointMaze, Reacher, PushT, PushObj), H 1 (Wall, WallRandom, Rope, Granular). frameskip 5 또는 1 | `train.yaml` 전역 기본값 `num_hist: 3`, `frameskip: 5`. 환경 YAML에는 두 값이 없음 | **확인 불가.** 환경별 값을 덮어쓰는 실행 스크립트를 찾지 못함 |
| D9 | 계획 평가 횟수 | 50쌍 (SR), 10개 인스턴스 (CD) | `conf/plan.yaml`: `n_evals: 10` (기본값) | 기본값이 다를 뿐임. 실험 스크립트 확인 필요 |
| D10 | 전이 모델 구조 | 깊이 6, 헤드 16, MLP 2048 | `conf/predictor/vit.yaml`: depth 6, heads 16, mlp_dim 2048, dropout 0.1, pool mean | **일치** |
| D11 | 파라미터 수 | 약 19M (A.9) | 모델 차원은 생성 코드에서 정해지며, 정적 검토에서 값을 확정하지 못함 | **확인 불가** (K8) |
| D12 | 학습 기본값 | AdamW, 100 에폭, 배치 32, decoder lr 3e-4, action lr 5e-4 | `train.yaml`: epochs 100, batch 32, decoder_lr 3e-4, action_encoder_lr 5e-4. 전이 모델은 `torch.optim.AdamW` (`train.py:316`). 인코더는 Adam | **일치** (D1 예외) |
| D13 | 관측 인코더 | DINOv2, 패치 특징 (384차원) | `conf/encoder/dino.yaml`: `dinov2_vits14`, `feature_key: x_norm_patchtokens` | **일치** |
| D14 | Reach, WallRandom, PushObj, GranularRandom 환경 | 표 1 (Reach), 표 3 (세 가지 분포 밖 구성) | 전용 설정 파일과 데이터 로더를 찾지 못함 (`conf/env/`에는 point_maze, pusht, wall, rope, granular, deformable_env만 있음) | **구현 공백.** 표 1 Reach와 표 3 결과는 공식 코드만으로 재현하기 어려움 |
| D15 | 평가 데이터 | OSF 링크 (본문) | README의 OSF 공유 링크 (view-only). 데이터 크기와 라이선스 미확인 | 받지 않음 |
| D16 | 체크포인트 | 본문 언급 없음 | README: PointMaze, PushT, Wall 체크포인트를 OSF에 올림. Reach, Rope, Granular는 없음. "add checkpoints" 커밋에는 가중치 파일이 없음 | **제한.** 추론 확인은 세 환경으로 제한 |

### 2.1 대조 요약
- **구조는 일치합니다** (D10, D13). ViT 전이 모델, DINOv2 패치, 인과 마스크, 디코더 분리 구조는 논문과 같습니다.
- **학습 설정은 여러 곳이 다릅니다** (D1, D3, D4, D5). 논문의 표 값을 그대로 쓰면 공식 코드와 같은 결과가 나오지 않을 수 있습니다. 어느 설정이 표 1을 만들었는지는 알 수 없습니다.
- **환경 공백이 있습니다** (D14). Reach와 분포 밖 구성 세 가지는 공식 코드에 흔적이 없습니다. 이 결과는 논문에서만 확인할 수 있습니다.

## 3. 실행 요구사항 (정적으로 읽은 설정)

- **환경:** Linux 전제. `environment.yaml`은 conda-forge의 openblas, libgcc, ffmpeg 패키지를 씁니다. README는 MuJoCo 210을 `wget`으로 받아 `LD_LIBRARY_PATH`에 넣으라고 합니다.
- **GPU:** `conf/train.yaml`과 `conf/plan.yaml`의 SLURM 설정이 `gres: gpu:h100`을 요구합니다. `models/vit.py:58`은 `.to('cuda')`를 씁니다. `plan.py`와 `utils.py`는 CUDA가 없으면 CPU로 떨어지지만, 인과 마스크에서 멈춥니다.
- **시뮬레이션:** Rope, Granular는 PyFlex를 Docker(`xingyu/softgym`, nvidia-docker)로 빌드합니다. README의 설치 절차는 GPU를 전제합니다.
- **데이터와 체크포인트:** OSF 공유 링크에서 받아 압축을 풉니다. 변형 환경(`deformable.zip`)은 분할되어 있어 합쳐야 합니다. 환경 변수 `DATASET_DIR`을 가리키게 해야 합니다(`conf/env/wall.yaml`). 체크포인트는 PointMaze, PushT, Wall만 제공됩니다.

## 4. 구현 가능성 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**판정: 전체 재현은 불가능합니다. 공식 체크포인트로 PushT 계획을 소규모로 확인하는 것은 조건부로 가능합니다. 구조를 새로 짜는 축소판은 가능합니다.**

| 작업 | 필요 조건 | 이 맥에서 | 노력 추정 (근거 없는 추정) |
|---|---|---|---|
| 공식 학습 (100 에폭) | H100급 GPU, SLURM (`gres: gpu:h100`), 대용량 데이터 | **불가**: GPU 전제 | — |
| Rope, Granular 시뮬 (PyFlex) | Docker, CUDA, SoftGym 빌드 (README) | **불가** | — |
| Reach, WallRandom, PushObj, GranularRandom | 공식 설정 부재 (D14) | **불가** (코드 없음) | — |
| 공식 체크포인트 + PushT 계획 평가 (소규모) | OSF 체크포인트 다운로드(크기 미확인), 환경 설치(mujoco210 또는 대체), CUDA 한 곳 (`vit.py:58`)을 CPU·MPS로 고침 | **조건부.** 승인 필요, 설치 필요 | 3–5일 |
| 자체 PyTorch 축소판 (ViT 전이 + CEM + PushT 래퍼, MPS) | torch, 시뮬 환경 (gym 계열), DINOv2 ViT-S 가중치 (다운로드 필요) | **가능** (작은 설정은 MPS에서 돌릴 만함) | 1–2주 |
| 구조 토이 (잠재 공간에서 CEM 계획, 무작위 특징) | numpy | **가능** | 1–2일 |

- **MPS와 CUDA 차이:** 환경 메모리(memory)는 충분하지만 `vit.py:58`처럼 `.to('cuda')`가 박혀 있어 한 줄 이상 고쳐야 합니다. 이 점을 고치면 추론은 MPS에서 돌아갈 가능성이 큽니다. 측정은 하지 않았습니다.
- **논문 시간 측정 (A.8):** 배치 32 추론 0.014 s, 계획 53 s는 A6000 기준입니다. 이 맥에서의 속도는 측정하지 않았습니다.
- **재현을 승인받아야 할 항목:** 체크포인트 다운로드, 새 가상환경, 공식 코드 실행. 프로젝트 규칙에 따라 실행 전에 확인을 받습니다.

## 5. 이 문서의 한계
- 정적 읽기만 했습니다. 설정이 실제 실험에 쓰였는지는 확인하지 못했습니다.
- 공식 저장소 커밋은 2025-03-24 것입니다. 이후 변경은 반영되지 않았습니다.
- 데이터와 체크포인트 크기는 확인하지 않았습니다.
