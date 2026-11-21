# 04. 공식 코드 대조 (정적 읽기), 라이선스, 구현 가능성

## 0. 대상과 방법

- 저장소: `rotematari/dlofeel-state` (저자 본인 계정), 커밋 `eb7f7e8c806a62a464a329e8d59de9a3e9d1b797` (2026-10-18, 기본 브랜치 `main`의 최신).
- 얕은 복제 `repo/` (`git clone --depth 1 --no-recurse-submodules`). 서브모듈 `third_party/sparsh`는 가져오지 않았습니다.
- **실행하지 않았습니다.** 설치, import, 테스트, 학습, 추론, 데이터와 가중치 다운로드를 하지 않았습니다. 복제본은 신뢰하지 않는 데이터로 취급했습니다.
- 읽은 것: `README.md`, `pyproject.toml`, `LICENSE`, `NOTICE`, `CITATION.cff`, `src/dlofeel_state/configs/default.yaml`, `configs/eval/k16.yaml`, `tests/fixtures/parity/config.yaml`(릴리스 K = 16 설정), `models/backbones/sparsh_backbone.py`, `models/backbones/pose_encoder.py`, `models/bspline.py`, `losses/weighted.py`, `losses/chamfer.py`, `losses/der.py`(곡률 함수), `engine/train.py`(옵티마이저, 손실 조합, 스케줄), `engine/metrics.py`, `engine/evaluate.py`, `data/dataset.py`(목표 재표본), `data/spline_utils.py`, `docs/data.md`, `docs/training.md`, `docs/model.md`, `docs/loss.md`, `docs/DATA.md`(커밋 blob).
- 라이선스 확인: 저장소 `LICENSE`는 MIT이고 Copyright (c) 2026 Rotem Atari입니다. `NOTICE`와 README는 Sparsh(Meta)를 CC-BY-NC 4.0으로 적습니다.
- 주의: 이 복제본의 워킹 트리에는 `docs/DATA.md`가 제대로 체크아웃되지 않았습니다(대소문자가 같은 `data.md`와 충돌). 필요한 내용은 `git show HEAD:docs/DATA.md`로 읽었습니다. 로컬 `git status`는 ` M docs/DATA.md`를 보여줍니다. 이 복제본은 분석용이며 수정하지 않았습니다.

## 1. 설치 요구사항과 라이선스

- Python ≥ 3.12, `torch ≥ 2.10`, `torchcurves ≥ 0.2.1`(B-spline 계산), `lightning`, `wandb`, `ikpy`, `viser`, `pyvista`, `rosbags` 등 (`pyproject.toml`). 설치 대상이 많습니다. 실행 전에 설치 목록을 승인받아야 합니다(프로젝트 규칙).
- Sparsh 서브모듈(`third_party/sparsh`, 고정 커밋 `fee6a05`)과 DINO 사전학습 가중치 `dino_vitsmall.ckpt`가 필요합니다. 저장소 README는 학습 시 필요하고 평가 시에는 체크포인트가 자체 설정을 가져온다고 합니다.
- 릴리스 체크포인트 `best.pt`(137 MB)는 저장소에 없습니다. 코드 주석은 "Phase 3에 HF 호스팅 예정"이라고 씁니다.
- 데이터: 패킹 파일 한 분할 약 40 GB(`docs/data.md` §4.2). 원본 rosbag 처리 스크립트(`scripts/process_bags.py`)는 오프라인입니다.
- **라이선스 게이트.** Sparsh와 사전학습 가중치는 CC-BY-NC 4.0입니다. 저장소 자체 코드(MIT)로도 상업적 이용은 Sparsh 조건에 묶입니다(`README.md` License 절). 이 점은 재사용 판단에 영향을 줍니다.

## 2. 논문 ↔ 코드 대조 (D 항목)

차이의 심각도: **높음**은 결과 해석이나 주장에 영향을 주는 차이, **중간**은 재현 설정이 다르다는 차이, **낮음**은 이름이나 서술의 차이입니다.

| 번호 | 항목 | 논문 | 저장소 (근거) | 심각도 |
|---|---|---|---|---|
| D1 | 학습·평가 입력의 자세 출처 | §II.A: 자세 `A_t`는 로봇 순기구학으로 계산. §II.B: 모션캡처는 라벨링 전용, 배포 시에는 촉각과 고유감각만 | 입력 `ee_position`/`ee_orientation`은 모션캡처(NatNet) 기반의 EE 포즈이고, 고정 좌표계(`fixed_position`/`fixed_orientation`, 모션캡처 강체)로 변환합니다(`data/dataset.py` 약 402–410행, `docs/data.md` §2 표와 §6). 관절 상태(`joint_positions`)는 학습에 쓰지 않는다고 문서에 있습니다. 배포(실시간) 코드는 저장소에 없습니다. | **높음** |
| D2 | 학습률과 백본 계수 | §II.C: 기본 `1e-4`, 백본 ×0.05. §III.C: 백본 `1e-5` | 릴리스 K = 16 설정: `lr = 7.147e-5`, `backbone_lr_factor = 0.5`, 따라서 백본 `3.57e-5`(`tests/fixtures/parity/config.yaml`). 기본 설정(`default.yaml`): `lr = 7e-5`, 계수 0.5. 옵티마이저는 AdamW이고 OneCycleLR입니다(`engine/train.py` 약 708–755행). 논문과 맞지 않습니다. | 중간 |
| D3 | 에폭 수 | 10 | 릴리스 K = 16: 10. 기본 설정(`default.yaml`): 20 | 낮음 |
| D4 | 물리 손실의 탄젠트 가중치 | `w_e = 0.05` (§III.C) | 릴리스 K = 16 설정(`tests/fixtures/parity/config.yaml`)과 기본 설정(`default.yaml`) 모두 `w_edges = 0.1`입니다. `w_n = 1`, `w_ℓ = 0.5`, `w_κ = 0.01`, `λ_w = 1`(`mse_weight`), `λ_c = 0.25`(`chamfer_weight`)는 논문과 같습니다. | 중간 |
| D5 | 벡터 항의 정규화 | 식 (9), (10), (12): 점마다 `‖·‖²₂`(세 좌표의 합), 점 수로 평균 | `WeightedLoss`는 `nn.MSELoss`를 씁니다(`losses/weighted.py`). 좌표 축까지 평균하므로 벡터 항(위치, 탄젠트, 곡률)이 1/3배 작아집니다. 스칼라 길이 항(식 11)은 같습니다. 가중치 `w_e`, `w_n`, `w_κ`의 실효값이 논문과 다릅니다. | 중간 |
| D6 | 평가 지표 "RMSE" | 표 1 "RMSE(mm)". 식은 없음 | `engine/metrics.py`의 `l2_mean`은 샘플마다 점별 유클리드 거리의 평균(`mean_point_distance_per_sample`)입니다. 제곱근 평균제곱이 아닙니다. `docs/loss.md`와 `docs/training.md`는 같은 값을 "RMSE"라고 부르고, 저장소 README도 "Per-vertex RMSE 30.52 ± 15.22 mm"라고 씁니다. | 중간(명칭 혼동) |
| D7 | 표준편차 정의 | 미정의 | `metrics.py`는 `np.std`(모집단, ddof = 0). 표 1의 "±"가 같은 정의라는 가정(`03` §0)을 지지합니다. | 확인됨 |
| D8 | 목표 곡선 재표본 | §II.B: 14개 마커를 정렬(모멘텀 탐욕)하고 B-spline 피팅으로 K개 재표본. 차수 미기재 | 원시 경로 `dataset.py` 약 422행: `fit_spline_unordered(..., k=5, smoothing=0)`, 즉 5차 스플라인. 패킹 `scripts/pack_dataset.py`의 기본 차수는 3입니다. 정렬 모멘텀 가중치는 0.2(`spline_utils.py`). 두 경로의 차수가 다릅니다. | 중간 |
| D9 | 제어점 수 | K = 16 (§II.B, §II.C) | 릴리스 K = 16 설정은 16(`tests/fixtures/parity/config.yaml`). 기본 설정(`default.yaml`)은 `n_control_points = 32`, `num_output_pts = 32`. 표 1 제안 행의 파라미터 수는 K = 32와 맞습니다(D16). | 중간 |
| D10 | 데이터 분할 수 | 학습 196,678, 시험 21,853 | 세션 목록 합계는 196,968 / 21,563입니다(`docs/data.md` §3). 커밋된 `docs/DATA.md`는 196,678 / 21,853을 "이 릴리스의 권위 있는 수치"라고 하고, 290개는 "분할 경계 위치"라고 씁니다. 논문에는 이 설명이 없습니다. | 중간 |
| D11 | 센서 드롭아웃 증강 | 미구현이라고 씀(§IV: 향후 과제, 확률적 드롭아웃 학습) | 코드에 `drop_augment`(15개 드롭 마스크, `data/augment.py`)가 있습니다. 릴리스 K = 16 설정은 `false`입니다. 기본값은 `False`입니다(`configs/train.py`). 논문과 모순은 아니지만 코드 기능이 논문에 없습니다. | 낮음 |
| D12 | 자세 인코더 구조 | §II.C: 3층 MLP | `pose_encoder.py`: 선형층 4개(7 → 512 → 512 → 512 → K·128), 각 은닉층 뒤 LayerNorm, ReLU, 드롭아웃 0.1 | 낮음 |
| D13 | 논문에 없는 학습 세부 | 기재 없음 | 드롭아웃 0.1, 그래디언트 클리핑 0.5, 잠재 차원 128, 가중치 감쇠 `1.07e-4`(릴리스 설정), 참조 이미지 빼기는 `use_diff_images`와 `use_blank_ref`로 구현됨 | 낮음 |
| D14 | 사전학습 Sparsh 규모 | §II.C: 46만 장 이상 | 코드는 가중치 파일명만 가리킵니다. 규모는 확인하지 않았습니다. 라이선스는 CC-BY-NC(§1). | 확인 불가 |
| D15 | 시간 차분 입력 | 직전 프레임 한 장 | `past_window_steps = 1`, `max_time_delta = 0.13` s. 0.13 s를 넘는 이전 프레임의 처리 방식은 논문에 없습니다. | 낮음 |
| D16 | 파라미터 수 | 표 1: 제안 6.38 M 학습 / 24.80 M 전체. "w/o B-spline" 5.33 M / 23.74 M | 릴리스 K = 16: 5,327,367 학습 / 23,747,607 전체(README). 저장소 `docs/training.md` 예시 로그(기본 설정 추정): 6,377,991 / 24,796,167. 학습 가능 차이 1,050,624 = `pose_enc` 마지막 층 `512 × 128 × (32 − 16) + 128 × (32 − 16)`과 정확히 같습니다. 전체 차이 1,048,560은 2,064만큼 맞지 않습니다. 결론: **표 1 제안 행은 K = 32 구성으로 보입니다.** K = 16이라면 본문과 맞지 않습니다. 저자 확인 필요. | **높음** |
| D17 | 추론 지연 | 61.5 ms, 16 Hz, 배치와 장비 미기재 | README: 같은 모델의 로그 64.20 ms(배치 32, RTX 5090, 10회 평균, 3회 워밍업, 순전파만). `docs/training.md` 예시 60.52 ms. 저장소 README는 61.5 ms를 "인용할 값"이라고 씁니다. | 낮음 |
| D18 | 그리퍼 방향 범위 | §III.B: ±45°가 물리적 한계 | 저장소 세션 이름에 `deg115`, `deg135`, `deg135_135`, `deg30_135`, `deg75_45` 등이 있습니다. 이름의 각도가 같은 물리량인지 확인할 수 없습니다. 판단 보류. | 확인 필요 |
| D19 | 오차의 단위 | 표 1은 mm | `evaluate.py` 주석과 `metrics.py` 독스트링은 "mm", `docs/training.md`는 "metres"(미터), 평가 임계 `low_rmse_threshold = 0.055`는 "(m)"이라고 적습니다. 원본 마커 최소 거리는 0.02 m입니다(`docs/data.md` §2). 보고된 30.52가 mm인지는 저장소만으로 확정할 수 없습니다. | 중간 |
| D20 | 평가 기준선 | §III.D: 평가는 GT 정점과 비교 | 평가는 재표본된 K점 목표와 비교합니다(D8). 원시 14개 마커 기준 오차는 계산하지 않습니다. 재표본 자체의 오차는 보고되지 않습니다. | 중간 |
| D21 | 배포 파이프라인 | 실시간 추정 16 Hz(§III.F) | 온라인 ROS 노드나 실시간 추론 스크립트는 없습니다(`scripts/`는 오프라인 처리). 지연 측정은 더미 입력(`torch.randn`)으로 한 값입니다(`engine/train.py` 약 194–226행). | 중간 |
| D22 | 정규화 함수 | 없음 | `WeightedLoss.normal_weight_distance`는 정의만 있고 호출되지 않습니다(주석 처리). 영향 없음. | 낮음 |
| D23 | 혼합 정밀도 | §III: 혼합 정밀도 학습 | `autocast`는 `device.type == "cuda"`일 때만 켜집니다(`engine/train.py`). 논문과 맞습니다. 다만 평가 결과가 CPU와 MPS에서 같은지는 확인하지 않았습니다. | 확인됨 |
| D24 | 촉각 중간 층 융합 | §II.C, 식 (2): 층 `S = {3, 6, 9, 12}`의 가중 합 | 릴리스 K = 16 설정과 기본 설정 모두 `fusion_layers = [9, 10, 11, 12]`입니다(`tests/fixtures/parity/config.yaml`, `default.yaml`). 층 3과 6은 쓰지 않습니다. 가중 합은 소프트맥스 가중치로 구현되어 식 (2)와 같습니다. | **높음** |

**맞는 것 (확인됨).** 손실 구성 `L = 0.25 L_chamfer + 1.0 L_weighted`(`engine/train.py` 약 794행, 식 6과 같음), 위치, 탄젠트, 길이, 곡률 항의 구조(식 9–12, D5 제외), 곡률 이항 벡터(`der.py`의 `2 e_{i-1}×e_i / (ℓ_{i-1}ℓ_i + e_{i-1}·e_i)`, 식 7과 같음), 교차 주의와 자기 주의 디코더, 학습 가능한 소프트맥스 층 가중치(식 2, 층 집합은 D24 참조), 센서별 학습 임베딩, 사인파 위치 부호화, 공유 인코더, 마지막 2블록과 최종 정규화만 학습, Chamfer의 양방향 평균(식 13).

## 3. 주장과 코드의 대조 요약

- 논문 §II.B의 "모션캡처는 라벨링에만" 주장은 저장소의 입력 구성과 맞지 않습니다(D1). 이 충돌은 `03` C13, C14에 있습니다.
- "캘리브레이션 불필요"(§II.C)는 학습 파이프라인에서 모션캡처 기반 고정 좌표계와 센서별 무접촉 기준 이미지에 의존합니다(`03` C13).
- 표 1의 제안 모델 파라미터는 K = 32 구성과 맞습니다(D16). 같은 표의 다른 행(w/o B-spline)은 K = 16과 맞습니다. 표 행 구성이 서로 다른 설정일 수 있습니다.

## 4. 구현 가능성 (이 맥: Apple M4 Pro, 약 24 GiB 통합 메모리, MPS, CUDA 없음)

**판정: 이 맥에서 공식 학습은 불가능합니다. 추론 파리티 점검은 조건부로 가능합니다. 자체 재구현은 2–3주(추정)입니다.**

| 작업 | 판정 | 근거 |
|---|---|---|
| 공식 학습 (10 에폭, 약 19.7만 샘플, 배치 32) | **불가** | RTX 5090 기준으로 학습 시간이 보고되지 않았습니다. 데이터가 약 40 GB이고, 혼합 정밀도 경로는 CUDA 전제입니다(MPS 경로는 검증되지 않음). 맥에서 쓸 만한 시간 안에 끝낼 수 없다고 판단합니다(추정, 측정 없음). |
| 릴리스 K = 16 체크포인트 추론, 저장소 픽스처 8프레임 파리티 | **조건부** | 픽스처 8프레임(7.2 MB)은 저장소에 있습니다. 체크포인트 137 MB는 저장소에 없고, 호스팅 경로가 미정입니다. 설치(torch ≥ 2.10, torchcurves, Sparsh)는 승인이 필요합니다. CPU나 MPS에서 가능한지는 측정하지 않았습니다. |
| 시험 전체 (21,853 프레임)의 추론과 표 1 재계산 | **불가(현실적으로)** | 데이터 40 GB 이상과 추론 시간이 필요합니다. 예측 결과 파일이 공개되지 않았습니다. |
| 센서 절제, 노이즈 실험 (표 2, 그림 6) | **조건부(추론만)** | 체크포인트와 소량 프레임으로 추론만 가능합니다. 재학습은 불가입니다. |
| 재학습 기반 절제 (w/o B-spline, w/o L_weighted, 센서 제거 재학습) | **불가** | 학습 비용이 GPU 전제입니다. |
| 자체 재구현 (MLP 자세 인코더, 교차 주의, B-spline 층, 물리 손실) | **조건부** | `torchcurves` 없이 B-spline을 직접 짤 수 있습니다. 촉각 인코더는 Sparsh 가중치(CC-BY-NC)가 필요하고, 대체 인코더(ResNet 등)는 논문 표 1의 ResNet 행과 같아집니다. 추정 2–3주, 측정 전입니다. |
| 손실 함수 단위 검증 (식 9–13 대 코드, 무작위 곡선) | **가능** | CPU에서 수 초입니다. 설치는 numpy와 torch CPU가 필요합니다. 이번에는 하지 않았습니다(`03` §4). |
| 실로봇 실험 | **불가** | Kinova Gen3, Robotiq, DIGIT 4개, OptiTrack 세팅이 필요합니다. |

- 재학습 비용이 크기 때문에 학습 설정을 바꿔 가설을 검증하기 어렵습니다. 그래서 D2, D4, D5, D8의 효과는 이 맥에서 측정할 수 없습니다.
- 노력 추정(측정 전): 릴리스 체크포인트의 파리티 점검 1–2일(승인과 설치 시간 제외). 자체 재구현 2–3주. 공식 재학습은 이 맥에서 범위 밖입니다.

## 5. 재실행

없음. 이 분석에서는 복제본의 코드를 실행하지 않았습니다. 재현 계획은 개인 분석 문서에 따로 있습니다.
