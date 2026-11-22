# 04. 공식 코드 대조와 구현 가능성

대상: `Yuansen-Huang/KPGrasp` (GitHub, 얕은 복제 `89716dd`, 2026-10-18 커밋). 복제본은 `repo/`에 있고 정적으로만 읽었습니다. 설치, 실행, import, 가중치와 데이터 다운로드는 하지 않았습니다. 저장소 안의 `tools/check_release.py`와 `sample.py --dry-run`도 실행하지 않았습니다.

차이는 **D**, 일치는 **K**로 표시합니다. 모든 항목에는 저장소 파일 경로를 적었습니다.

## 0. 요약

- **학습과 추론의 핵심 절차는 논문과 같습니다.** 흐름 정합 MSE 하나, 오일러 5단계, 100개 샘플, 가능도 상위 10, IK의 5회 반복과 5 mm 기준 (K1–K9).
- **키포인트 정의가 다릅니다.** 논문은 J = 21 (손바닥 5), 단일 단계입니다. 공식 저장소는 28 키포인트 (손바닥 7), 세 단계 (pregrasp, grasp, squeeze)를 한 토큰에 묶습니다 (D1–D3). 공식 저장소가 논문 결과의 모델을 그대로 담은 것인지는 확인할 수 없습니다.
- **평가 코드가 없습니다.** GSR, OSR, CDC, PD (표 1)와 IK 뒤의 DGA 평가 (표 2)를 계산하는 코드가 저장소에 없습니다 (D9, D10). 표 1, 2는 이 저장소만으로 재현할 수 없습니다.
- **학습 데이터를 만드는 단계가 빠져 있습니다.** 키포인트 레코드 (FK 결과)를 만드는 코드가 없고, 레코드를 HDF5로 묶는 코드만 있습니다 (D11).
- **라이선스는 CC BY-NC 4.0입니다.** 논문의 CC BY 4.0과 다릅니다. 상업적 사용이 제한됩니다 (§1).
- **체크포인트 가중치는 공개되지 않았습니다.** 256.5 MiB짜리 `Final_base`의 URL이 아직 없습니다 (`docs/CHECKPOINTS.md`).

## 1. 저장소, 라이선스, 이력

| 항목 | 값 | 파일 |
|---|---|---|
| 라이선스 | CC BY-NC 4.0 (LICENSE 전문). GitHub API는 SPDX를 인식하지 못해 NOASSERTION | `LICENSE` |
| 제3자 구성 요소 | DiT 블록과 시간 임베딩 (CC BY-NC 4.0), GLIDE (MIT), MinkowskiEngine (MIT), PointNet++ 계열 FPS (MIT), Muon (MIT), Shadow Hand XML과 메시 (Apache 2.0) | `NOTICE.md`, `licenses/` |
| DiT 파생 코드 | "같은 제한" (CC BY-NC) | `NOTICE.md`, `src/dexgrasp/network/policy/transformer.py` |
| 파일 수 | 복제본 91개 파일, `.git` 740 KB. 데이터와 가중치 없음 | 복제본 |
| 이력 | 얕은 복제 (depth 1). 전체 커밋 수는 확인하지 않음 | — |
| 저장소 생성 | 2026-09-22 (arXiv v1 2026-06-08 이후) | GitHub API |

**해석:** 논문은 CC BY 4.0이지만 코드는 CC BY-NC 4.0입니다. 논문 설명만 보고 다시 짜는 것은 코드 라이선스의 대상이 아니지만, 이 분석은 법률 자문이 아닙니다. 공개 저장소에 이 코드를 복사하면 NC 조건을 따라야 합니다.

## 2. 논문과 코드 대조

### 2.1 일치 (K)

| 번호 | 항목 | 논문 | 코드 | 판정 |
|---|---|---|---|---|
| K1 | 흐름 손실 | 식 (1): 선형 보간 경로, 속도 타깃 x_1 − x_0, MSE | `network/flow_method.py` `FlowMatch.calculate_loss`: CondOT 경로, 타깃 `path.dx_t`, `nn.MSELoss` | 일치 |
| K2 | 손실 구성 | 흐름 손실 하나 (§4.2) | `config/algo/mink.yaml` `loss_weight: loss_fm: 1.0`, 다른 손실 없음 | 일치 (C14 지지) |
| K3 | 추론 방식 | 노이즈 100, 오일러 5단계 (§4.2) | `configs/experiments/Final_base.yaml`: `test_grasp_num: 100`, `sample_step: 5`, `solver_type: Euler`; `network/ode_solver.py` `EulerSolver` | 일치 |
| K4 | 상위 10 선택 | 가능도 상위 10 (§4.2) | `sample.py`: `scores.topk(test_topk)`, `test_topk: 10` | 일치 |
| K5 | 접촉 정제 없음 | 추론은 ODE → 가능도 → IK (§1, §4.2) | `model.py` `KeyPointGraspModel.sample`: 노이즈 → ODE → 로그 가능도 반환. 접촉·침투 항 없음. IK는 `ik.py` 후처리 | 일치 (C15 지지) |
| K6 | 가능도 | Hutchinson, 탐침 1개 per 궤적 (§4.2) | `ode_solver.py`: `z = randint*2−1` (Rademacher), 단계마다 `_divergence`, `log_p -= dt·div` | 일치. 상수항 log p_prior(x_0)는 생략 (순위에는 영향 없음) |
| K7 | 물체 인코더 | 1,024점, MinkUNet, FPS 128 토큰, C = 512 (§4.2) | `Final_base.yaml`: `num_points: 1024`, `token_length: 128`, `out_feat_dim: 512`, `voxel_size: 0.005` | 일치 |
| K8 | DiT | 블록 8, 헤드 8, FFN 2,048, 자기·교차 주의 (§4.2) | `Final_base.yaml`: `depth: 8`, `num_heads: 8`, `hidden_size: 512`; `transformer.py` `DiTBlock` `mlp_ratio=4.0` → 은닉 2,048; 자기·교차 주의 모두 있음 | 일치 |
| K9 | 학습 설정 | Muon, 학습률 2×10^−4, 200,000 스텝, 전체 배치 4,096, 8 GPU (§4.2) | `Final_base.yaml`: `opt: muon`, `lr: 0.0002`, `max_iter: 200000`, `batch_size: 4096`; README: 8 GPU, GPU당 512 | 일치 (학습 시간 2일은 미검증) |
| K10 | IK | Mink, quadprog, 최대 5회 반복, 5 mm 이하 성공 (부록 A) | `kinematics/ik.py`: `mink.solve_ik(..., "quadprog", 1e-5)`, `for i in range(5)`, `IK_SUCCESS_ERROR_THRESHOLD = 0.005` | 일치 |
| K11 | SVD 강체 정합 | §4.1 | `kinematics/ik.py`: `np.linalg.svd(H)`로 `compute_transformation` | 일치 |
| K12 | 시간 조건 | τ는 AdaLN으로 주입 (§4.2, 그림 3) | `transformer.py`: 시간 임베딩과 AdaLN (`9 * hidden` 조절 벡터) | 일치 |
| K13 | 스케일링 설정 | 데이터 0.001–1, 블록 2–16, 배치 8–4,096 (§5.3) | `configs/experiments/` 아래 `scaling_data_*` (0.001, 0.01, 0.1), `scaling_layers_*` (2, 4, 16), `scaling_batch_*` (1, 8, 64) | 부분 일치 (D15) |

### 2.2 차이 (D)

| 번호 | 항목 | 논문 | 공식 저장소 | 영향 |
|---|---|---|---|---|
| D1 | 키포인트 수 | J = 21 (§4.1 "Keypoint Definition", 부록 F.1 Sharpa도 21) | `n_kp: 28` (`Final_base.yaml`, `hand/shadow.yaml`), `kp_name_lst` 28개 (`kinematics/ik.py`). README: "28-keypoint 구현을 유지" | **높음.** 논문 수치와 체크포인트가 같은 규약인지 확인할 수 없음. 저장소는 28을 "체크포인트와 일치해야 함"이라고 씀 (`docs/DATA.md`) |
| D2 | 단계 수 | 한 파지의 키포인트 K ∈ R^{J×3}, 속도 v ∈ R^{J×3} (§4.1, §4.2). 본문에 단계 개념 없음 | 세 단계 (pregrasp, grasp, squeeze)를 채널 방향으로 연결: `token_channels = n_frame·3 = 9`. 샘플 출력 3단계 (`model.py` `KeyPointGraspModel`, `sample.py` "three grasp phases") | **높음.** 논문의 출력 공간과 다름. 저장소는 파지 과정을 함께 예측 |
| D3 | 손바닥 키포인트 | 5개 (손바닥 중심 1, 뿌리 4) (§4.1) | 7개 (`palm_1`–`palm_7`) (`kinematics/ik.py`) | 중간. SVD 정합의 기준점이 달라짐 |
| D4 | 입력 정규화 | 언급 없음 | 키포인트를 실행 평균·표준편차로 정규화 (`utils/RMS.py`, 최대 2,000회 갱신, 추론 때 역변환) | 중간. 논문 설명에 없음 |
| D5 | 학습률 스케줄 | 학습률 2×10^−4 (§4.2) | 선형 워밍업 5,000 스텝 (0.2 → 1.0), 이후 코사인 감쇠 → 2×10^−5 (`train_ddp.py`) | 낮음~중간. 설명에 없음 |
| D6 | 최적화기 | "Muon" (§4.2) | 2차원 이상 은닉 행렬만 Muon, 정규화·AdaLN·최종층·임베딩은 AdamW (β = 0.9, 0.95), 가중치 감쇠 0.01 (`train_ddp.py`) | 낮음~중간. 설명에 없음 |
| D7 | 키포인트 임베딩 | 키포인트마다 MLP (§4.2) | 단일 선형층 `nn.Linear(in, hidden)` + 학습 위치 임베딩 (`transformer.py`) | 낮음 |
| D8 | 물체 전역 특징 | 물체 정보는 교차 주의로 전달 (§4.2) | `Final_base`는 `use_global_feat: true`. 전역 특징을 시간 임베딩에 더해 AdaLN 조건으로 씀 (`transformer.py`) | 중간. 논문에 없는 조건 경로 |
| D9 | 평가 코드 | GSR, OSR, CDC, PD, 다양성 (표 1) | 없음. `mujoco` 의존성만 있고 힘 테스트와 지표 계산이 없음. 물리 평가는 DexGraspBench를 가리킴 (README) | **높음.** 표 1을 이 저장소로 재현할 수 없음 |
| D10 | DGA 평가 | IsaacGym 평가 (§5.2) | 없음 (`src/`에 IsaacGym 참조 없음) | **높음.** 표 2 재현 불가 |
| D11 | 키포인트 레코드 생성 | FK로 K 계산 (§4.1 "Parameterization Conversion") | 없음. `docs/DATA.md`: "이 패커는 원시 Dexonomy 파지에서 키포인트를 만들지 않는다". `data_proc.py`는 이미 있는 레코드를 HDF5로 묶을 뿐 | **높음.** 학습 데이터 생성 단계가 빠짐 |
| D12 | 문서 링크 | — | `NOTICE.md`가 `docs/EXPERIMENTS.md`를 가리키지만 파일이 없음 | 낮음 (문서 오류) |
| D13 | 공개 체크포인트 | 표 1의 76.3% | `Final_base` 가중치 공개 URL 미지정. 256.47 MiB (SHA-256 기록). README는 76.3%를 "역사적 벤치마크 결과"라고 씀 | **높음.** 논문 수치의 재확인 경로가 없음 |
| D14 | 객체 집합 | Dexonomy 10.7k 객체 (§5.1) | `object_path`: DGN_5k, objaverse_5k (합 약 10k) | 낮음. 대략 일치 |
| D15 | 배치 스케일링 설정 | 배치 8, 64, 512, 4,096 (§5.3, 그림 5) | 배치 1, 8, 64 (`scaling_batch_*`) + 4,096 (본 설정). **512 설정 파일 없음** | **중간.** 본문의 배치 512 (73.5%)를 저장소로 확인할 수 없음 (`03` V18) |
| D16 | 1단계 FlowMap | 1단계는 LSD (FlowMap) 증류 (부록 D) | `Final_lsd3.yaml`: `flow.name: FlowMap`, `sample_step: 1`. 논문 부록 D의 1단계 행과 대응하는 것으로 보임 (확인은 못 함) | 낮음 |
| D17 | 절제 설정의 배치 | "다른 설정은 그대로" (§5.3) | `Final_RTJ`, `Final_RTJ34token`, `Final_SE32`, `Final_base_1token`, `Final_lsd3`의 `batch_size`가 **4**. 주 설정은 4096. README는 단일 GPU 학습에서 `algo.batch_size=8`을 쓰라고 함 | **중간.** 설정 파일을 그대로 학습하면 절제 조건이 주 설정과 달라짐 |
| D18 | SE(3) 대조의 샘플링 | "다른 설정은 그대로" (§5.3) | `Final_SE32.yaml`: `rt_sample_steps: 20`, `joint_sample_steps: 20`. 주 설정은 5단계 | **중간.** SE(3) 기준선은 샘플링 단계가 4배 많음. 비교 조건이 같지 않음 |

### 2.3 확인 불가 (U)

| 번호 | 항목 | 이유 |
|---|---|---|
| U1 | 모델 파라미터 67.19M (§4.2) | 모델을 만들려면 MinkowskiEngine과 PyTorch3D가 필요. 설치와 실행은 하지 않음 |
| U2 | 부록 A의 IK 통계 (97.37%, 0.669 mm) | 데이터와 IK 실행이 필요 |
| U3 | DGA 재학습 수치 (58.5 → 59.1) | 저장소와 논문 모두 학습 결과 파일 없음 |

## 3. 저장소 자체 검증 (`docs/VALIDATION.md`)

저장소가 주장하는 검증 범위는 다음과 같습니다. 이 분석은 이를 다시 하지 않았습니다.

- 6개 체크포인트 상태 사전 엄격 로드 통과 (GPU 환경).
- 주 모델과 원본 대비 같은 입력·시드에서 최대 키포인트 차이 0, 점수 차이 0.
- 한 배치 샘플링 (파지 10개), 세 단계 IK 처리.
- **전체 재학습과 물리 벤치마크는 실행하지 않았다고 명시.**

## 4. 구현 가능성 판정

**판정: 이 맥에서 논문 수치를 재현하는 것은 불가능합니다. 방법을 소형으로 다시 짜는 것은 가능하고 노력은 2–3주입니다.**

이 맥은 Apple M4 Pro, 약 24 GiB 통합 메모리, MPS, CUDA 없음입니다. 논문 학습은 RTX 4090 8장에서 약 2일입니다. 실측하지 않았지만 MPS의 연산량은 이보다 훨씬 작을 것입니다.

| 목표 | 이 맥 | 필요 자원 | 노력 (추정) | 판정 |
|---|---|---|---|---|
| 논문 학습 (표 1, 그림 5, 200k 스텝) | 불가 | CUDA GPU 8장 × 약 2일, 키포인트 레코드 (D11) | — | **불가** |
| 공식 가중치로 샘플링 (76.3%의 재확인) | 불가 (현재) | 가중치 공개 URL (D13 미충족). 공개되면 MinkowskiEngine 0.5.4, PyTorch3D 0.7.8 빌드 (CUDA 기준, 맥 빌드는 미확인) | 가중치가 공개되면 1–2일 (Linux GPU 권장) | **불가 (가중치 미공개)** |
| 표 1 지표 재계산 (GSR 등) | 조건부 | 샘플 파일 (가중치 필요), Dexonomy 평가 로직 (D9), MuJoCo 3.4 (맥 CPU 가능) | 평가기 재구현 1주 | **조건부** |
| IK 통계 재계산 (부록 A) | 조건부 | mink, mujoco (CPU 가능), 공식 IK 파일 실행은 설치 승인 필요, 데이터 | 1–2일 | **조건부 (설치 승인 필요)** |
| 키포인트 vs pose-joint 소형 비교 (표 3 방향) | 가능 | MPS (소형), 자체 구현, 손 한 개의 소형 데이터 | 2–3주 | **가능 (방법 이해용. 주장 검증은 아님)** |
| 데이터 준비 (키포인트 레코드 생성) | 조건부 | Dexonomy 원자료 (HF 타르볼 합 약 14.8 GiB, 메타데이터만 확인), FK 구현 | 1–2주 | **조건부** |

### 4.1 근거

- 학습이 CUDA 전제이고 이 맥에 CUDA가 없다는 점은 판정을 바꾸지 않습니다 (`INSTALL.md`, `README.md`).
- 가중치가 없어서 추론 재현도 현재 불가입니다 (`CHECKPOINTS.md`).
- 평가 코드가 없으므로 표 1, 2를 재계산하려면 Dexonomy와 DGA의 평가 로직을 다시 짜야 합니다 (D9, D10).
- 데이터가 빠진 단계 (D11)와 28 대 21 키포인트 규약 (D1)은 논문 방법을 똑같이 옮기는 것을 어렵게 합니다.

### 4.2 공식 설치를 하지 않은 이유

이 분석은 "설치와 실행 금지" 조건을 따랐습니다 (`CLAUDE.md`, 재현 승인 범위). 따라서 MinkowskiEngine과 PyTorch3D의 맥 빌드 가능성은 확인하지 않았습니다. 이것은 "불가능"이 아니라 "미확인"입니다.

## 5. 재현 가능성 요약

| 주장 | 저장소로 확인 가능 | 필요한 것 |
|---|---|---|
| 구조와 학습 손실 (K1–K9) | 예 (정적 읽기로 확인) | — |
| 표 1 (76.3%) | 아니오 | 가중치, 평가기 (D9, D13) |
| 표 2 (DGA 63.8%) | 아니오 | 가중치, IsaacGym 평가 (D10) |
| 표 3 (절제) | 아니오 | 절제 설정 파일은 5개 있음 (D17, D18). 가중치는 모두 미공개 |
| 그림 5 (스케일링) | 부분 | 배치 512 설정 없음 (D15). 학습 필요 |
| 부록 A (IK 0.669 mm) | 가능 (CPU 예정) | 데이터와 IK 실행 (설치 승인) |
| 실로봇 (83/100) | 아니오 | 하드웨어, 실험 코드 (저장소에 없음) |
