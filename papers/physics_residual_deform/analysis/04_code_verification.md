# 04. 공식 코드 대조: 논문 ↔ `shivanshpatel35/pgrd`

- 대상: `repo/` (얕은 복제, 커밋 `e294d96`, 2026-07-15, 공식 저장소의 HEAD 한 개). 논문은 v1(2026-07-15)입니다.
- 방법: **정적 읽기만** 했습니다. 실행, import, 설치, 체크포인트·데이터 다운로드를 하지 않았습니다. 코드와 설정 파일을 읽고 `grep`으로 위치를 찾았습니다. 저장소 내용은 신뢰하지 않는 데이터로 다뤘습니다.
- 판정 기호: **일치** (논문과 같은 내용을 코드에서 확인), **차이** (논문과 다르거나 논문에 없는 동작), **미확인** (정적으로 확인할 수 없음).
- 차이 항목은 D1–D20입니다. 논문 본문 인용은 `01_paper_analysis.md`의 절 번호를 씁니다.

## 1. 저장소 상태와 라이선스

| 항목 | 값 |
|---|---|
| 저장소 | `shivanshpatel35/pgrd` (논문 제1저자 계정) |
| 생성 / 마지막 push | 2026-04-18 / 2026-07-15 (GitHub API) |
| 별 | 14 |
| 라이선스 | **LICENSE 파일 없음**, GitHub API `license: null` |
| 얕은 복제 | 커밋 1개 (`e294d96`, "Update README.md to correct formatting ...") |
| 파일 | 블롭 109개. `.gitmodules` 없음 |
| 학습·평가 진입점 | `train.sh` → `experiments/train/train_eval.py`, `eval.sh` → `experiments/train/eval.py` |
| CMA-ES | `cmaes_optim/cmaes_optim.py` |
| 3DGS 렌더링 | `experiments/train/gs.py` |

라이선스가 없으므로 코드는 참조만 합니다. 이 분석의 어떤 산출물에도 코드를 옮겨 쓰지 않았습니다.

## 2. 일치하는 항목

| 논문 | 코드 위치 | 판정 |
|---|---|---|
| 스프링-질량 + semi-implicit Euler (§III-A) | `meta_material/sim/spring_mass.py` (`ImplicitBatchSim`, 주석 "Semi-implicit Euler integration") | 일치. `meta_material/sim/factory.py`가 이 클래스만 반환 |
| 파라미터 5개: 강성, 감쇠, 임계 거리, 노드당 최대 스프링, 바닥 마찰 (§III-A) | `experiments/cfg/cmaes_bounds/paper.yaml`, `experiments/cfg/cmaes.yaml` | 일치 (탐색 범위 5개 모두 같음) |
| CMA-ES로 최소화, 목적함수는 점별 MSE (§III-A) | `cmaes_optim/cmaes_optim.py`: 인자 기본값 `track_weight=1.0`, `chamfer_weight=0.0` | 일치. 단방향 Chamfer 항은 옵션으로 있으나 기본 꺼짐(D13) |
| 속도 잔차 $\hat V = V^{sim} + \Delta V$, 위치는 $\Delta V \cdot dt$ 적분 (§III-B) | `experiments/train/train_eval.py` 602–605행: `v = v_sim + residual_v`, `residual_x = residual_v * sim.sim_dt`, `x = x_sim + residual_x` | 일치 |
| 파지 입자 잔차 0 (§III-B) | `train_eval.py`: `residual_v = torch.where(mask_to_apply..., 0, residual_v)` | 일치 |
| 게이트 블렌딩 $\delta = \tanh$, $g = \sigma$, $0.1 \times$ (§III-C) | `experiments/train/temporal_transformer.py` 137–139행: `((1 - gate) * points_feats + gate * delta) * residual_scale`, 기본 `residual_scale=0.1` | 일치 |
| 트랜스포머 임베딩 64, 헤드 4, 층 2, FFN 128 (Table IV) | `temporal_transformer.py` 기본값 `d_model=64`, `nhead=4`, `layers=2`, `dim_ff=2*d_model=128` | 일치 (설정 파일에 재정의 없음) |
| PTv3 인코더 + NeRF식 디코더 (§III-C) | `meta_material/material/pbd_adaptive.py`: `PTv3Encoder`(`his_type == 0`로 고정), `CondNeRFModel` | 일치 (이름은 D1 참조) |
| 다중 스텝 롤아웃 학습, 재설정 없음 (§III-B) | `train_eval.py`: `x = x_sim + residual_x`를 다음 스텝 입력으로 사용. `experiments/cfg/sim/residual.yaml`의 `num_steps_train: 5`, `train.sh`의 `sim.num_steps_train=5` | 일치 (5 스텝 학습, 평가 37 스텝은 D20과 함께 확인 필요) |
| 손실은 위치 MSE 하나 (식 1) | `experiments/cfg/train/default.yaml`: `loss_factor_x: 1.0`, `loss_factor_v: 0.0` | 일치 |
| 입력 중 '현재·과거·시뮬 예측' 결합 (§III-C) | `pbd_adaptive.py`: `channel=6 * (2 + self.n_history)` | 일치하나 채널 수가 다름(D2) |
| 카메라 30 Hz, 4초 에피소드 (§IV-A) | `experiments/cfg/sim/residual.yaml`: `dt: 0.1`, `train.sh`: `dataset_skip_frame=3` | 일치 (30 Hz ÷ 3 = 10 Hz = dt 0.1, D4) |

## 3. 차이와 미확인 항목 (D1–D20)

### 3.1 모델·학습

**D1. 잔차 네트워크 이름과 설정 키가 논문과 다르게 보입니다 (낮음, 문서화 문제).**
`experiments/cfg/model/residual.yaml`에 `dynamics_type: 'pbd'`가 있고 기본 `residualnet`은 `pointnet_pbd_elasticity`(→ `PointNetPBDAdaptiveMetaNP2G`)입니다. 그러나 `dynamics_type`은 `.py`에서 참조되지 않습니다. 실제 백본은 `ImplicitBatchSim`(스프링-질량, `sim.backend: spring`)입니다. "PBD"는 잔차 네트워크 클래스 이름일 뿐입니다. 논문은 백본이 스프링-질량이라고 씁니다. 코드의 실행 경로는 이와 맞습니다.

**D2. 입력 채널 수가 논문 Table IV(18)와 다릅니다 (중간).**
`pbd_adaptive.py`는 PTv3 입력 채널을 `6 * (2 + self.n_history)`로 둡니다. `n_history`는 `train_eval.py` 215행에서 `cfg.sim.n_history`를 씁니다. `experiments/cfg/sim/residual.yaml`의 값은 `n_history: 2`입니다. 따라서 **채널 수는 24**입니다. 논문 Table IV의 18은 `n_history = 1`일 때 나옵니다. 같은 18은 PointNet 변형의 `6 * (1 + n_history)`(n_history=2)와도 맞습니다. 논문 설명(현재·과거 한 스텝·시뮬 예측 = 18)과 실제 설정 중 어느 쪽인지 정적으로는 확정할 수 없습니다.

**D3. 학습 하이퍼파라미터가 논문에 없고, 저장소의 기본값과 실행 스크립트 값이 다릅니다 (정보).**

| 항목 | 기본 설정 (`train/default.yaml`) | `train.sh` (paper_uiuc) | `eval.sh` (flag, 예시) |
|---|---|---|---|
| 반복 수 | 10,000 | 10,000 | 100,000 (`resume_iteration=3000`) |
| 배치 | 32 | 16 | 2 |
| 학습률 | 1e-4 (잔차), 1e-1 (`friction_lr`, 사용처 없음 D14) | 1e-4 | - |
| 그래디언트 클리핑 | 0.1 | - | - |
| 가중치 감쇠 | 0.0 | - | - |
| 롤아웃 | `num_steps_train: 5` | 5 | - |
| 데이터 스킵 | `dataset_skip_frame: 1` | 3 | 3 |
| 학습·평가 구간 | 0–150, 650–665 | 0–100, 100–120 | 0–10, 100–120 |

옵티마이저 종류(Adam 계열로 보이나 확인 못 함), 스케줄, 시드 기본값(`seed: 0`)은 `train_eval.py`와 설정에 있습니다. 논문에는 없습니다.

**D4. 시간 간격 dt = 0.1 (일치로 보이나 논문 미기재, 정보).**
`sim/residual.yaml`의 `dt: 0.1`이고 `train.sh`는 `dataset_skip_frame=3`입니다. 30 Hz 카메라를 3프레임마다 쓰면 10 Hz이고 dt = 0.1 s입니다. 4초 에피소드는 40프레임이고, 37 스텝 평가와 맞습니다. 이는 추정입니다.

**D5. 데이터 분할이 논문 본문과 다르게 읽힙니다 (중간).**
본문(§IV-A)은 학습 100, 검증 20 에피소드이고, 합쳐서 120 에피소드, 244초(약 4.07분)라고 씁니다. `train.sh`는 **객체 하나**(paper_uiuc)에 대해 학습 0–100, 평가 100–120을 씁니다. 객체별 분할이면 평가 에피소드는 객체당 20입니다. 본문 문장을 전체 합계로 읽으면 6개 객체 전체가 4분뿐이라 객체당 약 40초가 됩니다. 두 해석 중 객체별이 `train.sh`와 맞습니다. 다만 `eval.sh` 예시는 flag에서 학습 0–10만 씁니다. 본문과 맞지 않는 설정이 같은 저장소에 있습니다.

**D6. 바닥 클램프가 추가로 있습니다 (낮음, 논문에 없음).**
`train_eval.py` 606–607행: `x[..., 1].clamp(min=sim.ground_y)`. 식 (논문 §III-B)에는 없습니다. 시뮬의 `ground_y`는 `spring_mass.py` 486행에서 최저점 - 1e-5로 정합니다. 접촉 판정의 일부로 볼 수 있습니다.

**D7. 데이터셋 이름에 따른 분기가 학습과 시뮬에 있습니다 (중간~높음).**
- `train_eval.py` 389–390행: `use_flag_dataset_behavior = "flag" in dataset_context`. 이 경우 잔차를 **상단 띠(y > 임계)**에서 "집게가 닫혀 있는 스텝"에 0으로 가립니다(`mask_to_apply`). 논문은 flag의 위쪽 막대를 "강체 제약, 모든 점이 집게 궤적을 따름"으로 설명합니다(§IV-A). 코드의 마스크 규칙은 논문 설명과 다르고 논문에 없습니다.
- `is_teddy_dataset`: `spring_mass.py` 715행에서 teddy일 때 찌르기 힘 모드(`poke_stiffness`)를 씁니다. 논문은 teddy를 비파지 조작으로 설명하지만(§IV-A, Fig. 4) 데이터셋 이름 분기라는 사실은 쓰지 않습니다.
- 이 분기는 일반화 주장(§I)과 함께 읽어야 합니다. 새 객체는 이름 규칙을 따로 맞춰야 할 수 있습니다.

### 3.2 평가 지표

**D8. ±의 계산 방식이 논문에 없습니다 (중간).**
`experiments/train/eval.py`의 집계는 에피소드별 지표를 모아 **스텝마다** 평균, 중앙값, 25/75 백분위수, 표준편차(에피소드 간)를 구합니다. `total_metrics.txt`에는 스텝 평균을 낸 `mean`, `median`, 그리고 **스텝별 표준편차를 스텝 평균한 `std`**를 씁니다. 논문의 ±가 이 `std`인지 확인할 수 없습니다. `03_analysis.md`의 E1 가정(±는 에피소드 간 SD, n=20)은 이 코드와 맞지만, 확정은 아닙니다.

**D9. Chamfer 거리가 단방향입니다 (중간).**
`experiments/train/metric.py` 59–64행의 `chamfer_dist`는 예측 점마다 가장 가까운 정답 점을 찾고 평균합니다(예측 → 정답 한 방향). 논문 §IV-B는 **양방향** 최근접 거리라고 씁니다. 같은 파일의 `cmaes_optim.py`에도 단방향 Chamfer가 있습니다. 논문의 CD 수치는 단방향 값일 가능성이 큽니다. 표 I의 CD 수치를 양방향으로 다시 계산하면 값이 달라집니다.

**D10. EMD는 점 수가 같을 때만 정확한 할당입니다 (낮음).**
`metric.py` 67–81행: `cdist` 행렬에 `linear_sum_assignment`를 적용하고 쌍 거리의 평균을 냅니다. 예측 점 수와 정답 점 수가 같으면 최적 할당 EMD입니다. 다르면 min(N, M)개 쌍만 맞춥니다. 정답 점 수가 예측(1000 입자)과 같은지는 정적으로 확인하지 못했습니다(미확인).

**D11. MDE와 "mse"는 다릅니다 (낮음).**
`metric.py`의 `mse_dist`는 **평균 제곱 좌표 오차**입니다. 논문의 MDE(cm, 대응점 평균 거리)는 `metric_eval.py` 302–304행의 `avg_d`(평균 유클리드 거리)와 정의가 맞습니다. 논문 표 I의 MDE가 `avg_d`인지 `mse_dist`의 제곱근인지는 본문에서 확인할 수 없습니다.

**D12. 단위 cm는 코드에서 확인하지 못했습니다 (미확인).**
`metric.py`의 `inverse_preprocess`는 좌표를 회전, `preprocess_scale` 배, 평행 이동한 뒤 `dx = num_grids_flexible[-1] = 0.02`를 쓰는 변환을 합니다. 최종 단위가 cm인지 코드에 명시되어 있지 않습니다. `preprocess_scale`은 `sim/residual.yaml`에서 1.0이고, 코드의 대체 기본값은 1.2입니다.

**D13. CMA-ES 목적함수에 단방향 Chamfer 옵션이 있습니다 (정보).**
기본 실행에서는 MSE만 씁니다(`chamfer_weight=0.0`). 논문과 일치합니다. 옵션은 논문에 없습니다.

### 3.3 구현 누락

**D21. GPU 의존과 설치 경로 누락 (높음).** `environment.yml`은 `torch==2.8.0+cu128`, `dgl==2.4.0+cu124`, `torch-scatter`, `nvidia-cuda-*` 라이브러리와 `pyzed`를 요구합니다. 코드에서 `cuda`가 나오는 곳은 35행이고(예: `metric.py`의 `xyz.cuda()`), 시뮬은 Warp 커널(`wp.launch`)을 씁니다. README는 `third-party/diff-gaussian-rasterization-w-depth`를 `pip install -e`하라고 하지만 하위 모듈이 없습니다. 맥에서 Warp의 CPU 백엔드와 MPS 호환은 설치하지 않고는 확인할 수 없습니다(미확인).

**D22. F-score의 경계 허용폭 0.008은 논문에 없습니다 (정보).** `metric.py` 150행 근처의 `bound_th = 0.008`입니다.


**D14. Diff. Spring-Mass 기준선 구현이 없습니다 (높음, 표 I 한 행의 재현 불가).**
논문 §IV-C는 "공간 변화 파라미터를 미분으로 추가 학습하는 변형"을 기준선으로 씁니다. 저장소에는 `friction_lr`(설정값 0.1)이 있지만 `.py`에서 사용하지 않습니다. 공간 변화 파라미터를 미분으로 학습하는 코드도 찾지 못했습니다. 따라서 표 I의 "Diff. Spring-Mass" 행은 저장소만으로 재현할 수 없습니다. 반면 "Spring-Mass" 기준선은 잔차 네트워크를 끈 시뮬 분기(`eval.py`의 `sim-only` 집계)로 재현할 수 있습니다.

**D15. 계획, 언어 목표, 깊이·파지 파이프라인이 없습니다 (높음, §V 전체).**
`mppi`, `anygrasp`, `nano banana`, `depth_anything`, `dinov2`, `raystr3r`, `cotracker`, `grounded sam`, `keyboard` 키워드를 저장소 전체 `.py`와 `.md`에서 검색했습니다. 계획 루프(MPPI), 케이블 재배선 실험, 언어 조건 목표 생성, 키보드 상호작용 제어 코드가 없습니다. 3DGS 렌더링(`gs.py`)만 있습니다. §V 결과는 코드로 확인할 수 없습니다.

**D16. 데이터 수집과 전처리가 없습니다 (높음).**
RealSense 4대 수집, Grounded SAM 2 분할, CoTracker 추적, 깊이 리프트, 반복 롤아웃으로 대응 유지(부록 D)는 저장소에 없습니다. 가공 데이터는 Google Drive에서 받도록 되어 있습니다(README). 원자료에서 재현할 수 없습니다.

**D17. 카메라 코드가 논문 하드웨어와 다릅니다 (낮음).**
`experiments/train/droid/camera_utils/`에 ZED 카메라 읽기 코드(`pyzed`)가 있습니다. 논문은 RealSense D455를 씁니다. 이 코드가 논문 데이터 수집에 쓰였는지는 확인할 수 없습니다.

**D18. 절제 실험의 인코더 대체가 정확히 무엇인지 확인하지 못했습니다 (낮음).**
논문 Table III의 "PointNet++"는 저장소에서 `pointnet_nerf_elasticity`(→ `PointNetNeRFRadiusAdaptiveMetaNP2G`, `PointNetEncoder` 사용)로 보입니다. `pointnet_utils.py`에 계층적 집합 추상화(PointNet++)가 있는지는 확인하지 못했습니다. 절제에 쓴 설정 파일은 저장소에 없습니다.

**D19. 객체 이름 대응 (정보).**
Table I의 Duster는 `experiments/cfg/sim_params/default.yaml`의 `cleaner_uiuc`일 가능성이 큽니다(논문 명은 duster). 나머지 다섯 객체는 이름이 같습니다(rope, paper, flag, sloth, teddy). 논문 본문은 이 대응을 쓰지 않습니다.

**D20. 윈도 길이 W와 평가 길이 37 스텝을 확인하지 못했습니다 (중간).**
`temporal_transformer.py`는 `transformer_window`를 읽지만 설정 파일에는 값이 없고 기본값은 `None`입니다. `None`이 전체 시퀀스를 뜻하는지 고정 길이를 뜻하는지 정적으로 확인하지 못했습니다. 논문 Table IV에도 W가 없습니다. 평가 길이 37은 `eval.py`의 에피소드 길이에서 나오는 값으로 보이지만, 정적으로 확인하지 못했습니다(미확인).

## 4. 구현 가능성 판정

**결론: 이 맥(Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)에서 공식 학습과 재현은 불가능합니다. 축소 재구현은 가능하지만 공식 결과의 재현은 아닙니다.**

근거와 항목별 판정:

| 무엇 | 판정 | 근거 |
|---|---|---|
| 공식 학습 (객체당 CMA-ES + 잔차 10,000 반복) | **불가** | CUDA 12.8 PyTorch, DGL cu124, torch-scatter, CUDA 라이브러리를 `environment.yml`이 요구합니다. 코드에 `cuda` 호출이 35곳 있습니다. 학습 시간과 GPU 종류를 논문이 보고하지 않습니다. 데이터는 외부 드라이브에 있습니다 (D21) |
| 공식 체크포인트로 추론·표 I 재계산 | **조건부** | 체크포인트와 데이터 다운로드 승인이 있어야 합니다(이 작업에서는 금지). 설치 시 CUDA 의존을 CPU·MPS로 옮겨야 합니다. 설치가 끝나지 않으면 판정 불가. 미분 래스터라이저는 하위 모듈이 빠져 있어 설치 경로가 불완전합니다(D21) |
| 설치 가능성 자체 | **불명확** | `diff-gaussian-rasterization-w-depth`가 저장소에 없습니다. `pyzed`가 필요합니다(D17). 설치 없이 확인할 수 없습니다 |
| 스프링-질량 백본 + 속도 잔차 적분 (축소판, CPU) | **가능** | 백본은 semi-implicit Euler 한 스텝과 스프링 힘입니다. 로프 1,000 입자 규모는 CPU에서 굴릴 수 있을 것으로 봅니다. 잔차 네트워크는 작게 줄이면 됩니다. 추정 노력 1–2주(측정 전) |
| MPPI 계획 (학습된 모델 위) | **가능 (재구현)** | 코드가 없으므로 새로 짜야 합니다(D15). MPPI는 샘플과 가중 평균이라 CPU에서 가능합니다. 추정 노력 3–5일(측정 전) |
| 케이블 재배선 실로봇 | **불가** | xArm 7, 실로봇 설정, 카메라 4대가 필요합니다 |
| 언어 조건 계획 | **불가** | 목표 영상 생성 모델 호출과 AnyGrasp 등 외부 모델이 필요합니다. 외부 API 호출은 이 작업에서 금지입니다 |
| 3DGS 상호작용 렌더링 | **조건부** | `gs.py`가 있으나 CUDA 미분 래스터라이저가 필요합니다(D21). 맥에서 CPU 렌더링은 느릴 수 있습니다 |

**노력 추정** (측정 전, 추정): 축소 재구현 1–2주, MPPI 계획 추가 3–5일. 공식 설정 재현(GPU 전제)은 CUDA 환경이 있을 때 객체당 CMA-ES와 잔차 학습을 합해 수 일 단위로 봅니다. 이는 논문이 시간을 보고하지 않아 추정입니다.

## 5. 확인하지 못한 항목

- D2 (채널 수 18 대 24): 논문 Table IV의 의도를 저자에게 확인해야 합니다.
- D5 (데이터 분할): 객체별인지 합계인지 저자에게 확인해야 합니다.
- D8 (±의 정의): 논문 부록 또는 저자 확인이 필요합니다.
- D9 (CD 단방향): 논문 수치를 양방향으로 다시 계산하려면 예측과 정답 궤적 원자료가 필요합니다. 원자료는 외부 드라이브에 있습니다.
- D12 (단위 cm), D20 (W, 37 스텝): 실행하지 않고는 확인할 수 없습니다.
- 옵티마이저 종류, 학습 스케줄, 학습 시간은 `train_eval.py`를 더 읽으면 일부 확인할 수 있을 것입니다. 이번 작업에서는 대조 우선순위 때문에 끝까지 읽지 않았습니다.
