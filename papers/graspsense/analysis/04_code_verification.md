# 04. 코드 검증: 논문 ↔ 구현 대조, 수치 재계산, 구현 가능성

- **대상:** `eliizavetas/GraspSense` @ `16dc5fb` (2026-06-01, MIT). 저장소는 2026-05-19에 만들어졌습니다. v2(2026-04-13)보다 뒤이고, 논문과 arXiv 페이지에는 링크가 없습니다. 소유자 이름(Elizaveta)과 README가 가리키는 연구실 작업 환경으로 보아 1저자의 저장소로 추정합니다. 공식 공개본이라는 명시는 없습니다.
- **비교용:** 논문 Eq. 1–2와 대조하려고 `PKU-EPIC/DexGraspNet` @ `bd1b13d`의 `grasp_generation/utils/initializations.py`, `energy.py`를 읽었습니다.
- **범위:** GPU 실행 없이 코드 읽기와 CPU 재계산만 했습니다. 공개 코드에는 논문 Table I–IV를 만든 부분이 없어서, 표를 코드로 재현할 수는 없습니다.

## 1. 모듈 매핑

README가 스스로 밝힌 상태: "perception and scene-preparation" 골격은 동작하고, "real physically grounded force-map construction", "USD primvar export", "grasp generation, grasp ranking, motion planning, and impedance-based grip execution"은 진행 중입니다.

| 논문 | 구현 위치 | 상태 |
|---|---|---|
| (A) Qwen 과제 이해 → (O, a, λ) | `modules/vlm/vlm_module.py`, `prompts.py` | ⚠️ Qwen2.5-VL-3B-Instruct를 별도 프로세스로 실행하고, 실패하면 정규식 파서로 대체합니다. λ의 수치 변환은 없고 모드 문자열만 있습니다(D3) |
| (B) YOLO-World 검출 | `modules/detection/yolo_detector.py` (ultralytics `yolov8s-worldv2`) | ✅ |
| (B) SAM 분할 | `modules/segmentation/sam_segmentor.py` (ViT-H) | ✅ |
| (B) SAM 3D 복원 | `modules/segmentation/sam3d_reconstructor.py` → `facebookresearch/sam-3d-objects`를 별도 conda env에서 실행 | ✅ 논문 [17] 인용과는 다른 저장소입니다(D6) |
| (B) GLB → USD, SDF 충돌체 | `scripts/convert_mesh_to_usd_isaaclab.py` (`collision_approximation="sdf"`) | ✅ |
| (B) 실제 크기 정규화, O로 찾는 재질값, 자동 질량 | 같은 파일 | ❌ 크기 정규화 코드가 없고, 질량 0.5 kg·마찰 0.6/0.4·반발 0이 고정 기본값입니다(D4) |
| (C) 힘 지도 (PCA, 광선 두께, 테두리 보정, 가산점, primvar) | `modules/force_map/force_map_builder.py` | ❌ placeholder. 메쉬를 읽지 않습니다(D1) |
| (D) DexGraspNet 후보, TaskGrasp 필터, 재순위화 Eq. 4–6 | – | ❌ 없음 (README "in progress") |
| (E) diffusion 접근, 임피던스 Eq. 8–11 | – | ❌ 없음 |

## 2. 불일치와 논문에 적히지 않은 구현 요소

| # | 내용 | 영향 |
|---|---|---|
| D1 | **힘 지도는 메쉬와 무관한 placeholder입니다.** 20층 × 36구간 격자에 고정 Gaussian 가산점(테두리 +0.6, 바닥 +0.4)과 각도 방향 10% 사인 변동을 넣고, 재료 계수(paper 0.35, plastic 0.55, glass 0.85)를 곱한 뒤 10 N에서 자릅니다. `mesh_path`는 메타데이터에 기록만 합니다. 클래스 주석에 PCA, 광선 투사 두께, 테두리 보정, 꼭짓점 투영, primvar 저장이 모두 TODO로 적혀 있습니다. | 논문의 핵심 기여 1이 공개 코드에 없습니다. placeholder의 재료 순서(종이 < 플라스틱)는 Table I(종이컵 ≫ 플라스틱 컵)과 반대이고, 상한 10 N은 Table I의 989 N과 맞지 않습니다. |
| D2 | **기여 2–3(재순위화, 손가락별 임피던스)과 DexGraspNet·TaskGrasp 연결이 없습니다.** | Table II–IV를 코드로 재현할 수 없습니다. |
| D3 | 과제 이해 모델이 **VLM(Qwen2.5-VL-3B)**입니다. 논문 본문은 Qwen[14](2023 technical report, LLM)를 인용하고, Fig. 2는 "LLM-model Qwen 2.5"라고 적었습니다. 행동 유형도 다릅니다. 코드는 {take, pick_up, lift, find, hold, unknown}이고, 논문은 {pick up, pick and place, hand over}입니다. λ ∈ [0, 1]로 바꾸는 코드는 없습니다. Qwen이 실패하면 정규식이 "gently/carefully/fragile"을 찾아 모드를 정합니다. | 논문 실험의 λ = 0.3/0.7/1.0을 이 코드로 만들 수 없습니다. 세 물체 이름을 찾는 조회표(`_MATERIAL_OBJECTS`)가 재질을 정합니다. |
| D4 | 논문은 "메쉬를 실제 크기로 정규화하고, O로 찾은 조회표에서 ρ, μs, μd, e_r을 붙이고, 질량을 자동 계산"한다고 적었습니다. 코드는 크기 정규화가 없고, 질량 0.5 kg, 정지마찰 0.6, 동마찰 0.4, 반발 0이 모든 물체에 같은 기본값입니다. | 1.9 g 플라스틱 컵이라면 질량이 263배 큽니다. 논문 실험에 쓴 설정이 따로 있었는지는 알 수 없습니다. |
| D5 | 물체는 **강체**(`RigidBodyPropertiesCfg`)와 SDF 메쉬 충돌체입니다. | 벽의 변형이나 파손은 시뮬레이션할 수 없습니다. 논문의 "visible wall deformation"(Sec. IV-B)이 이 설정에서는 나올 수 없고, 논문에도 변형체 설정 설명이 없습니다. |
| D6 | SAM 3D는 `facebookresearch/sam-3d-objects`(arXiv:2511.16624)입니다. 논문 [17]은 arXiv:2306.03908(3D 장면 분할, `Pointcept/SegmentAnything3D`)입니다. v1 Fig. 2에는 Meta 로고가 있었고 v2에서 지워졌습니다. | 인용 오류입니다. 재현할 때 [17]을 따라가면 다른 모델을 받게 됩니다. |
| D7 | 시뮬레이터가 셋입니다. 실험은 Isaac Sim, 후보 검증은 Isaac Gym(DexGraspNet 방식), 공개 코드의 USD 변환은 Isaac Lab(`isaaclab.sh`)입니다. | 환경을 맞추는 비용이 큽니다. 접촉 모델이 시뮬레이터마다 달라 후보 검증 결과가 실험 환경에서 그대로 성립하는지 확인이 필요합니다. |
| D8 | 논문 이후에 추가된 기능: 실시간 카메라 모드(`scripts/run_live_camera_pipeline.py`), Isaac Sim 장면 뷰어. SAM 3D와 Isaac Lab의 CUDA·Omniverse 충돌을 피하려고 단계를 둘로 나눴습니다. | 파이프라인 앞단의 실용 정보입니다. |
| D9 | DexGraspNet 대조: Shadow Hand 자유도는 **22**입니다(`joint_angles_mu` 길이 22, `hand_pose = 3 + 6 + n_dofs`). 논문 Eq. 1은 θ ∈ R²⁴입니다. 에너지는 `E_fc + w_dis·E_dis + w_pen·E_pen + w_spen·E_spen + w_joints·E_joints`(기본 100, 100, 10, 1)로, 논문 Eq. 2에는 접촉 거리항 E_dis와 가중치가 빠져 있습니다. | 손목 관절 2개를 넣었는지 등은 알 수 없습니다. Eq. 2만 보고 구현하면 DexGraspNet과 다른 최적화가 됩니다. |

## 3. 수치 재계산

| # | 주장 (출처) | 재계산 | 판정 |
|---|---|---|---|
| C1 | F*_min 8.6×, 12.3×, 8.2× 향상 (Sec. IV-A) | 41.2/4.8 = 8.58, 3.7/0.3 = 12.33, 565/69 = 8.19 | ✅ Table I 영역 값의 비율 |
| C2 | Table II 여유와 위반 | 기준선 세 행이 여유 > 1인데 위반 1–2개. 역산 f_max: 기준선 3.43 / 0.25 / 32.9 N, 제안 5.28 / 0.63 / 45.6 N | ⚠️ 정의상 불가능한 조합. 제안 방법이 더 세게 잡음 |
| C3 | over-stiffness가 허용 하중의 2.5배 초과 (Sec. IV-B) | 0.74 / 0.3 = 2.47. 여유 0.3/0.74 = 0.41, 0.3/0.21 = 1.43 | ✅ |
| C4 | Table IV: 모드별 여유 | f_max = λ × 0.3 N이 세 자리까지 일치하고, 여유 × λ = 0.999, 1.001, 1.03 | ⚠️ 제어 법칙의 동어반복 (`03` §1.2) |
| C5 | 실험 2는 "힘 지도로 고른 파지"를 씀 | 실험 1의 선택은 테두리 3.7 N(f_max 0.63 N), 실험 2의 기준은 벽 0.3 N(f_max 0.21 N). Fig. 4의 손끝도 벽에 있음 | ⚠️ 서로 다른 파지 |
| C6 | under-stiffness는 "마찰 부족으로 미끄러짐" | 1.9 g 컵, μ = 0.3, 손가락 둘이면 손가락당 0.031 N으로 충분. 0.08 N은 그 2.6배 | ⚠️ 정역학으로 설명 안 됨 |
| C7 | under-stiffness 실패 vs gently 성공 | 0.08 N vs 0.09 N(12.5% 차이). F*_k = F_max이면 두 조건은 같은 강성(0.3 K_base) | ⚠️ 경계가 좁고 F_max 정의 없음 |
| C8 | 시행 수 | 실험 1은 rollout 6회(물체 3 × 방법 2), 실험 2는 5회. x/5는 손가락 수 | 통계 없음 |
| C9 | Shadow Hand 관절 24 (Eq. 1) | DexGraspNet 코드는 22 | ⚠️ D9 |
| T1 | 재순위화가 재료 정보를 쓴다 | α = 0이면 k_m과 무관하게 같은 후보를 고름(400회 중 100%) | `03` §2 |
| T2 | 계획·실행의 허용 하중이 같다 | 꼭짓점 간격 5 mm에서 테두리 전이 구간 접촉점의 5.6%가 실행 단계에서 1.5배 넘게 관대하게 보임 | `03` §4 |
| T3 | Table I 값의 크기 | σ t²로 보면 벽 두께 종이 0.31–0.40 mm, 플라스틱 0.09–0.11 mm, 유리 볼 1.2–1.5 mm에 해당 | 자릿수는 그럴듯함 (`03` §5) |

## 4. 구현 가능성 판단

### 4.1 공개 코드를 그대로 돌리는 경우

| 항목 | 요구 사항 | 비고 |
|---|---|---|
| 돌아가는 범위 | 명령 + RGB → Qwen → YOLO-World → SAM → SAM 3D → USD 장면 | 힘 지도는 placeholder이고, 파지 선택과 제어는 없습니다 |
| GPU | NVIDIA CUDA GPU | SAM 3D가 Kaolin, PyTorch3D 같은 CUDA 패키지를 쓰고, Isaac Lab도 CUDA가 필요합니다 |
| 소프트웨어 | Linux, conda env 2개(main, sam3d), Isaac Sim + Isaac Lab | README가 SAM 3D와 Isaac Lab을 같은 프로세스에 두면 CUDA·Omniverse 충돌이 난다고 경고합니다 |
| 가중치 | SAM 3D 체크포인트(HF gated), SAM ViT-H, YOLO-World v2, Qwen2.5-VL-3B | 자동 다운로드가 아직 없습니다 |
| 라이선스 | 코드 MIT, SAM 3D(SAM License), YOLO-World GPL-3.0 / ultralytics AGPL-3.0 | 상업적 사용에는 조건 확인이 필요합니다 |

**판단:** 공개 코드로 **논문 결과(Table I–IV)를 재현할 수는 없습니다.** 핵심 기여 세 개가 구현되지 않았고, 논문에도 필요한 값이 대부분 없습니다. 공개 코드로 할 수 있는 것은 앞단(언어 → 3D 메쉬 → Isaac 장면)까지이고, 그것도 NVIDIA GPU가 있는 Linux 머신이 필요합니다.

### 4.2 이 맥(M4 Pro, Apple GPU)에서

| 단계 | 가능 여부 | 근거 |
|---|---|---|
| 표 재계산, 재순위화·조회 toy | ✅ 완료 | numpy, 2초 |
| 힘 지도 알고리즘을 실제 메쉬에 구현 (PCA, 광선 두께, 테두리 보정, 꼭짓점 투영) | ✅ 가능 | 광선-삼각형 교차를 numpy로 직접 짜면 설치가 필요 없습니다. 컵 메쉬는 절차적으로 만들 수 있습니다 |
| Qwen2.5-VL-3B 과제 이해 | 조건부 | PyTorch MPS로 돌 수 있어 보이지만, torch·transformers 설치와 수 GB의 가중치(3B 파라미터, bf16 기준 약 7 GB로 추정)가 필요합니다. 확인하지 않았습니다 |
| SAM, YOLO-World | 조건부 | CPU·MPS로 추론은 될 것으로 보이며, 설치가 필요합니다 |
| SAM 3D 복원 | ❌ | CUDA 전용 의존성(Kaolin, PyTorch3D CUDA 빌드) |
| DexGraspNet 후보 생성, Isaac Sim/Lab/Gym 실험 | ❌ | CUDA 필요, Isaac은 macOS를 지원하지 않습니다 |

### 4.3 아이디어만 다시 구현하는 경우

| 구성 요소 | 난이도 | 메모 |
|---|---|---|
| 재순위화 (Eq. 4–6) | 낮음 | 수십 줄입니다. 분산 항은 변동계수로 바꾸고, 계획과 실행의 조회 방식을 통일해야 합니다(`03` §2, §4) |
| 힘 지도 (두께 → 허용 하중) | 중간 | 수백 줄입니다. F_base의 꼴과 가산점은 직접 정해야 합니다. σ t²에서 시작할 수 있습니다(`03` §5) |
| 손가락별 임피던스 | 낮음–중간 | 시뮬레이터의 관절 PD 이득을 손가락별로 바꾸면 됩니다 |
| **힘 지도의 검증** | 높음 | 이 연구에서 빠진 부분입니다. 변형체 시뮬레이션(FEM)이나 실제 컵의 눌림·파손 하중 측정이 있어야 의미 있는 재현이 됩니다 |

**요약:** 알고리즘 자체는 짧고 단순해서 다시 구현하기 쉽습니다. 하지만 논문 결과를 재현하는 것은 불가능하고(코드와 값 부재), 재현해도 같은 힘 지도로 판정하는 한 순환 평가입니다. 가치 있는 작업은 힘 지도를 독립된 물리 기준(FEM, 실측)과 비교하는 것입니다.
