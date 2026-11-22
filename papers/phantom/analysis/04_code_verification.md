# Phantom 공식 코드 대조 (04)

분석일 2026-11-22. 대상: `MarionLepert/phantom` 커밋 `a8bb81c` (2025-08-29), 얕은 복제(`repo/`). **정적 읽기만 했습니다.** 실행, import, 설치, 서브모듈 초기화, 데이터 다운로드를 하지 않았습니다. 대조 기준은 논문 v2의 3.2–3.3절, 4절, 6.6–6.8절입니다.

## 1. 한눈에

- 공식 저장소는 **사람 영상 처리 파이프라인**(bbox → 손 2D·3D → 행동 → 평활화 → 인페인팅 → 로봇 오버레이)입니다. 논문의 핵심 아이디어(HaMeR+ICP 행동, E2FGVI 인페인팅, 깊이 기반 로봇 오버레이, 추론 시 오버레이)는 코드에 있습니다.
- **논문에 없는 단계가 있습니다.** 가우시안 평활화(위치는 GP, 회전은 SLERP), 손 검출 실패 시 앞 값 유지와 중립 자세 대체, DINO 기반 bbox 검출(D3, D4, D6).
- **논문에 있는데 코드에서 찾지 못한 것이 있습니다.** Hand Mask 확산 모델과 팔 마스크 증강(D5, 표 1과 표 10의 비교 대상), 엄지·검지 마지막 두 관절의 1자유도 제한(논문 3.2절과 달리 엄지는 제한이 없음, D1), 그리퍼 하위 20% 백분위 닫힘(D2).
- **학습 코드는 없습니다.** README는 Diffusion Policy를 외부에서 쓰라고 합니다. 표 8의 학습 설정은 저장소 설정 파일에 없습니다(D7).
- 기본 설정 파일 `configs/default.yaml`은 "PHANTOM paper settings"라고 적혀 있고 `constrained_hand: true`, `depth_for_overlay: true`, `output_resolution: 240`로 논문 설정과 맞습니다(D8).

## 2. 대조 결과

### 2.1 논문과 다른 것 (D1–D7)

| 번호 | 논문 | 코드 | 판정 |
|---|---|---|---|
| D1 | 3.2절: 엄지와 검지의 **마지막 두 관절**을 **1자유도**로 제한 | `hand.py` `PhysicallyConstrainedHandModel.joint_limits`: 검지 관절 6, 7(마지막 두 관절)은 x, z 축 ±π/40, y 축 ±π/8로 제한 (1자유도가 아니라 3축 좁은 범위). 엄지 관절 2, 3(마지막 두 관절)은 x, y, z 모두 ±π로 **제한 없음** | **불일치.** 엄지 쪽은 제한이 없고, 검지 쪽은 1자유도가 아님 |
| D2 | 3.2절: 파지 중 슬립을 줄이려고 **그리퍼 거리 하위 20% 백분위를 완전히 닫힘**으로 설정 | 저장소 전체에서 `percentile`, `quantile` 검색 결과 없음. `action_processor.py`의 `_refine_actions`는 검출 실패 구간만 앞 값으로 채움. `robotinpaint_processor.py:436`의 `min + 0.2 × (max − min)`은 오버레이의 그리퍼 열림 분류용 임계값이며 다른 목적 | **코드에서 찾지 못함** (정적 검색 한정, 서브모듈 미확인) |
| D3 | 논문 3.2절에 평활화 단계 없음 | `smoothing_processor.py`: 위치와 그리퍼 거리에 가우시안 과정(GP) 평활화, 회전 행렬에 가우시안 SLERP 평활화(sigma 2, 커널 9) | **논문에 없음.** 평활화 파라미터가 결과에 영향을 줄 수 있음 |
| D4 | 논문에 검출 실패 처리 없음 | `action_processor.py`: 검출 실패 구간은 마지막 유효 값 유지(carry-forward), 전체 실패면 중립 자세(`_get_neutral_actions`), 중립 그리퍼 폭 0.085 m | **논문에 없음** |
| D5 | 6.8절: 확산 모델로 6-DOF 자세에서 손 마스크 생성, 64×64 학습 후 초해상도, 어텐션 주입. 6.8.2: 팔 마스크를 몇 픽셀 흔드는 증강 | 확산 모델, 학습 스크립트, 증강 코드를 찾지 못함 (`diffusers`, `UNet`, `ddpm` 검색 결과 없음). `robotinpaint_processor.py`는 `_dilate_mask`로 팽창만 함 | **코드에서 찾지 못함.** 표 1·2·10의 Hand Mask 결과를 이 저장소로 재현할 수 없음 |
| D6 | 논문은 손 검출을 HaMeR 경로로 설명 | `bbox_processor.py`: DINO 검출기 (신뢰도 0.2, 겹침 0.3). 논문에 없음 | **논문에 없음.** 입력 전처리 단계 |
| D7 | 표 8: Diffusion Policy 설정 (관측 2, 행동 8, 해상도 240, 배치 200/256, 학습률 1e-4, 증강) | 이 저장소에 학습 코드, 설정 없음. README는 외부 Diffusion Policy를 쓰라고 함 | **확인 불가** (저장소 밖) |
| D8 | 4절: Franka (OSC), Kinova (IK) | `twin_robot.py`는 기본 로봇 Panda, 그리퍼 Robotiq85, 로봇 이름 예시에 Kinova3. Kinova 단일 팔 IK 경로는 찾지 못함 (Kinova3는 `twin_bimanual_robot.py`의 양팔 설정에 있음) | **부분 대응.** 논문의 Kinova 단일 팔 실험 설정은 확인 못함 |
| D9 | 3.3절: 깊이로 가려짐 처리 | `depth_for_overlay: true` (기본 설정), `robotinpaint_processor.py:514` 깊이 기반 오버레이 | **일치** |

### 2.2 논문과 맞는 것 (확인)

| 항목 | 논문 | 코드 |
|---|---|---|
| 손 메시와 키포인트 | HaMeR, 21 관절, 778 정점 (3.2) | `detector_hamer.py`, 서브모듈 `phantom-hamer` (MIT) |
| 손·팔 분할 | SAM2 (3.2, 3.3) | `detector_sam2.py`, `segmentation_processor.py` |
| 점군 정합 | ICP (3.2) | `hand_processor.py` `icp_registration` (open3d, 복셀 0.005 m). 논문에 복셀 값 없음 |
| 엄지 끝과 검지 끝 거리로 그리퍼 | 3.2 | `action_processor.py` `_compute_gripper_opening`: 엄지 끝과 검지 끝의 유클리드 거리 |
| 제한된 손 모델 사용 | 3.2 | 기본 `constrained_hand: true`, `PhysicallyConstrainedHandModel` (D1의 제한 값은 있음) |
| 인페인팅 | E2FGVI (3.3, 4.4) | `handinpaint_processor.py`, 서브모듈 `phantom-E2FGVI` |
| 로봇 오버레이 | MuJoCo 기반 렌더링 (4) | `robotinpaint_processor.py`, robosuite 기반 `twin_robot.py` (robosuite는 MuJoCo 위에서 돌아감) |
| 카메라 외부 파라미터 | 알려져 있음 (3.1) | `camera/camera_extrinsics.json`, 설정의 `camera_extrinsics` |
| 해상도 | 240 (표 8) | `output_resolution: 240` (설정), 입력 1080 (Zed2 HD1080, README) |
| 행동 좌표 변환 | 카메라 → 로봇 (3.2) | `_convert_pts_to_robot_frame` (`T_cam2robot`) |

### 2.3 논문에 설명이 없는 나머지 구현 세부

- **실행 모드:** `bbox`, `hand2d`, `hand3d`, `hand_segmentation`, `arm_segmentation`, `action`, `smoothing`, `hand_inpaint`, `robot_inpaint`, `all` (설정 파일 주석). `all`은 전 단계를 돌립니다.
- **Masquerade용 확장:** 양팔 설정(`shoulders`), Epic-Kitchens 모드(`epic.yaml`), 여러 로봇 이름(Panda, Kinova3, UR5e, IIWA, Jaco). 이 확장은 Phantom 논문 범위가 아닙니다.
- **깊이 요구:** `hand3d`는 깊이가 필요하며 README는 왼손에서만 시험했다고 적습니다. 오른손 실험은 논문에 없습니다.

## 3. 구현 가능성 (§4 판정)

**판정: 이 맥(Apple M4 Pro, 약 24 GiB 통합 메모리, MPS, CUDA 없음)에서 논문 결과를 재현할 수 없습니다. 구성 요소 일부는 CPU나 MPS에서 검사할 수 있습니다.**

| 구성 | 이 맥에서 | 이유 |
|---|---|---|
| 손 자세 → 행동 변환 (중점, 평면 방향, 그리퍼 거리) | **가능 (CPU, numpy)** | 순수 기하. 입력은 합성 키포인트로 대신 가능 |
| 깊이 기반 가려짐 처리와 로봇 오버레이 | 조건부 | MuJoCo 렌더링은 가능. RGBD 데이터와 카메라 파라미터가 필요 |
| HaMeR, SAM2, E2FGVI 실행 | 조건부 (MPS 여부 미확인) | 설치와 가중치 다운로드 승인 필요. 서브모듈 초기화와 MANO 모델 등록 필요 (README 3단계) |
| Hand Mask 확산 모델 (표 9) | 불가 | 코드 없음 (D5). 직접 구현해도 학습 자원이 큼 |
| Diffusion Policy 학습 (표 8) | 불가 (논문 규모) | 외부 저장소, CUDA 전제 설정. MPS에서 소규모 학습은 가능하지만 논문 설정 재현은 아님 |
| 실로봇 평가 (표 1, 2, 5, 6) | 불가 | Franka, Kinova 하드웨어와 장면 필요 |
| 데이터 수집 (사람 영상 268–950개) | 불가 | 데이터는 저장소에 없음 |

**노력 추정 (측정 전, 추정):**
- 파이프라인 전체 (서브모듈과 가중치 설치, 샘플 영상 한 개 처리): 1–2주.
- 빠진 Hand Mask 확산 모델 구현 포함 표 1 재현: 수 주 이상 (GPU 권장).
- 실로봇 재현: 장비와 장면이 있어야 하며, 이 저장소 범위 밖입니다.

## 4. 라이선스와 출처

- **저장소:** MIT, `LICENSE` 파일 확인 (Copyright 2025 Stanford Interactive Perception and Robot Learning Lab). GitHub API는 MIT로 표시합니다.
- **서브모듈:** `phantom-hamer` MIT, `sam2` Apache-2.0, `phantom-E2FGVI`는 API에서 NOASSERTION이므로 원본 E2FGVI 라이선스를 따로 확인해야 합니다(미확인). `phantom-robosuite`와 `phantom-robomimic`은 조회하지 않았습니다.
- **논문:** arXiv 라이선스 CC BY 4.0. 논문 그림은 공개본에 넣지 않음 (`README.md` 라이선스 절).
- 이 저장소의 코드는 정적으로 읽었을 뿐이고 복사하지 않았습니다. `repo/`는 gitignore에 있습니다.
