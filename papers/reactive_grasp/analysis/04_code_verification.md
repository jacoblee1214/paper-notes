# 04. 코드 검증: 공개된 같은 연구실 코드 대조, 수치 재계산, 구현 가능성

- **논문 공식 코드:** 미공개입니다. 프로젝트 페이지에 "Code (Soon)"만 있고, `sangminkim-99/reactive-grasp-whole-hand`(@ `3ab1aa5`)는 웹페이지 저장소입니다.
- **대조 대상:** 공동 교신저자 Yonghyeon Lee 연구실의 공개 저장소 `rophi-lab/RoPhiTutorial` @ `85aaa4c`(2026-08-02, 논문보다 7주 앞섬)입니다. `controller/`, `utils/grasping/`, `configs/`, `env/`만 sparse clone했습니다(`repos/`, gitignore).
  - 같은 하드웨어(Flexiv Rizon 4 + Robotis 5F)용 force-closure 제어기, MuJoCo 평가 제어기와 환경, 기준선 제어기가 있습니다.
  - **이 저장소는 논문의 공식 코드가 아닙니다.** 시뮬레이션 설정값이 논문과 정확히 같아서 같은 계열 구현으로 보이지만, 논문 실험에 쓴 버전이라는 보장은 없습니다. 아래 D 항목은 "공개된 가장 가까운 구현과의 차이"로 읽어야 합니다.
- **범위:** 코드 읽기, 설정 대조, CPU 재계산만 했습니다. 공개 결과 파일이나 학습된 무언가는 없습니다(이 방법은 학습이 없습니다).

## 1. 모듈 매핑

| 논문 | 공개 코드 위치 | 일치 여부 |
|---|---|---|
| 링크 표면 표본, 2점/cm² | `utils/grasping/link_surface_sampler.py`, `surface_density_per_cm2: 2.0` | ✅ |
| 거리장 사전 계산 + Warp 질의 (Sec. III-B) | `utils/grasping/udf_contact_detector.py: UDFGrid` (128³, padding 2 cm, Warp GPU) | ⚠️ SDF가 아니라 **UDF**(D2) |
| 링크마다 임계값 안의 가까운 K개 (Eq. 1) | `detect_contacts(..., dist_threshold, max_contacts_per_link)` | ✅ |
| QP 목적함수 ½‖Gf‖² (Eq. 3) | `force_gen.generate_contact_forces`: z = [f; w], w = Gf, 비용 ½‖w‖² | ✅ (lifted 형식) |
| 마찰 피라미드 (Eq. 4–5) | 접점당 부등식 4개 `±f_t1 ± f_t2 − μ f_n ≤ 0`, `f_n ≥ 0` | ✅ |
| Σf_n ≥ f_n,min (Eq. 6) | `A_min_fn` | ✅ |
| 토크 한계 (Eq. 7–8) | `A_torque = Jr_contactᵀ`, `torque_limit: 0.2` | ✅ |
| Actuation consistency (Eq. 9) | `enable_achievable_force`: `U, s = svd(Jr_act)`, `rank = sum(s > 1e-6)`, `N = U[:, rank:]`, `±null_space_margin` | ✅ 식과 같음. 손바닥은 `actuated_mask`로 제외 |
| 손바닥은 수동 반력 | `actuated_mask = c_link_indices != palm_link_idx` | ✅ |
| OSQP | `osqp.OSQP().setup(..., max_iter=1000, polish=False)`를 매 호출마다 새로 생성 | ✅ (설정은 논문에 없음, D7) |
| τ* = J_actᵀ f* (Eq. 10) | `_compute_tau_from_J_and_f` | ⚠️ 접근 토크가 더해짐(D3) |
| 실패 시 직전 토크 유지 | 시뮬레이션 평가 `MjEvalGraspControl`: `tau_qp = self._prev_tau_qp` | ✅ (실물 제어기는 다름, D5) |
| 기준선 feedforward / impedance | `control_type: feedforward_force_control`, `task_space_impedance_control`, 제안 방법은 `real_time_feedforward_force_control` | ✅ |
| Sec. IV 설정 μ 0.7, f_n,min 10 N, τ_max 0.2, d_thresh 1 cm, K 5, ε 0.5 N | `configs/robotis_5f/grasp_squeeze/default.yaml` | ✅ 여섯 값 모두 같음 |

## 2. 차이와 논문에 적히지 않은 구현 요소

| # | 내용 | 영향 |
|---|---|---|
| D1 | 시뮬레이션 평가 설정(`grasp_squeeze/default.yaml`)이 논문 Sec. IV와 여섯 값 모두 같습니다(μ, f_n,min, τ_max, d_thresh, K, ε). 반면 `controllers/grasp_eval.yaml`은 K = 3입니다. | 같은 계열 구현이라는 강한 증거입니다. 다만 설정 파일이 두 벌이라 논문 실험이 어느 쪽이었는지는 논문 값으로만 판단할 수 있습니다. |
| D2 | **거리장이 부호 없는 UDF**입니다(`build_distance_field`, backend `auto`/`udf`). 법선은 UDF 기울기에서 얻어서, 손 표본점이 물체 안으로 들어가면 법선 방향이 뒤집힙니다. 부호로 뒤집어 주는 옵션(`flip_obj_normal_by_sign`)은 기본으로 꺼져 있습니다. | 논문은 "signed distance field"라고 씁니다. 저자들이 한계로 든 "얇은 물체에서 몇 mm 오차로 법선이 뒤집힌다"는 현상이 UDF에서 특히 쉽게 생깁니다. 침투한 점의 힘 방향이 반대로 할당될 수 있습니다. |
| D3 | **토크 = QP 토크 + 손끝 접근 토크(+ 인력 토크)**입니다. 접점이 없는 손가락의 손끝을 UDF 최근접 표면 쪽으로 0.1 m/s로 끌어당기는 속도 QP 토크(`APPROACH_KD = 0.1`, 지수 평활 0.1)가 더해집니다. 인력 토크는 `attract_stiffness: 0`이라 꺼져 있습니다. 접점이 하나도 없으면 접근 토크 + PD 닫힘만 씁니다. | 논문 Eq. 10은 τ* = J_actᵀ f*뿐입니다. "닿지 않은 손가락을 닫는" 동작이 QP 밖에서 따로 만들어집니다. 파지가 자리 잡는 단계의 성공(01 §3.3)에 이 장치가 얼마나 기여하는지는 알 수 없습니다. |
| D4 | QP에 **중력 wrench 항**(‖Gf + w_g‖²)과, 질량을 알면 f_n,min을 m·g 이상으로 올리는 옵션이 있습니다. 접점별 ε 가중 정규화 ½λΣε_i²‖f_i‖²(`eps_translation/rotation`)도 있습니다. | 논문은 중력·외력을 보상하지 않는 준정적 조임이라고 씁니다. 시뮬레이션은 중력을 끄므로 영향이 없습니다. 정규화는 기본으로 꺼져 있어, 03 §1의 비유일성이 코드에도 그대로 있습니다. |
| D5 | **실물 제어기 설정이 논문 서술과 다릅니다**(`controllers/three_finger_reactive_force_closure.yaml`). <br>· K = 3 (논문: 시뮬레이션과 같은 K = 5) <br>· `enable_achievable_force: false` (Eq. 9 끔) <br>· `enable_gravity_wrench: true`, f_n,min 5 N을 m·g 이상으로 올림 <br>· 힘 제어 50 Hz <br>· "세 손가락" reaching <br>· QP가 실패하면 Eq. 9를 빼고 f_n,min을 절반으로 낮춰 다시 풀고, 그래도 실패하면 손 토크 0 + 정지 | 공개 튜토리얼은 논문 실물 실험(다섯 손가락, K = 5)보다 앞선 버전으로 보입니다. 논문 실물 실험의 정확한 설정은 공개 자료로 확인할 수 없습니다. |
| D6 | 시뮬레이션 QP 실패 시 직전 토크를 유지합니다. | 논문과 같습니다. |
| D7 | OSQP를 매 주기 새로 만들어 0에서 시작합니다(warm start 없음, `max_iter=1000`, `polish=False`). | 논문에 없는 설정이지만, 03 §1의 해 선택(최소 노름에 가까운 해)을 정합니다. 1,000회 안에 수렴하지 못하면 "solved inaccurate"(status 2)도 성공으로 받습니다. |
| D8 | MuJoCo 손 모델의 접촉 geom은 `condim=3`, `friction="0.7 …"`, `timestep 0.001`, Euler 적분, solver 반복 5회입니다. | 시뮬레이터의 마찰계수가 제어기 μ와 같습니다. MuJoCo 기본 피라미드 원뿔(사전 지식)이면 시뮬레이터의 마찰 모델이 Eq. 5와 같은 모양이라, 시뮬레이션에서는 "보수적 근사"가 아닙니다. |
| D9 | 섭동 환경(`MjMeshesGraspEvalEnv`)은 키 입력으로 시작하는 대화형 도구입니다. 회전각을 N(0, σ_rot = 2 rad)에서 뽑고, 스프링 Kp = 1000·m에 힘 상한 10 N을 둡니다. 논문의 고정 각도(30/60/90°), 6방향 하중 시험, 790개 일괄 평가 스크립트는 공개 코드에 없습니다. | 논문 수치를 재현할 하네스가 공개되지 않았습니다. |

## 3. 수치 재계산 (`analysis/table_check.py`, `fig_digitize.py`, `qp_toy.py`, `pose_error_toy.py`)

| # | 확인 | 결과 | 판정 |
|---|---|---|---|
| C1 | Fig. 3 막대 27개의 정수 정합, 전체 성공률, 본문 서술 | 모두 정수 k/n 가능. Ours 92.3% [90.2, 93.9], Impedance 70.0%, Feedforward 64.7%. "32–56% vs 82–95%", "1/3(34.4%)" 일치 | ✅ |
| C2 | Fig. 4 판독(본문 두 점으로 보정, Ours 0° = 100.0 검증) | 0°는 2% 격자, 30–90°는 0.2% 격자와 맞음. 90°에서 Ours 88.6 vs Fingertip-only 72.4, p ≈ 1×10⁻¹⁰ | ✅ |
| C3 | 다른 그림의 시행 수 | Fig. 7의 750회 값은 정수 가능(K = 1 vs 5 p ≈ 0.13). 정밀 파지 38.5→42.1%는 68 파지 1회로 불가능하고, 5회 반복이면 131→143, p ≈ 0.39 | ⚠️ 시행 수 미기재 |
| C4 | Table I과 본문 | 201 s, 78회, 8회 놓침, 6회 회복. 교란당 놓침 10.3% [5.3, 19.0]. 작업공간 이탈 2회를 빼면 회복 6/6 | ✅ |
| C5 | 계산 속도 | t ≈ 2.45 + 0.140·N ms. K = 5(65 접점) 예측 86 Hz vs 페이지 84 Hz. 실물 루프는 30 Hz | ✅ |
| C6 | 하중 대 용량 | μ f_n,min = 7 N(7.1 kg 상당) ≫ 0.1 mg. Fig. 5 잔차 0.428 N이 0.44 kg 미만 물체에서는 하중보다 큼 | 해석 주의 |
| C7 | Eq. 9의 실효 크기 | toy에서 d = 25–205, ε√d = 2.5–7.2 N. ε = 0.5에서 실현 잔차가 거의 줄지 않음 | ⚠️ `03` §2 |
| C8 | 자세 오차 아래 접점 | (30°, 5 mm), d_thresh 1 cm에서 재현율 88%, 법선 오차 중앙값 28°, 비접촉 점 오검출 49% | ⚠️ `03` §5 |

## 4. 구현 가능성 판단

### 4.1 공식 코드로 논문을 재현하는 경우
**지금은 불가능합니다.** 공식 코드가 "Soon" 상태이고, 섭동·하중 시험 하네스와 790개 평가 스크립트도 없습니다.

### 4.2 공개된 같은 연구실 코드로 가까이 재현하는 경우

| 항목 | 요구 사항 | 이 맥(M4 Pro, CUDA 없음) |
|---|---|---|
| MuJoCo 평가(`MjMeshesGraspEvalEnv` + `MjEvalGraspControl`) | mujoco, pinocchio, trimesh, osqp, warp, omegaconf, LCM 통신 계층 | 조건부로 가능해 보입니다. MuJoCo와 OSQP는 CPU에서 돌고, UDF 계산은 `udf_device`를 cpu로 바꾸거나 `backend: mesh`(trimesh CPU)를 써야 합니다. 설치 승인이 필요하고, 실제 동작은 확인하지 않았습니다 |
| 목표 파지 | Dexonomy[5] 파지와 YCB 메쉬. 튜토리얼에는 일부 물체의 파지 예측 파일만 있음 | 790개 파지를 다시 만들려면 Dexonomy를 돌려야 합니다(별도 확인 필요) |
| 섭동·하중 시험 | 직접 작성(논문 Fig. 2 프로토콜) | 수십~수백 줄 스크립트 |
| 실물 | Flexiv Rizon 4 + Robotis 5F + RealSense, FoundationPose(GPU), Point2Pose + 2DGS 스캔 | 불가능(하드웨어, NVIDIA GPU) |

### 4.3 아이디어를 직접 구현하는 경우

| 구성 요소 | 난이도 | 메모 |
|---|---|---|
| 기하 접점 추정(링크 표본 + SDF 질의 + 링크별 top-K) | 낮음 | numpy만으로도 됩니다. UDF가 아니라 **SDF**를 쓰고 침투 부호를 처리해야 합니다(D2) |
| 힘 분배 QP | 낮음 | OSQP나 cvxpy로 수십 줄입니다. 해를 정하는 정규화 항(예: λ‖f‖²)과 ε의 크기(√d 조정)를 함께 정하는 것을 권합니다(`03` §1–2) |
| 토크 변환 + 접근 토크 + 전환 상태기계 | 중간 | 논문 Eq. 10만으로는 닿지 않은 손가락이 닫히지 않습니다. 공개 코드의 접근 토크(D3) 같은 장치가 필요합니다 |
| reactive reaching(Eq. 11) | 중간 | 충돌 제약이 있는 관절 속도 QP. Lee et al. [20] 코드는 찾지 못했습니다 |
| 실시간 물체 자세 추적 | 높음 | FoundationPose 변형(NVIDIA GPU). 정확도가 접점 추정을 좌우하지만, 성공률은 큰 오차에도 버팁니다(`03` §5) |

**요약:** 알고리즘은 짧고, 같은 연구실 코드가 핵심을 거의 그대로 보여 줍니다. 시뮬레이션 쪽은 이 맥에서도 시작할 수 있어 보입니다. 다만 논문 수치를 재현하려면 평가 하네스를 직접 만들어야 하고, 실물 결과는 하드웨어와 NVIDIA GPU 없이는 재현할 수 없습니다.
