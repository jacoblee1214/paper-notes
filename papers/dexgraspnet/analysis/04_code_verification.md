# 04. 논문 ↔ 공식 코드 대조: DexGraspNet

대상: `PKU-EPIC/DexGraspNet` main 브랜치, 커밋 `bd1b13d` (2025-01-06), `repo/`에 얕은 clone. **정적 읽기만 했습니다.** 실행, import, 설치, pickle 로드를 하지 않았습니다. 대조 기준은 논문 v2(`paper/paper.txt`)입니다. 차이 항목은 D1–D15, 확인 질문은 §5입니다.

## 1. 저장소 범위

| 경로 | 내용 | 상태 |
|---|---|---|
| `grasp_generation/` | 생성(`scripts/generate_grasps.py`), 에너지(`utils/energy.py`), 초기화(`utils/initializations.py`), 최적화(`utils/optimizer.py`), 손 모델(`utils/hand_model.py`), Isaac 검증(`utils/isaac_validator.py`, `scripts/validate_grasps.py`) | 본 대조의 중심 |
| `asset_process/` | 추출, ManifoldPlus, 정규화(`normalize.py`), CoACD 분해 | 물체 준비 대조 |
| `grasp_generation/mjcf/` | ShadowHand MJCF와 메시, 접촉 후보 `contact_points.json`, 자기 관통 키포인트 `penetration_points.json` | 접촉 후보 140개 확인 |
| `data/` | 샘플 `.npy` 5개와 메시 5개 | 로드하지 않음 (pickle) |
| `thirdparty/pytorch_kinematics/` | 수정된 포크 | 대조 대상 아님 |
| 없음 | Q1 계산, 엔트로피, DDG·GraspTTA 학습·평가, 벤치마크 집계, 데이터셋 통계 | D12 |
| 다른 분기 | `allegro`, `mano`, `mano-table`, `table` | 클론하지 않음 |

## 2. 일치하는 항목

| 논문 | 코드 | 확인 |
|---|---|---|
| 식 (1) \(E_{fc} = \|Gc\|^2\) | `energy.py` `cal_energy`: `E_fc = norm * norm` | 일치 |
| 식 (2) \(E_{dis}\) (접촉 거리 합) | `energy.py`: `E_dis = sum(distance.abs())` | 일치 (절댓값) |
| 역 관통 에너지 (물체 표면점이 손 안에 있는 깊이) | `energy.py`: `object_surface_points` (2000점)를 `hand_model.cal_distance`로 평가, 음수는 0 | 일치 |
| 식 (4) 자기 관통 \(E_{spen}\) | `hand_model.py` `self_penetration`: `max(0.02 − d, 0)` 쌍 합 | 일치 (임계는 D10) |
| 식 (5) \(E_{joints}\) | `energy.py`: 관절 한계 초과량 합 | 일치 |
| 식 (6) 가중치 100, 100, 10, 1 | `generate_grasps.py` 기본값 `w_dis, w_pen, w_spen, w_joints` | 일치 |
| 접촉 후보 140개 | `contact_points.json` 링크별 합: 10×3 + 10×3 + 10×3 + 10×3 + 10×2 = 140 | 일치 |
| 접촉점 n = 4 | `--n_contact` 기본 4 | 일치 |
| 자유도 d = 22 | `joint_names` 22개 | 일치 |
| 최대 6000 반복 | `--n_iter` 기본 6000 | 일치 |
| 초기화: 볼록 껍질을 0.2 m 팽창 | `initialize_convex_hull`: `vertices += 0.2 * unit normal` | 일치 |
| 초기화: 손바닥이 물체를 향함, 관절은 절단 정규분포 | `rotation_hand = euler(0, −π/3, 0)`, `torch.nn.init.trunc_normal_` (σ = 0.1 × 범위) | 일치 (σ는 논문 미기재) |
| 단위 구 정규화 | `normalize.py`: 중심 이동 후 `dmax × 1.03`으로 나눔 | 일치 |
| 생성 후 검증: 100 스텝 | `IsaacValidator(sim_step=100)` | 일치 |
| 6방향 중력 | `test_rotations` 6개 (축 정렬 회전) | 일치 |
| 성공 = 100 스텝 후 접촉 존재 (6방향 모두) | `run_sim`: 접촉 쌍 존재 플래그, `sum(result[i*6:(i+1)*6]) == 6` | 일치 (접촉 존재만 봄, D6) |
| 관통 임계 0.1 cm | `validate_grasps.py`: `penetration_threshold = 0.001` (m) | 숫자 일치, 정의 다름 (D3) |

## 3. 차이 D1–D15

| 번호 | 논문 | 코드 | 영향 | 근거 |
|---|---|---|---|---|
| **D1** | §III-B.3: MALA를 "단순화하여 단순 경사 하강법으로 \(T, R, \theta\) 갱신". 잡음은 없음 | `optimizer.py` `Annealing.try_step`: 경사를 EMA 제곱 평균(μ = 0.98)으로 나누는 RMSProp형 전처리, 스텝 크기 0.005 × 0.95^(⌊t/50⌋). `accept_step`: 온도 18 × 0.95^(⌊t/30⌋)의 Metropolis 수락·기각 | 논문 설명보다 복잡함. 수락·기각 단계는 "단순 GD"가 아님. 재구현 시 이 세 가지를 모두 넣어야 결과가 나옴 | `optimizer.py` 전체 |
| **D2** | v2 §III-B.3: "접촉점을 매 반복 후보에서 무작위 샘플". v1(MALA 설명 [19]): "작은 확률로 재샘플" | `try_step`: 접촉 인덱스 각각을 확률 `switch_possibility = 0.5`로 재선택 | 매 반복 전부 재샘플도, 작은 확률도 아님. 수렴 특성에 영향 | `optimizer.py` `switch_mask` |
| **D3** | §III-C: 최대 관통 깊이(물체 점에서 손 메시까지) 0.1 cm 초과면 실패 | `validate_grasps.py`: `estimated = E_pen_array < 0.001`. `E_pen`은 **생성 때 계산한 에너지 값**(2000 물체 표면점의 손 SDF 양수 합) | 최대 깊이가 아니라 합이라서 기준의 의미가 다름. 시뮬레이션 관통을 재지 않음 | `validate_grasps.py` `E_pen_array` |
| **D4** | §III-C: 시뮬레이션 설정을 설명. 사전 보정 단계는 없음 | `validate_grasps.py` 기본 실행(`--no_force` 없이)은 시뮬레이션 전에 **관절만** `hand_state[:, 9:] += grad × 500`으로 바꿈. 접촉점을 법선으로 1 mm 옮긴 목표와의 차이 손실을 만들고, 그 경사로 관절을 한 번 갱신. 저장은 보정 전 `qpos`를 씀 (`new_data_dict["qpos"] = data_dict[i]["qpos"]`) | **저장된 자세와 검증된 자세가 다름.** 데이터셋의 파지 품질은 보정 뒤 기준으로 판정되었지만, 파일에는 보정 전 값이 들어 있음 | `validate_grasps.py` 약 120–125행, 190행 |
| **D5** | §III-C: 접촉 링크를 법선 방향으로 움직인 목표 자세를 위치 제어 목표로 설정해 능동 힘을 줌 | `isaac_validator.py` `add_env(target_qpos=None)`에는 구현이 있음. 그러나 `validate_grasps.py`는 `target_qpos`를 넘기지 않음 | 논문에 적힌 능동 힘 단계는 공식 검증 경로에 없음 | `isaac_validator.py` 164–169행, `validate_grasps.py` `add_env` 호출부 |
| **D6** | "still in contact after 100 steps"(§III-C) | 접촉 쌍이 하나라도 있으면 성공. 들어 올림·버팀 여부는 보지 않음 | 논문의 "파지 유지"보다 느슨함. 벤치마크의 "1방향 이상" 기준과 함께 보면 판정 기준이 세 가지 | `isaac_validator.py` `run_sim` |
| **D7** | §IV: 검증을 통과한 파지만 저장 | `result_list`에 `qpos`와 `scale`만 저장 (E 항목 제거) | 일치 (README와도 맞음) | `validate_grasps.py` |
| **D8** | §III-A: "remesh into manifolds", "collision meshes" | `normalize.py`: `is_watertight and volume > 0.05` (단위 구 기준). 이 필터는 논문에 없음 | 두께가 얇은 물체를 버리는 필터. 논문은 "두께 없는 메시도 처리 가능"을 강조하므로 설명이 필요. 필터 전 물체 수는 본문에 없음 | `asset_process/normalize.py` 약 28행 |
| **D9** | §III-A: "augment each object by uniformly scaling with 5 fixed sizes" | `object_model.py`: `scale_choice = [0.06, 0.08, 0.10, 0.12, 0.15]` m. 각 파지(배치 원소)마다 무작위 선택 | "5355 물체"가 메시 수인지, 메시 × 크기 수인지 모호함. 공식 코드는 파지마다 한 크기 | `object_model.py` 45, 66행 |
| **D10** | Fig. 2a: 키포인트 구 반지름 δ = 1 cm. 식 (4)의 δ 값은 본문에 없음 | `hand_model.py`: `dis = 0.02 − dis` (쌍 거리 2 cm). 두 구의 반지름 합으로 보면 δ = 1 cm와 맞음 | 정의만 다르고 값은 맞을 가능성이 높음. 본문에 명시가 필요 | `hand_model.py` `self_penetration` |
| **D11** | 본문에 없는 상수들 | `generate_grasps.py` 기본값: 스텝 크기 0.005, 주기 50, 온도 18, 온도 감쇠 0.95 (주기 30), μ = 0.98, 접촉 재선택 0.5, 거리 [0.2, 0.3] m, 원뿔 ±π/6, 관절 jitter 0.1, 물체 표면점 2000 | 논문 재현에 필요한 상수 대부분이 본문에 없음 | `generate_grasps.py` 인자 정의 |
| **D12** | Q1(Eq. 7), 관절 엔트로피, DDG와 GraspTTA 학습·평가, 벤치마크 집계, 데이터셋 통계 | 저장소에 해당 코드 없음 | 표 II–IV의 재현은 불가. 값을 확인할 방법이 없음 | 저장소 전체 검색 (`Q1`, `entropy`, `DDG`, `GraspTTA`) |
| **D13** | 없음 | `generate_grasps.py`의 `--thres_fc/--thres_dis/--thres_pen`은 파싱만 되고 `generate()`에서 쓰이지 않음. `main.py`의 로거만 사용 | 설정이 무시됨 | `generate_grasps.py`, `main.py` |
| **D14** | 3980 CAD 객체, 133 범주 (ShapeNetCore + ShapeNetSem) | `extract_utils.py`의 ShapeNetCore 범주 사전: 12개 중 `car`가 주석 처리되어 11개 활성. ShapeNetSem은 제외 목록(`bad_categories`) 방식 | 133 범주와 3980 개의 수는 코드만으로는 검증 불가 | `asset_process/utils/extract_utils.py` 8–25행, 195행 |
| **D15** | 논문은 같은 결과를 재현한다고 주장 (§IV) | `validate_grasps.py`: `dtype=np.bool8`(NumPy 1.24에서 제거). README는 Python 3.7, PyTorch 1.10, CUDA 11.3 요구. `generate_grasps.py`: `torch.device('cuda')` 하드코딩 | 현대 환경에서는 그대로 돌지 않음. CPU나 MPS 경로가 없음 | `validate_grasps.py` 164행, `generate_grasps.py` `generate()` |

## 4. 구현 가능성 (이 맥: Apple M4 Pro, 24 GiB, MPS, CUDA 없음)

**판정: 공식 파이프라인(1.32M 파지 생성과 Isaac Gym 검증)은 이 맥에서 실행할 수 없습니다. 에너지 식과 최적화는 torch로 옮길 수 있고, 검증은 MuJoCo로 바꾸면 축소판이 가능합니다.**

- **막히는 곳 1, CUDA 하드코딩:** `generate()`가 `torch.device('cuda')`를 쓰고, 검증은 `CUDA_VISIBLE_DEVICES`를 요구합니다(D15).
- **막히는 곳 2, Isaac Gym:** NVIDIA GPU와 Linux가 필요합니다. 검증 판정의 원천입니다. 이 맥에서는 대체할 수 없습니다.
- **막히는 곳 3, 외부 CUDA 의존:** pytorch3d(초기화의 표면 샘플링), TorchSDF(Kaolin 변형, CUDA 커널)를 설치해야 합니다. 설치는 하지 않았습니다.
- **가능한 것 (조건부):** `energy.py`의 E_fc, E_dis, E_pen, E_spen, E_joints를 CPU torch로 옮겨 식을 확인하고, `optimizer.py`의 갱신과 수락을 소형 물체 몇 개에 적용. 검증은 `grasp_generation/mjcf/shadow_hand_wrist_free.xml`(저장소에 있음)과 mujoco 4.0으로 바꿀 수 있음. 판정 기준(100 스텝, 6방향, 관통)은 논문대로 옮기되, 접촉 물리가 달라서 성공률 수치는 비교할 수 없음.
- **노력(추정):** 식과 최적화를 torch로 옮기고 단위 시험을 붙이는 데 1–2주. MuJoCo 검증을 붙이고 D1, D2, D4를 확인하는 데 1–2주 더. 합 2–4주. 공식 1.32M 재생성은 A100급 GPU 약 950시간과 Isaac Gym 환경이 필요하므로 이 맥에서는 시작하지 않습니다.

## 5. 저자에게 물을 질문

1. 표 I의 DDGdata "물체당 >100"과 본문 "6.9K grasps, 565 objects"가 맞는지. 맞다면 물체당 값의 정의는 무엇인지 (D 항목 없음, `03` §1).
2. 검증 판정에서 관통은 생성 시 에너지 합 E_pen인지, 시뮬레이션에서 잰 최대 깊이인지 (D3).
3. 데이터셋에 저장된 `qpos`는 `validate_grasps.py`의 관절 보정 **전** 값입니다. 검증된 자세와 저장된 자세는 어떤 관계인지 (D4).
4. 최적화 갱신은 RMSProp형 전처리와 Metropolis 수락을 쓰는지, 논문 설명처럼 단순 경사 하강인지 (D1).
5. 벤치마크의 시도 수, 반복 횟수, 테스트셋 분할 (`03` §5).
6. 엔트로피의 단위(bit)와 표본 수 (`03` §3).
7. 정규화 필터 `volume > 0.05`가 데이터셋에 어떤 영향을 줬는지 (D8).
8. 논문 라이선스(arXiv CC BY 4.0)와 프로젝트 페이지·저장소의 CC BY-NC 4.0이 어떤 관계인지.

## 6. 이 대조에서 확인하지 못한 것

- 공식 저장소의 실제 실행 결과. 실행하지 않았습니다.
- `allegro`, `mano`, `mano-table`, `table` 분기의 내용. 논문 Fig. 4의 MANO·Allegro 결과가 이 분기에서 나온다는 것만 README로 확인했습니다.
- 데이터셋 파일 자체(1.32M 파지). 샘플 5개는 pickle이라 열지 않았습니다.
