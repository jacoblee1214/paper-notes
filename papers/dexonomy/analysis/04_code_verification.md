# 04. 공식 코드 정적 대조: Dexonomy

대상: `https://github.com/JYChen18/Dexonomy`, 얕은 복제 커밋 `d561ff9`(2026-03-26, "move BODex lfs to hf"). 복제본은 `repo/`에 있고 gitignore했습니다.

**방법.** 정적 읽기만 했습니다. 설치, import, 실행, 파이썬 파일 로드를 하지 않았습니다. 복제본은 신뢰하지 않는 데이터로 다뤘습니다. 읽은 파일은 `dexonomy/config/base.yaml`, `config/op/{init,grasp,eval,grasp_admm,tmpl,total}.yaml`, `config/hand/shadow.yaml`, `op/gen_init.py`, `op/gen_grasp.py`, `op/eval_data.py`, `sim/mujoco_env.py`, `script.py`, `getting_started/2_running_tutorial.md`, `README.md`, `assets/hand/shadow/right.xml`와 `real_right.xml`입니다.

**라벨.** K는 논문과 코드가 일치하는 항목, D는 논문과 코드가 다른 항목, U는 논문에 값이 없어서 코드 값만 있는 항목입니다.

## 1. 라이선스 (분석 결과)

| 출처 | 표시 | 확인 방법 |
|---|---|---|
| 논문 (arXiv abs) | **CC BY 4.0** (`creativecommons.org/licenses/by/4.0/`) | 2026-11-22 abs 페이지 |
| 논문 PDF | ©, copyright, All rights reserved, 회사 고지 없음 | `pdftotext`에서 검색 |
| 공식 저장소 README | "This work **and the dataset** are licensed under **CC BY-NC 4.0**" | 복제본 README 마지막 절 |
| 공식 저장소 파일 | LICENSE 파일 없음. GitHub API 라이선스 필드 null | 복제본 최상위 파일 목록 |
| HF 데이터셋 카드 | `cc-by-nc-4.0` | HF API |

**판단.** 논문 자체는 CC BY 4.0이지만, 공식 저장소 README는 "이 작업(this work)"을 CC BY-NC 4.0으로 묶습니다. 이 "작업"이 논문을 가리키는지 코드를 가리키는지 README만으로는 알 수 없습니다. 같은 종류의 판단을 다른 논문 분석에서도 했습니다. 그래서 **논문 그림은 공개 산출물에 넣지 않았습니다.** 판단 기준은 "의심이 있으면 자체 그림만"입니다.

- 코드: LICENSE 파일이 없으므로 기본적으로 저작권이 유보된 것으로 보입니다. README의 CC BY-NC 4.0이 적용된다고 보더라도 비상업 조건입니다. 복사해 쓰지 않았고, 저장소에 넣지 않았습니다.
- 재구현: 논문 설명만 따라 다시 짜는 것은 코드 복사가 아닙니다. 다만 저장소의 상수 값(K와 U 항목)을 옮기는 것은 코드에서 가져온 것이므로 비상업 조건을 따라야 합니다. 법률 판단은 사용자가 따로 받아야 합니다.
- 데이터셋: 카드가 CC BY-NC 4.0입니다. 받지 않았습니다.

## 2. 일치 항목 (K)

| 라벨 | 논문 | 코드 (파일:설정) | 판정 |
|---|---|---|---|
| K1 | IV-C: 접촉 여유 1 mm | `sim/mujoco_env.py` MuJoCo_OptCfg `obj_margin = hand_margin = 0.001` | 일치 |
| K2 | IV-C: 접촉 판정 2 mm | `config/op/grasp.yaml` `contact.thre: 0.002` | 일치 |
| K3 | IV-C: 200 스텝 | `grasp.yaml` `step: 20`, `substep: 10` | 일치 (조기 종료는 D4) |
| K4 | IV-B 식 (7): 손끝 거리와 법선 차의 합 | `op/gen_init.py` `loss = 1000·loss_dist + loss_normal` (제곱 합) | 구조 일치 (가중치는 U1) |
| K5 | VII-A: 여섯 방향 x 2초 | `eval_fc`: 6방향 x 10 x `step_sim(substep=50)` = 500 스텝, timestep 0.004 s | 일치 (2 s) |
| K6 | VII-A: 5 cm, 15° | `config/op/eval.yaml` `trans_thre: 0.05`, `rot_thre: 15` | 일치 |
| K7 | VII-B-2: 어려운 조건 마찰 0.3, 0.002 | `eval.yaml` `hard: True` → `miu *= 0.5, 0.1` (`mujoco_env.py` `_set_friction`) | 일치 (0.6x0.5 = 0.3, 0.02x0.1 = 0.002) |
| K8 | VII-A: 물체 100 g | `config/base.yaml` `obj_mass: 0.1` | 일치 |
| K9 | IV-B: 최근접점 질의 (Warp) | `op/gen_init.py` `coll_env.query_point`, `util/warp_util.py` | 일치 |
| K10 | IV-C: 주석된 손가락은 접촉해야 함 | `grasp.yaml` `contact.check_body: True` | 일치 |
| K11 | III, IV-B–C: 접촉 품질 임계 (식 (6)) | `init.yaml` `qp.thre: 0.01`, `grasp.yaml` `qp.thre: 0.001` | 구조 일치 (값은 U6) |
| K12 | IV-E: 성공 파지의 자세를 새 템플릿으로, 접촉은 가까운 실제 접촉으로 갱신 | `op/gen_grasp.py` `tmpl_upd_mode: hybrid` 분기 | 일치 (임계값은 U2) |
| K13 | IV-D: 압착 자세는 접촉 힘에서 계산 | `sim/mujoco_env.py` `get_squeeze_qpos` (`mj_applyFT`) | 일치 (세부 구현은 다름) |
| K14 | IV-D: 힘 QP에서 g = 0 | `data/obj_loader.py`: force_closure 과제의 `ext_wrench = 0`이 `ContactQP`에 들어감 (`gen_grasp.py` 71행) | 일치 |
| K15 | VII-A: 외력은 여섯 직교 방향 | `eval_fc`의 `ext_fdir_lst` = ±x, ±y, ±z | 일치 |
| K16 | VII-A, IV-B: 동작 순서 pre-grasp → grasp → squeeze | `eval_data.py` 기본 `skip_pregrasp: False` | 일치 |

## 3. 차이 항목 (D)

| 라벨 | 논문 | 코드 | 영향과 메모 |
|---|---|---|---|
| D1 | 부록 B: 물체당 **10 epoch** x 시도 10 | `config/op/init.yaml` `epoch: 5`. `script.py`가 덮어쓰지 않음 | 공식 데이터 생성 설정이 논문과 다를 수 있음. 재현할 때 `op.epoch`를 확인해야 함 |
| D2 | IV-E·부록 B: 사전 파지 여유 **2 cm** | `grasp.yaml`의 `pregrasp.cdist: 0.045` (선형 증가), `ho_thre: 0.04` | 4.5 cm 목표 여유. 논문 2 cm와 다름 |
| D3 | 식 (8): 단일 이득 **k_f** | `mujoco_env.py` `apply_contact_forces`: 법선 **500**, 접선 **100** (비등방 스프링) | 논문의 단순화와 다름. 법선이 더 강함 |
| D4 | IV-C: **200 스텝** 적용 | `gen_grasp.py`: 접촉 차이 변화 < 1e-4로 수렴하고 충돌 필터가 통과하면 **조기 종료** | 실제 스텝 수는 200보다 적을 수 있음 |
| D5 | VII-A: 성공 판정에 관통 **> 1 cm** 실패 | `eval_data.py`와 `mujoco_env.py`의 평가 경로에 관통 검사 없음. 파지 단계 필터는 손-물체 거리 ≥ 0 (`grasp.yaml` `ho_thre: 0`) | 평가 성공률의 정의가 논문과 다를 수 있음. 관통 판정은 확인 못함 |
| D6 | IV-E: 새 템플릿을 추가 (상한 언급 없음) | `config/base.yaml` `tmpl_max_evo: 2`. `eval_data.py`는 세대 2 이하만 새 템플릿으로 링크 | 논문에 없는 제한. 세대 3 이상의 성공 파지는 템플릿이 되지 않음 |
| D7 | IV-B: "최종 에너지 함수가 임계 아래" | `gen_init.py` `_loss_filter`: **접점별 최댓값** `loss_dist < pos_thre`, `loss_normal < normal_thre` | 형식이 다름 (합이 아니라 최댓값). 아래 메모 |
| D8 | II-B, IV-D: 다른 손 (논문은 Shadow, Allegro) | 저장소는 Shadow, Allegro, Leap, MANO, Unitree G1 설정과 Inspire 설정을 둠 | 확장이며 논문과 모순되지 않음 (참고만) |

**D7 메모 (단위 확인).** `gen_init.py`에서 `loss_dist = ||obj_scale·Δ||²`이고 `loss_normal = ||Δn||²`입니다. 설정의 임계값은 `pos_thre: 0.001`, `normal_thre: 0.3`입니다. 설정 주석은 "0.001 m"과 "0.3 rad"라고 씁니다. 그러나 제곱 값에 적용하면 실제 판정은 거리 약 **3.2 cm** (√0.001 = 0.0316 m), 법선 각 약 **0.56 rad** (2·asin(√0.3/2) ≈ 32°)입니다. 실행하지 않고 식으로만 확인한 값입니다. 설정 주석과 판정식이 다릅니다.

## 4. 논문에 값이 없는 항목 (U)

| 라벨 | 항목 | 코드 값 | 위치 |
|---|---|---|---|
| U1 | 전역 손실 가중치, 옵티마이저 | 1000 : 1 (거리 제곱 : 법선 제곱), Adam, 학습률 1e-2, 100 스텝 | `gen_init.py` |
| U2 | 템플릿 접촉 갱신 임계 | 거리 < 3 cm, 법선 각 < 45° | `gen_grasp.py` (hybrid) |
| U3 | 실물 Shadow 손의 커플링 | `assets/hand/shadow/real_right.xml`에 tendon 관련 줄이 43개 있고 풀리(pulley) 기하와 커플링 사이트가 들어 있음. 시뮬 기본 `right.xml`은 tendon 관련 줄이 6개 | 부록 E는 "해결책을 모른다"고 씀. 저장소에는 실물용 모델이 이미 있음. 동작 확인은 하지 않음 (정적 grep 기준) |
| U4 | 평가의 외력 크기 | 외력 = 방향 x (물체 질량 x 9.8) ≈ **0.98 N** | `mujoco_env.py` `set_active_obj_ext_fdir` |
| U5 | MuJoCo 접촉 옵션 | 원뿔 elliptic, noslip 2, impratio 10, condim 4, 시뮬 timestep 0.004 | `mujoco_env.py` |
| U6 | 필터 임계 전체 | 평면 여유 2 cm, 힘 QP 전체 마찰 [0.2, 0.001], 샘플 10 물체 병렬, 최종 10개 | `init.yaml` |
| U7 | 샘플 규모 | 물체당 점 1024, 평면 내 회전 8, 물체 100개 (`n_cfg`), 배치 10 | `init.yaml` |
| U8 | 지역 정제 부분 확인 | 접촉 점 거리 판정에 `cdist: 0.001` (목표 접촉 거리 1 mm) | `grasp.yaml` |

## 5. 공식 코드 구조와 계산 장치

- 명령 체계: `dexrun op=tmpl` (주석 변환), `dexsyn` (`script.py`가 init → grasp → eval을 순서대로 실행, 실패하면 재시도). `op=grasp_admm`은 GraspADMM 변형입니다 (`02` §4).
- 장치: `op=init`은 `device: cuda` 기본이며 `cpu` 분기가 있습니다 (`gen_init.py` 280행). `op=grasp`는 MuJoCo CPU만 씁니다 (튜토리얼 2절). `op=eval`도 MuJoCo CPU입니다.
- 필요한 자산: `assets/object/DGN_5k` (`DGN_5k_processed.tar.gz` 1.555 GB), 손 메시 (MuJoCo Menagerie submodule). BODex는 궤적 계획에만 씁니다 (선택).
- 다른 설정: `tmpl_upd_mode: hybrid`, `tmpl_max_buf: 1024`, `skip_traj: True` (궤적 생성 생략이 기본).

## 6. 구현 가능성 (이 맥 기준, 근거는 코드와 문서)

이 맥은 Apple M4 Pro, 약 24 GiB 통합 메모리, MPS, CUDA 없음입니다.

| 단계 | 코드 근거 | 이 맥에서 | 판정 |
|---|---|---|---|
| 템플릿 변환 (`op=tmpl`) | 설정 1줄 | CPU | 가능 (설치 승인 후) |
| 전역 정렬 (`op=init`) | `device: cpu` 분기 있음. Warp CPU 실행 여부는 확인 못함 | CPU 느림, 규모 축소 필요 | 조건부 (소규모 가능성) |
| 지역 정제 (`op=grasp`) | MuJoCo CPU만 사용 (튜토리얼) | 가능 | 조건부 (MuJoCo 설치 승인 후) |
| 평가 (`op=eval`) | MuJoCo CPU | 가능 | 조건부 (같음) |
| 전체 데이터 생성 | 8 x RTX 3090, 3일 (논문 V절) | 불가 | 불가 (규모) |
| 유형 조건부 학습 | DexLearn: CUDA, MinkowskiEngine (CUDA 빌드) | 불가 | 불가 |
| 실로봇 | UR10e, Shadow 손, Azure Kinect | 불가 | 불가 |

**실행하지 않았습니다.** 위 판정은 코드와 문서에서 읽은 결과입니다. 설치와 실행 승인 전까지 확인되지 않은 항목은 "조건부"입니다.

## 7. 확인하지 못한 항목 (미확인)

- GraspADMM (`op=grasp_admm`, arXiv:2603.13832): 논문 범위 밖이고 읽지 않았습니다. 기본 설정에서 `square_res: True`는 Shadow 손에 맞춘 것이라고 튜토리얼에 씁니다.
- DexonomyWeb, Dexonomy_gen (제3자): 읽지 않았습니다.
- U3의 커플링 모델이 실물 J1 굽힘을 재현하는지: 정적으로 건만 확인했습니다.
- 데이터셋의 파일 내용과 규모(9.5M 파지): 받지 않아서 확인하지 못했습니다.
- 공식 평가기의 관통 기준 (D5): 평가 경로에 없음을 확인했을 뿐, 다른 파일에 있는지는 확인하지 못했습니다.
