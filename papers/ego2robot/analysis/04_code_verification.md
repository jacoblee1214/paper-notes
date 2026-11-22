# 04. 논문 ↔ 공식 코드 대조와 구현 가능성 (Ego2Robot)

**대상.** https://github.com/www-Ye/ego2robot_blog, 커밋 `244a4ee` (2026-11-02, 유일한 커밋 "Add Ego2Robot synthesis and RoboTwin evaluation code"), 파일 1,492개. `repo/`에 얕게 clone했고 **정적으로만 읽었습니다**(실행, import, 설치 없음). 구조: `ego2robot/`(합성 파이프라인 Python 약 1.8만 줄), `scripts/`(LeRobot 변환 등), `configs/`(데이터셋별 YAML), `evaluation/robotwin/`(RoboTwin 2.0 포크 + 분리형 평가), `index.html`(프로젝트 페이지). 최상위 `LICENSE`가 없습니다(GitHub 라이선스 null). `evaluation/robotwin/LICENSE`는 RoboTwin의 MIT입니다.

## 1. 한눈에 보기

- **공개된 것:** 행동 정렬, 팔 분할, 손 제거, 베이스 탐색 + IK + 렌더링, L1 선별, LeRobot 변환, RoboTwin 분리형 평가(변형 설정, 과제 목록, 집계 스크립트).
- **공개되지 않은 것:** L2 통계 필터, L3 VLM 일관성 필터(D7, D8), 속도 정렬과 카메라 프레임 변환 등 훈련 데이터 로더(D11), 모델·학습 코드, 체크포인트, 합성 데이터 18,561 h(D11).
- **대조 결과:** 단계 대부분의 숫자(필터 임계, 격자, SAM 3 조각, SG 필터, 블렌딩 속도)는 일치하지만, **베이스 탐색의 선택 기준(D1)**, **실로봇 설정(D9)**, **평가 변형의 파라미터 세 곳(D13–D15)** 이 논문 서술과 다릅니다.

## 2. 일치 항목 (논문 ↔ 코드)

| 논문 | 코드 | 판정 |
|---|---|---|
| 가상 손끝 0.7·index + 0.3·middle, TCP, 폭, 자세 (식 1–3), 오른손 s = +1 / 왼손 −1 | `action_processor.py:113–154` 기본 모드 `index_middle`(손목 → 가상 손끝을 forward로, y = z × f, x = y × z) | 일치(D6의 모드 분기 제외) |
| 퇴화: w < 0.01 m, z와 d 평행 시 직전 자세 유지 (A.3) | `_build_universal_gripper_frame`: `width_freeze_threshold = 0.01`, `y_norm < 1e-3` | 일치 |
| 속도 필터 τ = max(5·중앙값, 0.9/fps), 회전 10.0/fps, 2회 (A.3, 식 7) | `_filter_world_space_outliers`: `velocity_mult = 5.0`, `pos_floor = 0.9/fps`, `rot_floor = 10.0/fps`, `max_rounds = 2` | 일치(fps는 30으로 고정, `self.fps = 30.0`) |
| Savitzky–Golay 창 min(21, n), 차수 min(3, 창−1) | `smoothing_processor.py:477–490` | 일치 |
| 가우시안 SLERP σ = 10, 커널 21 | `smoothing_processor.py:244–248`(양팔 경로). 단팔 + `constrained_hand`는 커널 41(`:128`) | 일치(양팔) |
| SAM 3 400프레임 조각, 50 겹침, 비트 OR, 틈 ≤ 3 보간, 면적 < 50% 교체(창 11), 5x5 닫힘 | `segmentation_processor.py:133, 158–191, 259–316` | 일치 |
| ProPainter 이웃 10, 참조 간격 10, 하위 영상 80, 팽창 4, fp16 | `handinpaint_processor.py:82–90` | 일치 |
| 손 마스크 키포인트 80% 밖이면 폐기 (A.1) | `hand_processor.py:483`: 마스크 안 키포인트 비율 < 0.2 | 일치 |
| 점프 필터 4 x 중앙값, 0.003 m/frame (A.1) | `hand_processor.py:564` | 일치 |
| 틈 처리 블렌드: 3.25 mm/frame, 1.08 deg/frame, n = max(5, min(90, ⌈0.6 n_pos + 0.4 n_rot⌉)) (A.1) | `robot_inpaint/processor.py:580–593` | 일치(`max_consec` 상한이 추가) |
| 베이스 후보 격자 좌우 7, 앞뒤 7, 상하 5, 피치 {30,45,60} x 요 {−45..45} x 롤 {−15,0,15} (A.4) | `processor.py:2632–2705` | 숫자 일치(축 이름은 D3) |
| 후보 제외: 카메라 거리 < 0.20 m, 도달 0.9r 초과, 베이스와 0.08 m 미만 | `processor.py:2650–2657` | 일치 |
| 도달 페널티 5.0·\|ρ̄ − 0.65\| | `processor.py:2677–2681` | 일치(역할은 D1) |
| 키프레임 최대 20개, mink IK | `processor.py:2476–2478` | 일치 |
| 팔별 상위 5개 → 25개 조합 검증 | `processor.py:2864`(`_INDEP_TOPK = 5`), `:2889–2915` | 일치(앞단은 D4) |
| L1: IK 오차 0.05 m, 양팔 교차 접촉 > 1, 로봇 마스크 70% | `processor.py:63–65`: `TRACKING_ERROR_THRESHOLD = 0.05`, `BIMANUAL_OVERLAP_THRESHOLD = 1`, `ROBOT_COVERAGE_THRESHOLD = 0.70` | 일치(주석은 75%, D10) |
| 짧은 유효 구간 < ⌊0.3·fps⌋ 침식 | `processor.py:1060–1117`(`min_run = max(1, round(0.3 * fps))`) | 거의 일치(`round` 대 ⌊⌋) |
| 무효 프레임 60% 초과 에피소드 폐기 | `scripts/convert_to_lerobot.py:1203`(`--invalid_threshold` 기본 0.6, `:313`) | 일치(위치는 변환 스크립트) |
| 15개 형태, 도달 반경(Table 3) | `ARM_REACH` 14종(`processor.py:2417–2425`), `robot_pool` 15종(`configs/*.yaml`) | 일치(YAM 제외, D5) |
| 평가 변형 YAML(배경, 조명, 로봇 색, 카메라, 높이, 방해물, ARX, UR5, Franka, 의역) | `evaluation/robotwin/task_config/*.yml`, `eval_suites/axes.yml` | 존재(세 파라미터는 D13–D15) |
| 의역 지시문 505개, 미지 물체 50과제, 과제 50개, 단계 제한 | 의역 505개(`description/task_instruction`의 `paraphrased` 합, base50 기준), `unseen_object50.txt` 50행, `_eval_step_limit.yml`(open microwave 1500, put bottles dustbin 1700 등) | 일치 |
| 형태 간 베이스 간격 0.6 / 0.59 / 0.65 m | `demo_clean_arx.yml`, `demo_clean_ur5.yml`, `demo_clean_franka.yml` | 일치 |
| 축 집계: Visual, Scene, Embodiment, Task 단순 평균 | `axes.yml`의 `aggregate`, `eval_suites/README.md` | 일치 |

## 3. 차이와 빈칸 D1–D16

### 합성 파이프라인

**D1. 베이스 선택 기준이 다릅니다 (식 4, 식 8 대 `processor.py`).** 논문은 IK 가능 비율(지시함수의 평균)을 최대화하고 점수 S = FR − 5.0·|ρ̄ − 0.65|를 쓴다고 씁니다(식 4, 8, A.4). 코드의 최종 선택은 **키프레임 IK의 최대 위치 오차를 최소화**합니다: `max_err`(+ 0.05 x 자세 오차)로 후보를 비교하고 `early_break = 0.15 m`에서 중단하며 오차 < 0.05 m이면 조기 종료합니다(`processor.py:2769–2843`). 65% 도달 페널티는 IK 이전의 기하 점수에서만 쓰입니다(`:2677–2691`). 즉 "IK 가능 비율"이 아닌 "추적 오차"가 기준이고, 두 기준은 같은 후보를 고르지 않을 수 있습니다.

**D2. 논문에 없는 탐색 단계들이 있습니다.** (i) 영상 중심 근처에 로봇 베이스가 투영되면 최대 8.0의 페널티(`:2670`), (ii) 피치·요 이탈 페널티 (p − 45)²/500 + y²/500(`:2691` 부근), (iii) 위치·자세 다양성 top-k 선택(`_diverse_topk`, `:2553`), (iv) 3인칭 시점 검출 시 좌우 후보 교환(EEF 접근축이 카메라를 향하는 비율 > 80%, `:2743–2744`), (v) 오차 0.05–0.5 m이면 **무작위 국소 정제** 60–100회(위치 ±0.11 → ±0.03 m 균등, 각도 ±20° → ±5° 균등, `:2936–2973`), (vi) 양팔 최소 거리·교차 금지(`:2543, 2965`), (vii) 후보가 모두 걸러지면 고정 대체 후보. 정제는 난수를 쓰는데 시드 고정은 확인하지 못했습니다(`np.random.uniform`, `:2952`).

**D3. 격자 축 이름이 뒤바뀌어 있습니다 (A.4).** 논문: 좌우 {0.3…1.2} x r(7), **앞뒤** {−0.1, 0, 0.1, 0.3, 0.5, 0.7, 0.9} x r(7), **상하** {0.4, 0.2, 0, −0.2, −0.4} x r(5). 코드는 카메라 좌표(Y 아래, Z 장면 쪽)로 `tx_opts`(7), `ty_opts = [−0.1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9]`(7, Y-down), `tz_opts = [0.4, 0.2, 0.0, −0.2, −0.4]`(5, Z)입니다(`:2632–2640`). 숫자와 개수는 일치하지만 7개 값 축은 Y(아래), 5개 값 축은 Z(앞)여서 논문의 "앞뒤"와 "상하" 이름이 서로 바뀐 것으로 보입니다. 3인칭 시점용 격자(`ty` 6개, `tz` [0.8…0.0])도 따로 있습니다.

**D4. 후보 수 서술이 다릅니다.** 논문은 "팔별 상위 5개를 독립 선별 후 25개 조합 검증"(A.4). 코드는 기하 점수 상위 **50개**(`N_BIMANUAL = 50`, `:2551`)를 팔별로 IK 샌드박스 평가하고 그중 상위 5개로 조합합니다(`:2864`). 5와 25는 최종 단계의 수입니다.

**D5. YAM의 도달 반경이 코드에 없습니다.** 표 3은 YAM 0.866 m이지만 `ARM_REACH`에 키가 없어 `ARM_REACH.get(self.robot, 1.0)`이 **1.0 m**를 씁니다(`processor.py:2417–2425`). 도달 한계 0.9r, 65% 목표가 YAM에서 논문 값(0.78 m, 0.56 m)이 아니라 0.90 m, 0.65 m로 계산됩니다. 나머지 14종 값은 표 3과 일치합니다.

**D6. 손 → 그리퍼 변환의 경로가 둘이고 설정이 다릅니다.** 기본 `index_middle`은 식 1–3과 같습니다(`action_processor.py:126–127, 153`). 그러나 (i) 실로봇 설정 `configs/arx.yaml`은 `grasp_finger_mode: "index_only"`로, 가상 손끝 = 검지만, 파지 중심 = (엄지 끝 + 검지 끝)/2, 접근 방향을 엄지 관절의 손바닥 법선에서 만듭니다(`:121–151`). (ii) 프레임별 외부 파라미터가 없는 경로(`_process_hand_sequence`, `:583–628`)는 중지 혼합을 쓰지 않고 엄지·검지 끝만 쓰며 z 기준에 `2/3·index_tip + 1/3·index_dip`을 씁니다. 어느 데이터 소스가 어느 경로를 쓰는지(`per_frame_extrinsics` 유무)는 추적하지 않았고, 논문 수치가 어느 경로로 만들어졌는지는 알 수 없습니다. 식 1–3은 기본 모드와 맞습니다.

**D7. L2 통계 필터가 공개 코드에 없습니다 (A.5).** Q1/Q99 ± 3(Q99 − Q1) 필터와 잔차·가속·저크 급변 필터는 `quantile`, `percentile`, `jerk` 검색 결과가 없습니다. 무효 프레임 60% 규칙만 `convert_to_lerobot.py`에 있습니다. 그래서 공개 코드로 합성하면 18,561 h에 적용된 L2를 재현할 수 없습니다.

**D8. L3 VLM 일관성 필터가 공개 코드에 없고, VLM은 하위 과제 분할에만 쓰입니다 (A.5, A.2).** `subtask_processor.py`는 Qwen3.5(DashScope API, 원격 호출은 환경 변수로 명시 허용해야 함)로 영상을 하위 과제로 나누는 일만 합니다. 합성 영상을 4 fps로 샘플해 "일치/불일치 + 신뢰도" JSON을 받는 감사 단계는 없습니다. 하위 과제 분할 설정도 다릅니다: 논문은 60초로 사전 분할(`presplit_video.py` 기본 60초는 일치), 코드의 VLM 청크는 `vlm_chunk_sec = 40`, `vlm_overlap_sec = 10`(`subtask_processor.py:266–267`)입니다.

**D9. 실로봇(ACone) 설정이 논문 서술과 다릅니다.** 논문은 놀이 영상을 Path B(WiLoR, DynHaMR, Qwen3.5)로 처리하고 베이스는 식 4로 탐색한다고 씁니다(§5.4, E.2). 공개된 `configs/arx.yaml`은 (i) `use_calibration_base: true`로 **베이스 탐색을 통째로 건너뛰고** 보정된 고정 베이스를 씁니다(`processor.py:2360–2377`. 코드 주석은 "기본 꺼짐. 실제 ACone에서 찍었을 때만 켜라. 사람 손 위치가 ACone 베이스 범위와 안 맞아 IK가 풀리지 않는다"고 씀), (ii) `dynhamr_enabled: false`(시간 최적화 미사용), (iii) DynHaMR 깊이 범위 `dynhamr_z_min: 0.15`(논문 [0.05, 0.4]의 0.05), 틈 보간 `dynhamr_gap_fill: 3`(논문 ≤ 10프레임 선형 보간), (iv) `grasp_finger_mode: "index_only"`(D6)입니다. 논문의 실로봇 수치가 이 설정으로 만든 것인지 알 수 없습니다.

**D10. L1 세부.** 코드 주석은 "로봇 마스크 > 75%"(`processor.py:1325`)이지만 상수는 0.70입니다. 자기 충돌 검사는 별도 프레임 플래그로는 찾지 못했고, IK 단계의 충돌 회피 제약(ACone `CollisionAvoidanceLimit`)과 양팔 교차 접촉 수만 있습니다. 손 검출에는 논문에 없는 가장자리 필터(키포인트의 70% 미만이 영상 안이면 폐기, `hand_processor.py:488–492`)도 있습니다.

**D11. 훈련 쪽이 통째로 없습니다.** (i) 소스별 프레임 부분 샘플링 60%/45%/25%(§3): 검색 결과 없음. (ii) 카메라 프레임 상대 EEF 변환(식 10, C.2): 없음. LeRobot 변환 README가 "자세는 월드 좌표이며 필요하면 데이터 로더에서 변환하라"고 씁니다. (iii) 소스별 샘플링 가중치(그림 7), 사전학습·미세조정 코드(Qwen3.5-4B + DiT, 흐름 매칭), 체크포인트, 합성 데이터 18,561 h, 실로봇 데이터: 없음. 저장소에서 합성 데이터나 가중치로 가는 링크도 찾지 못했습니다. 따라서 표 1, 2, 5, 그림 3, 4는 공개 자료만으로 재현할 수 없습니다.

**D12. 합성 영상의 해상도가 논문에 없습니다.** 공개 데이터 설정(`egodex`, `vitra`, `egoverse_*`)은 `input_resolution: 426`, `output_resolution: 240`, ACone 설정은 480입니다. 논문은 사전학습에 쓴 합성 프레임의 해상도, 프레임률, 미세조정 해상도(RoboTwin 기본)와의 차이를 쓰지 않습니다.

### 평가 틀

**D13. 책상 높이: 단방향 [−3, 0] cm 대 ±4 cm.** 본문 §4와 부록 B.1은 "±4 cm"입니다. 코드는 `random_table_height: 0.03`(`demo_random_height.yml`)이고 `table_z_bias = uniform(−random_table_height, 0) + bias`(`_base_task.py:125`)로 **내리는 방향만, 최대 3 cm**입니다. 따라서 "Height" 열은 ±4 cm가 아닙니다(RoboTwin의 Randomized도 0.03).

**D14. 카메라 오프셋: 구면 무작위 방향 x U(0, 5 cm) 대 "축별 최대 5 cm".** 논문 B.1은 "head camera displaced up to 5 cm per axis". 코드는 `camera.py:98–100`: 단위 무작위 방향 x `uniform(0, 0.05)`로 **전체 이동 크기가 최대 5 cm**입니다(축별 한계 ±5 cm 상자가 아님). 한 축의 기대 이동은 더 작습니다.

**D15. 방해물 개수: 기본 10 대 3–5.** 논문 B.1은 "3–5 distractor objects". 코드의 `get_cluttered_table(cluttered_numbers=10, ...)`는 인자 없이 호출되어(`_base_task.py:147–148`) 최대 10개를 놓으려 시도합니다(`max_try = 50`). 실제 배치 수는 배치 가능성에 따라 달라집니다.

**D16. 보고 방식: 코드는 시드·체크포인트별 평균과 표준편차를 낸다.** `README`(`evaluation/robotwin/README.md`)는 "세 평가 시드"와 "독립 미세조정 세 모델"의 평균(표준편차)을 보고하는 `aggregate_disentangled.py`를 설명하고(`:74–96`), `eval_suites/README.md`도 "mean and sample standard deviation across checkpoint/seed pairs"라고 씁니다. 논문 표 1, 2, 5는 표준편차 없이 단일 값이고 몇 개 시드·실행인지 쓰지 않습니다(`03` V8, V2). 논문이 단일 실행인지, 여러 실행의 평균인지 알 수 없습니다. 별도로 `use_seed: false`(평가 변형 YAML)이므로 평가 장면은 시드가 고정되지 않은 무작위입니다.

### 사소한 점

- `ActionProcessor`는 `self.fps = 30.0`, `self.dt = 1/15`를 고정합니다(`:92–93`). 데이터 소스의 실제 fps를 쓰지 않습니다(EgoDex, EgoVerse 설정은 `info.fps` 또는 기본 30).
- 로봇 XML은 16종(`trs_so_arm100` 포함)인데 `robot_pool`은 15종입니다. 논문 15와 맞습니다.
- README의 "Poses use world coordinates" 설명 외에 카메라 프레임 변환 설명이 없습니다(D11).

## 4. 구현 가능성 판정

**판정: 이 맥(Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)에서 논문의 수치(표 1, 2, 5, 그림 3, 4)는 재현할 수 없습니다. 합성 파이프라인의 순수 기하·신호처리 부분(행동 정렬, 필터, 베이스 후보의 기하 점수)은 CPU에서 구현·검증할 수 있고, IK와 MuJoCo 렌더는 설치 승인 뒤 조건부로 가능합니다.** (측정한 것이 아니라 코드와 요구 사항을 읽은 판단입니다.)

| 단계 | 필요한 것 | 이 맥에서 | 근거 |
|---|---|---|---|
| 식 1–3 행동 정렬, SG·SLERP 평활, 속도 필터 | numpy, scipy (설치됨) | **가능(CPU)** | `action_processor.py`, `smoothing_processor.py`는 순수 numpy/scipy. 합성 손 키포인트로 검증 가능. 공개 소스 영상 없이 |
| 베이스 후보 기하 점수(식 8의 도달 페널티) | numpy | **가능(CPU)** | 검색 격자 11,025개/팔, 순수 기하(`:2625–2705`) |
| IK 샌드박스(mink)와 MuJoCo 렌더링 | mujoco, mink, quadprog, 로봇 XML 360 MB | **조건부** | 이 세션의 Python(anaconda 3.12, homebrew 3.11·3.14, 시스템 3.9)에 mujoco, torch, mink, opencv, lerobot이 없음(확인 2026-11-22). 설치 전에 목록을 보여 주고 승인을 받아야 함. mink의 quadprog는 컴파일이 필요할 수 있음(미확인) |
| 팔 분할(SAM 3), 손 제거(ProPainter), 깊이(Depth Anything 3), 손 추정(WiLoR, Dyn-HaMR, AnyHand) | CUDA PyTorch 2.7+, CUDA 12.6+ (README), 각 가중치 | **불가(추정)** | README가 CUDA를 요구하고 `pipeline.py`가 `CUDA_VISIBLE_DEVICES`를 설정함. MPS 동작은 미확인. 가중치 크기 미확인(받지 않음) |
| 하위 과제 분할(Qwen3.5) | DashScope API 키, 원격 호출 | **불가(금지)** | API 호출은 승인 없이 하지 않음. 로컬 어노테이션이 있으면 `subtask` 단계를 생략할 수 있다고 README에 있음 |
| RoboTwin 2.0 분리형 평가 | SAPIEN, mplib, CuRobo, 자산 다운로드, 정책 | **불가(추정)** | `_install.sh`가 "deps, sapien/mplib patches, curobo"를 설치. CuRobo는 CUDA. 모델 체크포인트도 없음. 정책 없이는 환경만 가능 |
| 사전학습 + 미세조정(Qwen3.5-4B + DiT) | 8 x A100, 200K 스텝, 4개 혼합 설정 + 4개 미세조정 | **불가** | 학습 코드·체크포인트·데이터가 없고 GPU 시간이 논문에 없음 |
| 합성 데이터 18,561 h | 공개 여부 불명 | **불명** | 링크 없음. 다운로드 안 함. 디스크 여유 12 GiB(98% 사용, 확인 2026-11-22) |
| 실로봇 ARX ACone | 양팔 하드웨어, 카메라 보정 | **불가** | 하드웨어 필요(`configs/calibration/acone.json`은 저자 보정값) |
| 원천 데이터(EgoDex 등) | EgoDex 약 1.7 TB, 그 외 미확인 | **불가** | 디스크 12 GiB 여유. 다운로드 안 함 |

**노력 추정 (측정 없음, 추정).**
- 순수 numpy 부분의 재구현과 합성 궤적 검증(행동 정렬, 필터, 베이스 기하 점수, D1 기준 비교): 1–2일.
- MuJoCo + mink 설치 후 한 형태로 한 궤적을 렌더(손 제거·합성 제외, 로봇 오버레이 없이 IK 추적 오차만): 설치 승인 뒤 0.5–1일(추정).
- 한 소스의 짧은 영상 하나를 끝까지 합성(SAM 3, ProPainter, DA3 포함): CUDA 기기에서 설치와 가중치 준비 1–2일. 이 맥에서는 권장하지 않습니다.
- 논문 표 1의 재현: 공개 자료로는 불가능(D11). 가능하더라도 8 x A100 x 200K 스텝 x 4 설정 + 50과제 미세조정 + 변형 12설정 x 각 2,500 에피소드 평가는 이 분석의 범위(GPU 재현 제외)를 넘습니다.

## 5. 저자에게 물을 질문

1. 표 5의 각 값은 몇 에피소드·몇 시드의 평균인가(홀수 값이 많음)? 표 1, 2의 평균 방식은?
2. 베이스 선택은 IK 가능 비율인가 최대 추적 오차인가(D1)? 논문 수치는 어느 쪽으로 만들었나?
3. L2 통계 필터와 L3 VLM 필터는 18,561 h에 어떻게 적용했고 각 단계의 제거 비율은? 형태별 보존 시간은(D7, D8)?
4. 실로봇 합성은 `use_calibration_base`와 `dynhamr_enabled`를 어떻게 설정했나(D9)? 그리퍼 모드 `index_only` 사용 범위는(D6)?
5. 책상 높이는 ±4 cm인가 [−3, 0] cm인가, 카메라 오프셋과 방해물 수는(D13–D15)?
6. "카메라 프레임 상대 EEF"의 Δ는 EEF 프레임인가 월드 프레임인가(`03` K1)? 카메라 이동이 포함되는가(`03` K3)?
7. 사전학습 없음(VLM 초기화만) 기준선과 합성 데이터 양 곡선(시간 대 성능)이 있는가? 시드별 변동은?
8. 합성 데이터와 가중치의 공개 계획은?
