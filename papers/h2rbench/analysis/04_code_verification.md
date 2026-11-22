# 04. 공식 코드 대조와 구현 가능성

## 0. 대상과 방법

- 저장소: `xiaochy/H2RBench` (공식, 프로젝트 페이지의 Code 링크). 커밋 `155026875c6ddf3bedfedb8ef1fefb6e8c9a2744` (2026-11-03, 단일 커밋 "H2RBench: A Real-to-Sim Benchmark for Evaluating Human-to-Robot Transfer"). 얕은 복제를 `repo/`에 두었습니다(서브모듈 미초기화). 추적 파일 317개, 9.4 MB.
- **정적 읽기만 했습니다.** 코드를 실행, import, 설치하지 않았습니다. 데이터와 가중치는 받지 않았습니다. 사람 데이터셋 크기는 Hugging Face API 메타데이터로만 확인했습니다.
- 읽은 파일: `README.md`, `docs/*.md`, `pyproject.toml`, `.gitmodules`, `scripts/{eval_policy,eval_summary,generate_dataset}.py`, `src/polaris/config.py`, `environments/{__init__,droid_cfg,manager_based_rl_splat_environment}.py`, `environments/rubrics/{base,checkers}.py`, `utils_/eval_utils.py`, `client/{diffusion_policy,amplify,point_policy,ghost_lerobot}_client.py`, `server/{diffusion_policy,ghost}_server.py`(일부)와 PolaRiS-Hub의 과제 설정 파일 세 개. 정책 학습 코드는 이 저장소에 없습니다(D8, D16).
- 논문 쪽 기준은 v3 PDF(부록 A.3, A.4, Table 5, Table 6, B.3)입니다.

## 1. 설치 요구사항과 라이선스

- **의존성(`pyproject.toml`):** `isaaclab[all,isaacsim]==2.3.0`, `nvidia-curobo`(서브모듈), `diff-gaussian-rasterization`, `diff-surfel-rasterization`, `gsplat`, `pytorch3d`, `robosuite`(고정 커밋), `openpi-client`, `torch>=2.9.0`/`torchvision`(override, 기본 CUDA 13), 서브모듈 `third_party/openpi`, `src/curobo`, `src/polaris/policy/{lerobot,Point-Policy}`. Python 3.11. README는 CUDA 13 전제이고 "`sudo apt install ffmpeg`"를 안내합니다. Isaac Sim은 NVIDIA RTX GPU가 필요해서 이 맥(M4 Pro, CUDA 없음)에서는 설치 자체가 안 됩니다.
- **방법별 환경이 따로입니다.** README는 Diffusion Policy(`robodiff` conda), AMPLIFY(`amplify` conda), GHOST(`lerobot` pixi), Point Policy(`point-policy` conda)를 각각 다른 환경으로 안내하고, 평가는 ZMQ/WebSocket 서버–클라이언트로 연결합니다.
- **큰 파일은 저장소 밖입니다.** `content.zip`(cuRobo 자산), `put_red_cup_no_curtain.zip`, `nvidia_droid.zip`은 Google Drive에서 받으라고 안내합니다. **`PolaRiS-Hub/`에는 머그 과제 하나의 설정 파일만 있고 3D 자산이 없으며, 다른 세 과제(쌓기, 끼우기, 장난감 옮기기)의 폴더는 저장소에 없습니다**(D13).
- **라이선스.** `LICENSE`는 MIT, 저작권자는 "Arhan Jain, 2025"(PolaRiS 저자)입니다. H2RBench 저자의 저작권 표기는 없습니다. 이 저장소는 PolaRiS의 포크입니다(`pyproject.toml`의 이름 `polaris`, 작성자 Arhan Jain, README의 클론 명령이 가리키는 저장소 이름도 `polaris`). 사람 데이터셋 네 개에는 라이선스 필드가 없습니다.
- 크기: 사람 데이터 약 62.9 GB(에피소드 180 + 315 + 300 + 300, 받지 않음).

## 2. 논문 ↔ 코드 대조 (D 항목)

판정: **일치**, **차이**(코드 값이 논문과 다르거나 논문에 없음), **미확인**(저장소에서 찾지 못함).

| 번호 | 항목 | 논문 | 코드 | 판정 | 영향 |
|---|---|---|---|---|---|
| D1 | 진행도와 성공의 의미 | §3.4 "cumulative stages", "final-stage success rate", 진행도는 단계 평균 | `rubrics/base.py` L58–72: 각 기준이 **한 번이라도** 참이면 그 기준은 영구히 도달로 기록(`criteria_reached`), 진행도 = 도달한 기준 수 / 전체, 성공 = 전부 도달. 에피소드는 성공해도 끝나지 않음(`TerminationsCfg`는 time_out만, `droid_cfg.py` L517–520) | **차이 (논문에 없음)** | "최종 성공"은 어느 순간이든 모든 단계를 한 번씩 거치면 성립합니다. 끝 상태가 유지되는지는 보지 않음. 물체가 나중에 떨어져도 성공으로 남음 |
| D2 | 에피소드 길이와 주기 | 논문에 없음 | `EvalArgs.max_episode_length = 300`(`config.py` L69), `sim.dt = 1/120`, `decimation = 8`(`droid_cfg.py` L547–548) → 15 Hz, 에피소드 20 s. 환경 기본 30 s는 `eval_policy.py`가 덮어씀 | **차이 (논문에 없음)** | 평가 조건의 핵심 값이 논문에 없음. 정책의 열린 루프 구간 8 스텝이면 에피소드당 최대 약 37번 질의 |
| D3 | 평가 시드와 초기 조건 | "90 randomized rollouts across 3 seeds" (§5.1) | 기본 `seed = 42`(`config.py` L71), `set_seed`가 `np.random`을 고정하고 에피소드마다 순서대로 초기 위치를 뽑음(`eval_policy.py`, `eval_utils.py`). README 예시는 `--rollouts 30`. 학습 시드와 평가 시드가 같은지는 코드로 알 수 없음 | **미확인** | 평가 시드를 안 바꿨다면 모든 방법, 모든 학습 시드가 **같은 30개 초기 조건**을 씀. 90회 중 서로 다른 초기 조건은 30개. 방법 비교는 짝지은 설계라 장점이지만 "90 독립 롤아웃"이 아님 |
| D4 | 단계 임계값 | 논문 Table 6은 문장으로만 (예 "Reach for the red cup") | `environments/__init__.py`: 닿기(`reach`)는 **EE와 물체 원점 사이 0.2 m**(머그 L28, 그릇 L121, 도넛 L65), 들기는 초기 높이에서 +0.04 m(머그, 그릇) 또는 +0.15 m(도넛 L66), 놓기는 AABB 겹침 비율 ≥ 0.8(머그 L30), 0.5(그릇 L124), 도넛 말뚝 겹침 ≥ 0.1과 그리퍼 열림 검사 생략(`open_finger_threshold=2`, L67), 장난감 1.0(L103–105). 놓기 판정은 **그리퍼가 열려야** 참(`checkers.py` L46) | **차이 (논문에 없음)** | 닿기는 느슨합니다(20 cm). 놓기와 끼우기 판정은 3D 접촉이 아니라 XY 상자 겹침과 원점 거리입니다. 쌓기는 높이(z)를 보지 않음 |
| D5 | 단계 의존성 | Table 6 "step-by-step rubric", §3.4 "cumulative" | `Rubric(criteria=[...])`의 두 번째 원소가 의존 인덱스. 쌓기의 3단계 `reach(blue_bowl, 0.2)`(L123)는 **의존성이 없음**. 장난감 세 기준(L103–105)도 **의존성이 없고** 순서가 없음(어떤 순서든 상자에 넣은 개수) | **차이** | Table 6의 순차 표기("blue → orange → yellow")와 달리 순서를 강제하지 않음. 쌓기의 3단계는 초록 그릇을 든 채 파란 그릇 근처(20 cm)를 지나면 자명하게 참 |
| D6 | 끼우기 판정 구현 | "Insert the donut onto the peg" | `is_inserted`(`checkers.py` L222): 링과 말뚝 원점의 XY 거리 < 0.06 m, 링 z가 말뚝 z보다 낮고 |Δz| < 0.1 m, 링 속도 < 0.005 m/s가 연속 5스텝. 상태(`stable_count`)가 **클로저에 있어 에피소드 사이에 초기화되지 않음**(L232). 환경 리셋은 `Rubric.reset()`만 호출 | **차이 (논문에 없음)** | 임계값은 원점 기준이라 물체 모양에 따라 느슨할 수 있음. 이전 에피소드의 정지 카운트가 남으면 다음 에피소드에서 5스텝 정지 조건이 즉시 충족될 수 있음(저확률, 영향 작음) |
| D7 | 분포 밖 배치 실험 | B.3: **쌓기**, 초록 그릇이 로봇 학습 때 오른쪽에서만 초기화, 평가는 왼쪽 | 배치 변형 설정 `task_config_y_left.yaml`(y ∈ [−0.335, −0.188]), `task_config_y_right.yaml`(y ∈ [0.132, 0.216])은 **머그 집기-놓기** 폴더에만 있음. README 예시 실행 폴더가 `pick_mug/object_placement/mug_left` | **차이 / 미확인** | 저장소로는 쌓기 분포 밖 실험을 재현할 수 없고 머그용 설정만 있음. 논문 B.3이 쌓기라고 쓴 것과 어긋남 |
| D8 | Phantom 구현 | 정면 카메라만, 사람 팔 인페인팅 + 로봇 팔 렌더 후 모방 학습 (A.3.1, C.1.1) | 저장소에 Phantom 파이프라인(손 검출, 인페인팅, 리타게팅, 렌더)이 **없음**. "Phantom"은 `DiffusionPolicy` 클라이언트와 서버(`DiffusionUnetHybridImagePolicy`, 이미지 240 × 426 + EE 상태 10차원, 관측 2스텝, 청크 8)로만 존재 | **미확인** | Table 3의 Phantom 열을 만든 학습·데이터 변환 코드는 저장소 밖. 원 저장소(MarionLepert/phantom)에서 H2RBench 형태의 설정으로 다시 짜야 함 |
| D9 | 카메라와 입력 | Table 5: Phantom 정면; Point Policy, AMPLIFY 정면 + 좌측; GHOST 정면 + 좌측 + 손목 | README 카메라 표와 클라이언트: AMPLIFY `cam0`(정면) + `cam1`(좌측), GHOST `cam0`, `cam1`, `wrist_cam`(+ 깊이), Point Policy `cam0`, `cam1`(+ 깊이). Diffusion Policy(Phantom)는 `cam1`이고 README와 클라이언트 주석이 "front camera"라고 쓰지만 AMPLIFY 클라이언트는 같은 `cam1`을 "left"라고 씀 | **일치 (Point Policy, AMPLIFY, GHOST), 모호 (Phantom)** | Phantom이 쓰는 시점이 정면인지 좌측인지 코드 주석이 서로 다름. 또 GHOST와 Point Policy만 깊이를 받음 |
| D10 | 사람 데모 수 | Table 2 최대 사람 데모: 집기-놓기 100, 쌓기 300, 끼우기 300, 순차 300 | README의 데이터셋: 머그 180 에피소드, 장난감 315, 도넛 300, 그릇 300 (Hugging Face 파일 수와 일치). GHOST 예시 체크포인트 이름 `pick_toys_r400h0`(로봇 400, 사람 0으로 읽힘), Point Policy 예시 `pick_toys_simrobot_100` | **차이** | 집기-놓기는 공개분(180)이 논문 최대(100)보다 많고, 장난감은 315 대 300. 부분 집합 선택 규칙이 문서에 없음. 순차 조작 로봇 데모 100(논문) 대 400(체크포인트 이름)은 예시 이름일 수 있음 |
| D11 | 행동 변환 | §3.1 "unified observation-action interface", 이진 그리퍼 | 클라이언트마다 EE 행동 → 관절 위치 변환(cuRobo IK)이 다름. `DiffusionPolicy`와 `AMPLIFY`: IK 실패 시 **현재 관절 유지**(`diffusion_policy_client.py` L134–140, `amplify_client.py` L179–182), 그리퍼 **0.5로 이진화**(L142, L185). `ghost`와 `point_policy`: IK 성공 여부를 보지 않고 해를 그대로 쓰며(`ghost_lerobot_client.py` L308–310, `point_policy_client.py` L38–40), 그리퍼 값을 이진화 없이 덮어씀 | **차이** | 방법 사이에 제어 계층 차이가 생김. 방법 간 비교의 공정성(§3.1, A.3)에 영향. IK 실패율은 기록되지 않음 |
| D12 | 열린 루프 길이와 하이퍼파라미터 | "recommended hyperparameters from their original publications" (§5.1, A.3) | README의 모든 평가 예시가 `--policy.open_loop_horizon 8`. AMPLIFY 클라이언트 기본값은 1(`amplify_client.py` L85). Diffusion Policy 설정은 horizon 16, n_obs 2, n_action 8, 추론 100스텝(`pick_red_mug_diffusion_policy.yaml`). Point Policy, GHOST, AMPLIFY의 학습 하이퍼파라미터는 외부 저장소(포크와 브랜치)에 있고 AMPLIFY는 서브모듈도 아님(`.gitmodules`는 lerobot과 Point-Policy만) | **미확인** | 모든 방법이 같은 열린 루프 8 스텝이라면 원 논문 설정과 다를 수 있음. 학습 설정은 저장소만으로는 확인 불가 |
| D13 | 과제 환경 | 네 과제 (Table 2) | `environments/__init__.py`에 환경 **6개** 등록: `PutRedCup-no-curtain`, `StackBlock`, `InsertDonut`, `HangMug`, `PickPlaceToys`, `StackBowls`. 논문에 없는 `StackBlock`, `HangMug` 포함. 저장소에 들어 있는 장면 폴더는 `put_red_cup_no_curtain`(설정 파일만, 자산 없음) 하나 | **차이 / 미확인** | 쌓기, 끼우기, 장난감 장면의 3D 자산과 초기 조건 파일은 확인 불가. 추가 환경은 문서에 설명이 없음 |
| D14 | 문서와 주석의 어긋남 | | GHOST 클라이언트 독스트링은 "oracle goal projection"이라고 쓰지만 코드는 서버가 `policy.high_level.predict_and_project`로 목표를 예측하는 경로를 씀(`ghost_server.py` L258–286). README의 GHOST와 Point Policy 예시는 장난감 체크포인트를 쓰면서 `--environment DROID-PutRedCup-no-curtain`을 넘김 | **차이 (문서)** | 독스트링이 오라클 목표라고 읽히지만 실제 평가 경로는 예측된 하위 목표. README 예시는 그대로 실행하면 환경이 맞지 않음 |
| D15 | 평가 요약 도구 | | `scripts/eval_summary.py`는 진행도의 최소 비영 값에서 단계 수를 **추정**(`infer_num_stages`: 1/최솟값 반올림)하고 단계별 도달 비율과 성공률을 출력 | **일치 (논문 표 구조와 호환)** | 단계 수를 최소 비영 진행도에서 추정하므로 첫 단계만 달성한 에피소드가 있어야 정확함. 논문 Table 3의 Prog.와 SR 계산은 이 도구와 같은 구조 |
| D16 | 결과와 학습 코드 | Table 3, 4, 7–14, Fig. 3–5, 15, 16 | 평가 결과 CSV, 체크포인트, 네 방법의 학습 스크립트, 사람 영상 전처리(손 자세, 하위 목표 추출)가 저장소에 **없음**. 실세계 평가 코드와 결과도 없음 | **미공개** | 저장소만으로 논문 표를 재현할 수 없음. 벤치마크 환경과 평가 하네스만 있음 |

요약: 평가 하네스(환경 등록, 진행도 규칙, 클라이언트–서버 구조, 데이터 생성 스크립트)는 논문의 벤치마크 설명과 대체로 맞습니다. 차이는 **논문에 없는 평가 규칙**(D1, D2, D4–D6)과 **방법 간 제어 계층 차이**(D9, D11, D12), **실험 설정이 코드로 확인되지 않는 부분**(D3, D7, D8, D10, D13)에 몰려 있습니다. 재현의 가장 큰 장벽은 학습 코드, 체크포인트, 평가 결과, 세 과제의 자산과 실세계 데이터가 저장소에 없다는 점(D13, D16)입니다.

### 2.1 단계 규칙 정리 (코드 기준, 논문 Table 6 대조)

| 과제 (환경 ID) | 단계 | 논문 Table 6과 |
|---|---|---|
| 집기-놓기 (`PutRedCup-no-curtain`) | 닿기 < 0.2 m → (닿기 후) 들기 +0.04 m → (들기 후) 접시 위 겹침 ≥ 0.8, 그리퍼 열림 | 3단계, 일치 |
| 쌓기 (`StackBowls`) | 닿기(초록) → 들기(초록) +0.04 m → 닿기(파랑, 의존 없음) → (들기 후) 파랑이 초록과 겹침 ≥ 0.5, 그리퍼 열림 | 4단계 일치. 3단계 의존성과 그릇 역할이 모호 |
| 끼우기 (`InsertDonut`) | 닿기 → 들기 +0.15 m → 말뚝 겹침 ≥ 0.1 → 끼움(XY < 0.06, z < 0.1, 정지 5스텝) | 4단계, 일치 |
| 순차 (`PickPlaceToys`) | 장난감 3개가 각각 상자에 100% 겹침 + 그리퍼 열림 (순서 없음, 의존 없음) | 3단계, 순서 표기와 다름 |

### 2.2 "진행도 바닥" 추정 (정적 추정, 측정 아님)

닿기 기준이 20 cm이고 시작 시 EE 위치가 물체와 가까우면 첫 단계가 거의 항상 참일 수 있습니다. Table 3에서 끼우기 Prog.가 35–39인데 SR이 2–8%인 세 방법(Phantom, AMPLIFY, Point Policy)과 일치하는 해석이지만, 시작 위치를 코드로 확인하지 못했으므로 **추정**입니다. 순차 조작은 닿기 단계가 없어 이 바닥이 없습니다(AMPLIFY 8.1, SR 0.0). 그러면 과제 사이 Prog. 비교는 같은 척도가 아닙니다.

## 3. 구현 가능성과 노력

**판정: 이 맥(Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)에서 벤치마크의 학습과 평가는 재현할 수 없습니다. 논문 수치의 재계산, 평가 규칙의 정적 분석, 평가 설계 시뮬레이션은 할 수 있고 완료했습니다.**

| 구분 | 내용 | 맥에서 |
|---|---|---|
| **불가 (GPU와 로봇)** | Isaac Sim과 Isaac Lab 2.3 설치(NVIDIA RTX 필요), 가우시안 스플랫 렌더(CUDA 확장 `diff-gaussian-rasterization`, `gsplat`), cuRobo, 네 방법 학습(방법별 conda, pixi 환경), 실세계 Franka + GELLO 평가와 실세계 데이터 | 불가 |
| **불가 (자원 미공개)** | 쌓기, 끼우기, 장난감 장면의 자산과 초기 조건(저장소 밖 Google Drive), 학습 하이퍼파라미터 일부, 평가 결과 CSV, Phantom 파이프라인 | 불가 |
| **불가 (용량과 승인)** | 사람 데이터 약 62.9 GB 받기. 이번 분석에서는 데이터를 내려받지 않았음 | 받지 않음 |
| **가능 (완료)** | 표와 그림의 수치 재계산, 일관성 검사, 통계 검정, Sim-vs-Real 지표의 잡음 시뮬레이션(`analysis/recompute.py`, 수 초) | 완료 |
| **가능 (미실시)** | 진행도 규칙(`Rubric`)을 순수 파이썬으로 다시 구현해 궤적 데이터에 적용하는 도구. 규칙 구현 자체는 이 맥에서 돌지만 입력 궤적이 없음 | 가능, 시작 전 |
| **가능 (미실시)** | MuJoCo 기반 2–3 과제 축소판(스크립트 로봇 데모, 사람 영상 대신 합성 경로)으로 "로봇 데모 고정, 사람 데모 수만 변경"하는 평가 설계 축소 실험 | 가능, 노력 큼 |

노력 추정(측정 전, 추정):
- 리눅스 + RTX 머신에서 벤치마크 환경 설치와 한 방법(Diffusion Policy) 평가까지 **2–5일**(Isaac Lab 2.3과 CUDA 13, 서브모듈 6개 중 일부는 SSH URL, Google Drive 자산, 방법별 환경). 네 방법 전부 학습과 평가는 **2–4주**(하이퍼파라미터가 외부 저장소와 체크포인트 이름에 흩어져 있고 GPU 시간은 논문에 없음).
- 이 맥에서 MuJoCo 축소판(과제 둘, 합성 사람 데이터)과 진행도 규칙 도구는 **2–3주**. 논문 결과의 재현이 아니라 평가 설계 검증용입니다.
- 실세계 검증(Sim-vs-Real)은 Franka와 GELLO, 과제별 실세계 데모와 평가가 필요해 이 분석 범위 밖입니다.

## 4. 정적 읽기의 한계

- `repo/`는 단일 커밋이고 학습, 결과 파일이 없으므로 D 항목의 "차이"는 **평가 하네스에 한정**됩니다. 논문 Table 3의 수치가 이 코드로 나왔는지는 확인할 수 없습니다.
- 임계값과 시작 위치는 코드를 읽어 얻은 것이고 실제 시뮬에서 검증하지 않았습니다(D4, 2.2).
- 외부 저장소(GHOST, Point-Policy 포크, lerobot 포크, AMPLIFY, Phantom)의 코드는 존재와 라이선스만 확인했고 읽지 않았습니다.
