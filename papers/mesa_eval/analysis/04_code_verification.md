# 04. 공식 코드 대조와 구현 가능성

공식 저장소 `pairlab/mesa-env`를 **정적으로만** 읽었습니다(실행, import, 설치 안 함. 데이터 파일은 `json`과 `ast`로 파싱만 함). 기준 커밋 `e4a82e2a88547afdab4c97130ca47efdb523c50f`(2026-11-05), 얕은 복제는 `repo/`(gitignore)에 있습니다. 구조 통계는 `analysis/repo_static_stats.py`(표준 라이브러리, `repo/`가 있어야 실행), 결과는 `analysis/repo_static_stats_results.json`입니다.

## 1. 저장소 개요

| 항목 | 내용 |
|---|---|
| 라이선스 | MIT(LICENSE 파일, `pyproject.toml`도 MIT). 데이터는 Hugging Face에서 CC BY 4.0, 시뮬레이션 자산은 "other" |
| 크기 | 188 MB 복제(과제 BDDL JSON 14,250개가 161 MB), `mesa/` 패키지 파이썬 25.4k줄 |
| 구성 | `mesa/sim`(BDDL 환경, 과제 생성기), `mesa/mimicgen`(생성기, 궤적 최적화), `mesa/task_suites`(평가 세트와 과제 정의), `mesa/serving`(websocket 정책 서버), `scripts/`(생성, 평가, 수집), `docs/`(Sphinx) |
| 환경 | Python 3.10 고정, `mujoco==3.4.0`, `robosuite==1.5.1`, `uv.lock`, 문서는 "Ubuntu 24.04에서 검증" |
| 포함하지 않는 것 | 학습 코드(openpi, Isaac-GR00T 포크는 별도 저장소), 기준선 정책 가중치(Hugging Face), 평가 초기 상태(Hugging Face `mesa-init-states` 0.17 GB), 자산 13.6 GB, 언어 따르기율 계산 코드(D5) |

## 2. 논문 ↔ 코드 차이 D1–D14

심각도: 상은 결과 해석에 영향, 중은 재현 시 주의, 하는 서술 차이.

| # | 논문 | 코드 | 차이와 영향 | 심각도 |
|---|---|---|---|---|
| D1 | 500스텝 한도(부록 B.7 "baselines' 500-step limit") | `mesa/utils/eval_helpers.py:22-23` `max_steps=500`, `per_subtask_extra_steps=300`; `scripts/eval_server.py:183-185`는 새 하위 과제가 달성될 때마다 `max_steps += 300` | 논문 본문, 부록 F에 없는 규칙. 평가 과제의 하위 과제 수가 단일 팔 70과제 중 2개 41, 3개 29, 양팔 57과제 중 2개 27, 3개 10, 4개 20이라 한도가 500에서 최대 1,400스텝까지 늘 수 있음. 문서(`docs/using_mesa/running_inference.md`)에는 설명됨 | 중 |
| D2 | 성공 판정 시점은 정의되지 않음(E.5) | `eval_server.py:187-189` 한 번이라도 `info["success"]`가 참이면 성공으로 래치, 이후 20스텝은 마지막 행동 반복; `bddl_base_domain.py:1105-1108` | 성공은 "한 번이라도 목표 술어 충족"이고 최종 상태 성공이 아님. 술어에 정지 조건이 있어(`base_object_states.py` `check_ontop`: 접촉, 높이, 속도) 순간 통과는 막지만, 이후 무너져도 성공으로 남음 | 하 |
| D3 | "분산 객체 쌍으로 가능한 분산 과제를 열거"(§3.3) | `mesa/sim/task_gen/task.py:676` `_build_distractor_goal_specs`: 모든 집을 수 있는 객체 × (모든 `in` 영역, 모든 `on` 객체)에서 정답 목표만 뺌 | **대상 객체를 다른 곳으로 옮기는 목표**(BDDL당 평균 3.3개), 자기 자신 위에 놓기(`on(bowl, bowl)`) 같은 항, 호환되지 않는 조합까지 포함. BDDL당 분산 목표 평균 22.9개(mesa-70), 14.2개(bimesa-id). "분산 객체 쌍"이라는 서술과 다름 | 중 |
| D4 | D는 "실패한 롤아웃이 분산 과제를 끝낸 것"(§3.3) | `eval_server.py:175-180, 209`, `bddl_base_domain.py:1107`: 에피소드 내 어느 스텝이든 분산 목표가 한 번이라도 충족되면 `distractor_completed`; 성공 에피소드에도 기록. 요약 `eval_helpers.py:321-361`은 전체 에피소드 대비 비율 | 성공과 분산 완료가 한 롤아웃에서 겹칠 수 있어 S + D가 중복 계수될 수 있음. 논문 L이 겹침을 제외하는지 코드로 알 수 없음. 순간 충족도 포함 | 중 |
| D5 | L = S/(S + D), 표 3과 표 18, 19의 Wilson 구간 | 저장소 어디에도 L 계산 코드 없음(`language_follow`, `lfr` 검색 결과 0). `final_summary.json`은 `overall_success_rate`와 `overall_distractor_completion_rate`만 | 논문 표 3과 구간은 에피소드 로그에서 외부에서 계산. 롤아웃 로그와 계산 스크립트가 공개되지 않아 표 3을 코드로 재현할 수 없음(`03` V4로 정수 일관성만 확인) | 중 |
| D6 | 지시문 무작위화 절제(부록 B.2): "for each evaluation we sample a random instruction from the benchmark" | `eval_server.py:261-262`, `eval_set.py:39-45`: 과제당 한 번 `get_random_task_instruction()`(140개 단일 팔 과제 전체에서 균등), 그 과제의 모든 롤아웃이 같은 틀린 지시문 | 과제 단위 한 번 뽑기라 과제당 50롤아웃이 한 지시문을 공유(분산이 큼). 우연히 맞을 확률 1/140(지시문 중복 무시) | 하 |
| D7 | MESA-70 = 70과제 7,000시연, MESA-Aux 736과제(§4.1, E.4) | `task_sets.py:389-487` `TRAIN_SETS["mesa-70"]` 71과제, `demos_per_task = [100]*70 + [15]` = 7,015시연. 추가 1과제 `plate_bowl_drainer_right_region`(15시연, 평가 세트 밖). `mesa-aux` 735과제 | 합 806은 맞으나 70 + 736이 아니라 71 + 735. "7,000시연"은 7,015. `plate_bowl_drainer_left_region`(ID 과제)이 있는 상태에서 오른쪽 영역 과제가 추가됨(코드 주석: 시연 생성이 부족해 보충) | 하 |
| D8 | 보조 데이터는 보류 평가 설정과 겹치지 않음(§5.4) | `docs/mesa_bench/available_datasets.md:11`: 구 LeRobot v2.1은 **보류 Composite 과제 2개**(`cup_bowl_drainer_left_region`, `open_and_jam_slide_cabinet_contain_region`)를 포함하고 일부 에피소드 영상이 없다고 경고. `mesa-all` 목록(852과제)도 평가 Composite 10, Category 2과제를 포함 | 논문의 보조 공동학습(표 4, 20)이 어느 판본을 썼는지 논문에 없음. v2.1이면 Composite 2과제가 학습에 들어간 것이라 Composite의 "보조 효과 없음" 해석에 영향. 현재 v3.0에서는 제거됨(`mesa-aux`와 평가 세트의 교집합은 0, 정적 확인) | 중 |
| D9 | 궤적 시작점 최적화 식 1–3: 수직축 회전 하나로 전체 궤적을 변환 후 앞부분 절단(D.4) | `mimicgen/utils/trajectory_optimization.py:255-335`가 식과 일치(후보 i는 앞 절반, 닫힌 형식 각도). 그러나 (a) 회전을 궤적 뒤로 갈수록 풀어 주는 **페이드아웃 스케줄**(`get_schedule` 217; none/linear/smoothstep/shaped/knee)이 있고 술어별로 `configs/utils.py:100-152`에서 설정(기본 none, 슬라이드 캐비닛/전자레인지 `in`은 shaped β = 0.5 비율 0.5, `close`는 β = 0.3 비율 0.3), (b) `open` 술어는 시작점 최적화를 끔(147행), (c) 회전 최적화에 손목 180도 뒤집기 선택(`trajectory_optimization.py:363-377`), (d) 위치 최적화를 원천 선택의 순위 계산 안에서도 적용(`stitching_data_generator.py:441-453`) | 논문의 서술은 단순화. 생성 품질에 영향을 주는 설정이 논문에 없어 부록 B.3(그림 8)의 절제를 그대로 재현하려면 코드 설정을 따라야 함 | 중 |
| D10 | 시작점 최적화를 MimicGen에 추가(§3.2) | 비-이어붙이기 생성기 `mimicgen/data_generator.py:359`는 `PoseUtils.optimize_trajectory_for_pos(...)`를 호출하고 반환값을 `x, _ =`로 풀지만, 이 함수는 `pose_utils.py`에 없고(`trajectory_optimization.py:255`에만 있음) 단일 배열을 반환 | `generation.stitching = False`(기본값, `configs/config.py:31`)이고 `optimize_start = True`(기본)이면 `DataGenerator` 경로는 AttributeError가 날 것으로 보임(실행 안 함, 정적 판단). 공개 코드에서는 이어붙이기 경로(`stitching_data_generator.py`)만 동작 가능한 것으로 읽힘 | 중 |
| D11 | 원천 시연 20 Hz, Quest 이득 1.5(부록 D.2) | `scripts/data/collect_data.py:76` `gain = 1.5`는 일치. 같은 파일 94행 `control_freq = 15`(환경 기본값은 `config/envs/default.json:5` 20), `quest_agent.py:25` 기본 이득 1.0 | 기본값만 다르고 논문 설정은 명령행 인자로 가능. 수집 주파수 기본값이 15 Hz라 논문 설정을 그대로 따르려면 인자를 줘야 함 | 하 |
| D12 | GPT-5.1 용기 호환 라벨, LLM 요 대칭 라벨(부록 E.1, D.4) | 저장소에 LLM 호출 코드 없음. 요 대칭은 `object_info.py:2378` `AXISYMMETRIC_CATEGORIES` 정적 집합(사과, 토마토, 와인, 캔 등 손으로 정리된 목록과 주석), 용기 호환은 `valid_distractor_map`(`task.py:760-770`)과 객체 정보 | LLM 라벨링은 결과 데이터만 있고 절차가 재현 불가. 라벨 검증도 보고되지 않음 | 하 |
| D13 | 평가: 과제당 50롤아웃(E.5), 행동은 상대 관절각(F.1) | `eval_helpers.py:21` 기본 `num_rollouts_per_task = 10`, `:40` 기본 `controller_type = "osc_pose"`. 논문 설정은 `docs/mesa_bench/evaluating_trained_policies.md`에서 `--num-rollouts-per-task 50 --controller-type joint_pos`로 안내 | 기본값만 다름. 인자를 빼면 평가 크기와 컨트롤러가 논문과 달라짐 | 하 |
| D14 | MimicGen 설정 세부(D.3) | `configs/utils.py`에서 `action_noise = 0.01`, `subtask_term_offset_range = (10, 20)`(마지막 하위 과제 제외), `nn_k = 3`, 선택 전략은 술어별(기본 객체 최근접, `in`/`on`/`stack` 로봇 이동 거리, 슬라이드 캐비닛 `in`과 `close`는 EEF-객체 상대 이동) | 논문은 k = 3과 선택 전략 종류를 적었으나 **행동 잡음 0.01과 술어별 전략**은 쓰지 않음. 재현 시 필요한 정보 | 하 |

### 일치 확인 (차이 없음)

- 평가 세트 크기: `EVAL_SETS` = mesa-70 70, spatial 10, instance 20, composite 20, category 20, bimesa-id 57, spatial 38, instance 18, composite 15, object(= Category) 17. 모든 과제에 평가 BDDL이 정확히 50개씩(`repo_static_stats`)이고 `instance_idx = episode_idx % 50`으로 롤아웃 i가 모든 방법에서 같은 변형을 씁니다(E.5와 일치).
- 정책 입력: 카메라 `leftshoulder`와 `robot0_eye_in_hand` 224 × 224, 상태 `robot0_joint_pos`와 그리퍼 폭, `replan_steps = 5`(F.1과 일치). 특권 정보는 정책에 가지 않음(서버 요청은 이미지, 상태, 프롬프트뿐, `eval_server.py:150-155`).
- 공간 분할: 평가 BDDL의 대상 객체 영역이 ID는 y∈[−0.4, 0], OOD(`*_right_*`)는 y∈[0, 0.4]이고 x∈[0.05, 0.4](E.2와 일치).
- roll/pitch/yaw 제거: `grasp`, `in`, `on`, `stack`에서 roll, pitch를 버리고 `on`은 yaw도 버림(`configs/utils.py:113-146`), `mat2euler(axes="sxyz")`로 yaw만 남김(`pose_utils.py:335`)(D.4와 일치).
- 조인트 컨트롤러 `joint_pos.json` kp = 500(부록 B.6의 "500에서 234"와 일치).
- 어긋남 없는 개수: 양팔 학습 6,833 = 4,690 + 843 + 1,300(문서 `available_datasets.md`).

## 3. 정적 점검 요약

- **평가 harness는 읽기 쉬운 편**입니다. 정책은 websocket 서버로 분리되고(`mesa/serving`, `third_party/openpi-client`), 평가 서버는 단일 프로세스(`eval_server.py`)와 병렬(`eval_server_parallel.py`) 두 가지입니다.
- 에피소드 요약 JSON에는 `distractor_completed`와 `completed_distractor_tasks` 목록이 있어, 외부에서 L, 중복 계수, 분산 과제 종류별 혼동을 다시 계산할 수 있습니다(공식 평가를 돌릴 수 있을 때).
- **재현을 어렵게 하는 점**: 평가 초기 상태와 자산이 저장소 밖(Hugging Face)이고, L 계산, 로그, 기준선 학습 설정(openpi 포크의 학습 설정 파일)이 저장소 밖이며, 논문 부록 F의 하이퍼파라미터 확인은 포크 저장소를 읽어야 가능합니다(이 분석에서는 읽지 않음).

## 4. 구현 가능성 판정 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**논문의 핵심 수치(표 2, 3의 정책 평가)를 이 맥에서 재현하는 것은 불가능합니다. 통계 재계산과 코드 정적 대조는 완료했습니다. 평가 harness 자체와 데이터 생성(MimicGen)을 CPU로 돌려 보는 것은 조건부로 가능합니다.**

| 구성 요소 | 이 맥에서 | 필요한 것 | 노력(추정, 미측정) |
|---|---|---|---|
| 표 재계산, 검정, 일관성 검사 | **완료** | `analysis/recompute.py` | 완료 |
| 공식 코드 정적 대조 | **완료** | - | 완료 |
| 평가 harness 설치와 무작위 정책 스모크 | 조건부 | `uv sync`(Python 3.10, mujoco 3.4.0, robosuite 1.5.1 설치), 자산 약 13 GB(`--no-bimesa` 약 5 GB), 초기 상태 0.17 GB 다운로드, 공식 코드 실행 승인. 문서는 Ubuntu 24.04만 검증, macOS의 오프스크린 렌더링은 미확인 | 1–2일 |
| 새 과제 생성(BDDL), MimicGen 데이터 생성 | 조건부 | 위 설치, 원천 시연 데이터 다운로드(크기 미확인), CPU 시뮬레이션 | 며칠 |
| 공개 체크포인트(π0.5 12.4 GB, π0 등) 추론 평가 | 조건부, 소규모만 | openpi 설치(PyTorch 경로 확인 필요, MPS 미검증), 체크포인트 다운로드. 한 롤아웃에 정책 호출 약 100회(500스텝 ÷ 5), 3B 모델의 MPS 지연 미측정. 전체 7,000롤아웃(약 70만 호출)은 비현실적, 과제 몇 개 × 10롤아웃 수준만 | 1–2주(소규모) |
| 기준선 학습(π0, π0.5, GR00T 50k 스텝) | **불가** | H100 2장 21시간 또는 L40S 4장 48시간(F.3), 시연 데이터 11.7 GB | GPU 필요 |
| DP 학습(80M, 320k 스텝) | 사실상 불가 | 시간 미측정. 성능이 거의 0이라 가치도 낮음 | - |
| 양팔 GPT 에이전트 평가 | **불가** | 유료 API(사용 승인 없음), 1,450롤아웃, 지연 13배 | - |
| LFR, 유형 보정, 과제 부트스트랩을 새 정책 로그에 적용 | 가능 | 에피소드 요약 JSON(평가를 돌린 경우) | 수 시간 |

선택 사항이던 소형 toy는 만들지 않았습니다. 중심 주장(표 값, 순위, 유의성)은 표 재계산으로 검증되었고, 가장 의미 있는 검증은 정책 평가 자체라 CPU toy로 대체할 수 없습니다.
