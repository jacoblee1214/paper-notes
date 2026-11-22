# 04. 공식 코드 대조 (GaTech-RL2/armory)

기준: `GaTech-RL2/armory` `main` 브랜치, 커밋 `e876202ede99723f4be40d8d7cab31847bbd14a9` (2026-08-04, "Update badge links in README.md"). 얕은 클론을 `repo/`에 받아 **정적으로만 읽었습니다**(실행, import, 설치, 가중치·데이터 다운로드 없음). `repo/`는 `.gitignore`입니다. 논문과 같은 기간에 다른 브랜치(`paper_sweep_experiments`, `rohan-real-0513`, `ablate_mirror`, `submission_checkpoint`, `train-time-rtc` 등)가 있고 마지막 push는 2026-11-15입니다. 설정 파일 한 개(`configs/client/sim_sweep/one_fast/10_robots.json`)를 GitHub API로 읽었고, 다른 브랜치의 코드는 읽지 않았습니다. 클론한 내용은 신뢰하지 않는 데이터로 취급했습니다.

## 1. 저장소와 라이선스

| 항목 | 내용 |
|---|---|
| 라이선스 | Apache-2.0 (`LICENSE` 파일, GitHub API 일치). 서브모듈은 MIT(LIBERO 포크), Apache-2.0(OpenPI 포크, Isaac-GR00T 포크) |
| 구조 | `src/armory/`(스케줄러·서빙·백엔드), `armory-client/`(로봇용 경량 클라이언트), `src/evaluation/`(LIBERO 러너·지표·Toxiproxy), `scripts/`(로컬·Modal·Slurm), `configs/`, `tests/` |
| 규모 | Python 약 13.5k줄(서브모듈 제외), 얕은 클론 1.6 MB |
| 백엔드 | OpenPI(π0/π0.5), Isaac GR00T N1.7 포크 |
| 의존성 | Python 3.11 고정, `uv.lock`의 `required-environments = sys_platform == 'linux'`(Linux 전용), 서버는 `jax[cuda12]` (Linux). 서브모듈 3개(`git submodule update --init`) 필요 |
| CPU 모드 | `policy:mock`(프로파일 지연만큼 sleep하는 모의 정책)와 모의 환경 제공. 스케줄러 비교용 (README "mock mode") |
| 테스트 | `tests/scheduling/mirror_test.py`(거울: 청크 추적, step_forward, 마감, fast_forward, 트윈, 완료 후 도착 보정), `action_chunk_broker_test.py` 등. **LA 점수·가중치·후보 생성을 직접 검사하는 테스트는 없음** (`lookahead` 문자열이 나오는 테스트는 설정 해석 한 곳뿐) |

## 2. 논문과 코드의 차이 (D1–D12)

| # | 논문 | 코드 (파일:위치) | 영향 |
|---|---|---|---|
| **D1** 스케줄러 | 비교 대상은 RR, EDF, LA 셋. 부록 A.1의 EDF는 "마감 순 정렬 후 앞 b개" | `serving/scheduler.py`의 레지스트리에 7개: `max-batch`, `greedy-deadline`, `dynamic-action`, `lookahead-actions`, `round-robin`, `random`, `starvation`. **EDF가 두 가지**: `MaxBatchScheduler`("edf_prefix", 논문의 의사코드와 같음)와 `GreedyDeadlineScheduler`("edf_with_latency_fit": 최초 마감 전에 추론이 끝나는 가장 큰 배치 크기를 고르고, 없으면 처리량이 가장 좋은 크기). 기본값(`SchedulerConfig.scheduling_algorithm`)은 `greedy-deadline` (`baselines.py` 27–86, `protocol.py` 16) | 논문 표의 "EDF"가 어느 쪽인지 불명. 후자라면 A.4의 "EDF는 가능한 한 크게 채운다"와 맞지 않음. `gen_configs.py` 기본 스윕은 `max-batch greedy-deadline round-robin` 셋 다 포함 |
| **D2** LA 점수 | 식 10: 에폭 동안 **모든 로봇**의 비기아 스텝 가중 합. 식 13: 에폭 보상 / d_infer. A.1: "1초 창 안의 총 실행 로봇 시간" | `lookahead_actions.py`: `HORIZON = 1.0`, `GAMMA = 1`. `_evaluate`가 후보 배치를 큐에 넣은 트윈을 창 끝까지 굴리고 `score_sum = Σ score_j · weight_j`를 **스케줄의 총 GPU 시간**으로 나눔(258쪽 부근). 그런데 `mirror.py`의 `step_forward`는 `reward = 1.0 if is_available and chunk.origin == "searched"`: **후보 청크(origin "searched")가 만든 실행 스텝만** 점수에 들어감. 기존 큐의 스텝은 0 | 식 10과 다름(전 로봇 총합이 아니라 후보 청크 몫). L = 1에서 식 13이 구성 정보를 잃는 문제(`03` V10)를 코드가 푸는 방식. **토이는 이 읽기(A)에서 가중 LA가 느린 계층을 굶기고 표 4와 어긋나며, 한계 이득 읽기(B)가 표 4와 맞음(`03` §3)**. 실행하지 않고는 판정 불가 |
| **D3** 거울 | 로봇별 i_j, î_j, Q_j, 지연(EMA) 추적 | `mirror.py`: `Robot.steps`(최대 30), `Robot.chunks`(최대 5), `ControlStep`, `ActionChunk`(origin: queued/searched/completed). 지연은 `EMALatencyTracker(alpha = 0.1)`(`latency.py`). 서버 시작 시 배치별 지연 프로파일을 받아 `update_infer`로 시드 | **일치.** 지연 추정이 EMA라는 부록 A.6의 서술과 같음. 청크 큐 길이 5와 스텝 이력 30은 논문에 없음 |
| **D4** 후보 배치 | A.1: "크기 1..B_max 배치를 열거, 휴리스틱으로 가지치기" | `_candidate_batches`: 마감 순으로 정렬한 뒤 **가중치 값이 같은 로봇끼리 한 계층**으로 묶고, 크기별로 "계층별 몇 개를 EDF 순으로 뽑을지"의 조합(`compositions`)을 후보로 함. 즉 가중치가 모두 같으면 **후보는 EDF 접두 배치 b개뿐**(크기 1..b) | 가중치가 같은 LA@1은 "EDF + 점수로 배치 크기를 고름"과 같음. LA의 이득(느린 계층 +28조각, `03` V3)이 로봇 선택이 아니라 **배치 크기 선택**에서 온다는 가설과 양립. 두 계층일 때만 로봇 선택이 달라짐 |
| **D5** 자격과 드롭 | A.9: H_min 미달 요청, 같은 요청 재전송은 배치에서 제외 | 스케줄러 쪽 `Mirror.schedulable_robot_ids`가 `can_serve`(예상 시작 인덱스 ≥ 마지막 청크 시작 + H_min)로 후보를 거름(`mirror.py`, `schemas.py` 35). 엔진 쪽 `engine.py` 138줄도 같은 `can_serve`로 한 번 더 거름 | **일치.** 토이가 이 필터를 느슨하게 모델링해서 EDF·RR의 평균 배치 크기(토이 10 대 논문 4.6–4.7)가 어긋남 |
| **D6** 지연 표 | 표 1, 표 2 | `backends/inference_profiles.json`: `pi05.l40s` 배치 1–20 (0.0730, 0.1098, 0.1424, 0.1775, 0.2111, ...), `pi05.h100` 1–5, `gr00t.l40s` 1–20. `mock.py`가 이 값만큼 sleep | **일치**(표 1, 2의 모든 값). 코드는 배치 6–20까지의 값도 있음(논문 표는 1–5). 시뮬·실로봇 중 어느 설정의 측정인지 표기 없음 |
| **D7** 탐색 예산 | A.1: "LA는 GPU가 바쁜 동안 계속 탐색하고, GPU가 비면 최선의 배치를 보냄. L개 에폭의 모든 스케줄을 열거" | `LookaheadActionsScheduler` 기본값: `max_depth = 1`(L = 1 일치), `max_in_flight = 1`, `step_budget_nodes = 8`, `scheduling_buffer = 0.05`, `idle_durations = ()`(유휴 삽입 없음). `dispatch_budget = max(0, max_in_flight − in_flight)`이므로 **GPU가 바쁠 때 한 탐색 결과는 버려지고**, 비었을 때(슬랙 ≈ 0) 탐색은 최소 한 번의 `step(8)`, 즉 **후보 8개까지만** 평가하고 끝남(`get_next_batches`, `IncrementalSearch.step`) | 후보가 8개를 넘으면(두 계층 + b = 5이면 20개, b = 3이면 9개) 뒷부분(예: 빠른 로봇만 3대)이 평가되지 않음. 실행 확인 전까지는 정적 읽기의 의심. 토이에서는 영향이 작거나 불명확(`03` §3 TS-4) |
| **D8** 실험 설정 | 시뮬: 5분 창, 에피소드 타임아웃 30초, b = 3, 과제 2개, H_max 6/10, H_min 1, 20 Hz. LA 가중치 1, 3, 5 | `main`의 `gen_configs.py` 기본값은 `time_limit = 120`, `max_steps_per_episode = 300`, `max_batch_sizes = (5,)`, `num_steps = 10`, H_max 6/10, H_min 1, 20 Hz. 브랜치 `paper_sweep_experiments`의 `sim_sweep/one_fast/10_robots.json`은 `max_steps_per_episode = 600`(30초 × 20 Hz), `time_limit = 300.0`, `task_subset_size = 2`, H_max 6/10, H_min 1로 **논문 설정과 일치**. 두 경우 모두 로봇 설정에 **`weight` 필드가 없음**(기본 1.0). `gen_configs.py`에도 가중치 옵션이 없음 | 논문의 LA@3, LA@5(가중 스윕)를 만드는 설정은 읽은 범위에서 찾지 못함. `Robot.weight`는 클라이언트에서 서버로 전달됨(`types.py` 116, `client.py` 48–68). 기본 설정으로는 모든 LA가 LA@1 |
| **D9** 실로봇 | PiPER 10대, 턴테이블·벽돌 과제, π0.5 H = 20, f_c = 30 | `main`에 `piper`, `agilex`, `turntable` 문자열이 있는 파일이 없음. 실로봇 클라이언트·과제 코드·미세조정 체크포인트가 없음. 브랜치 `rohan-real-0513`("realworld" 병합)이 있으나 읽지 않음 | 표 5, 그림 4, 그림 5, 10대 시연은 이 저장소로 재현 불가 |
| **D10** 지표 | 처리량 = 분당 성공 수(시뮬) 또는 분당 레고 개수(실), 기아율 = 기아 스텝 비율 | `evaluation/metrics/summary.py`: 처리량 = `sum(success) / experiment_duration`(초당 성공 수, 60배하면 분당), 기아율 `planner_starvation_rate = starvation_steps / observed_steps` (전 에피소드), 그 외 "첫 행동 이후만 센" `post_first_starvation_rate`도 있음. 성공률은 잘린(truncated) 에피소드를 제외 | 논문이 어느 기아율 정의를 쓰는지 불명. 처리량은 일치 |
| **D11** 네트워크 | A.6: 편도 지연마다 로그정규 분포에서 요청마다 새 표본 | `evaluation/toxiproxy.py`: 로그정규(`rng.lognormal`)로 표본을 뽑아 Toxiproxy 지연으로 설정. **`resample_every_requests` 기본값 10**(10 요청마다 새 표본) | 논문의 "every request"와 기본값이 다름. 실제 값은 알 수 없음. 지터 결과(그림 8a)에 영향 가능 |
| **D12** 모델 설정 | π0.5, 디노이징 스텝 수 언급 없음 | `EngineConfig.num_steps = 10` (`serving/config.py`), `gen_configs`도 10 | 표 1의 지연이 10스텝 기준일 가능성. 논문에 없음 |

## 3. 일치한 것

푸시 기반 통신, 서버 쪽 거울과 EMA 지연 추정, 프로세스 분리(WS, 스케줄러, GPU 엔진이 ZMQ·공유 메모리로 통신), 가중치 필드의 전파(`Robot.weight` → `SlotRequest.weight` → 거울), L = 1 기본값, 1초 창(`HORIZON`), 지연 프로파일(표 1, 2), H_min 필터, 처리량의 정의, 기본 시뮬 설정(H_max 6/10, H_min 1, 20 Hz)입니다.

## 4. 구현 가능성 판정 (구현할 수 있는지)

**이 맥(Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)에서 논문 수치의 재현은 불가능합니다. 스케줄링 로직의 독립 재구현과 모의(mock) 환경 확인은 가능하고, 일부는 끝냈습니다.**

| 항목 | 이 맥에서 | 근거와 필요한 것 |
|---|---|---|
| 실로봇 10대(표 5, 그림 4) | **불가** | AgileX PiPER 10대, RealSense, 턴테이블 과제, 미세조정 π0.5 체크포인트. 코드가 저장소에 없음(D9) |
| L40S에서 π0.5 서빙과 LIBERO 시뮬(표 4) | **불가** | JAX CUDA, π0.5 체크포인트, LIBERO 다중 클라이언트. 논문은 서버 L40S와 별도 클라이언트 노드를 씀. `uv.lock`이 Linux 전용(D13) |
| 거울·스케줄러 로직의 재구현 | **가능, 일부 완료** | `analysis/toy_mirror_sim.py`: 표 1 지연으로 RR, EDF, LA를 구현해 표 4 기아율을 재현(MAE 1–2 %p, `03` §3). 정확한 재구현(청크 추적, 트윈, 탐색)은 약 1–2주(추정) |
| 공식 코드의 mock 모드(CPU) | **조건부** | `policy:mock`와 모의 환경은 GPU 없이 스케줄러 비교를 지원함. 그러나 `uv sync`가 Linux 전용 잠금이고 Python 3.11, ZMQ, 다중 프로세스를 요구. 새 설치와 공식 코드 실행 승인이 필요. 이 맥에서의 속도·타이밍 정확도는 측정 전(스케줄러가 시간 정밀도에 민감: 슬랙 50 ms) |
| 가중 LA의 점수 해석 확인(D2) | **조건부** | mock 모드에서 `lookahead-actions`를 one-fast로 돌려 느린 계층의 기아를 보면 읽기 A와 B를 가를 수 있음. 새 설치 승인 필요 |
| 그림 3(지평·기아가 처리량에 미치는 영향) | **불가(GPU, 로봇)** | 정책과 환경이 필요 |

**노력 추정:** 토이(완료) 1–2일. 거울과 LA를 충실히 다시 짜서 mock 환경과 대조하는 데 1–2주. 공식 코드를 Linux GPU 기기에서 LIBERO까지 돌리면 환경 구성 1–2일, Modal 사용 시 비용과 계정이 필요(README 상 Modal 배포). 실로봇 10대는 이 저장소 밖의 장비 문제입니다.

### D13. 구현 환경 메모
- `pyproject.toml`: `required-environments = ["sys_platform == 'linux'"]`, `requires-python >=3.11,<3.12`, 서버 extra에 `jax[cuda12]`. 평가 extra에 `mujoco==3.7.0`, `robosuite==1.4.1`, `robomimic==0.2.0` 고정(LIBERO).
- `scripts/sbatch/`는 Georgia Tech Phoenix 클러스터용 Slurm 스크립트이고 `scripts/modal/`는 Modal(L40S 서버, T4 클라이언트)용입니다.

## 5. 확인하지 못한 것 (정적 읽기의 한계)

- D2의 읽기 A/B, D7의 후보 절단은 **실행하지 않고는 판정할 수 없습니다.** 코드를 잘못 읽었을 가능성이 있습니다(특히 `origin` 태그가 production 경로에서 어떻게 붙는지).
- 브랜치 `rohan-real-0513`(실로봇)과 `paper_sweep_experiments`의 코드 본문은 읽지 않았습니다. `main`이 논문 제출(2026-07-31) 직후(2026-08-04)에서 멈췄고 다른 브랜치가 2026-11-15까지 갱신되어, 논문 결과를 만든 정확한 커밋이 `main`인지 확인하지 못했습니다.
- 서버 쪽 `scripts/serve.py`, 엔진의 JAX 경로(`openpi_adapter`), 클라이언트의 청크 브로커 외 부분은 훑지 않았습니다.
