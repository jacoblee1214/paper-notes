# 04. 공식 코드 대조와 구현 가능성

## 0. 대상과 방법

- 저장소: `Atarilab/motiondisco` (공식, README가 논문과 CoRL 2026 인용을 적음). 커밋 `0df8eab8623f58b447cd8402836ceffcdd472360` (단일 "Initial commit", 2026-11-06). 얕은 복제를 로컬에 두었고(약 37 MB, 서브모듈은 받지 않음) **정적으로만 읽었습니다.** 코드를 실행, import, 설치하지 않았고 LLM API도 부르지 않았습니다.
- 서브모듈 `submodules/faro`(Atarilab/faro, BSD-3-Clause)에 **논문의 KSO와 TO 본체와 장면 정의**가 있습니다. 서브모듈은 받지 않았으므로 솔버 내부는 읽지 못했습니다(D17).
- 읽은 파일: `README.md`, `NOTICE`, `examples/loco_manipulation/{README.md, run_search.sh, evaluate.py, scoring.py, to_eval.py, ik_eval.py, count_valid_plans.py, prompts.py, initial.py, env.py}`, `shinka/configs/{config.yaml, evolution/*, database/*, task/loco_manipulation.yaml, variant/*}`, `shinka/database/parents.py`, `shinka/core/{config.py, async_runner.py}` 일부.
- 판정: **일치**(논문 값과 코드 값이 같음), **차이**(다름 또는 논문에 없는 기능), **미확인**(저장소에서 찾지 못함). 코드는 논문 게시(2026-06-04) 5개월 뒤에 단일 커밋으로 올라왔으므로, 코드의 값이 논문 실험 시점의 값이라는 보장은 없습니다(D18). 그 점을 감안해 "논문 실험 설정이 코드와 다를 가능성"으로 읽어 주십시오.

## 1. 설치 요구사항과 라이선스

- 라이선스: Apache-2.0(저장소 루트 LICENSE). 탐색 엔진은 ShinkaEvolve(Apache-2.0) 파생이고 NOTICE에 수정 파일 목록이 있습니다. FARO는 BSD-3-Clause, 그 밖의 제3자 코드와 자산은 각자의 라이선스를 따른다고 README에 적혀 있습니다.
- 설치(README): 서브모듈 포함 clone → FARO의 `install.md`에 따라 conda 환경(`motiondisco`)에 acados와 moto를 빌드하고 FARO를 editable 설치 → `pip install -e .`. LLM 키(`ANTHROPIC_API_KEY` 등) 또는 로그인된 `claude` CLI(`claude-max/` 모델), 임베딩용 Ollama 서버(`run_search.sh`의 주석). "장면을 처음 쓸 때 moto 코드 생성에 장면·목표 모드당 약 20분, KSO 지평 구간마다 acados 빌드 1회"가 걸립니다(README).
- 공식 저장소의 평가 시간 한도는 평가 하나당 1시간 15분(`task/loco_manipulation.yaml`의 `job_config.time`)입니다.

## 2. 논문 ↔ 코드 대조 (D 항목)

| 번호 | 항목 | 논문 | 코드 | 판정 | 영향 |
|---|---|---|---|---|---|
| D1 | "유효(valid)" 계획의 정의 | §4.1: 순차 기구학 검사를 통과하고 **TO가 수렴** | `count_valid_plans.py` L9: "A 'valid' contact plan = **TO actually ran on it**"(수렴 여부 무관, 오류로 끝난 시도도 포함 가능). 한편 `to_eval.py`의 `ToSolve.valid`는 "수렴했거나 물리 검사(`trajectory_check`)를 통과"하고 목표 조건을 만족(L235). 저장소 안에 정의가 셋 | **차이** | 표 1의 Valid %가 어느 정의로 센 것인지 불명. TO를 시도했지만 수렴하지 않은 계획이 Valid %에 들어갈 수 있음 |
| D2 | "첫 유효해 시간" 지표 | 표 1: "time required to find the first valid solution", 본문: "within a few minutes" | `count_valid_plans.py`의 `_time_to_first_to_in_run`(L219): **gen_1이 끝난 시각**부터, gen_2 이후 첫 **TO 시도**(수렴 여부 무관, L199)까지의 파일 시각 차. 주석: gen_1에서 acados 솔버가 만들어지므로 컴파일을 뺀다 | **차이 (논문에 정의 없음)** | 표의 분(1.34–7.49)은 컴파일(장면당 약 20분)을 뺀 값. gen_1이 이미 유효했어도 시계가 그 뒤에서 시작 |
| D3 | 비용식 | 식 5: J = Σᵢ‖r‖₁ + w_j Σᵢ‖q⃛ᵢ‖² (정규화 없음, w_j 값 없음) | `to_eval.py` L7–8, L604–612: `to_cost = 1e2 · Σ(constraint violation) / num_nodes + 1e-2 · finite_diff_jerk / num_modes`. 노드 수와 모드 수로 나눔. `NODES_PER_MODE = 20`(L43, 주석은 FARO 기본 10과 다르다고 적음) | **차이** | 코드의 비용은 계획 길이로 정규화되어 길이가 다른 계획을 비교 가능하게 함. README는 "averages constraint violation over the horizon, so padded plans look better"라고 인정. 표 1 비용의 규모가 이 식에 의존 |
| D4 | 비용 → 점수 F, 수렴 처리 | 점수 Fₙ의 정의가 없음. "KSO 실패 시 점수 0" (§3.2.4) | `scoring.py` L77: `legacy_score = 1/(1+to_cost)` (to_cost > 0일 때), 아니면 0. **TO 수렴 여부를 보지 않음**(`evaluate.py` L526, `scoring.py` docstring: "ignores TO convergence"). README: "the default legacy rule is the sharp gate the CoRL ablations use" | **차이 (논문에 정의 없음)** | 비수렴 TO도 양의 점수를 받아 "최선 비용"과 부모 선택에 들어갈 수 있음. `count_valid_plans.py`는 최선 비용을 수렴한 것 위주로 고르되 수렴이 없으면 비수렴으로 대체하고 `*`로 표시(논문에 언급 없음) |
| D5 | 텍스트 피드백 내용과 "w/o TF" 구현 | §3.2.4, 식 6: **가장 긴 실현 가능 접두부**와 처음 실패한 모드, 잔차와 저크 비용 | `evaluate.py`(L646–700) + `violation_feedback.py`(822줄), `step_feedback.py`, `grip_feedback.py`, `repair_probe.py`: KSO와 TO의 제약 위반을 계획의 어휘(모드, 채널, 접촉면, "gap/slip/overhang" 등)로 번역, 걸음 보폭 진단, 양손 파지 진단, 걸음 삽입 수리 시도. "w/o TF"는 환경변수 `SHINKA_HIDE_IK_FEEDBACK=1`(L386)로, 위 설명 블록과 `kso_*` 필드를 숨김. 단 `ik_feasible`, `to_cost`, `to_converged`, 목표 위반은 계속 노출(L615–623) | **차이 (코드가 훨씬 풍부)** | 논문이 말하는 피드백은 식 6 한 줄이지만 코드의 TF는 훨씬 넓음. 이 풍부한 피드백이 논문 실험에 쓰였는지는 알 수 없음. TF 효과의 해석이 달라짐 |
| D6 | KSO 판정의 잡음과 접두부 탐색 | 식 6: "iteratively shortening the sequence until the longest feasible prefix" | `ik_eval.py` L25–28: KSO 판정이 1e-2 경계 근처에서 잡음이 있어 동일 입력도 통과 또는 실패 → 전체 계획 해를 최대 `KSO_ATTEMPTS = 3`회(L130), 접두부 탐색은 길이마다 `PREFIX_ATTEMPTS = 4`회, 수리 시도 `PROBES = 2`회(L131–134). 허용 오차 `_FEASIBILITY_TOL = 1e-2`(L155) | **차이 (논문에 없음)** | 접두부 실현 가능성이 단조라는 가정은 잡음 있는 솔버에서 깨질 수 있음. 판정은 확률적. 재시도 횟수가 평가 시간에 들어감 |
| D7 | 사전 검사와 점수 0 | §3.2.3: 구문과 접촉 모드 정의 일관성 검사 | `evaluate.py`: (i) **정확히 같은 계획은 이미 평가했으면 점수 0**(L399 부근 `_check_plan_duplicate`), (ii) 이산 목표를 최종 모드가 만족하지 않으면 점수 0, (iii) 확장된 모드 수가 `MAX_NUM_NODES`(기본 **50**, L88)를 넘으면 점수 0, (iv) 일정 오류 | **차이 (논문에 일부만)** | 노드 수(분모)에 이런 즉시 실패 노드가 포함되는지 불명. 50모드 상한은 본문에 없음 |
| D8 | 접촉 유형(프롬프트) | §6 한계와 부록 A.1 프롬프트: "contact_type = 1 (unilateral) is the only supported value for now" | 저장소 `prompts.py` L133–142의 시스템 프롬프트와 `initial.py` 주석은 "1 = sticking, **2 = sliding**"을 LLM에 안내. 같은 저장소의 `examples/loco_manipulation/README.md`는 "1 = unilateral (only value supported for now)" | **차이 (논문 프롬프트와 현재 코드 프롬프트가 다름, 저장소 내부도 불일치)** | 현재 프롬프트는 논문 부록 A.1의 프롬프트와 다른 버전. 논문 실험의 프롬프트가 코드에 그대로 있다고 보기 어려움(D18). 평가기가 유형 2를 어떻게 처리하는지는 FARO 쪽이라 미확인 |
| D9 | LLM과 설정 | §3.2: 모든 LLM 구성요소가 **Claude Opus 4.7** | `run_search.sh` L127–138: `claude-max/claude-opus-5-5`, 추론 강도 high, `max_tokens = 32768`, 임베딩 `local/embeddinggemma`. `variant/loco_diverse.yaml`: opus-5-5 + sonnet-5-5 혼합, 온도 0.5/1.0. 기본 `medium_budget.yaml`: gpt-5-mini, gemini, gpt-5.4 혼합. 문서 예시 `docs/support_claude_max.md`는 `claude-opus-4-7` | **차이** | 코드의 기본 모델 문자열이 논문의 4.7과 다름. 논문 시점 설정과 현재 스크립트가 다르다는 신호 |
| D10 | 변이 유형 확률 | 부록 A.2: diff 0.6, full 0.3, crossover 0.1 | `run_search.sh`는 `evolution=small_budget` 사용: `patch_types: [diff, full]`, 확률 **0.5/0.5**, crossover 없음. 0.6/0.3/0.1은 `medium_budget.yaml`(최대 50 세대)의 값. `loco_diverse`는 0.3/0.5/0.2 | **차이** | 논문의 세 값은 `run_search.sh`의 실제 프리셋과 맞지 않고 `medium_budget`과 맞음 |
| D11 | 섬, 이주, 부모 선택, 새로움 판정 | 부록 A.2: 섬 균등 추출, 이주(엘리트 보호), wᵢ = σ(λ(F−α)/m)/(1+o), **LLM 새로움 판정** | `parents.py` L368–393: α = 점수 중앙값, MAD 하한 1e-6, `s = σ(λ·Δ/m)`, `h = 1/(1+자식 수)`, 여기에 `rarity` 인자 곱(기본 설정에서는 1로 추정). λ는 `DatabaseConfig` 기본 **10**. `island_small.yaml`(run_search): 섬 4개, 이주 간격 15, 이주율 0.1, 엘리트 유지. `island_medium.yaml`: 섬 2개, 이주율 0. **LLM 새로움 판정은 `novelty_llm_models`가 있어야 켜지는데**(`async_runner.py` L492) 기본·`run_search.sh`·`loco_diverse`에 설정이 없음(`ollama_local.yaml`과 `shinka_ollama.yaml`에만 있음). `loco_diverse.yaml` 주석: 코드 임베딩 새로움 판정은 "never rejected anything" | **일치 (식), 차이 (새로움 판정, 이주)** | 부모 선택식은 논문과 같음. 새로움 기각은 기본 설정에서 동작하지 않음. 노드 예산 11–40이 이주 간격 15와 4개 섬에 비해 작아 이주가 0–2번뿐이고 섬당 노드 2–10개 (T1 참고) |
| D12 | TO 솔버와 설정 | §3.1: acados [26], HIPPO [27]. 설정 수치 없음 | `to_eval.py`: `NODES_PER_MODE = 20`, `TIME_PER_MODE = 1.0` s, 최대 NLP 반복 `SHINKA_TO_MAX_ITERS = 300`(L60), 실현 가능성 허용 `TO_FEASIBILITY_TOL = 0.05`(L63), 모드 지속 시간을 변수로 | **논문에 없음** | 비용과 "수렴"의 의미가 이 허용 오차에 의존. 솔버 자체는 FARO와 moto (서브모듈, 읽지 못함) |
| D13 | 탐색 예산(노드 수) | 표 1: "given the same search budget", 값 없음 | `run_search.sh` L109–122: 과제별 `SCENE_MAX_NODES`(Banana 27, Box 11, Climb 13, Long-Dist. 20, Clutter 15, Parkour 1 40, Parkour 2 34, Under-Table 14), 기본 40. 주석: `num_generations`가 노드 예산. 이 값을 분모로 쓰면 표 1의 Valid % 16칸 중 13칸이 버림으로 재현됨(`03` V1). `run_search.sh`의 실험 이름 `loco_abl3_ik_fb`(주석: 이전 릴리스의 이름 유지) | **일치 (13/16칸), 불일치 3칸** | 예산이 과제마다 11–40으로 다르고 논문에 없음. Parkour 1(40)과 Banana MD(70.30)의 칸은 이 값과 안 맞음 |
| D14 | SC 기준선 | §4: 단일 LLM 호출(SC) | 저장소에서 SC 전용 설정이나 스크립트를 찾지 못함(`single`, `SC`, `baseline` 검색에서 일치 없음). SC가 첫 LLM 제안(gen_1)인지 독립 호출의 최선인지 판별 불가 | **미확인** | 표 1의 SC 열(4/8 성공)은 코드로 재현 경로가 보이지 않음 |
| D15 | 실로봇 추종 정책 | §4.3: DeepMimic 보상 + 도메인 무작위화 RL, 영 샷 실로봇 | 저장소에 RL 학습, 배포 코드 없음(`ppo`, `isaac`, `mjlab`, `unitree` 검색). 있는 것은 **추종 데이터셋 생성**: `generate_dataset.py`, `dataset_export.py`(물체 질량 0.5–2배, 크기 0.85–1.15배, 로봇 시작 위치·방향, 타이밍을 무작위화해 TO를 다시 풀고 MuJoCo용 궤적을 내보냄) | **미공개 (부분)** | 실로봇 결과(§4.3, 부록 B.2)는 코드로 재현할 수 없음. 데이터 생성 쪽만 공개 |
| D16 | 논문에 없는 기능 | — | 계획 점수 `score_mode=quality`(수렴 게이트, 부분 점수), 다양성 모드(`variant=loco_diverse`: 4개 고립 섬, MAP-Elites형 보관, 전략 계열 필터), 창 단위 증분 TO와 캐시, 걸음 수 자동 보정, 데이터 캠페인, 장면 캠페인, 100모드 이상 계획(예: 113모드) | **논문에 없음** | README의 "30 to more than 100 contact modes"는 논문 과제(README 갤러리의 논문 과제 24–43모드)보다 큼. 논문 결과는 `legacy` 기본값이 재현한다고 적혀 있음 |
| D17 | 솔버 본체 | KSO, TO(§3.1) | FARO 서브모듈에 있음. 이 읽기에서는 서브모듈을 받지 않음 | **미확인** | 식 3의 구조, 충돌 제약, 접촉 모델은 코드로 확인 못 함 |
| D18 | 코드 이력 | — | 단일 "Initial commit"(2026-11-06). README는 "Research code accompanying ... (CoRL 2026)" | **확인** | 논문 시점 대비 변경 이력을 볼 수 없음. 코드 값을 논문 실험 값으로 단정할 수 없음 |

요약: **구조(접촉 계획 프로그램, KSO → TO 파이프라인, ShinkaEvolve 부모 선택식, 평가 단계)는 논문과 맞습니다.** 차이는 (1) 논문이 정의하지 않은 지표(Valid %, 시간, 점수 F, 비용 정규화: D1–D4), (2) 피드백의 폭과 KSO의 재시도(D5, D6), (3) 프리셋이 논문 값과 어긋남(D9, D10, D11), (4) 예산 미기재(D13)에 모여 있습니다. 재현에 가장 큰 장벽은 FARO/moto 설치와 LLM 비용, 실로봇 추종 코드의 부재입니다(D15).

## 3. 한 번에 보는 위험 (재현할 때 고정해야 할 값)

논문 결과를 다시 얻으려면 최소한 다음을 고정해야 합니다: 장면별 노드 예산, `legacy` 점수 모드, TF 끄기(`SHINKA_HIDE_IK_FEEDBACK=1`), LLM 모델과 추론 강도, 섬 프리셋과 변이 확률, KSO 재시도 횟수, TO 반복 한도, "유효" 계수 방식(D1)과 시간 시계(D2).

## 4. 구현 가능성 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**판정: 논문 결과의 전체 재현은 이 맥에서 불가능합니다. 탐색 루프의 축소 재구현과 수치 재계산은 가능하고 일부는 끝냈습니다. 공식 파이프라인(CPU 최적화 + 외부 LLM)은 조건부입니다.**

| 수준 | 판정 | 근거 |
|---|---|---|
| 표 1 재계산, 일관성 검사 | **가능, 완료** | `analysis/recompute.py`(수 초) |
| 부모 선택식 토이 | **가능, 완료** | `analysis/toy_parent_weights.py`(1초 미만, numpy) |
| 공식 탐색 파이프라인(한 과제, LLM + KSO + TO) | **조건부** | 솔버는 CPU라 GPU는 필요 없음. 단 (1) 새 conda 환경과 acados/moto 빌드(맥 arm64 빌드 가능 여부는 FARO `install.md`를 읽지 못해 미확인), (2) 공식 코드 실행 승인, (3) LLM API 비용(과제당 11–40회 호출, 최대 32k 토큰 출력), (4) 임베딩 서버. 평가 시간은 V6의 추정으로 과제당 약 13–142분(독립·순차 가정) + 장면당 컴파일 약 20분 |
| 탐색 루프의 축소 재구현 (장난감 접촉 계획: 상자와 발판을 2차원 격자로 단순화, "LLM" 자리에 스크립트 변이, 기하 필터와 비용을 직접 구현) | **가능** | numpy/torch CPU. 섬 선택, 새로움 기각, 점수 변환, 피드백 유무의 효과를 시험 가능. 노력은 1–2주(추정, 측정 전) |
| 실제 KSO/TO(다중 사격, 접촉 명시적 최적화) 자체 재구현 | **사실상 불가에 가까움** | 수개월 규모. 공식 FARO와 moto를 쓰는 편이 현실적 |
| 추종 정책 RL 학습과 실로봇 | **불가** | 학습 설정이 본문에 없고 저장소에도 없음(D15). 휴머노이드 로봇과 시뮬레이터 GPU 학습이 필요 |

공식 파이프라인은 "논문 수치 재현"이 아니라 "탐색이 어떻게 움직이는지 보는" 목적에는 맞습니다. 단 이 맥에서 새 환경 구축, 공식 코드 실행, LLM 호출은 승인이 필요하고 이 분석에서는 하지 않았습니다.
