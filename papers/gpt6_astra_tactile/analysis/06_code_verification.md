# 06. 재구현 코드 검증 및 통합 보고

- 작성 역할: 코드 검증/통합 에이전트 (결과 검증용)
- 작성일: 2026-10-23
- 대상 코드: `code/` (`tactile_gpt_policy` 0.1.0). 재구현 대상 논문은 Wang et al., "GPT-6-Astra Understands Touch and Outperform Vision–Tactile–Language–Action Models", arXiv:2610.28457v1, 2026-10-20 (이하 "논문")
- 검증 범위: numpy 기반 toy 환경 `mock_insert_hole`과 오프라인 규칙 정책 `MockHeuristicBackend`. 실제 UniVTAC(arXiv:2602.10093) 시뮬레이터와 실제 LLM(GPT-6-Astra, Claude 등)은 **한 번도 실행하지 않았다.** 이 문서의 수치는 모두 toy에서 나온 것이므로 논문 수치와 비교할 수 없다.
- 실행 환경: Python 3.12, numpy, pyyaml, pytest. openai, anthropic, wandb, mlflow, torch, h5py, isaaclab은 설치되어 있지 않다(`importlib.util.find_spec`으로 확인). 패키지를 새로 설치하지 않았고 git commit도 하지 않았다.
- 모든 명령은 `code/` 디렉터리에서 실행했다.

---

## 0. 요약

| 항목 | 결과 |
|---|---|
| 단위·통합 테스트 | `python3 -m pytest` → **113 passed in 17.6 s**. 실패 0건, 수정 전후 모두 통과 |
| 증류 (40 demo 구성 / 10 demo 검증) | V+P+T: E_tac 8개, 필수 헤더 누락 없음. V+P: E_tac 0개, Contact interpretation 헤더 제외 |
| run_eval, V+P+T, 100 seed | **100/100 = 100.0%, Wilson 95% CI [96.3, 100.0]**, 평균 32.9 step, 결정 14.9회, LLM 호출 14.9회 |
| run_ablation, V+P, 같은 100 seed | **21/100 = 21.0%, CI [14.2, 30.0]**, 평균 270.6 step, 결정 102.4회. 실패 79건은 모두 step_limit |
| 차이 (V+P+T − V+P) | **+79.0 pp, Newcombe 95% CI [69.3, 85.8]**. 두 조건의 seed 목록이 같음을 확인 |
| 판정 | 촉각 유무에 따른 차이는 분명하다. 다만 V+P+T가 천장(100%)에 닿아 있고, 차이의 크기는 mock 설계와 휴리스틱의 탐색 범위로 정해진다(§3.3). 파이프라인이 제대로 연결됐다는 증거일 뿐, LLM이 촉각을 이해한다는 증거는 아니다 |
| 정적 점검 | SDK lazy import, 액션 bounds/format 검사, seed 결정성, 에피소드 사이 history·backend 세션 초기화, V+P 프롬프트에 촉각·privileged가 섞이지 않음: 모두 통과 (§4) |
| 수정한 버그 | 1건. V+P 조건의 mock K_τ 텍스트에 촉각 용어("deformation increases")가 남아 있었다 (§5.1) |

논문 Table 3은 Insert Hole에서 V+P+T 99%, V+P 31%를 보고한다(arXiv:2610.28457, Tab.3). 이번 toy 결과(100% 대 21%)도 방향은 같다. 하지만 toy는 SPEC §5.5(MockHeuristicBackend 항목)에서 "V+P+T ≥ 0.9, V+P ≤ 0.6"을 calibration 목표로 정하고 만든 것이다. 따라서 방향이 같다는 사실은 논문 주장을 뒷받침하는 근거가 되지 못한다.

---

## 1. 산출물 위치

| 경로 | 내용 |
|---|---|
| `code/results/configs/insert_hole_mock.yaml` | `configs/tasks/insert_hole_mock.yaml`을 복사하고 `context_dir`만 `results/contexts/...`로 바꾼 것 |
| `code/results/configs/eval_mock_100.yaml` | V+P+T, 100 seed, `distill_if_missing: false`(고정된 context만 사용), `resume: false` |
| `code/results/configs/ablation_mock_100.yaml` | V+P+T와 V+P, 같은 seed, 조건별 context 사용 |
| `code/results/contexts/insert_hole_mock/{V+P+T,V+P}/` | 증류한 C_τ: `skill.md`, `context.json`, `tactile_refs.npz` |
| `code/results/eval_vpt/` | run_eval 결과: `benchmark_V+P+T.json`, `insert_hole/V+P+T/{episodes.jsonl, summary.json, steps/*.jsonl}`, `run_config.json` |
| `code/results/ablation/` | run_ablation 결과: `ablation_insert_hole.json`, 조건별 `episodes.jsonl`, `summary.json`, `steps/` |
| `code/results/verify_static.py` | §4의 결정성, reset, bounds, 누출, eval과 ablation 일치 여부를 점검하는 스크립트 |
| `code/results/static_checks.json` | 위 스크립트의 출력 |
| `code/results/rerun_verification.sh` | 이 문서의 모든 결과를 다시 만드는 스크립트 |
| `code/results/*.log` | pytest, distill, run_eval, run_ablation, verify_static의 원본 로그 |

디렉터리 이름이 `insert_hole_mock`인 이유는 task yaml이 `${task}_mock/${condition}` 템플릿을 쓰고 task 이름이 `insert_hole`이기 때문이다. 결과 디렉터리 전체 크기는 약 39 MB이며, 대부분 step 단위 JSONL이다.

---

## 2. 단위·통합 테스트

```
$ python3 -m pytest
113 passed in 17.60s
```

파일별 테스트 수: test_action_parser 20, test_mock_env 14, test_descriptors 11, test_distill 11, test_backends 9, test_delta 7, test_runner 7, test_types 7, test_prompt_builder 6, test_render 6, test_policy 5, test_univtac_adapter 5, test_history 4, test_e2e_mock 1.

- 처음 실행했을 때부터 113개가 모두 통과했다. §5.1의 수정을 적용한 뒤 다시 실행해도 113개가 통과했다.
- `pyproject.toml`의 `addopts = "-q"` 위에 `-q`를 한 번 더 주면 `-qq`가 되어 요약 줄이 출력되지 않는다. 재실행 스크립트에서는 `-q`를 빼고 실행한다. 동작에는 영향이 없다.
- computed task 지시서에 나온 테스트 파일명(test_tactile.py, test_env.py, test_eval.py, test_knowledge.py, test_parser.py)은 실제로 존재하지 않는다. 구현자 A와 B는 SPEC §11의 이름을 따랐다. 검증 범위는 SPEC §11 표와 맞으므로 이름 차이는 문제로 보지 않았다.

---

## 3. 통합 E2E (MockHeuristicBackend + mock_insert_hole)

### 3.1 실행 명령과 출력

**(i) 오프라인 증류 (논문 §3.2, §4.1의 40 demo 구성 + 10 demo 검증 재현)**

```
$ python3 scripts/distill_task_skill.py --task-config results/configs/insert_hole_mock.yaml \
    --backend-config configs/backend.yaml --condition V+P+T --out results/contexts/insert_hole_mock/V+P+T
{"n_refs": 8, "missing_headers": [], "validation": {"n_points": 46, "intent_agreement": 0.652,
 "direction_agreement": 0.717, "event_coverage": 1.0}}          # 약 10 s
$ ... --condition V+P ...
{"n_refs": 0, "missing_headers": [], "validation": {... 같은 값 ...}}  # 약 0.8 s
```

- 데모 seed 0–49는 40/10으로 나뉘고, 테스트 seed 100000–100099와 겹치지 않는다.
- 이벤트 수(구성 데모 40개 합계): blocked 75, deform_up_no_advance 75(V+P+T만 해당), retract 75, realign 86, advance 80, success 40. contact_onset은 0이다. mock에서는 물체를 이미 쥔 상태로 시작하기 때문이다(A와 B가 보고한 내용과 같다).
- V+P+T의 mock calibration은 촉각 단서 4개를 축방향 전단으로 정규화한 뒤 expert 보정량으로 ridge 회귀한다. R²는 [0.934, 0.954, 0.931, 0.941]이고, 사용한 쌍은 51개다.
- 검증 지표(intent 0.652, direction 0.717)는 두 조건에서 완전히 같다. mock `validate_act`가 관측을 보지 않고 매 질의마다 상태를 초기화한 채 "advance"를 기본으로 답하기 때문이다(`context.json`의 `validation.failures` 16건이 모두 "expert adjust vs model advance"). **따라서 이 지표로는 증류 품질을 판단할 수 없다.**

**(ii) run_eval, V+P+T, 100 seed**

```
$ python3 scripts/run_eval.py --eval-config results/configs/eval_mock_100.yaml \
    --backend-config configs/backend.yaml --out results/eval_vpt --no-resume
insert_hole      V+P+T  100/100 = 100.0%  CI95 [96.3, 100.0]  decisions/ep 14.9
{"condition": "V+P+T", "mean_success": 1.0, "out_dir": "results/eval_vpt"}
real 0m48s
```

**(iii) run_ablation, V+P+T와 V+P, 같은 100 seed**

```
$ python3 scripts/run_ablation.py --ablation-config results/configs/ablation_mock_100.yaml \
    --backend-config configs/backend.yaml --out results/ablation
insert_hole      V+P+T  100/100 = 100.0%  CI95 [96.3, 100.0]  steps 33
insert_hole      V+P     21/100 =  21.0%  CI95 [14.2, 30.0]  steps 271
[{"a": "V+P+T", "b": "V+P", "diff_pp": 79.0, "diff_ci95_pp": [69.29, 85.83], "wilson_overlap": false, "same_seeds": true}]
real 1m02s
```

### 3.2 수치 요약

| 조건 | 성공 | 성공률 (Wilson 95% CI) | 평균 step | 평균 결정 수 | 평균 LLM 호출 | parse 실패율 | 종료 사유 | 입력/출력 토큰/ep (mock 추정) | wall/ep |
|---|---|---|---|---|---|---|---|---|---|
| V+P+T | 100/100 | 100.0% [96.3, 100.0] | 32.9 (최소 26, 최대 35) | 14.9 (최대 16) | 14.9 | 0 | success 100 | 88.0k / 0.81k | 0.48 s |
| V+P | 21/100 | 21.0% [14.2, 30.0] | 270.6 (중앙값 300) | 102.4 | 102.4 | 0 | success 21, step_limit 79 | 329.9k / 5.86k | 0.14 s |

- 차이: +79.0 pp, Newcombe hybrid-score 95% CI [69.3, 85.8]. 두 조건의 Wilson CI는 겹치지 않는다. CI 계산법의 출처는 Wilson(1927, JASA 22:209–212)과 Newcombe(1998, Stat. Med. 17:873–890)이다. runner의 Wilson 값은 99/100에서 [94.55, 99.82], 31/100에서 [22.78, 40.63]으로, `test_runner.py`에서 손으로 계산한 값과 일치함을 확인했다.
- run_eval의 V+P+T 결과와 ablation의 V+P+T 결과는 서로 다른 프로세스에서 실행됐지만 100개 seed 모두에서 (success, steps, n_decisions)가 같았다(`static_checks.json`의 `eval_vs_ablation.identical_per_seed = true`).
- V+P+T 에피소드의 의도(intent) 분포는 advance 1130, retract 180, adjust 180이고, 에피소드당 blocked는 평균 2.3회다. 정리하면 "하강 → 막힘 → 2 mm 후퇴 → 촉각 기반 보정"을 1–2회 반복하고 성공한다. V+P는 advance 3652, retract 3294, adjust 3294로, 격자 탐색이 끝없이 반복되는 패턴이다.
- 평균 LLM 호출 수가 결정 수와 같다. mock backend가 항상 올바른 형식으로 답하기 때문에 E2E에서는 format 재질의 경로가 한 번도 실행되지 않았다. 이 경로는 `test_policy.py::test_format_retry_then_success`와 `test_format_failure_and_refusal_hold`에서만 검증된다.
- 토큰 수는 mock이 텍스트 문자 수를 4로 나눠 추정한 값이고 이미지 토큰은 빠져 있다(`backends.py:374`). 실제 API 비용을 추정하는 데 쓰면 안 된다. 그래도 규모를 가늠하는 데는 쓸 수 있다. V+P는 결정 수가 약 7배 많아서, 실제 LLM으로 실행하면 실패 에피소드 하나가 300 step 동안 약 100회를 호출하게 된다. 논문은 호출 수나 비용을 보고하지 않는다(미확인).

### 3.3 해석: 차이는 있지만 mock 구조가 만든 차이다

결과는 "100% 대 0%"처럼 극단으로 갈리지 않았다. 그러나 다음 세 가지 이유로 차이의 크기를 의미 있게 해석할 수 없다.

1. **V+P+T는 천장에 닿아 있다.** step 수가 26–35로 좁게 몰려 있고 실패가 하나도 없다. mock 환경은 숨은 4자유도 오차(측방 2D, 축 각도 2D)를 촉각 단서 4개에 일대일로, 거의 선형으로 대응시킨다(A의 설계 메모). mock 정책은 이 대응을 데모에서 ridge 회귀로 학습한다(R² 약 0.93–0.95). 그러므로 V+P+T 조건은 사실상 "역함수를 학습한 뒤 한두 번 보정하는" 문제이며, LLM이 촉각을 해석하는 능력과는 관계가 없다.
2. **V+P의 21%는 휴리스틱의 탐색 범위가 상한을 정한다.** V+P mock 정책은 측방 {0, 0.5, 1.0}×`lateral_range_mm`(2.67 mm)와 기울기 {0, 0.5, 1.0}×`angle_range_deg`(5.27°)를 8방향으로 조합한 288개 후보를 seed별 무작위 순서로 시도한다(`backends.py::_search_candidate`). 진입 조건은 |e|<1 mm 그리고 θ<3°다. 따라서 각도 오차가 약 8.3°를 넘으면 원리상 맞출 수 없다. 실측에서도 그렇게 나왔다. 숨은 오차 크기별 V+P 성공은 다음과 같다(같은 100 seed, `default_rng(seed)`에서 다시 샘플링).

   | 숨은 각도 오차 | n | V+P 성공 |
   |---|---|---|
   | 4–6° | 39 | 14 (35.9%) |
   | 6–8° | 30 | 7 (23.3%) |
   | 8–10° | 31 | **0 (0%)** |

   | 숨은 측방 오차 | n | V+P 성공 |
   |---|---|---|
   | 1.5–2.3 mm | 36 | 13 |
   | 2.3–3.1 mm | 29 | 6 |
   | 3.1–4.0 mm | 35 | 2 |

   결국 V+P 성공률은 "격자 범위와 300 step 예산 안에서 운 좋게 맞는 비율"이다. 탐색 전략을 바꾸면 크게 달라진다. A의 단순 격자 탐색은 약 2%였고, 같은 mock에서 B의 20 seed 실행은 25%였다. 논문의 31%와 가까운 것은 우연이다.
3. **calibration 목표가 미리 정해져 있었다.** SPEC §5.5(MockHeuristicBackend 항목)는 mock에서 "V+P+T ≥ 0.9, V+P ≤ 0.6"이 나오도록 설계하라고 명시한다. 이번 결과는 그 목표를 만족한다는 확인일 뿐이다.

이번 E2E가 실제로 보여 주는 것은 다음과 같다. (a) 증류, 고정 context, 폐루프 평가, 집계, ablation이 끊김 없이 이어진다. (b) V+P 조건에서 촉각 정보가 정책에 실제로 차단된다. 차단되지 않았다면 두 조건의 결과가 같았을 것이다. (c) 같은 seed와 같은 context에서 결과가 결정적으로 재현된다. 논문의 핵심 주장, 즉 범용 멀티모달 모델이 촉각 표현을 보고 접촉 상태를 추론한다는 주장은 이 실험으로 **검증되지 않았다.**

---

## 4. 정적·행동 점검

`python3 results/verify_static.py` → `results/static_checks.json` (약 6 s)

### 4.1 SDK lazy import

```
$ python3 -I -c "import tactile_gpt_policy.policy.backends"
ModuleNotFoundError: No module named 'tactile_gpt_policy'
```

지시서에 적힌 명령을 그대로 실행하면 실패한다. 하지만 코드 결함은 아니다. `-I`(isolated mode)는 cwd를 `sys.path`에서 빼고 `PYTHONPATH`도 무시한다. 그리고 패키지를 설치하는 것(`pip install -e .`)은 금지되어 있다. 같은 의도로 경로만 명시해서 다시 확인했다.

```
$ python3 -I -c "import sys; sys.path.insert(0,'.'); import tactile_gpt_policy, ...backends, ...policy, ...eval.runner, ...knowledge.distill, ...envs.univtac_adapter, ...tactile.render; ..."
loaded heavy: []
OpenAIBackend ImportError: OpenAIBackend requires the OpenAI SDK: pip install openai (API key via OPENAI_API_KEY)
AnthropicBackend ImportError: AnthropicBackend requires the Anthropic SDK: pip install anthropic (API key via ANTHROPIC_API_KEY)
```

패키지 전체를 import한 뒤에도 openai, anthropic, wandb, mlflow, torch, h5py, matplotlib, isaaclab, PIL 중 어느 것도 `sys.modules`에 올라오지 않는다. 실제 backend는 생성자를 호출하는 시점에 SDK를 import하며, SDK가 없으면 설치 방법을 안내하는 ImportError를 낸다. `test_runner.py::test_runner_import_has_no_sdk_side_effects`도 같은 내용을 검사한다.

### 4.2 액션 bounds와 format 검사 (논문 §3.4의 "checks on command bounds and format")

입력 `{"dpos_mm":[35,-1,50], "drot_deg":[30,0,0], "gripper":1.7, "horizon":9}`(코드 펜스 포함), EEF z=0.59 m:

```
dpos_mm → [20.0, -1.0, 10.0]   drot_deg → [10, 0, 0]   gripper → 1.0   horizon → 5   target z → 0.600 m
warnings: clipped dpos_mm[0]: 35.0 -> 20.0 / clipped dpos_mm[2]: 50.0 -> 20.0 / clipped drot_deg norm: 30.0 -> 10.0 /
          clipped gripper: 1.7 -> 1.0 / clipped horizon: 9 -> 5 / clipped target_pos_mm[2]: 610.0 -> 600.0
```

범위를 넘는 값은 축별 clip(dpos), 노름 clip(drot), workspace clip 순서로 처리된다. workspace clip이 적용되면 dpos가 다시 계산된다(z 20 → 10 mm). 아래 입력은 모두 `ActionParseError`로 거부된다: 키 누락, 여분 키, enum에 없는 intent, horizon 2.5, NaN, JSON이 아닌 텍스트. clip을 쓸지 reject를 쓸지는 논문에 나와 있지 않으므로 [가정]이다(action_parser docstring에 명시됨).

### 4.3 seed 결정성

조건마다 seed 100000, 100007, 100042를 env와 policy를 새로 만들어 두 번씩 실행했다. 두 번의 실행은 reset 이미지 바이트, 모든 결정의 LLM 원문 텍스트, 목표 위치, 성공 여부, step 수까지 **6건 모두 동일**했다. 프로세스를 달리한 run_eval과 ablation의 V+P+T 결과도 100 seed 모두 같았다(§3.2).

### 4.4 에피소드 사이 초기화

runner는 env와 backend 객체를 에피소드 사이에 재사용한다(`_factories`가 backend를 한 번만 만든다). 따라서 상태가 넘어가는지 직접 확인했다. 같은 env, backend, policy 객체로 seed 100003을 실행한 뒤 seed 100011을 실행한 결과를, 새 객체로 seed 100011만 실행한 결과와 비교했다.

| 조건 | 결과 동일 | 첫 에피소드 뒤 history 길이 | 두 번째 에피소드 뒤 history 길이 | 두 번째 에피소드 결정 수 | 두 번째 에피소드 첫 record t |
|---|---|---|---|---|---|
| V+P+T | true | 15 | 13 | 13 | 0 |
| V+P | true | 81 | 113 | 113 | 0 |

`TactileGPTPolicy.reset()`은 `HistoryBuffer.reset()`과 `backend.new_session(episode_id)`를 호출한다. `new_session`은 MockHeuristicBackend의 mode, pending, candidates, gain 같은 에피소드 상태와 rng를 `(seed, crc32(episode_id))`로 다시 만든다. 따라서 결과는 에피소드 실행 순서와 무관하다. 논문 §4.1의 "episode별 독립 세션"과 일치한다.

### 4.5 V+P 프롬프트 누출 점검

V+P 조건에서 처음 3번의 결정 요청을 가로채 텍스트와 이미지를 확인했다.

- 이미지는 `scene_rgb`, `wrist_rgb`만 있고 촉각 이미지와 E_tac 이미지는 없다.
- 텍스트에서 `indentation`, `marker`, `shear`, `curl`, `tactile_l/r`, `e_tac`은 한 번도 나오지 않는다.
- privileged 키 10개(`hole_center_m`, `lateral_err_mm`, `angle_err_deg`, `contact_force`, `seed` 등) 가운데 `blocked`만 15회 나온다. 이 단어는 privileged 값이 아니다. proprio의 tracking error로 계산한 delta flag(`tracking_error_mm > 2`)와 K_τ 본문에 쓰인 같은 단어다. V+P에서도 쓸 수 있는 정보이므로 누출이 아니다. 한편 `test_prompt_builder.py`는 privileged에 sentinel 문자열을 넣어 누출 여부를 별도로 검사한다.

---

## 5. 발견한 문제

### 5.1 수정함: V+P 조건의 K_τ에 촉각 용어가 남음

- 위치: `code/tactile_gpt_policy/policy/mock_skill.py`, "Corrective strategies for obstructions" 줄.
- 증상: V+P 조건으로 증류한 `skill.md`에 "Condition: blocked or **deformation increases** without advancement"라는 문장이 들어 있었다. V+P 정책은 변형(deformation)을 관측할 수 없다. SPEC §6.1("V+P이면 촉각 플래그를 쓰지 않는다")과 §6.2(V+P에서는 Contact interpretation 제외)의 의도에도 어긋난다.
- 영향: mock 정책은 K_τ의 `mock_calibration` 블록만 읽기 때문에 수치에는 영향이 없다(수정 전과 후 모두 V+P 21/100). 그러나 실제 LLM으로 V+P ablation을 할 때 mock 증류 결과를 K_τ 템플릿으로 재사용하면, 정책이 볼 수 없는 신호를 기준으로 쓰라는 지시가 들어가 V+P 성능이 왜곡될 수 있다.
- 수정(최소 변경): `has_tac`가 False일 때는 "blocked (commanded advance not achieved)"를 쓰도록 바꿨다. V+P+T의 `skill.md`는 수정 전후 바이트 단위로 같음을 diff로 확인했다. 수정 뒤 두 context를 다시 증류하고 eval과 ablation을 다시 실행했으며, 수치는 같았다. pytest도 113개 모두 통과했다.

### 5.2 수정하지 않음: 관찰과 권고

1. **`context_dir`가 cwd 기준 상대경로다.** task yaml에 있는 `outputs/contexts/${task}_mock/${condition}`은 `--out`과 관계없이 cwd 아래에 캐시된다. 프롬프트나 증류 코드를 바꿔도 이전 캐시가 조용히 재사용될 수 있어 위험하다. 이번 검증에서는 결과 전용 yaml을 따로 두고, `distill_if_missing: false`로 고정 context만 쓰도록 했다. 기존 `code/outputs/contexts/insert_hole_mock/`(A와 B의 이전 실행에서 생긴 것)는 지우지 않고 그대로 두었다. 권고: 상대경로는 out_dir을 기준으로 해석하고, context에 코드나 프롬프트 해시를 기록해 불일치하면 경고하도록 한다.
2. **증류 검증 지표가 실제로는 아무것도 측정하지 않는다**(§3.1). mock `validate_act`는 관측과 상관없이 같은 답을 낸다. 실제 LLM을 붙이기 전에 최소한 "막힘 직후에는 retract나 adjust를 기대한다" 같은 규칙 기반 baseline과 비교하는 기준이 필요하다.
3. **K_τ의 "Events per demo" 수치가 `n_events`와 맞지 않는다.** V+P+T 텍스트에는 blocked 1.7, retract 1.0이 적혀 있지만, `n_events`로 계산하면 blocked 75/40=1.9, retract 75/40=1.9다. V+P 텍스트에는 1.9, 1.9로 적혀 있다. 구간 병합 기준이 조건마다 다른 것으로 보이지만 원인은 추적하지 않았다(미확인). mock 동작에는 영향이 없다.
4. **V+P 정책의 탐색 범위를 데모 보정량의 분포에서 가져온다**(§3.3). 숨은 오차의 분포 범위(각도 4–10°)보다 좁아서 큰 오차는 원리상 해결할 수 없다. "V+P 성능"이라기보다 "이 휴리스틱의 성능"이다. 숫자를 인용할 때는 반드시 이 점을 같이 적어야 한다.
5. **SPEC과 지시서의 파일명 차이**(테스트, 프롬프트). 구현은 SPEC을 따랐고 기능 범위는 SPEC §11과 같다.

---

## 6. 남은 한계 (검증하지 못한 것)

1. **실제 UniVTAC(arXiv:2602.10093)에서는 실행하지 않았다.** Isaac Sim과 isaaclab이 없다. `UniVTACEnv`는 Isaac이 없으면 명확한 RuntimeError를 내는 것까지만 테스트됐다. 어댑터에 TODO(unverified)로 남은 항목은 모두 미확인이다: depth 단위와 부호, marker (2,63,2)의 축 의미, gripper 방향, 'ee' 액션의 batch 차원, HDF5 레이아웃, instruction 문구 등(A의 open issue). 논문 Table 2와 3의 수치를 재현했는지 여부도 당연히 미확인이다.
2. **실제 LLM으로는 실행하지 않았다.** OpenAI와 Anthropic backend는 SDK와 API 키가 없어서 메시지 직렬화 단위 테스트(mock client)만 거쳤다. 다음 항목은 미확인이다: GPT-6-Astra의 공개 model id(현재 placeholder `gpt-6-astra`), Responses API의 `reasoning.effort="xhigh"` 지원 여부, Anthropic SDK에서 `output_config`와 fallbacks를 지원하는 최소 버전.
3. **평가 프로토콜 차이**: UniVTAC의 `eval_policy.py`는 예외가 난 에피소드를 집계에서 빼고 `seeds.json`으로 seed를 건너뛴다. 이 runner는 논문 §4.1에 따라 모든 종료를 실패로 센다. seed_base도 100000으로, UniVTAC의 1e6과 다르다. 논문 수치와 비교하려면 이 둘을 맞춰야 한다.
4. **mock은 UniVTAC Insert Hole의 ±30° 구멍 기울기를 운동학에 반영하지 않는다**(A의 open issue). 그래서 논문 Fig. 2에 나오는 "~30° 축 탐색" 현상은 재현되지 않고, 4–10° 범위의 숨은 보정만 재현된다.
5. **비용과 지연 시간**: 실제 API 기준 비용(`cost_usd`)은 단가를 알 수 없어 null이다. 논문은 호출당 지연 시간과 에피소드 wall-clock을 보고하지 않는다(미확인). 실제 LLM으로 V+P를 실행하면 실패 에피소드마다 약 100회 호출이 필요하므로, 파일럿은 `--n-episodes 5–10`부터 시작하기를 권한다.
6. **나머지 4개 태스크**(lift_can, grasp_classify, lift_bottle, pull_out_key)는 yaml만 있고 mock 환경이 없다. 이 태스크들의 파이프라인은 E2E로 실행해 보지 않았다.

---

## 7. 재실행 방법

```bash
cd code
bash results/rerun_verification.sh      # pytest → 증류 2회 → run_eval 100 → ablation 100×2 → verify_static → lazy-import 확인
```

오프라인에서 약 3분 걸리고, 네트워크와 API 키가 필요 없다. 단계별로 실행하려면 다음과 같이 한다.

```bash
python3 -m pytest
python3 scripts/distill_task_skill.py --task-config results/configs/insert_hole_mock.yaml \
  --backend-config configs/backend.yaml --condition V+P+T --out results/contexts/insert_hole_mock/V+P+T
python3 scripts/distill_task_skill.py --task-config results/configs/insert_hole_mock.yaml \
  --backend-config configs/backend.yaml --condition V+P   --out results/contexts/insert_hole_mock/V+P
python3 scripts/run_eval.py --eval-config results/configs/eval_mock_100.yaml \
  --backend-config configs/backend.yaml --out results/eval_vpt --no-resume
python3 scripts/run_ablation.py --ablation-config results/configs/ablation_mock_100.yaml \
  --backend-config configs/backend.yaml --out results/ablation
python3 results/verify_static.py
```

실제 LLM으로 바꾸려면 `--backend-config configs/backend_openai.yaml` 또는 `configs/backend_anthropic.yaml`을 쓰고(SDK 설치와 API 키 필요), 프롬프트나 backend를 바꿀 때마다 `results/contexts/`를 지워 다시 증류해야 한다. 실험 추적은 `--logging wandb` 또는 `--logging mlflow`로 켤 수 있다(run_eval만 지원하며 SDK가 설치되어 있어야 한다. SDK가 없으면 경고를 내고 none으로 대체된다). UniVTAC에서 실행하는 절차는 `code/README.md`의 1안과 2안을 참고하라. 두 안 모두 아직 검증하지 않았다.
