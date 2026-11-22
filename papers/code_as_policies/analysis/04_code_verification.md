# 04. 논문 ↔ 공식 코드 정적 대조와 구현 가능성

공식 저장소는 `google-research/google-research`의 `code_as_policies/` 폴더만 희소 얕은 clone(`--depth 1 --filter=blob:none --sparse`)으로 `repo/`에 받았습니다. 폴더의 마지막 커밋은 `8c116f61`(2024-02-16, "lmpc demo")입니다. 클론한 모노레포의 HEAD는 `87e25a5`(2026-11-21)입니다.

**실행, import, 설치, API 호출을 하지 않았습니다.** 읽은 것은 노트북 8개의 코드 셀과 저장된 출력입니다. 노트북 저장 출력은 실행 기록이므로 재실행 결과로 보지 않고, 표 값과 비교하는 기록으로만 썼습니다.

## 1. 저장소 구성

| 파일 | 내용 | 논문과 연결 |
|---|---|---|
| `LMP Examples.ipynb` | 방법 절의 예제(§III-A–D)와 LMP 세션 | 설명 예제 |
| `Experiment_ Robot Code-Gen Benchmark.ipynb` | RoboCodeGen 37문제, 구성 16개 셀(모델 4 × 계층·평면 2 × 프롬프트 2) | 표 I, Fig. 4 |
| `Experiment_ HumanEval Benchmark.ipynb` | HumanEval 164문제(공식 평가기 호출) | 표 II, V |
| `Experiment_ Reasoning with Code vs Natural Language.ipynb` | 객체·위치 선택(28·23문항) | 표 IV |
| `Experiment_ Reactive Controllers on Toy Tasks.ipynb` | CartPole과 임피던스 제어(부록 F) | 부록 F, 정적 확인만 |
| `Interactive_Demo.ipynb` | PyBullet UR5e와 Robotiq 2F85 시뮬 환경, LMP 래퍼 | 표 III, VI의 환경(일부) |
| `lmpc_demo.ipynb`, `coc_demo.ipynb` | 후속 데모 | 논문 밖 |

- 저장소에 `.py` 파일은 없습니다. 모든 코드는 노트북 안에 있습니다.
- 라이선스는 모노레포의 Apache-2.0입니다. `Interactive_Demo.ipynb`의 첫 셀에 "Copyright 2022 Google LLC. SPDX-License-Identifier: Apache-2.0"가 적혀 있습니다.

## 2. 대조표

| 논문 항목 | 코드 위치 | 판정 |
|---|---|---|
| §III-A 힌트(import)와 예제(주석 → 코드)로 프롬프트 구성 | `prompt_f_gen`, `prompt_f_gen_flat`(프롬프트 정의부) | 일치 |
| §III 생성 코드 실행: import·`__`·exec·eval 차단 후 `exec(code, globals, locals)` | `exec_safe()`(Robot 노트북 셀 6, HumanEval 노트북 셀 12) | **D1** |
| §III 계층 함수 생성: AST에서 정의되지 않은 함수를 찾아 재귀 생성 | `lmp_fgen(..., recurse=True)`, `ast.parse`, `astunparse` | 일치, 깊이 상한은 미확인 (D5) |
| §IV-A 모델: code-davinci-002 temperature 0 | `default_query_kwargs`: engine `code-davinci-002`, temperature 0 | 일치 (max_tokens는 논문에 없음, D5) |
| §IV-A 문제 37개, 함수 헤더에 docstring과 타입 힌트 없음 | 셀 11: `n problems: 37` | 일치 |
| §IV-A 평가는 사람이 쓴 단위 테스트 | `eval_problems`가 문제별 `success`를 기록(테스트 정의부는 세부 확인 생략) | 일치(부분 확인) |
| 표 I 값 | 저장 출력 `Success rate: k/37` | **D2, D3, D4** |
| 표 II·V HumanEval 164문제, greedy와 샘플 | `read_problems()`(164), `temperature 0.8` 샘플, `human_eval.evaluate_functional_correctness` | 구조는 일치, 저장된 pass@k 없음 (D6) |
| 표 IV 객체 28, 위치 23 | 셀 9, 22: `len` 28과 23 | n은 일치, 논문 미기재 (D7) |
| §IV-D 시뮬 환경: UR5e + Robotiq 2F85, 10색 블록·10색 그릇 | `Interactive_Demo.ipynb`의 PyBullet `PickPlaceEnv` | 환경 일치, 평가 루프 없음 (D8) |
| §IV-D 기준선: CLIPort(30k 시연), 종료 oracle, NL 계획기 | 저장소에 없음 | **미공개** (D9) |
| §IV-D 과제 정의 14개(seen/unseen 속성과 지시문) | 부록 K 목록(논문 본문)과 프롬프트 링크 | 텍스트 일치, 평가 샘플링 코드 없음 |
| §IV-B 궤적 그리기(UR5e, MDETR) | 없음 | **미공개** (D10) |
| §IV-C 테이블탑 실로봇(UR5e 흡입, MDETR) | 없음 | **미공개** (D10) |
| §IV-E 모바일 조작(Everyday Robots, ViLD) | 없음 | **미공개** (D10) |
| 부록 F 카트폴 제어기, 임피던스 제어 | 노트북 `Reactive Controllers` | 정적으로 존재 확인, 출력 미검토 |

## 3. 차이 목록 (D1–D11)

영향은 표 값이나 주장에 미치는 크기로 나눴습니다. 큼은 주장의 근거를 바꿀 수 있고, 중간은 비교의 공정성에 영향을 주며, 작음은 기록만 필요합니다.

**D1. 실행 전 검사의 구현 (작음~중간, 보안 주장).** 논문은 "import, `__` 변수, exec·eval 호출을 확인한 뒤 exec로 실행한다"고 씁니다(§III). 노트북의 `exec_safe()`는 코드 문자열에 `import`나 `__`가 들어 있으면 `assert`로 멈추고, 실행 시에는 `exec`와 `eval`을 빈 함수로 바꿉니다. 검사는 문자열 포함 여부이므로 주석과 문자열 리터럴도 걸러집니다. 실행 환경은 `globals`만 바꾼 `exec`라서 내장 함수는 그대로 쓸 수 있습니다. 따라서 이 검사는 샌드박스가 아니라 실수를 막는 장치입니다. 실행하지 않고 읽은 결과입니다.

**D2. GPT-3 6.7B 행에 구문 수정 단계가 있음 (중간, 표 I).** 노트북의 `text-curie-001` 네 셀(`bug_fix=True`)은 생성 코드를 `code-davinci-edit-001`에 넣어 구문을 고칩니다. 지시문은 "Fix syntax errors. Keep same inputs and outputs. Only small changes. No comments."입니다. 논문 본문과 부록 D에는 이 단계가 없습니다. 같은 행의 다른 구성도 모두 이 단계를 거쳤으므로 행 안의 비교는 같은 조건이지만, 다른 행과는 다른 조건입니다. `03` §1.3.

**D3. 코드-davinci-002 계층 열 (중간, 표 I).** 노트북은 36/37 = 97.3%, 표 I은 95(35/37)입니다. 한 문제 차이입니다. `03` §1.2.

**D4. cushman 평면 열의 구성 (중간, 표 I).** 표 I은 54인데, 노트북의 평면 생성·평면 프롬프트는 59.5(22/37)입니다. 54는 평면 생성·계층 프롬프트 조합(20/37)입니다. 표 I의 Flat 행이 모델에 따라 다른 조합을 가리킵니다. 결과적으로 cushman에서 계층 생성의 이득(+3)은 같은 기준에서 -2(57 대 59)로 바뀝니다.

**D5. 보고되지 않은 생성 설정 (작음).** 노트북의 기본값은 `max_tokens = 512`, `frequency_penalty = 0`, 정지 토큰 `# define function:`과 `# example:`입니다. 논문은 max_tokens와 정지 토큰을 적지 않았습니다. 재귀 생성의 깊이 상한도 적지 않았습니다. 읽은 `lmp_fgen` 본문에서는 상한을 찾지 못했지만, 전체 함수를 끝까지 읽지는 않았습니다.

**D6. HumanEval의 pass@k 원자료 없음 (작음, 검증 불가).** 노트북은 `results` JSONL을 쓰고 `evaluate_functional_correctness.py`를 셸로 호출합니다. 하지만 저장된 출력에는 최종 pass@1과 pass@k가 없습니다. 따라서 표 II와 V의 P@N은 이 저장소로 확인할 수 없습니다. 샘플 수 N도 논문에 없습니다.

**D7. 표 IV의 문항 수와 Total 계산 (작음).** 노트북은 객체 선택 28, 위치 선택 23문항을 씁니다(논문 미기재). 표 I의 Total 58은 CoT의 과제 행 평균이고, 항목 합계(58.8)가 아닙니다(`03` §3).

**D8. 시뮬레이션 환경은 있으나 평가 루프는 없음 (중간).** `Interactive_Demo.ipynb`는 PyBullet 환경(`PickPlaceEnv`, UR5e, Robotiq 2F85)과 LMP 래퍼를 제공합니다. `get_obj_names()`는 장면의 객체 목록 전체를 돌려주고 `is_obj_visible()`은 목록에 있으면 참이므로, 인지는 시뮬레이터 정답입니다. 논문이 말하는 "scripted object detector"와 맞습니다. 그러나 표 III·VI의 과제 샘플링(속성과 초기 장면), 50 시행 루프, 성공 판정은 없습니다. 노트북에 있는 `range(50)`은 파지 대기 루프입니다.

**D9. 기준선이 공개되지 않음 (큼, 표 III·VI).** CLIPort 학습 코드와 30k 시연, oracle 종료 변형, NL 계획기 프롬프트 실행 코드가 없습니다. NL 계획기 프롬프트는 프로젝트 페이지에 `.txt`로 링크되어 있지만, 실행과 평가 코드는 없습니다. 따라서 표 III의 기준선은 이 저장소로 재현할 수 없습니다.

**D10. 실로봇과 모바일 코드가 없음 (큼, §IV-B, C, E).** UR5e 그리기, UR5e 테이블탑 실로봇, Everyday Robots 모바일 조작의 API와 MDETR·ViLD 호출, 흡입 그리퍼 드라이버는 없습니다. 프롬프트 파일(`draw_ui.txt`, `mobile_ui.txt` 등)은 프로젝트 페이지에 링크가 있습니다. 논문도 이 실험들은 정량 결과가 없으므로, 공개되지 않은 것이 재현 가능성에 주는 손실은 "정성적 시연의 재확인 불가"입니다.

**D11. 노트북 출력과 실행 환경 (작음).** 저장된 출력은 `python3.7`의 `/usr/local/lib/python3.7`, Colab, `shapely` 경고 등을 담고 있습니다. 결과는 당시 환경에서 나온 기록이며, 현재 환경에서 같은 값이 나온다는 보장은 없습니다. 그래서 표 I과의 대조는 값의 기록으로만 씁니다.

## 4. 구현 가능성

판정 기준은 이 맥(Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)입니다. 새 설치와 1 GB 이상의 가중치 다운로드는 승인이 필요하다고 보았습니다. 이 분석에서는 아무것도 설치하지 않았습니다.

**논문 방법의 핵심(프롬프트로 코드를 생성하고 실행하는 것)은 이 맥에서 구현할 수 있습니다.** 다만 논문 수치를 그대로 얻을 수는 없습니다.

- **가능(설치와 모델 준비 필요):** RoboCodeGen 37문제를 오픈 가중치 코드 LLM으로 다시 측정합니다. 필요한 것은 `numpy`, `shapely`, `astunparse`, 그리고 코드 모델 가중치입니다. 모델 크기가 1–7B이면 MPS에서 37문제 × 4구성 × 2프롬프트를 돌릴 수 있습니다. 노력은 1–2일(추정)입니다. 결과는 원래 Codex 값과 다르므로, "계층 생성이 평면 생성보다 낫다"는 주장만 독립적으로 시험할 수 있습니다.
- **조건부(설치와 장시간 실행 필요):** PyBullet 시뮬(`pybullet` 설치)에서 CaP 방식으로 과제 14개 × 50 시행을 돌립니다. LLM 호출은 시행당 여러 번이므로 총 수천 번입니다. 로컬 7B 모델이면 수 시간에서 하루가 걸릴 것으로 추정합니다(측정하지 않음). 기준선은 NL 계획기만 가능하고, CLIPort는 GPU와 30k 시연이 필요하므로 불가입니다.
- **불가:** 원래 실험의 `code-davinci-002`와 `text-davinci-002`는 OpenAI API 모델입니다. 제가 알기로 Codex 계열 모델은 2024년에 종료되었지만, 이 분석에서 확인하지 않았습니다(API 호출 금지). 실로봇(UR5e, Everyday Robots), CLIPort 학습은 맥에서 불가능하거나 하드웨어가 필요합니다.
- **노력(추정):** 오픈 모델 재측정 1–2일, 시뮬 축소판 1–2주. 공식 노트북을 그대로 실행하려면 OpenAI API 키와 비용이 필요하며, 모델 종료 여부를 먼저 확인해야 합니다.

## 5. 저자에게 물을 것

1. 표 I의 코드-davinci-002 계층 값(95)과 노트북 저장 출력(36/37)의 차이는 어느 실행을 가리키나요?
2. cushman 평면 열(54)은 평면 생성과 계층 프롬프트 조합인가요, 평면과 평면 조합(59.5)인가요?
3. GPT-3 6.7B 열에 구문 수정 단계(`code-davinci-edit-001`)가 들어간 이유와 그 영향은 무엇인가요?
4. "164문제 중 6.5%"와 "44% 대 56%"는 어떤 문제 집합인가요? 두 수치는 정수 개수와 맞지 않습니다.
5. 표 III의 UA-UI 공간 기하 CLIPort 값 0.01은 표 VI의 1.3과 어떻게 다른가요?
6. 표 VI의 Total 행에서 NL 계획기는 장기 과제만, CaP는 전체를 평균한 이유는 무엇인가요?
7. 시뮬 평가 루프(속성 샘플링, 50 시행, 성공 판정)와 CLIPort 기준선 코드를 공개할 계획이 있나요?

## 6. 재계산 스크립트와의 관계

- 표 값 대조는 `analysis/recompute.py`로 했고, 결과는 `analysis/recompute_results.json`에 있습니다.
- 노트북 출력의 k/37 값은 `recompute.py`의 `NB` 사전에 직접 적었습니다. 노트북 셀 번호를 함께 적어 두었습니다.
