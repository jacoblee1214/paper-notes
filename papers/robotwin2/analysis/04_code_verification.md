# 04. 코드 대조: 논문 ↔ 공식 저장소 (RoboTwin-Platform/RoboTwin)

**방법.** 얕은 clone (`repo/`, 커밋 `ea8b211`, 2026-09-24)을 **정적으로만** 읽었습니다. 실행, import, 설치, 에셋 다운로드는 하지 않았습니다. 저장소 내용은 신뢰할 수 없는 데이터로 취급했습니다. 설정 파일, 평가 스크립트, 환경 기반 클래스, 코드 생성 루프를 읽었습니다.

**읽은 파일.** `env_cfg/task_config/demo_clean.yml`, `demo_randomized.yml`, `_config_template.yml`, `_eval_step_limit.yml`, `envs/_base_task.py` (랜덤화와 clutter, 높이), `scripts/collect_data.py` (데이터 수집 루프), `scripts/eval_policy_xpolicylab.py` (평가 시드 루프와 `expert_check`), `scripts/_task_config_template.json`, `scripts/requirements.txt`, `scripts/_install.sh`, `code_gen/task_generation_mm.py`, `code_gen/test_gen_code.py`, `code_gen/run_code.py`, `code_gen/observation_agent.py`, `README.md`.

**기호.** D1–Dn은 논문과 코드의 차이 (내부 점검 라벨). 판정은 "일치", "불일치", "미확인"입니다. "미확인"은 에셋이나 학습 코드가 없어서 정적으로 확인할 수 없는 항목입니다.

## 1. 한눈에 보기

| 항목 | 논문 | 코드 | 판정 |
|---|---|---|---|
| 탁상 높이 랜덤화 | "plausible range에서 균등" (§2.2), "최대 3 cm" (부록 C) | 높이 변화는 `uniform(−h, 0)`, 한쪽으로만 낮춤. Hard 설정 h = 0.03 | **부분 일치** (방향 차이) (D1) |
| 평가 시드 | 과제당 100회 롤아웃 (§4.5) | `test_num` 기본값 100 | 일치 (D8) |
| 학습 데모 수 | 과제당 clean 50 (§4.5) | `episode_num` 50 | 일치 (D9) |
| 평가 시 전문가 검사 | 언급 없음 | 기본값으로 켜짐. 전문가가 실패하거나 불안정한 시드는 **건너뜀** | **불일치 / 논문 누락** (D7) |
| Easy/Hard 언어 분할 | "평가 조건만 다르다" (부록 K) | Clean `seen`, Hard `unseen` | **불일치** (D6) |
| 데이터 수집 | "100,000 궤적" | 성공한 에피소드만 저장 | 일치 (성공 필터, 논문에 없음) (D9) |
| 코드 생성 루프 | 10회 실행, 5회 개선, 성공 임계 0.5 (§2.1) | `run()` 기본 10회, `generate_num` 5, 임계 0.5 | 일치 (D10) |
| 코드 생성 모델 | DeepSeek-V3, moonshot-v1-32k-vision (부록 G) | `gpt="deepseek"`, `moonshot-v1-32k-vision-preview` | 일치 (D11) |
| 조명 무작위 | 색 온도, 종류, 개수, 위치 (§2.2) | `random_light`, 2% 강한 무작위 | 일치 (강도 수치는 논문에 없음) (D3) |
| 질감 수 11,000 | §2.2 | 에셋 zip 미다운로드 | 미확인 (D4) |
| 객체 수 731 | §3.1 | `description/objects_description` 항목 117개, 객체 에셋은 zip | 미확인 (D16) |
| 실세계 카메라 섭동 ≤1 cm | §4.4 | 저장소 설정의 `random_head_camera_dis` 0, 실세계 설정 없음 | 미확인 (D13) |
| 라이선스 | 명시 없음 | MIT | - |

## 2. 랜덤화 (§2.2 대 코드)

**D1. 탁상 높이 (방향).**
- 논문: "uniformly randomize table height within a plausible range" (§2.2). 부록 C: "table height variation (up to 3 cm)".
- 코드: `envs/_base_task.py:112`, `self.table_z_bias = np.random.uniform(low=-self.random_table_height, high=0) + table_height_bias`. 높이 변화는 **0에서 −h까지**로 한쪽으로만 간다. Hard 설정 `demo_randomized.yml`의 `random_table_height: 0.03`과 맞는다.
- 판정: 크기는 부록 C와 맞지만, 본문의 "균등 분포 (양쪽)"라는 표현과는 다르다. 높이를 올리는 변화는 코드에 없다.

**D2. Clutter.**
- 논문: 무관 물체를 배치하고, 비슷한 물체는 제외한다고 적는다 (§2.2).
- 코드: `get_cluttered_table(cluttered_numbers=10, ...)`는 기본 10개 물체를 `self.obj_names`에서 무작위로 고르고, 충돌과 `prohibited_area`를 검사한다 (`envs/_base_task.py:318–380`). 비슷한 물체 제외 규칙은 이 함수 안에서 보지 못했다.
- 판정: 10개 개수와 충돌 검사는 일치. **시각·의미 유사도 제외는 정적 읽기에서 확인하지 못했다** (미확인).

**D3. 조명.**
- 논문: 색 온도, 광원 종류, 개수, 위치 무작위 (§2.2).
- 코드: `random_light: true` 및 `crazy_random_light_rate: 0.02` (`demo_randomized.yml`). 강한 조명 변화는 2%의 확률로 적용된다 (`envs/_base_task.py:84, 425`).
- 판정: 일치. 2%라는 강도 수치는 논문에 없다.

**D4. 배경 질감 (11,000).**
- 논문: 표면 설명 1,000개 → SD v2 20장 → 사람 검수 → 11,000장 (§2.2).
- 코드: `scripts/_download_assets.sh`가 `background_texture.zip`을 받는다 (zip 미포함). `create_table_and_wall`이 `random_background`일 때 질감을 쓴다 (`envs/_base_task.py:277–292`). 생성 과정과 개수는 저장소에 없다.
- 판정: **미확인.** 개수 확인에는 에셋 다운로드가 필요하다.
- 참고: `clean_background_rate`는 `np.random.rand() <= rate`일 때 깨끗한 배경을 쓴다. Hard 설정의 `clean_background_rate: 0.02`는 **98%가 질감 배경**이라는 뜻이다. 논문에는 이 비율이 없다.

**D5. Clean 설정.** `demo_clean.yml`은 모든 랜덤화 값을 끈다. 일치.

## 3. 평가 프로토콜 (§4.5 대 `scripts/eval_policy_xpolicylab.py`)

**D6. Easy/Hard의 언어 분할 차이.**
- 논문: Easy와 Hard는 "평가 조건만 다르다" (부록 K). 학습은 같다.
- 코드: `demo_clean.yml`은 `eval_instruction: seen`, `demo_randomized.yml`은 `eval_instruction: unseen`이다. `resolve_eval_instruction`이 이 값을 읽어 평가 언어를 정한다 (`eval_policy_xpolicylab.py:225–234`).
- 판정: **불일치.** Hard에서는 평가 언어도 바뀐다. 논문이 쓴 Easy/Hard 수치가 이 두 설정 파일로 만들어졌는지는 확인할 수 없다. 만약 그렇다면 Hard 낙폭 (RDT −20.8, Pi0 −30.1)에는 장면 효과와 언어 효과가 섞여 있다.

**D7. 전문가 검사로 시드를 건너뜀 (가장 큰 발견).**
- 논문: 평가 절차에서 시드 선택을 언급하지 않는다.
- 코드: `expert_check`는 기본값 **True**다 (`eval_policy_xpolicylab.py:534`). 시드마다 전문가 코드를 먼저 실행한다. `plan_success`와 `check_success()`가 모두 참이어야 평가 에피소드가 된다. 아니면 `expert_failed`로 **건너뛴다**. 플래너가 불안정하면 `unstable`, 예외가 나면 `expert_error`로 건너뛴다 (`:600–661`).
- 시드 상한은 `max(test_num × 50, worker_num × 20)`이다. `test_num = 100`이면 5,000 시드까지 본다 (`:370–371`).
- 결과 파일에 `Skipped Seeds`가 기록된다 (`:506`).
- 판정: **논문에 없는 선택 규칙.** 보고된 성공률의 분모는 전문가가 성공한 장면이다. **건너뛴 시드 수와 Easy/Hard별 비율은 논문에 없다.** Hard 설정에서 전문가 실패가 더 많다면, Easy와 Hard는 서로 다른 장면 집합에서 평가된 것이다. 이 효과는 측정하지 않았다.
- 참고: 이 규칙이 논문의 Table 10 (부록 K)을 만든 경로인지는 확인할 수 없다. 논문의 평가 경로가 `eval_policy_multitask.py`일 수도 있다. 두 스크립트를 모두 읽었고 `eval_policy_multitask.py`에도 `test_num`과 `eval_instruction` 처리가 있다 (`:591, 768–777`). 그 파일의 `expert_check` 기본값은 이 분석에서 확인하지 않았다.

**D8. 롤아웃 수.** 기본 `test_num = 100`이다 (`eval_policy_xpolicylab.py`, `eval_policy_multitask.py:173, 768`). 논문 §4.5 "100 rollouts"와 일치.

**D9. 학습 데모와 데이터 수집.**
- 학습 데모: `episode_num: 50`. 논문 "50 clean expert demonstrations"와 일치.
- 데이터 수집: `collect_data.py`의 루프는 `suc_num < episode_num`인 동안 시드를 늘리며 **성공한 에피소드만 저장**한다. 실패 궤적은 저장되지 않는다. 논문의 "10만+ 궤적"은 성공 필터를 거친 전문가 궤적이다. 이 사실은 논문에 없다.

**D15. 스텝 제한.** `env_cfg/task_config/_eval_step_limit.yml`은 과제별 제한을 400–1500으로 둔다 (`open_microwave` 1500, `hanging_mug` 900 등). 목록에 없는 과제는 1000이다 (`_base_task.py:146–149`). 논문에는 스텝 제한이 없다.

## 4. 코드 생성 (§2.1, 부록 G 대 `code_gen/`)

**D10. 반복 구조.**
- `task_generation_mm.py`: `generate_num = 5`, `success_threshold = 0.5`, 각 반복에서 `run(task, args)`. `run`은 `from test_gen_code import *`로 가져오며 기본 `check_num = 10`이다 (`test_gen_code.py:231`). 논문 "10회 실행"과 일치.
- 다른 경로 `run_code.py`는 `check_num = 50`을 쓴다 (`:33`). 두 경로가 함께 있다. 논문 실험이 어느 경로였는지는 모른다.
- 5회는 **총 반복 수**이고 "연속" 개선이 아니다 (`for id in range(generate_num)`). 본문의 "five consecutive refinements"와 다르다.
- 최고 코드 저장은 같은 `run()`의 `success_rate`로 한다 (`task_generation.py:160–190`). 보고된 ASR은 같은 실행으로 계산되는지 논문에 없다. 선택 편향 가능성 (`01` §4.1).

**D11. 모델.** `gpt_agent.py`와 `task_generation*.py`의 `gpt="deepseek"`, `observation_agent.py`의 `base_url="https://api.moonshot.cn/v1"`, `model="moonshot-v1-32k-vision-preview"`. 논문 부록 G와 일치. 이 분석은 API를 호출하지 않았다.

**D12. 실패 분석 (VLM 관찰자).** 부록 G.4의 130개 시퀀스 데이터셋은 저장소에 없다. 미확인.

## 5. 설치와 실행 요구사항 (맥 판정의 근거, 

- `scripts/requirements.txt`: torch 2.4.1, numpy 1.26.4, sapien 3.0.0b1, mplib 0.2.1, open3d 0.18.0, scipy 1.10.1, gymnasium 0.29.1 등. `numpy 1.x ABI`를 명시한다.
- `scripts/_install.sh:56–59`: `curobo` v0.7.8을 소스에서 빌드한다 (`git clone --branch v0.7.8 https://github.com/NVlabs/curobo.git`). cuRobo는 CUDA 확장을 전제로 하는 것으로 알려져 있으나 이 분석에서 빌드 스크립트는 읽지 않았다. 확인하지 못함.
- 논문 부록 E도 "GPU-accelerated motion planner" cuRobo를 쓴다고 적는다.
- 따라서 **맥 (CUDA 없음)에서 전문가 실행과 평가를 재현하는 것은 확인되지 않은 채로 어렵다**.

## 6. 논문과 저장소의 기타 차이

**D13. 실세계 카메라 섭동.** 논문 §4.4는 시뮬레이션 카메라 자세에 1 cm 이하의 3D 섭동을 준다고 쓴다. 저장소의 시뮬레이션 설정은 `random_head_camera_dis: 0`이다 (`camera.py:100`에 기능은 있음). 실세계 학습 설정은 저장소에 없다. **미확인.**

**D14. 학회 표기.** 저장소 README: "RoboTwin 2.0 (ICML 2026)". 논문 arXiv 페이지와 프로젝트 페이지: 학회 표기 없음. PMLR v306 공식 목록과 icml.cc 포스터 페이지가 확인되었다 (`README` 참고).

**D16. 객체 라이브러리.** 논문: 731 객체, 147 범주. 저장소: `description/objects_description`에 117개 항목이 있다 (범주인지 객체인지 정적으로 구분하지 못함). 객체 에셋은 zip으로 받는다. **미확인.**

**D17. 과제 목록.** `env_cfg/eval/all_tasks.yml`의 과제 50개와 논문 표 10의 행 50개를 이름 정규화 후 대조했다. 차이 없음 (일치).

## 7. 구현 가능성 (정적 읽기 기준)

- 평가 파이프라인은 공개되어 있다 (MIT). 시뮬레이터와 정책 서빙은 XPolicyLab 기반이다.
- 정책 학습 코드는 이 저장소에 없고 (RDT, Pi0 등은 외부 저장소), 학습 설정은 부록 D에 있다.
- 맥에서 재현은 

## 8. 저자에게 물을 질문

1. Easy와 Hard 평가에서 `expert_check`를 켰는가? 건너뛴 시드 수는 얼마인가 (설정별)?
2. Hard 평가의 언어 분할은 `unseen`인가? 아니면 `seen`인가? 부록 K의 "평가 조건만 다르다"와 어떻게 맞춰야 하는가?
3. 실세계 시행 수 (셀당 50?)와 시드 분산은 얼마인가?
4. 초록의 367%와 228%는 한 셀 (정돈 + 미지 배경, 4과제)에서 나온 값인가?
5. 탁상 높이 변화는 올리는 방향도 있는가, 아니면 낮추기만 하는가?
6. 코드 생성 ASR은 최고 코드를 고른 같은 10회 실행으로 계산했는가?
