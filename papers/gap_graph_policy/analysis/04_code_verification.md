# 04. 공식 코드 대조와 구현 가능성

## 0. 대상과 방법

- 저장소: `graph-robots/graph-as-policy`(공식, 프로젝트 페이지와 논문 첫 쪽의 링크). 커밋 `e572392cce48475d1ca4ba0f564d306e437a69cc`(2026-11-11, "README, docs: point to the released self-learning pipeline"). 얕은 복제를 `repo/`에 두었고 서브모듈(`third_party/*`)은 가져오지 않았습니다. 파이썬 파일 약 4만 줄입니다.
- **정적 읽기만 했습니다.** 코드를 실행, import, 설치하지 않았습니다. 가중치, 데이터, 서브모듈은 받지 않았습니다. 자매 저장소(open-robot-skills, agentic-self-learning)는 복제하지 않고 공개 파일 목록과 README, `asl/loop.py` 한 파일만 GitHub API로 읽었습니다.
- 읽은 파일: `README.md`, `NOTICE.md`, `LICENSE`, `pyproject.toml`, `gap-core/pyproject.toml`, `docs/design.md`, `docs/source/{getting-started/overview,developers/roadmap,developers/architecture,running/environments,authoring/generation,benchmarks/benchmarking}.md`, `gap/agent/{multi_agent,config,llm,parallel}.py`, `gap/runtime/{executor,validate}.py`, `gap/benchmark/config.py`, `gap/env_config.py`, `gap/envs/libero_env.py`, `gap/connector/core.py`(일부), `examples/benchmark/*.yaml`, `examples/grocery_packing/`, `gap-core/src/gap_core/rpc/server.py`.
- 논문은 v3, 저장소 README 머리글은 "BETA CODE RELEASE, July 1, 2026"입니다. 논문 v3(2026-11-04)보다 이른 베타에서 출발한 코드이고, 이후 커밋(2026-11-04의 인용 갱신, 2026-11-11의 자기 학습 저장소 안내)이 있습니다.

## 1. 설치 요구사항과 라이선스

- README 요구사항: **NVIDIA RTX 4090(24 GB VRAM 이상), Linux, EGL**, LLM API 키(OpenRouter 또는 Vertex), uv. 처음 실행 때 가중치 약 3.5 GB를 받고, SAM3 가중치는 접근 승인이 필요한 HuggingFace 저장소입니다(README 55–60행). LIBERO 퀵스타트만 약 10 GB VRAM이 필요합니다(`docs/source/getting-started/installation.md`). cuRobo는 선택이지만 벤치마크 계열(`grocery` extra)이 씁니다.
- `gap skills install --all`은 번들마다 별도 가상환경(sam3, cuRobo, vlm 등)을 만듭니다. 다운로드, 설치 모두 이 분석에서는 하지 않았습니다.
- **라이선스.** `LICENSE`(Apache 2.0 전문), `pyproject.toml`(`Apache-2.0`), README 배지와 GitHub API는 Apache-2.0입니다. `NOTICE.md`는 "gap and open-robot-skills are released under the MIT License"라고 적고, `gap-core/pyproject.toml`은 MIT입니다(D14). 서브모듈 `robots_realtime`은 고정 리비전에 라이선스 파일이 없다고 `NOTICE.md`가 적습니다. 가중치는 각자의 라이선스를 따릅니다(SAM3는 Meta 맞춤 라이선스, cuRobo는 리비전에 따라 NVIDIA 소스 코드 라이선스).

## 2. 논문 ↔ 코드 대조 (D 항목)

판정: **일치**는 논문 서술과 코드가 같음, **차이**는 다름 또는 논문 본문에 없는 동작, **없음**은 이 저장소에 해당 코드가 없음, **미확인**은 저장소만으로 판단할 수 없음.

| 번호 | 항목 | 논문 | 코드·문서 | 판정 |
|---|---|---|---|---|
| D1 | 릴리스 범위 | 벤치마크 8개(II-a/b, III, IV, V 포함), 모션·파지에 GraspGen·M2T2 사용(§4.1, §5.5) | `docs/source/getting-started/overview.md`의 "Scope of v1"은 "language → verified pick-and-place graphs on LIBERO and real Franka/UR"입니다. `docs/source/developers/roadmap.md` 30–47행은 execution-feedback 그래프 수리, 학습형 파지 계획기(graspgen/m2t2), 양팔 조작, Isaac 기반 리허설, 비 pick-place 영역(스토브 노브, USB 삽입, 장기 과제)을 "cut for scope"로 열거합니다. `docs/design.md` 370–375행도 같습니다 | **없음** (Make Popcorn, USB-C, Wash Crates 그래프와 GraspGen은 v1 범위 밖) |
| D2 | 실로봇·예제 | 실로봇 Franka 3과제 + UR5 케이블(§5.7) | 예제는 LIBERO 퀵스타트, 식료품 포장, 식료품 주문 이행, 벤치마크, `steered_policy`(VLA 위임), `real_franka_pick_place`, `cable_ur`입니다. `cable_ur`는 "perception-only, motion structurally impossible(읽기 전용 RTDE)"로 설명됩니다(`examples/README.md`). ROS 번역 스킬, 손 설계 케이블 그래프, 양팔 그래프는 없습니다 | **없음** (논문 §5.7의 ROS 연동과 실행 경로는 저장소에서 재구성할 수 없음) |
| D3 | 자기 학습 루프 | 알고리즘 1: N개 병렬 리허설 → 실패 분석 → LLM Graph Update, 갱신은 새 인스턴스 100개에서 성공률·주기 시간이 떨어지면 거부, 정체까지 반복(§4.2) | 이 저장소에는 없습니다(`gap/agent/config.py` 모듈 docstring "Deleted relative to the source: rehearsal, scene_spec, ..."). README 109행 이하와 머리글이 별도 저장소 `agentic-self-learning`을 안내합니다. 그 `asl/loop.py`는 헤드리스 Claude Code 또는 Codex 세션 하나를 이어서 쓰며 훈련 배치 피드백(`FEEDBACK.md`)을 주고, 보류 에피소드로 주기적으로 평가하고(에이전트에게 숨김), 완벽 배치 연속 달성·반복 한도·비용 한도에서 멈추고, 훈련 점수가 가장 좋은 반복을 전달합니다(6–13행, 281–290행). 거부는 검증기·허용 목록 위반에만 쓰입니다(125–139행). 시뮬레이터는 RoboSimStudio | **차이** (논문 알고리즘 1과 구조가 다른 후속 루프. 채택 규칙 "100개 새 인스턴스에서 하락이면 거부"는 찾지 못함) |
| D4 | 다중 에이전트 파이프라인 | 오케스트레이션 에이전트가 구간을 나누고 스킬 에이전트가 부분 그래프를 만든 뒤 이어 붙임(§1, §4, 부록 F) | `gap/agent/multi_agent.py` 1–16행: coordinator(최상위 `WorkflowSpec`) → 부분 그래프마다 `subgraph_agent` → `checkpoint_agent`(전체 한 번, 기본 켜짐) → 조립 → 검증과 LLM 스크립트 수정 루프(`max_validation_retries` 2). `gap/agent/config.py` 94–127행: 에이전트별 재시도 2, 그래프 재생성 최대 2 | **일치** (구조), 논문 본문에 없는 단계(checkpoint agent, 수정·재생성 루프) 있음 |
| D5 | LLM·온도 | GaP와 CaP-X에 Gemini-3.1-Flash-Lite, 온도 0.1(§5.2) | 기본 모델은 `gemini-3.1-flash-lite-preview`(`gap/agent/llm.py` 59–62행)로 같습니다. 온도 기본값은 `LlmConfig.temperature = 0.7`(123행)이고 부분 그래프 에이전트는 0.3(`config.py` 99행)입니다. 릴리스 게이트 설정 `examples/benchmark/grocery_acceptance.yaml`은 `gemini-3.1-pro-preview`를 지정합니다. 디스크 캐시를 켜면 온도 > 0에서 반복 생성이 첫 응답을 되풀이합니다(`docs/source/authoring/generation.md` 325–329행) | **차이** (0.1은 설정으로 명시해야 함, 게이트 설정은 다른 모델) |
| D6 | 평가 프로토콜 | 표 1 칸당 100 task instance(§5.2) | 위치 변동 격자(`examples/benchmark/posvar.yaml`)는 과제 10개 × 시드 50개(초기 상태 50개가 YAML에 박힘)로 칸당 500회이고, 인수 게이트(`grocery_acceptance.yaml`)도 10 × 50 = 500회, 임계값 0.90입니다. `regenerate_code_per_trial: false`(기본)라 과제당 그래프 하나를 50회 실행합니다(`config.py` 38행). 게이트 설정은 과제별로 손으로 쓴 `objects`(대상 문구, 기대 라벨, 형상 힌트)를 코드 생성에 줍니다 | **차이** (칸 구성 방식이 다름. 논문의 100 instance가 어떻게 뽑혔는지 저장소로는 알 수 없음) |
| D7 | 포장(Pack) 성공 지표 | 6개 물건 중 6번 파지 시도 후 바구니에 들어간 개수(§5.1) | `gap/benchmark/config.py` 41–43행: 포장 성공은 "teleport-on-In monotonic delivery", 바구니에 들어간 물건은 환경이 치워 버립니다(`gap/envs/libero_env.py` 833–850행의 `pack_all_into`, `completion_rate`). 예제 루프는 VLM 완료 질의 또는 `_MAX_PASSES = 30`으로 끝납니다(`examples/grocery_packing/.../route_next_object.py` 55행) | **차이** (6번 시도 제한과 같은 규칙은 찾지 못함) |
| D8 | 스킬 라이브러리 | MORSL 초기 51개: 인지 15, 파지 5(Contact GraspNet, GraspGen, M2T2), 모션 8, 비전 유틸 15, 기타 8(ROS 번역기 포함)(§4.1) | open-robot-skills 공개 파일 목록에서 SKILL.md는 43개(스킬 33, 도구 번들 8: curobo, curve, gemini-er, geometry, grounding-dino, molmo, sam3, vlm, 정책 2: pi05-libero, molmoact-libero)입니다. v1의 파지는 기하 방식(OBB 위에서 내려오기)이고 학습형 파지는 "post-v1"입니다(D1). OWL-ViT, ROS 번역 스킬은 목록에 없습니다 | **차이** (개수와 구성이 다름. 분류 기준이 달라 직접 비교는 어려움) |
| D9 | 벤치마크 자산 | 8개 벤치마크를 "open"으로 공개(§1) | 서브모듈 `Variational-Automation-Benchmark`(브랜치 `packing_v0.1`)와 LIBERO-PRO, robosuite 포크, `robots_realtime`을 쓰며 모두 복제본에서 비어 있습니다. `NOTICE.md`는 앞의 둘을 MIT로 적습니다. Make Popcorn, USB-C, Wash Crates 자산은 저장소에서 찾지 못했습니다 | **미확인** (서브모듈을 받지 않아 내용 확인 불가) |
| D10 | 기준선과 표 생성 | TiPToP, CaP-X, π0.5, MolmoAct2, 손 설계 그래프(§5.2, §5.8) | π0.5·MolmoAct2는 정책 스킬로 있고 `steered_policy`와 `posvar.yaml`의 `llm_plus_policy`, `policy_only` 모드가 하이브리드 방식(§5.3.1)과 같은 설계입니다. TiPToP, CaP-X 구성, 손 설계 Wash Crates·케이블 그래프, 표 1–5의 집계 스크립트는 없습니다 | **일부 일치** (VLA 하이브리드는 있음, 나머지는 없음) |
| D11 | 배포 시 에이전트 | "배포 후에는 에이전트가 필요 없다"(§1, §5.6) | 실행기는 코딩 에이전트 없이 `workflow.json`을 돌립니다. 다만 인지 스크립트(`perceive_dino_vlm.py`, `perceive_gemini_er`)는 실행 중에 VLM을 호출하므로 README는 "LLM key … drives the multi-agent codegen harness"와 "in-graph VLM"을 모두 요구합니다. SAM3와 Grounding DINO 가중치도 필요합니다 | **일치** (코딩 에이전트 기준), VLM 서비스 의존은 있음 |
| D12 | 시뮬레이션 관절 이동 모드 | 서술 없음 | 벤치마크 워커는 정책이 없으면 `GAP_LIBERO_JOINT_MOTION_MODE = "teleport"`로 설정합니다(`gap/agent/parallel.py` 370–372행). teleport는 목표 관절각을 직접 쓰고 최대 10 시뮬 스텝 정착합니다(`docs/source/running/environments.md` 170–180행). 손에 물체를 든 동안에는 자동으로 폐루프 추적으로 바뀝니다. `env_config.py` 112–122행의 기본값은 `closed_loop`이고 문서(170행, 214행)는 기본값이 `teleport`라고 적어 코드와 문서가 서로 다릅니다 | **차이** (논문에 없는 설정. 표 1 GaP 행이 어느 모드에서 나왔는지는 저장소로 알 수 없음) |
| D13 | 정적 타입 검사 | 타입이 있는 입출력, 정적 타입·간선 검사(§2, §4.2) | `gap/runtime/validate.py` 1–70행: 워크플로 수준 W1–W8, 부분 그래프 수준 S1–S11, 스킬 계약 SK1–SK4. 타입 호환은 이름이 같아야 하고(`_types_compatible`, 315–330행) `Any`·`dict`는 항상 통과하며 int와 float는 호환입니다. W8은 입력 이름이 상류 출력에 선언되어 있는지를 봅니다 | **일치** (검사는 이름 기반이고 `Any`는 느슨함) |
| D14 | 라이선스 표기 | — | `LICENSE`·`pyproject.toml`·README는 Apache-2.0, `NOTICE.md`는 MIT, `gap-core/pyproject.toml`은 MIT | **차이** (메타데이터가 서로 다름. 저장소가 Apache-2.0을 기본으로 보이나 `NOTICE.md` 문구는 수정이 필요해 보임) |
| D15 | 실행기 의미론 | 방향 그래프, 독립 가지는 동시 실행, 조건 간선(§3) | `gap/runtime/executor.py` 1–20행: 준비된 노드를 동시에 실행하는 "super-step" 스케줄러, 조건 간선, 스트리밍 노드. 이미 끝난 노드로 가는 조건 간선은 루프로 처리하고 `GAP_ITERATION_CAP`(기본 10000)이 폭주를 막습니다(`examples/grocery_packing/README.md`) | **일치** (논문 서술보다 넓음: 후방 간선과 루프 포함) |
| D16 | 문서의 시뮬레이터 | v3: RoboSimStudio(§4.2) | `overview.md` 43행은 여전히 "internal **Isaac simulation**"이라 쓰고, 문서 첫 화면은 "an updated version is planned by 1 Aug 2026"이라 적고 있습니다. 프로젝트 페이지 문구도 "self-learns in NVIDIA Isaac"입니다. README만 RoboSimStudio(자기 학습 저장소)를 안내합니다 | **차이** (문서와 논문 v3 사이 시차. 논문 v1은 Isaac-Lab이었음, `01` §0) |

## 3. 논문 주장의 코드 쪽 확인

- **"그래프 검증이 연결성을 보장한다"(§5.4).** D13의 규칙이 이를 구현합니다(도달 불가 노드, 끊어진 간선, 상류 없는 입력, 종료 값 일관성). 이 검사가 실행 중 오류를 모두 막는다는 증거는 아니며, 타입이 `Any`인 입출력은 통과합니다.
- **"에이전트가 속이는 유인을 줄인다"(§1).** 코드에서 생성(coordinator, subgraph agent)과 체크포인트 작성(checkpoint agent)이 별도 단계이고 체크포인트가 시뮬 정답(`world_snapshot`)을 읽습니다. 실로봇 커넥터에는 정답 상태가 없어 체크포인트를 건너뜁니다(`docs/source/running/checkpoints.md`). 속임 방지의 효과를 측정한 절제는 저장소에 없습니다.
- **"Gemini 호출 30회 미만, 약 $0.01"(§5.6).** 파이프라인의 호출 수는 부분 그래프 수, 재시도(에이전트당 2), 수정 루프(2), 재생성(2)에 따라 달라서 저장소만으로 상한은 정해지지 않습니다. 4개 부분 그래프의 퀵스타트 기준 기본 호출 수는 coordinator 1 + subgraph 4 + checkpoint 1 = 6회에 재시도가 더해집니다(추정, 로그 미확인).
- **"실로봇에서 같은 그래프가 그대로 동작한다"(프로젝트 페이지).** 커넥터 계층이 시뮬(LIBERO)과 실제(Franka, UR)를 같은 도구 인터페이스로 감싸는 구조(`gap/connector`, `gap/envs/franka_real_env.py`, `ur_zed_env.py`)는 있으나, 실로봇 예제는 README가 `docs/safety.md`를 먼저 읽도록 요구합니다.

## 4. 구현 가능성 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**논문 수치의 재현은 불가능합니다. 방법의 핵심(타입 그래프, 정적 검증, 채택 루프)은 이 맥에서 축소 구현할 수 있고, 재계산과 채택 규칙 토이는 끝냈습니다.**

| 작업 | 판정 | 근거 |
|---|---|---|
| 표 1–5 재계산과 검정 | 가능, 완료 | `analysis/recompute.py`(수 초) |
| 채택 규칙의 잡음 토이 | 가능, 완료 | `analysis/toy_accept_rule.py`(수 초, numpy) |
| 표 1 재현(LIBERO 변형 5,500회) | 불가 | README는 RTX 4090(24 GB) + Linux + EGL을 요구하고 SAM3, cuRobo가 CUDA 전제입니다. 이 맥에는 CUDA와 EGL이 없습니다 |
| 공식 엔진 설치와 `gap generate`(LLM으로 그래프 생성) | 조건부 | 새 가상환경, 패키지 설치, LLM API 호출(키와 비용), 공식 코드 실행의 승인이 필요합니다. 문서는 "시뮬레이터가 필요 없고 LLM 호출과 정적 검증뿐"이라고 적지만(`generation.md` 22–27행) 스킬 레지스트리(open-robot-skills) 체크아웃이 필요하고 이 맥에서의 동작은 미확인입니다 |
| `gap run --validate-only`(그래프 정적 검증) | 조건부(추정) | 엔진 설치가 먼저 필요합니다. 순수 파이썬 검증이라 CPU에서 돌 가능성이 높지만 미확인입니다 |
| 자기 학습 루프(논문 알고리즘 1) | 불가(코드 없음) | 이 저장소에 없고, 별도 저장소는 RoboSimStudio(GPU)와 Claude Code 세션이 필요합니다 |
| Make Popcorn, USB-C, Wash Crates, GraspGen | 불가 | 그래프·스킬·자산이 v1 밖이고, 실로봇이나 양팔 시뮬이 필요합니다 |
| 실로봇(Franka, UR5) | 불가 | 하드웨어와 컨트롤러 브리지 |
| 축소 재구현: 타입 그래프 실행기·검증기 + MuJoCo 장난감 pick-and-place | 가능 | mujoco 4.0과 torch 2.15가 설치되어 있습니다(새 설치 없음). 인지는 잡음이 있는 가짜 검출기로 대체합니다 |

- **노력 추정(측정 전):** 그래프 실행기·검증기 순수 파이썬 재구현 3–5일, MuJoCo 장난감 환경과 "그래프 대 단일 스크립트" 절제 1–2주, 자기 학습 루프의 규칙 기반 축소판(LLM 대신 규칙 편집)과 채택 규칙 비교 1–2주. 공식 파이프라인 전체는 RTX급 GPU 서버와 LLM 비용이 필요합니다.
- **LLM이 필요한 부분**(코드 생성, 인지 VLM)은 외부 API 또는 로컬 모델이 필요합니다. 이 분석은 API를 호출하지 않았고, 로컬 모델은 받지 않았습니다.

## 5. 재실행

정적 읽기라 재실행할 코드가 없습니다. 이 문서의 수치 근거는 `analysis/recompute.py`와 `analysis/toy_accept_rule.py`입니다.

```bash
python3 analysis/recompute.py
python3 analysis/toy_accept_rule.py
```
