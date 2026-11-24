# 02. 선행연구와 관련 저장소

이 문서는 GaP의 구성 요소가 어디서 왔는지, 가까운 대안이 무엇인지, 공식 코드와 저장소 상태가 어떤지를 정리합니다. 번호는 논문 참고문헌 번호(`paper/paper.txt`의 References, v3)입니다.

## 1. 선행연구 계보 (구성 요소별 출처)

| GaP의 요소 | 출처 (논문 번호) | 관계 |
|---|---|---|
| LLM이 로봇 정책 코드를 씀 | Code as Policies [30], CodeDiffuser [31, 44], ALGARA [43], GRAPPA [45], Maestro [46] | GaP의 출발점입니다. 저자는 이들을 "구조가 느슨하다(자유 형식 파이썬)"고 봅니다(§1, §2) |
| 코딩 에이전트 벤치마크와 기준선 | CaP-X [32] | 표 1의 CaP-X 행. 단일 에이전트, 이미지와 문장만 입력. 저자는 이를 "GaP에서 다중 에이전트와 자기 학습을 뺀 절제"로 해석합니다(§5.2) |
| 그래프 구조: 타입이 있는 데이터 흐름 | ROS [3] | "타입 있는 데이터 흐름 위상"을 가져옵니다(§2) |
| 그래프 구조: 반응형 분기와 실패 복구 | 행동 트리 [41], BTGenBot [42] | 조건 분기와 복구 루프(§2). 정적 검사와 시뮬 기반 최적화는 GaP의 추가 요소입니다 |
| 계층적 과제·모션 계획의 영감 | TAMP [33, 34], Shakey [39, 40] | 안전을 위한 그래프 계층(§1) |
| LLM + TAMP 기준선 | TiPToP [35] (cuTAMP [73], M2T2 [68], Gemini 인지) | 표 1, 2, 3의 TipTop 행(§5.2). 표 3에는 MORSL 스킬로 바꾼 변형이 있습니다 |
| LLM 기반 TAMP 계열 | LLM3 [47], Proc3s [48], VLM 유도 TAMP [49, 50] | 관련 연구 수준의 언급(§2) |
| 스킬 라이브러리와 자기 개선 에이전트 | Voyager [55], AFlow [54] | 스킬을 누적하는 에이전트 워크플로(§2) |
| 언어 기반 실패 추론과 프롬프트 최적화 | promptolution [56], GEPA [57], Feedback Descent [58] | 반복 최적화의 문헌 배경. GaP는 이 방법들을 실제로 쓰거나 비교하지 않습니다(부록 A 서술만) |
| 물리 실험으로 LLM 계획 개선 | Blox-Net [59], BrickGPT [74] | 리허설 기반 개선의 직접 선례(§2, 부록 A) |
| 에이전트 스킬 규약과 하네스 개념 | Agent Skills(Skill.md) [60], 장기 실행 하네스 설계 글 [29] | MORSL의 스킬 형식과 "하네스" 용어(§1, §4.1) |
| 벤치마크 기반 | LIBERO [36], LIBERO-Pro [71], Minkowski 합 [72] | 벤치마크 I–III의 장면. 위치 변동과 바구니 교환은 이를 확장합니다(§5.1) |
| 일반 VLA 기준선 | π0.5 [10], MolmoAct2 [9] | 표 1의 VLA 행과 하이브리드 행(§5.2) |
| MORSL의 구성 모델 | SAM 2·3 [61, 62], Grounding DINO [63], OWL-ViT [64], Molmo [65], Gemma 3 [66], Contact-GraspNet [67], GraspGen [4], M2T2 [68], cuRobo·cuRobo v2 [69, 70] | 인지 15, 파지 5, 모션 8개 스킬의 바탕(§4.1) |
| 내부 시뮬레이션 | RoboSimStudio [5] (arXiv:2610.18835) | v3에서 명시. v1은 Isaac-Lab, v2는 Newton과 Isaac Sim으로 적었습니다(`01` §0) |

**가까운 대안과의 차이.** 가장 가까운 것은 CaP 계열(CaP-X)과 TAMP 계열(TiPToP)입니다. CaP-X는 단일 에이전트가 자유 형식 코드를 쓰고, GaP는 노드와 간선이 타입으로 검증되는 그래프를 다중 에이전트가 쓰며 시뮬레이션 리허설로 갱신합니다. TiPToP는 형식 계획기(cuTAMP)를 쓰고 GaP는 LLM이 그래프를 고칩니다. 표 3은 "MORSL 스킬을 얹은 TiPToP"(0.53, 0.60)가 원래 TiPToP(0.24, 0.34)보다 높다는 것도 보여 줍니다. 즉 일부 이득은 스킬 라이브러리에서 오고 그래프 구조와 자기 학습만의 효과가 아닙니다(`03` V5). 이 효과를 분리한 대조(같은 스킬에서 구조만 바꾸는 것)는 논문에 없습니다.

## 2. 코드와 저장소 (GitHub 표)

조회일 2026-11-24(`gh api` 메타데이터). 공식 저장소만 얕게 복제했고, 나머지는 메타데이터와 공개 파일 목록만 확인했습니다. 어떤 코드도 실행, import, 설치하지 않았습니다.

| 저장소 | 관계 | 라이선스 | 별 · 마지막 push | 대조 기준과 상태 |
|---|---|---|---|---|
| [graph-robots/graph-as-policy](https://github.com/graph-robots/graph-as-policy) | 공식 구현(엔진, 에이전트 코드 생성, 실행기, 검증, 벤치마크 하니스). README 첫머리가 "BETA CODE RELEASE, July 1, 2026" | LICENSE와 `pyproject.toml`은 Apache-2.0, `NOTICE.md`는 MIT라고 적음, `gap-core/pyproject.toml`은 MIT (`04` D14) | 222 · 2026-11-23 | `e572392cce48475d1ca4ba0f564d306e437a69cc` (2026-11-11, 얕은 복제, 서브모듈 비어 있음). 정적 읽기만 (`04`) |
| [graph-robots/open-robot-skills](https://github.com/graph-robots/open-robot-skills) | 공식 스킬·도구·정책 번들(MORSL에 해당) | Apache-2.0 | 77 · 2026-11-23 | 공개 파일 목록에서 SKILL.md 43개(스킬 33, 도구 8, 정책 2) 확인. 복제 안 함 (`04` D8) |
| [graph-robots/agentic-self-learning](https://github.com/graph-robots/agentic-self-learning) | 자기 학습 루프(2026-10-22 생성, 후속 논문 arXiv:2611.09030에 딸림) | Apache-2.0 | 6 · 2026-11-11 | README와 `asl/loop.py` 한 파일만 API로 읽음. 복제 안 함 (`04` D3) |
| [robosimstudio/robosimstudio](https://github.com/robosimstudio/robosimstudio) | 논문 v3의 내부 시뮬레이터 [5] (arXiv:2610.18835) | Apache-2.0 | 673 · 2026-11-23 | 메타데이터만. 확인 안 함 |
| [ehehee/Variational-Automation-Benchmark](https://github.com/ehehee/Variational-Automation-Benchmark) (서브모듈, `packing_v0.1` 브랜치) | 위치 변동·순열·바구니 교환이 박힌 LIBERO 포크(`NOTICE.md`는 MIT로 적음) | MIT (NOTICE 기준) | 확인 안 함 | 서브모듈 비어 있음(복제 안 함). 확인 안 함 (`04` D9) |
| [kabilankb/so101-gap](https://github.com/kabilankb/so101-gap) | 개인 개발자의 GaP 이식 데모(LeRobot SO-101 MuJoCo 장면, Jetson AGX Orin 한 대). 2026-11-10 생성 | Apache-2.0 | 11 · 2026-11-10 | `33f6cc1` (얕은 복제). 정적 읽기만 |
| 프로젝트 페이지 [graph-robots.github.io/gap](https://graph-robots.github.io/gap/) | 논문 프로젝트 페이지(결과표, 그래프 시각화, 영상) | — | — | 2026-11-24 확인. 읽음 |
| 문서 [graph-robots.github.io/graph-as-policy](https://graph-robots.github.io/graph-as-policy/) | 공식 문서. 첫 화면에 "Beta Code release (1 July 2026)", "an updated version is planned by 1 Aug 2026"이 적혀 있음 | — | — | 2026-11-24 확인. 읽음 |

- 후속 연구 arXiv:2611.09030(*A Controlled Study of Models, Skills, Feedback, and Curriculum in Agentic Robotics*, 같은 연구 그룹)은 공식 README가 자기 학습 파이프라인의 논문으로 안내합니다. 이 분석의 대상은 아닙니다.
- 기준선(TiPToP, CaP-X, π0.5, MolmoAct2)과 표 1–5를 만든 평가 스크립트, 케이블 삽입·Wash Crates·Make Popcorn 그래프는 공식 저장소에서 찾지 못했습니다(`04` D1, D9, D10).

## 4. 참고: 이 논문의 주장이 기대는 선행 결과

- "단일 에이전트는 환각과 제약 위반이 잦고, 구조화하면 개선된다"는 §1의 동기는 CaP-X [32]와 GaP 자체의 절제(§5.4)에 기댑니다. 절제는 시행 수가 적혀 있지 않아 이 문헌과 같은 정밀도로 비교할 수 없습니다.
- "LLM이 만든 코드에 시뮬레이션 피드백을 되먹임하면 개선된다"는 점은 Blox-Net [59] 등에 있고, GaP의 새로움은 이를 그래프 단위의 타입 있는 편집(다섯 동작)으로 제한한 데 있습니다. 이 제한이 자유 편집보다 낫다는 대조는 논문에 없습니다.
- 처리량 지표(성공률 나누기 주기 시간)는 §2에서 자동화 연구의 표준으로 소개됩니다. 표 3의 "cycles/hr"과는 읽는 방식이 다릅니다(`03` V8).
