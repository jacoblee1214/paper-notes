# 02. 선행연구와 관련 저장소

MotionDisco의 구성 요소가 어디서 왔는지, 가까운 대안이 무엇인지, 코드가 어디 있는지를 정리합니다. 번호는 논문 참고문헌 번호입니다.

## 1. 선행연구 계보 (구성 요소별 출처)

| MotionDisco의 요소 | 출처 (논문 번호) | 관계 |
|---|---|---|
| 접촉 모드열을 고정하고 연속 궤적을 최적화 | 접촉 계획 [1] (Tonneau 2018), 논리 기하 계획 [2] (Toussaint 2018), 저자 그룹의 휴머노이드 TAMP [3] (Ciebielski 2025) | 식 1–3의 접촉 모드 + 모드별 다양체 구조. [3]과 [25]가 같은 그룹의 직접 선행입니다 |
| 직접 다중 사격과 구조 활용 솔버 | acados [26], HIPPO [27] (Zhao, Righetti, Khadiv) | KSO는 acados, TO는 HIPPO 계열(공식 코드는 `moto`)을 씁니다(§3.1, `04` D12) |
| LLM이 계획 프로그램을 씀 | SayCan [14], Code as Policies [15], PRoC3S [16], InstructFlow [17] | 고수준 절차를 LLM이 제공하거나 코드로 번역. MotionDisco는 이를 접촉 계획 프로그램으로 좁혔습니다 |
| LLM + 진화 | Eureka [18](보상 코드 진화), AlphaEvolve [21], ShinkaEvolve [22], AdaEvolve [23], EvoX [24] | 탐색 골격은 ShinkaEvolve를 그대로 채택했습니다(§3.2.2). 공식 저장소의 `shinka/`는 ShinkaEvolve에서 파생됐다고 NOTICE에 적혀 있습니다 |
| 가장 가까운 연구 | Shcherba 외 [20] (CoRL 2025, TAMP에서 LLM 프로그램 탐색으로 궤적 최적화 보조) | [20]은 조작 술어 수준, MotionDisco는 보행과 조작을 함께 다루는 접촉 계획 수준이라고 구분합니다(§2.2) |
| 사람 → 휴머노이드 리타게팅 | ASAP [4], OmniH2O [5], OmniRetarget [6], 잠재 확산 [7], SPIDER [12], DynaRetarget [13] | MotionDisco가 피하려는 데이터 경로. [13]은 동적 정제 후 RL 추종이라 가장 가까운 "정제" 대안입니다 |
| 추종 정책 학습 | DeepMimic [8], BeyondMimic [9], HDMI [10], ResMimic [11] | §4.3의 실로봇 추종은 DeepMimic식 보상. 세부 설정은 본문에 없습니다 |
| 사람 행동을 LLM 계층 계획 + RL로 | Wu 외 [19] | 계층 계획 + 시뮬레이션 추종. 접촉 계획을 프로그램으로 탐색하지는 않습니다 |
| 시뮬레이션 준비 장면(향후 과제) | SimRecon [29], HoloScene [30], ReplicateAnyScene [31] | 인지 없이 알려진 장면을 가정하는 한계의 해결 방향(§6) |

**가까운 대안과의 차이.** (a) 리타게팅 계열은 사람 동작이 있어야 하고 접촉 정보 복원이 약합니다(§2.1). MotionDisco는 접촉 계획을 탐색으로 얻고 TO로 동적 실현성을 보장합니다. 대신 장면과 접촉 모델(직사각형 패치의 단방향 고착 접촉, 상자 물체)에 제약이 큽니다(§6). (b) 샘플링 기반 TAMP나 MCTS형 접촉 탐색은 논문 비교 대상이 아닙니다. 공식 저장소의 서브모듈 FARO에 MCTS 목적 함수가 있다고 코드 주석에 언급되지만(`scoring.py`), 논문은 그 비교를 싣지 않았습니다. (c) 표 1의 기준선은 "단일 LLM 호출"과 "피드백 없는 탐색"뿐이라, 기존의 비LLM 접촉 계획기(MCTS, 샘플링)나 같은 예산의 무작위 변이와는 비교하지 않았습니다.

## 2. 코드와 저장소 (GitHub 표)

2026-11-24에 `gh api`로 확인했습니다(메타데이터만, 별도 다운로드 없음).

| 저장소 | 관계 | 라이선스 | 별 | 마지막 push | 대조 기준 | 상태 |
|---|---|---|---|---|---|---|
| [Atarilab/motiondisco](https://github.com/Atarilab/motiondisco) | **공식 구현** (README가 논문과 CoRL 2026 인용을 적음). 탐색 엔진 `shinka/`, loco-manipulation 과제 `examples/loco_manipulation/`, FARO를 서브모듈로 참조 | Apache-2.0 | 11 | 2026-11-06 | 커밋 `0df8eab8623f58b447cd8402836ceffcdd472360` (단일 "Initial commit", 2026-11-06 13:31 +0100). 얕은 복제 약 37 MB | 정적 읽기만 (`04`). 실행, 설치, import 안 함 |
| [Atarilab/faro](https://github.com/Atarilab/faro) | 서브모듈: 장면 레지스트리, KSO, TO의 구현 (README: "contact-explicit trajectory optimizer"). **논문의 KSO와 TO 본체가 여기에 있습니다** | BSD-3-Clause | 16 | 2026-11-07 | 복제하지 않음 (서브모듈이라 공식 저장소 얕은 복제에서는 비어 있음) | 읽지 못함. D 항목 중 솔버 내부는 미확인 |
| [HaizhouZ/moto](https://github.com/HaizhouZ/moto) | HIPPO(RA-L)의 구현이라고 설명되는 TO 솔버 (공저자 개인 저장소). 공식 README의 설치 안내가 가리킴(FARO install.md 경유) | BSD-3-Clause | 22 | 2026-10-09 | 복제하지 않음 | 미확인(참조만) |
| [SakanaAI/ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve) | 탐색 엔진의 원본. NOTICE: 원본 커밋 8adc053(2026-10-05)에서 파생 | Apache-2.0 | 1,538 | 2026-11-23 | 복제하지 않음 | 부모 선택 식은 공식 저장소 쪽 사본으로 확인 |
| 프로젝트 페이지 `atarilab.github.io/motiondisco.io` | 저장소 homepage 필드. 본문은 JS로 그려져 HTML에는 제목만 보입니다 | — | — | — | — | 내용 확인 못 함(본문 없음) |
| 영상 `youtu.be/DHiVz34QYlw` | 논문 초록의 보충 영상 | — | — | — | — | 보지 않음 |

- 논문 본문(arXiv v1)에는 코드와 프로젝트 페이지 링크가 없습니다. 공식 저장소는 논문 게시(2026-06-04)보다 5개월 뒤인 2026-11-05/06에 생겼습니다(단일 커밋, 이력 없음).
- 의존 도구 acados는 라이선스를 조회하지 않았습니다. LLM 호출(Anthropic API 또는 `claude-max/` 구독 CLI), 임베딩 서버(Ollama의 embeddinggemma), 컨다 환경이 필요합니다(`04` §4).

## 4. 참고: 이 논문의 주장이 기대는 선행 결과

- **LLM 진화 탐색의 표본 효율**은 ShinkaEvolve [22]가 주장하는 성질(섬 모형, 새로움 기각)을 그대로 상속합니다. 이 논문의 "절제"는 피드백 유무만 바꾸고 섬, 새로움, 영감 샘플링의 기여는 분리하지 않았습니다(표 1은 세 조건뿐).
- **KSO와 TO의 신뢰성**은 FARO와 HIPPO의 성질에 기댑니다. 공식 저장소의 주석은 KSO 판정이 1e-2 경계 근처에서 잡음이 있다고 적습니다(`ik_eval.py`). 같은 입력이 실행에 따라 통과 또는 실패한다는 설명이며, 논문 식 6의 접두부 탐색이 단조성을 가정한다는 점과 함께 `04` D6에서 다룹니다.
