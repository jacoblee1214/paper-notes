# 02. 선행연구와 계보

번호 [n]은 논문 참고문헌 번호입니다(`paper/paper_raw.txt`). 인용한 논문을 각각 열어 검증하지는 않았습니다. 계보는 DAPPT 본문의 인용과 공식 저장소 상태를 기준으로 정리했습니다.

## 1. 계보

DAPPT는 세 줄기가 만나는 곳에 있습니다.

**(a) 시뮬레이션 기반 손 조작 RL (sim-to-real).** 강체 시뮬레이션에서 학습하고 실로봇에 옮기는 손 조작은 큐브 회전 [10], 파지 [1–4], 손 안 조작 [5–9]에서 성과를 냈습니다. DAPPT가 보기에 이 방법들은 보상을 사람이 정의해야 하고, 도구 사용 같은 다른 기술로 옮기기 어렵습니다(§II "Learning with Manually-Defined Rewards").

**(b) 목표 포즈 추종으로 일반 기술을 얻는 방법.** 6D 포즈 도달 기술을 먼저 학습하는 계열이 있습니다. SimToolReal [16]은 도구 조작을 "object-centric" 목표 도달로 학습하고, 이 논문이 가장 가까운 기준선입니다. [11] Play2Perfect, [24] LUCID도 같은 흐름에 있습니다(§II). DAPPT는 이들과 달리 **도메인 파라미터로 궤적 집합을 학습**하고, 사람 시연 없이 새 도구 포즈에 대응한다고 주장합니다.

**(c) 인간 영상 기반 궤적 추종.** 인간 손-물체 궤적을 추출해 로봇 궤적으로 리타겟하고, RL 또는 샘플링으로 추종시키는 계열입니다(§II "Learning with Human Hand Videos", [25–32]). DAPPT는 영상 궤적 대신 절차적 궤적을 씁니다.

**이 논문의 위치.** 입자·변형체를 시뮬레이션하지 않고, 강체 도구 궤적만 추종하게 한 뒤 도메인 파라미터 탐색으로 최적 행동을 고릅니다. 이 설계가 입자 시뮬레이션 비용을 얼마나 줄이는지는 본문에서 측정하지 않았습니다(시뮬레이터 속도 비교 없음).

## 2. 구성 요소별 출처

| 구성 요소 | 출처 | 논문에서의 역할 | 가까운 대안 |
|---|---|---|---|
| 교사 학습 | PPO [37] | 비대칭 critic, 상태 기반 교사 | 기타 on-policy RL |
| 시연 보조 | DexPilot [35] (teleop) | 10–20 시연, BC 목표를 학습 초기에 더함 | 인간 영상 리타겟 [25–32] |
| 증류 | DAgger [38] | 교사 → 시각 정책 | 직접 BC |
| 인코더 | PointNet++ [39] | 1024점 포인트 클라우드 | 다른 점 인코더 |
| 궤적 계열 | 절차적 생성 (이 논문) | 도메인 파라미터로 궤적 집합 정의 | 인간 영상 궤적 [14, 15] |
| 기준선 | SimToolReal [16] | GPT-6 궤적 입력, "최선" 보고 | DexMachina [14], DexTrack [15] |
| 시뮬레이터 | IsaacLab (학습), Genesis (평가) | 강체 학습, MPM 평가 | — |
| 조작 방법 | 저강성 PD, Kd 보정 [41] | 실로봇 sim-to-real 간격 | 힘 기반 제어 |
| 도메인 무작위화 | 무작위 힘, 행동 잡음, 질량·마찰 | 로버스트 정책 | — |

## 4. GitHub와 자료 (확인 2026-11-22)

| 자료 | 링크 | 라이선스 | 별 / 마지막 활동 | 비고 |
|---|---|---|---|---|
| **DAPPT 저장소** | https://github.com/princeton-vl/DAPPT | 없음 | 별 1, 생성 2026-11-15, 마지막 push 2026-11-17 | README 한 장("Code coming soon")뿐. 커밋 2개(`42bc70a`, `d0699b8`). 얕은 clone 후 정적으로만 읽음 |
| 프로젝트 페이지 | https://beininghan.github.io/DAPPT-web/ | - | - | 시뮬·실로봇 롤아웃 영상, 코드 버튼은 `#`(자리표시자) |
| 시뮬레이터 Genesis | https://github.com/Genesis-Embodied-AI/genesis-world | Apache-2.0 | 별 30,283, 마지막 push 2026-11-20 | 평가에 쓰임. 논문 버전 명기 없음 |
| 시뮬레이터 IsaacLab | https://github.com/isaac-sim/IsaacLab | NOASSERTION(GitHub가 라이선스 파일 판정 못 함) | 별 8,694, 마지막 push 2026-11-21 | 학습에 쓰임. CUDA GPU 필요 |

**판정.** 논문 수치를 재현할 공식 코드가 없습니다. 따라서 `04_claim_verification.md`에서 주장 대조를 합니다(§4 참조). 저장소는 생긴 지 일주일 정도이고 "Code coming soon"이라 공개 시점은 불명확합니다. 공개되면 `04_code_verification.md`로 바꿔 다시 대조해야 합니다.

## 5. 학회 판정

**결과: 학회 표기 없음 (arXiv).**

확인한 것:
- arXiv abs 페이지: comments, journal-ref 모두 없음 (2026-11-22).
- 논문 본문: 학회 이름 없음. 저자 소속은 Princeton University뿐.
- 프로젝트 페이지: 학회 이름과 "submitted" 문구 없음. 인용 블록의 `journal = {Princeton University}`는 학회가 아닌 소속입니다.
- 공식 목록 `conferences/corl2026/papers.json` (CoRL 2026 accepted papers, 755편, 2026-11-14 수집): 제목 키워드(Procedural, Domain-Aware, Dexterous Tool Use, Sim-to-Real Dexterous)로 검색해 일치 없음.
  - 주의: 이 스냅샷은 논문 제출(2026-11-15)보다 하루 먼저 수집되었습니다. 따라서 **없다는 것이 CoRL 탈락을 뜻하지 않습니다.** 채택 목록을 다시 받아 확인해야 합니다.
- 웹 검색: 2차 집계 사이트(teahose.com)만 나오고, 학회 공식 목록은 없습니다. RSS, ICRA, IROS, ICLR, NeurIPS의 공식 목록은 이 분석에서 개별로 검색하지 않았습니다.

**판정 근거.** 공식 목록 확인이 없고 저자·프로젝트 페이지도 학회를 말하지 않으므로 기준에 따라 `학회 표기 없음 (arXiv)`로 둡니다. 이 판정은 2026-11-22 기준이며, CoRL 2026 채택 목록을 다시 받으면 바뀔 수 있습니다.
