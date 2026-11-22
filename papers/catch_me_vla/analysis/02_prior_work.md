# 02. 선행연구, 가까운 대안, 저장소

## 1. 계보

논문이 스스로 위치를 잡는 세 갈래(§1, §2)와, 각 갈래에서 VLA-Feedback이 가져온 것입니다. 인용 번호는 논문 참고문헌입니다.

| 갈래 | 대표 연구 | VLA-Feedback이 가져온 것 / 다른 점 |
|---|---|---|
| 행동 청킹 VLA와 열린 루프 실행 | π0 [5], GR00T N1 [41], RT-2 [1], OpenVLA [2], PD-VLA [6] | 계획기로 GR00T를 그대로 씁니다. 청크는 계획기가 만들고, 청크 중간 관측은 계획기가 아니라 피드백 모듈이 씁니다. |
| 비동기·실시간 실행 | RTC [26], DynamicVLA [27], AsyncVLA [28], DiscreteRTC [29], VLASH [12], FASTER [8] | "생각하면서 행동하기"로 대기 지연은 줄이지만 청크는 여전히 열린 루프(논문 §2)라고 봅니다. **실험에는 이 계열 기준선이 없습니다.** |
| 이중 시스템(System 2/1) VLA | Fast-in-Slow(FiS-VLA) [9], DP-VLA [10], RoboDual [11], Hume [51], HiRT [52], HyperVLA [53] | 느린 계획기와 빠른 행동 모듈을 다른 주파수로 돌린다는 점은 같습니다. 다른 점은 빠른 모듈이 별도 행동을 만들지 않고 **같은 계획기의 마지막 디노이징 단계만 대신한다**는 것입니다. |
| 확산 가이던스와 조건부 디노이징 | Classifier-free guidance [13], ControlNet [14], 로봇의 가이던스 [44–47] | 중간 디노이징 상태를 추가 정보로 조종한다는 발상을 빌립니다. 가이던스는 청크 생성 시점에만 적용되고 실행 중 관측은 못 쓴다는 점이 동기입니다. |
| 행동 공간 잔차 보정 | A2C2 "Leave No Observation Behind" [54], Policy Decorator [55] | **가장 가깝습니다.** [54]의 초록은 "매 제어 스텝 실행되는 가벼운 청크 보정 머리가 최신 관측, 기본 행동, 청크 내 위치 특징, 기본 정책 특징을 받아 스텝별 보정을 낸다"고 쓰고, 기본 정책 재학습이 필요 없다고 합니다. 입력 구성이 이 논문의 피드백 모듈과 거의 같습니다. 논문은 차이를 "행동 공간에서의 사후 잔차 대 디노이징 속도 갱신"으로 설명하지만 **직접 비교는 없고** 그림 5a의 Δa 변형이 이를 대신합니다. |

## 2. 가까운 대안과 차이

| 방법 | 새 관측을 쓰는 지점 | 계획기 재실행 | 학습 | 이 논문과의 차이 |
|---|---|---|---|---|
| **A2C2** [54] | 기본 행동에 더하는 스텝별 보정 머리(최신 관측, 기본 행동, 청크 내 위치, 기본 정책 특징을 입력) | 불필요 | 기본 정책 재학습 없음(초록 기준) | 입력이 거의 같고 출력만 "보정량" 대 "마지막 디노이징 속도"입니다. 이 논문은 손실이 최종 행동 회귀라 대수적으로 같습니다(`03` §8). 초록 기준으로 Kinetix 12과제와 LIBERO-Spatial에서 RTC 대비 +23, +7 %p를 보고하므로 직접 비교 후보입니다(논문 본문은 읽지 않음). |
| **RTC** [26] | 청크 경계의 안내(guidance)로 이전 청크와 정합 | 필요(비동기) | 학습 없음(추론 시 VJP) | 청크 사이를 잇는 문제를 풉니다. 청크 내부에서 새 관측을 쓰는 문제는 풀지 않습니다. 이 논문의 문제와 직교하므로 결합 가능하지만 시험하지 않았습니다. |
| **VLASH** [12] | 실행 시작 시점의 미래 상태로 계획 | 필요(비동기) | 파인튜닝(상태 오프셋 증강) | 상태 지연을 보정합니다. 영상 관측은 청크 중간에 갱신되지 않습니다. |
| **FiS-VLA** [9] | 빠른 모듈이 매 스텝 갱신 | 부분(느린 경로 4스텝마다) | 공동 학습(이 논문에서는 비전 동결 + LoRA) | 이중 시스템 기준선. 이 논문의 FiS-VLA 결과는 LIBERO 41.5/53.5로 원 논문의 보고보다 크게 낮습니다(원 논문 수치는 확인하지 않음, 미확인). 동일 조건 비교로 읽기 어렵습니다. |
| **DynamicVLA** [27], **AsyncVLA** [28] | 비동기 실행과 연속 추론 | 필요 | 각자 | 동적 물체용 모델. 같은 과제에서 비교하지 않았습니다. |
| **짧은 실행 길이의 GR00T** (기준선 아님) | 청크의 앞 몇 행동만 실행 후 재계획 | 필요(매번) | 없음 | 가장 단순한 대안입니다. 계획기 지연(약 80 ms)이 제어 간격(100 ms)보다 짧아 10 Hz에서도 가능해 보입니다. **이 논문에 없는 핵심 기준선**입니다(`03` C2). |

## 4. GitHub와 코드 자료 (확인 2026-11-22, GitHub API)

| 저장소 | 관계 | 라이선스 | 별 / 마지막 푸시 | 기준 | 상태 |
|---|---|---|---|---|---|
| [jidaxian010/VLA-Feedback-release](https://github.com/jidaxian010/VLA-Feedback-release) | **공식 구현** (피드백 모듈, 학습·평가 스크립트, Robosuite 환경) | Apache-2.0 (NOTICE에 Isaac-GR00T 파생 명시) | ★7 / 2026-09-10 | `38263397` ("minor readme fix", 얕은 복제) | 정적 읽기 완료 (`repo/`) |
| [jidaxian010/Isaac-GR00T](https://github.com/jidaxian010/Isaac-GR00T) | 공식 저장소의 서브모듈(NVIDIA 포크, "작은 변경") | NOASSERTION (원본 Apache-2.0) | ★1 / 2026-05-19 | `88a3bb37` (2026-05-14, "add droid joint") | 서브모듈 커밋만 얕게 받아 정적 읽기 (`repo/third_party/`) |
| [huggingface.co/jidaxian010/vla-feedback-sim](https://huggingface.co/jidaxian010/vla-feedback-sim) | 공식 체크포인트(`cp_sim_obs`, `cp_libero_obs`) | Apache-2.0 | 다운로드 0 / 2026-09-10 | 체크포인트 2개, 합 **15.5 GB** (각 7.77 GB) | 메타데이터만 확인, 받지 않음 |
| [vla-feedback/vla-feedback.github.io](https://github.com/vla-feedback/vla-feedback.github.io) | 프로젝트 페이지 저장소 | 없음 | ★0 / 2026-11-10 | — | 페이지 `https://vla-feedback.github.io` 읽음 |
| [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) | 계획기 원본(GR00T N1.5) | Apache-2.0 | ★8403 / 2026-11-17 | — | 링크만 |
| [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | 정적 평가 벤치마크(서브모듈 `8f1084e3`) | MIT | ★2565 / 2025-03-15 | — | 링크만 |
| [CHEN-H01/Fast-in-Slow](https://github.com/CHEN-H01/Fast-in-Slow) | FiS-VLA 기준선의 공개 코드로 보임(저장소 이름으로 추정, 논문 링크 없음) | MIT | ★165 / 2025-08-01 | — | 링크만 (논문에서 쓴 코드와 같은지 미확인) |
| [openvla/openvla](https://github.com/openvla/openvla) | OpenVLA 기준선 | MIT | ★7331 / 2025-03-23 | — | 링크만 |
| [Physical-Intelligence/real-time-chunking-kinetix](https://github.com/Physical-Intelligence/real-time-chunking-kinetix) | RTC 실험 코드(가까운 대안, 이 논문은 비교 안 함) | MIT | ★637 / 2025-12-08 | — | 링크만 |

- 공식 저장소에는 **베이스라인 체크포인트, 시연 데이터, 실로봇 코드와 체크포인트, OpenVLA·FiS-VLA 파이프라인이 없습니다**(`04` §1). 프로젝트 페이지의 이메일 주소는 기록하지 않았습니다.
- 프로젝트 페이지는 CoRL 2026 표기, 영상, 논문과 코드, 체크포인트 링크, BibTeX를 담고 있습니다.

## 5. 학회 확인

- arXiv comments: "10th Conference on Robot Learning (CoRL 2026), Austin TX, USA". 프로젝트 페이지 동일.
- CoRL 2026 공식 채택 목록(corl.org, 수집 2026-11-14, 687편): 제목 정확히 일치(`match: title_exact`), 발표 **포스터**(Tue 11/10 5:00–6:30 PM, 세션 Poster post-p2, 번호 5), OpenReview forum id가 있습니다.
- 판정: `venue: "CoRL 2026"`, `venue_note: "포스터 (corl.org 공식 채택 목록)"`.

## 6. 참고문헌 점검 (논문 쪽)

- 이 논문이 근거로 든 최근 arXiv 항목 10개의 존재와 제목을 abs 페이지로 확인했습니다: [3] 2602.18397(VLA-Perf), [4] 2605.08168, [7] 2604.04161, [8] 2603.19199(FASTER), [9] 2506.01953(Fast-in-Slow), [12] 2512.01031(VLASH), [27] 2601.22153(DynamicVLA), [28] 2511.14148(AsyncVLA), [29] 2604.25050(DiscreteRTC), [54] 2509.23224(A2C2). 모두 실제 논문이고 제목이 일치합니다.
- **연도 표기 의심 2건(미확인):** [5] π0를 "2026"으로 적었지만 arXiv 2410.24164는 2024년 제출입니다. [2] OpenVLA와 [33] EquiBot, [52] HiRT는 PMLR 권 270(CoRL 2024)인데 날짜를 "Nov 2025"로 적었습니다. 서지 도구가 PMLR 게시일을 쓴 것으로 보이며 내용에는 영향이 없습니다.
- 식 (9)의 "변화가 균등하게 도착한다 → 평균 대기 `R·Δt/2`" 분석은 FASTER [8]의 초록("반응 시간은 첫 행동까지의 시간과 실행 길이가 정하는 균등 분포")과 같은 구도입니다. 논문은 식 (9)를 자체 정의처럼 쓰고 [8]에서 가져왔다고 밝히지 않습니다(초록 기준, 본문 미확인).
- 논문은 [4]를 "비동기 추론 방법 이해" 서베이로 인용해 "이중 시스템의 지연-용량 절충"의 근거로 씁니다. 내용은 읽지 않았습니다.
