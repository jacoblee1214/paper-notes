# 02. 선행연구와 관련 자료

번호는 논문 참고문헌 번호입니다(`paper/paper.txt`의 References).

## 1. 선행연구 계보 (구성 요소별 출처)

| 이 논문의 요소 | 출처 | 관계 |
|---|---|---|
| 행동 청킹, 열린 루프 실행 | ACT [14] (Zhao 2023) | 청크 H개를 한 번에 예측해 추론 빈도를 줄입니다. 이 논문의 전제입니다 |
| 비동기 추론, 행동 큐 | SmolVLA [16], Bidirectional decoding [15], "Thinking while moving" [17] | 서버-클라이언트와 큐로 추론과 실행을 겹칩니다. §3.1의 생산자-소비자 정의가 이 계열 위에 있습니다 |
| 실시간 청킹(RTC), 수정 불가능 접두 | Real-time chunking [18] (Black 2025) | 이미 확정된 행동 접두를 고정하고 나머지를 인페인팅합니다. 이 논문의 수식 표기(d, s, m)와 기준선(π0 + RTC)이 여기서 옵니다 |
| 보정 헤드, 학습 시 접두 조건화 | Leave no observation behind [19], VLASH [20], Training-time action conditioning [21] | RTC의 변형입니다. 이 논문의 접두 조건화(식 1)는 "AR 모델은 접두를 프롬프트에 넣으면 된다"는 형태로 같은 목표를 달성합니다 |
| 확산·흐름 정책 | Diffusion policy [22], Streaming diffusion policy [23] | 병렬 디코딩이라 지연이 거의 일정합니다. 이 논문은 이것이 선행 실시간 연구의 대상이었다고 봅니다 |
| AR VLA | RT-2 [2], OpenVLA [3], π0-FAST [6] | 순차 디코딩. 지연이 토큰 수에 비례합니다 |
| 행동 토큰화 | RT-1 빈 토큰화 [26], FAST [6] (DCT + BPE), FASTer [27], BEAST [28] (B-스플라인), OAT [29] (순서 있는 토큰화) | FAST+ 토크나이저를 400 ms 구간에 재사용합니다. [29]는 BPE로 잃는 well-definedness 문제를 다룹니다. 이 논문의 제약 디코딩이 같은 문제를 푸는 방법입니다 |
| 제약 디코딩 | 격자 빔 서치 [33], 동적 빔 할당 [34], 제약 순차 추론 [35], RT-2 [2] | 규칙으로 로짓을 가리는 기존 방식. 이 논문은 "B 토큰 안에서 용량 C를 정확히 채운다"를 한정 동전 교환 DP로 표현합니다 |
| 다중 궤적 디코딩, KV 공유 | 검증자 없는 시험 시 샘플링 [36], 디코딩의 메모리 병목 [37] | 접두 KV를 공유하고 가능도로 고릅니다 |
| 사전학습 구조 | PaliGemma [38], π0 [4], π0.5 [5], 지식 격리(knowledge insulation) [39] | π0-FAST와 π0는 같은 VLM을 쓰고, π0.5는 웹 데이터를 추가해 학습한 후속입니다 |
| 평가 환경 | LIBERO [24], DROID [25], Kinetix [31] | LIBERO는 시뮬레이션, DROID는 제로샷 실로봇, Kinetix는 RTC 원논문의 2D 시뮬레이션입니다 |
| 지연 상한 | MEM [32], AWQ [40], 추측 디코딩 [41] | §3.2는 MEM을 반응성 근거로, 한계 절은 AWQ와 추측 디코딩을 지연 해법으로 듭니다 |

**가까운 대안과의 차이.** 같은 목표(비동기 실행과 부드러운 청크 연결)를 흐름 정책 쪽에서 푸는 방법이 RTC·VLASH·학습 시 접두 조건화입니다. 이 논문은 정책 종류가 다르고(AR) 지연이 가변이라는 점이 달라서, 지연 상한을 "토큰 수 상한 B"로 바꿔 보장한다는 것이 핵심 차이입니다. 비교 기준선은 π0 + RTC와 π0.5 + RTC뿐이고, VLASH, 보정 헤드, 학습 시 접두 조건화와의 직접 비교는 없습니다.

## 2. GitHub와 자료 표

| 저장소·자료 | 관계 | 라이선스 | 별 | 마지막 push | 상태 |
|---|---|---|---|---|---|
| [oddqueue/realfast](https://github.com/oddqueue/realfast) | 저자 계정의 저장소. 이름이 방법과 같음 | Apache-2.0 (openpi에서 상속) | 1 | 2026-11-08 | **방법 코드가 없습니다.** `main`은 Physical-Intelligence/openpi의 포크이고 업스트림 대비 앞선 커밋 0개(뒤처짐 49개, 최신 커밋 2025-11-21). 다른 가지 `page`에 프로젝트 페이지 HTML(커밋 e575b2d, 2026-06-19)만 있음. 복제하지 않음 |
| [oddqueue.github.io/realfast](https://oddqueue.github.io/realfast) | 프로젝트 페이지(논문 첫 쪽의 주소) | 페이지 템플릿 README는 CC BY-SA 4.0 | | | "Code (Coming soon!)" 표기(2026-11-24 확인). 본문은 논문 요약과 영상·그림 |
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | 논문이 구현 기반으로 명시(부록 A, 각주 4). π0, π0-FAST, π0.5 JAX 구현과 LIBERO·DROID 예제 | Apache-2.0 | 14,747 | 2026-08-24 | 읽기 전용 참고. 이 저장소의 LIBERO README와 평가 스크립트 값을 `04` C21에서 씀 |
| [Physical-Intelligence/real-time-chunking-kinetix](https://github.com/Physical-Intelligence/real-time-chunking-kinetix) | RTC 원논문의 Kinetix 시뮬레이션(각주 5) | MIT | 637 | 2025-12-08 | 그림 3b의 환경과 같은 계열. 논문의 RTC "구현"은 공식 릴리스에 없어서 저자가 openpi 위에 직접 구현했다고 적음(부록 A) |
| [mit-han-lab/vlash](https://github.com/mit-han-lab/vlash) | VLASH [20]의 공식 구현 | Apache-2.0 | 527 | 2026-04-22 | 비교 기준선으로 쓰이지 않음 |
| [FLAIROx/Kinetix](https://github.com/FLAIROx/Kinetix) | Kinetix [31] 환경 | MIT | 286 | 2026-09-24 | 참고 |
| [huggingface/lerobot](https://github.com/huggingface/lerobot) | SmolVLA [16]와 비동기 큐 구현 계열 | Apache-2.0 | 29,064 | 2026-11-24 | 참고 |

- 별과 마지막 push는 `gh api`로 2026-11-24에 조회했습니다.
- GitHub 검색("realfast autoregressive real-time execution", "RealFAST pi0-fast real-time chunking constrained decoding")에서 다른 비공식 구현은 나오지 않았습니다.
- `oddqueue/realfast`의 `main`에는 논문의 요소(제약 디코딩, DP 표, 접두 프롬프트, 비동기 서버·클라이언트)가 하나도 없습니다. 그래서 코드와 논문을 대조하는 대신 `04_claim_verification.md`로 논문의 주장을 표·그림·본문 수치로 대조했습니다.

## 4. 참고: 이 논문의 주장이 기대는 선행 결과

- "AR 정책은 수렴이 빠르고 지시 따르기가 더 낫다"는 FAST 논문 [6]의 결과를 인용한 것입니다. 이 논문의 실험은 지시 따르기를 따로 측정하지 않습니다(`04` C17).
- "RTC에서 지연이 같을 때 m이 크면 성능이 떨어진다"는 RTC 원논문의 주입 지연 결과를 인용합니다. 이 분석은 RTC 원논문을 읽지 않았습니다.
