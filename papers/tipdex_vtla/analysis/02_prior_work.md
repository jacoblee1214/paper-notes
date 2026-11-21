# TipDex-VTLA: 선행연구와 관련 저장소

## 1. 계보

- **VLA 기반.** RT-1 [3], RT-2 [48], Open X-Embodiment [29], Octo [9], OpenVLA [17], π0 [2] (흐름 매칭 Action Expert), π0.5 [1] (이질 공동 학습). TipDex의 기저 구조는 π0.5 계열의 흐름 매칭 Action Expert입니다(공개 메타데이터 기준, `04` D8).
- **촉각 VTLA.** AnyTouch [6], ForceVLA [37]/ForceVLA2 [21], DECO [20] (촉각 어댑터), FTP-1 [38], AT-VLA [19], TacVLA [42], ResTacVLA [43] (촉각 잔차를 시각 예측에 대해 게이트).
- **촉각 예측과 실행 중 보정.** N0-VTLA [27] (잠재 촉각 토큰), ForeTac-VLA [30], UniTacVLA [45], TouchWorld [47] (촉각 예측과 고주파 잔차), ReTouch [44], TacPAC [26] (예측과 관측을 맞춰 미실행 행동 보정), T-Rex [28] (캐시된 흐름 상태와 가변 주기 행동 생성).
- **시각 강건성.** ImageNet-C [13], AugMix [14], DeepAugment [15], LIBERO-Plus [5] (조명과 시점 교란 평가), RobustVLA [10], StableVLA [7], ChromaGuard [33], E-VLA [41], 증거 게이트 정규화 [36], BAS-VLA [23].
- **이 논문이 채우는 자리.** 시각 강건성 계열은 시각 특징을 고치고, 촉각 계열은 촉각을 붙이거나 예측합니다. 두 신호의 신뢰도를 함께 실행 중 보정 강도로 쓰는 연구는 논문 §2.3이 말한 대로 드뭅니다. 가장 가까운 것은 ResTacVLA [43]의 시각 기준 게이트와 TacPAC [26]의 예측 검증입니다.

## 2. 공식 코드와 저장소

| 저장소 | 관계 | 라이선스 | 최근 활동 (API, 2026-11-21 확인) | 상태 |
|---|---|---|---|---|
| [tiplab/TipDex-VTLA](https://github.com/tiplab/TipDex-VTLA) | 공식 추론 구현 (README "Official inference implementation"). 프로젝트 페이지 "Code" 링크가 가리킴 | Apache-2.0 (코드). 모델 패키지는 Gemma 조건 (`licenses/LICENSE_GEMMA.txt`) | 생성 2026-11-19, 최근 push 2026-11-20 16:25 UTC, 별 0, 커밋 13개, HEAD `7e68bc5` | **정적 읽기 전용 얕은 복제** (`repo/`, gitignore). 설치, 실행 안 함 |
| [sez956/TipDex-VTLA](https://github.com/sez956/TipDex-VTLA) | 프로젝트 사이트 소스 (GitHub Pages). 설명: "research project website and demonstrations. No manuscript PDF." 코드는 위 tiplab 저장소를 가리킴 | 없음 (API license: null) | 생성 2026-11-10, 최근 push 2026-11-20 | 사이트만 확인. 코드 비교에는 쓰지 않음 |
| [HF TipDex-VTLA/TipDex-VTLA](https://huggingface.co/TipDex-VTLA/TipDex-VTLA) | 모델 패키지: 백본 9309(에그 과제), VASE, TACE, 촉각 예측기, 토크나이저, 정규화 통계 | 모델 카드 참조 (Gemma 조건) | HTTP 200 (메타데이터만) | **다운로드 안 함** (가중치 금지) |
| [HF datasets TipDex-VTLA/TipDex-Data](https://huggingface.co/datasets/TipDex-VTLA/TipDex-Data) | 원본 50 궤적 + 조명 교란 짝 50 궤적 | 확인 안 함 | HTTP 200 (메타데이터만) | **다운로드 안 함** |
| [tiplab.github.io/TipDex-VTLA](https://tiplab.github.io/TipDex-VTLA/) | 프로젝트 페이지 (README가 가리킴). 인용 블록 "Unpublished manuscript" | — | HTTP 200 | 본문 확인 |

- 두 프로젝트 페이지가 있습니다. 논문 본문과 abs 페이지에는 프로젝트 주소가 없습니다. 공식 코드 README와 tiplab 페이지가 같은 주소를 가리키므로, 이 분석에서는 tiplab을 공식으로 봅니다. sez956 사이트는 같은 내용을 옮긴 것으로 보이나 소유자와 관계를 확인하지 못했습니다.
- "Checkpoints 正在加工"(준비 중)이라는 표시가 프로젝트 페이지에 있었습니다. 같은 시점에 HF 모델 저장소는 접근됩니다.
- 학습 루프, 평가 스크립트, UniVTAC 실행 코드, 실로봇 드라이버는 저장소에 없고, 손실 정의 일부만 있습니다. 이 저장소는 추론 예제와 배포 프로토콜입니다 (`04` D1, D12).

## 4. 구성 요소별 출처

| 구성 요소 | 출처 | 논문에서 새로운 부분 |
|---|---|---|
| 흐름 매칭 Action Expert | Lipman 외 [22] | 없음 |
| 느린-빠른 흐름 실행 | T-Rex [28] | 보정을 빠른 단계에만 넣는 방식은 기존 방법을 따름 |
| 촉각 예측기와 예측-검증 | TacPAC [26], ForeTac-VLA [30] | 적격 예측 선택과 명령 일관성 검사를 함께 씀 |
| 시각 행동 신뢰도 q_v | 교란 강건성 연구 [5, 7, 10] | 얼린 기준 정책의 짝 이미지 행동 차이를 감독으로 씀 (Eq. 1) |
| 게이트 보정 | ResTacVLA [43] | q_v와 e를 함께 쓰는 게이트 (Eq. 4) |
