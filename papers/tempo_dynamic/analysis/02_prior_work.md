# 02. 선행연구와 자료

## 1. 계보

TEMPO는 세 갈래의 연결입니다. 번호는 논문 참고문헌 번호입니다.

| 갈래 | 대표 연구 | TEMPO가 가져온 것 |
|---|---|---|
| 사전학습 VLA와 흐름 매칭 정책 | π0 [1], π0.5 [2], OpenVLA [3], RT-1·RT-2·Octo [14, 16, 17] | 백본 π0.5(PaliGemma + 행동 전문가 흐름 매칭, 10단계 디노이징). 입력은 현재 관측 한 장 |
| 비동기·실시간 추론 | SmolVLA [20], **RTC** [4], **VLASH** [5], Leave No Observation Behind [21], DynamicVLA [22], F2F-AP [23] | 지연을 줄이는 쪽은 그대로 두고(VLASH를 기본 백본으로), "무엇을 입력으로 주느냐"를 바꿉니다. 비교 기준선 RTC, VLASH |
| 시각 표현과 영상 기반 특징 | 에고센트릭·로봇 영상 시각 사전학습 [24–27], RoboAffordances [28], HPT [29], UVT [30], AnchorAlign [31], **SAM 2** [32], Reducio [33], VidTwin [34], VideoLaVIT [35] | 움직임 코드의 원천으로 얼린 영상 모델(SAM 2.1-tiny)을 씁니다. UVT는 같은 연구실의 학습 목표 |

### 구성 요소별 출처

| TEMPO 구성 | 출처 | 논문이 밝힌 것 |
|---|---|---|
| 흐름 매칭 손실, 행동 전문가, 접두·접미 구조 | π0 / π0.5 | "손실은 바뀌지 않음"(§4.3) |
| AdaRMS 조건 주입 | π0.5의 시간 조건 | ACT 이력을 잔차로 더함(§4.3) |
| 비동기 백본 | VLASH [5] | "본 방법은 이 선택에 무관"(§4.3) |
| 움직임 코드 | SAM 2.1-tiny의 메모리 어텐션 [32] | 캐시와 교차 어텐션(식 1). W 미기재 |
| 행동 이력의 버킷 평균 | (논문이 출처를 적지 않음) | 파라미터 없는 집계(§4.2) |
| TEMPO + UVT | UVT [30] | 출력 쪽 학습 목표를 바꾸는 이전 연구 |
| 프로브 방식 | VLM 프로브 연구 [36, 37] | 고정 토큰 위치의 은닉 상태에 MLP 회귀 |
| TEMPO-Bench 객관식 변형 | MVBench [6], VLM4D [7] 형식 | 움직임 방향·크기 질문 |

### 가까운 대안 (논문이 비교하지 않은 것)

1. **최근 프레임을 그대로 쌓기.** 영상 모델을 거치지 않고 과거 프레임을 같은 인코더에 넣는 방식입니다. 공개 코드는 TEMPO_MOT 구성에 이런 경로(SigLIP 3프레임 시간 어텐션)를 이미 담고 있습니다(`04` D3). 논문 표에는 이 변형이 없습니다.
2. **기준선에 같은 행동 이력을 주기.** 별칭 과제에서 TEMPO가 이기는 것이 "이력이 있어서"라면, VLASH에 TEMPO_ACT만 붙인 행이 가장 직접적인 대조입니다. 표 6의 ACT만 행은 Handover 하나에서만 있습니다.
3. **광류·예측 프레임으로 움직임을 주기.** F2F-AP [23]는 광류로 미래 프레임을 합성해 지연을 보상합니다. 논문은 직교적이라고만 서술하고 실험 비교는 없습니다(§2).
4. **상태 이력을 생성 시작점으로 쓰기.** 이력의 사용처가 정책 입력이 아니라 흐름 매칭의 출발점(샘플러)인 방향입니다.

## 2. 이 연구가 놓인 자리

- 질문은 "동적 조작에서 VLA의 병목이 지연인가 표상인가"입니다. 비동기 방법(RTC, VLASH)이 지연을 줄여도 마르코프 가정은 남는다는 논거로 시작합니다.
- 두 입력은 모두 정책이 이미 받는 데이터(과거 프레임, 자신의 명령)에서 만듭니다. 그래서 새 센서 없이 얹을 수 있다는 점이 설계의 장점입니다.
- 다만 "백본에 손대지 않는다"는 서술과 공개 코드의 구조가 다릅니다(`04` D2, D3).

## 4. GitHub와 자료 표

| 자료 | 관계 | 라이선스 | 별 | 마지막 push | 커밋 | 비고 |
|---|---|---|---|---|---|---|
| [tempo-robot/TEMPO](https://github.com/tempo-robot/TEMPO) | **공식 코드** (openpi 포크, PyTorch) | Apache-2.0 (+ Gemma 약관 `LICENSE_GEMMA.txt`) | 26 | 2026-10-15 | `dd8342a1234e7166b856ffec2d7917b51cce05e4` (20커밋) | 정적 읽기만(`repo/TEMPO`). README TODO: 학습 데이터, 체크포인트, YAM 배포 미공개 |
| [tempo-robot/TempoBench](https://github.com/tempo-robot/TempoBench) | **공식 데이터** (TEMPO-Bench) | API는 NOASSERTION. `LICENSE`는 "All rights reserved ... internal research use" 자리표시자 | 5 | 2026-08-07 | `7d1d1d4ff7dc3c6ab17590187e108fe663676346` (3커밋) | 클론 25 MB(영상 514개 약 750 MB는 Git LFS, 받지 않음) |
| [프로젝트 페이지](https://tempo-robot.github.io/) | 공식 | 페이지 템플릿 CC BY-SA 4.0(푸터) | - | - | - | 논문, 코드, 데이터 링크와 롤아웃 영상 |
| [mit-han-lab/vlash](https://github.com/mit-han-lab/vlash) | 백본·기준선 VLASH [5] | Apache-2.0 | 526 | 2026-04-22 | - | TEMPO 저장소에 포함되지 않음 |
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | TEMPO 코드의 기반 | Apache-2.0 | 14,742 | 2026-08-24 | - | π0, π0.5 구현 |
| [facebookresearch/sam2](https://github.com/facebookresearch/sam2) | TEMPO_MOT의 SAM 2.1-tiny | Apache-2.0 | 20,213 | 2026-05-30 | - | TEMPO README가 별도 clone을 요구 |
| [DeFisch/UVT](https://github.com/DeFisch/UVT) | TEMPO+UVT의 UVT [30] | 없음(API null) | 3 | 2026-03-11 | - | 같은 연구실 |
| [i2rt-robotics/i2rt](https://github.com/i2rt-robotics/i2rt) | YAM 로봇 SDK | MIT | 184 | 2026-11-23 | - | 논문 [38]의 하드웨어 |

- RTC는 공개 코드가 없어 논문 저자가 "논문대로 충실히 다시 구현했다"고 합니다(부록 B.3). 그 구현은 공개되지 않았습니다.

## 5. 학회와 라이선스 확인

- **학회:** CoRL 2026 구두. corl.org 공식 채택 목록(2026-11-14 수집본)에서 제목이 정확히 일치하고 발표 형식이 oral, 세션 O2, 월요일 11/9, 포스터 48번으로 나옵니다. arXiv 코멘트와 프로젝트 페이지의 "Oral" 표기도 같습니다.
- **논문 라이선스:** arXiv abs 페이지(2026-11-23 확인)의 "view license" 링크가 `http://creativecommons.org/licenses/by-nc-sa/4.0/`입니다. **CC BY-NC-SA 4.0**입니다. 따라서 이 분석의 공개본은 논문 그림을 쓰지 않고 우리가 그린 그래프만 씁니다. 표의 수치는 인용과 재계산을 위해 옮겼습니다.
- **PDF 권리 표기:** `paper/paper.txt`에서 ©, copyright, All rights reserved를 찾지 못했습니다.
- **공식 코드 라이선스:** TEMPO는 Apache-2.0이지만 PaliGemma/Gemma 약관(`LICENSE_GEMMA.txt`)과 π0.5 가중치(Physical Intelligence 배포)에는 별도 조건이 붙습니다. TempoBench는 LICENSE가 자리표시자여서 사용 조건이 확정되지 않았습니다(`04` D19).
