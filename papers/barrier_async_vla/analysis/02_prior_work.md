# 02. 선행연구, 가까운 대안, 저장소

## 1. 계보

| 흐름 | 대표 연구 (논문 참고문헌 번호) | 이 논문과의 관계 |
|---|---|---|
| 흐름 기반 VLA와 청크 | π0 [2], π0.5 [3], GR00T [1], SmolVLA [5], OpenVLA [4] | 백본. 흐름 생성기의 v_θ와 x̂_1 구조를 씁니다 |
| 비동기 추론 | 비동기 추론 [9–13]: Running VLAs at real-time speed [10], Leave no observation behind [11], 비동기 fast-slow VLA [13] | 같은 문제(관측과 실행의 어긋남)를 다룹니다 |
| 학습 기반 지연 적응 | VLASH [14] (미래 상태 인식 비동기), DeFlect [28] (지연에 강건한 실행), StreamingVLA [29], Realtime-VLA Flash [30], DynamicVLA [15], AsyncVLA [32] | 학습 파이프라인이나 구조를 바꿔야 해서 동결 VLA에 맞지 않습니다 |
| 추론 기반 연속성 (VJP) | **RTC [12]** (Black, Galliker, Levine), 안내 기법 [33, 34] | 가장 가까운 대안입니다. FreeBar는 VJP 대신 닫힌형 투영을 씁니다 |
| 장벽과 CBF | CBF 이론 [35] (Ames et al. 2019), 잠재 정책 장벽 [16] (Sun & Song 2025) | 제어 장벽 함수의 투영 형태를 빌렸습니다. 안전 보장이 아니라 정렬 제약입니다 |

RTC 원 저장소(Physical-Intelligence/real-time-chunking-kinetix, MIT)는 FreeBar의 Kinetix 실험 코드의 뿌리입니다. 공식 저장소의 `kinetix/src/model.py`에 RTC 경로(`realtime_action`, VJP 기반)가 같이 들어 있습니다.

## 2. 가까운 대안: RTC와 차이

| 항목 | RTC [12] | FreeBar (논문) |
|---|---|---|
| 정렬 방식 | 고정 접두부에 VJP로 보정 | 닫힌형 투영 λ (식 5) |
| 추론 비용 | 스텝마다 VJP (역전파 유사) | 참조와의 차이와 스칼라만 |
| 조절 변수 | 안내 세기 β | 바리어 세기 α와 감쇠율 β (본문은 β 값 없음) |
| 수평선 | 지수 감쇠 가중치 (prefix attention) | 세 구간 (강함, 부드러움, 자유) |
| 시간 감쇠 | RTC 고유 상수 | τ = 1−t로 점점 조임 |

## 4. GitHub와 코드 자료 (확인 2026-11-22, GitHub API)

| 저장소 | 무엇인가 | 라이선스 | 별 / 마지막 푸시 | 기준 커밋 | 상태 |
|---|---|---|---|---|---|
| [NamelessCrew/FreeBar](https://github.com/NamelessCrew/FreeBar) | 논문 공식 구현 (설명: "Official implementation of CoRL 2026 paper"). Kinetix 평가, GR00T 포크의 FreeBAR 액션 헤드, RoboCasa 포크 | Apache-2.0 (최상위 LICENSE) | ★2, 포크 0 / 2026-09-19 (업데이트 메타데이터 2026-11-11) | `e24dd74` (2026-09-19, "Initial release", 단일 커밋) | **얕은 복제(`repo/`)로 정적 읽기 완료.** 실행, 설치 안 함 |
| [Physical-Intelligence/real-time-chunking-kinetix](https://github.com/Physical-Intelligence/real-time-chunking-kinetix) | RTC 원 실험 코드 (Kinetix). FreeBar Kinetix 부분의 뿌리 | MIT | ★636 / 2025-12-08 | — | 링크만 (클론 안 함) |
| [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) | GR00T 원본 (FreeBar의 포크 기반) | Apache-2.0 | ★8402 / 2026-11-17 | — | 링크만 |
| [robocasa/robocasa](https://github.com/robocasa/robocasa) | RoboCasa 원본 (포크 기준 커밋 `9a3a786`) | API는 NOASSERTION. 포크 사본의 LICENSE는 MIT | ★1881 / 2026-11-18 | `9a3a786` (포크 기준) | 링크만. 자산(약 10 GB)과 `models/`는 포크에 없음 |
| 프로젝트 페이지 | 찾지 못함. `namelesscrew.github.io/FreeBar`는 404. 논문에도 링크 없음 | — | — | — | — |

메모:
- 저장소 이름 `NamelessCrew`는 익명 심사용 이름처럼 보입니다. 저자 계정과의 연결은 확인하지 못했습니다.
- 저장소의 README에는 채우지 않은 템플릿 자리가 남아 있습니다: arXiv 링크 `XXXX.XXXXX`, 인용 `Coming soon`, 연락처 `<email>`, 제목 `<Full Paper Title>`. 템플릿 주석에 "Fill in before publishing"이라고 적혀 있습니다. 논문과 저장소 이름도 FreeBar와 FreeBAR로 다릅니다.
- 실로봇 실험(컨베이어, 두더지 잡기)의 코드는 저장소에 없습니다. RoboCasa용 RTC 기준선과 표 2의 가벼운 기준선 코드도 없습니다.

## 5. 같은 CoRL 2026 목록의 비동기 실행 논문

`conferences/corl2026/papers.json`(공식 목록, 2026-11-14 수집)에서 비동기 실행 관련 제목을 찾았습니다. 이 논문과 직접 비교할 대상은 다음과 같습니다(제목 기준, 내용 미확인).

- DiscreteRTC (arXiv:2604.25050, 포스터): 이산 확산 정책에서의 비동기 실행
- F2F-AP (arXiv:2604.02408, 포스터): 흐름-대-미래 비동기 정책
- Start Right, Arrive Right (arXiv:2606.19774, 포스터): 초기 잡음 선택에 의한 비동기 실행
- Real-Time Execution with Autoregressive Policies (arXiv:2606.13355, 포스터)
- EagleVLA (arXiv:2607.12659, 스포트라이트): 전방 정렬 비동기 추론
- AsyncVLA (arXiv:2602.13476, 포스터), FlashVLA (arXiv:2608.27384, 포스터)

## 6. 참고문헌 점검 (논문 쪽)

- [2] π0가 "2026"으로 적혀 있지만 arXiv 번호 2410.24164는 2024년 제출입니다.
- [33] MiniVLA(`openvla-mini` 저장소 링크)를 VJP 기반 안내의 출처로 인용합니다. 안내 기법의 원 출처로 적기에는 맞지 않아 보입니다.
- [1] GR00T N1을 인용하지만 공식 코드는 `GR00T_N1_5` 클래스를 씁니다(README도 GR00T N1.5). 백본 버전 표기가 다릅니다.
