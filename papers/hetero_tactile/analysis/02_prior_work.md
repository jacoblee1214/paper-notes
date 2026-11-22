# 02. 선행연구, 공개 자원, 학회 상태

## 0. 학회와 발표 상태 (확인 2026-11-22)

- arXiv 메타데이터(v2) comments: "15 pages, 5 figures. Paper has been accepted to CoRL 2026". 논문 PDF 첫 쪽에 "10th Conference on Robot Learning (CoRL 2026), Austin TX, USA" 표기가 있습니다(v2에서 추가).
- **CoRL 2026 공식 채택 목록에 있습니다.** CoRL 2026 공식 채택 목록에서 제목 일치, 포스터(Poster post-p2, 2026-11-10 화 17:00–18:30, 포스터 번호 141), OpenReview 포럼 `0DopZPPLvw`로 확인했습니다.
- 프로젝트 페이지와 GitHub README도 "Accepted CoRL 2026"로 표기합니다. 판정: `venue: "CoRL 2026"`, 포스터.

## 1. 선행연구 계보 (구성 요소별 출처)

| 구성 요소 | 가까운 선행 | HTT가 하는 일 |
|---|---|---|
| 마스크 복원 사전학습 | MAE (He 외 2022), Sparsh (Higuera 외, CoRL 2024), T3 (Zhao 외, CoRL 2024) | 센서마다 MAE 복원 손실을 쓰고 인코더는 이 손실로만 갱신 |
| 센서별 인코더 + 공유 트렁크 | T3 (같은 구조의 광학식 전용 버전, 공식 코드의 `shared_trunk.py`는 T3에서 가져옴) | 배열식 센서를 같은 트렁크에 붙임 |
| 센서 불변 표현 | SITR (Gupta 외, ICLR 2025, 합성 데이터), AnyTouch (Feng 외, ICLR 2025, 정적·동적, 다중 광학 센서), AnyTouch 2 (ICLR 2026) | 광학 대 배열이라는 더 큰 이질성을 쌍 데이터로 다룸 |
| 예측 기반 정렬 | I-JEPA·V-JEPA 계열의 마스크 임베딩 예측, stop-gradient 목표(BYOL, SimSiam 계열) | 교차 센서 예측기로 가려진 목표 임베딩을 소스에서 예측 |
| 쌍 촉각 수집 | UMI (Chi 외, RSS 2024), Octopi 1.5, UniForce (Chen 외, 2026) | 양 손가락에 다른 센서를 달아 파지 단위 쌍을 모음 |
| 이종 촉각 | UniTac-NV (비시각 센서), 촉각 간 번역(Grella 외, CASE 2025), 지도 학습 융합(Gao 외, IROS 2020) | 자기지도 + 쌍 정렬 + 하류 전이 평가 |
| 시뮬 평가 | ManiFeel (Luu 외, 2025) | peg 삽입, bulb 설치 |

가까운 대안과의 차이: AnyTouch는 광학 센서만 다루고 동적 정보(영상)와 텍스트·시각 정렬을 쓰며, SITR는 합성 데이터로 센서 불변성을 얻습니다. 둘 다 배열식 센서를 받지 못합니다. HTT의 새로운 점은 배열식 입력과 쌍 정렬이지만, 절제(SPH, MAE) 결과로 보면 개선의 큰 부분은 구조에서 옵니다(`03` V3).

## 2. 코드와 저장소 (GitHub 표, 확인 2026-11-22)

| 자원 | 주소 | 라이선스 | 별 | 마지막 push | 비고 |
|---|---|---|---|---|---|
| 공식 코드 | github.com/jxbi1010/HTT | CC BY-NC 4.0 (LICENSE와 NOTICE; GitHub API는 NOASSERTION) | 8 | 2026-09-05 | 커밋 `2e5b0b1`, 얕은 복제 (정적 읽기만). 학습·평가·베이스라인 로더·예제 포함 |
| 프로젝트 페이지 | jxbi1010.github.io/htt-gh-page | 불명 (htt-gh-page 저장소에 라이선스 없음) | - | 2026-09-05 | 논문, 코드, 데이터 링크 |
| 데이터셋과 체크포인트 | huggingface.co/datasets/AllenBi21/HTT-dataset | CC BY-NC 4.0 (카드) | - | 2026-09-03 | 데이터 4.41 GB, 체크포인트 `htt_4sensors_best.pth` 약 69 MB (받지 않음), 다운로드 1,464회 |
| 모델 저장소 | huggingface.co/AllenBi21/HTT | CC BY-NC 4.0 | - | 2026-09-03 | 같은 체크포인트 |
| T3 | github.com/alanzjl/t3 | MIT | 64 | 2024-06-21 | 공식 코드의 `shared_trunk.py` 출처 |
| AnyTouch | github.com/GeWu-Lab/AnyTouch | MIT | 97 | 2026-01-13 | 비교 기준선 |
| Sparsh | github.com/facebookresearch/sparsh | NOASSERTION | 225 | 2025-02-27 | 관련 |
| ManiFeel | github.com/purdue-mars/manifeel | MIT | 68 | - | 시뮬 평가 벤치마크 |
| UMI | github.com/real-stanford/universal_manipulation_interface | MIT | 1,647 | 2026-05-26 | 쌍 수집 장비의 기반 |

- 실로봇(Sharpa 손, ACT 정책)과 ManiFeel 평가 코드, 그림 3의 Procrustes 검색 코드는 공식 저장소에 없습니다(`04` D13, D14).
- 공식 코드 라이선스는 비상업적 연구 용도입니다. V-JEPA 파생 파일(`utils/__init__.py`)이 CC BY-NC 4.0이라 저장소 전체가 같은 조건입니다(NOTICE).

## 4. 참고: 이 논문의 주장이 기대는 선행 결과

- "광학 전용 사전학습은 힘 과제를 보존하지 못한다"(§5.3)는 T3 9DTact 힘 MAE 2.678 대 Scratch 1.255와 SITR GSMini 1.373 대 1.260에 기댑니다. 이 두 칸은 T3·SITR가 센서 분포 밖(9DTact은 T3 사전학습에 없음)에서 평가된 결과라 "사전학습 일반"의 증거로는 약합니다.
- "배열 센서 쪽의 힘 단서와 정렬해 슬립 표현이 낫다"는 설명(§5.3)은 TAC-02 한 칸(+12.0, 시드 SD 7.27)에 기댑니다(`03` V4).
