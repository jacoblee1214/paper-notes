# 02. 선행연구와 관련 저장소

## 1. 선행연구 계보 (구성 요소별 출처)

| 구성 요소 | 논문이 기대는 출처 | 이 논문에서의 위치 |
|---|---|---|
| 촉각 반사 제어기(교사) | TactileReflex [1] (같은 저자 그룹, arXiv:2605.23568, IROS 2026), 고전적 미끄러짐 검출·방지 [2][3][4][8] | 수집 때 그리퍼를 25 Hz로 조절하는 **교사**. 부록 C는 제어 논리를 바꾸지 않고 실행 파이프라인만 고쳤다고 적고, 새 알고리즘 기여로 세지 않습니다 |
| 교사-학생과 특권 정보 | Vapnik의 특권 정보 학습 [24], Learning by Cheating [25], 사족 보행 교사-학생 [26], RMA [27], 시각 기반 보행 [28] | "증류"라는 표현은 해석상의 비유라고 §2가 한정합니다. 교사는 학습된 정책이나 더 풍부한 관측이 아니라 **공학적으로 설계한 결정론적 제어기**이고 수집 때만 존재합니다 |
| 학생 정책 | ACT [5], Diffusion Policy [6], π0.5 [7] | 모두 촉각 없는 입력(RGB 2개 + 7차원 로봇 상태). 실험은 ACT(주), π0.5(교차 확인) |
| 촉각을 정책 입력으로 | 3D-ViTac [10], KineDex [11], MimicTouch [13], VTLA [14], VLA-touch [15], Tactile-VLA [16], OmniVTLA [17], DreamTacVLA [18], TaF-VLA [19], TacVLA [20], Calandra [9] | 대비 대상. 이들은 삽입·도구 사용 등에 촉각을 모달리티로 넣습니다. 이 논문의 부록 A는 단순한 촉각 이미지 추가가 나아지지 않았다는 정성 서술만 있습니다 |
| 빠른 반응 층과 안전 층 | RDP [29], ImplicitRDP [30], CompliantVLA-adaptor [31], VLSA [32], SafeDiff [33], Safe-Night VLA [34] | 배포 때 중재기(알고리즘 1)가 이 계열과 비슷한 위치입니다. 차이는 같은 제어기가 수집 때도 쓰였다는 점이라고 §2가 정리합니다. 각주 2는 RDP 계열의 빠른 층이 "학습된 통계 모델"이라는 점을 대비로 듭니다 |
| 촉각 없이 추론 | HapticVLA [23] (동시기, 보상 가중 행동 전문가에서 촉각 토큰을 증류) | 가장 가까운 동시기 연구로 인용 |
| 원격조작과 수집 도구 | DexUMI [21], TypeTele [22], LeRobot [35] | 수집 도구 연구와 달리 이 논문은 "촉각 반사가 수집을 바꾼다"에 초점. LeRobot 호환 형식으로 기록 |
| 센서 | MC-Tac [36] | 이중 카메라 기반 촉각 센서, 25 Hz 루프의 입력 |
| 동역학 손실의 동기 | CHOMP [37]의 궤적 평활 비용 | 2차 차분 일치 항(§3.2)을 "궤적 최적화의 평활 비용과 비슷하다"고 소개 |

**논문에 비교가 없는 가까운 개념(이 분석자의 정리).** (1) 사람과 자동 제어기가 제어를 나눠 갖는 보조 원격조작(shared autonomy)은 인용되지 않았습니다. 반사 모드의 리더 그리퍼는 이진 의도 신호이고 제어기가 세기를 정하므로 이 틀과 가깝습니다. (2) 같은 반사 제어기로 모은 데이터를 **수동 데이터와 섞거나** 반사 데이터를 늘려(30개 이상) 데이터 규모 곡선을 그리는 비교가 없습니다. (3) 힘 센서가 있는 장비로 힘 목표를 직접 기록하는 힘 중심 시연(촉각 대신 F/T 기반)과의 비교가 없습니다.

## 2. 코드와 저장소 (GitHub 표)

기준일 2026-11-26. 스타 수, 라이선스(API의 `license` 값), 마지막 push는 GitHub API 값입니다.
**이 논문의 학습·평가 코드(ACT/π0.5 학습, 중재기, 동역학 손실, 평가 스크립트)는 찾지 못했습니다.** 프로젝트 페이지에는 코드 링크가 없습니다. 교사 제어기의 저장소는 공개되어 있어 정적으로 읽었습니다(`04`).

| 저장소 | 관계 | 라이선스 | 스타 | 마지막 push | 상태 |
|---|---|---|---|---|---|
| [shayfeng/tactile-reflex-open-source](https://github.com/shayfeng/tactile-reflex-open-source) | **교사 제어기 구현**(TactileReflex [1]의 MC-TAC 제어기, 약 25 Hz 최적화판 `tactile_exp_v2.py`, DM-TAC·Xense 백엔드). 이 논문의 학습·평가·중재기 코드는 없음 | 없음(API license null, LICENSE 파일 없음) | 2 | 2026-10-10 | 커밋 `a3699b579555e03f1a12328db341cfd1da7014dd`(2026-10-10 "Align MC-TAC vertical slip proxy and calibration", 크기 65 KB)를 얕게 복제해 정적으로만 읽음(`repo/`, 실행·설치 안 함). 이 논문의 arXiv v1(2026-09-22)보다 뒤의 커밋입니다 |
| [shayfeng/TactileReflex](https://github.com/shayfeng/TactileReflex) | TactileReflex 프로젝트 웹사이트 저장소(HTML 등, 약 27 MB) | 없음 | 0 | 2026-10-10 | 링크 목록만 확인. 복제하지 않음 |
| 프로젝트 페이지 [shayfeng.github.io/better-curriculum](https://shayfeng.github.io/better-curriculum/) | 이 논문의 프로젝트 페이지(저자 개인 사이트 저장소 shayfeng/shayfeng.github.io 안, 영상은 Bilibili). 코드·데이터 링크 없음 | 없음 | 0 | 2026-11-22 | HTML 한 장을 읽음 |
| [shayfeng/SO101_Piper_Teleop](https://github.com/shayfeng/SO101_Piper_Teleop) | 같은 계정의 SO101 리더 + Piper 팔로워 원격조작 패키지(LeRobot 복사본 포함). 논문이 인용하지 않아 이 논문의 수집에 쓰였는지는 확인할 수 없음 | 없음 | 0 | 2026-07-04 | 파일 목록과 README만 확인 |
| [shayfeng/pi05_xense_tactile_fusion](https://github.com/shayfeng/pi05_xense_tactile_fusion), [shayfeng/TacO-Xsense](https://github.com/shayfeng/TacO-Xsense) | 같은 계정의 **다른 연구선**: Xense 촉각 센서를 쓰는 ACT, π0.5 촉각 융합 기록. 이 논문의 센서(MC-Tac)와 과제와 다름 | 없음 | 0, 1 | 2026-08-18, 2026-08-16 | 관련 없음으로 판단, 읽지 않음 |
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | π0.5 학습·추론 기반 [7] | Apache-2.0 | 14,766 | 2026-08-24 | 참고 |
| [huggingface/lerobot](https://github.com/huggingface/lerobot) | 데이터 형식 [35] | Apache-2.0 | 29,090 | 2026-11-26 | 참고 |
| [tonyzhaozh/act](https://github.com/tonyzhaozh/act) | ACT [5] | MIT | 2,355 | 2024-07-23 | 참고 |
| [agilexrobotics/piper_sdk](https://github.com/agilexrobotics/piper_sdk) | Piper 팔·그리퍼 SDK | MIT | 425 | 2026-10-14 | 교사 제어기의 의존성 |
| [Tacxels/MC-Tac](https://github.com/Tacxels/MC-Tac) | MC-Tac 센서 설계 [36] | BSD-3-Clause | 22 | 2023-09-21 | 참고 |

- 수집한 데이터(반사 30, 수동 100 풀, 접촉 선별 100), 학습된 가중치, 평가 영상은 공개 링크가 확인되지 않습니다(2026-11-26).
- 교사 저장소의 README는 "논문 실험 이후의 개선과 센서 확장을 포함하며 논문과 구현 세부·보정 임계값이 다를 수 있다"고 적습니다(`04` §0).

## 4. 가까운 대안과 비교 공백

논문이 비교하지 않은 것들입니다. 이 분석이 결과를 가정하지 않고 빈 칸으로만 적습니다.

- 같은 수집 세션에서 번갈아 모은 반사 데이터와 수동 데이터(순서 무작위화)
- 반사 데이터에 사람이 한 번 더 걸러 낸(영상 선별) 데이터와 수동 선별 데이터의 대조(두 풀의 선별 조건을 맞춤)
- 같은 재질의 주기로 돌린 촉각 없는 정책(예: 그리퍼 청크를 짧게 한 π0.5 또는 ACT)과의 외란 대조
- 촉각을 입력으로 쓰는 정책을 같은 반사 데이터로 학습한 경우(부록 A의 촉각 이미지 변형이 어떤 시연으로 학습했는지는 서술이 없음)
