# 02. 선행연구와 관련 저장소

이 문서는 논문의 구성 요소가 어디서 왔는지, 가까운 대안이 무엇인지, 공식 코드가 있는지를 정리합니다. 번호는 논문 참고문헌 번호(`paper/paper.txt`의 References)입니다.

## 1. 선행연구 계보 (구성 요소별 출처)

| 논문의 요소 | 출처 (논문 번호) | 관계 |
|---|---|---|
| 이진 촉각 신호로 시뮬-실제 격차 줄이기 | DexTouch [25] (Lee 외, RA-L 2024), Rotating without Seeing [3, 26] (Yin 외, RSS 2023) | 논문이 직접 "following prior work [25]"라고 밝히는 출발점입니다(§3.1). 본문 소개(§1)는 같은 주장을 [8, 9]에 걸어 두는데, 두 논문 제목(Taccel, Taxim)은 시각 기반 촉각 시뮬레이터입니다(`03` K6) |
| 촉각 시뮬레이터와 연속 센서 보정 | Taccel [8], Taxim [9], Ding 외 [20], Lach 외 [21], Chen 외 [22], Narang 외 [23], Church 외 [6, 24] | 연속 신호 시스템 식별 계열. 논문은 "수십 개 이종 센서에는 확장이 어렵다"고 봅니다(§2) |
| 촉각 사건 시뮬(SDF + 벌점 압력) | Kaolin [38], Xu 외 [40](미분 가능 촉각 시뮬, CoRL 2022) | 식 1의 노드 압력과 SDF 질의의 기반 |
| 지수 전파 | Vinogradova & Feuillebois [39] | 탄성체 표면력의 지수 감쇠를 식 1의 교차 반응 모델로 인용 |
| 센서: TwinTac, LEAP Hand, 손바닥 전체 촉각 | TwinTac [37](같은 연구실, IROS 2025), LEAP Hand [36], 촉각 피부 [41] (Zhao 외, Nature MI 2025) | 하드웨어는 같은 연구실의 이전 센서를 곡면으로 개조 |
| 센서 3D 위치를 붙인 촉각 표현 | Spatially anchored tactile [34], 3D-ViTac [35] | 고해상도 지문 센서의 손 운동학 정렬. 논문은 이를 "희소 이진 센서에는 직접 적용 안 됨"으로 구분합니다(§2) |
| 특권 정보 사전학습 | 시뮬 전용 정답으로 인코더 사전학습(§3.2) | RMA 계열의 비대칭 학습과 점군 예측 사전학습의 조합. 논문은 이 계열을 인용하지 않고 형상·접촉 감독으로 설명합니다 |
| 전문가 RL → 증류 | PPO 전문가 + Diffusion Policy [10] (Chi 외), BC·ACT [42] | 물체별 RL 전문가의 성공 궤적 모음을 단일 확산 정책으로 증류 |
| 시뮬 | MuJoCo Playground [43] | 병렬 PPO 학습과 평가 |
| 파지 합성과 형상 | DFC [27], BODex [28], Dexonomy [29], T(R,O) Grasp [30] | 시각 정책이 전역 형상을 쓴다는 배경(§2) |
| 이진 촉각 문제 | Dang 외 [4] (blind grasping, 2011), 파지 안정성 [1, 7] | 시각 없는 파지의 고전 동기 |

**가까운 대안과의 차이.** 같은 이진 촉각 + 시뮬 학습 계보의 직전 연구는 DexTouch [25]입니다. 이 논문은 (1) 이진 사건의 시작·해제 시점을 센서마다 맞추는 보정, (2) 택셀 위치를 붙인 인코더와 사전학습, (3) 물체별 전문가 → 확산 정책 증류를 추가합니다. 세 요소를 따로 떼어 실로봇에서 비교한 결과는 (2)의 사전학습 한 가지뿐입니다(표 5). 보정의 정책 수준 효과와 증류 방식(확산 정책 대 단일 RL 정책)의 실로봇 비교는 없습니다(`03` K2, `04` C3, C9).

## 2. 코드와 저장소 (GitHub 표)

| 저장소 | 관계 | 라이선스 | 별 / 최근 push | 상태 |
|---|---|---|---|---|
| (공식 코드 없음) | 프로젝트 페이지의 Code 버튼이 `#`입니다(2026-11-25 확인). `gh search`(제목, 프로젝트 주소 문구)로도 찾지 못했습니다 | - | - | clone 없음 |
| [Dex-Blind-Grasp/Dex-Blind-Grasp.github.io](https://github.com/Dex-Blind-Grasp/Dex-Blind-Grasp.github.io) | 프로젝트 페이지 소스(Astro). 코드가 아닙니다. 하드웨어 저장소는 "coming soon" | API 기준 없음 | 최근 push 2026-09-06 (`761643d`) | 읽기만 |
| [OrigamiOuO/twintac](https://github.com/OrigamiOuO/twintac) | TwinTac 디지털 트윈(Isaac Gym Flex). 논문의 [37]과 같은 계열 이름이지만 작성자가 "TwinTac Contributors"라서 저자 소속 저장소인지 확인하지 못했습니다 | MIT | 4 / 2026-06-30 (`a99eecb`) | 읽기만, clone 안 함 |
| [google-deepmind/mujoco_playground](https://github.com/google-deepmind/mujoco_playground) | 시뮬레이션과 병렬 PPO의 기반 [43] | Apache-2.0 | 2,342 / 2026-11-24 | 공개 |
| [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy) | Diffusion Policy [10] | MIT | 4,745 / 2024-12-24 | 공개 |
| [leap-hand/LEAP_Hand_API](https://github.com/leap-hand/LEAP_Hand_API), [LEAP_Hand_Sim](https://github.com/leap-hand/LEAP_Hand_Sim) | LEAP Hand [36] 구동과 시뮬 | API는 판정 불가(NOASSERTION), Sim은 MIT | 179 / 2025-10-20, 216 / 2024-05-18 | 공개 |
| [NVIDIAGameWorks/kaolin](https://github.com/NVIDIAGameWorks/kaolin) | SDF 질의 도구 [38] | Apache-2.0 | 5,200 / 2026-11-23 | 공개 |
| [tonyzhaozh/act](https://github.com/tonyzhaozh/act) | ACT 기준선 [42] | MIT | 2,355 / 2024-07-23 | 공개 |

- 논문 소유 코드, 체크포인트, 데이터, 시뮬 자산(물체 메시, 손 모델 보정 값)은 공개되어 있지 않습니다. 프로젝트 페이지의 "Open-source hardware repository (coming soon)"가 유일한 약속입니다.
- 이 분석은 어떤 저장소도 복제하지 않았습니다(공식 코드가 없고, 관련 저장소는 구현 가능성 판단에 필요한 메타데이터만 확인).

## 4. 참고: 이 논문의 주장이 기대는 선행 결과

- "이진 촉각이 연속 촉각보다 전이가 쉽다"는 주장은 DexTouch [25]와 Rotating without Seeing [26]의 선행 결과와 부록 C.2의 시뮬 비교(표 6: 59.2% 대 60.4%, 차이 구분 불가)에 기댑니다. 표 6은 시뮬 안에서의 비교라서 "전이가 쉽다"는 부분(실로봇)을 직접 시험하지 않습니다.
- "센서 위치 정보가 도움이 된다"는 부록 C.4(표 8: 36.2 대 33.8 seen, 20.0 대 19.8 unseen)에 기대지만 두 차이는 500회 가정에서 구분되지 않습니다(`03` K3).
