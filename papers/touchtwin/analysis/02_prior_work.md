# 02. 선행연구와 관련 저장소

이 문서는 TouchTwin의 구성 요소가 어디서 왔는지, 가까운 대안이 무엇인지, 공식 코드가 있는지를 정리합니다. 번호는 논문 참고문헌 저자와 연도로 적었습니다(`paper/paper.txt`의 References).

## 1. 선행연구 계보 (구성 요소별 출처)

| TouchTwin의 요소 | 출처 (논문 인용) | 관계 |
|---|---|---|
| 사람 영상 → 시뮬 재생으로 데이터 증폭 | Wang 2026 (DexAgent), Mao 2026 (OmniHOI), Choi 2026 (Grasp-and-lift), Dan 2025 (X-Sim, CoRL 2025), Allshire 2025 (CoRL) | 논문 §2가 모두 상호작용을 **영상에서 복원한다**고 봅니다. TouchTwin은 같은 틀에 **측정된 촉각**을 넣습니다 (§1) |
| 재생 조각을 섞어 데이터 증폭 | MimicGen (Mandlekar 2023, CoRL), DexMimicGen (Jiang 2025, ICRA) | 궤적 조각을 재배열하는 방식입니다. TouchTwin의 섭동은 물리를 다시 푼다는 점이 다릅니다 (§2.6) |
| 장갑으로 촉각 기록 | Sundaram 2019 (Nature, 장갑), Yang 2022 (Touch and Go), Song 2025 (OpenTouch), Li 2026 (TouchScale) | 기록 데이터의 출처입니다. TouchScale 500시간 중 일부를 씁니다 |
| 촉각으로 손-물체 동역학 모델링 | Zhang 2021 (IROS) | 촉각을 동역학 식별에 씁니다. TouchTwin은 촉각으로 시뮬 접촉을 직접 정합니다 |
| 영상에서 촉각 예측 | TouchAnything (Zhou 2026b), TouchSight (Zhou 2026a), EgoTac (Zhang 2026a), EgoTactile (Zeng 2026a, ICML 2026) | 같은 픽셀에서 장갑 신호를 재현하도록 학습합니다. 힘이 픽셀에 없으면 복원하지 못한다고 TouchTwin이 지적합니다 (§1) |
| 촉각으로 손 영상 생성 | Glove2Hand (Zhang 2026b, CVPR 2026), Dexterous tactile world model (Zeng 2026b) | 촉각 조건의 영상 생성입니다. TouchTwin은 시뮬 재생을 조건으로 씁니다 |
| 촉각을 로봇 정책 감독에 씀 | Luo 2026 (Touch2Robot), Ni 2026 (TactiDex), Yu 2024 (MimicTouch, CoRL 2024), Xu 2025 (DexUMI, CoRL 2025), N0-VTLA (NeoteAI·Fudan TEAI 2026) | N0-VTLA는 TouchTwin의 정책 기준선입니다 (표 3) |
| 물체 디지털 트윈에 촉각 | ForceTwin (Engelbracht 2026) | 가장 가까운 "힘이 있는 디지털 트윈"입니다. 물체 쪽 식별이고 손 쪽 촉각 재생은 아닙니다 |
| 물리 엔진으로 영상 생성 조건 | GPT4Motion (Lv 2023), PhysGen (Liu 2024, ECCV), Code Video Model (2026 기술 보고서), Code World Model (Chen 2026), RoboRender (Huang 2026) | TouchTwin 표 1의 "sim control"과 같은 계열입니다 |
| 영상 생성기 | Team Wan 2025 (본문은 Wan2.2) | 논문이 인용한 arXiv:2503.20314 제목은 "Wan: Open and Advanced Large-Scale Video Generative Models"입니다. 이 제목은 Wan2.1 보고서로 알려져 있어 본문의 Wan2.2와 맞지 않습니다. 인용 번호를 확인해야 합니다 (`03` K7) |
| 물리 엔진 | MuJoCo (Todorov 2012, IROS) | 재생과 섭동에 씁니다 |
| 손 자세 추정 | WiLoR (Potamias 2024) | 손 모델을 영상에 맞추는 단계에서 씁니다 |

**가까운 대안과의 차이.** 같은 데이터 흐름(사람 영상 → 시뮬 → 정책)에서 TouchTwin이 바꾼 것은 **힘의 출처**입니다. 영상 복원 파이프라인(Dan 2025, Mao 2026 등, 논문 §2의 서술)에서는 접촉 힘이 영상에서 정해지지 않습니다. TouchTwin에서는 장갑 기록이 접촉과 닫힘 힘을 정합니다. 앞의 것들이 실제로 어떤 힘을 내는지는 이 분석에서 확인하지 않았습니다. 둘 사이의 직접 비교는 논문에 없습니다. 표 3의 "vision-based replays"는 같은 파이프라인에서 촉각만 뺀 대조이며, 다른 방법(예: Dan 2025, Mao 2026)과의 비교가 아닙니다.

## 2. 코드와 저장소 (GitHub 표)

논문과 프로젝트 페이지에 **공식 코드 링크가 없습니다.** 프로젝트 페이지 링크는 arXiv PDF와 저자 페이지뿐입니다(확인: `touchtwin.github.io` HTML). 아래는 GitHub 검색 결과입니다.

| 저장소 | 관계 | 라이선스 | 최근 push | 상태 |
|---|---|---|---|---|
| [TouchTwin/touchtwin.github.io](https://github.com/TouchTwin/touchtwin.github.io) | 프로젝트 페이지 소스(GitHub Pages). 이 조직의 유일한 저장소 | 없음 (API: null) | 2026-11-20 | 코드 아님. 클론하지 않음 |
| [Yuvraj06Jain/TouchTwin](https://github.com/Yuvraj06Jain/TouchTwin) | **이름만 같은 다른 프로젝트.** 휴대폰 화면을 노트북 터치패드로 쓰는 Rust 프로그램(KDE Plasma, Wayland) | GPL-3.0 | 2026-11-19 | 논문과 무관. 클론하지 않음 |
| TouchScale 데이터 (Li 2026, arXiv:2610.10288) | 기록의 출처. 프로젝트 페이지 링크는 arXiv 메타데이터에만 있음 | 확인 못 함 | — | 데이터 다운로드 안 함 (500시간 규모) |

- **공식 코드가 없으므로 `repo/` 클론은 하지 않았습니다.** 그래서 `04_code_verification.md`가 아니라 `04_claim_verification.md`를 씁니다.
- "TouchScale" 이름으로 된 GitHub 저장소가 여럿 있으나(예: ashertrockman/TouchScale, phai-lab/touchscale), 논문 저자와 관련 있는지 확인하지 못했습니다. 표에 넣지 않았습니다.
- 가중치, 데이터, 손 모델 파일은 공개 여부를 확인하지 못했습니다.

## 4. 참고: 이 논문의 주장이 기대는 선행 결과

- **촉각 장갑이 접촉 힘을 기록한다**는 가정은 Sundaram 2019와 Song 2025의 장갑 기록 방식에 기댑니다. 880 taxel의 힘 지도 값이 실제 힘과 어떻게 맞는지는 TouchTwin이 검증하지 않았습니다(Fig. 2 e는 장갑 힘 지도와 시뮬 촉각의 대조를 보여 주지만, 보정 방법은 없습니다).
- **시뮬 손 피부 접촉**(Fig. 9)은 물리 엔진의 접촉 모델에 기댑니다. MuJoCo의 접촉 파라미터(마찰, 접촉 강성)가 얼마나 민감한지는 논문에 없습니다.
- **섭동 결과**는 MuJoCo 물리에 의존합니다. 같은 실험을 다른 엔진에서 하면 결과가 달라질 수 있습니다.
