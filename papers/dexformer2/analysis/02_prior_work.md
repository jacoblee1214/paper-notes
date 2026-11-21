# 02. 선행연구, 가까운 대안, 공식 저장소

## 1. 계보: 이 논문이 기대는 흐름

| 흐름 | 대표 연구 | 이 논문과의 관계 |
|---|---|---|
| 손 재배치와 시뮬→실 전이 | Andrychowicz et al. 2020 (도메인 무작위화, [1]), DeXtreme [9], RMA 손 회전 [18] | 시뮬 학습과 무작위화의 기본 틀. 이 논문은 조합 축을 추가 |
| 물체 일반 파지 | UniDexGrasp [28], UniDexGrasp++ [24], UniGraspTransformer [26], RobustDexGrasp [32], GraspXL [31] | 물체 일반화를 다룸. 이 논문은 같은 물체 집합에서 **임베디먼트**를 바꿈 |
| 형태 무관 제어 (locomotion) | NerveNet [25], Huang et al. 2020 [10], MetaMorph [8], My body is a cage [12], URMA / One Policy to Run Them All [3], Bohlinger & Peters 2025 [2] | 관절 토큰 어텐션, 형태 조건화의 원류. 이 논문의 MorphNode와 같은 발상 |
| 적응형 이력 정책 | RMA [11], Transformer-XL [6], GRU [5] | 이력 조건화. TXL은 이 논문의 메모리, GRU는 기준선 |
| 임베디먼트 교차 파지 | CrossDex [30] (팔 고정, 인간 손 고유 파지 행동), GET-Zero [17] (팔 없음, 손 설계 간 전이), DexGrasp-Zero [27] (손목 데카르트, 형태 정렬 그래프), **DexFormer 1.0** [33] | 가장 가까운 선행. DexFormer 1.0은 같은 저자, 이력 조건이지만 형태 descriptor 없음 |
| 그래프 구조 인코딩 | Graphormer 계열 [29] | 트리 거리 편향 |
| 시뮬 도구, 원칙 | Isaac Lab [14], PPO [20], DAgger [19], t-SNE [23], YCB 물체 [4] | 학습, 증류, 평가에 쓴 도구 |

**두 축이 합쳐진 자리.** 기존 임베디먼트 교차 파지는 팔 또는 손을 고정하거나 손마다 행동 매핑을 따로 둡니다. 형태 무관 locomotion은 팔을 뺀 다리 로봇을 다룹니다. 이 논문은 팔과 손을 같이 바꾸고, 정준 관절 슬롯과 descriptor로 이 둘을 한 정책에 담습니다.

**가장 가까운 대안을 고른 기준.** 같은 저자 계열(DexFormer 1.0), 같은 문제(교차 임베디먼트 파지), 공개된 코드 유무. DexFormer 1.0 코드는 찾지 못했습니다(2026-11-21, GitHub 검색).

## 2. 공식 저장소와 관련 저장소 (GitHub 표)

| 저장소 | 역할 | 라이선스 | stars | 마지막 push | 상태 |
|---|---|---|---|---|---|
| [DavidLXu/dexformer2](https://github.com/DavidLXu/dexformer2) | 공식. 영상 렌더 코드, 기록 궤적, 매니페스트 | API "Other". LICENSE 파일은 MIT (저장소가 쓴 코드). 3자 자산은 각자 라이선스 | 3 | 2026-11-14 | 얕은 clone(`repo/`, 커밋 d964967), 정적 읽기 |
| [DavidLXu/dexformer2-web](https://github.com/DavidLXu/dexformer2-web) | 프로젝트 웹사이트 소스 | 없음 | 0 | 2026-11-19 | 클론 안 함 |
| [DavidLXu/DexFormer-web](https://github.com/DavidLXu/DexFormer-web) | DexFormer 1.0 웹사이트 | 없음 | 0 | 2026-02-25 | 클론 안 함 |
| [프로젝트 페이지](https://davidlxu.github.io/dexformer2-web/) | 영상, 요약 | - | - | - | "Code (soon)" |
| [PKU-RL/CrossDex](https://github.com/PKU-RL/CrossDex) | CrossDex [30] 공식 | (확인 안 함) | 90 | 2025-06-12 | 클론 안 함 |
| [YliangWu/DexGrasp-Zero-code](https://github.com/YliangWu/DexGrasp-Zero-code) | DexGrasp-Zero [27] 코드 | (확인 안 함) | 19 | 2026-08-12 | 클론 안 함 |

- **공식 저장소의 범위.** README: "This repository currently holds only the video-rendering code for the DexFormer 2.0 demonstration films, with the recorded trajectories they replay. Training code, the policy, evaluation scripts and model weights are not here yet."
- 정책 체크포인트 매니페스트는 있으나 `included: false`입니다. 공개 URL은 없습니다.
- 이 저장소의 궤적 데이터는 CC BY 4.0이고, 데이터 설명서에 따르면 시뮬레이터(Isaac Sim)로 기록한 관절과 물체 상태입니다.
- 자세한 대조와 정적 확인은 `04_claim_verification.md`에 있습니다.

