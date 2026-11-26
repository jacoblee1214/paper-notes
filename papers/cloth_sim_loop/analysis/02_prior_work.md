# 02. 선행연구와 관련 자료

## 1. 계보

논문(§2)은 선행연구를 상태 추정, 순방향 모델, 롤아웃 기반 행동 최적화 세 축으로 나눕니다. 그 위에 구성요소별 출처를 정리하면 다음과 같습니다.

| 구성요소 | 이 논문의 선택 | 가까운 출처 (논문 참고문헌 번호) | 이 논문이 더한 것 |
|---|---|---|---|
| 순방향 모델 = 물리 시뮬레이터 | FLASH 롤아웃 | 예측 샘플링과 MuJoCo(Howell et al. [19]), 병렬 시뮬레이터 Isaac Sim [31], Genesis [33], Newton [67] | 변형체(천)에서 시뮬레이터를 직접 순방향 모델로 닫힌 고리에 넣음. 학습 동역학 대신 시뮬레이터 호출 |
| 시뮬레이터 | FLASH (FEM + 비평활 접촉) | 같은 연구실의 FLASH [20] (arXiv:2604.17513, 저자 5명이 겹침) | 시뮬레이터 자체는 다른 논문. 이 논문은 4종 비교(표 1)와 실제 천 궤적 대조(C.1)를 더함 |
| 샘플링 최적화 | 사전 안내 MPPI | MPPI [40, 43, 44], CEM [42], 편향 제안 [45–50] | 사전 정책 주변 국소 샘플링(σ = 5 mm). 새로운 알고리즘이 아니라 구성 |
| 사전 정책 | 시뮬 안에서 스크립트 교사를 증류한 512/256/128 MLP | DAgger 계열 [16], 시뮬 교사 증류 | 천마다 한 번 증류(§3.4) |
| 상태 추정 | DINOv2 + 정점 학습 토큰 + 자기어텐션 | 범주 수준 정규 공간 GarmentNets [24], 템플릿 TRTM [26], 확산 기반 DPM(UniClothDiff [18]), 3D 가우시안 Cloth-Splatting [27] | RGB 한 장, 합성 데이터만, 고정 위상 메시의 정점 직접 회귀. 옷마다 학습 |
| 시각 특징 | 동결 DINOv2-small | DINOv2 [66], DeFM [69] | — |
| 천 폐루프 제어 | 시뮬 롤아웃으로 행동 정제 | AdaFold [54](RGB-D 입자 표현 + 학습 모델 MPC), Kotsovolis와 Demiris [55] | 직접 시뮬레이터 질의 |
| 훈련 시간 보정 | (비교 대상) | FoldNet [57], SPARS [58], BiFold [59], Hietala et al. [60] | 추론 시간 정제 vs. 훈련 시간 복구 |

**가장 가까운 대안.** 논문이 직접 지목하는 것은 UniClothDiff [18]입니다. 같은 문제를 시뮬 데이터로 학습한 확산 동역학 모델과 부분 점군의 전체 상태 복원으로 풉니다. 이 논문의 선택은 "학습 동역학 대신 시뮬레이터를 그대로 순방향 모델로" 쓰는 것이고, 상태 추정 비교(표 2)에서 [18]의 상태 추정기 DPM을 기준선으로 씁니다. 다만 [18]의 동역학 모델을 같은 MPPI에 넣은 종단 비교(순방향 모델만 바꾼 실로봇 행)는 없습니다. 표 3의 "시뮬레이터"를 바꾼 행은 Isaac Sim뿐이고 학습 동역학 모델 행은 없습니다(`03` V4).

## 2. GitHub와 코드 표 (2026-11-26 확인)

| 저장소 | 관계 | 라이선스 | 별 | 마지막 push | 상태 |
|---|---|---|---|---|---|
| 이 논문의 공식 코드 | 논문, 프로젝트 페이지, arXiv 어디에도 주소 없음 | — | — | — | 확인하지 못함 (아래 참고) |
| [silr-cloth/silr-cloth.github.io](https://github.com/silr-cloth/silr-cloth.github.io) | 프로젝트 페이지(정적 사이트, 기본 브랜치 `public`) | 없음 (API license: null) | 0 | 2026-11-19 | 공개. 코드가 아니라 웹페이지 소스 |
| FLASH (arXiv:2604.17513) | 이 논문의 시뮬레이터 | 논문은 CC BY-NC-SA 4.0(arXiv 라이선스 링크) | — | — | 해당 논문 본문이 "구현은 재현성을 위해 공개될 예정"이라고 적음. 저장소와 논문의 웹사이트 주소는 2026-11-26에 확인하지 못함 |
| [newton-physics/newton](https://github.com/newton-physics/newton) | 비교 대상(VBD 기반 Newton) | Apache-2.0 | 6,003 | 2026-11-26 | 공개 |
| [isaac-sim/IsaacSim](https://github.com/isaac-sim/IsaacSim) | 비교 대상(Isaac Sim, XPBD) | NOASSERTION (API) | 4,544 | 2026-11-16 | 공개 (NVIDIA 라이선스 조건은 저장소 안내를 따름) |
| [Genesis-Embodied-AI/Genesis](https://github.com/Genesis-Embodied-AI/genesis-world) | 비교 대상(PBD, 표에서 제외) | Apache-2.0 | 30,319 | 2026-11-26 | 공개 |
| [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | 시각 인코더 | Apache-2.0 | 13,570 | 2026-11-05 | 공개 |
| [facebookresearch/sam2](https://github.com/facebookresearch/sam2) | 천 분할(자산 구축) | Apache-2.0 | 20,245 | 2026-05-30 | 공개 |
| [Tongxuan259/UniClothDiff](https://github.com/Tongxuan259/UniClothDiff) | DPM 기준선의 원 구현 | MIT | 21 | 2025-11-29 | 공개 (논문은 "자기 천·위상·접힘 분포로 재학습"이라고 씀) |
| [albiLo17/Adafold](https://github.com/albiLo17/Adafold) | 가장 가까운 천 MPC 선행연구 | MIT | 1 | 2025-06-15 | 공개 |
| [google-deepmind/mujoco_mpc](https://github.com/google-deepmind/mujoco_mpc) | 예측 샘플링 계열의 참고 구현 | Apache-2.0 | 1,777 | 2026-11-24 | 공개. 이 논문이 쓰는 것은 아님 |

**공식 코드를 확인하지 못했습니다(2026-11-26).** (1) 프로젝트 페이지의 "Code" 버튼은 이 날짜에 저장소 주소로 연결되지 않습니다. (2) 논문 본문과 부록에는 저장소 주소가 없고 프로젝트 주소만 초록에 있습니다. (3) GitHub 저장소·코드 검색에서 이 논문의 arXiv 번호나 프로젝트 이름을 쓰는 코드 저장소를 찾지 못했습니다(프로젝트 페이지 저장소와 다른 사람의 논문 목록만 나옴). 그래서 이 분석은 코드 대조 대신 `04_claim_verification.md`로 갑니다.

**FLASH가 핵심 의존성입니다.** 표 1의 주 결과와 실로봇 전부가 FLASH 위에서 돌아가며, 시뮬레이터 코드가 공개되지 않으면 이 논문의 닫힌 고리는 외부에서 그대로 재현할 수 없습니다. FLASH 논문은 FLASH 코드의 공개를 "예정"이라고 적습니다(해당 논문 본문 서두의 구현 문장).

## 4. 이 논문의 위치 (요약)

- 새로운 알고리즘 요소보다 **구성의 새로움**(RGB → 정점 상태 → 시뮬레이터 → MPPI → 실행)이 주장의 중심이고, 개별 부품(DINOv2, MPPI, FEM 시뮬레이터)은 기존 것입니다.
- "첫 프레임워크"(서론)는 "천 조작에서 변형체 시뮬레이터를 추론 시간 순방향 모델로 쓰는 온라인 수평선 후퇴 제어 고리"라는 좁은 정의 안에서의 주장입니다(`04` C19). 같은 정의 밖의 AdaFold류(학습 모델 MPC)와는 순방향 모델의 종류로 구분됩니다.
- 가장 큰 설계상의 비용은 제어 주기입니다. 수건 한 스텝이 약 1.07 초(0.93 Hz), 반바지는 약 2.7 초이고 그중 MPPI가 대부분입니다(B.3, `03` V11). 준정적 조작에 한정한다는 한계와 짝을 이룹니다.
