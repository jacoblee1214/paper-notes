# 02. 선행연구와 구성 요소의 출처

번호는 논문(v2) 참고문헌 번호입니다.

## 1. 계보: 움직이는 물체와 VLA의 지연

| 흐름 | 대표 | AHEAD와의 관계 |
|---|---|---|
| 정적 장면 가정의 VLA | RT-2 [3], OpenVLA [1], Octo [2], π0 [4], GR00T N1 [17] | 관측 한 장에서 행동을 냅니다(§2). AHEAD는 OpenVLA 7B를 동결해 그대로 씁니다. |
| 인식–행동 루프 단축 | ACT [7], Diffusion Policy [8], OpenVLA-OFT [43], 이중 시스템 HiRT [9], RoboDual [10] | 행동 청크와 빠른 재계획으로 오래된 창을 줄이지만 없애지는 못한다고 논문이 봅니다(§1). 기준선으로 Realtime ACT, Streaming DP, VLA + Fast Replan을 씁니다. |
| 동시기 지연 보상 | DynamicVLA [11], VLASH [12] | 빠른 추론이나 비동기 실행. "루프를 가속하지 예측하지는 않는다"고 분류하고(§2) 비교하지 않았습니다. |
| 세계모델과 상상 롤아웃 | Dreamer [13, 18, 19], TD-MPC2 [20], Genie [14], 상호작용 시뮬레이터 [21], WorldVLA [15], Ctrl-World [28], VLAW [29] | 전체 장면을 고정 horizon으로 굴리고 픽셀 공간이거나 공동 학습이 필요하다고 봅니다. AHEAD는 일부 토큰만, 가변 horizon으로, 동결 VLA 특징 공간에서 굴립니다. |
| 사전학습 특징 공간 예측 | **LaDi-WM [16]**(CoRL 2025), DreamVLA [44], V-JEPA 2 [31] | 가장 가까운 선행입니다. LaDi-WM은 잠재 확산, 전체 토큰, 고정 horizon이고, AHEAD는 flow matching, 광류 조건, 선택적 롤아웃, 적응 중단입니다(§2). LaDi-WM은 v2에서 OpenVLA 특징으로 다시 학습해 기준선에 넣었습니다. |
| 고전 intercept | Morrison 외 [26], Kopicki 외 [27] | 손으로 설계한 상태 표현입니다. Track-and-Intercept 기준선이 이 계열의 이미지 공간 판입니다. |
| 비디오 사전학습 | GR-2 [30], V-JEPA 2 [31], R3M [32], Ego4D [33]를 쓰지 않는 이유 | 조작에 맞춘 코퍼스(EPIC-Kitchens, SSv2, DROID, Bridge V2, CALVIN, OXE)를 고릅니다(§2, 부록 B.8). |

## 2. 가까운 대안과 논문이 비교한 것

| 대안 | 논문의 처리 | 비고 |
|---|---|---|
| LaDi-WM(OpenVLA 특징) | 표 1, 3에 포함(v2) | 광류·운동 조건 없음, K = 5 고정, 20 확산 단계(부록 C.3). 세계모델 학습 약 96 GPU-시간으로 AHEAD의 사전학습과 같은 규모. |
| Track-and-Intercept | 표 1, 3에 포함(v2) | CoTracker3 + 5프레임 등가속 외삽 + 프레임 워프. 정속 4종 평균 86.8로 AHEAD 93.5에 가깝습니다. 논문도 "운동 사전 지식만으로도 상당 부분이 해결된다"고 씁니다(§5.2). |
| DreamVLA | 표 1–4에 포함 | 공개 구현과 가중치를 시나리오 분포로 미세조정(약 80 GPU-시간). |
| VLASH, DynamicVLA, F2F-AP류의 비동기·지연 보상 | 비교 안 함 | VLASH와 DynamicVLA는 관련 연구에서 동시기 작업으로 언급(§2). |
| 참 미래 프레임을 동결 VLA에 넣는 oracle | 보고 안 함 | 예측 토큰이 어디까지 상한에 근접하는지 알 수 없습니다(`04` C17). |
| 단순 지연 보상(물체 중심 외삽 + 프레임 워프)을 학습 없이 | Track-and-Intercept가 해당 | 위 표와 같음 |

## 4. 구성 요소별 출처

| 구성 요소 | 출처 | 논문 안의 위치 |
|---|---|---|
| 동결 VLA와 병렬 디코딩 | OpenVLA [1], OpenVLA-OFT [43] | §4, 표 6 |
| 광류 | RAFT [38], 대안 CoTracker3, Farnebäck | §4.1, 부록 B.5, 표 12 |
| 조건부 flow matching, Euler 5단계 | Lipman 외 [39], Esser 외 [40] | §4.2 |
| 샘플 분산 불확실성 | 샘플 평균·분산(식 4). 대안으로 MC 드롭아웃 [56], 앙상블 [69]를 부록 B.2와 한계에서 논의 | §4.3, 부록 B.2 |
| 등가속 속도 전파 | 고전 운동학 (식 3). 가속도는 두 광류의 차분(식 10) | §4.2, 부록 B.6 |
| 언어 조건 살리언시 | 교차 어텐션 [62] + 모션 하한(식 1, 8) | §4.1, 부록 B.4 |
| 사전학습 코퍼스 | EPIC-Kitchens [34], SSv2 [35], DROID [36], Bridge V2 [37], CALVIN [58], OXE [60] | 표 7 |
| 자기 시점 카메라 운동 제거 | 우세 호모그래피 차감 | 부록 B.8 |
| 정렬 층(행동 오차로 학습) | 동결 정책 π를 통과하는 지도(식 11). 증류 형태 | 부록 B.7 |
| 시뮬레이션과 로봇 | MuJoCo [41], Franka Panda [42], xArm 7, RealSense D435 | §5.1, 부록 C |

## 5. GitHub와 코드 자료 (확인 2026-11-26)

공식 코드와 프로젝트 페이지는 논문 본문, arXiv 페이지, GitHub 검색(논문 제목, 방법 이름, 저자 이름)으로 확인하지 못했습니다. 이 날짜 기준으로 확인하지 못했다는 뜻입니다. 같은 저자들의 이전 작업(ExpReS-VLA, arXiv:2511.06202)의 프로젝트 페이지 저장소는 있으나 AHEAD와는 별개입니다.

| 자료 | 링크 | 라이선스 | 별 / 마지막 push | 비고 |
|---|---|---|---|---|
| **AHEAD 공식 코드** | 확인하지 못함 | - | - | `repo/` 없음 |
| AHEAD 프로젝트 페이지 | 확인하지 못함 | - | - | 논문과 arXiv 페이지에 링크 없음 |
| 저자 이전 작업 ExpReS-VLA 프로젝트 페이지 | https://github.com/expresvla/expresvla.github.io | MIT | 0 / 2026-11-09 | AHEAD 자료 없음 |
| OpenVLA (동결 백본) | https://github.com/openvla/openvla | MIT | 7350 / 2025-03-23 | |
| OpenVLA-OFT (병렬 디코딩) | https://github.com/moojink/openvla-oft | MIT | 1450 / 2025-09-09 | |
| RAFT (광류) | https://github.com/princeton-vl/RAFT | BSD-3-Clause | 4131 / 2025-08-24 | |
| CoTracker (기준선·절제) | https://github.com/facebookresearch/co-tracker | 표기 없음(NOASSERTION) | 5212 / 2026-03-03 | |
| DreamVLA (기준선) | https://github.com/Zhangwenyao1/DreamVLA | 표기 없음 | 377 / 2026-01-06 | |
| LaDi-WM (가장 가까운 선행) | https://github.com/GuHuangAI/LaDiWM | 표기 없음 | 65 / 2025-11-30 | |
| VLASH (동시기 작업) | https://github.com/mit-han-lab/vlash | Apache-2.0 | 526 / 2026-04-22 | |

별과 마지막 push 날짜는 GitHub API 메타데이터입니다. 가중치와 데이터는 받지 않았습니다.

**학회.** arXiv comments는 "Accepted at the 10th Conference on Robot Learning (CoRL 2026)"이고 논문 첫 쪽 각주도 같습니다. corl.org 공식 채택 목록(2026-11-14 수집)에 같은 제목이 **포스터**로 있습니다(포스터 세션 Wed 11/11, 번호 20).
