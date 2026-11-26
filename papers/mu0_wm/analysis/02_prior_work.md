# 02. 선행연구와 관련 저장소

이 문서는 μ0의 구성 요소가 어디서 왔는지, 가까운 대안이 무엇인지, 공식 코드와 프로젝트 페이지가 어디에 있는지를 정리합니다. 대괄호 번호는 논문 참고문헌 번호(`paper/paper.txt`의 References, v3)입니다.

## 1. 선행연구 계보 (구성 요소별 출처)

| μ0의 요소 | 출처 (논문 번호) | 관계 |
|---|---|---|
| 중간 표현으로 모션을 쓰는 흐름 | 2D 흐름·궤적: ATM [10], Im2Flow2Act [11], Pixel Motion Diffusion [12], General Flow [31], FLIP [48]. 3D 흐름: 3DFlowAction [13], PointWorld [14], LaMP [15], 3PoinTr [16], TraceGen [17]. 물체 궤적: Track2Act [18] | 서론이 세 가지 한계(소영역 과소 표집, 카메라·물체 운동 혼합, 에피소드 단위 캡션)를 지적하는 대상 |
| 가장 가까운 선행: 3D 트레이스 세계 모델 | **TraceGen [17]** (같은 연구실, CVPR 2026) | 고정 격자 트레이스, 에피소드 단위 캡션, 추론 시 깊이 필요. μ0는 의미 키포인트, 이벤트 캡션, 선택적 깊이로 바꿈(§1, 부록 F) |
| 픽셀 공간 세계 모델과의 대비 | UniPi [7], VPP [8], Cosmos [9], GR-1 [32], Ctrl-World [33], 세계-행동 모델 [34–36], DreamDojo [2] | 외형 복원에 용량을 쓴다는 논지의 상대. VPP [8]는 부분 디노이징 특징을 정책에 쓰는 선례(부록 B.4) |
| 잠재 특징 세계 모델 | DINO-WM [38], LACE [37] | 제어가 어렵다는 것이 논문의 지적 |
| VLM 기반 웨이포인트·궤적 | HAMSTER [28], Spatial Trace [42], RoboPoint [43], Magma [44], RT-Trajectory [39] | 표 1의 2D 기준선(HAMSTER)과 VLM API 기준선(Gemini, GPT)의 계열 |
| 생성 영상에서 트랙 추출 | Dream2Flow [29], NovaFlow [47], Gen2Act [46], AVDC [45] | 표 1의 3D 기준선(Dream2Flow)의 계열 |
| 백본 | SmolVLM2-2.2B와 SmolVLA의 전문가 구조 [24], LeRobot [57] | 동결 VLM + 교차 주의 전문가를 그대로 가져와 트레이스 전문가로 씀 |
| B-spline 타깃과 강성 손실 | Trace Anything [25] | 3차 B-spline 제어점 표현과 "같은 부위 쌍 거리 불변" 정규화를 가져옴. 부위 라벨을 정답 분할 대신 DINO 군집으로 대체(부록 B.3) |
| 키포인트별 DINO 특징 융합 | Forecasting Motion in the Wild [27] | 질의 점 위치의 DINO 특징을 토큰에 더함(부록 B.2). ADE·FDE 정의도 같은 출처 |
| 조건화 | adaLN-Zero (DiT [26]) | 흐름 시간 τ 주입 |
| 3D 복원, 추적, 깊이 | VGGT [49], TAPIP3D [50], Depth Anything V2 [56] | TraceExtract의 부품. 3D 추적이 처리 시간의 45–58%(A.6) |
| 의미 키포인트 | DINOv2 [22] 군집, Trokens (v2 추가 인용) | TraceExtract §2.1, 부록 A.1 |
| 사전학습 데이터 | EgoDex [51], EgoVerse [52], AgiBot World [53], DROID [54] | 사람 52%, 로봇 48%(부록 A.7). 라이선스 열에 비상업·변경금지 조건이 섞여 있음(EgoDex CC-BY-NC-ND-4.0, Agibot CC BY-NC-SA 4.0) |
| 행동 전문가와 비교 대상 | π0 [20], π0.5 [21], Diffusion Policy [30] | 행동 전문가는 π0.5의 자기 주의 구조(B.4). 비교는 RoboCasa365 [19], [55]와 UR3 실로봇 |
| 사건 분할 | Savitzky–Golay 필터 [23] | 가속도 정점-골짜기 분할(식 2) |

**가까운 대안과의 차이.** 논문이 스스로 가장 가깝다고 보는 TraceGen은 (i) 고정 격자, (ii) 에피소드 단위 캡션, (iii) 추론 시 깊이 필요, (iv) 손으로 설계한 트레이스 재생을 쓰고 질의 가능한 세계 모델이 아니라는 점에서 다르다고 설명합니다(부록 F). 두 모델의 같은 줄기(같은 연구실, 같은 평가 틀)이므로 표 2와 그림 6의 TraceGen + 행동 전문가는 사실상 "격자 트레이스 + 같은 행동 머리" 대조입니다. 다만 TraceGen 쪽은 RGB에서 추정한 깊이(Depth Anything V2)를 입력으로 받는다고 부록 D.5가 밝힙니다.

## 2. 코드와 저장소 (GitHub 표)

기준일 2026-11-26. 별과 갱신일은 GitHub API, 크기는 Hugging Face API 메타데이터에서 읽었고 가중치와 데이터는 받지 않았습니다.

| 저장소 | 관계 | 라이선스 | 별 / 마지막 push | 대조 기준 | 상태 |
|---|---|---|---|---|---|
| [Yoonkyo/mu0](https://github.com/Yoonkyo/mu0) | 공식 구현(학습, 평가, 정책 학습). LeRobot 기반(SmolVLA 백본, 데이터/학습 인프라) | Apache-2.0 (GitHub API, 저장소 `LICENSE`는 LeRobot의 Hugging Face 저작권 표기를 유지) | 72 / 2026-10-27 | `963d5f13a94df31921d22887ff8611763b48178d` | 정적 읽기만 (`repo/`, 실행·설치 안 함) |
| [Yoonkyo/TraceExtract](https://github.com/Yoonkyo/TraceExtract) | 공식 데이터 엔진(depth/pose, 3D 추적, DINO 키포인트, 캡션) | Apache-2.0 | 19 / 2026-10-27 | `c02eb8efefd988dda7f0046e38efb00762a84e08` | 정적 읽기만 (`data/TraceExtract/`) |
| [프로젝트 페이지](https://mu0-wm.github.io/) | 논문, 코드 두 곳, 모델 링크, 영상, 동시 연구 목록 | 페이지 하단 템플릿 고지는 CC BY-SA 4.0(논문 라이선스와 별개) | | | HTML을 읽음 |
| [HF collection furonghuang-lab/mu0](https://huggingface.co/collections/furonghuang-lab/mu0) | 모델 2개와 논문 연결 | | | | 메타데이터만 |
| HF `furonghuang-lab/mu0` | 사전학습 체크포인트 `final_ckpt/model.safetensors` 5.62 GB, 정규화 통계, `sample_train_set.tar` 0.75 GB, `test_set.tar` 4.51 GB | Apache-2.0 태그 | 좋아요 1 / 2026-10-07 | | 받지 않음 (크기만) |
| HF `furonghuang-lab/mu0-policy` | RoboCasa 정책 체크포인트 `mu0_policy.tar` 10.97 GB (8과제 60k, 65과제 150k) | 표시 없음 | / 2026-07-02 | | 받지 않음 (크기만) |
| robosuite, robocasa (서브모듈) | 시뮬레이터와 RoboCasa 데이터·자산(자산 약 10 GB, 문서 기준) | 각각 | | | 가져오지 않음 |
| Depth-Anything-V2, Grounded-SAM-2 (서브모듈) | 추론 시 깊이·키포인트 합성 | 각각 | | | 가져오지 않음 |

- 공개 코드에는 TraceExtract가 쓰는 SpatialTrackerV2/VGGT, CoTracker, MoGe 계열 복사본(저장소 `models/`, `third_party/`)이 들어 있습니다.
- 기준선(π0, π0.5, Diffusion Policy, TraceGen + 행동 전문가)을 RoboCasa에서 돌리는 설정은 이 저장소에 없습니다(`04` D6).
- 프로젝트 페이지는 논문 외에 "2026 Best Paper Award @ IROS 2026 RoBoWoMo Workshop"을 적고 있습니다(워크숍 상). 학회 판정에는 쓰지 않았습니다.

## 4. 참고: 이 논문의 주장이 기대는 선행 결과

- "트레이스가 행동 전문가에 유용하다"는 가정은 TraceGen [17]의 결과와 VPP [8]의 부분 디노이징 특징 사용 선례에 기댑니다. μ0 논문 안에서 이 가정을 직접 가르는 실험은 표 12(트레이스 있음/없음)와 그림 6의 "VLM + 행동 전문가"입니다.
- "동결 VLM + 교차 주의 전문가는 효율적"이라는 설계는 SmolVLA [24]의 결과에 기댑니다.
- 3D 추적과 복원의 품질은 TAPIP3D [50]와 VGGT [49]의 성능에 그대로 묶입니다(논문 §6 한계가 인정).
