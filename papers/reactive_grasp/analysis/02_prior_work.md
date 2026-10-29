# 02. 선행연구와 계보

번호 [n]은 논문의 참고문헌 번호입니다. arXiv ID와 학회는 논문 참고문헌에 적힌 값을 옮겼고, 각 논문을 따로 열어 검증하지는 않았습니다. GitHub 정보는 2026-10-29에 `gh api`로 확인했습니다.

## 1. 계보도

```
[파지 합성: "어디를 잡을까"]
  force-closure 최적화 DexGraspNet [1] · GraspQP [4] · Dexonomy [5] (분류 체계 전 유형)
  → 생성 모델 DexDiffuser [2] · DexGraspNet 2.0 [3]
  ※ 실행은 대개 open-loop: 정해 둔 자세로 가서 미리 계산한 힘을 주거나 일정량 더 쥔다

[힘 분배: "얼마나 세게, 어디에"]
  Kerr & Roth 1986 [26] (다지 손 힘 분배) → Bicchi 1993/1994 [31][19] (whole-limb: 능동 vs 수동 성분)
  → Prattichizzo & Bicchi [32] · Haas-Heger 수동 반력 [27]
  마찰 피라미드 선형화 [29][30]

[반응형 파지 제어: "닿은 뒤 다시 맞추기"]
  수동성 기반 물체 임피던스 [6] · 적응 손가락 협응 [7] · force-closure MPC [8]
  → 촉각 기반 pinching [9] · TacDexGrasp [10] · CoorGrasp [11] (촉각이 있는 접점에서 일반 개수 재할당)

[접점 추정: "어디에 닿았나"]
  촉각 스킨 [12–16] · 관절 토크 잔차 [17][21][22] · 학습 추정 RobustDexGrasp [18] · 비전+고유감각 [23]
  사람 손-물체 모델링의 기하 접촉 정의 GRAB [24] · ContactOpt [25]

          ▼ 셋을 합침
  ★ 이 논문: 기하 접점(손 전체 링크) + 매 주기 힘 재분배 QP + 같은 그룹의 reactive reaching [20]
```

**차별점 요약:** CoorGrasp[11]는 접점 수와 위치가 일반적이고 힘을 온라인으로 재할당하지만, **촉각이 측정되는 곳에서만** 그렇게 합니다. 이 논문은 촉각을 "알려진 물체 모델 + 추적 자세 + 관절각"으로 바꿔 계측하지 않은 손 표면 전체로 넓혔습니다. 대신 물체 메쉬와 실시간 자세 추적이 전제됩니다.

## 2. 구성 요소별 출처

| 역할 | 출처 | 이 논문에서의 쓰임 |
|---|---|---|
| 목표 파지와 유형 | Dexonomy [5] (RSS 2025) | 시뮬레이션 power grasp 790개, 정밀 파지(Type 7), 기준선의 초기 힘 할당 방식 |
| pre-grasp 여유 | GraspQP [4] (CoRL 2025) | 침투 깊이 기준으로 3 cm 여유 |
| reactive reaching | Lee et al. [20] (ICRA 2026, 같은 그룹) | 손끝·손바닥 속도장 + 관절 공간 QP(Eq. 11)를 다섯 손가락으로 확장 |
| whole-limb 힘 해석 | Bicchi [19][31], Prattichizzo & Bicchi [32], Haas-Heger [27] | actuation-consistency 제약(Eq. 9)의 동기 |
| 마찰 피라미드 | Stewart & Trinkle [29], Anitescu & Potra [30] | Eq. 5 |
| QP 솔버 | OSQP [33] | 매 주기 힘 분배 |
| SDF 질의 | NVIDIA Warp [28] | 약 3,000점 질의 1 ms 미만 |
| 시뮬레이터 | MuJoCo [34] | 중력 끈 평가 |
| 물체 스캔·추적 | Point2Pose [35](같은 그룹, ECCV 2026), 2D Gaussian Splatting [36], Poisson [37], FoundationPose [38] | 스캔 → 메쉬 → SDF. FoundationPose의 메쉬 렌더러를 Gaussian 렌더러로 바꿔 30 Hz 추적 |
| 하드웨어 | Flexiv Rizon 4 + Robotis 5F(20 DoF) + RealSense D455 | 손끝 촉각 센서는 쓰지 않음 |

## 3. 이 저장소의 다른 논문과의 관계

| 논문 | 관계 |
|---|---|
| **GraspSense** (`graspsense/`) | 둘 다 "물리적으로 근거 있는 dexterous 파지 힘"을 다루지만 방향이 반대입니다. GraspSense는 재료와 두께로 **허용 하중의 상한**(힘 지도)을 미리 계산해 파지를 고르고, 이 논문은 **최소 조임(Σf_n ≥ 10 N)** 아래 힘을 실시간으로 재분배합니다. GraspSense의 영역별 상한 F*를 이 QP에 접점별 상한 f_n ≤ F*로 넣으면, 깨지기 쉬운 물체를 조임 하한과 상한 사이에서 조절하는 제어기가 됩니다. 평가 방식도 대조됩니다. GraspSense는 같은 힘 지도로 선택과 판정을 해서 순환적이었고, 이 논문은 시뮬레이션 하중 시험이라는 독립 판정이 있습니다. 다만 중력을 끈 시험이라 그 판정도 약합니다(01 §4.2). |
| **Contact-Grounded Policy** (`contact_grounded_policy/`) | 둘 다 "명령한 것과 실제로 일어나는 접촉력은 다르다"는 문제를 다룹니다. CGP는 촉각 데이터에서 매핑을 학습했고, 우리 분석에서 그 실체가 준정적 강성 역행렬임을 보였습니다. 이 논문은 같은 문제를 모델 기반으로 다루어, Jᵀ로 만들 수 없는 성분을 Eq. 9로 억제합니다. 두 논문 모두 compliance를 명시적으로 모델링하지 않는다는 점이 공통 한계입니다. |
| **DexScrew** (`dexscrew/`) | DexScrew는 촉각 신호로 손끝 회전 기술의 실패 꼬리를 줄였습니다. 이 논문은 촉각 없이 기하로 접점을 얻습니다. 실험에 쓴 Robotis 5F 손에도 손끝 촉각 센서가 있지만 쓰지 않았습니다(Sec. IV-C.1). 촉각을 함께 쓰면 접점 위치(기하)와 접촉 여부(촉각)를 교차 검증할 수 있습니다. |
| **FlashDexRetarget** (`flashdexretarget/`) | FlashDexRetarget도 물체를 128³ SDF 격자로 구워 손끝 거리와 법선을 관측에 넣고, 링크별 접촉 wrench를 privileged 관측으로 씁니다. 이 논문은 같은 종류의 기하 접점 정보를 실제 로봇에서 촉각 대신 쓰는 사례입니다. FlashDexRetarget을 실물로 옮길 때 접촉 wrench 관측을 무엇으로 바꿀지 고민했던 부분(teacher-student)에 대한 하나의 답입니다. 다만 자세 오차가 접점 정보를 크게 흔든다는 점(`03` §5)도 함께 가져와야 합니다. |

## 4. 논문이 다루지 않았지만 관련 있는 연구 (내 판단)

| 주제 | 관련성 | 근거 수준 |
|---|---|---|
| **실시간 파지 힘 최적화** (Buss, Hashimoto, Moore 1996, 양정치 행렬 위 경사 흐름. Boyd & Wegbreit 2007, 빠른 접촉력 계산) | 매 주기 힘을 다시 푸는 문제의 고전적 정식화입니다. 목적함수에 힘 크기 항이 있어 해가 유일하게 정해집니다. 이 논문의 QP는 그 항이 없어 해를 솔버가 고릅니다(`03` §1). | 내 사전 지식. 서지 확인 필요 |
| **내부력 분해** (Yoshikawa & Nagai) | G의 null space(내부력)와 Jᵀ의 null space(구동 불가 성분)를 구분하는 언어입니다. Eq. 9는 후자를 다룹니다. | 내 사전 지식. 서지 확인 필요 |
| **접촉 compliance 기반 힘 분배** (Bicchi [19]의 compliance 모델, Haas-Heger [27]) | 저자들이 한계로 든 "수동 compliance와 preload 미반영"을 직접 다루는 방법입니다. | 논문 인용 |

## 5. GitHub 정리

| 저장소 | 무엇인가 | 공식 여부 | 라이선스 | ★ | 마지막 push | 확인 커밋 |
|---|---|---|---|---|---|---|
| sangminkim-99/reactive-grasp-whole-hand | 프로젝트 웹페이지(index.html, static). 코드 없음, 페이지에 "Code (Soon)" | 공식(웹페이지만) | 표기 없음 | 0 | 2026-09-25 | `3ab1aa5` |
| rophi-lab/RoPhiTutorial | 공동 교신저자 연구실의 MuJoCo·실물 공용 파이프라인. Flexiv Rizon 4 + Robotis 5F, UDF 접점 검출, force-closure OSQP, feedforward·impedance 비교 제어기, 섭동 환경 | 논문 공식 코드 아님(같은 연구실). sim 설정값이 논문과 일치 | 표기 없음 | 1 | 2026-08-02 | `85aaa4c` (`controller/`, `utils/grasping/`, `configs/`, `env/` sparse clone) |
| rophi-lab/oop-public | 같은 연구실. Robotis 손 description(URDF, xacro) 포함 | 관련(하드웨어 모델) | MIT | 5 | 2026-10-27 | 메타데이터만 |
| JYChen18/Dexonomy | 목표 파지 합성 [5] | 구성 요소 공식 | 표기 없음 | 158 | 2026-03-26 | `d561ff9` (메타데이터만) |
| leggedrobotics/graspqp | GraspQP [4]. 1저자의 포크 `sangminkim-99/graspqp`도 있음 | 구성 요소 공식 | MIT | 66 | 2026-07-13 | `e2de494` (메타데이터만) |
| zdchan/RobustDexGrasp | 학습 기반 링크별 접촉 추정을 쓰는 관련 연구 [18] | 관련 공식 | 비표준 | 179 | 2025-08-20 | `b00a6cd` (메타데이터만) |
| NVlabs/FoundationPose | 자세 추적 [38]. 1저자의 포크 `sangminkim-99/FoundationPose`도 있음 | 구성 요소 공식 | 비표준(NVIDIA) | 3,663 | 2026-04-29 | `a1b694b` (메타데이터만) |
| point2pose/point2pose.github.io | Point2Pose [35] 웹페이지 | 구성 요소(웹페이지) | BSD-3-Clause | 5 | 2026-09-24 | `1801ce1` (메타데이터만) |
| NVIDIA/warp | SDF 질의 [28] | 의존성 | Apache-2.0 | 7,253 | 2026-10-29 | `104afd1` (메타데이터만) |
| osqp/osqp | QP 솔버 [33] | 의존성 | Apache-2.0 | 2,215 | 2026-01-12 | `1572ae0` (메타데이터만) |
| google-deepmind/mujoco | 시뮬레이터 [34] | 의존성 | Apache-2.0 | 15,948 | 2026-10-28 | `9916cc5` (메타데이터만) |
| ROBOTIS-GIT/robotis_hand | Robotis 손 ROS 패키지 | 하드웨어 공식 | Apache-2.0 | 67 | 2026-10-26 | `6f9e22a` (메타데이터만) |

CoorGrasp[11], TacDexGrasp[10], Lee et al. [20]의 공개 저장소는 검색에서 찾지 못했습니다.
