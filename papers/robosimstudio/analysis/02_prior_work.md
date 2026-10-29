# 02. 선행연구와 계보

번호 [n]은 논문(v2)의 참고문헌 번호입니다. GitHub 정보는 2026-10-29에 `gh api`로 확인했고, 공식 저장소는 코드를 직접 읽었습니다. 나머지 논문은 따로 열어 검증하지 않았습니다.

## 1. 계보도

```
[시뮬레이터]  MuJoCo [9] → Isaac Gym / Isaac Lab [10–12], ManiSkill3 [13], MuJoCo Playground [14], Genesis [6]
              Warp [16] → Newton [3] (MuJoCo-Warp [17], VBD [18], XPBD [19] 통합) · SuperDex [4] (암시적 FEM, 외부 엔진)
                                         │
[Real2Sim]   digital twin / cousin: RialTo [44], URDFormer [45], Digital Cousins [46], Articulate-Anything [47],
             SIMPLER [48], SplatSim [49]  ·  생성형: GenSim [31, 32], RoboGen [50], Holodeck [51], Video2Policy [33]
                                         │
[에이전트 로보틱스]  SayCan [20], Code as Policies [21], VoxPoser [23] → Eureka [27], DrEureka [30]
             → 범용 코딩 에이전트(ReAct [7], Claude Code [1], Codex [2]) → CaP-X [36], GaP [5], ASPIRE [38], ENPIRE [39]
                                         │
                    ★ RoboSimStudio: 시뮬레이터의 모델 선택을 "편집 가능한 항목"으로 노출해 사람·에이전트 공동 Real2Sim
```

**차별점:** 기존 Real2Sim 연구는 대부분 재구성이나 생성 파이프라인 하나를 제안합니다. RSS는 파이프라인이 아니라 **작업대**입니다. solver·접촉 모델·물성을 바꾸고 결과를 보는 루프를 사람과 코딩 에이전트에게 함께 제공하는 것이 핵심입니다. 같은 그룹(Goldberg lab)의 GaP [5], CaP-X [36], Push-T 에이전트 연구 [42]와 이어지는 "코딩 에이전트가 로봇 문제를 직접 실험하며 푼다"는 흐름의 시뮬레이션 쪽 기반이기도 합니다.

## 2. 구성 요소별 출처

| 역할 | 출처 | 이 논문에서의 쓰임 |
|---|---|---|
| 물리 백본 | Newton [3] (Warp [16] 위) | 장면 조립, 결합, 렌더링 |
| 관절·강체 solver | MuJoCo-Warp [17] (CPU에서는 MuJoCo C) | 로봇 팔, 강체 |
| 천·케이블 solver | VBD [18] | 기본 "그 밖의 영역" solver |
| 연성체 FEM | SuperDex [4] (`facebookresearch/project_superdex`) | 스펀지, 가방. CPU에서는 천·로프도 담당 |
| 두 번째 외부 엔진 | Genesis [6] | IPC 천·연성체, SPH 물(선택) |
| 브라우저 화면 | Viser [8] | studio, 인스펙터, 기즈모 |
| IK / 계획 | PyRoKi [55] / cuRobo [56] | 팔 IK(기본), 충돌 회피 계획(`motion="plan"`, CUDA) |
| 정책 개선 | Graph-as-Policy [5] (`KeplerC/graph-as-policy`) | 스펀지 닦기 정책을 시뮬레이션 피드백으로 개선 |
| 리더 암 | GELLO [58] (논문 인용) · 저장소는 Feetech STS3215 서보의 YAM 리더 + `lerobot` 버스 드라이버 | 힘 피드백 원격 조종 |
| 마모 모델 | Preston [57] | 띠 연마, 마커 줄 지우기 |
| 에이전트 루프 | ReAct [7], SWE-agent [43] | "고치고-실행하고-관찰" 루프의 근거 |

## 3. 이 저장소의 다른 논문과의 관계

| 논문 | 관계 |
|---|---|
| **Do as I Do** (`do_as_i_do/`) | 둘 다 "일상 영상 → 물리 시뮬레이션"입니다. Do as I Do는 영상에서 손·물체 4D 궤적과 SAM 3D 메쉬를 자동으로 복원한 뒤 MuJoCo Warp sampling으로 리타겟팅합니다. RSS는 장면 자체를 코딩 에이전트가 손으로 짓습니다. Do as I Do가 다루지 못한 **변형 물체와 접촉 모델 선택**을 RSS가 채우고, RSS가 수작업으로 하는 **기하 복원**을 Do as I Do가 자동화합니다. 서로 보완하는 관계입니다. |
| **FlashDexRetarget** (`flashdexretarget/`) | FlashDexRetarget은 Isaac Sim 강체 장면에서 리타겟팅합니다. 둘 다 **물리적이지 않은 보조**를 씁니다. FlashDexRetarget은 물체를 레퍼런스로 끄는 PD 힘을 학습 중에만 쓰고 평가에서는 끕니다. RSS 공개 twin은 스펀지를 손에 고정하는 latch를 과제 내내 씁니다(`04` D3). 보조를 평가에서 빼는지가 결과의 물리적 타당성을 가릅니다. |
| **GraspSense** (`graspsense/`) | GraspSense는 SAM 3D 메쉬로 Isaac Lab 장면을 만들고 힘 지도로 파지를 고르지만, 코드상 물체는 강체였습니다. RSS의 SuperDex FEM은 GraspSense가 검증하지 못한 "벽 변형·손상" 주장을 실제 변형체로 확인할 수 있는 도구입니다. |
| **AlphaNIST** (`alphanist/`) | AlphaNIST는 Isaac Lab에서 NIST 조립 보드를 학습합니다. RSS의 커넥터 삽입(1 mm 미만 여유)과 같은 접촉 정밀도 문제를 다룹니다. RSS는 CPU에서 SDF·hydroelastic 접촉을 쓸 수 없어, 이런 과제는 GPU가 필요합니다(`04` §4). |
| **Contact-Grounded Policy** (`contact_grounded_policy/`) | CGP는 Meta 내부 FEM 시뮬레이터로 visuotactile 정책을 학습했습니다. RSS가 쓰는 SuperDex는 Meta의 공개 암시적 FEM 엔진입니다(`facebookresearch`). 같은 계열인지는 확인하지 못했습니다(가설). |

## 4. 논문이 다루지 않았지만 관련 있는 연구 (내 판단)

| 주제 | 관련성 | 근거 수준 |
|---|---|---|
| **시스템 식별(system identification)·베이지안 보정** | "영상과 맞을 때까지 값을 고친다"는 문제는 고전적인 물성 식별 문제입니다. 관측 가능성과 식별성 분석 없이 눈으로 맞추면 여러 값이 똑같이 그럴듯해 보입니다(`03` §2). | 내 해석 |
| **미분 가능한 시뮬레이션으로 물성 추정** | 같은 Warp·Newton 위에서 경사 기반으로 물성을 맞추는 방법은 에이전트의 시행착오를 줄일 수 있습니다. | 내 사전 지식. 서지 확인 필요 |
| **도메인 랜덤화** (Tobin et al. 2017 등) | Fig. 7e의 강성·배치 변형은 고전적 도메인 랜덤화입니다. 식별할 수 없는 값을 랜덤화로 처리하는 것은 합리적이지만, 범위가 물리적으로 의미 있어야 합니다(`04` C6). | 내 사전 지식 |

## 5. GitHub 정리 (2026-10-29 `gh api` 기준)

| 저장소 | 무엇인가 | 공식 여부 | 라이선스 | ★ | 마지막 push | 확인한 커밋 |
|---|---|---|---|---|---|---|
| [robosimstudio/robosimstudio](https://github.com/robosimstudio/robosimstudio) | 이 논문의 도구 전체(라이브러리, studio, 에이전트 페이지, 벤치마크, GaP 연결) | **공식** | Apache-2.0 (로봇 자산은 MIT·BSD) | 270 | 2026-10-28 | `81f57c9` (HEAD, clone), `73739ab` (논문 시점 첫 커밋, 트리만) |
| robosimstudio/robosimstudio.github.io | 프로젝트 페이지 | 공식 | 표기 없음 | 0 | 2026-10-15 | `47b0796` (메타데이터만) |
| [newton-physics/newton](https://github.com/newton-physics/newton) | 물리 백본 (Linux Foundation 프로젝트) | 의존성 | Apache-2.0 | 5,861 | 2026-10-28 | `23638bf` (메타데이터만) |
| [NVIDIA/warp](https://github.com/NVIDIA/warp) | GPU/CPU 커널 언어 | 의존성 | Apache-2.0 | 7,253 | 2026-10-29 | 메타데이터만 |
| [google-deepmind/mujoco_warp](https://github.com/google-deepmind/mujoco_warp) | MuJoCo GPU판 | 의존성 | Apache-2.0 | 1,592 | 2026-10-28 | 메타데이터만 |
| [facebookresearch/project_superdex](https://github.com/facebookresearch/project_superdex) | 암시적 FEM 엔진 (스펀지·가방) | 의존성 | Apache-2.0 | 770 | 2026-10-29 | `353abe6` (메타데이터만) |
| [viser-project/viser](https://github.com/viser-project/viser) | 브라우저 3D 화면 | 의존성 | Apache-2.0 | 2,838 | 2026-09-25 | `56712d3` (메타데이터만) |
| [chungmin99/pyroki](https://github.com/chungmin99/pyroki) | IK (jax) | 의존성 | MIT | 1,812 | 2026-04-01 | RSS가 `95afccc`에 고정 |
| [NVlabs/curobo](https://github.com/NVlabs/curobo) | 충돌 회피 계획 (CUDA) | 선택 의존성 | Apache-2.0 | 1,943 | 2026-09-10 | RSS가 `v0.8.0`에 고정 |
| [KeplerC/graph-as-policy](https://github.com/KeplerC/graph-as-policy) | GaP 런타임 [5] | 같은 1저자, 선택 의존성 | Apache-2.0 | 1 | 2026-10-24 | RSS가 `09e45f8`에 고정 |
| [Genesis-Embodied-AI/genesis-world](https://github.com/Genesis-Embodied-AI/genesis-world) | 두 번째 외부 엔진 | 선택 의존성 | Apache-2.0 | 30,136 | 2026-10-28 | 메타데이터만 |
| [wuphilipp/gello_software](https://github.com/wuphilipp/gello_software) | GELLO 리더 암 [58] | 관련(논문 인용, 저장소는 YAM 리더 사용) | MIT | 561 | 2026-09-14 | 메타데이터만 |
| [i2rt-robotics/i2rt](https://github.com/i2rt-robotics/i2rt) | YAM 로봇 모델 출처 | 자산 출처 | MIT | 178 | 2026-10-28 | 메타데이터만 |
| [KeplerC/microduck-with-rss](https://github.com/KeplerC/microduck-with-rss) | 사진 한 장 Real2Sim 데모(README 소식) | 같은 1저자 | Apache-2.0 | 2 | 2026-10-26 | 메타데이터만 |

**한계:** 4절의 서지는 내 사전 지식입니다. 블로그나 git에 올리기 전에 원문을 확인해야 합니다.
