# 02. 선행연구와 계보

번호 [n]은 논문의 참고문헌 번호입니다. 별도 표기가 없는 서지는 논문 참고문헌 목록을 옮긴 것이고, 각 논문을 열어 검증하지는 않았습니다. GitHub 정보는 2026-10-29에 `gh api`로 확인했습니다.

## 1. 계보도

```
[다지 파지 합성]  DexGraspNet [19] · UniDexGrasp++ [21] · Dexonomy [26] · SpringGrasp [17]
        │
        └─ Lightning Grasp [50] (같은 1저자, 절차적 파지 합성, contact field) ─┬─ DexterityGen [3]
                                                                            └─ ★ LG+ (이 논문 부록 A.1, collision-aware IK)
[sim-to-real 손안 조작 RL]  OpenAI [1], Rubik's Cube [35] · DeXtreme [11] · 촉각 회전 [12] · DexScrew [34]
[장기 과제 연결]  Sequential Dexterity [13] (정책 사슬)
[애니메이션식 접근]  Liu, "Dextrous manipulation from a grasping pose", SIGGRAPH 2009 [14] (교신저자 C. K. Liu)
[관절형 물체·도구]  DexArt [36] · 두 손 뚜껑 [37] · 문·노트북 등 큰 관절 물체 [36–42]
                    공중·고정 초기 파지에서의 도구 조작: Xu et al. [47], Atar et al. [48]
                    강체 도구 zero-shot: SimToolReal [5]
[원격조작 데이터]  DexPilot [6] · Telekinesis [7] · Open-TeleVision [8] · Bunny-VisionPro [9] · GeoRT [10](같은 1저자)
        ▼
★ Mana: 탁자 위 얇은 관절형 도구의 잡기 → 작동 전체를, 절차적 파지 keyframe + MP/RL 사이 채우기 + 점군 diffusion 증류로
```

**차별점 요약:** 가장 가까운 선행연구 [47, 48]은 손 안에 이미 안정적으로 쥔 상태에서 시작합니다. Mana는 탁자 위 두께 1 cm 도구를 **집어 올리는 것부터** 작동까지 다루고, 사람 시연 없이 시뮬레이션 데이터만 씁니다. 방법상으로는 긴 RL 탐색 문제를 "파지 keyframe 그래프 + 짧은 전이"로 쪼갠 것이 핵심입니다. 이는 애니메이션의 keyframe과 inbetweening, 그리고 교신저자 C. K. Liu의 2009년 "파지 자세에서 출발하는 손 조작"[14]의 연장선입니다.

## 2. 구성 요소별 출처

| 역할 | 출처 | 이 논문에서의 쓰임 |
|---|---|---|
| 파지 합성 | Lightning Grasp [50] arXiv:2511.07418 (같은 1저자) | contact field 질의 → 접촉 영역 → 손끝 위치·관절 최적화. 공개 코드 있음(§5) |
| 얇은 물체 파지 | LG+ (부록 A.1, 새로 제안) | LM IK에 침투 해소 목표(p → p + d) 추가, CUDA. **공개되지 않음** |
| 시뮬레이터 | IsaacLab [51] arXiv:2511.04831 | 파지 안정성 필터, RL, 데이터 생성 |
| 경로 계획 | RRT-Connect [52] | GPU 병렬 구현으로 pre-grasp 접근 |
| RL | PPO (부록 A.4) | 목표 도달형 전이, privileged 상태 |
| 분할 | SAM 3 [53] arXiv:2511.16719 | 배포 시 도구 마스크 |
| 깊이 | Fast-FoundationStereo [54] arXiv:2512.11130 | 스테레오(D435 좌우 IR) 깊이 → 점군 |
| 점군 인코더 | Perceiver [55] | 질의 토큰 4개 |
| 생성 정책 | DDPM [56] | transformer diffusion head, 다음 행동 하나 |
| 원격조작 기준선 | GeoRT [10] (같은 1저자) | 위치 기반 리타겟 원격조작 |
| 상태 다양성의 근거 | "Emergent dexterity via diverse resets" [58] (ICLR 2026) | keyframe을 촘촘히 뽑는 이유로 인용 |
| 원격조작 한계의 근거 | DexJoCo [57] arXiv:2605.16257 | "원격조작은 약 30%" 관찰의 근거로 인용 |

## 3. 이 저장소의 다른 논문과의 관계

| 논문 | 관계 |
|---|---|
| **DexScrew** (`dexscrew/`) | Mana가 [34]로 인용합니다. 둘 다 "힘이 필요한 손안 조작"을 시뮬레이션 RL로 풉니다. DexScrew는 단순화한 시뮬레이션에서 RL 기술을 배우고, 그 기술을 텔레옵 버튼으로 써서 실물 촉각 데이터를 모읍니다. Mana는 시연 없이 시뮬레이션 데이터만으로 증류합니다. 그 대신 도구 스프링 시스템 식별과 큰 힘 랜덤화로 sim-to-real 간극을 메웁니다. |
| **FlashDexRetarget** (`flashdexretarget/`) | 둘 다 "기준 자세 사이를 RL로 채우는" 데이터 생성입니다. FlashDexRetarget은 사람 시연의 레퍼런스를 정책 하나로 추적하고, 물체 보조 힘 커리큘럼으로 탐색을 돕습니다. Mana는 사람 시연 대신 절차적 파지 keyframe을 쓰고, 긴 문제를 짧은 전이로 쪼개 탐색 부담을 줄입니다. 쪼갠 대가로 단계 연결은 원격조작에 맡깁니다(01 §4.2). |
| **GraspSense** (`graspsense/`) | 둘 다 "접촉 위치에 따라 허용·필요한 힘이 달라진다"를 다룹니다. GraspSense는 힘 지도를 명시적으로 계획하지만 강체 시뮬레이션이라 손상을 검증하지 못했습니다. Mana는 힘을 명시적으로 계획하지 않고, RL과 랜덤화로 마찰원뿔 안 힘 분배를 암묵적으로 배웁니다. |
| **Reactive Grasp** (`reactive_grasp/`) | 같은 문제의식(위치 기반 제어·원격조작으로는 힘을 못 맞춘다)에서 출발합니다. Reactive Grasp는 기하 접점 추정 + QP로 손 전체 power grasp의 힘을 실시간 조절합니다. Mana는 정밀 파지에서 학습 정책으로 힘을 냅니다. Mana의 한계(10 Hz, 촉각 없음, 미끄러짐 감지 불가)는 Reactive Grasp식 저수준 힘 조절 층과 결합할 자리로 보입니다(해석). |
| **DexPolicy** (`dexpolicy/`) | 둘 다 궤적·레퍼런스가 있는 dexterous RL의 탐색 문제를 다룹니다. DexPolicy는 탐색 잡음 σ의 스케줄로 풀고, Mana는 문제를 keyframe 사이 짧은 전이로 쪼개서 풉니다. |
| **AlphaNIST** (`alphanist/`) | 둘 다 "다양한 시작 상태"를 견고성의 핵심으로 봅니다. AlphaNIST는 OmniReset(arXiv:2603.15789, ICLR 2026)의 reset bank와 성공률 기반 시작 상태 샘플링을 씁니다. Mana는 [58] "Emergent dexterity via diverse resets and large-scale RL"(ICLR 2026)을 인용하며 keyframe 밀도를 늘립니다(Fig. 7 가운데). 저자·학회·주제로 보아 [58]은 OmniReset과 같은 논문으로 보입니다(추정, arXiv 번호 미확인). |
| **Lightning Grasp** (arXiv:2511.07418) | Mana 데이터 생성의 "동력원"입니다(프로젝트 페이지 표현). Mana 본문에서 확인되는 의존 관계는 두 가지입니다. ① 기능 영역 위 파지를 LG로 생성하고(Sec. 3.1), ② 지면 근처 얇은 물체에서 LG의 운동학 최적화가 충돌을 고려하지 않는 한계를 LG+로 고쳤습니다(부록 A.1). 공개 LG 코드의 IK에는 침투 해소 항이 없고 충돌은 사후 필터로만 쓰입니다(`04` §2). |

## 4. 논문이 다루지 않았지만 관련 있는 연구 (내 판단)

| 주제 | 관련성 | 근거 수준 |
|---|---|---|
| **DexMimicGen / MimicGen 계열** (소수 시연을 대량 시뮬레이션 데이터로 늘리기) | Mana도 "시뮬레이션에서 대량 궤적 생성 → 정책 증류"입니다. 시연 대신 절차적 keyframe을 쓴다는 점만 다릅니다. 비교 기준선으로 자연스럽습니다. | 내 사전 지식. 서지 확인 필요 |
| **Teacher–student 증류** (privileged RL teacher → 시각 student) | 부록 A.3·A.4가 RL 정책을 "(teacher)"라고 부릅니다. 데이터 생성 RL이 teacher, 점군 diffusion이 student인 구조입니다. DAgger 같은 on-policy 증류 대신 성공 궤적만으로 오프라인 증류한 점이 차이입니다. | 논문 표현에서 확인. 해석 |
| **Hybrid position–force control, impedance control** | 논문은 "precise hybrid position-force control"이 필요하다고 쓰지만(Sec. 2), 제어는 관절 PD와 학습 정책입니다. 명시적 힘 제어 층을 두는 대안과 비교하지 않았습니다. | 내 해석 |

**한계:** 4절의 서지 정보는 내 사전 지식에 근거하므로, 게시 전에 원문을 확인해야 합니다.

## 5. 관련 GitHub 저장소

| 저장소 | 무엇인가 | 공식 여부 | 라이선스 | ★ | 마지막 push | 확인 커밋 |
|---|---|---|---|---|---|---|
| (Mana 코드) | 데이터 생성기, LG+, RL 환경, 정책 | — | — | — | — | **미공개**. 프로젝트 페이지에 코드 링크가 없고, 1저자 공개 저장소 목록에도 없음 |
| zhaohengyin/lightning-grasp | Lightning Grasp 공개 구현(Allegro, LEAP, Shadow, D'Claw 설정) | 같은 저자의 구성 요소 공식 | 자체 LICENSE(CC BY-NC 문구, GitHub 표기 NOASSERTION) | 226 | 2026-06-12 | `af43818` (shallow clone, `repos/`) |
| facebookresearch/GeoRT | 원격조작 기준선 | 기준선 공식(같은 1저자) | NOASSERTION | 236 | 2025-09-02 | `fb2075e` (메타데이터만) |
| facebookresearch/sam3 | 분할 | 구성 요소 | NOASSERTION(자체 라이선스) | 12,139 | 2026-10-23 | `f1647e8` (메타데이터만) |
| NVlabs/Fast-FoundationStereo | 스테레오 깊이 | 구성 요소 | NOASSERTION(자체 라이선스) | 1,570 | 2026-09-11 | `476f424` (메타데이터만) |
| isaac-sim/IsaacLab | 시뮬레이터 | 구성 요소 | NOASSERTION(BSD-3 계열) | 8,511 | 2026-10-29 | `6d37745` (메타데이터만) |
| tylerlum/simtoolreal | SimToolReal [5] 공식 구현(강체 도구, 같은 교신저자 C. K. Liu) | 관련 연구 공식 | MIT | 424 | 2026-09-21 | `313d5ae` (메타데이터만) |

Lightning Grasp 저장소 README의 요점은 다음과 같습니다.
- "Dexterity Gen과 앞으로 나올 연구를 움직이는 엔진"이라고 소개합니다.
- CUDA 커널은 **미리 컴파일된 바이너리만** 배포하고, C++ 소스는 나중에 공개한다고 적었습니다.
- 지원 환경은 Ubuntu 22.04/24.04, NVIDIA GPU, CUDA 12입니다.
- "hand–object 침투를 줄이는 position iteration을 운동학 미세조정에 넣은 판"을 2026-05-24에 다음 릴리스로 예고했습니다. 2026-10-29 기준 main에는 반영되지 않았습니다. LG+(부록 A.1)와 같은 기능일 가능성이 있습니다(추정).
