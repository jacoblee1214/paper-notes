# 02. 선행연구와 계보

번호 [n]은 논문의 참고문헌 번호입니다. GitHub 저장소는 2026-10-29에 `gh api`로 확인했고, 논문 각각의 내용은 따로 열어 검증하지 않았습니다(§4는 내 판단).

## 1. 계보도

```
[고전] GraspIt! [16] (2004) ─ 손 모델 + grasp 품질 지표, 시뮬레이터 안 탐색
        │
[미분 가능 에너지 최적화: 침투 E_pen ↔ 끌어당김 E_attract 균형]
   DFC (Liu et al., RA-L 2021) [13] ─ 미분 가능한 force-closure 추정
     → DexGraspNet [19] (ICRA 2023) ─ 대규모 데이터셋, A100에서 SPS < 3 (Fig. 1)
     → DexGraspNet 2.0 [22] (CoRL 2024) ─ 어수선한 장면, 생성 모델
   SpringGrasp [6] (RSS 2024) ─ 형상 불확실성 아래 compliant 손끝 grasp
   BODex [5] (ICRA 2025) ─ cuRobo 위 bilevel 최적화, 손끝 접점, SPS 30–50
   Dexonomy [4] (RSS 2025) ─ grasp taxonomy 전 유형 합성
        │
[self-balancing ε-wrench 기준] Khandate et al. [11] (RSS 2023), Chen et al. [7] (SIGGRAPH 2023),
                              DexterityGen [21] (RSS 2025, 같은 1저자: CPU 클러스터 grasp 탐색)
        │
  ★ Lightning Grasp: 기하(contact field + BVH)와 탐색(0차 국소 탐색)을 분리, GPU, SPS 수백–수천
```

**차별점 요약:** 기존 해석적 방법은 침투 에너지와 끌어당김 에너지를 함께 최적화합니다. 그래서 메쉬 기하 계산이 최적화의 매 스텝에 끼어 있고, 두 에너지의 가중치에 민감합니다(Sec. 3.3). 이 논문은 "손이 닿을 수 있는 곳"을 contact field로 미리 계산해 두고, 탐색은 그 안에서 접점만 고르게 바꿨습니다. 같은 1저자의 DexterityGen[21]에서 쓰던 CPU 기반 grasp 탐색을 GPU로 다시 설계한 결과로 소개합니다(Sec. 1).

## 2. 구성 요소별 출처

| 역할 | 출처 | 이 논문에서의 쓰임 |
|---|---|---|
| 안정성 기준 (FSWO/GSWO) | Khandate et al. [11], Chen et al. [7], DexterityGen [21] | 마찰 없는 self-balancing wrench. 공개 코드는 투영 경사하강 NNLS로 풂 (`04` D2) |
| BVH 구성 | LBVH, Karras [10] (HPG 2012) | contact field box를 LBVH로 정리 (Alg. 1). 구현은 비공개 CUDA 커널 |
| 손 자기 충돌 | GJK [9] | 손 메쉬를 볼록 분해한 뒤 병렬 GJK |
| 손 운동학 | PyTorch [17] | 정·역운동학, 다중 사슬 DLS IK (Eq. 7–8) |
| 손 모델 | Shadow [8], LEAP [18], Allegro [14], DClaw (ROBEL [1]) | 평가한 손 네 종 |
| 물체 | YCB [3], 인터넷 CC 자산 4개 | 평가 물체 |
| 기준선 | DexGraspNet [19], SpringGrasp [6], BODex [5] | Fig. 1의 속도·다양성 비교. 수치는 범위로만 제시 |

### 관련 GitHub (2026-10-29 `gh api` 확인)

| 저장소 | 무엇인가 | 공식 여부 | 라이선스 | ★ | 마지막 push | 확인 커밋 |
|---|---|---|---|---|---|---|
| zhaohengyin/lightning-grasp | 이 논문의 시스템(`lygra`). Python + 미리 컴파일한 CUDA 커널 | 공식 | CC BY-NC 4.0 (LICENSE 파일. GitHub는 "NOASSERTION"으로 표시) | 226 | 2026-06-12 | `af43818` (clone) |
| 같은 저장소 Release v1.0.0 | 손·물체 자산(44 MB), 커널 바이너리 4종(Python 3.8–3.10, CUDA 12.1–12.6, 각 약 64 MB) | 공식 | 위와 같음 | – | 2025-11-11 | 메타데이터만 |
| PKU-EPIC/DexGraspNet | 기준선, 에너지 최적화 데이터셋 | 기준선 공식 | 표기 없음 | 478 | 2025-01-06 | `bd1b13d` (메타데이터만) |
| PKU-EPIC/DexGraspNet2 | 생성형 후속 [22] | 공식 | 표기 없음 | 165 | 2025-01-23 | `26ecd76` (메타데이터만) |
| Stanford-TML/SpringGrasp_release | 기준선 | 기준선 공식 | 표기 없음 | 45 | 2025-10-15 | `07c1131` (메타데이터만) |
| JYChen18/BODex | 기준선, cuRobo 기반. README "3090 한 장으로 하루 수백만 grasp" | 기준선 공식 | 표기 없음(NOASSERTION) | 152 | 2026-04-21 | `06b9a3c` (메타데이터·README만) |
| JYChen18/Dexonomy | 관련 [4]. README CC BY-NC 4.0, RTX 3090 테스트 | 공식 | CC BY-NC 4.0 (README) | 158 | 2026-03-26 | `d561ff9` (README만) |
| leggedrobotics/graspqp | 관련(논문 미인용). 이 저장소 `reactive_grasp/`의 pre-grasp 출처 | 공식 | MIT | 66 | 2026-07-13 | `e2de494` (메타데이터만) |
| graspit-simulator/graspit | GraspIt! [16] | 공식 | 표기 없음(NOASSERTION) | 222 | 2021-04-19 | `83a6d4c` (메타데이터만) |
| tengyu-liu/GenDexGrasp | 관련(논문 미인용), 손 일반화 생성 모델 | 공식 | 표기 없음 | 211 | 2025-09-03 | 메타데이터만 |

공식 저장소에는 포크가 몇 개 있지만, 독립 구현이나 CPU 이식판은 찾지 못했습니다(`gh search repos "lightning grasp"`). 이슈는 16건이고, 그중 2건이 열려 있습니다. 내용이 분석과 관련된 이슈는 #6(IsaacGym에서 침투·불안정), #12(Shadow 중간 마디 법선), #14(중력 미반영), #15(큰 물체 한계: 점 2,048개 기준 표면적 약 0.2 m²)입니다.

## 3. 이 저장소의 다른 논문과의 관계

| 논문 | 관계 |
|---|---|
| **GraspSense** (`graspsense/`) | GraspSense는 DexGraspNet으로 물체당 후보 50개를 만들고, Isaac Gym으로 거른 뒤 힘 지도로 다시 순위를 매깁니다. 후보 생성이 병목인 구조라서 Lightning Grasp가 그 자리를 바로 대신할 수 있습니다(Shadow 손 지원). 다만 GraspSense가 비판받은 지점, 즉 물리 검증 없이 기하 점수만으로 고른다는 점은 Lightning Grasp도 같습니다. |
| **Reactive Grasp** (`reactive_grasp/`) | 실시간 힘 조절 제어기에 Dexonomy 목표 파지와 GraspQP pre-grasp를 입력으로 씁니다. Lightning Grasp는 "어디를 잡을지"를 빠르게 만드는 쪽이고, Reactive Grasp는 "잡은 뒤 힘을 어떻게 나눌지"를 다룹니다. Lightning Grasp의 FSWO(마찰 없음, 중력 없음)는 Reactive Grasp의 마찰 원뿔 QP보다 약한 기준이라서, 그 제어기로 실현 가능성을 확인하는 조합이 자연스럽습니다. |
| **ZeroDex** (`zerodex/`) | 일반 형상의 파지 후보를 BODex로 만듭니다. Fig. 1 기준으로 Lightning Grasp는 BODex보다 SPS 6–33배, forward 시간 20–60배 빠릅니다. 단, ZeroDex는 어포던스 항을 BODex 목적함수에 더하는데, Lightning Grasp에서는 그 자리가 "허용 접촉 영역 지정"(Sec. 5.7 Modularity)이 됩니다. |
| **FlashDexRetarget** (`flashdexretarget/`) | 둘 다 "로봇 데이터 생성을 빠르게"가 목표입니다. FlashDexRetarget은 사람 시연을 물리 시뮬레이션으로 검증한 궤적으로 바꾸고, Lightning Grasp는 정적 grasp 자세를 물리 검증 없이 대량으로 만듭니다. Lightning Grasp 출력은 RL의 시작 상태나 grasp 사전(prior)으로 쓸 수 있습니다. |
| **Mana** (같은 1저자, arXiv:2606.13677) | 이 저장소에서 따로 분석 중이라 여기서는 읽지 않았습니다. Lightning Grasp README는 이 엔진이 "DexterityGen과 앞으로 나올 연구들"을 받친다고 씁니다. |

## 4. 논문이 다루지 않았지만 관련 있는 연구 (내 판단)

| 주제 | 관련성 | 근거 수준 |
|---|---|---|
| **Reachability map / capability map** (Zacharias et al., 2007 계열) | 매니퓰레이터 작업공간을 격자로 미리 계산해 두는 고전 기법입니다. contact field는 그 아이디어를 손 표면 접촉(위치 + 법선)으로 옮긴 것으로 볼 수 있습니다. | 내 사전 지식. 서지 확인 필요 |
| **Ferrari–Canny ε-metric, force closure 고전** | 마찰 원뿔을 넣은 grasp 품질의 표준입니다. FSWO는 그보다 약한 "자기 균형" 조건이라 비교 기준이 필요합니다. | 내 사전 지식. 서지 확인 필요 |
| **GenDexGrasp, DexDiffuser [20] 같은 학습형 생성기** | 학습형 생성기는 Lightning Grasp가 만든 데이터를 학습 자료로 쓸 수 있고, 거꾸로 Lightning Grasp의 물체 배치 정책(Sec. 5.7의 data-driven search)이 될 수도 있습니다. | DexDiffuser는 논문 인용. GenDexGrasp는 내 판단 |
| **GPU BVH 라이브러리** (NVIDIA Warp, Open3D, cuBVH 등) | 비공개 커널을 공개 라이브러리로 다시 만들 때 후보가 됩니다(`04` §4.3). | 내 판단. macOS 지원 여부는 라이브러리마다 확인 필요 |

**한계:** 4절의 서지와 라이브러리 지원 범위는 내 사전 지식에 근거합니다. 게시하기 전에 원문을 직접 확인해야 합니다.
