# 02. 선행연구와 계보

GNR은 물리 기반 리타겟팅의 "시연마다 최적화" 비용을, 풀린 궤적으로 학습한 조건부 생성기로 상각(amortize)하려는 연구입니다. 아래 서지는 논문 참고문헌에서 가져왔습니다. §4의 "논문이 인용하지 않은 연구"는 내 판단이고, 서지 일부는 원문을 다시 확인하지 않았습니다(해당 항목에 표시).

## 1. 계보도

```
[운동학 리타겟팅]  keypoint·손끝 IK
  DexPilot(Handa 2020) · AnyTeleop(Qin 2023) · GeoRT(Yin 2025, 신경망 IK)
        │  동역학 무시 → 물리에서 실행 불가 (HOT3D에서 IK 18.3%)
        ▼
[물리 기반 리타겟팅]
  ├─ RL: DexTrack(Liu 2025) · ManipTrans(Li 2025) · DexMachina(Mandi 2025) · CHORD(Zhu 2026)
  │        시연(또는 모음)마다 정책 학습, 보상 설계
  └─ sampling MPC: MPPI(Williams 2017) · iCEM(Pinneri 2021) · CMA-ES · DIAL-MPC(Xue 2024)
           → SPIDER(Pan 2025, annealed sampling + 접촉 유도) → Do as I Do(Paliwal 2026)
           시연마다 독립 최적화, 해가 다음 시연으로 옮겨 가지 않음
        │
        ▼  "풀린 궤적에서 배워 다음 시연을 싸게"
[상각]
  Generative Predictive Control(Kurtz & Burdick 2025): 과제별로 MPC를 flow matching 정책으로 증류
  GNR(2026): 시연(IK 레퍼런스 + 기하)을 조건으로 잔차 궤적을 생성
             open-loop pass@S  또는  MPC 초기값(GNR+MPC)
```

## 2. 구성 요소별 출처

| 구성 요소 | 출처 | GNR에서의 역할 |
|---|---|---|
| 리타겟팅 정식화 (제약 최적화, 비용 J) | SPIDER (Pan et al., 2025) | Sec. 3.1의 문제 정의 |
| 시드 생성기, 기준선 | DIAL-MPC (Xue et al., 2024) | HOT3D 300×10, ShapeFilter 500×6, NIST 3단계 (Table 5) |
| 생성 모델 | Flow matching (Lipman et al., 2022) | 잔차 Δ의 조건부 생성, Euler 10단계 |
| 백본 | Temporal U-Net (Ronneberger 2015), FiLM (Perez 2018) | 19.57M 파라미터 (HOT3D) |
| 기하 인코더 | Point Transformer V3 (Wu et al., 2024) | 점 1,024개 |
| 시뮬레이터 | SuperDex Physics (The Mochi Team, 2026) | 암시적 접촉, 분포 접촉력, 비볼록 충돌 |
| 사람 시연 | HOT3D (Banerjee et al., 2025), OptiTrack 15대 + 착용형 외골격 | 다양성 축, 정밀도 축 |
| 기하 변형 | SAM 3D (Chen et al., 2026b) | 재생성, 가림 보완, 회전 후 재생성 |
| 성공 기준 150 mm | CHORD (Zhu et al., 2026) | HOT3D γ_track |
| 정밀 과제 | NIST assembly board (Lian et al., 2021) | M8 노브 체결 |
| 다중 embodiment 벤치마크 | SPIDER 공개 벤치마크 | 손 5종, Table 16 |
| pass@S | LLM 코드 생성의 pass@k 관례 (논문은 출처를 밝히지 않음) | 성공 지표 |

## 3. 이 저장소의 다른 논문과의 관계

### 3.1 세 가지 리타겟팅 방식 비교

| | 운동학 (IK, GeoRT) | 물리 기반 최적화 (RL, sampling MPC) | 생성 (GNR) |
|---|---|---|---|
| 출력 | 관절 자세 열 | 물리에서 검증된 제어열 | 제어열 후보 S개 (시뮬레이터로 거름) |
| 시연 하나당 비용 | ms 단위 (GNR 표의 IK 39 core-s는 시뮬레이터 재생 포함) | HOT3D DIAL-MPC 2,689 core-s, RL은 GPU-h 단위 | HOT3D S=256에서 354 core-s, 별도 학습 비용 |
| 시연 사이 지식 전이 | 없음(규칙 고정) | MPC는 없음. 다중 레퍼런스 RL은 정책 안에서 공유 | 생성기가 공유 |
| 물리 실현 가능성 | 보장 없음 | 최적화 목표에 포함 | 생성 후 시뮬레이터 거부 표집으로 확보 |
| 전형적 실패 | 침투, 접촉 누락, 물체 낙하 | 국소해, 예산 부족, 보상 해킹 | 학습 분포 밖 외삽, 무거운 꼬리 (`03` §2–3) |
| 필요한 것 | 손 모델 | 시뮬레이터, 비용·보상 | 시뮬레이터 + **시드 최적화기** + 학습 GPU |

GNR은 물리 기반 최적화를 대체하지 않고 그 위에 얹습니다. 시드 데이터는 DIAL-MPC가 만들고, GNR은 그 해의 분포를 배워 제안 분포로 씁니다. 그래서 "생성 리타겟팅"보다 **학습된 제안 분포를 쓰는 sampling 리타겟팅**이라고 부르는 편이 정확합니다. 논문 제목의 "neural retargeting"은 GeoRT 같은 신경망 운동학 리타겟팅과 이름이 겹치지만, 출력이 물리 검증된 제어열이라는 점이 다릅니다.

### 3.2 개별 논문

| 논문 (폴더) | 관계 |
|---|---|
| **FlashDexRetarget** (`flashdexretarget/`) | 목표가 같습니다. 둘 다 "시연마다 최적화"의 선형 비용을 없애려고 합니다. FDR은 레퍼런스 조건부 RL 정책 하나로 시연 모음 전체를 추적하고(폐루프, Isaac Sim, GPU-h), GNR은 MPC 해로 학습한 생성기에서 open-loop 후보를 뽑습니다(CPU core-h). 둘 다 HOT3D를 쓰지만, FDR의 엄격한 기준 SR_MT-Obj는 물체 3 cm / 30°이고 GNR HOT3D 기준은 150 mm / 40°입니다. 성공률을 직접 비교하면 안 됩니다. GNR은 FDR을 인용하지 않습니다(FDR v1 2026-10-01, GNR v1 2026-10-08 공개). FDR은 RL 학습 비용을 한 번 치르고, GNR은 MPC 시드 비용을 치릅니다. 두 방식 모두 "어떤 시연에서 실패하는가"는 비슷한 꼬리 문제를 보입니다(FDR 1,000개에서 변환 비율 61%, GNR MPC-failure 48%). |
| **Do as I Do** (`do_as_i_do/`) | GNR이 대체하려는 쪽입니다. SPIDER식 annealed sampling MPC(MuJoCo Warp, 1,024표본)로 단안 영상 복원 레퍼런스를 시연마다 따로 리타겟합니다. GNR Table 3에도 500개, 100 mm 기준으로 실렸습니다. Do as I Do의 transition reward처럼 비용을 고치는 방향과, GNR처럼 초기값·제안 분포를 고치는 방향은 함께 쓸 수 있습니다. |
| **DexForce** (`dexforce/`) | 둘 다 "힘 라벨이 붙은 손 조작 데이터"를 만듭니다. DexForce는 손끝 6축 센서로 kinesthetic 시연의 **실측 힘**을 얻고(소규모), GNR은 시뮬레이터에서 **계산한 접촉력**을 대규모로 얻습니다. GNR 라벨은 접촉 강성(기본값의 1/10)에 의존하고 실측과 대조하지 않았습니다. DexForce의 x_f = x_o + k_f f 같은 힘 기반 행동을 GNR 데이터로 만들 수 있을지가 둘을 잇는 질문입니다. |
| **Mana**, **Lightning Grasp** (`mana/`, `lightning_grasp/`) | Zhao-Heng Yin 계열입니다. GNR이 운동학 리타겟팅의 예로 인용한 GeoRT의 저자이기도 합니다. Mana는 절차적 파지 keyframe 사이를 motion planning과 짧은 RL로 채우고, Lightning Grasp는 contact field로 파지를 빠르게 합성합니다. 둘 다 사람 시연 없이 파지를 만드는 쪽이고, GNR은 사람 시연을 물리로 옮기는 쪽입니다. GNR의 파지 재설정 단계(App. C.3)는 Lightning Grasp류 합성 파지로 바꿀 수 있는 자리입니다. |
| **GraspSense** (`graspsense/`) | 둘 다 SAM 3D로 물체 메쉬를 만듭니다. GraspSense는 인식용이고, GNR은 같은 물체의 기하 변형(물체당 200개)을 만드는 데이터 증강에 씁니다. SAM 3D 메쉬의 물리 타당성을 검증하지 않는다는 점이 공통 약점입니다. |
| **RoboSimStudio** (`robosimstudio/`) | 같은 SuperDex(`facebookresearch/project_superdex`)를 씁니다. RSS는 FEM 변형체용으로, GNR은 강체 손 조작 시뮬레이터로 씁니다. |
| **Contact-Grounded Policy** (`contact_grounded_policy/`) | 같은 Meta Reality Labs Research입니다. CGP는 내부 FEM 시뮬레이터로 촉각 정책을 학습했습니다. GNR의 시뮬레이션 접촉력 라벨은 CGP 같은 접촉 기반 정책의 학습 데이터가 될 수 있습니다. 두 시뮬레이터가 같은 계열인지는 확인하지 못했습니다(가설). |

## 4. 논문이 다루지 않았지만 관련 있는 연구 (내 판단)

- **FlashDexRetarget (arXiv:2610.01849):** §3.2. 같은 문제를 RL 쪽에서 상각합니다. 같은 HOT3D 클립과 같은 기준으로 비교하면 "생성기 + 거부 표집"과 "다중 레퍼런스 정책" 중 어느 쪽이 싼지 알 수 있습니다.
- **학습된 warm start로 궤적 최적화 가속:** Memory of Motion(Lembono et al., ICRA 2020)처럼, 풀린 문제에서 초기값 분포를 배워 최적화를 빠르게 하는 연구가 있습니다. GNR+MPC는 이 계열의 접촉 많은 손 조작 버전입니다. 논문은 이 계열을 인용하지 않습니다(서지는 다시 확인하지 않았음).
- **LLM의 test-time scaling과 검증기:** pass@k와 "많이 뽑고 실행해서 거르기"는 코드 생성에서 온 방식입니다(Codex, Chen et al., 2021). 무거운 꼬리 때문에 수익이 줄어드는 현상도 같은 구조입니다(`03` §2). 논문은 출처를 밝히지 않습니다.
- **잔차 학습:** IK 기준에 잔차를 더하는 설계는 FDR의 손목 잔차 행동, residual policy learning과 같은 발상입니다. App. E.4의 잔차 대 절대 비교는 이 선택을 뒷받침합니다.

## 5. GitHub 정리 (2026-11-03 `gh search repos`, `gh api` 확인)

공식 코드는 없습니다. 논문 제목, "GNR retargeting", 저자 계정(`YY-GX`, `gaodechen`)을 검색했고, 프로젝트 페이지 링크와 논문 본문에도 코드 주소는 없습니다.

| 저장소 | 무엇인가 | 공식 여부 | 라이선스 | ★ | 마지막 push | 확인 커밋 |
|---|---|---|---|---|---|---|
| [YY-GX/GNR](https://github.com/YY-GX/GNR) | 프로젝트 페이지 정적 사이트(영상, 그림, 스크립트). 코드·데이터 없음. README: "Code and Dataset buttons are omitted until release" | 공식(2저자 계정) | 표기 없음 | 0 | 2026-10-30 | `d6cf09e` (README와 파일 목록) |
| [facebookresearch/project_superdex](https://github.com/facebookresearch/project_superdex) | SuperDex 엔진·Studio·Teleop. `assets/bots/arm_hand_combos/fr3_dg5f_*`에 FR3 + DG-5F 모델이 있음. 지원 OS에 macOS(ARM) 표기 | 논문이 쓴 시뮬레이터(GNR 코드 아님) | Apache-2.0 | 807 | 2026-11-03 | `156fe13` (README와 파일 목록만) |
| [LeCAR-Lab/dial-mpc](https://github.com/LeCAR-Lab/dial-mpc) | DIAL-MPC 원 구현. JAX + Brax(MuJoCo) 기반, CUDA 권장 | 시드 생성기 원본(GNR은 SuperDex 위에 다시 구현한 것으로 보임) | Apache-2.0 | 1,012 | 2025-05-28 | `871c84f` (README만) |
| [facebookresearch/spider](https://github.com/facebookresearch/spider) | SPIDER 물리 기반 리타겟팅, 다중 embodiment 벤치마크 | 비교 기준선 | CC BY-NC 4.0 (LICENSE 파일) | 592 | 2026-09-22 | `4471700` (메타데이터만) |
| [facebookresearch/hot3d](https://github.com/facebookresearch/hot3d) | HOT3D 데이터 API. 데이터는 projectaria.com·Hugging Face 배포(약관 확인 필요) | 데이터 | Apache-2.0 (코드) | 292 | 2026-06-16 | `146b34a` (메타데이터만) |
| [Pointcept/PointTransformerV3](https://github.com/Pointcept/PointTransformerV3) | 기하 인코더 | 구성 요소 | CC BY-NC 4.0 (LICENSE 파일) | 1,986 | 2026-10-31 | `4e277e6` (메타데이터만) |
| [facebookresearch/sam-3d-objects](https://github.com/facebookresearch/sam-3d-objects) | 기하 변형 생성 | 구성 요소 | SAM License | 7,626 | 2026-06-02 | `f91db41` (메타데이터만) |

clone한 저장소는 없습니다. GNR 코드가 없어서 코드 대조(04_code_verification) 대신 주장 검증(`04_claim_verification.md`)을 했습니다.
