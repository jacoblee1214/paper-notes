# 02. 선행연구와 계보

번호 대신 저자-연도로 적었습니다(논문 참고문헌 형식). arXiv ID는 논문 참고문헌에 적힌 값을 옮겼습니다. 이 문서의 각 논문을 따로 열어 검증하지는 않았고, GitHub 정보는 2026-10-29에 `gh api`로 확인했습니다.

## 1. 계보도

```
[사람 손 synergy]  Santello et al. 1998 (자세 대부분을 소수 협응 패턴이 설명)
        │
[eigengrasp 계획]  Ciocarlie et al. 2007, Ciocarlie & Allen 2009
        │
[협응을 행동 공간에]  DFG (Agarwal 2023) · DextrAH-G (Lum 2024) · VAE synergy (Dimou 2021, 2024)
                     PCHands (Puang 2025) · Cross-embodiment grasping (Yuan 2025) · SAR (Berg 2024) · LAMP (Yang 2026)
        │  한계: 저차원 행동은 미세 조정이 안 됨 (Lum 2024, Lee 2026 ADEPT, Gupta 2026a Grasp to Act)
        ▼
[잔차로 표현 복구]  LUCID (Gupta 2026b, arXiv:2606.11628): eigen 계수 + 관절 잔차 = Eigen-Residual-Δ
        │  질문: 이득이 "협응 명령"인가 "상관된 탐색"인가?
        ▼
  ★ EigenDEXplore: 행동은 관절 그대로, 사람 방향은 탐색 잡음에만

[구조화된 탐색 (사람 사전 없음)]  gSDE (Raffin 2021) · pink noise (Eberhard 2023) · colored noise PPO (Hollenstein 2024)
                                Lattice (Chiappa 2023, 정책이 만든 actuator 간 상관)
```

**차별점 요약:** 협응 사전 분포를 쓰는 선행연구는 그것을 **행동 표현**에 넣었습니다. 구조화된 탐색 연구는 **사람 사전 없이** 시간 상관이나 정책 유래 상관을 썼습니다. 이 논문은 둘을 갈라서 "사람 사전 분포를 탐색 공분산에만 넣는다"를 독립 변수로 만들었습니다. 비교 대상이 같은 저자 그룹의 선행 방법(DextrAH-G, Grasp to Act, LUCID)이고, 벤치마크도 같은 그룹 것(SimToolReal, DexMachina 공저)이 많습니다.

## 2. 구성 요소별 출처

| 역할 | 출처 | 이 논문에서의 쓰임 |
|---|---|---|
| 사람 자세 데이터 | EgoSuite-Open100K (Lightwheel 2026), GRAB (Taheri 2020), ARCTIC (Fan 2023) | PCA 사전 분포. 공개 기저는 세 데이터 가중 혼합(`04` D4) |
| 운동학 리타겟 | ManipTrans (Li 2025, CVPR) | 사람 손 → 로봇 관절. 공개 코드에는 리타겟·fitting 파이프라인이 없음 |
| PPO | Schulman 2017, rl_games | log-prob·entropy·KL을 low-rank + diag Gaussian으로 계산 |
| 비구조 RL 벤치마크 | SimToolReal (Kedia 2026, RSS), DeXtreme (Handa 2023, ICRA) | 행동 공간 비교, 실물 |
| 레퍼런스 추적 RL | DexMachina (Mandi 2026, ICML) | 커리큘럼 유무 비교 |
| 표본 기반 리타겟 | SPIDER (Pan 2025, arXiv:2511.09484) | 후보 섭동만 교체 |
| 리타겟 과제 데이터 | OakInk2 (Zhan 2024) | SPIDER 과제 10개 |
| 실물 자세 추정 | FoundationPose (Wen 2024), AprilCube (Park & Agrawal 2026) | 도구 waypoint, 큐브 자세 |
| 저차원 분포 | `torch.distributions.LowRankMultivariateNormal` | Eq. 2를 그대로 구현(`04` §1) |

## 3. 이 저장소의 다른 논문과의 관계

| 논문 | 관계 |
|---|---|
| **DexPolicy** (`dexpolicy/`) | 가장 가까운 주제입니다. 둘 다 dexterous RL의 **탐색 잡음**만 바꾸고 행동 표현은 그대로 둡니다. DexPolicy는 σ의 **크기**를 시간에 따라 줄이고, EigenDEXplore는 σ의 **모양**(공분산 방향)을 바꿉니다. DexPolicy 분석에서는 학습되는 σ가 짧은 예산에서 거의 움직이지 않았습니다(0.202 → 0.185). EigenDEXplore도 σ_q와 σ_e를 학습하지만 초기값에서 크게 벗어나지 않을 수 있습니다. 그렇다면 **초기 분산의 크기**가 결과를 좌우하므로, 공개 SimToolReal preset의 두 배 분산(`04` D1)은 더 중요한 교란이 됩니다. 두 방법은 함께 쓸 수 있습니다(σ_q·σ_e를 따로 스케줄링). |
| **DDPG** (`ddpg/`) | DDPG의 OU 잡음은 **시간** 상관, EigenDEXplore는 **관절 간** 상관입니다. README가 권하는 AR(1) 변형(ρ = 0.94, 지속 약 16 step)은 두 축을 합친 것입니다. DDPG 분석에서 OU의 dt가 빠져 상관 시간이 구현마다 100배 달랐던 것처럼, 이 변형도 지속 길이를 제어 주기·행동 청크와 함께 적어야 비교할 수 있습니다. |
| **Do as I Do** (`do_as_i_do/`) | Do as I Do의 리타겟은 SPIDER 기반 annealed sampling MPC입니다. EigenDEXplore의 SPIDER 실험은 바로 그 표본 분포만 바꿔 cold start 비용을 25% 줄였으므로, Do as I Do 파이프라인에 그대로 꽂을 수 있는 후보입니다. 다만 Do as I Do의 이득 대부분은 warmup에서 나왔습니다(+41%p). 둘을 함께 비교한 실험은 없습니다. |
| **FlashDexRetarget** (`flashdexretarget/`) | DexMachina식 물체 보조 커리큘럼을 쓰는 대규모 off-policy 리타겟입니다. EigenDEXplore의 DexMachina 결과(커리큘럼과 보조 보상이 있으면 이득이 +6%로 줄어듦)를 보면, FlashDexRetarget처럼 보조 장치가 많은 파이프라인에서는 구조화 탐색의 여지가 작을 것으로 예상됩니다. FlashSAC는 tanh-Gaussian 대각 정책이라, Eq. 2를 쓰려면 tanh 이전의 low-rank 공분산과 엔트로피 목표를 다시 정해야 합니다. |
| **AlphaNIST** (`alphanist/`) | 이진 보상과 성공률 기반 시작 상태 샘플링으로 탐색 문제를 푼 연구입니다. "보상 설계가 적을수록 이득이 크다"는 EigenDEXplore의 결론과 같은 방향의 문제의식입니다. 다만 AlphaNIST는 평행 그리퍼라 사람 손 사전 분포를 쓸 수 없습니다. |
| **DexScrew** (`dexscrew/`) | IsaacGym PPO로 손가락 회전 기술을 배웁니다. 기술 하나를 배우는 설정이라, 이 논문의 DeXtreme 실험(재배향, +40%)과 가장 비슷한 조건입니다. |

## 4. 논문이 다루지 않았지만 관련 있는 연구 (내 판단)

| 주제 | 관련성 | 근거 수준 |
|---|---|---|
| **CMA-ES·natural ES** | 표본 분포의 공분산을 온라인으로 적응시키는 고전 방법입니다. EigenDEXplore는 공분산을 사람 데이터로 **미리 고정**한 셈이고, 저자들이 한계로 든 "학습 중 기저 적응"이 바로 이 방향입니다. SPIDER 같은 표본 최적화에서는 CMA식 공분산 갱신과 비교해야 공정합니다. | 사전 지식, 서지 확인 필요 |
| **MPPI 공분산 학습 / 학습된 proposal** | 표본 MPC에서 proposal 분포를 학습하는 연구가 있습니다. 사람 사전 분포 대 과제에서 학습한 proposal 비교가 빠져 있습니다. | 사전 지식 |
| **SAC류 엔트로피 목표** | 대각 Gaussian의 엔트로피 목표(−d)를 low-rank 공분산에 그대로 쓰면 의미가 달라집니다. off-policy로 옮길 때 다시 정해야 할 부분입니다. | 내 해석 |
| **Action chunking 정책의 초기 잡음** | flow·diffusion 정책은 행동 청크를 Gaussian 잡음에서 생성합니다. 이 초기 잡음 공분산을 사람 협응 공분산으로 바꾸는 것은 같은 아이디어의 생성 모델 버전입니다. | 내 해석 |

## 5. 관련 GitHub (2026-10-29 확인)

| 저장소 | 무엇인가 | 공식 여부 | 라이선스 | ★ | 마지막 push | 확인한 커밋 |
|---|---|---|---|---|---|---|
| hgupt3/eigendexplore | 방법 패키지, 6개 손 PCA 자산, 벤치마크 어댑터·패치 | 공식 | MIT | 53 | 2026-10-11 | `5d57180` (clone, 커밋 1개) |
| eigendexplore/eigendexplore.github.io | 프로젝트 페이지 | 공식 | 표기 없음 | 0 | 2026-10-13 | `5ee962b` (메타데이터만) |
| hgupt3/simtoolreal | SimToolReal 포크(Isaac Lab 포팅), 벤치마크가 고정해 씀 | 공식 벤치마크 원본의 포크 | MIT | 0 | 2026-09-21 | `fddff74` (고정 커밋, 학습 설정 파일만 읽음) |
| tylerlum/simtoolreal | SimToolReal 공식 구현(위 포크의 원본) | 벤치마크 공식 | MIT | 424 | 2026-09-21 | 메타데이터만 |
| isaac-sim/IsaacGymEnvs | DeXtreme 호스트 | 벤치마크 공식 | 기타(표준 아님) | 2,950 | 2024-10-26 | `aeed298` (고정 커밋, PPO 설정 파일만 읽음) |
| facebookresearch/spider | SPIDER 호스트 | 벤치마크 공식 | 기타(표준 아님) | 590 | 2026-09-22 | `7123845` (고정 커밋, 메타데이터만) |
| Denys88/rl_games | 두 RL 벤치마크의 PPO | 의존성 | MIT | 1,407 | 2026-10-29 | DeXtreme는 1.6.5, SimToolReal은 vendored `da1c321`(메타데이터만) |
| mandizhao/dexmachina | DexMachina (공개 벤치마크에는 없음) | 관련 공식 | MIT | 277 | 2026-03-23 | 메타데이터만 |
| DLR-RM/stable-baselines3 | gSDE (Raffin 2021) 구현 | 관련 | MIT | 13,943 | 2026-10-26 | 메타데이터만 |
| martius-lab/pink-noise-rl | pink noise 탐색 (Eberhard 2023) | 관련 공식 | MIT | 50 | 2025-11-20 | 메타데이터만 |
| amathislab/lattice | Lattice 잠재 탐색 (Chiappa 2023) | 관련 공식 | MIT | 49 | 2024-02-23 | 메타데이터만 |

**한계:** 4절의 서지와 CMA-ES 계열 비교는 내 사전 지식에 근거합니다. 블로그나 git에 올리기 전에 원문을 직접 확인해야 합니다.
