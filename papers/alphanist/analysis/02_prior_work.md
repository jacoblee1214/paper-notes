# 02. 선행연구와 계보

번호 [n]은 AlphaNIST의 참고문헌 번호입니다. SGS[7](arXiv:2610.12465)는 원문을, OmniReset[10]은 공개 구현(UW-Lab/UWLab @ `2b40afb`)의 보상·PPO 설정을 직접 확인했습니다. 나머지는 논문의 참고문헌 목록을 옮긴 것입니다.

## 1. 계보도

```
[게임: 결과 보상만으로 전략 발견]  AlphaGo [8] → AlphaGo Zero [9]          ← 논문의 동기

[조립 시뮬레이션과 학습]
  Factory [2] (RSS 2022, Isaac Gym 접촉) → IndustReal [29] (sim-to-real) → AutoMate [20] (전문가·범용 정책)
  → Refinery [25] (초기 조건 샘플링으로 미세조정) · ARCH [26] (계층형 하이브리드)
     공통점: horizon 단축, 미리 잡은 상태, 시연·하위 기술 지정 같은 구조를 둠

[대규모 on-policy RL에서 시작 상태 다루기]
  OmniReset [10] arXiv:2603.15789 (ICLR 2026)  다양한 reset bank, 고정 dense 목적
  → SGS [7] arXiv:2610.12465 (CoRL 2026)      성공률 중간인 시작 상태에 집중, 100만 env 이상, 과제마다 정책 하나
        │
        ▼
  ★ AlphaNIST: OmniReset bank + SGS 샘플링 + 공통 이진 보상 + 정책 하나로 20개 과제

[다중 과제 RL]  PCGrad [19], PopArt [21] ↔ "단순 공유 망도 충분하다" Kurin et al. [22]  ← AlphaNIST는 후자 쪽
[망 구조]       SimBa [28] (residual + LayerNorm, ICLR 2025)
[시뮬레이터]    Isaac Lab [11], PhysX [17], Newton [18], mjlab [12]
```

**차별점 요약:** 같은 그룹(UW·NVIDIA)의 두 선행 작업을 합친 형태입니다. OmniReset은 시작 상태의 **범위**를, SGS는 그 안에서의 **배분**을 다루었고, 둘 다 과제마다 정책을 따로 두었습니다. AlphaNIST의 새 요소는 ① 20개 과제를 정책 하나로 묶은 것, ② 과제별 shaping을 없애고 공통 이진 보상만 쓴 것, ③ 다중 부품 episode입니다.

## 2. 구성 요소별 출처

| 역할 | 출처 | 이 논문에서의 쓰임 | 확인 |
|---|---|---|---|
| reset bank | OmniReset [10] | 조립 정체 e_i를 함께 저장해 20개 과제로 확장(Eq. 3) | 논문 |
| 시작 상태 샘플링 | SGS [7] | Eq. 4, "SGS의 규칙을 유지" | SGS 원문: Beta 커널 + 온도 T = 2 softmax → (p(1−p))^¼. 식 4는 지수 ½ (`03` §1) |
| 공통 이진 보상 | SGS의 "도메인별 공유 보상"(성공 + 작은 정규화) | 성공 1, 1 mm 임계, 성공 시 종료 | SGS: λ_succ = 100, 실패 −1, 행동·변화율·속도 정규화. AlphaNIST가 SGS를 "작업별 파지 shaping을 유지"한다고 한 것과 다름(`04` C13) |
| dense 비교군 계수 | OmniReset [10] | c₂ = c₃ = 0.01 | OmniReset 구현: 가중치 0.1 × Isaac Lab dt(0.1 s) = 실효 0.01. 단 성공 보상을 매 step 주고 성공으로 종료하지 않음(`04` C14) |
| PPO 설정 | (명시 없음) | 32 step, 5 epoch, minibatch 4 | OmniReset `rsl_rl_cfg.py`와 같음(γ 0.99, λ 0.95, lr 1e-4도 같을 가능성, 미확인) |
| actor·critic 망 | SimBa [28], ResNet [27] | 폭 256, 확장 1024, block 2개, 분리 망 | 논문 Fig. 3 |
| 로봇·제어 | Franka + libfranka 강성·감쇠, IndustReal [29]·AutoMate [20]의 하드웨어 맞춤 시뮬레이션 | 25 Hz Δq 목표 + 그리퍼 | 논문 |
| 시뮬레이터 | Isaac Lab [11], Factory [2]의 접촉 | GPU 병렬 환경 | 논문 |
| 과제 | NIST ATB1 [1] | 20개 조립 쌍(너트 4, 기어 3, 막대 4, 페그 4, 커넥터 5) | 논문 Fig. 2, 5 |

## 3. 이 저장소의 다른 논문과의 관계

| 논문 | 관계 |
|---|---|
| **FlashDexRetarget** (`flashdexretarget/`) | 둘 다 "정책 하나를 이질적인 과제 모음 전체에 학습하면 각자 학습할 때보다 낫다"를 보입니다. FlashDexRetarget은 손-물체 시연 50–1,000개, AlphaNIST는 조립 과제 20개입니다. **방향은 반대**입니다. FlashDexRetarget은 off-policy(FlashSAC)로 바꿔 300M step에 끝내고 PPO는 거의 0이었습니다(Fig. 4d). AlphaNIST는 on-policy PPO를 64K–256K 환경, 512억–2,050억 step으로 밀어붙입니다. 시작 상태 쪽도 대조됩니다. FlashDexRetarget은 레퍼런스를 **균등하게** 뽑아 개선 여지를 남겼고(`flashdexretarget/analysis/04_code_verification.md` D13), AlphaNIST는 성공률 기반 샘플링이 256K에서 +13.1%p를 냈습니다. 보조 장치도 다릅니다. FlashDexRetarget은 물체 보조 힘 커리큘럼과 dense 기하 보상을 쓰고, AlphaNIST는 이진 보상만 씁니다. 공통 교훈은 "다양한 시작 상태(레퍼런스 위 무작위 프레임, reset bank)가 학습을 가능하게 한다"는 것입니다. |
| **DDPG** (`ddpg/`) | AlphaNIST는 DDPG 계열(off-policy actor-critic)이 아니라 PPO입니다. 이진 보상, 긴 horizon, 수만 개 병렬 환경이라는 조건에서 off-policy를 쓰지 않은 이유는 논문에 없습니다. `ddpg` 분석의 교훈(구현마다 기본값이 다르니 설정을 명시할 것)이 여기서도 그대로 적용됩니다. AlphaNIST는 γ, λ, 학습률을 적지 않았습니다. |
| **DexScrew** (`dexscrew/`) | 둘 다 시뮬레이션 RL로 나사·너트 조작 기술을 배웁니다. DexScrew는 너트를 회전 관절로 단순화해 다지 손으로 돌리고, AlphaNIST는 나사산 접촉을 그대로 시뮬레이션해 평행 그리퍼로 체결합니다(한쪽 손끝 두드림 같은 전략이 나옴). DexScrew는 실제 텔레옵과 촉각 BC로 넘어갔고, AlphaNIST는 아직 시뮬레이션 단계입니다. |

## 4. 논문이 다루지 않았지만 관련 있는 연구 (내 판단)

| 주제 | 관련성 | 근거 수준 |
|---|---|---|
| **Reverse curriculum** (Florensa et al. [24], 논문이 인용) | 목표 근처에서 시작해 점점 멀리 가는 방식입니다. AlphaNIST의 reset bank + Beta 샘플링은 이것의 자동판으로 볼 수 있습니다. 쉬운 시작점이 많이 섞여 있다는 점(`03` §3)도 같은 맥락입니다. | 논문 인용 + 내 해석 |
| **Prioritized Level Replay, Sampling for Learnability(SFL)** | SGS가 직접 비교한 방법입니다(SGS 2절, 표 1). SFL은 p(1−p)를 쓰는데, AlphaNIST 식 4의 √(p(1−p))는 SFL과 SGS(T = 2)의 중간 지수입니다. | SGS 원문 |
| **POET, ACCEL 등 비지도 환경 설계** | 성공률 기반 배분의 상위 계보입니다. | 내 사전 지식, 서지 확인 필요 |
| **희소 보상 + 종료 설계**(정지 상태 보상, 종료 시 보상 처리) | 성공 시 종료에 매 step 양의 보조 보상을 붙이면 완료를 미루는 유인이 생깁니다. potential-based shaping(Ng et al., 1999)은 이 문제를 피하는 고전적 방법입니다. | 내 사전 지식(Ng et al.은 서지 확인 필요) + `03` §2 toy |

## 5. 관련 GitHub 저장소 (2026-10-28 `gh api` 기준)

공식 코드는 아직 없습니다. AlphaNIST는 "심사·수정 기간 뒤 공개 예정"(Sec. I)이고, SGS 페이지는 "Code (coming soon)"입니다.

| 저장소 | 무엇인가 | 공식 여부 | 라이선스 | ★ | 마지막 push | 확인한 커밋 |
|---|---|---|---|---|---|---|
| alpha-nist/alpha-nist.github.io | 프로젝트 웹사이트(영상, 그림) | 공식, 코드 없음 | 표기 없음 | 1 | 2026-10-15 | `c568417` |
| SGS-RL/sgs-rl.github.io | SGS 프로젝트 웹사이트 | 공식, 코드 없음 | 미확인 | 8 | 2026-10-09 | `cb2423d` |
| UW-Lab/UWLab | OmniReset[10] 구현이 들어 있는 UW 연구실 Isaac Lab 확장 | 구성 요소 공식 | BSD-3-Clause | 340 | 2026-09-28 | `2b40afb` (보상·PPO·종료 설정 파일을 읽음) |
| isaac-sim/IsaacLab | 시뮬레이션 틀[11], Factory·AutoMate 환경 포함 | 구성 요소 공식 | BSD-3 계열 | 8,504 | 2026-10-28 | `210ee79` (`reward_manager.py`의 dt 곱셈 확인) |
| newton-physics/newton | GPU 물리 엔진[18] | 관련 공식 | Apache-2.0 | 5,855 | 2026-10-28 | `d632fb7` (메타데이터만) |
| suniique/assembly-nist | **독립 MuJoCo 구현**: Franka, 20개 과제, 이진 보상, Beta 샘플링(식 4 그대로, ε_q = 1e-4, L = 50), SimBa식 망, 사전학습 checkpoint 5개 | 비공식(저자와 무관하다고 명시) | MIT | 64 | 2026-10-23 | `eb5d15e` (clone, README·sampler·train 설정 확인) |
| ZoreAnuj/AlphaNIST | 비공식 구현. 실험은 UR7e와 다른 조립 과제(모터 마운트 등)로 바꿈 | 비공식 | 자체 문구(README 참조) | 0 | 2026-10-22 | `e0eb350` (README만) |
| dcbenwu/alpha-nist-repro | README뿐인 재현 계획 저장소 | 비공식 | 표기 없음 | 3 | 2026-10-14 | `9551812` (README만) |

**독립 MuJoCo 구현(suniique/assembly-nist)에서 볼 점:**
- 8,192 환경, 10K PPO iteration에서 평가 성공 98.4%라고 보고합니다(AlphaNIST 프로토콜: 시드 5 × 과제당 32회, 확률적 행동). 시작 상태별 값도 있습니다. 12.5K iteration 기준으로 놓인 부품 98.1%, 집기 98.8%, 운반 98.8%, 부분 조립 99.0%, 끼우기 직전 98.7%입니다. README 표의 부품 종류별 값을 과제 수로 가중 평균하면 보고된 전체 값(98.4%, 98.7%)과 0.03%p 안에서 맞습니다(`04` C12). 시뮬레이터, 자산, 접촉 모델이 달라 AlphaNIST와 직접 비교할 수는 없다고 README도 적었습니다.
- 처음 시도에서 막힌 지점을 기록해 두었습니다.
  - 탐색 유지: 학습되는 표준편차 초기값 1.0, entropy 0.005로 바꾸자 정체가 풀렸습니다.
  - 손목 속도: 관절별 행동 스케일을 조정해야 15초 안에 방향을 바꿀 수 있었습니다.
  - 물리 수정: 너트에 반경 방향 0.2 mm 여유를 주고, 기어·나사산은 0.5 ms step을 썼습니다.
  - 논문에 없는 설정이 성능을 좌우한다는 증거입니다.
- 학습 자원: GPU learner 1장과 CPU worker 960개(약 30 노드)로 10K iteration에 약 26시간 걸렸습니다. 평가는 16 CPU 코어로 3,200 episode에 약 1시간입니다. 접촉 플러그인은 x86-64용(`-mf16c`)이라 이 맥(arm64)에서는 그대로 빌드되지 않습니다.

**한계:** 4절의 POET·ACCEL, Ng et al. 서지는 내 사전 지식이라 게시 전에 원문을 확인해야 합니다. 비공식 저장소의 성능 주장은 저자 자기보고이며 이 분석에서 실행해 확인하지 않았습니다.
