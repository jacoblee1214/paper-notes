# 02. 선행연구, 후속 계보, 공개 구현

논문 참고문헌은 저자-연도로 표기합니다(이 논문은 번호 인용을 쓰지 않습니다). 후속 연구의 arXiv ID는 TD3(`1802.09477`, FlashSAC 코드 주석에서 확인)만 확인했고, 나머지는 내 사전 지식이라 게시 전에 확인해야 합니다.

## 1. 계보도

```
[연속 행동 actor-critic]
  Prokhorov & Wunsch 1997 (adaptive critic designs)
  NFQCA (Hafner & Riedmiller 2011)     신경망 actor-critic, 배치 학습
  DPG (Silver et al. 2014)              결정적 정책 경사 정리, 선형·타일 코딩 근사
  SVG(0) (Heess et al. 2015)            Q-critic + 확률적 정책 (DPG는 그 결정적 극한)
        │
[DQN의 안정화 장치]  DQN (Mnih et al. 2013, 2015): 재생 버퍼 + 표적망
        │
  ★ DDPG (Lillicrap et al. 2016) = DPG + 재생 버퍼 + soft 표적망 + 배치 정규화 + OU 잡음
        │
  ├─ TD3 (Fujimoto et al. 2018, arXiv:1802.09477)   과대추정 대응: twin Q의 최솟값, 표적 정책 평활화, actor 지연 업데이트
  ├─ SAC (Haarnoja et al. 2018)                      확률적 정책 + 최대 엔트로피, 자동 온도 조절
  ├─ D4PG (Barth-Maron et al. 2018)                  분포형 critic, n-step, 분산 수집
  ├─ 파라미터 공간 잡음 (Plappert et al. 2018)        OpenAI baselines DDPG의 기본 탐색
  ├─ CrossQ (Bhatt et al. 2024)                      critic에 batch renorm, 표적망 제거 → DDPG의 BN 주장을 다시 꺼냄
  └─ 대규모 off-policy: FlashSAC (Kim et al. 2026)   범주형 critic, twin-Q 앙상블, 큰 버퍼·critic (이 저장소 flashdexretarget/)
```

**DDPG의 위치:** 알고리즘 자체(Eq. 6)는 DPG와 같습니다. 기여는 "신경망으로 키울 때 무엇이 필요한가"를 보인 공학적 조합과 대규모 평가입니다. 이후 계보는 대부분 DDPG의 약점을 고치는 방향입니다. TD3는 과대추정, SAC는 탐색과 민감도, D4PG는 분포와 규모, CrossQ는 표적망 대신 정규화를 다룹니다.

## 2. 구성 요소별 출처

| 구성 요소 | 출처 | DDPG에서의 쓰임 |
|---|---|---|
| 결정적 정책 경사 (Eq. 6) | Silver et al. 2014 | actor 업데이트 |
| off-policy Q 학습 (Eq. 4–5) | Watkins & Dayan 1992 | critic 업데이트 |
| 재생 버퍼, 표적망 | Mnih et al. 2013, 2015 (DQN) | 버퍼 10⁶. 표적망은 hard copy 대신 soft update(τ = 0.001) |
| 배치 정규화 | Ioffe & Szegedy 2015 | 관측 단위 차이 해소 |
| 자기상관 탐색 잡음 | Uhlenbeck & Ornstein 1930, Wawrzyński 2015 | OU 과정 θ = 0.15, σ = 0.2 |
| Adam | Kingma & Ba 2014 | actor 10⁻⁴, critic 10⁻³ |
| ReLU | Glorot et al. 2011 | 은닉층 |
| 비교 기준 계획기 | iLQG (Todorov & Li 2005), MPC (Tassa et al. 2012) | 정규화 점수 1 |
| 재생 버퍼 actor-critic 선행 | Wawrzyński 2009, Wawrzyński & Tanwani 2013 | cheetah 환경 출처이기도 함 |

## 3. 이 저장소의 다른 논문과의 관계

| 논문 | 관계 |
|---|---|
| **FlashDexRetarget** (`flashdexretarget/`) | FlashSAC 학습기는 DDPG 계보의 직계 후손입니다. 재생 버퍼, soft 표적망(τ = 0.01), 결정적 경사를 대신하는 재매개화 경사, TD3의 twin-Q(코드 주석에 arXiv:1802.09477 인용)를 씁니다. 그 논문의 Fig. 4(d)에서 버퍼(10M → 50M)와 critic 폭(256 → 1024)이 성능 차이의 대부분을 만든 결과는, DDPG가 처음 보인 "off-policy + 재생 버퍼"의 데이터 효율이 규모를 키워도 유지된다는 사례입니다. |
| **DexScrew** (`dexscrew/`) | PPO(on-policy)로 손가락 기술을 학습합니다. DDPG 논문은 TRPO 같은 on-policy 방법이 "데이터 효율이 상당히 낮아 보인다"고 했고(Sec. 5), FlashDexRetarget은 다중 레퍼런스 설정에서 같은 스텝 예산이면 PPO가 거의 0이라는 결과를 냈습니다. 한 과제만 학습하는 DexScrew 설정에 바로 적용되지는 않습니다. |

## 4. 공개 구현 (GitHub)

DeepMind는 이 논문의 코드를 공개하지 않았습니다. 아래 표는 2026-10-28에 `gh api`로 확인한 값입니다. "대조"는 이 분석의 `04`에서 실제로 코드를 읽어 논문과 비교한 구현입니다.

| 저장소 | 공식 여부 | 프레임워크 | 라이선스 | 별 | 마지막 push | 확인 커밋 | DDPG 파일 | 대조 |
|---|---|---|---|---|---|---|---|---|
| [DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3) | 비공식 (유지보수 활발) | PyTorch | MIT | 13,940 | 2026-10-26 | `0cdcd20` | `stable_baselines3/ddpg/ddpg.py` (TD3의 하위 클래스) | ✅ |
| [openai/spinningup](https://github.com/openai/spinningup) | 비공식 (교육용) | PyTorch / TF1 | MIT | 12,019 | 2024-08-05 | `038665d` | `spinup/algos/pytorch/ddpg/ddpg.py` | ✅ |
| [sfujim/TD3](https://github.com/sfujim/TD3) | TD3 저자 공식, DDPG는 재구현 | PyTorch | MIT | 2,118 | 2023-07-14 | `34770cc` | `DDPG.py`(논문 재현), `OurDDPG.py`(재조정) | ✅ |
| [openai/baselines](https://github.com/openai/baselines) | 비공식 (유지보수 중단) | TF1 | MIT | 16,773 | 2024-08-01 | `ea25b9e` | `baselines/ddpg/` | ✅ |
| [rll/rllab](https://github.com/rll/rllab) | 비공식, 벤치마크 논문용 (유지보수 중단) | Theano | 라이선스 파일 비표준 | 3,079 | 2023-06-10 | `ba78e4c` | `rllab/algos/ddpg.py`, `exploration_strategies/ou_strategy.py` | OU·기본값만 |
| [vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl) | 비공식 (단일 파일) | PyTorch / JAX | 비표준 표기 | 10,611 | 2026-04-20 | `fe8d8a0` | `cleanrl/ddpg_continuous_action.py` | 목록만 |
| [google-deepmind/acme](https://github.com/google-deepmind/acme) | DeepMind 라이브러리 (이 논문 코드는 아님) | TF / JAX | Apache-2.0 | 4,082 | 2026-10-23 | `2e09e3d` | `acme/agents/tf/ddpg/` | 목록만 |
| [tensorflow/agents](https://github.com/tensorflow/agents) | Google 라이브러리 | TF2 | Apache-2.0 | 3,031 | 2026-01-16 | `eb24cf5` | `tf_agents/agents/ddpg/` | 목록만 |
| [thu-ml/tianshou](https://github.com/thu-ml/tianshou) | 비공식 | PyTorch | MIT | 11,042 | 2026-04-03 | `f240205` | `tianshou/algorithm/modelfree/ddpg.py` | 목록만 |
| [rlworkgroup/garage](https://github.com/rlworkgroup/garage) | 비공식, rllab 후속 (유지보수 중단) | TF / PyTorch | MIT | 2,130 | 2023-05-04 | `2d59480` | `src/garage/torch/algos/ddpg.py` | 목록만 |
| [pfnet/pfrl](https://github.com/pfnet/pfrl) | 비공식 | PyTorch | MIT | 1,277 | 2026-03-02 | `3578e33` | `pfrl/agents/ddpg.py` | 목록만 |
| [keras-rl/keras-rl](https://github.com/keras-rl/keras-rl) | 비공식 (유지보수 중단) | Keras | MIT | 5,548 | 2023-09-17 | `216c314` | `rl/agents/ddpg.py` | 목록만 |

**고르는 기준:** 지금 실험에 쓸 기준선이라면 **SB3**입니다. 유지보수가 활발하고, DDPG가 TD3의 특수 경우로 구현되어 비교 실험이 쉽습니다. 논문 설정에 가장 가까운 코드를 보려면 **sfujim/TD3의 `DDPG.py`**입니다. 다만 저장소의 `main.py`로 실행하면 τ와 배치가 TD3 기본값으로 덮어써집니다(`04` D3). 알고리즘을 공부하려면 **Spinning Up** 문서와 코드가 가장 읽기 쉽습니다. OpenAI baselines와 rllab은 역사적 참고용입니다.

## 5. 논문이 다루지 않았지만 관련 있는 연구 (내 판단)

| 주제 | 관련성 | 근거 수준 |
|---|---|---|
| **결정적 정책 경사의 off-policy 편향** (Degris et al. 2012의 off-policy actor-critic 분석 등) | Eq. 6은 행동 분포 ρ^β에서 기댓값을 취하므로 엄밀한 경사가 아닙니다. 버퍼가 클수록 이 편향이 커질 수 있습니다(`03` §1). | 내 해석, 서지 확인 필요 |
| **Pink noise 탐색** (Eberhard et al., ICLR 2023) | OU(빨간 잡음)와 백색 잡음 사이의 상관 구조를 고르는 문제를 정면으로 다룹니다. OU dt 모호성(`04` D5)과 바로 연결됩니다. | 내 사전 지식, 서지 확인 필요 |
| **LQR을 이용한 actor-critic 단위 시험** | 선형-이차 문제는 최적 제어기를 알 수 있어서 구현 검증에 쓸 수 있습니다. | 내 해석 |
| **HJB 잔차를 쓰는 연속 시간 가치 학습** | critic이 만족해야 하는 Bellman 식의 연속 시간 극한(HJB 편미분방정식)을 PINN식 손실로 넣는 계열입니다. PINN 연구와 이어집니다. | 내 사전 지식, 서지 확인 필요 |

**한계:** 1절과 5절의 후속 연구 서지는 TD3를 빼면 내 사전 지식에 근거합니다. 블로그나 git에 올리기 전에 원문을 직접 확인해야 합니다.
