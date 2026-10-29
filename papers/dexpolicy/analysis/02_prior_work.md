# 02. 선행연구와 계보, 관련 GitHub

번호 [n]은 논문의 참고문헌 번호입니다. 공식 코드, 기반 코드(ViViDex), Hugging Face 메타데이터는 직접 확인했습니다(§4). 그 밖의 논문은 따로 열어 검증하지 않았습니다.

## 1. 계보도

```
[사람 시연 → dexterous RL]
  DexMV [3] · DexVIP [4] · H-InDex [5] · DexTrack [6]
  TCDM / PGDM 계열 코드 (ViViDex README가 TCDM·DexMV 코드 기반이라고 밝힘)
        ▼
  ViViDex [1] (ICRA 2025): 영상 궤적 → state policy RL(PPO, SB3) → visual policy 증류
        ▼  state-policy 단계만 가져옴
  ★ DexPolicy: σ를 학습 step의 함수로 정함 (PPO / GRPO 연속 학습 / flow 평균 PPO)

[on-policy 최적화와 탐색]
  TRPO [7] → PPO [2] → "What matters" 대규모 연구 [10]
  탐색 구조: gSDE [11], colored-noise PPO [12], 행동 잡음 연구 [13], PPO-CMA [14]
  critic 없는 그룹 상대 advantage: GRPO [15] (언어 모델 RL에서 옴)

[flow 정책]
  flow matching [16] → FPO [17], FPO++ [18], ReinFlow [19]
```

**차별점 요약:** 탐색을 다룬 선행연구는 대개 잡음의 **구조**(상태 의존, 시간 상관, 공분산 적응)를 바꿉니다. DexPolicy는 구조는 그대로 두고 **크기 하나를 학습 step에 따라 정하는 것**만 바꿉니다. 최적화 하이퍼파라미터도 고정합니다. 그만큼 단순하고 다른 알고리즘에 쉽게 붙일 수 있습니다. 대신 효과가 과제와 예산에 민감합니다(01 §3.2).

## 2. 구성 요소별 출처

| 역할 | 출처 | 이 논문에서의 쓰임 |
|---|---|---|
| 과제·환경·기준선 PPO | ViViDex [1], `zerchen/vividex_mujoco` | state-policy 재배치. PPO 설정(lr 1e-5, n_steps 4096, batch 256, 5 epoch, log σ 초기 −1.60, 망 [256, 128])을 그대로 씀 |
| PPO 구현 | Stable-Baselines3 1.1.0 (README) | `PhaseScheduleCallback`이 rollout마다 `log_std`를 덮어씀 |
| 탐색 크기를 학습에 맡기지 않는 근거 | PPO-CMA [14] (분산이 너무 빨리 줄어드는 문제), Andrychowicz et al. [10] | 반대로 이 과제에서는 학습된 σ가 거의 줄지 않음을 동기로 듦 |
| critic 없는 advantage | GRPO [15] | 에피소드 나이별 reward-to-go 표준화(Eq. 5). 공개 코드는 다름(`04` D1) |
| flow 평균 | flow matching [16], FPO [17], FPO++ [18] | 8단계 결정론적 적분으로 평균만 만듦. ratio는 Gaussian |
| 하드웨어 물체 자세 | FoundationPose [22] | RGB-D 모델 기반 6D 자세 |
| 물체 | YCB [21] | 머스터드, 머그, 바나나, 설탕, 토마토, 피처, 나무 블록, 클램프 |

## 3. 이 저장소의 다른 논문과의 관계

| 논문 | 관계 |
|---|---|
| **FlashDexRetarget** (`flashdexretarget/`) | 같은 문제(사람 시연을 로봇 궤적으로)를 다른 쪽에서 풉니다. FlashDexRetarget은 레퍼런스 수백 개를 off-policy(FlashSAC) 정책 하나로 추적했고, 같은 예산에서 PPO는 거의 0이었습니다(Fig. 4(d)). DexPolicy는 물체마다 PPO 정책을 따로 두고, 학습된 σ가 줄지 않는 문제를 스케줄로 우회합니다. 두 결과를 합치면, 넓은 상태 분포나 정밀 추적에서 **PPO의 탐색 크기 조절이 약점**이라는 같은 관찰이 됩니다. SAC 계열은 목표 엔트로피로 온도를 자동 조절해서 이 문제를 구조적으로 다룹니다. |
| **Do as I Do** (`do_as_i_do/`) | 영상 → 손-물체 궤적 → 로봇이라는 흐름이 같습니다. Do as I Do는 sampling MPC에서 노이즈 커널을 반복·horizon 방향으로 annealing했습니다(SPIDER 계열). DexPolicy는 같은 "넓게 시작해 좁히기"를 RL 학습 step 축에서 합니다. |
| **DexScrew** (`dexscrew/`) | 둘 다 PPO로 손가락 기술을 배웁니다. DexScrew 분석에서 정리한 검정력·시행 수 계산(`dexscrew` A1)이 이 논문의 seed 3개, 하드웨어 모델 1개 설계에 그대로 적용됩니다. |
| **Contact-Grounded Policy** (`contact_grounded_policy/`) | 결정론적 평가와 작은 시행 수라는 통계 문제가 같습니다. CGP 분석의 "점추정 1위 ≠ 유의한 우위" 교훈이 DexPolicy의 PPO 결과(+3.7%p, p ≈ 0.39)에 그대로 해당합니다. |
| **DDPG** (`ddpg/`) | 탐색 잡음을 어떻게 정하느냐가 성능과 재현성을 가른다는 점이 같습니다. DDPG의 OU 잡음 dt 누락처럼, DexPolicy도 스케줄의 시간축(누적 환경 step인지, 연속 학습에서 어디서 다시 세는지)이 재현의 핵심 세부입니다. |
| **ZeroDex** (`zerodex/`) | 같은 Inspire 손을 씁니다(ZeroDex는 실물, DexPolicy는 RH56 모델과 실물). |

## 4. 관련 GitHub 저장소 (2026-10-28 `gh api` 확인)

| 저장소 | 무엇인가 | 공식 여부 | 라이선스 | ★ | 마지막 push | 확인 커밋 |
|---|---|---|---|---|---|---|
| AIGeeksGroup/DexPolicy | 방법 핵심 코드: σ 스케줄 preset과 SB3 callback, 궤적 그룹 GRPO, flow 평균 trainer, 테스트 | 공식 | MIT | 2 | 2026-10-04 | `98bf668` |
| huggingface.co/AIGeeksGroup/DexPolicy | 머스터드 seed 0 체크포인트 6개(PPO/GRPO/FPO × 기준선/DexPolicy)와 메타데이터 JSON | 공식 | 표기 없음 | – | 2026-09-19 | `575645a` |
| zerchen/vividex_mujoco | ViViDex MuJoCo 구현(환경, 보상, PPO 설정, 학습 스크립트) | 기반 코드 | MIT | 21 | 2025-04-09 | `9790140` |
| zerchen/object_sim | ViViDex 물체·궤적 자산(서브모듈, DexYCB·HOI4D 궤적 npz) | 기반 자산 | MIT | 2 | 2025-04-03 | `96a275c` |
| zerchen/vividex_sapien | ViViDex SAPIEN 구현 | 관련 | MIT | 24 | 2025-04-09 | `b22a07e` |
| DLR-RM/stable-baselines3 | PPO 구현 (DexPolicy는 1.1.0 고정) | 의존성 | MIT | 13,941 | 2026-10-26 | `0cdcd20` |
| akanazawa/fpo | 원래 FPO [17] 구현 (이 논문의 "FPO"와 다른 알고리즘) | 관련 | 표기 없음 | 477 | 2026-01-13 | `418c255` |
| ReinFlow/ReinFlow | flow 정책 온라인 RL 미세조정 [19] | 관련 | MIT | 376 | 2026-04-24 | `e722e15` |

- 공식 저장소는 README에서 "method-core release"라고 밝힙니다. 시뮬레이션 환경, 물체 메쉬, 궤적, 학습 가중치(머스터드 6개 제외), 로봇 드라이버, 실제 로봇 데이터는 없습니다.
- ViViDex 환경과 자산은 공개되어 있어서, 스케줄 callback을 ViViDex 학습 스크립트에 붙이면 PPO 비교는 다시 만들 수 있습니다. 다만 논문의 Target(3 cm)과 Lift 정의, 레퍼런스 시퀀스 선택, 하드웨어 embodiment는 공개되지 않았습니다(`04` D6, D7).
- FPO++ [18]와 colored-noise PPO [12]의 공식 저장소는 검색으로 찾지 못했습니다.

## 5. 논문이 다루지 않았지만 관련 있는 연구 (내 판단)

| 주제 | 관련성 | 근거 수준 |
|---|---|---|
| **SAC의 자동 엔트로피 조절** (Haarnoja et al., 2018) | 목표 엔트로피에 맞춰 온도를 학습하면 탐색 크기가 학습 신호에 따라 자동으로 줄거나 늘어납니다. "학습된 σ가 안 줄어든다"는 PPO의 문제를 다른 방식으로 푸는 직접적인 대안입니다. | 사전 지식. 서지 확인 필요 |
| **엔트로피 계수 스케줄링** | PPO에서 σ를 직접 고정하는 대신 엔트로피 보너스를 줄이는 방법입니다. 같은 효과를 학습 가능성을 유지한 채 얻을 수 있는지가 자연스러운 비교군입니다. | 내 해석 |
| **ε-greedy·잡음 감쇠 같은 고전적 탐색 스케줄** | DQN의 ε 감쇠, DDPG·TD3의 잡음 크기 설정과 같은 계열입니다. 이 논문의 새로움은 "on-policy PPO에서 log σ를 얼리고 스케줄"하는 점과, 결정론적 평가로 효과를 분리한 점입니다. | 사전 지식 |
| **PGDM / TCDM** (Dasari et al.) | ViViDex 설정의 `pregrasp: template`, `hand_imitation` 패키지 이름이 이 계열에서 온 것으로 보입니다. | 코드 이름으로 추정. 서지 확인 필요 |

**한계:** 5절의 서지 정보는 내 사전 지식에 근거합니다. 블로그나 git에 올리기 전에 원문을 직접 확인해야 합니다.
