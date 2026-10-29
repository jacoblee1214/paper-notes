---
title: "DexPolicy 읽기: 잡음을 줄이면 나아진다, 그런데 무엇과 비교했나"
date: 2026-10-28
tags: [robotics, dexterous-manipulation, reinforcement-learning, exploration, ppo, paper-review]
paper: https://arxiv.org/abs/2610.00360
code: https://github.com/AIGeeksGroup/DexPolicy
project: https://aigeeksgroup.github.io/DexPolicy/
summary: "PPO의 행동 잡음 σ를 학습 step에 따라 줄이는 DexPolicy를, 표를 재계산하고 공개 코드·기반 환경과 대조하며 toy로 다시 따져 읽었습니다."
---

> **논문:** Wang, Qian, Li, Zhang et al. *DexPolicy: Scheduled Exploration for Trajectory-Guided Dexterous Manipulation.* arXiv:2610.00360 (2026). Peking University, AI2 Robotics.
> 논문 라이선스(CC BY-NC-SA 4.0) 때문에 논문 그림은 싣지 않았습니다. 그림과 영상은 [프로젝트 페이지](https://aigeeksgroup.github.io/DexPolicy/)에 있습니다. 이 글의 그래프는 논문 수치, 공개 코드의 스케줄 정의, 직접 만든 toy로 그렸습니다. 코드: [AIGeeksGroup/DexPolicy](https://github.com/AIGeeksGroup/DexPolicy) @ `98bf668`.

## 한 줄 요약

사람 영상 궤적을 RL로 다듬는 ViViDex의 state-policy 단계에서, PPO가 학습하던 행동 잡음 σ를 **얼리고 학습 step에 따라 0.20 → 0.04로 줄이는 것**만으로 최종 결정론적 성공률이 오른다는 논문입니다. 손실, 구조, 보상, 최적화 설정은 그대로 둡니다. PPO, critic 없는 GRPO 연속 학습, flow 평균을 쓰는 PPO 변형("FPO")에 적용했고, 실제 로봇 실험도 했습니다.

## 1. 동기와 방법

ViViDex 기준선의 PPO는 log σ를 학습합니다. 그런데 머스터드 세 실행에서 σ가 5M step 뒤에도 0.202에서 0.185로만 줄었습니다. 논문은 이 관찰에서 출발해 σ를 직접 정합니다.

![σ 스케줄](images/noise_schedules.png)

평가는 결정론적 실행(σ_eval = 0)입니다. 학습 잡음이 다른 두 정책을 같은 실행 규칙으로 비교하려는 설계이고, 이 점은 좋습니다.

## 2. 표는 맞는다

Table I의 Overall, 8개 물체 평균, 본문에 공개된 seed별 값, Table II·III·V·VII을 다시 계산하면 모두 맞습니다. 숫자를 지어낸 흔적은 없고, 개발용 seed와 seed별 실패도 본문에 솔직하게 적혀 있습니다.

## 3. 그런데 이득은 작거나 몰려 있다

![seed 산포](images/sim_seed_spread.png)

| 방법 | Overall | 물체별 변화 (%p, 머스터드/머그/바나나/설탕/토마토) | seed 수준 Welch p |
|---|---|---|---|
| PPO | 32.0 → 35.7 | +10.7 / +15.4 / −6.4 / 0 / −1.3 | 0.39 |
| GRPO | 14.1 → 45.4 | +3.7 / +16.4 / +0.2 / +83.8 / +52.8 | 0.04 |
| FPO | 49.4 → 68.1 | +48.3 / +5.7 / +1.4 / +7.7 / +30.4 | 0.27 |

- **PPO는 사실상 구분되지 않습니다.** 물체 2개가 오르고 2개가 내렸습니다. 8개 물체 평균은 seed 3개 중 1개에서만 올랐습니다(본문).
- **GRPO와 FPO의 이득은 두 물체가 만듭니다.** FPO 머그는 seed 0이 65%에서 0%로 무너졌습니다.
- seed 3개의 표본 SD가 50%p를 넘는 칸이 있습니다. 논문도 유의성은 주장하지 않는다고 밝힙니다.

## 4. 기준선은 무엇이었나

가장 중요한 점은 기준선입니다. 기준선은 ViViDex 공개 설정의 기본 학습량(1억 step)의 1/20인 5M step으로 학습되었습니다.

이 예산에서 Adam(lr 1e-5)이 log σ를 움직일 수 있는 폭은 대략 다음과 같습니다.

```
(5,000,000 / 256) × 5 epoch = 97,656 step  →  최대 약 0.98
```

0.20에서 0.04로 가려면 log σ가 1.62 움직여야 합니다. **학습 σ는 이 예산에서 원리상 DexPolicy의 최종 잡음에 도달할 수 없습니다.** 실제로는 최대 폭의 9%만 움직였습니다. 따라서 PPO 비교는 "학습 σ 대 스케줄"이라기보다 **"거의 고정된 큰 σ ≈ 0.19 대 스케줄"**입니다. 저자도 "잘 조정한 고정 저잡음 기준선이 없다"를 한계로 적었습니다.

작은 toy로 확인했습니다. 4차원 밴딧에서 PPO-clip을 돌렸고, lr × step은 논문과 같은 0.9로 맞췄습니다. seed는 10개입니다.

![잡음 스케줄 toy](images/exploration_toy.png)

- **탐색이 필요 없는 과제에서는** 모든 조건이 100%입니다.
- **탐색이 필요한 과제에서는** σ를 줄이는 방식이 고정 σ보다 낫습니다.
- **Eq. 4 모양 자체는 특별하지 않았습니다.** 학습 σ와 비슷했고, 선형 감쇠가 가장 좋았습니다.
- **toy의 학습 σ는 실제로 줄었습니다**(0.20 → 0.07–0.11). 논문 기준선이 거의 안 줄었다는 것은, 그 과제에서 log σ의 기울기가 약했다는 뜻으로 읽힙니다.

## 5. 공개 코드와 다른 곳

공식 저장소는 "method-core release"입니다. 스케줄 정의(Eq. 4는 `no_mid_jump_5m` preset과 정확히 같음)와 SB3 callback, flow 평균 trainer는 논문과 맞습니다. 다른 점은 세 가지입니다.

1. **공개 GRPO advantage가 논문 식이 아닙니다.**
   - 논문 Eq. 5는 같은 에피소드 나이끼리 할인 reward-to-go(γ = 0.95)를 표준화합니다.
   - 공개 코드는 궤적의 할인 없는 총 보상을 rollout 전체에서 표준화해, 그 궤적의 모든 step에 같은 값을 줍니다.
   - Hugging Face 메타데이터는 `grpo_rtg` 모듈을 가리킵니다.
   - 가상 rollout에서 두 advantage의 상관은 0.73, 부호 불일치는 7.6%였습니다.
2. **GRPO 연속 학습 스케줄**(0.20 → 0.10 → 0.03)이 코드에 없습니다.
3. **성공 기준이 다릅니다.** 기반 환경인 ViViDex 공개 코드의 목표 성공 판정은 최종 위치 **5 cm**입니다. 논문 Target은 3 cm이고, DexPolicy의 평가 코드는 공개되지 않았습니다.

## 6. 실제 로봇: 크지만 모델 하나

![하드웨어](images/hardware_ci.png)

하드웨어 Target 평균은 PPO 8.3 → 43.3%, GRPO 10.0 → 63.3%, FPO 25.0 → 85.0%입니다. 시행 단위로 보면 차이는 확실합니다(Fisher p < 10⁻⁴). 하지만 **조건마다 학습한 모델이 하나**라서, 시뮬레이션에서 본 큰 seed 분산이 검정에 들어가지 않습니다. 하드웨어 정책은 다른 embodiment(RealMan RM75 + Inspire RH56)로 학습해서, 시뮬레이션과의 효과 크기 차이도 직접 비교할 수 없습니다.

## 7. 직접 구현할 수 있나

| 경로 | 판단 | 메모 |
|---|---|---|
| 공식 코드 그대로 | 방법 핵심만 | 환경·보상·평가·레퍼런스 선택·하드웨어 모델은 없고, 체크포인트는 머스터드 seed 0의 6개뿐입니다. Python 3.10, PyTorch 2.4.1, Gym 0.25.2, SB3 1.1.0 고정 |
| 공개 환경으로 다시 만들기 | 조건부 가능 | ViViDex MuJoCo 환경과 물체·궤적 자산(MIT)이 공개되어 있습니다. 스케줄 callback을 붙이면 PPO 비교를 다시 만들 수 있고, CUDA는 필요 없어 보입니다. 다만 성공 기준은 ViViDex 정의가 되고, 오래된 버전 조합의 설치를 확인해야 합니다 |
| 아이디어만 | 매우 쉬움 | log σ를 얼리고 rollout마다 덮어쓰기: 수십 줄. 결론을 내려면 고정 저잡음과 긴 예산의 대조군이 필요합니다 |

## 8. 정리

- **좋은 점:** 결정론적 평가로 "학습 잡음"과 "실행 잡음"을 분리했습니다. 학습 return, 결정론적 성공, 잡음 내성이 어긋난다는 점도 보였습니다. GRPO는 큰 실행 잡음(σ 0.10)에서 이득이 뒤집혔습니다.
- **조심할 점:** PPO 이득은 구분되지 않습니다. 기준선은 짧은 예산의 거의 고정된 σ이고, 공개 GRPO 코드는 논문 식과 다릅니다.
- **가져갈 것:** PPO를 쓸 때 학습 σ의 궤적과 최대 이동 폭(Adam step × lr)을 꼭 확인합니다. 정책 비교에는 결정론적 성공과 공통 잡음 성공을 함께 보고합니다.

분석 문서, 재계산·toy 스크립트, 리포트 PDF는 함께 공개한 저장소에 있습니다.

## 자료

- [통합 리포트 PDF](report/report_public.pdf)
- [01. 논문 분석: DexPolicy (arXiv:2610.00360)](analysis/01_paper_analysis.md)
- [02. 선행연구와 계보, 관련 GitHub](analysis/02_prior_work.md)
- [03. 핵심 식 해석과 toy 검증](analysis/03_analysis.md)
- [04. 코드 검증: 논문 ↔ 구현 대조, 수치 재계산, 구현 가능성](analysis/04_code_verification.md)
