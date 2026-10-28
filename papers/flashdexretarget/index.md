---
title: "FlashDexRetarget 읽기: 시연 수백 개를 정책 하나로, 그런데 무엇이 성능을 만들었나"
date: 2026-10-28
tags: [robotics, dexterous-manipulation, retargeting, reinforcement-learning, paper-review]
paper: https://arxiv.org/abs/2610.01849
code: https://github.com/Holiday-Robot/FlashDexRetarget
project: https://holiday-robot.github.io/FlashDexRetarget/
summary: "사람 손-물체 시연 모음을 RL 정책 하나로 로봇 궤적으로 바꾸는 FlashDexRetarget을, 표를 재계산하고 공개 코드와 대조하며 읽었습니다."
---

> **논문:** Lee, Kim, Hwang, Oh et al. *FlashDexRetarget: Accelerating Dexterous Manipulation Data Generation through Multi-Motion Retargeting.* arXiv:2610.01849 (2026). KAIST AI, Holiday Robotics.
> 논문 그림은 CC BY 4.0에 따라 출처를 밝히고 실었습니다(© 저자). 나머지 그래프는 논문 수치와 공개 코드의 식으로 직접 그렸습니다. 코드: [Holiday-Robot/FlashDexRetarget](https://github.com/Holiday-Robot/FlashDexRetarget) @ `6f95e13`.

![논문 Fig. 1](images/paper_fig1_teaser.png)

## 한 줄 요약

사람 손-물체 시연을 로봇손 궤적으로 옮기는 물리 기반 리타겟팅은, 지금까지 **시연 하나에 최적화 하나**였습니다. 그래서 비용이 시연 수에 비례합니다. 이 논문은 레퍼런스 조건부 RL 정책 **하나**가 시연 모음 전체를 추적하게 하고, 평가를 통과한 rollout을 로봇 데이터로 저장합니다. XHand 50개 모션에서 90%를 29 GPU-h에 변환했고, CHORD는 46%에 2,847 GPU-h가 들었습니다.

## 1. 무엇을 바꿨나

휴머노이드에서는 이미 정책 하나로 대규모 모션을 추적합니다. 손-물체로 옮기면 문제가 네 가지 생기고, 논문은 문제마다 장치를 하나씩 붙였습니다.

| 문제 | 장치 |
|---|---|
| 물체 모양이 시연마다 다름 | 물체 점군 128점(현재·레퍼런스), 손끝·손목 SDF 거리와 법선을 관측에 추가 |
| 다음 프레임만으로는 동작 방향을 모름 | 미래 10프레임을 Conv1D로 128차원 latent로 압축 |
| 복원된 접촉이 부정확 | 이진 접촉 대신 "사람 손끝보다 멀지만 않으면 만점"인 한쪽 방향 거리 보상 |
| 상태 분포가 넓고 두 손 역할이 다름 | FlashSAC(off-policy)의 버퍼 10M → 50M, critic 256 → 1024, 손별 actor-critic |

![논문 Fig. 2](images/paper_fig2_overview.png)

## 2. 표를 다시 계산해 봤다

Table I은 내부적으로 일관됩니다. 모든 성공률 × 50이 정수이고, 엄격한 기준이 느슨한 기준보다 항상 낮습니다. 50개 기준 신뢰구간을 계산해도 차이는 확실합니다(SR_SPIDER 45/50 vs 24/50, Fisher p ≈ 9×10⁻⁶). 다만 읽을 때 걸러야 할 점이 세 가지 있습니다.

![Table I 다시 그리기](images/headline_table1.png)

- **헤드라인은 XHand 값입니다.** CHORD 대비 계산량은 XHand 98배, Sharpa 75배입니다. 성공률 차이도 XHand +44%p, Sharpa +36%p입니다.
- **진짜 격차는 엄격한 기준에 있습니다.** 손마다 물체를 3 cm / 30° 안에 유지해야 하는 SR_MT-Obj에서 기준선은 0.02–0.32, FlashDexRetarget은 0.86입니다. SR_SPIDER(10 cm / 0.5 rad)로만 보면 격차가 작아 보입니다.
- **SR_SPIDER는 생각보다 관대합니다.** 코드를 보면 위치 오차에서 궤적 평균 오프셋을 빼고 계산합니다. SPIDER 공식 스크립트와 같은 정의라 비교는 공정하지만, 물체가 처음부터 끝까지 일정하게 8 cm 어긋나도 위치 오차는 0입니다.

성공 1건당 비용으로 바꿔 보면 차이가 더 분명합니다. SR_MT-Obj 기준으로 FlashDexRetarget은 0.67 GPU-h, DexMachina는 27.9, Do as I Do는 33.0이고, CHORD는 2,847 GPU-h에 1건입니다.

## 3. 점군 보상: 주장은 맞지만 사각지대가 있다

물체 보상은 물체 크기를 반지름 ρ = 0.058 m로 정규화한 뒤, 대응점 오차가 가장 큰 세 점의 평균으로 측정합니다. 위치와 회전을 가중치 없이 한 단위로 묶는 깔끔한 설계입니다. "30° 회전이 약 3 cm"라는 주장은 2ρ·sin 15° = 3.00 cm로 정확합니다.

그런데 이 값은 **가장 먼 점이 회전축에 수직일 때의 상한**입니다. 공식 코드의 식을 그대로 옮겨 여러 형상에 적용해 보니, 길쭉한 물체를 장축으로 돌릴 때가 사각지대였습니다.

![회전축에 따른 점군 보상 오차](images/reward_axis.png)

20×3×3 cm 막대를 장축으로 30° 돌려도 오차는 0.62 cm(보상 0.94)이고, 180°를 돌려도 3 cm에 이르지 않습니다. 반면 평가 기준은 쿼터니언 각도라 같은 회전을 실패로 셉니다. 대칭인 병이라면 오히려 맞는 성질이지만, 숟가락이나 주걱처럼 장축 회전이 중요한 도구에서는 보상과 평가가 어긋납니다. FlashDexRetarget의 평균 회전 오차(MORE)가 기준선보다 나쁜 이유 중 하나일 수 있습니다. 다만 모션별 결과가 공개되지 않아서 가설로 남겨 둡니다.

## 4. 무엇이 성능을 만들었나: ablation과 코드

![논문 Fig. 4](images/paper_fig4_ablation.png)

논문 ablation에서 가장 큰 요인은 학습 알고리즘과 용량입니다. 같은 300M 스텝에서 PPO는 거의 0이고, 기본 FlashSAC는 50개 중 약 17개, 버퍼와 critic을 키우면 43개입니다. 두 기하 보상을 함께 빼면 약 15개로 떨어집니다.

공개 코드를 논문과 대조하면, 논문에 없지만 기본으로 켜져 있는 장치가 세 가지 있습니다.

1. **물체 보조 힘 커리큘럼:** 물체를 레퍼런스 자세로 당기는 PD 힘을 주고, 보상과 완주율 조건을 만족하면 0.9배씩 줄입니다(DexMachina 방식). 평가는 보조 없이 하므로 저장된 궤적은 물리적으로 유효합니다. 하지만 학습에 대한 기여는 측정되지 않았습니다.
2. **잔차 행동 공간:** 손목은 운동학 레퍼런스 + 작은 잔차로 움직입니다. FlashDexRetarget이 손 기준(SR_MT)과 물체 기준(SR_MT-Obj)에서 같은 성공 수를 낸 이유로 보입니다.
3. **접촉 wrench 관측:** 링크별 접촉 힘·토크·접촉점이 정책 입력에 들어갑니다. 논문 식에는 없는 privileged 정보이고, 실제 배포를 위한 증류에서 대체해야 할 입력입니다.

그 밖에 이미 성공한 클립도 균등하게 다시 뽑아서, 우선순위 샘플링의 여지가 있습니다. 1,000개 모음으로 키우면 변환 개수는 늘지만 비율은 82%에서 61%로 줄어드는데(Fig. 6), 이 결과와 관련 있어 보입니다.

![논문 Fig. 6](images/paper_fig6_scaling.png)

## 5. 직접 구현할 수 있나

**재현 정보는 충분히 공개되어 있습니다.** 설정 파일, 50개 벤치마크의 클립 목록과 프레임 수, 평가 코드까지 있습니다. 다만 장비 요구가 큽니다.

| 경로 | 판단 | 요구 사항 |
|---|---|---|
| 공식 코드 그대로 | 가능 (GPU 서버) | Linux, Isaac Sim 5.1, GPU 약 110 GB 또는 GPU 16 GB + 호스트 RAM 약 100 GB, XHand 50개 약 30 GPU-h, TACO·OakInk2·HOT3D와 MANO 직접 다운로드 |
| CPU만으로 | 일부만 가능 | 보상 식 검증은 완료했습니다. 시연 import → 운동학 리타겟 → 학습용 파일 생성은 MuJoCo와 CPU SDF 옵션으로 돌릴 수 있어 보입니다. RL 학습은 불가능합니다 |
| 아이디어만 재구현 | 쉬움, 성능은 어려움 | 보상 두 식과 미래 인코더는 짧습니다. 그러나 성능을 내려면 보조 커리큘럼, 잔차 행동, 접촉 관측과 대용량 off-policy 인프라가 함께 필요합니다 |

논문에 적히지 않은 요구도 있습니다. 50M 행 replay buffer는 88 GB라서, 논문이 쓴 RTX 3090(24 GB)에서는 버퍼를 호스트 RAM에 둘 수밖에 없습니다. "29 GPU-h"는 계산량으로는 작지만, 128 GB급 RAM을 가진 한 대에서 하루 이상 연속으로 돌려야 하는 작업입니다.

## 6. 정리

- 아이디어는 명확하고 효과도 큽니다. 시연마다 최적화하던 리타겟팅을 정책 하나로 바꾸면, 엄격한 기준에서 성공률이 크게 오르고 비용은 수십~수천 배 줄어듭니다.
- 헤드라인 수치는 XHand 기준이고, SR_SPIDER는 관대한 지표라는 점을 알고 읽어야 합니다.
- 공개 코드에는 논문에 없는 장치가 있습니다. 재구현한다면 이 장치들을 먼저 가져와야 합니다.
- 결과물은 일반화 정책이 아니라 **데이터**입니다. 학습에 쓴 모션 자체를 변환한 것이고, 처음 보는 모션에서의 일반화는 다루지 않았습니다.

분석 문서, 재계산 스크립트, 리포트 PDF는 함께 공개한 저장소에 있습니다.

## 자료

- [통합 리포트 PDF](report/report_public.pdf)
- [01. 논문 분석: FlashDexRetarget (arXiv:2610.01849)](analysis/01_paper_analysis.md)
- [02. 선행연구와 계보](analysis/02_prior_work.md)
- [03. 핵심 식 해석과 toy 검증](analysis/03_analysis.md)
- [04. 코드 검증: 논문 ↔ 구현 대조, 수치 재계산, 구현 가능성](analysis/04_code_verification.md)
