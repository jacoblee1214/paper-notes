---
title: "Do as I Do 읽기: SAM 3D 추적은 사실 EMA 필터다"
date: 2026-10-28
tags: [robotics, dexterous-manipulation, retargeting, flow-matching, paper-review]
paper: https://arxiv.org/abs/2606.19333
code: https://github.com/malik-group/do-as-i-do
summary: "일상 영상에서 다지 로봇손 데이터를 만드는 Do as I Do를 코드까지 대조해 읽고, 핵심 추적 기법의 정체를 유도했습니다."
---

> **논문:** Paliwal, Etukuru, Liang, Abbeel, Shafiullah, Malik. *Do as I Do: Dexterous Manipulation Data from Everyday Human Videos.* arXiv:2606.19333 (2026), CC BY 4.0.
> 이 글의 논문 그림은 위 논문에서 가져왔습니다. 코드 대조 기준은 `malik-group/do-as-i-do@824591b`입니다.

![Fig. 1](images/paper_fig1_teaser.png)
*논문 Fig. 1: 일상 영상(위)에서 실제 로봇 실행(아래)까지.*

## 한 줄 요약

단안 RGB로 찍은 일상 영상에서 손과 강체 물체의 4D 궤적을 복원하고, 물리 시뮬레이터 안의 sampling MPC로 22-DoF 로봇손 궤적으로 바꾸는 파이프라인입니다. 인터넷 영상, egocentric 영상, 생성 영상을 가리지 않고 쓸 수 있다는 점이 핵심입니다.

![Fig. 2](images/paper_fig2_overview.png)
*논문 Fig. 2: 분할, 깊이, 점 추적 → 복원 → 리타겟팅 → 배포.*

## 1. 복원: SAM 3D를 학습 없이 비디오 추적기로

SAM 3D는 이미지 한 장에서 물체의 형상(shape)과 자세(pose)를 같은 latent 공간에서 함께 생성합니다. 프레임마다 따로 돌리면 메쉬와 자세가 매번 달라지기 때문에, 저자들은 flow matching의 Euler 적분 매 스텝에 다음 blending을 넣었습니다.

```
x_t = (1−α)·(x_{t−Δ} + Δ·v_θ) + α·[(1−t)·ε + t·x_ref]
```

shape 블록은 x_ref를 기준 프레임 형상으로 두고 α=1로 고정합니다. pose 블록은 x_ref를 이전 프레임 자세로 두고 α_p만큼 당깁니다. α_p는 점 추적으로 구한 회전 속도에 따라 `max(0.1, 0.7 − 0.09·|Δθ|)`로 정합니다(단위는 deg/frame).

### 이 식의 정체

어떤 velocity든 x̂₁ = x_t + (1−t)·v라는 "x1-예측" 형태로 다시 쓸 수 있습니다. 마지막 스텝(t: 1−Δ → 1)에서는 Δ = 1−t이므로 Euler 결과가 곧 x̂₁이고, 이때 z_ref(1) = x_ref입니다. 따라서 최종 출력은 다음과 같습니다.

```
x₁ = (1−α_p)·x̂₁ + α_p·x_{k−1}
```

즉 프레임 단위로 보면 **지수이동평균(EMA) 필터**입니다. 앞선 24번의 blending이 x̂₁도 이전 pose 쪽으로 끌어당기기 때문에, 실제 평활화 강도는 α_p보다 큽니다.

![w_eff](images/toy_w_eff.png)
*toy 모델: 네트워크가 현재 상태에 의존하는 정도(γ)가 클수록 실효 가중치가 α_p보다 커집니다.*

"빠르게 움직이면 평활화를 줄인다"는 적응형 규칙은 HCI 분야의 One-Euro 필터(Casiez et al., CHI 2012)와 같은 발상입니다. 1D toy로 비교해 보면, 고정 α는 빠른 회전 구간에서 크게 뒤처지고 적응형 α는 지터와 지연을 함께 줄입니다.

![toy tracking](images/toy_tracking.png)

| 방식 | RMSE (°) | 빠른 구간 최대 지연 (°) |
|---|---|---|
| 가이던스 없음 | 3.99 | 7.8 |
| 고정 α = 0.7 | 12.75 | 30.5 |
| 적응형 α | **2.93** | 7.5 |

이 관점에서 개선 방향도 바로 보입니다. 2D 회전 속도 대신 3D 속도를 쓰거나, EMA 대신 상수속도 예측(α-β 필터)을 x_ref로 쓰면 지연을 줄일 수 있을 것입니다. 아직 검증하지 않은 가설입니다.

## 2. 리타겟팅: 성공률 향상의 대부분은 Warmup

![Fig. 4](images/paper_fig4_retarget.png)
*논문 Fig. 4: Warmup, Random Force, Transition Reward가 해결하는 실패 유형.*

SPIDER의 annealed sampling MPC 위에 세 가지를 얹었습니다. 복원 레퍼런스 655개 기준 성공률은 다음과 같습니다.

| | 성공률 |
|---|---|
| Annealed Sampling (SPIDER) | 25% |
| + Warmup | **66%** |
| + Perturbation | 67% |
| + Transition Reward | 71% |

전체 +46%p 중 +41%p가 **Warmup** 하나에서 나옵니다. Warmup은 레퍼런스 앞에 3초를 덧붙이고, 물체를 공중에 고정한 채 손만 자세를 잡게 하는 장치입니다. 잡음 섞인 궤적으로 궤적 최적화를 하는 사람이라면 가장 먼저 가져올 만한 교훈입니다.

## 3. 수치를 읽을 때 주의할 점

- **추적기 자체의 이득:** 같은 파이프라인에 FoundationPose를 끼워 비교하면 이득이 작습니다. HOI4D에서는 동률입니다. 차이가 드러나는 곳은 in-the-wild 영상(사람 선호 67%)입니다.
- **느슨한 성공 기준:** 성공 기준은 "평균 위치 오차 < 10 cm"입니다. 레포에 포함된 whisking 결과를 직접 계산해 보면 위치 오차 3.8 cm, 회전 오차 0.28 rad로 기준을 통과합니다. 그런데 물체를 레퍼런스(약 14 cm)의 1/3 정도(약 4.5 cm)만 들어 올립니다.
- **본문에 없는 구현:** 렌더 기반 post-optimization이 기본으로 켜져 있습니다. Warmup은 손을 편 채 10 cm 떨어진 곳에서 시작합니다. 교란은 물체를 쥔 구간 안쪽에만 가해집니다. 이 장치들의 효과는 ablation에서 분리되지 않았습니다.
- **보고되지 않은 것:** 시드 반복, 신뢰구간, 실제 로봇 성공률이 없습니다.

## 4. 가져갈 것

1. **flow 기반 VLA action head:** 위 항등식은 action chunk를 flow matching으로 생성하는 정책에도 그대로 적용됩니다. 이전 chunk에서 남은 action을 x_ref로 두고, 상황(속도나 접촉)에 따라 α를 바꾸는 것은 학습 없이 시도할 수 있는 실험입니다.
2. **잡음 레퍼런스 궤적 최적화:** Warmup 방식의 초기화를 쓰고, 상위 10%만 남겨 순위를 정규화한 sampling 가중치를 씁니다. 후자는 보상 스케일에 둔감해서 보상 항을 자주 바꾸는 연구 단계에 편합니다.
3. **인터넷 영상 큐레이션 예산:** 100DOH 기준으로 쓸 수 있는 클립은 약 4%였습니다.

---

*분석 자료(리포트 PDF, 발표 자료, 유도와 toy 코드, 코드 대조표)는 함께 공개한 git 폴더에 있습니다.*

## 자료

- [통합 리포트 PDF](report/report_public.pdf)
- [01. 논문 분석: Do as I Do (arXiv:2606.19333)](analysis/01_paper_analysis.md)
- [02. 선행연구와 계보](analysis/02_prior_work.md)
- [03. 유도와 해석](analysis/03_derivations.md)
- [04. 코드 검증: 논문 ↔ 구현 대조](analysis/04_code_verification.md)
