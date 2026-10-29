---
title: "EigenDEXplore 읽기: 사람 손 협응을 탐색 잡음에만 넣으면, 그런데 잡음의 크기는 같았나"
date: 2026-10-29
tags: [robotics, dexterous-manipulation, reinforcement-learning, exploration, paper-review]
paper: https://arxiv.org/abs/2610.07681
code: https://github.com/hgupt3/eigendexplore
project: https://eigendexplore.github.io
summary: "사람 손 PCA 방향을 행동이 아니라 탐색 잡음에만 넣는 EigenDEXplore를, 공개 PCA 자산과 벤치마크 설정을 직접 계산하고 표본 최적화·정책 기울기 toy로 다시 읽었습니다."
---

> **논문:** Gupta, Lum, Wang, Pan, Liu, Bohg, Song. *EigenDEXplore: Structured Exploration for Dexterous Manipulation with Human Priors.* arXiv:2610.07681v2 (2026). Stanford.
> 논문 그림은 라이선스(arXiv 비독점 배포) 때문에 싣지 않았습니다. 그림과 영상은 [프로젝트 페이지](https://eigendexplore.github.io)에서 볼 수 있습니다. 이 글의 그래프는 공개 코드의 PCA 자산과 설정, 논문 수치로 직접 그렸습니다. 코드: [hgupt3/eigendexplore](https://github.com/hgupt3/eigendexplore) @ `5d57180`.

## 한 줄 요약

Dexterous RL은 보통 관절마다 독립인 Gaussian 잡음으로 탐색합니다. 이 논문은 사람 손 자세를 로봇에 리타겟해 PCA로 얻은 "협응 방향"을 **행동 공간이 아니라 탐색 잡음에만** 더합니다. 탐색 분포는 Σ = diag(σ_q²) + Eᵀdiag(σ_e²)E 하나이고, 정책 출력과 배포는 그대로입니다. SimToolReal(+34%), DeXtreme(+40%), DexMachina, SPIDER(비용 −16%), 실물(과제 진행 33% → 82%)에서 모두 이겼다고 보고합니다.

## 1. 수치는 맞는다

본문의 헤드라인 비율을 다시 계산하면 모두 맞습니다.
- SimToolReal 10.6 / 7.9 = +34.2%
- DeXtreme 12.3 / 8.8 = +39.8%
- SPIDER 여섯 손 평균 −16.05%
- 실물 SimToolReal: Fig. 8의 과제별 평균 네 개의 평균이 33% → 82%
- 실물 DeXtreme 8.45 → 11.90 = +40.8%

공개된 6개 손의 PCA 자산(`pca.safetensors`)을 직접 읽어 보면, 분산 90%에 필요한 k도 논문과 같습니다(Sharpa 9/22, XHand 6/12, Allegro 10/16, Inspire 3/6).

![손별 누적 분산](images/pca_spectra.png)

## 2. 그런데 공개 SimToolReal 설정은 탐색이 두 배다

논문 Sec. 3.2는 "초기 잡음 분산의 합을 Joint-Space와 같게 맞추고, 두 성분이 반씩 나눈다. σ_e는 √λ에 비례한다"고 씁니다. 공개 벤치마크 설정을 공분산으로 바꿔 계산해 보면 다음과 같습니다.

![탐색 예산](images/noise_budget.png)

- **DeXtreme 설정:** IID σ √0.5 + eigen RMS √0.5, 지수 1이라 논문 서술과 정확히 같습니다.
- **SimToolReal 설정:** IID σ를 Joint-Space와 같은 1로 두고 그 위에 eigen 잡음(RMS 1)을 더합니다. 그래서 관절당 초기 분산이 **2.0**입니다(Joint-Space 1.0). 스펙트럼도 λ^¼로 평평합니다.
- **SPIDER 설정:** σ_e ∝ λ이고(코드 주석이 "eigenvalue power 1"이라고 밝힘), 기존 IID는 그대로 둔 채 eigen 잡음을 더합니다.

논문 실험이 이 설정으로 돌았다면, SimToolReal의 +34%와 실물 33 → 82%는 "구조"와 "두 배 탐색"이 섞인 결과입니다. SPIDER에는 같은 스펙트럼을 무작위 방향에 실은 대조군이 있어서 크기만으로 설명되지 않는다는 근거가 있지만, RL 벤치마크에는 그런 대조군이 없습니다.

## 3. 왜 이기는가: 표본 최적화와 정책 기울기는 다르다

**표본 최적화(SPIDER, CEM, MPPI).** 평활화된 비용의 기울기에 대해 E[f(μ+ε)ε] = Σ∇J가 성립합니다. 즉 Σ가 기울기를 전처리합니다. Sharpa에서 사람 자세 방향의 보폭은 2.2배가 되고, 등방 방향은 평균적으로 손해가 없습니다. 예산이 적은 SPIDER식 toy로 확인했습니다.

![toy 비용 비](images/toy_cost.png)

- 사람 분포를 따르는 목표에서는 이기고(0.88, 0.67), 등방 목표에서는 집니다(1.29, 1.38).
- 같은 스펙트럼의 무작위 방향은 사람 목표에서도 IID보다 나쁩니다. 논문 App. B.2와 같은 결론입니다.
- **총량 2배는 영역에 따라 부호가 바뀝니다.** 기준 잡음이 작으면 구조보다 크기 효과가 더 크고(0.36), 적당하면 오히려 손해입니다(1.32). 총량이 다른 두 설정을 비교하면 구조의 효과를 따로 읽을 수 없습니다.

**정책 기울기(PPO).** REINFORCE의 기대값은 ∇E[R]이라서, 조밀한 이차 보상에서는 기대 기울기가 Σ와 **무관**합니다. toy에서는 배치 추정의 정확도가 오히려 떨어졌습니다(cos 0.42 → 0.30). 그러면 PPO에서 이득은 어디서 올까요. 희소한 성공을 **발견**하는 데서 옵니다. 목표 방향으로 3σ 이상 나아갈 확률이 사람 방향에서 17배가 됩니다. 다만 같은 계산에서 분산 2배는 44배라서, 크기 차이가 특히 큰 교란이 됩니다.

## 4. 보조 장치가 있으면 이득이 사라진다

DexMachina Fig. 5의 막대를 판독했습니다(본문 값 4개와 오차 0.004 이하).

![DexMachina 판독](images/dexmachina_gain.png)

- 커리큘럼과 보조 보상이 있으면 평균 +6%이고, Schunk 상자 한 칸을 빼면 거의 차이가 없습니다.
- 과제 보상만 쓰면 평균 0.504 → 0.648, +28.6%입니다. 서론의 "+28%"는 이 설정의 값입니다.
- 위의 정책 기울기 분석과 같은 그림입니다. 이득은 보상 설계가 적을 때, 즉 발견이 병목일 때 납니다.

## 5. 그 밖에 읽을 때 걸러야 할 것

- **"EgoSuite 사전 분포"는 세 데이터셋의 혼합입니다.** 공개 기저의 manifest는 ARCTIC, GRAB, EgoSuite를 [0.6, 0.2, 0.2]로 섞었고, 원 시퀀스는 3,596개입니다. 코퍼스와 fitting 파이프라인이 없어서 데이터셋 비교(Fig. 7a)는 재현할 수 없습니다.
- **기준선이 원 설정보다 불리할 수 있습니다.** 원 SimToolReal은 Sharpa의 외전 관절 범위를 좁혔는데, 이 논문은 전체 범위를 씁니다(v2에 추가된 문장). 실물 기준선도 공개 체크포인트가 아니라, 원래 예산의 절반(60B)만 학습한 정책입니다.
- **가장 큰 격차는 표본이 가장 적은 곳에서 나옵니다.** 연장 학습(4.9 대 14.1)과 실물은 방법마다 정책 하나입니다. v2는 서론에서 "연장 실행에서 더 큰 이득"이라는 문장을 뺐습니다.
- **DexMachina 실험 코드는 공개되지 않았습니다.**

## 6. 직접 구현할 수 있나

| 경로 | 판단 |
|---|---|
| 방법 패키지 | 바로 사용 가능합니다. torch `LowRankMultivariateNormal` 기반이고, 손 6종 기저와 CPU 예제가 들어 있습니다(MIT). |
| 공식 벤치마크(DeXtreme, SimToolReal, SPIDER) | 고정 커밋과 패치로 환경을 재현할 수 있지만, 모두 Linux와 NVIDIA GPU가 필요합니다. SimToolReal 60B는 1주 규모이고, Fig. 3a를 재현하려면 설정부터 고쳐야 합니다. |
| 아이디어만 재구현 | 매우 쉽습니다(수십 줄). 진짜 비용은 내 손에 맞는 사전 분포(사람 데이터 + 리타겟)입니다. 텔레옵 로그의 관절 PCA가 싼 대안입니다. |

비교 실험을 한다면 세 가지를 반드시 맞춰야 합니다. 탐색 총량(trΣ/d), 스펙트럼 지수, 같은 스펙트럼의 무작위 방향 대조군입니다. 공개 설정끼리도 이 셋이 달랐습니다.

## 7. 정리

- 아이디어는 단순하고 강력합니다. 행동 표현을 건드리지 않고 탐색 공분산만 사람 협응으로 바꾸는 것이라, 기존 PPO·표본 MPC에 거의 공짜로 붙습니다.
- 수치는 정확합니다. 다만 공개 SimToolReal 설정은 탐색 총량이 두 배라서, 헤드라인 결과에서 "구조"의 몫은 따로 재야 합니다.
- 이득은 표본 최적화에서는 기울기 전처리, PPO에서는 희소한 성공의 발견에서 나옵니다. 보상 설계가 충분하거나 자유도가 적은 손에서는 기대하기 어렵습니다.

분석 문서, 재계산·toy 스크립트, 리포트 PDF는 함께 공개한 저장소에 있습니다.

## 자료

- [통합 리포트 PDF](report/report_public.pdf)
- [01. 논문 분석: EigenDEXplore (arXiv:2610.07681)](analysis/01_paper_analysis.md)
- [02. 선행연구와 계보](analysis/02_prior_work.md)
- [03. 핵심 식 해석과 CPU 검증](analysis/03_analysis.md)
- [04. 코드 검증: 논문 ↔ 구현 대조, 수치 재계산, 구현 가능성](analysis/04_code_verification.md)
