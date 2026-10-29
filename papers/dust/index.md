---
title: "Dust 읽기: 역전파 없이 Transformer를 학습한다, 그런데 무엇과 비교했나"
date: 2026-10-29
tags: [optimization, zeroth-order, evolution-strategies, transformer, pretraining, paper-review]
paper: https://qlabs.sh/research/dust
code: https://github.com/qlabs-eng/dust
summary: "활성값을 토큰마다 흔들어 forward만으로 gradient를 추정하는 Dust를, 부록 표 재계산, 연산량 환산, 공식 코드 대조, numpy toy로 읽었습니다."
---

> **논문:** Samip Dahal, Bishwas Mandal, Serdar Gülbahar, Akshay Vegesna. *Dust: Pretraining Transformers Without Backpropagation.* Q Labs Research, 2026년 10월. https://qlabs.sh/research/dust (arXiv 판 없음)
> 논문 페이지에 라이선스 표기가 없어서 논문 그림은 싣지 않았습니다. 이 글의 그래프는 모두 부록 표와 공식 코드(MIT, 커밋 `b20f7c0`)의 설정으로 직접 계산해 그렸습니다.

## 한 줄 요약

Dust는 선형층 출력(활성값)에 **토큰마다 독립적인 가우스 잡음**을 더하고, **그 토큰의 손실 변화**로 잡음에 보상을 줍니다. 보상으로 가중한 잡음의 평균이 출력 오차의 추정치가 되고, 이것과 입력의 외적이 가중치 gradient가 됩니다. 토큰 하나가 population 구성원 하나가 되므로, forward 한 번에 수천 개 구성원을 평가합니다.

## 1. 숫자는 맞다

부록 Table 1(test loss), Fig. 4 표(모델 크기), Table 8(cosine 법칙 적합)에서 본문 수치 12개를 다시 계산하면 모두 맞습니다. Table 1의 "∞" 행(power law 극한)도 다섯 칸으로 다시 적합하면 ±0.003 안에서 같은 값이 나옵니다. 공식 코드의 draw 배분은 부록 Table 3·4와 8개 항목이 모두 일치합니다. 예를 들어 K = 16,384일 때 writer 쌍 7,040, MLP 7,040, attention 출력 1,664, 임베딩 640입니다.

## 2. "backprop을 넘는다"는 업데이트 6번과 61번 구간이다

![Table 1](images/ladders.png)
*부록 Table 1을 다시 그림(제목의 step 수는 배치 16,384 토큰 기준 우리 계산). 오른쪽은 20M power law 극한의 한 칸 빼기 재적합.*

배치가 16,384 토큰이므로 예산별 update 횟수는 다음과 같습니다.

| 예산 | update 수 | 토큰/파라미터 |
|---|---|---|
| 100k | 6 | 0.003 |
| 1M | 61 | 0.027 |
| 10M | 610 | 0.27 |
| 20M | 1,221 | 0.53 |

측정값으로 Dust가 backprop보다 낮은 칸은 100k와 1M뿐이고, 차이는 0.04–0.07입니다. 이 정도의 극초기 학습에서는 gradient 방향의 질보다 학습률, 모멘텀, 업데이트 잡음이 손실을 더 크게 좌우합니다. 시드별 값이 보고되지 않아서, 차이가 시드 변동보다 큰지도 알 수 없습니다.

20M에서 power law 극한 4.431이 backprop 4.633보다 낮다는 결과는 적합에 크게 의존합니다. 다섯 칸 중 한 칸을 빼고 다시 적합하면 극한이 2.57에서 4.68까지 흔들립니다. 64 draw 칸을 빼면 극한이 backprop보다 높아집니다. 또 사다리 맨 위(16k) 칸만 학습으로 따로 튜닝되었습니다(부록 F). 저자들도 이 극한을 "측정된 극한이 아니라 간격이 줄어든다는 증거"로 읽으라고 적었습니다.

## 3. 같은 토큰, 다른 연산

![Compute](images/compute.png)
*update 한 번의 연산량을 forward 횟수로 환산했다. 부록의 draw 배분과 "블록 l부터 다시 계산" 규칙으로 계산.*

- **update 한 번의 비용:** K = 16,384에서 forward 약 13,600회입니다. forward 1회와 backward(약 2회)로 이루어진 backprop step의 약 4,500배입니다.
- **같은 연산이면:** 1M 토큰에서 이 설정과 같은 연산량이면, backprop은 약 45억 토큰을 볼 수 있습니다.
- **"10³–10⁴배 효율"의 뜻:** 가중치 공간 ES(EGGROLL) 대비 population 효율입니다. backprop 대비가 아닙니다. 저자들도 연산 효율은 이 논문의 목표가 아니라고 밝혔습니다.

## 4. cosine 법칙은 어디서 오나

![Cosine](images/cosine_16k.png)
*부록 Table 8의 법칙. 점선은 학습에 실제로 쓴 population 16k.*

논문은 Dust 추정치와 backprop gradient의 cosine이 cos(K) = c_max/√(1 + c/K)를 따른다고 보고합니다. 이 법칙은 유도할 수 있습니다. 잡음이 작은 선형 근사에서는 다음과 같습니다.

  c = (d_out + 1) · Σ_t ‖x_t‖²‖g_t‖² / ‖Σ_t g_t x_tᵀ‖²

즉 c는 "출력 차원 × 토큰별 gradient가 서로 얼마나 상쇄하는가"입니다. numpy toy에서 적합한 c는 이론값과 맞았습니다(308 vs 314).

![Toy](images/toy.png)
*numpy toy. (a) 이론식과 측정 cosine. (b) 토큰 사이 영향 β가 있을 때, 자기 토큰만 보상하면(γ = 0) c_max가 국소 gradient와 전체 gradient의 cosine으로 떨어진다.*

c_max < 1의 원인도 toy로 확인했습니다. 각 토큰의 잡음을 **자기 토큰 손실로만** 보상하면(γ = 0), 추정기는 "자기 토큰에 대한 국소 gradient"를 추정합니다. 그래서 c_max는 국소 gradient와 전체 gradient의 cosine과 같아집니다(0.933 vs 0.936). 미래 토큰까지 보상하면 c_max는 1이 되지만 분산이 약 200배 커집니다. 부록 Table 8의 패턴도 이 해석과 맞습니다.

- c_max = 1: head와 query(영향이 자기 토큰뿐), key와 value embedding(미래 credit을 받음)
- c_max < 1: MLP와 출력 투영(γ = 0인데 뒤 토큰에 영향)

또 학습 population 16k에서의 cosine은 head를 빼면 0.16–0.57입니다. "backprop gradient에 가깝다"는 population이 10만을 넘는 측정 구간의 이야기입니다.

## 5. 정리

| 주장 | 판단 |
|---|---|
| 활성값 섭동이 가중치 공간 ES보다 population을 훨씬 싸게 만든다 | 지지됨 |
| population에 따른 cosine 법칙, 1B 토큰까지 유지 | 지지됨 (법칙은 유도 가능) |
| 큰 population에서 backprop보다 낮다 | 극초기(6·61 step), 같은 토큰 기준에서만 |
| 20M 극한이 backprop보다 낮다 | 적합에 크게 의존 |
| 근사가 backprop과 다른 것이 오히려 좋은 성질 | 주된 원인은 구조적 편향(γ = 0) |

## 6. 가져갈 것

1. **미분할 수 없는 모듈이 섞인 학습:** 시뮬레이터 step, IK, 외부 코드 실행처럼 미분할 수 없는 모듈이 있으면, 그 앞 층 출력만 흔드는 혼합 방식이 가능합니다. 필요한 draw 수는 그 층의 출력 차원에 비례합니다.
2. **0차 추정을 쓰기 전:** 작은 대리 모델에서 c를 계산해 필요한 population을 먼저 정합니다. 목표 cosine이 0.9이면 K ≈ 4.3c입니다.
3. **학습법을 비교할 때:** "같은 토큰"과 "같은 연산" 두 기준을 함께 보고합니다.
4. **power law 외삽을 보고할 때:** 한 칸 빼기 민감도를 같이 적습니다.

---

*재계산 스크립트, numpy toy, 논문 ↔ 코드 대조표, 공개용 리포트 PDF는 함께 공개한 분석 폴더에 있습니다.*

## 자료

- [통합 리포트 PDF](report/report_public.pdf)
- [01. 논문 분석: Dust, 역전파 없이 Transformer 사전학습하기](analysis/01_paper_analysis.md)
- [02. 선행연구: Dust는 어디에서 왔는가](analysis/02_prior_work.md)
- [03. 재계산과 해석](analysis/03_analysis.md)
- [04. 논문 ↔ 코드 대조와 주장 검증](analysis/04_code_verification.md)
