# 01. 논문 분석: Dust, 역전파 없이 Transformer 사전학습하기

- **논문:** Samip Dahal, Bishwas Mandal, Serdar Gülbahar, Akshay Vegesna. *Dust: Pretraining Transformers Without Backpropagation.* Q Labs Research, 2026년 10월.
  - 본문: https://qlabs.sh/research/dust
  - 부록: https://qlabs.sh/research/dust/appendix.html
- **원문 형태:** arXiv 판이 없는 웹 논문입니다(2026-10-29 arXiv 검색 기준).
  - 본문과 부록 HTML, 그리고 Chrome으로 인쇄한 PDF를 `paper/`에 저장했습니다.
  - 절·식·표 번호는 웹 페이지의 번호를 그대로 씁니다.
- **라이선스:** 논문 페이지에 라이선스 표기가 없습니다. 그래서 논문 그림은 개인용 리포트·PPT에만 넣고, 게시본에서는 뺍니다.
- **코드:** https://github.com/qlabs-eng/dust
  - MIT 라이선스이고, 분석 기준 커밋은 `b20f7c0`(2026-10-06)입니다.
  - README에 따르면 "추정기와 튜닝된 기본값은 유지하되 실행 최적화는 뺀 최소 구현"입니다. 1M·10M 예산, population 256–16k만 들어 있습니다.

## 1. Abstract

Transformer 언어모델을 **역전파 없이 forward만으로** 사전학습하는 0차(zeroth-order) 방법 Dust를 제안합니다. 두 가지가 핵심입니다.

- **가중치가 아니라 활성값을 흔듭니다:** 각 선형층의 출력에 토큰마다 독립적인 가우스 잡음을 더합니다. 이것을 node perturbation이라고 합니다.
- **보상을 토큰 단위로 줍니다:** 각 토큰의 잡음은 그 토큰의 손실 변화로 보상받습니다. 따라서 forward 한 번에 토큰 수만큼의 "가상 population"을 평가합니다.

저자들의 주장(TL;DR)은 다음과 같습니다.

1. Transformer 사전학습에서 backprop과 경쟁할 수 있는 최초의 0차 방법이고, population이 크면 여러 설정에서 backprop보다 낫습니다.
2. 가중치 공간 ES(EGGROLL)보다 10³–10⁴배 효율적입니다(외삽 기준).
3. 모델이 클수록 population 효율이 높습니다. 243M 모델이 120배 작은 모델보다 대부분의 population에서 낫습니다.
4. population이 커질수록 Dust의 gradient가 backprop의 gradient와 더 잘 정렬되고, 1B 토큰까지 정렬이 유지됩니다.

## 2. Method

### 2.1 활성값 공간의 섭동 (§2.1)
선형층 y_t = W x_t의 출력에 y_t → y_t + σ a_t (a_t ~ N(0, I))를 모든 토큰에 독립적으로 더합니다.

- 토큰 t에서 이 섭동은 그 층 가중치에 저랭크 섭동을 준 것과 같습니다. 따라서 가중치를 실제로 복사하지 않아도 됩니다.
- 시퀀스 길이 2,048이면 forward 한 번에 2,048개 "구성원"을 평가하는 셈입니다.
- 잔차 혼합 스칼라 2L개만 가중치 공간 ES로 학습합니다.

### 2.2 보상과 추정 (§2.2, Eq. 1–3)
- **보상:** c_s = ℓ̃_s − ℓ_s입니다. ℓ̃_s는 같은 forward 묶음 안 draw들의 평균 손실입니다.
- **미래 토큰 credit:** r_t = Σ_{s≥t} γ^{s−t} c_s (Eq. 1). γ = 0이면 자기 토큰만 봅니다.
- **층 출력 오차 추정:** ĝ_t = −(1/Kσ) Σ_i r_t^(i) a_t^(i) (Eq. 2). K는 population, 즉 draw 수입니다.
- **가중치 gradient:** Ĝ_W = Σ_t ĝ_t x_tᵀ (Eq. 3). backprop과 같은 외적이고, 출력 오차를 chain rule이 아니라 population에서 얻는다는 점만 다릅니다.

### 2.3 간섭 줄이기와 튜닝 (§2.3, Eq. 4, 부록 E·F)
- **층 종류별 분리:** 층 종류마다 다른 forward pass에서 흔듭니다. 블록 l의 draw는 깨끗한 forward를 캐시해 두고 블록 l부터만 다시 계산합니다.
- **attention 내부(q, k, v, gate, value embedding):** 손실이 아니라 추정된 attention 출력 오차와의 정렬 c_s = −⟨ĝ_s, Δo_s⟩로 보상합니다(Eq. 4). k, v, gate, ve는 미래 토큰이 읽으므로 γ ≈ 0.98–0.99를 쓰고, 나머지는 γ = 0입니다.
- **LM head:** 캐시된 logit에서 256열 묶음(slab)씩만 흔들고 cross-entropy만 다시 계산합니다.
- **하이퍼파라미터 튜닝:** 층별 잡음 크기, credit decay, draw 배분을 정합니다. 배치 한 개에서 backprop gradient와의 cosine을 최대화해 후보를 고르고, 짧은 학습으로 확정합니다.
- **draw 배분(K = 16,384, 1M 기준):**
  - writer 쌍(attn·MLP 출력 투영) 7,040, MLP hidden 7,040, attention 출력 1,664, 토큰 임베딩 640
  - 앞 블록에 많이 줍니다(블록 1: 2,432 → 블록 8: 128).
  - head는 별도로 65,536 slab draw를 씁니다.

## 3. Experiments

### 3.1 설정 (§3.1, 부록 A)
- **모델:** modded-nanoGPT형 decoder 8층, 폭 512, 37.7M 파라미터.
  - RMSNorm, RoPE, squared-ReLU MLP
  - sliding window(블록 4개 중 3개는 1,024)
  - 블록 절반에 value embedding
- **데이터와 학습:** FineWeb, BPE 4,096. 배치 16,384 토큰(8 × 2,048), 1 epoch, 모멘텀 SGD, 상수 학습률.
- **평가:** 셀마다 3 시드. 검증 손실 기준으로 고른 체크포인트의 test 손실을 보고합니다.
- **비교 대상:** backprop(같은 grid로 튜닝), EGGROLL-Transformer(가중치 공간 ES, 같은 구조).

### 3.2 주 결과 (Table 1, Fig. 2)

| 예산 | Dust 64 | 256 | 1k | 4k | 16k | Dust ∞(fit) | EGGROLL 16k | Backprop |
|---|---|---|---|---|---|---|---|---|
| 100k | 7.240 | **7.165** | **7.151** | **7.164** | **7.170** | 7.161 | 7.263 | 7.216 |
| 1M | 6.290 | 6.067 | **5.934** | **5.911** | **5.916** | 5.892 | 6.708 | 5.959 |
| 10M | 5.397 | 5.231 | 5.111 | 5.072 | 5.049 | 5.013 | 6.033 | 4.989 |
| 20M | 5.265 | 5.139 | 4.960 | 4.912 | 4.802 | **4.431** | 5.865 | 4.633 |

굵은 값은 backprop보다 낮은 칸입니다. 저자 해석은 다음과 같습니다.

- 100k와 1M에서는 Dust가 backprop보다 낮습니다.
- 10M에서는 적합 극한이 backprop 바로 위입니다.
- 20M에서는 사다리가 아직 내려가는 중이고, power law 극한 4.431(95% 구간 3.89–4.58)이 backprop 4.633보다 낮습니다.
- EGGROLL은 population을 256배로 늘려도 Dust@64에 못 미칩니다.

### 3.3 Adam (§3.3, Fig. 3)
1M 토큰에서 세 방법을 모두 Adam으로 다시 튜닝했습니다. Dust의 극한은 5.248(구간 5.17–5.30)로, backprop보다 낮다고 보고합니다. 그림에 표시된 backprop 값은 5.361입니다. EGGROLL은 Adam으로 거의 나아지지 않습니다.

### 3.4 과대매개변수화 (§4.1, Fig. 4)
10M 토큰에서 2M, 7.3M, 38M, 243M 네 가지 크기를 비교했습니다. 저자 해석은 다음과 같습니다.

- 256 draw 이상에서는 2M → 7M → 38M으로 갈수록 손실이 낮아집니다. 243M은 조금 나쁩니다.
- 1k 이후 population 증가로 얻는 이득은 큰 모델이 약 30% 더 큽니다.
- 모델 크기를 "탐색 공간의 크기와 기하"로 봐야 한다고 해석합니다.

### 3.5 gradient 정렬 (§4.2, Fig. 5, 부록 G)
backprop으로 학습한 10M, 100M, 1B 토큰 체크포인트에서, Dust 추정치와 backprop gradient의 cosine을 population 64–128k에 걸쳐 쟀습니다.

- cos(K) = c_max/√(1 + c/K)가 층 종류마다 RMSE 0.06 이하로 맞습니다(Eq. 5).
- c_max는 head·Q·K·value emb에서 1.0이고, MLP·attn out·V에서 0.86–0.88, token emb 0.73, value gate 0.67입니다(Table 8).
- 같은 forward 수의 EGGROLL은 head를 빼면 cosine 0.05 미만입니다.

## 4. Limitations

### 4.1 저자가 밝힌 한계 (§1, §5)
- 연산 효율은 목표가 아니었습니다. backprop을 대체하려면 연산 효율이 몇 자릿수 더 좋아져야 합니다.
- 미분 불가능한 구조, 외부 프로그램이 들어간 망, 루프 구조 학습은 향후 과제입니다.
- population이 크면 backprop보다 낮아지는 메커니즘은 아직 모릅니다.

### 4.2 추가로 확인한 한계 (`03`, `04`에서 근거)
1. **"backprop을 넘는다"는 고정 토큰 비교이고, 연산량은 수천 배입니다.**
   - K = 16k의 update 한 번은 forward 약 13,600회, 즉 backprop step(약 forward 3회)의 약 4,500배입니다(우리 환산).
   - 같은 연산량이면 backprop은 1M 대신 수십억 토큰을 볼 수 있습니다(`03` §2).
2. **실험이 학습 극초기에 몰려 있습니다.**
   - step 수는 100k 예산에서 6, 1M에서 61, 20M에서 1,221입니다.
   - 토큰/파라미터는 최대 0.53입니다. Chinchilla 기준의 약 2.7%입니다.
   - backprop 손실도 균등 예측 손실(ln 4096 = 8.32)에서 13–44%만 내려간 구간입니다(`03` §1).
3. **20M의 "극한이 backprop보다 낮다"는 결론이 불안정합니다.**
   - 다섯 칸 중 한 칸을 빼고 다시 적합하면 극한이 2.57–4.68로 흔들립니다. 64 draw 칸을 빼면 극한이 backprop보다 높습니다.
   - 사다리 맨 위(16k) 칸만 자기 상수로 따로 튜닝되었습니다(부록 F)(`03` §3).
4. **학습 population에서의 cosine은 "가깝다"고 하기 어렵습니다.** Table 8 법칙으로 K = 16k의 cosine을 계산하면 head 0.98을 빼고 0.16–0.57입니다(`03` §4).
5. **c_max < 1인 층에는 공통점이 있습니다.**
   - c_max < 1인 층은 자기 토큰 손실만으로 보상받는 층(γ = 0이고, 출력이 뒤 토큰에도 영향을 주는 층)입니다.
   - c_max = 1인 층은 영향이 자기 토큰에만 미치거나(head, Q), 미래 credit을 받는 층(K, value emb)입니다.
   - 우리 toy는 이 해석과 맞습니다(`03` §5).
6. **과대매개변수화 결론은 약합니다.**
   - 243M은 모든 population에서 38M보다 나쁩니다. backprop에서도 243M(5.048)이 38M(5.015)보다 나쁩니다.
   - 10M 토큰은 243M에게 토큰/파라미터 0.04입니다.
   - backprop과의 차이는 64 draw에서 2M 0.525 → 243M 0.671로 커집니다(`03` §1).
7. **불확실성이 표에 없습니다.** 시드별 값과 표준편차가 보고되지 않았습니다. 100k의 Dust-backprop 차이(0.05–0.07)가 시드 변동보다 큰지 판단할 수 없습니다.
