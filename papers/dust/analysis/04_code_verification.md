# 04. 논문 ↔ 코드 대조와 주장 검증

- **공식 코드:** https://github.com/qlabs-eng/dust
  - 커밋 `b20f7c03eac630b6441ba6f254128c1761dc45ad`(2026-10-06), MIT 라이선스
  - 구성: `dust.py` 607줄, `model.py` 201줄, `baselines/backprop.py` 148줄
  - `code_official/`에 clone했고, git에서는 제외합니다.
- **README의 범위:** "논문의 추정기와 튜닝된 기본값은 유지하되, 실험에 쓴 실행 최적화는 뺀 최소 구현"이라고 적혀 있습니다.
  - 1M·10M 예산과 population 256, 1k, 4k, 16k만 지원합니다.
  - 실행에는 NVIDIA GPU(bf16)가 필요합니다.

## 1. 논문과 코드가 일치하는 부분

| 항목 | 논문 | 코드 | 위치 |
|---|---|---|---|
| draw 배분 (K = 16,384, 1M) | 부록 Table 3·4 | writers [38, 32, 16, 10, 6, 4, 2, 2] × 64 = [2432, …, 128], hubs [6, 6, 4, 2, 2, 2, 2, 2] × 64, embedding 640, local (q, v, gate, k, ve) = (128, 1280, 384, 768, 512), head 65,536 | `recipe()` |
| 10M@16k 별도 배분과 γ | 부록 Table 4, F: γ = 0.99 | writers [34, 28, 14, 8, 6, 4, 2, 2] × 64, hubs [12, 10, 8, 4, …] × 64, `gamma=0.99` | `DEFAULT_CONFIGS['10M']['allocation']` |
| K의 정의 | writer·MLP·attention 출력·embedding의 손실 보상 draw 합 | 2·7,040 + 1,664 + 640 = 16,384 (10M@16k도 16,384) | `table_check.py` V4에서 재구성 |
| 잡음 크기 | 0.2, 깊은 절반 attention 출력 0.4, 내부 0.05, head 0.05, 잔차 스칼라 0.03 | 같음 | `step()`, `local_errors`, `head_error`, `scalar_gradients` |
| 가중치 gradient | Ĝ = Σ_t ĝ_t x_tᵀ, embedding은 scatter-add | `reassemble()` | |
| attention 내부 보상 | Eq. 4, Q는 자기 토큰, 나머지는 γ로 미래 토큰 합산 | `local_errors`: `score` → `credit` 행렬(γ^lag, 창 안) | |
| 잔차 스칼라 | 2L개는 가중치 공간 ES | 양측(antithetic) ES 8 draw | `scalar_gradients` |
| 학습 step | 1M = 61, 10M = 610 | 같음 (배치 8 × 2,048) | `DEFAULT_CONFIGS` |
| 학습률 그룹 | 부록 Table 6: 토큰 임베딩 1000, value emb 0.3, 잔차 0.03 | 같음. backprop은 임베딩 1000, 나머지 base | `make_optimizer` (두 파일) |
| backprop 기준선의 공정성 | 같은 모델·데이터·평가 | 같은 `make_model`(출력 투영 초기화 포함), 같은 데이터 순서(seed + 1), 같은 평가 | `baselines/backprop.py` |

## 2. 논문과 코드가 다른 부분, 또는 논문에 없는 세부 (D1–D8)

| # | 항목 | 논문 | 코드 | 영향 |
|---|---|---|---|---|
| D1 | 보상 baseline | c_s = ℓ̃_s − ℓ_s (같은 묶음 draw 평균과의 차, Eq. 1 앞) | 깨끗한 손실 − 섭동 손실을 구한 뒤, 묶음(n ≥ 2) 안 평균을 뺌. n = 1이면 중심화 없음 | n = 1이면 논문 식은 0이 되므로 코드가 맞는 구현. 묶음 중심화는 추정 크기를 (1 − 1/n)배로 줄임(`03` §5.3) |
| D2 | head 보상의 baseline | 명시 없음 | **draw 묶음이 아니라 배치의 8개 시퀀스에서 같은 위치의 평균**을 뺌(`reward.mean(1)`) | 다른 시퀀스의 손실을 baseline으로 쓰는 셈. 편향은 작지만 논문 서술과 다름 |
| D3 | gate의 섭동 | 일반적인 한쪽 섭동으로 서술 | gate만 양측 차분(z + σa, z − σa), 나눗셈 2σ | 분산 감소. 논문에 없음 |
| D4 | writer 쌍 | "층 종류마다 다른 pass"(§2.3) | attn.c_proj와 mlp.c_proj를 **같은 pass에서 함께** 흔듦(부록 Table 3의 "Writer pair") | 같은 블록의 두 출력 사이에 간섭이 생김. toy에서 함께 흔들면 c가 1.6배(`03` §5.3) |
| D5 | 지원 범위 | 100k, 1M, 10M, 20M × 64–16k, Adam, 모델 크기 4종 | 1M·10M × 256–16k, SGD만. 64 draw(n = 1), 100k, 20M, Adam, 모델 크기 설정 없음 | 논문 핵심 수치 중 20M 극한(4.431), Adam 결과, 과대매개변수화는 공개 코드로 재현할 수 없음 |
| D6 | EGGROLL-Transformer | 같은 구조로 구현, 자체 grid 튜닝 | 저장소에 없음 | Table 1의 EGGROLL 행과 Table 2 외삽을 확인할 수 없음 |
| D7 | 시드·튜닝 절차 | seed 42로 grid, 상위 후보에 43·44 (부록 F) | `--seed` 인자 하나. grid 스크립트 없음 | 3 시드 평균과 선택 절차를 재현하려면 직접 작성해야 함 |
| D8 | 실행 최적화 | 블록 l부터 재계산, GPU 분할 | 최소 구현이라 블록 l 섭동도 forward 전체를 다시 계산(`self.losses(x, y)`) | 결과는 같고 시간만 더 듦. 연산량 환산(`03` §2)은 논문 방식(블록 l부터) 기준 |

## 3. 수치 정합성 (V1–V4)

| # | 확인 | 결과 |
|---|---|---|
| V1 | 본문 수치 12개(§3.2, §4.1, §4.2) ↔ 부록 표 | 모두 일치 |
| V2 | Table 1의 ∞ 행을 power law로 재적합 | 4칸 모두 ±0.003 이내로 일치. 다만 20M은 한 칸 빼기에 매우 민감(`03` §3) |
| V3 | Table 2(EGGROLL 외삽) | 직선 적합 6칸 일치, power law는 1M만 일치(10M 10,200배 vs 16,000배, 20M 5,700배 vs 8,100배) |
| V4 | 코드의 draw 배분 ↔ 부록 Table 3·4 | 8개 항목 모두 일치, K = 16,384 |

## 4. 주장별 판정

판정: ✅ 지지됨 · ⚠️ 부분적으로 지지되거나 해석에 주의 · ❌ 근거 부족

| # | 주장 (위치) | 판정 | 근거 |
|---|---|---|---|
| C1 | Transformer 사전학습에서 backprop과 경쟁할 수 있는 최초의 0차 방법 (TL;DR, §1) | ⚠️ | 같은 토큰 수에서 손실이 비슷한 것은 맞음. 다만 실험은 최대 1,221 step, 토큰/파라미터 0.53 이하이고, 연산은 backprop의 수천 배(`03` §1–2) |
| C2 | population이 크면 여러 설정에서 backprop보다 낮다 (§3.2) | ⚠️ | 100k(6 step), 1M(61 step)에서만 측정값으로 낮음. 차이 0.04–0.07이고 시드별 값 없음. 10M·20M은 측정값이 모두 backprop보다 높음 |
| C3 | 20M 극한 4.431 < backprop 4.633 (§3.2, Fig. 2) | ❌ | 한 칸 빼고 재적합하면 2.57–4.68. 64 칸을 빼면 backprop보다 높음. 맨 위 칸은 별도 튜닝(`03` §3). 저자도 "측정된 극한이 아님"이라고 씀 |
| C4 | 가중치 공간 ES보다 10³–10⁴배 효율 (TL;DR, 부록 D) | ✅ (population 기준) | 외삽 산술은 대체로 재현. "효율"은 같은 손실에 필요한 population 배수이고, draw당 연산은 0.82 forward라 연산 기준으로도 비슷함 |
| C5 | Adam에서도 backprop보다 극한이 낮다 (§3.3) | ⚠️ | 그림만 있고 표가 없음(backprop 5.361, Dust 극한 5.248). 공개 코드에 Adam 설정 없음(D5) |
| C6 | 큰 모델이 population 효율이 더 높다 (§4.1) | ⚠️ | 2M → 38M은 맞음. 243M은 모든 population에서 38M보다 나쁘고, backprop과의 차이도 64 draw에서 0.525 → 0.671로 커짐. 243M에게 10M 토큰은 토큰/파라미터 0.04 |
| C7 | population이 커질수록 gradient가 backprop에 정렬, 1B 토큰까지 유지 (§4.2) | ✅ | 법칙 적합 RMSE 0.0032–0.0367 확인. 다만 학습 population(16k)의 cosine은 head를 빼면 0.16–0.57(`03` §4) |
| C8 | 근사가 backprop과 정확히 같지 않은 것이 오히려 좋은 성질이다 (§4.2) | ❌ (해석) | c_max < 1의 주요 원인은 자기 토큰만 보상하는 구조적 편향으로 설명됨(toy에서 c_max = cos(국소, 전체 gradient)). 더 좋은 궤적이라는 근거는 C2의 극초기 결과뿐(`03` §5–6) |
| C9 | 활성값 섭동이 가중치 섭동보다 population을 싸게 만든다 (§2.1) | ✅ | toy에서 같은 forward 수로 cosine 0.41 vs 0.18(K = 64). 이론상 분산이 파라미터 수가 아니라 d_out × 토큰 비정합도에 비례 |

## 5. 구현 가능성

### 5.1 공식 코드로 재현하는 경우
- **요구 사항:** 공식 코드는 CUDA GPU만 받습니다(`parser.error('Training requires a CUDA GPU.')`).
- **1M × 256 단일 GPU:** 61 step × update당 약 229 forward이므로 forward 약 1.4만 번입니다. 최소 구현에서는 블록 l부터가 아니라 전체 forward를 다시 계산하므로 더 걸립니다. GPU 한 장에서 시간 단위로 예상합니다(우리 추정).
- **10M × 16k:** 610 step × 13,565 forward ≈ 830만 forward이고, 8 GPU를 씁니다.
- **이 기기:** GPU 재현이라 하지 않습니다.

### 5.2 CPU로 가능한 것
- **numpy toy(`dust_toy.py`):** 추정기 식, cosine 법칙, γ의 편향-분산 교환을 확인했습니다(이 문서 §4의 C8, C9 근거).
- **아주 작은 Transformer:** 2층, 폭 64, 문맥 64 정도라면 CPU PyTorch로 Dust vs backprop을 몇백 step 돌릴 수 있습니다. torch 설치가 필요합니다.
