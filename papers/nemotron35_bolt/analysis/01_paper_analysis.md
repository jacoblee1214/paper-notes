# 01. 논문 분석: Nemotron 3.5 Bolt 74B-A9B 24GB (arXiv:2610.33039)

- 제목: *Nemotron 3.5 Bolt 74B-A9B 24GB: A Hybrid Vision-Language Model for Local AI*
- 저자: NVIDIA (기여자 100여 명, Contributors 절)
- 공개: arXiv v1, 2026-10-15 (cs.CV, 19쪽, 부록 없음) · 연구 페이지 https://research.nvidia.com/labs/nemotron/Nemotron-3.5-Bolt-24GB/ · 가중치 Hugging Face `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-*` (OpenMDW-1.1)
- 라이선스: arXiv abs 페이지는 **CC BY 4.0**입니다. 그런데 PDF 첫 쪽 아래에 **"© 2026 NVIDIA. All rights reserved."**가 함께 찍혀 있습니다. 권리 표기가 충돌하므로 논문 그림은 개인용 리포트·PPT에만 쓰고 게시본에서는 뺐습니다(`thinkworld/`와 같은 판단).
- 분석 범위: 본문 19쪽, 연구 페이지, Hugging Face 모델 카드와 `config.json`·`hf_quant_config.json`(받은 날짜 2026-10-30), 공개 GGUF 세 개의 **헤더만**(HTTP Range로 텐서 목록과 메타데이터, 가중치는 받지 않음), llama.cpp 소스 일부(`889f2ec`). 학습 코드는 공개되지 않았습니다.

---

## 1. Abstract (한 줄 요약과 주장)

**한 줄 요약:** Nemotron 3.5 Super VL(120B 전체, 12B 활성)을 Star Elastic으로 74B-A9B로 줄이고, 후학습으로 회복한 뒤, 전문가(expert) 가중치를 2.0625비트(IQ2_XXS)로 양자화 인지 증류(QAD)해서 **텍스트 전용 배포 메모리를 24 GB 안으로 맞춘** 로컬용 하이브리드 Mamba-Transformer MoE 비전-언어 모델입니다. 병렬 초안 모델 LiLiCorr를 함께 내서 RTX Spark에서 84 TPS를 냅니다.

논문이 내세우는 것은 네 가지입니다(Sec. 1).

1. 메모리 예산을 파라미터 예산으로 바꿔 층별 expert 수와 MoE 폭을 고르는 압축(Star Elastic, Sec. 2).
2. "작은 모델을 높은 비트로"보다 "큰 모델을 낮은 비트로"가 같은 크기에서 더 정확하다는 판단과, 이를 위한 2비트 QAD(Sec. 5).
3. 같은 메모리대(Qwen3.6 35B-A3B, Gemma 4 26B-A4B)보다 에이전트·웹·터미널 벤치마크에서 높은 정확도(Table 7).
4. llama.cpp와 vLLM 양쪽 런타임 지원과 LiLiCorr 초안 모델(Sec. 6–7).

---

## 2. Method

### 2.1 전체 흐름 (Fig. 2, Fig. 5, Fig. 7)

```
Nemotron 3.5 Super VL 120B-A12B (40 Mamba / 40 MoE / 8 attention, 512 experts × 폭 2688, top-22)
  │ Star Elastic: 예산 조건부 라우터가 층별 expert 수·MoE 폭을 고름 + 부모 KD (8B 토큰에서 수렴)
  ▼
압축 체크포인트 74B-A9B (층별 E ∈ {240…496}, I ∈ {1536…2560}, top-9) → KD 회복 64B(8K) + 48B(128K) = 총 120B 토큰
  │ 비전: 부모의 비전 타워를 다시 붙이고 projector만 학습(6.6M 이미지) → 비전 중심 SFT → SLERP 80:20
  │ RL: MCP-Atlas RL → RLHF → SWE-RL → (RLHF / 연장 SWE-RL) → SLERP 75:25  = 공개 BF16
  ▼
QAD 1단계: 이전 SWE-RL BF16을 교사로 26B 토큰 (Jaccard 손실)
QAD 2단계: 최종 BF16을 교사로 10M 토큰 (토큰 효율)          = 공개 IQ2 "24GB"
  + LiLiCorr 초안 모델(1.13B, 8칸 블록) → llama.cpp(GGUF) / vLLM(ModelOpt) 배포
```

### 2.2 압축: Star Elastic (Sec. 2, Eq. 1–3, Table 1)

- **정렬과 접두 마스크:** 보정 데이터 8M 토큰으로 expert(REAP 점수)와 MoE 중간 채널(Star Elastic 점수)을 중요도 순으로 정렬합니다. 후보 크기 c_j마다 "앞에서 c_j개만 남기는" 마스크 m_j[i] = 1{i ≤ c_j}를 둡니다(Eq. 1). 작은 마스크가 큰 마스크의 부분집합이 됩니다.
- **미분 가능한 선택:** 라우터(축마다 MLP 1 → 256 → 40·C_a)가 낸 로짓에 Gumbel-softmax를 씌워 마스크를 가중 평균합니다(soft masking, Fig. 1). 덕분에 KD 손실이 아키텍처 선택까지 미분으로 전달됩니다.
- **예산 손실:** 라우터 가중 파라미터 추정 P(ψ; b)가 목표 b에서 ε = 5% 넘게 벗어날 때만 상대 오차 δ로 벌칙을 줍니다(Eq. 2). 전체 목적은 L_KD + λ·L_budget이고 λ = 1입니다(Eq. 3). 예산 손실은 라우터만, KD는 라우터와 학생을 모두 학습시킵니다.
- **얻은 구조 (Table 1):** Mamba·MoE·attention 층 수, hidden 4096, latent MoE 폭 1024, shared expert 폭 5376은 부모와 같습니다. 층별 routed expert 수 ∈ {240, 304, 368, 432, 496}, MoE 폭 ∈ {1536, 2048, 2304, 2560}이고, top-K는 22 → 9로 고정했습니다.
- **목표:** 배치 1, 128K 문맥에서 24 GB 미만입니다(Sec. 2.3). 학습은 GB200 128장 × 약 64시간입니다.

### 2.3 비전 적응과 회복 학습 (Sec. 3–4, Fig. 5–6)

- 압축은 **텍스트 타워에만** 적용했고, KD도 텍스트 벤치마크 회복만 겨냥했습니다. 그 결과 압축 직후 비전 점수가 무너집니다(Fig. 6: WebArena 57.4 → 1.35, ScreenSpot-Pro 41.6 → 0, RefCOCO 80.45 → 3.09).
- 비전 projector만 학습(6.6M 이미지, 영상·텍스트 없음)하면 대부분 돌아오고, 비전 중심 SFT(54.1B 유효 토큰, 비전 토큰 47.5%)로 더 회복합니다. 그 뒤 SLERP로 병합합니다.
- RL은 MCP E2E RL(512 배치, MCP-Atlas 48 → 56), RLHF(Nemotron 3 Ultra GenRM), SWE-RL(1,024 궤적/스텝, 최대 196,608 토큰)입니다.

### 2.4 2비트 양자화와 QAD (Sec. 5, Table 2, Eq. 4)

- routed·shared expert 전체를 IQ2_XXS(2.0625비트)로, Mamba 입출력 projection을 NVFP4(4.5비트)로, attention·latent MoE·임베딩·LM head를 Q8_0(8.5비트)으로, router와 conv를 F32로 둡니다.
- Table 2의 가중치 합은 22.01 GB이고, 런타임(BF16 SSM 상태 0.08, FP8 KV 0.54, 활성 버퍼 0.25)을 더하면 22.87 GB, 초안 모델 0.96을 더하면 23.83 GB입니다. **비전 인코더·projector 1.68 GB는 이 합계에서 제외**합니다(Sec. 8.2).
- **QAD:** 학생(2비트)과 고정된 교사(BF16) 출력 분포의 가중 Jaccard 거리 L = 1 − Σmin(p_s, p_t) / Σmax(p_s, p_t)를 최소화합니다(Eq. 4). KL, 역 KL, JSD, MSE, cosine보다 낫다고 보고합니다(수치 없음). 학습률은 1e-5(1.2B 토큰 warmup)에서 1e-7까지 cosine입니다.
- **2단계:** 1단계는 이전 SWE-RL 체크포인트를 교사로 26B 토큰, 2단계는 최종 BF16을 교사로 10M 토큰입니다(Fig. 7).

### 2.5 LiLiCorr 초안 모델 (Sec. 6, Table 3–6)

- DFlash식 병렬 초안 본체(5층, 그룹 동적 conv, 1,024 슬라이딩 창 + attention sink) 위에 **2층 상관 헤드(37M)**를 둡니다.
- 블록 B = 8(초안 7칸), 위치마다 후보 K = 8개를 고르고, 인접 위치 후보끼리의 cosine 호환 점수를 한 번에 계산한 뒤, 왼쪽부터 추가 계산 없이 고릅니다.
- 학습 데이터는 7.55M 샘플(입력 19.14B 토큰)이고, GB200 96장(학습 64, 타깃 서빙 32)을 썼습니다. NVFP4 가중치 전용 PTQ로 DGX Spark 처리량이 11% 올라갑니다.

---

## 3. Experiments

### 3.1 압축 품질 (Fig. 4)
9개 지표 평균이 부모 57.58에서 압축 체크포인트 54.74로 2.85점(4.94%) 내려갑니다. 이 비교는 SFT·RL 회복과 양자화 **전** 체크포인트입니다.

### 3.2 양자화 품질과 경쟁 모델 (Table 7, Fig. 8–9)

| | Super VL BF16 | Bolt BF16 | **Bolt 24GB (2비트)** | Qwen3.6 35B-A3B BF16 | Gemma4 26B-A4B | Lightning 30B-A3B |
|---|---|---|---|---|---|---|
| GDPVal-AA-V2 | 34.48 | 31.13 | **28.47** | 25.75 | 15.33 | 18.26 |
| Terminal-Bench 2.1 | 61.94 | 53.93 | **50.98** | 44.38 | 37.22 | 23.46 |
| BrowseComp | 71.50 | 70.00 | **66.75** | 48.74 | 26.30 | 36.81 |
| PinchBench | 90.25 | 90.35 | **87.82** | 88.07 | 74.70 | 83.43 |
| AA-Omniscience accuracy | 25.98 | 21.80 | **16.90** | 19.47 | 22.17 | 16.63 |
| IFBench | 73.04 | 68.71 | **66.17** | 63.71 | 77.25 | 72.88 |

(전체 14행은 `04` C4와 `analysis/table_check.py`)

해석:
- **2비트 손실은 대부분 2–5점입니다.** 14개 중 11개에서 BF16 대비 −2.4 ~ −4.9점입니다. 14개 평균은 부모 56.68 → Bolt BF16 53.17 → 2비트 51.09입니다.
- **두 곳에서는 2비트가 BF16보다 높습니다.** AA-Omniscience non-hallucination(+5.13), WebArena(+0.61)입니다. 2단계 QAD가 최종 BF16(RLHF 포함)을 교사로 썼기 때문일 수 있지만, 논문은 설명하지 않습니다.
- **경쟁 모델 대비 승패:** Qwen3.6·Gemma4·Lightning 각각에 대해 비교 가능한 13–14개 중 11개에서 이깁니다. AA-Omniscience accuracy에서는 Qwen3.6과 Gemma4 모두에게 지고, PinchBench에서는 Qwen3.6 BF16(88.07)보다 0.25점 낮습니다.
- **비교 조건이 섞여 있습니다.** 경쟁 모델의 정확도는 BF16으로 쟀습니다(Fig. 9 캡션). 그런데 "같은 메모리대"라는 묶음은 경쟁 모델이 4비트일 때의 크기 기준입니다(연구 페이지). 경쟁 모델에게 유리한 조건이라 Bolt의 우위 주장은 보수적인 쪽입니다.
- **가장 강한 24 GB급 경쟁자가 빠져 있습니다.** 2쪽 그림의 Qwen3.8 27B Q4는 GDPval 약 44로 Bolt(약 28)보다 훨씬 높지만, 속도가 28 TPS라서 Table 8에만 나오고 Table 7 정확도 비교에는 없습니다.
- **Fig. 8과 Table 7 사이에 점수 변화가 있습니다.** QAD 1단계(26B 토큰) 뒤 BrowseComp 69.25가 최종 66.75로 2.5점 내려가고, Terminal-Bench는 48.88 → 50.98로 오릅니다. 2단계(토큰 효율 교사)의 득실은 본문에 정리되어 있지 않습니다.

### 3.3 초안 모델 (Table 5–6)
SPEED-Bench 880문항에서 평균 수락 길이(AL)는 BF16 drafter 4.736(temp 0), NVFP4 4.681(−1.2%)입니다. 코딩 5.7, 글쓰기 3.7로 범주 차이가 큽니다. 실제 에이전트 재생(AA AgentPerf)에서는 LiLiCorr 4.18 대 DSpark 3.68(+13.6%)입니다.

### 3.4 속도 (Table 8)
RTX Spark(Surface Laptop Ultra 128 GB)·llama.cpp에서 Bolt 24GB + LiLiCorr는 83.99 TPS입니다. Qwen3.6 35B-A3B Q4 + MTP(96.63), Gemma 4 26B-A4B Q4 + MTP(100.56)보다 13–16% 느리고, 에이전트 재생 시간도 600.8초로 25–32% 깁니다. Qwen3.8 27B + DFlash(28.26 TPS, 1,043초)보다는 빠릅니다.

---

## 4. Limitations

### 4.1 저자가 밝힌 것
- 양자화는 "의도적으로 손실이 큽니다". 목표는 BF16에 가까운 것이 아니라 크기 대비 정확도입니다(Sec. 5.1).
- MCP-Atlas는 0.75 임계값 이진 판정이라 RL의 개선이 공식 점수에 반영되지 않았습니다(Sec. 4.2).
- 비전 인코더는 24 GB 목표에서 제외했습니다(Sec. 8.2).

### 4.2 분석하면서 추가로 발견한 것
1. **"24GB"는 텍스트 전용 예산입니다.** 비전(1.68 GB, GGUF mmproj 실측 1.73 GB)을 넣으면 25.52 GB로 24 GB를 넘습니다. 모델 카드의 공식 실행 레시피도 "28GB"(65K 문맥, q8_0 KV)와 "32GB"(256K)이고, 권장 하드웨어는 32 GB 이상 GPU 또는 64 GB Spark입니다(`04` C1, D3).
2. **디코드 대역폭의 절반은 Mamba 입출력 projection입니다.** 토큰 하나에 읽는 4.82 GB 중 2.47 GB(51%)가 NVFP4 Mamba 가중치이고, 2비트 routed expert는 0.44 GB뿐입니다. 2비트 expert는 주로 **용량**을 줄이고, **속도**는 Mamba 정밀도가 좌우합니다(`03` §2).
3. **"초안 블록 검증 비용 ≈ 토큰 하나"는 라우팅이 겹칠 때만 성립합니다.** 독립·균등 라우팅이면 8토큰 검증은 1.57배의 바이트를 읽습니다(`03` §3).
4. **비교 표가 선택적입니다.** GDPval이 가장 높은 Qwen3.8 27B가 정확도 표에 없습니다. Qwen3.6의 MCP-Atlas는 공식 62.8 대신 저자 재현 51.2를 씁니다. "PinchBench #1"(블로그·연구 페이지)은 Table 7과 맞지 않습니다(`04` C4).
5. **시드·분산 정보가 없습니다.** Table 7, 8은 단일 측정입니다(Table 5만 5시드). 0.25점 차이 같은 순위는 구분하기 어렵습니다.
6. **맥(Apple Silicon)에서 그대로 돌리기 어렵습니다.** llama.cpp의 Metal 백엔드는 NVFP4 행렬곱을 지원하지 않아(`889f2ec`), 디코드 바이트의 절반인 Mamba projection이 CPU로 넘어갑니다. Ollama도 74B-A9B는 Apple Silicon을 지원하지 않는다고 적고 있습니다. 24 GiB 맥은 메모리도 부족합니다(`04` §4).
7. **E4B는 같은 계열이 아닙니다.** 함께 발표된 "Bolt E4B 4GB"는 Gemma-4-E4B를 후학습한 모델(GGUF 아키텍처 `gemma4`, 8B 전체 / 4B 유효)이고, 공식 GGUF는 6.18 GB입니다(`04` D10).
