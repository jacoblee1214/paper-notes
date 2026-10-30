---
title: "Nemotron 3.5 Bolt 24GB 읽기: 이름의 24GB는 텍스트 전용이고, 속도는 2비트가 아니라 Mamba가 정한다"
date: 2026-10-30
tags: [llm, vlm, quantization, moe, mamba, speculative-decoding, local-inference, paper-review]
paper: https://arxiv.org/abs/2610.33039
project: https://research.nvidia.com/labs/nemotron/Nemotron-3.5-Bolt-24GB/
summary: "120B-A12B VLM을 2비트 74B-A9B로 줄여 24 GB에 넣은 Nemotron 3.5 Bolt를, 공개 config와 GGUF 헤더로 메모리 표를 대조하고 디코드 대역폭을 계산하며 읽었습니다."
---

> **논문:** NVIDIA. *Nemotron 3.5 Bolt 74B-A9B 24GB: A Hybrid Vision-Language Model for Local AI.* arXiv:2610.33039v1 (2026-10-15).
> arXiv 라이선스는 CC BY 4.0이지만 PDF에 "© 2026 NVIDIA. All rights reserved." 문구가 함께 있어, 논문 그림은 싣지 않았습니다. 그림은 [연구 페이지](https://research.nvidia.com/labs/nemotron/Nemotron-3.5-Bolt-24GB/)와 논문에서 볼 수 있습니다. 이 글의 그래프는 논문 표 수치와 공개 GGUF 헤더로 직접 그렸습니다. 가중치와 런타임은 공개되어 있고(OpenMDW-1.1), 학습 코드는 공개되지 않았습니다.


## 한 줄 요약

Nemotron 3.5 Super VL(120B-A12B, 하이브리드 Mamba-Transformer MoE)을 **Star Elastic**으로 74B-A9B까지 줄입니다. 층마다 expert 수와 폭을 예산 조건부 라우터가 고르고, top-K를 22에서 9로 낮춥니다. 그 뒤 SFT·RL로 성능을 회복하고, expert를 **2.0625비트(IQ2_XXS)**로 양자화 인지 증류(QAD)합니다. 여기에 1.13B 병렬 초안 모델 **LiLiCorr**를 붙여 RTX Spark에서 84 TPS를 냅니다.

가중치를 받지 않고 확인할 수 있는 것을 확인했습니다. 공개 `config.json`, GGUF 파일의 헤더(HTTP Range로 앞부분만), llama.cpp 소스를 대조하고, 메모리와 디코드 대역폭을 계산했습니다.

## 1. 메모리 표는 공개 파일과 정확히 맞는다. 다만 "24GB"는 텍스트 전용이다

공개 GGUF의 텐서 843개를 형식별로 묶으면 논문 Table 2와 구성 요소마다 0.01 GB 안에서 맞습니다. 파라미터도 config로 세면 74.043B 전체, 9.06B 활성이고, 같은 식에 부모 설정을 넣으면 120.67B / 12.23B가 나옵니다. 이름의 "74B-A9B"와 "120B-A12B"는 정확합니다.

![메모리 예산](images/memory_budget.png)

| 구성 요소 | Table 2 (GB) | GGUF 헤더 (GB) | 형식 |
|---|---|---|---|
| routed experts | 17.05 | 17.053 | IQ2_XXS |
| Mamba 입출력 | 2.47 | 2.466 | NVFP4 (ggml type 40) |
| 임베딩 + LM head | 1.14 | 1.141 | Q8_0 |
| shared / latent / attention | 0.45 / 0.36 / 0.30 | 0.454 / 0.357 / 0.303 | IQ2_XXS / Q8_0 / Q8_0 |
| **가중치 합** | **22.01** | **22.010** | |

그런데 숫자를 끝까지 더하면 이름의 뜻이 좁아집니다.

- 가중치 22.01 + KV·SSM 상태 0.87 + 초안 0.96 = **23.84 GB이고, 텍스트 전용입니다.** 비전 인코더(1.68 GB)를 넣으면 25.52 GB입니다.
- 24 GiB(= 25.77 GB)로 읽으면 비전까지 들어가지만 여유는 0.25 GB입니다.
- 모델 카드의 공식 llama.cpp 레시피 이름은 **28GB**(65K 문맥)와 **32GB**(256K)이고, 권장 장치는 32 GB 이상 GPU 또는 64 GB 이상 Spark입니다. 24 GB 장치용 레시피는 없습니다.

작은 차이도 있습니다. config의 SSM 캐시 정밀도는 `float32`라 Table 2의 BF16 값(0.08 GB)의 두 배이고, `kv_cache_quant`는 비어 있어 FP8 KV는 런타임에서 따로 맞춰야 합니다. 예산을 넘기지는 않지만, 남는 예산이 0.17 GB인 설계에서는 무시할 크기가 아닙니다.

## 2. 2비트 expert는 용량을, NVFP4 Mamba는 속도를 정한다

배치 1 디코드는 메모리 대역폭에 묶입니다. 토큰 하나를 만들 때 읽는 가중치를 GGUF 바이트와 층별 expert 수로 계산했습니다.

![토큰당 읽는 바이트와 초안 검증 비용](images/per_token_bytes.png)

- 토큰당 4.82 GB를 읽고, 그중 **51%가 NVFP4 Mamba 입출력**입니다. routed expert는 층마다 9개만 읽으므로 9%입니다.
- routed expert는 저장 용량의 77%를 차지합니다. 그러니 2비트 양자화는 "24 GB에 넣기"에는 결정적이지만, 속도에는 적게 기여합니다.
- Mamba 입출력까지 2.0625비트로 내리면 토큰당 바이트가 약 28% 줄어드는 계산입니다. 논문은 Mamba를 4.5비트로 둔 이유를 설명하지 않습니다.

초안 블록 검증 비용도 같은 틀로 볼 수 있습니다. 논문은 "가중치 읽기가 병목이라 블록 검증이 토큰 하나 생성과 거의 같다"고 씁니다. MoE에서는 블록의 토큰마다 다른 expert를 고르면 층마다 그 합집합을 읽어야 합니다. 8토큰 블록의 비용은 라우팅이 서로 독립이면 단일 토큰의 1.57배, 완전히 같으면 1.0배입니다. routed expert 비중이 작아서, 이 모델에서는 논문의 서술이 대체로 맞습니다. 다만 expert 비중이 큰 다른 MoE에 그대로 옮길 수는 없습니다.

## 3. 2비트의 비용과 경쟁 비교

![Table 7 벤치마크별 변화](images/table7_deltas.png)

- **BF16 → 2비트:** 14개 벤치마크 중 11개에서 2.4–4.9점 떨어지고, 평균은 53.17 → 51.09(−2.08점)입니다.
- **부모 → 2비트:** 평균 56.68 → 51.09입니다. 압축(−3.51점)이 양자화(−2.08점)보다 큰 손실이고, Terminal-Bench가 −10.96점으로 가장 크게 떨어집니다.
- **AA-Omniscience:** 2비트에서 non-hallucination은 5.13점 오르고 accuracy는 4.90점 떨어집니다. "모르면 답하지 않는" 쪽으로 성향이 바뀐 것으로 읽힙니다(가설).

경쟁 모델과의 비교는 대체로 맞지만, 읽을 때 조건을 같이 봐야 합니다.

| 주장 | 확인한 것 |
|---|---|
| 같은 메모리대 모델보다 정확하다 | Qwen3.6-35B-A3B에는 14개 중 11개, Gemma 4 26B-A4B와 Nemotron 3.5 Lightning 30B-A3B에는 값이 있는 13개 중 11개에서 앞섭니다. AA-Omniscience accuracy는 Qwen3.6과 Gemma4 모두에게 집니다. |
| "PinchBench #1" (블로그·연구 페이지) | Table 7 안에서는 Qwen3.6 88.07 > Bolt 24GB 87.82입니다. 다른 리더보드 기준으로 보입니다. |
| MCP-Atlas 우위 | Qwen3.6은 공식 62.8 대신 저자 재현값 51.2를 썼습니다. 공식값이면 Bolt(52.20)가 집니다. |
| 비교 범위 | Qwen3.8 27B는 논문 2쪽 그림에서 GDPval 약 44로 Bolt(약 28)보다 높고, 속도 표(Table 8)에도 있지만, 정확도 표(Table 7)에는 없습니다. |
| 84 TPS가 "비슷하다" | Qwen3.6(Q4 + MTP) 대비 0.87배, Gemma 4 대비 0.84배로 13–16% 느립니다. dense인 Qwen3.8 27B(28 TPS)보다는 2.97배 빠릅니다. |

경쟁 모델의 정확도는 BF16으로, 크기는 4비트로 놓고 비교한 점도 함께 적어 둡니다. 조건을 밝혔으니 잘못은 아니지만, 표를 옮겨 쓸 때 쉽게 섞이는 종류의 차이입니다. seed나 반복 횟수가 없어서 1–2점 차이의 유의성은 판단할 수 없습니다.

QAD는 두 단계입니다. 1단계는 이전 SWE-RL 체크포인트를 교사로 26B 토큰, 2단계는 최종 BF16을 교사로 10M 토큰입니다. 1단계 끝(Fig. 8)과 최종(Table 7)을 비교하면 BrowseComp는 69.25 → 66.75로 내려가고 Terminal-Bench는 48.88 → 50.98로 오릅니다. 2단계의 득실은 본문에 설명이 없습니다.

## 4. 공개 파일과 다른 점

- **초안 모델:** HF config의 대상 층은 6개 [1, 18, 35, 51, 68, 85]이고 파라미터는 1.113B입니다. 논문은 "다섯 층 + 최종 hidden state", 1.13B라고 씁니다.
- **문맥 길이:** 설계 예산은 128K인데 모델 카드는 256K를 광고합니다.
- **비전:** mmproj GGUF는 1.734 GB로 Table 2의 1.68 GB보다 조금 큽니다. 모델 카드는 영상 입력을 지원한다고 쓰지만, 논문의 비전 회복 학습에는 영상 데이터가 없습니다.
- **"Bolt E4B 4GB":** 같이 발표된 소형 모델의 GGUF 아키텍처는 `gemma4`(7.52B)입니다. 이름은 같은 Bolt지만 이 논문의 하이브리드 Mamba 계열이 아닙니다.

## 5. 직접 돌려 볼 수 있나

| 환경 | 판단 | 이유 |
|---|---|---|
| NVIDIA GPU 32 GB+, RTX/DGX Spark | **가능** | 공식 28GB·32GB 레시피가 있고, 필요한 llama.cpp·vLLM PR이 모두 병합됨 |
| NVIDIA GPU 24 GB | 애매 | 공식 레시피 없음. 텍스트 전용·짧은 문맥이면 경계선 |
| 24 GiB 통합 메모리 맥 (M4 Pro, Metal) | **공식판은 사실상 불가** | 32K 문맥에 22.96 GB가 필요해 기본 GPU 한도(RAM의 약 2/3로 가정)를 넘음. 게다가 llama.cpp Metal 백엔드는 NVFP4 행렬곱을 지원하지 않아 Mamba 입출력이 CPU로 넘어감. Ollama 라이브러리 페이지도 74B-A9B의 Apple Silicon 미지원을 명시 |
| 같은 맥 + 커뮤니티 1비트대 GGUF (18.5 GB) | 조건부 | NVFP4를 쓰지 않아 Metal에서 돌 수 있음. GPU 한도를 올리고 비전·초안 없이 짧은 문맥으로. 품질은 검증되지 않음 |
| 학습(압축·QAD) | 불가 | 코드 미공개, GB200 128장 × 64시간 규모 |

맥 판단은 설치나 다운로드 없이 계산으로만 했습니다. 실제 GPU 한도(llama.cpp 시작 로그의 `recommendedMaxWorkingSetSize`)는 장치마다 확인해야 합니다. MLX 3비트 변환도 31.65 GB라 24 GiB 맥에는 들어가지 않습니다.

## 6. 정리

- 메모리 표와 파라미터 수는 공개 파일과 정확히 맞습니다. 다만 "24GB"는 텍스트 전용 가중치와 상태의 크기이고, 24 GB 장치에서 돈다는 뜻은 아닙니다.
- 2비트 expert는 용량을 줄이고, 속도는 NVFP4 Mamba 입출력이 정합니다.
- 경쟁 비교는 대체로 맞지만 PinchBench 순위, MCP-Atlas 재현값, 빠진 경쟁 모델, 13–16% 느린 속도는 같이 읽어야 합니다.
- NVIDIA 하드웨어에서는 바로 쓸 수 있고, 학습 파이프라인은 재현할 수 없습니다.

분석 문서, 계산 스크립트, 리포트 PDF는 함께 공개한 저장소에 있습니다.

## 자료

- [통합 리포트 PDF](report/report_public.pdf)
- [01. 논문 분석: Nemotron 3.5 Bolt 74B-A9B 24GB (arXiv:2610.33039)](analysis/01_paper_analysis.md)
- [02. 선행연구와 계보, 공개 저장소](analysis/02_prior_work.md)
- [03. 파라미터·메모리 산술, 디코드 대역폭, 초안 검증 비용](analysis/03_analysis.md)
- [04. 공개 모델·런타임 대조, 수치 재계산, 구현 가능성](analysis/04_code_verification.md)
