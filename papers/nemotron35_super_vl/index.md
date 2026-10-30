---
title: "Nemotron 3.5 Super VL 기술 보고서 읽기: 120B-A12B 하이브리드 MoE를 공개 체크포인트로 검증하기"
date: 2026-10-30
tags: [llm, vlm, mixture-of-experts, mamba, quantization, nvfp4, speculative-decoding, tech-report-review]
paper: https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3.5-Super-and-Lightning-Technical-Report.pdf
code: https://github.com/NVIDIA-NeMo/Nemotron
summary: "NVIDIA의 Nemotron 3.5 Super VL·Lightning 기술 보고서를, 공개 config와 safetensors 헤더로 구조를 확인하고 표·블로그 수치를 재계산하며 읽었습니다."
---

> **보고서:** NVIDIA. *Nemotron 3.5 Super and Lightning: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Models for Agentic AI.* 기술 보고서, 2026-10-27 (arXiv 없음, 79쪽).
> 보고서는 "© 2026 NVIDIA. All rights reserved."라서 보고서 그림은 싣지 않았습니다. 이 글의 그래프는 보고서 표의 수치, 공개 `config.json`, safetensors 헤더로 직접 그렸습니다. 모델 가중치는 받지 않았고 모델을 실행하지도 않았습니다.

## 한 줄 요약

Nemotron 3 Super(120B-A12B, 텍스트)를 출발점으로 **Ultra 3에서 1T 토큰 증류 → Indic 계속 사전학습 → 1M 문맥 → 256K 문맥 SWE 중간학습**을 거친 뒤, 사후학습에서 시각을 붙이고 전문 교사 여러 개를 **MOPD(multi-teacher on-policy distillation)**로 합친 멀티모달 에이전트 모델입니다. NVFP4 양자화와 vLLM 서빙 최적화까지 한 보고서에 담았습니다. 제목의 "Lightning"은 같은 세대의 30B-A3B 텍스트 전용 소형 모델입니다.

새로운 모델 구조를 제안하는 논문이라기보다 **레시피와 시스템 보고서**입니다. 그래서 이 글은 "보고서의 숫자가 공개물과 맞는가"를 중심으로 읽었습니다.

## 1. 구조는 config로만 확인된다

보고서 본문에는 층 구성을 정리한 표가 없습니다. 공개 `config.json`과 safetensors 파일 64개의 헤더(텐서 이름과 모양만 HTTP range 요청으로 읽음)에서 정리하면 다음과 같습니다.

| 항목 | 값 |
|---|---|
| 층 | 88층 = Mamba-2 40 + LatentMoE 40 + attention 8 |
| MoE | hidden 4096 → latent 1024, routed expert 512개 중 top-22, shared expert 1개 |
| Attention | query 32헤드, **KV 2헤드**, head_dim 128 |
| 시각 | C-RADIO v4-H (ViT-H/16, 32층) |
| 전체 | **124.45B** (보고서 BPE 식의 N = 124.453B와 정확히 일치) |
| 토큰당 활성 | **12.23B** (임베딩 제외). 이름의 "A12B"와 맞음 |

흥미로운 점은 활성 가중치의 구성입니다. 전체 파라미터의 90%는 routed expert지만, 토큰 하나가 지나가는 가중치에서는 **Mamba projection(4.38B, 35.8%)이 routed expert(4.84B, 39.6%)만큼 크고**, attention은 2.3%뿐입니다. NVFP4가 적용되는 expert는 활성 가중치의 54%이고 나머지는 FP8입니다.

배치 1 decode에서는 토큰마다 약 9.76 GB를 읽어야 합니다. 대역폭 273 GB/s인 DGX Spark에서는 투기적 디코딩 없이 약 28 tok/s가 상한입니다. 모델 카드가 DGX Spark에 병렬 drafter(LiLiCorr)를 항상 쓰라고 권하는 이유가 이 계산으로 설명됩니다.

## 2. KV는 문맥과 함께 늘고, Mamba 상태는 고정이다

![요청당 상태 메모리](images/state_memory.png)

attention이 8층이고 KV 헤드가 2개뿐이라 KV는 토큰당 4,096 B(FP8)입니다. 1M 토큰에서도 4 GiB입니다. Mamba 상태는 요청당 80 MiB(FP16)로 고정이고, **문맥이 약 2만 토큰보다 짧으면 상태가 KV보다 큽니다.**

보고서는 vLLM이 attention과 Mamba의 캐시 페이지 크기를 바이트로 맞추기 때문에 "Mamba 한 페이지가 약 4K 토큰의 KV"라고 적었습니다. config로 계산하면 한 층 상태 2 MiB ÷ 한 층 KV 512 B/토큰 = 정확히 **4,096 토큰**입니다. prefix caching이 4K 토큰 간격으로만 Mamba 상태를 저장하는 이유이고, 보고서가 공유 prefix 경계 근처에 체크포인트를 하나 더 두도록 vLLM을 고친 배경입니다.

## 3. "Ultra를 넘는다"는 어디까지 맞나

![T10에서 Super 3.5 − Ultra 3](images/t10_vs_ultra.png)

초록은 SWE-bench Pro 51.78%로 Nemotron 3 Ultra(49.66%, 활성 55B)를 넘었다고 강조합니다. T10의 27개 공통 벤치마크를 모두 비교하면 **17승 10패**이고, 7개 모델 중 1위는 GraphWalks 하나입니다. 크게 앞서는 곳(GraphWalks +26.4, NV Chat +18.4, Apex +13.4, BrowseComp +11)은 3.5에서 새로 넣은 데이터와 교사(그래프 장문맥, 정렬 교사, 추론 교사, 검색 에이전트)와 바로 연결됩니다. 크게 뒤지는 곳(IFBench −12.1, Tau2 Telecom −9.8, AA-LCR −6.6, MMLU-ProX −6.2)은 지시 따르기, 대화형 도구 사용, 장문맥, 다국어처럼 용량이 필요한 영역입니다.

헤드라인 SWE-bench Pro 수치는 프로토콜에도 민감합니다. 같은 OpenCode 하네스인데 T10은 51.78%, 하네스 6개를 비교한 Fig. 8은 **41.04%**입니다. Fig. 8의 6개 하네스 최고값(Claude Code 49.45%)도 Ultra의 T10 값보다 낮습니다. NeMo Gym 평가 레시피를 확인해 보면, T10 측정에도 HEAD 조상 밖의 Git 기록을 지우는 설정과 부정행위 금지 문구가 들어 있습니다. 그래서 "미래 커밋 유출"로는 설명되지 않습니다. 남는 차이는 하네스 프롬프트, OpenCode 버전(v1.17.11 vs v1.17.8), 조상 기록 제거 여부, 반복 횟수이고, 보고서는 이 차이를 설명하지 않습니다.

## 4. NVFP4: 중앙값보다 꼬리

![T23 복원율](images/t23_recovery.png)

공개 NVFP4 체크포인트의 양자화 레시피(expert NVFP4, Mamba·attention·lm_head FP8, 시각·임베딩·latent BF16, BPE 4.82)는 `hf_quant_config.json`과 **모듈 단위로 정확히 일치**합니다.

복원율은 숫자가 셋입니다. 초록의 99.62%는 "36개 벤치마크" 기준이고, T23 37행 전체의 중앙값은 99.70%, §4.5와 결론은 같은 구성을 99.52%(35개)로 적었습니다. 숫자보다 중요한 것은 꼬리입니다. SciCode problem 93.6%, Tau3 Banking 94.2%, MMLU-ProX 95.9%로 긴 다단계 추론과 다국어에서 4–6% 떨어집니다. 보고서 부록은 그 원인을 **expert 가중치의 FP4 양자화가 멈출지 계속할지의 경계를 흔드는 것**으로 분석합니다. MOPD 전 모델은 tau3-banking rollout의 19.7%가 끝없이 반복했고, MOPD 뒤에는 776개 중 1개였습니다.

공개물과 어긋나는 곳도 하나 있습니다. 보고서가 검증한 Mamba 상태 정밀도는 FP16 + stochastic rounding인데, 공개 config의 기본값은 float32입니다. 모델 카드의 vLLM 명령은 FP16으로 덮어쓰지만 SGLang 명령은 BF16을 씁니다.

## 5. ReplaySSM과 MOPD를 numpy로 확인하기

**ReplaySSM.** 투기적 디코딩에서는 몇 토큰이 채택될지 모르므로 후보 위치마다 Mamba 상태를 써야 합니다. ReplaySSM은 체크포인트 상태와 이후 입력만 버퍼에 두고 필요할 때 상태를 재구성합니다. 공개 config 크기로 세면 draft 5토큰에서 상태 트래픽이 약 5배 줄지만, 측정된 처리량 이득은 4.7–6.4%입니다. 해당 운영점에서 상태 쓰기는 단계 시간의 작은 몫이라는 뜻으로 읽힙니다. FP16 상태를 16토큰마다 한 번만 반올림하면 오차가 3.5–4배 줄어 "덜 자주 양자화한다"는 주장과 맞습니다. 다만 iid 입력 toy에서는 stochastic rounding이 반올림보다 오히려 오차가 컸고, 보고서에도 둘을 직접 비교한 결과는 없습니다. 본문의 재구성 식은 곱 범위를 j > i+1로 적었는데, 같은 문단의 갱신식에서 유도하면 j > i입니다.

**MOPD.** 학생 rollout의 각 토큰에서 Â = log π_T − log π_S를 advantage로 씁니다. 학생 분포에서 본 reverse KL의 토큰 분해입니다. 2단계에서는 교사를 0.2·π_T + 0.8·π_S로 바꾸므로, advantage가 아래에서 log 0.8 = −0.223으로 막힙니다. 무작위 분포 200쌍으로 확인한 최솟값도 −0.22314였습니다.

## 6. 어디서 돌릴 수 있나

![공개 양자화 크기와 메모리](images/memory_fit.png)

| 용도 | 요구 사항 |
|---|---|
| NVFP4 추론 | GB200·H200·RTX PRO 6000·DGX Spark 1장, 또는 H100 2장. vLLM·SGLang·TRT-LLM 모두 CUDA |
| BF16 추론 | H100 4장 또는 GB200 2장 |
| LoRA | 8× H100 노드 하나(GPU당 약 39 GiB) |
| 전체 SFT | 4노드 × 8 H100 |

24 GiB 통합 메모리의 Apple Silicon 노트북에서는 가장 작은 공개 양자화(MLX 3비트, 39.8 GiB)도 들어가지 않습니다. 같은 세대 Lightning 30B-A3B의 4비트 GGUF(17.6 GiB)만 빠듯하게 들어갑니다. 미세조정과 평가 레시피(NeMo AutoModel, NeMo Gym)는 잘 공개되어 있어서, GPU 노드만 있으면 진입 장벽은 낮습니다.

## 7. 블로그에만 있는 주장

발표 블로그의 "5배 처리량", "Dynamo로 8배 사용자·TTFT 85% 감소", "LiLiCorr 2.3배", "사후학습 약 3.8M 샘플"은 보고서에서 확인되지 않습니다. 보고서의 수치(Qwen3.5 대비 2.34배, Qwen3.8 대비 1.90배, GB200 기준)는 블로그보다 보수적입니다. 3.8M 샘플은 보고서의 오염 제거 수치(20,671개 = 약 0.09%)로 역산한 약 23M과 맞지 않습니다.

## 정리

- 공개 체크포인트는 보고서의 구조·양자화 설명과 거의 정확히 맞습니다. 어긋나는 곳은 SSM 상태 정밀도와 몇몇 숫자의 기준 집합입니다.
- 성능 주장은 "활성 12B가 55B를 다수 벤치마크에서 이긴다"까지는 맞습니다. SWE-bench Pro 헤드라인은 프로토콜에 따라 10%p가 움직입니다.
- 양자화 결과는 중앙값과 함께 꼬리를 봐야 합니다.

분석 문서, 재계산 스크립트(`param_check.py`, `table_check.py`, `mechanism_toy.py`), 게시용 리포트 PDF는 함께 공개한 분석 노트 저장소에 있습니다.

## 자료

- [통합 리포트 PDF](report/report_public.pdf)
- [01. 보고서 분석: Nemotron 3.5 Super (VL)와 Lightning 기술 보고서](analysis/01_paper_analysis.md)
- [02. 선행연구, 공개 자료, 이 저장소의 다른 논문과의 관계](analysis/02_prior_work.md)
- [03. 수치 해부와 메커니즘 확인](analysis/03_analysis.md)
- [04. 공개물 검증: 보고서 ↔ 공개 체크포인트·레시피 대조, 수치 재계산, 구현 가능성](analysis/04_code_verification.md)
