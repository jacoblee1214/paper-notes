# 04. 공개 모델·런타임 대조, 수치 재계산, 구현 가능성

- **대상:** 학습 코드는 공개되지 않았습니다. 그래서 "코드"는 공개된 산출물로 대신합니다.
  - Hugging Face 설정 파일: BF16 `config.json`(`e09a4ce`), 24GB `config.json`·`hf_quant_config.json`(`cf2effc`), 초안 `config.json`(`449d241`)
  - 공개 GGUF 세 개의 헤더(본 모델 `e0fecb6`, 초안 `182d298`, Unsloth UD-IQ1_S `34841e3`)와 E4B GGUF 헤더(`1db54d5`)
  - llama.cpp 소스 `889f2ec`(2026-10-30)의 `ggml.h`, `ggml-metal-device.m`, `ggml-cpu.c`
- **범위:** 가중치는 받지 않았습니다. GGUF는 HTTP Range로 앞부분(헤더)만 받았습니다. 모델을 실행하지 않았으므로 속도·정확도는 재현하지 않았습니다.

## 1. 논문 ↔ 공개 산출물 매핑

| 논문 | 공개 산출물 | 일치 여부 |
|---|---|---|
| 40 Mamba / 40 MoE / 8 attention, hidden 4096 (Table 1) | config `block_configs` 88개, GGUF `block_count` 88 | ✅ |
| expert 수 ∈ {240, 304, 368, 432, 496}, 폭 ∈ {1536, 2048, 2304, 2560}, top-9 | config: E 분포 12/6/9/10/3층, I 분포 1/11/15/13층, top-9 | ✅ Fig. 3의 층별 순서와도 맞음 (Fig. 3 축의 1,792는 눈금일 뿐 쓰이지 않음) |
| latent 1024, shared 5376, KV 헤드 2 | config `moe_latent_size`, `moe_shared_expert_intermediate_size`, `num_key_value_heads` | ✅ |
| 74B-A9B, 부모 120B-A12B | 식 74.043B / 9.06B, GGUF 74,042,720,848 / 부모 식 120.67B / 12.23B | ✅ (`03` §1.1) |
| Table 2 형식·크기 | GGUF 텐서 843개 형식별 합(가중치 22.010 GB) | ✅ 구성 요소마다 0.01 GB 안 (`03` §1.2) |
| IQ2_XXS 2.0625 bpw, Q8_0 8.5, NVFP4 4.5 | `hf_quant_config` 27,921항목(ModelOpt 0.37.0 `MIXED_PRECISION`), GGUF bpw | ✅ |
| 148 GB BF16 → 22.1 GB, 85.1% 감소 | 74.04B × 2 B = 148.09 GB, GGUF 22.01 GB | ✅ (22.01이면 85.13%) |
| LiLiCorr: B = 8, 후보 K = 8, 헤드 2층·폭 1024·8헤드·FFN 2048, conv (2, 16), 창 1024 + sink | 초안 config·GGUF `lilicorr.*` 키, `dflash.*` 키 | ✅ |
| 초안 1.13B(공유 임베딩·LM head 제외), 0.96 GB | 초안 GGUF 1,112,828,867 파라미터, HF safetensors 0.9625 GB | ⚠️ 파라미터가 1.5% 적음 (D4) |
| 런타임: llama.cpp(GGUF), vLLM(ModelOpt) | llama.cpp PR #28717·#29807·#30506, vLLM PR #57934·#59604 모두 병합 | ✅ |

## 2. 차이와 논문에 적히지 않은 요소

| # | 내용 | 영향 |
|---|---|---|
| D1 | **SSM 캐시 정밀도:** Table 2는 BF16(0.08 GB)으로 계산했지만, config의 `mamba_ssm_cache_dtype`은 `float32`입니다(0.17 GB). | 0.09 GB 차이라 예산을 넘기지는 않지만, 0.17 GB 남는 예산(Sec. 8.2) 안에서는 무시할 수 없습니다. |
| D2 | **문맥 길이:** 설계 예산은 128K인데(Sec. 2.3), GGUF·모델 카드는 최대 256K를 광고합니다. 공식 레시피는 65K(28GB) 또는 262K(32GB)입니다. | 128K 예산 수치를 256K 사용에 그대로 적용하면 안 됩니다. |
| D3 | **"24GB"와 공식 하드웨어 요구:** 모델 카드의 권장 배포는 "RTX/DGX Spark 64 GB·128 GB 또는 RTX 32 GB+·48 GB+ GPU"이고, 레시피 이름이 28GB·32GB입니다. 24 GB GPU용 레시피는 없습니다. | 이름의 24GB는 **텍스트 전용 가중치+상태의 크기**이지, 24 GB 장치에서 돈다는 뜻이 아닙니다(`03` §5). |
| D4 | **초안 모델:** HF config의 `target_layer_ids`는 [1, 18, 35, 51, 68, 85](GGUF는 1 더한 [2, 19, 36, 52, 69, 86]) 여섯 개입니다. 논문은 "다섯 층 + 최종 hidden state"라고 썼는데, 85(86)는 88층 중 마지막이 아닙니다. 파라미터도 1.113B로 논문 1.13B와 1.5% 차이입니다. | 사소하지만 재현 시 확인할 부분입니다. |
| D5 | **GGUF의 NVFP4 = ggml type 40:** llama.cpp `889f2ec`의 `ggml_metal_supports_mul_mat_op`는 src0이 NVFP4이면 `false`를 반환하고, `GET_ROWS`도 NVFP4를 제외합니다. CPU 백엔드에는 `ggml_vec_dot_nvfp4_q8_0`(ARM 경로 포함)이 있습니다. 논문이 인용한 SSM 최적화 PR 두 개는 CUDA 전용입니다. 반면 `SSM_SCAN`·`SSM_CONV`는 Metal에서 지원되므로, 막히는 것은 NVFP4뿐입니다. Ollama 라이브러리 페이지(`nemotron-3.5-bolt`, 2026-10-30 확인)도 "Apple Silicon / Metal is not currently supported for 74B-A9B"라고 적고 있습니다(E4B는 지원). | **맥(Metal)에서는 Mamba 입출력 80개 행렬곱이 CPU로 넘어갑니다.** 디코드 바이트의 51%가 여기서 나오므로(`03` §2) 속도 손해가 클 것으로 예상합니다. 실행으로 확인하지는 않았습니다. |
| D6 | `hf_quant_config`의 `kv_cache_quant_algo`는 `None`입니다. Table 2의 FP8 KV는 런타임 설정입니다. llama.cpp 기본값은 F16(128K에서 1.07 GB)이고, 공식 28GB 레시피는 q8_0 KV를 지정합니다. | 런타임 설정을 Table 2와 같게 맞추지 않으면 KV가 두 배입니다. |
| D7 | **비전:** config의 비전 타워는 RADIO ViT-H/16(32층, hidden 1280, 최대 13,312 패치, 입력 512)이고, 모델 카드는 C-RADIO v4-H라고 씁니다. mmproj GGUF는 1.734 GB로 Table 2의 1.68 GB보다 0.05 GB 큽니다. 모델 카드는 영상 입력도 지원한다고 쓰지만, 논문의 비전 적응에는 영상 데이터가 없습니다(Sec. 3). | 영상 성능은 논문에서 검증되지 않았습니다. |
| D8 | **llama.cpp 레시피의 초안 종류:** 모델 카드 레시피는 `--spec-type draft-dflash`이고 초안 GGUF의 아키텍처도 `dflash`입니다. 그래도 GGUF에 `lilicorr.*` 텐서 26개와 `lilicorr.candidate_topk = 8`이 들어 있고, llama.cpp PR #30506("spec: add lilicorr support")이 병합되어 있습니다. | DFlash 본체 + LiLiCorr 헤드로 읽힙니다. 실제로 헤드가 쓰이는지는 실행해야 확인됩니다. |
| D9 | 압축 코드(Star Elastic 라우터), QAD 학습 스크립트, 데이터 혼합 비율, WebArena 내부 하네스가 공개되지 않았습니다. | 학습 파이프라인은 재현할 수 없습니다. |
| D10 | **동시 발표된 "Bolt E4B 4GB":** 공식 GGUF의 아키텍처는 `gemma4`이고 파라미터는 7.52B입니다. 모델 카드도 "Gemma-4-E4B-it를 후학습, 8B 전체 / 4B 유효, 64K 문맥"이라고 씁니다. 공식 GGUF(Q4_K_M)는 6.18 GB입니다. | 이름은 같은 Bolt지만 하이브리드 Mamba 계열이 아닙니다. "4GB"가 무엇을 뜻하는지 모델 카드만으로는 분명하지 않습니다. |

## 3. 수치 재계산 (`analysis/table_check.py`, `config_check.py`, `spec_decode_toy.py`)

| # | 주장 (출처) | 재계산 | 판정 |
|---|---|---|---|
| C1 | 24 GB 안에 들어간다 (Abstract, Sec. 8.2) | 가중치 22.01 + 런타임 0.87 + 초안 0.96 = 23.84 GB(텍스트 전용). 비전 포함 25.52 GB | 텍스트 전용에서만 ✅ |
| C2 | 74B-A9B, 부모 120B-A12B | 74.043B / 9.06B, 부모 120.67B / 12.23B. GGUF와 4,176개 차이 | ✅ |
| C3 | Table 2 구성 요소별 크기 | GGUF 헤더와 구성 요소마다 0.01 GB 안 | ✅ |
| C4a | 2비트 손실 | BF16 → 24GB 평균 −2.08점(14개), 11개에서 −2.4 ~ −4.9. 두 개는 상승 | 참고 |
| C4b | 같은 메모리대 모델보다 정확 (Abstract, Table 7) | Qwen3.6·Gemma4·Lightning 대비 각각 11승. AA-Omniscience accuracy는 Qwen3.6·Gemma4 모두에게 짐 | 대체로 ✅ |
| C4c | "PinchBench #1, BrowseComp #2" (블로그·연구 페이지, 같은 메모리대 기준) | Table 7 안에서 PinchBench는 Qwen3.6 BF16 88.07 > Bolt 24GB 87.82 | ⚠️ Table 7과 맞지 않음. 공개 리더보드 기준으로 보임 |
| C4d | MCP-Atlas 비교 | Qwen3.6 공식 62.8 대신 저자 재현 51.2를 써서 Bolt(52.20)가 앞섬. 공식값이면 짐 | ⚠️ 재현값 의존 |
| C5 | 84 TPS로 "더 적은 활성 파라미터의 모델과 비슷" (Abstract) | Qwen3.6 Q4 + MTP 대비 0.87배, Gemma4 Q4 + MTP 대비 0.84배. 에이전트 재생 시간은 1.25–1.32배 | "비슷"보다는 13–16% 느림 |
| C6 | Table 5 Overall = 범주 평균 | 네 열 모두 ±0.0006 안 | ✅ |
| C7 | NVFP4 초안은 수락 길이를 거의 떨어뜨리지 않음 | −1.2%(temp 0), −0.9%(temp 1) | ✅ |
| C8 | LiLiCorr가 DSpark보다 낫다 (Table 6) | AL +13.6% (4.18 vs 3.68). 본문의 "약 20% 처리량"은 장문맥 별도 실험이고 수치 표가 없음 | 부분 확인 |
| C9 | 학습 산술 (Sec. 2.3) | 8 + 64 + 48 = 120B 토큰, 배치 2048 × 8K = 128 × 128K = 16.78M 토큰, 128 GPU × 64 h = 8,192 GPU-h | ✅ |
| C10 | 압축 품질 2.85점(4.94%) (Sec. 2.4) | 57.58 − 54.74 = 2.84, 2.84/57.58 = 4.93% | ✅ (반올림) |
| C11 | QAD 1단계 → 최종 (Fig. 8 vs Table 7) | BrowseComp 69.25 → 66.75, Terminal-Bench 48.88 → 50.98, GDPval 28.99 → 28.47 | 2단계의 득실이 본문에 없음 |

## 4. 구현 가능성 판단

### 4.1 공개 모델을 그대로 쓰기 (추론)

| 경로 | 판단 | 근거 |
|---|---|---|
| NVIDIA GPU 32 GB+ 또는 RTX/DGX Spark 64 GB+, llama.cpp | **가능** (공식 지원) | 모델 카드 28GB·32GB 레시피, 필요한 llama.cpp PR이 모두 병합됨 |
| NVIDIA GPU 24 GB (예: RTX 4090) | **애매**. 공식 레시피가 없음 | 텍스트 전용·짧은 문맥·비전 없음이면 가중치 22.0 + 상태 약 0.5 + 버퍼가 24 GB에 걸침 |
| vLLM (SM12x, RTX 5090, Spark) | 가능 | b12x 커널 PR #59604 병합 |

### 4.2 이 맥 (Apple M4 Pro, 통합 메모리 24 GiB = 25.77 GB, CUDA 없음)

`config_check.py`의 계산입니다. **가정:** macOS와 기본 앱에 4 GB를 남기고, llama.cpp compute buffer를 0.5 GB로 둡니다. GPU 기본 한도(`recommendedMaxWorkingSetSize`)는 RAM의 2/3(약 17.2 GB)로 가정했고, 이 맥에서 직접 재지 않았습니다(`iogpu.wired_limit_mb = 0`, 즉 기본값).

| 파일 | 가중치 | 32K 문맥 필요량 | GPU 기본 한도 안 | RAM − 4 GB 안 | 판단 |
|---|---|---|---|---|---|
| 공식 24GB IQ2_XXS | 22.02 GB | 22.96 GB | ✗ | ✗ | **사실상 불가.** 메모리가 모자라고, Mamba NVFP4가 Metal에서 돌지 않음(D5) |
| LM Studio 커뮤니티 GGUF, Ollama `nemotron-3.5-bolt:latest` | 22.02 GB / 25 GB(비전 포함) | — | ✗ | ✗ | 공식 IQ2와 같은 파일 구성이라 같은 판단. Ollama는 Apple Silicon 미지원을 명시 |
| Unsloth UD-IQ1_M | 20.31 GB | 21.25 GB | ✗ | ✓ | 조건부. GPU 한도를 올려야 함 |
| Unsloth UD-IQ1_S | 18.51 GB | 19.45 GB | ✗ | ✓ | **조건부 가능.** NVFP4를 쓰지 않아 모든 형식이 Metal 지원 대상. `sudo sysctl iogpu.wired_limit_mb=20480` 정도로 한도를 올리고, 비전·초안 없이, 문맥 32K 이하. 1비트대 품질은 검증되지 않음 |
| 커뮤니티 REAP50 13.38 GB | 13.38 GB | 14.32 GB | ✓ | ✓ | 들어감. 비공식 추가 가지치기라 품질 미확인 |
| Bolt E4B Q4_K_M | 6.18 GB | 약 6.7 GB | ✓ | ✓ | **바로 가능.** 단, Gemma-4 계열이라 이 논문의 모델이 아님(D10) |
| MLX 3비트 (rapid-mlx) | 31.65 GB | — | ✗ | ✗ | 불가 |

**결론:**
- 이 맥에서 논문의 공식 24GB 모델을 원래 의도대로(GPU에서, 비전과 초안 포함) 돌리는 것은 **불가능합니다.**
- Unsloth IQ1_S를 GPU 한도를 올려 텍스트 전용·짧은 문맥으로 돌리는 것은 **해 볼 수 있습니다.** 다만 메모리 압박이 크고, 품질이 공식판과 다릅니다.
- 실제로 해 보려면 사용자가 확인할 것이 세 가지입니다.
  1. llama.cpp 시작 로그의 `recommendedMaxWorkingSetSize` 값
  2. 한도를 올렸을 때 macOS가 안정적인지
  3. 디스크 여유(18–22 GB)
- 계산상 대역폭 상한(M4 Pro 공칭 273 GB/s, 토큰당 약 4.8 GB 읽음)은 초안 없이 약 57 TPS입니다. 실제로는 더 낮을 것입니다.

### 4.3 학습 쪽 재현
- 압축(Star Elastic)·QAD 학습 코드가 공개되지 않았고, 규모도 GB200 128장 × 64시간입니다. **재현할 수 없습니다.**
- 공개된 부분으로 해 볼 수 있는 것은 두 가지입니다. ① Model Optimizer로 작은 모델에 PTQ·QAD를 거는 것, ② REAP식 expert 가지치기를 작은 공개 MoE에 적용하는 것. 둘 다 NVIDIA GPU가 필요합니다.

### 4.4 아이디어만 다시 구현하기
- **메모리 예산 → 파라미터 예산 역산(Sec. 8.2)**과 **텐서별 혼합 정밀도 설계**는 이 저장소의 `config_check.py`처럼 계산 몇십 줄이면 됩니다. 로봇 탑재 VLM의 메모리 계획에 그대로 쓸 수 있습니다.
- Jaccard 손실(Eq. 4)은 한 줄입니다. 효과를 보려면 GPU 학습이 필요합니다.
