# 04. 공개물 검증: 보고서 ↔ 공개 체크포인트·레시피 대조, 수치 재계산, 구현 가능성

- **대상:**
  - Hugging Face `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16` @ `fb0ee0b` (2026-10-22)와 `-NVFP4` @ `da5b265` (2026-10-23)의 `config.json`, `hf_quant_config.json`, safetensors 색인, 색인이 가리키는 64개 파일의 헤더(HTTP range 요청으로 헤더만 읽음, `analysis/fetch_hf_meta.py`), 모델 카드
  - cookbook `NVIDIA-NeMo/Nemotron` @ `99a7e45`, AutoModel 문서 `NVIDIA-NeMo/Automodel` @ `571df6d`
  - 평가 레시피 `NVIDIA-NeMo/Gym` `nemotron_recipes/super-3.5/` (레시피가 고정한 평가 커밋 `1e66890`)
- **범위:** 학습 코드 전체는 보지 않았습니다. 보고서가 학습 프레임워크(NeMo-RL, ModelOpt, Megatron)만 가리키고 이 모델의 학습 스크립트는 공개하지 않았기 때문입니다. 가중치는 받지 않았고, 모델 실행·API 호출도 하지 않았습니다.

## 1. 보고서 ↔ 공개물 매핑

| 보고서 | 공개물 | 일치 여부 |
|---|---|---|
| 120B total, 12B active | LLM 120.67B, 활성 12.23B(임베딩 제외) | ✅ (`03` §1) |
| BPE 계산의 N = 124.453B | shard 헤더 합 124.453B | ✅ 정확히 일치 |
| 512 routed expert, LatentMoE | `n_routed_experts 512`, top-22, latent 1024, expert 2688, shared 1 | ✅ top-k와 차원은 보고서에 없음(D5) |
| C-RADIO 시각 인코더 | `vision_model` ViT-H/16, 32층, 폭 1280. cookbook은 C-RADIO v4-H로 명시 | ✅ |
| MTP: 재귀 공유 헤드 1개, attention + MoE, lm_head 공유 (§4.7) | `language_model.mtp.layers.0`에 attention·MoE, 2.94B, 별도 lm_head 없음 | ✅ |
| 1M 문맥 | `max_position_embeddings 1,048,576` | ✅ |
| T12 Aggressive 레시피 | `hf_quant_config.json`: routed expert up/down 각 20,480개와 shared expert가 NVFP4. Mamba in/out, attention QKVO, lm_head가 FP8. 임베딩, conv1d, latent proj, 라우터, MTP attention, 시각·projector는 제외 | ✅ 모듈 단위로 정확히 일치 |
| MTP expert NVFP4, MTP attention BF16 (§4.7) | MTP experts·shared expert NVFP4, MTP q/k/v/o 제외 | ✅ |
| KV cache FP8 | `kv_cache_quant_algo: FP8` | ✅ |
| Mamba SSM cache FP16 + SR | config `mamba_ssm_cache_dtype: float32` | ⚠️ D1 |
| 권장 샘플링: 텍스트 temp 1.0, top_p 0.95 | `generation_config.json` temp 1.0, top_p 0.95 | ✅ (시각용 0.6은 카드에만 있음) |
| 평가 재현 레시피 (§3.8.1 각주 14) | Gym `super-3.5/`에 벤치마크 설정 22개, `reproducibility.md` | ✅ 시각 8개와 NV Chat은 없음. 보고서도 "아직 공개 도구에 없음"이라고 밝힘 |

## 2. 불일치와 보고서에 적히지 않은 요소

| # | 내용 | 영향 |
|---|---|---|
| D1 | **SSM 상태 정밀도가 세 가지입니다.** 보고서 평가와 서빙은 FP16 + stochastic rounding입니다(T12, §4.5, §5.6). 공개 NVFP4 `config.json`의 기본값은 `float32`입니다. 모델 카드의 vLLM 명령은 `--mamba-ssm-cache-dtype float16 --enable-mamba-cache-stochastic-rounding`으로 덮어쓰고, SGLang 명령은 `--mamba-ssm-dtype bfloat16`(+SR)을 씁니다. | 플래그 없이 띄우면 요청당 상태가 160 MiB로 두 배가 되어 동시 처리 수가 줄어듭니다. SGLang의 BF16 상태(가수 8비트)는 보고서가 검증한 FP16(가수 10비트)과 다르므로, SGLang 경로의 정확도는 보고서 수치로 보증되지 않습니다 |
| D2 | 모델 카드는 SWE 중간학습을 **67B 토큰**, 보고서는 **65.4B 토큰**이라고 적었습니다(§1, §2.5). 혼합 비율(T4)은 같습니다. 카드 벤치마크 표의 CritPt는 Qwen3.8·GLM-5.3이 "—"인데 T10에는 15.70, 9.70이 있습니다. | 사소하지만 카드와 보고서 중 어느 쪽이 최종인지 알 수 없습니다 |
| D3 | AutoModel 문서는 "120.7B 학습 가능(시각 동결, 99.31%), bf16 약 227 GB, 63 shard"라고 적었습니다. 실제 저장소는 전체 124.45B, 248.9 GB이고 색인이 가리키는 파일은 64개입니다. | 문서 수치는 **MTP(2.94B)를 뺀 모델 기준**으로 계산하면 맞습니다(120.67 / 121.51 = 99.31%, 226.3 GiB). 64번째 파일은 RADIO의 `vision_model.summary_idxs`(int64 2개) 하나만 담은 `model-summary-idxs.safetensors`라, 가중치 shard는 문서대로 63개입니다. 디스크 계획에는 248.9 GB를 써야 합니다 |
| D4 | NVFP4 색인의 `total_size`는 75.010 GB, 보고서의 S는 74.987 GB입니다. | 0.03% 차이라 BPE는 둘 다 4.82입니다. 색인 메타데이터나 scale 텐서 집계 방식 차이로 보입니다 |
| D5 | **보고서에 구조 표가 없습니다.** 층 패턴, top-22, KV 헤드 2, Mamba 차원, 어휘 131,072는 config에서만 확인됩니다. | 재구현·메모리 계획에는 config가 필수입니다. Super 3 보고서(2604.12374)에 있을 가능성이 있지만 확인하지 않았습니다 |
| D6 | **복원율 수치가 기준 집합마다 다릅니다.** T23은 37행이고 전체 중앙값이 99.70%입니다. 초록·§4.1의 99.62%는 "36개 벤치마크" 기준입니다. §4.5(T18)와 결론(§6)은 같은 공개 구성(FP8 KV)을 99.52%(35개)로 적었습니다. | 숫자 자체보다 꼬리(93.6%, 94.2%, 95.9%)가 중요합니다(`03` §5) |
| D7 | **공개 체크포인트의 KV 구성은 QAD가 겨냥한 구성이 아닙니다.** QAD 학생은 NVFP4 KV용 고정 scale(1/6)로 학습했고, 공개본은 KV scale 없이 FP8 KV를 씁니다(vLLM은 scale 1.0을 기본으로 씀). | 보고서가 직접 밝힌 사실입니다(§4.5). NVFP4 KV를 쓰려면 원래 QAD 체크포인트가 필요한데, 공개되지 않았습니다 |
| D8 | MTP는 따로 양자화한 뒤 본체에 이어 붙였습니다(splice). 텐서 이름·모양·dtype이 정확히 같아야 합니다. | 보고서가 밝힌 사실입니다(§4.7). 사용자가 본체만 다시 양자화하면 MTP를 같은 방식으로 다시 붙여야 합니다 |
| D9 | **블로그에만 있는 주장이 있습니다**(C10): 5배 처리량, Dynamo 8배 사용자·TTFT 85% 감소, LiLiCorr 2.3배, Baseten 600 TPS, Hermes Bench·LangChain Evals 점수, CharXiv Nano Omni 63%와 Python 도구 84%, 사후학습 약 3.8M 샘플. | 보고서 수치(C1, C9)는 블로그보다 보수적입니다. 3.8M 샘플은 보고서의 오염 제거 비율로 역산한 약 23M과 맞지 않습니다 |
| D10 | ReplaySSM 재구성 식의 곱 범위가 본문에서는 ∏_{j>i+1}이지만, 같은 문단의 갱신식에서 유도하면 ∏_{j>i}입니다. | 표기 오류로 보입니다. 구현은 TensorRT-LLM PR #14203에 있어 식 자체의 영향은 없습니다(`03` §3.1) |
| D11 | **SWE-bench Pro 평가 프로토콜이 둘입니다.** T10(51.78%)은 Gym 레시피로 측정했습니다(OpenCode v1.17.11, 3회, `apply_anti_cheating=True`로 HEAD 조상 밖의 refs·reflog 삭제, 프롬프트에 부정행위 금지 규칙). F8은 하네스 기본 프롬프트, Git 기록 제거, OpenCode v1.17.8, 2회이고 41.04%입니다. | 같은 하네스 이름으로 10.7%p 차이가 나는데 보고서는 설명하지 않습니다. 하네스 기본 설정에서는 6개 하네스 모두 Ultra의 T10 값(49.66%)보다 낮거나 같습니다 |
| D12 | T6에서 Super Base와 Lightning Base의 GSM8K(91.74)와 PIQA(85.09)가 소수 둘째 자리까지 같습니다. | 복사 오류일 가능성이 있지만 검증할 방법이 없습니다 |

## 3. 수치 재계산 (`analysis/table_check.py`, `param_check.py`, `mechanism_toy.py`)

| # | 주장 (출처) | 재계산 | 판정 |
|---|---|---|---|
| C1 | SWE-bench Pro 51.78 > Ultra 49.66, 활성 약 4.6배 적음 (초록) | +2.12%p, 55/12 = 4.58배 | ✅ (단 D11) |
| C1′ | MTP boosting AL +21.8% (초록, T8) | greedy +10.05%, sampling +21.79% | ✅ |
| C1″ | NVFP4 2.43배, Qwen3.5 2.34배, Qwen3.8 1.90배 (초록, F10–F11) | GB200 2.432 / 2.336 / 1.898. H100에서는 1.858 / 2.327 / **1.313** | ✅ GB200 기준. H100에서 Qwen3.8 대비는 1.31배뿐 |
| C1‴ | 4,856 tok/s/GPU (초록) | T24 최고점(동시 64, MTP 5, ReplaySSM 켬). F10은 4,324, F11은 4,331.88 | ✅ 설정이 다른 세 값 |
| C2 | Indic 최대 +9.82%p (T2), 2.83T 토큰 (T1) | T2 변화량 5개 모두 일치, T1 소계(2,273.8 / 224.6 / 327.2 = 2,825.6B) 일치, 번역 비용 −59.9%(2.49배) | ✅ |
| C3 | 중간학습: 다른 하네스 평균 SWE-V +8.6, TB +6.8 (§2.5) | 8.56, 6.82. RULER-1M 79.9 → 33.5 → 81.0 | ✅ |
| C4 | 테스트 시간 선택 +9.2 / +9.9 / +8.3%p (§3.11) | 일치 | ✅ |
| C5 | 오염 제거: 20,671개(약 0.09%), 30개 중 22개 벤치마크의 1,808문항(6.0%) | SFT 14,793 + RL 5,878 = 20,671. 0.085–0.095%로 역산하면 사후학습 표본 21.8–24.3M. 1,808 / 6.0% ≈ 30,133문항 | ✅ 내부 일관. 블로그 3.8M과 불일치(D9) |
| C6 | Super가 Ultra를 넘음 (초록·결론) | T10 공통 27개 중 17승 10패, 평균 순위 4.24/7. 1위는 GraphWalks 하나. SWE-V·Tau2 Telecom·AA-LCR·MMLU-ProX는 7개 중 7위 | ⚠️ 부분적 |
| C7 | Text Arena (T11) | Super NVFP4 1420 ± 12의 구간이 GLM-5.3을 제외한 모든 모델과 겹침 | 구분 불가 |
| C8 | NVFP4 중앙값 복원율 99.62%, 길이 +0.44% (§4.1) | 37행 전체 99.70%, 길이 +0.43%. 36개 부분집합이면 99.62%. 같은 구성의 다른 값 99.52%(§4.5, §6). Lightning 99.25%(보고 99.24%), 길이 −3.63% | ✅ (D6) |
| C8′ | BPE 4.82 = 8S/N (§4.1), NVFP4 KV 저장량 −43.75% (§4.5) | 8 × 74.987 / 124.453 = 4.820, 색인 기준 4.822. 1 − 4.5/8 = 43.75% | ✅ |
| C9 | ReplaySSM 이득 4.7–6.4% / 6.4–12.5% (T24), 분리 서빙 +10.3% / +61% (T25) | T24 다섯 행 재계산 범위 일치. T25 +10.3%, +61.4%, TTFT 2.29배 증가 | ✅ |
| C10 | 블로그 주장 | 26T 토큰 = Super 3의 25T + 3.5 단계 1.209T(재구성 26.21T) ✅. F7 기준 GPU-h 4.5–4.9배 적음 ✅이지만 정확도 5–9%p 낮음. 5배·8배·2.3배·600 TPS·3.8M은 보고서에 없음 | 부분 |
| C11 | SWE-bench Pro 하네스 (F8 vs T10) | OpenCode 41.04 vs 51.78 (10.74%p). F8 평균 46.54, 최고 49.45 < Ultra 49.66 | ⚠️ D11 |
| V1 | Mamba 페이지 ≈ 4K 토큰 KV (§5.3) | 2 MiB / 512 B = **4,096** | ✅ |
| V2 | ReplaySSM: 트래픽 T+1 → 약 1+1/L, 양자화 오차 감소 (§5.4) | T = 5, L = 16에서 6.00 → 1.18 (5.1배). FP16 반올림 오차 3.5–4배 감소 | ✅ (SR 이점은 toy에서 안 보임) |
| V3 | proximal teacher 하한 log(1−α) ≈ −0.22 (§3.6) | toy 최솟값 −0.22314 = log 0.8 | ✅ |

## 4. 구현 가능성 판단 ("이 맥에서 돌릴 수 있는가, 고칠 수 있는가")

이 맥: Apple M4 Pro, 통합 메모리 24 GiB(GPU가 쓸 수 있는 양은 Metal 기본 한도인 약 18 GiB로 가정), 메모리 대역폭 273 GB/s, MPS 있음, **CUDA 없음**.

### 4.1 공개 모델로 추론하기

| 경로 | 요구 사항 | 이 맥 |
|---|---|---|
| BF16 (공식, vLLM·SGLang) | 4× H100 또는 2× GB200 (카드). 가중치 231.8 GiB | ❌ |
| NVFP4 (공식) | W4A4는 Blackwell, 그 밖에는 W4A16. 1× GB200, 1× H200, 2× H100, 1× RTX PRO 6000(96 GB), 1× DGX Spark(128 GB, LiLiCorr 권장). 가중치 69.9 GiB. vLLM·SGLang·TRT-LLM은 모두 CUDA 필요 | ❌ (CUDA 없음, 메모리 부족) |
| 커뮤니티 GGUF (llama.cpp, Metal 지원) | Q4_0 67.0 GiB, Q4_K_M 65.3 GiB, Q8_0 119.7 GiB. 시각은 mmproj 별도 | ❌ 메모리 부족 |
| 커뮤니티 MLX | 4비트 64.6 GiB, 3비트 39.8 GiB | ❌ 메모리 부족 |
| 이론적 최소 | 18 GiB에 가중치만 넣으려면 파라미터당 1.24비트 | ❌ 품질을 기대할 수 없는 수준 |
| **같은 세대 Lightning 30B-A3B (텍스트 전용)** | GGUF Q4_0 17.6 GiB | △ 한도에 겨우 들어감. 긴 문맥 불가. llama.cpp 또는 mlx-lm 설치 필요(하지 않음) |

**판단:** Super VL은 **이 맥에서 어떤 공개 형식으로도 실행할 수 없습니다.** 메모리가 부족한 것이 결정적이고, 공식 서빙 스택(vLLM, SGLang, TRT-LLM)과 NVFP4 커널은 CUDA 전용입니다. 메모리가 충분한 Apple Silicon(64 GB 이상, 3–4비트 양자화)이라도 대역폭 계산상 배치 1 decode는 수십 tok/s가 상한입니다(`03` §1.2). 이 맥에서 Nemotron 3.5를 직접 만져 보려면 Lightning 30B-A3B 4비트가 현실적인 유일한 선택지입니다. 로컬 실행을 목표로 만든 Bolt 24GB 계열은 별도 분석 대상입니다. NVIDIA 호스팅 API(build.nvidia.com)는 이번에 호출하지 않았습니다.

### 4.2 미세조정

| 방법 | 공개 레시피의 요구 사항 | 이 맥 |
|---|---|---|
| 전체 SFT (AutoModel) | 4노드 × 8 H100 80 GB, GPU당 약 65 GiB, 400 step 약 21분(문서 기준) | ❌ |
| LoRA (AutoModel) | 1노드 × 8 H100, GPU당 약 39 GiB (rank 64, 시각 동결) | ❌ |
| LoRA (Megatron-Bridge, finance cookbook) | 8× H200 또는 8× H100, 디스크 약 800 GB | ❌ |
| 멀티모달 GRPO (NeMo RL + Gym cookbook) | 다중 노드 | ❌ |

**판단:** 미세조정은 **최소 8× H100급 노드 하나(LoRA)**가 필요합니다. 공개 레시피가 잘 갖춰져 있어 GPU 노드만 있으면 진입 장벽은 낮습니다. 이 맥에서는 불가능합니다.

### 4.3 보고서 학습 과정 재현

1T 토큰 증류, GB200 128노드의 MOPD, GB200 512장의 QAD가 들어갑니다. 교사 체크포인트(에이전트, 추론, 비디오, 정렬)와 MOPD 전 체크포인트도 공개되지 않았습니다. **학습 재현은 사실상 불가능합니다.** 사후학습 데이터 일부(CC BY 4.0)와 RL 환경(Gym)은 공개되어 있어, 레시피를 작은 모델로 옮겨 보는 것은 가능합니다.

### 4.4 평가 재현

Gym 레시피로 T10의 텍스트·에이전트 벤치마크 22종을 재현할 수 있습니다. 조건은 다음과 같습니다. 모델을 서빙하는 GPU 4장(GB200 4장에서 검증, 80 GB GPU는 메모리 설정 변경 필요), SWE 과제마다 CPU 4개·RAM 30 GB인 Docker 샌드박스(동시 16개), judge 모델 API(GPT-5.6 Luna 등), Exa 검색 API(BrowseComp)입니다. 시각 벤치마크 8종과 NV Chat Benchmark는 레시피가 없습니다.

### 4.5 CPU로 한 것과 더 할 수 있는 것

| 단계 | 상태 | 근거 |
|---|---|---|
| 구조·파라미터·메모리 계산 (헤더만 읽음) | ✅ 완료 | `param_check.py` |
| 표·본문·블로그 수치 재계산 | ✅ 완료 | `table_check.py` |
| ReplaySSM, MOPD 하한 toy | ✅ 완료 | `mechanism_toy.py` |
| shard 하나(약 4 GB)를 받아 expert 가중치의 NVFP4 양자화 오차 분포 측정 | 가능 (다운로드 필요, 하지 않음) | 블록 scale MSE vs Max(T13)를 가중치 수준에서 확인할 수 있음 |
| ReplaySSM을 PyTorch(MPS/CPU)로 구현해 Mamba-2 실제 decode와 대조 | 가능 | 공개 config 차원, 무작위 가중치로 충분 |
| 작은 모델(0.5–1B)로 MOPD 2단계 레시피 축소 실험 | MPS로 가능할 수 있음 (설치 필요) | 교사 2–3개를 같은 작은 모델의 과제별 미세조정으로 대체 |
