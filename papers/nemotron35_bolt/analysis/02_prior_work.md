# 02. 선행연구와 계보, 공개 저장소

번호 대신 논문 참고문헌의 저자·연도를 씁니다. Star Elastic, Nemotron Elastic, Puzzle 75B-A9B, LiLiCorr는 arXiv abs 제목을 직접 확인했습니다(2026-10-30). 나머지는 본문의 참고문헌을 옮긴 것입니다.

## 1. 계보

```
[하이브리드 Mamba-Transformer]  Nemotron-H (arXiv:2504.03624) → Nemotron Nano 2 (2508.14444)
        → Nemotron 3 (2512.20856): Nano 30B-A3B / Super 120B-A12B (2604.12374) / Ultra (2606.15007)
        → Nemotron 3.5 Super VL 120B-A12B (기술 보고서 PDF, arXiv 없음)  ← Bolt의 부모
[압축]  Minitron/Puzzle 계열 → Nemotron Elastic (2511.16664) → Star Elastic (2605.07182)
        Nemotron-Labs-3-Puzzle-75B-A9B (2607.04371): 같은 75B-A9B 급 하이브리드 MoE 압축
[expert 가지치기]  REAP (Lasby et al., ICLR 2026): MoE expert 중요도 점수
[저비트]  ParetoQ (2502.02631), Quest (2502.05003): "큰 모델 저비트 > 작은 모델 고비트"
         QAD (Xin et al., 2026): NVFP4 정확도 회복용 양자화 인지 증류 → 이 논문에서 2비트로 확장
[병렬 초안]  DFlash (Chen et al., ICML 2026) → DSpark (2607.05147) → LiLiCorr (2608.20530) / DFlash 2 (Inco AI)
          ★ Nemotron 3.5 Bolt 74B-A9B 24GB
```

**차별점:** 압축(Star Elastic), 2비트 QAD, 병렬 초안을 **하나의 배포 메모리 예산**에 맞춰 묶었다는 점입니다. 개별 기법은 대부분 같은 그룹의 선행 작업이고, 이 보고서의 새로움은 2비트 QAD 확장(Jaccard 손실, 2단계 교사)과 예산 역산(Sec. 8.2)입니다.

## 2. 구성 요소별 출처

| 역할 | 출처 | 이 논문에서의 쓰임 |
|---|---|---|
| 부모 모델 | Nemotron 3.5 Super VL 120B-A12B | 압축 대상, KD 교사, 비전 타워 |
| 구조 탐색 + 압축 | Star Elastic (Taghibakhshi et al., arXiv:2605.07182) | 예산 조건부 라우터, Gumbel-softmax 접두 마스크(Eq. 1–3) |
| expert 정렬 | REAP (Lasby et al., ICLR 2026) | routed expert 중요도 순서 |
| 2비트 형식 | llama.cpp IQ2_XXS (ggml-org 텐서 인코딩) | expert 가중치 2.0625 bpw |
| 4비트 형식 | NVFP4 (E4M3 스케일, 16원소 블록) | Mamba 입출력 projection |
| 양자화 인지 증류 | QAD (Xin et al., 2026), NVIDIA Model Optimizer | Jaccard 손실로 2비트 확장 |
| 병렬 초안 | DFlash, DSpark, LiLiCorr (Rusanovsky et al., arXiv:2608.20530) | 1.13B 초안, 상관 헤드 |
| 비전 인코더 | C-RADIO v4-H (config: RADIO ViT-H/16, 32층, hidden 1280) | 부모의 비전 타워를 재부착 |
| 서빙 | llama.cpp(GGUF), vLLM(+vllm-gguf-plugin, b12x 커널) | 로컬·고처리량 두 경로 |
| 평가 | SPEED-Bench, AA-AgentPerf Local, MCP-Atlas, WebArena 등 | Table 5–8 |

## 3. 이 저장소의 다른 논문과의 관계

솔직하게 말하면 직접 연결은 약합니다. 이 논문은 로봇 학습이 아니라 로컬 LLM/VLM 배포 논문입니다. 연결되는 지점은 다음 정도입니다.

| 논문 | 관계 |
|---|---|
| **ThinkWorld** (`thinkworld/`), **AW-0** (`agentic_wam/`) | 둘 다 VLM 백본(Qwen3-VL 계열 등) 위에 행동 헤드를 얹은 VLA입니다. Bolt 같은 압축·저비트 VLM은 **로봇 탑재 추론의 백본 후보**가 될 수 있지만, 이 논문은 로봇 과제나 연속 행동 출력을 전혀 평가하지 않았습니다. |
| **PaceVLA** (`pacevla/`) | PaceVLA는 행동 쪽 실행 빈도를 줄여 처리량을 올렸고, Bolt는 언어 쪽 디코드 속도를 초안 모델로 올립니다. 둘 다 "실패를 포함한 처리량"과 "토큰당 지연"을 따로 봐야 한다는 점에서 평가 설계 교훈이 겹칩니다. |
| **Hydra-0** (`hydra0/`) | 대규모 생성 모델을 실제 배포 예산에 맞추는 문제의식만 공유합니다. |
| **DexPolicy, FlashDexRetarget 등 RL 논문** | 공통점은 거의 없습니다. 다만 Bolt의 SWE-RL·MCP RL은 GRPO 계열이라, `dexpolicy/`에서 본 그룹 advantage 설계 문제와 같은 질문을 던질 수 있습니다(이 논문은 advantage 식을 공개하지 않음). |

## 4. 공개 저장소와 모델 (2026-10-30 확인)

**Hugging Face (NVIDIA 공식)**

| 저장소 | 무엇인가 | 라이선스 | 다운로드 | 마지막 수정 | 확인 revision |
|---|---|---|---|---|---|
| nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF | IQ2 GGUF 22.02 GB + mmproj 1.73 GB | OpenMDW-1.1 | 178,518 | 2026-10-15 | `e0fecb6` |
| nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB | 같은 IQ2를 HF 형식(ModelOpt `MIXED_PRECISION`, 90 shard, 24.12 GB) | OpenMDW-1.1 | 12,511 | 2026-10-20 | `cf2effc` |
| nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16 | BF16 (149.8 GB, 비전 포함) | OpenMDW-1.1 | 6,520 | 2026-10-23 | `e09a4ce` |
| nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr | 초안 모델 HF (NVFP4/BF16, 0.96 GB) | OpenMDW-1.1 | 4,829 | 2026-10-19 | `449d241` |
| nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr-GGUF | 초안 모델 GGUF (0.998 GB, 아키텍처 `dflash`) | OpenMDW-1.1 | 21,949 | 2026-10-15 | `182d298` |
| nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB | 동시 발표된 소형 모델. Gemma-4-E4B 후학습, GGUF Q4_K_M 6.18 GB | OpenMDW-1.1 | 64,263 | 2026-10-23 | `1db54d5` |

**커뮤니티 변환(이 맥 판단에 쓴 것)**

| 저장소 | 무엇인가 | 크기 |
|---|---|---|
| unsloth/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-GGUF (`34841e3`) | UD-IQ1_S·IQ1_M·IQ2_XXS…Q8_0, NVFP4를 쓰지 않는 혼합 양자화 | 18.51 / 20.31 / 22.02 GB … |
| rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-3bit (`7816420`) | MLX 3비트 | 31.65 GB |
| anik-jha/Nemotron-3.5-Bolt-74B-A9B-16GB-GGUF (`e24fd75`) | REAP로 expert를 더 잘라낸 비공식 변형 | 13.38 GB(REAP50), 9.06 GB(REAP25) |
| lmstudio-community/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-GGUF (`2116154`), …-LiLiCorr-GGUF | LM Studio용 재배포. IQ2_XXS 파일 크기가 NVIDIA GGUF와 64바이트 차이 → 같은 NVFP4 구성으로 보임 | 22.02 GB + mmproj 1.73 GB |
| Ollama 라이브러리 `nemotron-3.5-bolt` (`:latest`, `:e4b`) | 74B-A9B는 25 GB(256K, 텍스트·이미지), E4B는 7.2–7.9 GB. 페이지에 "Apple Silicon / Metal is not currently supported for 74B-A9B" 명시 | 25 GB / 7.2–7.9 GB |

**GitHub**

| 저장소 | 역할 | 라이선스 | ★ | 마지막 push | 확인 커밋 |
|---|---|---|---|---|---|
| ggml-org/llama.cpp | GGUF 추론. 논문이 인용한 PR #28717·#29807(CUDA SSM), #30506(LiLiCorr) 모두 병합 | MIT | 132,831 | 2026-10-30 | `889f2ec` (소스 일부 읽음) |
| vllm-project/vllm | 고처리량 서빙. PR #57934(LiLiCorr), #59604(IQ2_XXS·Q8_0 b12x) 병합 | Apache-2.0 | 95,910 | 2026-10-30 | `8493b34` (메타데이터만) |
| vllm-project/vllm-gguf-plugin | vLLM GGUF 실행 | Apache-2.0 | 55 | 2026-10-24 | 메타데이터만 |
| local-inference-lab/b12x | SM12x(DGX/RTX Spark, RTX 5090)용 IQ2·Q8_0 커널 | Apache-2.0 | 268 | 2026-10-06 | 메타데이터만 |
| NVIDIA/Model-Optimizer | QAD·PTQ, 초안 학습 | Apache-2.0 | 5,471 | 2026-10-30 | `cb9ab4d` (메타데이터만) |
| NVIDIA-NeMo/Nemotron | 사용 cookbook | Apache-2.0 | 2,248 | 2026-10-28 | `99a7e45` (메타데이터만) |
| NVIDIA-NeMo/Automodel · RL · Gym | 부모 계열의 SFT·RL·환경 | Apache-2.0 | 1,064 · 2,095 · 1,305 | 2026-10-30 | 메타데이터만 |
| ollama/ollama | `nemotron-3.5-bolt` 배포 | MIT | 185,334 | 2026-10-30 | 메타데이터만 |
| unslothai/unsloth | 미세조정·GGUF | Apache-2.0 | 78,795 | 2026-10-30 | 메타데이터만 |
| scaleapi/mcp-atlas · ArtificialAnalysis/aa-agentperf-local | 평가 | MIT · Apache-2.0 | 163 · 122 | 2026-10 | 메타데이터만 |

**공개되지 않은 것:** Star Elastic 압축 코드와 라우터, QAD 학습 스크립트, 데이터 혼합 비율의 정확한 값, 평가 하네스(WebArena 내부 구현)입니다. 가중치와 런타임은 공개되어 있지만 **학습 파이프라인은 재현할 수 없습니다.**

## 5. 논문이 다루지 않았지만 관련 있는 것 (내 판단)

| 주제 | 관련성 | 근거 수준 |
|---|---|---|
| **MoE 배치 검증 비용** (예: expert offloading, speculative decoding for MoE 연구들) | 초안 검증 때 토큰마다 다른 expert를 읽어 비용이 늘어나는 문제는 MoE 추론 연구에서 알려진 주제입니다. `03` §3의 toy가 같은 질문을 다룹니다. | 내 해석, 서지 확인 필요 |
| **Apple MLX 저비트 양자화** | 맥에서는 GGUF IQ2보다 MLX 2–4비트 경로가 현실적인데, 3비트 변환도 31.65 GB라 24 GiB 맥에는 들어가지 않습니다. | HF 파일 크기로 확인 |
| **Gemma 3n/4 E-시리즈의 PLE(층별 임베딩)** | "E4B = 유효 4B"는 임베딩을 저장 장치로 옮기는 Gemma 방식이라, "4GB"가 메모리 표기인지 파일 표기인지 따로 확인해야 합니다. | 내 사전 지식, 확인 필요 |
