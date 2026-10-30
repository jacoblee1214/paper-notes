# 01. 보고서 분석: Nemotron 3.5 Super (VL)와 Lightning 기술 보고서

- 제목: *Nemotron 3.5 Super and Lightning: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Models for Agentic AI*
- 저자: NVIDIA (기여자 목록은 보고서 Contributors 절)
- 공개: 보고서 PDF 첫 쪽 날짜 **2026-10-27**, research.nvidia.com에 PDF로만 공개(arXiv 없음, 79쪽). 모델은 Hugging Face에 2026-10-12에 올라왔고, 발표 블로그 두 편은 2026-10-15에 나왔습니다. 보고서가 모델과 블로그보다 약 2주 늦습니다.
- 라이선스: 보고서는 **"© 2026 NVIDIA. All rights reserved."**(1쪽 하단)이고 CC 라이선스 표시가 없습니다. 그래서 보고서 그림은 개인용 리포트와 PPT에만 씁니다. 모델 가중치는 **OpenMDW-1.1**(MIT식 허용 라이선스, 조문은 `02` §4), 학습·평가 코드(NeMo 계열)는 Apache-2.0입니다.
- 분석 범위: 보고서 본문 1–53쪽, 부록 A–C. 발표 블로그(blogs.nvidia.com)와 기술 블로그(developer.nvidia.com), Hugging Face 모델 카드 2개(BF16, NVFP4), 공개 `config.json`·safetensors 색인·헤더(가중치는 받지 않음), NVIDIA-NeMo cookbook과 NeMo Gym 평가 레시피도 함께 봤습니다.

> **"Lightning"이 가리키는 것.** 같은 3.5 세대의 작은 텍스트 전용 모델 **Nemotron 3.5 Lightning 30B-A3B**입니다(Nano 3 Base에서 초기화, Hugging Face 공개 2026-08-01). 이 보고서는 두 모델을 함께 다루지만 분량은 대부분 Super VL에 씁니다. Lightning은 증류, MTP, 양자화(§4.9), 병렬 drafter(DFlash, DSpark) 부분에만 나옵니다. 이 저장소의 `lightning_grasp/`(로봇 grasp 논문)와는 관계가 없습니다.

---

## 1. Abstract (한 줄 요약과 주장)

**한 줄 요약:** Nemotron 3 Super(120B-A12B, 텍스트)를 출발점으로, **Ultra 3 Base에서 1T 토큰 증류 → Indic 계속 사전학습 → 1M 문맥 → 256K 문맥 SWE 중간학습**을 거친 뒤, 사후학습 단계에서 **시각 입력을 붙이고 전문 교사 여러 개를 MOPD로 합쳐** 멀티모달 에이전트 모델로 만들었습니다. 그다음 **NVFP4 양자화와 vLLM 서빙 최적화(ReplaySSM, Mamba 체크포인트 prefix caching)**까지 한 보고서에 담았습니다.

보고서가 내세우는 수치(초록, §1)는 다음과 같습니다.

1. SWE-bench Pro **51.78%**로 Nemotron 3 Ultra(49.66%)를 넘음. 활성 파라미터는 약 4.6배 적음.
2. Indic 계속 사전학습으로 인도 언어 벤치마크 최대 **+9.82%p**(T2).
3. 사후학습 MTP boosting으로 acceptance length가 greedy에서 +10.1%, 확률적 draft sampling에서 **+21.8%**(T8).
4. NVFP4 양자화 정확도 복원율 중앙값 **99.62%**(36개 벤치마크, §4.1). 단, 결론(§6)에는 99.52%로 적혀 있습니다(`04` C8).
5. GB200 한 장에서 사용자당 50 tok/s 이상을 유지하며 최대 **4,856 tok/s/GPU**(T24). 같은 조건에서 NVFP4가 BF16의 2.43배, Qwen3.5 122B-A10B의 2.34배, Qwen3.8 Flash-Next의 1.90배(F10, F11).
6. 공개물: base·BF16·NVFP4 체크포인트, 병렬 drafter(Super용 LiLiCorr, Lightning용 DFlash·DSpark), 데이터셋, cookbook, 학습 레시피.

---

## 2. Method

### 2.1 구조 (보고서에는 구조 표가 없어 공개 config로 보완)

보고서 본문은 "120B total, 12B active, hybrid Mamba/attention, 512 routed experts, MTP"(§4.2)라고만 적고 층 구성을 표로 주지 않습니다. 아래 값은 공개 `config.json`과 safetensors 헤더에서 읽었습니다(`analysis/param_check.py`, `04` D5).

| 항목 | 값 |
|---|---|
| 층 | 88층 = Mamba-2 40 + LatentMoE 40 + attention 8. 패턴 `MEMEMEM*EMEMEMEM*…` (attention은 대략 11층마다 1개) |
| LatentMoE | hidden 4096 → latent 1024로 줄인 뒤 routed expert 512개 중 top-22, expert 중간 차원 2688, shared expert 1개. 라우터 계산은 FP32 |
| Attention | query 32헤드, **KV 2헤드**, head_dim 128 (GQA 16:1) |
| Mamba-2 | 128헤드 × 64, 상태 128, 그룹 8 |
| 문맥 | 최대 1,048,576 토큰 |
| 어휘 | 131,072 |
| 시각 | C-RADIO v4-H (ViT-H/16, 32층, 폭 1280, 최대 패치 13,312, 2×2 다운샘플) + projector(중간 20,480) |
| MTP | 재귀적으로 공유하는 MTP 층 1개(attention + MoE). lm_head는 본체와 공유 |
| 파라미터 | 전체 **124.45B**(보고서 BPE 계산의 N과 일치) = LLM 120.67B + MTP 2.94B + 시각 0.65B + projector 0.19B. routed expert가 전체의 90.6% |
| 활성 | 토큰당 **12.2B**(임베딩 제외) / 12.8B(포함). 이름의 "A12B"와 맞음 |

### 2.2 사전학습 파이프라인 (§2)

```
Nemotron 3 Super Base (25T 사전학습, arXiv:2604.12374)
  │ MTP 초기화: 원래 MTP 헤드 2개 중 첫째만 남겨 재귀 공유 헤드 1개로 (§2.1)
  │ MTP warm-up 100B 토큰 (본체 동결) + hidden-state mixing
  ▼
증류 1T 토큰 ← 교사 Ultra 3 Base (550B-A55B)                                   §2.2
  │ Skew Reverse KL (α = 0.1), top-128 logit + sink 토큰, KL 0.9 + CE 0.1
  │ LR 4.5e-6 → 4.5e-5 (200B 토큰 동안 증가) → 4.5e-6 (800B 동안 선형 감소)
  ▼
Indic 계속 사전학습 100B (Indic 10% + 직전 혼합 90%)                           §2.3
  ▼
장문맥 1M, 35B 토큰 (CP16 × TP8 × EP64, GB300)                                  §2.4
  ▼
SWE 중간학습 65.4B 토큰 @256K (T4 혼합)                                        §2.5
  ▼
장문맥 회복 9B 토큰 (RULER-1M 79.9 → 33.5 → 81.0)                              §2.4
  = Nemotron 3.5 Super Base
```

Lightning은 Nano 3 Base에서 시작해 Super 3 Base를 교사로 1T 증류(forward KL)하고, 장문맥 21B만 거칩니다. Indic CPT와 중간학습은 없습니다(§2.2, §2.4).

**MTP의 train–inference 불일치(§2.1).** 재귀 MTP를 표준 방식으로 학습하면 k번째 단계는 정답 은닉 상태로만 학습하지만, 추론에서는 앞선 MTP 단계가 만든 합성 상태를 입력으로 받습니다. Super는 학습 중 k번째 단계의 입력을 1…k−1 단계 출력에서 섞어 뽑는 **hidden-state mixing**으로 이 차이를 줄였습니다. 여러 단계를 실제로 펼쳐 학습하는 TTT보다 계산이 적다는 것이 근거입니다.

### 2.3 Indic 계속 사전학습 (§2.3, T1–T3, F2–F3)

- **데이터:** Nemotron-CC 영어 문서(4,096 토큰 미만, 분할 없이 통째로)를 Gemma-4-31B-it로 번역합니다. 번역 → 품질 점수(q ≥ 75면 채택) → 오류 수정 → 재채점의 4단계이고, 로마자 표기본도 같은 모델로 만듭니다(F2). 저자원 10개 언어는 Bodhan-MT로 번역합니다. 총 **약 2.83T 토큰, 22개 언어, 34개 언어·문자 조합**입니다(T1, 합계 2,825.6B 재계산 일치).
- **비용:** 번역 비용을 영어 입력 100만 토큰당 0.835 → 0.335 GPU-h로 줄였습니다(−59.9%, 2.49배).
- **결과(T2):** 5개 Indic 벤치마크가 모두 올랐습니다(+4.98 ~ +9.82%p). 다만 합성 데이터와 자연 데이터의 기여는 나누지 않았습니다(본문 명시).
- **혼합 비율(T3):** Lightning으로 200B 토큰 예산에서 Indic 10/20/30%를 비교하면 평균 63.1 / 64.6 / 65.6으로 단조 증가합니다. 공개된 Lightning에는 Indic 단계가 없습니다.

### 2.4 SWE 중간학습 (§2.5, T4–T5)

사전학습은 짧은 문맥이라 저장소 전체, 이슈–PR, 커밋 이력, 빌드 로그를 담기 어렵습니다. SFT는 대부분 합성 데이터이고, 너무 길면 이후 RL 이득을 줄입니다. 중간학습은 그 사이에서 **자연 SWE 데이터를 256K 문맥으로** 학습합니다.

| 혼합(T4) | 비율 |
|---|---|
| 이슈–PR(저장소 맥락 + 병합 패치) | 41% |
| 저장소 전체(서로 import하는 파일을 붙여서) | 20% |
| 사전학습 재생 | 20% |
| 에이전트 궤적(도구 응답은 loss mask) | 10% |
| 장문맥 데이터 | 5% |
| 커밋 diff | 3% |
| 빌드·CI 로그 | 1% |

효과는 같은 실험용 SFT(33B 토큰, 2K step)를 base와 중간학습 체크포인트에 각각 돌려 비교했습니다(T5). OpenHands 하네스에서는 차이가 ±1점(실행 간 변동 수준)입니다. 다른 5개 하네스에서는 SWE-bench Verified가 평균 **+8.6점**, Terminal-Bench 2.1이 평균 **+6.8점** 올랐습니다(재계산 8.56, 6.82). 가장 큰 개선은 Codex 하네스입니다(48.4 → 61.8). DeepSWE는 6.4 → 10.4입니다. 다시 말해 **중간학습의 주효과는 점수보다 하네스 간 강건성**입니다.

### 2.5 사후학습 파이프라인 (§3, F4)

```
Super 3.5 Base (텍스트 전용)
  │ vision projector warm-up
  ▼ SFT 2단계 (§3.1): 1단계 = 시각·추론·대화·장문맥·안전·에이전트 혼합, 2단계 = 부족한 능력 보강
  │   packing 524,288 토큰, GBS 32, LR 1e-5 고정, MTP 보조 loss 0.3
  ▼ RL (§3.2): RLVR + RLHF 동시, GBS 8,192 × rollout 16, 최대 생성 80K → 128K
  │   router replay, FP32 LM head (학습·추론 엔진 간 KL 오차를 10배 이상 줄임)
  ▼ step-120 체크포인트에서 교사들을 가지치기
  ├─ 에이전트 교사 (§3.3.1): Terminal·SWE·Search 샌드박스, GRPO 64 × 16, rollout당 최대 2시간
  ├─ 추론 교사 (§3.3.2): 25,600문항(과학 90%), 생성 131,072, 80 update
  ├─ 비디오 추론·추적 교사 (§3.4.1): Video-QA + SA-V 추적, 라우터·시각 모듈 동결, step 60
  ├─ 시각 도구 사용 교사 (§3.4.2), 시각 추론 RLVR (§3.4.3)
  └─ 정렬 교사 (§3.5): 길이 벌칙을 넣은 GenRM, 고정 참조 응답(Gemma 4 31B-it)과 비교
  ▼ MOPD 2단계 (§3.6)
  ▼ MTP boosting (§3.7, 본체 동결)
  = Nemotron 3.5 Super (BF16) → NVFP4 PTQ + QAD (§4)
```

- **SFT 데이터 출처(§3.1):** 시각 데이터는 Nemotron 3 Nano Omni 코퍼스에 Qwen3.5로 만든 데이터를 더했습니다. 시각 에이전트 데이터는 Kimi K2.6·K3로 만들었고(이미지 crop·회전·확대 도구, SpatialClaw 비디오 궤적 14,169개, 행동 81,366개), 추론 데이터는 Kimi K3(수학·STEM), DeepSeek V4 Pro(증명), GLM-5.2(경쟁 프로그래밍)가 교사입니다. **5,000 토큰보다 짧은 추론 응답은 버렸습니다.** Kimi K3 궤적은 여러 epoch에 걸쳐 계속 좋아졌다고 보고합니다(DeepSeek V4 Pro 궤적은 1 epoch에서 정체).
- **시각 에이전트 학습 효과(T7):** SpatialClaw 하네스 기준 공간·비디오 벤치마크 평균 50.5 → 54.8(+4.3)입니다.
- **추론 모드(§3.9):** off, regular, medium-effort 세 가지입니다. medium은 regular보다 토큰을 약 2.2배 적게 쓰고 정확도는 약 9% 낮습니다.

### 2.6 MOPD: 다중 교사 on-policy 증류 (§3.6, Eq. 1–2)

학생이 직접 생성한 응답의 각 토큰에서, 그 도메인을 맡은 교사의 확률과 학생 확률의 로그 차이를 advantage로 씁니다.

```
Δθ ∝ E_{y ~ π̄_S(·|q)} [ Σ_t Â_t ∇θ log π_S(y_t | c_t) ]                          (Eq. 1)
Â_t = sg[ log π_T(y_t | c_t) − log π̄_S(y_t | c_t) ]                               (Eq. 2)
```

- 이것은 학생 분포에서 본 **reverse KL D(π_S ‖ π_T)를 토큰 단위로 최소화**하는 것과 같습니다. verifier도 교사 샘플링도 필요 없고, 교사 forward 한 번이면 모든 토큰에 dense 신호가 생깁니다. 학생과 교사가 같은 곳에서는 신호가 0입니다.
- 비동기 학습이라 한 step 늦은 응답은 π̄_S/π̄_{S−1}로 중요도 가중하고 [0.2, 5] 밖은 마스크합니다.
- **도메인 라우팅:** 각 프롬프트는 해당 교사로 가고, 교사가 없는 환경은 **self-anchor**(학생 시작점의 동결 복사본)로 보냅니다. self-anchor는 학생이 시작점에서 멀어질 때만 신호를 줍니다.
- **1단계**(30 step): 에이전트 35%, 추론 25%, 정렬 20%, 시각 도구 15%, self-anchor 5%. 512 프롬프트 × rollout 1, LR 2e-6.
- **2단계**(40 step): 에이전트·추론 교사는 은퇴시키고 그 환경을 self-anchor로 돌립니다. 정렬 30%, 시각 도구 20%, 시각 추론 20%, 비디오 20%, self-anchor 10%. 1,024 프롬프트, LR 3e-6. 빨리 오르고 나중에 떨어지는 도메인을 1단계에, 느리게 오르는 도메인을 2단계에 둔 배치입니다.
- **proximal teacher(2단계만):** 교사를 α π_T + (1−α) π̄_S(α = 0.2)로 바꿉니다. 그러면 advantage가 아래에서 log(1−α) ≈ −0.223으로 막혀, 교사가 매우 낮게 보는 토큰도 무한히 큰 음의 신호를 받지 않습니다(`03` §4에서 확인).
- **부수 효과(부록 B.1):** MOPD 전 체크포인트는 tau3-banking에서 rollout 776개 중 153개(19.7%)가 끝없이 반복하며 멈추지 못했습니다(runaway loop). MOPD 뒤에는 1개입니다. 이 차이가 NVFP4 양자화 강건성에도 이어집니다(§2.8).

### 2.7 MTP boosting (§3.7, T8–T9, F6)

본체를 동결하고 MTP 헤드만 두 번 학습합니다. SFT 체크포인트에서 4k step을 먼저 학습하고, 최종 체크포인트에서 25k step을 더 학습합니다. hidden-state mixing을 유지하고, loss를 CE 대신 **동결 본체를 교사로 한 증류(LK loss)**로 바꿨습니다. 데이터는 본체가 temp 1로 생성한 on-policy rollout(8K 토큰 상한)입니다.

| draft 길이 7, SPEED-Bench | greedy AL | sampling AL |
|---|---|---|
| 최종 체크포인트, 원래 MTP | 4.287 | 3.787 |
| + SFT 단계 MTP 헤드 이식 | 4.528 | 4.436 |
| + 25k boosting | **4.718** (+10.1%) | **4.612** (+21.8%) |

깊은 draft 위치일수록 개선이 큽니다(F6). 다만 T9에서 boosting 후 Super의 greedy AL 평균 4.718은 Gemma-4 31B(4.837)와 Qwen3.5 122B(4.799)보다 낮습니다. sampling AL 4.612는 가장 높습니다.

### 2.8 양자화 (§4, T12–T23, 부록 B)

**목표:** 중앙값 정확도 복원율 99% 이상, 출력 길이 증가 5% 이하. Super VL NVFP4는 Blackwell에서 W4A4, 그 밖에서는 W4A16으로 배포하고, Lightning은 어디서나 W4A16입니다.

| 모듈 (T12) | 공개 레시피(Aggressive) |
|---|---|
| C-RADIO, VL connector, 임베딩, Mamba conv | BF16 |
| routed + shared expert GEMM | **NVFP4** |
| latent down/up projection | BF16 |
| attention QKV·out, Mamba linear, lm_head | FP8 |
| KV cache, BMM | FP8 |
| Mamba SSM cache | **FP16 + stochastic rounding** |
| 전체 BPE | **4.82** (S = 74.987 GB, N = 124.453B) |

- **PTQ(§4.2):** ModelOpt로 이미지–텍스트 SFT 데이터 1,024개를 보정에 썼습니다. 블록 scale은 MSE 탐색이 Max보다 낫습니다(T13 복원율 100.06% vs 98.58%).
- **AutoQuant(§4.3, Eq. 3):** 민감도(양자화 오차 × task-loss 기울기²) 합을 평균 비트 제약 아래 최소화합니다. lm_head가 NVFP4 민감도의 거의 절반이고, attention·Mamba projection이 다음이며, expert는 상대적으로 강건합니다.
- **QAD(§4.4):** PTQ 학생에서 시작해 양자화된 GEMM 가중치만 KL로 학습합니다. 100% SFT 데이터로 200 iteration이 가장 좋았고, RL rollout을 섞은 효과는 작았습니다. 100 iteration에 GB200 512장으로 약 2시간(1,024 GPU-h)입니다.
- **KV cache(§4.5, T18):** NVFP4 KV(블록 16개당 FP8 scale, 원소당 4.5비트)는 FP8 대비 저장량이 43.75% 적습니다. 복원율은 NVFP4 KV 99.36%, FP8 KV 99.52%로 비슷했습니다. 그래도 공개 체크포인트는 소프트웨어 성숙도 때문에 FP8 KV를 기본으로 씁니다.
- **어디가 민감한가(§4.6, 부록 B):** 정확도 손실은 주로 **expert 가중치**에서 나옵니다. FP4로 양자화하면 멈출지 계속할지의 경계(stop-vs-continue margin)가 흔들리고, 원래 불안정하던 모델에서 runaway loop가 늘어납니다. MOPD 전 모델에서 expert를 FP8로 올리면 runaway가 28.2% → 17.9%로 줄었습니다. Mamba out_proj를 FP4로 만들면 출력이 길어지고, lm_head는 FP4에서도 정확도는 같지만 종료 판단이 흔들려 FP8 이상을 권합니다. 장문맥 구분 과제(MRCR-128k)는 attention score 해상도에 민감합니다.
- **MTP 양자화(§4.7):** MTP expert를 NVFP4로 바꾸면 AL이 4.612 → 4.592입니다. MTP는 따로 양자화한 뒤 본체 체크포인트에 이어 붙였습니다(splice).

### 2.9 서빙 (§5)

- **작업 부하:** 20턴 에이전트 대리 부하(공유 prefix 51,200, 턴당 입력 2,048, 출력 1,024)에서 **사용자당 50 tok/s 이상**인 점 가운데 GPU당 처리량이 가장 높은 점을 고릅니다.
- **Mamba prefix caching(§5.3):** vLLM은 attention과 Mamba의 캐시 페이지 크기를 바이트로 맞춥니다. Mamba 상태 한 페이지가 attention KV 약 4K 토큰과 같으므로(`03` §2에서 정확히 4,096으로 재계산), Mamba 상태는 4K 토큰 간격으로만 저장됩니다. 투기적 디코딩과 함께 쓰면 최대 4K 토큰을 다시 계산해야 해서, 공유 prefix 경계 근처에 Mamba 체크포인트를 하나 더 두도록 바꿨습니다.
- **ReplaySSM(§5.4):** 투기적 디코딩은 몇 토큰이 채택될지 모르므로 후보 위치마다 Mamba 상태를 써야 합니다. 대신 체크포인트 상태와 이후 입력(k, v, α)만 작은 버퍼에 저장하고, 필요할 때 상태를 재구성합니다. 상태 쓰기를 버퍼 길이 L에 걸쳐 분할상환하므로 상태 트래픽이 T+1에서 약 1+1/L로 줄고, 양자화 횟수도 줄어듭니다. 측정 이득은 출력 처리량 +4.7–6.4%, 사용자당 +6.4–12.5%입니다(T24). Dao lab의 동시 발견과 같은 기법입니다.
- **분리 서빙(§5.6, T25):** Vera Rubin(VR200) 11장에서 prefill 7 + decode 4로 나누면 GPU당 처리량 +10.3%, 사용자당 속도 +61%입니다. TTFT는 0.49 → 1.12 s로 늘지만 5 s 목표 안입니다.
- **병렬 drafter(§5.7):** LiLiCorr는 블록 8칸(7개 draft)을 한 번에 제안하고, 칸마다 상위 K 후보 사이의 호환성을 작은 Transformer로 점수 매겨 왼쪽부터 고릅니다(Eq. 4–5). drafter는 1.13B(헤드 37M)이고, 동시 처리량이 낮은 DGX Spark 같은 환경이 목표입니다. AL은 greedy 4.77, sampling 4.37입니다(T28).

---

## 3. Experiments

### 3.1 사후학습 모델 평가 (T10, 27개 공통 텍스트 벤치마크 + 시각 11개)

| 영역 | Super 3.5 | Ultra 3 | 관찰 |
|---|---|---|---|
| SWE-bench Pro | **51.78** | 49.66 | 표의 7개 모델 중 5위. Qwen3.8 60.37, GLM-5.3 59.70 |
| SWE-bench Verified | 71.10 | 73.93 | 7개 중 **7위** |
| Terminal-Bench 2.1 | 60.67 | 54.78 | Qwen3.8 85.80 |
| Tau2 Telecom | 73.19 | 83.00 | 7개 중 **7위** |
| BrowseComp | 78.91 | 67.85 | 두 하네스 모두 Ultra보다 +11점 |
| GraphWalks (≤1M) | **89.71** | 63.28 | 7개 중 **1위** (유일한 1위) |
| AA-LCR | 68.63 | 75.19 | 7개 중 7위 |
| IFBench | 70.86 | 82.94 | −12.1 |
| MMLU-ProX | 76.91 | 83.10 | 7개 중 7위 |
| NV Chat Benchmark | 52.67 | 34.27 | +18.4 (내부 벤치마크) |
| 시각 11개 | | — | 대부분 Qwen3.8·Qwen3.5보다 낮음(MathVision은 5개 중 5위). OCR-Reasoning, OCRBench-v2(EN), Video-MME-v2에서만 2위 |

- **Super vs Ultra:** 27개 공통 벤치마크에서 Super가 17개 우세, 10개 열세입니다. 평균 순위는 4.24/7입니다(`04` C6). "Ultra를 넘는다"는 초록의 표현은 SWE-bench Pro 한 줄에 대해서는 맞지만, Ultra가 확실히 나은 영역(장문맥 AA-LCR, 지시 따르기 IFBench, 다국어 MMLU-ProX, Tau2)도 분명합니다.
- **기준선 설정:** 기준 모델의 온도·top_p·최대 토큰은 각 모델 카드 권장값을 그대로 썼다고 밝힙니다(§3.8.1). 자체 측정이므로 공식 리더보드 값과 다를 수 있습니다.

### 3.2 Text Arena (T11)

Super NVFP4는 투표 2,328개(Ultra의 1/5)로 1420 ± 12입니다. 신뢰구간이 GLM-5.3(1474 ± 5)을 제외한 모든 모델과 겹칩니다. 보고서도 "preliminary"라고 적었습니다.

### 3.3 하네스 강건성과 테스트 시간 확장 (§3.10–3.11, F8–F9)

- **F8:** 하네스 6개의 기본 설정에서 SWE-bench Pro는 Claude Code 49.45%가 최고, OpenCode가 **41.04%로 최저**입니다. T10의 51.78%도 OpenCode 하네스 값인데 10.7%p 높습니다. F8은 Git 기록 제거와 부정행위 금지 지시를 명시하고, T10은 NeMo Gym 레시피를 씁니다. 이 차이를 보고서는 설명하지 않습니다(`04` C11).
- **F9 (병렬 에이전트 + 자기 검증):** 같은 체크포인트가 후보 3–5개를 만들고 LLM-as-a-Verifier(20단계 등급 토큰의 확률 가중 점수, 쌍 비교, pivot 토너먼트)로 하나를 고릅니다. Terminal-Bench 62.7 → 71.9, DeepSWE 24.6 → 34.5, BrowseComp(400문항 부분집합) 71.5 → 79.8입니다. SWE 3종은 +1.1–2.4%p로 작습니다.

### 3.4 양자화 결과 (T22, T23)

- Super NVFP4(T23, 37행): 표 전체 중앙값 99.70%이고, 보고값 99.62%는 36개 부분집합에서 나옵니다. 최저는 SciCode problem 93.58%, Tau3 Banking 94.17%, MMLU-ProX 95.90%입니다. 출력 길이는 +0.43%(보고 +0.44%)입니다.
- Lightning W4A16(T22): 중앙값 99.25%(보고 99.24%), 출력 길이 −3.63%, 최저 90.22%입니다.

### 3.5 서빙 성능 (F10–F11, T24–T25)

| 비교 (사용자당 ≥ 50 tok/s) | GB200 | H100 |
|---|---|---|
| NVFP4 / BF16 (F10) | 4,324 / 1,778 = **2.43배** | 970 / 522 = 1.86배 |
| Super / Qwen3.5 122B (F11) | 4,331.88 / 1,854.35 = **2.34배** | 988.87 / 424.94 = 2.33배 |
| Super / Qwen3.8 Flash-Next (F11) | 4,331.88 / 2,282.73 = **1.90배** | 988.87 / 752.95 = **1.31배** |

- 초록의 4,856 tok/s/GPU는 T24의 최고점(동시 64, MTP 5, ReplaySSM 켬, 사용자당 89.9 tok/s)입니다. F10(4,324), F11(4,331.88)과는 설정이 달라 값이 다릅니다.
- 성능 모델 설명(§5.2): attention 층이 적고 KV 헤드가 2개뿐이라 요청당 메모리가 문맥과 함께 느리게 늘어납니다. 그만큼 한 번의 가중치 읽기를 더 많은 요청이 나눠 씁니다. LatentMoE는 expert 병렬 통신량도 줄입니다.
- **SWE-bench Pro 정확도–계산(F7):** B200 4장에서 2,193과제를 끝내는 데 Super NVFP4는 8.3 GPU-h, Qwen3.8 37.1, DeepSeek V4 Flash 40.4, Inkling Small 98.9 GPU-h입니다. 다만 정확도는 Super가 51.3%로 셋보다 5.2–9.1%p 낮습니다.

### 3.6 Base 모델 (T6)

Super 3.5 Base와 Lightning 3.5 Base만 비교하고, **출발점인 Nemotron 3 Super Base와의 비교는 없습니다.** 그래서 증류·Indic·중간학습이 base 능력을 얼마나 바꿨는지는 T2(Indic), T5(중간학습)로만 부분적으로 볼 수 있습니다. 두 모델의 GSM8K(91.74)와 PIQA(85.09)가 소수 둘째 자리까지 같습니다. Lightning의 GPQA Diamond 33.33%는 198문항 중 66문항으로, 4지선다 우연 수준(25%)에 가깝습니다.

---

## 4. Limitations

### 4.1 저자가 밝힌 한계
- Indic 단계의 이득에서 합성 데이터와 자연 데이터의 기여를 나누지 않았습니다(§2.3). Lightning 혼합 비율 실험은 Super의 최적 비율을 정하지 않습니다.
- 오염 제거 파이프라인은 저렴한 유사도 신호로 후보를 고른 뒤에만 judge가 판정하므로 재현율에 상한이 있습니다. 문제 문장만 보고, 합성 오염 실험은 대리 평가입니다(부록 C.6). 합성 변형 탐지율은 Terminal-Bench 87.5%, GPQA 79.1%입니다.
- 양자화 원인 분석(부록 B)은 추론에 기반한 귀속이고, MRCR-128k가 유일한 구분 과제이며, 모든 구성이 FP8 KV를 쓴다는 단서를 붙였습니다.
- 하네스에 따라 순위가 바뀌므로 하네스와 설정을 함께 보고해야 한다고 스스로 강조합니다(§3.10).

### 4.2 분석하며 본 한계
1. **구조 표가 없습니다.** 층 구성, expert 수, KV 헤드 같은 값은 config를 봐야 알 수 있습니다(`04` D5).
2. **같은 수치가 두 값으로 나옵니다.** NVFP4 복원율이 초록·§4.1에서는 99.62%, §4.5·결론에서는 99.52%입니다. 처리량도 4,856 / 4,324 / 4,331.88로 세 값입니다(`04` C1, C8).
3. **헤드라인 SWE-bench Pro 수치는 프로토콜에 민감합니다.** 같은 OpenCode 하네스인데 T10 51.78%, F8 41.04%입니다. F8의 최고값 49.45%는 T10의 Ultra 49.66%보다 낮습니다(`04` C11).
4. **불확실성 표시가 거의 없습니다.** T10은 3–8회 평균이지만 표준편차나 신뢰구간이 없습니다. 벤치마크마다 1–2점 차이는 실행 간 변동(§2.5에서 "약 1점")과 구분하기 어렵습니다.
5. **MOPD와 교사 각각의 기여를 분리한 ablation이 없습니다.** 비디오 교사 같은 개별 교사 점수(§3.4.1)만 있고, 최종 모델 점수가 어느 단계에서 왔는지 보여 주는 단계별 표가 없습니다.
6. **블로그에만 있는 주장이 많습니다.** "5배 처리량", "Dynamo로 8배 사용자·TTFT 85% 감소", "LiLiCorr 2.3배", "Baseten 600 TPS", "사후학습 약 3.8M 샘플"은 보고서에서 확인되지 않습니다. 3.8M은 보고서의 오염 제거 수치(20,671개 = 약 0.09%)에서 역산한 약 23M과 맞지 않습니다(`04` C5, C10).
7. **로봇·embodied 평가가 없습니다.** 시각 평가는 문서·차트·OCR·비디오 QA·GUI(OSWorld) 중심입니다. 공간 추론(ERQA 등)은 SpatialClaw 하네스 효과(T7)로만 나옵니다. 로봇 VLA 백본으로 쓸 수 있는지는 이 보고서로 판단할 수 없습니다.
