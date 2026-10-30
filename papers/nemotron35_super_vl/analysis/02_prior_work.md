# 02. 선행연구, 공개 자료, 이 저장소의 다른 논문과의 관계

arXiv 번호와 제목은 arXiv API로 확인했습니다(2026-10-30). 보고서가 인용하지 않은 계열 문서는 "인용 없음"으로 표시했습니다. 선행 보고서들은 따로 열어 검증하지 않았고, 아래 설명은 이 보고서가 그 문서를 어떻게 쓰는지에 한정합니다.

## 1. 계보도

```
Nemotron-H (arXiv:2504.03624, 2025-04)  Mamba-2 + attention 하이브리드 계열의 출발점        ← 인용 없음
   ▼
Nemotron 3 (백서 arXiv:2512.20856)                                                           ← 인용 없음
  ├─ Nano 3 (arXiv:2512.20848)        → Lightning 3.5의 초기화 (§2.2)
  ├─ Super 3 (arXiv:2604.12374)       → Super 3.5의 초기화, 25T 사전학습, 재귀 공유 MTP 정식화 (§2.1)
  ├─ Ultra 3 (arXiv:2606.15007)       → 증류 교사(550B-A55B), RL 알고리즘, MOPD 원형, 오염 제거·추론 모드 레시피
  └─ Nano 3 Omni (arXiv:2604.24954)   → 시각 데이터 코퍼스, vision projector warm-up (§3.1)
                                              ▼
            ★ Nemotron 3.5 Super VL (120B-A12B) · Lightning (30B-A3B)  ← 이 보고서 (2026-10-27)
                                              │
            같은 3.5 세대: Bolt 74B-A9B 24GB (arXiv:2610.33039, 로컬용 하이브리드 VLM) → 별도 분석 `nemotron35_bolt/`

[사후학습 계열]  Nemotron-Cascade (arXiv:2512.13607, 인용 없음) → Cascade 2 (arXiv:2603.19220, MOPD의 토큰 단위 갱신 근거로 인용)
                 On-policy distillation (Lu & Thinking Machines, 2025) · trust-region policy distillation (Xie et al., 2026)
[투기적 디코딩]  speculative sampling (Leviathan 2023) · MTP (Gloeckle 2024) · LK loss (Samarin 2026)
                 DFlash · DSpark · LiLiCorr (Rusanovsky 2026) · ReplaySSM (Liou, …, Dao 2026)
[기타]           Skew Reverse KL (Ko 2024) · LLM-as-a-Verifier (Kwok 2026) · Nemotron Elastic (arXiv:2511.16664, 인용 없음)
```

**차별점 요약:** 구조 자체는 Super 3와 같습니다(LatentMoE, Mamba-2, MTP). 3.5에서 새로 생긴 것은 (1) 더 큰 교사(Ultra 3)에서 받는 1T 토큰 증류, (2) Indic과 SWE 중간학습이라는 데이터 단계, (3) 텍스트 전용 base를 **사후학습에서** VLM으로 바꾼 점, (4) 시각·비디오 교사를 포함한 2단계 MOPD, (5) 양자화·서빙 공학(QAD, MTP splice, ReplaySSM, Mamba 체크포인트 prefix caching)입니다. 새로운 모델 구조를 제안하는 논문이라기보다 **레시피와 시스템 보고서**에 가깝습니다.

## 2. 구성 요소별 출처

| 구성 요소 | 출처 | 이 보고서에서 바뀐 점 |
|---|---|---|
| Mamba-2/attention 하이브리드 + LatentMoE | Super 3 (2604.12374) | 바뀐 점 없음. 구조 표도 없음(`04` D5) |
| 재귀 공유 MTP | Super 3 | 헤드 2개 중 첫째만 남김, hidden-state mixing (§2.1) |
| 대형 교사 증류 | Ultra 3 Base | Skew Reverse KL(α = 0.1)과 top-128 logit (§2.2) |
| Indic 합성 데이터 | NeMo Data Designer, Gemma-4-31B-it, Bodhan-MT | 2.83T 토큰 코퍼스, 4단계 번역·검증 (§2.3) |
| SWE 중간학습 | Singhal et al. 2026 (별도 보고서) | 65.4B 토큰 @256K, T4 혼합 |
| 시각 인코더 | C-RADIO v4-H | projector warm-up 뒤 SFT에서 언어·시각을 함께 학습(§3.1). 비디오 교사 RL에서는 동결(§3.4.1). AutoModel 미세조정 레시피도 시각 모듈을 동결 |
| RL (GRPO, 비동기) | Ultra 3, NeMo-RL, NeMo Gym | router replay, FP32 LM head 추가 (§3.2) |
| MOPD | Ultra 3, on-policy distillation 계열 | 2단계, 도메인별 교사 + self-anchor, proximal teacher (§3.6) |
| MTP boosting | Ultra 3 (temperature forward-KL) | LK loss로 교체 (§3.7) |
| NVFP4 PTQ/QAD | ModelOpt | 시각 포함 보정, MTP 별도 양자화 후 splice (§4) |
| ReplaySSM | TensorRT-LLM PR #14203, Liou et al. (Dao lab) 동시 발견 | vLLM 통합 (§5.4) |
| 병렬 drafter | DFlash, DSpark, LiLiCorr | Super용 LiLiCorr 학습을 증류로 변경 (§5.7.1) |
| 테스트 시간 선택 | LLM-as-a-Verifier (Kwok 2026) | 같은 체크포인트가 검증자 (§3.11) |

## 3. 이 저장소의 다른 논문과의 관계

솔직히 말하면 **직접적인 방법론 연결은 약합니다.** 이 저장소의 로봇 논문들은 2–3B 규모 VLM이나 비디오 생성 모델을 백본으로 쓰고, 실시간 제어 지연이 핵심 제약입니다. Super VL은 활성 12B, 체크포인트 75–249 GB라 로봇 위에서 돌리는 백본 후보가 아닙니다. 연결 지점은 "오프라인 도구"와 "학습 레시피" 쪽입니다.

| 폴더 | 그 논문의 핵심 | Super VL과의 관계 |
|---|---|---|
| `thinkworld/` | Qwen3-VL-2B VLA가 손목 움직임 토큰(CoD)을 생성하고, 예측-관찰 일치 보상으로 GRPO | 둘 다 **VLM 자체를 RL로 고칩니다.** ThinkWorld의 결과 기반 보상과 Super의 MOPD는 상보적입니다. 과제별로 RL한 교사 여러 개를 한 학생으로 합치는 MOPD 레시피(§3.6)는, 과제별 VLA 정책을 하나로 합칠 때 그대로 참고할 수 있습니다. 다만 2B 학생에 MOPD를 쓴 증거는 이 보고서에 없습니다 |
| `pacevla/` | GR00T N1.5 action expert에 속도 모드 decoder 4개, LVLM(Gemini 3.8 Flash) 라벨로 학습한 judge·라우터 | PaceVLA는 데이터셋 전체의 chunk 라벨링을 상용 API로 했습니다. Super VL(OpenMDW, 비디오 입력, 1M 문맥)은 이 **오프라인 라벨러를 자체 호스팅 모델로 바꾸는 후보**입니다. 다만 로봇 국면 판정 정확도는 이 보고서의 평가에 없습니다 |
| `agentic_wam/` | 코드·미래 영상·행동을 MoT 하나에서 함께 denoise (SDAR-1.7B 코드 LM, Cosmos-Predict2.5-2B, Qwen3-VL-2B ViT). 하이브리드 모드에서 Molmo·SAM 3·GraspNet 사용 | AW-0의 "코드로 계획" 부분은 작은 LM을 씁니다. Super VL의 시각 도구 사용(SpatialClaw, crop·zoom·SAM3 호출)과 에이전트 학습은 **도구를 호출하는 상위 계획기** 쪽과 맞닿습니다. 지연 0.23 s 수준의 joint denoising과는 시간 규모가 다릅니다 |
| `hydra0/` | 범용 비디오 world model(Cosmos 2.5, Wan2.2)로 행동을 이미지 평면 점 궤적(action flow)으로 생성 | 직접 관계는 거의 없습니다. 같은 NVIDIA 연구 생태계이고, 보고서가 "© NVIDIA All rights reserved"라 그림을 게시하지 못한다는 라이선스 처리만 같습니다 |

**하이브리드 SSM 관점의 연결.** 로봇 정책의 긴 이력(수천 프레임 관측)을 다룰 때, KV는 문맥에 비례해 늘고 Mamba 상태는 고정입니다. Super의 서빙 분석(§5.2–5.4)과 `03` §2의 요청당 메모리 계산은 하이브리드 백본 VLA를 설계할 때 메모리 예산을 잡는 근거가 됩니다.

## 4. 공개 자료 (GitHub, Hugging Face)

조회일 2026-10-30. 별 수와 마지막 push는 GitHub API, Hugging Face 값은 Hub API 기준입니다. 코드 저장소는 clone하지 않고, 필요한 파일만 받아 `paper/cookbook/`, `paper/gym_recipe/`에 두었습니다.

| 자료 | 내용 | 라이선스 | 별 / 좋아요 | 마지막 push·수정 | 확인한 커밋·리비전 |
|---|---|---|---|---|---|
| [NVIDIA-NeMo/Nemotron](https://github.com/NVIDIA-NeMo/Nemotron) | cookbook(vLLM·SGLang·TRT-LLM 배포, LoRA, RL, 에이전트 예제) | Apache-2.0 | ★2,248 | 2026-10-28 | `99a7e45` |
| [NVIDIA-NeMo/Automodel](https://github.com/NVIDIA-NeMo/Automodel) | Super VL 전체 SFT·LoRA 레시피 | Apache-2.0 | ★1,064 | 2026-10-30 | `571df6d` |
| [NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL) | GRPO, MOPD 학습 프레임워크 | Apache-2.0 | ★2,095 | 2026-10-30 | `2bb62a8` |
| [NVIDIA-NeMo/Gym](https://github.com/NVIDIA-NeMo/Gym) | 평가·RL 환경, `nemotron_recipes/super-3.5/` 재현 레시피 | Apache-2.0 | ★1,305 | 2026-10-30 | `abdc0b5` (레시피가 고정한 평가 커밋 `1e66890`) |
| [NVIDIA/Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer) | NVFP4 PTQ·QAD·AutoQuant | Apache-2.0 | ★5,471 | 2026-10-30 | `cb9ab4d` |
| [NVIDIA-NeMo/DataDesigner](https://github.com/NVIDIA-NeMo/DataDesigner) | Indic·시각 합성 데이터 파이프라인 | Apache-2.0 | ★2,382 | 2026-10-30 | `f61f71d` |
| [NVIDIA-NeMo/Evaluator](https://github.com/NVIDIA-NeMo/Evaluator) | base 모델 평가(T6) | Apache-2.0 | ★357 | 2026-10-19 | `ea35ba3` |
| HF `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16` | 사후학습 BF16, 가중치 shard 63개, 248.9 GB | OpenMDW-1.1 | ♥107 | 2026-10-22 | `fb0ee0b` |
| HF `…-Super-VL-120B-A12B-NVFP4` | 공개 양자화본, 75.0 GB | OpenMDW-1.1 | ♥246 | 2026-10-23 | `da5b265` |
| HF `…-Super-VL-120B-A12B-Base-BF16` | base | OpenMDW-1.1 | ♥57 | 2026-10-22 | `5d80455` |
| HF `…-Super-VL-120B-A12B-NVFP4-LiLiCorr` | 병렬 drafter, 4.9 GB | OpenMDW-1.1 | ♥11 | 2026-10-16 | `292bf1e` |
| HF `…-Lightning-30B-A3B-BF16` / `-NVFP4` / `-Base-BF16` | Lightning 본체 (65.8 / 25.2 / 65.8 GB) | OpenMDW-1.1 | ♥257 / 482 / 28 | 2026-08-24 / 09-10 / 08-11 | `a9904d2` / `bee7596` / `434456c` |
| HF `…-Lightning-30B-A3B-NVFP4-DFlash` / `-DSpark` | Lightning drafter | OpenMDW-1.1 | ♥23 / 42 | 2026-09-01 / 10-12 | `8abcc4d` / `6d127b5` |
| HF `nvidia/Nemotron-Post-Training-Dataset-v2` | MTP boosting 시드 프롬프트 | CC BY 4.0 (gated: auto) | — | 2025-08-21 | `5c89e01` |
| HF `nvidia/Nemotron-RL-Super-Training-Blends` | RL·MTP boosting 프롬프트 | CC BY 4.0 | — | 2026-09-29 | `79c4deb` |
| HF `nvidia/Nemotron-Pretraining-Indic-CC-v1`, `-Indic-Specialized-v1` | Indic 코퍼스 일부 | 확인 안 함 | — | 2026-09-04, 10-26 | — |

**모델 라이선스 OpenMDW-1.1** (Linux Foundation, `paper/openmdw_1.1.html`에서 원문 확인): 모델과 관련 산출물을 저작권·특허·DB·영업비밀 범위에서 무료로 "제한 없이" 다룰 수 있게 허락합니다. 재배포할 때는 라이선스 사본과 고지를 유지해야 하고, 모델이 특허·저작권을 침해한다고 먼저 소송을 걸면 권리가 종료됩니다. 모델 출력에는 제한이 없습니다. 모델 카드는 "commercial or non-commercial use"가 가능하다고 적었습니다. 이 라이선스는 **모델에만** 적용됩니다. 보고서 PDF는 별도로 "All rights reserved"입니다.

**선행 보고서 (arXiv)**

| arXiv | 제목 | 이 보고서에서의 역할 |
|---|---|---|
| 2504.03624 | Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models | 하이브리드 계열의 시작 (인용 없음) |
| 2512.20856 | NVIDIA Nemotron 3: Efficient and Open Intelligence | 3세대 개요 (인용 없음) |
| 2512.20848 | Nemotron 3 Nano | Lightning 초기화 |
| 2604.12374 | Nemotron 3 Super | Super 3.5 초기화, MTP 정식화 |
| 2606.15007 | Nemotron 3 Ultra | 증류 교사, RL·MOPD·오염 제거 레시피 |
| 2604.24954 | Nemotron 3 Nano Omni | 시각 데이터, projector warm-up |
| 2512.13607 | Nemotron-Cascade | cascade RL (인용 없음) |
| 2603.19220 | Nemotron-Cascade 2 | MOPD의 토큰 단위 on-policy 증류 근거 |
| 2511.16664 | Nemotron Elastic | many-in-one 모델 (인용 없음). Lightning MTP를 Super 3 MTP에서 차원을 줄여 만든 방식(§2.1)과 발상이 비슷하지만 관계는 확인하지 않음 |
| 2610.33039 | Nemotron 3.5 Bolt 74B-A9B 24GB | 같은 세대의 로컬용 모델. 이 폴더에서는 다루지 않음 |
