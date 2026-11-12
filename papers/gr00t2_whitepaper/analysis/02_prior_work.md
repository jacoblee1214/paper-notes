# 02. 선행연구, 구성 요소의 출처, 공개 자료

## 1. GR00T 계보

| 세대 | 공개 시점 (확인) | 구조 (공개 자료 기준) | GR00T 2와의 차이 |
|---|---|---|---|
| GR00T N1 | 2025-06 (코드 태그 `n1-release`, 2025-06-11; arXiv 2503.14734) | VLM + 확산 행동 헤드 (VLA) | 영상 예측 없음 |
| GR00T N1.5 | 2025-12-15 (`n1.5-release`) | 같은 계열, 동결 VLM 개선 | 영상 예측 없음 |
| GR00T N1.6 | 2026-04-15 (`n1.6-release`), HF `GR00T-N1.6-3B` | 같은 계열 | 영상 예측 없음 |
| GR00T N1.7 | 2026-04-18 (`n1.7-release`), HF `GR00T-N1.7-3B` (2026-11-10 갱신) | VLM(`Cosmos-Reason2-2B`) + 32층 DiT 행동 헤드, 행동 호라이즌 40 (config) | 백서 주 비교 대상 "GR00T 1.7(VLA)" |
| **GR00T 2** | 백서 2026-11 (코드 "12월 공개 예정") | Cosmos-Predict2.5-2B 기반 WAM, 영상과 행동 공동 flow matching | 아래 §3 |

백서는 GR00T 1.7을 "VLA", GR00T 2를 "WAM"으로 나눕니다(Sec. 2, Tab. 1). 두 모델은 같은 사후학습 레시피와 같은 데이터로 학습했다고 하며(Sec. 3.2, Tab. 4의 "모델 형식만 다름"), 이 대조가 백서의 핵심 비교입니다.

## 2. 구성 요소별 출처

| 구성 | 백서의 인용 | 확인 방법 |
|---|---|---|
| 백본 비디오 DiT | Cosmos-Predict2.5-2B (NVIDIA et al., 2025a) | HF `nvidia/Cosmos-Predict2.5-2B` 존재 확인 (메타데이터만) |
| 영상 VAE | Wan 2.1 (Wan Team, 2025); Turbo는 Wan 2.2 | 백서 인용만 확인 |
| 언어 VLM | Cosmos-Reason1 (NVIDIA et al., 2025b) | N1.7 config는 `Cosmos-Reason2-2B`를 씀 (차이) |
| 사후학습 RL | GRPO (Shao et al., 2024), CPS 샘플링 (Wang and Yu, 2025), RTC (Black et al., 2025) | 인용만 확인 |
| 세계 모델 RL | DreamDojo (Gao et al., 2026) | 인용만 확인 (공개 여부 미확인) |
| 데이터 엔진 | Flora (Leng et al., 2026), WGO-Bench (Macrodata Labs, 2026) | 미공개 |
| 사람 데이터 | Human-1M, HF `nvidia/PhysicalAI-Human-1M-100K` (CC-BY-4.0) | 공개 100k 시간 부분집합만 (§4) |
| 원격조작 데이터 | ABC-130k (XDOF, 2026, HF `XDOF/ABC-130k`), DROID (Khazatsky et al., 2024) | 외부 공개 데이터 |
| 시뮬레이터 | Isaac Sim (Apache-2.0), IsaacLab-Arena, Newton, MuJoCo Warp | 공개 소프트웨어 |
| 기준 모델 | π0.5 (PI 2025), GR00T 1.7, FastWAM (Yuan 2026), MolmoAct2 (Fang 2026), FLUX 3 Action (Black Forest Labs 2026), LingBot-VA (Li 2026) | Tab. 1–3, 19 |

## 3. 가까운 대안 (이 저장소 안의 분석과 연결)

| 폴더 | 논문 | 관계 |
|---|---|---|
| `openwam/` | OpenWAM (arXiv 2609.07398): 영상 생성 prior + ActionDiT | WAM 설계 선택을 부품 단위로 비교한 같은 계열. GR00T 2의 "영상 공동 예측"과 직접 비교 가능 |
| `agentic_wam/` | AW-0 (arXiv 2610.30136): 영상과 행동을 한 Transformer에서 함께 denoise | 공동 생성 WAM의 다른 구현 |
| `thinkworld/` | ThinkWorld (arXiv 2610.37666): 미래 예측 보조 손실 | 미래 예측을 보조 손실로만 쓰는 경우. GR00T 2는 생성 자체를 쓴다는 점이 다름 |
| `dreamx/` | DreamX (arXiv 2611.05377) | 세계 모델과 정책 결합 |
| `pacevla/` | PaceVLA (arXiv 2610.25961): GR00T N1.5 action expert 위에 decoder 4개 | GR00T 계열 VLA 확장 |
| `imle_vla/`, `adavla/`, `motion_latent_action/`, `uvt_visuomotor/` | 행동 생성기·잠재 행동 VLA | 행동 헤드 대안. GR00T 2는 행동 헤드를 영상 토큰과 같은 트랜스포머에 둠 |
| `ultra_humanoid/` | ULTRA (arXiv 2603.03279): 휴머노이드 전신 제어 | GR00T 2 Turbo + SONIC 조합과 비교할 대상 |
| `restacvla/` | ResTacVLA (arXiv 2607.03387) | 촉각 VLA. GR00T 2에는 촉각이 없음 |

## 4. GitHub, Hugging Face, 웹 자료 (확인일 2026-11-12, 메타데이터만)

| 자료 | 라이선스 | 별·좋아요 | 마지막 갱신 | 상태 |
|---|---|---|---|---|
| 백서 PDF (`research.nvidia.com/labs/gear/gr00t-2/assets/gr00t2_whitepaper.pdf`) | © 2026 NVIDIA, All rights reserved (표지 쪽 하단) | — | 2026-11 (PDF 생성 2026-11-09) | 원문 135쪽, 23.5 MB |
| 프로젝트 페이지 `research.nvidia.com/labs/gear/gr00t-2/` | 페이지 라이선스 표기 없음 | — | 2026-11 | 버튼: "Code Available in December" (코드 12월 공개 예정) |
| [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) | Apache-2.0 | ★8,355, 포크 1,534 | 커밋 `3b49067` 2026-11-10 | 설명은 "N1.7". GR00T 2 코드·태그 없음. 최신 릴리스 `n1.6.1-release`(2026-04-23) |
| HF `nvidia/GR00T-N1.7-3B` | nvidia-open-model-license (`other`) | ♥188, 다운로드 58,508 | 2026-11-10 (sha `373d34e`) | 공개 중. config는 `analysis/meta/gr00t_n1d7_config.json` |
| HF `nvidia/GR00T-N1.6-3B` | 라이선스 미기재 | ♥93 | 2025-12-15 | 공개 중 |
| HF `nvidia/PhysicalAI-Human-1M-100K` | CC-BY-4.0 | ♥40, 다운로드 5,447 | 2026-11-07 (sha `7e312e6`) | 100,000 시간 공개 부분집합, 3,405,573 에피소드 |
| HF GR00T 2 모델 (`nvidia/GR00T-2*`, `nvidia/GR00T-N2*`) | — | — | 검색 결과 없음 | **공식 가중치 없음** (커뮤니티 `GR00T-2` 이름 저장소는 모두 비공식) |
| arXiv 제목 검색 "GR00T 2" | — | — | — | 0건 (2026-11-12) |

메타데이터 수집: `analysis/fetch_meta.py` → `analysis/meta/gr00t2_meta.json`.

**공개 상태 판정:** 2026-11-12 기준으로 GR00T 2의 가중치, 학습·추론 코드, 데이터 레시피는 공개되지 않았습니다. 프로젝트 페이지의 "Code Available in December"가 가장 최근 공식 일정입니다. 백서 Tab. 13 주석의 "released checkpoint"는 이 상태와 맞지 않아 `04`에서 불일치로 적었습니다.

## 5. GR00T N1.7 공개 config와 GR00T 2 백서의 구조 대조 (정적 비교)

`analysis/meta/gr00t_n1d7_config.json`(2 KB)에서 가져온 값과 백서 값을 나란히 놓았습니다. 두 모델은 다른 아키텍처이며, 같은 축에서 비교만 합니다. 어느 쪽도 실행하지 않았습니다.

| 항목 | GR00T N1.7 (공개 config) | GR00T 2 (백서) |
|---|---|---|
| 모델 이름 | `Gr00tN1d7` | (이름 미정, 백서 본문은 "GR00T 2") |
| 언어·시각 백본 | `model_name: nvidia/Cosmos-Reason2-2B`, `tune_llm: true`, `tune_visual: true` (미세조정) | Cosmos-Reason1(동결), 영상 DiT는 학습 |
| 중간층 | `select_layer: 16`, `use_vl_self_attention` 4층 | 영상과 행동 토큰이 한 시퀀스 |
| 행동 헤드 | DiT 32층, 헤드 32개 × 48차원, `output_dim` 1024 | 같은 트랜스포머의 행동 토큰(출력 디코더는 임베디드먼트별) |
| 행동 호라이즌 | `action_horizon: 40` | 백서에 일반값 없음. 설정마다 다름 (UMI 직접 학습은 48스텝 중 30 실행, App. E.2; DexUMI 후처리는 40스텝 stride 2, App. E.5) |
| 추론 단계 | `num_inference_timesteps: 4` | 영상 1–2, 행동 2–4 |
| 입력 해상도 | `image_target_size: [256, 256]` (잘라내기 230) | 224×320, 카메라 최대 4대 |
| 임베디드먼트 수 | `max_num_embodiments: 32`, `max_action_dim: 132` | 임베디드먼트별 MLP(Sec. 2), 수는 백서 미기재 |
| 정밀도 | `dtype: bfloat16`, `use_flash_attention: true` | BF16 학습, 서빙은 TensorRT·FP8 (Sec. 4.6.1) |
| 학습 목표 | 행동 flow matching | 영상 + 행동 flow matching |
| 파라미터 수 | 이름에 3B | 백서는 "2B-parameter transformer" |

**읽는 법:** GR00T 2는 N1.7의 "VLM + 행동 헤드" 구조를 버리고, 영상 생성 백본 하나로 행동까지 생성합니다. 그래서 N1.7 config의 `tune_llm`, `select_layer` 같은 항목은 GR00T 2에 대응하는 값이 없습니다.

## 6. 데이터 공개 범위

| 데이터 | 백서 규모 | 공개 상태 |
|---|---|---|
| Human-1M 사전학습 필터본 | 402,730 시간 | 미공개 (필터 기준과 목록은 App. C.1) |
| Human-1M 100K 공개 부분집합 | 100,000 시간 (HF, CC-BY-4.0) | 공개 |
| 스튜디오 사람 데이터 3,000 시간 | — | 미공개 |
| UMI-10K | 10,184 시간 | 미공개 (백서 App. C.2에 통계만) |
| 원격조작 (YAM, Sharpa, Wuji, G1) | 23.5k 시간 | 대부분 미공개. ABC-130k(XDOF)는 HF 공개 |
| 평가 원자료 (A/B 아레나, 실물 시도 로그) | 16,044 롤아웃 | 미공개 (집계 표만) |

## 7. 이 저장소의 다른 WAM·VLA와 비교할 때 쓸 기준

- **같은 정도의 공개성:** GR00T 2는 가중치와 코드가 없습니다. 같은 계열인 OpenWAM과 AW-0는 공식 코드를 공개했습니다(각 폴더 README 기준). GR00T 2는 코드가 아직 없어 재현 가능성이 크게 낮습니다(`openwam/`, `agentic_wam/`의 분석 참조).
- **공동 예측의 증거:** Fig. 10의 사람 영상 비율 실험은 **영상 목표가 있는 모델**의 실험이라 WAM 효과만 따로 떼어 보기는 어렵습니다. GR00T 2 Turbo의 H.3 실험(시뮬레이션 데이터, WAM 대 VLA)이 가장 직접적인 대조인데, 시도 수가 30회라 결론이 약합니다(`03` §4).
