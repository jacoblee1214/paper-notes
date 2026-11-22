# 선행 연구, 계보, 자료 (Wan, arXiv:2503.20314)

분석일 2026-11-22. 공식 코드는 얕은 clone(`repo/`, HEAD `9737cba`, 2026-03-05)으로 정적으로만 읽었습니다.

## 1. 계보: 이 논문이 기대는 것

| 축 | 선행 연구 | 논문에서 쓰는 방식 | 출처 |
|---|---|---|---|
| 백본 | DiT (Peebles & Xie, 2023), U-Net 계열 영상 확산 (VDM, Ho et al. 2022) | DiT 기반, 텍스트는 교차 어텐션, 시공간 전체 어텐션 | §2, §4.2.1 |
| 목표 | Flow matching (Lipman et al. 2022), rectified flow (Esser et al. 2024, SD3) | 속도 예측 MSE, logit-normal 시간 샘플링 | §4.2.2, 식 (1)–(3) |
| 텍스트 | T5 계열, CLIP, LLM 인코더 (HunyuanVideo) | umT5 선택, 소거로 Qwen2.5·GLM-4·Qwen-VL과 비교 | §4.2.1, §4.7.2 |
| 영상 VAE | SVD (1×8×8), CogVideoX, Mochi (6×8×8), HunyuanVideo, LTX-Video | 3D 인과적 VAE, 4×8×8, 16채널, 특징 캐시 | §4.1, 그림 7 |
| 가속 | DiTFastAttn, FasterCache (캐시), SageAttention (INT8 어텐션), DeepSpeed Ulysses, Ring Attention, USP | 확산 캐시, FP8 GEMM, INT8-FP8 어텐션, 2D 컨텍스트 병렬 | §4.3.2, §4.4 |
| 조건 제어 | ControlNet, I2V-Adapter, VACE (Jiang et al. 2025, 같은 팀) | I2V는 마스크 채널 연결과 CLIP 디커플드 교차 어텐션, VACE는 VCU 통합 | §5.1, §5.2 |
| 스트리밍 | LCM, VideoLCM, Streamer (슬라이딩 창) | 4단계 일관성 증류와 창 기반 스트리밍 | §5.6 |
| 평가 | VBench (Huang et al. 2023), FVD·FID의 한계 지적 | VBench 리더보드 점수 인용, 자체 Wan-Bench(14지표) | §4.6, §4.7 |

### 1.1 closed 대 open 계열

- 상용(논문 시점): Sora (2024.02), Kling, Luma, Runway Gen-3, Vidu, Hailuo, Pika 1.5, Movie Gen, Veo 2 (§2).
- 오픈: Mochi, CogVideoX, HunyuanVideo, LTX-Video, Open-Sora-Plan, Step-Video, SVD (§2, 그림 1, 표 2–4 비교 대상).
- 논문의 자기 위치: 공개 모델 중 최상위, 상용 모델과 "경쟁 또는 우위" (§2 마지막 문단).

### 1.2 가까운 대안 (구조 비교)

| 모델 | 텍스트 조건 | 영상 압축 | 비고 |
|---|---|---|---|
| Wan 2.1 (이 논문) | umT5, 교차 어텐션 | 4×8×8, 16채널 | 1.3B과 14B, 공개 추론 코드 |
| HunyuanVideo | MLLM 텍스트 인코더, MM-DiT (결합 어텐션) | 논문 표기 없음(그림 7 비교) | 논문이 자주 비교하는 오픈 대안 |
| CogVideoX | T5, 교차 어텐션 | 4×8×8 (논문 비교) | 표 4의 CogVideoX1.5-5B |
| Mochi | T5 | 6×8×8, 잠재 12 | 그림 7 비교 |
| Step-Video | — | 8×16×16, 잠재 64 | 그림 7 비교 |

## 2. 이 논문이 기여하는 것과 아닌 것

- **기여:** (a) 공개 가능한 3D 인과적 VAE 설계와 특징 캐시, (b) 해상도와 길이를 단계로 올리는 학습 순서, (c) 1.3B의 소형 효율, (d) I2V, 편집, 개인화, 카메라, 실시간, V2A까지 하나의 계열로 묶은 것, (e) 평가 체계(Wan-Bench, 사람 평가).
- **기여가 아닌 것(또는 근거가 얇은 것):** 일반적인 DiT와 flow matching 자체, 확산 캐시와 FP8의 원리, 실시간 증류 (LCM은 기존 방법).
- **공개 범위:** 추론 코드와 가중치 (Apache-2.0). 학습 코드, 데이터, Wan-Bench 평가 코드와 데이터, 실시간·V2A·개인화·카메라 코드는 저장소에 없습니다(`04`).

## 4. GitHub, 프로젝트 페이지, 가중치 자료 (확인 2026-11-22)

| 자료 | 링크 | 라이선스 | 별 / 마지막 활동 | 비고 |
|---|---|---|---|---|
| **공식 코드 (Wan2.1)** | https://github.com/Wan-Video/Wan2.1 | Apache-2.0 | 17,473 / 2026-03-05 | `repo/` 얕은 clone (HEAD `9737cba`, 2026-03-05). 추론만 있음. 학습 코드 없음 (`04` D1) |
| 후속 코드 (Wan2.2) | https://github.com/Wan-Video/Wan2.2 | Apache-2.0 | 18,817 / 2026-09-21 | 이 논문의 후속. 클론하지 않음, 이 분석의 대상 아님 |
| 프로젝트 페이지 | https://wan.video/ | — | — | 응답 200. 제품 소개 페이지이고 이 논문 전용 페이지는 찾지 못함. arXiv 초록에는 프로젝트 링크가 없음 |
| HF 가중치 (1.3B T2V) | https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B | Apache-2.0 (메타데이터) | 좋아요 690, 다운로드 97,816 (메타데이터) | 받지 않음 (약 2.6 GB, bf16 추정) |
| HF 가중치 (14B T2V) | https://huggingface.co/Wan-AI/Wan2.1-T2V-14B | Apache-2.0 (메타데이터) | 최종 수정 2025-03-12 | 받지 않음 (bf16 약 26.6 GiB, 추정) |
| 학술 상세 | arXiv:2503.20314 (cs.CV) | non-exclusive 1.0 | v1 2025-03-26, v2 2025-04-19 | 학회 표기 없음 (기술 보고서) |

- 커뮤니티 확장 (README 언급): DiffSynth-Studio (FP8, LoRA, VRAM 최적화), LightX2V (가속). 이 분석에서는 확인하지 않았습니다.
- 저장소의 추론 코드는 `generate.py` 하나로 T2V, I2V, FLF2V, VACE, T2I를 구동합니다(`repo/generate.py`, `repo/wan/`).

## 5. 이 논문의 위치 한 문단

Wan은 2025년 초 공개 오픈 영상 모델 가운데 평가 지표 최상위권에 있었고, 이후 WAM 계열 논문(Fast-WAM, OpenWAM, WAMotor, DreamZero)의 기본 영상 백본으로 자주 쓰였습니다. 그래서 이 저장소에서 Wan은 독립된 생성 모델이라기보다 **로봇용 세계 행동 모델(WAM)의 백본**으로 읽는 것이 실용적입니다. 추론 코드는 공개되었지만 학습과 평가 코드는 공개되지 않아, 로봇 데이터로 파생 모델을 만들려면 별도 구현이 필요합니다.
