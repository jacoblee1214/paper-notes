# Cosmos-Predict2.5 논문과 공식 코드 대조 (D1–D22)

대상: `repo/cosmos-predict2.5/` (커밋 `a2c298b`, 2026-06-08), `repo/cosmos-transfer2.5/` (커밋 `2ff49d0`, 2026-06-07). 두 저장소 모두 `--depth 1`, 정적 읽기만 했습니다. 설치, 실행, 가중치 다운로드는 하지 않았습니다. 저장소 내용은 신뢰하지 않는 자료로 다뤘습니다.

**주의:** 이 대조는 **공개된 설정 파일**을 기준으로 합니다. 논문 실험에 쓴 최종 설정이 저장소에 그대로 있는지는 확인할 수 없습니다. 같은 파일 안에도 여러 단계(사전학습, SFT, 증류, 행동 조건)의 설정이 있어서, 여러 값이 보일 때는 어느 단계인지 함께 적었습니다.

## 1. 대조 결과

| # | 항목 | 논문 | 코드 (파일:줄) | 판정 |
|---|---|---|---|---|
| D1 | 2B 층수 | 표 3: 32층, 14B 36층 | `cosmos_predict2/_src/predict2/configs/video2world/defaults/net.py` 의 `COSMOS_V1_2B_NET_MININET`: `num_blocks = 28`, 14B는 36 (일치) | **불일치 (2B)**. 14B는 일치. §6.1의 "7블록 간격 4개 = 28"과도 맞음 |
| D2 | AdamW β, weight decay | β2 = 0.999, wd 0.001 (§4.1) | `.../reason_embeddings/model_2B_reason_1p1_rectified_flow.py:291–293`, `.../action/configs/action_conditioned/experiment/exp_2B_action_conditioned_rectify_flow.py:286–288`: `weight_decay=1e-3`, `betas=[0.9, 0.999]` | 일치 (실험 설정). 기본값(`cosmos_predict2/_src/predict2/configs/common/defaults/optimizer.py:24–26`)은 lr 1e-4, wd 0.1, β2 0.99로 다름 (실험 설정이 덮어씀) |
| D3 | 학습률 | 2B 3e-5, 14B 1.3e-5 (§4.1) | 2B: `lr=3e-5` (`model_2B_reason_1p1_rectified_flow.py:291`), 일치. 같은 디렉터리 다른 실험: `lr=2 ** (-14.5)` (`:113`, 주석은 "4.3158e-05"). 행동 조건 파일 `exp_2B_..._rectify_flow.py:111`: `lr=2 ** (-14.5)`인데 주석은 "3.0517578125e-05" (그건 2^-15). 14B RF 설정 `model_14b_reason_1p1_rectified_flow.py` 등: `lr=2 ** (-14.5)` (≈ 4.3e-5). 14B의 1.3e-5는 찾지 못함 | **부분 일치 (2B), 불일치 (14B)**. 주석과 값이 서로 다른 곳도 있음 |
| D4 | 워밍업 | 2000 스텝 (사전학습) | 사전학습 단계 `warm_up_steps=[2_000]` (`model_2B_reason_1p1_rectified_flow.py:119`, 14B `model_14b_reason_1p1_rectified_flow.py:317`), 후속 단계 `[100]` (`:298`) | 일치 (사전학습). 후속 단계는 다름 |
| D5 | 시간 이동 β | 1 (256p)에서 5 (720p)까지 **점진적** (§4.1) | 제품 설정 RF: `shift=5`, `use_dynamic_shift=False` (`model_2B_reason_1p1_rectified_flow.py:309–310`). 256p 단계에서 β=1로 바꾸는 설정은 찾지 못함 | **부분**. 720p 값(5)만 확인, 점진 스케줄은 확인 못함 |
| D6 | 고잡음 샘플 5% | 가장 높은 2% 잡음에서 5% 샘플 (§4.1) | `cosmos_predict2/_src/predict2/models/text2world_model_rectified_flow.py:104–107, 754–762`: `high_sigma_ratio = 0.05`, 구간 `980–1000` (1000 중 20 = 2%). RF 설정 `use_high_sigma_strategy=True` (`model_2B_reason_1p1_rectified_flow.py:477`) | **일치** |
| D7 | 로짓 정규 분포와 식 (4) | logit-normal, t_s = βt/(1+(β−1)t) | `train_time_distribution="logitnormal"` (`text2world_model_rectified_flow.py:101`). 이동 식 `sigmas = shift*s/(1+(shift−1)*s)` (`cosmos_predict2/_src/predict2/models/fm_solvers_unipc.py:106`) | **일치** (식 4와 같은 형태) |
| D8 | 텍스트 인코더 | Cosmos-Reason1의 여러 블록을 이어 붙여 1024로 사영 (§3.2) | 제품 설정: `embedding_concat_strategy=FULL_CONCAT` (`model_2B_reason_1p1_rectified_flow.py:170`), `crossattn_proj_in_channels=100352` (= 28 × 3584), `crossattn_emb_channels=1024` (`:146–148`) | **일치**. 참고: 인코더 클래스의 기본값은 `MEAN_POOLING`, `n_layers_per_group=5` (`text_encoders/text_encoder.py:44–45`). 제품 설정이 덮어씀 |
| D9 | 절대 위치 임베딩 제거 | 상대 위치(3D RoPE)만 (§3.2) | `configs/video2world/defaults/net.py`: `pos_emb_cls="rope3d"`, 2B/14B 설정에서 `extra_per_block_abs_pos_emb = False` (`:86, :93`). 7B 기본은 `True` (`:76`) | **일치** (2.5의 2B/14B). 7B(Predict1)에는 절대 위치가 남아 있음 |
| D10 | 시각 토크나이저, 패치 | WAN2.1 VAE, 시간 4 × 공간 8 × 8 압축, 패치 1×2×2 (§3.2) | `patch_spatial=2`, `patch_temporal=1` (`defaults/net.py:64–65`). 토크나이저 `Wan2pt1VAEConfig` 등록 (`configs/common/defaults/tokenizer.py`) | 패치: **일치**. 압축 비율 4×8×8은 설정에서 직접 확인하지 못함 (VAE 구현 내부) |
| D11 | 93프레임 → 24 잠재 프레임 | §3.2, §4.1 | `num_video_frames=93` (`exp_2B_..._rectify_flow.py:247, 415, 517`) | **일치** |
| D12 | RL (VideoAlign + GRPO), 보상 정규화 | §4.2.2: 입력당 8개 샘플, 256 스텝, 배치 32, 확산 손실 정규화 | 저장소에 RL 학습 코드가 없음. `grpo`와 `videoalign` 검색에는 `Stage-c_GRPO-reason_embeddings-...` 체크포인트의 S3 로드 경로만 나옴 (교사 경로와 `checkpoints_predict2.py`, `camera/.../exp_2b.py`) | **미검증** (코드 없음). 보고서의 v1·v2 정규화 설명도 다름 (`01` §0) |
| D13 | 4스텝 증류 | rCM (연속시간 CM + DMD 하이브리드), 4스텝 (§4.2.3) | 증류 실험 설정 `predict2/distill/configs/experiment/experiments_dmd2_predict2p5.py`: DMD2 중심. `rcm`은 콜백 이름과 주석 처리된 시각화만 있음 (`callbacks.py:26, 64`, `experiments_dmd2_predict2p5.py:71`). README도 "DMD2 distillation"이라고 씀 | **불일치 (방법 이름)**. rCM 구현은 확인 못함. 4스텝 자체는 설정에서 확인 못함 |
| D14 | 모델 병합 (soup, TIES, DARE) | §4.2.1, 그림 4 | 병합 코드를 찾지 못함 (`merge` 검색은 어텐션 모듈 내부 용어만 나옴) | **미검증** |
| D15 | 텍스트 조건 드롭아웃 | 논문에 없음 | 제품 RF 설정: `text dropout_rate=0.2` (`model_2B_reason_1p1_rectified_flow.py:155`) | 코드에만 있음 (정보) |
| D16 | 행동 조건: 행동 임베더를 시간 임베딩에 더함 | §6.6: 행동 MLP 출력을 DiT 시간 임베딩에 더함 | `cosmos_predict2/_src/predict2/action/networks/action_conditioned_minimal_v1_lvg_dit.py:68–75, 112–141`: `action_embedder_B_D`, `action_embedder_B_3D`를 `t_embedding`과 `adaln_lora`에 더함 | **일치** |
| D17 | 행동 주입 대안 (교차 주의, 채널 이어 붙이기) | 표 20 비교 | 행동 파일에 두 대안을 고르는 설정을 찾지 못함 | **미검증** (코드에 없음 또는 미공개) |
| D18 | 행동 조건 추론 | 표 19 | 문서: "Action conditioned inference does not yet support multi-GPU", `context_parallel_size=1` (`docs/inference_robot_action_cond.md`) | 정보 (CP 없음). 예제 명령과 JSON 구조 확인 |
| D19 | Ulysses형 컨텍스트 병렬 | §4.3 | `cosmos_predict2/_src/predict2/networks/a2a_cp.py`(all-to-all) | 일치 (파일 이름 수준) |
| D20 | 로봇 체크포인트 이름 | 표 1 | `docs/inference_robot_action_cond.md`, `docs/inference_robot_multiview-agibot.md`, `docs/post-training_video2world_gr00t.md` 존재. README 표가 `robot/action-cond`, `robot/multiview-agibot`, `14B/robot/gr00tdream-gr1`를 같은 이름으로 나열 | **일치** |
| D21 | 라이선스 | 초록: 소스 코드, 체크포인트, 벤치마크를 NVIDIA Open Model License로 공개 | 코드: `LICENSE` = Apache-2.0 (두 저장소 모두). README: "NVIDIA Cosmos source code is released under Apache 2.0", "models under NVIDIA Open Model License" | **불일치 (초록 표현)**. 코드는 Apache-2.0, 가중치만 Open Model License |
| D22 | Transfer2.5 제어 블록 배치 | §6.1: 블록 4개를 7블록마다 | `cosmos_transfer2/_src/transfer2/networks/minimal_v4_lvg_dit_control_vace.py:538–597`: `control_layers = range(0, num_blocks, vace_block_every_n)` (VACE형). 실험 설정 기본은 2, 일부 4 (`configs/vid2vid_transfer/experiment/exp_large_scale.py:236, 327`). 2B 릴리스의 7은 찾지 못함 | **부분**. 구조는 맞지만 7이라는 값은 확인 못함 |

## 2. 로봇 관련 항목 요약

- **행동 조건 (D16, D18, D20)** 은 보고서와 코드가 맞습니다. 세부 설정 (코드 기본 `action_dim`이 `10 * 8`인 것 등)은 Bridge 설정과 같은지 확인하지 않았습니다.
- **증강 (표 13, §6.2)** 에 쓴 Transfer 모델의 학습 설정(로봇 다시점 Transfer, 엣지와 블러 제어)은 `cosmos_transfer2/` 아래 로봇 설정으로 있을 것입니다. 이 분석에서는 확인하지 않았습니다.
- **정책 학습 (§6.2)** 의 확산 정책은 이 저장소에 없습니다. 논문이 인용한 Diffusion Policy 구현을 따랐을 것으로 보이지만 확인하지 못했습니다.
- **RL 보상 (D12)**, **병합 (D14)**, **rCM (D13)** 은 재현에 필요한 코드가 저장소에 없거나 다른 방법입니다.

## 3. 저장소의 자체 상태

- 두 저장소의 README 첫머리는 Cosmos 3 공지입니다. "이 저장소는 더 이상 활발히 개발되지 않으며, 제한적 유지보수만 합니다." 저장소 설명은 Cosmos 3으로 옮겨 가라고 말합니다(`02` §4).
- README 릴리스 노트로 보면 2026년 2월에 Robot/Policy(LIBERO, RoboCasa) 체크포인트와 그 후학습 레시피가 나왔습니다. 논문에는 없는 모델입니다.
- `pyproject.toml`과 `uv.lock`이 있고, Docker 파일도 있습니다. 이 분석에서는 설치하지 않았습니다.

## 4. 구현 가능성 판정

**판정: 이 맥에서는 공식 코드를 실행할 수 없습니다. 설치 요구 조건에서 막힙니다. 정적 대조와 재계산은 완료했습니다.**

| 항목 | 요구 사항 (저장소 `docs/setup.md`) | 이 맥 (Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음) |
|---|---|---|
| OS, 아키텍처 | Linux x86-64, glibc ≥ 2.35 | macOS arm64 → **불충족** |
| GPU | NVIDIA Ampere 이상 (RTX 30, A100 이상) | 없음 → **불충족** |
| 드라이버, CUDA | 드라이버 ≥ 570.124.06, CUDA 12.8.1 | 없음 → **불충족** |
| 다시점 추론 | 최소 8 GPU, 각 80 GB (`docs/inference_auto_multiview.md`) | 불가 |
| 행동 조건 추론 | 단일 GPU (다중 GPU 미지원). 필요 VRAM은 문서에 없음 | 불가 (CUDA) |
| 2B 로컬 추론 | 필요 VRAM 명시 없음 | 불가 (CUDA). MPS 포팅은 별도 작업 |
| 학습 (논문 값) | H100 4096장 (표 9) | 불가 |

- **무엇이 가능한가:** 정적 읽기, 설정 대조(완료), 논문 표 재계산(완료), 행동 조건 확산 구조의 작은 CPU 재구현 (재구현은 가능하지만 이 보고서의 재현은 아님).
- **노력 추정 (측정 전):**
  - CUDA 기기에서 설치와 2B 행동 조건 추론 smoke test: 반나절에서 1일 (추정). Bridge 100 에피소드 추론 시간은 측정 안 함.
  - 표 19 재현 (100 에피소드, 두 모델 비교): GPU 기기 1–2일 (추정). 평가 지표 구현 포함.
  - 표 13 재현 (실로봇): 로봇 하드웨어가 필요해서 이 저장소 밖 (불가).
  - 정책 평가 실험 (C17 검증): 정책 여러 개와 시뮬레이터 또는 실로봇 데이터가 필요해서 수주 (추정, 자원 의존).
  - 세계 모델 재구현 (DiT, flow matching, 행동 임베더): 수주에서 수개월, 데이터와 GPU 전제 (추정).

## 5. 보고서 밖의 한계 (코드 쪽)

- `robot/action-cond` 추론은 단일 GPU 전제입니다. 긴 자기회귀 롤아웃의 누적 오차를 재는 설정은 확인하지 못했습니다.
- 보고서가 쓴 "Cosmos-Predict2.5-2B/robot/action-cond" 체크포인트는 Bridge(7차원 행동) 기반입니다. 로봇 다른 데이터셋으로 바꿀 때 행동 차원 설정을 바꿔야 합니다(코드 기본 `action_dim`은 `10 * 8`, D16 참조).
- 보고서 이후 생긴 Cosmos 3과의 비교 결과가 필요합니다. 이 보고서만으로는 2.5를 기준선으로 고정할 수 없습니다.
