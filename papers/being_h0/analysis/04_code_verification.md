# 04. 공식 코드 대조와 구현 가능성

## 0. 대상과 방법

- **대상 저장소:** `github.com/BeingBeyond/Being-H0`, 기본 브랜치 `main`, 커밋 **`ae92e46f671f48d91161ef1c8cc956bcdffab606`** (2026-05-04, "Update README"). 얕은 복제는 `repo/`에 있고 `.gitignore`에 들어 있습니다.
- **방법:** 정적 읽기만 했습니다(`grep`, `sed`). 설치, 실행, `import`, 가중치와 데이터 다운로드는 하지 않았습니다. 다운로드는 Hugging Face 설정 파일(JSON, 수 KB)과 라이선스 파일만 받았습니다(`04` §1).
- **읽은 파일:** `beingvla/models/vla/training_mixins.py`, `being_vla_model.py`, `config.py`, `generation_mixins.py`, `beingvla/models/motion/mano_adapter.py`, `m2m/tokenizer/config.py`, `m2m/tokenizer/model.py`, `m2m/tokenizer/quantizer/part_grvq.py`, `beingvla/dataset/datasets.py`, `dataset_internvl.py`, `beingvla/inference/vla_internvl_inference.py`, `m2m/aligner/eval_policy.py`, `run_server.py`, `environment.yml`, `README.md`.
- **경로 표기:** 아래 `repo/…` 경로와 줄 번호는 위 커밋 기준입니다.
- **HF 설정 파일 출처:** `BeingBeyond/Being-H0-GRVQ-8K/{wrist,finger}/motion_args.json`, `BeingBeyond/Being-H0-8B-2508/config.json`, `Being-H0-8B-Align-2508/config.json` (2025-08-02 버전).

## 1. 설치 요구사항과 라이선스

| 항목 | 내용 | 출처 |
|---|---|---|
| 코드 라이선스 | **MIT** (LICENSE 파일, `Copyright (c) 2025 BeingBeyond`). GitHub API도 MIT | `repo/LICENSE`, API |
| 환경 | conda `beingvla`, Python 3.10.18, 중요 패키지는 `environment.yml`에 고정 | `repo/environment.yml` |
| 추가 설치 | `pip install flash-attn --no-build-isolation`, `manotorch`(git), `chumpy`(git) | `repo/README.md` Setup |
| MANO | 별도 등록 후 다운로드. "MANO 라이선스" 적용, `beingvla/models/motion/mano/`에 배치 (저장소에는 `.gitkeep`만 있음) | `repo/README.md`, `repo/beingvla/models/motion/mano/.gitkeep` |
| 토크나이저 가중치 | `BeingBeyond/Being-H0-GRVQ-8K`, MIT, **0.21 GB** (wrist 104.1 MB, finger 105.8 MB). 설정 JSON은 받음 | HF API, `grvq_*_motion_args.json` |
| 사전학습 VLA | `Being-H0-1B-2508` **2.37 GB**, `-8B-2508` **16.05 GB**, `-14B-2508` **30.86 GB**. 모두 MIT | HF API |
| 후학습 체크포인트 | `Being-H0-8B-Align-2508` **16.05 GB**, MIT | HF API |
| 후학습 데이터 | `BeingBeyond/h0_post_train_db_2508`, **1.17 GB, 10,820 파일**, 태그 MIT, 원천 데이터 라이선스는 설명에 없음 | HF API |
| 프로젝트 페이지 | CC BY-SA 4.0 (사이트 템플릿 표기). 논문 라이선스가 아님 | 사이트 하단 |

- **원천 데이터 라이선스는 확인하지 못했습니다.** 저장소와 모델은 MIT지만, UniHand 원천(EgoDex, TACO, OakInk2, ARCTIC 등)은 각자 라이선스가 따릅니다. 원천 데이터를 쓰는 재학습 결과를 공개하기 전에 각각 확인해야 합니다.

## 2. 논문 ↔ 코드 대조 (D 항목)

| ID | 논문 | 공식 코드 | 판정 |
|---|---|---|---|
| D1 | 사전학습과 후학습 절차를 자세히 씀 (§4.1.4, §6.1.1: AdamW 1e-5, 배치 128, A800 32장) | **학습 코드와 스크립트 없음.** README 체크리스트 "Training code and scripts"가 미완료(`- [ ]`). 학습 데이터 경로는 `dataset_internvl.py:170`의 내부 절대 경로 `/share/dataset/UniHand/` | **코드 없음**. 학습 설정 대조 불가 |
| D2 | 토크나이저 손실 L = L_recon + λ1 L_commit + λ2 L_wrist, λ1 0.02, λ2 1.0 (식 6) | `m2m/tokenizer/config.py:69–71`: `recons_loss` 기본 `l2`, 공개 설정 JSON은 **`l1_smooth`**, `commit_weight` 0.02 (일치), **`vel_weight` 1.0** (속도 손실 항, 논문 식 6에 없음) | **차이:** 속도 손실과 `l1_smooth`는 논문에 없음. 학습 루프가 없어 실제 합은 확인 못함 |
| D3 | 15 FPS, 1초 창 (§4.1.4) | 공개 설정 `window_size` 15 (일치), `down_t` 2 → 4 단계. 그러나 같은 JSON의 **`fps: 20`** (`config.py:15` 기본 20). 창 길이가 맞으므로 `fps`는 쓰이지 않는 메타데이터로 보임 | 대체로 일치. 메타데이터 불일치 |
| D4 | 부분별 토크나이저, 손목과 손가락 분리 (§4.1.2) | `m2m/tokenizer/model.py:101–119`: 부분 인덱스 설정은 `"mano" not in motion_feat`과 **`mano114` 분기만** 있음. **`mano162`(논문 기본값) 분기가 없음**. 추론은 부분별 체크포인트를 따로 읽어 문제가 없음 | **학습 경로의 D162 부분 분할은 확인 불가** |
| D5 | 128 토큰/초 (2 부분 × 2 그룹 × 8층 × 4) | 계산 일치: `calculate_num_tokens` (`config.py:117–134`), `calculate_codebook_size` (`:136`), 공개 설정 `shared_codebook: true` | **일치** |
| D6 | 어휘 로짓 마스크 P = 50%, 손실 백분위 [15%, 95%] (식 8–10) | 구현은 있음: `training_mixins.py:91–100`(마스크), `126–158`(백분위 필터). 그러나 **값은 외부에서 설정**(`optimize_rate` 기본 0.0, `curr_top_rate` 1.0, `curr_bottom_rate` 0.0). 코드에는 손실 0 제외(`loss > 0.0`)가 **추가**로 있음 | 메커니즘 일치, 수치 확인 불가. 추가 조건 1개 |
| D7 | 소프트 모드: 예측과 GT의 평균 후 재양자화 (§4.1.5) | `generation_mixins.py:259–275`, `_apply_soft_motion_constraints` (`:295`), `mano_adapter.py:322`: `soft_mano = alpha * pred + (1 - alpha) * gt`, 기본 `alpha = 0.5`. **GT 블록이 매 단계 입력됨** | 일치. 표 4 해석 주의 (§3) |
| D8 | 448×448, 동적 타일링, 썸네일, 픽셀 셔플 0.5 (§4.1.3) | `config` 공개 JSON: `force_image_size 448`, `downsample_ratio 0.5`, `use_thumbnail true` | 일치. 학습 전처리에 **JPEG 압축 증강** 있음(`datasets.py:149–152`, `RandomJPEGCompression`) — 논문에 없음 |
| D9 | 약한 원근 정렬, 식 11–12 (scale·translate), 어안 FoV 90° 정규화 (§4.2.1) | 추론 `vla_internvl_inference.py:226–242`: `H = K_target · K_raw⁻¹`를 `cv2.warpPerspective`에 씀(회전 없으면 식 11–12와 같은 꼴). 목표 K는 f = 736.6339, 주점 (960, 540)으로 데이터셋과 같음(`dataset_internvl.py:175`). **원본 K 기본값은 하드코딩** `[[2304, 0, 1729], [0, 1942, 971], …]` (`:233`). 어안 정규화 코드는 없음. 학습 데이터는 이미 정렬된 `aligned_raw_videos`로 읽음 | 식은 일치, 정렬 전처리 코드는 없음 (사전 정렬). 어안 정규화 **없음** |
| D10 | 시점 분포 균형: 깊이 스케일 λ_s와 1/λ_s 영상 조정, 광축 회전 (식 13–14) | **구현 없음.** 데이터 균형은 학습 파이프라인(미공개)에 있을 것으로 보이나 확인 불가 | **확인 불가** (코드 없음) |
| D11 | 로봇 전이: 프로프리오 MLP f_p, 학습 가능한 쿼리와 헤드 f_r (식 15) | 두 방식이 있음: `gen_action_type = "action_token"`(정렬 체크포인트, `<ACTION_CHUNK_i>` 토큰이 쿼리 역할, `being_vla_model.py:147–152`, `385`)과 `"prop_hidden"`(기본 체크포인트, 한 벡터에서 청크 전체를 냄, `:154–160`). 논문은 한 방식만 설명 | **차이:** 논문은 한 방식, 코드는 두 방식. 정렬 체크포인트는 쿼리 방식 |
| D12 | 행동 차원 (Franka 7 + Inspire 6 = 13, §6.1.3) | 공개 설정 `action_dim` 13, `proprio_dim` 13. 프로프리오 인코더 `Linear(13, H/2) → ReLU → Linear(H/2, H)` (`being_vla_model.py:140–144`), 행동 헤드 `Linear(H, H/2) → ReLU → Linear(H/2, 13)` (`:148–152`) | 일치 (구조와 차원). 실제 로봇 좌표계는 코드에 없음 |
| D13 | 로봇 행동 손실은 L1 (식 15) | `being_vla_model.py:314–343`: `loss_func` 기본 `"l1"`(`config.py:70`). **행동 손실이 LM 손실을 대신함** (`outputs.loss = action_loss`) | 일치 |
| D14 | Franka Research 3, Inspire 손, RealSense, Gello 원격 조작 (§6.1.3) | 로봇 드라이버, Gello, RealSense 관련 코드 **없음**(검색 0건). 정책 평가는 zarr 데이터 경로를 받는 `eval_policy.py`와 `run_server.py`만 있음 | 로봇 코드 **없음** |
| D15 | 행동 청크 길이는 본문에 없음 (식 15의 N_a) | 공개 설정 `action_chunk_length` 16 | 논문에 없는 값 (공식 기본값) |
| D16 | (본문 없음) | 추론 장치: `torch.cuda.is_available()`이 아니면 **CPU**(`vla_internvl_inference.py:169–175`). MPS 경로 없음. dtype **bf16 고정**(`:120`, `:289`). `use_flash_attn` 기본 True이나 `has_flash_attn`이 없으면 끔(`modeling_intern_vit.py:185–187`) | 추론은 CUDA 또는 CPU만. 맥은 CPU로만 가능 |
| D17 | 표 2–6 (생성 지표, FID, M2T) | `m2m/utils/metric.py`에 MPJPE, PA-MPJPE, R-precision, FID 함수가 있으나 **이를 호출해 논문 표를 만드는 스크립트가 없음**. `tmr_model_path`는 빈 문자열, TMR 체크포인트 미공개. `scripts/evaluation/`은 로봇 정책 평가만 | **평가 재생산 불가** |
| D18 | (본문 없음) 데이터 공개 범위 | 후학습 데이터 한 세트만 공개. 테스트 세트는 "라이선스 때문에 저장소에 없다"(README). UniHand 2.5M 전체는 공개되지 않음 | 부분 공개 |

## 3. 평가 코드와 장기 생성 해석

- **표 3–6은 재생산할 수 없습니다.** 지표 함수는 있으나 테스트 분할 생성, TMR 임베딩, FID 계산을 묶은 스크립트가 없습니다(D17). 독립 구현은 가능하지만 분할 정의(5% 보류, head/tail 구성)가 공개되지 않아 값이 그대로 나오지 않을 수 있습니다.
- **표 4(소프트 모드)는 정답 블록을 섞습니다(D7).** 코드는 `_apply_soft_motion_constraints`에서 생성된 블록과 정답 블록을 평균하고 다시 양자화합니다. 따라서 표 4는 자유 롤아웃의 장기 오차가 아니라 정답 주변에서 얼마나 벗어나는지를 잽니다. 표 3(블록 모드)과 직접 비교하면 안 됩니다.

## 4. 구현 가능성 (이 맥: Apple M4 Pro, 24 GiB, MPS, CUDA 없음)

**판정: 학습과 실로봇 재현은 이 맥에서 불가능합니다. 추론과 토크나이저 검사는 조건부입니다. 실행은 사용자 승인 뒤에 해야 합니다.**

| 작업 | 판정 | 근거 |
|---|---|---|
| 사전학습 (UniHand-2.5M, 1B·8B·14B) | **불가** | 학습 코드 미공개 (D1). 논문은 A800 80GB 32장 (§6.1.1) |
| 후학습 정렬 (Align 8B) | **불가** | 같은 이유 (D1). 데이터 1.17 GB는 받을 수 있으나 학습 코드가 없음 |
| 토크나이저 재구성 검사 (GRVQ-8K, 표 5 재구성) | **조건부** | 가중치 0.21 GB, MANO 파일 등록, 테스트 원천 데이터 직접 받기 필요. CPU로 돌 가능성은 있으나 미측정 |
| 1B 추론 (CPU, bf16) | **조건부** | 가중치 2.37 GB. CPU 경로만 있고 MPS 없음 (D16). 속도 미측정 |
| 8B 추론 (CPU, bf16) | **조건부 (무거움)** | 가중치 16.05 GB. 24 GiB 통합 메모리에서 활성 메모리와 함께 빠듯함. 미측정 |
| 14B 추론 (bf16) | **불가** | 가중치 30.86 GB > 24 GiB. 양자화 없이는 못 올림 |
| 실로봇 정책 평가 | **불가** | Franka Research 3, Inspire 손, 제어 코드가 없음 (D14) |
| 표 3–6 생성 지표 재현 | **불가** (스크립트 미공개) | D17. 독립 구현과 테스트 분할 정의를 직접 만들어야 함 |

**노력 추정 (측정 전, 추정치):**
- CPU 추론 포팅(MPS 경로 포함): 1–2주. 1B로 먼저 확인하면 하루에서 며칠.
- 토크나이저 재구성 검사(표 5 MPJPE 한 줄을 맞춰 보는 수준): 3–5일. MANO 등록과 원천 데이터 준비가 대부분입니다.
- 학습 재구현(토크나이저, 사전학습, 정렬): 학습 코드가 없어 수 개월 규모로 추정합니다. 32장 A800 학습 시간은 이 맥에서 맞출 수 없습니다.

## 5. 재실행

정적 분석이므로 재실행할 스크립트는 없습니다. 같은 대조를 다시 하려면 `repo/`를 같은 커밋으로 얕게 복제하고 위 줄 번호를 확인합니다.
