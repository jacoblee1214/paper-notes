# 공식 코드 대조 (논문 ↔ 저장소)와 구현 가능성

대상: `github.com/showlab/MetaWAM`, 커밋 `4e52edeb31f99f9dcd0dc6974fbdbb241c642d69` (2026-11-11), Apache-2.0. `repo/`에 얕은 복제했다. **정적으로만 읽었다.** 코드를 실행하지도, 설치하지도, import하지도 않았다. 모든 파일 경로는 `repo/` 기준이다.

## 1. 라이선스와 권리

- **저장소:** Apache-2.0. `LICENSE` 파일이 있고, 파일 헤더에도 SPDX 표시가 있다. `ATTRIBUTIONS.md`는 제3자 라이선스 목록이다 (GitPython BSD-3-Clause 등).
- **논문:** arXiv abs의 license 링크는 CC BY 4.0. 원문 PDF에 ©, All rights reserved, 회사 고지가 없다 (`analysis/01` §0).
- **제3자 자산:** 저장소의 `assets/images/`에는 CoRL 로고와 실로봇 롤아웃 이미지 한 장이 있다. 공개본에는 쓰지 않는다.

## 2. 대조 결과 (D1–D12)

판정: **일치**, **불일치** (논문과 코드 값이 다름), **논문 미기재** (코드에는 있고 논문 본문에는 없음), **코드 없음** (논문 결과를 만든 코드가 저장소에 없음).

| 번호 | 항목 | 논문 | 공식 코드 (파일:위치) | 판정 |
|---|---|---|---|---|
| D1 | 학습률 | 5절: "AdamW, 학습률 10^-6" | `cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py` `metawam_base`: `optimizer=dict(lr=1e-4)`. 스케줄 (`_src/imaginaire/functional/lr_scheduler.py` 143행 근처): 1000 스텝 선형 워밍업으로 1e-4까지, 이후 30K 주기 코사인 감쇠 (f_min = 0.3). 12K 스텝 끝에서 약 7.8e-5 | **불일치.** 논문 값이 설정과 다르다. 실제 학습에 어느 값을 썼는지는 확인 못 함 |
| D2 | 신호 손실 가중치 λ_act, λ_prop, λ_val | 식 (5)에 기호만 있고 값이 없다 | 같은 파일: `frequency_action_slot_loss_weight=2.0`, `frequency_future_proprio_slot_loss_weight=0.1`, `frequency_value_slot_loss_weight=0.01`. Huber δ = 1.0 (`frequency_active_slot_huber_delta`) | **논문 미기재** (값 자체). δ와 λ_off는 일치 |
| D3 | 대역 밖 페널티 λ_off (식 6) | λ_off = 10^-4, 식에는 절댓값이 없다 | `cosmos_policy/models/policy_text2world_model.py` 1139–1174행: `frequency_offslot_sparsity_weight=1e-4`, 페널티는 `coeff_pred.abs() * (1 - slot_mask)`의 평균. 허용 슬롯은 과제·현재 고유감각 각 3계수, 행동 유효 슬롯, 미래 고유감각 유효 슬롯, 가치 1계수 | **불일치 (표기).** 식 (6)은 부호 있는 평균으로 쓰여 있으나 코드는 절댓값 (L1). 논문 식이 오기일 가능성이 크다. 허용 슬롯 세부는 논문에 없다 |
| D4 | 채널 대역 배치 (식 2) | 과제, 현재 고유감각, 행동, 미래 고유감각, 가치의 분리 대역 | 같은 파일 62–63행: ch0 과제 (1), ch1–3 현재 고유감각 (3), ch4–11 행동 (8), ch12–14 미래 고유감각 (3), ch15 가치 (1). 합 16채널 | **일치.** 채널 수와 배치는 논문 본문에 없었다 |
| D5 | 행 0 전용 쓰기 (k_H = 0) | "kH = 0 row"에 쓴다 (4.2절) | 같은 파일 93–122행 `_idct2_row0_coeff_per_channel_ortho`: H 방향 IDCT가 DC 기저 상수 (1/√H)로 줄어든다 | **일치.** 결과적으로 공간 잠재는 H 방향으로 상수다 (`analysis/03` §3, 토이 V1) |
| D6 | 배치와 누적 | RoboCasa 1024 (64 × 4 GPU × 누적 4), LIBERO 1200, 실로봇 1024 (부록 B) | `metawam_robocasa` 배치 64, `metawam_libero` 배치 75 (× 4 GPU × 누적 4 = 1200), `metawam_realworld` 배치 64. 스크립트 `scripts/train_*.sh`의 기본 누적 4와 GPU 4 | **일치** |
| D7 | 행동 청크와 실행 길이 | RoboCasa 청크 32, 16 실행. LIBERO 16. 실로봇 16 전부 실행 | `chunk_size=32` (RoboCasa), `16` (LIBERO, 실로봇). 평가 `num_open_loop_steps` 16 (RoboCasa, LIBERO). 실로봇 `FRANKA_OPEN_LOOP_HORIZON` 기본 16 (`REALWORLD.md` 232행) | **일치** |
| D8 | 행동 차원 | 본문 기재 없음 | `robocasa_dataset.py` 177행: `action_dim = 7` (RoboCasa 원래 12차원 중 앞 7). 32 × 7 = 224 = 8 × W (W = 28 가정). 행동 대역이 정확히 찬다 | **논문 미기재** (차원과 용량) |
| D9 | 영상 증강 | 본문 기재 없음 | `use_stronger_image_aug=True` (RoboCasa, 실로봇), `False` (LIBERO) | **논문 미기재** |
| D10 | 진단 지표 (표 1), 신호 복원 (표 5), Wan 분석 (부록 A.2) | 결과 제시 | 해당 코드 없음. `grep`으로 Meta-Quadrants, Meta-Compact, 상대 재구성 오차, Wan2.1/2.2 디노이저 경로를 찾지 못함 | **코드 없음.** 표 1, 5, 6, 7 재현 경로 없음 |
| D11 | 절제 기준선 (표 5의 분리 프레임, 메타 쿼드런트, 메타 컴팩트) | 네 배치 비교 | `use_single_meta_frame=False`가 상위 Cosmos Policy의 프레임 주입 레이아웃을 쓴다 (`experiments/robot/cosmos_utils.py` 82–96행). 메타 쿼드런트와 메타 컴팩트 구현은 없음 | **부분 코드.** 분리 프레임 행이 같은 설정으로 재학습됐는지 확인 불가 |
| D12 | 추론 샘플링 노이즈 범위 | 본문 기재 없음 | 학습 `sigma_max=200, sigma_min=0.01`, 추론 `sigma_max=80, sigma_min=4` (`_inference_sde`) | **논문 미기재** |

추가 관찰.

- **LIBERO 레이아웃.** 설정 주석은 LIBERO를 "2 views, 6 latent frames"로 적는다 (`cosmos_policy_experiment_configs.py` 21행). 논문 식 (1)과 표 5의 8프레임은 RoboCasa와 실로봇 설정이다. LIBERO 설정은 논문에 없다 (**논문 미기재**).
- **체크포인트 입력 체계.** 저장소 README는 RoboCasa와 LIBERO의 데이터 통계와 T5 임베딩이 Cosmos Policy와 같다고 한다. 실로봇 체크포인트는 "우리 로봇 설정에 묶인다"고 명시한다 (`REALWORLD.md`).
- **시뮬 평가 프로토콜.** `run_robocasa_eval.py`는 장면 5개 (`layout_and_style_ids`)를 쓰며, 논문 5.1절 설명과 같다. LIBERO 평가 예시는 시드 195, 196, 197을 쓰며, 논문 부록 B.1과 같다.

## 3. 저장소의 구조와 의존성

- 기반: Cosmos Policy 코드 (`NVlabs/cosmos-policy`, Apache-2.0). 데이터 로더, 평가 스크립트, 학습 골격이 그대로 있다.
- 새 코드: `policy_text2world_model.py`의 슬롯 패킹과 손실, `policy_video2world_model.py`의 입력 메타 프레임 마스크, `minimal_v1_lvg_dit.py`의 슬롯 마스크 임베더, `cosmos_utils.py`의 읽기 함수. 저장소 README의 "Code Map"과 일치한다.
- 학습과 추론 의존성 (`pyproject.toml`의 `cu128` extra): `torch==2.7.0` (CUDA 12.8 휠), `flash-attn==2.7.3`, `transformer-engine==2.2.0`, `nvidia-*-cu12` 여러 개. 기본 Python 3.10. `SETUP.md`는 "NVIDIA GPU with CUDA support, Linux x86-64"를 요구한다.
- 체크포인트 크기: RoboCasa와 LIBERO, 실로봇 모두 `.pt` 3.91 GB, T5 임베딩 0.46 GB (RoboCasa) 또는 0.04 GB (LIBERO). 실로봇 임베딩은 3 MB.

## 4. 구현 가능성 판정

**환경:** Apple M4 Pro, 약 24 GiB 통합 메모리, MPS 사용 가능, CUDA 없음.

**판정: 부분 가능.** 논문의 핵심 수치는 이 맥에서 재현할 수 없다. 설계 산술, 읽기 식, 슬롯 용량은 CPU로 확인했다. 공식 추론을 MPS로 옮기는 작업은 조건부다.

| 수준 | 판정 | 근거 | 노력 (추정, 측정 전) |
|---|---|---|---|
| 표 재계산과 검정 | 가능, 완료 | `analysis/recompute.py` | 완료 |
| 식 (2)–(3) 읽기·쓰기 토이 | 가능, 완료 | `analysis/toy_readout_check.py` | 완료 |
| 공식 추론 (RoboCasa 한 과제 또는 LIBERO 한 스위트) | 조건부 | `flash-attn`, `transformer-engine`이 CUDA 전용. MPS 포팅과 `.pt` 3.91 GB 다운로드 승인 필요. 2B 모델의 MPS 추론 속도는 측정 안 함 | 포팅 1–3주 (추정) |
| 공식 시뮬 평가 (표 2, 3 전체) | 불가 (이 맥) | 시뮬레이터 설치, 3 시드 × 장면 5개 × 과제 24개, GPU 여러 장 전제 | 불가 |
| 공식 학습 (RoboCasa 12K 스텝, 4 × H200, 64 h; LIBERO 56 h; 실로봇 47 h) | 불가 | GPU 4장 전제, CUDA 전용 커널 | 불가 |
| 실로봇 (Franka, 3 카메라, 200+ 시연) | 불가 | 로봇, 카메라, 시연 데이터, 저장소 밖 장비 | 불가 |
| 디노이저 프라이어 진단 (표 1) | 조건부 | 동결 Cosmos-Predict2 디노이저가 필요 (2B). 추론 가능성은 위와 같음 | 포팅 후 1–2일 (추정) |
| 소형 CPU 토이 (슬롯 마스크 라우팅, 행 0 대 2D 패킹) | 가능 | numpy 또는 작은 torch 모델 | 1–2일 (추정) |

**가장 가까운 실행 경로:** 공식 체크포인트 한 개 (RoboCasa)의 추론 평가. 필요한 것은 CUDA 커널 대체 (flash-attn → SDPA, transformer-engine 제거), MPS 포팅, 3.91 GB 다운로드, RoboCasa 시뮬 설치다. 이 중 어느 것도 아직 승인받지 않았다.

## 5. 한계와 미확인 항목

- 학습률 불일치 (D1)의 실제 원인은 확인하지 못했다. 논문 값이 오기인지, 설정이 나중에 바뀌었는지 알 수 없다.
- 표 1, 5, 6, 7의 진단 실험은 코드가 없어 값을 검증할 수 없다. 표 값 자체의 정확성은 저자 보고에 의존한다.
- 공식 저장소는 2026-11-11에 처음 공개됐고 스타가 1개다. 이슈와 외부 검증 기록이 없다.
- 체크포인트의 SHA256은 HF 저장소 `SHA256SUMS`에 있지만 받지 않아 검사하지 않았다.
