# 04. 코드 대조와 구현 가능성 (Ctrl-World)

공식 저장소 `Robert-gyj/Ctrl-World`, 커밋 `99fb206` (2026-04-08), MIT. `repo/`에 얕은 clone하고 **정적으로만 읽었습니다.** 실행, import, 설치, 가중치와 데이터 다운로드는 하지 않았습니다. 저장소 안 파일은 모두 신뢰하지 않는 입력으로 취급했습니다.

## 1. 대조 범위

| 대상 | 파일 |
|---|---|
| 학습 설정 | `config.py`, `config_eval.py` |
| 모델과 손실 | `models/ctrl_world.py` (Action_encoder2, forward), `models/unet_spatio_temporal_condition.py` (frame-level cond) |
| 학습 루프 | `scripts/train_wm.py` |
| 데이터 샘플링 | `dataset/dataset_droid_exp33.py` |
| 행동 어댑터 | `models/action_adapter/train2.py`, `model2_15_9.pth` (1.5 MB) |
| 정책 루프 평가 | `scripts/rollout_interact_pi_eval.py`, `scripts/rollout_interact_pi.py` |
| 문서 | `readme.md`, `requirements.txt` |

## 2. 논문 ↔ 코드 차이 (D1–D14)

| 번호 | 논문 | 코드 | 위치 | 영향 |
|---|---|---|---|---|
| D1 | 식 3: 미래 x0에 대한 가중 없는 MSE, 확산 스케줄 α_t | EDM 전처리 (c_skip, c_out, c_noise), 로그정규 잡음 σ (P_mean 0.7, P_std 1.6), 손실 가중 (σ²+1)/σ² | `models/ctrl_world.py` 176–187, 204–209 | 식 3을 그대로 구현하면 학습 목표가 다름. 논문의 식은 설명용 간략형으로 보임 |
| D2 | 손실은 **미래 프레임**에만 계산 (식 2–3) | `predict_x0[:, num_history:]`는 현재 프레임(인덱스 num_history)을 포함. 현재 프레임은 잡음 0–0.2로 덮인 조건 프레임인데 손실 대상에 들어감 | `ctrl_world.py` 204–209, `dataset` 159–166 | 학습 대상이 "다음 H 프레임"보다 한 프레임 넓음. 논문 기술과 다름 |
| D3 | 과거 프레임 **k = 7** (App A) | `num_history = 6` (`config.py` 64). 샘플링도 6개 (`dataset` 159–160) | `config.py:64` | 맥락 길이 차이. 표 1–2 재현 시 주의 |
| D4 | 과거 간격 1–2초 (§5.1) | 학습 샘플링: 간격 4 또는 8 프레임 (5 Hz 기준 0.8–1.6초). 15%는 간격 0 | `dataset` 147–154 | 대략 일치 |
| D5 | 추론 과거 프레임 간격 | `history_idx = [0,0,-12,-9,-6,-3]` (`config.py` 88). 간격 3 프레임 | `config.py:88` | 학습 간격 4 또는 8과 다름. 단위(프레임 대 상태 인덱스) 확인 필요 |
| D6 | 행동 15개(1초)를 5개로 다운샘플 (§4.1, App A) | 정책 출력을 `policy_skip_step = 2`로 건너뛰고 `pred_step = 5`만 씀. 한 상호작용에서 정책 출력 8개 사용 (`valid_num = skip*(pred_step-1)`) | `config.py` 81–82, `rollout_interact_pi_eval.py` 284–291 | 행동 청크 길이와 인덱스 선택이 본문과 다름 |
| D7 | 정책-루프 롤아웃, t = j·H로 진행 (Alg. 1) | 상호작용마다 `start_id = i*(pred_step-1)`로 4단계씩 전진 (겹침) | `rollout_interact_pi_eval.py` 354–355 | 롤아웃의 시간 정렬이 Alg. 1과 다름 |
| D8 | 세계모델 평가 지표 PSNR, SSIM, LPIPS, FID, FVD (표 1–2) | **평가 지표 계산 코드 없음** (`psnr|lpips|fvd|ssim|fid` 검색 0건) | 저장소 전체 | 표 1–2는 저장소로 재생성 불가 |
| D9 | 정책 개선: LLM 지시 바꾸어 말하기, 초기 자세 재설정, 사람 판정 필터, π0.5 2k 스텝 SFT (§4.2, §5.4) | **관련 코드 없음.** `train_wm.py`는 세계모델 학습만. 정책 SFT, LLM 호출, 궤적 선별 스크립트 없음. README는 VLAW(arXiv:2602.12063)를 참고하라고만 씀 | 저장소 전체 (`finetune|sft|paraphrase|judge` 검색 0건, `create_meta_info.py`의 주석 import만 있음) | 표 4–7과 그림 9는 저장소로 재현 불가 |
| D10 | 정책 출력을 그대로 세계모델에 넣는다고 읽힘 (§5.1, App B) | 과제별 **그리퍼 상한** (pickplace 0.75, towel 0.95, tissue 0.97, drawer 0.6 등)과 **z 하한** (pickplace 0.23, 기본 0.18)을 적용 | `config.py` 76–79, `rollout_interact_pi_eval.py` 264, 276 (`rollout_interact_pi.py` 261 동일) | 본문에 없는 후처리. 정책 성공률과 과제 간 비교에 영향 가능. 과제별로 손으로 고른 값 |
| D11 | 상호작용 횟수: 표 1은 10회, 그림 6은 20회 | 평가 설정 `interact_num` 12 (`config.py` 83), 15 (`config_eval.py` 121, 138) | `config.py:83`, `config_eval.py:121,138` | 실험마다 길이가 다름. 표 3의 롤아웃 길이는 본문에 없음 |
| D12 | π0.5 등 정책 평가 설정 (App B) | 정책 체크포인트 인자 `--pi_ckpt`의 기본값은 클러스터 경로 (`config.py` 14, `rollout_interact_pi_eval.py` 309). README 명령은 `--pi_ckpt`에 **세계모델 체크포인트 경로**를 씀 | `readme.md` (§3 명령) | 문서 오류. π0.5 체크포인트가 필요 |
| D13 | 행동 어댑터: "두 층 MLP"로 관절 속도를 관절 위치로 바꿈 (App B) | 은닉 512, 활성 SiLU, 선형 3개 (은닉층 2개). 정규화 상하한 상수는 코드에 고정 | `models/action_adapter/train2.py` 45–58 | 층 수 해석 차이 (은닉층 2개면 일치). 기능상 일치 |
| D14 | 텍스트 조건은 "지시" (§4.1) | CLIP 텍스트 조건. 학습 중 5% 확률로 조건 제거 (분류기 없는 안내, 논문 기술 없음). 행동 인코더 주석은 50% 확률이라고 쓰였지만 실제로는 항상 더함 | `models/ctrl_world.py` 170–172, `Action_encoder2.forward` 주석 | 주석과 코드 불일치. 무시할 만함 |
| D15 | 학습률 1e-5, 배치 64 (2×8 GPU), 10만 스텝, 해상도 192×320 (App A) | 학습률 1e-5 (`config.py` 39), 배치 4 × 16 GPU = 64 (`config.py` 42), **최대 스텝 500,000** (`config.py` 45), 해상도 320×192 (`config.py` 60–61), fp16 혼합 정밀도 (`config.py` 43) | `config.py` | 학습률, 배치, 해상도는 일치. 스텝 수는 기본값이 다름 (논문 10만). 실제 체크포인트 설정은 확인 불가 |
| D16 | 정책 평가에 DROID 시연 기반 새 장면 (`droid_new_setup`) | 저장소에 `droid_new_setup`과 `droid_new_setup_full` 예시 있음 (각 과제 시작 상태 포함) | `dataset_example/` | 일치. 전체 실험의 초기 상태는 `_full`에 있음 |

**일치 확인 (차이 아님):** 행동 차원 7, 입력 해상도 192×320의 3시점 잠재 24×40 (`72×40` 연결, `dataset` 183), 과거 프레임 잡음 0.3 (`ctrl_world.py` 191–193, App A 일치), 관절에서 Cartesian으로 가는 Franka FK (`rollout_interact_pi_eval.py`).

## 3. 저장소에 없는 것 (재현 차단 항목)

1. 평가 지표 (D8): PSNR, SSIM, LPIPS, FID, FVD와 검증 세트 256 클립 선택.
2. 정책 개선 전체 (D9): 지시 바꾸어 말하기, 초기 자세 재설정, 사람 판정, 궤적 선별, π0.5 미세조정.
3. 실로봇 판정 기록과 판정 기준 (App B의 과제별 기준은 문서로만 있음).
4. 학습 로그와 체크포인트 설정 (D15의 스텝 수 확인 불가).
5. 정책 출력 후처리의 과제별 값 선택 근거 (D10).

## 4. 구현 가능성 판정 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**판정: 논문 수치의 재현은 이 맥에서 불가능합니다. 학습은 불가능합니다. 추론 시연(롤아웃 한두 개)은 조건부로 가능하지만 이 분석에서는 실행하지 않았고 미검증입니다.**

| 작업 | 판정 | 이유 |
|---|---|---|
| 학습 (100k 스텝) | **불가** | 본문 2×8 H100 2–3일, README는 1–2노드 A100/H100 요구. 재학습 메모리와 시간이 맥을 넘음 |
| 표 1–2 영상 지표 재현 | **불가** | 평가 코드 없음 (D8). DROID 검증 세트(약 370 GB 전체 데이터 필요) |
| 표 3 정책 평가 재현 | **불가** | π0.5 JAX 체크포인트 실행 (`rollout_interact_pi_eval.py`가 JAX 사용), 장면 셋업, 사람 판정이 필요. JAX-Metal 경로는 확인 안 됨 |
| 표 4–7 정책 개선 재현 | **불가** | 코드 없음 (D9), π0.5 SFT GPU 필요 |
| 추론 시연 (replay, keyboard, 단일 롤아웃) | **조건부 (미검증)** | 체크포인트 ~8 GB, SVD ~8 GB, CLIP ~600 MB 다운로드 필요 (승인 대상). `pipe.to("cuda")`는 문서 예시에만 있고 실행 경로는 장치 인자를 따름 (`pipeline_ctrl_world.py` 339, 657 docstring 확인). bf16 가중치와 `dtype=torch.bfloat16` (`config.py` 67)은 MPS에서 지원 여부 확인 필요. 메모리는 여유로 보임 (UNet 1.5B bf16 약 3 GB 추정). 상호작용 한 번: A100 약 10초, H100 약 5초 (README). MPS 시간 미측정 |
| CPU toy (본문 핵심 주장의 축소 검증) | **가능** | `03` 참고. 사용자 승인 후 numpy 또는 torch(설치됨)로 가능 |

**노력 추정 (근거 없는 추정):**

- 추론 시연 포팅과 체크포인트 다운로드, MPS 확인: 2–4일. 다운로드 약 16 GB는 사용자 승인 후.
- 학습 축소판 (작은 UNet과 합성 궤적): 1–2주, 논문 수치와 직접 비교는 불가.
- 정책 평가 재현: 연구 인프라 수준 (실로봇과 π0.5 GPU 필요). 이 맥에서는 제외.

## 5. 실행하지 않은 것과 이유

- `scripts/*.py`와 `models/*.py` 실행: 저장소 규칙상 정적 읽기만. CUDA 기기 없음.
- 의존성 설치: `requirements.txt`는 `diffusers==0.34.0`, `transformers==4.48.1`, `torch==2.7.1`, `accelerate` 등을 요구. 설치 전 목록 승인 대상.
- 체크포인트와 데이터 다운로드: 승인 대상 (1 GB 이상).
