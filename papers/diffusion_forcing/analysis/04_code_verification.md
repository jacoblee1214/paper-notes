# 04. 논문과 공식 코드 대조

대상 저장소: `github.com/buoyancy99/diffusion-forcing` (MIT). 두 브랜치를 얕게 받아 정적으로만 읽었습니다(`repo/`, 실행·설치 안 함).

| 브랜치 | 커밋 | 날짜 | 내용 |
|---|---|---|---|
| `main` | `475e0bc` | 2025-02-22 | README는 "v1.5 code base, 최신 재구현(트랜스포머)"라고 함. 영상, 미로 계획의 재구현 |
| `paper` | `dd65516` | 2025-02-17 | 논문 실험 코드(미로 계획, 시계열, 로봇, RNN 기반). README는 "논문 재현은 paper 브랜치" |

> **범위 주의(D1).** 논문 표 1(미로), 표 2(시계열), §4.4(로봇) 수치를 대조하려면 `paper` 브랜치를 기준으로 봐야 합니다. `main`의 미로 계획은 README가 직접 "논문과 다르다"고 적었습니다. 아래 대조는 두 브랜치를 함께 씁니다.

## 1. 핵심 주장과 코드의 대응

| 논문 | 코드 | 판정 |
|---|---|---|
| 프레임마다 독립 잡음 수준 k_t (Alg. 1, 식 3.1) | `main`: `df_base.py` `_generate_noise_levels` → `torch.randint(0, timesteps, (num_frames, batch))`. `paper`: `diffusion_transition.py` forward가 프레임마다 `randint` | 일치 |
| 관측이 없는 토큰은 최대 잡음 (§3.1 마스킹) | `main`: `discard` 마스크인 토큰을 `timesteps - 1`로 설정 | 일치 |
| 인과 시퀀스 모델 (RNN 또는 트랜스포머) | `main`: `causal: True` 기본, `paper`: GRU 기반 `diffusion_transition.py` | 일치 |
| 샘플링 스케줄 행렬 K (Alg. 2, §3.2–3.3) | `main`: `_generate_scheduling_matrix`의 `pyramid`, `full_sequence`, `autoregressive`, `trapezoid` | 일치 (스케줄 종류는 같음, 세부는 D12) |
| 확산 DDIM 샘플링 (부록 D.6) | `Diffusion.sample_step`: `sampling_timesteps < timesteps`이면 DDIM | 일치 |
| 계획 안내 (MCG, 부록 D.5) | `paper` `df_planning.py` `interact`: 여러 샘플로 기울기 추정 | 부분 일치 (D6, D7) |

## 2. 차이 목록 (D1–D15)

**D1. 브랜치 범위(위 참고).** `main`의 미로 계획과 시계열, 로봇 코드는 논문과 다릅니다. README는 "MCTG 재구현 안 함", "미로 계획 설정 변경", "시계열과 로봇은 paper 브랜치"라고 적었습니다. 표 값을 재현하려면 `paper` 브랜치가 필요합니다.

**D2. 잡음 수준 범위.** 논문 Alg. 1은 k_t ∈ {0, …, K}이고 K = 1000입니다(부록 D.6). 코드는 `randint(0, timesteps)`이라 최댓값이 999입니다. 순수 잡음 수준 K가 한 칸 모자랍니다. 인덱스 규약 차이로 보이지만 확인하지 못했습니다. 실험 결과에 미치는 영향은 작을 것으로 보입니다.

**D3. 손실 가중(SNR).** 논문 식 3.1은 단순 MSE이고, 부록 D.1은 "fused SNR" 가중을 설명합니다. 코드 기본값은 `main`에서 `use_fused_snr: False`(기본)이고 영상만 `True`입니다. `paper`에서는 `use_snr: False`, `use_cum_snr: False`가 기본입니다. 논문 본문은 어떤 실험에 어떤 가중을 썼는지 적지 않습니다. 또 `paper`의 `diffusion_transition.py` 259행은 가중에 `snr_clip * 0.5`를 곱하는데 논문에 이 상수가 없습니다.

**D4. 잡음 클리핑.** `clip_noise`가 영상에서 6, 기본에서 20이고, 샘플링 중에도 x를 같은 값으로 자릅니다(`main` `diffusion.py`). 논문에는 없습니다. 실험 재현에 영향을 줄 수 있는 설정입니다.

**D5. 목적함수 매개변수화.** 논문 식 3.1은 ε 예측 형태로 씁니다. 코드 기본은 `main`에서 기본 `pred_x0`, 영상과 시계열 `pred_v`입니다(`df_video.yaml`, `df_prediction.yaml`). 부록 D.3의 v-매개변수화 언급과 맞지만, 어떤 실험이 어떤 매개변수화를 썼는지는 본문에 표로 없습니다.

**D6. 계획 안내의 목표 거리 식.** 부록 D.5는 보상 모델 Σ_t ‖p_T − g‖로 안내한다고 씁니다. `paper` 코드(`df_planning.py` 약 281–296행)는 위치 차원의 목표 거리 `dist`에 시간 가중 `weight`를 곱해 `episode_return_sparse = -(g_scale * n_frames * dist * weight).sum(1).mean()`을 만들고, 기울기로 `plan[t] += guidance_consts[t] * grads[t] * guidance_scale`을 씁니다. 식의 형태는 일치합니다. 다만 `g_scale = 1`, `n_frames` 곱, 마지막 가중 0 처리는 논문에 없습니다.

**D7. 안내 계수의 빠진 값.** 논문에는 안내 세기 `guidance_scale`과 MCG 샘플 수 `ddim_repeats`의 값이 없습니다. `paper` 설정 기본값은 `guidance_scale: 0.`, `ddim_repeats: 0`입니다. 즉 MCG가 기본으로 꺼져 있습니다. `main` README의 명령은 `guidance_scale=2`를 씁니다. 표 1의 "Ours"가 어떤 값으로 나왔는지 논문으로는 알 수 없습니다.

**D8. `main`의 목표 안내 식은 논문과 다릅니다.** `main` `df_planning.py` `goal_guidance`는 목표 거리에 `tanh(d/2)` 압축, 조건 토큰 가중 20, 목표 가중 8과 2, 전체 ×1000을 씁니다. 논문에 없는 계수입니다. `use_reward: True`(밀집 보상) 분기는 `NotImplementedError`입니다(본문 4.2의 보상 안내는 코드에 없음).

**D9. 평가 에피소드 수.** `paper` `interact()`는 `batch_size`만큼의 환경을 `envs.seed(0)`로 병렬 실행합니다. 에피소드 수는 검증 배치 크기에 따라 달라지지만 논문은 적지 않았습니다. 표 1 ±의 의미를 정하지 못하는 이유입니다(03 §0).

**D10. 시계열 설정.** 대부분 일치합니다. `calc_crps_sum: 100`(논문 F.4의 100 궤적과 일치), 조기 종료 `patience: 6`(부록 D.9와 일치), GRU 4층과 MLP 1층(부록 D.9와 일치), `pred_v`(부록 D.3과 일치). **5회 시드 반복 루프는 코드에 없습니다.** 시드 설정은 실행 명령에서 정하는 것으로 보입니다. 확인하지 못했습니다.

**D11. 영상 프레임 스택.** 부록 D.6은 DMLab 4, Minecraft 8, 미로 10이라고 적습니다. `paper` `df_video.yaml`은 `frame_stack: 4 # 4 for dmlab, 8 for minecraft`로 일치합니다. `main`에는 DMLab의 `frame_stack`이 설정되어 있지 않아 기본값 1이 적용됩니다. `main`의 DMLab 기본 실행은 논문과 다릅니다. Minecraft는 README 명령에서 `algorithm.frame_stack=8`을 줍니다.

**D12. 안정화 샘플링.** 논문 §3.3과 부록 B.4는 최근 토큰을 작은 잡음 수준 k로 유지해 롤아웃을 안정화한다고 씁니다. `main` 기본 `scheduling_matrix: autoregressive`는 `_generate_pyramid_scheduling_matrix(horizon, sampling_timesteps)`입니다(`df_base.py` 195행). README는 "기본으로 안정화를 쓴다"고 적었습니다. 그러나 k의 구체 값이 논문 부록 B.4와 일치하는지는 확인하지 못했습니다. **부분 확인.**

**D13. 공개 범위.** 미로 사전학습 체크포인트와 영상 미니 데이터는 README의 Google Drive 링크로 제공됩니다(받지 않았습니다). 영상 전체 데이터(TECO DMLab, Minecraft)는 크기가 커서 README가 별도 다운로드를 요구합니다. **로봇 데이터와 실로봇 코드는 공개되지 않았습니다.** README는 "로봇 코드는 요청 시 제공"이라고 적었습니다. 따라서 §4.4의 80%, 76%, 48%는 확인할 수 없습니다.

**D14. 메타데이터.** LICENSE는 MIT이고 저작권 행은 "Copyright (c) 2024 Boyuan Chen"입니다. GitHub API의 라이선스 필드는 자동 판독에서 `NOASSERTION`으로 나왔지만 파일 본문은 MIT입니다. README의 BibTeX는 `year={2025}`이고 `volume={37}`, 쪽 24081–24125입니다. NeurIPS 2024(vol. 37)와 맞지만 연도가 다릅니다. 사소한 불일치입니다.

**D15. 지원하지 않는 설정.** `main`의 `_generate_noise_levels`는 `"random_all"`만 처리합니다. 다른 값을 주면 `noise_levels`가 정의되지 않아 오류가 납니다. 기본 설정에서는 문제가 없습니다.

## 3. 일치하는 설정 (확인 목록)

- 노이즈 스케줄: 영상 sigmoid, 미로 linear, 나머지 cosine (부록 D.4) — `main` `df_video.yaml`(sigmoid), `df_planning.yaml`(linear), `df_base.yaml`(cosine). 일치.
- DDIM 샘플링 단계: 영상 100, 비영상 50 (부록 D.6) — `main` `df_video.yaml` 100, `df_base.yaml` 50. 일치.
- 미로 프레임 스택 10 (부록 D.6) — `main` `df_planning.yaml` `frame_stack: 10`. 일치.
- 시계열 CRPS 샘플 100, patience 6, GRU 4층 MLP 1층 — `paper` 설정. 일치.

## 4. 구현 가능성 판정

**판정: 부분 가능. 구조 재구현은 이 맥에서 할 수 있지만 논문 표 수치를 다시 얻는 것은 대부분 불가능합니다.**

이 맥은 Apple M4 Pro, 통합 메모리 약 24 GiB, MPS 사용 가능, CUDA 없음입니다. 공식 README는 "Macbook M1 GPU가 있으면 자동으로 쓴다"고 적었습니다(MPS 경로). 모델 규모와 학습 시간의 근거는 부록 D.10입니다.

| 재현 대상 | 필요한 자원 (논문·README) | 이 맥에서 | 노력 추정 |
|---|---|---|---|
| 미로 계획 (표 1) | 2080Ti 11GB, 모델 4.33M, 학습 4–8시간 (D.10). MuJoCo 210, d4rl 구버전 (extra_requirements) | 조건부. 사전학습 체크포인트 평가는 CPU/MPS로 가능할 수 있음. 학습은 시간 측정 필요 | 환경 설치 1–2일, 체크포인트 평가 반나절 (추정) |
| 시계열 (표 2) | 2080Ti, 4–8시간 (D.10), GluonTS 데이터 | 조건부. Exchange처럼 작은 데이터셋은 CPU/MPS로 시간 내 가능할 수 있음 | 데이터와 설치 반나절, 5시드 학습은 수십 시간 (추정) |
| 영상 (그림 3, 정성) | A100 8장 12시간 (D.10). 영상 전체 데이터는 README가 "giant datasets"라고만 적음 (크기 미확인) | **불가** | — |
| 로봇 (§4.4, 80%) | Franka, 실로봇, 데이터 미공개 | **불가** | — |
| 시계열 CRPS 재계산 | 학습된 체크포인트 필요 | 공개 체크포인트 없음 (확인 못함) | — |

**구조 재구현은 가능합니다.** 프레임마다 독립 잡음, 마스크 토큰을 최대 잡음으로 두기, 스케줄 행렬 샘플링은 수백 줄 이하의 PyTorch로 쓸 수 있습니다. 2D 장난감으로 CPU에서 확인할 수 있습니다. 이 판정은 기존 작업 방식을 따르며, 재현 실행은 사용자 확인 후에만 합니다.

## 5. 검증하지 못한 것

- D2의 인덱스 규약 차이가 결과에 주는 영향 (실행하지 않음).
- D12의 안정화 잡음 수준 k 값 일치 여부 (부록 B.4와 세부 코드 경로를 끝까지 대조하지 않음).
- 표 1, 표 2의 실제 학습 설정 (논문이 설정 파일을 공개하지 않음).
- `paper` 브랜치 `df_robot.py`의 세부 (실로봇 데이터가 없어 실행 불가).
