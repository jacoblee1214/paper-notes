# 04. 공식 코드 대조와 구현 가능성

## 0. 대상과 방법

- 저장소: github.com/houlong-xiong/REACT, 커밋 `eaad1d1b80f31b08c461dd067272e62bd9addb69`(2026-10-18, 기본 브랜치 main, 얕은 clone 깊이 1). openpi 커밋 `215abfb`(2026-08-24)에서 분기한 포크이고 그 뒤 두 커밋이 REACT입니다(README의 "Changes to openpi" 절). 크기는 소스 약 2.6 MB입니다.
- 방법: **정적 읽기만** 했습니다. 코드를 실행, import, 설치하지 않았고 가중치와 데이터도 받지 않았습니다. clone은 `repo/REACT/`(gitignore)입니다. 아래 줄 번호는 이 커밋 기준입니다.
- 읽은 파일: `src/openpi/models_pytorch/pi0_pytorch.py`, `src/openpi/models/pi0_config.py`, `src/openpi/react/{rolling,dual_decoupling,react_test}.py`, `src/openpi/training/config.py`, `src/openpi/policies/{policy,acone_policy,franka_policy}.py`, `scripts/{train_pytorch,serve_policy}.py`, `docs/react.md`, `examples/react_acone_ros2/`, `README.md`.

## 1. 설치 요구사항과 라이선스

| 항목 | 내용 |
|---|---|
| 요구 | README: "REACT needs an NVIDIA GPU." `uv sync`(Python 3.11, `uv.lock` 약 0.8 MB), PyTorch용 `transformers_replace`를 `site-packages`에 덮어써야 함(gemma의 per-token adaRMS 변경), JAX 모델은 REACT 설정을 거부합니다(`pi0_config.py:85`) |
| 입력 체크포인트 | openpi의 `pi05_base`(JAX)를 `examples/convert_jax_model_to_pytorch.py`로 변환해 `pytorch_weight_path`에 지정. REACT로 학습한 체크포인트는 공개되지 않았습니다(저장소 이슈 #1 "Any information about pretrained checkpoints", 2026-11-07, 답 없음) |
| 라이선스 | Apache-2.0(`LICENSE`), `LICENSE_GEMMA.txt` 동봉. 논문 PDF는 CC BY 4.0 |
| 서브모듈 | `third_party/aloha`, `third_party/libero`(openpi 원본과 같음) |
| 테스트 | `pytest src/openpi/react`(CPU, 작은 모델, 10개). 환경 설치가 필요해 실행하지 않았습니다 |

## 2. 논문 ↔ 코드 대조 (D 항목)

판정: 일치, 차이(본문과 다름), 없음(코드에 없음), 확인(읽었으나 동치 여부를 이 분석이 확정하지 않음).

| ID | 항목 | 논문 | 코드 | 판정 |
|---|---|---|---|---|
| D1 | 계단 시간 | τ_j = (⌊j/S⌋+1)/K (식 1) | `pi0_pytorch.py:204–212`: `block_index = arange(H)//S; time = (block_index+1)/K`, 배치 전체에 같은 값 | 일치 |
| D2 | 롤링 한 단계 | x̃ = x − v/K (식 2) | `react_denoise_step`(`:473–481`): `x_t − v_t / react_num_blocks`, v는 `denoise_step`에 위치별 시간 `(B,H)`를 넘김 | 일치 |
| D3 | 앞 블록 실행, 이동, 꼬리 잡음 | 식 3 | `rolling.py:71–`(`roll`): 앞 S개를 복사해 반환, 나머지를 앞으로 이동, 끝에 `randn` 블록 추가 | 일치 |
| D4 | 워밍업 | B_k = (1−τ_k)s̄_0 + τ_k ε, 첫 관측으로 K−1회 갱신, 앞 블록 버림 (부록 C.4) | `RollingBuffer.reset`(`:57–69`)이 같은 식. `ReactPolicy.warmup`과 DD `_warm_up`이 K−1회 `roll`. 기본 `--react-hold state`(서버)는 `obs["state"]`를 타일(ALOHA 계열용), ARX와 Franka는 전용 함수 선택 필요 | 일치 |
| D5 | 손실 정규화 | 식 5: 블록 평균의 평균, 분모 S·D (D = 행동 차원) | `forward`는 `F.mse_loss(u_t, v_t, reduction="none")`로 `(B,H,32)`를 반환하고 `train_pytorch.py:536`이 `losses.mean()`. 행동은 32차원으로 패딩되어 있어 평균이 패딩 차원을 포함. 블록 크기가 같아 블록 가중은 같음 | 차이(상수배, 패딩 차원 포함. 상대 가중에는 영향 없음) |
| D6 | 시뮬 학습 설정 | 표 6: 최대 LR 2.5e-5, 최종 2.5e-6, 코사인, 워밍업 10%, AdamW β 0.9/0.95, ε 1e-8, wd 1e-10, 클립 1.0, 배치 32, 30k 스텝 | `training/config.py` `pi05_react_aloha`: `CosineDecaySchedule(warmup_steps=3_000, peak_lr=2.5e-5, decay_steps=30_000, decay_lr=2.5e-6)`, `AdamW(clip_gradient_norm=1.0)`, `batch_size=32`, `num_train_steps=30_000`. AdamW 기본값(β 0.9/0.95, ε 1e-8, wd 1e-10)은 `optimizer.py:66–74`. bf16은 `pytorch_training_precision` 기본값 `"bfloat16"`(`config.py:598`) | 일치 |
| D7 | 실로봇 학습 길이 | 20 에폭, ARX 배치 128, Franka 256(Reaction Game 32) (표 6) | `pi05_react_acone`·`pi05_react_franka`: 배치 128, 256은 맞고 `num_train_steps=30_000`, 워밍업 3,000은 자리표시자. `docs/react.md`가 "20 에폭에 맞춰 스텝과 스케줄을 직접 설정"하라고 적음. Reaction Game 배치 32는 설정에 없음 | 차이(문서화됨) |
| D8 | Franka 행동 표현 | 위치 + **연속 6D 회전 표현(Zhou et al.)**, 팔당 3+6, 그리퍼 2개, 합 20차원 (부록 C.2) | `franka_policy.py` 머리말: 팔당 N=9 = `(x, y, z, roll_sin, roll_cos, pitch_sin, pitch_cos, yaw_sin, yaw_cos)`. 합 20차원은 같지만 코드 문서 기준 회전은 **roll, pitch, yaw의 sin/cos 6개**이고 Zhou 6D(회전행렬 두 열)가 아님. 변환이 데이터 쪽에서 이뤄지는지는 저장소에서 확인되지 않음 | 차이(차원 수는 같고 회전 표현이 다름) |
| D9 | 블록 수 | 절제에 K=1 포함 (그림 11, 12) | `pi0_config.py:58–61`: `react_enabled`이면 `react_num_blocks < 2`는 `ValueError`. K=1 칸(13.6, 11.0, 6.3, 9.9%)은 공개 설정으로 만들 수 없음. K=1은 모든 위치의 τ가 1(순수 잡음만 학습)인 설정 | 차이(코드 검증이 K=1을 거부) |
| D10 | 기준선 | H50E50, H50E10, H10E10, Async+Training RTC (표 1–3) | 저장소에 기준선 학습·추론 설정, 학습 시 RTC 구현이 없습니다. H50E50, H50E10은 REACT를 끄고 `action_horizon=50`으로 openpi 기본 학습·추론에 청크 길이만 조정하면 만들 수 있지만, 저장소가 제공하지 않습니다. `grep rtc`는 `src/openpi/react`와 모델 파일에서 0건 | 없음 |
| D11 | 평가 스크립트 | RoboTwin 2.0 7과제 × 2조건 × 100회, 실로봇 30회 | RoboTwin 평가 클라이언트나 서버 스크립트가 없습니다. `pi05_react_aloha`의 `repo_id`는 `your_hf_username/your_robotwin_dataset` 자리표시자입니다. 정책 서버(`serve_policy.py`)와 ARX ROS2 예제만 있습니다 | 없음 |
| D12 | 지표 코드 | jerk norm(중앙차분 3회), Reaction Game 타이밍 스크립트 (부록 C.7) | 저장소에서 jerk 계산이나 Reaction Game 코드를 찾지 못했습니다(파일 전체 검색에서 `jerk`는 0건, `reaction`은 문서 문장에서만) | 없음 |
| D13 | 학습 시 관측과 블록의 짝 | §3.3: 한 번의 순전파가 K단계를 지도, 부록 C.5 일치 논증 | `forward`(`:354–`): 한 청크의 **모든 블록이 같은 관측 o와 짝**이고 노이즈는 위치마다 독립 `sample_noise(actions.shape)`. 추론 시 버퍼의 블록들은 서로 다른 관측에서 온 것이라, 학습에 없는 조합입니다. 논문이 주장하는 일치는 (시간, 위치) 쌍에 한정됩니다 | 확인(학습 분포와 추론 분포의 차이, 03 §7) |
| D14 | DD 스케줄 | §3.2: 센싱은 S 카메라 주기마다 한 프레임, VLM 워커 풀과 latest-ready 캐시, DiT는 가장 신선한 완성 임베딩 사용 | `dual_decoupling.py`: 센싱 루프 주기 `S/camera_fps`(`:423`). 요청 큐 `maxsize = 워커 수`(`:238`), 가득 차면 **가장 오래된 요청을 버리고 새 것을 넣음**(`_submit`, 통계 `dropped_observations`). 캐시는 `capture_time <= 현재`이면 무시(`_poll_results`, 통계 `out_of_order`). **다음 DiT 갱신은 남은 행동이 `lead_steps`(기본 1) 이하일 때 시작**(`:76`, `:316`). 센싱 루프와 실행 루프는 독립 타이머 | 일치(세부 정책은 논문에 없음: 큐 드롭, `lead_steps`) |
| D15 | 캐시에 들어가는 것 | §3.2: "VLM prefix 임베딩" | π0.5에서는 `discrete_state_input = pi05`(`pi0_config.py:49–50`)로 **고유수용 상태가 이산 토큰으로 접두에 들어갑니다**. DD의 캐시(`_pack_prefix`가 KV 캐시와 `prefix.state`를 복사)는 캡처 시점의 상태까지 포함하므로, 묵은 임베딩은 묵은 상태도 의미합니다. 논문은 "observation–state mismatch"를 언급하지만 상태 포함 여부는 적지 않았습니다 | 확인(논문에 서술 없음) |
| D16 | 블록 인과 마스크 | 논문에 없음 | `react_block_causal_mask`(기본 꺼짐, `:339–342`): 블록 k가 0..k만 봄. 켜지 않으면 논문과 같은 전체 양방향 | 일치(옵션은 논문 밖) |
| D17 | 평가 시 노이즈 스텝 | 기준선 N=10 (부록 C.2) | `sample_actions(..., num_steps=10)`(`:443`) 기본 10 | 일치 |
| D18 | 테스트 | 해당 없음 | `react_test.py`: 계단 시간, 위치별 시간과 스칼라 시간의 동치, 학습이 계단을 쓰는지, 한 단계가 Euler 한 단계인지, 롤링 버퍼, `ReactPolicy`, 홀드 액션, DD(`:228`) 등 10개. CPU에서 작은 모델로 돎 | 일치(실행하지 않음) |

### 2.1 논문 주장 중 코드로 확인되는 것

- **모델 구조가 같다**는 서술("changes the training timestep layout and the inference-time scheduling rather than the model architecture", 부록 C.2)은 코드와 맞습니다. 추가된 것은 위치별 시간 임베딩(adaRMS의 per-token 조건)과 마스크 옵션뿐입니다(`embed_suffix:295–329`).
- **"일 갱신당 DiT 한 단계"**는 `react_denoise_step`이 `denoise_step`을 한 번 호출하는 것으로 확인됩니다.
- DD 통계(`embedding_age_ms`, `out_of_order`, `dropped_observations`, `reused_embeddings`)가 코드에 있어 §3.2의 임베딩 나이와 "순서 뒤바뀜 0.0%"는 같은 계측으로 낼 수 있는 값입니다. 다만 그 로그 자체는 공개되지 않았습니다.

### 2.2 코드에 없는 것(재현에 영향)

- 기준선 설정과 학습 시 RTC(D10), RoboTwin 평가 스크립트(D11), jerk와 Reaction Game 코드(D12), REACT 체크포인트와 학습 데이터(이슈 #1)가 없습니다. 따라서 표 1–3과 8은 공개 코드만으로 재현할 수 없고, 이 분석은 논문 표 값의 산술과 통계만 검증했습니다(03).

## 3. 정적 읽기로 본 구조 요약

1. 학습: `Pi0Config(react_enabled=True, react_num_blocks=5)` → `forward`가 `react_time`으로 위치별 τ를 만들고 `x_t = τ·noise + (1−τ)·actions`, `u_t = noise − actions`, 손실 `mse(u_t, v_t)`.
2. 서빙(w/o DD): `ReactPolicy.infer`가 VLM 접두를 계산(`react_embed`), 첫 호출이면 워밍업, 이후 `roll(react_denoise(prefix, x))`로 S개 행동 반환. 클라이언트가 S개를 다 실행한 뒤 다음 요청.
3. DD: VLM 워커 프로세스(spawn), KV 캐시를 CUDA IPC로 DiT 프로세스에 전달, 실행 스레드는 30 Hz로 행동 1개씩 적용.

## 4. 구현 가능성 (이 맥: Apple M4 Pro, 약 24 GiB 통합 메모리, MPS, CUDA 없음)

**판정: 논문 수치의 재현은 이 맥에서 불가능합니다. 구조의 축소판 재구현과 처리량·통계 검증은 가능하고, 일부는 이미 끝냈습니다.**

| 수준 | 판정 | 근거 |
|---|---|---|
| 실로봇 재현 (표 2, 3, 8의 ARX X5, Franka R3) | 불가 | ARX X5 또는 Franka Research 3 두 대, 카메라 3대, ROS2, Reaction Game 장치(부록 C.1, C.7)가 필요 |
| RoboTwin 2.0 7과제 × 100회 평가 (표 1) | 불가 | REACT 체크포인트와 평가 스크립트가 없음(D10, D11). 시뮬 자체의 요구 사양은 이 분석에서 확인하지 않음 |
| 공식 코드로 학습 (π0.5 약 3.3B 파라미터, 30k 스텝) | 불가 | README가 NVIDIA GPU를 요구. 3.3B 모델의 옵티마이저 상태까지 올리면 24 GiB를 넘길 가능성이 큼(미측정). 논문은 batch 32–256과 torch.compile을 씀 |
| 공식 체크포인트 추론 | 불가 | 체크포인트 미공개(이슈 #1) |
| 정적 읽기, 논문 표 재계산, 처리량 산술 | 가능, **완료** | `analysis/recompute.py`(수 초) |
| 계단 학습·롤링 추론 구조 확인 토이 | 가능, **완료** | `analysis/toy_rolling_modes.py`(약 1초, numpy). 정확한 베이즈 극한에서 버퍼에 고착되는지 확인 |
| 공식 `pytest src/openpi/react` (작은 모델, CPU) | 조건부 | `uv sync`와 transformers 덮어쓰기 설치, 공식 코드 실행 승인이 필요. 이 분석에서는 하지 않음 |
| 학습 가능한 축소판: 작은 DiT를 소형 조작 시뮬(MuJoCo)에서 계단 학습·롤링 추론으로 H50E50, H50E10, REACT, 비동기 변형을 비교 | 가능(조건부) | torch 2.15와 mujoco 4.0이 환경에 있어 새 설치 없음. 노력 추정은 코드 400–700줄, 2–4일(측정 전). 효과는 π0.5와 같지 않을 수 있습니다 |
| DD 스케줄러 시뮬(지연 모델, 위상 정렬 효과) | 가능 | 이벤트 시뮬 수십 줄. 추정 반나절(측정 전). 하드웨어 지연은 모델값 가정 |

노력 추정은 모두 측정 전이며, 코드와 환경 수정 시간을 포함하지 않습니다.

## 5. 재실행

```bash
python3 analysis/recompute.py          # 표 재계산과 검정 (수 초)
python3 analysis/toy_rolling_modes.py  # 롤링 버퍼 토이 (약 1초)
python3 analysis/plot_figures.py       # 그래프 네 장
```
