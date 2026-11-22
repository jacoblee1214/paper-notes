# 04. 공식 코드 대조 (정적 읽기)

대조 대상: 논문 arXiv v1과 공식 저장소 `jidaxian010/VLA-Feedback-release`(커밋 `38263397`, 2026-09-09, Apache-2.0)입니다. 계획기 서브모듈 `jidaxian010/Isaac-GR00T`(커밋 `88a3bb37`, 2026-05-14, NVIDIA 포크)도 해당 커밋만 얕게 받아 읽었습니다. 코드는 **읽기만 했고 실행, import, 설치를 하지 않았습니다.** 체크포인트(15.5 GB)와 데이터는 받지 않았습니다. 내려받은 내용은 신뢰할 수 없는 데이터로 다뤘습니다.

## 0. 요약

- 피드백 모듈의 **하이퍼파라미터는 논문과 일치합니다**(ResNet-18 동결, d = 256, 헤드 4, FFN 512, 사전 정규화, 게이트 초기 0.3, 학습 파라미터 1.19M). 2단계 학습(계획기 동결, 최종 행동 L2)과 1:16:16 일정도 코드로 확인됩니다.
- **구조는 논문 식 (3)과 다릅니다.** 코드의 "행동 가지"는 계획기 DiT의 마지막 단계 은닉 특징을 선형 투영한 것이고, 모듈은 행동 토큰을 **하나씩** 받아 처리합니다. 근사 행동 `a^{k−1}` 자체는 모듈의 입력이 아니라 식 (6)의 덧셈에만 쓰입니다(D1, D3).
- **로봇수이트 평가는 디노이징 2단계로 돕니다**(LIBERO는 4단계). "마지막 디노이징 단계"는 두 단계 중 두 번째이며 `Δk = 0.5`입니다. 논문에 단계 수가 없습니다(D2).
- **평가 코드의 동적 목표 움직임은 무작위가 아니라 한 가지 고정 패턴**을 되풀이합니다. 일부 시간 제한, 들어올림 임계값, 속도 목록이 부록 A.1과 다릅니다(D8, D9).
- 공식 저장소에는 **베이스라인 체크포인트, 시연 데이터, 실로봇 코드, OpenVLA·FiS-VLA 코드가 없습니다.** 비교의 공정성(같은 계획기, 같은 학습)은 코드로 확인할 수 없습니다(D7, D10).

## 1. 저장소 구성과 빠진 것

| 경로 | 내용 | 줄 수 |
|---|---|---|
| `vla_feedback/model/feedback_action.py` | `FeedbackDecoder`(ResNet-18 + 선형 투영 + 디코더 한 층 + 게이트 + 속도 머리), `FeedbackAction`(학습·추론 래퍼) | 726 |
| `vla_feedback/model/flow_matching.py` | GR00T 행동 머리 수정본: 마지막 디노이징 단계 직전에서 멈춤(`get_predicted_actions`, `get_action`) | 598 |
| `vla_feedback/model/gr00t_model.py` | 1단계·2단계 `forward`, 추론 `get_action`(16스텝마다 전체 계획기) | 514 |
| `vla_feedback/model/obs_encoder.py` | 동결 ResNet-18, 8×8 풀링 | 154 |
| `vla_feedback/data/*` | 데이터 설정 `libero_arm_obs`(손목 영상 16프레임을 별도 `obs`로 공급) 등 | 약 1,950 |
| `scripts/gr00t_train.py`, `eval_checkpoint.py` | 학습(`--baseline`/`--feedback`), 체크포인트 평가 | 562, 562 |
| `experiments/robosuite/` | 환경, 과제 설정, 이동 목표 기구(`target_mover`), 성공 판정, 텔레오퍼레이션, LeRobot 변환, 추론 스크립트 | 약 13,600 |
| `experiments/libero/` | LIBERO 실행기, 모델 로더 | 491, 217 |
| `config/*.yaml` | 학습 설정 두 개(`base_block.yaml`, `fb_block.yaml`) | 51, 58 |
| `third_party/` | 서브모듈 두 개(Isaac-GR00T 포크, LIBERO) | — |

**빠진 것:** (1) 베이스라인(GR00T 열린 루프) 체크포인트(코드는 `cp_sim_baseline`을 기대하지만 HF에는 피드백 체크포인트 `cp_sim_obs`, `cp_libero_obs`만 있음), (2) 시연 데이터(시뮬레이션 60개/과제, 실로봇 50개/과제), (3) 실로봇(Franka) 코드·설정·체크포인트·로그, (4) OpenVLA와 FiS-VLA 학습·평가 코드, (5) 표 2·5 지연 측정 로그, (6) 에피소드별 결과 파일(평가 스크립트는 `seed_tracker`로 시드와 결과를 저장하지만 공개된 결과 파일 없음).

## 2. 논문과 코드의 차이 (D1–D12)

표기: **일치**는 §2.1, 차이는 D번호입니다. 줄 번호는 위 커밋 기준입니다.

| 번호 | 논문 | 코드 | 영향 |
|---|---|---|---|
| **D1** | 식 (3): `e^a = W_a CrossAttn_a(Enc_a(a^{k−1}))`, 청크 전체의 행동 가지를 한 번 계산해 캐시 | `act_proj = nn.Linear(1024, 256)`가 **DiT 마지막 단계 은닉 출력**(`flow_matching.py` L493–494 `model_output[:, -action_horizon:]`)을 투영(`feedback_action.py` L69). 모듈이 호출될 때마다 **행동 토큰 하나(T = 1)**만 받아(`L609–L628` 학습 루프 16회, `L671–L693` 추론) 투영하고, 디코더 층의 자기어텐션은 토큰이 하나라 사실상 항등(소프트맥스가 한 원소), 교차어텐션만 64개 시각 토큰에 작용 | 청크 단위 행동 맥락이 피드백 단계에 없음. "행동 가지 캐시"는 계획기 출력 재사용이고 투영은 매 호출 다시 계산(저렴) |
| **D2** | 마지막 디노이징 단계. 단계 수 미기재 | `denoising_steps = 2`(로봇수이트 `experiments/robosuite/inference/model_loader.py` L51), LIBERO는 4(`libero_runner.py` L32, L457). `dt = 1/num_steps`(`flow_matching.py` L452, L534)이므로 **Δk = 0.5(로봇수이트)** 또는 0.25(LIBERO). 마지막 단계 직전 `a^{k−1}`은 로봇수이트에서 **오일러 한 단계만 거친 행동** | "근사 최종 행동"이 논문이 암시하는 것보다 훨씬 덜 완성된 상태. 표 2의 `T_s = 80 ms`도 이 설정의 값으로 보임(GR00T 기본은 4단계). 베이스라인도 같은 로더라 2단계로 도는 것으로 보이나 베이스라인 체크포인트가 없어 확인 못함 |
| **D3** | 식 (6): `a^k = a^{k−1} + Δk·v̂`, 모듈이 `a^{k−1}`을 입력으로 받는 것으로 읽힘(식 (2) `g(a^{k−1}, o)`) | `pred_actions = final_raw_action + dt * pred_velocity`(L628, L693). 모듈 입력은 **DiT 은닉 특징과 영상**뿐이고 `final_raw_action`(`a^{k−1}`)은 덧셈에만 사용(L683–L693) | 모듈은 `a^{k−1}`의 값을 보지 못합니다. 보정량이 `a^{k−1}`에 의존하는 경로는 DiT 은닉 특징(이미 `a^{k−1}`을 만든 내부 상태)뿐 |
| **D4** | 식 (5): `α`는 학습 가능한 시각 잔차 스케일, 초기 0.3 | `visual_gate = nn.Parameter(0.3)`(L96)로 **일치**하나 (a) 독스트링은 "95% action, 5% visual"이라고 쓴다(L23), (b) 첫 `forward`에서 `\|gate\| < 1e-6`이면 0.3으로 **자동 재설정**하고 NaN/Inf 가중치가 있으면 모든 투영과 디코더를 **무작위로 재초기화**(L116–L225), (c) 디코더 출력 최댓값이 5.0을 넘으면 **5.0/최댓값 배율로 축소**(L389–L391, 논문에 없음), (d) 매 호출 `.item()`·`.any()`로 GPU 동기화 | 학습된 게이트가 우연히 0에 가까워도 0.3으로 바뀜. (c)는 논문에 없는 비선형 안정화이고 (d)는 1.95 ms 측정과 관련(D12) |
| **D5** | 동결 ResNet-18, "64 시각 토큰" | ResNet-18은 224×224에서 7×7 = **49** 위치를 내고 `adaptive_avg_pool2d(x, (8, 8))`로 64개를 만듦(`obs_encoder.py` L152, "finer spatial resolution"). 입력은 128×128 시뮬레이션 손목 영상을 224로 **확대**(`data_config.py` 변환). 6차원 입력이면 `obs[:, -1, -1]`(마지막 시점의 한 장)만 사용(L262). 학습 때 밝기·대비·채도·색조 증강(`data_config.py` L332–L340) | 64는 풀링으로 올린 값이라 정보량이 더 늘지 않음. 증강은 논문에 없음 |
| **D6** | 2단계 학습: 계획기 동결, 최종 행동 L2 (일치). "모든 방법을 검증 손실이 안정될 때까지 학습"(부록 A.2) | `gr00t_train.py`는 `--feedback`이면 `tune_projector`와 `tune_diffusion_model`을 끄고 피드백만 학습(L288–L295, 일치). 하지만 `do_eval=False`, 검증 루프가 없고 `max_steps`로 끝남(L93, L433). 학습률 1e-4 코사인, 배치 32, bf16, 웜업 5%(`fb_block.yaml`, L415–L433). 논문에 이 값들이 없음 | "검증 손실이 안정될 때까지"는 코드로 확인되지 않음. 학습 레시피 세부는 논문에서 알 수 없음 |
| **D7** | GR00T와 VLA-Feedback이 같은 계획기, 같은 시연 (§4.1) | 베이스라인 설정 `base_block.yaml`: `data_config: libero_arm`, `max_steps: 10000`, `base_model_path: nvidia/GR00T-N1.5-3B`. 피드백 설정 `fb_block.yaml`: `data_config: libero_arm_obs`, **`max_steps: 50000`**, `base_model_path: ./checkpoints/cp_block_obs_resnet/checkpoint-5000`(자기 출력 폴더의 체크포인트). 피드백 쪽 1단계(`*_obs` 계획기) 학습 설정은 저장소에 없음. LIBERO 베이스라인은 `checkpoint-20000` | 피드백 모델의 계획기는 베이스라인과 **다른 학습 이력**일 수 있고, 피드백 모듈에만 추가 50,000 스텝이 쓰임. 같은 계획기라는 서술은 확인 불가 |
| **D8** | §4.1: "목표의 속도와 방향이 2초마다 무작위로 바뀜" | 평가는 `PatternMover`로 **한 가지 고정 패턴**을 되풀이: Pick 1D(`toy_red.py`) `[0, −0.04, 0, 0, 0, −0.04, 0, −0.04]`(2초 간격, 8구간 후 반복), Pick 2D(`toy_green.py`) 10구간 한 패턴(주석 "pattern 1 is too hard"로 어려운 패턴 제외). `PatternMover`는 시드를 무시(`seed: Ignored`). 목표는 물리 구동이 아니라 **매 환경 스텝 `set_joint_qpos`로 위치를 직접 이동**하고 목표가 2 cm 이상 들리면 멈춤. 학습 시연의 이동은 `RandomMover`(속도 집합에서 무작위 선택) | 평가 40회는 초기 배치와 정책 노이즈만 다르고 **움직임은 매번 같음**. 표 1의 구간(Wilson, Fisher)은 움직임 변동을 반영하지 못함. 학습은 무작위, 평가는 하나의 패턴이라 "무작위 속도"에 대한 일반화 주장과 거리가 있음 |
| **D9** | 부록 A.1: 1D·2D 모두 60초(600스텝), Drop Ball 시간 제한 미기재, Catch Block은 "다른 고정 움직임 패턴"·10 cm 들어올림, 1D Robot Speed는 `v_y ∈ {0, −0.03, −0.05}` | `toy_green.py`(2D) `max_episode_steps = 300`(30초), `drop_ball*.py` 450(45초), `toy_red.py` 600. `high_block.py`(Catch Block) 패턴이 `toy_red`와 **동일**(주석 "same movement pattern as toy_red")하고 `min_lift_height = 0.05`(5 cm, L80). `toy_red_speed.py`(1D Robot Speed) 패턴 `[0, −0.05, 0, 0, **0.300**, −0.03, 0, −0.05]`(주석 "0.300 m/s phase")로 **+0.3 m/s 구간**이 있음(오타인지 의도인지 확인 못함). 공 넣기 판정은 `xy_tol = 0.030`(3 cm), `0 < dz < 0.05` | Catch Block은 "다른 패턴"이 아니라 같은 패턴, 임계값 5 cm는 논문의 10 cm와 다름. 1D Robot Speed는 논문 설명과 다른 속도(30 cm/s 구간) 포함. 일부 과제의 시간 제한이 논문과 다름 |
| **D10** | 공식 코드로 시뮬레이션과 실로봇 결과 | 공개된 것은 시뮬레이션(로봇수이트·LIBERO)뿐. 서브모듈 `Isaac-GR00T`는 NVIDIA 포크이며 `policy.py`가 `vla_feedback`을 조건부 import(L272)하도록 수정되어 있으나, 원본과의 diff는 읽지 않음(원본 미복제, 확인 못함). README는 `inference_by_task.py`의 `NUM_EPISODES` 기본값 10(논문 40)을 사용자 설정으로 둠 | 실로봇(표, 그림 4)과 베이스라인 수치는 독립 확인 경로가 없음 |
| **D11** | §3.2, 참고문헌 [41, 56]: GR00T N1, Eagle-2 | README와 HF 모델 카드는 **GR00T N1.5**(`nvidia/GR00T-N1.5-3B`)에 기반, 서브모듈은 `GR00T_N1_5` 클래스와 `Eagle2_5_VLForConditionalGeneration`(Eagle 2.5) 설정(`backbone/eagle2_hg_model/config.json`) 사용 | 논문은 모델 버전을 N1으로 인용. 결과 해석에서 버전을 구분해야 함(N1.5와 N1의 차이는 이 문서에서 대조하지 않음) |
| **D12** | 표 5: 피드백 추론 1.95 ms(시뮬레이션과 하드웨어 동일) | `GR00T_TIMING=1`일 때 구간별 시간을 찍는 코드가 있으나(`gr00t_model.py` L316–L416), 공개된 측정 결과가 없고, 피드백 경로에 `.item()`/`.any()` 동기화와 NaN 검사가 매 호출 들어감(D4). 하드웨어 합 71.01 ms의 카메라·IPC 부분은 실로봇 코드가 없어 확인 불가 | 1.95 ms가 이 검사를 포함한 값인지 알 수 없음 |

### 2.1 코드로 확인된 일치 항목 (M1–M9)

- M1 동결 ImageNet ResNet-18 → `obs_encoder.py`. 매 첫 `forward`에 ImageNet 가중치를 강제로 다시 불러옴(체크포인트 오염 방지).
- M2 투영 d = 256, 디코더 한 층, 헤드 4, FFN 512, 사전 정규화, dropout 0(`feedback_action.py` L82–L92). 속도 머리 256 → `action_dim`(LIBERO·시뮬 설정 7).
- M3 학습 파라미터 수: 입력 차원 1024(GR00T 행동 머리 `hidden_size`)에서 직접 세어 1.186M(`03` V10).
- M4 게이트 초기 0.3, 학습 가능한 스칼라(L96, L404).
- M5 2단계 학습에서 백본·계획기를 동결하고 피드백만 학습, 손실은 `F.mse_loss(pred_actions, gt_actions) × mask`의 마스크 평균(L628–L637). 식 (8)과 일치하고 **속도가 아니라 최종 행동에 대한 회귀**입니다.
- M6 추론 일정 1:16:16: `gr00t_model.get_action`이 `time_step % 16 == 0`일 때만 VLM과 DiT를 실행하고 나머지 스텝은 캐시한 `final_model`, `final_raw_action`에 최신 영상으로 피드백만 적용(L324, L377 이하).
- M7 피드백 모듈은 **손목 카메라(`eye_in_hand`) 영상만** 사용하고 계획기는 두 카메라를 씀(`data_config.py` `PandaHandObsConfig`).
- M8 제어 주파수 10 Hz(`control_freq = 10`, 과제 설정 전부).
- M9 Pick 1D의 성공 기준은 상대 들어올림 10 cm(`toy_red.py` L90, `min_lift_height = 0.1`), 시간 제한 600스텝(60초).

### 2.2 D1–D3의 의미

공식 코드에서 피드백 모듈은 "계획기 마지막 단계의 은닉 상태(1024차원) 토큰 하나와 현재 손목 영상의 8×8 특징에서 속도를 내는 작은 회귀기"입니다. 출력에 `dt`가 곱해져 `a^{k−1}`에 더해지므로 최종 행동은 `a^{k−1} + dt·W(…)`입니다. 손실이 최종 행동 L2 회귀이고 머리가 선형이므로 이는 같은 입력에서 "행동 잔차 `Δa`를 직접 예측하는 선형 머리"와 대수적으로 같은 함수류입니다(`03` §8). 논문의 "디노이징 공간 갱신"은 코드에서는 **입력 특징을 DiT 은닉 상태로 잡았다는 점**과 **출력에 `dt`를 곱했다는 점**으로 구현됩니다. 후자는 최적해에 영향이 없습니다.

## 3. 라이선스와 공개 범위

- **공식 저장소:** Apache-2.0(`LICENSE`), `NOTICE`에 `Copyright 2026 Yiheng Ji`와 NVIDIA Isaac-GR00T 파생 파일 목록(`concat.py`, `data_config.py`, `dataset.py`, `transforms.py`, `flow_matching.py`, `gr00t_model.py`, `eval_checkpoint.py`, `gr00t_train.py`, `module_fetcher.py`)이 있습니다. LIBERO는 MIT(별도 저작).
- **체크포인트:** HF 모델 카드는 Apache-2.0, `base_model: nvidia/GR00T-N1.5-3B`. 원 모델(GR00T N1.5) 가중치의 사용 조건은 확인하지 않았습니다(미확인).
- **서브모듈 포크:** GitHub API 라이선스 NOASSERTION(NVIDIA 원본은 Apache-2.0).
- **공개 산출물에는 공식 코드와 체크포인트를 넣지 않습니다.** `repo/`는 gitignore합니다.

## 4. 구현 가능성 판정 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**판정: 피드백 모듈은 이 맥에서 구현하고 작은 계획기 위에서 시험할 수 있습니다. 논문 수치(GR00T 3B 기반 시뮬레이션·실로봇)의 재현은 이 맥에서 불가능합니다.**

| 대상 | 이 맥에서 | 이유 / 필요한 것 |
|---|---|---|
| 피드백 모듈만 새로 구현 (ResNet-18 + 선형 투영 두 개 + 디코더 한 층 + 게이트 + 속도 머리, 약 1.19M) | **가능** | 순수 PyTorch, 코드 약 100줄. torch는 설치되어 있고 torchvision은 확인하지 않음(ResNet-18은 설치되어 있지 않으면 대체 가능) |
| 작은 계획기(MLP 또는 소형 DiT) + 피드백 모듈 + MuJoCo 이동 목표 과제 | **가능** | mujoco 설치됨. 환경은 직접 짜면 설치 없이 가능, robosuite는 설치 필요(승인) |
| 모듈 단독 추론 시간 측정 (MPS 또는 CPU) | 가능 | 표 5의 1.95 ms는 GPU 값이라 비교 대상이 아님 |
| 공식 코드로 로봇수이트·LIBERO 평가 (체크포인트 15.5 GB) | **불가** | CUDA 가정(`flash_attention_2`, README의 `cu128` torch 2.7.0, `flash-attn==2.7.4.post1`), `pyproject.toml` 고정 목록(`tensorflow==2.15.0`, `transformers==4.51.3`, `numpy<2`, Python 3.10, `torchcodec`, `evdev` 등). 이 맥에서는 설치가 실패하거나 CUDA 경로가 없음. 다운로드는 사용자 승인 필요 |
| 공식 코드로 학습(계획기 + 피드백) | **불가** | 3B 모델 `bf16` 학습, CUDA, 시연 데이터 미공개 |
| 베이스라인(GR00T) 대비 격차 재현 | **불가** | 베이스라인 체크포인트와 시연 데이터가 공개되지 않음 |
| 표 1·2·3·5, 그림 4·5의 재계산·검정 | **가능, 완료** | `analysis/recompute.py` |
| 실로봇(Franka) | **불가** | 코드, 데이터, 체크포인트 없음 |

**노력 추정(측정하지 않음):** 피드백 모듈 PyTorch 구현과 장난감 계획기 시험은 2–4일. 공식 코드를 CUDA 기기에서 로봇수이트 평가까지 돌리려면 환경 구성 1–2일, 체크포인트 다운로드, 평가 반나절–1일이 더 듭니다(데이터가 없어 학습은 포함하지 않음).

## 5. 확인 방법 (실행하지 않음)

공식 코드를 승인 뒤 CUDA 기기에서 돌린다면, 다음 순서가 D 항목을 가장 싸게 확인합니다.
1. `inference_by_task.py`에서 `MODE = "feedback"`, `NUM_EPISODES = 40`으로 Pick 1D를 평가하고, `seed_tracker`가 저장한 속도 이력으로 움직임이 에피소드마다 같은지 본다(D8).
2. 같은 체크포인트로 `denoising_steps`를 2와 4로 바꿔 성공률 차이를 본다(D2).
3. 피드백 모듈의 첫 `forward`가 게이트·가중치 재초기화를 호출했는지 출력 로그(`WARNING: FeedbackDecoder weights corrupted`)를 확인한다(D4).
4. `GR00T_TIMING=1`로 피드백 경로 시간과 `.item()` 동기화 포함 여부를 측정한다(D12).
