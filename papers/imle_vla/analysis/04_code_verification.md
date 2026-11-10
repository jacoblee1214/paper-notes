# 04. 코드 대조: 논문 ↔ 공식 저장소

**대상:** `kianhk6/IMLE-VLA`, 커밋 `0f08e32`(2026-10-28, README 학습 연산량 갱신). 얕은 복제본은 `repo/`(gitignore)에 있습니다. 라이선스 Apache-2.0.
**범위:** 코드를 읽고 논문의 식, 설정, 수치 정의와 대조했습니다. 실행하지 않았습니다. 학습과 LIBERO 평가는 CUDA GPU와 시뮬레이터가 필요해 이 맥에서 하지 않았습니다(재현 보류).
**구성:** 저장소에는 openpi(JAX, Apache-2.0)와 LeRobot(PyTorch, Apache-2.0) 포크, Triton 엔진(`openpi/realtime-vla-imle/`, Dexmal realtime-vla 기반, MIT)이 들어 있습니다.

## 1. 논문 항목과 일치하는 부분

| 논문 | 코드 | 판정 |
|---|---|---|
| 식 3–4: m개 후보 중 argmin, 갱신은 선택 후보 | openpi `src/openpi/models/pi0.py` 426행(`jnp.argmin`), 선택은 `stop_gradient` 뒤 재순전파. LeRobot `lerobot/src/lerobot/policies/pi05/modeling_pi05.py` 1214행(`dists.argmin`), 선택 후 다시 순전파 | 일치 |
| Algorithm 1: 선택 단계는 기울기 없음, 선택 후보만 역전파 | openpi `_compute_loss_imle`의 Phase 1(`stop_gradient`)과 Phase 2(선택 잡음 재순전파). LeRobot `_forward_imle`의 Phase 1(`torch.no_grad`, eval 모드)과 Phase 2(train 모드) | 일치. LeRobot은 선택 단계를 eval 모드로 돌립니다. 드롭아웃이 있다면 논문과 다른 설정이지만, 현재 설정에서는 영향이 명시되지 않았습니다 |
| 식 1, 2: z ∼ N(0, I), 잠재 차원 = A의 차원 | openpi: `jax.random.normal(noise_rng, (B*k,) + actions.shape[1:])`(chunk × action_dim). LeRobot: `(B*k, chunk_size, max_action_dim)` | 일치(패딩 차원 포함, D10) |
| III-A: VLM 동결, 행동 헤드만 학습 | openpi 논문 설정 `frozen=["paligemma"]`(LIBERO 설정 `libero_50_action_05_lr_005_min_2_factor_correct`, DROID 설정 `16_horrizon_05_lr_2_factor_96_batch`). LeRobot은 `use_lora`가 꺼져 있으면 VLM 전체를 동결(`modeling_pi05.py` 1826–1832행) | 일치 |
| 추론 1회: 접두부 KV 캐시 한 번, 행동 expert 한 번 | openpi `_sample_actions_imle`(518행): 접두부 KV 한 번 계산, `time = 1.0`, 접미부 한 번 통과. LeRobot `sample_actions_imle`(1358행): 접두부 `past_key_values` 한 번, `denoise_step`을 `t = 1.0`으로 한 번 | 일치. t = 1 고정이 논문에 명시되지 않았지만 코드에서는 항상 1입니다 |
| 실물 H=12(π0.5는 H=8) | openpi `examples/droid/main.py` 기본 `open_loop_horizon = 8`, 저장소 README의 실행 명령 `--open_loop_horizon 12` | 일치 |
| 55 Hz의 엔진: Triton + CUDA 그래프, L40S 약 18 ms | `openpi/realtime-vla-imle/`(`pi05_infer_imle.py`), `openpi/docs/triton.md` 83행 "about 18 ms per forward pass on an L40S" | 엔진 존재 일치. 실측 스크립트(Table I 전체)는 없음(D14) |
| 실물 jerk: 관절 위치의 세 번째 차분 / dt³, 관절 평균, 그리퍼 제외 | `openpi/examples/droid/main.py` 99–100행: `JerkMeter("proprio", signal="position", n_dims=7)`(3차). `openpi/examples/jerk_metric.py`: `|Δⁿ| / dtⁿ`, 평균 절댓값 | 식 일치(집계 차이는 D11) |

## 2. 차이와 누락 (D1–D15)

판정 기준: **차이** = 논문과 코드가 다름, **누락** = 논문에 없어 재현에 필요한 정보, **불일치** = 문서끼리 또는 논문과 문서가 맞지 않음.

| ID | 판정 | 내용 | 근거 | 영향 |
|---|---|---|---|---|
| D1 | 차이(작음) | LIBERO 학습 데이터 출처가 두 가지입니다. LeRobot은 `HuggingFaceVLA/libero`, openpi는 `physical-intelligence/libero` | `lerobot/examples/imle_vla/train_libero.sh`, `openpi/src/openpi/training/config.py`의 LIBERO 설정 | 같은 원데이터의 다른 변환일 가능성이 큽니다. 어느 쪽으로 논문 결과를 냈는지는 불명 |
| D2 | 누락 | LIBERO 평가 경로가 세 가지입니다(LeRobot PyTorch `eval_libero.sh`, openpi 웹소켓 `examples/libero/main.py`, Triton `eval_triton_imle.py`). 논문은 어느 경로인지 쓰지 않습니다 | 세 스크립트 | bf16 등 수치 차이 가능성. 수치 차이는 작아 보이지만 확인 전에는 단정하지 않습니다 |
| D3 | 차이(중요) | 표 I의 기준선 "π0.5 JAX 15 Hz"와 IMLE 55 Hz는 엔진이 다릅니다. IMLE 55 Hz는 Triton·CUDA 그래프 엔진에서 나온 값입니다. 같은 엔진의 π0.5 10단계 측정은 저장소에 없습니다 | `docs/triton.md`(엔진 설명), `openpi/realtime-vla/benchmark.py`(Dexmal의 기존 벤치마크, 어느 행을 재었는지 확인되지 않음) | 3.67배 중 헤드 효과는 2.2배로 줄어듭니다(`03` §4) |
| D4 | 누락 | EMA를 씁니다. LeRobot LIBERO는 `use_ema=true`, `ema_decay=0.999`, 평가도 EMA 가중치(`eval_libero.sh`). openpi DROID 설정은 `ema_decay=0.99`, LIBERO 설정은 0.999. 논문은 EMA를 언급하지 않습니다 | `train_libero.sh`, `eval_libero.sh`, `configuration_pi05.py` 123–124행, openpi `config.py` | 재현 시 EMA를 빼면 결과가 달라질 수 있습니다. 논문의 수치가 EMA 가중치인지 불명 |
| D5 | 차이 | 두 LIBERO 모델이 같은 모델의 실행 지평 차이가 아닙니다. H=10은 chunk 50, m=2, 학습률 1.25e-5. H=30은 chunk 100, m=5(`imle_k=5`), 학습률 1.25e-4. 논문은 "m=2 throughout"이라고 씁니다 | `train_libero.sh` 두 레시피(h50, h100), `lerobot/README.md` 체크포인트 표 | Table II의 H=30 행과 Fig. 3이 어느 모델의 결과인지 불명. 논문 본문의 "m=2 throughout"과 H=30 레시피가 맞지 않습니다 |
| D6 | 누락(위험) | 저장소 README는 suite별 체크포인트가 "그 suite에서 가장 좋은 것"이라고 씁니다. 선택 기준(검증 셋인지, 평가 성공률인지)과 절차는 없습니다 | `lerobot/README.md`, 체크포인트 표 | 평가 성공률로 고르면 Table II 수치가 과장됩니다. 논문의 "highest average"는 이 선택에 달렸을 수 있습니다 |
| D7 | 불일치(문서) | 학습 스텝이 문서끼리 맞지 않습니다. LeRobot README는 "약 8시간 후 step 5000 체크포인트"라고 쓰지만 `train_libero.sh`는 10,000 스텝입니다. openpi README는 DROID "step 57000"을 공개했다고 쓰지만 DROID 설정 `num_train_steps`는 256,000입니다 | `lerobot/README.md` 학습 절, `train_libero.sh`, `openpi/README.md`, openpi `config.py` | 어느 스텝의 체크포인트로 논문 결과를 냈는지 불명 |
| D8 | 차이 | 같은 LIBERO 설정의 학습률과 감쇠가 두 구현에서 다릅니다. openpi 논문 설정은 peak 1.25e-5, 최종 1.25e-6, 워밍업 333. LeRobot H=10 레시피는 1.25e-5에서 1.25e-7로, 워밍업 1000 | `train_libero.sh`, openpi `config.py` | 두 구현에서 같은 결과가 나온다고 가정할 수 없습니다 |
| D9 | 차이(무해할 가능성) | 코드에만 있는 옵션: `imle_softmax_temp`(설정에만 있고 학습·추론 경로에서 읽히지 않음), `noise_locked_reactive`(추론, 기본 꺼짐), `continuity_k_candidates`(추론, 기본 1). 논문에는 없습니다 | `configuration_pi05.py` 53, 60, 66행 | 기본값으로 실행하면 논문과 같은 경로입니다. 논문 결과에 쓰였는지는 확인되지 않음 |
| D10 | 차이(작음) | 선택 거리 정의가 두 구현에서 다릅니다. openpi는 모델의 행동 차원(패딩 포함, DROID 32) 전체로 거리를 계산합니다. LeRobot은 `original_action_dim`으로 잘라 거리를 계산하지만, 손실(`loss`)은 패딩을 포함한 전체에 대해 계산합니다 | openpi `pi0.py` 413행(거리), 426행(argmin), 445행(갱신 손실). LeRobot 1164–1166행(차원), 1213행(선택 거리), 1262행과 1273행(갱신 손실) | 패딩 차원의 목표가 0이라 영향은 작을 것으로 보입니다. 논문의 "행동 거리" 정의와 어느 쪽이 맞는지는 명시되지 않음 |
| D11 | 누락 | jerk 집계 방식과 dt가 논문에 없습니다. 클라이언트는 모든 에피소드와 성공 에피소드를 따로 집계하고, 기본 dt는 1.0입니다 | `openpi/examples/jerk_metric.py` 헤더 주석, `examples/droid/main.py` 97–100행 | 비율(2.2–3.0배)은 dt에 무관하지만 집계 방식에 따라 값이 달라집니다 |
| D12 | 차이(중요) | VLA-WC의 정의가 다릅니다. 논문은 "성공 에피소드당 VLA 시간"이지만, 코드는 `client.infer` 호출 시간을 모든 에피소드에서 더하고(웹소켓 왕복 포함) 출력합니다. 성공 에피소드만 거르는 코드는 보이지 않습니다 | `openpi/examples/libero/main.py` 95–97, 170–173, 220–224행 | 실물 Table V의 절대값을 재현하는 경로가 없습니다(`03` §5.1) |
| D13 | 불일치 | 장비 표기가 맞지 않습니다. Table I(55 Hz)와 프로젝트 페이지, 저장소 문서는 L40S입니다. 본문 IV-D는 실물을 A6000에서 돌렸고 π0.5 15 Hz와 IMLE 55 Hz를 그대로 씁니다 | 논문 IV-A, IV-D, 프로젝트 페이지 | 실물의 빈도는 L40S 측정을 옮긴 값일 가능성이 큽니다(확인 불가) |
| D14 | 누락 | Table I의 PyTorch+compile 20 Hz와 Triton π0.5 25 Hz를 재는 스크립트가 IMLE 저장소에 없습니다. Dexmal 저장소의 `benchmark.py`가 있지만 어느 실행이 이 값을 냈는지 기록이 없습니다 | `openpi/realtime-vla/benchmark.py` | 기준선의 재현성이 떨어집니다 |
| D15 | 누락 | LIBERO-plus(Fig. 2) 평가 코드와 설정이 없습니다. 논문에 수치 표도 없습니다 | 저장소 전체 검색(`LIBERO-plus`, `libero_plus`, `liberoplus`: 없음) | Fig. 2는 재현할 수 없습니다 |

## 3. 재현에 필요한 빈칸 정리

논문과 저장소를 합쳐도 다음 값은 확정되지 않습니다.

- 학습 스텝, 배치, 학습률, 감쇠가 두 구현에서 다름(D7, D8). 논문 결과에 쓴 체크포인트 스텝 불명.
- EMA 사용 여부(D4), 체크포인트 선택 규칙(D6), 평가 경로(D2).
- H=30 결과의 모델(D5).
- 실물 VLA-WC의 집계(D12)와 jerk 집계(D11).
- 기준선의 엔진과 측정 스크립트(D3, D14).

## 4. 구현 가능성 판정

- **공식 재현 (학습과 LIBERO 평가):** 가능성은 있습니다. 코드와 가중치(HF `kinster/IMLE-VLA`의 LIBERO 10개와 DROID)가 공개되어 있습니다. 다만 CUDA GPU 1장 이상, LIBERO 시뮬레이터, 그리고 원시 학습 데이터(수 GB)가 필요합니다. 이 맥(Apple M4 Pro, 약 24 GiB 통합 메모리, MPS, CUDA 없음, torch 미설치)에서는 **불가**합니다. 이번 작업은 재현을 보류했습니다.
- **Triton 55 Hz 측정:** CUDA 장비가 필요합니다. 이 맥에서는 불가입니다.
- **실물 재현:** Franka 장비와 DROID 학습 설정(RLDS, 수 TB 규모)이 필요해 불가입니다.
- **아이디어 재구현:** 쉬운 편입니다. cIMLE 손실은 약 30–50줄입니다(후보 m개 생성, argmin, 선택 후보 재순전파). 추론은 접두부 KV 캐시를 두고 행동 expert를 한 번 돌리면 됩니다. 어려운 부분은 (a) π0.5 흐름 체크포인트를 그대로 쓰는 초기화와 t = 1 입력, (b) EMA와 체크포인트 선택 규칙, (c) chunk 길이와 m 선택입니다. 이 셋은 논문만으로 정할 수 없어 임의로 정해야 합니다.
- **종합 판정:** 공식 재현은 GPU가 있으면 가능하지만 이 맥에서는 불가입니다. 아이디어 재구현은 중간 난이도이며, 재구현 결과를 논문 수치와 비교하려면 D4–D8을 먼저 확인해야 합니다. 저장소에는 학습 명령, 설정 이름, 엔진, 체크포인트가 있어 빈칸이 임의 선택보다는 확인 문제로 좁혀집니다.

## 5. 라이선스와 권리

- 저장소 Apache-2.0(LICENSE, NOTICE). Triton 엔진은 MIT(Dexmal). Gemma·PaliGemma 가중치는 별도 라이선스(LICENSE_GEMMA.txt).
- 이 폴더는 저장소의 코드를 복사하지 않았습니다. `repo/`는 gitignore이고, 공개본에는 넣지 않습니다.
