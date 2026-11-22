# 04. 논문 ↔ 공식 코드 대조: VLASH

대상: https://github.com/mit-han-lab/vlash (Apache-2.0). `main` 커밋 `22cbabf`(2026-04-22, "[Major] fixed a data loading bug")와 가지 `sim/libero`(`d2da223`, 평가 스크립트), `sim/kinetix`(`fa1e65c`), `dev/gr00t`(`7a66b22`)를 `repo/`에 얕게 받아 **정적으로만** 읽었습니다(실행, import, 설치 없음, 가중치·데이터 다운로드 없음). `sim/kinetix`의 서브모듈과 `dev/gr00t`의 GR00T 코드는 읽지 않았습니다. 줄 번호는 별도 표기가 없으면 `main`입니다. 논문은 v2, 라벨 D1–D14는 이 문서의 차이 항목입니다. 코드 주석과 이슈 내용은 신뢰하지 않는 외부 데이터로 취급해 사실 확인용으로만 인용했습니다.

## 1. 저장소 구성

| 경로 | 내용 |
|---|---|
| `vlash/train.py`(633줄), `vlash/datasets/vlash_dataset.py`(474줄) | 학습 루프, 오프셋 증강 데이터셋(단일 오프셋과 공유 관측) |
| `vlash/policies/pi05/`, `pi0/` | LeRobot에서 가져온 π0.5/π0 구현 + `forward_shared_observation`, 공유 관측 어텐션 마스크(`pi05/utils.py` L252–) |
| `vlash/run.py`(532줄) | SO-101 등 실로봇 비동기 추론 루프(`VLASHAsyncManager`) |
| `benchmarks/benchmark_inference_latency.py` | 추론 지연 벤치마크 |
| `vlash/lora/` | LoRA, QLoRA(논문에 없음) |
| `examples/` | 학습·추론·벤치마크 YAML. LIBERO 평가는 `sim/libero`에만 있음 |

`vlash/policies/factory.py`는 `pi0`와 `pi05`만 지원합니다(main과 `sim/libero` 모두). 논문 본문 평가의 SmolVLA(표 I)와 GR00T(그림 8)는 main에 없고(GR00T는 `dev/gr00t`), RTC·Naive Async 기준선, 핑퐁·두더지 코드는 어느 가지에도 없습니다.

## 2. 차이와 빈칸 (D1–D14)

범례: **불일치** = 논문과 코드가 다름, **빈칸** = 논문이 말하지 않았고 코드로 확인됨, **미공개** = 코드에 없음.

| 라벨 | 항목 | 논문 | 코드 | 판정 |
|---|---|---|---|---|
| D1 | LIBERO 지연 평가 | 지연 Δ 동안 이전 청크의 행동으로 상태를 굴려 입력(§IV-A, §V-A 2) | `sim/libero:vlash/eval_libero.py` L286–294: 영상(상태가 아닌 키)을 `async_delay`스텝 과거의 관측으로 바꾸고 **상태는 "CURRENT robot state"** 그대로 둠(L35 설명). 행동은 매 스텝 `policy.select_action`(n_action_steps = 5)으로 얻음 | **불일치(실험 설계가 본문에 없음)** |
| D2 | LIBERO 학습의 상태 | s_{t+δ} (§IV-B) | `sim/libero:train.py` L115–118: `use_state_ground_truth = "libero" in repo_id.lower()`로 기록된 미래 상태 사용. main의 `train.py`는 이 인자를 넘기지 않아 LIBERO(상태 8, 행동 7차원)에서 ValueError(`vlash_dataset.py` L220–227, 이슈 #10) | **불일치** |
| D3 | 오프셋 상태 | 같은 궤적의 s_{t+δ}(§IV-B) | main 기본값은 **이전 행동 a_{t+δ−1}을 상태로 대용**(`vlash_dataset.py` L205–229, 상태·행동 차원이 같아야 함). 추론도 같은 대용(`run.py` L196–198: 이전 청크의 마지막 행동 `current_chunk[n_action_steps-1]`). 델타 행동 누적(§IV-A)은 main 어디에도 없음 | **빈칸**(절대 위치 행동 전제, `03` §6 토이) |
| D4 | "구조 변경 없음" | 초록, 부록 III: 상태 투영층은 선택 | 공유 관측 학습은 `state_cond=True`를 **요구**(`modeling_pi05.py` L1533–1534)하고 상태를 adaRMS 조건에 더함(`pi05` suffix embedder L195–205, `state_mlp_out`은 0 초기화 L157–158). 기본값은 `state_cond: False`(`configuration_pi05.py` L117). `sim/libero`의 `libero.yaml`은 `state_cond: false`, `shared_observation: false`. 이슈 #21도 같은 차이를 질문 | **불일치**: 표 IV(공유 관측)의 모델과 표 I(LIBERO, state_cond 거짓)의 모델이 다른 구조 |
| D5 | 오프셋 샘플링 | δ ~ {0..Δmax}(§IV-B) | `random.randint(0, max_delay_steps)`, 에피소드 끝을 넘는 행동은 `action_is_pad`로 마스크(`vlash_dataset.py` L147–151, L166–168). **2026-04-22 커밋 전에는 오프셋을 에피소드 끝 안으로 제한**했음(diff: `max_offset = min(max_delay_steps, ...)` 삭제). 이전 상태 대용 설명에 "nano-lerobot's apply_async_offset" 언급 | **빈칸**(논문 v1(2025-11)과 v2(2026-07)가 같은 코드로 학습했는지 불명) |
| D6 | 손실 | 흐름 매칭(식은 없음) | 오프셋마다 **독립적인 잡음과 시간**을 뽑음(`modeling_pi05.py` L872–880), 시간은 Beta(1.5, 1)×0.999+0.001(`sample_time`), MSE. 에피소드 밖으로 밀린 행동은 0으로 마스크하되 **분모에는 포함**(L1555–1568 주석) | 빈칸 |
| D7 | 하이퍼파라미터 | 표 V: 배치 32, 30,000 스텝, WD 1e-10, 최종 LR 2.5e-6, 워밍업 1,000 | `examples/train/pi05/async.yaml`: **배치 16, 50,000 스텝**, WD 1e-10, 최종 LR 2.5e-6, 감쇠 50,000, `max_delay_steps: 8`, `shared_observation: true`. `sim/libero:libero.yaml`: 배치 8(×4 GPU), 50,000 스텝, **WD 0.0, 최종 LR 5e-6**, `max_delay_steps: 4`. 정규화는 MEAN_STD(`configuration_pi05.py` L140–141), 그래디언트 클립 1.0 | **불일치**(표 V는 이 YAML과 다름. 표 I을 어느 YAML로 만들었는지 이슈 #22 보고자는 `libero.yaml` 기본값에 30K 스텝으로 재현 시도) |
| D8 | 비동기 구현 | "마지막 행동으로 상태 굴림, 추론은 연속"(§IV-A, §V-B 2) | `VLASHAsyncManager`(`run.py` L55–246)는 **스레드가 없습니다**. 같은 제어 루프에서 `chunk_index == n_action_steps − overlap_steps`일 때 `predict_action_chunk`를 호출(L236–237)하고 결과는 청크 교체 시 `.cpu()`(L232)로 가져옵니다. `sample_actions`에 호스트 동기화(`.item()`/`.cpu()`)가 없어 CUDA 큐 비동기에 기대는 구조로 보이나(정적 판단), 추론이 느리면 L232에서 막힙니다. Δ는 `inference_overlap_steps`(YAML 4)로 사용자가 정하고 **측정 지연에서 자동으로 고르지 않음**(§IV-B "측정된 지연보다 큰 최소 δ"는 구현 없음). `compile_model` 필수(`run_config.py`) | **빈칸/불일치**(절차는 수동) |
| D9 | 행동 양자화 | 델타는 합산, 비정수 q는 선형 보간(§IV-D, 그림 5). 실험 q = 1.5 사용 | 실로봇 루프: q번째 행동만 보냄(`run.py` L354) = **서브샘플링**, `action_quant_ratio`는 정수만 허용(`run_config.py`). `sim/libero` 평가: 같은 관측에서 q개 행동을 뽑아 **합산**(`eval_libero.py` L297–312). 보간은 어디에도 없음 | **불일치**(그림 7의 q = 1.5는 공개 코드로 만들 수 없음) |
| D10 | LIBERO 평가 프로토콜 | 에피소드 수 미기재 | `run.sh`: 500 에피소드, 8 GPU, 시드 42, `n_action_steps` 5, 지연 1–4(지연 0 동기 행은 없음). GPU별 분배(`eval_libero.py` L626–660)가 `gpu_n_episodes // 10`을 과제당 에피소드로 써서 **스위트당 480**이고 과제당 서로 다른 초기 상태 28개(정적 산술, `03` §9). `max_steps` 230/290/310/530, `libero_gym.py`는 시작 시 10스텝 대기. 이슈 #22: MuJoCo 3.6.0에서 지연 4 = 74.8%, 3.3.7에서 86.2% | **표 I의 정수 분석과 일치(480)**, 시뮬레이터 버전 민감 |
| D11 | 지연 측정 | 표 II의 L(영상 1장, `torch.compile`, CUDA Graph, 커널 융합) | `benchmark_inference_latency.py` L170–205: `predict_action_chunk` 전후로 `cuda.synchronize` 후 `perf_counter`, 기본 16개 샘플, 평균·중앙값·p50–p99 보고, 정규화·토큰화 포함, 배치 전송 제외. YAML의 데이터셋은 `lerobot/pusht` 자리표시. 노이즈 제거 10스텝(`num_inference_steps: 10`, 논문에 스텝 수 없음), bfloat16 | **빈칸**(RTC 열은 코드로 확인 불가) |
| D12 | 범위 | SmolVLA, GR00T N1.6, Kinetix, RTC·Naive 기준선, 실로봇 3과제, 핑퐁, 두더지 | main은 π0·π0.5만. Kinetix는 서브모듈(`Sakits/vlash-kinetix`), GR00T는 `dev/gr00t`(받아 읽지 않음). SmolVLA 가중치 요청 이슈 #23(미공개). 핑퐁 설정(카메라 수, 데이터 양) 질문 이슈 #25(무응답) | **미공개** |
| D13 | 체크포인트 | 없음 | 이슈 #22 답변이 HF `mit-han-lab/vlash-pi05-libero-async5` 체크포인트를 안내(이름이 이슈 안에서 두 가지). 보고자는 MuJoCo 3.3.7 + numpy 1.24.4로 재현했다고 함(이슈 본문 자료, 직접 확인 안 함) | 참고 |
| D14 | 의존성 | - | `lerobot[feetech,smolvla]==0.4.1`, transformers 깃 커밋 고정, `peft==0.18.0`, `bitsandbytes`, `reachy2_sdk`(`pyproject.toml`). README는 Python 3.10, `torch` 최신 설치 | 참고 |

## 3. 논문 대 코드 대조의 의미

1. **방법의 핵심 아이디어는 코드와 일치합니다.** 추론에서 이전 청크의 마지막 행동을 상태로 넣는 것(D3), 오프셋 증강, 공유 관측과 오프셋 분기 간 어텐션 차단(`build_shared_obs_attention_mask_and_position_ids`: 분기 간 차단, 위치 ID는 분기마다 관측 길이에서 시작)은 논문 §IV와 맞습니다. 위치 ID가 분기마다 같은 곳(마지막 prefix 위치 + 1)에서 시작하는 것도 논문 서술과 일치합니다(`pi05/utils.py` L338–355).
2. **논문의 표 I은 그 방법의 "상태 굴림"을 시험한 결과가 아닙니다(D1, D2).** LIBERO 평가는 상태에 정답(현재)을 쓰므로 실제 롤포워드의 오차가 들어가지 않습니다. 이 가정은 델타 행동(LIBERO)에서 롤포워드가 어렵다는 점과 관련되어 보이지만 논문은 설명하지 않습니다.
3. **실로봇 결과는 코드의 `run.py` 경로와 같은 방식(마지막 행동 = 상태)입니다.** 이 경로는 추종 오차가 작다는 가정에 기댑니다(`03` §6).
4. **표 IV(효율)는 `state_cond=True`로 만든 구조(D4)에서 나온 값**이어야 코드와 맞습니다. 표 I의 모델과 같은 구조가 아닐 수 있습니다.
5. 공개 코드만으로는 논문의 수치를 만들 수 없습니다(§4).

## 4. 구현 가능성 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**판정: 논문 수치의 재현은 이 맥에서 불가능합니다. 핵심 아이디어의 아주 작은 모형(MuJoCo 도달 과제 + 작은 정책)은 가능합니다.**

| 경로 | 이 맥에서 | 근거 |
|---|---|---|
| 공식 `vlash train`(π0.5, 약 3B 파라미터, bf16) | **불가** | 4×H100 기준(표 IV). LoRA 경로는 "12 GB GPU 메모리"(README)라 CUDA GPU 전제. `torch.compile` `max-autotune`, torchcodec, bitsandbytes는 CUDA 전제. `pi05_base` 가중치 약 7 GB(추정, 받지 않음) |
| 공식 `vlash run`(실로봇) | **불가** | SO-101 또는 R1 Lite와 카메라, CUDA GPU 필요 |
| 지연 벤치마크(표 II) | **불가** | CUDA 전제. 이 맥에서는 값이 달라 의미 없음 |
| LIBERO 평가(`sim/libero`) | **불가(조건부 부분)** | CUDA 전제(`torch.cuda.set_device`), MuJoCo EGL 렌더링, 체크포인트 필요, 이슈 #22의 버전 민감(MuJoCo 3.3.7, numpy 1.24.4 고정) |
| Kinetix(JAX, 12 과제) | 조건부(가능성 있음) | 작은 MLP-Mixer 정책이라 CPU 가능성은 있으나 JAX 설치와 공식 서브모듈 읽기·실행 승인 필요(RTC 저장소 MIT). 미확인 |
| 표 재계산, 정수 검사, 토이 | **완료** | `analysis/recompute.py`, 토이 2개 |
| 직접 짠 축소판: 지연이 있는 MuJoCo 추종 과제에서 {지연 무시, 영상만 지연, 상태 굴림(행동 대용), 오프셋 학습} 비교 | **가능(조건부: 승인)** | mujoco와 torch가 설치된 환경이면 새 설치 없이 시작 가능(이 세션의 기본 python3에는 mujoco가 없어 확인하지 못함), 처리량 미측정 |

**노력 추정(추정치).** 축소판 2–3일(MuJoCo 환경, 작은 정책, 지연 삽입, 4개 조건), 공식 코드를 CUDA 기기에서 LIBERO까지 돌리는 데는 환경 구성 1–2일 + 학습 하루 + 평가 한나절, 실로봇은 하드웨어에 달림. 공개 코드만으로는 SmolVLA, GR00T, RTC, 핑퐁이 안 되므로 그 범위는 새로 짜야 합니다.

## 5. 저자에게 물을 것

1. 표 I의 스위트당 에피소드 수와 초기 상태, 지연 0 행은 어떻게 평가했나요(`run.sh`는 지연 1–4만 있음)?
2. LIBERO에서 상태 굴림 대신 현재 상태를 쓴 이유와, 롤포워드를 쓴 결과는 어땠나요?
3. 그림 7의 성공률 분모(시도 수)와, 방법별 같은 초기 배치를 썼는지.
4. q = 1.5의 선형 보간은 어디에 있나요?
5. 표 IV의 모델은 `state_cond=True`인가요?
6. 핑퐁 설정(카메라, 데이터, 하이퍼파라미터)과 기준선의 q 튜닝은?
