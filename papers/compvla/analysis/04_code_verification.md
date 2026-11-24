# 04. 공식 코드 대조와 구현 가능성

## 0. 대상과 방법

- 대상: [compvla/CompVLA](https://github.com/compvla/CompVLA), 커밋 `8393e22e01ee90cd884c8198a4151c9166aca34e` ("Initial release", 2026-11-05), 얕은 복제(`repo/`, gitignore). 약 4,900줄 중 라이선스와 README를 빼면 모델(`models/compvla/`, 약 2,060줄), 학습(`train.py`, 557줄), 데이터 변환(`datasets_convert/`, 약 1,500줄), 보조(`utils/Lie.py`, 139줄)입니다.
- 방법: **정적 읽기만** 했습니다. 코드를 실행, 설치, 임포트하지 않았고 가중치와 데이터를 받지 않았습니다. 논문(v2)의 식, 부록 표와 파일 줄 번호를 대조했습니다.
- 논문 대조 기준은 v2입니다. 코드는 v2 제출 하루 전에 올라왔습니다(2026-11-05 대 11-06).

## 1. 설치 요구사항과 라이선스

| 항목 | 내용 |
|---|---|
| 라이선스 | 코드 Apache-2.0(LICENSE 파일 있음). LeRobot, OpenPI에서 가져온 파일은 원 저작권 표기 유지(README). 데이터 CC BY 4.0 (README) |
| 환경 | Python 3.11 conda, ffmpeg 7.1.1, `torch==2.7.0+cu128`(RTX 5090 기준), `lerobot[pi]` v0.4.3(git 고정), `requirements_extra.txt`(scipy) |
| 게이트 | PaliGemma 토크나이저(`google/paligemma-3b-pt-224`)는 접근 승인과 Hugging Face 로그인이 필요. 초기화 가중치 `lerobot/pi0_base`는 리비전 `26b99b9` 고정 |
| 장치 | 설정 파일 `device: cuda`, 학습 스크립트와 README는 CUDA 전제 |
| GPU 메모리(README, RTX 5090 32 GB, 배치 2, bf16, gradient checkpointing) | 전체 미세조정(4.08B 파라미터) 32 GB에서 OOM. PaliGemma 고정(1.16B) 13.5 GB. LoRA(2.76M) 9.3 GB |
| 데이터 | 원자료 zip 과제당 1개, 총 5.8 GB(풀면 7.1 GB). 변환 후 LeRobot v3.0 |

## 2. 논문 ↔ 코드 대조

### 2.1 차이와 논문에 없는 설정 (D 항목)

| 번호 | 항목 | 논문 | 코드 | 판정 |
|---|---|---|---|---|
| D1 | 임피던스 제어기 (식 13–17, 부록 A.2) | GIC 단순화 변형: 가상 목표, 방향 프레임, 6×6 K_d, D_d, wrench, τ = JᵀF | 저장소에 없음. `impedance`, `stiffness`, `Exp(` 등을 담은 제어 코드가 없고, 모델은 정규화 해제된 (V, g, k, d)를 반환하는 데서 끝남 | **빠짐** |
| D2 | 기준선과 평가 코드 (§5.1) | π0, ForceVLA(3카메라), TA-VLA(DePost-1), DP, ACT를 같은 GIC에 연결 | 없음. `use_compliance_expert=false`가 "속도 + 그리퍼 π0"이고 나머지는 없음. 로봇 인터페이스, 평가 루프, 안전 정지 처리 없음 | **빠짐** |
| D3 | 손실 결합 (식 12) | L = L_a + L_c | 두 expert의 요소별 MSE를 이어붙여 한 번에 평균(`modeling_compvla.py` `forward`). 요소 수는 속도 7 + 관절 7 = 14, compliance 12 + 관절 7 = 19라서 각 요소 가중이 같습니다. 유효 가중은 L_a : L_c = 14 : 19(약 0.42 : 0.58)이고 전체 크기는 1/33 | **차이** (비율 1 : 1이 아님. AdamW라 크기 차이는 영향이 작을 수 있으나 비율은 다름) |
| D4 | 궤적 조건 (식 8) | X̄_{t+i} = X_t⁻¹ X_{t+i}, 정답 TCP 궤적 | 조건은 속도 청크를 적분해 만듭니다(`action_to_traj`). **기본값 `traj_condition_space="normalized"`는 평균/표준편차로 정규화된 속도를 그대로 적분**하므로 조건이 물리 단위의 상대 자세가 아닙니다. `physical`을 고르면 정규화를 풀어 적분하고, 이 경우 정답 자세와 같습니다(속도가 연속 자세의 Log/차분이라 적분이 정확히 되돌림) | **차이** (기본 설정). 논문 결과가 어느 모드인지 서술 없음 |
| D5 | 조건 채널 처리 | "노이즈 입력의 사용하지 않는 패딩 채널에 삽입" | 일치. 노이즈와 데이터 양쪽 조건 채널에 같은 traj를 넣어 흐름 목표가 0이 되고, 손실에서 제외(`losses_cp[:, :, :cp_dim]`과 관절 채널만 사용) | 일치 |
| D6 | 라벨: 작은 힘에서 가상 변위 (§4.1, 식 18) | 변위 크기 = 힘 크기 / k_low. f < f_min에서 k_low = k_max이므로 변위는 f/k_max | f < f_min(병진 1 N, 회전 0.1 Nm)이면 **변위를 0으로 둠**(`compute_stiffness_and_vt_disp`) | **논문에 없음** (소음 억제로 보임) |
| D7 | 힘/토크 필터 (부록 B.3) | "Butterworth low-pass" (차수, 차단주파수 없음) | 10 Hz 표본에서 3차, 차단 0.8 Hz, `filtfilt`(영위상, 비인과). 변환 fps를 바꾸면 재조정 필요하다고 주석 | **논문에 없음** |
| D8 | 크기 클리핑 (부록 B.3) | "magnitude clipping" (값 없음) | 힘 50 N, 토크 10 Nm로 노름 스케일. K_min과 합쳐 변위 상한은 병진 250 mm, 회전 1 rad | **논문에 없음** |
| D9 | 부호 | 언급 없음 | 보상 후 wrench에 부호를 뒤집음(`-wrench_tcp`, 주석 "Invert for compliance control") | 논문에 없음 (의미: 로봇이 환경에 가하는 힘 방향) |
| D10 | 그리퍼 명령 (식 7, g_t ∈ {0, 1}) | 이진 | 폭의 변화에서 두 단계 문턱으로 0/1 라벨을 만듦(`derive_gripper_cmd`, 문턱은 fps = 10에 맞춤). 모델은 이를 연속값으로 flow matching 회귀하고 이진화는 추론 코드(없음)에 달림 | 논문에 문턱과 이진화 서술 없음 |
| D11 | 안전 한계 | 언급 없음 | 예측 강성 양수 보장(K_d ≻ 0, §3), 상하한 클램프가 모델 코드에 없음. 출력은 정규화 해제된 값 | 제어기(D1)에서 처리될 수 있으나 확인 불가 |
| D12 | 감쇠 (식 15) | c_i = 2√k_i (단위 유효 관성) | 제어기 없음. 유효 관성이 1 kg이 아니면 감쇠비는 1/√Λ_eff (`03` S2) | 확인 불가 |
| D13 | 시연 수와 선별 (§5.1: "100 demonstrations") | 과제당 100개 | README: 공개 에피소드 Erase 107, Pivot 115, PnP 109, Dispense 111. 변환은 폴더 이름 정렬 순 앞 100개를 쓰고, 유효성 검사(누락, NaN, 정지 구간)에 실패한 에피소드는 건너뜀. 저장된 수가 100 미만일 수 있음 | 논문에 선별 서술 없음 |
| D14 | 모델 범위 | "각 과제 100 시연" (과제별 모델인지 서술 없음) | 학습 스크립트가 `TASK` 하나를 골라 과제별 데이터셋과 출력 폴더로 학습 | 과제별 학습으로 보임 |
| D15 | 이미지 | 카메라 3대 (전면, 측면, 손목) | 카메라 키 `end_effector`, `front`, `front_right`로 저장. 모델 입력은 224×224로 비율 유지 리사이즈(패딩) | 이름 차이만. 해상도는 논문에 없음 |
| D16 | 보조 관절 예측 (부록 A.1) | "관절각을 보조 목표로 포함, 추론 때 버림" | 두 expert 모두 7개 관절 값(다음 스텝 관절각)을 보조 예측하고 추론 시 버림 | 일치 (두 expert 모두라는 점이 서술에 없음) |
| D17 | 시간 샘플링 | 식 9의 τ ~ U(0, 1) | Beta(1.5, 1) × 0.999 + 0.001(OpenPI 방식)이며 두 expert가 같은 τ를 공유 | **차이** (식 10, 11의 U(0, 1) 서술과 다름) |
| D18 | 실행 길이 | 청크 T = 25 (부록 A.1) | `n_action_steps = chunk_size = 25`가 기본. 실행 길이는 제어기 쪽 문제이므로 논문 실험의 값은 불명 | 확인 불가 |

### 2.2 논문과 일치하는 것

| 항목 | 논문 | 코드 |
|---|---|---|
| 학습 스텝, 배치, 정밀도 | 100,000, 32, bf16, 그래디언트 체크포인트 | `steps: 100000`, `batch_size: 32`, `dtype: bfloat16`, `gradient_checkpointing: true` |
| 학습률 | 최대 2.5e-5, 웜업 1,000, 최소 2.5e-6, 30,000 스텝 감소 | `optimizer_lr 2.5e-5`, `scheduler_warmup_steps 1000`, `scheduler_decay_lr 2.5e-6`, `scheduler_decay_steps 30000` (LeRobot 스케줄러가 웜업을 감소 구간에 포함하는지는 확인하지 않음) |
| 청크와 추론 | T = 25, 10 denoising 스텝 | `chunk_size 25`, `num_inference_steps 10` |
| 라벨 상수 (식 18) | 병진 (500, 200, 1, 15), 회전 (30, 10, 0.1, 2.5) | `K_PARAMS = {force: (500, 200, 1, 15), torque: (30, 10, 0.1, 2.5)}` 구간 선형 감소 |
| 동기화 | 10 Hz, 선형 보간, 가장 가까운 프레임 | `fps: 10`, `resample_to_timeline`, `read_video` |
| 힘 보상 | 중력과 관성 보상, 센서에서 TCP로 adjoint 변환 | `compensate_gravity`, `compensate_inertia`, `Adjoint(T_SENSOR_TO_TCP).T` |
| 조건 차원 | 13 (회전 9 + 위치 3 + 그리퍼) | `traj_dim 13` |
| 초기화 | π0 base, compliance expert는 action expert 복사 | `from_pretrained`가 `cp_expert`를 `gemma_expert`에서 복사 |
| 정책 입력 | F/T 미사용 | `observation.state` 13차원(R 9, p 3, 그리퍼 1), 이미지 3 |
| 추론에서 expert 순서 | action expert → 궤적 적분 → compliance expert | `sample_actions` 일치 |

### 2.3 논문 주장의 코드 쪽 확인

- **"외부 F/T 없이 추론"**: 정책 상태 피처에 F/T가 없고, F/T는 `postprocess_wrench`에서 행동 라벨(K, d)을 만드는 데만 쓰입니다(`raw_to_lerobot.py`).
- **"컴플라이언스 expert는 기준 궤적만 조건으로 받는다"**: compliance expert는 별도의 forward를 하고 action expert 토큰과 어텐션을 주고받지 않습니다. 연결은 `action_to_traj`가 만든 13차원 조건뿐입니다.

## 3. 저장소에 없는 것

임피던스 제어기, 로봇 인터페이스, 평가 루프, 기준선(ForceVLA, TA-VLA, DP, ACT), 학습된 체크포인트, 학습 로그. README는 "모델, 학습 코드, 원자료 변환 코드"를 담는다고 적고 있어 이 범위와 일치합니다.

## 4. 구현 가능성 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**판정: 논문 결과는 이 맥에서 재현할 수 없습니다. 라벨 규칙, 제어 법칙, 토이 수준의 구현은 가능하고 일부는 이 분석에서 끝냈습니다.**

| 단계 | 판정 | 근거 |
|---|---|---|
| 실로봇 평가 (표 1–4) | 불가 | Franka FR3 두 대(리더-팔로워 햅틱), F/T 센서, RealSense 3대, 컨트롤러 구현(D1) 필요 |
| 공식 코드로 전체 미세조정 | 불가 | 파라미터 4.08B. bf16 가중치 약 8 GB, 기울기 약 8 GB, AdamW 상태 약 16 GB로 약 33 GB(추정, README는 RTX 5090 32 GB에서 배치 2가 OOM). 24 GiB 통합 메모리에 맞지 않음. 설정이 CUDA와 `torch 2.7+cu128`을 전제 |
| LoRA 또는 PaliGemma 고정 학습 | 사실상 불가 | CUDA 기준 메모리는 9–14 GB로 맞아 보이나 MPS 동작은 확인되지 않았고 속도는 측정하지 않았습니다. 논문 결과는 전체 미세조정입니다 |
| 원자료 변환(CPU) | 조건부 | 5.8 GB 다운로드, 새 패키지 설치(opencv, pandas, pyarrow, lerobot), 공식 코드 실행에 대한 승인이 필요. 변환 자체는 CPU로 도는 코드 |
| 라벨 규칙 재구현 | 가능, 일부 완료 | 식 18과 d = f/k_low는 수십 줄. 라벨 맵과 1축 접촉 토이는 완료(`03` S1) |
| 제어기(부록 A.2) 재구현 | 가능 | 식 13–17은 numpy/torch로 100–150줄(추정). 식 (5)와 분리 오차 검산은 완료(`03` S2). 시뮬은 MuJoCo(설치 완료)로 가능하나 로봇 모델이 필요 |
| 이중 expert flow matching 축소판 | 가능 (조건부) | 작은 transformer로 합성 접촉 데이터에서 구조(궤적 조건 expert)를 재현하는 것은 CPU/MPS로 가능해 보임 |

**노력 추정 (측정 전):** 라벨 + 제어기 + 1–3축 접촉 시뮬 재구현 3–5일. 합성 데이터로 구조 축소판 재현 1–2주. 논문 수치 수준의 재현은 FR3와 F/T 센서, 40 GB 이상 GPU, 시연 수집이 필요해 이 저장소 범위를 벗어납니다.

## 5. 재실행

정적 대조이므로 실행할 스크립트는 없습니다. 읽은 파일: `README.md`, `configs/config_compvla.yaml`, `scripts/*.sh`, `models/compvla/*.py`, `datasets_convert/{raw_to_lerobot.py, utils.py, features.yaml, tasks.json}`, `utils/Lie.py`, `train.py`(구조만).
