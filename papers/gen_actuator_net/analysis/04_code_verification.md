# 04. 공식 코드 대조와 구현 가능성

## 0. 대상과 방법

- 저장소: [intelligent-soft-robots/genan](https://github.com/intelligent-soft-robots/genan), 커밋 `c443b8bd3f7459912f398d390fb3b7c3812d870a`(2026-10-13, "Initial commit"), `main` 브랜치. 프로젝트 페이지의 "Code" 링크가 가리키는 저장소입니다.
- 얕은 복제(`repo/`, gitignore)를 **정적으로만** 읽었습니다. 코드를 실행하거나 import하지 않았고 패키지를 설치하지 않았습니다.
- 파이썬 약 2.2만 줄(JAX/Flax NNX + MJX, Hydra 설정, skrl 포크의 JAX PPO)입니다. 논문은 구현 프레임워크를 적지 않습니다. 시뮬레이터가 MJX라는 점만 §4에 있습니다.
- `exps/`(실험 데이터와 GenAN 체크포인트·학습 로그)와 `logs/`(정책 학습 로그·체크포인트)는 **Git LFS 포인터**(약 130 바이트)입니다. 받지 않았습니다. 포인터에 적힌 크기만 읽었고 `exps` 약 8.1 GB, `logs` 약 1.7 GB입니다. 그래서 실제 학습에 쓴 설정 파일(`config.yaml`, `overrides.yaml`)은 확인할 수 없고, 아래 대조는 **저장소의 기본 설정**과 코드를 논문과 비교한 것입니다. 단, 보고된 성공 횟수를 하드코딩한 스크립트의 실행 이름(예: `..._action_pen5000_disag_pen01_...`)은 기본값이 쓰였다는 간접 증거입니다(D13).
- 라벨: D = 논문과 코드의 대조 항목. 판정은 일치 / 차이 / 확인 불가입니다.

## 1. 설치 요구사항, 라이선스, 크기

- **라이선스:** `LICENSE` 파일이 없고 GitHub API의 license가 null이며 `pyproject.toml`에도 표기가 없습니다. 코드와 MJCF 자산의 재사용 조건은 불명입니다. 같은 조직의 다른 저장소는 BSD-3-Clause입니다(`02` §2).
- **환경:** `python==3.10.*`만 지원합니다(README, `requires-python = ">=3.10,<3.11"`; 이유는 로봇 프레임워크가 3.11 이상을 지원하지 않기 때문). PAM 로봇 프레임워크(o80, `launch_pam_mujoco`)를 문서대로 먼저 설치해야 합니다. 의존성: flax, jax(암시), mujoco, mujoco-mjx, optax, orbax-checkpoint, hydra-core 1.3 + hydra-ax-sweeper, `skrl @ git+https://github.com/jan1854/skrl.git@genan`, h5py, tensorboardX.
- **데이터와 체크포인트:** 위 LFS 크기. 데이터 `raw_data.pkl`은 pickle이어서(저자 클래스로 직렬화) 신뢰할 수 없는 pickle을 열지 않는다는 이 분석의 원칙상 읽지 않았습니다.
- **시험:** `test/`에 단위 시험이 있고(`test_an_training.py` 987줄 등), 위치 손실의 "이상적인 액추에이터면 손실 0" 시험(`test_pos_acc_loss_gt`)과 토크 손실 시험(`test_torque_pred_loss_gt`)이 있습니다. 실행하지 않았습니다.

## 2. 논문 ↔ 코드 대조 (D 항목)

| 번호 | 항목 | 논문 | 코드 (파일:줄) | 판정 |
|---|---|---|---|---|
| D1 | 위치 손실의 정규화 | 식 1은 `‖q̂ₜ₊₁ − qₜ₊₁‖²`(정규화 없음). 정규화는 다중 스텝 손실에서만(식 35, 38) | 기본 `normalize_loss: true`(`loss/position_accuracy_loss.yaml`). 오차를 영토크 모델의 관절별 평균 위치 오차로 나눈 뒤 제곱(`actuator_network/loss.py:368–413, 481`), `rollout_len: 1`에서도 적용 | 차이. 식 1에 없는 관절별 가중. 단일 스텝에서는 관절 스케일을 맞추는 효과 |
| D2 | 기울기 계산 | 식 2로 미분 불가능한 시뮬레이터에서도 기울기를 얻을 수 있다(§3.1) | MJX를 통과해 자동미분. 시뮬레이션은 double 정밀도, 입력 이력에는 `stop_gradient`(`loss.py:337–365`, `mjx_backend.py:_compute_torques`) | 일치(동등한 구현). 식 2 자체는 쓰지 않음 |
| D3 | GenAN 층 수 | 표 1: 은닉층 2, 층당 512, tanh | `Linear(32,512)` → `Linear(512,512)`×`num_hidden_layers`(=2) → `Linear(512,4)`, 활성화는 각 선형층 앞(`models.py:46–73`, `actuator_net/step1_hist3.yaml`). 은닉 512층이 3개, 파라미터 54.4만 개(논문 표기대로면 28.2만 개) | 차이 또는 표기 차이. 표 1이 "입력층 뒤 은닉층 수"를 세지 않았을 가능성 |
| D4 | 정책·가치 망 층 수 | 표 3–5: 은닉층 4/3/4(reacher/ball-in-a-cup/탁구), 뉴런 64/128/128 | `layers = [num_neurons] * (num_hidden_layers + 1)`(`rl_tasks/agent.py:31`)이므로 은닉층이 5/4/5. 가치망도 같은 구조 | 차이 또는 표기 차이(D3와 같은 방향의 +1) |
| D5 | 정책 출력 | 식 4: 정책 출력을 tanh로 눌러 Δu = Δu_max·tanh(â) | 평균에 tanh를 적용하고 가우시안 샘플(로그 표준편차 초기 0, 상한 0.5)을 더한 뒤 `[−1,1]`로 클립(`agent.py:39–53`) | 차이(경미). 샘플 잡음은 눌리지 않고 잘림 |
| D6 | PPO 하이퍼파라미터 | 표 3–5: 할인율, λ 0.95, 학습률, 엔트로피, ratio clip, rollouts, mini batches, epochs, grad clip 1.0, value clip 0.2, kl 0.008, 정규화기, 병렬 1024 | `agent/skrl_ppo_tuned_{reacher,ball_in_cup,table_tennis}.yaml`과 `skrl_ppo.yaml`의 값이 표와 모두 같음(예: 탁구 0.9815, 9.699e-5, 0.1029, 128/16/10) | 일치 |
| D7 | 행동·불일치 페널티 가중치 | c_act = 1250, c_disag = 0.025(모든 과제) | 코드 5000과 0.1이고 4관절 **평균**을 씀(`pamy_env.py:355–372`). 5000/4 = 1250, 0.1/4 = 0.025로 논문의 **합**과 같음. Low action penalty는 실행 이름의 `pen1000`(= 250 × 4)과 맞음 | 일치(합과 평균의 차이). 단, 불일치는 표준화 상태의 출력(`output_not_destd`)의 표준편차이고 논문 서술은 토크 단위의 `fθ` 출력 |
| D8 | action repeat | 정책 100 Hz, 행동을 5스텝 반복(A.4), Δu_max = 0.01 | 5번 반복하는 루프 안에서 매 서브스텝마다 `control += action·0.01`(`pamy_env.py:316, 441–444`). 정책 스텝당 제어가 최대 0.05 변하고, 보상이 5개 서브스텝 합으로 누적됨 | 해석 차이 가능. 논문 서술만으로는 서브스텝당 0.01인지 정책 스텝당 0.01인지 정해지지 않음 |
| D9 | 보상 항 | reacher `−‖q−g‖`, 탁구 타격 800 + 착지 5000, 관절 한계 페널티(c_lim = 1) | reacher L2 노름, 탁구 `ball_hit_reward` + `goal_dist_reward`(마지막 스텝의 착지점), 관절 한계는 한계 10° 안에서 0→1 선형 증가를 4관절 합(`pamy_env.py:336–353`) | 일치(10° 폭은 논문에 없음) |
| D10 | ball-in-a-cup 속도 페널티 | A.4.2: r_vel = −‖q̇‖², c_vel = 0.0025 | 환경 기본 `velocity_penalty_weight: 0.0`(`env/pamy_ball_in_cup.yaml:13`). 정렬·공 높이 보상 항도 있으나 0 | 차이 또는 확인 불가(실제 학습 설정은 LFS 로그) |
| D11 | ball-in-a-cup 초기 제어 범위 | 표 4는 reacher(표 3)와 같은 `u_init` 범위(−0.5, −0.6, −0.6, −0.5)–(0.5, 0, 0.4, 0.5) | 기본 `reset_ctrl_lower/upper`가 (−0.5, −0.8, −0.2, −0.2)–(0.5, −0.6, 0.2, 0.2)(`env/pamy_ball_in_cup.yaml:7–8`) | 차이 |
| D12 | ball-in-a-cup 성공 정의 | "공을 컵에 넣음" | 컵 좌표계에서 수평 거리 6 cm 미만, 높이 −7 cm–+0.5 cm, 컵 기울기 25° 미만(`pamy_ball_in_cup.py:322–337`, yaml `success_max_cup_angle_deg: 25`). 실물 성공은 이 판정이 아니라 사람이 센 값일 수 있음 | 확인 불가(논문에 임계값 없음) |
| D13 | 실물 성공 횟수 집계 | 100회 시행의 성공률(Fig. 5) | ball-in-a-cup은 횟수를 하드코딩(75, 56, 74, 55, 76, 실행 이름에 2026-01-21/22 날짜)(`scripts/compute_success_metrics_ball_in_cup.py:20–26`). reacher는 로그에서 계산하며 평균·최대·L2 기준과 1–5°의 여러 임계값을 모두 계산(`compute_success_metrics_reacher.py`), 논문은 평균 2°를 보고. Wilson 구간은 같음(`util/success_rate_cis.py`). 탁구 집계 스크립트는 없음 | 일치(값이 Fig. 5와 같음). 탁구는 확인 불가 |
| D14 | 탁구 환경 세부 | A.4.3: 기록 공 궤적 재생, 접촉 시 반발 모델, EKF | 공 궤적에 무작위 평행이동(±5 cm, ±1 cm, ±1 cm)을 더함(`backend/mjx.yaml:14–15`), 반발 계수 0.78을 세 축 모두와 접선 속도에도 곱하고 스핀은 없음(`util/table_tennis.py:82, 104`), 정책 관측에 정규화 시간이 들어감(`env/pamy_table_tennis.yaml`의 `finite_horizon_time_obs: true`). 공 궤적 데이터는 LFS(`assets/ball_trajectories.hdf5`)라 개수를 확인할 수 없음 | 차이(논문에 없는 설정 3개) |
| D15 | 탐색 궤적 생성기 | A.1: 0.5 s마다 제어를 뽑아 3차 스플라인, 잡음 σ ~ U(0.0005, 0.003), 클립 | 기본 `motion_generator: ou`(Ornstein–Uhlenbeck, `collect_data.yaml`). 스플라인 설정(`motion_generator/spline.yaml`)은 논문과 같은 값(간격 0.5 s, 잡음 0.0005–0.003) | 확인 불가(어느 생성기로 모았는지는 LFS 설정) |
| D16 | 속도 한계 처리 | A.1: 한계 초과 시 Δu를 10으로 나눠 한계의 80 % 아래로 | `0.1·Δu + N(0, 0.002²)`를 `[−0.01, 0.01]`로 클립, 해제 기준 80 %(`scripts/collect_data.py:169–203`). 속도 한계 (250, 200, 400, 400)°/s는 같음 | 일치(잡음 항은 논문에 없음) |
| D17 | 데이터 전처리 | 유한차분 속도·가속도(부록 B) | 후진 차분 속도, 중앙 차분 가속도 + MuJoCo 역동역학(`process_data.py`)은 일치. 그 앞에서 위치를 균일 격자로 보간(`data_processing/filter/filter.py`)하고 **제어를 1스텝(2 ms) 당김**(`actuation_delay: 1`, `process_data.py:93`) | 일치 + 논문에 없는 처리 2개 |
| D18 | 학습·평가 궤적 수 | 학습 2,500개(80/20), 테스트 800개 | 학습 데이터 앞 200개를 버림(`num_warmup_trajectories: 200`, `train_actuator_network.py:368`). 평가 스크립트가 학습 2,000 + 검증 500, 테스트 `1000 − 200 = 800`을 씀(`actuator_net_episode_accuracy.py:177–180`). 즉 논문의 2,500과 800은 원래 모은 개수(추정 2,700과 1,000)에서 200개씩 뺀 값 | 일치(추론). 제외 서술은 논문에 없음 |
| D19 | 분할·조기 종료·최적화 | 궤적 단위 80/20, 검증 손실 최소 모델, Adam 1e-4, 배치 512 | 고정 시드 42 순열, `floor(0.8N)`, 구성원별로 검증 손실이 가장 낮은 상태 보존, `adamw`(weight decay 0.0)(`train_actuator_network.py:37–52, 150–230`). `num_epochs`는 필수 인자(`???`)라 150은 확인 불가 | 일치(epoch는 확인 불가) |
| D20 | GenAN 입력 | q와 u의 H=3 이력을 현재값 대비 차이로, 표준화 | 위치·제어 이력만 쓰고 속도는 쓰지 않음(`velocity_input: false`), 차이 이력과 입력 표준화(`pack_an_input.py`, `step1_hist3.yaml`). 입력 32차원 | 일치 |
| D21 | 앙상블 | 5개, 시드·데이터 순열이 다름. 시뮬 스텝마다 무작위 구성원 | 구성원별 초기화(vmap 분할 RNG)와 순열(`loss.py:get_permutation`), 인스턴스·스텝마다 무작위 구성원(`mjx_backend.py:_compute_torques`) | 일치 |
| D22 | 시뮬레이션 모델과 인코더 | MJX 500 Hz, 4자유도 팔, 컵 107 g, 공 54 g, 줄 20 cm | MJCF에 관절 감쇠·관성 보정 없음, 모터 4개(토크 구동), `timestep 0.002`. 컵 세 부품 질량 합 0.10703 kg, 공 0.054 kg, 줄 `range [0, 0.2]`(`assets/.../cup.xml`, `cup_ball.xml`, `util/model.py`). 라켓 0.371 kg(논문에 없음). 인코더 눈금 20000/18400/14400/14400을 흉내 내는 `simulate_encoder_quantization` 옵션(필수 인자)이 정책 **관측**에 양자화를 넣음(`util/encoder_simulation.py`, `physics_backend.py:386`) | 일치 + 논문에 없는 양자화 옵션(사용 여부는 LFS 로그) |
| D23 | 공 추적 잡음 | 컵: σ 0.5 cm, 누락 5 %, 버퍼 5 | `simulated_ball_tracking_noise_std 0.005`, `dropout_rate 0.05`, `ball_pos_buffer_length 5`(`backend/mjx.yaml`) | 일치 |
| D24 | 정책 학습 예산, 평가 시행 수 | 논문에 학습 스텝 수 없음. 전이 평가 100회 | `num_timesteps` 기본 300,000(`train_rl_agent.yaml`), `evaluate_transfer.yaml`의 `num_episodes` 기본 20 | 확인 불가(실제 값은 LFS) |
| D25 | UAN 기준선 | 표 2: c_pos 5.769, c_exp 17.56, c_smooth 0.6789, 병렬 1024, 에피소드 2 s, 탐색 200세트(v2는 500) | 튜닝 설정에 `reward_scale_pos 5.769`, `reward_scale_smoothness 0.6789`, 지수 70.23(= 17.56 × 4, 평균 대 합의 차이), PPO 값은 표 2와 같음. Ax 탐색 `max_trials: 500`, 조기 종료 `max_epochs_without_improvement: 200`(`train_actuator_network_rl_tuned.yaml`, `tune_actuator_network_rl.yaml:31–35`) | 일치(탐색 횟수는 설정 500, v3 서술 200, 어느 쪽이 실제인지 확인 불가) |
| D26 | SysID 기준선 | 표 6–7: 격자 2–50 ms, 8 재시작, 3,000 스텝, 배치 128, 학습률 0.015, 롤아웃 300, 풀리 반경 (38.5, 25.1, 23.5, 17.0) mm | `sysid_muscle_model.yaml`, `actuator_net/analytic_muscle.yaml`의 값이 같음. `max_lag_time_constant: 0.05`로 압력 지연 상한이 50 ms에 걸려 있어 식별값 50 ms가 이 상한과 같음 | 일치(V11의 격자 끝 관찰과 연결) |
| D27 | 데이터 구성 | 하나의 탐색 데이터셋 설명(§3.1, A.1) | `exps/`에 네 폴더: `2026-01-21_green_cup`(컵), `2026-01-22_black_racket`, `2026-01-23_black_racket_testset`, `2026-05-05_black_heavy_racket`. 폴더 이름의 날짜로 보면 reacher·ball-in-a-cup 정책 학습 로그는 2026-01-21/22, 탁구 정책 학습 로그는 2026-05-05입니다. 테스트 셋은 2026-01-23 | 확인 불가. 과제마다 말단 장치와 데이터가 다르다면 논문에 표가 필요. A.2의 "전이 실험 뒤에 테스트 셋을 모음"이 탁구 이전의 전이 실험에 대한 서술일 가능성 |

### 2.1 논문 주장의 코드 쪽 확인

- 토크 라벨 = MuJoCo 역동역학 + 유한차분(부록 B): 코드 `compute_torques`가 `mujoco.mj_inverse`를 씁니다(`process_data.py`). 일치.
- 평가 지표: 시뮬 정확도는 데이터 상태에서 시뮬레이터를 초기화해 기록된 명령을 재생하고 위치 오차를 잽니다. 1 step과 500 steps 오차를 따로 로깅합니다(`simulation_accuracy_evaluator.py`). 평가는 각 궤적의 **마지막** 구간을 씁니다(`[:, -total_length:]`). 일치.
- reacher 성공 기준 "평균 2° 미만": `successes_avg_deg[2.0]`이 같은 기준입니다. 일치.

## 3. 한계와 확인하지 못한 것

- 실제 학습에 쓴 설정 파일과 GenAN·정책 체크포인트는 LFS라 확인하지 못했습니다. D10, D11, D15, D18, D24, D27의 "확인 불가"가 여기서 옵니다.
- 정책 학습 로그의 스칼라(보상 곡선 등)와 시드 수도 LFS에 있어 보지 못했습니다.
- 데이터 원자료(`raw_data.pkl`, 약 1.1 GB 외)는 열지 않았습니다. 수집 궤적 수(추정 2,700)는 위 D18의 추론입니다.

## 4. 구현 가능성 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**판정: 논문의 핵심 아이디어(위치 손실로 학습하는 GenAN)의 축소 재구현과 식 검증은 CPU로 가능하고 이미 끝냈습니다. 공식 코드로 학습·평가하는 것은 조건부(새 환경, LFS 약 8 GB 받기, 승인 필요)이고, 정책 학습과 실물 재현은 이 맥에서 하지 않습니다.**

| 수준 | 판정 | 근거 |
|---|---|---|
| 실물 재현(Fig. 5 성공률, Fig. 4 정확도) | 불가 | PAMY2 로봇, 공 추적(Vicon, 시각 추적), 발사기, 압력 제어 프레임워크가 필요 |
| 공식 코드로 정책 학습(3과제 × 설정 5개) | 사실상 불가 | MJX 1,024 병렬 환경 PPO는 GPU 전제. CPU 시간은 측정하지 않음 |
| 공식 코드로 GenAN 학습(150 epoch, 학습 2,000 궤적) | 조건부 | python 3.10 새 환경, jax(CPU), mujoco-mjx, orbax, hydra-ax, skrl 포크 설치와 LFS 데이터(약 1 GB) 받기 필요. A100에서 25분이라 CPU에서는 수 시간 이상(추정, 미측정) |
| 공식 데이터로 PyTorch 재구현(위치 손실 GenAN) | 조건부 | LFS 데이터 받기와 신뢰할 수 없는 pickle 처리 필요. 구현은 `M(q)`를 MuJoCo에서 받아 식 2로 기울기를 얻는 방식이면 단순함 |
| 식 2 검증, 질량행렬 이방성, 손실 비교 장난감 | **가능, 완료** | `analysis/toy_position_vs_torque_loss.py`(약 20초, mujoco + torch) |
| 합성 액추에이터로 앙상블·불일치 페널티 장난감 | 가능 | 노력 2–3일(추정) |
| MuJoCo CPU로 3과제 시뮬레이션 재구현 | 조건부 | 노력 2–4주(추정). 정책 학습은 GPU 병렬 없이 느림 |

노력 추정은 모두 측정 전의 추정입니다.

## 5. 재실행

```bash
python3 analysis/toy_position_vs_torque_loss.py                       # 자체 4링크 사슬(기본)
python3 analysis/toy_position_vs_torque_loss.py --assets <MJCF 폴더>   # 저자 저장소의 xml 폴더를 데이터로 읽을 때
```
