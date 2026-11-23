# 04. 공식 코드와 논문 대조

대상: `github.com/youngcv/artgym` 커밋 `63b94fb3364596db51b7e4651b3c3c98ff994710`(2026-09-14, 전체 커밋 4개)와 서브모듈 `make_data`(`eafa7b6e`), `func_lygra`(`55f60c7e`). **정적으로 읽기만 했습니다.** 실행, import, 설치를 하지 않았고 에셋, 체크포인트, grasp 캐시, 가중치를 받지 않았습니다. 복제본은 `repo/`(gitignore)에 있습니다. 이 문서의 차이 번호는 **D**입니다. 결론은 "저장소 기본 설정이 논문과 다른 곳"이지 "논문이 틀렸다"가 아닙니다. 논문 값이 다른 설정으로 학습된 결과일 가능성이 있고, 저자 확인이 필요합니다.

## 1. 저장소 구성

| 위치 | 내용 |
|---|---|
| `isaacgymenvs/tasks/artmanip.py`(2,504줄) | 환경: 관측, 보상, 목표 전환, 평가 모드 |
| `isaacgymenvs/cfg/{task,object,hand,train}/` | 설정(YAML). 범주별 `object/*.yaml`이 보상 가중치, 임계값, 무작위화 |
| `isaacgymenvs/train.py`, `learning/` | SAPG 교사 학습, 특권 인코더 네트워크 |
| `isaacgymenvs/distill.py`(1,573줄) | 학생 증류(닫힌 루프) |
| `isaacgymenvs/valid_grasp.py`, `eval_consecutive.py` | grasp 검증, 연속 사이클 평가 |
| `isaacgymenvs/deploy/` | Sharpa 손 RPC 서버와 클라이언트(실로봇) |
| `rl_games/` | 수정된 `rl_games` 포크(MIT) |
| `make_data/`, `func_lygra/` | 서브모듈(물체 생성, 기능적 grasp 합성) |

- **없는 것:** 체크포인트, 에셋(`assets/objects`)과 grasp 캐시(`caches/initial_grasp`)는 저장소에 없고 서브모듈로 직접 만들게 합니다. 기준선 코드는 논문에도 없습니다.
- **라이선스:** 루트에 LICENSE 없음(GitHub license: null). `rl_games`는 MIT. `func_lygra`의 LICENSE는 CC BY-NC(Lightning Grasp 상류에서 이어짐). 즉 상업 이용 가능 여부가 불확실하고, grasp 합성 쪽은 비영리입니다.
- **설치 전제(`install.md`):** Linux, RTX 4090, Python 3.8, torch 2.1.0+cu118, IsaacGym TacSL 패키지(Google Drive 링크를 `gdown`으로 받음). RTX 50 계열은 `--custom_tcn`이 필요하다고 README가 적음.

## 2. 일치 확인 (논문 값 = 공개 설정)

| 항목 | 논문 | 코드 |
|---|---|---|
| 관측 차원 | 초기 57, 고유감각 59, 목표 1, 특권 21 | `policyObsDim: 117`(= 57 + 22 + 22 + 1 + 15), `privilegedObsDim: 21`, 초기 관측 구성(손 22, 링크 포즈, 손끝 15, 박스 2x3)이 일치 |
| 보상 가중치 (Tab. 7) | 진행 1000/50/100/100, 안정 50/2/1, 정규화 0.001/1, 낙하 −1, 성공 +50 | `object/*.yaml`의 `GoalDistance1`, `ObjPosDeviation`, `ObjRotDeviation`, `StableContact`, `HandQposDeviation`, `Smooth`, `Drop`, `Success`가 모두 일치 |
| τ, 목표값, 접촉 수 | τ 0.005/0.10/0.05/0.05, 목표 0.04/−1.0/0.3/0.4, `C_target` 3(라이터)/5 | `success_threshold`, `goals`, `contact_num` 일치 |
| 도메인 무작위화 (Tab. 8) | 질량, 감쇠, 강성, 마찰 [2.0, 4.0] | 기본값 x 배율과 일치(예: 칼 감쇠 1000 x [0.7, 1.3] = 700–1300, 마찰 3.0 ± 1.0) (`03` V11) |
| 네트워크 | MLP [512,256,128], LSTM 512 1층, 특권 인코더 [256,128,16], 잠재 16 | `artmanipPrivLSTMPPO.yaml` 일치 |
| PPO/SAPG | γ 0.998, GAE 0.95, KL 0.016, 클립 0.2, 호라이즌 16, 미니 에폭 4, 평론가 계수 4, 엔트로피 0, 탐색 보상 계수 0.005, off-policy 비율 1.0 | 일치 |
| 학생 인코더 | TCN, 은닉 192, 5층, 커널 3, 팽창 증가 2, 드롭아웃 0.05, 헤드 [128], 이력 50 | 일치(`student_encoder`, `proprioHistoryLen: 50`) |
| 시뮬레이션 | 120 Hz, 제어 30 Hz, 1200 스텝 | `dt: 1/120`, `controlFrequencyInv: 4`, `episodeLength: 1200` |
| 에셋 범위 (Tab. 6) | 링크별 범위 | `make_data/configs/batch/*.toml`과 일치, 칼 관절 하한도 생성기 식에서 재계산되어 일치(`03` V12) |
| grasp 합성 | 인스턴스당 후보 1000개, 1초 안정 검증 | `func_lygra` `target_n_result: 1000`, README 검증 `--episode-length 30`(= 1 s) |
| 닫힌 루프 증류 | 학생 잠재가 정책을 구동 | `distill.py`가 학생 인코더 출력을 `actor_encoder_obs_override`로 정책에 넣어 행동을 뽑음 |
| 실배포 목표 전환 | 수동 | `deploy.md`: "goal switching is manual-only" |
| 학습 시 목표 반복 | 열기 후 닫기를 반복 | `goals: [x, 0.0]` 두 값을 교대 |

## 3. 차이 D1–D16

| ID | 항목 | 논문 | 공개 코드(위치) | 영향 |
|---|---|---|---|---|
| D1 | 행동 스케일 | α = 1/40 (식 2, §3.3) | 목표 갱신 `prev + dofSpeedScale x dt x a`, `dofSpeedScale: 1.0`, `dt = 1/120`이므로 α = 1/120 (`tasks/artmanip.py` `pre_physics_step`, `cfg/task/artmanip.yaml`) | 제어 스텝당 목표 이동이 1/3. 최대 목표 속도 0.75 대 0.25 rad/s(`03` V10). 학습 설정이 다르면 정책이 다른 속도 범위를 배움 |
| D2 | 보상 식 형태 | 안정항 `−||v||²`, `−||ω||²`, 손 편차 `−||q − q_init||²`, 평활 `−||a_t − a_{t−1}||²`(Tab. 7) | 선속도·각속도는 **제곱 없는 노름**을 각각 [0, 1], [0, 10]으로 클립, 손 편차도 제곱 없는 노름을 [0, 5]로 클립, 평활은 행동이 아니라 **관절 목표 변화량**의 제곱합을 [0, 50]으로 클립 (`compute_reward`) | 같은 가중치라도 실제 크기가 다름. 표 7의 가중치가 코드 가중치와 숫자로는 일치하므로 식 표기 차이일 가능성도 있음 |
| D3 | 낙하 임계(위치) | 0.05 m (App. C.3) | 칼, 라이터 0.05, **스테이플러, 집게 0.10** (`object/*.yaml` `pos_devia_threshold`) | 두 범주에서 낙하 판정이 두 배 느슨. 회전 임계 1.57은 일치 |
| D4 | 물체 외력 | 무작위화 대상은 질량, 마찰, 관절 파라미터, 관측 잡음(§4.1, App. C.5) | 매 스텝 확률 0.001–0.1(환경별 로그 균등)로 **물체 링크에 무작위 외력**(`질량 x N(0,1) x forceScale 2.0`, 감쇠 0.99/0.08 s)을 가함 (`pre_physics_step`, `forceScale`, `forceProbRange`). 평가 명령도 `--randomize true`라 같은 외력이 들어감 | 시뮬 평가가 논문 서술보다 어려운 조건일 수 있음. 시뮬 SR이 실로봇보다 낮게 나오는 범주(스테이플러, 집게)를 설명할 후보 중 하나이나 확인하지 않음 |
| D5 | 라이터 관절 구동력 | 서술 없음 | 라이터에서 관절 위치가 −1.4 ≤ q < −0.6이면 관절에 **균등 [−1, 0] 구동력**을 가함(`object_is_lighter`, `dof_actuation_forces`) | 뚜껑이 열리는 구간에서 어시스트나 스냅 효과를 흉내 낸 것으로 보이나 논문에 없음. 실물 라이터와의 일치 여부를 알 수 없음 |
| D6 | 평가 허용 오차 | 학습 τ의 2배로 평가(App. C.3) | 평가 모드도 같은 `success_threshold`(= τ)를 씀. `2τ`를 적용하는 코드를 찾지 못함 (`compute_reward`, `_finalize_eval_stage`) | 정적으로 찾지 못한 것이며 호출 쪽 설정으로 바뀔 가능성은 배제하지 못함. 맞다면 시뮬 SR이 더 엄격한 기준 |
| D7 | 목표 유지 시간과 시간 제한 | 서술 없음(시뮬), 실로봇은 도달 즉시 수동 전환 | 학습: 목표 허용 범위 안에서 **무작위 0–2 s** 유지해야 성공으로 세고 다음 목표로 전환, 10 s 안에 못 닿으면 리셋, 20회 성공(= 10 사이클)에서 리셋 (`successHoldDurationRangeSec`, `goalSwitchTimeoutSec`, `maxConsecutiveSuccesses`). 평가: `--goal-switch-interval-secs` 1.5(단일 인스턴스)와 1.0(일괄 스크립트) 유지, 단계별 10 s 제한 (`README.md`, `eval_consecutive.py`) | 시뮬 CSC는 단계마다 1.0–1.5 s 유지가 들어가고 실로봇 CSC는 유지가 없음. 같은 지표 이름이지만 사이클 수가 직접 비교되지 않음(`03` K3, V5) |
| D8 | 보상 커리큘럼 일정 | 시작 에폭 100, 길이 1000 (식 7, App. C.3) | `warmupSteps: 200`, `totalSteps: 2000` (`cfg/task/artmanip.yaml`). 시작 가중치 0.1배, 대상 항(`ObjPosDeviation`, `ObjRotDeviation`, `StableContact`)은 일치 | 코드는 `epoch_num`(에폭)을 단위로 씀. 논문 값의 2배 |
| D9 | 학습률 | 교사 2e-4(Tab. 9), 학생 2e-4 | 교사 `learning_rate: 1e-4`(적응형 KL 스케줄), 학생 README `--lr 1e-4` | 적응형 스케줄이 KL 임계 0.016으로 조정하므로 초기값 차이일 수 있음 |
| D10 | 환경 수와 연동된 설정 | 환경 20,000, 미니배치 40,000, 탐색 블록 4,000 (Tab. 9) | README 환경 16,000(GPU당), 미니배치 32,000, 블록 3,200 | 비율은 같음(환경 수의 2배, 1/5). 논문이 합계인지 GPU당인지 표에 없음 |
| D11 | 증류 손실 | `L = ||z' − z||²` (식 5) | `F.mse_loss`(평균) + `cosine_coef x (1 − cos)`, README는 `--cosine-coef 0.1`, 인자 기본값은 1.0. 학생 인코더를 교사로 초기화하는 옵션이 기본(`--student-init teacher`). 기울기 클립 1.0 (`distill.py`) | 논문의 손실 서술에 없는 코사인 항이 있음. 평균 대 합계 정규화 차이도 |
| D12 | 학생 입력 | 고유감각 59차원(관절 22, 손끝 15, 이전 행동 22)의 이력 50 (식 4, App. C.4) | 학생 이력의 스텝당 차원은 **44**(관절 22 + 이전 행동 22, 손끝 위치 없음). 초기 관측 57은 일치 (`utils/student_obs_utils.py`, `_set_proprioception_buf`) | 학생이 손끝 위치 이력을 보지 않음. 교사와 기반 정책 입력은 손끝 포함. 논문 Fig. 3과 App. C.4의 "고유감각" 정의와 다름 |
| D13 | 평론가 입력 | 서술 없음 | 평론가가 별도 특권 인코더를 갖고 **5차원 이진 접촉 정보**를 추가로 받음 (`critic_policy_contact_dim: 5`, `a2c_sapg_priv_network_builder.py`) | 학습 전용(배포에 안 쓰임). 논문의 정보 흐름 그림에 없음 |
| D14 | 시험 인스턴스 분할 | 학습 30, 시험 5 (§4.1) | 분할 방법이 문서화되지 않음. `instance_id_list: ['']`이면 에셋 폴더의 **모든** 인스턴스를 학습에 씀. README의 학생 평가 예시는 `knife_30`(생성 명령 `--num 30`으로 000–029)에서 `--instance-id 030`을 씀 | 코드로 분할을 재현하려면 저자 확인이 필요. 시험 인스턴스가 학습에 섞이지 않는지 정적으로 확인하지 못함 |
| D15 | 관절 관측 잡음 | 표준편차 0.01 (App. C.5) | 잡음이 **[−1, 1]로 정규화한 관절값**에 더해짐(`_get_hand_proprio_obs`) | 단위가 라디안이 아니라 관절 범위 비율. 가벼운 차이 |
| D16 | 실배포 초기 관측 | 물체를 렌더링 화면에 맞춰 손으로 배치(§4.1, App. C.4) | 배포 문서가 "proprio-only 학생에게는 캐시된 초기 상태(`--use-measured-init-*` false)가 시뮬 리셋 분포에 더 가깝다"고 안내(`deploy.md`) | 일치하지만 의미가 있음: 초기 관측의 손 관절, 손끝 위치, 링크 자세는 **측정값이 아니라 디지털 트윈의 grasp 값**일 수 있음. 실제 물체 자세 오차를 정책이 보지 못함(한계 절과 같은 방향) |

## 4. 구현 가능성 판정 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**전체 재현은 불가능합니다. 이 맥에서 할 수 있는 것은 설정·평가 코드를 읽고 지표를 다시 계산하는 일, 그리고 축소 모형(관절 하나의 물체를 쥔 손의 간단한 시뮬레이션)을 직접 짜 보는 일입니다.**

| 단계 | 필요 | 이 맥 |
|---|---|---|
| 에셋 생성(`make_data`) | numpy와 표준 라이브러리만 import하는 것으로 보임(TOML 읽기 방식은 미확인) | **가능성 높음**(설치 승인 필요). 박스 2개 URDF를 만드는 수준이라 CPU로 수 초 |
| 기능적 grasp 합성(`func_lygra`) | NVIDIA GPU(CUDA 커널, Lightning Grasp 상류), conda py38/39 | **불가**. 범주당 8–24시간(App. C.2), CC BY-NC |
| 교사 학습(SAPG, 환경 20,000개, 2B 스텝) | IsaacGym(Linux, CUDA GPU), RTX 5090 두 장 약 48시간 | **불가**. IsaacGym은 macOS를 지원하지 않고 PhysX GPU 파이프라인이 필요 |
| 증류(학생 TCN) | 교사 체크포인트와 시뮬레이터 | **불가**(체크포인트 없음) |
| 시뮬 평가(`eval_consecutive`) | IsaacGym + 체크포인트 + grasp 캐시 | **불가** |
| 실로봇 배포 | Sharpa Wave 손(22 DoF)과 SDK | **불가**(장비) |
| 논문 표 재계산 | 표 값 | **완료**(`03`, `recompute.py`) |
| 축소 모형(MuJoCo + torch, 관절 하나의 물체, 작은 PPO) | 직접 구현 | **가능**. 단 22 DoF 손과 접촉 모델이 다르므로 논문 수치 재현이 아니라 **방법의 어느 부분(보상 커리큘럼, 관절 감쇠 무작위화, 잠재 증류)이 작은 설정에서도 효과를 보이는지**를 보는 용도 |

- **노력 추정(측정 전 추정):** 에셋 생성 코드 실행 반나절. MuJoCo 축소 모형(손 대신 2–3 손가락 단순 그리퍼, 관절 하나의 물체, 작은 PPO와 증류)은 2–3주. 공식 파이프라인 전체(IsaacGym 환경 재구현 포함)는 CUDA 장비 없이는 현실적이지 않음.
- **가장 가치가 큰 확인:** D1(행동 스케일), D4(외력), D6(평가 허용 오차), D7(유지 시간)이 시뮬 성능 수치에 얼마나 영향을 주는지. 저자에게 학습 설정 파일을 확인하는 것이 시뮬레이션을 돌리는 것보다 빠릅니다.

## 5. 정적 읽기에서 확인하지 못한 것

- 훈련된 정책이 사용한 설정(커밋 이전의 내부 설정일 수 있음). 이 대조는 **공개 저장소의 기본값**과의 차이입니다.
- 에셋과 grasp 캐시를 만드는 서브모듈이 다른 환경 설정 값(예: 인스턴스 폴더 이름과 분할)을 쓰는지(D14).
- `deploy/` 하위의 Sharpa SDK 인터페이스(외부 의존).
