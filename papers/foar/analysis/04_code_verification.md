# 04. 논문 ↔ 공식 코드 정적 대조와 구현 가능성

공식 저장소 `https://github.com/Alan-Heoooh/FoAR`를 `repo/`에 얕은 clone(`--depth 1`, 커밋 `869abd3`, 2025-11-13 "fix typo (issue #6)")으로 받았습니다. **실행, import, 설치, 가중치와 데이터 다운로드를 하지 않았습니다.** 읽은 파일은 다음과 같습니다. `policy/policy.py`, `policy/tokenizer.py`, `policy/diffusion.py`, `train.py`, `eval.py`, `eval_agent.py`, `dataset/realworld.py`(접촉 라벨과 시간 창), `utils/ensemble.py`, `utils/constants.py`, `dataset/constants.py`, `device/robot/flexiv.py`, `script/command_train.sh`, `script/command_eval.sh`, `requirements.txt`, `assets/docs/INSTALL.md`, `README.md`.

## 1. 대조 범위와 빠진 것

- **저장소에 있는 것:** FoAR 본체(점군 인코더, 힘 인코더, 예측기, 융합, 확산 헤드), 학습과 평가 스크립트, 로봇·카메라·그리퍼·F/T 센서 드라이버, 시간 앙상블.
- **저장소에 없는 것:** 표 I–IV의 기준선 코드(ACT, Diffusion Policy, RISE, RISE 변형, FoAR (3D-cls)), 썰기·외란 실험 코드, 평가 지표 계산 코드, 체크포인트, 시연 데이터(README는 닦기·껍질 샘플을 Google Drive로 제공한다고만 적음).
- **라이선스:** LICENSE 파일이 없습니다. GitHub 라이선스 API는 `null`입니다. 저장소 설명은 RA-L 2025와 IROS 2025를 함께 적었지만, IROS 2025 판본은 확인하지 못했습니다.

## 2. 대조표

| 항목 | 논문 | 코드 | 판정 |
|---|---|---|---|
| 점군 인코더 | 희소 3D, 얕은 ResNet, 512차원 (§III-B, App. A) | `ResNet14`(MinkowskiEngine), `out_channels=512`, 복셀 0.005 (`policy/tokenizer.py`, `train.py`) | 일치 |
| 점군 토큰 수 | N_p (수치 없음) | `max_num_token=100` (`tokenizer.py`) | 논문에 없음 (D11) |
| 힘 인코더 | 3층 MLP (64, 128, 512), 정현파 시간 부호화, Transformer (App. A) | `Linear(6,64)–LN–ReLU–Linear(64,128)–LN–ReLU–Linear(128,512)–LN–ReLU`, `PositionalEncoding1D`, Transformer 4+1 (`tokenizer.py`, `policy.py`) | 일치 (LN은 논문에 없음) |
| 힘 입력 길이 | To = 200 (100 Hz) (§III-A, App. A) | `num_obs_force=200`, `force_torque_freq=100` (`train.py`, `eval.py`) | 일치 |
| Transformer 크기 | 4 인코딩 + 1 디코딩, d_model 512, d_ff 2048 (App. A) | `num_encoder_layers=4`, `num_decoder_layers=1`, `dim_feedforward=2048` | 일치 |
| 예측기 백본 | ResNet18 (App. A) | `resnet18(pretrained=True)` (`policy.py`) | 일치. 사전학습 가중치 사용은 논문에 없음 (D9) |
| 예측기 힘 MLP | 두 층 MLP (128, 512) (App. A) | `Linear(6·200, 128)–ReLU–Linear(128, **256**)` (`policy.py`, `force_fc`) | **불일치** (D2) |
| 예측기 출력 | 이어 붙여 선형층 → ϕ (App. A) | `fc = Linear(512+256, 1)`, `sigmoid` | 일치 |
| 융합 | h_t = [h^s; ϕ·h^f + (1−ϕ)·h*] (§III-B) | `force_readout = w·force_readout + (1−w)·flat_embed`, `cat(sparse, force)` (`policy.py`) | 일치 |
| ϕ의 기울기 | 명시 없음 | 융합 가중으로 쓰이고 `detach()` 없음. 행동 손실의 기울기가 예측기로 흐름 | 논문에 없음 (D10) |
| 행동 헤드 | 확산, 학습 100, 추론 DDIM 20 (App. A) | `DDIMScheduler(num_train_timesteps=100)`, `num_inference_steps=20` 기본값 (`diffusion.py`) | 일치 |
| 행동 청크 | Ta = 20 (App. A) | `num_action=20` | 일치 |
| 손실 가중치 | **α = 0.1** (§III-B, §IV-A, App. A) | 학습 기본값 `alpha=10.0` (`train.py` default_args와 argparse). `command_train.sh`는 `--alpha`를 주지 않음 → **10.0** | **불일치** (D1) |
| 접촉 라벨 시간 창 | [t−2 s, t+2 s] (App. A) | 과거 200샘플(2 s) + 미래 `num_action_force`샘플. 스크립트는 300 (3 s). 즉 [t−2 s, t+3 s] (`dataset/realworld.py` L116–124, L322–324; `command_train.sh`) | **불일치** (D3) |
| 접촉 임계값 δ_demo | 값 없음, "과제마다 다를 수 있음" (App. A) | 스크립트: 힘 8 N, 토크 8 N·m (기본 5). 라벨은 두 노름의 OR | 논문에 값 없음 (D4) |
| 배치, lr, 워밍업, 코사인 | 240, 3e-4, 2000, 코사인 (App. A) | `batch_size=240` (GPU당 120, 2 GPU), `lr=3e-4`, `num_warmup_steps=2000`, `get_cosine_schedule_with_warmup` (`train.py`) | 일치 |
| 옵티마이저, 에폭 | 논문에 없음 | AdamW(β = 0.95, 0.999), wd 1e-6, 에폭 1000, 저장 100 (`train.py`, `command_train.sh`) | 논문에 없음 (D15) |
| 반응 제어 임계값 | δϕ = 0.9, δf = 8 N, δt = 5 N·m, ε = 0.006 m, Tf = 5 (§IV-A, App. A) | `cls_threshold=0.9`, `force_threshold=8`, `torque_threshold=5`, `epsilon=0.006`, `num_motion_calc_steps=5` (`eval.py`) | 일치 |
| 반응 제어 로직 | Alg. 1 (ϕ로 분기, 힘 조건부 보정, 두 버퍼) | `eval.py` 루프: `prop < cls_threshold`에서 분기, `cur_force < force_threshold and cur_torque < torque_threshold`일 때만 보정, 보정 방향은 최근 5 단계 평균 위치 | 일치 (D5와 함께 볼 것) |
| 시간 앙상블 | ACT [52]류 (§III-C) | `EnsembleBuffer`, `ensemble_mode act` (k = 0.01 기본값) (`utils/ensemble.py`, 스크립트) | 일치. k 값은 논문에 없음 (D16) |
| 로봇 제어 명령 | "단순 end-effector 위치 제어", 컴플라이언스 없음 (§III-C, App. A) | `send_tcp_pose` → `send_impedance_online_pose` → `switch_mode("cart_impedance_online")` → `sendCartesianMotionForce` (`device/robot/flexiv.py` L190–212). 모드 상수는 `NRT_CARTESIAN_MOTION_FORCE` | **해석 차이** (D5) |
| 그리퍼 힘 | 논문에 없음 | `set_force(30)` (`eval_agent.py`) | 논문에 없음 (D14) |
| 힘 정규화 | 논문에 없음 | `FORCE_MIN/MAX` ±80 N, ±10 N·m, 카메라 좌표로 투영 (`dataset/constants.py`, `dataset/realworld.py`, `eval.py`) | 논문에 없음 (D12) |
| 행동 정규화와 작업 공간 | 논문에 없음 | `TRANS_MIN/MAX`, `WORKSPACE_*`, `SAFE_WORKSPACE_*` 클리핑 (`utils/constants.py`, `eval.py`) | 논문에 없음 (D13) |
| 추론 질의 간격 | N_inference (Alg. 1, 값 없음) | 스크립트 `--num_inference_step 8`, 기본 20 | 논문에 값 없음 (D8) |

## 3. 불일치와 빈칸

### 3.1 불일치

- **D1 (중요) 손실 가중치 α.** 논문은 α = 0.1이라고 §IV-A와 App. A에서 씁니다(§III-B 식은 α를 기호로만 둡니다). 학습 코드의 기본값은 10.0이고, 공식 학습 스크립트는 값을 바꾸지 않습니다. 따라서 공식 학습 설정으로 실행하면 예측기 손실 가중치는 논문보다 100배 큽니다. 논문의 α = 0.1이 학습 결과에 쓰였는지, 코드 기본값 10이 쓰였는지는 체크포인트 없이 확인할 수 없습니다. 어느 쪽이든 논문과 공개 설정이 다릅니다.
- **D2 예측기 힘 MLP 폭.** App. A는 (128, 512)라고 적었지만 코드는 (128, 256)입니다. 결합 차원은 512 + 256 = 768입니다. 논문 설명대로면 1024입니다. 작은 차이지만 정량 비교에 영향을 줍니다.
- **D3 접촉 라벨 시간 창.** 논문은 [t−2 s, t+2 s]입니다. 코드는 과거 2 s와 미래 `num_action_force` 샘플을 씁니다. 공식 스크립트는 300샘플(3 s)입니다. 미래 쪽이 1초 길어, 접촉 직전 3초부터 양성 라벨이 붙습니다. 논문 설명(2초)보다 1초 일찍 양성이 되며, 예측기가 "곧 접촉"을 더 일찍 배우는 방향으로 작용합니다.
- **D5 "단순 위치 제어"와 임피던스 계열 모드.** 논문은 컴플라이언스나 임피던스 제어를 쓰지 않았다고 명시합니다(§III-C, App. A, §V). 코드는 Flexiv의 `cart_impedance_online` 모드를 씁니다(상수 이름과 주석의 "impedance"를 근거로 한 해석). 이 모드가 실제로 어떤 강성과 최대 접촉 힘으로 움직이는지는 저장소에 없습니다(`set_max_contact_wrench`는 정의만 있고 호출되지 않음). 힘 임계값 반응과 함께 보면, 위치 명령이 임피던스 계열 모드로 전달된다는 점이 논문 설명과 다를 수 있습니다. 실험 결과에 영향을 주는 요인이므로 저자에게 물을 질문입니다.
- **D6 평가 스크립트의 인자 불일치 (정적 판독).** `script/command_eval.sh`는 `--max_steps 500`을 넘기지만, `eval.py`의 argparse는 `--max_timesteps`만 정의합니다. 약어 일치도 되지 않으므로 argparse 오류가 예상됩니다. 실행하지 않았으므로 확정은 아닙니다. 설령 통과하더라도 기본 키 `max_steps`는 명령줄 값과 연결되지 않습니다(`eval.py` L42, L331).
- **D7 `--epsilon` 타입.** 파서에서 `type=int`인데 기본값은 0.006입니다. 명령줄에서 `--epsilon 0.006`을 주면 int 변환에 실패합니다. 기본값으로 실행하면 문제 없습니다.

### 3.2 논문에 없는 설정 (재현에 필요)

- **D4** δ_demo 값. 스크립트는 8 N과 8 N·m이지만 논문은 값을 적지 않았습니다.
- **D8** 추론 질의 간격. 스크립트 8, 기본값 20. 논문은 값을 적지 않았습니다. 질의 간격은 반응 제어의 반응 시간을 바꿉니다.
- **D9** 예측기 ResNet18의 사전학습 가중치(ImageNet). 논문은 밝히지 않았습니다.
- **D10** 융합 가중 ϕ의 기울기. 코드는 `detach()` 없이 행동 손실을 예측기로 역전파합니다. 논문 식은 이를 명시하지 않습니다. 이 선택이 α와 함께 예측기 성능을 정합니다.
- **D11** 점군 토큰 최대 수 100, 위치 부호화 범위 800 (복셀 0.005 기준). 논문에 없습니다.
- **D12** 힘 정규화(±80 N, ±10 N·m)와 카메라 좌표 투영. 논문에 없습니다.
- **D13** 행동 정규화 범위와 안전 작업 공간 클리핑. 논문에 없습니다.
- **D14** 그리퍼 힘 30 N, 초기 자세(`ready_pose`)와 그리퍼 임계값 0.02 m.
- **D15** AdamW(β = 0.95, 0.999), wd 1e-6, 에폭 1000, 저장 주기 100. 논문에 없습니다.
- **D16** 시간 앙상블의 k = 0.01 (`ensemble_mode act`). 논문에 없습니다.

### 3.3 저장소의 완전성

- 학습과 평가의 기기 주소가 코드에 고정되어 있습니다. 평가는 로봇 주소, PC 주소, 그리퍼 포트, 카메라 시리얼이 `eval.py`(L190–194)에 적혀 있고, 학습 스크립트는 분산 학습의 마스터 주소를 `command_train.sh`에 고정합니다. 자기 환경에서는 이 값들을 바꿔야 합니다.
- 학습 코드는 NCCL 백엔드와 `torch.cuda.set_device`를 씁니다(`train.py`). 이 맥에서는 그대로 돌릴 수 없습니다.
- `load_state_dict(..., strict=False)`를 학습 재개와 평가에서 씁니다(`train.py` L131, `eval.py` L185). 키가 맞지 않아도 오류가 나지 않습니다. 가중치 적재가 완전했는지 확인하기 어렵습니다.
- 로봇 드라이버는 `flexivrdk` 바이너리(`.so`)가 필요합니다. 저장소에는 없습니다(INSTALL.md가 RDK v0.9의 파일을 복사하라고 씀).
- `requirements.txt`는 `torch==1.13.0`, `open3d==0.18.0`, `diffusers==0.27.2` 등을 고정하고 Python 3.8을 전제합니다. MinkowskiEngine과 PyTorch3D는 requirements에 없고 별도로 빌드합니다(INSTALL.md 3–4단계). 설치 문서는 CUDA 11.4 또는 11.7을 권장합니다.
- 실로봇 설정(`calib/`, 카메라 외부 파라미터, 워크스페이스)은 저장소에 없습니다.

## 4. 구현 가능성 (이 맥: Apple M4 Pro, 약 24 GiB 통합 메모리, MPS, CUDA 없음)

### 4.1 판정

**논문 결과 재현은 이 맥에서 불가능합니다. 구조의 PyTorch 재구현과 반응 제어 로직의 단위 검사는 가능합니다. 로봇 평가는 어떤 경우에도 이 맥에서 할 수 없습니다.**

### 4.2 항목별

| 항목 | 이 맥에서 | 막히는 곳과 근거 |
|---|---|---|
| 표 I–IV의 실로봇 평가 | **불가** | Flexiv Rizon 팔, OptoForce F/T, Dahuan 그리퍼, RealSense 카메라가 필요합니다. 저장소 밖 장비입니다 |
| 공식 학습 (2 GPU, 1000 에폭) | **불가** | NCCL과 `cuda.set_device`를 씁니다. MinkowskiEngine은 CUDA 빌드를 전제로 INSTALL에 적혀 있습니다. 데이터 크기(시연 수 50회, 프레임 수는 미확인)를 받지 않아 시간도 추정하지 못했습니다 |
| 공식 추론과 평가 | **불가** (체크포인트 없음) | 체크포인트가 저장소와 논문에 없습니다. 학습으로 만들어야 하므로 위 항목과 같습니다 |
| 데이터(닦기·껍질 샘플) 받기 | 가능하나 하지 않음 | README의 Google Drive 링크이며 크기를 확인하지 못했습니다. 재현 승인 없이 받지 않습니다 |
| 융합 식, 예측기, 확산 헤드를 PyTorch(MPS 또는 CPU)로 옮기기 | **조건부 가능** | 점군 인코더는 MinkowskiEngine 대체가 필요합니다. 희소 합성곱을 밀집 복셀로 바꾸면 되지만 메모리와 속도를 보장하지 못합니다. 나머지 구성은 표준 PyTorch입니다 |
| 반응 제어 Alg. 1의 상태 기계 | **가능** | `eval.py`의 분기, 보정 방향, 두 버퍼를 합성 힘 신호와 행동 청크로 CPU에서 검사할 수 있습니다. 실제 접촉 동역학은 없으므로 검사 범위는 로직에 한정됩니다 |
| 설치 | **어려움** | MinkowskiEngine(CUDA 또는 저자 포크의 빌드), PyTorch3D, torch 1.13, Python 3.8. 맥 arm64에서 이 조합의 빌드 성공은 확인하지 못했습니다 |

### 4.3 노력 추정 (측정 전, 추정)

- **반응 제어 로직 검사:** 1–2일.
- **구조 재구현 (MinkowskiEngine 대체 포함):** 1–2주. 점군 인코더 대체가 가장 크고, 속도 측정은 하지 않았습니다.
- **공식 학습 재현:** 측정 불가. CUDA 전제이고 데이터 크기를 모릅니다.
- **실로봇 평가:** 저장소 밖 장비가 필요하므로 이 맥에서는 해당 없음.

### 4.4 계획에 쓸 판정

- 재현 계획은 사용자 승인이 필요합니다(`CLAUDE.md`). 이 분석에서는 실행하지 않았습니다.
- 가장 가치 있는 확인은 D1(α), D3(라벨 창), D5(제어 모드)입니다. 세 가지는 코드 읽기만으로도 결과 차이를 설명할 수 있습니다.

## 5. 저자에게 물을 질문

1. 본문 실험에 쓴 α는 0.1입니까, 코드 기본값 10입니까?
2. 접촉 라벨의 미래 창은 2 s입니까, 3 s입니까? δ_demo는 과제별로 얼마입니까?
3. 실험에서 Flexiv `cart_impedance_online` 모드의 강성과 최대 접촉 힘은 얼마였습니까? 논문의 "단순 위치 제어"와 어떤 관계입니까?
4. 표 V(껍질, MLP 인코더)의 FoAR 값 0.588은 표 I의 0.756과 어떤 차이로 생겼습니까?
5. 표 III와 IV의 시행 수와 설정을 알려 주실 수 있습니까?
