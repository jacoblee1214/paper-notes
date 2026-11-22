# 04. 공식 코드 대조와 구현 가능성

## 0. 대상과 방법

- 저장소: `daimeipo/LeaP` (공식. README와 프로젝트 페이지가 논문 제목, 저자, arXiv:2606.17408을 적음). 커밋 `8eb3bc948b054db3145efb7ad89d37447d6a2c0e` (2026-09-20, "Host the official code and project website under daimeipo"). 얕은 복제를 `repo/`에 두었습니다. 추적 파일 60개, 파이썬 약 1,550줄. 첫 커밋 `55fede3`(2026-09-19, "Release LeaP code and CoRL 2026 project website")부터 6개 커밋이 있고 모두 릴리스 정리와 링크 변경입니다. 조직 쪽 사본(SEU-VIPGroup/LeaP)은 같은 코드입니다.
- **정적 읽기만 했습니다.** 코드를 실행, import, 설치하지 않았습니다. 가중치, 데이터, 시뮬레이터 자산은 저장소에 없고 받지 않았습니다. 프로젝트 페이지의 슬라이드(PPTX 13.7 MB)와 영상(10.3 MB)도 받지 않았습니다.
- 읽은 파일: `README.md`, `configs/leap.yaml`, `requirements.txt`, `docs/VALIDATION.md`, `docs/index.html`, `THIRD_PARTY_NOTICES.md`, `leap/policy.py`, `leap/flow_net.py`, `leap/flow_matcher.py`, `leap/pointnet.py`(인코더와 `create_mlp`), `leap/normalizer.py`(정규화 모드), `leap/robotwin/{train,data,checkpoint,deploy}.py`, `scripts/robotwin.py`(일부), `tests/test_policy_sampling.py`(일부).
- 파라미터 수는 코드의 레이어 모양을 옮겨 적어 손으로 계산했습니다(`analysis/param_count.py`, 코드를 import하지 않음).
- 이 저장소는 "연구판을 하나의 자립 패키지로 정리한 것"입니다. `leap/policy.py` 머리말은 "4단계 연구 상속을 한 클래스로 합쳤고 동작은 논문 주 모델과 같다"고 적고, README는 연구판 체크포인트가 키 이름이 달라 변환이 필요하다고 씁니다.

## 1. 설치 요구사항과 라이선스

- 코어(`leap/`)는 torch ≥ 2.1, torchvision, termcolor, numpy < 2, zarr < 3입니다. RoboTwin 파이프라인은 h5py, PyYAML과 **RoboTwin 2.0 시뮬레이터**(SAPIEN, 에셋, 데이터)가 필요합니다(`requirements.txt`, README).
- README의 학습 환경은 Python 3.10, torch 2.4.1+cu121입니다. 시뮬은 "Linux, NVIDIA GPU, 호환 드라이버, 컴파일용 CUDA 툴킷(nvcc, cusparse.h), Vulkan"을 요구하고 PyTorch3D와 cuRobo(v0.7.8)를 소스에서 컴파일합니다. RoboTwin은 커밋 `9b5491d`(구 `script/eval_policy.py` API)에 고정됩니다.
- **라이선스.** LeaP 기여분은 MIT(`LICENSE` 파일 있음, GitHub API도 MIT). 서드파티는 원 라이선스를 유지합니다: Diffusion Policy(MIT), DP3(MIT), FlowPolicy(MIT), A2A(Apache-2.0, SimpleFlowNet과 흐름 매처의 출처). `licenses/`에 원문이 들어 있습니다. 가중치와 데이터는 배포하지 않습니다(README, 프로젝트 페이지).

## 2. 논문 ↔ 코드 대조 (D 항목)

판정: **일치**는 논문과 코드가 같음, **차이**는 다름, **논문에 없음**은 코드에만 있는 설정, **미공개**는 저장소에서 찾지 못함.

| 번호 | 항목 | 논문 | 코드 | 판정 | 영향 |
|---|---|---|---|---|---|
| D1 | NLL의 정규화 (식 5) | 1/(2H) Σ_h Σ_j [log σ²_j + (a_{h,j} − μ_j)²/σ²_j], 차원 j에 대해 **합** | `_gaussian_nll`: `0.5 * (log_var + (a − mu)²/var + log 2π).mean()`, B, H, d_a 전체 **평균**(`policy.py`) | **차이** | NLL이 d_a(= 14, D10의 파라미터 수 일치로 추정)배 작게 들어감. 식 7을 제곱 노름(합)으로 읽으면 흐름 손실은 ÷112, NLL은 ÷14로 줄어 둘의 상대 비중이 코드에서 8배(=H) 큼. 식 그대로 재구현하면 β = 1의 의미가 달라짐. 상수 log 2π는 그래디언트에 영향 없음 |
| D2 | 로그 분산 범위 | 제한 없음 (식 2) | `log_var = clamp(log_var, −10, 5)` (`LeaPPrior.forward`) | **논문에 없음** | σ ∈ [e⁻⁵, e^2.5] ≈ [0.0067, 12.2]. 학습 안정화 장치이며 클램프된 구간에서는 NLL 그래디언트가 끊김 |
| D3 | 프라이어 입력 | g_φ의 입력은 z_s 하나(식 2, A.2: "two-layer MLP", LayerNorm → 512 → ReLU → Dropout 0.1 → 2d_a) | head는 논문 그대로. 입력은 **관측 3스텝**의 관절 상태를 공유 상태 인코더(Linear(14,128)–ReLU–Linear(128,128))로 보낸 z_s를 이어 붙인 384차원. 이 상태 인코더는 **생성기 조건에도 공유**(`z_obs = [pc_feat; z_s]`) | **일치 (세부 보강)** | 프라이어가 흐름 손실의 그래디언트를 상태 인코더로 보내므로 프라이어와 생성기가 인코더를 같이 학습. A.2가 이 공유를 서술하지 않음 |
| D4 | 정렬 손실 (식 6) | 대칭 InfoNCE, 투영 두 개(Linear–ReLU–Linear, 128차원), 온도 τ 값 **없음** | 같은 구조. `align_temperature = 0.1`, 정규화된 z₀와 a(펼친 H·d_a = 112차원)를 입력, 가중치 0.3 (`configs/leap.yaml`) | **일치, τ는 논문에 없음** | 재현에 필요한 값이 논문에 빠짐 |
| D5 | 옵티마이저, 스케줄, 에폭 | AdamW, lr 1e-4, 코사인, 워밍업 500, 배치 256, 3000 에폭, EMA (표 2) | AdamW **betas (0.95, 0.999), eps 1e-8, weight decay 1e-6**, 같은 lr과 스케줄, **3001 에폭**, EMA 감쇠 min(0.9999, 1 − (1+age)^−0.75), 그래디언트 클리핑 없음 (`train.py`, `leap.yaml`) | **논문에 없음 (세부)** | 3000과 3001은 무시할 차이. betas와 weight decay, EMA 스케줄은 논문에 없음 |
| D6 | 검증 분할 | "과제당 전문가 데모 50개" (§4.1, A.1) | 기본 `val_ratio = 0.02`: 50개 중 `round(50 × 0.02) = 1`개를 무작위로 떼어 **학습은 49개**, 검증 손실을 50 에폭마다 기록 (`data.py`). 정규화 통계는 전체 데이터로 계산 | **차이 (논문에 없음)** | README는 "연구판의 기본값이며 분할과 정규화가 연구 데이터셋과 같다"고 적음. 논문 결과가 49개 데모로 학습됐을 수 있음 |
| D7 | 평가 프로토콜 | 3시드, 시드마다 100회 (표 1) | `scripts/robotwin.py eval`이 공식 `script/eval_policy.py`로 **유효 에피소드 100개** 평가. `--seed`는 RoboTwin 공식 벤치마크 시드. 정책 확률 소스의 시드는 외부에서 설정해야 함 (README, `policy.predict_action` 주석) | **논문에 없음** | "3시드"가 학습 시드인지 평가 시드인지, 둘의 관계가 논문에 없음 |
| D8 | 데이터, 행동 공간, 정규화 | 행동 공간, 정규화 서술 없음 | 상태 = 관절 위치 벡터(`joint_action/vector`, 14차원으로 추정), **행동 = 다음 프레임의 관절 위치**(관절 공간), 점군 1024점 크롭, 장면은 `demo_clean` 기반의 "clean scenes", 모든 값은 **min–max로 [−1, 1]에 정규화**(`limits` 모드). 소스와 NoPrior의 N(0, I)도 이 정규화 공간 (`data.py`, `normalizer.py`, README) | **논문에 없음** | 소스 이득의 크기가 정규화와 행동 공간에 결합됨(`03` §2.3). RoboTwin의 clean 대 randomized 설정이 표 1에 어느 쪽인지 논문에 없음 |
| D9 | 추론 | 프라이어에서 z₀를 뽑고 오일러 NFE 6, 6스텝 실행 (§3.2, 표 2, 3) | `predict_action`: 학습과 같은 재매개화 소스(`eval()`에서도 확률적), 오일러 6스텝, `action_pred[:, 2:8]`(= n_obs − 1부터 6스텝) 실행. Dropout은 `eval()`에서 꺼짐 | **일치** | |
| D10 | 생성기와 파라미터 수 | SimpleFlowNet, adaLN(스케일, 시프트, 게이트), 표 5: LeaP 13.22M, NoPrior 13.01M, 프라이어 head 0.21M | 은닉 512 × 4층, mlp 비율 4, 시간 임베딩 256, 조건은 시간 임베딩에 합산. 레이어 모양으로 센 결과: 전체 **13.217M**(= 논문 13.22M), 프라이어 head 0.2123M(1.61%), 정렬 투영 0.062M, 점군 인코더 0.076M, 생성기 12.85M | **일치** | NoPrior는 head를 빼면 13.005M으로 논문 13.01M보다 0.005M 작음(반올림 한 칸). 논문 0.21M은 head만이고 정렬 투영(0.062M)은 LeaP 총합에 포함 |
| D11 | 흐름 손실 (식 1, 7) | 선형 경로, 목표 속도 a − z₀, t ~ U[0, 1], σ_FM = 0 | `SimpleConditionalFlowMatcher`: x_t = (1−t)x₀ + t x₁, u = x₁ − x₀, `t ~ U[0,1]`, `sigma = 0.0`, 손실은 원소 평균 MSE | **일치** | 평균이므로 D1과 같은 스케일 문제 |
| D12 | 손실 가중치 (식 4, 표 3) | NLL 1.0, Align 0.3 | `nll_loss_weight = 1.0`, `align_loss_weight = 0.3` | **일치** | |
| D13 | 공개 범위 | 기준선 A2A, VITA, NoPrior, BridgePolicy, 절제 변형, 확산 브리지 생성기, 실로봇 | 코드에 **LeaP 본 모델만** 있음. NoPrior 스위치, 시각 입력 프라이어, full Σ, GMM, MSE, 고정 σ, 정렬·NLL 제거, 브리지, DP3/ACT, 실로봇(Franka) 코드와 체크포인트, 데이터 모두 **없음** | **미공개** | 표 1의 LeaP 열 외 모든 열과 표 4–9, 그림 3–7은 이 저장소로 재현할 수 없음. 표 6–9의 정의를 확인할 방법이 없음 |
| D14 | 재현 검증의 범위 | 표 1의 성공률 | `docs/VALIDATION.md`: 단위 테스트 12개(CPU), 연구판과의 가중치 매핑 후 **손실, 그래디언트, 추론의 수치 일치**, 시뮬 한 에피소드 연동(0/1 성공). "전체 3001 에폭 학습, 100회 평가는 **입증하지 않았다**" | **논문 수치 재현 미검증** | 연구판 체크포인트와 `structureflowpolicy_fixedattn_film_state_a2afm` 연구 저장소는 공개되지 않음. 동등성은 저자 주장 |
| D15 | BridgePolicy 코드 | "BridgePolicy has no public code, is reimplemented" (§4.1, A.2) | 해당 없음 (LeaP 저장소 밖). 공개 저장소 `jianghcsr/BridgePolicy`가 2026-05-25 생성, 설명 "official implementation of BridgePolicy", README는 ICML 2026 표기. LeaP arXiv v1은 2026-06-16 | **서술과 공개 상황이 어긋날 수 있음 (미확인)** | LeaP 실험 시점에 공개였는지는 알 수 없음. 표 1의 BridgePolicy 열이 공식 구현이 아닌 재구현값이라는 점은 변하지 않음 |
| D16 | 라이선스 | — | MIT, A2A 유래 파일은 Apache-2.0(헤더에 명시), 가중치와 데이터 없음 | **일치** | 코드 재사용에 장애 없음 |

### 2.1 논문 주장의 코드 쪽 확인

- **"프라이어는 고유수용감각만 본다"**: 코드에서 `LeaPPrior.forward(state)`의 입력은 상태 하나뿐입니다(점군 특징은 생성기 조건에만 들어감, `_encode_obs`). 일치합니다.
- **"같은 (μ, σ²)를 H 스텝에 공유, 잡음은 스텝마다 독립"**: `LeaPPrior.sample`이 `mu.unsqueeze(1) + exp(0.5 log_var).unsqueeze(1) * eps`(eps 모양 (B, H, d_a))로 구현합니다. 일치합니다.
- **"추론에서도 확률적 소스"**: 단위 테스트가 `eval()`에서도 소스가 확률적이고 시드로 재현됨을 확인합니다(`tests/test_policy_sampling.py`). 일치합니다.
- **"μ를 직접 실행하면 0%"**: 코드에 해당 경로가 없고(D13), μ는 H 스텝 공유라 상수 청크입니다(`03` §2.2).

## 3. 정적 관찰 (논문과의 차이는 아니지만 읽을 때 중요한 것)

1. 프라이어의 σ는 NLL 최적해에서 청크 내부 분산을 포함합니다(`03` §2.2). 코드의 `_gaussian_nll`이 `actions`(B, H, d_a)를 `mu.unsqueeze(1)`과 비교하므로 μ는 청크 평균으로 학습됩니다.
2. 대조 손실은 같은 배치 안의 다른 샘플을 음성으로 씁니다. 배치 256개 안에 같은 에피소드의 인접 프레임이 있으면 음성이 사실상 양성에 가까워 학습 신호가 약해질 수 있습니다. 논문과 코드 모두 이를 다루지 않습니다(관찰이며 검증하지 않았습니다).
3. README는 `flow_matcher_sigma`가 경로 잡음이지 학습된 소스 표준편차가 아니라고 구분합니다. 논문의 σ_FM = 0과 소스 σ_φ는 서로 다른 양입니다.
4. 출시 검증은 CPU 단위 테스트와 한 에피소드 연동이 전부입니다(D14). "논문 수치를 재현했다"는 근거는 저장소에 없습니다.

## 4. 구현 가능성 (이 맥)

**맥 환경:** Apple M4 Pro, 통합 메모리 약 24 GiB, MPS 있음, CUDA 없음. 저장소 루트 가상환경에 torch 2.15와 mujoco 4.0이 있습니다. 이 분석에서는 공식 코드를 실행하지 않았고 파이썬 표준 라이브러리, numpy, scipy, torch(토이)만 썼습니다.

**판정: 논문 표의 재현은 이 맥에서 불가능합니다. 방법 자체(소스 프라이어와 세 손실)는 작은 PyTorch 재구현으로 구현할 수 있고, 핵심 주장 하나에 대한 토이는 이미 돌렸습니다.**

| 수준 | 가능성 | 근거 |
|---|---|---|
| 방법의 소형 재구현과 토이 (소스 변형 비교) | **가능 (완료)** | `analysis/toy_source_prior.py`, 약 1.8분, CPU. `03` §3 |
| 표 수치 재계산, 검정, 코드 정적 대조 | **가능 (완료)** | `03`, 이 문서 §2 |
| LeaP 코어를 자체 PyTorch로 재구현 (정책 약 150줄) | **가능** | `leap/policy.py`가 단일 파일이고 레이어 모양이 공개됨. 노력 0.5–1일 (추정) |
| 소형 벤치(상태 기반 Push-T류, MuJoCo 단순 과제)에서 소스 변형 비교 (N(0, I), 상태 평균, 고정 σ, 학습 σ) | **가능** | 데이터 생성과 학습이 CPU/MPS에서 가능한 규모. 노력 1–2주 (추정, 측정 전). 논문 표와의 직접 비교는 아님 |
| 공식 LeaP 코어 단위 테스트 12개 실행 | **조건부** | torch는 이미 있으나 termcolor, zarr < 3, h5py, numpy < 2가 환경에 있는지 확인 전. 공식 코드 실행 승인과 설치 목록 확인이 먼저 (이번 분석은 하지 않음) |
| RoboTwin 15과제 데모 수집, 학습, 100회 평가 (표 1) | **불가** | 시뮬레이터가 Linux, NVIDIA GPU, Vulkan, CUDA 툴킷과 cuRobo·PyTorch3D 컴파일을 요구함. 학습 자체(13M 파라미터)는 작지만 데이터와 평가가 시뮬에 묶여 있음 |
| 표 1 기준선 (A2A, VITA, NoPrior, BridgePolicy), 절제 (표 6–9) | **불가 (코드 미공개, D13)** | 변형별 구현을 처음부터 써야 하고 시뮬이 필요 |
| 실로봇 (그림 4) | **불가** | Franka Research 3, 과제 세트, 데모 100개 |

**노력 추정 (측정 전, 추정치).**

- 토이 수준 (완료): 반나절.
- LeaP 재구현 + 소형 벤치 비교: 1–2주.
- RoboTwin 파이프라인 재현: Linux GPU 장비가 있다면 환경 구성 1–2일, 15과제 × 50 데모 수집 수 시간–수일, 학습은 45회(15과제 × 3시드) 필요하고 한 번당 갱신 횟수는 데모 길이에 따라 약 6만–18만(과제당 윈도 수 ÷ 256 × 3000, 추정). 이 맥에서의 시간은 측정하지 않았습니다. 시뮬이 맥에서 돌지 않으므로 이 맥에서는 의미가 없습니다.

## 5. 재실행

이 문서의 대조는 파일 읽기와 레이어 모양 계산입니다. 파라미터 수 계산은 다음과 같습니다.

```bash
python3 analysis/param_count.py
```

공식 저장소를 다시 얻으려면 다음 명령을 쓰되 실행하지는 마십시오(읽기 전용).

```bash
git clone --depth 1 https://github.com/daimeipo/LeaP.git repo
git -C repo checkout 8eb3bc948b054db3145efb7ad89d37447d6a2c0e   # 얕은 복제에서는 fetch가 필요할 수 있음
```
