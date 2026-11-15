# 04. 공식 코드 대조와 구현 가능성

## 0. 대상과 방법

- 저장소: `Dongwon-Son/TAM` (공식, 프로젝트 페이지의 Code 링크). 커밋 `0cc882865c5509f7ba88690f792ea1368713cf7b` (2026-09-05, "Clarify that DAgger fine-tuning takes place entirely in simulation"). 얕은 복제를 `repo/`에 두었습니다(`GIT_LFS_SKIP_SMUDGE=1`, 서브모듈 없음). 파일 1,618개.
- **정적 읽기만 했습니다.** 코드를 실행, import, 설치하지 않았습니다. 가중치와 데이터는 저장소에 없습니다. 공개 추론 체크포인트(릴리스 자산 `tam_public_multirobot_ckpt3472000.tar.gz`, 16.8 MB, 게시 2026-06-04)는 메타데이터만 확인했고 받지 않았습니다.
- 읽은 파일: `README.md`, `docs/deployment.md`, `pyproject.toml`, `req.txt`, `src/simadaptor/config/{train,encoder,data}.py`, `models/adaptor.py`, `physics/actuator.py`, `physics/smoothing.py`, `tam/entrypoints.py`, `tam/presets.py`, `deploy/inf_util.py`, `deploy/mapping_server.py` (일부), `scripts/train/tam/{train_impl,dagger_finetune_impl}.py` (손실과 설정).
- 논문 식 B.3과 B.4는 PDF를 렌더링해 확인했습니다(추출 텍스트에서는 부호가 흐릿했습니다).

## 1. 설치 요구사항과 라이선스

- 의존성(`pyproject.toml`): einops, flax, matplotlib, mujoco, mujoco-mjx, numpy, optax, scipy, tqdm, tyro, wandb, zarr. 선택 extras는 `train`(torch, 데이터 로딩 CPU 전용), `deploy`(pyzmq), `cpu`(jax), `cuda`(jax[cuda12]), `dev`(pytest).
- README 설치 절차는 `pip install -U "jax[cuda12]"`를 씁니다. "학습은 JAX로 GPU 연산을 한다"고 적혀 있습니다. CUDA가 없는 맥에서는 공식 학습 경로를 그대로 쓸 수 없습니다.
- **라이선스.** `pyproject.toml`은 `license = { text = "MIT" }`라고 적지만 저장소 루트에 LICENSE 파일이 없습니다. GitHub API의 license 값은 `null`입니다. 자산 일부(google_robot, unitree_z1, flexiv_rizon4, kuka_iiwa_14)에만 LICENSE 파일이 있습니다. 코드를 재사용하려면 저자에게 라이선스를 확인해야 합니다.

## 2. 논문 ↔ 코드 대조 (D 항목)

판정: **일치**는 논문 값과 코드 값이 같음, **차이**는 다름(또는 논문에 없는 기능), **미확인**은 저장소에서 찾지 못함.

| 번호 | 항목 | 논문 | 코드 | 판정 | 영향 |
|---|---|---|---|---|---|
| D1 | 학습 제어 주파수 | 1 kHz (Table B.2 "History stream 1 kHz") | `hz_randomization_enable=True`, 선택지 200/500/1000 Hz, 기준 1 kHz (`config/train.py`). `tam-train-*` 진입점이 `--hz-filter 200 500 1000`을 기본으로 넘김 (`tam/entrypoints.py` L187) | **차이 (논문 본문에 없음)** | 학습 데이터가 여러 주기를 섞어 씀. 1 kHz 배포와 학습 분포가 다를 수 있음. 논문 재현 시 이 기본값을 명시해야 함 |
| D2 | DAgger 손실 | §3.3 "same analytic teacher-torque objective", Huber δ = 1e-2 (Table B.2) | DAgger 기본 `tau_huber_delta = 1.0` (`dagger_finetune_impl.py` L119), 추가로 `zero_residual_weight = 0.5`의 영잔차 항 (L118, L1249). 두 값 모두 논문에 없음 | **차이** | δ = 1.0은 δ = 0.01의 100배라서 손실 대부분이 MSE 영역입니다. 논문 §4.4의 DAgger 결과가 어느 설정에서 나왔는지 확인이 필요함 |
| D3 | 토크 어댑터의 관절 공유 | §3.2 "separate MLPs embed the local ... features per joint" 와 D.2.1 "torque adaptor MLP weights are shared across the joint axis" | `models/adaptor.py`의 `SimAdaptorJointwiseFlat`: q, q̇, τ의 Dense 층은 관절 축에 공유됨. 교차 관절 정보는 평균 풀링(`joint_global_proj`)과 "자기 제외 토크 문맥"(`other_joint_tau_context`)으로 들어감 | **논문 내부 불일치**. 코드는 D.2.1과 맞음. 자기 제외 문맥과 관절별 초망(hypernetwork) 출력 헤드는 논문에 설명이 없음 | 논문 설명만으로는 구조를 재구현할 수 없음 |
| D4 | 어댑터 학습 중 히스토리 마스킹 | Table B.1의 증강 목록은 위치, 속도 잡음과 지연만 씀 | `models/adaptor.py`에서 학습 중 창의 앞부분을 무작위로 0으로 가림(`data_mask = arange(nT) >= randint(...)`) | **차이 (논문에 없음)** | 짧은 히스토리 강건성 증강. 효과는 미측정 |
| D5 | 교사 토크 샘플 수 S | Table B.2 "Nominal torque samples per state S = 64" | `tau_map_sample_no: int = 256` (`config/train.py` L72). 진입점도 이 값을 넘김 (`entrypoints.py` L136) | **차이** | 기본 설정은 논문의 4배 샘플. 논문 값이 실제 실험에 쓰였는지 불명 |
| D6 | 이력 길이 | 12 s 롤링 버퍼 (Table B.2, §3.2) | 데이터 설정 `history_duration = 12.0` (`config/data.py`)과 일치. 그러나 README의 fused DAgger와 매핑 서버 예시는 `--attention-history-s 4.0`을 씀 | **일치 (학습), 차이 (배포 예시)** | 공개 예시로 배포하면 논문보다 짧은 어텐션 창을 씀. 논문 결과 재현에는 12 s 설정이 맞는지 확인 필요 |
| D7 | 물리 잔차 특징 e_τ 설정 | ±50 샘플, 가중치 2.0/1.0, 거리 감쇠 1/(1+\|s\|), 리지 1e-6 (Table B.2) | `physics/smoothing.py`: 가중치 `q_weight=2.0`, `qd_weight=1.0`, `taper = 1/(1+|s|)`, `ridge = 1e-6`, `masked_fit_max_neighbors_each_side=50` (`config/encoder.py`) | **일치** | — |
| D8 | 이력 인코더 구조 | 5층, 8헤드, d = 256, 패치 400/200, RoPE, d_z = 64 (Table B.2) | `EncoderConfig`: `num_layers=5`, `num_heads=8`, `d_model=256`, `patch_size=400`, `patch_stride=200`, `rope_base=5000`. `TrainConfig.emb_dim=64` | **일치** (RoPE 기저 5000은 코드 고유) | — |
| D9 | 마찰 모델 | 식 B.3 (두 가지 방향, σ 차) | `physics/actuator.py`: `amp_dir · (σ(slope·(q̇+shift)) − σ(slope·shift))`, 방향별 매개변수 | **일치** (렌더링한 식 B.3에도 음의 가지 앞 부호 없음) | — |
| D10 | 불감대 | 식 B.4, B.7 | `physics/actuator.py`에 순방향(L99–100)과 역방향(L144–155)이 있고, 기울기 `DEADZONE_SLOPE = 1e-2`로 가역성을 맞춤. 토이 T2에서 식 B.4와 B.7의 왕복 오차는 4e-16 | **일치** (수식 수준) | — |
| D11 | 배포 분할 | 인코더 5 Hz, 워크스테이션 GPU, 어댑터 1 kHz, 컨트롤러 CPU (§B.2) | 워크스테이션이 임베딩을 계산해 컨트롤러로 보냄(`docs/deployment.md`). `inf_util.py`의 `export_simadaptor_weights_cpp`가 어댑터 가중치를 C++로 내보냄. 컨트롤러 브리지는 저장소에 없음 | **일치 (구조), 미확인 (1 kHz 실시간 코드)** | 1 kHz 어댑터 구현은 저장소 밖 |
| D12 | 최종 토크 클리핑 | τᶜ = clip(τ⁰ + Δτ, τ_min, τ_max) (§2) | 배포 경로에서 최종 클리핑 함수를 찾지 못함. `actuator_ctrlrange`는 메타데이터 제외 목록에만 나옴 | **미확인** | 한계 한도 동작은 확인 불가 |
| D13 | 불감대 기반 보정 축소 | 논문에 없음 | `adaptor_deadzone_scaling`이 정의되어 있으나 호출은 주석 처리 (`inf_util.py` L847) | **논문에 없음, 미사용** | 실행 경로에 영향 없음 |
| D14 | 기준선 구현 | Online SysID (입자 필터, K = 512, MJWarp), PhysPred, RMA | 저장소에서 입자 필터와 SysID 코드를 찾지 못함 (`particle`, `sysid` 검색 결과 없음) | **미공개** | 표 1–2와 표 3의 기준선은 재현할 수 없음 |
| D15 | 실로봇 체크포인트 | Panda 실로봇 실험에 쓴 가중치 | 공개 체크포인트는 "sanitized multirobot inference checkpoint" (README). 실로봇 Panda 가중치는 README에 없음 | **미공개 (확인)** | 표 1–2의 재현은 가중치만으로 불가 |
| D16 | 로봇 목록 | 5개 사전학습 로봇 (Panda, Piper, RBY-1, iiwa14, Flexiv Rizon4), 테스트 Google Robot과 Unitree Z1 | `presets.py`에 7개 로봇 (같음). README 예시는 이 목록 | **일치** | — |
| D17 | 평가 스크립트 | 표 3 (MuJoCo Menagerie 자산), 소스→OSC 시뮬 평가 | `tam-eval-source-to-osc-sim` (`source_to_osc_sim_experiment.py`, 3,448줄)과 `assets/` 자산 | **일치 (시뮬 평가 경로 있음)** | 표 3의 100회 평가 정확한 프로토콜은 확인 못 함 |

요약: 구조(인코더, 어댑터, e_τ, 마찰, 불감대, 배포 분할, 로봇 목록)는 논문과 맞습니다. 차이는 학습 기본값(D1, D2, D5)과 논문에 없는 증강과 구조 세부(D3, D4)에 몰려 있습니다. 재현에 가장 큰 장벽은 기준선 코드 미공개(D14)와 실로봇 가중치 미공개(D15)입니다.

### 2.1 논문 주장의 코드 쪽 확인

- **C18 (1 kHz 적응).** 어댑터 가중치 C++ 내보내기 함수가 있어 구조와 맞습니다. 실시간 루프는 저장소 밖입니다(D11, D12).
- **정책 재학습 없음.** README와 학습 진입점(`tam/entrypoints.py`의 `train_robot`, `train_multi_robot`, `dagger_finetune`)의 이름과 설명을 보면 TAM 가중치만 학습합니다. 정책 학습 코드는 찾지 못했습니다.
- **DAgger가 시뮬레이션 전용.** 커밋 메시지와 README가 일치합니다. 실로봇 라벨이 없습니다.

## 3. 안정성 증명 검토 (부록 G)

- 식 (G.9)의 소거 $x^\top A x = 0$은 Ṁ − 2C의 반대칭으로 맞습니다. 식 (G.10)–(G.13)의 에너지 천장 논법과 상한 B_s도 맞습니다. 식 (G.3)은 τ = M q̈_r + C q̇_r + g − K_D s + Δτ_TAM과 M q̈ + C q̇ + g = τ + d₀에서 나옵니다. 증명은 옳습니다.
- **그러나 전제가 실험 제어기와 다릅니다.** 논문도 "일반 PD나 OSC 제어기는 식 (G.3)을 만들지 않을 수 있다"고 씁니다. 실험의 관절 임피던스와 OSC는 이 형태가 아닙니다. 또한 식 (G.4)의 토크 상한은 액추에이터 모델을 거쳐 옮겨야 합니다. 논문은 "명령 클리핑만으로는 가정을 세울 수 없다"고 씁니다(D12와 함께 보면 확인이 안 됨). 따라서 G.1의 결과는 조건부입니다.
- 코드는 G.1의 전제를 확인하지 않습니다.

## 4. 구현 가능성 (이 맥)

**맥 환경:** Apple M4 Pro, 통합 메모리 약 24 GiB, MPS 있음, CUDA 없음. 이 환경에서 torch 2.15와 mujoco 4.0은 설치되어 있고 승인되어 있습니다. 이 분석에서는 numpy만 썼습니다. 현재 기본 python(anaconda)에는 torch와 mujoco가 없습니다.

**판정: 전체 재현은 불가능하고, 구조와 토이는 구현할 수 있습니다(조건부).**

| 수준 | 가능성 | 근거 |
|---|---|---|
| 논문 구조의 축소판 (2-링크 토크 잔차, 히스토리 입력, DAgger 1회) | **가능 (완료)** | 토이 T1, 약 3.5분, numpy. `03` §3 |
| 역변환과 교사 토크 검증 (식 B.4–B.8, 식 1) | **가능 (완료)** | 토이 T2. `03` §3.1 |
| 공식 코드 정적 검토와 설정 대조 | **가능 (완료)** | 이 문서 §2–3 |
| 공식 코드로 추론 (공개 체크포인트, 시뮬 평가) | **조건부** | 새 설치(JAX CPU, mujoco-mjx, pyzmq), 공식 코드 실행 승인, 체크포인트 16.8 MB 다운로드 승인 필요. 맥 CPU에서 JAX 추론 속도는 미측정 |
| 공식 코드로 학습 (다중 로봇 200,000 스텝) | **조건부 (사실상 어려움)** | README가 JAX CUDA를 전제. CPU JAX 학습 시간은 미측정. MPS용 JAX(jax-metal)는 실험적이라 설치하지 않았음 |
| 논문 구조를 torch MPS로 재구현 (MJX 대신 자체 시뮬레이터) | **조건부** | 토이 수준은 가능. Panda 규모 학습은 MPS에서 시간 추정이 필요함 (미검증) |
| 표 1–2 실로봇 재현 | **불가** | Franka Panda, 토크 명령 인터페이스, 컨트롤러 브리지, RGB-D와 점군 추정 필요. 저장소 밖 |
| 표 1–2 기준선 (SysID, PhysPred, RMA) | **불가 (코드 미공개, D14)** | 입자 필터와 RMA 정책 코드가 없음 |

**노력 추정.**

- 토이 수준 재구현(이미 끝남): 1–2일.
- 인코더와 어댑터를 torch로 옮기고 MuJoCo로 Panda 시뮬을 구성해 DAgger 1회까지 돌리기: 2–3주. 이 중 씬 구축과 MJX 대체가 가장 큽니다. 추정치이며 측정 전입니다.
- 공식 코드를 맥 CPU에서 실행해 추론만 검증: 2–4일 (설치와 승인 포함). 학습은 측정 전입니다.

## 5. 재실행

이 문서의 검사는 파일 읽기로 끝납니다. 재실행할 스크립트는 없습니다. 공식 저장소를 다시 얻으려면 다음 명령을 쓰되 실행하지는 마십시오.

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --no-recurse-submodules https://github.com/Dongwon-Son/TAM repo
```
