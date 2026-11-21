# 04. 공식 코드 대조와 구현 가능성

## 0. 대상과 방법

- **대상:** `nicklashansen/tdmpc2`, commit `332f19e4aba4bffe2dc736f45ea7f7bdc585903a` (2026-10-20). 얕은 복제는 `repo/`에 있습니다(gitignore).
- **방법:** 정적 읽기만 했습니다. `tdmpc2/tdmpc2.py`, `tdmpc2/common/{world_model,layers,math,scale,parser,buffer,__init__}.py`, `tdmpc2/config.yaml`, `tdmpc2/trainer/online_trainer.py`, `tdmpc2/envs/__init__.py`, `docker/Dockerfile`, `docker/environment.yaml`, `datasets/download_mt80.sh`를 읽었습니다. 코드를 실행하거나 import하거나 설치하지 않았습니다. 체크포인트와 데이터셋은 받지 않았습니다.
- **논문 기준:** `analysis/01_paper_analysis.md`와 `paper/paper.txt`의 표 6, 8, 9와 부록 H, 식 3–8.

## 1. 설치 요구사항과 라이선스

| 항목 | 내용 | 확인 |
|---|---|---|
| 라이선스 | `LICENSE` 파일: MIT, "Copyright (c) Nicklas Hansen (2023)". GitHub API spdx: MIT | 확인 |
| 기본 환경 | Docker `pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime`. conda 환경 `python=3.11`, `torch==2.7.1`, `mujoco==3.1.2`, `dm-control==1.0.16`, `gymnasium==0.29.1`, `tensordict==0.8.3`, `torchrl==0.8.1`, `hydra-core==1.3.2`, `wandb==0.17.4` 등 | 확인. 모두 고정 버전 |
| Meta-World | README: MuJoCo 2.1.0과 `gym==0.21.0` 필요. `environment.yaml`에서 해당 줄은 주석 처리됨 ("unmaintained") | 설치 어려움 |
| MuJoCo 라이선스 키 | README는 `tdmpc2.com/files/mjkey.txt`를 "unrestricted MuJoCo 2.1.0 license (courtesy of Google DeepMind)"로 안내 | **받지 않음.** 재배포 키의 조건을 확인하지 않았음 |
| ManiSkill2 | README: `python -m mani_skill2.utils.download_asset all`로 자산 다운로드, `MS2_ASSET_DIR` 설정 | 자산 크기 미확인 |
| 데이터셋 | `datasets/download_mt30.sh`, `download_mt80.sh`: Hugging Face `nicklashansen/tdmpc2`의 `mt80/chunk_0..19.pt` (20개 조각) | **받지 않음.** 크기는 미확인이며 논문의 545M 전이로 보아 대형으로 추정 |
| 체크포인트 | README: 단일 과제는 `model_size=5`, 다과제는 1–317M 여러 크기 | 받지 않음 |

## 2. 논문 ↔ 코드 대조 (D 항목)

표 8과 코드의 기본 설정 값은 대부분 일치합니다. 일치하는 항목을 먼저 적고, 차이(D)를 뒤에 적습니다.

### 2.1 일치 확인

| 항목 | 논문 (표 8, 식) | 코드 (`config.yaml`, 해당 파일) |
|---|---|---|
| 배치, UTD | 256, 1 | `batch_size: 256`, 온라인 루프에서 단계당 갱신 1회 |
| 일관성 계수 (joint-embedding) | 20 | `consistency_coef: 20` |
| 보상, 가치 계수 | 0.1, 0.1 | `reward_coef: 0.1`, `value_coef: 0.1` |
| λ (시간 가중) | 0.5 | `rho: 0.5` |
| 학습률 | 3e-4 | `lr: 3e-4`, 옵티마이저 Adam |
| 기울기 클리핑 | 20 | `grad_clip_norm: 20` |
| 할인율 공식 | γ = clip(1 − 5/T, [0.95, 0.995]), T = 500 → 0.99 (식 7) | `_get_discount`: `(frac−1)/frac`, `frac = T/5`. 같은 식 |
| 시드 단계 | max(5T, 1000) (식 8) | `envs/__init__.py`: `cfg.seed_steps = max(1000, 5*cfg.episode_length)` |
| 계획 호라이즌, 반복, 샘플, 엘리트, 정책 궤적 | 3, 6(+2 if A ≥ 20), 512, 64, 24 | `horizon: 3`, `iterations: 6`, `num_samples: 512`, `num_elites: 64`, `num_pi_trajs: 24`. 반복 +2는 `tdmpc2.py`의 `iterations += 2*int(action_dim >= 20)` |
| 표준편차 범위, 온도 | 0.05–2, 0.5 | `min_std: 0.05`, `max_std: 2`, `temperature: 0.5` |
| 운동량 | 없음 | 없음 (`_prev_mean`은 평균 이동에만 씀) |
| 정책 로그 표준편차 | −10–2 | `log_std_min: -10`, `log_std_max: 2` |
| 엔트로피 계수 | 1e-4 | `entropy_coef: 1e-4` |
| 이산 회귀 빈 수 | 101 | `num_bins: 101`, 빈 크기 `(vmax−vmin)/(num_bins−1)` |
| 인코더 층, 차원 | 2, 256 | `num_enc_layers: 2`, `enc_dim: 256` |
| MLP 차원, 잠재 차원 | 512, 512 | `mlp_dim: 512`, `latent_dim: 512` |
| 과제 임베딩 | 96, 노름 1 | `task_dim: 96` (다과제 일부는 예외, D6), `nn.Embedding(..., max_norm=1)` |
| Q 개수, 드롭아웃 | 5, 1% (첫 선형층 뒤) | `num_q: 5`, `dropout: 0.01`, `mlp(..., dropout=dropout*(i==0))` |
| SimNorm V | 8, τ = 1 | `simnorm_dim: 8`, softmax에 온도 없음 |
| Q 목표 EMA | 0.99 유지 (표 8) | `tau: 0.01`, `lerp_(online, tau)` → 유지율 0.99 |
| 엔트로피 정규화 | 이동 백분위 (5%, 95%) | `RunningScale`의 `_percentiles = [5, 95]` |
| 재생 버퍼 | 1M, 균등 | `buffer_size: 1_000_000`, `SliceSampler` (우선순위 없음) |
| TD 목표 | 무작위 두 Q̄의 최솟값 (§3.1) | `Q(..., return_type='min', target=True)`에서 `randperm(...)[:2]` |
| 정책 손실 | 최대 엔트로피 (식 4), λ^t 가중 | `pi_loss = ... * rho**t`, 정책만 갱신 |
| 모델 구성 (표 9) | 1M, 5M, 19M, 48M, 317M | `common/__init__.py`의 `MODEL_SIZE`: 같은 값 (317M은 Q 8) |
| 파라미터 수 (5M) | 5,389,930 | 재계산 일치 (`03` K13) |

### 2.2 차이

| 번호 | 논문 | 코드 | 영향 |
|---|---|---|---|
| **D1** | 표 8: 인코더 학습률 1e-4 | `enc_lr_scale: 0.3` → 3e-4 × 0.3 = **9e-5**. `tdmpc2.py`의 옵티마이저에서 인코더 그룹만 적용 | 표 8의 값과 0.9배 차이. 재현 시 9e-5를 써야 공식과 같음 |
| **D2** | 식 3: \(\lVert z_t - \mathrm{sg}(h(s'_t)) \rVert^2_2\) (차원 합) | `consistency_loss += F.mse_loss(z, _next_z) * rho**t` (차원과 배치 평균) | 같은 계수 20에서 손실 크기가 잠재 차원 512배 작음. 계수 20은 코드 기준으로 해석해야 함. 식과 표 8의 "coef 20"만으로 재현하면 틀림 |
| **D3** | §3.2: 계획은 (µ, σ)를 이전 해에서 한 칸 이동해 warm start | `mean[:-1] = _prev_mean[1:]`만 있고, σ는 매 결정마다 `max_std`로 재설정 | 논문 설명보다 warm start가 약함. 이는 계획 분포의 초기 폭 차이 |
| **D4** | §3.2: 실행 행동은 N(µ*_t, σ*_t)에서 샘플 | 엘리트 중 `gumbel_softmax_sample`로 한 시퀀스를 고르고 그 첫 행동에 `std * randn` 잡음 (평가 모드는 잡음 없음) | 행동 선택 규칙이 논문 설명과 다름. 재현하려면 공식 규칙을 따라야 함 |
| **D5** | 식 6: Q(z, a) (정의 없음) | 계획 평가는 `Q(..., return_type='avg')`, 즉 무작위 두 Q의 평균. TD 목표는 'min' | 식 6의 Q 정의는 논문에 없음. 코드는 평균 |
| **D6** | 표 8, 9: 과제 임베딩 96 | `cfg.task_dim = 96 if task == 'mt80' or model_size in {1, 317} else 64` (코드 주석 "slight inconsistency"). mt30의 5M, 19M, 48M은 **64** | 표 9의 "96"은 mt30 중형 모델에는 맞지 않음 |
| **D7** | 표 9: 19M 잠재 차원 768 | `if task == 'mt30' and model_size == 19: latent_dim = 512` ("slightly smaller checkpoint") | 공개 mt30 19M 체크포인트는 표 9와 다른 구성일 수 있음 |
| **D8** | §3: "S 단계 동안 갱신 없음" | 시드 단계 직후 `num_updates = seed_steps` 번을 한꺼번에 갱신한 뒤 단계당 1회 | 설명은 맞지만, 사전 학습 갱신의 양은 논문에 없음 |
| **D9** | (논문에 없음) | 보상과 Q의 마지막 선형층 0 초기화 (`init.zero_`); π의 Adam `eps=1e-5`; `RunningScale` 값 하한 1; symlog 범위 vmin −10, vmax +10; 인코더 입력은 S + 과제 임베딩 차원 (단일 과제는 임베딩 0) | 재현 시 명시해야 하는 구현 세부. 특히 vmin, vmax와 초기화 |
| **D10** | §3.3 "단일 설정" | 다과제에서 배치 1024 (README 명령과 §4.1 일치) | 설정 변화는 논문에 적혀 있음. 초록의 표현만 과장 (`03` K5) |

- D1은 설정 값의 차이입니다. D2–D5는 **계산 규칙의 차이**이므로 재현 결과에 직접 영향을 줍니다. D6–D7은 다과제 구성의 차이입니다. D8–D10은 문서화되지 않은 구현 세부입니다.
- 논문 본문의 "α를 이동 통계로 조정한다"(§3.1)는 코드에서 Q와 엔트로피 항을 `RunningScale`로 정규화하는 것과 같은 목표입니다(`01` §2.3). 불일치로 보지 않습니다.

### 2.3 논문 주장의 코드 쪽 확인

- **"과제 임베딩의 노름 1 제한"**: 코드에서 `max_norm=1`로 확인됩니다(D 항목 아님).
- **"SimNorm이 잠재를 희소하게 한다"**: 코드 구현은 `SimNorm(dim=8)`이며 설명과 일치합니다.
- **"TD-MPC는 발산하지만 TD-MPC2는 안정"**: 이 저장소의 코드로는 판단하지 않습니다. 기울기 자료는 공개되지 않았습니다(`03` C11).
- **"계획 처리량 2배"**: 코드에 `torch.compile`과 CUDA graph 호출이 있어 계획 처리량 최적화가 포함돼 있습니다. 측정은 하지 않았습니다.

## 3. 안정성 증명 검토

논문에 안정성 증명은 없습니다. 부록 G는 기울기 노름 곡선(그림 21)을 씁니다. 이 저장소에 기울기 기록은 없습니다. 해당 없음.

## 4. 구현 가능성 (이 맥)

**판정 요약:** 논문의 전체 재현은 이 맥에서 불가능합니다. 공식 코드는 CUDA를 전제하고, 다과제 학습은 GPU 여러 장과 수백 M 전이의 데이터가 필요합니다. 단일 과제 DMControl의 축소 실행은 포팅과 설치 뒤에 조건부로 가능합니다. 토이 규모의 재구현(SimNorm, two-hot, 잠재 계획)은 가능합니다.

### 4.1 이 맥을 막는 요소 (코드에서 확인)

| 요소 | 위치 | 영향 |
|---|---|---|
| `cuda:0` 하드코딩 | `tdmpc2.py` (디바이스 설정과 할인율 텐서), `common/buffer.py` (디바이스와 저장 장치 선택), `common/scale.py` (값과 백분위 텐서) | 디바이스 인자화와 MPS 포팅 필요. CUDA를 지정하는 줄은 저장소 전체에 13줄(8개 파일) |
| `torch.compile(mode="reduce-overhead")` (CUDA graph) | `tdmpc2.py`의 `_update`, `plan`, `cfg.compile` | MPS에서는 `compile: false`로 끄는 것이 필요. 속도에 영향 |
| `torch.compiler.cudagraph_mark_step_begin()` | `trainer/online_trainer.py`, `trainer/offline_trainer.py`, `tdmpc2.py` | CUDA 전제. MPS에서는 호출 제거 필요 |
| `torch==2.7.1`, `tensordict==0.8.3`, `torchrl==0.8.1`, `mujoco==3.1.2` | `environment.yaml` | 맥에서 같은 버전을 설치할 수 있는지 미확인. 설치 승인 전에는 확인 불가 |
| Meta-World, ManiSkill2, MyoSuite | README와 `environment.yaml` 주석 | Meta-World는 MuJoCo 2.1과 gym 0.21, 라이선스 키 필요. 별도 설치 |
| 데이터셋과 체크포인트 | `datasets/download_mt*.sh`, Hugging Face | 다과제는 대형 데이터. 단일 과제 체크포인트는 크기 미확인 |

- 이 맥(M4 Pro, 24 GiB, MPS, CUDA 없음)에서 README의 요구는 맞지 않습니다. README는 "GPU, 12 GB RAM(단일 과제), 128 GB RAM(mt80)"을 요구합니다. 317M 학습은 "24 GB GPU 메모리"를 요구합니다.

### 4.2 재현 대상별 판정

| 대상 | 판정 | 근거 |
|---|---|---|
| 104과제 × 시드 3 × 4M (전체 온라인) | **불가** | CUDA 전제, 대규모 계산. 이 맥에서 학습 시간을 측정하지 않았으나 시간 규모가 분 단위가 아님 |
| 다과제 80과제 317M 학습 | **불가** | 33 GPU일 (표 1, RTX 3090). 데이터 545M 전이, 128 GB RAM 요구 |
| 공식 단일 과제 체크포인트 추론과 평가 | **조건부** | 포팅 (4.1의 항목), 패키지 설치와 체크포인트 다운로드 승인 필요. 정적 읽기 결과로는 추론 경로가 짧음 |
| 단일 과제 DMControl 축소 실행 (예: 시드 1, 50k 환경 단계) | **조건부** | 포팅과 설치 뒤 측정 필요. 1M 단계는 한 과제에도 시간이 많이 듦 (추정, 측정 전) |
| 토이 재구현 (SimNorm, two-hot, 잠재 계획, 2-링크 또는 점질량) | **가능** | torch 직접 구현은 사용자 승인 범위 안 (재현 승인 범위 안: 직접 짠 torch 구현은 허용) |

### 4.3 노력 추정 (측정 전)

- **포팅 (CUDA 제거, MPS 또는 CPU):** 1–2일. 디바이스 인자화와 `compile` 끄기가 주된 작업입니다.
- **환경 설치 (DMControl 경로):** 설치 승인 후 반나절–1일. MuJoCo 2.1과 Meta-World 경로는 별도로 하루 이상 추정합니다.
- **축소 실행 한 번 (DMControl 한 과제, 수만 단계):** 측정 전이라 추정하지 않습니다.
- **재구현 (torch, 축소판):** 3–5일 추정. 식 2–8과 SimNorm, two-hot, MPPI를 짜고 토이에서 맞춥니다.

## 5. 재실행

```bash
# 정적 대조의 근거를 다시 보는 법 (코드는 실행하지 않음)
cd repo && git log -1 --format='%H %cI %s'
grep -n "enc_lr_scale\|consistency_coef\|task_dim\|latent_dim" tdmpc2/config.yaml tdmpc2/common/parser.py
```
