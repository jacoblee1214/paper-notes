# FlashSAC 코드 대조 (논문 ↔ 공식 저장소)

## 0. 대상과 방법

- 저장소: https://github.com/Holiday-Robot/FlashSAC , 커밋 `87edc90` (LICENSE 파일 커밋, 2026-04-09). `repo/`에 `--depth 1`로 얕게 복제했다. 라이선스 MIT. 커밋 이력은 1개뿐이다.
- 방법: **정적 읽기만 했다.** 코드를 실행하지 않았다(학습, 테스트, `uv sync`, 설정 로드 모두 안 함). 논문 식과 부록 표를 코드 줄과 대조했다.
- 줄 번호는 위 커밋 기준이다. 파일 경로는 `repo/` 기준이다.
- 파라미터 수와 기타 값은 코드의 층 정의를 손으로 세어 `analysis/derivation_checks.py`에서 계산했다(03 §8).
- 공개 학습 곡선 CSV(`repo/results/`)는 재계산에 썼다(03).

## 1. 대조 표 (D1–D30)

판정: **일치**는 논문 설명과 코드가 같다. **차이**는 코드와 논문 본문·표가 다르다. **문서 불일치**는 논문 안에서 서로 다르다. **미기재**는 논문에 없고 코드에 있다. **부재**는 논문에 있으나 공개 코드에 없다.

### 1.1 목적 함수와 학습 루프

| # | 항목 | 논문 | 코드 (줄) | 판정 |
|---|---|---|---|---|
| D1 | 비판자 손실 | 식 (4)는 MSE. §4.2는 분포형 비판자와 교차 엔트로피 | `flash_rl/agents/flashSAC/update.py:249` `ce_loss = -(target_probs · pred_log_probs).sum`. 출력은 `layer.py` `EnsembleCategoricalValue`의 101 bin softmax | 문서 불일치 (식 표기만 다름). 구현은 §4.2 설명과 일치 |
| D2 | 최소 이중 Q | 식 (5)의 min Q | `update.py:15–31` `_select_min_q_log_probs`: 샘플마다 기대 Q가 작은 쪽 비판자의 분포를 고른다 | 일치 (기대값 기준 min) |
| D3 | 목표 값 | 식 (5): r + γ(min Q − α log π) | `update.py:56` `target_bin_values = reward + gamma*(bin_values - actor_entropy)*(1-done)`, `actor_entropy = α·log π`. `update.py:57` 지지 [min_v, max_v]로 자른다 | 일치. 지지로 자르는 단계는 논문에 없다(미기재) |
| D4 | n-step | 표 9: n = 1 (GPU). 표 11: 시각 n = 3 | `scripts/run_isaaclab.sh:48` `n_step=3`. 다른 GPU 스크립트(`run_maniskill.sh`, `run_genesis.sh`, `run_mujoco_playground.sh`)와 `configs/agent/flashSAC.yaml`의 기본값은 1. 버퍼의 n-step 반환은 `torch_buffer.py:103–132` | 문서 불일치: IsaacLab만 n=3 |
| D5 | 종료와 절단 | 명시 없음 | 목표는 `terminated`만 쓴다(`update.py` 배치 키). 보상 정규화의 누적 반환 리셋은 `terminated or truncated`(`reward_normalization.py` `_update_reward_stats`) | 미기재 (표준적 처리) |
| D6 | 블록 구조 | 그림 2: Linear → BN → ReLU → Linear → BN → ReLU → (+잔차). 본문 "pre-activation BN" | `layer.py:91–107` `FlashSACBlock`: `w1 → norm1 → relu → w2 → norm2 → relu → +residual`. 그림과 같다 | 일치. 본문의 "pre-activation"은 명칭 혼동: BN은 선형 뒤, ReLU 앞이다 |
| D7 | 최종 정규화 | RMSNorm으로 특징 노름을 제한 | `network.py` `post_norm = UnitRMSNorm` 후 `predictor`. `layer.py:62–75`: `F.rms_norm`, 가중치 노름 √d | 일치 |
| D8 | 가중치 투영 | 각 가중치 벡터를 단위 노름, (γ, β)는 노름 √d(본문은 "d"로 보이나 PDF 기호 손실) | 단위 행 투영: `layer.py:19–24` (`F.normalize`, 입력 차원 기준). BN (γ,β): `layer.py:52–57` (노름 √d). 호출은 `update.py:155`(액터), `update.py:269`(비판자), `agents/utils/network.py` `normalize_parameters`. 온도 파라미터는 투영하지 않음 | 일치 (√d 여부는 PDF 추출로 확인 불가; 코드는 √d) |
| D9 | 교차 배치(비판자) | 현재와 다음 상태를 한 배치로 묶어 BN 통계 공유 | `update.py:214` `obs_all = cat(obs, next_obs)`, 표적·예측 비판자 모두 이 배치로 계산 | 일치 |
| D10 | 교차 배치(액터 갱신) | 논문 본문은 비판자 쪽만 말함 | `update.py:105` `actor_obs_all = cat(obs, next_obs)`로 액터를 통과시킨 뒤 앞 절반만 씀. 그래서 액터 BN 통계는 다음 상태 포함 2B 샘플로 계산됨 | 미기재 (코드 특이점) |
| D11 | 액터 갱신 때 비판자 모드 | 명시 없음 | `update.py:116–123`: `critic(..., training=False)`. 액터 손실의 Q는 BN 이동 평균을 쓰고, 표적·예측 비판자 호출(`update.py:219`, `:242`)은 `training=True`로 배치 통계를 씀 | 미기재 (모드가 다름) |
| D12 | 표적 비판자 EMA | 식 (3), τ = 0.01 | `agents/utils/network.py` `_ema_update_fn`: `torch._foreach_lerp_(params, src, τ)`. 매개변수만 갱신하고 BN 버퍼(이동 평균)는 갱신하지 않음 | 일치 (버퍼 제외는 미기재) |
| D13 | 액터 지연 갱신 | 표 9: update delay 2 | `agent.py:501` `do_actor_update = (step % actor_update_period == 0)` | 일치 |
| D14 | 온도와 목표 엔트로피 | 식 (7): H̄ = ½|A| log(2πeσ²), σ = 0.15 | `agent.py:355` 식 그대로. 손실은 `update.py` `update_temperature`(`α·(H − H̄)`). 코드의 H는 tanh 보정 포함 log π의 기대값 | 일치 (식은 가우시안 기준. 같은 σ에서 tanh 보정 엔트로피는 차원당 0.022 nats 낮다, 03 §8) |
| D15 | 잡음 반복 | §4.3: 길이 k 동안 ε 고정. "짧은 로컬 상태" | `agent.py:245` `reinit = (cur_count == 0) | (cur_count >= cur_n)`. 카운터와 길이 `cur_n`은 스칼라 텐서라 **모든 환경이 같은 시점에 갱신**. 잡음 벡터는 `randn_like(mean)`로 환경마다 뽑힘(`agent.py:247`) | 미기재 (환경 간 동기화) |
| D16 | Zeta 분포 | P(k) ∝ k^(-s), s = 2, 최대 16 | `agent.py:394` `_build_truncated_zeta_cdf(mu=2.0, max_n=16)`; `agent.py:206` `pmf = ns ** (-mu)` | 일치. E[k] = 2.13, 매 갱신 비율 0.47 (03 §8) |
| D17 | 보상 정규화 | 식 (6): r̄ = r / max(√(σ²_G+ε), G_max,t / G_max) | `reward_normalization.py:32–35` `denominator = max(sqrt(G_var+eps), G_r_max/G_max)`; `G_r_max`는 지금까지 |G_r|의 최댓값(`:20`)이라 줄지 않음. G_max = 5 (`configs/agent/flashSAC.yaml`) | 일치. 지지 [-5, 5]와 맞춘 설계 (02 §5) |
| D18 | 학습률 | 표 9: 코사인 감쇠 3e-4 → 1.5e-4 | `configs/agent/flashSAC.yaml`: 워밍업 비율 1e-6 → `int(1e-6 × 97,658) = 0`이라 워밍업 없음. 감쇠 길이는 전체 업데이트 수 | 일치 (워밍업 없음은 미기재) |
| D19 | 기울기 클리핑 | §4.2: "기울기 노름을 명시적으로 제한" | `grep clip_grad` 결과 0건. 기울기 노름은 가중치 투영을 통해서만 간접 제한 | 차이 (표현이 과함) |
| D20 | 혼합 정밀도 | 본문: 벽시계 5–10% 감소 | GPU 설정 `use_amp: true`(`run_isaaclab.sh`): `autocast(fp16)` + `GradScaler`(`update.py`). 통계는 float32(`layer.py` 주석). CPU 설정은 `use_amp=false` | 일치. 5–10% 수치는 코드에 측정 코드 없음 (D29) |
| D21 | 컴파일 | §4.1: JIT | `agent.py` `_resolve_compile_mode`: torch ≥ 2.9는 `max-autotune`, 그 아래는 `reduce-overhead`. README는 Python 3.11에서 2.9.1을 쓴다고 함 | 일치 (설치 환경 종속) |
| D22 | 병렬 환경·버퍼 | 1024 env, 버퍼 10M | `run_isaaclab.sh`: `num_train_envs=1024`, `buffer_max_length=10_000_000`, `buffer_device_type='cuda'`, `buffer_min_length=100_000`. 기본 설정은 10k. `train.py:123–125`: 버퍼가 찰 때까지 무작위 행동 | 일치. 최소 버퍼 크기(100k)와 무작위 시작은 논문에 없음 (미기재) |
| D23 | UTD | 본문: 2/1024. 표 9: "2/2048" | `run_isaaclab.sh`: `updates_per_interaction_step=2`, 상호작용 1회에 전이 1024개(`train.py:121` `env_step = interaction_step * num_train_envs`). 즉 2/1024 | 문서 불일치 (표 9 오기로 보임). 코드는 본문과 같다 |
| D24 | 배치 크기 | GPU 2048, CPU 512, 시각 256 | GPU 스크립트 2048, CPU 스크립트 512(`run_mujoco.sh` 등). 시각 설정 없음 (D26) | 일치 (시각 제외) |
| D25 | 시드 | 명시 없음 | 스크립트 시드 `(0 1000 2000 3000 4000)`, 공개 CSV도 같은 5개(FlashSAC 전부) | 미기재 (논문은 개수를 적지 않음) |
| D26 | 시각 기반 학습 | §5.3: 3층 합성곱 인코더, 프레임 스택 3, 3단계 반환, 액션 반복 2 | 픽셀 관련 코드 0건(`from_pixels`, `conv`, `frame_stack` 검색). `dmc.py`는 상태 관측만 쓴다. `run_dmc.sh`는 배치 512, n = 1로 상태 기반 설정 | **부재**: 공개 코드에 시각 설정이 없다. 시각 CSV(`visualRL/`)는 있다 |
| D27 | 평가 빈도 | 학습 곡선 | `configs/flashSAC_base.yaml`: 평가와 로그 주기 = 총 상호작용 수 / 10. 곡선은 실행당 11점. GPU는 약 5M step 간격, CPU는 약 0.1M 간격 | 미기재 (곡선 해상도) |
| D28 | 평가 방식 | 명시 없음 | 평가는 결정적 행동 `tanh(mean)`(`agent.py:241`), 평가 에피소드 1024개(GPU), 50개(CPU) | 미기재 |
| D29 | 벽시계 측정 | 부록 A: 환경 시간 + 업데이트 시간 합산, 업데이트 시간은 한 과제에서 측정해 재사용 | `train.py`, `flash_rl/common/logger.py`, `flash_rl/evaluation.py`에 시간 측정 호출 없음. 공개 CSV에도 시간 열이 없다 | **부재**: 벽시계 측정 코드가 공개되지 않음 |
| D30 | 실물 배포 | §5.4, 부록 D: PD 제어 200 Hz, Unitree SDK, ROS 2 노드, Pybind | 배포 코드 없음. `uv.lock`과 CSV, ManiSkill 스크립트의 과제 이름에만 Unitree 문자열 있음 | **부재**: 실물 배포 코드 없음. 학습 환경(IsaacLab 설정)만 있음 |

### 1.2 설정 값 요약

| 항목 | 논문 (표 9–10, 본문) | 공개 스크립트 (GPU / CPU) |
|---|---|---|
| 환경 수 | 1024 / 1 | 1024 / 1 |
| 배치 | 2048 / 512 | 2048 / 512 |
| UTD (업데이트/상호작용 단위) | 2/1024 (표 9는 2/2048) | 2 / 1 |
| 버퍼 | 10M / 1M | 10M / 1M |
| 블록 수 (액터, 비판자) | 2, 2 | 2, 2 (`configs/agent/flashSAC.yaml`) |
| 은닉 폭 (액터, 비판자) | 128, 256 | 128, 256 |
| 이중 비판자, 원자 수 | 2, 101 | 2, 101 |
| 표적 τ, 액터 지연 | 0.01, 2 | 0.01, 2 |
| n-step | 1 (표 9), 3 (시각) | **3 (IsaacLab 스크립트)**, 1 (나머지) |
| 감마 | IsaacLab 0.99, Playground 0.97 | 스크립트별 0.99 |
| 학습률 | 3e-4 → 1.5e-4 (코사인) | 같음 |
| σ_tgt, 온도 초기 | 0.15, 0.01 | 같음 |
| ζ 지수, 최대 반복 | 2, 16 | 같음 |
| 시드 | (미기재) | 0, 1000, 2000, 3000, 4000 |

### 1.3 파라미터 수 (손으로 셈, 03 §8)

| 항목 | 액터 | 비판자 앙상블(2개) | 합계 |
|---|---|---|---|
| Shadow Repose (관측 157, 행동 20) | 290,402 | 2,251,150 | 2,541,552 |
| G1 평지 (관측 123, 행동 37) | 290,368 | 2,242,378 | 2,532,746 |
| HalfCheetah (관측 17, 행동 6) | 268,590 | 2,171,686 | 2,440,276 |

논문 §4.1의 "2.5M 파라미터"는 **액터와 비판자 앙상블을 합친 값**과 맞는다. 액터만은 0.3M 남짓이다. 표준 SAC 기준선(0.2–0.5M)과 비교할 때는 비판자 두 개를 함께 세어야 같은 기준이다.

## 2. 재현 검증이 하지 않은 것 (정적 읽기의 한계)

- 학습 곡선과 최종 점수가 코드에서 나온다는 것은 확인하지 못했다. 공개 CSV는 스크립트와 같은 시드, 같은 과제 이름을 쓰지만, 실행 기록을 대조하지는 않았다.
- 수치 안정성(부동소수점, torch.compile 결과, AMP 손실 스케일링)은 실행하지 않아서 확인하지 못했다.
- 시각 설정(D26)과 실물 배포(D30)는 코드가 없어서 검증할 수 없었다.

## 3. 구조 요약 (코드 기준 흐름)

1. `train.py` 루프: 상호작용 1회에 `process_transition` → 버퍼가 찼으면 `update_counter += updates_per_interaction_step`만큼 `agent.update()`.
2. `agent.py` `update()`: 버퍼에서 배치 샘플 → 보상 정규화(식 6) → `_update_networks`: 액터(2회마다 1회) → 온도 → 비판자 → 표적 EMA.
3. `update.py` `update_critic`: 교차 배치로 표적 분포 계산(`_compute_categorical_td_target`) → 교차 엔트로피 → Adam 갱신 → 가중치 투영.
4. 샘플링: `sample_actions`는 BN 이동 평균으로 평균·표준편차를 얻고, `temperature=1`로 잡음을 곱해 행동을 만든다. 잡음은 Zeta 길이만큼 유지된다.

## 4. 구현 판정 (재현 가능성)

**판정: 공식 재현은 이 맥에서 불가능하다. 알고리즘을 자기 설정에 옮기는 것과 CPU 소규모 검증은 가능하다.**

| 항목 | 필요 자원 | 이 맥(Apple M4 Pro, 약 24 GiB 통합 메모리, MPS, CUDA 없음)에서 | 노력 | 판정 |
|---|---|---|---|---|
| 공식 GPU 재현: IsaacLab, MuJoCo Playground, ManiSkill, Genesis 25과제 × 시드 5 | RTX 30/40/50 계열 또는 Blackwell GPU (README), Ubuntu 22.04/24.04, Python 3.10/3.11, IsaacLab 설치 | CUDA가 없다. 저장소가 CUDA 경로를 전제하고, `buffer_device_type='cuda'`. MPS로는 같은 설정을 돌릴 수 없다 | 큼. 대략 과제당 30–90 min × 시드 5 × 25과제 (논문 Fig. 3 벽시계 축 기준) ≈ 100 GPU시간 이상 | **불가** |
| 공식 CPU 재현: MuJoCo v4 / DMC / MyoSuite / HumanoidBench, 1 env, 배치 512 | Python 3.10/3.11, uv, gymnasium, mujoco (README는 MuJoCo 2.1 "mujoco210" 필요), dm_control, myosuite, humanoid-bench | `.venv`에 torch 2.15와 mujoco 4.0.0이 있으나 gymnasium은 없다. 설치는 승인 필요 | 중간. 토이 측정(2 ms/업데이트, 16 업데이트/iter)에서 외삽하면 실행당 약 1–2시간(1M 업데이트, 배치 512) | **조건부**: 패키지 설치 승인 후 과제 1–2개, 시드 1–2개 정도 |
| 시각 DMC 재현 | GPU, 렌더링(EGL), 시각 인코더 코드 | 공개 저장소에 시각 설정 코드 없음 (D26) | 큼 | **불가** (코드 없음) |
| 실물 G1 배포 | Unitree G1, GPU 학습, ROS 2, Unitree SDK | 로봇 없음. 배포 코드 없음 (D30) | 매우 큼 | **불가** |
| 알고리즘 재구현 (PyTorch, 이 저장소 구조 참고) | 없음 (CPU와 MPS 가능) | 토이로 확인 (03 §9). 교차 배치, 가중치 투영, 분포형 비판자는 약 300줄 | 중간 | **가능** |
| 공개 CSV 재계산 (정규화, 유의성, 소거 지표) | 없음 | 완료 (03) | 완료 | **완료** |

**그래서 실제로 할 수 있는 것은 두 가지다.** (a) 공개 CSV와 논문 표로 주장 점검(이 분석에서 완료), (b) CPU에서 알고리즘 요소를 작은 과제에 옮겨 보기(토이 완료). 공식 GPU 재현은 CUDA 장비가 있어야 하고, 실물 배포는 로봇이 있어야 한다.

## 5. 검증 공백과 이 코드를 믿기 전에 확인할 것

1. 벽시계 시간 측정 코드가 없으므로 "수 시간에서 수십 분"과 "3배 계산량" 주장은 코드로 확인할 수 없다(D29).
2. 시드 개수와 시드 선택이 논문에 없다. 공개 CSV가 5개 시드로 맞으면 공식 결과와 대조할 수 있다(D25).
3. 개수와 설정 표가 맞지 않는다: UTD(D23), n-step(D4), 과제 개수(03 §1).
4. 액터 갱신에서 BN 통계에 다음 상태를 넣고 비판자를 평가 모드로 쓰는 조합(D10, D11)은 논문이 다루지 않는다. 재구현할 때 같은 선택을 할지, 본문대로 할지 정해야 한다.
5. 기울기 클리핑이 없으므로 기울기 폭주가 생기면 가중치 투영 말고는 막을 장치가 없다(D19). 이 점은 소거 실험에서 기울기 노름이 0으로 기록된 것과 같이 확인이 필요하다(03 §7).
