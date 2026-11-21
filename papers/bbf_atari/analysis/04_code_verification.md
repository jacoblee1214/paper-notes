# 04. 공식 코드 대조와 구현 가능성

## 0. 대상과 방법

- **대상.** `google-research/google-research`의 `bigger_better_faster/` 폴더입니다. 얕은 희소 복제를 `repo/gr/`에 두었고, 체크아웃 HEAD는 `9394dfc803d4`(2026-11-20)입니다. 이 폴더의 마지막 커밋은 GitHub API `commits?path=`로 `5b09c22d73`(2026-02-12)으로 확인했습니다. 얕은 복제에는 이력이 없어 로컬에서는 확인하지 못했습니다.
- **읽은 파일.** `bbf/configs/BBF.gin`, `SR_SPR.gin`, `SPR.gin`, `DrQ.gin`, `rainbow.gin`, `der.gin`, `OTRainbow.gin`(설정 값), `bbf/agents/spr_agent.py`(에이전트, 재설정, 스케줄, 옵티마이저), `bbf/spr_networks.py`(네트워크, 증강), `bbf/train.py`, `bbf/eval_run_experiment.py`(평가와 정규화), `requirements*.txt`, `README.md`. 대조는 `grep`과 함수 단위 읽기로 했습니다.
- **실행하지 않았습니다.** 설치, import, 학습, 평가 모두 하지 않았습니다. 코드의 수치는 파일 안의 값과 기본값으로만 확인했습니다. 실제 실행 시 동작은 확인하지 않았습니다.
- **리플레이 비율 해석.** `set_replay_settings()`(`spr_agent.py` 1365–1400행)를 따라 손으로 계산했습니다. `n_envs=1`, 배치 32에서 `replay_ratio=64`이면 환경 스텝마다 그래디언트 배치 2개(`batches_to_group=2`)를 씁니다. 즉 RR = replay_ratio/32입니다. `SPR.gin`(64)은 RR 2, `rainbow.gin`(8)은 RR 0.25로, 논문의 정의(§2, §4)와 맞습니다.

## 1. 설치 요구사항과 라이선스

- **패키지 핀.** `requirements_jax.txt`는 `jax==0.3.14`, `jaxlib==0.3.14`입니다. `requirements.txt`는 `dopamine-rl==4.0.5`, `gym[atari,accept-rom-license]<=0.25.2`, `pyglet==1.4.11`, `protobuf==3.19.1`, `tensorflow`, `dm-haiku`, `flax>=0.6.3`, `wandb`를 요구합니다. 2022년 무렵의 버전 묶음입니다.
- **하드웨어.** README는 "운영체제와 CUDA 버전에 따라 JAX 설치에 추가 단계가 필요할 수 있다"고 적었습니다. CUDA를 전제로 한 설명입니다.
- **라이선스.** 모노레포 API가 Apache-2.0을 보고합니다. 소스 파일 머리말에도 Apache 2.0 표기가 있습니다. 하위 폴더에는 LICENSE 파일이 없습니다. 모노레포 라이선스를 따르는 것으로 봤습니다.

## 2. 논문 ↔ 코드 대조 (D 항목)

| 번호 | 논문 주장 (출처) | 코드 (파일:위치) | 판정 | 비고 |
|---|---|---|---|---|
| D1 | 주된 결과는 리플레이 비율 8 (§4 "Base agent") | `BBF.gin`: `replay_ratio = 64` → RR 2 | **불일치 (기본값)** | README의 학습 명령은 `BBF.gin`을 그대로 씁니다. RR=8 재현에는 리플레이 비율을 256으로 바꿔야 합니다. RR=8 설정 파일은 없습니다. |
| D2 | 재설정은 4만 그래디언트 스텝마다 (§4 "Receding update horizon") | `BBF.gin`: `reset_every = 20_000` (환경 스텝), RR 2에서 그래디언트 스텝 2개/환경 스텝 → 4만 | **일치 (RR 2 기준)** | RR 8에서는 `reset_every = 5_000`이어야 4만이 됩니다(`SR_SPR.gin`이 이 값을 씁니다). 주석 "Change if you change the replay ratio"가 이 점을 적고 있습니다. |
| D3 | 1M 스텝 실험에서 100K 이후에도 재설정을 계속 함 (§6 "Data Scaling") | `BBF.gin`: `no_resets_after = 100_000  # Need to change if training longer` | **설정 변경 필요** | 논문 실험을 재현하려면 이 값을 바꿔야 합니다. 주석이 이를 알려 줍니다. |
| D4 | 하이퍼파라미터 (본문 전체) | 코드에만 있음: lr 1e-4(`learning_rate`), Adam eps 1.5e-4, AdamW wd 0.1, 배치 32, 은닉 2048, 원자 51, `target_update_tau` 0.005, `spr_weight` 5, `jumps` 5, `min_replay_history` 2000, 리플레이 용량 20만, `max_noops` 30, `terminal_on_life_loss` True, `epsilon_eval` 0.001, 평가 에피소드 100 | **논문에 없음** | 논문 v3 본문과 부록에 하이퍼파라미터 표나 목록이 없습니다. 부록 표는 A.1 점수뿐입니다. |
| D5 | "데이터 증강" (§3, 관련 연구에서 DrQ 언급) | `spr_networks.py`: `drq_image_aug`는 가장자리 패딩 4 후 무작위 자르기, `_intensity_aug`는 강도 잡음 ±5%(`scale=0.05`), `data_augmentation=True` | **논문 본문에 없음** | 증강이 학습에 쓰인다는 것은 코드에서 확인됩니다. 논문은 어떤 증강인지 적지 않았습니다. |
| D6 | 합성곱 층은 50%, 마지막 층은 SR-SPR처럼 재설정 (§4 "Harder resets") | `BBF.gin`: `shrink_perturb_keys = "encoder,transition_model"`, `shrink_factor = 0.5`, `perturb_factor = 0.5`. `reset_encoder=False`(기본), `reset_head=True`, `reset_projection=True`(기본) | **일치, 세부 누락** | 논문은 전이 모델(`transition_model`)에 shrink-and-perturb를 적용한다는 것을 적지 않았습니다. 마지막 층 재설정은 기본값으로만 정해져 있습니다. |
| D7 | n-step 10에서 3으로, 재설정 뒤 1만 스텝 지수 감쇠 (§4) | `BBFAgent`: `update_horizon=3`, `max_update_horizon=10`, `cycle_steps=10_000`(`cycle_grad_steps` 기준), `reset_weights()`에서 0으로 초기화 | **일치** | 감쇠 함수는 `exponential_decay_scheduler`(로그 공간 지수)입니다. 1만 스텝은 4만 그래디언트 스텝의 25%입니다(§4와 맞음). |
| D8 | γ를 0.97에서 0.997로 증가 (§4) | `JaxDQNAgent.gamma = 0.997`, `BBFAgent.min_gamma = 0.97`, 같은 주기 | **일치** | |
| D9 | EMA 표적망을 학습과 행동 선택에 씀 (§5) | `target_update_tau = 0.005`, `target_action_selection = True` | **일치** | 계수 0.005는 논문에 없습니다. |
| D10 | NoisyNets를 뺌 (§4 "Removing noisy nets") | `BBFAgent.noisy = False`, `dueling=True`, `double_dqn=True`, `distributional=True`, 원자 51 | **일치** | 노이즈 제거의 FLOPs 2.5배, 메모리 1.6배 수치는 코드에 없어 확인하지 못했습니다. |
| D11 | Impala ResNet을 폭 4배로 (§4 "Larger network") | `RainbowDQNNetwork.width_scale = 4`, `ImpalaCNN.dims = (16, 32, 32)` → (64, 128, 128), `num_blocks = 2` | **일치** | 블록당 합성곱 2개, 단계당 합성곱 5개(첫 합성곱 1 + 블록 2×2), 세 단계면 합성곱 15개입니다. 논문의 "15층"과 맞습니다. |
| D12 | SPR 손실을 유지 (§5) | `spr_weight = 5`, `jumps = 5`, `data_augmentation=True`에서 SPR 손실 사용 | **일치, 값 누락** | 손실 가중치 5와 예측 단계 5는 논문에 없습니다. |
| D13 | AdamW 가중치 감쇠 0.1 (§5) | `create_scaling_optimizer(... weight_decay=0.1)`, `optax.adamw` | **일치** | |
| D14 | 정규화는 인간 정규화 점수 (§2, §4) | `eval_run_experiment.py`: `normalize_score`가 `(ret − random)/(human − random)`. 공개 CSV에서 26게임 전체 행 일치(03 K9) | **일치** | |
| D15 | 그림 3의 SR-SPR-ResNet 변형 (§4, 그림 3) | `SR_SPR.gin`: 은닉 512, CNN 인코더, 폭 1, shrink 0.8/perturb 0.2, `replay_ratio = 256`, `reset_every = 5_000`, γ 0.99, wd 0 | **부분 일치** | 기준선 SR-SPR의 CNN 설정은 있습니다. ResNet 변형(그림 3)의 설정은 없습니다. 그림 3의 SR-SPR 점은 이 파일로 재현할 수 없습니다. |
| D16 | "코드와 데이터를 공개합니다" (초록, 본문) | `scores/`: BBF RR 2와 8, 절제, 스티키, 55게임 학습 곡선의 CSV 25개 | **부분 일치** | SR-SPR, DrQ(ε), DER, IRIS, EfficientZero의 점수 파일이 없습니다. 표 A.1 기준선 열은 검증할 수 없습니다. |
| D17 | 실행 수 (표 A.1 캡션 50, 그림 5 캡션 15) | `scores/RR8_BBF.csv`: 게임당 52–60행, 시드 값은 행마다 다름 | **불일치** | 03 K1. |
| D18 | 평가 방식 (논문 본문에 명시 없음) | `DataEfficientAtariRunner.num_eval_episodes = 100`, `max_noops = 30` | **논문에 없음** | 평가 에피소드 수와 무작위 시작 행동 수는 코드에서만 확인됩니다. |

## 3. 기타 정적 관찰

- **`configs/SR_SPR.gin` 첫 import 블록이 깨져 있습니다.** `import bigger_better_faster.bbf.spr_networksimport ...`처럼 줄바꿈 없이 이어진 줄이 세 개 있습니다(`grep -c` 3). gin 파서가 이 파일을 읽지 못할 가능성이 큽니다. 실행하지 않았으므로 확인은 못 했습니다. SR-SPR 기준선을 재현하려면 먼저 고쳐야 합니다.
- **`configs/BBF.gin` 첫 줄 주석 "# SPR (Schwarzer et al, 2021)".** BBF 설정 파일을 SPR에서 복사한 흔적입니다. 동작에는 영향이 없습니다.
- **`SR_SPR.gin`의 `gamma = 0.99`, `weight_decay = 0.0`.** BBF(γ 0.97→0.997, wd 0.1)와 다릅니다. 논문이 SR-SPR을 같은 설정으로 비교했는지는 본문에 없습니다.

## 4. 구현 가능성 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**판정: 부분적으로 가능합니다. 논문 전체 재현은 이 맥에서 불가능합니다.**

| 무엇 | 가능 여부 | 근거와 필요 자원 |
|---|---|---|
| 논문 주된 결과 재현 (RR 8, 26게임 × 50시드) | **불가** | 논문 §5: BBF는 RR 8에서 한 실행에 "싱글 CPU와 A100 절반으로 10시간"입니다. 50시드 × 26게임 = 1,300회 실행 × 약 5 A100시간 ≈ 6,500 A100시간입니다(논문 수치로 계산한 추정). 맥에는 CUDA가 없습니다. |
| 공식 JAX 코드로 게임 1개, 시드 1개, RR 2, 10만 스텝 | **조건부** | 새 가상환경이 필요합니다. 핀 버전(jax 0.3.14, dopamine 4.0.5, gym 0.25.2)이 Apple Silicon에서 설치될지 확인하지 않았습니다(추정: 어려움). 실행 시간은 측정 전입니다. 논문은 RR 2에서 4만 스텝에 A100 절반 1시간이라고 했습니다. 10만 스텝이면 약 2.5시간(A100 절반 기준)입니다. 맥의 속도는 측정 전입니다. |
| BBF 구조를 PyTorch(MPS)로 다시 구현하고 게임 1개, 시드 1개 학습 | **조건부(구현 가능)** | 구조와 설정은 위 D1–D13으로 명세할 수 있습니다. 노력은 2–3주(추정)입니다. 학습 시간은 MPS 측정 전이라 추정하지 않습니다. |
| 절제 실험 1개 (예: 재설정 제거) 재현 | **불가에 가까움** | 논문의 절제 팔은 게임당 14–28회 실행입니다(03 §1.3). 26게임이면 수백 회 × 10만 스텝이라 위 두 항목보다 비용이 큽니다. |
| 점수 파일 재계산 (CPU) | **가능, 완료** | 03에서 수행했습니다. |

**노력 추정 (추정, 측정 전):**
- 공식 코드 설치와 한 게임 실행 설정: 1–2일. 패키지 설치 승인이 먼저 필요합니다.
- PyTorch 재구현과 게임 1개 검증: 2–3주.
- 전체 재현: 현실적이지 않습니다. GPU 클러스터가 필요합니다.

## 5. 재실행

공식 코드는 실행하지 않았습니다. 재계산은 `analysis/recompute.py`로 합니다(03 §5).
