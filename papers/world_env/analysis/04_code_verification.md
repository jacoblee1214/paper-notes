# 04. 공식 코드 대조: 논문 ↔ `amap-cvlab/world-env`

대상: GitHub `amap-cvlab/world-env` (Apache-2.0, 2025-10-31 생성, 마지막 push 2025-10-31, 별 47, 포크 0). 복제 커밋 `55bc757`(단일 커밋 `init`). 복제 위치 `repo/world-env/` (얕은 복제, 정적 읽기만). **실행, 설치, 가중치 다운로드는 하지 않았다.** 검사한 것은 파일 내용과 설정값이다.

대조 기준은 논문 **v6 (2026-04-27)**이다. 저장소 커밋은 2025년 10월 말이라 **v1–v3 시점의 방법과 맞출 가능성이 크다**. v4–v6에서 추가된 VGGT 주입, 실세계 실험, Octo·RIPT 비교는 저장소에 흔적이 없다 (D1, D9, D17).

## 1. 저장소 구성 (읽은 범위)

| 경로 | 내용 | 논문과의 관계 |
|---|---|---|
| `test.py` | Hydra 진입점. `algo.eval_only=False`면 RL 학습, `True`면 평가 | §4.3 RL 루프 |
| `lib/env_runner/openvla_oft_libero_runner.py` | 학습 롤아웃 `run_episode`(842–1122행)와 평가 롤아웃 `run_episode`(660–840행) | §4.1–4.2 가상 환경 루프 |
| `lib/algos/rl_optimizers/rl_optimizer_openvla_oft.py` | RLOO 이점 계산(103–113행), PPO 클리핑(184–186행) | §4.3, 식 (5)–(6) |
| `lib/algos/rl_optimizers/openvla_oft_interface.py` | 정책·스케일 헤드 적재, 학습 가능 파라미터 설정(384–386행) | §4.4 |
| `lib/enerverse/` | EnerVerse-AC 기반 세계모델 추론(`infer_image.py`), 학습 설정 yaml, `evac/` 모델 코드 | §4.1 세계 시뮬레이터 |
| `lib/LLaVA/predict_reward.py` | 반영기 추론: LLaVA 백본, 32프레임, `RewardHead`, sigmoid | §4.2 |
| `lib/algos/reward_functions/libero.py` | `SuccessReward`: `rollout_episode['success']` 반환 | §4.2 보상 |
| `config/` | RL 설정 (`train_rl_openvla_oft_all_task_{goal,object,spatial,long}.yaml`), `algo/openvla_oft_rl.yaml`, `paths.yaml` | §4.4 구현 세부 |
| `scripts/openvla_oft/eval/*.sh` | 평가 스크립트 4개 (LIBERO 스위트별) | §5 평가 |
| `INSTALL.md`, `requirements.txt` | Python 3.10, torch 2.4.0, CUDA 계열 패키지 (`cupy-cuda12x`, `tensorflow==2.15.0`, `ray==2.40.0`) | 재현 비용 |

공식 체크포인트와 학습 데이터 파이프라인은 저장소에 없다. README는 체크포인트를 ModelScope(`junjxiao/world-env`)에서 받으라고 하는데, **이번 분석에서는 받지 않았다.**

## 2. 논문과 코드가 맞는 부분 (확인됨)

- **RLOO 이점 식 (5).** 코드는 `baseline = (sum - r) / (N-1)`, `advantage = r - baseline` (`rl_optimizer_openvla_oft.py` 103–113행). 논문 식 (5)와 같다.
- **PPO 클리핑 0.1.** `ppo_clip_range: 0.1`, `ppo_clip_high: 0.1` (RL 설정 4개 모두). 논문 §4.4와 같다.
- **반영기 임계값 η = 0.5.** `task_done = score > 0.5` (`run_episode`, 975행). 논문 식과 같다.
- **반영기 점수 형태.** `success = torch.sigmoid(reward_head(h))` (`predict_reward.py` 412–413행). 논문 식 (4)와 같다.
- **반영기 프롬프트와 프레임 수.** "Watch the video and determine whether it completes the task ... answer only 'Yes' or 'No'" (185행), 32프레임 균등 추출 (161행). 논문 §9.2와 같다.
- **RL 학습 때 보상은 종료 시점의 반영기 점수 하나.** `SuccessReward`가 `rollout_episode['success']`를 돌려주고, 학습 롤아웃의 그 값은 마지막 반영기 점수다 (`run_episode` 1122행 `return score, ...`). 논문 §4.2의 단일 종료 보상과 효과가 같다. 다만 이름이 `success`라서 읽기 어렵다 (D3).
- **평가는 실제 시뮬레이터 영상을 쓴다.** 평가 롤아웃(660–840행)에서 세계모델 호출은 주석 처리되어 있다. 표 1·4의 정책 평가가 LIBERO 실제 렌더링을 쓴다는 뜻이다. 학습에서만 세계모델을 쓴다.
- **Laplace 샘플링과 스케일 헤드.** `use_laplace_sampling: true`, `laplace_log_prob`(169행). 논문 §4.1–4.3과 같다.

## 3. 차이 (D1–D17)

판정 기호: **차이** = 논문과 코드가 다름, **미기재** = 논문에 없는 설정이 코드에 있음, **누락** = 논문이 말하는 것이 코드에 없음, **주의** = 맞지만 해석에 주의가 필요함.

### D1. 기하 특징 주입(VGGT) 누락 — 누락 (논문 기여 2)

- 논문: 세계 시뮬레이터가 VGGT 잠재 특징과 CLIP 특징을 U-Net 교차어텐션으로 주입 (§1 기여 2, §4.1, 그림 4).
- 코드: `vggt`를 검색하면 **0건**이다 (`grep -rli vggt`). 세계모델은 EnerVerse-AC의 잠재 영상 확산 모델이다 (`lib/enerverse/train_config_iros_challenge_wm.yaml`의 `ACWMLatentDiffusion`, `UNetModel`, `AutoencoderKL`, `FrozenOpenCLIPImageEmbedderV2`).
- 함의: 논문의 핵심 기여 하나(기하 주입)는 공개 코드로 확인할 수 없다. 논문이 말한 U-Net 픽셀 확산과 저장소의 잠재 확산이 같은 모델인지도 확정할 수 없다.

### D2. 세계모델의 행동 조건 — 차이 (논문은 Akin to [25]로 정리)

- 논문 §4.1: 행동을 FK로 본체 상태로 바꾸고 영상 평면에 투영한 **행동 지도(action map)**를 조건으로 준다.
- 코드: `data_for_enerverse`에 `c2w`(카메라 외부 행렬)와 `intr`(내부 행렬)을 넣고 `actions`에 `observation['state']`를 쌓는다 (`run_episode` 900–910행 근처). 투영은 세계모델 쪽 `infer_image.py`에서 한다. 행동 지도 생성 코드는 이 저장소 안에서 확인하지 못했다.
- 판정: 방향은 같다(카메라 투영 기반 행동 조건). 세부 구현은 확인하지 못했다.

### D3. 보상 이름과 해석 — 주의

- `generate_rollouts`(`rollout_generator.py` 265행)의 `rollout_stats[init_hash].append(int(success))`는 연속 점수를 `int()`로 자른다. 점수가 1.0 미만이면 0이 된다. 이 값은 "이미 모두 성공한 초기 상태 건너뛰기"(223–228행)에 쓰인다. 연속 반영기 점수에 이진 성공 로직을 쓰는 것이라 의도와 다를 수 있다. 실행해 보지 않아서 실제 영향은 확인하지 못했다.

### D4. 반영기 시작 스텝 — 미기재

- 코드는 반영기를 `t > start_pred_step`일 때만 부른다 (975행). `TASK_START_STEPS`는 spatial 110, object 140, goal 150, long 260, libero-90 200이다 (94–100행).
- 논문: 임계값을 넘는 **어느 시점이든** 종료한다 (§4.2). 시작 스텝은 없다.
- 함의: 공개 코드의 종료는 시작 스텝 이후에만 가능하다. 표 1의 짧은 과제(예: Goal)에서는 종료 효과가 줄 수 있다. 논문은 이 제약을 말하지 않는다.

### D5. 세계모델 호출 간격과 실제 물리 — 차이 (해석 주의)

- 논문: 시뮬레이터는 행동을 받아 다음 관찰을 만들고, 정책은 그 관찰로 다음 행동을 낸다 (§4, 그림 3). 본체 상태는 FK로 계산한다.
- 코드 (학습 롤아웃): `env.step(step_actions)`(1005–1010행 근처)로 **LIBERO 물리가 계속 돈다**. 관찰의 실제 상태(`observation['state']`)와 손목 카메라는 LIBERO에서 온다. 세계모델이 만든 것은 `observation['full_image']`(외부 카메라 영상) 하나다 (`run_episode` 958–980행).
- 세계모델은 `chunk`(8)스텝마다 한 번 호출된다 (`action_nums % chunk == 0`).
- 함의: 공개 코드의 "가상 환경"은 **외부 카메라 영상만** 세계모델이 만들고, 상태와 물리는 시뮬레이터다. 논문의 "물리 상호작용을 대체한다"는 표현보다 좁다. 이 차이는 표 1의 RL 효과가 어디서 오는지 해석을 바꾼다 (`03` §4).

### D6. 학습률 — 차이

- 논문 §4.4: LoRA 1e-4, 헤드 1e-5.
- 코드: RL 설정 spatial, object, long은 `lr: 5e-5`, `header_lr: 5e-5`이고, goal은 `lr: 1e-5`, `header_lr: 1e-5`다. 기본값 `openvla_oft_rl.yaml`은 `lr: 1e-4`, `header_lr: 5e-5`다. 스위트마다 학습률이 다르고, 어느 값이 표 1 실험에 쓰였는지는 논문에 없다.

### D7. 스케일 헤드 고정 — 차이

- 논문 §4.4: "action/scale heads are trained with full parameters" (학습률 1e-5).
- 코드: 4개 RL 설정 모두 `fix_scale_head: true`. `openvla_oft_interface.py` 384–386행에서 `fix_scale_head`가 참이면 스케일 헤드를 학습 파라미터에서 뺀다. 즉 RL 동안 탐색 분산을 정하는 헤드는 고정이다.
- 함의: 논문이 말하는 "불확실성 인지 탐색"은 학습되는 것이 아니라 SFT 때 얻은 헤드를 쓰는 것이다. 논문 §4.3의 설명과 다르다.

### D8. 논문에 없는 RL 설정 — 미기재

- `scale_factor: 5.0` (RL 설정; 기본 `algo/openvla_oft_rl.yaml`은 1.0). 탐색 분산을 직접 바꾼다.
- `log_scale_clip: [-2.0, 0.5]`, `grad_norm_clip: 1.0`, `rollouts_per_env: 50`, `num_parallel_envs: 1`, `gradient_accumulation_steps: 4`.
- RL 반복 수: `train_rl_openvla_oft_all_task_{goal,object,spatial}.yaml`의 `training.n_steps`는 각각 **2, 6, 3**이다. 기본 `train_base_rl_openvla_oft.yaml`은 26이다. 논문은 반복 수를 쓰지 않았다 (그림 5는 20 스텝).
- 함의: 공개 설정으로 돌린 실행 길이가 매우 짧다. 이것이 실제 논문 실험의 길이인지, 시험 설정인지 저장소만으로는 확인할 수 없다. 표 1의 수치를 재현하려면 이 값을 알아야 한다.

### D9. 실세계 실험 코드 없음 — 누락

- 논문 v4–v6의 실세계 실험(표 3, Franka/로봇 세부, 네 과제)과 Octo 비교(표 6)는 저장소에 없다. 실세계 제어 코드, 로봇 설정, 데이터는 없다.

### D10. 세계모델 학습 데이터 — 차이

- 논문 §4.1: 세계모델 학습 데이터는 LIBERO 시연과 OFT가 만든 자율 탐색 궤적을 합친 것이다. 탐색에는 Laplace 노이즈를 준다.
- 코드: 학습 설정 `data_roots: ["/mnt/nas-data-5/<저자 계정>/dataset/libero_for_enerverse/libero_90/"]`, `sample_n_frames: 64`, `chunk: 8`. **LIBERO-90**을 쓴다. 논문은 LIBERO 네 스위트를 말한다.
- 탐색 데이터 생성 코드(`run_episode`의 `enerverse_path` 저장 블록, 1075–1110행 근처)는 **주석 처리**되어 있다. 반영기 학습 코드도 저장소에 없다 (추론 쪽 `RewardHead`만 있다).
- 함의: 세계모델을 저장소만으로 다시 학습할 수 없다. 데이터 양도 논문에 없다. "5개 시연만 쓴다"는 주장은 OFT 정책에만 맞고, 세계모델과 반영기가 쓴 데이터 양은 논문에 없다 (`01` §5).

### D11. 세계모델 학습 설정 — 미기재

- `base_learning_rate: 5e-5`, `ddim` 설정, `ae_batch_size: 2`, `batch_size: 8`, `gradient_clip_val: 0.5`. 논문 §4.4에 학습 step 수와 에폭이 없다.

### D12. 반영기 백본 — 미기재 (LLM 크기)

- `RewardHead(input_dim=3584, hidden_dim=4096)`. 3584는 Qwen2-7B 계열의 은닉 차원과 같다. 논문은 "LLaVA"만 말하고 LLM 크기를 쓰지 않았다. 시각 타워는 `siglip-so400m-patch14-384`다 (`predict_reward.py` 193행). 논문 §9.2의 "frozen vision encoder"와 맞지만, 시각 타워 이름은 논문에 없다.
- 헤드 초기화: 편향 -3.0, Xavier gain 0.01 (147–148행). 논문에 없다.

### D13. 체크포인트와 경로 — 누락 (재현 차단 요인)

- `config/paths.yaml`: `wm_path: null`, `reward_path: null`, `data_prefix: YOUR_DATA_PATH`.
- 여러 파일이 `/mnt/nas-data-5/<저자 계정>/...`, `/mnt/workspace/<저자 계정>/...` 같은 저자 환경 경로를 쓴다 (세계모델 설정, `libero_utils.py` 398–399행 CLIP, `predict_reward.py` 193행 SigLIP, 학습 로그 경로). 재현하려면 모두 바꿔야 한다.
- 평가 스크립트는 `pretrained_models/openvla-oft-5traj/...`와 `experiments/libero/...`의 LoRA 어댑터를 요구한다. 학습 결과물은 저장소에 없다.

### D14. README의 논문 식별자 오류 — 차이 (문서)

- 저장소 README는 "Official implementation of World-Env ..."를 `https://www.arxiv.org/abs/2505.17016`에 연결한다. **2505.17016은 다른 논문이다.** 인용 블록은 `eprint 2509.24948`로 맞다.

### D15. 논문 URL 표기 — 주의 (링크)

- v6 PDF 본문은 `https://github.com/amap-cvlab/worldenv`를 쓴다. 이 주소는 **404**다 (GitHub API와 HTTP 상태 모두). arXiv abs 페이지의 "this https URL" 링크와 v2 본문의 `junjxiao/world-env`는 각각 `amap-cvlab/world-env`(200)와 다르다. 올바른 주소는 `amap-cvlab/world-env`다.

### D16. 저장소와 논문 버전의 시차 — 주의

- 저장소: 단일 커밋 `init`, 2025-10-31. 논문 v4(2026-03-03)에서 이름이 "RehearseVLA"였다가 v5(2026-03-18)에서 "World-Env"로 되돌아갔다. 저장소 이름은 `world-env`다. 시차 때문에 저장소를 v6 방법의 구현으로 보면 안 된다.

### D17. 평가 설정 — 확인 불가 (수치)

- 저장소에는 결과 로그, 평가 에피소드 수, 시드 목록이 없다. 표 1–6의 수치를 저장소로 대조할 수 없다. `scripts/*.sh`는 `algo.model_seed=1` 하나만 쓴다.

## 4. 구현 가능성 판정 (이 맥 기준)

**결론: 이 맥(Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)에서 논문 수치의 재현은 불가능합니다. 공식 세계모델 추론 한 번은 가중치 다운로드와 CUDA 의존 제거 후에 조건부로 가능하고, 방법의 축소판(토이)은 구현할 수 있습니다.**

| 항목 | 판정 | 근거 |
|---|---|---|
| 논문 표 1 (5-demo LIBERO RL 학습) | **불가** | 8×H20 48시간 (논문 §4.4). 저장소에 CUDA 계열 의존(`cupy-cuda12x`, NCCL 기반 `torchrun`, `tensorflow==2.15.0`) |
| 표 1 평가 (OFT 5traj 체크포인트 + 공식 LoRA) | **조건부** (CPU 가능성 측정 전) | 7B 정책을 CPU에서 LIBERO 500 에피소드 평가는 시간이 매우 많이 든다 (추정). 체크포인트와 LoRA 다운로드 승인 필요 |
| 공식 세계모델 추론 (EnerVerse-AC 기반 잠재 확산, `chunk`=8) | **조건부** | 가중치 다운로드(ModelScope) 승인, 잠재 확산 코드의 MPS 포팅 필요. 속도와 메모리는 측정 전 |
| 반영기 추론 (LLaVA 계열 7B 추정 + 헤드) | **조건부 (메모리 빠듯)** | 백본 크기가 논문에 없다. 임베딩 차원 3584로 볼 때 7B급 추정. 24 GiB에서 세계모델과 같이 올리기 어렵다 (추정) |
| RL 학습 재현 (축소판 포함) | **불가 (GPU 필요)** | 공식 설정은 GPU 기준. 축소판도 정책 학습은 CPU에서 느리다 |
| 세계모델 재학습 | **불가** | 학습 진입점 없음(추론과 모델 클래스만 있음), 탐색 데이터 생성 코드는 주석 처리 (D10) |
| 개념 재구현 (RLOO + 반영기 종료 + 행동 조건 토이) | **가능** | `analysis/toy_advantage.py`로 이점 퇴화 메커니즘을 이미 확인했다. 소형 정책과 소형 세계모델로 확장은 1–2주 (추정, 측정 전) |

노력 추정은 **공식 추론 한 번에 2–4일(다운로드·포팅 포함, 추정)**, **축소 RL 루프 1–2주(추정)**다.

## 5. 재현 전 확인할 질문 (저자 문의 후보)

1. 세계모델 학습 데이터 규모 (시연 수, 탐색 궤적 수, LIBERO-90 사용 여부).
2. 반영기 학습 데이터와 임계값의 근거 (표 7의 평가 클립 수 n=40 가정이 맞는지).
3. RL 반복 수, 초기 상태 수, 최대 스텝, 시드.
4. VGGT 주입 코드가 있는 다른 저장소나 브랜치가 있는지.
5. 표 1–6의 학습 설정이 저장소 기본값(`scale_factor 5.0`, `fix_scale_head`)과 같은지.
