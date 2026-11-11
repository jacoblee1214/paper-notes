# 04. 코드 대조: 논문 ↔ Refined-Policy-Distillation/RPD

**정적 읽기만 했습니다.** 코드를 실행하지 않았고 아무것도 설치하지 않았습니다. 교사 쪽 의존성은 GitHub에서 소스 tarball을 받아 임시 폴더에서 읽었습니다(공개 자료에 넣지 않음).

## 1. 확인한 파일

| 파일 | 줄 수 | 내용 |
|---|---|---|
| `repo/src/rpd/ppo_rgb_rpd.py` | 756 | RPD 학습 본체. 손실은 633–672행, 교사 질의는 636–646행, 서버 기동은 752–756행 |
| `repo/src/rpd/ppo_rgb.py` | 610 | PPO 기준선. RPD와 달리는 것은 MSE 항, RPD 인자, 로그 메시지뿐 (`diff`로 확인) |
| `repo/train.py` | 약 90 | 실험 설정. `use_rpd`, 환경, 시드, 하이퍼파라미터, 기본 인자 |
| `repo/README.md`, `pyproject.toml`, `Makefile` | — | 설치, 의존성, 데이터 |

커밋: `10becdf` (2025-07-18, HEAD, shallow clone). 저장소 이력은 3개 커밋(`2f71ca8`, `39ba6a5`, `10becdf`)입니다. 논문 v1 (2025-03-06)과 v2 (2025-08-04) 사이에 코드가 바뀌었는지는 이력만으로 알 수 없습니다. 마지막 push가 2025-07-18이므로 v2 시점의 코드와 같을 가능성이 큽니다.

교사 쪽 대조는 `RobotControlStack/vlagents`의 **2025-07-18 상태(`f51338c`)**를 기준으로 했습니다. 이 상태에 `src/agents/policies.py`의 `OctoActionDistribution`, `OpenVLADistribution`과 `octodist`, `openvladist` 등록, `client.py`의 `RemoteAgent`, `evaluator_envs.py`의 `start_server`, `wrappers.py`의 `HumanCameraWrapper`가 있습니다. 현재 HEAD (`1b98af4`)에는 이 경로가 없으므로(D16) 재현하려면 2025 상태를 고정해야 합니다.

## 2. 논문 ↔ 코드 대조 (D 항목)

판정 기호: **일치**, **차이** (논문과 코드가 다름), **미확인** (이 조사에서 확인할 수 없음), **정보 없음** (논문이나 코드 어느 쪽에도 없음).

| ID | 항목 | 논문 | 코드 (위치) | 판정 |
|---|---|---|---|---|
| D1 | 손실 형태와 가중치 | 식 (3) L_RL − L_MSE, 가중치 1 (§III-A) | `loss = pg_loss − ent_coef·H + vf_coef·v_loss + ppd_lambda·l_ppd`, `ppd_lambda = 1.0` 상수 (633–672행, `train.py`의 `--ppd-lambda=1.0`) | **일치** (부호 규약: `pg_loss`는 surrogate의 음수) |
| D2 | MSE의 정규화 | 식 (5), (μ − E ā)² (합 형태로 읽힘) | `F.mse_loss(means, p.mean)`: 배치와 7차원 모두 평균 (655–656행) | **차이** (합 형태 대비 1/7). 논문에 정규화 설명 없음 |
| D3 | 교사 질의 빈도 | "상태당 교사 행동 한 개" (§III-A) | 미니배치마다 `model.act` 호출 (636–646행). 에폭 8 × 미니배치 49 = 반복당 392회, 상태당 **8회** | **차이** (논문은 한 번, 코드는 에폭마다 한 번). 같은 상태를 8번 다시 질의 (`03` §4.6) |
| D4 | 교사 출력의 성질 | 확률적 샘플 한 개 (§III-A, §III-C) | Octo: `sample_actions`로 확률적 샘플 1개, 배치 질의 (vlagents `octodist`). OpenVLA: `do_sample=False`이고 이미지마다 순차 추론 (vlagents `OpenVLADistribution`) | **일치** (§III-C의 "OpenVLA는 배치 추론 미지원"과 맞음). Octo는 확률적이므로 같은 상태의 목표가 에폭마다 바뀜 |
| D5 | MSE 가중치 스케줄 | 상수 (§IV-A, RPD). PPD만 λ 조정 언급 | `max_ppd_anneal_steps=None`이면 상수 `ppd_lambda`. 코사인 감쇠 코드는 있으나 `train.py`에서 주석 처리 (`# args.append("--max-ppd-anneal-steps=6_000_000")`). 주석 "starts at a value of 2"는 식 `0.5cos(0)+0.5 = 1`과 맞지 않음 | **일치** (주 실험은 상수). 주석 불일치는 코드 결함 |
| D6 | 분포 추정 분기 | PPD는 교사 샘플 10개로 다변량 가우시안을 맞춤 (§IV-A). RPD는 1개 (§III-A) | `Args.num_vla_samples` 기본 10이고 `train.py`는 1. 분기 `if num_vla_samples != 1`는 KL 경로. 그런데 서버 호출 `Obs(cameras=...)`에 `info["num_samples"]`를 넣지 않으므로 서버는 항상 1 (`obs.info.get("num_samples", 1)`). `stds`는 0 | **차이 / 주의**: 인자 `num_vla_samples`는 분기 선택에만 쓰이고 서버 샘플 수를 바꾸지 않음. KL 경로를 켜면 공분산 0이 됨. 실제로 쓰이는 경로는 MSE/L1/BC뿐 |
| D7 | 관측 공간 | 인간 카메라 256×256 RGB, 교사와 학생 같은 카메라, 상태 정보 없음 (§III-B) | 학생: `obs_mode="rgb"`, `HumanCameraWrapper`, `FlattenRGBDObservationWrapper(rgb=True, depth=False, state=args.include_state)`, 기본 `include_state=False`. 교사: `cameras["rgb_side"]`, octo 어댑터에서 `(256,256,3)` assert | **일치** |
| D8 | 지시문 | Table I 문자열 (§III-B) | `INSTRUCTIONS` 사전 (43–55행): 8개 과제 문자열이 Table I과 같음 | **일치** |
| D9 | 하이퍼파라미터 | "ManiSkill3 기준선의 상수 하이퍼파라미터, 배치 크기만 바꿈" (§IV-A) | `train.py`: num_envs 128 (Args 기본 512), num_steps 50, num_minibatches 48 (기본 32), update_epochs 8 (기본 4), total_timesteps 3×10^7 (기본 10^7), eval_freq 10, num_eval_envs 500 (기본 8), lr 3e-4, gae 0.9, clip 0.2, target_kl 0.2, ent 0, vf 0.5, max_grad_norm 0.5, norm_adv True | **차이 / 미확인**: 미니배치 수와 에폭 수도 기본값과 다름. "기준선"이 ManiSkill3의 값인지는 확인하지 않음 (ManiSkill 저장소의 기준선 파일은 조사하지 않음) |
| D10 | sparse의 γ | 0.8 → 0.99, 다른 변경 없음 (§IV-C) | `gamma = 0.99 if sparse else 0.8`. 같은 `train.py`가 `use_rpd`에 상관없이 적용되므로 PPO 기준선도 같음 | **일치**. 이전 메모의 "기준선에도 적용됐는지 불확실"은 해소 |
| D11 | 시드 | 5개 (Fig. 3–5, 7), OpenVLA 1개 | `train.py`의 `seed = 0` 기본, "mass experiments"는 사용자가 바꾸라는 주석. 시드 목록과 실행 스크립트는 없음 | **미확인** (실행 목록 없음) |
| D12 | 평가 | 주기적 검증 롤아웃 (Fig. 3–5, 7) | `eval_envs`는 500개, `eval_reconfiguration_freq=1` (매 리셋마다 물체 재배치), 결정적 평균 행동 `deterministic=True`, 50 스텝. 기록 키는 `eval_infos["episode"]`의 항목 | **일치** (성공 지표의 정확한 키는 ManiSkill이 정의. 이 조사에서 확인하지 않음) |
| D13 | 시뮬레이터 백엔드, 보상 | ManiSkill3 (§III-B), dense 정규화 보상 (§IV-C) | `sim_backend="gpu"`로 고정 (370–380행), `reward_mode = "normalized_dense" if dense else "sparse"`. 빈 `sim_backend`를 CPU로 바꾸는 옵션 없음 | **차이 (재현성)**: GPU 시뮬 강제 (`04` §4) |
| D14 | 교사 체크포인트 | Octo, OpenVLA 파인튜닝본 (§III-B) | Octo `hf://Juelg/octo-base-1.5-finetuned-maniskill`, horizon 1. OpenVLA `Juelg/openvla-7b-finetuned-maniskill`, `unnorm_key = "maniskill_human:7.0.0"`, `checkpoint_step = 40000`, `attn_implementation = "flash_attention_2"`, `device = "cuda:0"` | **일치** (HF 확인). OpenVLA는 CUDA와 flash-attn 2가 필요 |
| D15 | PPD 기준선 | 자체 구현, 공식 코드 없음 (§IV-A) | 저장소에 PPD 구현 없음. `vlad_loss_type`은 mse/l1/bc만 | **정보 없음**: PPD 결과는 이 저장소로 재현 불가 |
| D16 | 의존성 버전 | 명시 없음 | `agents @ git+https://github.com/juelg/agents.git`: 커밋 미고정. `mani_skill @ v3.0.0b15`는 고정. 현재 HEAD의 `vlagents`에는 `agents.*` 경로와 `octodist`/`openvladist`가 없음 | **차이 (재현성)**: 2025-07~08 상태의 `agents`를 고정해야 함 |
| D17 | 서버와 메모리 | 같은 기기면 공유 메모리, 아니면 HTTP (§III-C) | `RemoteAgent("localhost", port, ..., on_same_machine=True)`, 무작위 포트 10000–40000, `start_server`가 하위 프로세스로 교사 실행. `_to_shared_memory`와 `_from_shared_memory` 사용 (2025 `client.py`, `policies.py`) | **일치** |
| D18 | 로그 이름 | — | `losses/ppd_kl`에 실제로는 MSE가 기록됨 (690행). `losses/ppd_loss = ppd_kl × ppd_lambda` | **정보 없음** (이름만 혼동). 분석에서는 MSE로 읽음 |
| D19 | PPO 세부 | 식 (4) 클리핑, CleanRL 기반 (§IV-A) | 클리핑 surrogate (614–616행), 가치 손실 비클리핑(`clip_vloss=False`), 기울기 노름 0.5, 정규화 advantage (CleanRL 표준) | **일치** |
| D20 | 기준선 공정성 | "모든 변형 같은 하이퍼파라미터" | `ppo_rgb.py`는 `ppd_lambda`, `num_vla_samples` 인자가 없고 `train.py`의 `use_rpd=False` 경로가 이를 붙이지 않음. 나머지 인자는 같음 | **일치** (RPD와 PPO 차이가 MSE 항 하나로 한정됨) |
| D21 | RPD-BC, L1 | 식 (2) 최대우도, L1 (§IV-A) | `bc`: `-p.log_prob(means).sum(1).mean()` (학습되는 log-std 사용). `l1`: `F.l1_loss` | **일치** |
| D22 | 과제 최대 길이, 데이터 필터 | 최대 300 (§III-B, 파인튜닝용), RL 쪽은 기본 | 저장소에 없음 (데이터셋 변환은 HF 데이터셋과 외부 도구) | **미확인** |
| D23 | 교사 질의 시점의 관측 | 같은 상태 (§III-A) | `b_obs[mb_inds]["rgb"]`는 롤아웃 때 저장한 학생 관측. 질의 시점의 현재 정책이 아님 | **일치** (저장된 상태에 대한 질의) |
| D24 | 학습 길이와 곡선 범위 | 명시 없음 | 기본 3×10^7 스텝 (`train.py`). Fig. 3 x축은 약 1.25×10^7에서 끝나고 OpenVLA는 약 0.7×10^7 | **정보 없음** (곡선 길이가 `train.py`와 다름) |

### 2.1 두 가지 설명

- **D3과 D4의 결합.** 논문의 "한 샘플"은 질의 횟수를 줄이기 위한 선택으로 읽힙니다. 코드에서는 같은 상태를 에폭마다 질의합니다. Octo는 확률적이므로 에폭마다 다른 샘플이 들어옵니다. 기댓값 논리(`03` §4.3)는 유지되지만 계산 비용은 8배입니다.
- **D6의 의미.** `num_vla_samples`를 1보다 크게 설정해도 서버는 여전히 샘플 1개를 줍니다. KL 경로(PPD형)는 기본 설정에서 공분산이 0이 되어 제대로 동작하지 않습니다. 논문의 PPD는 이 경로가 아니라 자체 구현으로 돌렸을 것입니다.

## 3. 저장소의 실행 흐름 (정적 추적)

1. `train.py`가 `ppo_rgb_rpd.py`를 인자와 함께 `os.system`으로 호출 (`shlex.join`).
2. `__main__`에서 `vla_type`에 따라 `start_server("octodist" 또는 "openvladist", kwargs, port, ...)`를 하위 프로세스로 띄움. Python 경로는 `vla_conda_path`.
3. 5초 대기 후 `main()`: `RemoteAgent`로 연결, `model.reset(obs=Obs(), instruction=...)`.
4. 환경 2개 (학습 128, 평가 500), 래퍼 (`HumanCameraWrapper`, `FlattenRGBDObservationWrapper`, `ManiSkillVectorEnv`).
5. 반복마다: 롤아웃 50 스텝 → GAE → 에폭 8 × 미니배치 (PPO 손실 + 교사 질의와 MSE) → 기울기 클리핑 → Adam.
6. 평가는 `eval_freq`마다 `iteration % eval_freq == 1`일 때.

## 4. 구현할 수 있는지

### 4.1 공식 재현

- **필요한 것:** NVIDIA GPU (ManiSkill GPU 시뮬, `sim_backend="gpu"` 고정), conda 환경 두 개 (RPD와 교사 VLA), `mani_skill` v3.0.0b15, `agents` 2025 상태, HF 체크포인트 (Octo 약 93M, OpenVLA 7B), 시연 데이터 (RLDS 또는 원본 HDF5 재생).
- **노력:** 높음. 의존성 고정 (D16), 시연 재생으로 인간 카메라 영상을 만드는 단계 (README의 `replay_trajectory` 명령), `HumanCameraWrapper` 위치 (agents 저장소) 확인이 필요합니다.
- **계산:** 논문은 OpenVLA를 H100 4노드 (노드당 GPU 4장 중 2개씩 인스턴스, 병렬 8)에서 돌렸고 Octo는 A40 한 장에서 돌렸다고 씁니다(§III-C). Octo 한 번 실행 (3×10^7 스텝, D3 기준 교사 질의 2.4×10^8 이미지)의 벽시계 시간은 논문에 없습니다. 논문 그림 수준 (1.25×10^7 스텝, 5 시드)은 A40급 GPU 수 장으로 며칠 규모로 추정합니다. 추정치이며 측정하지 않았습니다.
- **판정:** 기술적으로 가능하지만 GPU가 있는 곳에서만, 버전 고정 작업을 거쳐야 합니다.

### 4.2 이 맥 (Apple M4 Pro, 약 24 GiB 통합 메모리, MPS, CUDA 없음, torch 미설치)

- **공식 재현: 불가.** `sim_backend="gpu"` 하드코딩 (D13)과 OpenVLA의 `cuda:0`과 `flash_attention_2` (D14)가 막습니다. Octo는 JAX 기반이고 Apple 쪽 GPU 가속 경로가 표준이 아닙니다.
- **CPU 검증: 제한적으로 가능.** 학생 PPO의 MSE 항은 교사 없이도 토이로 확인할 수 있습니다(`03` §5). 이것은 손실 수학의 확인이며 논문 수치 재현이 아닙니다.
- **MPS로 OpenVLA:** 7B 모델은 이 기기에서 실용적이지 않습니다.

### 4.3 아이디어 재구현 노력

- **손실 항만 옮기기:** CleanRL 스타일 PPO에 교사 질의 함수와 MSE 한 줄을 붙이는 것은 쉽습니다. 어려운 부분은 교사 서버와 GPU 시뮬입니다.
- **교사를 바꾸기:** 교사를 임의의 정책 함수로 바꾸면 RPD 항은 교사를 모르고 동작합니다. 교사 질의 비용이 줄어듭니다(예: 교사 질의를 롤아웃 때 한 번만).

### 4.4 판정 (한 줄)

**공식 재현은 NVIDIA GPU 환경과 버전 고정이 있을 때만 가능하고, 이 맥에서는 불가입니다. 아이디어(MSE 항) 재구현은 쉽고, 그 효과를 교사 없이 확인하는 토이는 CPU에서 가능합니다.**

## 5. 라이선스와 코드 사용

- RPD: Apache-2.0 (`LICENSE` 파일과 `pyproject.toml` 일치).
- vlagents: Apache-2.0 (GitHub API). 이 조사에서 받은 소스는 공개본에 넣지 않습니다.
- 체크포인트: MIT (태그), 데이터셋: Apache-2.0 (태그). 두 가지 모두 HF 모델 카드에 명시적 라이선스 파일이 있는지는 확인하지 않았습니다.
- 공개본에는 코드를 복사하지 않고 파일명과 줄 번호로만 인용합니다.

## 6. 재현에 필요하지만 논문과 코드에 없는 정보

- 시드 목록과 실행 명령, 교사 질의 방식의 실제 로그 (D11)
- 과제별 최대 에피소드 길이와 데이터 필터 (D22)
- ManiSkill3 기준선의 하이퍼파라미터 (D9)
- PPD 구현 (D15)
- 교사 샘플 방식의 `agents` 2025 상태 확인 (이 조사에서 확인한 것은 2025-07-18 `f51338c`까지)
- 학습 길이 (D24)
- 벽시계 시간과 GPU 사용량
