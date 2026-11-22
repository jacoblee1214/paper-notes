# 04. 코드 검증: 논문 ↔ 공식 저장소 (정적 읽기)

- 대상: `mani-skill/ManiSkill`, 클론 HEAD `62ff3a5` (커밋 2026-08-01), `setup.py` 버전 3.0.1. 클론 위치 `repo/` (gitignore).
- 방법: `grep`과 파일 읽기만 했습니다. 설치, import, 예제 실행, 학습은 하지 않았습니다. 클론 내용은 신뢰하지 않는 데이터로 취급했습니다.
- 논문 버전: arXiv v2 (2025-05-30). 공식 저장소는 v2 이후 커밋을 포함합니다. 논문과 코드가 다르면 **코드가 나중 버전**이라는 점을 감안해야 합니다.

## 1. 확인 결과 요약

| 항목 | 논문 | 코드 | 판정 |
|---|---|---|---|
| PPO 할인율 gamma | 기본 0.8 (App. IX-A) | `examples/baselines/ppo/ppo.py` 76행 `gamma: float = 0.8` | 일치 |
| PPO GAE lambda | 수치 없음 | 0.9 (78행) | 논문에 없음. 코드값만 기록 |
| PPO 기본 env 수 | state 최대 4096, RGB 256–1024 (App. IX-A) | 기본 512 (56행) | 기본값과 권장 범위가 다름. 과제별 튜닝이라 모순 아님 |
| PPO 롤아웃 길이 | 16 또는 32, 가능하면 줄임 (App. IX-A) | 기본 50 (64행) | 기본값과 권장 범위가 다름 (D1) |
| partial reset | 모든 실험에서 켬 (App. IX-A) | 기본 `partial_reset = True` (60행), 평가는 False (62행) | 일치 |
| reconfiguration freq | 학습 0 (App. IX-A) | 기본 `None` (68행), 평가 1 (70행) | D2: None이 0과 같은 뜻인지 확인 필요 |
| TD-MPC2 버퍼 | state 1M, vision 200K (App. IX-B) | `examples/baselines/tdmpc2/config.yaml` `buffer_size: 1_000_000` | state 값만 일치. vision 200K는 저장소에서 찾지 못함 (D3) |
| SIMPLER 4개 과제 | Table·§III-E: 4 과제 포팅 | `digital_twins/bridge_dataset_eval/put_on_in_scene.py`에 `register_env` 4개 | 일치 |
| 로봇 20+ | §III-F | `mani_skill/agents/robots/`에 패키지 24개 (`_template` 제외) | 일치 |
| Cartpole 솔버 설정 | 위치 4, 속도 0 (App. XI-A) | `cartpole.py` 98–101행 `solver_position_iterations=4, solver_velocity_iterations=0` | 일치 |
| Cartpole 시뮬 120 Hz, 제어 60 Hz | App. XI-A | `mani_skill/examples/benchmarking/envs/maniskill/cartpole.py` 75–81행 `sim_freq=120`, `control_freq=60`, solver 4/0. Isaac 쪽 `envs/isaaclab/cartpole_visual.py` 30행 `dt=1/120`, `render_interval=2`, decimation 2 | 일치 (벤치마크 스크립트 기준) |
| Koch v1.1 sim2real 과제 | §IV-D, App. VIII | `mani_skill/agents/robots/koch/` 로봇 정의는 있음. Koch 전용 큐브 과제·그린스크린 설정은 찾지 못함. 대신 `digital_twins/so100_arm/grasp_cube.py` (SO100, `SO100GraspCube-v1`) | D5: 논문의 sim2real 학습 설정이 이 저장소에 없음 |
| 실로봇 에이전트 | 텔레오퍼레이션, sim2real | `mani_skill/agents/base_real_agent.py` 존재 | 존재 확인 (동작 검증 아님) |
| 보상 | 큐브 보상 식 (App. VIII-E) | `pick_cube.py` 161행 `compute_dense_reward`는 TCP–큐브 거리 등 | 항의 구성은 대체로 맞음. 상수 계수는 원문 식과 다를 수 있음 (확인 보류) |

## 2. 불일치와 빈칸 (D 항목)

- **D1. PPO 기본값과 권장 범위.** 기본 `num_steps = 50`, `num_envs = 512`인데 논문은 16–32 스텝과 4096 env를 권장합니다. 과제별 설정은 wandb 기록에만 있다고 논문이 밝힙니다(App. IX-A). 공개 설정 파일 없이는 논문 수치를 그대로 재현하기 어렵습니다.
- **D2. reconfiguration 기본값.** 코드 기본 `None`, 논문 "0". 값이 같은지 `ManiSkillVectorEnv` 쪽 구현을 확인해야 합니다. 이 분석에서는 확인하지 않았습니다.
- **D3. 시각 TD-MPC2 버퍼 200K.** 저장소 기본 설정은 1M입니다. 시각 과제용 오버라이드가 다른 파일(예: wandb 설정)에 있을 수 있어 찾지 못했습니다.
- **D4. (해소) Cartpole 주파수.** 벤치마크 환경 설정에서 120/60 Hz와 솜버 4/0이 확인되어 논문 부록 XI-A와 일치합니다. 다만 이것은 `examples/benchmarking`의 설정이고, 논문 Fig. 2·4의 측정 스크립트 버전이 같은지는 확인하지 못했습니다.
- **D5. sim2real 학습 코드 부재.** 논문의 Koch v1.1 PPO 학습은 이 저장소에서 찾지 못했습니다. 디지털 트윈 예시는 SO100 과제입니다. 논문 Fig. 8·12의 실물 설정을 재현하려면 저자의 별도 설정이 필요합니다. 논문은 "reproducible setup"이라고 쓰지만 공개 경로는 확인되지 않았습니다.
- **D6. 벤치마크 재실행 조건.** 저장소 `mani_skill/examples/benchmarking/`에 ManiSkill과 Isaac Lab 비교 스크립트가 있습니다(`scripts/isaac_lab.sh`, `isaac_lab_gpu_sim.py`, `gpu_sim.py`, `profiling.py`). Isaac 쪽은 `omni.isaac.lab`(Isaac Lab 1.x 네임스페이스)을 import하므로 논문 v1.2.0 계열 설치가 필요합니다. 스크립트는 num_envs 4–16384, 카메라 수 2–4, 해상도 80–512 격자를 돕니다. 실행은 하지 않았습니다. 그래서 Fig. 2·4·31–35의 수치를 재현할 수 없습니다. 측정 횟수와 반복 설정은 `scripts/`에서 한 번씩만 실행되는 것으로 보입니다(확인 필요).

## 3. 저장소에서 확인된 강점 (논문 주장과 맞는 부분)

- 객체지향 API와 Pose 체이닝 (논문 §III-F): `mani_skill/` 하위 구조에서 actor, articulation 객체 중심 설계가 보입니다. 세부 API는 읽지 않았습니다.
- 공개 기준선 코드: `examples/baselines/` 아래 `act`, `bc`, `diffusion_policy`, `ppo`, `rfcl`, `rlpd`, `sac`, `stable_baselines3`, `tdmpc2`. 논문 §IV의 기준선 목록과 맞습니다.
- 디지털 트윈 환경 등록 (`register_env`)은 논문 §III-E와 맞습니다.

## 4. 이 맥에서 재현 가능성 (코드 기준)

| 항목 | 판단 |
|---|---|
| 설치 | Linux GPU 중심. `pyproject.toml`과 README의 설치 조건은 이 분석에서 확인하지 않았습니다 |
| CUDA | PhysX GPU 시뮬레이션은 CUDA 전제 (논문 §II). 이 맥은 CUDA가 없습니다 |
| CPU 시뮬레이션 | 저장소에 CPU 경로가 있는지는 확인 안 함. 논문은 ManiSkill2 CPU 대비 비교만 함 |
| 결론 | 이 맥에서 설치·실행은 조건부 불가. 코드 구조 확인까지만 가능 |

실행 판단(설치 가능 여부와 승인 절차)은 이 문서에서 다루지 않습니다. 이 문서에서는 설치 명령을 쓰지 않았습니다.

## 5. 저자 또는 저장소에 물을 질문

1. Table II와 III의 BC 1000 demos 행이 같은 이유 (복사 오류인지).
2. PPO 권장 설정 (env 수, 롤아웃 길이, reconfiguration) 공개 여부.
3. Koch v1.1 sim2real 학습 설정과 그린스크린 처리 코드 공개 여부.
4. 부록 XI-C의 "24x" 메모리 비율.
