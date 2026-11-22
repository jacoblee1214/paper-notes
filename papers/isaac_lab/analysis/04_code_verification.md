# 04. 공식 코드 대조 (정적 읽기)

## 0. 대조 기준과 범위

- **대조 대상:** `isaac-sim/IsaacLab` 태그 `v2.3.0`, 커밋 `3c6e67b` (2025-10-28, 논문 제출 2025-11-06 이전 마지막 릴리스). 저장소는 `isaac_lab/repo/`에 얕게 복제했다(LFS 제외).
- **읽은 것:** `source/isaaclab/isaaclab/` 아래 `actuators/`, `sensors/`(camera, ray_caster, imu, contact_sensor), `controllers/`(differential_ik, joint_impedance, operational_space, pink_ik, rmp_flow), `envs/`(manager_based_env, manager_based_rl_env, direct_rl_env, direct_marl_env), `assets/articulation/`, `sim/simulation_cfg.py`, `sim/spawners/sensors/`. `source/isaaclab_mimic/`(motion_planners, mimic_env_cfg), `source/isaaclab_rl/`(rl_games pbt), `source/isaaclab_tasks/`(locomotion 로봇 목록, dexsuite, anymal_c 등록), `scripts/benchmarks/`.
- **읽지 않은 것:** 실행 테스트, 자산 USD 파일, 렌더러 설정 세부, 텔레오퍼레이션 장치 드라이버, 실험 설정 파일의 값 전체.
- **실행하지 않았다.** 설치, import, 테스트 없음. 복제본은 신뢰할 수 없는 데이터로 취급했다.
- **버전 차이 (D1).** 논문은 버전을 쓰지 않는다. 최신 `main`(v3.0.0, Isaac Sim 6.2)은 구조가 바뀌었다. 여기서는 v2.3.0을 기준으로 삼는다. v3.0.0에서 같은 주장이 유지되는지는 확인하지 않았다.

## 1. 차이 목록 (D1–D12)

| ID | 논문 | 코드 (v2.3.0) | 판정 |
|---|---|---|---|
| D1 | 버전을 밝히지 않음. 2.x 계열과 Isaac Sim 5.x 시대 (논문 제출 시점) | 대조 대상 v2.3.0. 최신 `main`은 v3.0.0이고 README는 Isaac Sim 6.2, 다중 백엔드(Newton, OVRTX, OVPhysX)를 말함 | 버전 차이 큼. 대조 결과는 v2.3.0에만 적용 |
| D2 | 알고리즘 2: 물리 부분 스텝 안에 "Apply pre-simulation events (if any)" 단계 | 이벤트 모드는 `prestartup`, `startup`, `reset`, `interval`뿐 (`manager_based_env.py`, `direct_rl_env.py`). 부분 스텝 루프는 `_apply_action`, `write_data_to_sim`, `sim.step`, 렌더링, `scene.update`만 함 | 논문 절차에 있는 단계가 코드에 없음. 설계 차이 또는 이름 차이 |
| D3 | 3.2.2절 지연 PD: "buffering desired actions by a configurable number of simulation steps" | `DelayedPDActuator.reset`이 환경마다 지연을 `[min_delay, max_delay]`에서 무작위로 뽑음 (`actuator_pd.py` 339–350) | 논문에 없는 무작위화. 기본 설정과 영향은 확인하지 않음 |
| D4 | 3.2절: 마찰은 두 모델 (상수 Coulomb 계수, 정지 마찰 + 동적 + 점성) | `actuator_cfg.py`는 `friction`(정지), `dynamic_friction`, `viscous_friction` 세 계수를 한 설정에서 받음. `articulation.py` 827–903이 PhysX 관절 속성 0, 1, 2번에 기록 | 기능은 같음. 분류 설명이 다름 (경미) |
| D5 | 3.3.4절: 시각 촉각 센서 시뮬레이션 (TacSL 계열, 그림 10) "implemented in Isaac Lab" | v2.3.0 전체 트리에서 `tactile`, `tacsl`, `visuotactile` 검색 0건. v3.0.0 main 트리에는 `source/isaaclab_contrib/isaaclab_contrib/sensors/tacsl_sensor/`(visuotactile_sensor, visuotactile_render, cfg, data), 문서 `docs/source/experimental-features/visuo_tactile_sensor.rst`, 예제 `examples/sensors/tacsl_sensor.py` | 논문 시점 릴리스에 없음. v3.0.0에서는 contrib 실험 기능 |
| D6 | 3.7.3절: 환경은 네 계열 (classic, locomotion, manipulation, navigation) | `manager_based/` 아래 classic, **locomanipulation**, locomotion, manipulation, navigation 다섯 계열. 5.5.1절의 loco-manipulation 파이프라인과 연결됨 | 계열 목록에서 locomanipulation 누락 (경미) |
| D7 | 그림 7: 컨트롤러로 IK, OSC, NN 정책, OCS2, RMPFlow 표기. 본문 3.4절은 IK, 힘 제어, cuRobo | `controllers/rmp_flow.py` 있음. OCS2 코드는 v2.3.0 어디에도 없음 | 그림 표기 중 OCS2는 코드에 없음. RMPFlow는 있으나 본문 설명은 없음 |
| D8 | 3.7.3절: 직접 작성 다중 에이전트 API는 PettingZoo Parallel API를 "기반으로" 함 | `DirectMARLEnv(gym.Env)` (`direct_marl_env.py` 39행). docstring은 PettingZoo `ParallelEnv` 상속을 선택지로 설명 | API 형태만 따르고 상속하지 않음 (경미) |
| D9 | 7.1.5절: Newton 통합은 "experimental feature branch"에 있음, 기능 동등은 "coming months" | v2.3.0 Python 소스에 newton 참조 없음. v3.0.0 README에는 Newton 백엔드가 있음 | 논문 시점과 맞음. 이후 진행됨 |
| D10 | 4.1절: 처리량은 "Configurations for all evaluations are provided in the IsaacLab repository" | `scripts/benchmarks/`에 `benchmark_rlgames.py`, `benchmark_rsl_rl.py`, `benchmark_non_rl.py`, `benchmark_cameras.py`가 있음. 그림 13–18의 정확한 조합(환경 수 목록, GPU 수, 해상도 조합)을 담은 설정 파일은 찾지 못함 | 일부만 있음. 그림 재현 설정은 직접 맞춰야 함 |
| D11 | 식 (1): FPS = 환경 스텝 ÷ (시뮬 시간 + 학습 시간) | `benchmark_rlgames.py` 244–246은 세 가지 FPS를 기록: "Environment only", "Environment + Inference", "Environment + Inference + Policy update". `benchmark_non_rl.py` 187행은 `effective_fps = fps × num_envs × world_size` | 논문 식 (1)이 어느 로그 항목인지 밝히지 않음. 환경 수는 프로세스(GPU)당 값으로 보임 |
| D12 | 식 (2): FPS = 렌더링 스텝 ÷ 시뮬 시간. 그림 18은 rays/s | `benchmark_cameras.py`는 타일드 카메라와 레이캐스터 카메라 개수와 데이터 타입을 인자로 받음. 그림 17의 ANYmal-D + 환경당 메시 설정은 1:1로 확인하지 못함 | 측정 정의와 시나리오 일치 여부 미확인 |

## 2. 일치 항목

| 논문 | 코드 (v2.3.0) |
|---|---|
| 3.2절: 임플리시트 액추에이터는 PhysX 관절 PD를 쓰고 속도·토크 한계를 솔버 부등식으로 씀 | `ImplicitActuator`, `effort_limit_sim`, `velocity_limit_sim` (`actuator_pd.py` 34–115) |
| 3.2.2절: Ideal PD, DC 모터, 지연 PD, 원격 PD | `IdealPDActuator`, `DCMotor`(토크-속도 클립), `DelayedPDActuator`, `RemotizedPDActuator` (`actuator_pd.py`) |
| 3.2.2절: 신경망 액추에이터 LSTM과 MLP | `ActuatorNetLSTM`, `ActuatorNetMLP` (`actuator_net.py`) |
| 3.1절: 지연 갱신 (lazy update) | `articulation_data.py`의 타임스탬프 기반 버퍼 (`if buffer.timestamp < self._sim_timestamp`) |
| 2.3절: 프리셋으로 품질과 속도를 고름 | `SimulationCfg.rendering_mode`: performance, balanced, quality (`sim/simulation_cfg.py` 322행). 논문에는 이름이 없음 |
| 3.3.2절: 카메라 좌표 규약 world, ROS, OpenGL과 핀홀·어안 모델 | `camera.py` 290–299의 `convention`, `sensors_cfg.py`의 `PinholeCameraCfg`와 `FisheyeCameraCfg` |
| 3.3.3절: 레이캐스터 카메라는 RGB와 의미 분할을 아직 지원하지 않음 | `RayCasterCamera.UNSUPPORTED_TYPES`에 `rgb`, `instance_id_segmentation`, ... (`ray_caster_camera.py` 49행) |
| 3.3.1절: IMU 가속도는 유한 차분, 중력 투영 측정을 제공 | `imu.py` 177–189 (`lin_acc_w = (lin_vel_w - prev)/dt + gravity_bias`), `GRAVITY_VEC_W` |
| 3.3.1절: 접촉 센서는 짧은 이력을 보관 | `contact_sensor.py`의 `net_forces_w_history`, `track_air_time` |
| 3.4.1절: 차분 IK 방법 네 가지 | `differential_ik_cfg.py` 35행: `pinv`, `svd`, `trans`, `dls` |
| 3.4.2절: 관절 임피던스 fixed, variable, variable_kp | `joint_impedance.py` 121–181의 `impedance_mode` 분기 |
| 3.4.2절: OSC의 동역학 분리, 중력 보상, 영공간 제어 | `operational_space_cfg.py` 35–79 (`inertial_dynamics_decoupling`, `gravity_compensation`, `nullspace_control`) |
| 3.4.1절: Pink IK와 NullSpacePostureTask | `controllers/pink_ik/` (`null_space_posture_task.py`) |
| 3.4.3절: cuRobo MotionGen 계획 | `isaaclab_mimic/motion_planners/curobo/curobo_planner.py` (위치는 Mimic 확장) |
| 5.5절: SkillGen 단일 플래그 | `mimic_env_cfg.py` 76행 `use_skillgen: bool = False` |
| 5.2절: DexPBT, 8 정책 | `isaaclab_rl/rl_games/pbt/pbt_cfg.py` `num_policies = 8`, 기본 `mutation_rate` 0.25. Dexsuite 설정 `rl_games_ppo_cfg.yaml`의 `num_policies: 8` |
| 3.7.3절은 보행 로봇 11종: Unitree A1, G1, H1, Go1/2, ANYmal B/C/D, Cassie, Digit, Spot | `isaaclab_tasks/.../locomotion/` 아래 a1, anymal_b, anymal_c, anymal_d, cassie, digit, g1, go1, go2, h1, spot (11) |
| 3.7절: 매니저 기반과 직접 작성 두 워크플로 | `ManagerBasedRLEnv(ManagerBasedEnv, gym.Env)`, `DirectRLEnv(gym.Env)` |
| 3.7.3절 등 등록된 작업 이름 | `Isaac-Velocity-Rough-Anymal-C-v0`, `Isaac-Velocity-Rough-Anymal-C-Direct-v0` (`anymal_c/__init__.py`, 테스트) |
| 3.7.1절 (대략, 알고리즘 2) | 알고리즘 2의 순서: 행동 처리 → 부분 스텝 반복 → 종료와 보상 → 리셋 → 명령과 관측. 이벤트 단계 하나만 다름 (D2) |

## 3. 라이선스 (정적 확인)

- `LICENSE`: BSD-3-Clause (copyright 2022–2026 The Isaac Lab Project Developers). 직접 확인.
- `LICENSE-mimic`: Apache-2.0. 확장 `isaaclab_mimic`에 적용. `README.md` 확인.
- 의존성과 자산: `docs/licenses/` 아래 문서에 있음. Isaac Sim 라이선스와 cuRobo 라이선스는 별도 독점 조건을 가짐.
- GitHub API는 저장소 전체를 `NOASSERTION`으로 표시한다. 여러 라이선스가 섞여 있어서다.

## 4. 구현 가능성 (이 작업 환경: Apple M4 Pro, 약 24 GiB 통합 메모리, MPS, CUDA 없음)

**판정: 이 맥에서 Isaac Lab을 실행하는 것은 불가능합니다. 논문 수치 재현은 Linux와 NVIDIA GPU 서버가 있어야 합니다. 맥에서 가능한 것은 문서 대조, 수치 재계산, CPU 토이까지입니다.**

근거는 다음과 같다.

- 설치 문서(`docs/source/setup/installation/index.rst`, v2.3.0)는 **Ubuntu 22.04 또는 Windows 11**, RAM 32 GB 이상, GPU VRAM 16 GB 이상을 요구한다. macOS는 지원 목록에 없다.
- Isaac Lab의 PhysX 경로는 CUDA GPU 텐서를 쓴다. Isaac Sim은 RTX GPU와 Omniverse 드라이버가 필요하다.
- 벤치마크 장비는 L40(48 GB), RTX PRO 6000(96 GB), GeForce 5090(32 GB)이다. 맥에는 이 중 어느 것도 없다.
- v3.0.0의 Newton 백엔드(커널리스, Isaac Sim 불필요)는 Warp에 기반한다. Warp의 CPU 백엔드로 맥에서 돌 수 있는지는 확인하지 않았다. 설치 승인 전에는 판정할 수 없다.

| 작업 | 맥 | 필요한 것 | 노력 추정 (측정 전, 추정) |
|---|---|---|---|
| 논문 그림 13 처리량 재현 (단일 GPU 기준) | 불가 | Linux, RTX 계열 GPU, Isaac Sim 설치, 벤치 실행 | 설치 1–2일 + 벤치 수 시간 (추정) |
| 그림 13 다중 GPU 재현 (2, 4, 8 GPU) | 불가 | GPU 8장 서버 | 장비가 있어야 판단 |
| 그림 18 레이캐스터 스케일링 재현 | 불가 | NVIDIA GPU, Warp CUDA | 설치 후 1–2일 (추정) |
| 문서와 코드 대조 (이 문서) | 가능 | 없음 | 완료 |
| 수치 재계산과 판독 (`analysis/recompute.py`) | 가능 | numpy | 완료 |
| 액추에이터 모델의 CPU 토이 (DC 모터, 지연 PD) | 가능 (별도 작업) | numpy | 1–2일 (추정) |
| Newton 커널리스 경로 확인 | 조건부 | Warp (CPU) 설치 승인, 버전 확인 | 설치 승인 후 판단 |

## 5. 기타

- 복제본은 `isaac_lab/repo/`에 있고 gitignore한다. 실행 파일과 설치 스크립트는 읽지 않았다.
- 저장소 최신 `main`의 별 수, 마지막 push는 `02` §4에 있다.
