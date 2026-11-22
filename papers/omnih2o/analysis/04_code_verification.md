# 04. 공식 코드 대조 (정적 읽기): D1–D14

대상: `repo/` (LeCAR-Lab/human2humanoid, commit `750f1fa`, 2025-02-21). **논문 결과가 나온 시점의 코드가 아닙니다.** 논문은 2024-06 arXiv v1, 학회본은 2024-10경입니다. 저장소 최종 수정은 2025-02이므로 설정 값이 논문 실험 때와 다를 수 있습니다. 실행, 설치, import를 하지 않았습니다. 모든 내용은 파일을 읽은 결과이며 파일 경로와 줄 번호를 붙였습니다. 줄 번호는 이 커밋 기준입니다.

## 0. 저장소 기본 사항

- 라이선스 표기가 셋으로 엇갈립니다. README는 "CC BY-NC 4.0, 상업적 이용 금지, Legged Gym과 RSL RL은 ETH Zurich와 NVIDIA의 상속 라이선스"라고 적고, `LICENSE-CC-BY-NC-4.0.md`가 있습니다. `rsl_rl/LICENSE`도 따로 있습니다. GitHub API의 `license`는 null입니다. 논문 공개 시에는 코드 파일을 넣지 않습니다. 이 분석은 코드 파일 내용을 인용하지 않고 설정 값과 함수 이름만 적습니다.
- 저장소에 정책 가중치(`.pt`)가 없습니다. 있는 `.pt`는 `resources/actuator_nets/anydrive_v3_lstm.pt` 하나이고, 코드에서 참조되지 않습니다.
- 저장소 안의 모션 파일은 `legged_gym/resources/motions/h1/stable_punch.pkl` 하나입니다. README의 학습·재생 명령 12개가 모두 이 파일을 `motion.motion_file`로 지정합니다. 논문의 AMASS 필터본(`amass_phc_filtered.pkl`, 14k 시퀀스)은 README 데이터 절에서만 언급되고, 받는 링크는 CMU Box입니다(받지 않음). 그래서 README 명령만으로는 논문의 학습 분포를 재현할 수 없습니다.

## 1. 구성 요소 존재 여부

| 구성 요소 (논문 위치) | 코드에 있는가 | 근거 |
|---|---|---|
| 교사 정책 학습 (§3, 표 4) | 있음 | `legged_gym/legged_gym/cfg/train/ppo_teleop.yaml`, README 1번 |
| 학생 DAgger 증류 (§3, 표 18) | 있음 | `rsl_rl/rsl_rl/algorithms/ppo.py` "Teacher student training" 분기 (약 342–352행), README 학생 명령 `train.distill=True`, `train.dagger.dagger_only=True` |
| 25스텝 히스토리 관측 (표 6) | 있음 | README 학생 명령 `env.add_short_history=True`, `env.short_history_length=25`. 관측 함수 `envs/base/legged_robot.py` (약 1237–1351행) |
| 보상 (표 15) | 있음 | `cfg/rewards/rewards_teleop_omnih2o_teacher.yaml`, 보상 함수 `envs/base/legged_robot.py` (약 3287–4060행) |
| 도메인 무작위화 (표 16) | 있음 | `cfg/domain_rand/domain_rand_teleop.yaml` |
| H1 자산 | 있음 | `legged_gym/resources/robots/` (78 MB) |
| 리타게팅 (§3, 그림 2) | 있음 | `scripts/data_process/*`, `phc/` |
| 실로봇 드라이버 | 일부 | `hardware_code/damiao_mvp/`, `damiao_wrist.cpp`, `inspire_hand.cpp`, `zed_odometry.py`. 정책 추론 루프의 전체 흐름은 읽지 않았습니다 |
| VR·카메라 입력 | 일부 | `scripts/ws_client.py`, `scripts/quest_camera.py`. 내용은 읽지 않았습니다 |
| LfD 학습 (확산 정책, 작업별 데이터) (§4.3, 표 3, 17) | **없음** | 문자열 검색에서 확산 정책 구현이 없습니다. README는 Diffusion Policy 저장소(real-stanford)를 가리킵니다 |
| GPT-4o 연동과 프롬프트 (§4.3, 부록 M) | **없음** | "openai", "gpt" 검색 결과는 `phc/`의 무관한 문자열뿐입니다 |
| OmniH2O-6 데이터셋과 로더 | **없음** | 저장소에 링크와 로더가 없습니다 |
| 학습된 정책 가중치 | **없음** | §0 |

## 2. 차이 (D1–D14)

| 번호 | 항목 | 논문 | 코드 (파일, 줄) | 판정과 영향 |
|---|---|---|---|---|
| D1 | 추적 보상 커널 | 식: exp(−0.25‖·‖²) (관절 위치), exp(−10‖·‖²) (속도), exp(−0.5‖·‖²) (몸체 위치, VR 점), exp(−0.01‖·‖²) (각속도), exp(−0.1‖θ⊖θ̂‖) (회전) (§부록 E 표 15 아래) | `envs/base/legged_robot.py` 3483–3591, 3756–3777, 3804–3982. 모든 항이 exp(−mean(err²)/σ)이고 σ는 `cfg/rewards/rewards_teleop_omnih2o_teacher.yaml`에서 옴. 관절 위치 σ = 0.5, VR 3점 위치 σ = 0.03, 몸체 위치 σ = 0.5 (하체) / 0.03 (상체), 회전 σ = 0.1, 속도 σ = 10 | **불일치 (큼).** 논문의 계수를 σ로 바꾸면 관절 위치는 σ = 4, VR 점은 σ = 2입니다. 코드의 VR 점 σ = 0.03은 논문보다 약 67배 엄격합니다. 가중치(표 15의 32, 16, 30, 50, 20, 8, 8)는 일치합니다. 어느 식이 실험에 쓰였는지 학습 로그 없이는 판단할 수 없습니다 |
| D2 | 최대 발 높이 보상 | +1000 × max{h_max − 0.25, 0} (보상, 표 15). 학회본은 "더 높은 걸음을 보상"이라고 설명 | `_reward_feet_max_height_for_this_air` (3985–4011행): clamp_min(0.25 − h_max, 0)의 합에 가중치 **−2500**. `cfg` 값 `feet_max_height_for_this_air: -2500`, `desired_feet_max_height_for_this_air: 0.25` | **방향과 형태가 다름.** 코드는 0.25 m보다 낮은 걸음에 벌점을 줍니다. 0.25 m 위의 높이에는 보상이 없습니다. 논문 설명과 반대입니다. 제목에서 말한 "큰 걸음 보상"의 구현이 아닙니다 |
| D3 | 넘어짐(stumble) 가중치 | −0.00125 (표 15) | `cfg`: `stumble: -1000.0*1.25` = **−1250**. 보상은 불리언 `any(‖F_xy‖ > 5|F_z|)` | **10^6 차이.** 논문 값이 오기(typo)일 가능성이 높지만 확인 못함 |
| D4 | 발 접촉력 항 | −0.75 × ‖F_feet‖²₂ (표 15) | `_reward_feet_contact_forces` (4056행): −0.75 × Σ max(‖F‖ − 500, 0). `max_contact_force: 500` | **형태 다름.** 제곱 대신 500 N 초과분의 선형 벌점입니다 |
| D5 | 미끄럼(slippage) | ‖v_feet‖²₂ × 1(F ≥ 1) (표 15) | `_reward_slippage` (4034행): norm(v) (제곱 아님) × 1(‖F‖ > 1), 가중치 −37.5 | 작음. 제곱과 노름의 차이 |
| D6 | 체공 시간 보상 | T_air − 0.25, 가중치 1000 (표 15) | `_reward_feet_air_time_teleop` (4013–4033행): 참조 루트 xy 속도 > 0.1일 때만 보상 | 논문에 없는 조건. 정지 명령에서 보상이 0이 됩니다 |
| D7 | 마찰 무작위화 | U(0.2, 1.1) (표 16) | `domain_rand_teleop.yaml`: `friction_range: [-0.6, 1.2]`. `_process_rigid_shape_props` (약 1906–1915행) 64개 버킷에서 값을 그대로 마찰 계수로 넣음 | **범위 불일치.** 음수 계수가 들어갑니다. PhysX가 어떻게 처리하는지 확인하지 못했습니다 (`U1`) |
| D8 | 토크 노이즈 | 0.1 × 토크 한계 (고정, 표 16) | `rfi_lim: 0.1`, `randomize_rfi_lim: True`, `rfi_lim_range: [0.5, 1.5]` (관절별 배율) | 논문보다 넓은 무작위화입니다 |
| D9 | 제어 지연 | U(20, 60) ms (표 16) | `ctrl_delay_step_range: [0, 3]`, 정책 스텝 단위 (`step()`의 action_queue, 약 185–200행). 정책 스텝 20 ms → 0, 20, 40, 60 ms | 하한이 0 ms입니다. 논문은 20 ms부터입니다. 작음 |
| D10 | 기준 자세 오프셋 단위 | [−0.02, 0.02], [−0.02, 0.02], [−0.1, 0.1] "cm" (표 16) | `motion_ref_xyz_range: [[-0.02,0.02],[-0.02,0.02],[-0.1,0.1]]` (단위는 m으로 보임) | 논문 표기 오류로 보입니다 (2 cm 대 0.2 mm) |
| D11 | 링크 질량 무작위화 범위 | U(0.7, 1.3) × 기본값 (모든 링크로 읽힘, 표 16) | `randomize_link_body_names`: pelvis, 고관절 6개, torso 8개만 | 범위 명시 누락 |
| D12 | 보상 curriculum | 곱셈 0.9999 / 1.0001, 초기 0.5, 상한 1, 40 / 120 기준 (부록 E) | `level_degree: 0.00001` (1e-5), `penalty_scale_range: [0.25, 1.0]`, 임계 50 / 120 (`_update_penalty_curriculum` 2302–2307행). 초기 0.5는 README 명령이 넘김 | **곱셈 계수가 10배 작음** (1e-5 대 1e-4). 하한 0.25는 논문에 없습니다. 임계 40은 코드에서 50입니다 |
| D13 | DAgger 손실 형태 | L = ‖a_privileged − a‖²₂ (§3 "Policy Distillation") | `ppo.py`의 kin_loss = mean(‖pred − gt‖₂) (약 352행). **노름이고 제곱이 아닙니다** | 형태 불일치 (작음~중간). 최적해는 같은 방향이지만 그래디언트 크기가 다릅니다. 경로는 일치 (아래 V6) |
| D14 | PPO 배치 | 배치 크기 64 (표 18) | `ppo_base.yaml` `num_mini_batches: 4`, `num_steps_per_env: 24`, `num_envs: 4096` → 미니배치 약 24,576 샘플. 추가로 `desired_kl: 0.01` (적응 학습률), `lam: 0.95` | 표 18의 "배치 64"를 코드 항목과 대응시킬 수 없습니다. 논문의 정의를 확인해야 합니다. 적응 학습률은 논문에 없습니다 |

## 3. 일치 확인 (V 항목)

- **V1 제어 주파수.** 시뮬 dt 0.005 s (200 Hz), decimation 4 → 정책 50 Hz (`legged_robot_config.py` 71, 245행). 논문 그림 3과 부록 A와 일치합니다.
- **V2 관측 차원.** 교사 913 (`ppo_teleop.yaml`, README), 학생 1665 = 90 + 63×25 (README `env.num_observations=1665`, `add_short_history=True`, `short_history_length=25`). 표 4, 6과 일치합니다. 학생 목표는 선속도 없는 `v-teleop-extend-vr-max-nolinvel` 관측입니다 (V7).
- **V3 PPO 하이퍼파라미터.** γ 0.99, clip 0.2, 엔트로피 0.005, 학습률 1e-3, grad norm 0.2, 에포크 5, MLP [512, 256, 128] (`ppo_teleop.yaml`). 표 18과 일치합니다. DAgger 초기 노이즈 0.001 (README `train.policy.init_noise_std=0.001`)도 일치합니다.
- **V4 보상 가중치.** 표 15의 가중치 중 다음은 코드와 같습니다. 토크 −0.0001, 토크 한계 −2, 관절 가속 −0.000011, 관절 속도 −0.004, 하체 행동 변화율 −3, 상체 −0.625, 관절 위치 한계 −125 (= −100 × 1.25), 관절 속도 한계 −50, 종료 −250 (= −200 × 1.25), 미끄럼 −37.5, 발 방향 −62.5, 공중 −200, 자세 −200, 체공 1000. 추적 항의 가중치 32, 16, 30, 50, 20, 8, 8도 같습니다. 불일치는 D1–D4입니다.
- **V5 무작위화.** PD 이득 0.75–1.25, CoM ±0.1 m, 링크 질량 0.7–1.3, 5 s 주기 1 m/s 밀기 (`domain_rand_teleop.yaml`). 표 16과 일치합니다.
- **V6 증류 경로.** README 학생 명령은 `train.distill=True`, `train.dagger.dagger_only=True`, 교사 체크포인트 로드를 씁니다. 그래서 RL 손실 없이 지도 손실만 씁니다. 논문 §3과 일치합니다. 기본 설정(`ppo_teleop.yaml`의 `distill: False`, `dagger_only: False`)은 RL과 증류 손실을 섞으므로 README를 따라야 합니다.
- **V7 선속도 제외.** 학생 관측 `v-teleop-extend-vr-max-nolinvel`에 선속도가 없습니다 (§4.1.2의 주장과 일치).
- **V8 하드웨어 항목.** Damiao 모터 드라이버, Inspire 손 드라이버, ZED 오도메트리 파일이 있습니다 (부록 A와 일치).

## 4. 확인하지 못한 항목 (U)

- **U1** 마찰 무작위화에서 음수 계수가 PhysX에 어떻게 들어가는지 (D7).
- **U2** 학생 관측의 3점 목표가 머리·양손 3점으로 구현됐는지. `v-teleop-extend-vr-max-nolinvel` 함수는 읽었지만 목표 텐서 조립을 끝까지 따라가지 않았습니다.
- **U3** 실로봇 정책 추론 루프(50 Hz, ROS, 지연 20 ms)와 VR 입력 변환.
- **U4** 논문 표 2 실로봇 실험의 설정(20 시퀀스, 어떤 체크포인트, 어떤 명령)과 저장소 설정의 대응.
- **U5** 논문 학회본의 "Oral"과 저장소 README의 명령 예시(단일 모션 `stable_punch.pkl`)가 논문 실험의 설정과 같은지.

## 5. 구현 가능성 판정 (코드 기준)

**이 맥(Apple M4 Pro, 24 GiB, MPS, CUDA 없음)에서 공식 코드로 학습하거나 재현하는 것은 불가능합니다.**

- 시뮬레이터: Isaac Gym Preview 4 (NVIDIA 전용, Linux와 CUDA 필요). macOS에서 실행할 수 없습니다. 이 맥에는 CUDA도 없습니다.
- 학습 규모: 환경 4096개, 교사 학습 10⁷ 반복 상한 (`max_iterations: 10000000`), 학생 증류도 GPU 전제입니다. 공식 README가 `cuda:0`을 고정합니다.
- 실로봇: Unitree H1, Damiao 모터, Inspire 손, Vision Pro 또는 VR. 이 저장소 밖의 장비입니다.
- 공식 코드에 없는 것: LfD 학습 코드, GPT-4o 연동, OmniH2O-6 로더, 학습 가중치. 논문 표 3과 4.3절은 코드만으로 재현할 수 없습니다.

**맥에서 할 수 있는 것:**
- 보상 함수의 수식과 설정 값 비교(D1–D4)는 numpy로 가능합니다. 실행 없이 식만 옮깁니다.
- 관측 차원 계산(V2), 표 재계산(`03`)은 완료했습니다.
- 교사-학생 증류의 최소 구현(작은 PyTorch MLP와 간단한 시뮬)은 가능하지만, 논문 설정과 같은 결과를 내기 어렵습니다. 공식 코드의 MPS 포팅은 2–4주로 추정합니다(측정 전, 추정치).

**노력 추정:** (a) 공식 코드를 정적으로 검토하고 차이를 기록하는 작업은 완료했습니다. (b) 맥에서 보상 식 비교 스크립트 작성은 0.5–1일. (c) 학생 증류 구조를 PyTorch로 옮긴 최소 재구현은 2–4주(추정). (d) 논문 수치 재현은 GPU 클러스터와 H1 실로봇 없이는 불가능합니다.
