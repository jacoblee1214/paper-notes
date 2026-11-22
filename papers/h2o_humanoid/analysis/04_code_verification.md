# 04. 공식 코드 대조 (정적 읽기): D1–D17, 구현 가능성 판정

대상: `repo/` (LeCAR-Lab/human2humanoid, 얕은 복제, commit `750f1fa052641f0fde43669d50cb4e407dabe6c8`, 2025-02-21, "fixing eval during training"). **논문(arXiv v1 2024-03-07)보다 6.5개월 뒤(저장소 생성 2024-09-29)에 공개되었고, OmniH2O와 공용인 코드베이스입니다.** README가 "Official Implementation for H2O and OmniH2O"라고 적습니다. 그래서 논문 H2O의 학습 설정과 코드의 기본 설정이 같다는 보장이 없습니다. 실행, 설치, import를 하지 않았습니다. 모든 내용은 파일을 읽은 결과이며, 줄 번호는 이 커밋 기준입니다. 복제한 파일은 신뢰하지 않는 데이터로 다뤘습니다.

## 0. 저장소 기본 사항

- **라이선스.** README와 `LICENSE-CC-BY-NC-4.0.md`는 CC BY-NC 4.0(상업적 이용 금지)이고, `legged_gym`과 `rsl_rl`은 ETH Zurich와 NVIDIA의 상속 라이선스입니다. GitHub API의 license는 비어 있습니다. 공개 게시본에는 코드 파일을 넣지 않고 설정 값과 함수 이름만 적습니다.
- **가중치가 없습니다.** `.pt` 파일은 `resources/actuator_nets/anydrive_v3_lstm.pt` 하나이고 코드에서 참조하지 않습니다.
- **H2O 레시피는 README의 명령 한 줄입니다.** 8번 항목 "Train H2O Policy (8point tracking, no history, MLP, with linear velocity in the state space)": `env.num_observations=138 env.num_privileged_obs=215 motion.teleop_obs_version=v-teleop-extend-max motion.teleop_selected_keypoints_names=[left_ankle_link,right_ankle_link,left_shoulder_pitch_link,right_shoulder_pitch_link,left_elbow_link,right_elbow_link] motion.extend_head=False ... rewards=rewards_teleop_omnih2o_teacher rewards.penalty_curriculum=True rewards.penalty_scale=0.5 env.add_short_history=False`. 138차원은 논문의 H2O와 같습니다(`03` V2). 보상 파일 이름이 `omnih2o_teacher`라서 **OmniH2O 교사용 보상**을 H2O에 씁니다.
- **기본 모션 파일은 `stable_punch.pkl` 한 개(2.7 MB)입니다.** README의 모든 명령이 이 파일을 지정합니다. 논문의 AMASS 필터본(`amass_phc_filtered.pkl`)은 CMU Box 링크로만 있고 받지 않았습니다.
- **열린 이슈에서 본 것(외부 글, 참고).** #65 `config_teleop`이 기본 보상 `rewards_teleop`을 찾지 못해 `rewards=rewards_teleop_omnih2o_teacher`를 직접 넘겨야 함(코드에서 확인: `cfg/config_teleop.yaml` 13행이 존재하지 않는 `rewards_teleop`을 가리킴). #41 공저자가 "평가 스크립트를 제공하지 않았다"고 답함. #45 필터 구현 질문, #39 음수 마찰 질문은 답이 없음.

## 1. 구성 요소 존재 여부

| 구성 요소 (논문 위치) | 코드에 있는가 | 근거 |
|---|---|---|
| H2O 정책 학습 (§V, 표 I, II) | 있음 | `legged_gym/scripts/train_hydra.py`, `cfg/*`, README 8번 |
| 관측 138차원 (§V-A) | 있음, 일치 | `legged_robot.py` 1012–1060행 `v-teleop-extend-max` |
| 도메인 무작위화 (표 II) | 있음, 값이 다름 | `cfg/domain_rand/domain_rand_teleop.yaml` (D7) |
| 보상 (표 I) | 있음, 값과 형태가 다름 | `cfg/rewards/rewards_teleop_omnih2o_teacher.yaml` (D2, D3) |
| 형상 맞춤과 리타게팅 (§IV-A) | 있음, 설정이 다름 | `scripts/data_process/grad_fit_h1_shape.py`, `grad_fit_h1.py` (D10) |
| 특권 모방 정책 (§IV-B) | 일부 | `phc/phc/data/cfg/env/h1_im_*.yaml` (D11, D12) |
| sim-to-data 필터링 스크립트 (§IV-B) | **없음** | 실패 키를 저장하는 코드(`im_amp.py` 126–128행)와 PNN용 `scripts/pmcp/forward_pmcp.py`만 있음. 필터본을 만드는 스크립트 없음 (D11) |
| 평가 지표 Succ, Eg-mpjpe, Empjpe, Eacc, Evel (§VI-A) | **없음** | `legged_robot.py` 373행 `extras['mpjpe']` 하나뿐. 공저자도 평가 스크립트가 없다고 답함 (D13) |
| 실로봇 배포 (HybrIK → 정책 → H1) (§VI-B) | **없음** | `hardware_code/`는 손과 손목 구동, ZED 오도메트리. HybrIK는 README가 외부 저장소를 가리킴. H1 제어 루프 없음 (D14) |
| 학습된 가중치 | **없음** | §0 |

## 2. 차이 (D1–D17)

판정 표기: **큼**(결과를 바꿀 수 있음), 중간, 작음.

| 번호 | 항목 | 논문 | 코드 (파일, 줄) | 판정과 영향 |
|---|---|---|---|---|
| D1 | 코드의 정체 | H2O (arXiv 2024-03) | OmniH2O와 공용, 생성 2024-09-29. H2O 레시피는 README 8번 명령 한 줄이고 보상 파일이 `omnih2o_teacher`. 기본 설정(`config_teleop.yaml`)은 OmniH2O용이며 13행 `rewards: rewards_teleop`은 없는 파일, `train_hydra.py`와 `play_hydra.py`의 기본 설정 이름이 서로 다름(`play_hydra.py`는 `config_base`) | **큼.** 코드 결과를 논문 H2O 값과 직접 연결할 수 없음. 논문 시점의 설정이라는 증거 없음 |
| D2 | 과제 보상 커널과 가중치 | 표 I: DoF 위치 exp(−0.25‖·‖²) × 24, DoF 속도 exp(−0.25‖·‖²) × 24, 몸체 위치 exp(−0.5‖·‖²) × 40, 회전 exp(−0.1‖·‖) × 16, 선속도 exp(−10‖·‖²) × 60, 각속도 exp(−0.01‖·‖²) × 60 | `cfg/rewards/rewards_teleop_omnih2o_teacher.yaml` 78–90행과 `legged_robot.py` 3509–3560, 3636–3668, 3756–3783, 3804–3818, 3863–3878, 3926–3940행. 모든 항이 exp(−**평균** 제곱 오차 / σ). 관절 위치 32 (σ = 0.5, 관절별 가중 0.5–2.0), 관절 속도 16 (σ = 10), 몸체 위치 30 (하체 σ = 0.5 × 0.5, 상체 σ = 0.03 × 1.0), 회전 20 (σ = 0.1), 몸체 선속도 **8** (σ = 10), 각속도 **8** (σ = 10) | **큼 (속도).** 노름을 합으로 읽으면 위치 커널은 크기가 맞음(`03` V9). 선속도와 각속도 보상 가중치는 8 대 60, 커널은 17–81배 완만해 학습 신호가 거의 없음. 관절 속도 16 대 24. 합산 가중치도 164(VR 점 50 포함) 대 224 |
| D3 | 벌점과 정규화 가중치 | 표 I: 토크 한계 −0.2, 관절 위치 한계 −100, 종료 −200, DoF 가속도 −8.4e−6, DoF 속도 −3e−3, 행동 변화율 −0.9(하나), 토크 −9e−5, 발 체공 +800, 발 접촉력 −0.1·‖F‖², 넘어짐 −1000, 미끄럼 −30·‖v‖² | 같은 파일 3–23행: 토크 한계 **−2** (10배), 관절 위치 한계 −125, 종료 −250, DoF 가속도 −1.1e−5, DoF 속도 −4e−3, 행동 변화율은 **하체 −3.0과 상체 −0.625로 분리** (`_reward_lower_action_rate` 3422행, 상체 3426행), 토크 −1e−4, 발 체공 +1000, 발 접촉력 −0.75 × 500 N 초과분의 **선형** 합(`_reward_feet_contact_forces` 4056행), 넘어짐 −1250, 미끄럼 −37.5 × ‖v‖ (제곱 아님, `_reward_slippage` 4034행) | 중간. 표 I의 값과 거의 모든 항이 다름. 토크 한계 벌점 10배와 접촉력 식 형태가 가장 큼 |
| D4 | 논문에 없는 보상 항 | 표 I에 없음 | 같은 파일: `dof_vel_limits` −50, `feet_ori` −62.5 (`_reward_feet_ori` 3386행), `in_the_air` −200 (양발이 공중이면 벌점, 3287행), `orientation` −200, `feet_max_height_for_this_air` **−2500** (착지 시 최대 발 높이가 0.25 m보다 낮으면 벌점, 3985행), 모든 보상 척도에 dt(0.02)를 곱함(2926행) | **큼.** 표 I은 보상의 전부가 아님. 높이 0.25 m 이하의 걸음에 큰 벌점 |
| D5 | "VR 3점" 보상이 H2O 설정에서 다른 몸체를 가리킴 | 해당 없음(논문에 없음) | `teleop_body_position_vr_3keypoints: 50`, σ = 0.03. 구현은 확장 몸체 목록의 **마지막 3개**(`diff[:, -3:]`, 3756–3783행). H2O 명령은 `extend_head=False`라 확장 몸체가 손 2개뿐이고(`legged_robot.py` 97–105행) 마지막 3개는 오른쪽 팔꿈치와 두 손 | 중간. 머리 포함 3점 VR 보상이 H2O 설정에서는 팔꿈치와 두 손으로 바뀜. 가중치 50이 가장 큰 단일 보상 |
| D6 | 조기 종료 | §V-D: 기저 높이 < 0.3 m, 투영 중력 x, y > 0.7, **평균 링크 거리 > 0.5 m** | `cfg/asset/asset_teleop.yaml`: `terminate_by_low_height: False` (30행, 높이 종료 꺼짐), `terminate_by_gravity: True` (0.7), 추가로 pelvis, shoulder, hip, knee 접촉 종료 (8행). 기준 거리 종료는 README 명령이 `max_ref_motion_distance=1.5` (기본 5.0, 48행)로 줌. 학습 중에는 **몸체별 최댓값**(`legged_robot.py` 442행), 평가(`env.test`, `im_eval`)에서만 평균(438–439행). 밀린 뒤 60스텝은 거리 종료 해제(2238행, 444–446행) | **큼.** 기준 거리 1.5 m(논문 0.5 m의 3배), 최댓값(논문은 평균), 높이 종료 꺼짐, 접촉 종료 추가. 학습 종료 기준이 논문과 크게 다름. 논문의 "평가 성공 = 학습 종료 기준"(`03` K8)은 코드에서 성립하지 않음 |
| D7 | 도메인 무작위화 (표 II) | 마찰 U(0.2, 1.1), 밀기 0.5 m/s, 지연 U(20, 60) ms, RFI 0.1 × 한계, 링크 질량 × U(0.7, 1.3) | `cfg/domain_rand/domain_rand_teleop.yaml`: 마찰 **[−0.6, 1.2]** (10행, 음수 33%), 밀기 **1.0 m/s** (6행, `_push_robots` 2232행은 속도를 직접 덮어씀), 지연 정수 0–3 스텝(40행, 0 ms 포함), RFI 0.1에 관절별 배율 U(0.5, 1.5)(35–37행), 링크 질량은 pelvis, 고관절 6개, torso 8개만(24–27행) | 중간~큼. 마찰 음수는 PhysX 처리를 확인하지 못함(이슈 #39). 밀기는 논문의 2배 |
| D8 | 논문에 없는 무작위화와 잡음 | 표 II에 없음 | 기준 키포인트 오프셋 x, y ±0.02 m, z ±0.1 m (43행, `randomize_motion_ref_xyz`). 관측 잡음 `add_noise: True`, `noise_level: 1.0` (`cfg/noise/noise_teleop.yaml`): 관절각 0.01, 관절속도 0.1, **선속도 0.2**, 각속도 0.5, 중력 0.1, 기준 위치 0.05. H2O 명령은 `noise.add_noise=False`를 주지 않음. 벌점 curriculum(`penalty_curriculum=True`, 척도 0.5 시작, 0.25–1.0, `_update_penalty_curriculum` 2302행) | 중간. 그림 4의 "Input Noises"는 있으나 크기가 논문에 없음. 선속도 잡음 0.2 m/s는 MoCap 속도(§VI-B)를 가정하면 매우 큼 |
| D9 | 행동과 제어 | a_t ∈ R^19 관절 목표, τ = K_p(a_t − q_t) − K_d q̇_t. PD 200 Hz, 정책 50 Hz (그림 4) | `_compute_torques` 2055–2080행: 목표 = 0.25 × a + **기본 자세** (`action_scale: 0.25`, `cfg/control/control_teleop.yaml` 24행). `dt: 0.005`, `decimation: 4` → 50 Hz (일치). P 이득 hip 200, knee 300, ankle 40, torso 300, shoulder 100, elbow 100 (논문에 값 없음) | 작음. 행동 정의가 "기본 자세 기준 잔차"임. 이득 값은 논문에 없어 코드가 유일한 출처 |
| D10 | 리타게팅 설정 | §IV-A: 12개 관절로 β′를 경사 하강, 12개 관절 위치 차이를 **Adam**으로 최소화, 휴리스틱으로 앉기 등 제거. 13k → 10k | `grad_fit_h1_shape.py`: 14개 관절(고관절과 머리 포함), Adam lr 0.1, 1000회, **스케일 변수도 함께 최적화**(81–84행). `grad_fit_h1.py`: 11개 관절(고관절 제외), **Adadelta lr 100**, 500회(135–137행), 필터 휴리스틱 없음. 스크립트에 `ipdb.set_trace()`가 두 곳(172, 176행)이고 모션마다 멈춘 뒤 `test.pkl`을 덮어씀. README 로그는 15,886개 모션 | 중간. 논문 설명(12점, Adam, 필터)과 다름. 그대로는 무인 실행되지 않는 개발 중 스크립트. 15,886 대 논문 13k |
| D11 | sim-to-data 필터링 | §IV-B: 특권 정책으로 10k를 모방, 실패 시퀀스 제거 → 8.5k | 필터 스크립트 **없음**. 실패 키를 일정 주기마다 저장(`im_amp.py` 126–128행)하고 `scripts/pmcp/forward_pmcp.py`는 이를 PNN 확장용으로 씀(경로가 `output/dgx/…`에 고정, `ipdb` 호출). 결과 파일은 Box 링크(`amass_phc_filtered.pkl`)로만 제공(시퀀스 수 미확인). 후속 논문(OmniH2O)의 필터본은 14k 규모로 읽힘(직접 확인 못 함) | **큼.** 이 논문의 핵심 기여(필터)를 코드로 재현할 수 없음. 파일의 수(8.5k 대 약 14k)가 논문과 다름 |
| D12 | 특권 정책 설정 | PULSE 방식, hard-negative mining, 입력 R^778, DR 없음 | `phc/phc/data/cfg/env/h1_im_1..4.yaml`: `numTrajSamples: 3` (→ 298 + 1,440 = 1,738차원, 778이 아님, `03` V2), `hard_negative: False` (28행), 해당 갱신 코드는 주석 처리(`humanoid_im.py` 321–322행), 제어 30 Hz (`controlFrequencyInv: 2`), 종료 거리 0.25 m (58행), 환경 1,536개 | 중간~큼. 논문의 특권 정책(778차원, hard-negative)과 저장소 설정이 다름 |
| D13 | 평가 지표 | 표 III, IV: Succ, Eg-mpjpe, Empjpe, Eacc, Evel | 지표 계산 코드 없음. `extras['mpjpe']` (몸체 평균 거리, 373행)만 있고 `has_eval`은 학습 중 기록용. 공저자가 이슈 #41에서 "평가 스크립트를 제공하지 않았다"고 답함. 다른 사용자가 필터본으로 학습했으나 후속 논문 수치에 못 미쳤다고 보고(#41) | **큼.** 표 III, IV를 코드로 다시 만들 수 없음 |
| D14 | 실로봇 시스템 | §VI-B: 웹캠 + HybrIK 30 Hz → 정책 → H1, MoCap 선속도 50 Hz | HybrIK 연동, 정책 추론 루프, H1 SDK 제어, MoCap 입력이 없음. `hardware_code/zed_odometry.py`는 ZED 오도메트리(OmniH2O 계열), `scripts/ws_client.py`와 `quest_camera.py`는 VR 계열. README는 외부 저장소를 가리킴 | **큼.** 실물 파이프라인 재현 불가. `play_hydra.py`는 ONNX 내보내기만 함(154–163행) |
| D15 | 네트워크와 PPO | 논문에 없음 | MLP [512, 256, 128] ELU, **비대칭 크리틱**(관측 138 대 특권 215), PPO: 클립 0.2, γ 0.99, λ 0.95, 학습률 1e−3 적응(KL 0.01), 엔트로피 0.005, 그래디언트 노름 0.2, 에폭 5, 미니배치 4, 롤아웃 24 스텝 × 4,096 환경 (`cfg/train/ppo_teleop.yaml`, `ppo_base.yaml`) | 정보. 논문이 비대칭 크리틱을 언급하지 않음. 값은 이 저장소가 유일한 출처이고 논문 시점 값이라는 보장 없음 |
| D16 | 지형 | 평지, 거침, 낮은 장애물 (표 II) | `terrain_proportions [0.2, 0.6, 0.2, 0, 0]` (`terrain_teleop.yaml` 28행): 평지 20%, 거침 60%, 낮은 장애물 20% | 일치(비율은 논문에 없음, 코드가 출처) |
| D17 | 시작 모션 | 학습은 AMASS 필터본 | 기본이 `stable_punch.pkl` 한 시퀀스(`motion_teleop.yaml` 9행, 8행은 주석 처리된 필터본). README 명령이 모두 이 파일을 씀. 외부 사용자는 README 교사 명령으로 학습하면 평균 에피소드 길이가 10–20에 머문다고(#58), 로봇이 넘어진다고(#65) 보고 | 중간. README 명령을 그대로 돌려서는 논문의 학습 분포가 아님 |

## 3. 일치 확인 (M)

- **M1 제어 주파수.** `dt: 0.005`, `decimation: 4` → 정책 50 Hz, PD 200 Hz. 그림 4와 같습니다.
- **M2 관측 구성.** `v-teleop-extend-max`: 관절각 19, 관절속도 19, 선속도 3, 각속도 3, 투영 중력 3, 과제 관측, 이전 행동 19. 과제 관측은 `compute_imitation_observations_teleop_max`(4210행)의 (현재 대비 기준 위치 차이, 기준 위치, 기준 선속도) × 8점 × 3 = 72입니다. 합 138(`03` V2).
- **M3 8개 키포인트.** README 명령은 발목 2, 어깨 2(`shoulder_pitch_link`), 팔꿈치 2에 손 2(팔꿈치 링크에서 0.3 m 앞 점, `extend_body_pos`, 100행)를 더해 8점입니다. 논문의 "어깨, 팔꿈치, 손, 발목"과 같습니다. 손은 별도 링크가 아니라 팔꿈치에서 정한 점입니다.
- **M4 지연과 RFI 구조.** 제어 지연이 행동 큐로 구현되고(`step()` 185–200행), RFI 잡음이 토크에 더해집니다(2078–2079행).
- **M5 PD 구조와 PD 이득 무작위화 범위 ×(0.75, 1.25), CoM ±0.1 m, 링크 질량 ×(0.7, 1.3).** 논문과 같습니다.
- **M6 보상 구조.** 벌점, 정규화, 과제 세 군이고 과제 보상이 지수 커널입니다(값은 D2).

## 4. 구현 가능성 판정 ("구현할 수 있는지")

이 맥은 Apple M4 Pro, 약 24 GiB, Apple GPU(MPS), **CUDA 없음**입니다.

**논문 수치(표 III, IV)와 실물 시연의 재현은 이 맥에서 불가능합니다. 논문 식과 공식 설정의 정적 대조와 표 재계산은 끝냈습니다.**

| 경로 | 이 맥에서 | 근거 |
|---|---|---|
| 공식 코드 학습과 평가 (4096 환경) | **불가** | README 설치가 Isaac Gym Preview 4, Python 3.8, NVIDIA GPU를 전제(Linux x86_64). macOS와 Apple GPU 지원 없음. 이슈 #66은 A4000 또는 V100에서 교사 1M 반복이 3주 이상으로 추정된다고 보고함(외부 글, 참고) |
| 표 III, IV 재현 | **불가** | 평가 코드 없음(D13), 필터본은 Box에서 받아야 함, 학습 하드웨어 없음. 논문에 시드, 하이퍼파라미터, 학습 시간이 없어 단순 재현이 되더라도 수치가 맞는지 판단할 기준이 약함 |
| 실물 H1 시연 | **불가** | Unitree H1 EDU, MoCap, 웹캠, ZED, Orin NX가 필요. 배포 코드도 저장소에 없음(D14) |
| sim-to-data 필터 재구현 | **조건부** | 필터 스크립트가 없어 다시 짜야 함. 특권 정책 학습(1,536 환경 × 30 Hz)은 이 맥에서 어려움. 축소판(작은 인체 모델, MuJoCo)에서 "특권 정책으로 거르기"의 효과를 보는 실험은 가능성이 있으나 측정 전 |
| 리타게팅 스크립트 (`grad_fit_h1_*.py`) | **조건부** | PyTorch CPU 코드이지만 AMASS(약 수십 GB, 등록 필요)와 SMPL 모델 파일, `smpl_sim`과 `poselib` 의존성, `ipdb` 제거가 필요. 모션당 500회 반복, 수천 모션이라 배치 없이는 느림(미측정). 새 설치와 데이터 다운로드 승인이 필요 |
| 보상 커널 식 비교, 상태 차원 재구성, 표 재계산 (numpy) | **가능, 완료** | `analysis/recompute.py`, `03` V1–V10 |
| 보상 커널 식의 학습 영향 확인 (MuJoCo + PyTorch PPO, 소형 추적 문제) | **가능(작게)** | mujoco 4.0, torch 2.15가 이미 있어 새 설치가 필요 없음. 속도 보상 σ 변경이 학습에 영향을 주는지 소형 모델에서 볼 수 있음. 처리량은 측정하지 않음 |

**노력 추정(모두 추정, 측정 전).**
- 정적 대조: 완료(이 문서).
- 보상 커널 식을 MuJoCo 소형 모델에 옮겨 PPO로 비교하는 축소판: 3–5주.
- H1 전신 추적을 MuJoCo + PyTorch로 다시 짜서 맥에서 학습하는 것: 학습 처리량을 먼저 재야 하며, 공식 규모(4,096 환경)는 이 맥에서 현실적이지 않음. 공식 규모 재현은 CUDA 기기와 H1이 있어야 함.

## 5. 저자에게 물을 것

1. H2O 논문 실험에 쓴 보상 가중치와 σ는 표 I의 값입니까, README 8번 명령(`omnih2o_teacher`)의 값입니까?
2. 평가 성공 기준(평균 0.5 m)과 학습 종료 기준(코드는 몸체별 최댓값 1.5 m)을 어떻게 설정했습니까?
3. 표 III, IV의 시드 수와 시드 간 변동은?
4. 정리 집합(8.5k)과 공개 필터본(`amass_phc_filtered.pkl`)의 시퀀스 수와 관계는?
5. 실로봇 평가의 시도 수, 실패 사례, 지연 측정은?
6. 음수 마찰 범위는 의도입니까?
