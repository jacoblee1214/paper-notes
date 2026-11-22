# 04. 공식 코드 대조와 구현 가능성

## 0. 대상과 방법

- 저장소: `inclusionAI/Universal-Manipulation-Exoskeleton` (공식. 프로젝트 페이지 Code 링크와 README가 서로 가리킴). 얕은 복제 커밋 `e087df5cd3b281418722e155d9975695f163698e` (2026-07-15, 메시지 "update"). GitHub API 기준 전체 커밋은 3개(2026-07-16 push까지: open-source, Add Apache License 2.0, update)이고, 얕은 복제는 HEAD 한 개만 가집니다. `repo/`에 두었고(약 227 MB, 대부분 STEP·STL 3D 모델) `.gitignore`에 있습니다.
- **정적 읽기만 했습니다.** 코드를 실행, import, 설치하지 않았습니다. 데이터, 체크포인트는 저장소에 없고 받지 않았습니다.
- 읽은 파일: `README.md`, `ume/docs/00–02`, `ume/robot/ume/v6_bimanual/{solver.py, ume_leader/controller.py, openarm_follower/{solver.py, controller_real.py}}`, `ume/robot/ume/v6_imu/{solver.py, ume_leader/controller.py}`, `ume/robot/openarm1/{solver.py, teleop_leader_tuning.py}`, `ume/robot/mobile_openarm/hexfellow_base_imu_teleop_redis.py`, `ume/learning/mobile_openarm_{ume,no_torque,umi}/*`, `ume/learning/mobile_openarm/eval_openarm_controller.py`, `ume/learning/model/act/{policy.py, detr/main.py, detr/models/{detr_vae.py, transformer.py}}`, `ume/learning/video_mp4_dataset.py` 일부, MJCF 두 개의 어깨 관절 선언(`robot.xml`, `openarm.xml`).
- 판정: **일치**는 논문과 코드가 같음, **차이**는 다름 또는 논문에 없는 기능, **미확인**은 찾지 못함.

## 1. 구성과 라이선스

- 파이썬 약 8,900줄과 STEP/STL 모델. 한 파일 단위로 통신은 Redis 스트림과 `process-compose` 프로세스 묶음(`collect_data.yaml`, `teleop_openarm*.yaml`, `eval_mobile_openarm*.yaml`)입니다.
- 의존성(`conda_envs/*.yaml`): python 3.12, numpy 1.26, scipy, opencv, av, redis, pytorch/torchvision, **pinocchio 3.4**, typer, wandb 등. 하드웨어 계층은 Linux의 SocketCAN, 시리얼(dialout), `/dev/video*`를 전제로 합니다(`02_software.md`).
- **라이선스: Apache-2.0** (루트 `LICENSE.md`, GitHub API로도 확인). ACT/DETR 하위 폴더는 원래 Apache-2.0(Facebook) 파일을 포함합니다. `openarm1/models/LICENSE`는 OpenArm 모델 별도 파일입니다.
- 공개되지 않은 것: 시연 데이터, 학습된 체크포인트, 성공률 평가 스크립트, 기준선 학습 로그, X-ARM·Franka 팔로워 코드.

## 2. 논문 ↔ 코드 대조 (D 항목)

| 번호 | 항목 | 논문 | 코드 | 판정 | 영향 |
|---|---|---|---|---|---|
| D1 | 되돌림 토크의 크기 | §2.2 "transparent" 피드백. τ_exo = J_exoᵀ F (식에 이득 없음) | `ume_leader/controller.py` L154–176: 팔로워 상호작용 토크에 **SCALE = 0.5**와 아래 D2의 게이트를 곱하고, 관절 그룹별로 **±4 Nm(DM4340 계열: 어깨·팔꿈치) / ±1 Nm(DM4310 계열: 손목·그리퍼)로 클립**한 뒤 중력 보상 토크에서 뺌 | **차이 (논문에 없음)** | 사람이 느끼는 토크는 정적 가정에서 팔로워 접촉 토크의 약 0.2–0.5배이고 12 Nm(어깨)·2.4 Nm(손목)에서 포화합니다(`03` T1). 9 Nm, 3 Nm 정격 모터의 44%, 33% 수준에서 막는 안전 한계로 보이나 근거는 코드에 적혀 있지 않음 |
| D2 | 추종 오차 게이트 | 없음 | L18–19, L155–163: 팔로워의 목표–실제 자세 오차(회전 벡터 노름)로 `gain = SCALE·(tanh(10·(err − 1°)) + 1)/2`. **오차 0에서도 0.207**, 1°에서 0.25, 5°에서 0.40, 20°에서 0.499 | **차이 (논문에 없음)** | 피드백이 자세 추종 오차에 의존하는 비선형 이득입니다. 접촉 없는 정지에서도 중력·코리올리 잔차 토크의 약 21%가 손에 전달됩니다. 전달 이득이 토크 크기에 따라 바뀌어 "투명성" 정량 평가가 필요합니다 |
| D3 | 팔로워 접촉 토크 정의 | §2.2 "measured joint torques of the follower" | `controller_real.py` L165–175: `τ_interaction = τ_actual − rnea(q, q̇, 0)`. 중력과 코리올리·원심(측정 속도 사용)을 뺀 값이고 **마찰·정지마찰 보정은 없음**. 그리퍼 관절(8번째)도 포함. `τ_actual`은 Damiao 모터의 응답 프레임 토크 필드(`damiao_util.py` L138, `actual_torque_nm`)로, 12비트 양자화입니다(DM4340은 ±28 Nm 범위에서 약 0.014 Nm, DM4310은 ±10 Nm에서 약 0.005 Nm 해상도) | 일치 (세부는 코드에만) | 정책 입력 τ(D11)도 같은 값. 모터 토크 추정은 감속기 마찰과 전류 오차를 포함하므로 "force/torque sensing"이 아닌 전류 기반 추정입니다 |
| D4 | 위치 IK와 야코비안 역 | §2.2: q = f_IK(R)(닫힌 형태로 읽힘), J† = (JᵀJ)⁻¹Jᵀ, J̄ = A⁻¹Jᵀ(JA⁻¹Jᵀ)⁻¹ | `openarm_follower/solver.py` L44–75: `ik_ball_joint`는 **현재 q에서 1스텝 가우스–뉴턴**(오차 = R·log(Rᵀ R*), `pinv(J)`, step_size 0.5)이고 2 kHz 루프에서 반복. 속도는 `pinv(J)`, 토크는 `pinv(Jᵀ)`. **관성 가중(A)은 없음** | 차이 (3×3 정칙이면 같은 결과) | 정칙에서는 논문 식과 수치가 같습니다. 특이 근처에서는 SVD 의사역행렬이 조용히 잘라 내거나 크게 증폭합니다(감쇠 없음). 공식 어깨 구성(OpenArm 어깨, MJCF)의 특이점은 `03` T1 |
| D5 | 순응성의 원천: 팔로워 제어기 | §2.4 "desired positions encode joint compliance information", 정책은 위치 목표 출력 | 팔로워는 MIT 모드 **고정 PD**(kp = [240, 240, 240, 240, 24, 31, 25, 16], kd = [3, 3, 3, 3, 0.2, 0.2, 0.2, 0.2], `controller_real.py` L14–15, `eval_openarm_controller.py` L14–15)에 중력 보상 토크를 더함. 텔레오퍼레이션에서는 속도 피드포워드도 보냄(L259, 268). 평가에서는 `dq = 0` | **일치하되 논문이 말하지 않는 사실** | 순응성은 정책이 아니라 고정 kp와 위치 목표의 어긋남에서 나옵니다. 정책이 강성을 바꿀 수 없고, 목표를 접촉 면 안쪽으로 얼마나 밀어 넣는지로 힘을 정합니다. "active compliant policy"는 이 의미입니다. 수집(속도 피드포워드 있음)과 평가(`dq = 0`)의 PD 항 차이가 τ_interaction 분포를 조금 바꿀 수 있으나 측정하지 않았습니다 |
| D6 | 외골격 보상 | 코리올리, 원심, 중력, 마찰 보상(부록 C.1) | `ume_leader/controller.py` L12–16, L103–113: `rnea(q, q̇, 0)`(중력+코리올리), 마찰 `clip(1.6·계수·q̇, ±0.4/±0.1 Nm)`, **정지마찰 보상 ±0.5 Nm**(어깨·팔꿈치만, 문턱 1°–10°/s 구간) | 일치 + 정지마찰 항은 논문에 없음 | 마찰 상수는 코드에만 있음. 수치 재현에 필요 |
| D7 | 그리퍼 | 논문 방법·표에 외골격 그리퍼 서술 없음(로봇 쪽 "1-DoF gripper"만) | 외골격에도 8번째 관절(J8)이 있고 그리퍼 토크를 되돌림(`interaction_tau_R_gripper = −wrench`), 목표 그리퍼 위치 = −외골격 위치 − 1.17(오른팔), 외골격 위치 − 1.12(왼팔). 평가·텔레오퍼레이션에서 그리퍼 명령은 현재 위치 ±0.16 rad로 클립(약 2.5 Nm 주석) | **차이 (논문에 없음)** | 그리퍼 파지력도 정책이 보는 τ_interaction에 들어감. 오프셋 상수는 장치마다 보정 필요 |
| D8 | 모바일 베이스 | IMU 오일러각을 선형 매핑해 속도 (§2.3) | `hexfellow_base_imu_teleop_redis.py` L102–112, L135–141, L170: 롤/피치/요 × 배율 2, **불감대 0.05 rad**, ±1 클립 후 최대 0.5, 가속 제한 0.8(Ruckig 궤적 생성기), 1 kHz | 일치 (불감대·가속 제한은 논문에 없음) | 정책의 베이스 행동은 가속 제한 전 `desired_vel_xyt` |
| D9 | 정책 구조: CVAE | "Action Chunking with Transformers (ACT) [13]" | `mobile_openarm_ume/model.py` L112–158: `self.model(qpos, img)`가 `actions=None`으로 호출되어 **학습에서도 잠재 변수 z = 0, KL 항 없음, L1 손실만**(`F.l1_loss`). 벤더링된 `ACTPolicy`(KL 포함, `policy.py`)는 쓰이지 않음 | **차이 (논문에 없음)** | 결정론적 청크 회귀입니다. 다봉 시연의 분산은 L1 평균으로 흡수됨. ACT 원본 대비 비교 시 주의 |
| D10 | 모델 크기와 설정 | 표 3: 인코더 4층, 디코더 7층, FFN 3200, 은닉 512, 8헤드, lr 1e-4/1e-5, 배치 200, 드롭아웃 0.1 | `act_train_ume.py`는 인자 없이 `build_ACT_model_and_optimizer`를 호출하고 `detr/main.py` 기본값이 적용됨: **인코더 4, 디코더 6, FFN 2048, 은닉 256**(L36–48), 8헤드, lr 1e-4, 백본 1e-5, 드롭아웃 0.1, AdamW 가중치 감쇠 1e-4. 배치 200 일치 | **차이 (은닉·FFN·디코더 층 수)** | 트랜스포머 파라미터가 논문 설정이 코드 기본값의 약 3.7배(`03` V10, 산술 추정). 논문 설정이 CLI 인자(`parse_args`)로 넘어갔을 수는 있으나 문서화되어 있지 않고 README 명령은 인자가 없음 |
| D11 | 입력 구성과 토크 임베딩 | §2.4 "process torque data in the same way as joint positions, then append their embeddings" | `model.py` L62, L93–107, `dataset.py` L88–92: 관측은 **1 스텝**(`obs_horizon=1`), 입력 벡터 = [왼팔 q(8), 왼팔 τ_interaction(8), 오른팔 q(8), 오른팔 τ(8)] = 32차원을 **하나의 Linear(32→256)**에 통과(분리 임베딩이나 이어 붙임이 아님). 가우스 정규화(전체 데이터 평균·표준편차), 카메라 3대(양 손목, 머리) 224×224 어안 중앙 크롭. 행동 = 목표 관절 위치 16 + 베이스 속도 3 = 19 | 차이 (구현 세부) | 토크가 단일 시점 값이라 접촉의 시간 변화(충격, 점진 증가)를 정책이 못 봅니다. "순간 토크로 판단"(§3.1)과 일치하나 이력은 안 봅니다 |
| D12 | 행동 청크와 실행 | 청크 길이·주기·실행 방식 서술 없음 | `act_train_ume.py` L42–43: 행동 20스텝, 기간 2.0초(10 Hz). 평가 `eval_openarm_controller.py` L26–28, L121–140: **2.0초마다 새 청크로 교체**(`chunk_renew_every_s`), 이음 0.45초(`stitch_duration`, 선형 보간), 시간 앙상블 없음. 추론 워커는 매 루프 청크를 발행 | **차이 (논문에 없음)** | 실질 폐루프 주기가 약 2–2.5초입니다. 접촉 반응(예: 박스 뒤집기의 두 상태 전환)이 청크 안에서 개루프로 지나갈 수 있음. UMI 기준선도 같은 평가 컨트롤러를 거치므로 청크 교체 주기는 같습니다(UMI 워커는 청크 20개 자세를 현재 q에서 순차 IK로 관절 목표로 바꿈) |
| D13 | UMI 기준선 | "UMI의 입출력 형식, 전신 IK 풀이기로 관절 위치로 변환"(§3), 서론은 UMI가 충돌 제약을 IK에 의존한다고 서술 | `mobile_openarm_umi/model.py` L62–63: 관측 18차원(좌우 말단 간 상대 자세 rot6d+pos3d), 행동 23차원(베이스 3, 좌 rot6d+pos3d, 우 rot6d+pos3d, 그리퍼 2). IK(`openarm_ik_solver.py`)는 **감쇠 최소제곱 100회, 스텝 상한 0.05, 관절 한계만**, 현재 q에서 시작. **충돌 제약, 자세 규제(영공간), 장애물 정보가 없음** | **차이 (기준선 설계)** | UMI의 0/20 결과(GPU, 냉장고, 뒤집기)에는 "충돌 회피 없는 IK + 말단 자세 정책"의 영향이 섞여 있습니다. 논문의 "UMI-style은 IK로 충돌을 처리한다"는 서술과 달리 이 구현은 IK 충돌 처리가 없어 UMI 방식 일반의 한계로 읽기 어렵습니다 |
| D14 | 범용 재타깃팅의 구현 범위 | OpenArm, X-ARM(실물), Franka(시뮬)(§2.2, 부록 C.2) | 공개 코드는 **OpenArm 쌍(`OpenArm313Solver`, "3-1-3")만**. X-ARM·Franka 코드, 설정, 모델 없음(README의 감사 인사에만 X-ARM). `ume/sim/mujoco_env.py`는 일반 MuJoCo 환경 | **미확인 (코드 없음)** | "universal"은 구조 주장이며 공개 코드로는 OpenArm에서만 검증 가능. 다른 로봇은 `ball_joint_q_slice`와 프레임 이름을 새로 정의해야 함 |
| D15 | 재현 환경 | 한 장 RTX 4090, 40k 스텝, 8시간(부록 D) | 학습 스크립트는 `wandb.init`를 무조건 호출(오프라인 줄은 주석), `.cuda()`와 `to_cuda` 하드코딩, 데이터로더 `num_workers=24`, 학습 루프는 **정지 조건 없는 `while True`**(체크포인트를 5,000 스텝마다 저장, 추론 기본 경로는 `ckpt_step_40000`) | 일치 (40k는 추론 기본 경로에서 간접 확인) | 학습 길이는 수동으로 끊어야 하고, 학습률 스케줄이 없습니다(상수 lr). 논문에 스케줄 서술 없음과 일치 |

참고(경미): `model_inference_worker.py` L17 주석은 "desired joints"를 관측으로 쓴다고 적지만 실제 코드는 τ_interaction을 씁니다(오래된 주석). 평가 때 τ_interaction은 `eval_openarm_controller.py`가 같은 정의(실제 토크 − rnea)로 계산해 `openarm_controller_state:*`로 발행하고, 학습은 수집 때 기록된 `openarm_follower_controller_state:*`를 읽습니다. 이름은 다르지만 정의는 같아 보입니다(L68–78 대 `controller_real.py` L165–175, 그리퍼 항 0 포함).

## 3. 일치한 것

- 외골격 구조와 부품: 전 구간 8관절(어깨 3, 팔꿈치 1, 손목 3, 그리퍼 1), Damiao 4340×4/4310×4 팔당, STEP·STL 파일, 모터 설정 절차(`01_hardware.md`)는 표 4의 부품 수와 맞습니다.
- 구면 어깨·손목을 야코비안으로 분해하는 구조(속도 `pinv(J) ω`, 토크 `pinv(Jᵀ) τ`, 위치 회전 오차 IK)는 논문 §2.2 설명과 같은 형태입니다. 팔꿈치는 직접 거울 대응(왼팔은 부호 반전)입니다.
- ACT 계열 트랜스포머(DETR 기반 인코더·디코더, ResNet18 백본, 학습 가능한 질의 20개)와 No-torque 기준선(입력 32→16차원, 토크 두 항 삭제)은 논문의 서술과 맞습니다. 다른 것은 위 D9–D11입니다.

## 4. 구현 가능성 판정

**이 맥(Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)에서 실물 재현은 불가능하고, 소프트웨어 파이프라인 일부는 조건부로 가능합니다. 논문의 성공률 표는 어떤 방식으로도 재현할 수 없습니다.**

| 수준 | 판정 | 근거 |
|---|---|---|
| 공식 코드 정적 대조, 보고 수치 재계산 | **가능 (완료)** | 이 문서와 `03` |
| 되돌림 이득 사슬·어깨 야코비안 사상 토이 | **가능 (완료)** | `analysis/toy_retarget_torque.py`, numpy, 약 30초 |
| 외골격·팔로워 텔레오퍼레이션 (표 1의 데이터 수집 전부) | **불가** | UME 하드웨어(모터 16개, 3D 프린트), OpenArm 양팔 + 모바일 베이스, CAN-FD 어댑터, Linux 실시간 제어. 코드가 하드웨어 레이어를 전제함 |
| 정책 학습 (표 1, 40k 스텝, 배치 200, 카메라 3대 224×224) | **불가에 가까움 (데이터 없음, GPU 전제)** | 시연 데이터가 공개되지 않았고 학습 스크립트가 CUDA 하드코딩, 논문은 4090 8시간. 맥에서 MPS 학습 시간은 측정하지 않았음. 같은 구조를 합성 데이터로 CPU/MPS에서 돌리는 축소 재구현은 가능 |
| 학습된 정책의 로봇 평가 (표 1 성공률) | **불가** | 체크포인트와 평가 환경(실물)이 없음 |
| 시뮬레이션에서 재타깃팅 검증 | **조건부** | MJCF 두 개는 공개됨. 코드가 pinocchio 3.4를 쓰므로 맥에 새 설치 필요(설치 전 확인). mujoco 4.0은 이미 설치되어 있어 MJCF 직접 로딩은 가능하나 외골격 측 입력(실제 사람)이 없음 |
| X-ARM/Franka 재타깃팅 | **불가 (코드 없음, D14)** | 새로 구현해야 함 |
| 되돌림 이득(D1, D2)을 사람 실험으로 측정 | **불가** | 장치와 사용자 필요 |

**노력 추정(측정 없음).**

- 축소 재구현(합성 접촉 과제에서 ACT에 토크 입력 유무를 비교, numpy/torch, CPU·MPS): 3–5일.
- MJCF 두 개로 어깨 사상 검증을 시뮬레이션에 연결(관절 한계 포함): 2–4일. pinocchio 설치 승인 후.
- 실물 하드웨어 재현(부품 조달, 3D 프린트, 모터 설정, 영점, 팔로워 조립): 수 주(저자 문서 기준, 조달 시간 제외하고도).
- 성공률 재현: 위 하드웨어 + 시연 수집(시연 26–157개 × 과제 4) + GPU 학습 + 평가 20회 × 방법 3 × 과제 4. 이 저장소의 제약(실로봇·GPU 불가)으로는 하지 않습니다.

## 5. 재실행

이 문서의 검사는 파일 읽기로 끝납니다. 공식 저장소를 다시 얻으려면 다음 명령을 쓰되 실행하지는 마십시오.

```bash
git clone --depth 1 https://github.com/inclusionAI/Universal-Manipulation-Exoskeleton.git repo
```
