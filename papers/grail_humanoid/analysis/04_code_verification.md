# 04. 논문 ↔ 공식 코드 대조와 구현 가능성: GRAIL (arXiv:2606.05160 v2)

## 0. 대상 코드

- **저장소:** https://github.com/NVlabs/GRAIL, 클론 커밋 `7d1a0442e79add501b16382693fb473c43f58d08` (2026-11-20, "Release SOMA reconstruction and G1 tracking examples"). `repo/`에 얕은 클론(depth 1)으로 받았고, 정적으로만 읽었습니다. 실행, 설치, import는 하지 않았습니다.
- **공식성:** 프로젝트 페이지(research.nvidia.com/labs/dair/grail/)와 arXiv 초록 옆 "Code" 링크가 이 저장소를 가리킵니다. 저장소 README의 뉴스에 "2026-06-03 Code release"가 있고, 이는 arXiv v1 제출일과 같습니다.
- **라이선스:** NVIDIA License (`LICENSE` 42–45행). 연구와 평가 목적("for research or evaluation purposes only")의 비상업 사용만 허용합니다. GitHub는 "Other"로 표시합니다.
- **규모:** `grail/` 패키지 파이썬 파일 238개, `tests/` 테스트 파일 29개. 저장소 전체 907 MB(에셋과 블렌더 파일 포함). 서브모듈은 없고 `imports/`에 사본이 직접 들어 있습니다(`imports/SONIC`, `imports/SONIC-manipulation`, `imports/agentic-retargeter`). 나머지 `imports/GEM-X`, `imports/FoundationPose`, `imports/GMR`, `imports/MoSE3`, `imports/MoGe`, `imports/soma-retargeter`, `imports/Sana`, `imports/Hunyuan3D-2.1`은 서브모듈 빈 디렉터리입니다.
- **주의:** 클론에 있는 `AGENTS.md`, `CLAUDE.md`, `.agent/`, README의 Codex 실행 안내는 저장소 작성자용 지시입니다. 이 분석의 지시로 따르지 않았고, 내용은 데이터로만 읽었습니다.

**공식 README가 적은 현재 상태 (2026-11-21 기준):** 2026-11-20에 SOMA 몸 모델 기반 에이전트 재구성과 G1 리타게팅 워크플로를 추가했습니다. 2026-08-25에 MiniMax-H3 영상 백엔드를, 2026-07-15에 추적 체크포인트를 추가했습니다. TODO에는 "quick-start demo script"가 열려 있습니다.

## 1. 일치하는 부분 (확인된 것)

| 항목 | 논문 | 공식 코드 | 근거 |
|---|---|---|---|
| 전체 흐름 | 3D 구성 → 영상 생성 → 4D 복원 → 리타게팅 → 추적 → 시각 정책 | 같은 단계가 `grail/pipelines/`(gen_2dhoi, recon_4dhoi, retargeting, tracking)에 있음 | `grail/pipelines/` 목록 |
| 공동 최적화 구조 | 초기 추정 후 잔차 파라미터로 공동 최적화 | `grail/optimization/hoi_optimizer.py`, `loss_computer.py`, 단계별 `opt_stage_specs` | `configs/recon_4dhoi/manip_soma.yaml` |
| 발 미끄럼 억제 | L_foot (GENMO 발 접촉 라벨) | `human_foot_contact`, `label_source: gem` | `manip_soma.yaml` |
| 골반 속도 정합 | L_vel (GENMO 전역 속도) | 설정 항목에 `root_speed_weight`, `joint_velocity_weight` | `manip_soma.yaml` (`human_traj_reg` 계열) |
| 어댑터 출력 차원 | 64차원 잔차 + 2차원 이진 손 원시 동작 (식 6, B.1) | `fsq_tokens.py`에 `hand_action_dim: 2`, `use_finger_primitive: True` | `imports/SONIC-manipulation/gear_sonic/utils/fsq_tokens.py` |
| PPO 설정 | actor lr 2×10⁻⁵, critic lr 10⁻³, 4 mini-batch (B.1) | 미세조정 설정 `actor_learning_rate: 2e-5`, `critic_learning_rate: 1e-3`, `num_mini_batches: 4` | `imports/SONIC/gear_sonic/config/exp/manager/universal_token/hoi/advanced_manip_ground_unified_contactfirst_finetune.yaml` |
| 워밍스타트 | 기존 정책에서 미세조정 (B.3) | `WARMSTART=.../manip_warmstart/last.pt` | `scripts/tracking/manipulation_finetune.sh` |
| 재구성 물체 추적 | FoundationPose 기반, SAM2 확인 | FoundationPose 어댑터, SAM2 어댑터 존재 | `grail/adapters/foundation_pose.py`, `sam.py` |

## 2. 차이 (D 항목)

D 번호는 이 문서 안에서만 씁니다. 각 항목은 "논문이 말한 것 → 공식 코드에서 보인 것 → 판단"으로 적습니다. 논문 재현 결과가 달라지는 원인이 될 수 있는 항목에 **(중요)**를 붙였습니다.

### 복원 (4D HOI)

- **D1 (중요). 사람 포즈 추정기.** 논문: GENMO가 카메라 공간 SMPL-X 포즈를 냄(§3.2.1). 코드: `configs/recon_4dhoi/*.yaml`의 `human_pose_backend: gem_x`, `body_model: soma03`, GEM-X native77. 몸 모델이 SMPL-X에서 SOMA(77 관절)로 바뀌었습니다. 발표 시점(v1)과 이후 릴리스 사이의 차이로 보입니다. 릴리스 데이터 카드도 pickup, sitting 등에는 SMPL-X, manip_ground와 manip_table에는 SOMA를 씁니다. 판단: 논문 수치는 GENMO+SMPL-X 경로에서 나왔고, 현재 공식 경로로는 같은 수치를 기대할 수 없습니다.
- **D2. 손 추정.** 논문: WiLoR로 MANO를 얻고, 빠진 프레임은 선형 보간과 Savitzky-Golay 필터, 손목 IK로 SMPL-X에 붙임(§3.2.1). 코드: 설정 파일에 WiLoR 항목이 없고, 설치 문서에 `WiLoR-mini` 설치 한 줄이 있습니다(`docs/source/installation.md`). Savitzky-Golay는 물체 궤적 후처리에서 확인됩니다(`grail/pose_est/utils.py`, `object_initialization.py`의 `savgol`). 판단: 손이 GEM-X에서 나오는지 WiLoR에서 나오는지는 확인하지 못했습니다(확인 필요).
- **D3. 물체 추적.** 논문: FoundationPose를 합성 데이터로 5 epoch 미세조정하고, 깊이 채널을 0으로 둠(§3.2.1). 코드: `object_tracking_backend: mose3_fp_hybrid` (MoSE3과 FoundationPose 혼합, `configs/object_tracking/mose3_fp_hybrid.yaml`). 지형(정적 물체) 시퀀스는 FoundationPose를 건너뜀(`loco_soma.yaml`). 미세조정 epoch 설정은 저장소에서 찾지 못했습니다. 판단: 논문의 추적기는 공식 혼합 경로와 다르고, 미세조정 방법은 공개되지 않았습니다.
- **D4 (중요). 정규화 항목.** 논문: L_reg = L_foot + L_vel + L_smooth, 3항(§3.2.2). 코드: 설정의 손실 항목이 훨씬 많습니다. `human_foot_contact`(30), `human_foot_ground`(300, 발 아래 평면 제약), `human_foot_ground_temporal`(1), `human_smoothness`(30), `human_pose_reg`(100), `human_global_init_reg`(3000), `verts_tracking`(0.3), `obj_smoothness`(50) 등입니다. 판단: 논문의 3항 정의는 코드의 일부만 옮긴 것입니다.
- **D5 (중요). λ 값.** 논문: 식 1의 λ_kp, λ_proj, λ_depth, λ_cont, λ_reg에 값이 없습니다(`01` §2.2). 코드: 단계별 yaml에 가중치가 있습니다. 키포인트 0.3, 깊이 점군 100, 접촉(2단계) 30,000, 물체 평활 50 등입니다. 판단: 논문 재현에는 이 yaml 값이 필요하지만 논문은 참조하지 않습니다.
- **D6 (중요). 물체 투영 손실 L_proj (식 3).** 논문: 최적화된 물체 꼭짓점의 투영이 초기 추적 투영에서 벗어나지 않게 함. 코드: 손실 레지스트리(`grail/optimization/loss_computer.py`의 `_LOSS_FN`)에 대응하는 명시적 항목이 없습니다. `obj_traj_reg`, `obj_rot_reg`, `obj_precontact_reg`는 있지만 투영 정합과 같지 않습니다. 판단: 공식 경로에서 L_proj가 같은 역할을 하는지 확인하지 못했습니다. Table 8의 "w/o L_proj" 결과를 공식 코드로 재현하기 어렵다는 뜻입니다.
- **D7 (중요). 접촉 손실 (식 5).** 논문: VLM 접촉 라벨 구간에 깊이 방향 Chamfer(CD_z)만 적용(§3.2.2). 코드: `contact`는 `phase_contact_map` 모드로 손과 물체 영역의 부분 질량 비율, 쌍과 덮개 온도 파라미터, 법선 코사인을 씁니다(`manip_soma.yaml`). 2단계 설정의 주석은 "Stage 2 uses depth for review only; Stage 1 retains its depth loss"라고 적었습니다. 판단: 논문의 CD_z 접촉은 공식 코드의 1단계 깊이 점군 손실(`depth_pointcloud`)과 2단계 접촉 맵 손실로 나뉘어 있고, 같은 식이 아닙니다.
- **D8. 깊이 정렬.** 논문: MoGe-2로 깊이를 추정하고 배경 깊이로 메트릭 정렬(§3.2.2). 코드: `grail/adapters/moge_depth_worker.py`, `grail/core/depth_config.py`가 있고 MoGe를 씁니다. 세부 정렬 알고리즘은 확인하지 못했습니다.

### 리타게팅과 장면

- **D9. 리타게팅 도구.** 논문: GMR로 SMPL-X를 G1 관절 공간에 옮김(§3.3). 코드: 주 경로는 NVIDIA SOMA Retargeter입니다(`grail/retargeting/soma.py`, "pinned public NVIDIA SOMA Retargeter"). GMR은 G1 MJCF 자산 경로(`imports/GMR/assets/unitree_g1/g1_mocap_29dof_with_hands.xml`)로만 쓰입니다. 판단: 논문의 리타게팅 결과는 GMR 기반이고, 공식 예제는 SOMA 기반입니다.
- **D10. 3D 장면.** 논문: Infinigen으로 두 장면 템플릿 생성(A.1, 그림 6). 코드: "Infinigen" 문자열이 저장소 어디에도 없습니다. 대신 `data/Scene/`의 `indoor1.blend`, `indoor2.blend`, `indoor2_notable.blend`, `long_table.obj`를 씁니다. 판단: 장면 생성 파이프라인이 논문과 다릅니다. 장면 템플릿 목록은 `configs/scenes/default.yaml`에 있습니다(확인 필요).
- **D11. 영상 생성기.** 논문: Kling 2.5 Turbo Pro, 1920×1080, 24fps, 5초(A.1). 코드: 기본 `kling-v3` (`grail/adapters/kling.py`, `grail/pipelines/gen_2dhoi.py`). 2026-08-25부터 MiniMax-H3 백엔드가 추가되었습니다. 영상 API 선택지는 `kling-ai` 또는 `fal-ai`입니다(`grail/core/types.py`). 판단: 영상 생성 결과는 생성기 버전에 따라 달라집니다.
- **D12. 체형.** 논문: 캐릭터는 G1에 맞춘 SMPL-X 에셋(§3.1). 코드: `data/body_profiles/unitree_g1/soma/soma03_custom_smplx_identity.npz`에 고정 identity 파일을 씁니다. 재구성 시 원래 몸 형상을 쓰고, 배치 단계에서 고정 G1 identity를 적용합니다(`manip_soma.yaml` 주석). 판단: 체형 처리 방식이 논문과 다릅니다(몸 형상을 추정하지 않는다는 점은 같음).

### 추적과 정책

- **D13 (중요). 어댑터 구조.** 논문: SONIC 인코더, FSQ, 디코더를 동결하고 3층 MLP(512, 256, 128)가 잔차 Δz(λ=0.1)를 더함(식 6, B.1). 코드: `imports/SONIC-manipulation/gear_sonic/trl/modules/sonic_decoder_lora.py`는 **디코더 LoRA를 PPO 우도 안에 넣는 변형**을 구현합니다. 설정군에는 `contactfirst_d_sonic_lora`, `b_hiphi_sonic_lora`, `contactfirst_d_sonic_noiseonly`(LoRA 없이 어댑터만 쓰는 대조군) 등이 있습니다. 미세조정 설정(`contactfirst_finetune`)은 `ablation_d`를 상속하고, 상속 체인에서 디코더 동결 여부는 확인하지 못했습니다. 판단: 공식 추적기가 논문의 "동결 디코더 + 잔차 어댑터"인지 "LoRA 디코더 미세조정"인지 본문과 설정만으로 확정할 수 없습니다. 재현하려면 실제 production 설정 파일의 freeze 값이 필요합니다.
- **D14. 높이맵 인코더 구조.** 논문: 11×11 높이맵을 3층 CNN(64, 128, 256, 커널 3, 스트라이드 2)으로 인코딩해 1,024차원을 얻고 융합 MLP(256)에 넣음(B.2). 코드: `imports/SONIC-manipulation/gear_sonic/trl/modules/universal_token_modules.py`의 `height_adapter`는 **평탄화 후 MLP**(기본 512, 256, SiLU)로 64차원 토큰을 만들고 잠재에 더합니다. CNN은 확인하지 못했습니다. 판단: 인코더 구조가 다릅니다. 이 차이는 논문 Table 2의 지형 결과와 관련될 수 있습니다.
- **D15. 학습 규모 (중요).** 논문: 30,000 iteration, 1,024 env/GPU, GPU 64장, 약 30시간(§3.3, B.1–B.3). 코드: 미세조정 기본값은 **5,000 iteration, 2,048 env, 단일 GPU**입니다(`manipulation_finetune.sh`의 `ITERATIONS=5000`, `NUM_ENVS=2048`; README의 지형 예제도 5,000). 판단: 공식 예제는 논문의 전 학습과 다른 설정입니다. 워밍스타트 미세조정 기준이고, 처음부터 학습하는 설정은 공개 예제에 없습니다.
- **D16. 평가 체계.** 논문: 실물 평가 스크립트와 실물 배포 코드(머리 카메라 RGB 증류, OAK-D W, 10 Hz, RTX 5090 데스크톱, §3.4). 코드: GRAIL 고유의 egocentric RGB 증류와 실물 배포 코드를 찾지 못했습니다. `imports/SONIC`에는 배포 스택(`gear_sonic_deploy/`)과 카메라 서버 서비스(`systemd/composed_camera_server.service`)가 있습니다. GRAIL의 egocentric 정책 전용 코드인지는 확인하지 못했습니다. 판단: 실물 파이프라인은 SONIC 사본에서 부분적으로만 확인됩니다.
- **D17. 손가락 관측.** 논문 Table 5는 "Hand-to-object transform per right hand"라고 적었습니다. 코드의 관측 정의는 확인하지 못했습니다(`03` K5).

### 데이터와 자원

- **D18 (중요). 데이터 규모와 구성.** 논문: 20,000 이상 시퀀스, 네 범주(§4.2). 공개 데이터 카드: 8개 범주, 36,097 모션(`03` V6). 궤적 주파수 25 Hz, 몸 모델 SMPL-X 또는 SOMA, 로봇 G1 29 자유도 + 손 7 × 2(카드 표). 판단: 공개본은 논문보다 크고 몸 모델이 혼합되어 있습니다. 논문 Table 1–3의 학습 데이터와 같은지 확인할 수 없습니다.
- **D19. 라이선스 조합.** 논문 PDF: arXiv 비독점 배포, "© 2026 NVIDIA. All rights reserved." 코드: NVIDIA License(비상업 연구·평가). 데이터: CC BY-NC 4.0(절차 생성 범주), CC BY 4.0(RoboCasa 파생 범주, 저작자 표기 필요). 내장 체크포인트: NVIDIA Open Model License(SONIC 가중치, 공식 카드 기준). 판단: 개인 연구와 비상업 용도로만 사용 가능합니다. 논문 그림과 데이터의 공개 재배포는 각각의 조건에 맞춰야 합니다.

### 한 줄 요약

공식 저장소는 논문의 **전체 구조와 대부분의 PPO 설정은 맞지만, 사람 포즈 추정기(GENMO→GEM-X), 리타게팅(GMR→SOMA Retargeter), 손실 목록과 가중치, 장면 생성, 추적기 변형(LoRA), 학습 규모**가 다릅니다. 논문 수치를 공식 코드로 그대로 재현할 수 없다는 뜻입니다. 공식 저장소는 "2026-06 제출 논문의 코드"가 아니라 "이후 확장된 파이프라인"입니다.

## 3. 주장 확인 (코드가 있으므로 C 항목 대신 D로 정리)

코드 저장소가 있으므로 `04_claim_verification.md`는 만들지 않았습니다. 주장별 상태는 다음과 같습니다.

| 주장 | 상태 | 근거 |
|---|---|---|
| 메트릭 4D HOI 복원 (깊이 모호성 감소) | 구조는 확인, 수치는 미확인 | D1–D8, `03` V1–V4 |
| 과제 일반 추적 (어댑터가 몸 추적을 해치지 않고 물체 상호작용을 가능하게 함) | 설정 존재 확인, 결과 재현 불가 | D13, Table 2 |
| 20,000 이상 시퀀스 | 공개 데이터는 36,097 모션 | D18, `03` V6 |
| 실물 84% pick-up, 90% 계단 | 실물 코드 확인 불가 | D16, `03` V1–V2 |

## 4. 구현 가능성 판정

**판정: 부분적으로 가능합니다. 논문 결과의 재현은 이 맥에서 불가능합니다.** 근거와 자원 추정은 다음과 같습니다.

### 4.1 이 맥(Apple M4 Pro, 약 24 GiB 통합 메모리, MPS, CUDA 없음)에서

| 작업 | 판정 | 이유 |
|---|---|---|
| 공식 파이프라인 실행 (재구성, 추적, 정책 학습) | 불가 | 저장소 README는 "Linux GPU host, NVIDIA Docker"를 전제합니다. GEM-X, FoundationPose, MoSE3, Isaac Lab은 CUDA를 씁니다. |
| 논문의 학습 (30,000 iter, 64 L40) | 불가 | GPU 64장, 약 30시간. 이 맥에서 같은 규모를 돌릴 수 없습니다. |
| Kling으로 영상 생성 | 조건부 | 유료 API 호출 비용과 승인이 필요합니다. 사용자 승인 없이 호출하지 않습니다. |
| 실물 Unitree G1 배포 | 불가 | 실물 로봇과 RTX 데스크톱이 필요합니다. |
| 공식 저장소의 정적 읽기와 재계산 | 완료 | `03`, 이 문서 |
| 어댑터 구조의 PyTorch 구현과 형상 검사 (MPS 또는 CPU) | 가능 | 3층 MLP, 64+2 출력, λ=0.1 잔차, FSQ는 고정. 입출력 차원 검사와 단위 테스트 수준. 추정 1–2주. |
| 높이맵 인코더 (11×11 CNN 대 MLP) 비교의 CPU 토이 | 조건부 | 합성 높이맵으로 형상과 연산 시간만 확인할 수 있음. 정책 성능은 재현하지 않음. 추정 1주. |
| 소형 G1 MuJoCo CPU 환경의 추적 학습 | 조건부 | G1 자산을 받아야 함(라이선스 확인, 승인 필요). CPU 처리량에 따라 추정 2–4주. 논문 규모와는 다름. |
| 공식 데이터셋 (36,097 모션) 내려받기 | 보류 | 이 분석의 제약(가중치와 데이터 다운로드 없음). 필요하면 사용자 승인 후 메타데이터부터. |

### 4.2 GPU, 로봇, 데이터가 필요한 것

- 복원 전체(GEM-X, FoundationPose, MoGe-2, SAM2): CUDA GPU. 시퀀스당 A100에서 약 14분(논문 Table 4).
- 추적 학습: GPU 여러 장과 Isaac Lab (`imports/SONIC`). 워밍스타트 미세조정이라도 NVIDIA GPU가 필요합니다.
- 실물 검증: G1, OAK-D W 카메라, RTX 데스크톱.

### 4.3 노력 추정 (전체)

- 논문 방법의 **알고리즘 핵심**을 PyTorch로 옮기고 CPU 또는 MPS에서 형상과 손실 값을 검사하는 것: 2–4주.
- 공식 코드의 추적 설정을 같은 맥에서 돌리는 것: 불가(위 참조).
- 논문 수치를 다시 얻는 것: 조직 자원(GPU 수십 장과 로봇)이 필요해 이 저장소의 범위 밖입니다.

## 5. 문제와 미확인 항목

- **미확인 1:** 손 추정이 GEM-X에서 나오는지 WiLoR에서 나오는지(D2).
- **미확인 2:** 물체 추적 FoundationPose 미세조정의 epoch과 설정(D3).
- **미확인 3:** 공식 추적기의 디코더 동결 여부(D13).
- **미확인 4:** 높이맵 인코더의 실제 구조(D14).
- **미확인 5:** GRAIL 고유 실물 배포 코드의 위치(D16).
- **미확인 6:** 공식 저장소가 2026-06 제출 시점의 코드를 그대로 보존하는지. 2026-08 이후 추가 사항은 논문 버전과 대응하지 않습니다.
- **미확인 7:** 논문 Table 2–3의 결과가 어느 커밋(또는 이전 버전)에서 나왔는지. `docs/` 또는 이슈에 기록이 있는지 확인하지 않았습니다.
