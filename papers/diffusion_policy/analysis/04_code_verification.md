# 04. 공식 코드 대조 (정적 읽기)

대상: `repo/` (github.com/real-stanford/diffusion_policy, 커밋 `5ba07ac6661db573af695b419a7947ecb704690f`, 2024-12-24, 얕은 복제). 정적 읽기만 했습니다. 코드를 실행하거나 import하거나 설치하지 않았습니다. 대조 대상은 논문 v5 본문, 부록 표 7·8, 그리고 `diffusion_policy/config/` 설정 파일입니다. 점검 번호 D1–D15는 이 문서의 내부 번호입니다.

## 1. 저장소 개요와 라이선스

- 루트에 `LICENSE` 파일이 있고 MIT입니다 ("Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab"). `tam_torque`의 공식 저장소와 달리 라이선스가 불명확하지 않습니다.
- 주요 진입점: `train.py` (설정 기반 학습), `eval.py` (체크포인트 평가, 기본 장치 `cuda:0`), `eval_real_robot.py`, `demo_real_robot.py`, `demo_pusht.py`.
- 정책 구현: `diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py` (CNN 확산 정책, 이미지), `diffusion_transformer_hybrid_image_policy.py` (트랜스포머, 이미지), `diffusion_unet_lowdim_policy.py` (상태), `diffusion_transformer_lowdim_policy.py` (상태 트랜스포머), `ibc_dfo_*`와 `bet_lowdim_policy.py` (기준선).
- 관찰 장치: 학습 설정의 `training.device`가 `cuda:0`입니다. 저장소의 README도 CUDA GPU를 전제로 합니다 (`conda_environment.yaml`에 `cudatoolkit=11.6`과 `pytorch3d` 포함).

## 2. 논문과 코드의 대조 (D1–D12)

| ID | 항목 | 논문 | 코드 (정적) | 판정 |
|---|---|---|---|---|
| D1 | 시각 인코더 풀링 | §3.2: 전역 평균 풀링을 **공간 소프트맥스**로 교체 | `model/vision/model_getter.py`의 `get_resnet`은 torchvision ResNet의 `fc`만 `Identity`로 바꿉니다. 저장소 안에 `SpatialSoftmax` 정의가 없습니다. 전역 평균 풀링이 그대로 남습니다 | **차이**. 논문 설명과 다른 구현 (또는 코드에 없는 구현) |
| D2 | 정규화 | §3.2: BatchNorm을 GroupNorm으로 교체 | `multi_image_obs_encoder.py`의 `replace_bn_with_gn`에 해당하는 `nn.GroupNorm` 치환 (`use_group_norm: True`) | 일치 |
| D3 | 학습 에폭 | §5.2: 상태 기반 4500, 이미지 3000 | 설정: CNN 이미지 (hybrid) 3050, CNN 이미지 8000, CNN 상태 5000, 트랜스포머 상태 (Push-T) 8000, 트랜스포머 상태 (Kitchen) 5000 | **차이**. 논문 수치와 일치하는 설정이 없음 |
| D4 | 평가 초기 조건 수 | 표 1 캡션: 50 × 3 시드 = 150. 각주: 로봇믹 과제는 **22**만 사용 (평가 코드 버그) | 과제 설정 `n_test: 50` (예: `lift_lowdim.yaml`, `tool_hang_lowdim.yaml`) | **차이 또는 수정됨**. 논문 결과는 22개 기준이고 저장소 기본은 50 |
| D5 | 노이즈 분산 방식 | 본문에 명시 없음 | `train_diffusion_unet_hybrid_workspace.yaml`의 주석: "Yilun's paper uses fixed_small_log instead, but easy to cause Nan". 실제 설정은 `variance_type: fixed_small` | **차이** (저장소 스스로 밝힘) |
| D6 | 추론 단계 수 | §3.4: DDIM 학습 100, 추론 **10**, 0.1초 (RTX 3080) | 시뮬 CNN 설정 `num_inference_steps: 100`. DDIM 설정 (`train_diffusion_unet_ddim_*`) 8. 실로봇 hybrid 8, 실로봇 image 100. 표 7은 실로봇 16 | **불일치** (세 값이 공존). `03` V15 |
| D7 | 하이퍼파라미터 표 (7, 8) | 표 7 (CNN): 학습률 1e-4, 가중치 감쇠 1e-6, To 2, Ta 8, Tp 16, 학습 단계 100. 표 8 (트랜스포머): Push-T 가중치 감쇠 1e-1, Kitchen 1e-3, 임베딩 256 (Push-T), 768 (Kitchen), To 4 (Kitchen) | CNN 기본값이 표 7과 일치. 트랜스포머 설정: Push-T 가중치 감쇠 0.1, Kitchen 1e-3, 임베딩 256 (Push-T), 768 (Kitchen), To 4 (Kitchen) 일치. 예외: 실로봇 Push-T는 표 7·8 모두 Ta = 6인데, 실로봇 설정 `n_action_steps: 8` | 대부분 **일치**, 실로봇 Ta **차이** |
| D8 | 평가 집계 | §5.2: 마지막 10개 체크포인트 (50 에폭마다)의 평균 | `checkpoint_every: 50`, `rollout_every: 50`은 일치. 체크포인트 보존은 상위 5개 (`test_mean_score`)와 마지막 체크포인트. README는 "논문 수치는 `max`와 `k_min_train_loss` 집계 키에 해당"이라고 씀 | **차이**. 논문 설명 (마지막 10개 평균)과 README 설명 (`k_min_train_loss`)이 다름 |
| D9 | 실로봇 학습 길이 | 부록 C.1.2: 고정 학습 시간 12시간 (Push-T), 부록 C.2.2: 1000 에폭 (붓기, 펴기) | 실로봇 설정 `num_epochs: 600` (hybrid와 image 모두) | **차이** (설정 기준. 시간 기준 학습은 설정으로 확인 불가) |
| D10 | Push-T 학습 에피소드 | 표 3: Push-T #PH = 200 | `pusht_lowdim.yaml`, `pusht_image.yaml`: `max_train_episodes: 90` | **차이** (학습 에피소드 90개 제한) |
| D11 | 최대 롤아웃 길이 | 표 3: Lift 400 (단일 값) | `lift_lowdim.yaml`: `400 if ph else 500 if mh` | 대체로 일치 (표는 ph만 적은 것으로 보임) |
| D12 | 이미지 전처리 (실로봇) | 표 7: 2 × 320 × 240 → 2 × 288 × 216 | 실로봇 설정: `resize_shape [240, 320]`, `crop_shape [216, 288]`. 시뮬은 84 → 76 무작위 자르기 | 일치 |
| D13 | ImageNet 정규화 | 본문 없음 | 설정 `imagenet_norm: True`. 표 7에도 없음 | 추가 설정 (논문에 없음) |
| D14 | 사전학습 | §3.2: 사전학습 없음 (ResNet-18) | 설정 `weights: null` | 일치 |
| D15 | 실험 로그 표 번호 | 본문 표 1, 2, 4 | README: "Table I, II and IV" | 명명 불일치 (v1 또는 학회판 표 번호로 보임, 사소함) |

- D1은 본문의 설계 설명과 코드가 다릅니다. 결과에 어떤 영향을 주는지는 실험 없이 판정하지 못했습니다. 공간 소프트맥스가 빠지면 공간 정보가 평균에 묻힐 수 있습니다. 실제 영향은 실험으로만 알 수 있습니다.
- D4와 D8은 표 1–2 수치를 해석할 때 중요합니다 (`03` §4, §5). 설정 기본값과 논문 실행 값이 다르고, 집계 방식도 달라서 표 수치를 다른 평가 설정과 바로 비교할 수 없습니다.
- D7의 BlockPush 제어 방식은 셋 중 어느 쪽인지 확정하지 못했습니다. 표 7은 Pos, 표 8은 Vel이고, 설정 `task/blockpush_lowdim_seed.yaml`은 `abs_action: False`만 적혀 있어 위치인지 속도인지 명확하지 않습니다 (`03` V13).

## 3. 실행 환경 (구현 가능성에 필요한 사실)

- **CUDA 전제.** 학습 설정 기본값 `training.device: "cuda:0"`, `eval.py`의 기본 장치 `cuda:0`. 저장소 README는 "Nvidia GPU가 있는 Linux"를 전제로 합니다. MPS를 명시 지원하는 코드는 확인하지 못했습니다 (실로봇 스크립트의 일부 문자열 일치는 장치 지원이 아니었습니다).
- **맥용 conda 환경이 있습니다.** `conda_environment_macos.yaml`: Python 3.9, PyTorch 1.12.1, torchvision 0.13.1, diffusers 0.11.1, gym 0.21.0, pymunk 6.2.1. 오래된 버전 고정이라 Apple Silicon의 MPS와 맞는지 확인하지 못했습니다. 새 가상환경을 만들려면 설치할 목록을 먼저 확인해야 합니다.
- **Push-T 데이터.** 데이터 경로 `data/pusht/pusht_cchi_v7_replay.zarr`는 저장소 밖에서 받아야 합니다. 받지 않았습니다.
- **체크포인트.** 실험 로그와 체크포인트는 프로젝트 페이지 아래에 있습니다 (`02` §2). 받지 않았습니다.

## 4. 구현 가능성 판정 (Mac M4 Pro, 24 GiB, MPS, CUDA 없음)

**판정: 논문 표 재현은 이 맥에서 불가능합니다. 구조 재구현과 작은 2차원 토이는 가능합니다.**

| 작업 | 판정 | 근거 |
|---|---|---|
| 표 1–2 시뮬 전체 재현 (로봇믹 9변형, Push-T, 시드 3, 평가 150) | **불가** (GPU 학습 전제, 시간 과다) | 설정 3050–8000 에폭, CUDA 기본값, 공개된 로그 (D3, D4) |
| Push-T 상태 기반 한 시드 학습, 공식 코드, MPS 포팅 | **조건부** | 소형 모델이라 MPS 가능성은 있음. 포팅과 구버전 환경 설치 (Python 3.9, torch 1.12.1)가 먼저 필요. 학습 시간은 측정 전 |
| 체크포인트 추론과 시뮬 평가 (공개 체크포인트 사용) | **조건부** | 체크포인트 다운로드와 새 환경 설치 승인 필요. 맥 속도는 측정 전 |
| 실로봇 재현 (표 6, 양손 작업) | **불가** | UR5, Franka Panda, 카메라, 원격조작 장치 필요 |
| 구조 재구현 (1D U-Net 확산 정책, 2차원 토이 다중 모드) | **가능** | numpy 또는 PyTorch만으로 가능. 설치 최소. 노력 추정 1–2일 (추정) |
| 행동 지평선과 DDIM 단계 수에 대한 토이 검사 (Fig. 5 유사) | **가능** | 토이에서 CPU로 수 분 이내 (추정) |
| 표 재계산과 집계 검사 | **완료** | `analysis/recompute.py` (`03`) |

- 노력 추정 (측정 전): 구조 재구현 1–2일, 공식 코드 MPS 포팅과 Push-T 상태 학습 1–2주. 추정치이며 측정하지 않았습니다.
- 차이 D1 (공간 소프트맥스)과 D5 (분산 방식)는 공식 코드를 그대로 쓰면 논문 구조와 다르게 됩니다. 재구현하려면 어느 쪽을 따를지 먼저 정해야 합니다.

## 5. 재현에 쓰기 전 확인할 것

1. D4의 평가 초기 조건 수를 논문 값 (22)과 저장소 기본값 (50) 중 무엇으로 맞출지 정합니다.
2. D8의 집계 방식이 마지막 10개 평균인지 `k_min_train_loss`인지 로그로 확인합니다 (`03` V1과 연결).
3. D6의 추론 단계를 정합니다 (100, 16, 10 중 하나).
4. D10의 Push-T 학습 에피소드 (90 대 200)를 정합니다.
