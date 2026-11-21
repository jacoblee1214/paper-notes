# UMI 공식 코드 대조 (정적 읽기)

**대상:** `real-stanford/universal_manipulation_interface`, HEAD `d095ba9590df789df5189eea5ee7e431689038a6` (2026-05-26), MIT. 얕은 복제 `repo/`(gitignore)를 **정적으로만** 읽었습니다. 코드를 실행, import, 설치하지 않았고 체크포인트와 zarr 데이터도 받지 않았습니다. 저장소의 `README.md`는 체크포인트(`cup_wild_vit_l_1img.ckpt`)와 zarr 데이터셋을 공개한다고 안내하지만, 이 분석에서는 내려받지 않았습니다.

**대조 기준:** 논문 v3(01 §2–3, 표 A1). 코드 쪽은 파일 경로와 설정 키를 적었습니다.

## 1. 결과 요약

| 번호 | 항목 | 논문 | 코드 | 판정 |
|---|---|---|---|---|
| D1 | 상대 엔드이펙터 자세 (PD2.1, PD2.2, PD2.3) | 현재 자세 기준 상대 SE(3) | `pose_repr_util.convert_pose_mat_rep`의 `relative` 분기 `inv(base) @ pose`, 기준은 마지막 관측 자세 | 일치 |
| D2 | 원본 어안 입력 (HD2) | 왜곡 보정 없이 원본 사용 | 평가 기본값 `sim_fov=None`이면 변환기 없음. 보정은 옵션(`--sim_fov`)으로만 켜짐 | 일치 |
| D3 | 두 그리퍼 상대 자세 (PD2.3) | `robot{i}_eef_pos_wrt{j}` 관측 | `umi_dataset.py` 273–296행에서 두 로봇 사이 상대 자세를 만듦 | 일치 |
| D4 | 확산 스케줄 (표 A1) | 학습 50 스텝, 추론 16 스텝, DDIM | `num_train_timesteps: 50`, `num_inference_steps: 16`, `beta_schedule: squaredcos_cap_v2` | 일치 |
| D5 | 시각 인코더 학습률 (부록 E) | 사전학습 인코더 학습률 10배 작게 | `train_diffusion_unet_image_workspace.py`에서 사전학습 인코더 lr × 0.1 | 일치 |
| D6 | 비전 인코더 (표 A1) | 컵 CLIP ViT-B/16, 던지기·접기 ResNet-34(사전학습 없음), 일반화 CLIP ViT-L/14 | 기본 UMI 설정은 `vit_base_patch16_clip_224.openai` 하나. ResNet-34 설정은 주석 처리된 예시. ViT-L은 공개 체크포인트 설명에만 있음 | **차이**: 저장소 기본은 던지기·접기 설정과 다름 (D6 상세) |
| D7 | 증강 (부록 E) | 무작위 자르기 0.95, 회전 ±5°, 색 흔들기 (밝기 0.3, 대비 0.4, 채도 0.5, 색조 0.08) | 자르기와 색 흔들기는 일치. 회전은 기본 UMI UNet 설정에서 **주석 처리**(주석: "RandomRotation is not compatible with BF16"). 트랜스포머 UMI와 양팔 설정에는 ±5° 활성 | **부분 차이** (D7 상세) |
| D8 | 학습 기간과 배치 (표 A1) | 50–350 에폭, 배치 224–1024, 4–8 GPU | 기본 UNet 설정 120 에폭, 배치 64. 양팔 설정은 다른 값을 쓸 수 있으나 표와 같은 과제별 설정은 확인하지 못함 | **차이** (D8 상세) |
| D9 | 행동 지평과 실행 스텝 (표 A1 "Ta") | Ta = 6 (과제), 주파수 10 Hz (대부분), 20 Hz (던지기) | 학습 `n_action_steps: 8`, `action_horizon: 16`. 평가 기본값 `steps_per_inference 6`, `frequency 10` | 부분 일치. 20 Hz 던지기 설정은 저장소에 없음 |
| D10 | 지연 보상 (PD1, 그림 5) | 관측은 프레임 시각 기준 보간, 실행은 미리 명령 | 관측 지연 설정 `camera_obs_latency 0.125`, `gripper_obs_latency 0.02`, `robot_obs_latency 0.0001` (`diffusion_policy/config/task/umi.yaml`). 명령 지연 `command_latency 0.01`(평가 기본값) | 구조는 일치. **수치는 예시 설정**. 논문 측정값(부록 A)과 같은 값이 아님 |
| D11 | 거울 반사와 좌우 교환 (HD3, §V-A) | 반사 + 좌우 교환이 최선 (20/20) | 평가 기본값 `mirror_swap=False`. 반사 마스크(`draw_predefined_mask`)와 거울 영역 처리 있음 | **확인 필요**: 기본 실행은 논문 최선 설정이 아님 (D11 상세) |
| D12 | SLAM 수정 (부록 D: 맵 초기화, 마커 초기화) | ORB-SLAM3 수정 | README가 `cheng-chi/ORB_SLAM3` 포크와 Docker `chicheng/orb_slam3`를 씀. 포크는 이 저장소 밖 | 대조 불가 (저장소 밖) |
| D13 | 옵티마이저 (표 A1 학습률) | 학습률 3e-4 | `optimizer.lr: 3.0e-4`, AdamW, betas (0.95, 0.999), weight decay 1e-6 | 일치 (학습률). 가중치 감쇠는 논문에 없음 |
| D14 | 데이터 필터링 (HD6) | 운동학 기반 필터 | 저장소에 필터 파이프라인 있음(`scripts_slam_pipeline`). 기본 설정의 필터 기준은 대조하지 않음 | 확인 못함 |

## 2. 세부 대조

### D6. 비전 인코더 설정

`diffusion_policy/config/train_diffusion_unet_timm_umi_workspace.yaml`의 `model_name`은 `vit_base_patch16_clip_224.openai`입니다. 같은 파일에 `resnet34`, `vit_tiny_patch16_224`, `efficientnet_*`가 **주석 처리된** 예시로 있습니다. 논문 표 A1에서 던지기와 접기는 ResNet-34(사전학습 없음)를 쓰므로, 저장소 기본 설정으로는 두 과제를 그대로 재현하지 못합니다. 설정을 바꾸면 되는 수준이지만, 논문 설정과 다르다는 점은 확인해야 합니다.

### D7. 회전 증강

`train_diffusion_unet_timm_umi_workspace.yaml`에서 `RandomRotation`은 주석 처리되어 있고 옆 주석은 BF16과 호환되지 않는다고 적습니다. 반면 `train_diffusion_transformer_umi_workspace.yaml`과 `train_diffusion_unet_umi_bimanual_workspace.yaml`에는 `degrees: [-5, 5]`로 활성화되어 있습니다. 즉 논문 증강 목록은 저장소 설정 세 개 중 두 개와 맞고, 기본 UNet 설정과는 맞지 않습니다.

### D8. 학습 기간과 배치

기본 UNet 설정은 `num_epochs: 120`, 배치 `64`입니다(`training` 블록). 논문 표 A1은 과제별 50–350 에폭, 전체 배치 224–1024입니다. 표의 배치는 여러 GPU를 합한 값이고, 저장소의 `batch_size`는 데이터 로더 기준이므로 직접 비교하기 어렵습니다. 그래도 기본 120 에폭은 논문 범위(50–350) 안에 있어 에폭 차이는 크지 않습니다. 과제별 설정이 저장소에 있는지는 확인하지 못했습니다.

### D9. 행동 지평과 주파수

학습 기본값은 `n_action_steps: 8`, `action_horizon: 16`입니다. 평가 스크립트 `eval_real.py`의 `--steps_per_inference`는 기본값 6이고, `--frequency`는 기본값 10 Hz입니다. 논문 표 A1의 Ta = 6은 평가 쪽 값과 맞습니다. 던지기의 20 Hz 설정은 저장소 기본값에 없습니다. `obs_down_sample_steps: 3`과 `dataset_frequeny: 0`(주석에 59.94)은 데이터 기록 주파수에 따라 달라지므로, 실제 주파수를 정하는 설정은 데이터 파이프라인을 봐야 합니다.

### D10. 지연 수치

`diffusion_policy/config/task/umi.yaml`의 `camera_obs_latency: 0.125`(초), `gripper_obs_latency: 0.02`, `robot_obs_latency: 0.0001`은 예시 설정입니다. 논문 부록 A는 UR5와 Franka FR2에서 카메라, 로봇 고유감각, 그리퍼 지연을 따로 측정하고, 그림 5(c)는 실행 지연을 그리퍼 120 ms, 팔 100 ms로 그립니다. 같은 값은 아니므로, 재현하려면 하드웨어마다 다시 측정해야 합니다. 이 점은 "지연 맞춤이 중요하다"는 주장(C4)의 실제 조건과 맞닿아 있습니다.

### D11. 거울 설정

평가 스크립트(`eval_real.py`, `scripts_real/eval_real_umi.py`)는 `--no_mirror`(기본 False), `--mirror_swap`(기본 False), `--mirror_crop`(기본 False)을 씁니다. 논문 §V-A는 "디지털 반사와 좌우 교환"을 최선으로 보고했습니다. 기본 실행은 교환이 꺼져 있으므로 논문 최선 설정과 같지 않습니다. 다만 이 플래그가 학습 데이터 변환에도 쓰이는지, 또는 어떤 과제 예시에서 켜지는지는 이 정적 읽기로 확정하지 못했습니다.

## 3. 저장소 구조와 실행 조건 (정적 확인)

- `README.md`: Ubuntu 22.04 전제, Docker로 SLAM 실행, 단일 GPU는 RTX 3090 24 GB에서 시험했다고 적습니다. 학습은 `accelerate`로 여러 GPU를 씁니다.
- `train.py`와 `diffusion_policy/workspace/train_diffusion_unet_image_workspace.py`: 옵티마이저와 사전학습 인코더 학습률 분리(D5)가 있습니다.
- `umi/real_world/`: UR5와 Franka 등 로봇 드라이버, 그리퍼 인터페이스, 카메라 파이프라인(`bimanual_umi_env.py`).
- `scripts_slam_pipeline/`: SLAM 결과에서 zarr 데이터셋을 만드는 단계.
- 라이선스: `LICENSE`는 MIT입니다(Columbia AI & Robotics Lab, 2023).

## 4. 구현 가능성 판정

**판정: 이 맥(Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)에서 논문 실로봇 실험과 정책 학습 재현은 불가능합니다. 정적 대조와 수치 재계산은 할 수 있고 이미 끝냈습니다.**

| 작업 | 자원 | 맥에서 | 근거 |
|---|---|---|---|
| 실로봇 과제 재현 (UR5, Franka FR2) | 로봇 두 종, 그리퍼, GoPro 장치 | 불가 | 하드웨어 필요 |
| 현장 시연 수집 | 사람, 장소, GoPro | 불가 | 사람과 장소 필요 |
| SLAM 파이프라인 실행 (`run_slam_pipeline.py`) | Docker(리눅스 컨테이너), ORB-SLAM3 포크 빌드 | 조건부 | 리눅스 arm64 컨테이너 빌드와 GPL 포크 빌드 필요. 공식 코드 실행 승인 전이라 보류 |
| 정책 학습 (기본 UNet ViT-B CLIP, 300 에피소드) | 공식 README는 RTX 3090 24 GB 단일 GPU를 시험했다고 씀 | 불가(실용) | CUDA 전제 코드. MPS에서 ViT-B 224 학습 시간은 측정하지 않았지만 수십 시간 이상으로 추정. 측정하지 않은 추정 |
| 체크포인트 추론 (cup_wild_vit_l) | 체크포인트(크기 미확인), MPS | 조건부 | 체크포인트 다운로드 승인 필요. 이 분석에서는 받지 않음 |
| 상대 자세 식과 행동 표현 검사 (numpy) | CPU | 가능 | 식만 옮기는 검사. 이 분석에서는 만들지 않음 (03 §6) |

**노력 추정 (측정 없음):**
- 공식 SLAM 파이프라인을 리눅스 컨테이너로 돌려 example 세션을 처리하기: 1–2주(환경 구축 중심).
- 체크포인트 추론 한 번 돌리기: 1–3일(승인 후 MPS 호환성 확인 포함).
- 학습 재현: 맥에서는 현실적이지 않습니다. 다른 GPU 환경이 있을 때 판단합니다.
- 로봇 배포: 저장소 밖 하드웨어가 필요하므로 이 저장소의 범위를 벗어납니다.
