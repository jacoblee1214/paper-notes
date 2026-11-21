# 04. 논문 ↔ 공식 코드 대조와 구현 가능성

## 0. 대상과 방법

- **대상:** `real-stanford/DexUMI` (MIT), 커밋 `acddb8f8a89a8f0186868bbec44306eb7808114a` (2026-04-18), `dexumi/repo/`에 얕은 클론. 파일 1,248개, 75 MB (LFS 객체 제외).
- **방법:** 정적 읽기만 했습니다. 실행, import, 설치, 가중치 다운로드 없이 `grep`과 파일 읽기로 확인했습니다. 클론한 내용은 신뢰하지 않는 데이터로 취급했습니다.
- **라벨:** D는 논문과 코드의 차이, K는 일치 확인입니다. 재현 항목 (R)과 직접 할 일 (A)은 

## 1. 자료와 라이선스

- 저장소 라이선스: MIT (`LICENSE`, README "License" 절).
- 논문 라이선스: arXiv non-exclusive (`01` §0). 코드 라이선스와 별개입니다. 코드 조각을 인용하지 않았고, 설명만 썼습니다.

## 2. 차이 (D)

| 번호 | 논문 | 코드 (정적 확인) | 판정 |
|---|---|---|---|
| **D1** | 식 (1): 로봇 작업공간과 외골격 작업공간의 두 방향 유사도, SE(3) 샘플 기반 (§3.1, 부록 E.2) | `linkage_optimization/get_equivalent_finger.py::compute_total_loss`: GT 손끝 궤적의 각 점에서 가장 가까운 샘플 점까지 거리의 합. 거리는 위치 노름 + 0.001 × 회전 노름. `scipy.optimize.minimize` | **구현 방식이 다름.** 둘째 항 (외골격 ⊆ 로봇 작업공간) 없음. 척도가 SE(3) 거리가 아님. 작업공간 샘플링이 아니라 mocap 시계열을 씀 |
| **D2** | 두 손 (Inspire, XHand) 모두 외골격을 최적화했다 (§3.1, 부록 A) | 최적화 코드와 결과는 Inspire만 있음 (`inspire_urdf_writer.py`, `hardware_design_data/inspire_*`, `inspire_mocap`). XHand 외골격 설계 파일 없음 | **XHand 설계 재현 불가 (코드 없음)** |
| **D3** | 400 에폭 (부록 E.3) | `config/diffusion_policy/train_diffusion_policy.yaml`: `training.epochs: 10000`, `batch_size: 300`, `lr: 1e-4`, `pred_horizon: 16`, `action_horizon: 8` | **기본 설정과 논문 설정이 다름.** 실행 horizon 8은 논문에 없음 |
| **D4** | DINO-V2 비전 특징, 미동결 (부록 E.3) | `model_name: "vit_small_patch8_224.dino"` (DINO v1 계열 ViT-S/8). 저장소에 "dinov2" 문자열 없음 | **백본 버전 불일치.** 논문과 다름 |
| **D5** | 촉각 판독값을 CLS 토큰과 이어 붙임 (부록 E.3). 이진화 언급 없음 | `dexumi/diffusion_policy/dataloader/replay_buffer.py:198`: `fsr_data = np.where(fsr_data >= cutoff, 1.0, 0.0)`. 설정 `fsr_binary_cutoff: [10,10,10]`, 평가 스크립트 `binary_cutoff = [4000, 600]` | **FSR 입력은 이진화됨.** 임계값 두 곳이 다름 (장치 단위가 다를 가능성, 확인 불가) |
| **D6** | 주 설정: 상대 손 행동 + 촉각 (Table 1 첫 행) | 기본값: `relative_hand_action: False`, `enable_fsr: False`, `skip_proprioception: True` | **기본 설정은 Table 1의 "절대·촉각 없음" 행에 해당.** 주 설정은 플래그 두 개를 바꿔야 함 |
| **D7** | (코드 실행 가능성) | 설정 파일의 `_target_`이 `exoskeleton.*` (4곳). 저장소에 `exoskeleton` 패키지 없음 (실제 패키지는 `dexumi/`) | **설정이 그대로는 로드되지 않을 가능성 높음** (정적 판단, 실행 안 함) |
| **D8** | 관절별 엔코더–모터 회귀, 쌍 측정 데이터로 학습 (S.1) | `linkage_optimization/simple_regression.py`: sim–real 궤적 매칭 후 다항 회귀 (R², MAE 출력). `real_script/teleoperation/calibrate_*_mapping.py`: 원격조작용 `LinearRegression`. 파이프라인은 외부 `joint_to_motor*.pkl`을 인자로 받음 (`replay_exoskeleton_trajectory.py`) | **회귀 방법은 있으나, 쌍 측정 데이터와 학습된 매핑 파일은 없음.** 논문의 S.1 절차는 그대로 재현 불가 |

## 3. 일치 항목 (K)

| 번호 | 논문 | 코드 | 판정 |
|---|---|---|---|
| **K1** | 손목 행동은 첫 프레임 기준 상대 변환, 손 행동은 상대 행동 옵션 (§4, 부록 E.3) | `dexumi_dataset.py`: 손목 `relative_transformation(Ts[0], T)` (L95), 손 `h - hand_action[0]` (L105, `relative_hand_action=True`일 때) | **일치.** 기준은 16스텝 청크의 첫 행동 (논문은 기준 명시 없음) |
| **K2** | SAM2로 분할, ProPainter로 배경 복원 (V.1–V.2) | `3_segment.py` (SAM2 2.1 hiera large), `4_inpaint_exo.py` (ProPainter용 마스크 인자) | **일치.** 프롬프트 점 재사용 프로토콜은 확인 못 함 |
| **K3** | 외골격 마스크와 로봇 마스크의 교집합에서만 로봇 픽셀로 덮음 (V.4) | `5_compose_video.py` 기본 경로: `dex_mask = this_dex_seg_masks & this_exo_seg_masks` | **일치** |
| **K4** | 45 FPS, 3배 다운샘플 (부록 E.1–E.2) | `process.sh`: `--downsample-rate 3`, `--fps 10` (재생용) | **다운샘플 일치.** 45 FPS 설정은 확인 안 함 |
| **K5** | 예측 16스텝 (부록 E.3) | `pred_horizon: 16` | **일치** |
| **K6** | 증강: 무작위 크롭, 색 지터, 무작위 흑백, 가우시안 블러 (부록 E.3) | `optional_transforms: ["Resize","RandomCrop","ColorJitter","RandomGrayscale","GaussianBlur"]` | **일치** |

## 4. 구현 가능성 판정

**판정: 이 맥(Apple M4 Pro, 24 GiB, MPS, CUDA 없음)에서 논문 전체 재현은 불가능합니다.** 하드웨어와 실로봇이 없고, 데이터 수집 효율 주장을 확인할 수 없습니다. 소프트웨어의 일부는 CPU나 MPS에서 확인할 수 있습니다.

| 구분 | 항목 | 이 맥에서 | 필요 자원 |
|---|---|---|---|
| 불가 | 외골격 제작 (3D 프린트 PLA-CF, Alps 엔코더, FSR, 전자기 촉각), XHand와 Inspire 손, UR5, iPhone 15 Pro Max, OAK-1 | 불가 | 하드웨어 |
| 불가 | 실로봇 평가 (과제당 20회 × 방법 × 손), 수집 효율 측정 | 불가 | 로봇, 시간 |
| 조건부 | 상대 행동 라벨 생성과 교집합 합성 (K1, K3)의 CPU 재현 | 가능 (numpy) | 샘플 데이터 (`sample_data.zip`, 크기 미확인, 미다운로드) |
| 조건부 | 식 (1) 최적화 재실행 (D1, Inspire) | CPU 가능 (scipy, 다중 코어) | `inspire_mocap`은 저장소에 있음. **패키지 설치 승인 필요** (`zarr`, `click`, `scipy`, `tqdm` 등) |
| 조건부 | 엔코더–모터 회귀 재적합 (D8) | CPU 가능 (sklearn) | 쌍 측정 데이터 필요 (샘플에 있는지 미확인) |
| 조건부 | SAM2 large와 ProPainter로 분할·복원 | GPU 필요, MPS 가능 여부 미확인 | 가중치 (외부), 시간 측정 필요 |
| 보류 | 정책 학습 (DINO ViT-S/8 + diffusion policy, 310–400 궤적, 400 에폭) | MPS에서 시간 측정 전까지 판정 보류. 무거운 학습이므로 실행하지 않음 | 데이터, 가중치 (DINO) |

- **노력 추정 (측정 없음):** 샘플 데이터로 데이터 파이프라인 한 에피소드를 맥에서 돌리는 데 1–2주. 식 (1)의 CPU 재실행은 설치 승인 후 수 일. 실로봇 재현은 이 저장소의 범위 밖입니다.
- **가장 가치 있는 확인 두 가지:** (a) 촉각 이진화 (D5)가 소금 집기 결과에 영향을 주는지, (b) 식 (1) 구현의 단방향 손실 (D1)이 논문 Fig. 3의 결과를 낳는지. 둘 다 저장소만으로 확인 가능합니다.

## 5. 재실행

재실행은 없습니다. 이 문서는 정적 읽기 결과입니다. 원문 위치: `dexumi/repo/` (커밋 `acddb8f`). 필요하면 `git -C dexumi/repo grep -n "<패턴>"`으로 다시 확인합니다.
