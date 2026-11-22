# 04. 공식 코드 대조와 구현 가능성

## 0. 대상과 방법

- **대상:** [j96w/DexCap](https://github.com/j96w/DexCap), 얕은 복제 HEAD `4b0bed0966c87368f3cde4476aadb7585c3b94b5` (2024-08-17, 커밋 메시지 "hardware updates"). 복제본은 `dexcap/repo/`에 있고 gitignore됩니다.
- **비교 대상:** 논문 arXiv v2 (2024-07-04) 본문, 부록 A–G, 표 I–VIII.
- **방법:** 파일 구조 확인, 주요 스크립트와 설정 읽기, `grep`으로 키워드 찾기. **실행, 설치, import, 가중치 다운로드는 하지 않았습니다.** 복제본 내용은 신뢰하지 않는 데이터로 다뤘습니다.
- **시점 주의 (D0):** 저장소의 마지막 커밋이 2024-08-17입니다. 논문 v2는 2024-07-04입니다. 저장소 README는 2024-08 하드웨어 갱신을 알리며, 그 뒤 수집 경로가 바뀌었습니다. 아래 대조는 **논문 쪽 시스템과 저장소 쪽 시스템이 같지 않을 수 있다**는 전제로 읽어야 합니다.

## 1. 설치 요구사항과 라이선스

- **라이선스:** MIT (LICENSE 파일 있음). 포함된 robomimic 사본도 MIT 계열입니다.
- **설치 환경 (README, 설치 폴더):**
  - 수집 PC (NUC): Windows, Rokoko Studio, Anaconda, `install/env_nuc_windows.yml`.
  - 작업 PC (Ubuntu, 헤드리스 서버 가능): Python 3.8, `install/env_ws_requirements.txt`, `STEP3_train_policy`에서 `pip install -e .`.
  - 설치 자체는 하지 않았습니다. `env_ws_requirements.txt`에는 torch, torchvision, diffusers==0.11.1, pybullet==3.2.6, h5py, open3d, scipy 등이 있습니다.
- **하드웨어 요구:** Rokoko 장갑 (Windows 소프트웨어), RealSense (T265, L515), 2024-08 갱신 경로에서는 HTC Vive 트래커 (OpenXR). 장비 없이 수집 단계는 실행할 수 없습니다.
- **공개 데이터 (Hugging Face, `chenwangj/DexCap-Data`):** 파일 6개, 합계 약 **107.8 GB**. 처리본 `hand_packaging_wild_1-20_5actiongap_10000points.hdf5` (29.1 GB), `hand_wiping_1-14_5actiongap_10000points.hdf5` (14.0 GB), 원시 `rawdata_packaging_wild_1-20.zip` (42.0 GB), `rawdata_wipe_1-14.zip` (22.7 GB). 라이선스 필드가 비어 있습니다. 받지 않았습니다.
- **공개 데이터의 범위:** 포장 (야외 수집)과 접시 닦기 두 과제뿐입니다. 논문의 스펀지, 공, 가위, 차 준비 데이터는 없습니다. 정책 설정 파일도 두 개뿐입니다 (`STEP3_train_policy/robomimic/training_config/`).

## 2. 논문 ↔ 코드 대조 (D 항목)

| 번호 | 항목 | 논문 | 공식 코드 | 판정 | 영향 |
|---|---|---|---|---|---|
| D0 | 수집 하드웨어 | Rokoko 장갑 + T265 세 대 + L515 (§III-A) | 두 경로: 원 경로 `STEP1_collect_data` (T265, RealSense, Rokoko), 2024-08 경로 `STEP1_collect_data_202408updates` (Vive 트래커, OpenXR, RealSense) | **다름** (새 경로가 논문 이후) | 재현 시 어느 경로가 논문 데이터인지 확인 필요 |
| D1 | 데이터 저장 형식 | 60 Hz 기록, 장면 RGB-D, 손 pose, 관절 (부록 B) | README: `pose.txt` (가슴), `pose_2.txt` (왼손), `pose_3.txt` (오른손), `*_hand_joint.txt`, 컬러와 깊이 프레임 | 맞음 (구조) | — |
| D2 | 점군 정렬 | 첫 SLAM 프레임 = 세계 좌표 (§IV-A) | README: `pose.txt` "world frame", `transform_to_robot_table.py`로 로봇 테이블에 맞춤 (키보드) | 맞음 | 사람이 정렬하는 단계가 있음 (부록 C와 같음) |
| D3 | 점군 인코더 | 기본 "Ours (DP-perc)" = **Perceiver** (표 I, 표 VII) | `robomimic/models/obs_nets.py`: 키 이름에 `pointcloud`가 있으면 **PointNet** (input 6, layers [32, 64, 128, 256], output 64)으로 **고정**. SetXFPCDEncoder 코드는 주석 처리. Perceiver는 공개 설정에서 쓰이지 않음 | **다름** | 공개 코드로 논문 기본 모델을 재현할 수 없음. 표 I의 DP-perc 결과 재현에 Perceiver 구현을 따로 연결해야 함 |
| D4 | 행동 지평 | 본문에 명시 없음 (표 VI "Input Horizon 3"만) | 설정: observation_horizon 3, action_horizon 10, prediction_horizon 20. README: "20 steps", 46차원 | 논문 공백 | 논문 재현 시 예측 지평을 임의로 정하면 안 됨 |
| D5 | 점 수 | 저장 5000 (부록 C), 정책 입력 1000 (부록 E, 표 VI) | `demo_create_hdf5.py`: `num_points_to_sample = 10000`. PointNet forward에서 1000 초과 시 `torch.randperm`으로 1000개 무작위 선택 (학습과 추론 모두, `base_nets.py`) | **다름** (저장 값 3종) | 추론이 호출마다 무작위. 시드를 고정하지 않으면 같은 입력도 행동이 달라짐 |
| D6 | 행동 차원 | 두 팔 "7-DoF" + 두 손 16-DoF = 46 (§IV-B), 본문 "6-DoF pose" (부록 D) | 설정의 `robot0_eef_pos` (3) + `robot0_eef_quat` (4) + `robot0_eef_hand`. README "46 dimensions" | 맞음 (쿼터니언 포함) | 용어 불일치. 46은 위치 3 + 쿼터니언 4로 읽어야 맞음 |
| D7 | 행동 정의와 프레임률 | a_t = s_{t+1}, 60 Hz를 20 Hz로 다운샘플 (§IV-A, §V-A b) | `demo_create_hdf5.py`: `action_gap = 5`. `dataset_utils.py`: 행동을 5프레임 뒤 상태로 두고, 프레임을 `j::action_gap`으로 나눠 **다섯 개 부분열을 각각 데모로 저장** | **다름** (논문에 없는 5배 분할) | 데이터가 논문의 5배로 늘어남. 20 Hz와 5프레임 간격이 맞는지 코드만으로 확인 못 함 |
| D8 | 학습 하이퍼파라미터 | 배치 16, 학습률 1e-4, 3000 에폭, DDIM 100/10, UNet [256, 512, 1024], 커널 5, 관찰 3 (표 VI) | 설정 파일 동일 (`diffusion_policy_pcd_*.json`). EMA는 꺼짐 (`ema.enabled = false`), 학습률 감쇠 없음 | 맞음 (EMA는 논문 공백) | EMA 설정은 결과에 영향을 줄 수 있음 |
| D9 | 사람 교정 | 잔차 모드 (α, β, β < 0.1), 텔레옵 모드, 발판 페달, 교정 데이터 혼합 (§IV-C, 부록 G) | 페달, 잔차, 텔레옵 로직 없음. `redis_glove_server.py`는 장갑 스트림 서버만. α, β 값 없음 | **공개 안 됨** | 교정 실험 (표 II 오른쪽, 표 III, IV) 재현 불가 |
| D10 | 로봇 실행 | 두 Franka + 두 LEAP, OSC 팔 제어, 관절 임피던스 손, 20 Hz, 도달 판정 h = 10 (§III-B, 부록 D) | `robomimic/envs/env_real_panda.py` (Panda 단일 래퍼, control_freq 20 기본값). LEAP 양손 제어기, OSC, 정책 롤아웃 루프 없음 | **공개 안 됨** (기본 환경만 있음) | 실로봇 결과 재현 불가 |
| D11 | 하드웨어 설계 | "모든 하드웨어 설계와 코드를 공개" (결론) | STL, STEP, URDF 없음 (`leap_hand_mesh`만 LEAP 손). 카메라 랙과 장갑 마운트 설계 없음 | **공개 안 됨** | 장비 제작 불가. 논문 결론의 문장과 다름 |
| D12 | 데이터 범위 | 여섯 과제, 과제당 데모 수 (표 I, 본문) | 공개 데이터: 포장 (야외)과 접시 닦기 두 과제만. 원시 zip도 같은 두 과제 | **부분 공개** | 스펀지, 공, 가위, 차 준비 결과는 공개 데이터로 확인 못 함 |
| D13 | 학습 설정 범위 | 과제별 학습 설정은 암시 | 설정 파일 2개 (포장, 접시). 스펀지, 공, 가위, 차 설정 없음 | 부분 공개 | D12와 같은 이유 |
| D14 | 기준선 | BC-RNN (img, img-mask, point, prec), DP (img, img-mask, point, point-raw) (표 I) | BC-RNN과 DP 기반은 robomimic 사본에 있음. 마스크 이미지는 `STEP2_build_dataset/utils.py`의 `mask_image`에서 만듦. BC-RNN-prec, DP-point-raw 설정은 확인 못 함 | 부분 | 기준선 몇 개는 설정을 직접 맞춰야 함 |
| D15 | 점군 증강 | 무작위 2D 평행 이동 (§IV-B) | 공개 설정의 `obs_randomizer_class`가 null. 평행 이동 코드 확인 못 함 (`obs_utils.py`에는 중앙 자르기만) | **확인 안 됨** | 증강이 없다면 논문과 결과가 달라짐 |
| D16 | 로봇 손 점군 | FK로 로봇 손 점을 점군에 더함 (§IV-B) | README: 사람 손이 보일 때 FK로 만든 로봇 손 메시를 점군에 추가. `dataset_utils.py`와 `pybullet_ik_bimanual.py` | 맞음 | — |
| D17 | 손끝 IK | 엄지와 세 손가락 IK, 새끼손가락 제외, 손목 6-DoF로 초기화 (§IV-A) | `pybullet_ik_bimanual.py`: 손 한 개당 4 × 4 = 16관절, 손목 초기화 | 맞음 (새끼손가락 제외는 세부 확인 못 함) | — |
| D18 | 교정 데이터 혼합 | 같은 확률로 D와 D' 샘플 (§IV-C) | 공개 코드에 교정 데이터 혼합 없음 | **공개 안 됨** | D9와 같은 이유 |

### 2.1 논문 주장의 코드 쪽 확인

- **"학습 코드와 데이터 처리를 공개한다" (§III-A, 결론):** 맞습니다. 수집 (STEP1), 데이터 처리 (STEP2), 학습 (STEP3)이 있습니다.
- **"손, 장비 설계와 교정 코드도 공개한다" (결론):** 맞지 않습니다 (D9, D10, D11). 공식 저장소는 학습 파이프라인 중심입니다.
- **"학습은 PyTorch 기반의 확산 정책으로 한다" (§IV-B):** 맞습니다 (robomimic 사본 + diffusion_policy.py).
- **"점군 대신 이미지를 쓰는 기준선이 움직이는 카메라에서 실패한다" (§V-B):** 기준선 코드 (마스크 이미지)는 있으므로 원인 분석은 재현 가능합니다. 실험 자체는 데이터가 없어서 재현 못 합니다 (D12).

## 3. 학습 설정 대조 요약

| 항목 | 논문 표 VI | 공식 설정 | 같은가 |
|---|---|---|---|
| 배치 | 16 | 16 | ✓ |
| 학습률 | 1e-4 | 1e-4 | ✓ |
| 에폭 | 3000 | 3000 | ✓ |
| DDIM 학습 / 추론 | 100 / 10 | 100 / 10 | ✓ |
| UNet 채널, 커널 | [256, 512, 1024], 5 | 같음 | ✓ |
| 관찰 지평 | 3 | 3 | ✓ |
| 점군 인코더 | PointNet (DP-point), Perceiver (DP-perc) | PointNet 고정 | DP-perc ✗ |
| 점군 다운샘플 | 1000 | 1000 (PointNet 내부), 저장 10000 | 저장 값 ✗ |
| EMA | 언급 없음 | 꺼짐 | — |

## 4. 구현 가능성 (이 맥)

**판정: 이 맥 (Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)에서 전체 재현은 할 수 없습니다.** 장비, 로봇, 교정 코드가 공개되지 않았고, 공개 데이터는 두 과제분 (약 108 GB)입니다. 데이터 처리와 학습 파이프라인의 일부는 CPU에서 할 수 있으나, 논문의 기본 모델 (Perceiver)은 공개 코드로 만들 수 없습니다.

| 구성 요소 | 필요 자원 | 이 맥에서 | 판정 | 노력 추정 (측정 없음) |
|---|---|---|---|---|
| 장비 (장갑, T265 ×3, L515, NUC) | 하드웨어 (약 4000 USD, 논문 기준) | 불가 | 불가 (범위 밖) | — |
| 실로봇 (두 Franka, 두 LEAP) | 로봇, 제어 코드 (D10) | 불가 | 불가 | — |
| 교정 (페달, α, β) | 로봇 실험 + 공개 안 된 코드 | 불가 | 불가 | — |
| 공개 데이터 다운로드 (108 GB) | 디스크, 네트워크, 다운로드 승인 | 조건부 | 조건부 (승인 후) | 다운로드 시간은 네트워크에 따라 다름 |
| 데이터 처리 (STEP2, PyBullet IK, hdf5 생성) | 원시 zip 42 GB + CPU | 조건부 | 조건부 (패키지 설치 승인 후, 포장 과제만) | 1–2일 (추정) |
| 학습 (DP-PointNet, 3000 에폭, 배치 16) | 긴 GPU 학습 | 이 저장소의 원칙상 GPU 재현은 하지 않음 | 보류 | MPS 속도 측정 전 판정하지 않음 |
| 표 재계산과 검정 | 없음 | 가능 | **완료** (`03`) | — |
| 논문 기본 모델 (Perceiver) 구현 | 공개 코드 없음 (D3) | 가능 (직접 구현) | 시작 가능 (직접 구현) | 1–2주 (추정, 논문에 인코더 세부 설정이 일부만 있어 불확실) |

- **재현 우선순위:** (1) 공개 데이터 중 포장 과제의 hdf5 생성 파이프라인을 CPU에서 확인 (설치 승인 필요). (2) 점군 인코더 차이 (PointNet 대 Perceiver)의 영향을 표 I의 Plate 과제 규모로 볼 것. 이 실험은 학습이 필요하므로 별도 승인 후에만 합니다.
- **실로봇 결과와 교정 효과는 이 저장소만으로 재현할 수 없습니다.**

## 5. 실행 기록

- 공식 저장소의 코드는 **실행하지 않았습니다.** 읽기와 `grep`만 했습니다.
- 공개 데이터와 가중치는 받지 않았습니다. 크기만 Hugging Face 메타데이터에서 확인했습니다.
- 패키지는 설치하지 않았습니다.
