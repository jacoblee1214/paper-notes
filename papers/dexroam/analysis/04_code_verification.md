# 04. 논문 ↔ 공식 코드 대조: DexRoam (정적 읽기만)

공식 저장소 두 개를 얕게 clone해 **읽기만** 했습니다. 실행, import, 설치, 체크포인트 로딩, 데이터·가중치 다운로드는 하지 않았습니다. 저장소 내용은 신뢰할 수 없는 입력으로 취급했습니다.

| 저장소 | 커밋 (전체 해시) | 날짜 (UTC) | 라이선스 | 규모 |
|---|---|---|---|---|
| zhourui9813/DexRoam (수집 + 정렬, `repo/`) | `e539387a847ac3f70285bfd9ceb35cfff98db614` | 2026-09-29 09:59 | MIT | 커밋 3개, 별 9, clone 약 293 MB(에셋 포함) |
| zhourui9813/DexRoam-Policy-Training (π0.5 + GR00T 학습, `repo_policy/`) | `a82d2c98dc0dff00c17e1aa4fb2bb93d1c5655fe` | 2026-09-29 05:25 | MIT | 커밋 2개, 별 4, clone 약 20 MB |

두 저장소는 논문 v1 제출(2026-09-28) 다음 날 올라왔고 그 뒤 코드 변경은 README 한 줄입니다. 대조 기준은 논문 v1 본문과 부록 표입니다. `repo/`와 `repo_policy/`는 위 URL을 `git clone --depth 1`로 받아 위 커밋으로 checkout한 상태입니다. 분석 스크립트(`analysis/recompute.py`의 V15)는 `repo_policy/`를 읽습니다.

## 1. 공개 범위

| 논문의 요소 | 저장소 | 상태 |
|---|---|---|
| 사람 전신·손 수집 (Quest 3 + ZED Mini, 제스처 제어, HDF5) (§III-A3, 부록 A2) | DexRoam | 있음 (`ego_wholebody_mocap/`, 실행 진입점 `python -m ego_wholebody_mocap`) |
| 마운트 3D 모델, 하드웨어 설명 | DexRoam, 프로젝트 페이지 | 있음 (`assets/3d_model/`, Google Docs 링크) |
| 몸체 정렬: 루트 프레임, 상체 목표, 머리, 베이스 (§III-B1) | DexRoam | 있음 (`human_robot_alignment/dexroam_alignment/geometry.py`) |
| 손 리타게팅 (dex-retargeting) | DexRoam | 있음 (`hands.py`, `configs/xhand_*.yml`, 정지한 v0.5.0 스냅샷) |
| **행동 의미 정렬: 상대 행동 `ΔT` 생성 (§III-B2)** | - | **없음** (`04` D1) |
| 시간 정렬: 과제 진행 재표집 (§III-B3) | DexRoam | 있음 (`resampling.py`) |
| 로봇 시연 수집(텔레옵) (§III-A2) | - | 없음 (샘플 HDF5 데이터만 공개) |
| 데이터 변환 (HDF5 → LeRobot v2.1) | Policy-Training | 있음 (`data_scripts/dexroam_hdf5_to_lerobot.py`) |
| GR00T N1.7 미세조정 (표 IX) | Policy-Training | 있음 (`dexroam_gr00t/`, 모달리티 설정, 학습·서버 스크립트) |
| π0.5 미세조정 (표 VIII) | Policy-Training | 있음 (`dexroam_pi05/`, 학습 설정, 서버 스크립트) |
| 세 훈련 전략(Cotrain, Pretrain + FT, Pretrain + Cotrain FT)의 데이터 혼합·일정 | - | 없음 (단일 데이터셋 경로만 지원) |
| 궤적 재생 평가(표 I, II), PICO + 트래커 경로(표 IV) | - | 없음 |
| MMD, SWD, PCA(그림 5, 6), LDLJ·SPARC(표 V), TCS 평가(표 X) | - | 없음 |
| 실로봇 정책 실행(클라이언트), Astribot 제어 | - | 없음 (정책 서버 스크립트만) |
| 데이터 | HF | 로봇 시연 50개(Push Chair & Close Laptop) 하나. 사람 데이터와 나머지 4과제는 없음 |

## 2. 항목별 대조 (D1–D16)

범례: 일치 = 논문과 코드가 같음, 차이 = 서술이나 값이 다름, 빈칸 = 논문에 없고 코드에만 있거나 그 반대, 미공개 = 코드에 없어 확인할 수 없음.

| ID | 논문 | 코드 | 판정 |
|---|---|---|---|
| D1 | §III-B2: 몸통, 양 손목은 `ΔT_t = (^H T_t)^{-1} ^H T_{t+1}`, 베이스는 연속 루트의 평면 변화 `Δξ`로 상대 행동을 만든다. 절제 표 III에서 이 단계를 빼면 0% | `dexroam_alignment/`의 출력은 `fk_poses_dict_from_state`(절대 자세 pose7)와 `mobile_dict/absolute_position`(절대 베이스 x, y, yaw)뿐이다. 문서(`Human_Robot_Alignment_Reference.md` §1)는 "행동 의미 정렬은 하류 데이터 변환에서 적용"이라 쓰는데, 변환 스크립트는 입력 HDF5에 이미 있어야 하는 `/fk_poses_relative_action/*`와 `/mobile_dict/relative_position`을 읽을 뿐 만들지 않는다. 두 저장소 전체를 검색해도 이 키를 쓰는 코드가 없다. 시간 재표집은 이 두 그룹을 명시적으로 건너뛴다(`ACTION_GROUP_NAMES`) | **미공개** (논문의 핵심 단계 하나가 코드에 없음). 절대 자세를 재표집한 뒤 로봇 속도에서 차분하는 순서는 §III-B3의 식(자세를 보간)과 맞지만, 상대 행동 계산 코드를 확인할 수 없다 |
| D2 | §III-B3: `N̄_r` = 그 과제 로봇 에피소드의 평균 길이로 사람 궤적을 재표집. 위치 선형, 회전 SLERP, 이미지·이산 신호 최근접, 베이스 yaw는 unwrap 뒤 보간 | `resampling.py`: `--robot-dir`의 에피소드 길이 평균을 `floor(mean + 0.5)`로 반올림(최소 2). 위치 `np.interp`, 쿼터니언은 부호를 맞춘 뒤 SLERP, 이미지와 비부동소수 데이터는 `_nearest_indices`, 베이스는 `np.unwrap` 뒤 선형. 손 관절(부동소수)은 선형. 코드에만 있는 모드: 에피소드별 `--speed-factor`, 고정 `--target-length` | **일치**와 빈칸. 두 추가 모드는 논문에 없다 |
| D3 | §III-B1: F_H의 원점은 골반의 지면 투영, 앞축은 골반 로컬 전방의 수평 투영, 수직축은 중력 반대 | `compute_human_root_frame`: `hips` 관절 행렬의 로컬 z 열을 수평 투영해 앞축, 위치의 z를 0(`ground_z`)으로 둔 것이 원점, 세계 위쪽은 `WORLD_UP = (0, 0, 1)`. 투영이 0이면 `chest`, 이전 앞축 순으로 대체 | **일치**. 고장 대비(대체 앞축)는 논문에 없다 |
| D4 | §III-B1: 머리는 F_H 기준 yaw, pitch만 2자유도 명령 | `compute_head_rpy`: 로봇 베이스 프레임으로 회전을 바꾼 뒤 `xyz` 오일러에서 `[yaw, pitch]`를 `wrap_to_pi`로 반환, roll은 버림 | **일치** |
| D5 | §III-B1: 손은 dex-retargeting으로 12자유도 XHand 명령. 하이퍼파라미터는 없음 | `configs/xhand_left.yml`: 벡터 리타게팅, 손목에서 다섯 손끝으로 가는 벡터, `scaling_factor 1.2`, `normal_delta 0.008`, `huber_delta 0.02`, `low_pass_alpha 0.5`. 손 방향은 `VR_TO_MANUS_ROT_MATS` 상수로 MANUS 규약에 맞추고 쿼터니언을 프레임 간 평활. 팔·몸통 목표의 손목 로컬 y 보정(±90°)도 상수 | **빈칸**. 논문에 없는 설정값. 손 모양 차이(1.2배 스케일 등)가 12차원 손 행동 분포에 들어간다 |
| D6 | 표 VI: 53차원 상태 = 팔 2개 + 몸통 (위치 + 6D 회전 9차원씩), XHand 관절 12 x 2, 머리 관절 2. 부록 D1: 단일 프레임. §II: 상태 **이력** `P_t` | 변환 스크립트: 상태는 `fk_poses_dict_from_state`의 절대 자세 3개(위치 + rot6d), `dex_hand_dict/*/state`(도 → 라디안) 24, 머리 2차원 = **`joints_position_command`의 마지막 두 열**(명령 열)이며 `state_history_length 1`. 사람 에피소드에서는 자세가 루트 프레임 기준 로봇 베이스 프레임의 절대 값 | **일치**(53차원, 단일 프레임). 차이: ① §II의 "상태 이력"은 코드·부록에서 단일 프레임. ② 사람 에피소드의 상태에 시연자별 신체 차이가 남는다(행동은 상대, 상태는 절대). ③ 머리 "관절 상태"는 명령 열 |
| D7 | 표 VII: 56차원 행동 = 손 12 x 2(관절 목표), 팔 2개와 몸통 9차원씩(상대 이동 + 6D 회전), 머리 2(관절 목표), 베이스 3 | 변환: 손 행동 = `dex_hand_dict/*/state`의 **다음 스텝 값**(마지막 스텝은 반복), 팔·몸통 = `fk_poses_relative_action`(xyz + quat → rot6d), 머리 = `joints_position_command[:, -2:]`의 다음 스텝, 베이스 = `mobile_dict/relative_position`. 순서와 차원(12, 12, 9, 9, 9, 2, 3)은 일치 | **일치**(차원). 해석: 로봇 데이터의 손 행동은 측정된 XHand 관절 피드백의 다음 값이고(부록 A1), 사람 데이터의 같은 키는 리타게팅한 손 명령이다. 두 신호의 성질이 다르다. 논문의 한계 절(손 명령을 MANUS 관절각으로 지도하면 도움이 될 수 있다)과 같은 지점 |
| D8 | §III-B4, 표 VIII: π0.5는 전 파라미터 학습, 행동 입출력 투영은 무작위 초기화, 표 VIII의 학습률 2.5e-5, 웜업 1,000 스텝, 가중치 감쇠 1e-10, 30,000 스텝, 전역 배치 4 | `dexroam_train_config.py`: `action_dim 56`, `action_horizon 32`, `max_token_len 256`, `batch_size 4`, `HighDimActionWeightLoader`. 학습 기본값은 `TrainConfig`: 학습률 코사인 2.5e-5(웜업 1,000, 끝 2.5e-6), AdamW 가중치 감쇠 1e-10, 기울기 클립 1.0, **EMA 0.99**, `num_train_steps 30,000`, `freeze_filter` 없음(전부 학습) | **일치**와 빈칸. 행동 호라이즌 32(30 Hz에서 약 1.07 s), EMA 0.99, 학습률 끝값은 논문에 없다 |
| D9 | §III-B4: π0.5는 스테레오 두 시점을 입력 | `DexRoamInputs`: 왼쪽 이미지를 `base_0_rgb`에, **오른쪽 이미지를 `left_wrist_0_rgb`**에 넣고 `right_wrist_0_rgb`는 0과 마스크 | **차이(세부)**. 사전학습 π0.5가 손목 카메라로 해석하는 슬롯에 오른쪽 눈 영상이 들어간다. 이 배정이 성능에 영향을 주는지는 논문에 없다 |
| D10 | 표 IX: GR00T N1.7-3B, GPU 4개, 전역 배치 24(GPU당 6), 40,000 스텝, 학습률 1e-4, 가중치 감쇠 1e-5, 웜업 비율 0.05(2,000 스텝). LLM·비전 동결, 프로젝터와 확산 디코더 학습 | `run_train_gr00t.sh`: `CUDA_VISIBLE_DEVICES=0`(GPU 1개), `GLOBAL_BATCH_SIZE=12`, `MAX_STEPS=40000`, `SAVE_STEPS=10000`. 저장된 실행 설정: 학습률 1e-4, 가중치 감쇠 1e-5, 웜업 비율 0.05, 코사인, bf16, DeepSpeed stage 2, `tune_llm false`, `tune_visual false`, `tune_projector true`, `tune_diffusion_model true`, **`tune_vlln true`**, 색 지터(밝기 0.3, 대비 0.4, 채도 0.5, 색조 0.08), 상태 드롭아웃 0.2, 어텐션 드롭아웃 0.2 | **차이**: GPU 4개 x 6 대신 1개, 배치 12(표 IX는 24). 배치 크기가 다르면 같은 스텝 수의 학습량이 다르다. **빈칸**: `tune_vlln`, 색 지터, 드롭아웃은 논문에 없다 |
| D11 | 부록 D3: 행동 청크. 본문은 크기를 밝히지 않음("행동 청크 크기는 바꾸지 않는다") | GR00T 모달리티 설정의 행동 `delta_indices = range(32)`(32스텝), 모델 설정의 `action_horizon 40`. 팔·몸통·베이스 행동 표현은 `DELTA`로 표기되는데, 이 표현의 변환은 처리기에서 `RELATIVE`일 때만 일어나고 코드에는 `DELTA` 분기가 없다(데이터가 이미 상대이므로 그대로 통과) | **빈칸**(청크 32스텝 ≈ 1.07 s, 코드 확인). `DELTA` 표기는 변환이 없다는 뜻으로 읽힘 |
| D12 | §III-B4: 세 훈련 전략과 "같은 하류 미세조정 예산". Q3: 사람 50개 + 로봇 {0, 10, 25, 50} | 학습은 하나의 LeRobot 데이터셋 경로(`mix_ratio 1.0`)만 받는다. 사람 에피소드를 같은 변환기로 바꾸려면 D1의 입력 키가 필요하다. 사전학습 → 미세조정 일정, 혼합 비율, 사람 표지(임베딩, 토큰)는 코드에 없다. 변환기는 한 번 실행에서 모든 에피소드에 같은 과제 문장을 붙인다 | **미공개**. 세 전략의 구현 차이를 코드로 확인할 수 없다 |
| D13 | §III-A3: ZED HD1080, 30 fps, 헤드셋 표시 약 20 Hz, 정책 입력은 스테레오 쌍 | `config.py`: `ZED_RES "HD1080"`, `CAMERA_CAPTURE_FPS 30`, `VR_PUSH_FPS 20.0`. **`STREAM_MODE = "left"`**가 기본값(선택지 left, right, both). 저장은 JPEG 품질 85 | **일치**와 차이(세부). 기본 설정에서 헤드셋에 왼쪽 눈 영상만 보인다. 정책은 스테레오 두 눈을 쓴다. "수집자가 보는 것 = 정책 관측"은 왼쪽 영상까지는 같다 |
| D14 | 표 IV, §III-B1: PICO + 모션 트래커 5개와의 비교 (추적기 기반 경로) | 코드 없음. 몸체 정렬 입력은 Quest 몸 관절 `hips`, `chest`, 손목 관절을 가정(`REQUIRED_JOINT_NAMES`) | **미공개** |
| D15 | 부록 E: MMD(RBF, 중앙값 대역폭), SWD(L = 128), PCA, TCS, 재생 지표 | 평가 코드 없음 (저장소 검색에서 `mmd`, `wasserstein`, `ldlj`, `sparc`, `tcs` 계열 구현 없음) | **미공개**. 그림 5, 6과 표 V, X는 재계산할 수 없다(표 값 재계산은 `03`) |
| D16 | 저장소 README | ① 정렬 README는 `docs/`, `tests/` 디렉터리를 구조에 적지만 clone에는 둘 다 없다(`docs/`는 저장소 루트에만). ② 정렬 요구 사항은 `dex_retargeting==0.4.6`을 고정하는데 포함된 스냅샷은 0.5.0이다. ③ `run_train_gr00t.sh`는 `--modality-config-path MODALITY_CONFIG`에서 변수 `$`를 빠뜨려(정의한 `MODALITY_CONFIG`가 전개되지 않음) 경로가 문자열 그대로 전달된다(런처는 없는 경로에서 `FileNotFoundError`를 내므로, 정적 판단으로는 이 스크립트를 그대로 쓰면 실패). ④ 체크포인트 폴더에 실행 설정 두 벌(`DEXROAM_GROOT_N1.7`, `EXP_NAME`)이 있고 둘째는 다른 데이터셋 경로를 가리킨다 | 차이(문서와 파일). 실행하지 않았으므로 ③은 정적 판단 |

## 3. 재생산 점검에서 눈에 띈 점

- **정렬의 입출력 계약이 코드 밖에 있다.** 정렬 출력(절대 자세, 절대 베이스)과 변환기 입력(상대 행동, 상대 베이스)이 한 단계 비어 있어, 사람 데이터를 정책 학습까지 연결하는 경로가 공개 코드만으로는 이어지지 않습니다(D1).
- **로봇과 사람 데이터의 손 신호가 다릅니다.** 변환기는 같은 키(`dex_hand_dict/*/state`)를 두 데이터에서 쓰지만 로봇은 관절 피드백, 사람은 리타게팅 명령입니다(D7). 접촉에서 로봇 손 상태가 포화된다는 논문의 한계와 같은 지점입니다.
- **정책 학습 쪽은 표와 큰 틀에서 일치합니다.** 차원(53, 56), 학습률, 웜업, 가중치 감쇠, 스텝 수, 동결 구성이 표 VIII, IX와 맞습니다(D8, D10). 다른 것은 GPU 수와 배치 크기(D10)입니다.
- **공개 데이터는 한 과제입니다.** HF의 로봇 에피소드 50개(Push Chair & Close Laptop, 파일 크기 합계 약 244 GB, 받지 않음)가 전부이고, 사람 데이터는 없어서 정렬 → 학습 경로를 공개 자료만으로 끝까지 따라가 볼 수 없습니다.

## 4. 구현할 수 있는지 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**판정: 논문 수치의 재현은 이 맥에서 불가능합니다. 정렬 파이프라인의 기하·재표집 부분은 CPU에서 구현·검증할 수 있고, 정책 학습·수집·실로봇은 GPU, 하드웨어, 데이터가 필요합니다.**

| 구성 | 이 맥에서 | 필요한 것 | 근거 |
|---|---|---|---|
| 사람 수집 (Quest 3 + ZED Mini) | 불가 | 하드웨어, ZED SDK 4.x(NVIDIA CUDA, Ubuntu 20.04/22.04 전제), ADB | README, D13 |
| 몸체 정렬의 기하(루트 프레임, 상대 변환, 머리), 시간 재표집 | 가능 (numpy, scipy, h5py는 설치됨) | 합성 또는 공개된 HDF5 형식 데이터. 공식 코드를 그대로 돌리면 `cv2`(영상 디코딩), 손 리타게팅은 `pin`·`sapien`이 필요해 새 설치가 필요하고, 우리는 설치도 실행도 하지 않았다 | D1, D2, D3 |
| 상대 행동 생성(`ΔT`) | 가능 (직접 구현) | 논문 §III-B2의 식. 공식 코드에는 없음 | D1 |
| 손 리타게팅 | 조건부 | `pin`, `sapien`(macOS arm64 휠 유무 미확인), XHand URDF는 저장소에 있음 | D5 |
| 데이터 변환 (HDF5 → LeRobot) | 조건부 | `ffmpeg`(libx264, 이 맥에는 없음), 사람·로봇 HDF5 | README |
| π0.5 미세조정 | 불가 | JAX + NVIDIA GPU(`uv sync`), 스텝 30,000, 전역 배치 4. 논문은 GPU 종류를 적지 않음 | D8, 표 VIII |
| GR00T N1.7 미세조정 | 불가 | CUDA, flash attention, DeepSpeed, 모델 3B + Cosmos-Reason2-2B. MPS로는 이식 대상이 아님 | D10 |
| 공개 데이터 | 불가(저장 공간) | 약 244 GB(HF 파일 크기 합계) | 위 표 |
| 실로봇 평가 (Astribot + XHand) | 불가 | 로봇 하드웨어, 제어 스택(미공개) | D14, D15 |
| 지표(MMD, SWD) 재구현 | 가능 | 부록 E의 식으로 numpy. 데이터는 없으므로 합성 데이터로만 (`03` §6 토이) | D15 |

**노력 추정(측정 없음):**
- 정렬 파이프라인의 numpy 재구현과 합성 검증: 2–4일.
- 사람 상대 행동 + 시간 재표집 효과를 보는 소규모 폐루프 시뮬레이션(MuJoCo + 작은 정책, CPU): 2–4주(규모를 키우면 더).
- 논문 수준의 전체 재현(로봇 하드웨어와 사람 수집 포함): 하드웨어 구축 수 주 + 과제별 시연 수집(과제당 사람·로봇 각 50개, 총 약 500개, 시연 수 50/50 가정) + 정책 학습(GPU 종류·시간 미상). 공개 자료만으로는 어렵습니다.

## 5. 저자에게 물을 것

1. 행동 의미 정렬(`ΔT` 생성, `fk_poses_relative_action`, `relative_position`)을 만드는 코드는 어디에 있는가(D1)? 시간 재표집과 차분의 순서는?
2. 주 결과(그림 7)의 로봇·사람 시연 수는 몇 개인가? 시행은 같은 초기 조건을 재사용하는가(`03` V14)?
3. 그림 9의 백본과 원점 값이 그림 7과 다른 이유는 같은 설정의 다른 평가 세션인가(`03` V9)? 표 III의 Throw Trash Cotrain이 .65(그림 7은 .60)인 이유는(`03` V7)?
4. 그림 8의 π0.5 Throw Trash 50개 로봇 칸 값이 그림 7과 다른 이유는(`03` V8)?
5. 세 훈련 전략의 사전학습·미세조정 스텝, 혼합 비율은?
6. GR00T 학습의 GPU 수와 전역 배치는 표 IX(4, 24)와 공개 스크립트(1, 12) 중 어느 쪽인가(D10)?
7. π0.5에서 오른쪽 눈 영상을 `left_wrist_0_rgb` 슬롯에 넣는 선택의 영향은 점검했는가(D9)?
8. 사람 에피소드 상태의 절대 자세는 시연자 신체 차이를 어떻게 다루는가(D6)?
