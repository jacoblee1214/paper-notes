# 04. 공식 코드 대조와 구현 가능성

대상: `repo/` (github.com/j96w/MimicPlay, 커밋 `334ad94`, 2024-04-23, MIT). **정적 읽기만** 했다. 설치, 실행, import를 하지 않았다. 내려받은 내용은 신뢰하지 않는 데이터로 다뤘다.

## 1. 확인한 범위

- 파일: `mimicplay/algo/mimicplay.py`(764줄), `mimicplay/models/obs_nets.py`, `mimicplay/configs/{highlevel,highlevel_human,lowlevel,BC_*}.json`, `mimicplay/scripts/dataset_extract_traj_plans.py`, `mimicplay/scripts/run_trained_agent*.py`, `mimicplay/scripts/human_playdata_process/demo_mp4.py`, README.
- 검색: `KL`, `divergence`, `triangul`, `cuda`, `ac_dim`, `hand_loc`, 기준선 이름.
- 저장소 크기 83 MB(얕은 복제). 체크포인트와 데이터는 저장소에 없고 Google Drive 링크뿐이다.

## 2. 논문과 코드의 차이 (D1–D10)

| 번호 | 논문 | 공식 코드 | 영향 | 판정 |
|---|---|---|---|---|
| D1 | 잠재 계획기의 KL 정렬 손실 L_KL = D_KL(Q^r ‖ Q^h), λ = 1000 (§3.2, 식 3–4, 표 5) | **KL 항이 없다.** `KL`, `divergence`, `kl_loss` 검색 결과 구현이 없다. `vae.kl_weight`(`mimicplay_config.py` 61행)는 VAE 설정이고 `vae.enabled = false`다 | Table 2의 "w/o KL" 비교는 공개 코드로 만들 수 없다. 사람 데이터 쪽 학습이 공개 설정과 다를 가능성이 있다 | **미구현** |
| D2 | 사람 3D 손 궤적: 두 대의 보정된 카메라 삼각측량 (§3.1, 부록 D) | `demo_mp4.py`는 두 시점 mp4를 받아 hand_object_detector로 손 상자를 얻는다(`hand_loc`). **삼각측량 코드는 없다**(`triangul` 검색 없음). 카메라 보정 파일도 없다 | 논문의 3D 궤적을 공개 코드로 똑같이 만들 수 없다 | **부분 구현** (2D 검출까지) |
| D3 | 목표 프레임: 현재에서 H 스텝 뒤, H ∈ [200, 600] (§3.2, 부록 A) | `configs/highlevel*.json`과 `lowlevel.json`: `goal_image_range [100, 200]`, `eval_goal_gap 150` | 학습 때 목표 간격이 논문의 절반 이하다. 증강 효과(§3.2)의 범위가 달라진다 | **불일치** |
| D4 | 계획 궤적의 미래 구간 길이 (본문에 명시 없음) | `dataset_extract_traj_plans.py`: 15프레임 간격 x 10점 = 미래 150프레임. 로봇 데이터 20 Hz에서는 7.5초, 사람 데이터 60 fps에서는 2.5초 | 같은 "150프레임"이 두 데이터에서 다른 시간 길이를 가진다. 논문 본문에서는 확인 불가 | 확인 불가 (본문 누락) |
| D5 | 학습 예산: 100k 반복, 배치 16, 에폭 1000, LR 감쇠 [300, 600] (§A, 표 5–6) | `highlevel.json`: 배치 32, 에폭 3000, `epoch_every_n_steps 100`, `epoch_schedule []`(감쇠 없음), LR 1e-4 | 학습량과 배치가 논문 표와 다르다. 공개 기본값이 논문 설정과 같다고 볼 수 없다 | **불일치** (재현 시 값을 명시해야 함) |
| D6 | 실로봇 제어: Franka, OSC, Deoxys, RoboTurk 전화 원격 조작, 17 Hz (§A) | **시뮬(LIBERO)만 있다.** `deoxys`, `franka`, `roboturk` 검색은 README 문장에서만 나온다 (코드 0건) | 표 1–3 (실로봇) 재현 경로가 없다 | **없음** |
| D7 | 학습과 추론은 GPU (논문 "single GPU machine") | `.cuda()` 호출 32곳: `mimicplay.py` 6, `GPT.py` 8, `demo_mp4.py` 9, `demo_hdf5.py` 9. 설정 `cuda: true`. `goal_image` 로드도 `.cuda()` 고정 | macOS에서는 MPS로 바꿔야 한다. 기기 중립 코드가 아니다 | **이식 필요** |
| D8 | 시뮬 평가: 5 시드 x 100 시도 (부록 C) | `run_trained_agent.py`: `--n_rollouts 100 --horizon 2000 --seed 0`, 영상 fps 20 | 평가 설정은 논문과 같다 | **일치** |
| D9 | 저수준 정책: GPT 4층, 4헤드, 블록 10, 임베딩 656, GMM K = 5 (표 6) | `lowlevel.json`: `n_layer 4`, `n_head 4`, `block_size 10`, `feat_dim 656`, `gmm_modes 5` | 구조가 같다 | **일치** |
| D10 | 계획기 GMM K = 5, 인코더 ResNet-18 (표 5) | `highlevel.json`: `gmm.num_modes 5`. 인코더는 robomimic `VisualCore`/`ResNet18Conv`(`obs_nets.py` 24–26행) | 일치 (인코더 깊이는 기본값을 따른 것으로 보임, 세부 확인 안 함) | **일치 (부분 확인)** |
| D11 | 사람 데이터 행동 차원 (논문 명시 없음) | `highlevel_human.json`의 `ac_dim 40`, 로봇은 30. 행동 차원은 데이터 `actions` 형태에서 읽는다(`file_utils.py` 116행) | 40이 어떤 좌표를 담는지 코드에서 확정하지 못함 (추정: 10점 x 4차원) | 확인 불가 |
| D12 | 기준선: GC-BC (BC-RNN), GC-BC (BC-trans), C-BeT, LMP, R3M-BC (§4) | BC-RNN-GMM(`BC_RNN_GMM`), GC-BC(BC-trans)(`BC_trans_scratch.json`)만 있다. C-BeT, LMP, R3M 전용 코드는 없다 | 시뮬 표 4에는 기준선 두 개만 있어 논문 본문과 맞는다. 실로봇 기준선은 재현 못 함 | 부분 (두 개) |
| D13 | 체크포인트와 데이터 | README: Google Drive 링크. 저장소에 없음 | 공개 체크포인트로 표 4 추론은 가능할 수 있으나 다운로드하지 않았다 | 다운로드 안 함 |

## 3. 대조에서 본 것 (정적 읽기)

- `mimicplay.py`의 계획기 클래스 `Highlevel_GMM_pretrain`는 BC_Gaussian을 상속하고 GMM 정책만 쓴다. KL 항, 사람-로봇 특징 쌍, 영역 정렬 배치가 없다. 계획기는 데이터셋 하나(시뮬에서는 로봇 시연)로 학습한다.
- 저수준 `MimicPlay` 클래스(`_forward_training`)는 `human_nets`에서 잠재 계획을 `torch.no_grad()`로 뽑아 입력에 붙인다. 계획기 동결 구조가 논문과 맞다(§3.3).
- 시뮬 표 4 설정은 "같은 로봇 데이터로 계획기와 정책을 함께 학습"이다(README와 부록 C). 실로봇 설정(사람 play로 계획기)과 다르다. 따라서 시뮬 결과는 논문의 핵심 주장(사람 play의 효과)을 직접 검증하지 못한다. 이 점은 README 설치 절과 논문 부록 C가 모두 밝힌다.

## 4. 구현 가능성 판정 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**판정: 전체 재현은 불가능합니다. 시뮬 평가와 축소판 구현은 조건부로 가능합니다.**

| 항목 | 맥에서 가능한가 | 근거 | 필요 조건 |
|---|---|---|---|
| 실로봇 실험 (표 1–3) | **불가** | D6, 실로봇 장비 없음 | Franka, 제어 스택, 사람 play 수집 장치 |
| 공식 학습 (계획기와 정책) | **조건부 (MPS 이식 후)** | D7 (`.cuda()` 32곳, Python 파일 전체), D5 (설정 불일치) | 이식 코드, 학습 시간 측정 전 (CPU/MPS 측정 없음) |
| 공식 시뮬 평가 (LIBERO, 표 4) | **조건부** | D8 설정 일치. 설치 필요 (robomimic 브랜치, robosuite 1.4.1 LIBERO 브랜치, LIBERO, BDDL). 체크포인트 다운로드 필요 | 패키지 설치 승인, 체크포인트 다운로드 승인, 크기 미확인 |
| 사람 데이터 처리 (손 검출) | **불가 (현재 형태)** | hand_object_detector가 PyTorch 1.0.1 / CUDA 10 전제 (공식 README) | 새 환경 구성, 모델 가중치 다운로드 |
| 직접 짠 축소판 (잠재 계획 + GMM 헤드, 장난감 손 궤적) | **가능** | CPU/MPS에서 작은 모델로 가능 | 설치는 numpy/torch 정도. 승인 후 |
| KL 정렬 효과 검증 (D1) | **가능 (축소판)** | 두 영역 특징 분포를 합성해 KL 항의 효과를 본다 | 직접 구현 (공식 코드에 없음) |

**노력 추정 (측정 전):**
- 시뮬 평가 한 과제 설치와 추론: 1–2일 (추정).
- MPS 이식 후 학습 (계획기 + 정책): 1–2주 (추정). 학습 시간은 측정하지 않았다.
- 표 4 전체 재현 (5 시드 x 100 시도, 기준선 2 x 시드 5): 시간 측정이 없어 추정하지 않는다.
- 실로봇: 저장소 밖 장비가 필요해 맥에서 불가.

## 5. 판정 근거에 대한 한계

- 체크포인트와 데이터를 받지 않았다. 따라서 공식 평가가 논문 수치를 내는지 확인하지 못했다.
- 라이선스: 저장소는 MIT, 논문은 arXiv 비독점 배포 라이선스(`README.md`의 라이선스 절 참조). 코드를 가져갈 때는 MIT 조건(저작권 고지)을 지킨다.
