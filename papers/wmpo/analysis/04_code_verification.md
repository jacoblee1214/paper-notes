# 04. 논문 ↔ 공식 코드 대조 (D1–D9)와 구현 가능성

대상은 `WM-PO/WMPO` 커밋 `c836d74` (2026-01-04, Apache-2.0)입니다. `repo/WMPO/`를 얕게 clone해 **정적으로만** 읽었습니다. 아무것도 실행, import, 설치하지 않았습니다. 경로는 저장소 루트 기준입니다. 체크포인트(세계모델 364 GiB, 데이터 530 GiB 규모)는 받지 않았습니다. 그래서 세계모델 학습 설정 파일 `train_mimicgen_cfg.py`를 보지 못했고, 그 내용을 요구하는 항목은 "미확인"으로 적었습니다.

## 0. 요약

- **확인됨:** GRPO 하이퍼파라미터(표 4), KL 제거, 그룹 크기 8, 동적 샘플링, 보상 모델의 8프레임 슬라이딩 윈도우 판정, SDXL 계열 2D VAE.
- **차이 또는 빈틈:** 과제별 보상 임계값 (논문에 값이 없고 선택 과정도 없음), 임계값 선택에 평가 결과 파일을 쓰는 스크립트, RL 중 세계모델을 고정하는 설정, 라벨 파싱이 두 스크립트에서 다름.
- **미확인:** 노이즈 프레임 조건화(50/1000)와 프레임 단위 AdaLN 행동 제어. 공개 벤더 OpenSora에서 대응 구현을 찾지 못했고, 세계모델 설정 파일이 체크포인트 안에 있습니다.

## 1. 대조 표

| # | 논문 주장 (위치) | 공식 코드 (파일) | 판정 |
|---|---|---|---|
| D1 | 보상 임계값은 "검증 실험으로 선택" (§3.3). 값은 없음 | `examples/mimicgen/*/train_wmpo_{128,1280}.sh`의 `actor_rollout_ref.wm.rm_threshold`: coffee 0.82, square 0.96, stack_three 0.66, three_piece_assembly 0.62. `verl/trainer/config/ppo_trainer.yaml:30`의 기본값 0.89 | **차이 (보고 누락).** 과제별 고정값이 논문에 없고, 기본값 0.89와도 다름. 표 1 수치가 어느 임계값에서 나왔는지 본문만으로는 알 수 없음 |
| D2 | 같은 임계값 선택 (§3.3) | `reward_model/find_thre.py`는 `results/final_results_thre_0.50.json`을 읽고 임계값 30개(0–1)를 훑어 정확도, F1을 계산. `reward_model/inference_videomae.py`는 `--threshold` 기본값 0.06, 함수 기본값 0.93, 실행 주석에는 0.1–0.6을 적음 | **선택 편향 가능성.** 임계값을 평가 결과 파일에서 고르는 경로가 코드에 있음. 검증 세트와 평가 세트가 분리되었는지 확인할 수 없음 |
| D3 | 라벨 파싱 (§3.3, 학습 데이터 라벨) | `reward_model/inference_videomae.py`는 `video_path.split('=')[-1]`을 읽고 `== "False"`를 반전해 쓰고, `find_thre.py`는 `split("_")[-1] == '1'`로 읽음 | **불일치 위험.** 두 스크립트가 같은 파일 이름 규칙을 쓰는지 확인할 수 없음. 라벨 방향이 뒤집히면 F1 해석이 달라짐 |
| D4 | GRPO 설정 (표 4, 부록 A) | 학습률 `optim.lr=5e-6`, 클리핑 `clip_ratio_low=0.2`, `clip_ratio_high=0.28`, 온도 `rollout.temperature=1.6`, 그룹 `data.n_samples=8`, 배치 `train_batch_size=64`, 미니배치 `ppo_mini_batch_size=128`, KL `algorithm.kl_ctrl.kl_coef=0.00`. `data.filter_accuracy=True`가 동적 샘플링을 켬 | **확인.** 표 4의 값과 모두 일치. 논문에 없는 설정: `grad_clip=1`, `traj_mini_batch_size=16`, `rollout.micro_batch_size=2`, `num_steps_wait=10`, `total_epochs=100` |
| D5 | 보상 모델: VideoMAE, 길이 8 클립, 임의 클립이 임계값을 넘으면 성공 (§3.3) | `reward_model/inference_videomae.py`: `window_size=8`, `stride=1`, 시그모이드 확률, `label = 1 if any(...)`. `VideoMAEConfig.from_pretrained("MCG-NJU/videomae-base", num_frames=8, num_labels=2)`. 학습 스크립트 `rm_img_size=224` | **확인.** 논문 설명과 일치 |
| D6 | 세계모델: OpenSora 백본, 3D VAE를 SDXL 2D VAE로 교체 (§3.2) | `dependencies/opensora/opensora/models/vae/video_sdxl/`가 존재. OpenSora STDiT(`stdit/stdit3.py`) 존재 | **확인 (구조 수준).** 교체된 VAE를 실제로 쓰는 설정은 세계모델 설정 파일에 있으며 미확인 |
| D7 | 노이즈 프레임 조건화, 50/1000 (§3.2) | 벤더 OpenSora 코드에서 조건 프레임 노이즈 주입 경로를 찾지 못함 (`grep -i "cond_noise\|noise_level\|noisy_cond"` 무결과) | **미확인.** 학습 설정 파일 `checkpoint_files/world_models/*/train_mimicgen_cfg.py`가 체크포인트 번들 안에 있어 받아야 확인 가능 |
| D8 | 프레임 단위 AdaLN 행동 제어: 프레임마다 γ, β, α 생성 (§3.2 식) | `dependencies/opensora/opensora/models/layers/blocks.py`, `stdit/stdit3.py`에 modulate 함수 존재. 프레임 단위 α를 행동에서 만드는 코드는 확인하지 못함 | **미확인 (정적 검색 한계).** 같은 이유로 D7과 함께 세계모델 설정이 필요 |
| D9 | RL 중 세계모델을 고정하지 않고 "정책 행동 정렬"과 지속 학습에서 세계모델을 번갈아 갱신 (§3.2, §4.5) | 공식 스크립트는 모두 `+actor_rollout_ref.wm.update_wm=False`. `verl/workers/fsdp_workers.py:516-517`에서 `update_wm`이 False면 추론용 세계모델을 `.eval()`로 만듦. 지속 학습 전용 스크립트는 저장소에서 찾지 못함 | **차이 (범위).** 공개 설정은 세계모델을 고정한 상태로 RL만 함. 정책 행동 정렬은 별도 단계로 있을 수 있으나 이 저장소에서 실행 경로를 확인하지 못함. 지속 학습 실험(§4.5)의 코드는 공개되지 않은 것으로 보임 |

추가 확인 사항 (차이는 아님):

- **초기 상태.** 평가와 학습의 초기 상태는 `verl/utils/dataset/*_d0_states*.pkl` 일곱 개에서 옴 (논문 "real environment D"와 맞음). 학습 세트와 평가 세트가 분리되었는지는 파일로 확인 불가.
- **평가.** `examples/mimicgen/*/evaluate.sh`는 `data.val_batch_size=128`로 128개 초기 상태를 씀 (논문 §4.1과 일치).
- **세계모델 학습 규모.** 논문은 사전학습 12M 스텝, 미세조정 3M 스텝 (표 3)을 적음. 공개 스크립트 `wm_training_steps_per_epoch=10000`, `total_epochs=100`은 단위가 달라 직접 대응하지 않음. 미확인.
- **전처리.** 행동 청크 `action_chunks_len=8`, 행동 토큰 `action_token_len=7`은 논문의 K = 8, 7자유도와 맞음. 논문 §3.1의 256 구간 이산화는 코드에서 확인하지 못함.

## 2. 공식 코드에서 쓰는 설정과 논문의 비교 (부록용 요약)

| 항목 | 논문 | 공식 스크립트 | 같은가 |
|---|---|---|---|
| 보상 임계값 | 값 없음 | 0.62–0.96 (과제별) | 아니오 (보고 누락) |
| 세계모델 상태 | 학습 시 정책 행동 정렬 + 필요 시 갱신 | RL 중 고정 | 범위 차이 |
| KL | 제거 (DAPO) | `kl_coef=0.00` | 예 |
| 그룹, 배치 | G = 8, 배치 64 (표 4) | `n_samples=8`, `train_batch_size=64` | 예 |
| 학습률, 클리핑, 온도 | 5e-6, 0.2/0.28, 1.6 | 같음 | 예 |

## 3. 판정

- 재현에 필요한 **RL 알고리즘 설정은 논문과 같습니다** (D4).
- 가장 중요한 빈칸은 **보상 임계값**입니다. 표 1의 수치는 과제별 임계값 위에서 나왔을 텐데, 값과 선택 절차가 논문에 없습니다 (D1, D2).
- 세계모델의 핵심 장치 두 개(노이즈 프레임 조건화와 프레임 단위 행동 제어)는 체크포인트 설정 없이 확인할 수 없습니다 (D7, D8).

## 4. 구현 가능성 (이 맥: Apple M4 Pro, 약 24 GiB 통합 메모리, MPS, CUDA 없음)

**논문 수치의 재현은 이 맥에서 불가능합니다. 방법의 축소판은 구현할 수 있고, 실제 세계모델 추론 한 번은 조건부로 가능합니다.**

| 작업 | 필요한 것 | 이 맥 | 판정 |
|---|---|---|---|
| 세계모델 사전학습 (OXE, 12M 스텝) | H100 수십 장 | 불가 | **불가** |
| 세계모델 정책 행동 정렬 (3M 스텝) | GPU 다수, 실제 궤적 P = 128–1280 | 불가 | **불가** |
| GRPO 학습 (표 1) | 기저 OFT 7B, 세계모델, 보상 모델, H100 32장 | 공식 코드는 `torch.cuda.current_device()`와 `bfloat16` CUDA 전제 (`fsdp_workers.py`) | **불가** (학습) |
| 공식 세계모델 추론 한 번 (상상 궤적 생성) | 세계모델 가중치 (수 GB 단위, 번들 크기 미확인), MPS 포팅 | 메모리 측면은 가능성 있음. 속도 미측정 | **조건부.** 가중치 다운로드 승인과 MPS 포팅 후 |
| 보상 모델 추론 (VideoMAE base, 8프레임) | 보상 모델 가중치 (소형), 검증 클립 | 소형이라 MPS나 CPU 가능성 높음 | **조건부.** 가중치와 검증 데이터 다운로드 후 |
| 공식 시뮬 평가 (MimicGen Coffee, 128 초기 상태) | MimicGen 시뮬 설치, 정책 체크포인트 | 설치 부담이 큼. 공식 코드 실행 승인 필요 | **조건부 (보류)** |
| 축소판: 작은 세계모델 + GRPO + 보상 모델 (MuJoCo 또는 2-D 토이) | 직접 작성 (torch, MPS) | 가능. 몇 시간에서 2주 | **가능** (노력 추정 1–2주, 측정 전) |
| 토이 검정 (보상 거짓 양성 기전, 표 검정) | numpy | 가능, 완료 | **완료** (`analysis/toy_reward_exploit.py`, `analysis/recompute.py`) |

**노력 추정 (측정 전, 추정).** 공식 세계모델 추론 한 번과 보상 모델 점수 계산 확인은 2–4일. 축소판 재구현은 1–2주. 논문과 같은 규모의 재현은 불가입니다.

