# 04. 공식 코드 대조와 구현 가능성

대상: `github.com/roboterax/video-prediction-policy`, 커밋 `75cf37ad0f537627e1a3aa9a04bab70324ec27ff` (2025-05-17). `repo/`에 얕게 복제했고 **정적으로만 읽었습니다.** 코드를 실행하거나 import하거나 설치하지 않았습니다. 가중치와 CALVIN 데이터는 받지 않았습니다.

## 1. 라이선스

- `LICENSE`: MIT, "Copyright (c) 2025 Tsinghua University". 저장소는 MIT입니다.
- 의존 모델의 라이선스는 따로 확인해야 합니다. SVD 기반 코드(`Stability-AI/generative-models`)는 MIT로 표시되어 있지만 **SVD 가중치의 조건은 확인하지 못했습니다.** Hugging Face 체크포인트 세 개와 데이터셋에는 메타데이터 라이선스가 없습니다.
- 공개 글에서는 코드를 붙이지 않고, 저장소 링크와 커밋 해시만 적습니다.

## 2. 대조 결과 (차이 D1–D15)

판정: **일치**, **차이** (논문과 코드가 다름), **누락** (코드가 없음), **미확인** (정적 읽기로는 판단 불가).

### 2.1 학습 데이터와 설정

| ID | 항목 | 논문 | 공식 코드 (파일) | 판정 |
|---|---|---|---|---|
| D1 | 1단계 데이터 혼합 | 표 8: sthv2 0.30, RT-1 0.15, Bridge 0.15, BC-Z 0.08, Taco 0.01, Jaco 0.01, Calvin 0.10, MetaWorld 0.05, Panda 0.05, dex 0.10 (10개) | `video_conf/train_calvin_svd.yaml`: `dataset: 'sthv2+bridge+rt1+calvin'`, `prob: [0.4, 0.2, 0.2, 0.2]` (4개). BC-Z, Taco, Jaco, MetaWorld, 자체 수집 데이터 없음 | **차이**. 공개 설정은 논문 혼합의 일부. 논문 결과 재현에는 다른 설정이 필요 |
| D2 | 영상 모델 조건의 fps | 논문에 없음 | 1단계 학습 `fps: 7` (`train_calvin_svd.yaml`, `train_svd.yaml`). 검증 `fps: 4` (`val_*.yaml`). 특징 추출은 `diffusion_extract.py:80`에서 `fps=4`로 **고정** | **차이** (정적). 학습 때 쓴 조건값과 추출 때 값이 다름. 효과는 측정하지 않음 |
| D3 | 2단계 특징의 무작위성 | 초기 잡음 "typically white noise" (§4.2) | `prepare_latents(...)`에 `generator`를 넘기지 않음 (`diffusion_extract.py:97`). 정의는 diffusers 상속 구현. 실행 시드는 평가 스크립트의 `seed_everything(0)` (`calvin_evaluate.py:231`)에 의존 | **미확인** (정적으로 확률적). 호출마다 다른 특징이 나올 수 있음 |
| D4 | 다층 특징 집계 | "여러 업샘플링 층을 보간해 채널로 쌓음" (§4.2) | `use_all_layer: True`, `extract_layer_idx: 1` (`VPP_Calvinabc_train.yaml`). `diffusion_extract.py:415–450`은 업샘플링 블록 i < 1 과 i = 1의 두 층만 모읍니다. 따라서 **두 층(블록 0, 1)만 쓰며** 그 합계 채널은 2 × 1280 = 2560 (`VPP_policy.py:84–89`) | **차이** (범위). 논문 표 10의 "Layer-3, 6, 9, 12"와 코드 인덱스의 대응은 확인하지 못함 |
| D5 | 확산 시점과 단일 전방 계산 | 한 번의 전방 계산 (§4.2), 표 11 시점 10/20/30 | `timestep: 20` (config). `step_time=1`이면 첫 UNet 호출 뒤 `break` (`diffusion_extract.py:118–152`). 즉 **20 스텝 스케줄의 첫 시점에서 한 번만** 계산 | **일치** |
| D6 | 시간 어텐션 | 시간 어텐션 있음 (식 5, App. C.2) | `use_temporal=True` 하드코딩 (`VPP_policy.py:100`) | **일치** |
| D7 | 행동 헤드 | 청크 10, 10 스텝 (§5.1) | `act_window_size: 10`, `num_sampling_steps: 10`, DDIM (`VPP_policy.py:388–410`). 학습 손실은 `diffusion_loss` (EDM식 노이즈 분포 `loglogistic`) | **일치** |
| D8 | 정책 학습 하이퍼파라미터 | 표 13 (CALVIN 열): 정책 배치 **76**, 에폭 **12**, 학습률 1e-4 | `VPP_Calvinabc_train.yaml`: `batch_size: 28`, `devices: 4` (유효 112), `max_epochs: 50`, `limit_train_batches: 3000`, 학습률 1e-4 | **차이** (배치, 에폭). 어느 쪽이 논문 결과에 쓰였는지 확인 불가 |
| D9 | Video Former 토큰 수 | 표 13: 조건 형태 225 × 384 | `num_latents: 224` = 16 × 14 (`VPP_Calvinabc_train.yaml`) | **사소한 차이** (1 토큰). 논문 표 13의 "토큰 16 × 14 × 384"와는 일치 |
| D10 | 영상 해상도와 길이 | 16 × 256 × 256 (§3) | `sequence_length: 16`, `video_size: [256, 256]` (`train_calvin_svd.yaml`) | **일치** |
| D11 | CALVIN 평가 채점 수 | 논문 본문에 없음 | `num_sequences: 1000` (`calvin_evaluate.yaml`) | 표준 설정으로 추정 |
| D12 | 실로봇 코드 | Franka Panda 실험, XHand 실험 | Panda 코드 없음 (`franka`/`panda` 문자열 없음). XHand/XBot 코드 있음 (`VPP_policy_xbot.py`, `step2_train_action_xbot.py`, `step3_deploy_real_xbot.py`, `policy_conf/VPP_xhand_train.yaml`) | **누락** (Panda) |
| D13 | MetaWorld 코드 | 표 2 (50과제) | `metaworld` 문자열 없음. 데이터 생성 스크립트도 없음 | **누락** |
| D14 | CALVIN 환경 | 시뮬 평가 | `calvin/` 서브모듈은 얕은 복제에서 비어 있음 (README는 `--recurse-submodules`를 요구) | **누락** (복제 방법 문제) |
| D15 | 실로봇 데이터 규모 | 논문 본문 2.5k(A.2) 대 4k(5.3) | 데이터 없음 | **확인 불가** (`03` C9) |

### 2.2 이식성 (맥 관련)

| 항목 | 위치 | 영향 |
|---|---|---|
| CUDA 하드코딩 | 파이썬 파일 48곳 `cuda` 언급. `policy_evaluation/utils.py:35` `.cuda()`, `make_prediction.py:209` `torch.device("cuda")`, `video_models/pipeline.py:332, 622` `.to("cuda")`, `VPP_policy_xbot.py:68` 기본 `device='cuda'`, `calvin_evaluate.py:230` `torch.cuda.set_device` | MPS로 옮기려면 코드 수정이 필요. 실행 전에 바꿔야 함 |
| 고정 경로 | `VPP_Calvinabc_train.yaml`: `/localssd/...`, `/cephfs/...`, `/home/disk2/...` | 모든 경로를 바꿔야 함 |
| 버전 고정 | `requirements.txt`: `torch==2.0.1`, `diffusers==0.28.0`, `transformers==4.47.0`, `pytorch-lightning==1.9.5`. README: Python 3.10, CALVIN은 `torch==1.13`을 요구하지만 README는 2.x도 된다고 씀 | macOS arm64에서 이 조합의 설치 가능성은 **설치 전 확인 필요** (미확인) |
| 문서 불일치 | README는 `step1_prepare_latent_data.py`라고 쓰지만 저장소에는 `step1_prepare_latent.py` | 명령을 그대로 따라 하면 실패 |

## 3. 논문과 코드가 같은 부분 (요약)

- 두 단계 구조(영상 모델 → 동결 특징 → 행동 헤드), 256 × 256 × 16 프레임, CLIP ViT-B/32 텍스트 임베딩, Video Former 깊이 6, 헤드 8, 학습률 1e-4, 행동 청크 10, DDIM 10 스텝: 코드와 논문이 같습니다.
- 한 번의 전방 계산(D5)과 시간 어텐션(D6)도 같습니다. 절제 실험과 논리가 맞습니다.

## 4. 구현 가능성 판정

**이 맥(Apple M4 Pro, 약 24 GiB 통합 메모리, MPS, CUDA 없음)에서 논문 수치를 재현하는 것은 불가능합니다. 추론 경로 일부(1단계 특징 한 번)는 측정할 수 있을 가능성이 있지만, 아직 측정하지 않았습니다.**

| 작업 | 이 맥에서 | 필요한 것 | 노력 (추정, 측정 없음) |
|---|---|---|---|
| 1단계 SVD 미세조정 (2–3일 × A100 8장, 논문 §5.1) | **불가** | GPU 8장 | — |
| 2단계 행동 학습 (6–12시간 × A100 4장) | **불가** | GPU 4장 | — |
| CALVIN ABC→D 평가 (1000 체인) | **불가** (이 맥) | Linux, EGL 렌더링, CALVIN 환경, ABC-D 데이터 (README "about 500 G"), 체크포인트 약 8 GB + 1 GB | Linux GPU에서 1–2일 (설치 포함) |
| 1회 특징 추출 시간 측정 (16프레임, 256²) | **조건부** | torch 2.0.1 설치, diffusers 0.28, `svd-robot-calvin-ft` 약 8 GB 다운로드, `.cuda()` 제거 | 1–2일 (코드 수정과 측정). 측정값은 없음 |
| 정적 대조 (이 문서) | **완료** | — | 완료 |
| 토이 (미래 예측 오차 대 역동역학) | **완료** | numpy | 완료 (`03` §4) |

- **재현 판정: 불가 (학습과 CALVIN 평가 전체). 이 맥에서 가능한 것은 특징 추출 한 번의 시간 측정과 구조 토이뿐입니다.**
- 조건부 항목은 설치, 다운로드, 코드 수정을 사용자가 승인한 뒤에 합니다. 공식 코드 실행과 새 가상환경은 승인 전에 하지 않습니다.
- 논문의 "7–10 Hz (RTX 4090)"은 이 맥에서 기대하기 어렵습니다. MPS 시간은 측정하지 않았습니다.

## 5. 확인하지 못한 것

- D2와 D3가 실제 결과에 주는 영향. 실행하지 않았습니다.
- D4에서 논문 표 10의 "층 번호"가 코드의 어떤 블록에 대응하는지.
- 프로젝트 페이지의 영상과 논문 그림 3–4의 질적 주장 (영상 자체는 보지 않음).
- `repo/calvin/` 서브모듈, 체크포인트, 데이터.
