# 04. 공식 코드 대조와 구현 가능성

## 0. 대상과 방법

- 공식 저장소: https://github.com/WEIRDLabUW/unified-world-model (기본 브랜치 `main`, 커밋 `b7d4067`, 2025-10-08).
- `repo/`에 얕은 복제(`--depth 1`)를 했습니다. **정적으로만 읽었고 실행, 설치, import를 하지 않았습니다.** 복제 내용은 신뢰하지 않는 데이터로 다뤘습니다.
- 읽은 핵심 파일: `models/uwm/uwm.py`(모델, 손실, 샘플러), `models/uwm/obs_encoder.py`, `models/common/transforms.py`(잘라내기, VAE), `datasets/droid/dataset.py`, `datasets/droid/convert_dataset_zarr.py`(행동 구성), `datasets/utils/mixture.py`, `datasets/webvideo/dataset.py`, `experiments/uwm/train.py`, `experiments/uwm/eval_droid.py`, `configs/*.yaml`, `scripts/*.sh`.
- 대조 기준은 논문 v3 본문, 부록 A–D, 표 V–IX입니다(`analysis/01_paper_analysis.md`).

## 1. 설치 요구사항과 라이선스

- **라이선스 없음.** 저장소에 LICENSE 파일이 없고 `pyproject.toml`에도 license 항목이 없습니다. GitHub API `/repos/WEIRDLabUW/unified-world-model/license`는 404입니다. 코드를 가져다 쓰거나 재배포하려면 저자에게 허가를 받아야 합니다. 이 분석은 코드를 공개본에 넣지 않습니다.
- 의존성(`requirements.txt`): `torch==2.2.2`, `torchvision==0.17.2`, `diffusers==0.30.0`, `transformers==4.44.0`, `tensorflow==2.15.0`, `tensorflow-datasets==4.9.7`, `robosuite==1.4.1`, `decord==0.6.0`, `zarr==2.18.2`, `numpy==1.26.4`, `hydra-core==1.3.2`, 그리고 `nvidia-*-cu12` 여러 개(cuBLAS, cuDNN 8.9.2.26, NCCL 2.19.3 등).
- `nvidia-*-cu12` 패키지는 Linux CUDA 휠이므로 macOS에서는 설치되지 않습니다. 설치는 시도하지 않았고, 이것은 정적 판단입니다.
- 배포 형식: `pyproject.toml`의 `[tool.setuptools]`가 `models`, `configs`, `datasets`, `experiments`, `environments` 패키지를 묶습니다.

## 2. 논문 ↔ 코드 대조

### 2.1 일치하는 항목

| 논문 | 코드 | 판정 |
|---|---|---|
| 식 1: `w_a = w_o' = 1`, MSE 두 항의 합 (표 V) | `loss = action_loss + dynamics_loss`, 두 항 모두 `F.mse_loss` (`uwm.py` `forward`) | 일치 |
| `t_a`, `t_o'`를 독립 균일 샘플 (§III.B, 식 1) | 두 시각을 각각 `randint(0, 100)`으로 뽑음 (`forward`) | 일치 |
| 학습 확산 100, 추론 DDIM 10, `squaredcos_cap_v2`, clip sample (표 V) | `num_train_steps: 100`, `num_inference_steps: 10`, `beta_schedule: squaredcos_cap_v2`, `clip_sample: True`, `DDIMScheduler` | 일치 |
| 식 4 순방향 동역학: `t_a = 0`, 깨끗한 행동 | `sample_forward_dynamics`: `action_t = timesteps[-1]` = 0, 행동 입력은 실제 값 | 일치 |
| 식 5 역동역학: `t_o' = 0`, 깨끗한 관측 | `sample_inverse_dynamics`: `next_obs_t = timesteps[-1]` = 0 | 일치 |
| 깊이 12, 폭 768, 헤드 12, MLP 비율 4, qkv bias, 레지스터 8 (표 V) | `configs/model/uwm.yaml`의 기본값 | 일치 (레지스터는 D5 참조) |
| 관측 2 프레임, 행동 16, 실행 8 (표 V, 부록 A.2) | `obs_num_frames: 2`, `action_len: 16`, `eval_droid.py`의 `action_horizon = action_len // 2` = 8 | 일치 |
| 동결 SDXL VAE (부록 A.1) | `AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")`, `requires_grad = False` | 일치 |
| 사전학습 상수 학습률, 미세조정 코사인과 워밍업 1000 (표 V) | `template_train_robomimic.yaml`: `constant`, `finetune`: `cosine`, `num_warmup_steps: 1000` | 일치 |
| AdamW, lr 1e-4, wd 1e-6, betas (0.9, 0.999), eps 1e-8 (표 V) | `template_train.yaml`의 `optimizer` | 일치 |
| 사전학습 100K 스텝 (§IV) | `num_steps: 100000` (DROID와 LIBERO) | 일치 |
| 행동 없는 영상의 행동 시각 `T`로 두기 (§III.D) | `forward`에서 `action_t[~mask] = num_train_steps - 1` (인덱스 99) | 거의 일치 (D1) |
| LIBERO 50개 초기화 평가 (§IV.C) | `num_rollouts: 50` | 일치 |

### 2.2 차이 (D 항목)

**D1. 행동 없는 샘플의 행동 처리.** 논문은 행동 시각을 `T`로 두고 행동을 단위 가우시안 노이즈로 채운다고 씁니다(§III.D, 부록 A.2). 코드는 다음과 같습니다.
- `datasets/utils/mixture.py`는 영상 샘플을 `video_buffer`에서 뽑습니다. 이 버퍼는 `convert_dataset_zarr.py --except_key WEIRD`로 만들며, 실제 DROID 행동(xyz, rot6d, 그리퍼 10차원)을 그대로 저장합니다. 영상 샘플에는 `action_mask = False`만 붙습니다.
- `datasets/webvideo/dataset.py`(인터넷 영상)는 행동을 0으로 채웁니다.
- `forward`는 마스크가 False인 샘플의 행동 시각을 99로 두고, 행동 손실은 **마스크 없이** 모든 샘플에 대해 계산합니다.
- 99 시각의 행동 신호 계수는 √ᾱ₉₉ ≈ 0.0156입니다(ᾱ₉₉ ≈ 2.4×10⁻⁴, `squaredcos_cap_v2` 정의로 직접 계산). 입력 행동의 정보는 1.6% 수준만 남습니다. 따라서 행동 정보 누출은 작지만, 논문 설명(무작위 노이즈로 채움)과 다릅니다. 영향은 추정하지 않았습니다. 확인하려면 실험 로그에서 행동 손실이 영상 샘플에서 어떻게 움직이는지 봐야 합니다.

**D2. 주변화 샘플링 시각 (정책과 영상 예측).** 논문 식 2–3은 `t_o' = T`에서 `o'_T ~ N(0,I)`로 시작합니다. 코드는 다음과 같습니다.
- `sample_marginal_action`(정책, 평가 경로인 `sample()`이 호출)은 다음 관측 시각을 `timesteps[0]`으로 둡니다. 추론 10 스텝, `leading` 간격(diffusers 0.30 기본)에서 이 값은 **90**입니다(정적 추론, 실행하지 않음).
- `sample_marginal_next_obs`는 행동 시각을 `timesteps[0]` = 90으로 둡니다.
- 두 경로 모두 시각 90에서 순수 `N(0,I)` 잠재를 넣습니다. ᾱ₉₀ ≈ 0.024이므로 √ᾱ₉₀ ≈ 0.155입니다. 즉 시각 90의 잠재에는 신호가 15.5% 들어 있어야 하는데 입력에는 없습니다.
- 영향: 배포되는 정책 경로에 작은 분포 차이가 있을 수 있습니다. 영향 크기는 측정하지 않았습니다. 확인하려면 같은 체크포인트에서 시각 90과 99의 입력을 비교해야 합니다(실행 필요, 이번 분석에서는 하지 않음).

**D3. 사전학습 부분집합 선택.** 논문은 DROID 2000 궤적을 "위치(location)"로 고른다고 씁니다(§IV.B.1). 코드의 스크립트는 `--filter_key WEIRD`(폴더 경로에 `WEIRD`가 들어간 에피소드 2000개)로 고르고, 영상은 `--except_key WEIRD`(나머지)를 씁니다. 필터 키가 "위치"와 같은지는 알 수 없습니다. 확인 필요.

**D4. LIBERO 미세조정 스텝.** 논문 §IV.C는 미세조정을 10K 스텝이라고 씁니다. 코드 `template_finetune_robomimic.yaml`은 `num_steps: 20000`이고, `finetune_uwm_robomimic.yaml`은 이 값을 덮어쓰지 않습니다. 확인 필요: 표 II를 어느 설정으로 냈는지.

**D5. LIBERO의 레지스터 수.** `configs/train_uwm_robomimic.yaml`과 `configs/finetune_uwm_robomimic.yaml`은 `num_registers: 24`입니다. 논문 표 V와 표 VII의 기준은 8개입니다. 설정상 LIBERO 사전학습은 레지스터 24로 돌므로 체크포인트도 24개일 것으로 보입니다(체크포인트는 확인하지 않음). 레지스터 수가 다르면 가중치를 불러오지 못합니다. 논문 본문은 LIBERO에 쓴 레지스터 수를 밝히지 않습니다.

**D6. 행동 표현.** 논문은 DROID 행동을 "델타 엔드이펙터(위치와 회전 증분) + 그리퍼 1차원"으로 씁니다(부록 C.1). 코드(`convert_dataset_zarr.py`의 `_load_episode`)는 `cartesian_velocity`의 앞 3차원을 xyz로, 뒤 3차원을 **오일러각으로 해석해** rot6d로 바꾸고, 그리퍼 위치(`gripper_position`)를 붙입니다. 합계 10차원입니다. 회전 증분을 오일러각 세 값으로 보는 방식은 논문에 없습니다. 이것이 로봇 명령의 실제 회전과 같은지는 확인하지 않았습니다.

**D7. 잠재 정규화.** `VAEDownsample`(`models/common/transforms.py`)은 SDXL VAE 잠재에 `scaling_factor`를 곱하고, 평균 0을 빼고, 상수 3.0으로 나눕니다. 이 값(3.0)은 논문에 없습니다. 재구현하려면 같은 정규화를 넣어야 합니다.

**D8. 무작위 자르기의 공유 범위.** `VideoTransform`은 카메라 키마다 `torchvision.transforms.RandomCrop`을 배치 전체에 **한 번** 적용합니다(`obs_encoder.apply_transform`). 그래서 같은 카메라의 모든 샘플과 두 관측(o, o')는 같은 잘라내기 위치를 씁니다. 카메라마다는 다릅니다. 논문 부록 A.2의 "o와 o'에서는 일관되고 카메라 사이에서는 다르다"는 문장과는 맞습니다. 샘플마다 다른 위치를 쓰지 않는다는 점은 논문이 쓰지 않은 세부 사항입니다. torchvision이 배치 텐서에 한 번 파라미터를 뽑는 동작은 정적 판단입니다.

**D9. 인터넷 영상의 빠진 시점 채우기.** 논문 부록 D.3은 "같은 영상의 무작위 자르기로 빠진 카메라를 채운다"고 씁니다. 코드 기본값은 `obs_padding: "same"`이며, 같은 클립을 모든 카메라 키에 그대로 넣습니다(`datasets/webvideo/dataset.py`, `configs/dataset/droid_webvideo.yaml`). 무작위 자르기는 없습니다. 표 IX를 어느 설정으로 냈는지 확인 필요.

**D10. LIBERO 입력 해상도.** `template_train_robomimic.yaml`은 `resize_shape: [240, 240]`입니다. DROID는 `[240, 320]`이고 잘라내기는 224×224입니다. 논문은 LIBERO의 해상도를 적지 않습니다. 영향은 작습니다.

### 2.3 논문 주장과 코드의 대응

- "행동과 영상의 독립 확산 시각으로 여러 분포를 얻는다"(C1): 식 4, 5와 정책 경로의 구조는 코드와 맞습니다. 정책과 영상 예측 경로의 시각은 D2 때문에 다릅니다.
- "행동 없는 영상을 같은 손실에 넣는다"(C4): 구조는 맞지만 D1과 같이 실제 행동이 입력에 있습니다.
- "LIBERO 미세조정 10K"(§IV.C): D4와 충돌합니다.
- "DROID 2000 궤적 위치 선택"(§IV.B.1): D3에서 확인이 필요합니다.

## 3. 안정성 증명 검토

UWM 논문에는 안정성 증명이 없습니다. 해당 없음.

## 4. 구현 가능성 (이 맥)

맥 환경은 Apple M4 Pro, 통합 메모리 약 24 GiB, MPS 사용 가능, CUDA 없음입니다.

**판정: 부분 가능.** 구조 재구현과 축소 실험은 가능합니다. LIBERO 체크포인트 추론 평가는 조건부로 가능합니다. 논문의 DROID 사전학습과 실로봇 재현은 불가능합니다.

| 항목 | 맥에서 가능한가 | 근거 |
|---|---|---|
| 공식 코드를 그대로 설치 | 불가 (정적 판단) | `nvidia-*-cu12` 휠, 2.2.2 torch, TensorFlow 2.15는 macOS에서 설치되지 않음. 설치 시도는 안 함 |
| 코드의 MPS 포팅 | 조건부 | `experiments/utils.py`의 NCCL 초기화, `torch.cuda.is_bf16_supported()`, `DistributedDataParallel`, 장치 `cuda:{rank}` 가정. 포팅 작업 필요 |
| DROID 2000 궤적으로 사전학습 | 불가 | 원 설정은 A100 4장 24시간(부록 A.3). 이 맥에서는 수 배에서 수십 배 느릴 것으로 추정하며 측정하지 않음 |
| LIBERO-90 사전학습 (100K 스텝) | 사실상 불가 | 같은 이유. 측정 없음 |
| LIBERO 체크포인트 추론과 시뮬 평가 | 조건부 | 체크포인트는 README의 Google Drive 링크에 있음(다운로드 승인 필요, 크기 미확인). mujoco와 robosuite 1.4.1 설치 필요 |
| 실로봇 재현 | 불가 | Franka와 DROID 장비가 필요 |
| 토이 (작은 DiT, 2D 또는 1D) | 가능 | numpy 또는 작은 torch로 시각 주변화를 검사 가능. 논문 수치와는 무관 |

**노력 추정 (측정 전, 추정).**
- 설치와 MPS 포팅, 의존성 정리: 2–4일.
- LIBERO 체크포인트 추론 평가 (다운로드 포함): 3–5일.
- 구조 재구현 (축소 모델 학습과 샘플러 검사, D2 확인 포함): 1–2주.
- 전체 재현 (DROID 사전학습): 불가. GPU 클러스터가 필요합니다.

## 5. 재실행

공식 코드는 실행하지 않았습니다. 이 문서의 설명은 다음 명령으로 다시 볼 수 있습니다.

```bash
cd uwm/repo && git log -1 --format='%H %cd'
grep -n "num_steps\|num_registers" configs/*.yaml configs/dataset/*.yaml
```
