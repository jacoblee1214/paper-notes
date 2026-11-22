# 04. 공식 코드 정적 대조와 구현 가능성

**방식:** 공식 저장소(github.com/Why-peace/F2F-AP, 커밋 `2c382e927d61`, 2026-09-05 "v1")를 얕게 받아 **읽기만** 했습니다. 코드 실행, import, 패키지 설치, 가중치·데이터 열기는 하지 않았습니다(저장소에 포함된 체크포인트와 데모 데이터도 열지 않고 파일 크기와 zarr 메타데이터 목록만 봤습니다). 논문은 v4입니다. 참조는 `파일:줄`이며 저장소 루트 기준입니다. `f2f-ap/f2f-ap` 저장소는 파일 548개의 내용이 같습니다(트리 해시 비교, `02` §2.2).

## 1. 저장소 개요

| 항목 | 내용 |
|---|---|
| 규모 | 추적 파일 548개, 작업 트리 184 MB (체크포인트 30.7 MB, 데모 zarr 18.7 MB, 데모 LMDB 23.6 MB 포함) |
| 라이선스 | **LICENSE 파일 없음.** 벤더링한 UMI·Diffusion Policy 폴더(`third_party/`)에도 원 MIT 파일이 없음 |
| F2F 고유 코드 | `f2f_ap/models/{diffusion_unet_timm_policy,dynamic_predictor}.py`, `f2f_ap/trainers/*`, `f2f_ap/real_world/workspace.py`, `scripts/*`, `configs/*` (약 4,800줄) |
| 벤더링 | `third_party/umi`, `third_party/diffusion_policy` (수정본으로 보임, 원 저장소와의 줄 단위 대조는 하지 않음) |
| 포함된 가중치·데이터 | `outputs/flow_predictor/model_best.ckpt`(예측기 한 개, 30.7 MB), `data/mini_demo/policy/dataset.zarr.zip`(에피소드 2개, 프레임 788개), `data/mini_demo/predictor`(LMDB 64샘플), 그리퍼 마스크 `data/finger_mask` |
| 의존 | Python 3.9, PyTorch 2.1.0 + CUDA 12.1, diffusers 0.18.2, timm 0.9.7, lightning 2.1.3, deepspeed 0.12.6, SAM2(선택) |

## 2. 논문과 일치하는 것

| 항목 | 논문 | 코드 |
|---|---|---|
| 예측기 구조 | 8층, D = 384, 8 헤드, MLP 1536, 패치 16, 입력 6×224×224 (C.2) | `configs/predictor/flow_transformer.yaml` depth 8, dim 384, heads 8, frame_stack 6, patch 16 (MLP 배수 4 = 1536, `transformer.py`) |
| 국소 상관·히트맵 | 반경 2, 56×56 히트맵 | radius 2, stride 4 → 56×56 (`dynamic_predictor.py:65–134`) |
| 점 수 | 4 | `num_flow_points: 4` |
| 예측기 학습 | Adam, 배치 32, lr 1e-4, 코사인 + 5 에폭 워밍업, 혼합 정밀도 | `lr: 1e-4`, `batch_size: 32`, `warmup_epoch: 5`, bf16-mixed |
| 정책 구조 | CLIP ViT-B + 1D U-Net, DDIM 16단계 | `vit_base_patch16_clip_224.openai`, `ConditionalUnet1D`, `num_inference_steps: 16` |
| 정책 학습 | AdamW, 배치 32, lr 3e-4(백본 3e-5), 워밍업 2000, 코사인 | `lr: 3.0e-4`, 사전학습 인코더는 ×0.1 = 3e-5 (`train_diffusion_unet_image_workspace.py:60–62`), `lr_warmup_steps: 2000` |
| 관측 구성 | T_obs = 2, T_pred = 6 | `img_obs_horizon: 2`, `predictor_horizon: 7`(프레임 7개 = 두 시점 × 6프레임) |
| 지연 측정값 | 카메라 125 ms | `camera_obs_latency: 0.125`(설정과 평가 스크립트에 하드코딩) |
| 정답 흐름 | K-Means(K = 4), 이전 프레임으로 초기화 | `scripts/data_processing/clustering_point_old.py` (K-Means + 헝가리안 매칭으로 시간 일관성) |
| 예측기 파라미터 수 | 미기재 | 구조로 센 값 약 15.2M = 체크포인트 30.7 MB(16비트 저장 가정)와 맞음 (`03` K5) |

## 3. 차이와 의심 (D1–D19)

심각도: **높음**(재현이나 결과 해석을 바꿈), **중간**, **낮음**. 확정이 아닌 것은 "의심"으로 적었습니다.

| 번호 | 항목 | 논문 | 코드 | 심각도 |
|---|---|---|---|---|
| D1 | 공개 범위 | 사족 로봇(Carrot, Length Meas.), RTC·VLASH·MF-UMI 이식 기준선, 절제 | 사족 쪽은 `f2f_ap/real_world/workspace.py`가 저장소에 없는 `ModelServer`를 import하고(L14) Go2 서버에 접속하는 구조라 단독으로 못 돎. 평가 스크립트는 UR5 두 개뿐. 기준선은 `policy_type`(`syc`, `asyc`, `vlash`)으로 선택하지만 RTC 호출은 주석 처리(`eval_real_ur5_no_space_mouse.py:292,359`). 학습된 정책 가중치·전체 데이터 없음 | 높음 |
| D2 | 기본 설정이 F2F가 아님 | F2F-AP가 방법 | 기본 설정 `policy_type: asyc`(`train_diffusion_unet_timm_umi_workspace.yaml:88`). `train_policy.py:18–23`는 `policy_type != 'f2f'`이면 `flow_predictor`를 지우고 지연도 0으로 만듦. README의 정책 학습 스모크 테스트 명령에는 `policy.policy_type=f2f`가 없어 **F2F도 대조 손실도 실행되지 않음** | 높음 |
| D3 | 평가 스크립트와 환경 불일치(의심) | - | `eval_real_ur5_no_space_mouse.py:266,313,336`가 `env.get_obs(future_steps=…)`를 호출하는데 벤더링한 `BimanualUmiEnv.get_obs(self)`는 인자가 없음(`third_party/umi/real_world/bimanual_umi_env.py:352`). 정적으로는 `TypeError`. `future_steps`는 저장소 안 다른 곳에 정의가 없음 | 높음 |
| D4 | 추론 주파수 | 5 Hz(D.2), 세 방법 모두 같은 주파수(E.1) | 평가 스크립트(L154–171)는 F2F·VLASH를 `steps_per_inference = 4`(제어 10 Hz → 0.4 s 주기 = **2.5 Hz**), 비동기·동기를 2(5 Hz)로 **하드코딩**. CLI 옵션 `-si`(기본 6)는 덮어씀. 다른 스크립트 `eval_real_ur5.py`는 기본 6이고 F2F 분기가 없음 | 높음 |
| D5 | 지연 H의 단위(의심) | H = 4 제어 스텝 = 0.4 s | `task.latency_steps: 4`(`configs/policy/task/umi.yaml:6`)가 시간 샘플러에서 **복제 버퍼의 원시 프레임 단위**로 쓰임: 미래 영상 = `current_idx + 4`(`third_party/diffusion_policy/common/sampler.py:157–159`), 저차원 상태 보간 `+ latency_steps`(L172), 행동 `slice_start = current_idx + 4`(L215). 제어 간격은 `down_sample_steps: 6` 프레임(60 Hz → 10 Hz). 버퍼가 약 60 Hz이면(UMI 기본. 데모 zarr가 에피소드 2개에 788프레임) 학습의 미래 오프셋은 약 0.067 s인데 평가의 행동 시각표는 4 × 0.1 = 0.4 s(`eval…:384–385`). UMI 원본의 `latency_steps`도 원시 프레임 단위(`0.125 × 59.94`)라서 단위 혼동 가능성이 큼 | 높음 |
| D6 | 대조 손실의 구현 | 공유 인코더, 두 투영기, 대칭 InfoNCE, 시간 마스크 τ_d (식 3) | (a) 정답 미래 영상 특징은 `torch.no_grad()`로 계산해 **그 쪽 인코더 경로로는 그래디언트가 없음**(`diffusion_unet_timm_policy.py:340–342`), 투영기만 학습 (b) 두 입력은 `obs_encoder` 출력 **전체**(영상 + 저차원 로봇 상태를 이어 붙인 값, `timm_obs_encoder.py:258–283`)이고 정답 쪽도 같은 현재 로봇 상태를 씀(L338) → 같은 배치 항의 두 임베딩이 동일한 상태 특징을 공유, 상태만으로 양성 쌍을 맞힐 수 있는 지름길 (c) 온도는 학습 가능 `logit_scale`, 초기값 ln(1/0.07), 최대 100으로 clamp (L89, L349) (d) 마스크는 같은 에피소드이고 스텝 차 < 10(원시 프레임, 약 0.17 s)인 쌍의 로짓을 −1e4로 채움(L353–369), 배치 32 무작위에서는 배치당 0.4–0.8개 항만 닿음(`03` K4) (e) 손실 가중치 0.1(`contrastive_loss_weight`) | 높음 |
| D7 | 예측기 "동결"의 의미 | 동결된 F2F가 흐름을 그림 (C.3) | 파라미터만 `requires_grad = False`이고 `eval()`은 생성자에서 한 번만(L106–109). 학습 루프의 `self.model.train()`(워크스페이스 L211)이 **하위 모듈 전체를 학습 모드로 되돌려** 예측기의 어텐션·FF 드롭아웃 0.2와 BatchNorm 배치 통계가 정책 학습 중 켜짐. 추론에서는 `eval()`. 학습 때 그려진 흐름이 추론 때와 분포가 다를 수 있음(의심) | 중간 |
| D8 | 키포인트 추출 | 식 (1) 소프트 샘플링 S(H), C.1 설명은 헝가리안·일대일 손실이 붕괴 | 추론은 `kmeans_sparse_refine`(임계 0.20, 피크 + 가중 K-Means 30회 반복, `no_grad`, `dynamic_predictor.py:105–134`). **학습은 정답 좌표로 특징을 샘플링(교사 강요, L157–160)**하므로 점 순서 대응 문제가 위치로 해결됨. 헝가리안 함수 `hungarian_vectorized`는 정의만 있고 호출되지 않음(`loss.py`) | 중간 |
| D9 | 흐름 그리기 | 4개 점, 두께 5픽셀, 분홍·초록 (E.3, 4.3) | 정책 경로는 `flow_to_purple_gradient_img(..., thickness=3)`(`dynamic_predictor.py:364`), 보라→연보라 그라데이션(`flow_utils.py:360–361`). 시각화 스크립트 경로는 두께 기본 1. 두께 5나 분홍·초록은 코드에 없음 | 중간 |
| D10 | 예측기 손실과 입력 | 마스크된 패치 복원 보조 손실(D.1) | 복원 손실은 마스크된 패치만이 아니라 **모든 패치·프레임**의 MSE(`dynamic_predictor.py:477`). 가중치는 `lbd_img = 10 × (1 − epoch/301) + 1`로 11에서 1로 감소(`train_predictor.py:65`), 히트맵 포컬 손실(α = 2, β = 2, `flow_transformer.yaml:loss_cfg`)과 오프셋·흐름 L1. 논문에 없는 설정 | 낮음 |
| D11 | 정답 흐름의 시간 길이 | H = 4 스텝 뒤(알고리즘 1) | `FlowDataset`은 끝점을 `local_idx + (horizon − 1) × downsample_steps`로 잡음(`flow_dataloader.py:117–120`) → (4 − 1) × 6 = 18프레임(60 Hz면 0.3 s)이고 H × 6 = 24프레임(0.4 s)이 아님. 정책 학습의 미래 영상 오프셋(+4프레임, D5)과 평가 시각표(0.4 s)까지 합쳐 **세 가지 다른 시간 간격**이 보임(의심) | 중간 |
| D12 | 입력 가림 처리 | 그리퍼 가림은 합성 데이터로 해결(B.3) | 추론에서 예측기 입력에 `mirror_gripper_mask`를 곱해 그리퍼 영역을 지움(`dynamic_predictor.py:226–235, 350–353`). 학습 데이터에는 이름 번호 > 999인 영상에 확률 0.75로 손가락 마스크를 붙임(`flow_dataloader.py:113–114`). 논문 본문에 없는 두 가지 | 낮음 |
| D13 | 하드코딩 | T_obs, T_pred를 설정값으로 | 생성자 인자를 무시하고 `self.obs_horizon = 2`, `self.predictor_input_len = 6`(`diffusion_unet_timm_policy.py:84–85`) | 낮음 |
| D14 | 논문에 없는 정책 설정 | 미기재 | 행동 지평선 16(`task/umi.yaml:12`), `n_action_steps: 8`, 학습 노이즈 50스텝 DDIM, EMA power 0.75(최대 0.9999), 입력 섭동 0.1, 200 에폭, 검증 비율 0.05, 오프셋 정규화 등 | 낮음 |
| D15 | 체크포인트 | 과제별 예측기(§5 한계 2) | 예측기 한 개(30.7 MB)만 있고 어느 과제용인지 설명 없음. `mirror_gripper_mask` 때문에 특정 카메라 배치용으로 보임. 정책 가중치 없음 | 중간 |
| D16 | 절제·기준선 스위치 | 표 2의 8개 절제, RTC·MF-UMI | 대조 가중치만 설정값. 미래 상태 제거, 히트맵 제거, 색·점 수, MF-UMI는 코드 수정 없이 불가. RTC 함수는 있으나 평가 호출 주석 처리, RTC 가이던스 값(s = 5, 가중치 10)은 논문에 없음 | 중간 |
| D17 | 평가 프로토콜 | SR, ET, 시도 수 | 스크립트가 성공·실패·시간을 기록하지 않음(키 `S`로 중단, 영상 저장). 시도 순서, 사람 변동 통제, 성공 기준은 코드로 확인 불가 | 낮음 |
| D18 | 베이스라인 이식에서 지연 정렬 | UMI 기준선은 UMI와 같음 | `train_policy.py:22–23`이 F2F·VLASH가 아니면 `latency_steps = 0`으로 둠. 원본 UMI는 카메라–로봇 지연 정렬에 `latency_steps`(원시 프레임, 약 7개)를 씀. 표 1의 UMI 기준선이 이 설정으로 학습됐다면 원본보다 불리한 구성일 수 있음(의심, 학습 설정 미공개) | 중간 |
| D19 | 실행 환경 | 학습은 RTX 4090 8장, 추론 RTX 3090 | `train_predictor.py:33–36`이 `accelerator='cuda'`, `strategy='deepspeed'`, `train_policy` 설정이 `training.device: cuda:0`, 평가 스크립트가 `torch.device('cuda')`(L262). CPU·MPS로는 설정 수정 없이 불가 | 중간 |

### 3.1 D5, D11: 시간 오프셋 정리 (60 Hz 버퍼 가정)

| 오프셋 | 논문 | 코드 |
|---|---|---|
| 지연 H | 4스텝 = 0.4 s | 평가 시각표 4 × 0.1 s = 0.4 s |
| 미래 영상 (대조 정답) | o_{t+H} | 샘플러 + 4프레임 ≈ 0.067 s |
| 상태·행동 시작 | s_{t+H} | + 4프레임 ≈ 0.067 s |
| 정답 흐름 끝점 | H 스텝 뒤 | + 18프레임 ≈ 0.30 s |

이 가정이 맞으면 코드는 논문과 같은 알고리즘이 아니라 "작은 지연으로 학습하고 큰 지연으로 실행"하는 구성입니다. 버퍼 주파수가 60 Hz가 아니거나 `down_sample_steps`가 원시 프레임을 뜻하지 않는다면 이 차이는 사라집니다. **저자에게 확인이 필요하고, 데이터 없이 정적으로는 확정할 수 없습니다.**

## 4. 구현 가능성 판정 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**이 맥에서 논문 수치(실로봇 표 1, 2)를 재현하는 것은 불가능합니다. 공식 코드를 그대로 돌리는 것도 어렵고, 핵심 아이디어의 축소 재구현과 추론 경로 점검은 가능합니다. 재현은 하지 않았습니다.**

| 수준 | 판정 | 근거 |
|---|---|---|
| 실로봇 표 1·2 | **불가** | UR5e, 그리퍼, GoPro, 사람이 건네는 물체, 사족 Go2 + 전신 제어기(코드 비공개, D1) |
| 공식 학습(예측기 deepspeed, 정책 accelerate) | **사실상 불가** | `cuda`, deepspeed, 4090 8장 규모, 학습 데이터 비공개 (D19). 데모 데이터는 에피소드 2개라 스모크 테스트 용도일 뿐 |
| 공식 코드로 **추론·시각화**(`eval_predictor.py`) | **조건부** | 새 환경 설치(torch, timm, einops, cv2 등), 코드 실행 승인, 체크포인트 사용 승인이 필요. 예측기 15M 파라미터, 추론 1회 약 130 GFLOPs(U-Net 제외, 어림, `03` K7)라 MPS에서 가능해 보이지만 BatchNorm·`grid_sample`·`unfold` 연산의 MPS 지원과 속도는 **측정 전** |
| 정책 구조 축소 재구현 (UMI식 1D U-Net, 흐름 합성, 대조) | **가능** | torch로 새로 구현, 작은 합성 데이터, CPU/MPS. 노력 추정 1–2주(측정 전) |
| 논문 핵심 가설의 수치 토이 | **완료** | `analysis/toy_latency_intercept.py`(수 초) |

**노력 추정(측정 전):**
- 공식 코드로 추론·시각화만: 환경 설정과 연산 호환 점검 1–2일 (승인 후).
- 흐름 예측기 학습 코드의 CPU 소규모 동작 확인: 설정 수정(`accelerator`, deepspeed 제거) 후 1–2일. 학습 규모의 의미 있는 결과는 데이터가 없어 불가.
- 직접 구현으로 핵심 가설(예측 흐름이 정책 입력에 더 낫다)을 MuJoCo 합성 과제에서 시험: 2–3주(추정).
- 같은 수준의 실험을 실로봇에서 재현하려면 UR5e 계열 팔, GoPro, 데이터 수집 장치(UMI), 사람이 필요하고 이 맥만으로는 불가.

**이 맥에서 이미 가능한 것 / 필요한 것**

- 가능(완료): 표 재계산, 통계 검정, 코드 정적 대조, 토이.
- 가능(승인 필요): 위의 추론·시각화.
- 필요(이 맥에 없음): CUDA GPU(학습), UR5e 또는 비슷한 팔 + UMI 장치(평가), 과제별 시연 100–200개 + 그리퍼 없는 영상(데이터), Go2 + 전신 제어기(사족 실험).
