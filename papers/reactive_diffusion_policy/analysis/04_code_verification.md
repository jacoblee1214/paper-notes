# 04. 공식 코드 대조 (정적 읽기)

대상은 `xiaoxiaoxh/reactive_diffusion_policy`의 커밋 `824c5e8de1fd1811106907a04b5f0186e0138c0b` (2026-04-12, 얕은 복제 `repo/`)입니다. 코드는 **읽기만 했습니다.** 실행, import, 설치, 가중치와 데이터 다운로드는 하지 않았습니다. 차이 번호는 D1–Dn입니다. 논문 번호는 `01_paper_analysis.md`와 같습니다.

## 1. 저장소와 라이선스

- 최상위 `LICENSE`는 MIT입니다(저작권: Han Xue, Wendi Chen, Yuan Fang, 2025). 같은 파일에 **Diffusion Policy와 UMI의 MIT 고지**가 이어서 들어 있습니다. 제3자 코드(`third_party/`: MindVision SDK, Flexiv RDK 예제, `mvcam`)는 각자의 라이선스를 따릅니다. GitHub API는 여러 라이선스가 섞여 NOASSERTION으로 분류합니다.
- 공식 README는 RDP 학습에 GPU 워크스테이션(예: RTX 3090)과 Ubuntu 22.04, ROS 2 Humble을 요구합니다. 촉각 센서(GelSight Mini 25 FPS, 고성능 CPU 권장)와 Flexiv Rizon 4 두 대, Meta Quest 3, RealSense 카메라가 필요합니다.
- 데이터와 체크포인트는 Hugging Face에 있습니다(`WendiChen/reactive_diffusion_policy_dataset`, `WendiChen/reactive_diffusion_policy_model`). **받지 않았습니다.**

## 2. 논문과 코드의 대조

판정: **일치**는 코드 값이 논문 본문 또는 표 VII와 같음, **차이**는 다름, **미기재**는 논문에 값이 없고 코드가 정한 값입니다.

| 번호 | 항목 | 논문 | 코드 (파일) | 판정 |
|---|---|---|---|---|
| D1 | LDP 추론 확산 반복 | 표 VII: 추론 8 (학습 100) | `train_latent_diffusion_unet_real_image_workspace.yaml`: `num_inference_steps: 100`. LDP 전용 재정의는 찾지 못했습니다 | **차이 (확인 필요).** 표 I의 LDP 100 ms는 100 단계와 더 잘 맞아 보이지만, 표 VII 오기인지 실제 설정인지 확정하지 못했습니다 |
| D2 | AT 인코더 다운샘플 | 표 VII: 4 | `at_peel.yaml`: `conv_layer_num: 1`. 인코더는 stride-2 conv 두 개(`EncoderCNN`)이므로 4배 | **일치** |
| D3 | AT 잠재 차원 | 피넬링 4, 닦기·들기 8 | `at_peel.yaml`: 4. `at_wipe_lift.yaml`: 8 | **일치** |
| D4 | AT 예측 길이, 은닉, λ_KL, 학습률, weight decay | 32 (24 Hz), 32, 1e-6, 1e-3, 1e-4 | `at_peel.yaml`: horizon 32, `rnn_latent_dims` 32, `kl_multiplier` 1e-6. `train_at_workspace.yaml`: AdamW lr 1e-3, wd 1e-4 | **일치** |
| D5 | AT 손실과 입력 | L1 재구성 + KL, 인코더는 행동만, 디코더는 촉각 | `model/vae/model.py` `compute_loss_and_metric`: `encoder_loss = (state - dec_out).abs().mean()` + `kl_multiplier`. 인코더 입력은 `batch["action"]`, 디코더의 `temporal_cond`는 `extended_obs`(촉각) | **일치.** 비대칭 구조가 코드에도 있습니다 |
| D6 | AT 에폭 | 600 | `train_at_workspace.yaml`: `num_epochs: 601` | 일치 (1 에폭 차이) |
| D7 | AT 이마 EMA | 논문 미기재 | `train_at_workspace.yaml`: `use_ema: False` | **미기재** |
| D8 | LDP 학습률, weight decay | 1e-4, 1e-6 | `train_latent_diffusion_unet_real_image_workspace.yaml`: lr 1e-4, wd 1e-6, betas [0.95, 0.999] | **일치** |
| D9 | LDP 에폭 | 400 | `num_epochs: 401` | 일치 (1 에폭 차이) |
| D10 | LDP 다운 차원, 조건 주입 | [512, 1024, 2048], FiLM 1D U-Net | `down_dims: [512, 1024, 2048]`, `diffusion_step_embed_dim: 128` | **일치** |
| D11 | LDP 커널 크기 | 3 | `latent_diffusion_unet_image_policy.py`: 조건부 `kernel_size = 3` (AT 인코더 사용 시) | **일치** |
| D12 | DP 커널, 에폭, 확산 반복 | 5, 600, 학습 100 | `train_diffusion_unet_real_image_workspace.yaml`: kernel 5, `num_epochs: 600`, 학습 100 | **일치** |
| D13 | LDP EMA | 논문 미기재 | `use_ema: True` (LDP와 DP 모두) | **미기재** |
| D14 | 학습 중 학습률 자동 스케일 | 논문 미기재 | `train_diffusion_unet_image_workspace.py` 103–107행: `lr = lr * accelerator.num_processes`. LDP도 같은 워크스페이스를 씁니다 | **미기재, 재현 영향 큼.** GPU 수에 따라 실제 학습률이 달라집니다. 논문의 GPU 수는 명시하지 않았습니다 |
| D15 | 관측 인코더 학습률 계수 | 논문 미기재 | `encoder_lr_coefficient: 0.1`. 코드에서 사전학습 인코더이고 LoRA를 쓰지 않을 때만 적용됩니다(78행). 기본 구성에서 해당 조건이 참인지 확인하지 못했습니다 | **미기재** |
| D16 | 워밍업 | 논문 미기재 | LDP 500 스텝, AT 100 스텝 | **미기재** |
| D17 | 지연 매칭 스텝 | "몇 스텝" (값 없음) | `task/real_peel_image_gelsight_emb_ldp_24fps.yaml`: `latency_step: 4`, `gripper_latency_step: 4`. 24 Hz에서 약 167 ms. 실행 루프가 `action_all[self.latency_step:]`로 앞부분을 버립니다 | **미기재 (값).** 구현은 논문 설명과 일치 |
| D18 | 상대 행동 | 피넬링·닦기 상대, 들기 절대 | 피넬링 LDP: `relative_action: True`. 들기 `..._absolute_24fps.yaml`: `relative_action: False` | **일치** |
| D19 | 제어와 청크 주기 | 24 FPS 제어, 청크 0.67 s, 1–2 Hz | `control_fps: 24`, `tcp_action_update_interval: 16` (0.67 s, 1.5 Hz). `inference_fps: 6`은 루프 주기로 보입니다 | 일치 |
| D20 | 촉각 PCA 차원 | 15 | `n_components: 15`. `data/PCA_Transform_GelSight`, `data/PCA_Transform_McTAC_v1`의 변환 행렬을 저장소에 제공 | **일치.** 부록 D의 PCA 데이터 자체는 없고 변환 행렬만 있습니다 |
| D21 | 이미지 해상도 | 240 × 320 | `image_shape: [3, 240, 320]` | 일치 |
| D22 | 학습 장치 | 논문 미기재 | `training.device: "cuda:0"` (세 설정 모두). 학습 스크립트는 `CUDA_VISIBLE_DEVICES`와 `accelerate launch`를 씁니다 | **재현 장벽.** 이 맥에서는 MPS로 바꾸는 코드 변경이 필요합니다 |
| D23 | 실행 스크립트의 경로 | — | `train_rdp.sh`: 저자 기기의 홈 디렉터리 절대 경로. 평가 스크립트에는 `/path/to/...` 자리 표시자 | 재현 시 경로를 바꿔야 합니다 |
| D24 | 시연 수 | 피넬링 60, 닦기 80, 들기 50 | 저장소 설정에서 확인하지 못함 (데이터는 Hugging Face) | 확인 불가 |
| D25 | TactAR 앱 | 논문 부록 B에 렌더링 방식과 지연 | 저장소 밖 (`TactAR_APP`, Unity, 라이선스 없음). 복제하지 않았습니다 | 확인 불가 |

## 3. 읽기 중 확인한 구조

- **AT (`model/vae/model.py`)**: `EncoderCNN`은 `Conv1d` stride 2를 `conv_layer_num + 1`번 씁니다. 디코더 `DecoderRNN`은 GRU이고 입력은 잠재 전역 조건과 촉각 시간 조건을 시간축에 이어 붙인 것입니다. 손실은 L1 재구성과 KL입니다. 논문 식 2와 맞습니다.
- **LDP (`policy/latent_diffusion_unet_image_policy.py`)**: `DiffusionUnetImagePolicy`를 상속합니다. 행동 차원은 AT 잠재 차원으로 바뀌고, 예측 길이는 AT 다운샘플 후 길이(8)입니다. 학습 목적은 DDPM 노이즈 예측(식 3)과 같은 틀입니다.
- **실행 (`env_runner/real_runner.py`)**: 청크를 `tcp_action_update_interval`마다 갱신하고, 앞 `latency_step` 스텝을 버립니다. 앙상블 버퍼 모드는 `'new'`입니다. 빠른 AT는 촉각을 받아 잠재 청크를 자기회귀로 디코딩합니다(`use_latent_action_with_rnn_decoder: True`).

## 4. 구현 가능성 판정

**판정: 이 맥에서 실험 재현은 불가능합니다. 구조 재구현과 학습 경로 점검은 조건부로 가능합니다.**

| 항목 | 판정 | 근거 |
|---|---|---|
| 실로봇 평가 (표 II–IV) | **불가** | Flexiv Rizon 4 두 대, Grav 그리퍼, Quest 3, GelSight Mini, MCTac, RealSense가 필요합니다 (README, §3). 이 맥에 장비가 없습니다 |
| 공식 코드의 학습 (LDP, AT) | **조건부 (코드 수정 필요)** | `cuda:0`이 설정에 박혀 있습니다(D22). 학습률은 GPU 수에 따라 스케일됩니다(D14). 데이터는 Hugging Face에서 받아야 하고, 용량은 확인하지 않았습니다. 이 맥의 MPS에서 학습 속도는 측정하지 않았습니다 |
| 공식 체크포인트 추론 | **조건부** | 체크포인트는 Hugging Face에 있습니다(받지 않음). 추론 자체는 로봇 없이 할 수 있지만 촉각 입력이 필요해 실제 센서 없이는 검증이 안 됩니다 |
| AT 구조의 CPU 재구현 (토이) | **가능** | 1D-CNN과 GRU, 비대칭 입력은 numpy 또는 torch로 옮길 수 있습니다. 이번 작업에서는 만들지 않았습니다 |
| 표 재계산과 검정 | **완료** | `03_analysis.md`와 `analysis/recompute.py` |

**노력 추정 (측정 전, 추정치).**

- 공식 코드를 MPS에 맞춰 LDP 한 번 학습 가능하게 하는 작업: 1–2일.
- 한 번의 LDP 학습 시간: 측정하지 않았습니다. 시연 60–80개 규모라면 수 시간에서 하루 이상일 것으로 추정합니다.
- AT 구조만 옮긴 CPU 토이: 1–2일.
- 로봇 두 대와 촉각 센서를 쓴 실험: 저장소 밖 장비가 필요하므로 이 맥에서 불가입니다.

## 5. 공개 범위와 라이선스 메모

- 공식 코드는 공개본에 넣지 않습니다(라이선스 고지가 복합적입니다). 공개본은 이 문서의 대조 결과와 재계산 스크립트만 포함합니다.
- 논문 라이선스는 arXiv 비독점(`nonexclusive-distrib/1.0`)입니다. CC BY가 아니므로 논문 그림은 공개본에 넣지 않습니다(`README.md`의 라이선스 절 참조).
- `paper/paper.pdf`(v3, 6.4 MB)는 25 MB보다 작아서 gitignore 대상이 아닙니다. v1과 v2 PDF는 gitignore에 넣었습니다.
