# 04. 논문 ↔ 공식 코드 대조

- 저장소: https://github.com/bjkim95/rofacto (Apache-2.0), 얕은 clone, 커밋 `f9504d0b839ca2eb8936407eb23f7ae9d9ba4078` (2026-11-11 KST, 저장소 전체 6 커밋 중 최신. 확인 2026-11-26)
- **정적으로만 읽었다.** 실행, import, 설치, 가중치·데이터 다운로드를 하지 않았다. `repo/`는 `git clone --depth 1 https://github.com/bjkim95/rofacto repo`로 다시 받을 수 있다.
- 규모: Python 약 5.2만 줄(`training/wan/train_dwm_wan.py` 6,860줄, `dataset.py` 4,258줄), 설정 YAML, 데이터 처리 스크립트, Cosmos3 백엔드, 모델 카드.
- 코드가 다루는 범위: **Wan2.1 구현(논문), 추가 Cosmos3 구현(논문에 없음), 데이터 전처리(DROID 렌더, RoboCasa 명목 상태, DexYCB), 학습·추론.** 코드에 없는 것은 §3.

## 1. 일치하는 것

| 논문 | 코드 | 위치 |
|---|---|---|
| Wan2.1-Fun-V1.1-14B-InP, 패치 임베딩 확장 + LoRA rank 64/α 64 | `WanTransformer3DModelWithConcat`가 입력 채널 36 + C로 패치 임베딩 확장. 학습 가능 패턴 `patch_embedding` + LoRA | `training/wan/models/wan_transformer3d_with_conditions.py:136–179`, `exps/smoke/*.yaml` |
| C = 16(메시만), 48(메시 + 두 깊이) | `condition_channels: 16 / 48`. 채널 수가 어긋나면 예외 | 같은 파일 262–270 |
| 마스크 전부 1(정적 스트림 보존) | `conditioning_mode: dwm`에서 `mask_latents = torch.ones(...)` | `train_dwm_wan.py:6446–6450` |
| 고정 카메라는 첫 프레임 반복 | `static_videos = videos[:, 0:1].repeat(...)` | `train_dwm_wan.py:6436` 부근 |
| AdamW(0.9, 0.95), 감쇠 0.01, lr 1e-4, 코사인 + 재시작, 워밍업 100, 클립 1.0, 배치 1 · 누적 4, bf16, 체크포인팅 | smoke 설정이 같은 값 | `exps/smoke/wan21_14b_smoke_48ch.yaml` |
| 목표 속도 ε − z_0, z_σ = (1 − σ) z_0 + σ ε | 같음 | `train_dwm_wan.py:6687–6699` |
| 해상도 480x832, 81프레임, 16 fps | 같음 | 설정 YAML |
| RoboCasa 명목 상태: 충돌 없는 shadow rollout | `disable_geom_collisions`로 shadow 환경 충돌을 끄고 행동을 재생 | `data_processing/robocasa/extract_dit4dit_nominal_state_conditioning.py`, `collect_dit4dit_clipstart_rollout_dataset.py:557` |
| AdaLN 기준선: 4프레임 쌓기(DROID 7차원 → 184차원 합동 입력의 쌓기), HRDexDB의 4프레임 델타 | `window4_stack`, `delta4_stack` 모드가 같은 규칙 | `training/wan/robot_state_conditioning.py:257–360` |
| DROID 클립은 겹치지 않는 81프레임 | `--clip_stride` 기본값이 `--clip_len` | `data_processing/droid/make_render_v2_clips81.py:92` |
| LightX2V 4 스텝, 가이던스 1 | 옵션으로 지원 | `README.md` 추론 절 |

## 2. 차이 D1–D12

| # | 항목 | 논문 | 코드 | 영향 |
|---|---|---|---|---|
| D1 | 새 패치 임베딩 채널의 초기값 | 서술 없음 | **0으로 초기화**하고 사전학습 36채널만 복사 | 조건 스트림이 학습 초기에 출력에 영향을 주지 않는다. 구현 세부이며 논문 설명(확장)과 모순은 아니다 |
| D2 | 손실 | 식 6, 순수 속도 MSE | `custom_mse_loss`: `abs(차이) ≤ 50` 마스크(임계값 50), 선택적 공간 마스크, 가중치 `weighting_scheme: none`(기본), 균일 시간 샘플링(`uniform_sampling` 기본 True) | 임계 50은 사실상 비활성이지만 식 6에는 없는 항이다. 공간 마스크의 사용 여부는 데이터 경로가 `loss_masks`를 줄 때만 결정된다(smoke 설정에서는 확인되지 않음) |
| D3 | 드롭아웃 | 서술 없음 | smoke 설정: **텍스트 드롭아웃 0.1**(배치 단위로 빈 프롬프트), **EEF 깊이 드롭아웃 0.1**(깊이 잠재를 0으로), 로봇 드롭아웃 0.0 | EEF 깊이를 10% 가린 채 학습한다. 표 3의 "깊이 추가" 행이 이 설정으로 학습됐는지 논문에서 알 수 없다. 텍스트 드롭아웃은 CFG를 켜는 추론(README 기본 50 스텝 · CFG 6)과 연결된다 |
| D4 | 학습 스텝 | 부록 B: DROID nominal 진단 모델 15,010, 나머지 10,000. 표 1 모델의 스텝 수는 서술 없음 | smoke 설정 80 스텝. **공개 체크포인트**: Wan RGB(DROID) 8,000, Wan 깊이(DROID + RoboCasa) 8,000, Cosmos3-Super 5,000(batch 32) | 공개 체크포인트는 논문의 10,000/15,010 스텝 모델과 다른 학습 길이다(`repo/docs/checkpoints.md`) |
| D5 | 공개 체크포인트와 논문 변형 | 표 3 변형 모두 DROID 학습. 표 1 Wan 모델은 DROID + RoboCasa, 메시 + 깊이 | RGB 체크포인트는 DROID만, 깊이 체크포인트는 DROID + RoboCasa. 문서가 "matched-data RGB/depth ablation이 아니다"라고 명시 | 공개 체크포인트로 표 3의 절제(원시 대 명목, 깊이 유무)를 다시 만들 수 없다. 표 1의 Wan 행은 깊이 체크포인트가 가장 가깝다 |
| D6 | **DROID 명목화 코드** | 원격조종 관절·그리퍼 목표를 장면 없는 Isaac Lab에서 재생(§4.1, 부록 A) | **저장소에 Isaac Lab 재생 스크립트가 없다.** 학습·AdaLN 로더가 미리 만든 `joint_position_isaac_nominal.npy`, `gripper_position_isaac_nominal.npy`를 읽는다. 메시 렌더 스크립트(`render_robot_mesh_videos.py`)는 기본 접미사 `""`로 manifest의 관절 파일을 읽고, 로더 `load_joint_gripper_from_h5`는 `observation/robot_state/joint_positions`(로그된 실현 상태)를 `action/joint_position`보다 먼저 찾는다(manifest의 관절 파일이 어느 항목에서 나왔는지는 export 단계 코드에 달려 있어 정적으로는 확정하지 못했다) | 논문의 핵심 기여인 DROID 명목 궤적은 저장소만으로 재생성할 수 없다. 렌더링 단계는 어떤 관절 배열이든 그릴 수 있으나, 기본 경로가 명목이 아니라 로그 상태를 읽는 순서다. 사용자가 `--joint-suffix`로 명목 파일을 지정해야 한다 |
| D7 | 정적 깊이 인코딩 | "미터 깊이를 FoundationStereo로 추정" (부록 A) | 코드 이름은 `static_disparity`. 깊이를 **역깊이로 변환해 [0.2 m, 3.0 m]로 클립 후 16비트 정규화**(`encode_inverse_depth_u16`, `--near-m 0.2 --far-m 3.0`) | 논문에 없는 전처리 세부. 모델이 보는 것은 미터가 아니라 클립된 역깊이다. 3 m 이상은 같은 값이 된다 |
| D8 | AdaLN 기준선의 EEF 상태 | 7차원 [x, y, z, roll, pitch, yaw, g], Panda 정기구학(부록 B) | `_panda_fk`가 DH 변환 8개(링크 1–7과 플랜지 오프셋 0.107)까지만 곱해 **플랜지 포즈**를 낸다(그리퍼 오프셋 0.1034와 −π/4 회전은 목록에 있으나 `range(8)`에서 제외). 로봇 베이스 좌표. 카메라 정보 없음. 상태 MLP의 마지막 층은 0 초기화 | AdaLN 기준선은 그리퍼 끝이 아니라 플랜지의 포즈를 받는다. 카메라 정렬 정보가 전혀 없다(`03` §2.1) |
| D9 | 캡션 | "장면 설명 T는 보이는 장면과 문맥만 설명" | 설정은 `general.text` 필드를 쓴다. 이 프롬프트(`clip_general.txt`)는 영상 **클립 전체에서 균일 추출한 프레임**을 보고 쓰며 행동·목표 동사를 금지하고 "로봇이 작업 공간 위로 뻗는다" 같은 일반 문장 하나로 끝낸다. 같은 저장소에 `scene_only`, `scene_physics`, `detailed` 변형도 있다 | 캡션이 정답 영상 전체를 보고 만들어지므로 미래 프레임의 물체 위치나 상태가 장면 설명에 들어갈 수 있다. 프롬프트가 행동 서술을 금지하지만 어느 변형으로 논문 결과를 냈는지는 설정 하나(`general.text`)로만 알 수 있다 |
| D10 | Cosmos3 백엔드 | 논문 없음 | README가 "additional Cosmos3 version"이라 밝힘. 공개 Cosmos3-Super는 **RGB만** 조건(깊이 없음), H2R 예시 로봇은 ALLEX/RB-Y1(모델 카드). 분산 학습 경로는 "experimental"이며 GPU 검증 미완료(`docs/cosmos3/verification.md`) | 논문 그림 6의 H2R(Panda + Inspire, 깊이 스트림 사용, 부록 F)이 어느 모델로 만들어졌는지 논문에 없고, 공개 H2R 모델은 RGB 전용이다 |
| D11 | 추론 기본값 | 4 스텝, 가이던스 1(LightX2V 증류, 부록 B) | README 기본은 50 스텝, 가이던스 6, 음의 프롬프트 `"bad detailed"`. 4 스텝은 별도로 받은 LoRA를 옵션으로 켠다 | 논문 보고 속도와 기본 명령의 속도가 다르다(`03` V9) |
| D12 | 데이터 규모 서술 | 41,642 + 9,380 클립 | 데이터 자체는 저장소에 없고 분할 스크립트만 있다. 분할 규칙(`build_dit4dit_clipstart_task_balanced_split.py`)은 "task balanced"로 이름만 확인 | 클립 수를 코드로 재현할 수 없다 |

## 3. 코드에 없는 것 (공개 범위)

| 논문의 요소 | 저장소 |
|---|---|
| 지표 계산(PSNR, SSIM, LPIPS), 로봇 제외 평가(마스크, 5픽셀 침식, GT 채움) | 없음 (`PSNR`, `LPIPS`, `ssim`을 저장소 전체에서 찾았으나 없음). `docs/cosmos3/verification.md`는 "metric re-evaluation is not part of this integration"이라 밝힘 |
| 과제 결과 판정(GPT-5.4, 12프레임), RoboLab·RoboCasa 선택 규칙 | 없음 |
| SVD 변형과 Ctrl-World pose 기준선 | 없음 (SVD 설정 없음. CogVideoX와 Wan의 pose 변형 설정은 있으나 논문 표 1과의 대응은 확인되지 않음) |
| RoboTwin 영점 전이, HRDexDB 미세조정, 이중 Franka 전용 설정 | 없음. HRDexDB와 RoboTwin 데이터 처리나 설정이 없다 |
| 궤적 편집(cuRobo) | 없음 |
| DROID 명목화(Isaac Lab 재생) | 없음 (D6) |
| 학습된 가중치 | Hugging Face에 별도(받지 않음) |

## 4. 구현할 수 있는지 (Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**판정: 이 맥에서 논문 수치의 재현과 학습은 불가능하다. 추론도 불가능하다. 맥에서 할 수 있는 것은 정적 읽기, 수치 재계산, 작은 토이, 렌더링 전처리 일부(조건부)다.**

| 구성 | 맥 가능 여부 | 이유 |
|---|---|---|
| Wan2.1 14B 추론 | 불가 | 가중치 bf16만 약 26 GiB. 저장소 문서는 A100 80GB 기준이고 CUDA를 전제로 한다(`setup_env.sh`: CUDA 12.x, flash-attn 휠) |
| 학습 | 불가 | 4 GPU + DeepSpeed ZeRO-2(CPU 오프로드). 데이터는 DROID(수백 GB 규모)와 RoboCasa 롤아웃. 전처리로 VAE·T5 잠재를 미리 만든다(GPU 전제) |
| DROID 명목화 | 불가 | Isaac Lab(NVIDIA GPU)이 필요하고 코드가 없다 |
| RoboCasa 명목 상태 추출 | 조건부 | robosuite/MuJoCo 기반이라 CPU 가능성이 있으나 RoboCasa-GR1 환경과 DiT4DiT 롤아웃 데이터가 필요하다. 데이터를 받지 않았다 |
| URDF 메시 렌더링 | 조건부 | yourdfpy와 렌더러는 CPU에서 가능할 수 있으나 DROID 카메라 보정·원시 영상이 필요하다 |
| 지표, 판정자 | 불가 / 불필요 | 평가 코드가 없다. 판정자는 외부 API 호출(GPT-5.4)이다 |
| 토이(`toy_leakage.py`)와 표 재계산 | **가능(완료)** | numpy/torch CPU |

**노력(추정):** CUDA GPU 4장이 있으면 환경 설치와 데이터 준비(DROID 렌더 파이프라인, Isaac Lab 명목 재생 직접 작성 포함) 1–2주, 한 번의 학습은 8,000 스텝 기준 며칠. 평가 코드(PSNR/SSIM/LPIPS, 로봇 제외 마스크)와 기준선은 직접 작성해야 하며 2–3일. 이 맥에서는 메모리 때문에 시간이 아니라 **불가능**이다.

## 5. 저자 질문

1. DROID 명목 궤적을 만드는 Isaac Lab 재생 스크립트의 공개 계획. 또는 명목 파일(`joint_position_isaac_nominal.npy`)의 배포 계획.
2. 표 1 · 3의 모델이 EEF 깊이 드롭아웃 0.1과 텍스트 드롭아웃 0.1로 학습됐는지.
3. 표 3 변형의 스텝 수와 표 1 모델의 스텝 수. 공개 8,000 스텝 체크포인트와의 관계.
4. 그림 6(DexYCB → 로봇)을 만든 모델과 사용한 조건 스트림.
5. 평가 스크립트(PSNR/SSIM/LPIPS, 로봇 제외, 판정 프롬프트)의 공개 계획.
