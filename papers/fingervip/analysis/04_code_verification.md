# 04. 논문과 공식 코드 대조, 구현 가능성

## 0. 대상과 범위

- **저장소:** https://github.com/zhzhen0626/FingerViP, 커밋 `aba80110bf5777cd926f394ac6e870c877e21c6c` (2026-09-24, "init FingerViP", 단일 커밋). 얕은 클론(`git clone --depth 1 <url> repo`)을 정적으로만 읽었습니다. 실행, import, 설치, 가중치·데이터 다운로드를 하지 않았습니다. 이 커밋의 `repo/`가 아래 파일 경로의 기준입니다.
- **구성:** `DiffusionPolicy/`(데이터 변환, 학습, 평가 추론), `Teleop/`(Vision Pro 원격조작, 데이터 수집, ROS 배치 노드), `Hardware/`(손끝 모듈 CAD와 BOM, 모터 보정). 전체 추적 파일 492개.
- **라이선스:** 루트 `LICENSE`는 MIT. `Hardware/LICENSE`는 CC BY-NC-SA 4.0이라 상업 이용과 동일조건 규칙이 따로 걸립니다. `Teleop/LICENSE`는 MIT(REAL 연구실), `DiffusionPolicy/diffusion_policy/LICENSE`는 MIT(REAL + Columbia 표기). 팔·손 자산에는 별도 `LICENSE`가 있습니다.
- **대조한 파일:** 설정 `config/train_fingervip_workspace.yaml`, `config/task/fingervip.yaml`, `config/task/base.yaml`; 정책 `policy/diffusion_transformer_timm_policy.py`; 인코더 `model/vision/transformer_obs_encoder.py`; 디노이저 `model/diffusion/transformer_for_action_diffusion.py`; 데이터 `dataset/fingervip_dataset.py`, `process_data.py`; 학습 `workspace/train_diffusion_transformer_timm_workspace.py`; 추론 `eval_fingervip.py`; 배치 `Teleop/eval_robot_node.py`, `Teleop/args/eval_args.yaml`. (경로는 `DiffusionPolicy/diffusion_policy/` 아래, 마지막 둘은 `Teleop/`.)

## 1. 논문과 맞는 항목 (K1–K12)

| 번호 | 논문 | 코드 위치 | 결과 |
|---|---|---|---|
| K1 | 얼린 CLIP ViT-B/16 인코더 (§3.3) | `train_fingervip_workspace.yaml`: `model_name: vit_base_patch16_clip_224.openai`, `frozen: True`, `pretrained: True`, `feature_aggregation: cls` | 일치. 설정 주석에 DINOv2 대체 줄이 있고 표 3의 변형은 이 설정을 바꿔 만든 것으로 보임 |
| K2 | 관측 이력 2, 행동 궤적 16, 실행 8 (§3.3, 표 8) | `fingervip.yaml`: `img_obs_horizon 2`, `low_dim_obs_horizon 2`, `action_horizon 16`; `eval_args.yaml`: `num_steps: 8` | 일치. 평가 노드의 코드 기본값은 `num_steps = 1`이고 설정 파일이 8을 지정 |
| K3 | 배치 64, AdamW, β (0.95, 0.999), 학습률 3e-4, 가중치 감쇠 1e-6 (표 8) | `train_fingervip_workspace.yaml`의 `dataloader`, `optimizer` | 일치 |
| K4 | 노이즈 제거 16스텝, 어텐션 헤드 8 (표 8) | `num_inference_steps: 16`, `n_head: 8`, 평가에서 `self.policy.num_inference_steps = 16`(`eval_fingervip.py` L141) | 일치 |
| K5 | 관측: 손끝 RGB 5 + 외부 RGB 1, 224×224, 관절 위치 26, 손가락 전류 5×4, 카메라 자세(위치 3 + 회전 6) (§3.3) | `config/task/fingervip.yaml`의 `shape_meta` | 일치 |
| K6 | 손끝 카메라 자세는 Pinocchio FK로 손 기저 좌표에서 계산, 회전은 앞 두 열 (§3.3) | `process_data.py` L336–L370: `pin.forwardKinematics`, 프레임 `thtip_camera` 등, `T_world_base.inverse() * T`(기저 변환 마스크 `1,1,1,1,1`을 README가 안내), `rotation[..., :2]` | 일치. 두 열을 행 우선으로 편 순서(`reshape(1,6)`)라서 논문 `vec(R[:, :2])`의 열 우선과 배열 순서가 다를 수 있으나 선형 투영 앞이라 학습에는 영향이 없음 |
| K7 | 자세는 위치와 방향을 각각 128차원으로 투영해 256차원 (§3.3) | `transformer_obs_encoder.py` L256–L263: 전류와 자세를 둘 다 쓸 때 `n_emb // 6 = 128`, 두 개를 이어 256 | 일치 |
| K8 | 손가락당 전류 4차원 → 128 (§3.3) | 같은 파일 L264–L267, L423–L427 | 일치 (가중치 공유 여부는 D6) |
| K9 | 다섯 손끝 특징은 공유 선형층, 외부 특징은 별도 선형층 (§3.3) | `key_projection_map['camera1_rgb']`와 `['tv_rgb']`, `share_rgb_model: True` | 일치 |
| K10 | 손끝 토큰 = 시각 투영 ⊕ 자세 ⊕ 전류, 외부는 전역 토큰, 관절 위치 임베딩을 이어 붙임 (§3.3) | `forward` L439–L441, L463–L468, L477–L496 | 일치. 합 차원이 768이 되도록 손끝 시각 투영이 384차원으로 설정됨(D8) |
| K11 | 관절 위치 투영 768 (표 8) | `n_emb: 768`, 관절 위치를 (B, T, 26)으로 펴서 선형 투영 | 일치 |
| K12 | 데이터 증강: ColorJitter(0.3, 0.4, 0.5, 0.08), 손끝 카메라에만 GaussianBlur(커널 3, σ ∈ [0.1, 0.3]) (표 9, 10) | `train_fingervip_workspace.yaml` `transforms` | 값 일치. 확률과 외부 카메라 설정은 D4 |

## 2. 논문과 다른 항목 (D1–D17)

"영향"은 정적 읽기에서의 판단이고 실행으로 확인하지 않았습니다.

| 번호 | 논문 | 코드 | 영향·메모 |
|---|---|---|---|
| D1 | 그림 3에 "DDPM Scheduler", 본문은 Diffusion Policy를 따르는 조건부 노이즈 제거. 학습 스텝 수와 샘플러 종류는 적지 않음 | `config/train_fingervip_workspace.yaml` L21–L22: `diffusers.DDIMScheduler`, `num_train_timesteps: 50`, `squaredcos_cap_v2`, `prediction_type: epsilon`, `clip_sample: True`. 정책 파일이 `DDPMScheduler`를 불러오지만 타입 힌트용이고 설정이 DDIM을 인스턴스화함 | 샘플링이 DDIM(기본 η = 0, 결정적)으로 16스텝. 논문 표기와 다름 |
| D2 | 언급 없음 | `diffusion_transformer_timm_policy.py` L204, `input_pertub: 0.1`: 학습 때 잡음에 0.1 배의 추가 가우시안을 더한 `noise_new`로 노이즈를 입히고 목표는 원래 `noise` (DDPM-IP 방식의 입력 교란) | 논문에 없는 학습 기법. 모든 방법에 같이 적용됐는지는 알 수 없음 |
| D3 | 224×224 RGB를 얼린 CLIP에 넣음 | `imagenet_norm: True`라 ImageNet 평균·표준편차로 정규화(L175–L178). timm `vit_base_patch16_clip_224.openai`의 사전학습 설정은 OpenAI CLIP 평균·표준편차를 쓰는 것으로 알려져 있으나 이 분석은 timm 설정을 읽지 않음 | CLIP 입력 통계와 다를 수 있음. 영향 크기는 모름. 논문에 정규화 설명 없음 |
| D4 | 표 9, 10: 증강 값만 있음. 블러는 "randomly apply" | 블러는 `RandomApply(p: 0.02)`(설정 L59)로 2%만 적용. 외부 카메라(`tv_rgb`)는 ColorJitter를 더 약하게 덮어씀(밝기 0.85–1.15, 대비 0.8–1.2, 채도 0.8–1.2, 색조 ±0.03, `transformer_obs_encoder.py` L202, L239) | 논문 표의 값은 손끝 카메라 기준. 외부 카메라와 블러 확률은 미기재 |
| D5 | 언급 없음 | `config/task/base.yaml` L25–L26, `fingervip_dataset.py` L237–L249: 외부 카메라를 뺀 각 카메라의 이미지 시퀀스를 확률 0.05로 0으로 대체(`random_dropout`) | 손끝 카메라 무작위 탈락 증강. 논문에 없음. 한 시점이 가려질 때의 강건성에 기여했을 수 있음 |
| D6 | "손가락별 4차원 전류를 독립적으로 투영해 손가락별 독립성 보존" (§3.3) | 선형층 하나 `Linear(4, 128)`를 다섯 손가락에 공유해 적용(`reshape(-1, 4)`). 자세도 `Linear(3, 128)`, `Linear(6, 128)` 하나씩 공유 | "독립"은 입력을 손가락별로 나눈다는 뜻이고 가중치는 공유. 손가락 구분은 아래 D7의 위치 임베딩에 의존 |
| D7 | 언급 없음 | `transformer_for_action_diffusion.py` L197–L204: 조건 토큰마다 학습 가능한 위치 임베딩 `cond_pos_emb`를 더함. 자세 인코딩 때문에 끄려던 줄(`no_cond_pos_emb=...add_camera_poses`)은 주석 처리 | 자세 인코딩을 뺀 절제("w/o camera poses")에도 토큰 위치별 구분이 남아 있어, 절제가 보는 것은 손가락 정체 정보가 아니라 기구학 자세 정보로 읽힘 |
| D8 | 시각 투영 차원, 트랜스포머 층 수, 드롭아웃은 적지 않음(표 8은 헤드 8만) | 손끝 시각 투영 384차원(`n_emb // 2`), 외부 시각 투영 768차원, 층 수 `n_layer: 7`, `p_drop_attn: 0.1`, 디코더는 자기+교차 어텐션 `nn.TransformerDecoder` | 재현에 필요한 값이 코드에서만 확인됨. 토큰 수는 (외부 1 + 손끝 5) × 이력 2 + 관절 토큰 2 + 시간 1 = 15 |
| D9 | 학습 에폭, 스케줄은 없음. 학습 시간 평균 37시간(RTX 4090, C.3) | `num_epochs: 100`, 코사인 스케줄 + 워밍업 2,000스텝, EMA(power 0.75, 최대 0.9999)를 평가에 사용(`eval_fingervip.py` L139), 시드 42, 인코더용 별도 학습률 `obs_encoder_lr: 3e-5`(인코더가 얼려 있어 적용 대상 없음) | 에폭 수와 EMA는 논문에 없음 |
| D10 | 1,012개 데모로 학습 (§B.1) | `val_ratio: 0.05`로 에피소드의 5%를 학습에서 제외(`get_val_mask`, 최소 1개). 워크스페이스의 검증 손실 계산은 주석 처리. 체크포인트는 `train_loss`가 낮은 것 20개(`topk.monitor_key: train_loss`) | 실제 학습 에피소드는 약 95%. 평가에 쓴 체크포인트 선택 규칙은 논문에 없음 |
| D11 | 전류는 mA 단위 그래프(그림 8), 전처리 설명 없음 | `process_data.py` L340–L361: 원시 값에 `2.69e-6`(A)을 곱하고 손가락 순서를 재배열, 손끝별 4개 전류를 `finger_camera_order = [4, 0, 2, 3, 1]`로 매핑. 이후 범위 정규화기 | 전류 단위·정규화 과정이 논문에 없음. 손가락 매핑은 일관됨(엄지→4번째 묶음 등) |
| D12 | 기준선 TVC, WC, TVC w/ WC, FTC w/ WC가 표 1에 있음 | 저장소에 손목 카메라 영상을 기록하거나 쓰는 코드가 없음(`wrist` 검색에서 손목 자세 계산과 팔 링크 이름만 나옴). 설정은 손끝 5 + 외부 1 구성만 | 이 기준선들은 공개 코드로 만들 수 없음. 손끝 전용(FTC)과 두 절제는 설정 변경으로 가능해 보임(미실행) |
| D13 | 표 2(지연·FLOPs), 표 3, 표 5(오프라인 행동 MAE) | 측정 코드 없음. `eval_fingervip.py`는 실로봇 추론용이고 "합성 입력 점검"(README)은 CUDA 전제 | 표 2, 3, 5는 공개 코드로 다시 만들 수 없음 |
| D14 | 제어 시스템 100 Hz (부록 B), 시스템 속도 10.2 FPS (표 2), 앞 8개 행동 실행 (§3.3) | `Teleop/base_robot_system.py` L48: `control_rate_hz` 기본 125. `eval_robot_node.py` L140: 정책 간격 `int(125 / 10) = 12` 제어 주기. 예측한 8개 행동을 각각 12 주기(약 96 ms)씩 유지하며 선형 보간하고, 한 묶음(8 × 12 = 96 주기, 약 0.77 s)이 끝난 뒤에야 새 예측을 받음(L683–L720, L745–L755) | 행동 한 스텝은 약 10 Hz로 갱신되나 관측 갱신(새 예측)은 묶음마다 약 1.3 Hz. 논문의 "10.2 FPS"가 이 중 무엇을 가리키는지는 코드로 정할 수 없음 |
| D15 | 데이터는 카메라 30 Hz, 지령 100 Hz로 수집 (부록 B) | `Teleop/robot_node.py`의 `_handle_robot_data`는 제어 루프 반복마다 한 프레임을 저장(`base_robot_system.py` L489–L549). 학습의 행동 스텝 간격(`down_sample_steps: 1`)과 배치 때 한 스텝 유지 시간(약 96 ms)이 같은지 정적으로 확인하지 못함 | 확인 불가. 학습 기록 주기와 배치 행동 간격이 다르면 시간 척도 불일치가 생길 수 있어 저자 확인이 필요 |
| D16 | 640×480 카메라, 224×224 입력 | `process_data.py` L94–L125: 640×480 중앙을 480×480으로 자른 뒤 224×224로 줄임(`INTER_AREA`) | 가로 방향 25%(좌우 각 12.5%)를 버림. v3 본문에는 자르기 언급이 없고 v1에 "center-cropped" 서술이 있었음 |
| D17 | 시연 1,012개와 학습된 정책 | 저장소와 README, 프로젝트 페이지에 데이터와 체크포인트 링크가 없음 | 코드만으로는 논문의 수치를 낼 수 없고 데이터를 새로 모아야 함 |

## 3. 저장소 구성과 완성도

- **README 흐름:** 손 조립(`Hardware/`) → Vision Pro 원격조작·데이터 수집(`Teleop/`) → `process_data.py`로 Zarr 변환 → `train_fingervip.py`로 학습 → ROS 노드로 배치. Ubuntu 20.04, Conda, Python 3.10, torch 2.4.1, timm 0.9.7, diffusers 0.18.2, pin 2.7.0, ROS 1(Python 3.10용 `cv_bridge`를 직접 준비)을 요구합니다. 모든 의존성이 CUDA 12 휠로 고정되어 있습니다.
- **하드웨어 자료:** 손끝 모듈 STEP·STL(손가락 4 + 엄지 1)과 BOM·조립 PDF. 나머지 손 구조는 RAPID Hand 원본을 쓰라고 안내합니다(`Hardware/mechanical_structure/README.md`). 모터는 Dynamixel XC330-M288-T입니다.
- **이슈:** 열린 이슈 1건(PCB 제작 문의, 2026-10-12).
- **다른 가지:** `config/task/panoptes.yaml`, `train_diffusion_transformer_snake_workspace.yaml` 등 RoboPanoptes·UMI에서 온 파일이 남아 있습니다. FingerViP 학습은 `train_fingervip_workspace.yaml` 하나입니다.

## 4. 구현 가능성 판정 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**판정: 논문 결과의 재현은 이 맥에서 불가능합니다. 병목은 연산량이 아니라 하드웨어와 데이터입니다. 정책 구조의 소형 재구현은 데이터 없이 합성 입력으로 가능하고, 정적 대조와 수치 재계산은 끝냈습니다.**

| 구분 | 필요한 것 | 이 맥 | 판단 |
|---|---|---|---|
| 하드웨어 제작과 보정 | RAPID Hand 부품 + 손끝 모듈(3D 프린트, IMX258 카메라 5, Dynamixel 모터, PCB), UR5e, D435i 2, Vision Pro, PCIe USB 카드 | 불가 | 장비와 제작이 필요. 노력 추정(측정 없음)은 수 주에서 수 개월 |
| 데이터 수집 | 과제 4개 × 약 250개 시연 (평균 길이 6–14초, 총 약 2.5시간) | 불가 | 공개 데이터 없음(D17). 위 하드웨어가 있어야 함 |
| 정책 학습 | RTX 4090급에서 정책당 평균 약 37시간 (부록 C.3), 방법 8 × 과제 4 정책이면 약 1,184 GPU 시간(정책 수는 가정) | 사실상 불가 | CUDA 전제 설정(`device: cuda:0`)과 CUDA 12 고정 환경. 데이터도 없음 |
| 실로봇 평가 | 방법당 184회, 같은 초기 조건 | 불가 | 로봇과 기준선 코드(D12) 필요 |
| 정적 읽기와 수치 재계산 | 논문 표, 저장소 | 가능, 완료 | `analysis/recompute.py`, 이 문서 |
| 정책 구조 재구현(합성 입력) | torch로 토큰 구성, 7층 디노이저, 16스텝 샘플링 | 가능 | 새 설치 없이 torch만 있으면 됨. ViT-B/16 사전학습 가중치는 따로 받아야 하며(미다운로드) 임의 가중치로는 구조 점검만 가능 |
| 추론 속도 | 임의 가중치로 측정 | 완료 | `analysis/toy_cost.py`: MPS fp32에서 Ours형 약 178 ms, 기준선형 약 111 ms (인코더 + 디노이저 16스텝, 임의 가중치, 카메라·전처리 제외) |
| 손끝 카메라 자세 계산 검사 | URDF 두 개(저장소에 있음), 순기구학 | 가능(조건부) | Pinocchio 설치 승인이 있으면 저장소 방식 그대로, 없으면 URDF를 직접 읽는 순기구학 구현이 필요 |

- **노력 추정 (측정 없음):** 합성 입력 정책 재구현은 1주 안팎. 사전학습 CLIP 가중치를 쓰는 소형 학습(합성 과제)은 가중치 받기 승인 뒤 2–3주. 논문 설정으로 처음부터 다시 만드는 일은 하드웨어를 포함해 3개월 이상으로 봅니다.
- **정책 속도 참고:** 위 측정값은 정책 한 번 호출에 해당하는 값입니다. 평가 노드가 새 예측을 묶음마다(약 0.77 s) 받으므로(D14) 호출 지연이 제어 주기에 곧바로 들어가지는 않습니다.

## 5. 확인하지 못한 것

- 코드를 실행하지 않았으므로 D1–D11의 영향 크기는 모릅니다.
- timm 설정 값(D3)은 파일로 읽지 않았고 알려진 값에 기대었습니다.
- 학습 데이터 기록 주기와 배치 행동 간격(D15), 표 2의 속도가 가리키는 대상(D14)은 코드만으로 정할 수 없습니다.
- `Teleop/`의 재타겟팅과 ROS 노드, `Hardware/`의 모터 보정은 정책 대조 범위가 아니어서 자세히 보지 않았습니다.
