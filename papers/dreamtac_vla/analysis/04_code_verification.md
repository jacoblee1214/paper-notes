# 04. 공식 코드와 논문 대조

대조 대상은 `github.com/michaelyeah7/learning-to-feel-the-future` 커밋 `efd1f75ae169fc77273759df79590d81584f9566` (2026-05-16, 얕은 복제)이고 논문은 arXiv v5입니다. **코드는 정적으로만 읽었습니다.** 실행, 설치, import를 하지 않았고 체크포인트와 데이터셋은 받지 않았습니다(`02` §2). 복제본의 내용은 신뢰하지 않는 데이터로 다뤘습니다.

번호 **D1–D17**은 논문과 코드의 차이 또는 일치 항목입니다. 경로는 저장소 루트 기준입니다.

## 1. 저장소 개요와 라이선스

| 항목 | 내용 |
|---|---|
| 구성 | `ModelTrain/`(ACT 기반 학습: `module/`, `detr/`, `vjepa2_compat/`), `dobot_control/`(로봇 제어, 촉각 특징 추출, HSA 손실), `experiments/`(추론), `scripts/`(데이터 수집), `robomimic-r2d2/`, `third_party/`(벤더링). 단일 커밋, 자체 코드는 약 9천 줄(`ModelTrain/`, `dobot_control/`, `experiments/`, 벤더링 제외) |
| 학습 진입점 | `train_peg.sh` → `ModelTrain/model_train.py --policy_class ACTJEPAAdapter --enable_hsa ...` (Peg-in-Hole 예시 한 개) |
| 라이선스 | `LICENSE`는 MIT(저자) + Dobot 부분 MIT. GitHub API 라이선스 감지는 `NOASSERTION`. README는 V-JEPA / V-JEPA 2 사전학습 가중치가 Meta의 CC BY-NC 4.0이라고 따로 적음. HF 모델·데이터셋 카드 머리말은 apache-2.0 |
| 없는 것 | IsaacSim·TacEx·cuRobo 시뮬레이션 코드, 세계 모델 사전학습 코드(`vjepa2_compat/`는 모델 정의와 가중치 로더만), π0·Diffusion Policy 평가, Tool Stabilization 과제 설정, `jepa_ckpt/vitl_peg_e150.pt` |

## 2. 차이와 일치 항목

### 2.1 방법의 중심 구성 요소 (Dream 단계)

| 번호 | 논문 | 코드 | 영향 |
|---|---|---|---|
| D1 | **예측 MLP F_η, 미래 촉각 잠재 H_dream = F_η(z_τ, a_draft) (식 5), 예측 손실 L_W (식 6, 7)** | 저장소 전체에서 F_η, L_W, 미래 프레임 사용 코드를 찾지 못함. 데이터로더(`ModelTrain/module/utils.py`)는 시각 t의 촉각 이미지 한 장만 반환하고 t+N 프레임이 없음. 손실은 `policy_jepa_adapter.py`의 `l1 + kl_weight × kl`뿐. README도 "world model이 학습된다"고 쓰지만 해당 코드는 없음 | **크다.** 논문의 "Dream"(미래 촉각 예측)은 공개 학습 경로에서 구현되어 있지 않음. V-JEPA 2는 **현재 프레임 인코더**로만 쓰임 (D3) |
| D2 | 초안 행동을 입력으로 한 F_η의 출력이 정책의 추가 토큰(그림 2, 3.2절) | `detr/models/detr_jepa_adapter.py`의 `forward`: 초안 행동의 첫 스텝(14차원)을 `draft_action_proj`(선형층)로 1024차원으로 옮겨 **촉각 패치 토큰 앞에 한 토큰으로 붙이고**(`module/vitg_encoder_adapter.py` `forward`) 어댑터와 어텐션 풀링을 통과. 예측 단계가 없고, 초안 행동이 곧 조건 | 설계가 다름. 초안 행동은 같은 관측의 함수라서 새 정보가 아니라 "같은 입력을 한 번 더 통과시키는 구조"에 가까움. 이득의 원인이 미래 예측인지 2-패스 계산인지 코드만으로는 가를 수 없음 |
| D3 | 세계 모델 W_ϕ는 촉각 시퀀스로 사전학습(EMA 교사, 마스크 예측, D절)하고 동결해 **미래 임베딩을 만드는 데** 사용 | `module/vitg_encoder.py`: V-JEPA 2 인코더에 **이미지 한 장을 2프레임으로 복제**해 넣고(`num_frames=2`, tubelet 2) 패치 토큰을 반환. 예측기(`predictor.py`, `ac_predictor.py`)는 정의만 있고 사용되지 않음. 사전학습 체크포인트(`jepa_ckpt/vitl_peg_e150.pt`)와 그 학습 코드는 저장소에 없음 | 이름은 "world model"이지만 학습 경로에서는 사전학습된 **촉각 이미지 인코더** |
| D4 | 2단계 학습: 1단계(Action + HSA) 뒤 W_ϕ를 얹고 2단계에서 미세조정 (3.1, 3.2절, 알고리즘 1) | 단계 구분이 없음. `train_peg.sh`는 한 번의 20,000스텝 학습이고, 초안 패스와 정제 패스가 **같은 전방 계산 안에서 매 스텝** 함께 수행됨 (`forward`의 `use_draft=True`가 기본값이고 학습·추론 호출 모두 이 인자를 넘기지 않음) | 알고리즘 1과 다른 학습 절차. "Dream 없음" 변형을 만드는 옵션도 CLI에 없음 |
| D5 | 초안 행동: 현재 관측(촉각 포함)과 영벡터 H_null로 계산 (3.1절 끝, 식 4) | `forward_draft`: **촉각 토큰 전체를 영벡터로 대체**하고 RGB(+상태, 잠재)만으로 초안을 만듦. 초안 계산은 `torch.no_grad()` | 논문의 초안은 현재 촉각을 쓰고 미래 항만 영. 코드의 초안은 촉각을 아예 보지 않음 |

### 2.2 HSA 손실

| 번호 | 논문 | 코드 | 영향 |
|---|---|---|---|
| D6 | h_τ: 촉각 **인코더의 토큰** 평균(3.1.1절). 정렬이 촉각 인코더 표현을 시각과 맞춤 | `dobot_control/tactile_feature_extraction.py`의 `extract_tactile_features`: 정책의 **공유 CLIP 인코더**(RGB용)에 촉각 이미지를 넣어 토큰 평균을 만듦. 정책의 촉각 경로(V-JEPA 어댑터)에는 HSA 그래디언트가 가지 않음. `train_peg.sh`는 `--freeze_clip`이라 CLIP 본체는 동결이고 학습되는 것은 투영 선형층 등 | HSA가 정렬하는 대상이 "촉각 토큰"(정책이 쓰는 표현)이 아니라 CLIP 공간에서의 촉각·시각 표현 |
| D7 | L_HSA = L_HSA-W + L_HSA-TP (식 2), TP 가중치는 1(C.4는 가시성 가중 α(t)) | `HSALossWithThirdPerson`의 `use_third_person` 기본값 **False**, `tp_weight` 기본 0.5. `train_module.py`의 `hsa_config`에는 `use_third_person` 키가 없어 학습 경로에서 3인칭 항이 **꺼져 있음** (`hsa_total = hsa_wrist`만). 3인칭 특징은 계산하되 손실에 쓰이지 않음 | 논문의 두 항 손실 대신 손목 항 하나. 3인칭 정렬 효과는 공개 코드 기본 설정으로는 학습되지 않음 |
| D8 | 손목·3인칭 모두 FK와 보정된 카메라로 센서 투영 상자를 구해 상자 안 토큰만 평균 (3.1.1절, C.2) | 손목 상자는 `K_wrist`가 있어야 계산되는데 학습 경로는 `camera_params`를 CLI에서 받지 않아 `K_wrist`가 없음 → `extract_wrist_features`가 **손목 토큰 전체 평균**으로 폴백. FK 기반 상자는 3인칭 전용(D7로 꺼짐) | 기본 설정에서는 어느 시각에도 투영 상자 선택이 적용되지 않음 |
| D9 | 음성: 같은 이미지 안 상자 밖 패치(hard negatives)와 배치 내 다른 샘플 (C.4) | `HSALoss.forward`: `hard_negatives=None`이므로 **배치 내 다른 샘플만** (배치 16이면 음성 15개) | 논문의 "in-image negatives"는 구현되어 있지 않음 |
| D10 | 온도 κ = 0.07, 가중치 1.0, 특징 추출기 ViT-B/16 (표 5) | `--hsa_temperature 0.07`, `--hsa_weight 1.0`, `open_clip` ViT-B-16(768차원) | **일치** |

### 2.3 데이터, 시스템, 평가

| 번호 | 논문 | 코드 | 영향 |
|---|---|---|---|
| D11 | 단일 팔, 7-DOF 행동 (4.1절, A.2의 "6D EE delta + gripper") | 상태 차원 14 하드코딩(`detr_jepa_adapter.py`의 `state_dim = 14 # TODO hardcode`), 정책 행동 차원 16(`train_module.py`의 `action_dim: 16`), 데이터셋 카드는 `action (T, 14)`와 양팔 Dobot X-Trainer, 카메라 3대(top, left_wrist, right_wrist). 행동은 **관절 값(라디안)** 이고 추론 루프가 양팔 관절을 보냄 | 논문의 행동 표현(EE 상대 자세 + 그리퍼)과 코드의 관절 공간 14차원 양팔 정책이 다름 |
| D12 | 과제 4개, 실로봇 과제당 시연 100개, 시뮬 과제당 1000개 | `ModelTrain/constants.py`의 촉각 과제는 `peginhole_tac_1107`(에피소드 길이 350), `gearassemb_tac_1107`(700), `usb_tac_1107`(500) 세 개. Tool Stabilization 없음. 공개 데이터셋 에피소드는 52 / 50 / 50 | 공개 자료로는 3과제, 과제당 50개 안팎. 논문 표 1의 4번째 과제와 시연 100개를 공개 자료로 대조할 수 없음 |
| D13 | 정책은 ACT 계열 (B.1) 이지만 알고리즘 1은 L_DIFFUSION, 식 3은 ℓ1, 표 5는 KL 가중치 10 | `policy_jepa_adapter.py`: **L1 + 10 × KL(CVAE)**. 확산 손실 없음 | 식 3과 표 5가 코드와 같고 알고리즘 1의 L_DIFFUSION 표기만 다름 |
| D14 | 50 Hz 추론, 지수 가중 시간 집계 (식 10, B.3) | `module/model_module.py`: 매 스텝 질의(`query_frequency = 1`)와 `exp(−0.01 k)` 가중 집계. `experiments/run_inference_adapter.py`의 루프에는 **주파수 제어(sleep, 타이머)가 없고** 각 구간 시간을 출력만 함. `.cuda()`가 하드코딩. 초안 패스가 CLIP RGB 인코딩을 한 번 더 하므로 RGB 인코딩이 스텝당 2회 | 50 Hz 달성 여부는 코드로 확인되지 않음. 지연 측정은 논문에도 없음 |
| D15 | 하이퍼파라미터 (표 5, B.1, B.2) | hidden 512, FF 3200, 인코더 4층, 디코더 7층, 헤드 8, 청크 45, 배치 16, lr 1e-5, KL 10, weight decay 1e-4, 어댑터 은닉 512 / 깊이 3 / 드롭아웃 0.1 / 스케일 0.1 / 어텐션 풀링, 20,000스텝, 색 지터 (0.3, 0.4, 0.5, 0.08) | **일치** |
| D16 | RGB 인코더: 동결 ResNet-18 (B.1) | `train_peg.sh`는 `--clip_model ViT-B-16 --freeze_clip` (CLIP). ResNet 경로는 `--clip_model` 미지정 시의 레거시. 언어는 **고정 문장 하나**를 CLIP 텍스트 인코더로 임베딩해 잠재 벡터에 더함 (과제별 상수) | 3절의 "CLIP ViT"와는 일치, B.1의 "ResNet-18"과는 다름(`03` K4). 언어 조건이 과제마다 상수라 VLA의 "언어 일반화"는 쓰이지 않음 |
| D17 | HSA 기하 (C.1): K, 외부 행렬 | `policy_jepa_adapter_with_hsa.py`가 `K_tp = [[647,0,653],[0,644,364],[0,0,1]]`, `E_tp = [[1,0,0,.7],[0,-1,0,-.49],[0,0,1,1.14],[0,0,0,1]]`을 **코드에 고정**. 같은 값이 C.1에 있음. 추론 코드는 top 이미지를 `[150:420, 220:480]`로 잘라 640 × 480으로 확대 | 논문과 코드가 같은 값. 다만 이 외부 행렬의 회전 블록 행렬식이 −1(`03` V11). 자르기가 학습 데이터에도 적용됐는지는 확인되지 않음(3인칭 항이 꺼져 있어 학습에는 영향 없음) |

## 3. 정적 읽기에서의 추가 관측

1. **옵티마이저 중복 가능성(미확인).** `ACTJEPAHsa.configure_optimizers`(`policy_jepa_adapter_with_hsa.py`)는 기본 옵티마이저의 파라미터 그룹을 그대로 가져온 뒤 CLIP 인코더의 학습 가능 파라미터(투영 선형층 등)를 새 그룹으로 또 넣습니다. 그런데 이 파라미터는 `build_ACTJEPAAdapter_model_and_optimizer`(`detr/main.py`)의 "everything else" 그룹에 이미 들어 있어 PyTorch가 "some parameters appear in more than one parameter group" 오류를 낼 것으로 읽힙니다. 실행하지 않아 **확인되지 않았고**, 공개 체크포인트가 이 경로로 만들어졌는지도 알 수 없습니다.
2. **학습·추론 불일치 가능성.** 초안 패스는 학습 시 정답 행동 시퀀스로 만든 CVAE 잠재(z)를 쓰고, 추론 시에는 z = 0입니다(`encode`). 초안 행동이 정제 패스의 조건이므로, 학습 중 초안이 정답에 가깝고 추론에서는 덜 가까울 수 있습니다(가능성. ACT의 CVAE가 이미 가진 불일치와 같은 종류이지만 이 경로에서는 조건 입력까지 영향을 받습니다).
3. **어댑터 파라미터 수는 논문과 같습니다.** ViT-L에서 어댑터 1.31M + 풀링 4.20M = 5.52M (`03` V10). 코드가 출력하는 값은 실행하지 않아 보지 못했습니다.
4. **CUDA 전제.** 모델 생성 시 `model.cuda()`(`detr/main.py`), 추론 시 `.cuda()`가 하드코딩되어 있습니다. 모든 정책 클래스가 같습니다.
5. **V-JEPA 가중치.** 학습에는 `jepa_ckpt/vitl_peg_e150.pt`(peg 과제용 150 에폭 사전학습 파일로 보임, 추정)가 필요한데 공개 목록에 없습니다. HF 정책 체크포인트(과제당 약 1.67 GB, USB는 ViT-G 변형 약 4.5 GB)는 정책 전체 가중치이며 세계 모델 부분이 따로 분리되어 있는지는 확인하지 않았습니다.

## 4. 구현 가능성 판정 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**공식 결과의 재현은 이 맥에서 불가능합니다. 방법의 축소판 구현과 수치 재계산은 가능하고(수치 재계산은 완료), 논문의 중심 구성 요소(Dream)는 공개 코드에 없어 새로 구현해야 합니다.**

| 수준 | 판정 | 필요한 것과 근거 |
|---|---|---|
| 표 1 실로봇 재현 | **불가** | Dobot X-Trainer(양팔)와 DIGIT 센서, RealSense, 과제별 물체(Peg, USB 허브, 기어, 도구), 학습된 정책 |
| 표 4 시뮬 재현 | **불가** | IsaacSim과 TacEx(NVIDIA GPU 전제), cuRobo(CUDA). 해당 코드는 저장소에 없음 |
| 세계 모델 사전학습 (V-JEPA 2 ViT-L/G, 2M 프레임) | **불가** | 학습 코드가 저장소에 없고, 데이터는 시뮬 대부분. GPU 필요 |
| 공식 코드로 정책 학습 (공개 데이터 3.9 GB) | **사실상 불가** | `model.cuda()` 하드코딩과 V-JEPA ViT-L 가중치(저장소에 없음)가 필요. 코드를 MPS로 바꾸는 수정, 가중치와 데이터 다운로드, 새 가상환경 승인이 필요. 논문은 20,000스텝이 GPU에서 8–12시간이라 적음. MPS 시간은 측정하지 않음 |
| 공식 코드로 추론 | **불가** | 로봇 스택과 체크포인트(과제당 1.67 GB)가 필요 |
| 정적 대조와 수치 재계산 | **가능, 완료** | `analysis/recompute.py` (수 초) |
| **구조의 축소판 재구현** (작은 인코더, 합성 또는 공개 데이터 일부, HSA InfoNCE와 초안 → 예측 → 정제 구조) | **가능 (CPU/MPS)** | torch만 사용. 중심 질문은 "예측 잠재가 현재 관측만 쓰는 것보다 얼마나 더하는가"와 "2-패스 계산 효과와 미래 예측 효과의 분리". 노력 추정: 합성 데이터 토이 1–2일, 공개 데이터 일부 + 작은 ViT 1–2주 (추정, 측정 전) |

- **GPU나 로봇이 필요한 부분:** 표 1과 표 4 전체, 세계 모델 사전학습, 크기 600M/1B 모델.
- **이 맥에서 실질적으로 가능한 부분:** 수치 재계산(완료), 논문 구조를 작은 규모로 다시 짠 CPU 실험, 공개 데이터 일부를 받은 뒤의 분석(다운로드 승인 필요).
