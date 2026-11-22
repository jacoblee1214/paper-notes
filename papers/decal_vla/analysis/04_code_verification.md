# 04. 공식 코드 대조와 구현 가능성

## 0. 대상과 방법

- 저장소: `AureleoPKU/DeCAL` (공식. 프로젝트 페이지의 Code 링크와 저장소 설명 "Official Implementation ... (CoRL 2026)"). 커밋 `109f3904c46edc96e532e8fd2078c917191b1840` (2026-11-22 02:33 +08, "Update Data link in README.md"). 얕은 복제를 `repo/`에 두었습니다(파일 106개, 29 MB, gitignore).
- **정적 읽기만 했습니다.** 코드를 실행, import, 설치하지 않았습니다. 가중치와 데이터는 받지 않았습니다.
- 읽은 파일: `README.md`, `pyproject.toml`, `launch/decal_finetune.sh`, `launch/decal_finetune_multi_task.sh`, `src/lerobot/policies/DeCAL/{configuration_decal,modeling_decal,tactile_utils,transform_decal}.py`(전체), `src/lerobot/datasets/factory.py`(delta_timestamps), `src/lerobot/datasets/transforms.py`(증강), `src/lerobot/datasets/v30/convert_unitree_json_to_lerobot_v30.py`(일부), `tests/policies/decal/deploy_policy_tac_server.py`(전체), `deploy_policy_tactile.py`(평가 지표 부분). `transformers_replace/`(Qwen3-VL 수정판)와 `pi0/`는 읽지 않았습니다.
- 저장소 이력: 커밋 16개(2026-08-03 ~ 2026-11-21). 이슈 1개(Hugging Face 논문 페이지 제안, 답 없음). 릴리스와 체크포인트 없음.

## 1. 설치 요구사항, 라이선스, 자원

- README: Python 3.10, CUDA 12.8, torch 2.7.1 (cu128), conda ffmpeg 7.1.1, `pip install -e .`, HF 로그인과 `InternVLA-A1-3B`(리비전 고정) 다운로드, **설치된 `transformers` 안의 모듈 파일을 저장소 파일로 덮어쓰기**(`cp -r .../transformers_replace/models ${TRANSFORMERS_DIR}`), WANDB 키. `pyproject.toml`은 `transformers==4.57.1`, `torch<2.8`을 고정합니다.
- 학습 스크립트: `accelerate launch` 8프로세스(`PROC_PER_NODE=8`), NCCL 설정, `MASTER_ADDR` 기본값이 사설 IP입니다. 배포 스크립트는 `policy.cuda()`를 직접 부릅니다.
- **라이선스가 서로 다릅니다.** `LICENSE` 파일과 GitHub API는 Apache-2.0입니다. `pyproject.toml`은 `license = { text = "CC BY-NC-SA 4.0" }`이면서 분류자는 Apache입니다. 복제한 코드에는 NVIDIA Cosmos 토크나이저(Apache-2.0 표기), openpi/lerobot 파생 코드(Apache)가 섞여 있습니다. 재사용 전에 저자에게 확인이 필요합니다(D16).
- 데이터: ModelScope `Aureleo/DeCALdata_OXT`(라이선스 Apache-2.0 표기, 저장 크기 약 2.2 TB = 2,201,603,755,218 바이트, 13개 과제, 궤적 1,279개, 프레임 1,989,891개, 30 Hz). 받지 않았습니다. 크기는 메타데이터 API 값입니다.
- 사전학습 체크포인트: `InternRobotics/InternVLA-A1-3B`(약 3B). **DeCAL 학습 결과 체크포인트는 공개되지 않았습니다.**
- 내 어림(`analysis/code_budget.py`): 파라미터 약 3.75B(이해 2.12B, 생성·행동 전문가 각 0.44B, **촉각 교차어텐션 28층 0.47B**, 촉각 인코더 0.10B 등), bf16 가중치 7.5 GB, fp32 15 GB. 한 청크 추론은 약 1.9 TFLOP(어림). 논문의 0.27초에서 역산한 유효 처리량은 약 7 TFLOPS입니다.

## 2. 논문 ↔ 코드 대조 (D 항목)

| 번호 | 항목 | 논문 | 코드 | 영향 |
|---|---|---|---|---|
| D1 | 촉각 인코더 | "shared ResNet-based encoder" (§4.3) | `MultiFingerTactileEncoder`(tactile_utils.py:23–136): 3x3 합성곱 3층(64, 128, 256 채널, 보폭 2)과 1x1 투영, 손가락 10개 위 2층 4헤드 변환기. 사전학습 ResNet이 아니라 **처음부터 학습하는 작은 CNN**. 정규화는 기본 BatchNorm(`use_tactile_batchnorm=True`) | 논문 문장과 다름. 인코더 파라미터 약 0.10B (은닉 2048) |
| D2 | 전역 촉각 토큰 | "손가락 간 변환기로 전역 접촉 패턴을 요약" (§4.3) | `global_token = encoded_local.mean(dim=1)`. 어텐션 풀링은 주석 처리됨(tactile_utils.py:87–131). `attn_weights`는 1/10 상수 | 전역 토큰은 평균입니다 |
| D3 | 게이트 | `σ = Gate(z_global)`, "lightweight projection" (식 2) | `LayerNorm → Linear(2048,2048) → SiLU → Linear(2048,1) → Sigmoid`(modeling_decal.py:358–367). **네트워크 하나를 28층이 공유**하고 샘플당 스칼라 하나. 기본값 `use_tactile_gate=False`, 학습 스크립트가 `true`로 켬 | 논문의 σ ⊙ (원소별 곱)이 아니라 스칼라 곱입니다. 층별 게이트가 아닙니다 |
| D4 | 교차어텐션 | "각 블록에 교차어텐션, 질의는 prefix 투영" (§4.3) | 28층 모두 `nn.MultiheadAttention(2048, 8헤드)`(modeling_decal.py:348–357, 472–481). 질의 = 입력 정규화 직후 은닉 h, 키·값 = 손가락 지역 토큰 10개(정규화 없음). 자기어텐션 뒤, MLP 앞에 잔차로 더함 | 구조는 일치. 층당 16.8M, 합 0.47B 파라미터이고 A1-3B 사전학습 가중치가 없는 **새 파라미터**입니다(README는 A1 가중치로 초기화, 새 파라미터는 무작위) |
| D5 | 생성 전문가의 촉각 입력 | 현재 6축 힘이 힘 인코더를 거침(§4.4) | 학습 스크립트는 `use_tactile_force=true`: 손가락당 12차원(이전 힘 6 + 현재 힘 6) → `Linear(12,1024)-LN-SiLU`. **코드 기본값은 `use_tactile_force=False, use_tactile_deform=False`로, Farneback 광학흐름**(OpenCV)을 입력으로 씁니다. 이 경로는 논문에 없습니다 | 스크립트대로 돌리면 논문과 같고, 기본값은 다름 |
| D6 | 손실 가중치 | λ_v, λ_t, λ_R, λ_M, λ_F 값 미기재(식 4, 7) | λ_gen = 0.01, λ_deform = λ_raw = λ_force = 0.01. **논문에 없는 보조 손실** `lambda_tactile_encoder_deform=0.01`: 인코더 지역 토큰으로 현재 변형 지도를 복원(modeling_decal.py:1229–1241). 시각 손실은 마스크된 MSE, 힘 손실은 MSE, 행동 손실은 가중 없음 | 가중치가 매우 작아 행동 손실이 지배합니다. 보조 복원 손실은 논문 식 4에 없습니다 |
| D7 | 흐름 정합 시간 규약 | â = (1−τ)ε + τ a, **τ ~ Beta(1.5, 1.0)** (식 5), 목표 a − ε | x_t = t·ε + (1−t)·a, 목표 ε − a, t = 0.999·Beta(1.5, 1.0) + 0.001. **t = 1이 잡음**(openpi 규약). 추론은 t=1에서 0으로 Euler 10스텝 | 같은 Beta를 반대 규약에 쓰므로 논문 식을 그대로 구현하면 시간 분포가 뒤집힙니다. 코드는 openpi 쪽 규약입니다 |
| D8 | 시간 이력 | "이전 15프레임(약 0.5초)과 현재" (부록 B) | `image_delta_indices = [-15, 0, 15]`: **t−15 한 프레임, t, t+15(미래 목표)**. 생성 전문가는 t−15와 t 두 프레임만 입력. VLM에는 현재 프레임만(transform_decal.py:60 `data[k][1]`) | "15프레임 이력"은 15프레임 전의 한 장입니다. 힘도 t−15와 t 두 값 |
| D9 | 예측 지평 | 식 1의 z_{t+H}에서 H는 행동 청크(50) | 시각·촉각 목표는 **t+15 한 시점**(0.5초 뒤)(forward에서 `images[:, :, 2]`, `tactile_images[:, :, 2]`, `force_future`) | 행동 지평(50, 약 1.7초)과 다릅니다 |
| D10 | "상상" 구현 | 미래 프레임을 KV 캐시를 쓰는 병렬 디코딩으로 생성, 추론 때 미래 잠재를 행동 전문가에 제공(§4.2, §4.4) | 미래용 질의 토큰이 없습니다. **입력 토큰(t−15, t의 Cosmos 토큰 + 촉각 토큰 10개)의 출력 은닉**을 시간 평균해 디코더로 t+15 특징을 회귀(`decode_cosmos`의 `features.mean(2)`). 추론 때 디코더는 `decode_image=False`가 기본이며 행동 전문가는 이 은닉의 KV만 봅니다 | "상상"은 보조 손실이 은닉 표현을 바꾸는 것이고, 추론 때 명시적 미래 예측이 행동에 들어가는 구조가 아닙니다 |
| D11 | Factorized Flow Matching | 팔·손이 독립 잡음에서 시작, 별도 MLP, 함께 변환 (§4.2) | `split_arm_hand_action=true`(기본 False)일 때 팔 12차원, 손 44차원이 각자 `Linear`로 토큰화되어 **청크 50개씩 2개의 토큰열**이 되고(총 100 + 상태 1), 시간 MLP는 공유, **같은 t**를 씁니다. 출력도 팔·손 각 `Linear` | 잡음은 표준 가우시안 한 텐서이므로 "독립 잡음"은 원래 성립합니다. 실질 차이는 토큰열을 둘로 나눈 것입니다 |
| D12 | 학습 일정 | 선형 워밍업 2,000 + 선형 감쇠, 5e-5 → 5e-6, 100k 스텝, 청크 50 | `CosineDecayWithWarmupSchedulerConfig`(**코사인**). AdamW β=(0.9, 0.95), 가중치 감쇠 0.01, 기울기 클립 1.0(논문 미기재). `decal_finetune.sh`는 **청크 32, 120,000 스텝**, `decal_finetune_multi_task.sh`는 청크 50, 100,000 스텝. 추론 디노이징 10스텝(논문 미기재) | 두 스크립트가 서로 다르고 논문 설정은 다중 과제 스크립트와 맞음 |
| D13 | 증강 | 기준선(DECO, ViTacFormer)만 증강을 썼다고 쓴다(부록 E.4) | 학습 스크립트는 `image_transforms.enable=true`: 밝기·대비·채도·색상 지터, 선명도, ±5° 이동 5% 아핀 중 최대 3개를 프레임마다 | DeCAL도 증강을 쓰는데 본문은 말하지 않습니다 |
| D14 | 시각 생성 지표 | Cosmos 특징 코사인, LPIPS (표 3) | `deploy_policy_tactile.py`가 `cosmos_feature_cosine`(Cosmos 인코더 특징), `lpips_mean`(alex), DINOv2 CLS 코사인도 계산. 지표에 분산·표본 수 출력 없음, 평균만 | 지표는 논문과 같지만 코사인이 학습 목표 공간(Cosmos)에서 측정됨. 기준선(A1)을 같은 스크립트로 돌리는 코드는 없음 |
| D15 | 데이터와 변환기 | 부록 C: 원시 로그 → 학습 | README 변환기는 Unitree JSON 형식만 받고 `Ur5_Sharpa`만 구현. **공개 데이터는 OXT zarr**이고 OXT → LeRobot 변환기는 저장소에 없음. 코드는 손가락 10개의 `_raw` 촉각 이미지를 필수로 읽지만 데이터 README의 모달리티 표에는 변형 깊이 지도와 힘만 있고 원본 촉각 이미지는 없음. README의 데이터 링크가 두 개이고 `DeCAL_dataset`은 존재하지 않음(API 응답 "존재하지 않는 데이터셋"), 배지 링크 `DeCALdata_OXT`만 존재 | 공개 데이터로 저장소를 그대로 학습시키는 경로가 끊겨 있습니다 |
| D16 | 라이선스 | — | LICENSE/GitHub: Apache-2.0, pyproject: CC BY-NC-SA 4.0, 데이터: Apache-2.0 | 상업적 사용 가능 여부가 불명확 |
| D17 | 배포 입력 | — | 서버(`deploy_policy_tac_server.py`): 미래 프레임 자리에 `torch.rand_like`를 채우고, 이전 힘이 없으면 현재 힘을 복제, 힘은 통계 최대 절댓값으로 나눔(`/ tactile_force_scale`), **정책을 float32로 올림**(학습은 bf16). 서버에는 지연 시간 측정이 로그 한 줄뿐 | 0.27초는 재현할 측정 스크립트가 없습니다 |
| D18 | 기준선·평가·분석 코드 | 표 1–3, 그림 5–7 | GR00T, InternVLA-A1(+촉각), ViTacFormer, DECO 학습·평가 코드, 실로봇 평가 절차, t-SNE·게이트 시각화·힘 예측 그림 생성 코드는 저장소에 없음. `deploy_policy_tactile.py`가 열린 루프 지표(행동 MSE, 시각·촉각·힘 복원 지표)를 내는 것이 유일한 평가 코드 | 표 1, 2, 6은 코드로 재현 불가(실로봇) |

### 2.1 논문 주장의 코드 쪽 확인

- 맞는 것: Qwen3-VL-2B 이해 전문가 28층(`qwen3_vl_28l`), 생성·행동 전문가 Qwen3 계열 28층 1024 은닉(`qwen3_28l`), Cosmos CI8x8 토크나이저로 256×256 → 32×32 잠재(채널 16), 8×8 보폭 8 합성곱으로 4×4(§4.4와 일치), 3개 카메라, 손가락 10개, 상태·행동 56차원(arm 12 + hand 44), 행동 청크 50(다중 과제 스크립트), 단방향 주의 마스크 블록, 촉각 복원 디코더 3개(변형, 원본, 6축 힘), 학습 하이퍼파라미터(학습률 5e-5, 워밍업 2000, 장치당 배치 4, 8프로세스).
- 틀리거나 달라진 것: D1(ResNet이 아님), D3(스칼라 공유 게이트), D7–D10(시간 규약, 이력, 지평, 상상 구조), D12, D13.
- 확인하지 못한 것: 게이트가 접촉에서 커지는지(그림 6, 가중치 없음), 0.27초(측정 코드 없음), 실로봇 성공률, `transformers_replace/`의 Qwen3-VL 수정 내용(읽지 않음).

## 3. 정적 검토에서 눈에 띈 점 (D 항목과 별개)

- **게이트 크기.** 코드의 게이트는 시그모이드 출력이므로 값이 0–1입니다. 논문 그림 6의 값은 10⁻⁴–10⁻³ 크기라서 교차어텐션 출력이 학습 중에 매우 커졌거나, 그림이 다른 양(예: 별도 스케일)일 수 있습니다. 가중치가 없어 확인할 수 없습니다(`03` §5 C4).
- 촉각 변형 입력은 인코더에 현재 프레임만 들어갑니다(`tactile_images[:, :, 1]`). 시간 정보는 생성 전문가의 힘 이력(t−15, t)에만 있습니다.
- 행동 전문가는 매 디노이징 스텝에서 촉각 지역 토큰을 쓰지 않고 이해·생성 KV 캐시만 봅니다(`denoise_step`).
- `predict_action_chunk`와 학습 `forward`는 힘이 3시점(이전, 현재, 미래)이 아니면 `ValueError`를 냅니다. 배포 서버가 이를 위해 가짜 미래 값을 채웁니다(D17).
- 사용하지 않는 분기: `use_consistency_action_loss`(기본 False, 스크립트에서 주석 처리), 단일 `action_in_proj` 경로.
- `pyproject.toml`에 저자 이메일이 있습니다. 이 문서에는 옮기지 않았습니다.

## 4. 구현 가능성 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**전체 재현은 불가능하고, 구조의 축소판과 수치 검증만 가능합니다.**

| 수준 | 판정 | 근거 |
|---|---|---|
| 실로봇 재현 (표 1, 2, 6, 그림 9) | 불가 | UR5 2대, SharpaWave 손 2개와 손끝 촉각 센서, 원격조작 장비(MetaGlove, VIVE) |
| 공식 코드로 학습 | 불가 | 약 3.75B 파라미터, 8×H100 전제, 학습 메모리 어림 45 GB 이상(활성값 제외). 데이터 약 2.2 TB와 OXT → LeRobot 변환기 부재(D15) |
| 공식 코드로 추론 | 사실상 불가 | DeCAL 체크포인트 미공개, `cuda()` 고정, `transformers` 덮어쓰기와 버전 고정, Cosmos jit 모듈의 MPS 호환 미확인. 가중치를 얻는다고 가정해도 메모리는 bf16 7.5 GB, fp32 배포 코드는 15 GB로 24 GiB에 빠듯함 |
| 표 수치 재계산과 일관성 검사 | 가능, 완료 | `analysis/recompute.py`(수 초) |
| 게이트 구조의 합성 토이 | 가능, 완료 | `analysis/toy_gate.py`(약 3.7분 측정, 다른 작업과 병렬 실행 중) |
| 구조 재구현(작은 MoT + 촉각 교차어텐션 + 게이트, 합성 접촉 과제) | 가능 | torch 2.15 MPS/CPU. 노력 추정 1–2주(측정 전) |
| 공개 데이터 일부로 축소 학습 | 조건부 | 2.2 TB 중 일부 다운로드 승인과 OXT 변환기 작성 필요. 노력 2–3주(추정) |

노력 추정은 모두 측정 전의 어림입니다.

## 5. 재실행

```bash
python3 analysis/recompute.py          # 수 초
python3 analysis/code_budget.py        # 즉시
/path/to/venv/bin/python analysis/toy_gate.py   # torch CPU, 수 분
```
