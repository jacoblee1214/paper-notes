# 04. 공식 코드 대조와 구현 가능성

## 0. 대상과 방법

- 저장소: `Barrybarry-Smith/TROGraspV2`(프로젝트 페이지의 Code 링크, README가 공식이라고 명시). 커밋 `449db5a16754412f3f31f516dbf4a601897f58ff`(2026-10-27, "add citation"). 커밋은 5개(2026-10-18 initial commit부터 2026-10-27)입니다. 얕은 복제를 `repo/`에 두었습니다(287 MB, 추적 파일 1,181개).
- **정적 읽기만 했습니다.** 코드를 실행, import, 설치하지 않았습니다. 가중치(Google Drive 링크)와 데이터(HF 6.19 TiB)는 받지 않았습니다. 저장소에 포함된 `.pt` 임베딩 파일은 열지 않았습니다.
- 읽은 파일: `README.md`, `install.sh`, `install_isaac.sh`, `config/{train,val,val_wuji_hand}.yaml`, `train.py`, `test.py`, `model/{troe_graph,tro_denoiser,vqvae_encoder}.py`, `model/vqvae/vq_vae_ft.py`(일부), `dataset/{TROMixedHandZarrDataset,TROFilteredDatasetTestV2}.py`, `utils/{rotation,controller,pyroki_ik,tro_hand_model}.py`, `get_edge_embedding.py`, `validation/{val_lift,convert_before_eval,collision_check}.py`, `validation/validation/{isaac_main,isaac_validator_lift}.py`, `vis_grasp.py`(grep). `model/tro_denoiser.py`에는 클래스가 세 쌍(`TROGraph*`, `TROEGraph*`, `TROLGraph*`) 있고 학습에 쓰이는 것은 `TROEGraphDenoiser`입니다(`troe_graph.py` 임포트).
- 판정: **일치(K)**는 논문과 코드가 같음, **차이(D)**는 다르거나 논문에 없는 기능, **미확인**은 저장소에서 찾지 못함.

## 1. 설치 요구사항과 라이선스

- `install.sh`: conda `TROV2`(Python 3.10), torch 2.4.1 + cu124, pytorch3d(소스 빌드), torch-cluster(`torch-2.4.1+cu124` 휠 인덱스), chamferdist, theseus-ai 0.2.3, torchlie, accelerate, pytorch-kinematics, viser, fpsample, bps_torch 등. **스크립트가 `~/.pip/pip.conf`를 알리바바 미러로 덮어씁니다.** 맥에서 그대로 실행하면 전역 pip 설정이 바뀝니다.
- `install_isaac.sh`: Isaac Gym Preview 4(Python 3.8 별도 환경, NVIDIA 다운로드). 시뮬 평가는 NVIDIA GPU와 Linux가 필요합니다. 맥에서 불가.
- 추론은 JAX(Pyroki IK, `jax.jit`)와 `accelerate`의 bf16 autocast를 씁니다. pytorch3d, torch-cluster, chamferdist는 CUDA 휠을 전제한 설치 절차입니다. macOS(MPS/CPU) 호환은 확인하지 못했습니다.
- **라이선스.** 저장소 루트에 LICENSE 파일이 없고 GitHub API의 license는 `null`입니다. `third_party/pyroki`만 MIT(LICENSE 있음)입니다. HF 데이터 카드는 "라이선스·재배포 조건 미선언"이라고 적습니다. 코드·데이터 재사용 조건이 불명확합니다.
- **하드코딩 경로.** `/mnt/fx/TRO-1B-Dataset/...`가 `config/val.yaml`, `dataset/TROFilteredDatasetTestV2.py`, `validation/{convert_before_eval,val_lift,collision_check}.py`에 박혀 있고 `train.sh`도 머신 전용 acc 설정을 가리킵니다. README는 `val_lift.py`의 서브프로세스 경로를 사용자가 고치라고 안내합니다(알려진 버그 수정 미반영).

## 2. 논문 ↔ 코드 대조

### 2.1 일치 (K 항목)

| 번호 | 항목 | 논문 | 코드 | 비고 |
|---|---|---|---|---|
| K1 | 아키텍처 크기 | 0.2B: 8층, 12헤드, 노드·간선 768, 출력층 [512, 256, 128](표 10) | `config/train.yaml`: `v_conv_dim: 768`, `e_conv_dim: 768`, `num_layers: 8`, `c_atten_head: 12`, `v_out_hidden_dims: [512, 256, 128]` | 일치 |
| K2 | 매개변수 수 | 표 4: 176.53M | 위 설정으로 층별 산술 계산(`analysis/recompute.py` V12): 176.53M | **소수점까지 일치** |
| K3 | 손실 | 위치·회전 잡음 MSE, γ_pos = γ_rot = 1(식 8) | `troe_graph.py` `forward`: `loss_total = loss_trans_noise + loss_rot_noise`. 유효 링크 마스크 평균, 성분 평균 | 일치. 가중치가 없는 합 |
| K4 | 학습 하이퍼 | lr 1e-4, 워밍업 1,000, 클리핑 1.0, bf16, 배치 48, N_t = 5, 코사인 노이즈 s = 0.008, T = 1,000(§9) | `train.yaml`: `lr: 1e-4`, `warmup_steps: 1000`, `grad_clip: 1.0`, `mixed_precision: bf16`, `batch_size: 48`, `N_t_training: 5`, `cosine_s: 0.008`, `M: 1000` | 일치(옵티마이저·스케줄은 D6) |
| K5 | 입력 크기 | 물체 512점, 환경 2,048점, P = E = 25(§9) | `TROMixedHandZarrDataset`: `num_object_points=512`, `num_env_points=2048`. config의 `object_patch: 25`, `env_patch: 25` | 일치. 환경 점은 물체 중심에서 가까운 2,048개(`topk(d2, largest=False)`)로 고르며 논문에 서술 없음 |
| K6 | 그래프 구성 | 로봇 노드 = [자세, BPS, 위치 특징], 간선 = 상대 6D 자세, 순서 OR → ER → RR, 모든 층 특징 모음(식 3–5, 7) | `V_robot_dims: [3, 3, 128, 128]`. `relative_pose_6d`(링크 틀 기준 상대 변환 + `so3_log_map` 회전). `TROEGraphLayer`가 OR → ER → RR 순. `v_robot_wide_fc`가 층 특징을 모음 | 일치(어텐션 세부는 D9) |
| K7 | IK | 8개 초기값 병렬, 링크 위치 오차 최소 선택(§9) | `test.py` `num_multistart = 8`, 첫 시드는 정준 구성, 나머지는 관절 한계 안 균일 난수. 후보를 FK해 평균 링크 위치 오차로 선택 | 일치 |
| K8 | 추론 시드 | η = 1.0(§9) | `val.yaml` `ddim_eta: 1.0` | 일치(λ와 단계 수는 D5) |
| K9 | BPS, 운동학 특징 형태 | 공유 BPS, T_rel ∈ R^{L×L×N×6}(관절 연결 쌍만, 나머지 0, 식 2) | `get_edge_embedding.py`: 시드 42의 기저점 128개(반지름 최대 0.15 m), 링크 표면 256점의 최소 거리. 엣지 텐서 `(L, L, 128, 6)`을 0으로 초기화하고 연결 쌍만 채움 | 형태 일치(값 생성 방식은 D8) |
| K10 | 8손 구성 | Allegro, Ability, Barrett, DClaw, LEAP, Schunk, Shadow, XHand | `train.yaml` `dataset.sources[0].hand_type` 8개, 링크 수 `robot_links`(21, 12, 9, 10, 17, 22, 22, 18) | 일치 |

### 2.2 차이와 미확인 (D 항목)

| 번호 | 항목 | 논문 | 코드 | 판정 | 영향 |
|---|---|---|---|---|---|
| D1 | **성공 판정 구조** | §3.4: "환경과 충돌 없이 표적을 정해진 높이 이상 들어 올림"을 Isaac Gym에서 | `isaac_validator_lift.py`의 시뮬은 **표적 물체와 테이블 상자(0.8 × 0.8 × 0.04 m)만** 만듭니다. 장애물·선반·상자는 없습니다. 환경 충돌은 별도 스크립트 `collision_check.py`가 **환경 점군 근접 규칙**으로 판정합니다: 손 표면 점 중 환경 점에서 3 mm 이내가 48개 이상이고 0.6% 이상이면 충돌, 지면 −3 cm 아래 점 8개 이상이면 충돌. 최종 성공 = (들어 올림) AND (충돌 아님) | **차이 (논문에 없음)** | 장애물과의 접촉은 물리가 아니라 점군 근접으로만 걸러집니다. 접촉과 관통을 구분하지 못하고 점 밀도에 민감합니다(`03` T2). 검사 대상은 모델이 본 환경 점 2,048개뿐(환경 점군 전체가 아님) |
| D2 | **성공 판정 수치** | "정해진 높이 임계값", "outer 자세 → inner 자세 위치 제어" | 들어 올림 ≥ 0.03 m **그리고** 수평 이동 ≤ 0.05 m, 손 들어올림 0.20 m, 마찰 3.0(손·물체·테이블 모두), 물체 밀도 500, PD 강성 2000·감쇠 80, 접근 단계 100 스텝은 **중력 끔** 뒤 켬(`isaac_validator_lift.py` L29–46, 169, 294, 515–543). outer = 열림 한계 쪽 25%, inner = 닫힘 한계 쪽 15% 추가(`controller.py` L233–237) | **차이 (논문에 없음)** | 마찰 3.0은 흔한 값(0.5–1.0)보다 매우 큽니다. 3 cm 들어올림은 "정해진 높이"를 값으로 정한 것. 다른 코드베이스와 SR을 직접 비교할 수 없음 |
| D3 | **SR 집계 단위와 분모** | 손당 20,000 장면(§5.2) | `val_lift.py`/`collision_check.py`: 손 SR = **물체별 SR의 단순 평균**(매크로, `sum(sr_dict.values())/len(sr_dict)`). 예측이 유한하지 않은 장면은 건너뛰고(`test.py` skip), 메시가 10만 정점을 넘거나 단순화가 안 되는 물체는 통째로 제외(`convert_before_eval.py` `[SKIP]`), 모두 분모에서 빠짐 | **차이 (논문에 없음)** | 물체당 장면 수가 다르면 장면 단위 SR(표 2)과 다름. 건너뛴 비율은 코드에 기록되지 않음. 제외가 어려운 물체에 몰렸다면 SR이 높게 나옴 |
| D4 | **다양성 정의** | 표 2 "Diversity (rad.)", 정의 없음 | `collision_check.py` L204–208: 성공한 파지의 q(손목 위치 m, 손목 회전, 관절각 rad)를 **모든 물체에 걸쳐 합친 뒤** 성분별 표준편차를 구하고 평균(`torch.std(all_success_q, dim=0).mean()`) | **차이 (논문에 정의 없음)** | 손목 위치(m)와 각도(rad)를 섞은 평균, 물체·장면 간 변동이 섞인 값(한 장면의 다양성이 아님). 손 자유도마다 값의 규모가 다름(Ability 0.509 대 Shadow 0.354). 장면당 샘플이 하나이므로 한 장면에서의 다양성은 측정되지 않음 |
| D5 | **DDIM 단계와 잡음** | 20단계, η = 1.0(§9) | `inference_step: 20`이면 `step = 1000 // 20 = 50`, `start_t = 949`, `arange(949, -1, -50)` = **19단계**(949, …, 49)이고 마지막 단계에서 `a_bar_prev = 1`로 x0를 냄. 확률 항에 `lambda: 0.2`를 곱함(`troe_graph.py` L467–469, 513–514) | **차이** | 실제 단계 19, 확률 항 20%로 축소(η = 1이지만 유효 잡음은 0.2). 논문 표기와 같은 설정이 아님. 다양성에 영향 가능 |
| D6 | **옵티마이저와 학습 길이** | AdamW, 워밍업 1,000, "스텝 수는 학습 데이터 전체를 덮도록"(§9) | `torch.optim.Adam`(가중치 감쇠 없음, `train.py` L142), 워밍업 뒤 **코사인으로 0까지 감쇠**(`build_step_scheduler`), `total_steps: 1000000`, 에폭 없는 무한 반복 샘플러(`__iter__`의 `while True`) | **차이** | 1M 스텝 × 8 GPU × 48 = 3.84억 샘플 = 444M의 0.87배(`03` V14). 균일 복원 추출이면 샘플의 약 58%만 봄 |
| D7 | **손실 건너뛰기** | 서술 없음 | `train.yaml` `max_loss: 3.0`: 한 스텝의 총 손실이 3.0을 넘거나 유한하지 않으면 그 배치는 기울기 0으로 건너뜀(`train.py` L206 이후 `local_valid`/`should_skip` 블록) | **차이 (논문에 없음)** | 학습 안정화용 필터. 높은 손실 샘플(어려운 샘플)이 체계적으로 걸러질 수 있음 |
| D8 | **위치·운동학 특징 생성** | 위치: N개 무작위 관절 구성에서 FK로 T_base ∈ R^{N×L×6}. 운동학: 샘플 구성들에서 관절 연결 링크쌍의 상대 변환(식 2) | `get_edge_embedding.py`: 위치 특징 `pos.pt`는 **q = 0 한 구성**의 링크 자세 (L × 6)(`get_pos_embeddings`, 수동 `breakpoint()` 시각 확인이 들어 있음). 운동학 특징은 **관절별 독립** `np.linspace(lower, upper, 128)` 스윕의 상대 변환이고 부모/자식 방향을 같은 값으로 채움(무방향) | **차이** | 논문 설명(N개 무작위 구성)과 다른 특징입니다. 위치 특징은 손마다 상수 하나라 "링크 지문"의 위치 항은 정적 식별자에 가깝습니다. 절제(표 6)가 어느 쪽을 껐는지 알 수 없음 |
| D9 | **어텐션 세부** | "edge-conditioned attention [35]"(식 7) | 쿼리 = 로봇 노드, 키 = 물체/환경/로봇 노드 특징만. 간선 특징은 **값(`EdgeValueNet`)에만** 들어가고 어텐션 로짓에는 들어가지 않음(`tro_denoiser.py` `TROEGraphLayer`). 패딩 링크 노드(`max_link_node`까지)에 **키 마스크가 없음**(마스크는 손실 평균에만 사용). 링크 최대 수가 학습 설정에서 26(Wuji 포함), 검증 설정에서 25(논문은 L ≤ 25) | **부분 차이** | 간선이 값 경로에만 있다는 점은 논문 서술에서 읽기 어려움. 링크 수가 적은 손(Barrett 9개)에서 패딩 노드가 RR 어텐션에 참여 |
| D10 | **IK 목표** | "예측 링크 SE(3)를 IK로 관절 값으로"(§4) | `PyrokiRetarget`은 **링크 원점 위치만**(보조 링크 포함) 맞춤. 손목 자세는 예측 루트 자세를 초기값으로 복사. 64회 반복, 조기 종료 없음. IK 오차에 임계값이 없어 오차가 큰 해도 시뮬로 감 | **차이** | 모델이 낸 링크 회전은 IK에서 쓰이지 않고 간선에만 영향. 위치 정합 오차가 성공률에 어떻게 반영되는지 보고되지 않음 |
| D11 | **회전 표현** | 6D 자세, 가우시안 잡음(식 6) | 링크 자세 = [위치 3, **rotvec 3**]. 회전에도 R³ 가우시안 잡음을 직접 더하고 노이즈를 예측(`troe_graph.py` `forward`). 상대 회전은 `so3_log_map`(`rotation.py`) | **서술 부족** | rotvec은 각도 π에서 불연속입니다(`03` T1: 무작위 회전의 6%가 π − 0.1 안, 초기 잡음의 2%는 노름 > π). SO(3) 확산이 아님 |
| D12 | **마운트와 충돌 투영** | 손바닥에 원통 마운트를 붙여 충돌 검사에 포함(§3.2) | `is_mount: False`가 기본(`val.yaml`, `val_wuji_hand.yaml`). 학습은 항상 마운트 없이(`troe_graph.py` L130). `project_pose_with_env_pc`(Adam 5회 투영, `lambda_coll = 25`)가 정의돼 있으나 **어디서도 호출되지 않음**. `is_mount: True`이면 `inference_config["hand_pc_path"]`를 읽는데 config에 키가 없음. `tro_hand_model.py`에서 읽은 마운트 분기는 XHand뿐 | **차이 / 미사용 코드** | 표 2의 평가가 마운트 포함인지 알 수 없음. 공개 설정으로는 팔 쪽 여유를 검사하지 않음 |
| D13 | **Wuji 체크포인트** | Wuji는 단일 시점 데이터 ≈ 600만 개로 미세조정(§5.6) | README는 `nine_hand_with_wuji.pth`를 "Wuji 손으로 학습한 체크포인트"로 소개하고, 이름이 8손 + Wuji 9손 모델을 가리킴(미세조정 여부는 README가 밝히지 않음). `train.yaml`의 `dataset.hand_type`에는 9손이 있고 `val_wuji_hand.yaml`은 링크 26개 설정. 단일 시점 학습·변환 코드 없음 | **차이 가능 (미확인)** | 공개 Wuji 결과가 논문의 단일 시점 미세조정과 같은 것인지 확인할 수 없음 |
| D14 | **공개되지 않은 코드** | 데이터 생성, 난이도, 베이스라인, 절제, 규모, 다중 시도, 제로샷, 지연 측정, 다양성 분석, 실로봇 | 저장소에 **없음**: PPO 주석 정책과 보상(식 9, 10), 장면 합성기, 난이도 점수(식 11–13, `grep`에서 `difficult`·`enclos` 0건), AnyDexGrasp·DexGraspNet2 평가·변환 코드(`vis_grasp.py`는 `/mnt/fx/AnyDexGrasp_exp` 등 외부 결과를 읽기만 함), 링크 지문 절제 스위치(`RobotGraph`에 off 플래그 없음), 객체·모델 규모 변형, 다중 시도 평가, 변형 손 URDF, 효율 측정, 표 9 다양성, VQ-VAE 학습 스크립트(`model/vqvae/`에 손실만) | **미확인** | 표 3–6, 9, 11 중 코드로 재현할 수 있는 것은 거의 없음 |
| D15 | **공개 데이터 범위** | 학습 객체 47,737, 파지 약 4.44억(§3.1, 표 1) | HF 카드: Zarr 객체 25,228개(6.19 TiB). 원천 31,166개에서 시험 5,938개를 뺀 것이라고 적음. 검증 장면 `.pt` 8,200개(손별 폴더) | **차이** | 공개 학습 데이터는 논문 학습 객체의 52.8%. 8,200개 파일이 표 2의 손당 20,000 장면을 몇 장면으로 담는지는 파일 내부 `num_scene`이라 보지 못했음 |
| D16 | **체크포인트 로딩** | — | `test.py` L324: `model.load_state_dict(ckpt["model_state"], strict=False)` | **주의** | 누락·불필요 키가 조용히 무시됨. 체크포인트와 설정이 어긋나도 오류 없이 무작위 초기값 층이 남을 수 있음 |
| D17 | **실험 이름이 암시하는 변형** | — | `convert_before_eval.py`·`collision_check.py`의 기본 경로 `train_0.2B_multi_hand_v6_noise`, 데이터 폴더 `TRO-1B-Dataset`. 학습 `train.yaml`의 이름은 `train_0.2B_eight_hands` | **미확인** | 평가에 쓴 모델이 `v6_noise`(잡음 증강?) 변형일 수 있으나 공개 데이터 로더에 점군 잡음 증강 코드는 없음(읽은 범위) |

### 2.3 논문 주장의 코드 쪽 확인

- **"링크 지문이 필수" (표 6):** 코드에는 위치 특징(D8: 손별 상수)과 운동학 특징(관절 스윕)이 항상 들어가며 절제 스위치가 없습니다. 효과는 코드로 검증할 수 없습니다.
- **"추론 샘플 효율"(PPS):** 모델 쪽은 표적 전용 제안을 만들므로 PPS가 낮게 나오는 것이 구조적이지만, PPS를 재는 코드는 없습니다.
- **"실시간 IK" (§4):** Pyroki 다중 시작 IK가 맞지만, 64회 반복 8시작 LM 해의 지연이 표 4의 0.336 s에 들어 있는지는 확인할 수 없습니다(`test.py`는 시간을 재지 않음).

## 3. 데이터 경로 요약

학습 데이터(Zarr)는 객체당 하나의 디렉터리이고 샘플은 `data/pc`(환경 점군), `data/joint`, `meta/meta_json`(`target` 7차원 물체 자세)입니다(`TROMixedHandZarrDataset._load_one_sample`). 파지마다 환경 점군이 들어 있습니다. 객체당 평균 약 270 MB(6.19 TiB / 25,228)입니다.

## 4. 구현 가능성 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**전체 재현은 불가능합니다. 표 재계산과 설계 점검은 이미 했고, 소형 재구현은 가능합니다.**

| 무엇 | 이 맥에서 | 필요한 것 |
|---|---|---|
| 표 2–5(시뮬 성공률) 재현 | **불가** | Isaac Gym Preview 4(NVIDIA GPU, Linux), 평가 코드 일부(D1–D4, D14), 베이스라인 구현 |
| 0.2B 모델 학습(1M 스텝) | **불가** | 8 × A100 80GB급, 6.19 TiB 데이터 |
| 데이터 생성(RL 주석, 4.44억 파지) | **불가** | 코드 미공개(D14), GPU 병렬 시뮬 |
| 실로봇(표 7, 8) | **불가** | xArm7, XHand, LEAP, RealSense 3대 |
| 공개 체크포인트로 추론 + IK 통계(`test.py`) | **조건부** | 새 설치(torch 2.4.1, pytorch3d, torch-cluster, chamferdist, theseus, JAX, pyroki), 공식 코드 실행 승인, 체크포인트(Google Drive, 크기 미확인)와 검증 장면(6.9 GB zst) 다운로드 승인. 맥(MPS/CPU)에서의 설치 성공과 속도는 **측정 전** |
| 충돌·들어올림 평가 재현 | **불가**(들어올림) / **조건부**(충돌 규칙) | 들어올림은 Isaac Gym. 충돌 규칙은 CPU로 되지만 추론 결과와 손 모델·URDF가 필요(추론을 먼저 해야 함) |
| 구조의 소형 재구현(torch) | **가능** | 합성 다중 손 + 장면 데이터로 10M급 모델을 MPS/CPU에서 학습. 자체 구현, 논문 수치 재현은 아님 |
| 표 재계산, 일관성 검사, 코드 정적 대조 | **완료** | `analysis/recompute.py`, 이 문서 |

- **노력 추정(측정 전, 추정):** 소형 재구현 2–3주(링크 지문·그래프 디노이저·합성 손 생성기), 체크포인트 추론 확인 2–5일(설치 호환이 변수), 시뮬 평가는 맥에서 대체 구현(MuJoCo)에도 장애물·마찰·컨트롤러 값(D2) 이식이 필요해 2–3주이며 Isaac 결과와의 정합은 보장되지 않습니다.
- **필요하면 맥이 아닌 곳에서 해야 하는 것:** 시뮬 성공률과 학습은 NVIDIA GPU 서버가 필요합니다.
- 맥 저장소 규칙상 GPU 재현은 하지 않고, 설치·실행·다운로드는 승인 뒤에 합니다.

## 5. 재실행

```bash
python3 analysis/recompute.py
python3 analysis/toy_checks.py
python3 analysis/plot_figures.py
```

코드 대조는 정적 읽기이므로 재실행 대상이 아닙니다. 대조에 쓴 커밋은 `449db5a`입니다(`git clone --depth 1`로 같은 상태를 받을 수 있습니다).
