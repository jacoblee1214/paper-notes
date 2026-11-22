# 04. 공식 코드 대조와 구현 가능성

대상은 `GaTech-RL2/EgoWAM` 커밋 `c87617fe37a6ed6a951e6b176ad552200c425c93`(2026-09-07 14:48 -0400, 저장소의 마지막 커밋)을 얕게 복제한 것입니다(`repo/`, 137 MB, gitignore). **정적으로 읽기만 했고 실행·import·설치를 하지 않았습니다.** 복제된 내용은 신뢰할 수 없는 데이터로 다뤘습니다. 번호 D는 논문과 다른 점, K는 일치 확인입니다.

읽은 범위: 하이드라 설정(`egowam/hydra_configs/` 전부 중 BC, Pixel, Pixel-PT, DINO, 3D Flow, 컵 데이터 설정과 학습 설정), `algo/hpt_wm.py`(손실 구성, 미래 목표 부착), `models/fm_policy.py`, `dit_image_decoder.py`, `point_flow_decoder.py`, `denoising_policy.py`, `rae_vae.py`, `cosmos_vae.py`, `wan_dit.py`(헤드 구성부), `rldb/zarr/point_flow_loader.py`와 `zarr_dataset_multi.py`(분할, 모드), 3D 흐름 추출 스크립트(`scripts/3dflow_process/`), `robot/rollout.py`, `docs/`, `external/README.md`, `external/RAE`의 인코더. 읽지 않은 것: `external/` 안쪽 전체(Track4World 모델, cosmos-policy 본체, RoboTwin), 데이터 변환(`eva_process`, `aria_process`), 시험 파일, 노트북.

## 1. 라이선스, 자산, 접근

- **라이선스**: 저장소 `LICENSE`는 MIT이고 저작권 표기는 "Copyright (c) 2025 EgoMimic-Team"입니다. `external/`에는 업스트림 라이선스가 각각 있습니다(RAE MIT, cosmos-policy Apache-2.0, RoboTwin MIT, Track4World는 Tencent 고지·NOASSERTION). 이 저장소(research_roadmap)에는 코드를 가져오지 않았습니다.
- **필요 자산**(`docs/install.md`): Pixel은 게이트된 Hugging Face 저장소 `nvidia/Cosmos-Predict2-2B-Video2World`의 토크나이저, Pixel-PT는 Wan2.1 T2V-1.3B·VACE-1.3B 가중치(수 GB), DINO는 RAE 1단계 디코더(저자 체크포인트 `boeyyyy/EgoWAM-checkpoints`)와 업스트림 RAE 자산(1.6 GB + 3.2 GB 등)과 DINOv2 가중치. BC와 3D Flow는 외부 자산이 필요 없습니다(데이터는 필요).
- **데이터**: 학습 데이터는 EgoVerse SQL 에피소드 표와 객체 저장소에서 첫 학습 때 동기화되며(`docs/data.md`) 접근에 계정이 필요합니다. 데이터 탐색기와 체크포인트 링크는 있으나 로봇 시연(EVA) 데이터의 공개 범위는 확인하지 못했습니다.
- **보안 관찰(D16)**: 저장소 문서 일부에 자격 정보로 보이는 문자열이 평문으로 들어 있습니다. 위치와 값은 공개본에 적지 않았습니다. 공개 읽기 전용 자격일 가능성이 크지만 확인하지 못했고, 저자에게 비공개로 알릴 사항입니다.
- **재현 지원**: 열린 이슈 4개(#3 그림 4 UMAP 설정 문의, #4 읽기 목록 초대, #5·#6 누락 코드 제안). 저자가 닫은 이슈 2개(#1 DINO와 3D 흐름 동시 사용은 하지 않았음, #2 자원 문의에 "L40S 한 장으로 약 3일, V100 8장도 가능").

## 2. 일치 확인 (K)

| 번호 | 항목 | 논문 | 코드 (위치) |
|---|---|---|---|
| K1 | 트렁크 | 256차원·16블록·8헤드, 드롭패스 0.1, 행동 토큰 64 + 미래 토큰 16, 관측 지평 1 | `hpt_wm_cotrain_3dflow.yaml` `trunk:` 등 모든 모델 설정 동일 |
| K2 | 행동 헤드 | 6블록·128차원·4헤드 CrossTransformer, 100 스텝, 50 샘플링 스텝, v-prediction, τ ~ Beta(1.5, 1.0) | `head_specs.shared`, `models/fm_policy.py:78–89` |
| K3 | 14차원 행동, 분위수 정규화 | §3.1 | `data_schematic.norm_mode: quantile` (`train_zarr_wm.yaml`), 헤드 `act_dim: 14` |
| K4 | 창 길이 | 로봇 45프레임(1.5 s), 사람 30프레임(1.0 s), 청크 100으로 재표본 | 데이터 설정 `action_horizon: 45`·`30`, `chunk_length: 100` |
| K5 | 옵티마이저 | AdamW lr 1e-4, wd 1e-4, 코사인 T_max 1400, η_min 1e-5 | 모델 설정 `optimizer`, `scheduler` |
| K6 | 배치와 길이 | 32/32, 100 스텝 × 2000 에폭, bf16 | 데이터 설정 `batch_size: 32`, `trainer/default.yaml` `limit_train_batches: 100`, `max_epochs: 2000`, `precision: bf16` |
| K7 | 손실 가중 | λ = 1 | `auxiliary_loss_weights` 값 1.0, `total_loss = action + shared + aux` (`hpt_wm.py` 손실 구성) |
| K8 | 세계 손실이 공유 트렁크로 역전파 | §4.3 | 미래 토큰 특징을 detach 없이 헤드에 전달(`hpt_wm.py:_compute_joint_state_loss`) |
| K9 | Pixel 헤드 | 16×16×16 잠재, 128 px, 6블록·384차원·6헤드 DiT | `hpt_wm_cotrain.yaml` `shared_future_image_latent`, `future_image_resize: 128` |
| K10 | DINO 목표 | DINOv2-B 패치 특징 16×16×768, [CLS]·레지스터 제외, 토큰별 LayerNorm | `rae_vae.py` `encode()`는 RAE 1단계 인코더 특징을 반환, RAE의 DINOv2 인코더는 LayerNorm affine을 제거(`external/RAE/src/stage1/encoders/dinov2.py:22–25`) |
| K11 | 3D 흐름 지평·격자·임계값 | 로봇 45, 사람 30 프레임, 28×40 = 1120 앵커, 임계값 2/10 mm, 사람 앞뒤 20프레임 제외 | `compute_3d_flow_*.py --rate 16`(448×640/16), `point_flow_movement_threshold_3d: 0.002/0.010`, `point_flow_head/tail_skip_frames: 0/20` |
| K12 | 3D 흐름 헤드 | 4블록·256차원·4헤드, 자기+교차 어텐션, 앵커 위치 조건 | `models/point_flow_decoder.py`, `CrossBlock`(자기 어텐션 + 교차 어텐션 + MLP) |
| K13 | 시각 t 카메라 프레임 안정화 | X̃ = (T_t)⁻¹ T_{t+T} X_{t+T}, 로봇은 고정 카메라 | `compute_3d_flow_compensated.py:262–263` `T_rel = w2c[tskip] @ c2w[t]`, `T_inv`를 점에 적용. 로봇은 별도 `compute_3d_flow_fixed_camera.py` |
| K14 | 증강 | 색 지터(0.1, 0.1, 0.1, 0.05), ImageNet 정규화 | `train_image_augs` |
| K15 | 헤드 파라미터 수 | Pixel 17.3M, DINO 19.6M, 3D Flow 9.0M | 설정 구조식으로 계산: 17,307,840 / 19,620,992(평범한 DiT) / 8,962,092 (`03` V12) |
| K16 | 추론은 세계 헤드 없이 | §4.3 | 배포 스크립트는 정답 미래가 없어 보조 헤드가 돌지 않음(`rollout.py:234–243`, `--action-only-inference`) |
| K17 | 제어 주기 | 30 Hz | `docs/hardware.md` |

## 3. 차이 (D)

심각도: **높음**(결과 해석에 영향), **중간**(재현에 영향), **낮음**(서술 차이).

| 번호 | 심각도 | 항목 | 논문 | 코드 (위치) |
|---|---|---|---|---|
| D1 | 높음 | 학습에 쓰는 에피소드 | 로봇 300–360 시연, EgoVerse-A 전체 분할(표 3, C.2) | `MultiDataset(mode="train", valid_ratio=0.2)`이 에피소드(데이터셋 키)의 **20%를 검증용으로 떼어 둠**. 공개 데이터 설정은 모두 `mode: train`/`valid`(`zarr_dataset_multi.py:364–392, 961–997`). 공개 설정 그대로면 학습은 로봇 시연의 약 80%만 씀. 논문이 실제로 이 분할을 썼는지는 확인 불가 |
| D2 | 높음 | 세계 손실의 형태와 시간 방향 | 식은 노이즈 예측 ‖ε − ε_ψ‖²(§4.2), B.2는 v-prediction. s^τ = (1 − τ)ε + τ s에서 τ = 1이 깨끗한 데이터, τ ~ Beta(1.5, 1.0) | 모든 헤드가 v-prediction, 목표 u = noise − data이고 보간은 x_t = t·noise + (1 − t)·data로 **t = 1이 잡음**. 같은 Beta(1.5, 1.0)이 잡음 쪽에 가중(평균 잡음 비율 0.6 대 논문 서술 0.4, `03` V14, `fm_policy.py:78–89, 204–216`). DiT 헤드는 출력에 입력 x_t를 더함(`dit_image_decoder.py:334`, 논문에 없음) |
| D3 | 높음 | DINO 헤드 구조 | RAE의 DiT^DH: 6블록·384차원 DiT + **2블록·2048차원 넓은 DDT 헤드**(표 2, B.2) | 공개 설정(`hpt_wm_cotrain_rae.yaml`)의 헤드는 `DiTImageDecoder`(6블록·384차원, 입력 768채널, 패치 2)로 **넓은 헤드가 없음**. 파라미터 수가 논문 A.4의 19.6M과 맞고 넓은 헤드가 있다면 1.3억 이상(`03` V12). 저장소는 RAE 2단계 DiT-XL 모델도 만들어 불러오지만(`use_stage2_features: true`, 3.2 GB) `encode()`의 2단계 경로는 주석 처리라 쓰지 않음(`rae_vae.py:504`). `external/README.md`는 "DiT-DH 디노이저를 헤드로 쓴다"고 적음 |
| D4 | 중간 | 3D 흐름 목표의 시간 구조 | "[t, t+T] 위의 밀집 3D 운동장", 목표 100 × 1120 × 3(B.2) | 목표는 **시작점 P_t에서 끝점 P_{t+k}로 가는 선형 보간**(`point_flow_loader.py:393–401, 463–468`). 시간축에 정보가 없고 점마다 변위 벡터 하나(1120 × 3)가 실제 목표. 목표는 `NormalizeFlowDelta`로 정규화된 변위이고 손실은 유효 점 수로 정규화(`fm_policy.py:230–266`). 설정의 `num_points: 512`·`[100, 512, 3]`은 낡은 값이고 실제 점 수는 데이터에서 읽음(1120, `fm_policy.py:146`) |
| D5 | 낮음 | 3D 흐름 경계 처리 | 앞뒤 20프레임 제외(사람) | 끝점 프레임은 `tskip = min(t + frame_skip, T − 1)`로 클램프(`compute_3d_flow_compensated.py:222`)라 마지막 구간의 지평이 짧아질 수 있음. 제외 구간(20)이 지평(30)보다 짧아 일부가 남음 |
| D6 | 중간 | 세계 헤드 조건 방식 (통제 변수) | "교체 가능한 세계 모델 헤드가 트렁크 임베딩으로 조건화"(§4.1). B.2는 Pixel·DINO만 "평균 풀링한 트렁크 임베딩" 명시 | Pixel·DINO 헤드는 미래 토큰 16개를 **평균 풀링**(`pooling: "mean"`, `hpt_wm_cotrain*.yaml`), 3D Flow 헤드는 16개 토큰 전체를 교차 어텐션으로 받고 **현재 프레임 점 위치(앵커)도 입력**(`point_flow_decoder.py`), Pixel-PT 헤드도 전체 토큰(`wan_dit.py:1747`). 헤드 간 정보 경로가 다름 |
| D7 | 중간 | 사람 행동의 차원 | 사람과 로봇 모두 14차원(그리퍼 포함) (§3.1) | 사람 행동은 관측 손 포즈에서 만든 12차원(그리퍼 없음, `human.py` 변환 주석)이고 14차원 헤드의 그리퍼 자리는 **0으로 패딩해서 감독**(`denoising_policy.py:109–120`, `padding: "zero"`). 로봇의 실제 그리퍼와 한 헤드에서 충돌하는 목표가 되고 도메인 임베딩으로만 구분됨 |
| D8 | 중간 | 표 2에 없는 학습 설정 | ResNet-18, 색 지터 | ResNet-18은 **ImageNet 사전학습 가중치**(`weights="DEFAULT"`, `hpt_nets.py:518–526`), 이미지 입력 480 × 640(`hpt_wm.py:1497`), 스템 모달리티 드롭아웃 0.1(`crossattn_modality_dropout`), 행동 헤드 드롭아웃 0.1, 트렁크는 처음부터(`pretrained: false`) |
| D9 | 중간 | 학습률 일정과 체크포인트 | T_max = 1400, 2000 에폭, 평가 체크포인트 미기재 | 에폭 단위 스케줄러(Lightning 기본)라 에폭 1400 이후 학습률이 다시 상승(`03` V13). 체크포인트는 100 에폭마다 저장(`callbacks/checkpoints.yaml`)하고 문서 예시는 에폭 1399·1799(`docs/training.md`). 어느 에폭을 배포했는지 논문에 없음 |
| D10 | 중간 | 배포 질의 주기 | "배포 추론 30 Hz, BC와 같은 지연"(표 2, B.3) | 제어 루프 30 Hz, 정책 질의 기본값 `QUERY_FREQUENCY = 30` 스텝, 100스텝 청크를 `DEFAULT_RESAMPLE_LENGTH = 45`로 줄여 **앞 30스텝을 열린 루프로 실행**(`rollout.py:154–155, 390–405`). 질의는 약 1초에 한 번. 평가에서 인자를 바꿨는지는 확인 불가 |
| D11 | 높음 | 3D 흐름 추출 코드 | Track4World + Aria VIO 포즈(§4.2, B.2) | 두 추출 스크립트가 불러오는 `track4world.pipeline.efep_export`가 **릴리스에 없음**(`compute_3d_flow_fixed_camera.py:117`, `compute_3d_flow_compensated.py:130`, 저장소 이슈 #5). 추출은 프레임 쌍(t, t + frame_skip) 단위 추론이고 카메라 포즈는 머리 포즈를 그대로 카메라 포즈로 사용(`aria_pose.py:22–31` `build_head_only_c2w`). 장치→카메라 외부 파라미터를 이 함수 안에서 곱하지 않음(다른 곳에서 반영되는지는 확인 못함) |
| D12 | 중간 | DINO 변형의 시각 스템 | ResNet-18 또는 DINO 인코더 (§4.1) | `docs/training.md`: "모든 대표 결과는 ResNet-18. DINOv2 스템은 장면 일반화가 크게 좋아지는 별도 절제이며 배포 설정이 아님". 그런데 같은 문서의 공개 체크포인트 예시는 `fold_clothes_dinov2_c_widetrunk_1799.ckpt`. DINO 대표 결과(옷 접기 OOD 17/20)가 어느 스템인지 논문에서 구분되지 않음 |
| D13 | 중간 | Pixel-PT 초기화 | "사전학습 VACE-1.3B 트랜스포머" (§4.2, 표 2), A.3은 "Wan", A.4는 "Wan2.1-1.3B" | 설정은 **Wan2.1 T2V-1.3B의 본체 블록만** 불러옴(`pretrained_path: hf://Wan-AI/Wan2.1-T2V-1.3B`, `load_main_blocks_only: true`, `hpt_wm_cotrain_pixelpt.yaml:230–238`). VACE 컨텍스트 블록 아님. 그래디언트 체크포인팅 필수, 2 GPU·GPU당 배치 16 |
| D14 | 중간 | BC 기준선의 코드 경로 | 같은 트렁크·행동 헤드·데이터 혼합(§4.1) | BC는 별도 클래스(`algo/hpt.HPT`), 토큰 후처리가 `action_token`(WAM은 `joint_token`), 로봇만 BC는 도메인별 헤드(`hpt_bc_eva.yaml`). 구조 값은 같지만 구현 경로가 다르고, 문서는 BC 공동 학습에서 "일부 파라미터가 손실에 들어가지 않아" `find_unused_parameters`가 필요하다고 적음 |
| D15 | 낮음 | 학습 시간 | 방법당 약 이틀(B.3) | 저자 답변(이슈 #2)은 "약 3일" |
| D16 | 보안 | 문서의 평문 자격 정보 | – | 위치 비공개(§1) |

종합: 구조와 대부분의 하이퍼파라미터는 논문과 맞습니다(K1–K17). **가장 중요한 차이는 D3(DINO 헤드)와 D6(헤드 조건)로, 세 목표 비교의 "통제 변수"가 논문 설명보다 약합니다.** D1·D9·D12는 보고 수치가 어떤 설정에서 나왔는지에 영향을 줍니다.

## 4. 구현 가능성 판정 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**논문 수치의 재현은 불가능합니다. 정적 대조와 표 재계산, 소형 기하·구조 구현은 가능하고 일부는 끝냈습니다.**

| 수준 | 판정 | 근거 |
|---|---|---|
| 실로봇 평가 (그림 3, 8, 9, 표 3) | **불가** | ARX5 양팔, Aria 안경, RealSense, 시연 수집 장비, EgoVerse 접근 계정 |
| 공식 학습 (과제·방법당 L40S 1–2장 2–3일) | **불가** | CUDA·DDP 전제(`trainer/ddp.yaml accelerator: gpu`), 게이트된 자산(Pixel), 데이터 동기화에 계정 필요, Pixel-PT는 21 GiB 이상 가중치·옵티마이저(`docs/training.md`) |
| 공식 코드로 1% 슬라이스 디버그 학습 | **조건부, 어려움** | `trainer=debug`도 `ddp`를 상속해 GPU 가속기를 지정하므로(`trainer/ddp.yaml`) CPU·MPS 실행에는 오버라이드가 필요하고 시험하지 않음. 코드는 CUDA 유무로 장치를 고름(`hpt_wm.py:1328`). 새 가상환경, `requirements.txt` 설치, 데이터 일부 다운로드, 공식 코드 실행 승인이 필요 |
| 3D 흐름 추출 파이프라인 | **불가** | 모듈 누락(D11), Track4World 추론은 GPU |
| 표 재계산과 검정 | **가능, 완료** | `analysis/recompute.py` (수 초) |
| 3D 흐름 안정화·임계값 토이 | **가능, 완료** | `analysis/toy_egomotion.py` (약 1초, numpy) |
| 3D 흐름 헤드(약 9M) + 합성 점 변위 학습 토이 | **가능 (미실시)** | torch 2.15, MPS 또는 CPU, 합성 데이터. 노력 3–7일(추정, 측정 전) |
| "행동 정렬이 깨진 두 구현체" 장난감 공동 학습: BC 대 WAM 보조 헤드 | **가능 (미실시)** | 평면 2D 도달 과제에서 구현체별 행동 오프셋을 주고 BC 공동 학습과 보조 미래 헤드를 비교. 노력 1–2주(추정, 측정 전) |
| Pixel-PT(1.4B 헤드, Wan) | **불가** | 메모리와 가중치 크기 |
| 논문 수준 재구현(HPT 트렁크 + 세 헤드 + 데이터 파이프라인) | **사실상 불가** | 데이터와 로봇이 핵심 병목 |

노력 추정(모두 추정, 측정 전): 합성 3D 흐름 헤드 3–7일, 장난감 공동 학습 비교 1–2주, 공식 코드를 이 맥에서 디버그 슬라이스까지 돌리는 데 새 환경 구성과 MPS 대응을 포함해 1–3주. 실로봇은 이 저장소의 범위 밖입니다. 어느 쪽이든 CPU 가능 판정은 이 저장소 루트 가상환경(torch 2.15, mujoco 4.0)을 전제하며 그 밖의 설치는 사용자 확인이 필요합니다.

## 5. 확인하지 못한 것

- 논문의 보고 수치를 만든 정확한 설정(D1, D9, D12). 저자 체크포인트의 `.hydra/config.yaml`을 받아 보면 가려지지만 받지 않았습니다(크기 미확인).
- Track4World 쪽 변경 5개 파일의 내용, `external/cosmos-policy`의 Wan VAE 인터페이스, EgoVerse 데이터 변환 코드.
- 로봇만 WAM 모델과 BC 모델의 데이터 설정이 같은지(같은 `eva_cup_saucer` 폴더를 가리키는 것까지만 확인).
- RoboTwin 브리지의 ACT-EE·DP-EE 구현(`external/README.md`의 서술만 읽음).
