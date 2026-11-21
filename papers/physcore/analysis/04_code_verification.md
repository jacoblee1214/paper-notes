# 04. 공식 코드 대조: PhysCoRe

공식 저장소 `lunarlab-gatech/PhysCoRe`를 얕은 복제 (`repo/`, 커밋 `ba7ddf7`, 2026-09-13 "initial open-source release")하여 **정적으로만** 읽었습니다. 실행, 설치, import, 데이터 다운로드는 하지 않았습니다. 대조 기준은 논문 v3 (`01_paper_analysis.md`)입니다.

확인 방법: `configs/*.yaml`과 `*.py`를 읽고, 논문 식과 표의 값을 맞췄습니다. 추적한 경로는 `convert_to_episode.py` → `dataset.py` → `episode_runtime.py` → `validate_RfD.py`입니다. 실행으로 확인한 것은 없습니다. 아래 "확인됨"은 코드에 그 값이 있다는 뜻입니다. 실제 실행 결과가 같다는 뜻이 아닙니다.

## 1. 일치하는 항목

| 항목 | 논문 | 코드 (파일) | 판정 |
|---|---|---|---|
| MfM 사전 학습 | 2000 에포크, 배치 4, 학습률 1e-4, 감쇠 0.01, BPTT 4 창, K = 10 | `configs/train_MfM.yaml` (`epochs 2000`, `batch_size 4`, `lr 1e-4`, `weight_decay 0.01`, `update_every 10`, `tbptt_chunks 4`) | 확인됨 |
| 손실 가중치 | λ = 0.2, β = 0.1 | `loss.material_confidence_log_weight 0.2`, `plasticity_weight 0.1` | 확인됨 |
| 입력 밴드 | 정규 좌표 4, 추적 변위 16 | `canonical_bands 4`, `tracked_bands 16` | 확인됨 |
| 그래프 U-Net | 은닉 128, 이웃 k = 8, 증거 가중치 8·4, r_c = 0.04 | `hidden 128`, `graph_unet_k 8`, `evidence_tracked_weight 8.0`, `evidence_control_weight 4.0`, `evidence_control_radius 0.04` | 확인됨 |
| 제어점 입력 | 거리 가중 풀링 한 블록 (v2/v3) | `aggregate_controls=True`, `exp(-dist/0.04)` (`mfm_training.py:596`) | 확인됨 (v1과 다름) |
| 재료 범위 | log E ∈ [5, 11], ν ∈ [0.05, 0.45], 시그모이드 | `log_E_min 5`, `log_E_max 11`, `nu_min 0.05`, `nu_max 0.45`, `model_MfM.py:412–415` | 확인됨 |
| 신뢰도 | c = 1 + exp(r) | `model_MfM.py:417` | 확인됨 |
| 소성 확률 | 시그모이드, τ = 0.5 | `model_MfM.py:419` | 확인됨 |
| RfD 입력 특징 | 19차원 (식 13) | `NUM_CELL_FEATURES = 19` (`model_RfD.py:144`) | 확인됨 |
| RfD FiLM 조건 | 16차원 (식 15) | `build_film_cond` 16 채널 (`model_RfD.py:430`) | 확인됨 |
| RfD 출력 | δ_max · tanh, 출력층 0 초기화 | `max_delta_v * torch.tanh(...)` (`model_RfD.py:421`), `zeros_` (`:401–402`) | 확인됨 |
| RfD 주기 | H_r = 10 서브스텝 | `gvc.h: 10` ("corrector fires every h MPM substeps") | 확인됨 (아래 D1과 함께 봄) |
| 앞 절반 식별 | MfM이 앞 절반으로 재료를 추정하고 고정 | `gvc.warmup_fraction: 0.5`, `validate_RfD.py` phase 1 | 확인됨 |
| 손실 가중치 (RfD) | w_C = w_L2 = 1 | `loss.chamfer_weight 1.0`, `tracked_l2_weight 1.0` | 확인됨 |
| 추적 지터 | 7 mm | `tracking_jitter_std 0.007` | 확인됨 |
| 격자, 감쇠, 밀도, 바닥 | G = 32, α = 0.999, ρ = 100, μ = 0.5, e = 0, 높이 0.02 | `num_grids 32`, `damping 0.999`, `rho 100`, `ground_friction 0.5`, `ground_elasticity 0.0`, `ground_height 0.02` | 확인됨 |
| 증강 개수 | 14 원본 × 80 = 1,120 (v2/v3) | `augment_recipe.yaml`: 그룹 `elastic` 70 + `plastic` 10 = 80, `run_recipe.py:186`에서 `len(sources) × Σ num_episodes` | 확인됨 (원본 14개 수는 데이터셋 README로 확인, §5) |
| 실제 데이터 | 12 에피소드 (탄성·탄소성) | HF README: `physcore.zip` 12 cases | 확인됨 (데이터는 받지 않음) |
| 원본 PhysTwin 에피소드 | 14 | HF README: `phystwin.zip` 14 cases (MIT) | 확인됨 |

## 2. 차이와 논문에 없는 것

판정은 **높음** (재현과 주장 해석에 영향), **중간** (수치에 영향 가능), **낮음** (구현 세부)입니다.

### D1. 시간 이산화가 논문 표 6과 다른 경로로 흐릅니다 (중간~높음, 정적 추적)

- 논문 표 6: 서브스텝 ∆t = 6.66e-4 s, **카메라 프레임당 H = 50** 서브스텝. 프레임 시간 = 0.0333 s.
- 코드 (실제 에피소드): `convert_to_episode.py:1123–1125`가 `simulation.dt = 1/fps`, **`steps_per_frame = 1`**을 씁니다. `dataset.py:717–733`는 이 값을 읽습니다. 그러고 나서 `rollout_runtime.py:49`의 `_real_world_chunk_dt`가 `chunk_dt = frame_dt / (rollout_steps × steps_per_frame)`를 계산합니다. `rollout_steps`는 `configs/*.yaml`에서 **25**입니다.
- 결과 (추적 기준): 서브스텝 ∆t ≈ 1.33e-3 s, **프레임당 25 서브스텝**. 프레임 시간은 논문과 같습니다. 그러나 서브스텝 수와 ∆t가 논문의 2배, 절반입니다.
- RfD 주기 h = 10 서브스텝은 코드에서 프레임당 2.5회, 논문 (H = 50)에서는 프레임당 5회입니다. 같은 "10 서브스텝"이라도 물리 시간이 2배 다릅니다.
- 확인하지 않은 것: 공개 에피소드 파일 (`episode_data.pt`, `config.yaml`) 안의 실제 값. 위 값은 `convert_to_episode.py`가 쓰는 설정에서 따라간 결과입니다. 기본값 `default_steps_per_frame: 2`, `default_dt: 0.000666`은 실제 에피소드에는 쓰이지 않는 것으로 보입니다.
- 영향: 명시적 MPM은 ∆t에 민감합니다. 같은 프레임 시간이라도 서브스텝 설정이 다르면 결과가 달라질 수 있습니다. 논문 표 6으로 재현하면 다른 설정이 됩니다.

### D2. 논문에 없는 명시적 강체 접촉과 조작 접촉 설정 (중간)

- 부록 A는 경계조건으로 **운동학적 핸들 (제어점)과 바닥**만 적었습니다.
- 코드 (`configs/train_RfD.yaml`, `augment_recipe.yaml`): 명시적 강체 접촉 `use_explicit_rigid_contact: true`, 표면 점 256 (`rigid_surface_fps`), 마찰 0.5, 속도 혼합 0.9, 끈적임 1.0, 엄격한 비관통 `rigid_strict_nonpenetration: true`, 투영 반복 2회 (`rigid_projection_iterations`).
- 조작 접촉 반경 `manipulation_controller_grid_contact_radius`는 기본 0.04 m, 케이스별 0.02–0.08 m입니다 (`per_sample_rollout_config`). 논문에 없습니다. 실제 데이터에 맞춰 조정한 값입니다 (README: "Radii are scene-dependent — retune for your own objects").
- 바닥 접촉의 접선 감쇠 0.25, 힘 제한 `max_force_norm 5000`, `apply_force_every_substep: true`도 논문에 없습니다.
- 영향: 탄소성 결과와 접촉이 많은 과제 (밧줄 들기)에 영향을 줄 수 있습니다. 논문 Limitations의 "complex contacts"와 연결됩니다.

### D3. 논문에 없는 관측 추적 보정 (PID / 스프링) (높음)

- `episode_runtime.py:577–640`의 `_build_correction_targets`는 **다음 청크의 관측 위치**로 보정 목표를 만듭니다. `:896–913`에서 목표 변위를 계산하고, `correction_targets` 손실과 `mpm_endpoint_loss`를 만듭니다.
- 설정: `correction_target_velocity_gain 100`, `damping 1.2`, `integral_gain 40`, `correction.mode: spring`, `omega_n 150`, `zeta 2.5`, `velocity_cap 5.0`. 논문에는 이 항들이 없습니다. 논문 식 4–5는 RfD 보정만 적었습니다.
- **검증 스크립트에서의 사용.** `validate_RfD.py`의 기본값은 `--correction-mode half` (`:580`)입니다. 그러면 phase 1 (앞 절반, 식별 구간)에서 RfD를 끈 상태로도 **관측 추적 PID 보정**이 돕니다 (`:280–310`, 주석: "material refresh, RfD OFF, per-frame PID correction, loss ignored"). phase 2 (뒤 절반, 평가)는 `use_gvc`에 따라 RfD를 켭니다.
- 절제 실험에 대한 함의: `--deactivate-gvc`로 RfD를 끄면 phase 1의 PID는 그대로 남습니다. 따라서 논문 표 3의 "MfM only" 조건에도 관측 추적 보정이 들어 있을 가능성이 큽니다 (확인 필요: 논문이 어떤 플래그로 표 3을 만들었는지 적지 않았습니다). 이 경우 표 3의 차이는 RfD만의 효과가 아닙니다.
- 식별 시간 (표 5, 11.4 s)에 이 보정이 포함되는지도 논문에 없습니다.

### D4. 체크포인트 선택과 평가 분할 (높음, 프로토콜)

- `configs/train_RfD.yaml`과 `validate_RfD.yaml`에서 `train_roots`와 `validation_roots`가 **같은 글롭** (`data_episodes/<dataset>/*/episode_0000`)입니다.
- `validate_RfD.py`는 체크포인트마다 **뒤 절반 (held-out 평가 구간)**의 손실을 계산합니다 (`:335–345`, 주석: "every checkpoint ... is scored on the held-out second half").
- `validate_RfD.py:486–560`은 이 **뒤 절반 검증 손실이 가장 낮은 에포크**를 `gvc_best_val.pt`로 복사합니다.
- 결과: (가) RfD는 같은 에피소드의 **앞 절반**으로 학습합니다 (논문 §4.2.4와 같음). (나) 평가는 **같은 에피소드의 뒤 절반**입니다. 시간상 분리되지만 객체와 궤적이 같습니다. (다) 체크포인트를 평가 구간 점수로 고릅니다.
- 논문은 체크포인트 선택 기준을 적지 않았습니다. 표 2의 PhysCoRe 값이 이 최선 체크포인트인지, 마지막 체크포인트인지 확인해야 합니다. 평가 구간을 선택에 쓰면 표 2는 낙관적일 수 있습니다.
- 평가 지표는 에피소드별 평균입니다 (`:340–345`, "Each episode contributes ONE unit"). 마지막 부분 창은 채점하지 않습니다.

### D5. RfD 블록의 활성화와 정규화 (낮음)

- 논문 식 16: `ReLU(γ ⊙ C h + β)`, FiLM은 `γ`와 `β`를 그대로 곱하고 더함.
- 코드 `model_RfD.py:313–333`: `GroupNorm` → FiLM `feats * (1 + γ) + β` → **SiLU**. 논문에 없는 GroupNorm이 있습니다.
- 영향: 재현 시 블록 구조가 다릅니다. 논문 식을 따라 구현하면 체크포인트를 읽지 못합니다 (`load_state_dict` strict).

### D6. 증강과 렌더링의 세부 (낮음~중간)

- 재료 분포 범위는 논문에 "fixed uniform ranges"라고만 있습니다 (부록 D). 코드 `augment_recipe.yaml`: 탄성 log E 평균 [6, 9], 표준편차 [0.5, 1.5], ν 평균 [0.15, 0.35]. 탄소성 log E 평균 [10, 10.5], ν 평균 [0.30, 0.38], 표준편차 0. Perlin 옥타브 4, 지속성 0.55, 라쿠나리티 2.0.
- 소성: `VonMisesPlasticity.sigma_y: 200.0` (논문은 τ_y를 적지 않았습니다). 단위는 확인하지 못했습니다.
- 렌더링 잡음: 깊이 잡음 std 0.002, 상대 0.001, 결측 1% (논문에 없음).
- 증강 중 제어 궤적 재생은 `rollout_steps: 25`를 씁니다 (D1과 같은 이야기).

### D7. 학습 세부 (낮음)

- RfD: 80 에포크, 학습률 1e-4 → 1e-5 (30–80 에포크에서 감쇠), 기울기 클리핑 0.5, 시드 42, `bptt_frames 5`. 논문 부록 E에는 RfD 에포크와 학습률이 없습니다.
- MfM: `cosine_decay: true`지만 `lr = final_lr = 1e-4`라서 사실상 고정입니다. 논문과 맞습니다.
- 증강 회전과 거울 뒤집기는 코드에 있습니다 (z축 회전, x축 50% 뒤집기). 논문 부록 E와 맞습니다.

### D8. 평가 지표, 기준선, 계획 코드의 부재 (재현 범위)

- **IoU, PSNR, SSIM, LPIPS의 계산 코드가 없습니다.** `gaussian_splatting/`은 3DGS 렌더러이고, `render_MfM_confidence_3dgs.py`는 신뢰도 색칠용입니다. 표 2의 시각 지표를 이 저장소에서 다시 계산할 수 없습니다.
- CD와 추적 손실은 `validate_RfD.py`에 있습니다 (편측 Chamfer, 가시 점 L2).
- **PhysTwin, PGND 기준선 실행 코드가 없습니다.** 표 2, 표 4, 표 5의 기준선 값은 이 저장소로 재현할 수 없습니다.
- **계획 실험 (§5.6)의 코드가 없습니다.** 목표 설정, 계획자, KUKA 연동 모두 없습니다.
- **식별 시간 측정 스크립트가 없습니다.** 표 5의 측정 조건을 확인할 수 없습니다.

### D9. 용어 혼동 (낮음)

- 코드에서 MfM은 `refiner`로, RfD는 `gvc`로 부릅니다 (주석: "`gvc` = legacy RfD, `refiner` = MfM"). 같은 모듈입니다.

## 3. 저장소 구조와 실행 경로 (정적)

- 학습: `train_MfM.py` (증강 에피소드, 2000 에포크), `train_RfD.py` (실제 에피소드 앞 절반, 80 에포크).
- 검증: `validate_MfM.py`, `validate_RfD.py` (뒤 절반, 체크포인트마다).
- 데이터 생성: `datagen/process2d/` (CoTracker3, Grounded-SAM-2 쪽), `datagen/convert3d/convert_to_episode.py` (Poisson 완성, 입자 채움), `datagen/augment/run_recipe.py` (재료 증강).
- 물리: `physcore/mpm/` (Warp, `mpm_model.py`), `physcore/constitutive/`, `physcore/particle_flow/` (롤아웃, RfD 엔진, 손실), `physcore/sim/` (기본 물체 생성, 강체 접촉).
- 모델: `physcore/model_MfM.py` (그래프 U-Net), `physcore/model_RfD.py` (spconv 희소 U-Net).

## 4. 구현 가능성 (이 맥: Apple M4 Pro, 약 24 GiB 통합 메모리, MPS, CUDA 없음)

**판정: 전체 재현 불가. 체크포인트 추론은 포팅 뒤 조건부. 소형 자체 구현 토이는 가능.**

| 작업 | 판정 | 근거 |
|---|---|---|
| 공식 코드 실행과 설치 | **불가** | README는 CUDA 12.8, RTX 5090 (`sm_120`), PyTorch3D 소스 빌드, diff-gaussian-rasterization과 simple-knn CUDA 확장을 요구합니다. `spconv-cu126`은 CUDA 전용입니다. 코드에 CUDA 지정이 66곳 있습니다 (`device: cuda`). 이 맥에는 CUDA가 없습니다 |
| RfD 희소 U-Net (spconv) | 포팅 필요 | 3D 격자 32³ (32 768 셀)라서 밀집 3D 합성곱으로 바꿀 수 있지만 MPS 메모리와 속도는 측정하지 않았습니다. 추정 1–2주 |
| MfM 그래프 U-Net 추론 (5.7 MB) | 조건부 | PyTorch 중심이라 MPS나 CPU로 옮길 여지가 있습니다. FPS 풀링과 이웃 탐색 코드를 확인해야 합니다. 측정 전 |
| MPM 시뮬레이터 (Warp) | 조건부 | `warp-lang`은 CPU 백엔드가 있지만 이 맥에서 작동하는지, 속도가 어떤지는 확인하지 않았습니다. 확인 전 |
| 공개 체크포인트 추론 (RfD 7.0 MB) | 조건부 | 데이터 `physcore.zip` 4.2 GB 다운로드와 승인이 필요합니다 (1 GB 이상 보고 규칙). 위 포팅이 끝나야 합니다 |
| 증강 1,120개, MfM 2000 에포크, RfD 80 에포크 학습 | **불가** | GPU 전제. 시간 측정 없음. 대략 수십 GPU 시간 이상으로 보입니다 (추정, 측정 없음) |
| 표 2 기준선 (PhysTwin, PGND) 재현 | **불가** | 두 기준선 코드는 공개되어 있지만 GPU 전제이고, 이 저장소에 측정 스크립트가 없습니다 |
| 계획 (표 4), 신뢰도 탐색 (그림 4) | **불가** | 로봇 (KUKA), 계획 코드 없음 |
| 작은 2D MLS-MPM 토이 + 잔차 보정 + 신뢰도 헤드 (numpy 또는 torch CPU) | **가능** | 논문 방법의 축소판. 중심 주장 (잔차가 예측 오차를 줄인다, 신뢰도가 관측 영역과 맞는다)을 토이에서 확인할 수 있습니다. 일반화 주장은 확인할 수 없습니다. 노력 1–2주 (추정) |

노력 추정 (측정 전, 추정입니다): 토이 1–2주, 체크포인트 추론 경로 포팅 2–3주 (+ 데이터 4.2 GB 승인), 전체 학습 재현은 이 맥에서 범위 밖입니다.

## 5. 라이선스와 권리

- **코드** (`LICENSE`): "MIT License", Copyright (c) 2026 Haocheng Yin and Shuohan Tao, Lunar Lab @ Georgia Tech. 아래에 제3자 저작권 줄이 있습니다: OmniPhysGS (Yuchen Lin, 2025, MPM 코어와 구성 법칙), PhysTwin (Hanxiao Jiang, 2025, 캡처와 2D/3D 데이터 처리). GitHub API는 이 파일을 인식하지 못해 `NOASSERTION`으로 보고합니다. 제3자 부분의 원 라이선스는 확인하지 않았습니다.
- **데이터셋 (HF)**: 라이선스 `other`, 이름 "mixed-mit-and-cc-by-4.0". `data/phystwin/` (14 케이스)는 PhysTwin 데이터셋의 MIT 재배포이며 저작권 표시를 유지해야 합니다. `data/physcore/`, `gaussian_output/`, `checkpoints/`, `configs/`는 CC BY 4.0입니다. 저장소 크기는 약 7.8 GB입니다 (`physcore.zip` 4.2 GB, `phystwin.zip` 3.4 GB).
- **논문**: arXiv CC BY 4.0 (abs 페이지), 원문 PDF의 저작권 표시는 찾지 못했습니다 (`pdftotext`로 ©, copyright, All rights reserved, 회사명 검색). 그림 2–5 (7–9쪽)은 데이터셋 촬영 프레임, 실로봇 장면, 그 위에 그린 신뢰도 렌더입니다 (7–9쪽을 렌더해서 확인). CC BY 4.0이어도 데이터셋 프레임과 실로봇 사진은 이 분석의 공개 기준에서 뺍니다. 그래서 **공개본에는 논문 그림을 넣지 않습니다**.

## 6. 점검 목록 (논문 저자에게 물을 수 있는 것)

1. 논문 표 6의 H = 50, ∆t = 6.66e-4가 실제 실험의 설정인가? 코드의 실제 에피소드 경로는 25 서브스텝 (D1).
2. 표 3의 "MfM only"는 관측 추적 PID를 끈 조건인가? (D3)
3. 표 2의 PhysCoRe 값은 어떤 체크포인트에서 나왔는가? 선택 기준이 평가 구간이었는가? (D4)
4. 식별 시간 11.4 s에 PID 보정이 포함되는가? 측정 조건은? (D3, 표 5)
5. 에피소드 수의 객체 유형별 분할, 에피소드별 값, 단위 (표 2, 5)
6. `sigma_y = 200`의 단위와 의미 (D6)
