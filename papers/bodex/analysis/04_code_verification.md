# 04. 공식 코드 대조 (정적 읽기)

대상: `JYChen18/BODex` (`repo/`, 얕은 클론, 커밋 `06b9a3c`, 2026-04-21). 논문 v3와 대조했습니다. 실행, import, 설치, pickle 로드는 하지 않았습니다. 클론한 내용은 읽기 전용으로만 다뤘습니다.

검증 라벨: **K**는 논문과 코드가 일치함, **D**는 차이, **U**는 이 저장소만으로 확인할 수 없음입니다.

## 0. 요약

- 핵심 구조는 논문과 맞습니다. QP 에너지 (`grasp_energy/qp.py`), 8면 피라미드 마찰 근사, 3단계 (300/100/100) 스케줄, 손끝 다섯 접촉, 1 cm 사전 파지, 여섯 방향 힘-폐쇄 목표는 모두 확인했습니다 (K1–K8).
- 그러나 **논문에 없는 설정과 처리가 많습니다** (D1–D11). 가장 중요한 것은 (a) 하위 QP의 β가 고정 하이퍼파라미터가 아니라 변수 k ≥ 0.2이고, (b) 힘 제약 (Σ f ≥ 1.0)에 논문에 없는 부분 집합 제약 두 개가 붙고, (c) 합성 마찰 계수가 0.1이며 평가 (MuJoCo 0.6)와 다르고, (d) QP를 5 반복마다 한 번만 풀고, (e) 최적화기가 "gradient descent"가 아니라 L-BFGS라는 점입니다.
- 평가 코드 (MuJoCo SSR, PD, CDC, SPD, FVR)는 이 저장소에 없습니다. 벤치마크는 별도 저장소 DexGraspBench이고, 그 README는 "main 브랜치는 논문과 설정이 조금 다르다"고 적습니다 (U1). 표 II–IV의 시뮬 수치를 이 코드로 재현할 수 없습니다.
- 라이선스: 코드는 NVIDIA License (cuRobo)와 CC BY-NC 4.0 (나머지)입니다. 상업 사용 불가 (§5).

## 1. 일치 (K)

| 번호 | 항목 | 논문 | 코드 (위치) |
|---|---|---|---|
| K1 | 마찰 원뿔 근사 | 8면 피라미드, LCQP (§IV-A, 식 8) | 비틀림 마찰 계수 0이면 8면 (`grasp_energy/qp.py` `num_friction_approx`) |
| K2 | 3단계 스케줄 | coarse 300, fine 100, 최종 100 (§IV-D) | `opt_progress: [0.0, 0.6, 0.8]` × `n_iters: 500` (`manip/sim_shadow/fc.yml`, `task/gradient_grasp_fc.yml`) |
| K3 | 목표 접촉 거리 | 사전 파지 1 cm 감소, 최종에서 0 (§IV-D) | `distance: [0.01, 0.01, 0.0]` (`fc.yml`, `sim_allegro/fc.yml`, `sim_leap/fc.yml`, `tabletop.yml`) |
| K4 | 접촉 점 | 손끝 (§VI) | `rh_ffdistal/2`, `rh_mfdistal/2`, `rh_rfdistal/2`, `rh_lfdistal/2`, `rh_thdistal/1` |
| K5 | coarse는 구, fine은 메쉬 (GJK) | §IV-C | `contact_query_mode -1` (구)와 `0` (메쉬, Coal GJK). `max_ge_stage: 0`이므로 QP 에너지는 coarse에서만 쓰고 fine에서는 거리 에너지 (Q′에 해당) |
| K6 | 배치 ReLU-QP | §IV-B, [8] | `opt/qp/reluqp.py` `BATCHED_RELUQP` (PyTorch ADMM) |
| K7 | 여섯 목표 wrench | 힘-폐쇄 ±x, ±y, ±z (§IV-A) | `gamma: 180`에서 ±f와 ±축 1, 2 → 여섯 방향 (`grasp_cost.py` `init_TWS`) |
| K8 | 에너지 = 잔차 제곱 | 식 7의 ‖·‖² | `grasp_energy` = sᵀ Q₂ s, Q₂에는 정규화 없음 (`qp.py` `forward`) |
| K9 | 손 간 하이퍼파라미터 동일 | §V-C-3 "all hyperparameters kept the same" | Allegro, Leap, Shadow 설정 파일의 ge_param과 contact_strategy 동일 |

## 2. 차이 (D)

| 번호 | 논문 | 코드 | 영향 |
|---|---|---|---|
| **D1** | 합성에 쓰는 마찰 계수 값 없음. §V-A의 0.6은 MuJoCo 평가용 | `ge_param.miu_coef: [0.1, 0]` (세 손 모두). 합성은 μ = 0.1로 함 | 합성은 보수적 마찰로 하고 평가는 0.6으로 함. 합성 기준과 평가 기준이 다름. 본문에 이 설정이 없음 |
| **D2** | 식 7의 β는 고정 양수 하이퍼파라미터 (값 없음). 목표 wrench는 βt_j | `k_lower: 0.2`, 마지막 변수 k가 목표 wrench의 크기. k ≥ 0.2 (`qp.py` `_init_LCQP_*`) | 크기가 최적화 변수. 목표 크기가 파지마다 달라질 수 있음 |
| **D3** | 식 9: Σ_i f_j,i,1 ≥ γ (γ 값 없음) | Σ 전체 다섯 접촉 ≥ 1.0, 세 손가락 (0,1,2) 합 ≥ 0.4, 엄지 (4) ≥ 0.5 (`fc.yml` `pressure_constraints`) | 부분 집합 제약이 논문에 없음. 손가락 쓰임을 강제함 |
| **D4** | 제약에 상·하한 없음 (식 8: 마찰 원뿔만) | 법선 힘 0.01 ≤ f_n ≤ 1 (`qp.py` `_init_LCQP_lu`) | 힘 상한 1은 논문에 없음 |
| **D5** | QP는 "각 반복에서" 풂 (§IV-B) | `solve_interval: 5`. 5번 중 1번만 풀고 나머지 4번은 이전 해 재사용 (`qp.py` `forward`) | 대부분의 반복에서 힘이 한 단계 이전 값. 논문 설명과 다름 |
| **D6** | 근사 미분 방법만 설명, 메쉬 질의 빈도 없음 | `if self.count % 5 == 0`로 메쉬 접촉을 5 반복마다 갱신 (`grasp_cost.py` `forward`, 모드 0) | 빈도가 논문에 없음 |
| **D7** | 식 3 목적 = Σ_j Q_j (가중치 없음) | 가중합: 힘 항 ×100 (평균), 거리 항 ×1000, 정규화 항 ×10 (`gradient_grasp_fc.yml` `grasp_cfg.weight`) | 가중치와 평균 (합 대신) 처리 |
| **D8** | x_s = 2x − x_p를 x 전체에 씀 (§IV-D) | 관절 부분만 2x − x_p. 루트 자세 7차원은 파지 자세를 그대로 씀 (`example_grasp/plan_batch_env.py`) | 루트는 스퀴즈 방향으로 밀지 않음. 논문 식과 다름 |
| **D9** | 초기화를 설명하지 않음 | `seed_num: 20`, 물체 표면 샘플 128개 (볼록 껍질, 충돌 없음), 루트 회전 ±180°, 이동 ±3 cm, 관절 초기값 고정 (`fc.yml` `seeder_cfg`) | 시드 생성 규칙이 논문에 없음. 결과에 큰 영향 가능 |
| **D10** | 최근접 점 미분을 "유한 차분"으로 근사 (§IV-C) | 접촉점 주변 6방향 섭동, 크기 Halton 샘플 [0.01, 0.02] (`grasp_cost.py` `update_perturb_info`) | 섭동 크기 (단위 미기재, 아마 m)가 논문에 없음 |
| **D11** | 최적화는 "gradient descent" (§IV-B) | L-BFGS (`opt/newton/lbfgs.py`), history 6, 모멘텀, 탐색 greedy, 기울기 정규화 (`gradient_grasp_fc.yml`) | 최적화기 이름과 종류가 논문과 다름 |

## 3. 확인 불가 (U)

| 번호 | 항목 | 이유 |
|---|---|---|
| **U1** | 시뮬 SSR, PD, CDC, SPD, FVR 계산과 판정 (3 s, 5 cm, 15°, 여섯 중력) | 이 저장소에 없음. DexGraspBench (별도 저장소, README에 "main은 논문과 다름")에 있음. 클론하지 않음 |
| **U2** | 시뮬 파라미터 (μ = 0.6, 비틀림 0.02, noslip 2, 질량 30 g, 중력 9.8) | U1과 같은 이유. 이 저장소의 `sim_*` 설정에는 마찰 0.1만 있음 |
| **U3** | 물체 2,397개와 "DGN_2k" 이름의 관계, 스케일 4개 | 자산은 HF 데이터셋 (`DGN_2k_processed`, 0.55 GB). 받지 않아 확인 못 함 |
| **U4** | 데이터셋 파지 수 (4.09 M, 3.08 M) | 생성 설정 (시드 수, 필터)과 자산이 없음 |
| **U5** | 속도 49 grasp/s (3090), 평가 30.6 grasp/s | GPU와 CUDA 그래프가 필요. 이 맥에서 측정 불가 |
| **U6** | Leap 손 설정의 결과 | 이 저장소의 `sim_leap/fc.yml`은 있으나 논문에 Leap 수치가 없음 |
| **U7** | 실로봇 배포 코드 (§V-E) | `example_grasp/plan_mogen_batch.py`는 있으나 실제 로봇 드라이버와 카메라 파이프라인은 저장소 밖 |

## 4. 구현 가능성 (이 맥 기준)

이 맥: Apple M4 Pro, 약 24 GiB 통합 메모리, MPS, CUDA 없음.

| 항목 | 판정 | 근거 |
|---|---|---|
| 공식 합성 파이프라인 실행 | **불가** | README 설치가 `pytorch-cuda=12.1`과 cuRobo CUDA 확장 빌드 (`setup.py`, `src/curobo/geom/cpp`)를 요구. 설정이 `use_cuda_graph: True`, `use_cuda_kernel: True`, `use_cuda_line_search_kernel: True` |
| 논문 규모 데이터셋 재생성 | **불가** | 3090에서 Shadow 49.8 grasp/s로 4.09 M 파지 ≈ 23 GPU 시간 (표 III 속도로 계산). 맥에서는 더 느림 (측정 전) |
| 소형 재구현: QP 에너지 + 3단계 스케줄 + 접촉 모델 (물체 1–2개, 파지 10–100개) | **조건부** | torch (MPS 또는 CPU)로 식 7–11과 스케줄을 옮김. 메쉬 최근접점은 trimesh 또는 Coal 설치가 필요함 (새 패키지 설치 승인 필요). 노력 **2–3주 (추정, 측정 전)** |
| 공식 코드의 해 대 논문 식 대조 (입력 고정) | **조건부** | 파일 형식과 텐서 차원을 맞춰야 함. CPU 가능성은 CUDA 커널 우회 여부에 달림 (측정 전) |
| 사전 생성 파지 (Shadow 0.37 GB, Allegro 0.32 GB, Leap 0.32 GB)로 표 II 일부 재계산 | **조건부** | 다운로드 승인 필요. 기준 값과 대조는 CPU로 가능. 시뮬 평가는 U1 (DexGraspBench)이 있어야 함 |
| MuJoCo 기반 SSR 재평가 | **조건부** | MuJoCo 설치 가능 여부는 승인 후 확인 (이 작업에서 설치하지 않음). DexGraspBench의 설정 차이 (main과 논문)를 먼저 확인해야 함 |
| 실로봇 재현 | **불가** | Shadow Hand, UR10e, Kinect 필요 |

**판정: 이 맥에서 BODex 공식 파이프라인은 돌지 않습니다. 논문의 식과 3단계 최적화를 소형으로 옮기는 것은 가능하고, 그 과정에서 D1–D11의 설정 선택이 결과에 주는 영향을 CPU로 볼 수 있습니다. 노력 2–3주 (추정).**

## 5. 라이선스

- `repo/LICENSE`: NVIDIA License, Copyright (c) 2023 NVIDIA. 비상업 사용 조건 (43–44행: "non-commercially"이며 NVIDIA와 계열사는 상업 사용 가능).
- `repo/README.md` 97–104행: "The code from cuRobo is used under the NVIDIA license. All other parts of this work are licensed under CC BY-NC 4.0."
- 결론: 코드와 데이터는 비상업입니다. 우리 저장소의 공개 산출물에 코드를 넣지 않았고, 설명과 짧은 식 인용만 했습니다. 논문 자체는 CC BY 4.0이므로 `01`과 `03`의 인용은 괜찮습니다.

## 6. 읽은 파일

- `src/curobo/rollout/cost/grasp_energy/qp.py` (223행): QP 에너지 전체
- `src/curobo/rollout/cost/grasp_energy/dfc.py`, `tdg.py`, `chq1.py`: 비교 에너지 (읽기만)
- `src/curobo/rollout/cost/grasp_cost.py` (268행): 목표 wrench, 단계 전환, 손실 가중치, 섭동
- `src/curobo/opt/qp/reluqp.py` (172행): 배치 ReLU-QP
- `src/curobo/opt/newton/lbfgs.py`: 최적화기 (이름만 확인)
- `src/curobo/content/configs/manip/sim_shadow/fc.yml`, `sim_allegro/fc.yml`, `sim_leap/fc.yml`, `tabletop.yml`
- `src/curobo/content/configs/task/gradient_grasp_fc.yml`, `base_grasp.yml`
- `example_grasp/plan_batch_env.py` (254행): 사전 파지와 스퀴즈 자세
- `README.md`, `LICENSE`

읽지 않은 것: `grasp_solver.py` (870행, 주요 스케줄 흐름만 확인), cuRobo 본체 (`src/curobo`의 나머지), `tests/`, `benchmark/`, `docker/`, `example_grasp/plan_mogen_*.py` (세부).
