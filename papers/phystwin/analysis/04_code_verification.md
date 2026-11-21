# 04. 코드 대조: 논문 ↔ 공식 저장소 (PhysTwin)

대상: `repo/` = github.com/Jianghanxiao/PhysTwin, 커밋 `81c718790a37e5e0102eb77af2c6edd34a9db25f` (2026-08-24, 얕은 clone, MIT). 정적으로만 읽었습니다. 실행, import, 설치, 데이터 다운로드는 하지 않았습니다. 원 논문은 2025-03 v1이므로 저장소는 **논문 이후 17개월 동안 바뀐 코드**입니다. 논문 결과가 어느 커밋에서 나왔는지는 알 수 없습니다.

차이의 분류: **불일치** = 논문 서술과 코드가 다름, **누락** = 논문에 없는 구성 요소가 코드에 있음, **서술 차이** = 같은 것을 다르게 부름, **미확인** = 데이터가 없어 확인 못 함.

## 1. 저장소 구조와 실행 경로

| 논문 구성 | 코드 위치 |
|---|---|
| 스프링-질량 시뮬레이터 (식 1, Warp) | `qqtt/model/diff_simulator/spring_mass_warp.py` |
| 0차 최적화 (CMA-ES, 1단계) | `optimize_cma.py`, `qqtt/engine/cma_optimize_warp.py` |
| 1차 최적화 (경사, 1단계 후반) | `train_warp.py`, `qqtt/engine/trainer_warp.py` |
| 손실 (chamfer, track, acc) | `spring_mass_warp.py` 1033–1120행, 384–520행 |
| 평가 (CD, 추적) | `evaluate_chamfer.py`, `evaluate_track.py` |
| 평가 (PSNR, SSIM, LPIPS, IoU) | `gaussian_splatting/evaluate_render.py` |
| 외형 (3D Gaussian, 1단계 외 2단계) | `gs_train.py`, `gs_run.sh`, `gaussian_splatting/` |
| LBS 변형 (부록 B) | `gs_render_dynamics.py`, `gaussian_splatting/dynamic_utils.py` |
| 형상 사전 (TRELLIS, 등록) | `process_data.py`, `data_process/shape_prior.py`, `data_process/align.py` |
| 데이터 구성 | `data_config.csv` (22행), `process_data.py` (70/30 분할) |
| 설정 | `configs/real.yaml`, `qqtt/utils/config.py` |

실행 순서: `script_process_data.py` → `script_optimize.py`(CMA) → `script_train.py`(1차) → `gs_run.sh`(외형) → `export_render_eval_data.py` → `evaluate.sh`.

## 2. 대조 결과 (D1–D19)

### 2.1 손실과 목적 함수

**D1. 단방향 Chamfer의 거리 형식 (서술 차이).** 논문 식 (3)은 "single-direction Chamfer distance"라고만 씀. 코드는 두 가지를 다르게 씀.
- 최적화(`compute_chamfer_loss`, `spring_mass_warp.py` 384–398행): 최근접 거리의 **제곱**을 `num_valid`로 나눔.
- 평가(`evaluate_chamfer.py` 56–66행): `chamfer_distance(..., single_directional=True, norm=1)`, 즉 **L1 거리**의 평균.
→ 논문 표 1의 CD는 평가 쪽(L1)으로 보임. 단위는 어디에도 없음(추정: 미터). 최적화 목적과 평가 지표가 다르므로 "최적화한 값을 보고했다"는 해석은 맞지 않음.

**D2. 가속도 정규화 항 (누락).** 논문 식 (3)과 본문에는 없음. 코드에는 `acc_loss`가 있고 가중치는 `acc_weight = 0.01`(`qqtt/utils/config.py` 48행). 총 손실은 `loss = chamfer + track + acc`(`spring_mass_warp.py` 515–520행). 인접 두 스텝의 속도 차이(가속도)에 Huber형 벌점을 둡니다(`compute_acc_loss` 464–510행).
→ 0차 최적화(CMA)의 목적에 이 항이 들어감. 논문 식으로는 설명되지 않는 정규화입니다.

**D3. 가속도 항의 활성 범위 (정적 판단, 실행 미확인).** `acc_count`는 CMA(`cma_optimize_warp.py` 368, 400행)에서만 설정됨. 두 번째 프레임부터 켜짐. 1차 학습 루프(`trainer_warp.py`)는 `set_acc_count`를 부르지 않으므로 0으로 남아 가속도 항이 꺼져 있음. 즉 **1단계 0차에서만 쓰이고 1차 경사 최적화에서는 쓰이지 않음.**

**D4. 0차 최적화 대상 (서술 차이).** 논문 §4.2.1은 0차 최적화가 "collision parameters and homogeneous spring stiffness"를 맞춘다고 씀. 코드의 CMA는 **다섯 개**를 맞춤: 동질 스프링 강성(`spring_Y`), 물체 반경과 최대 이웃 수, 제어점 반경과 최대 이웃 수(`cma_optimize_warp.py` 203–257행). 충돌 파라미터는 CMA에 없고 설정 파일 값으로 고정(`collide_elas 0.5`, `collide_fric 0.3` 등). 충돌 파라미터는 1차에서 Adam으로 학습(`trainer_warp.py` 154–162행). → 논문의 역할 분담은 맞지만, 0차에 들어가는 항목이 논문 서술과 다름.

**D5. 추적 손실 형식 (미지정).** 논문 식 없이 "tracking error"라고만 씀. 코드(`compute_track_loss`, `spring_mass_warp.py` 402–440행)는 **축별 smooth-L1(Huber, δ = 1.0)** 을 씀. 단위가 미터라면 δ = 1 m는 실제로 거의 모든 오차에서 L2 형태입니다. 평가(`evaluate_track.py`)는 **유클리드 거리의 평균**이므로 학습과 평가 지표가 다릅니다.

### 2.2 형상과 데이터

**D6. 형상 사전은 22개 중 13개에만 쓰임 (불일치, 중요).** `data_config.csv`의 셋째 열(`shape_prior`)이 `True`인 시나리오는 **13개**, `False`는 **9개**(천 8개와 택배 상자 1개). `process_data.py` 158–162행은 `False`이면 `data_process_sample.py`를 형상 사전 없이 실행합니다. 논문 §4.2.1은 모든 물체에 TRELLIS 형상 사전을 쓴다고 쓰고, 이 예외를 밝히지 않습니다. 논문 초록의 "complete geometry"는 천에 대해 성립하지 않습니다. 천과 상자는 관측된 부분 점군으로 맞춥니다.
→ 결과 표 1–2의 평균 22개 중 9개는 완성되지 않은 형상입니다. 시나리오별 결과 없이는 두 집단을 나눠 볼 수 없습니다.

**D7. 시나리오 수와 분할 (일치).** `data_config.csv`에 22행(밧줄 5, 봉제인형 4, 장난감 3, 천 8, 공룡 1, 택배 1). 분할은 `process_data.py` 169–170행에서 `int(frame_len * 0.7)`로 **시간 순서로** 나눕니다. 논문 7:3과 일치하고, "미래 예측"은 같은 영상의 뒤쪽 30% 프레임입니다.

**D8. 학습은 학습 구간만 사용 (일치).** CMA는 `max_frame = cfg.train_frame`까지만(`cma_optimize_warp.py` 366–372행), 1차는 `range(1, cfg.train_frame)`(`trainer_warp.py` 320–332행)만 씀. 논문의 "학습 구간만으로 구축" 주장과 맞음.

**D9. 제어점 선택 (미확인, 구현 위치 불명).** 논문은 손 마스크 추적 점을 3D로 올린 뒤 **최원거리 샘플링(FPS)** 으로 제어점을 고른다고 씀. 저장소에서 `farthest`, `fps`(대소문자 무시)로 검색하면 구현이 없음(`data_process/visualizer.py`의 fps는 영상 프레임률). 제어점은 `data_process_track.py`에서 **제어 마스크 안에서 모든 프레임에 보이는 추적 점을 고르고** `get_final_track_data(controller_threhsold=0.01)`로 거른 것으로 보임. 물체 점은 `data_process_sample.py`에서 **복셀 다운샘플링(0.005)** 으로 정함. → FPS 대신 다른 절차를 쓴 것으로 보이며, 정확한 절차는 데이터 없이 확정할 수 없음.

**D10. 기준선 확장 (미확인).** Spring-Gaus와 GS-Dynamics를 저자가 확장한 코드는 이 저장소에 없음. 논문은 Spring-Gaus를 **성공한 경우만** 보고했다고 씀(§5.1). 실패 횟수는 어디에도 없음.

### 2.3 평가 프로토콜

**D11. 2D 평가 시점 (서술 차이).** 논문: "center viewpoint"에서만 평가. 코드(`gaussian_splatting/evaluate_render.py` 66행): `for view_idx in range(1):  # only consider the first view`. 첫 번째 시점이 중앙 시점인지는 데이터 없이 확인 불가.

**D12. 사람 마스크로 가리기 (누락).** 코드는 PSNR, SSIM, LPIPS, IoU를 계산하기 전에 **사람 마스크를 정답과 렌더링 모두에서 가림**(`evaluate_render.py` 76–86행). 논문 §5.1에는 없음. 손이 프레임을 가리는 영역을 빼는 것이므로 점수가 올라가는 방향일 수 있음.

**D13. IoU 정의 (서술 없음).** 코드는 렌더링 알파(>0)와 정답 마스크(>0)의 IoU(`evaluate_render.py` 94행). 논문은 "silhouette alignment using IoU"라고만 씀.

**D14. 추적 평가의 대응 (서술 없음).** `evaluate_track.py`는 **프레임 0**에서 정답 추적 점과 가장 가까운 예측 정점을 KDTree로 찾아 고정 대응을 만들고, 이후 프레임에서 유클리드 거리의 평균을 냄. 점 개수는 `gt_track_3d.pkl`에서 오는데 데이터를 받지 않아 **9개인지 확인하지 못함 (미확인)**.

**D15. Chamfer의 정답 점 (일치).** 평가는 `object_visibilities`로 **보이는 물체 점만** 씀(`evaluate_chamfer.py` 56–60행). 논문의 "partial ground truth"와 같음.

### 2.4 시뮬레이터와 설정

**D16. 설정값은 논문에 없음 (누락).** 논문은 감쇠 계수 γ, δ와 시간 간격을 값으로 주지 않습니다. `configs/real.yaml`에 있음: `dt = 5e-5`, `num_substeps = 667`(= round(1/30 / 5e-5), 30 FPS와 일치), `dashpot_damping = 100`, `drag_damping = 3`, `init_spring_Y = 3e4`, `spring_Y` 범위 [0, 1e5], `collide_elas 0.5`, `collide_fric 0.3`, `object_collide_* 0.7 / 0.3`, `object_radius 0.02`(이웃 30), `controller_radius 0.04`(이웃 50), `collision_dist 0.02`, `chamfer_weight 1`, `track_weight 1`, 1차 반복 200회, Adam 학습률 1e-3.

**D17. 스프링 강성의 매개변수화 (서술 없음).** 저장된 `spring_Y`는 `torch.exp`를 씌워 기록하고(`trainer_warp.py` 415행), 시뮬레이터는 `wp.exp(spring_Y)`를 클램프함(`spring_mass_warp.py` 98, 124행). 최적화 변수는 **로그 공간**으로 보임. 초기화 변환은 확인하지 못함.

**D18. 감쇠 적용 (서술 차이).** 논문의 δ는 속도에 곱하는 계수로 적혀 있음. 코드는 `v2 = v1 · exp(−dt · drag_damping)`(`spring_mass_warp.py` 156–160행)로 **스텝마다 지수 감쇠**를 씀. 같은 역할이지만 δ 값을 `exp(−dt·3) ≈ 0.99985`로 봐야 함.

**D19. CMA 세부 (논문에 없음).** `optimize_cma.py`의 `--max_iter` 기본값 **20**, 시드 **42**, 정규화 변수 범위 [0, 1]. 물체 반경 [0.01, 0.05], 이웃 [10, 50], 제어점 반경 [0.01, 0.08], 이웃 [10, 80]. README는 CMA가 "약 12분"이라고 씀.

### 2.5 외형과 LBS

**D20. 등방성 Gaussian (일치).** `gs_run.sh` 33행의 `--isotropic`. 논문 §4.2.2의 "isotropic"과 맞음.

**D21. 렌더링 손실 가중치 (논문에 없음).** `gs_train.py` 154행: `(1 − λ) L1 + λ (1 − SSIM)`, `λ = 0.2`(`arguments/__init__.py` 96행). 깊이와 분할 손실은 0(`lambda_depth`, `lambda_seg`). 반복 10,000(`gs_run.sh` 27행). 논문에 값이 없음.

**D22. LBS의 K (논문에 없음).** 부록 B는 K-최근접 정점으로 보간한다고 씀. 코드는 `gs_render_dynamics.py` 234, 254행에서 **K = 16**, 가중치 `1/(거리+1e-6)`을 정규화(`dynamic_utils.py` 268–273행). 식 (8)과 같은 형식.

### 2.6 버전과 업데이트

**D23. 논문 이후 변경 (미확인).** README 업데이트에 2025-04부터 2026-08까지 변경이 있음. 예: 2025-10-26 자기충돌 가속(별도 브랜치 `release_collision_accelerate`에 있다고 함), 2026-08-24 Boba(ECCV 2026)와 Deform360(ECCV 2026) 발표. 논문 표의 결과가 어느 커밋에서 나왔는지, 그 뒤 변경이 결과를 바꾸는지는 **데이터와 실행 없이 알 수 없음.**

## 3. 대조 요약

| 구분 | 번호 | 요점 |
|---|---|---|
| 불일치 | D6 | 형상 사전이 22개 중 9개에 없음. 논문 본문에 예외 없음 |
| 누락 | D2, D16, D12, D21 | 가속도 정규화 항, 설정값, 사람 마스크, 렌더링 가중치 |
| 서술 차이 | D1, D4, D5, D11, D18 | Chamfer 제곱 대 L1, 0차 대상, 추적 손실 형식, 2D 시점, 감쇠 |
| 구현 위치 불명 | D9 | FPS 제어점 |
| 미확인 | D10, D14, D23 | 기준선 확장, 추적 점 개수, 논문 이후 변경 |
| 일치 | D7, D8, D15, D20 | 시나리오 수, 학습·테스트 분할, 보이는 점, 등방성 |

## 4. 구현 가능성 (이 맥 기준)

**이 저장소는 이 맥에서 바로 돌릴 수 없습니다.**

1. `spring_mass_warp.py` 6행이 `wp.set_device("cuda:0")`을 **하드코딩**합니다. Warp의 Mac 지원(CPU 또는 MPS)으로 바꾸는 것은 코드 수정이 필요합니다.
2. `use_graph`가 기본 `True`이고 CUDA 그래프 캡처를 씁니다(`spring_mass_warp.py` 768–779행). 끄면 디버그 모드가 되어 느려집니다.
3. 외형 단계와 재생 도구는 `diff-gaussian-rasterization`(CUDA 확장)을 씁니다. 렌더링 평가(PSNR, SSIM, LPIPS)를 하려면 이 래스터라이저가 필요합니다.
4. 데이터 처리 파이프라인은 Grounded-SAM2, TRELLIS, SuperGlue, CoTracker3, SDXL, RealSense 설치를 요구합니다(README "Setup"). 모두 GPU 전제입니다.
5. 학습된 결과(`experiments.zip`, `experiments_optimization.zip`)와 데이터(`data.zip`)는 Hugging Face에 있지만 받지 않았습니다.

**가능한 범위(조건부):** 공개된 결과 파일로 **재시뮬레이션과 평가(`evaluate_chamfer.py`, `evaluate_track.py`)만** CPU에서 돌리는 것. 여기에는 (a) 데이터와 결과 파일 다운로드, (b) Warp 장치 설정 변경, (c) pytorch3d CPU 설치가 필요합니다. 2D 지표는 래스터라이저 포팅이 필요해 이 경로에 넣지 않습니다.

## 5. 저자에게 물을 질문

1. 논문 표 1의 CD 단위와 제곱 여부 (D1).
2. 가속도 정규화 항은 1단계에만 쓰는지, 학습 결과에 얼마나 영향을 주는지 (D2, D3).
3. 천과 상자에 형상 사전을 쓰지 않은 이유, 그리고 그 9개를 따로 보고할 수 있는지 (D6).
4. Spring-Gaus 실패 사례 수 (C4).
5. 논문 결과를 낸 커밋 해시 (D23).
