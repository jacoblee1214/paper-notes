# 04. 공식 코드와 논문 대조, 구현 가능성

## 1. 대상과 방법

- 저장소: `erickun0125/constraint-decomposed-impedance-control` (MIT), 커밋 `d1f99f98db2c36f8673278ac0c60a871a89792f7` (2026-10-06, 단일 커밋), 얕은 복제 후 **정적으로만** 읽었습니다. 코드를 실행·import·설치하지 않았고, 체크포인트(각 약 355 MB, 5개)와 데이터는 받지 않았습니다.
- 구성: `con_dec_imp/`(방법, numpy·torch), `con_dec_imp_sim/`(Isaac Lab 환경), `scripts/`(수집·학습·평가·점수·표), `configs/`, `tests/`(약 2,000줄 단위 테스트), `examples/`(앙상블 4개와 노트북).
- 이 저장소는 **시뮬레이션 실험만** 구현합니다(README 첫 문단: "its simulation experiments"). 실로봇(FR3, ATI Axia80, 원격조작, DDPM 100단계 설정)은 들어 있지 않습니다.
- 대조한 것: Algorithm 1, 식 (8)–(10), (34), (54), 표 5–8의 하이퍼파라미터, 평가 지표(식 55–56), 표 1·9·10을 만드는 `make_tables.py`.

## 2. 일치하는 것

| 항목 | 논문 | 코드 |
|---|---|---|
| Body twist 차분 | 식 Alg. 1 line 3: 중앙차분, 양 끝 한쪽 차분 | `twists.py` `body_twists`: 같음, `log(T_{t-1}⁻¹T_{t+1})/(2dt)` |
| α와 가중 PCA | α = clip(‖u_v‖/‖u_ω‖, 0.03, 0.5), 비중심 SVD, λ_k = σ̃²/M | `subspace.py`: 같음 (norm에 ε = 1e-6 추가) |
| 차원 규칙 | m = #{√λ_k > max(κ√λ_6, v_abs)}, 시뮬 v_abs = 5 mm/s, κ = 4 | `settings.py` `ESTIMATION`: 같음 (0.005 m/s, 4.0) |
| 사영과 게인 | 식 (8), (10), (54) | `gains.py`: 같음. `modal_gains`가 k_⊥ = r k₀, d = 2ζ√k |
| 게인 값 | 표 6: k₀ = 250, r = 0.1, ζ = 0.766, m̄ = 0.586, Ī = 0.00622 | `GAINS`: ζ = 0.7655(= √m̄), 나머지 같음. 메트릭은 m̄로 정규화한 blkdiag(ℓ²I₃, I₃) (표 6 캡션의 "정규화 메트릭"과 일치) |
| 정책 구조 | 표 7: 예측/실행 24/16, 10 Hz, 관측 이력 1, ResNet-18, AdamW β = (0.95, 0.999), wd 1e-6, lr 1e-4 cosine, 배치 64, 100/10 DDIM | `configs/policy/*.json`, `diffusion_policy.py`: 같음 |
| 체크포인트 | 에폭 250/150/100/150 (단일), 400 (다중), 500 스텝/에폭 | README, `train_policy.py` docstring: 같음 |
| 제어 주기·게인 전환 | 150 Hz, 50 제어 스텝 (0.33 s) 동안 등방 → 분해 게인으로 전환 | `CONTROL`: 150 Hz, `gain_transition_steps` = 50 |
| 시연 | 물체당 400개, 배치 범위, 그리퍼 시작 (표 8) | `base.py` `OBJECT_POSE_RANGES` (±0.15, ±0.20, ±0.10 m, ±15°), `configs/collect/*.yaml` num_demos 400 |
| 지표 | 식 (55)–(56): ICF = ∫‖f_⊥‖dt, PCF = Q95, 블록별 사영 | `metrics.py`: 같음 (직사각 합, `np.percentile` 선형 보간) |
| 공통 성공 집합 평균 | 표 10 캡션: 세 제어기가 모두 성공한 시도의 평균 | `make_tables.py`: 같음 (추가 조건은 D7) |
| 실행 시간 측정 | 70.1 ms 중앙값 (§5.3) | `evaluate.py`가 `perf_counter`로 `Estimate.time`을 기록 (측정치는 실행 안 해 확인 못 함) |

## 3. 차이 D1–D17

심각도는 "논문 문면만 따라 구현하면 결과가 달라질 수 있는가"의 관점입니다.

| 번호 | 항목 | 논문 | 코드 | 영향 |
|---|---|---|---|---|
| D1 | twist 행렬에 쓰는 웨이포인트 | Alg. 1: 궤적 N개 × H개 전부, W ∈ R^{6×NH} | `evaluate.py`가 `grasped` 마스크(그리퍼 명령 > 0.5인 웨이포인트)만 W에 넣음. 선택된 이웃 사이의 실제 시간 간격으로 차분하고 λ를 M = 선택된 개수로 나눔(`estimator.py`, `twists.py` `selected_body_twists`) | 논문에 없는 전처리. 예시 앙상블은 모든 웨이포인트가 닫힘 명령이라 영향 없음. 실제 평가에서 얼마나 걸러지는지는 코드로 알 수 없음. 접근 구간이 섞이면 SE와 차원에 영향 가능 |
| D2 | 추정 차원 m = 0일 때 | 식 (10)을 문면대로 쓰면 U_∥ 비어 P_∥ = 0 → K_v = r k₀ Λ̄ (전 방향 저강성) | `decomposed_gains`와 `evaluate.py`: m = 0이면 Iso 게인 유지. 선택된 웨이포인트가 6개 미만이어도 추정하지 않고 Iso 유지 | 논문 평가는 모든 시도에서 차원이 맞다고 보고하므로(부록 H.1) 표 1에는 영향 없음. 배포 시 안전 동작에는 중요한 정의 차이 |
| D3 | 실행 궤적과 앙상블의 관계 | Alg. 1은 π에서 N개 샘플링. 어느 궤적을 실행하는지는 본문에 서술 없음 (그림 5는 실로봇의 "tracking sample"을 표시) | 실행은 영 노이즈 DDIM 샘플(`predict`), 앙상블은 가우시안 노이즈 η = 0 DDIM(`sample_ensemble`)이고 서로 다른 샘플 | Iso와 Ours가 같은 실행 궤적을 쓰므로 비교는 공정. 다만 실행 궤적이 앙상블의 한 원소가 아니라는 점은 논문에 없음 |
| D4 | 게인 설치 시점 | "grasp closure에서 한 번 추정, 고정"(§4, §5.1) | 모든 제어기가 Iso로 시작. 닫힘 명령 후 손가락이 닫히거나 10 제어 스텝이 지나면 닫힌 것으로 보고 15 스텝(손목 카메라 한 프레임) 뒤 grasp closure로 정의 → 이때 샘플링·설치. 분해 게인은 손가락이 닫혀 있는 동안만 작용하고 해제하면 Iso로 복귀 (`evaluation.py`, `actions.py`) | 타임아웃, 지연, 해제 처리는 논문에 없음. 재그립 시 재추정 |
| D5 | 평가 구간과 성공 기준 | "common articulation range"(수치 없음) | 구간은 grasp closure 다음 단계부터 성공 종료 또는 2000 스텝(13.3 s) 타임아웃까지(종료 스텝은 기록 안 함). 성공 문턱: revolute 관절 > 0.95π − 0.32 rad (152.7°), cylindrical 회전 > 145° 이고 이동 > 0.26 m, planar y < −0.12 m·z > 0.12 m·회전 > π/2 − 0.32 rad, universal 두 관절 > 65° (`tasks/*.py`) | 표 1 SR과 하중의 구간 정의에 필요한 수치가 논문에 없음 |
| D6 | 시연의 목표 회전 | 부록 H.3: revolute 힌지 회전 π, planar "반 바퀴(half-turn)" | `configs/collect`: revolute `open_target` π × `open_ratio` 0.95 = 0.95π, planar 목표 회전 1.5708 rad (π/2) | 문면과 다른 수치. planar는 "half-turn"이 π라면 코드 값은 그 절반 |
| D7 | 평균에 들어가는 시도 | "세 제어기가 모두 성공한 시도" | 추가로 세 제어기 모두 grasp closure가 검출돼야 함. 성공했지만 closure가 검출되지 않은 시도는 평균에서 제외하고 stderr 경고 (`make_tables.py`) | 소수 시도의 제외. 표 9의 n_c가 min(SR)보다 작은 칸(다중 cylindrical 95 대 96)과 관련 가능 |
| D8 | SE 정의의 경계 | "full-subspace maximum principal angle" | 추정과 기준의 차원이 다르면 SE = 90°로 처리 (`metrics.py`). 비교 메트릭의 평가 길이 α\*는 코드에만 있음 (revolute 0.243, cylindrical 0.153, planar 0.135, universal 0.065 m, `reference.py`) | 평균 SE 계산에 차원 오류 시도가 들어가면 90°가 섞이지만 논문 결과는 차원 오류가 없다고 보고하므로 영향 없음 |
| D9 | 표 7에 없는 학습 설정 | 표 7 | UNet down_dims 256/512/1024, kernel 5, groups 8, EMA power 0.75, warm-up 500 스텝, cosine num_cycles 0.5, min-max 행동 정규화, 이미지 crop 212×212, 공간 softmax 특징 차원 64, `clip_sample` true, squaredcos_cap_v2 | 정책을 처음부터 다시 학습하려면 필요. 224×224 관측이 학습 시 212 crop |
| D10 | 시연 수집 설정 | 스크립트 전문가의 목표 자세만 서술 | 수집 때는 별도 임피던스 `EXPERT_STIFFNESS` (회전 12, 병진 250) 와 `EXPERT_DAMPING` (0.24, 20)을 쓰고, 성공 문턱도 평가보다 엄격(revolute 0.95π − 0.16, 평가는 −0.32) | 시연의 노이즈 성격에 영향 가능 (논문 부록 H.3의 시연 설명에 없음) |
| D11 | 데이터 분할과 검증 | 표 7: 검증 손실 최소 이후 첫 체크포인트 | demo_0–359 학습, demo_360–399 검증. robomimic이 검증 배치에서도 학습률 스케줄을 진행하므로 검증 주기(10/50/200 에폭)가 스케줄에 포함됨(`train_policy.py` docstring) | 체크포인트를 같은 값으로 재현하려면 검증 주기까지 같아야 함. 논문 본문에는 없음 |
| D12 | 평가의 결정성 | "100회 시도" | 배치 리셋 시드 고정(1296387670)과 시도별 ensemble 시드(seed + 1 + index)로 초기 배치는 재현되지만 GPU 렌더링이 비결정적이라 "통계적으로 일치, 숫자 단위 일치는 아님"(README 명시) | 표 1의 소수 둘째 자리 재현은 원리상 불가 |
| D13 | 지연 시간 측정 | RTX 5070 Ti, 120회 실행의 중앙값 70.1 ms, 최대 76.5 ms | `evaluate.py`가 시도마다 `perf_counter` 시간을 `Estimate.time`에 기록. 배치 1 샘플링. 120회(10 관측 × 3 시드)를 만드는 스크립트는 없음 | 측정 구조는 있으나 표 11–13과 같은 별도 스크립트가 없음 |
| D14 | §5.3 분석 코드 | 표 11(N 스윕), 12(DTW 다양성), 13(잡음) | `scripts/`에 해당 실험 스크립트가 없음. 그림 4와 5, 표 11–13의 생성 코드 없음 | 부록 I의 앙상블 분석은 코드로 재현할 수 없음 (입력이 되는 N = 128 앙상블 4개만 `examples/data`에 있음) |
| D15 | 실로봇 | 표 2, 표 14, 그림 4, 5 | 코드 없음 (FR3 제어기, 힘 센서 처리, 원격조작, DDPM 100단계 설정, 물체별 κ_rel = 10/4) | 실로봇 결과는 저장소에서 검증할 수 없음 |
| D16 | Λ̄의 구성 | 식 (20)–(21): 공칭 Λ_b에서 trace 평균 | `inertia_metric(rotational_inertia, mass)`로 값을 직접 받음. 식 (21)의 계산 코드는 없음 | 값(0.00622, 0.586)은 표 6과 같고 유도 코드는 없음 |
| D17 | 단위 표기 | 표 6: k는 s⁻², d는 s⁻¹ | 정규화 메트릭 때문에 수치는 N/m, N·s/m이고, 정규화 메트릭에서만 s⁻² 단위로 읽힘 | 표기 문제. 물리 단위로 쓰려면 메트릭 정규화를 되돌려 읽어야 함 |

## 4. 구현할 수 있는지 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**판정: 방법 자체(앙상블 → 부분공간 → 게인)는 이 맥 CPU에서 이미 구현·실행해 보았고(`analysis/toy_ensemble_estimator.py`, `03` §8), 시뮬 실험 전체와 실로봇은 이 맥에서 재현할 수 없습니다.**

| 수준 | 판정 | 이유 |
|---|---|---|
| 방법 (Algorithm 1, 식 8–10, 54) | 가능, 완료 | numpy 수십 줄. 공식 예시 앙상블 4개로 실행(약 3초). 공식 `con_dec_imp`는 numpy·torch만 필요하나 실행은 하지 않음 |
| 공식 Isaac Lab 시뮬 평가 (표 1, 10) | 불가 | Isaac Sim 5.1과 Isaac Lab은 Linux·Windows와 NVIDIA GPU 전제 (논문은 RTX 5070 Ti, 평가 한 번 약 1분·7.5 GB, 전체 30분 미만). macOS 미지원 |
| 정책 학습 (단일 정책 예: revolute 에폭 250 × 500 = 125k 스텝, 배치 64) | 불가 | ResNet-18 + UNet Diffusion Policy를 이미지로 학습. 소요 시간은 측정하지 않았고 이 맥 CPU·MPS로는 비현실적이라 판단함. robomimic·diffusers 새 설치도 필요 |
| 공식 체크포인트 추론 | 조건부 | 체크포인트 약 355 MB 다운로드 승인, 새 가상환경, 공식 코드 실행 승인, 시뮬 관측(손목 카메라 이미지)이 필요해 시뮬 없이는 입력이 없음. 예시 앙상블은 이미 출력이라 추론 없이 사용 |
| 직접 짠 MuJoCo + torch 소형 재구현 (유니버설·회전문 한 개, 상태 기반 소형 Diffusion Policy 또는 규칙 기반 샘플러) | 가능 (시작 전) | mujoco 4.0, torch 2.15(MPS) 설치됨. 논문 §5.3 (i)이 상태 기반 정책 학습을 썼음. 노력은 2–3주(추정, 측정 전) |
| 실로봇 (FR3 + 힘센서, 표 2) | 불가 | 로봇, 센서, 시연 수집 필요 |
| 표 11–13 (N 스윕, 다양성, 잡음) 부분 재현 | 가능 (소형) | 정책 학습 없이 예시 앙상블 4개로 부분 확인(완료, 다양성·잡음은 미완) |

노력 추정(측정 전 추정): 방법과 예시 앙상블 검산은 1일(완료), MuJoCo 단일 물체 재구현은 2–3주, 시뮬 전체 재현은 GPU 워크스테이션이 있다면 공식 스크립트로 반나절(체크포인트 다운로드 포함)입니다.

## 5. 라이선스

- 코드: MIT(`LICENSE`). 그리퍼 모델(`flying_gripper.usda`, 6.8 MB)에는 `franka_description` 0.7.1의 Franka hand 메시와 관성 값이 포함되어 있고 Apache-2.0입니다(`NOTICE`, `licenses/`).
- 체크포인트(가중치) 라이선스는 저장소 라이선스와 같다고 README가 명시하지는 않고 "체크포인트는 `download_checkpoints.py`가 받는다"고만 적습니다(별도 표기 없음).
- 프로젝트 페이지 저장소는 라이선스 표기가 없습니다.
- 논문은 CC BY 4.0입니다(`README.md`와 `01` §0).
