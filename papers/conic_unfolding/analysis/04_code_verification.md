# 04. 공식 코드 대조와 구현 가능성

## 0. 대상과 방법

- **저장소:** https://github.com/alexoshin/deep-conic-optimization, 커밋 `cf16834997c3b7d89e877df8cbb1f4edc1238fb9`(2026-10-12, "Initial commit", 이 커밋 하나뿐). 얕은 복제를 `repo/`에 두고 **정적으로만** 읽었습니다. 실행, import, 설치를 하지 않았고 테스트도 돌리지 않았습니다. 코드를 다시 받는 방법은 `git clone https://github.com/alexoshin/deep-conic-optimization repo && git -C repo checkout cf16834997c3b7d89e877df8cbb1f4edc1238fb9`입니다.
- **범위:** 파이썬 파일 39개 약 6960줄과 설정 파일 21개(`dco/` 패키지, `configs/`, `tests/`). 논문 v3의 식 (3)–(14), 표 4와 표 5, 부록 D–F와 대조했습니다.
- **번호:** 코드와 논문의 차이는 D1–D14입니다. 일치한 항목은 §2.1에 모았습니다.

## 1. 설치 요구사항과 라이선스

- **요구:** `pyproject.toml`의 `requires-python = ">=3.14"`, 의존성은 clarabel, jaxtyping, numpy, pydantic ≥ 2, pyyaml, scipy, torch, tqdm입니다. README는 "CUDA 호환 GPU"를 전제하고 모든 설정 파일이 `device: cuda`, `precision: double`입니다.
- **명령줄:** `dco-generate`(데이터 생성, Clarabel로 기준해 계산), `dco-train`(펼쳐서 학습), `dco-benchmark`(학습된 정책과 기준선의 반복 수와 시간).
- **라이선스:** `LICENSE.md`는 Apache License 2.0 전문이고 GitHub API도 Apache-2.0입니다. `pyproject.toml`에는 license 필드가 없습니다.
- **미공개:** 사전 학습 가중치, 데이터셋, 논문의 표 2, 표 3, 표 6, 그림 1–6을 만드는 스크립트는 저장소에 없습니다(D11).
- **이 맥에서의 환경(읽기만):** venv의 Python은 3.12.7이고 torch 2.15가 있으며, clarabel과 jaxtyping은 없습니다. Python 3.14는 Homebrew에 따로 있으나 torch를 확인하지 않았습니다. Julia는 없습니다.

## 2. 논문 ↔ 코드 대조

### 2.1 일치한 것

| 항목 | 논문 | 코드 |
|---|---|---|
| ADMM 갱신식 (3a)–(3e) | 식 3 | `dco/functional.py`의 `reduced_rhs`, `recover_z_tilde`, `second_block`, `dual_update`. z̃ = b − Ax̃(간접 방식)는 (3b)와 대수적으로 같음 |
| 환원된 선형계 (4) | M = P + σI + AᵀD(ρ)A | `dco/linsys.py` `ReducedSystemOperator.matvec`: `P x + σ x + Aᵀ(ρ ⊙ (A x))` |
| 행렬 없는 수반 규칙 (5)–(6) | ∇ρL = (Aλ̃) ⊙ (Ax̃), ∇ξL = −λ̃ | `dco/linsys.py` `ImplicitCGLayer.backward`(243–245행): `L_rhos = (A λ) * (A x)`, `L_rhs = −λ`. 반복 간 수반 CG 웜스타트(`AdjointWarmStartCache`)는 논문에 없음(D8) |
| 안정한 PSD 역전파 (9)–(10) | Loewner 행렬, f'(0) = 0 | `dco/cones.py` `PSDProject.backward`(127–160행). 같은 고유값에서 ½(f'(λ_i) + f'(λ_j)), 아니면 나눈 차분 |
| σ = 1e-6, 간접 모드, Jacobi 전처리 | §2.2, 부록 D | `config.py`: `sigma: 1e-6`, `cg_preconditioner: jacobi`, `linsys_solve_method: indirect` |
| 정책 구조 | 부록 E: 로짓 증분 재귀, 영 초기화, ρ 입력 7개, α 입력 5개, 비분리 원뿔 평균화 | `dco/policy.py`: `n_rho_inputs = 7`, `n_alpha_inputs = 5`(202–204행), `zero_last_layer=True`, `rectify_rhos`(95행). ρ 범위는 설정 파일에서 [1e-3, 1e3] |
| 웜스타트 | 부록 E: 변수·제약별 특징 + 전역 문맥, 64×64 | `dco/warmstart.py` `RowWiseMLPWarmstart` (`N_VAR_BASE_FEATURES = 4`, `N_CON_BASE_FEATURES = 3 + 4`, `GLOBAL_CONTEXT_DIM = 4`, 영 초기화) |
| 손실 (13) | log(‖r_d‖²/detach + ‖r_p‖²/detach) 합 | `dco/loss.py`(112–125행): `metric: residuals_sq`, `normalization: initial_gap`, `transform: log`(기본값). 식 13의 ‖·‖²가 제곱 노름이므로 일치. 합의 범위는 D7 |
| 학습 설정 (표 4) | Adam, 1e-3 → 1e-4 코사인, 슈퍼배치 5, 클립 1.0, 100/20, 32×32, 64×64, 초기 (100, 0.1, 0.1, 0.1), α 1.6 | `configs/training/*.yaml`와 같음. 검증 손실이 가장 작은 체크포인트를 고르는 것(`best_metric: loss`)도 같음 |
| 클래스별 K와 에폭 (표 5) | 400/30, 300/60, 100/50, 100/100, 300/60, 150/60, 150/60 | `train_iters`, `n_epochs`가 모두 같음 |
| 데이터 분할 | 훈련 100, 검증 20, 시험 100은 독립 시드 | `train.py`: 정렬된 파일에서 앞 100개, 뒤 20개. 시험 설정은 `seed: 1`, 훈련은 `seed: 0` |
| 문제 생성기 (부록 F) | Lyapunov, Max Cut, Lovász ϑ, KF, MPC의 분포와 매개변수 | `dco/problems/*.py`가 같음(Lyapunov의 슈어형 대각 U(0.1, 2.0) 부호 반전, 진동형 ζ ∈ [0.05, 0.5], ω ∈ [0.5, 5], KF 이상치 확률 0.2와 σ = 20, MPC의 γ, γ_term, 상태·입력 상자, 쌍대 노름 150 이상 거절) |
| 표 5의 크기 | 변수, 제약, nnz | 생성기에서 읽은 구조(Lovász nnz(A) = 5150 + E, KF 변수 N(n_x + n_w + n_y + 4) + n_x)가 표 5와 같음 (`03` V7) |

### 2.2 차이 (D1–D14)

| 번호 | 항목 | 논문 | 코드 (파일:위치) | 영향 |
|---|---|---|---|---|
| D1 | "튜닝된 상수" 기준선 | 표 1: 상수는 "best-tuned fixed values of ρ and α" (§3.2) | 모든 `configs/training/*.yaml`의 `baseline_policy_kwargs`가 같은 기본값(ρ_eq 100, 나머지 0.1, α 1.6)이고, `dco-benchmark`의 `constant` 팔은 이 기본값입니다(`cli/benchmark.py` `build_arms`). 클래스별로 튜닝한 값과 격자는 저장소에 없음 | 표 1의 상수 열을 설정만으로 만들 수 없음. KF에서 상수와 적응형 열이 같은 값이라는 점(`03` §2)은 이 열이 기본값 설정일 가능성과 맞음 |
| D2 | 실패한 풀이의 처리 | 표 1은 중앙값 [IQR]과 성공 수만 적음. 부록 J는 실패에 예산 상한을 부과한다고 적음 | `cli/benchmark.py` `charged`(128–132행)는 실패한 풀이에 `max_iter`(10000)를 부과합니다. 그러면 적응형 Lyapunov(47/100), DI(49/100)의 중앙값은 10000이어야 하는데 표 1은 788, 684 | 표 1의 두 칸은 성공한 인스턴스만의 중앙값으로 읽힘(`03` V2). 도구의 요약(`summary.json`)과 표 1이 다른 계산 |
| D3 | 벤치마크 프로토콜의 기본값 | 허용오차 1e-3, 10000 반복 또는 60 s, 추론 CG 절대 허용오차 1e-4 (§3.2, 부록 D) | `dco-benchmark`는 `--eps-abs 1e-3`, `--max-iter 10000`이 기본이지만 `--time-limit`은 없음, CG는 설정 파일 값을 그대로 씀(`config.py` 100–101행: `cg_eps_abs 0.0`, `cg_eps_rel 1e-8`) | 논문 프로토콜로 재현하려면 `--time-limit 60 --cg-eps-abs 1e-4 --cg-eps-rel 0`을 명시해야 함. README 예시에는 없음 |
| D4 | "학습" 팔의 구성 | 표 1 캡션: "Learned uses our proposed MLP policies". 부록 I: "정책과 웜스타트 네트워크(표 1)". 그림 5: 정책 단독이 웜스타트 포함보다 대체로 낫거나 같음 | 모든 훈련 설정이 `initialization: init.ws_row_mlp`이고 정책과 웜스타트를 함께 Adam으로 학습(`train.py`). 정책 단독 설정은 없음(`init.default`로 바꾸면 가능) | 표 1의 "Learned"는 정책 + 웜스타트. 정책 단독 열과 웜스타트 단독 열은 표가 아니라 그림 5(수치 없음)에만 있음 |
| D5 | DI 시험 집합의 σ₀ | 부록 F.1: σ₀ = 0.01은 훈련·검증, 시험은 σ₀ ∼ U(0.005, 0.02) | `configs/datasets/test/cs_double_integrator_N30_test.yaml`이 `sigma_0: 0.01`로 고정(고정하면 생성기의 기본 U(0.005, 0.02)가 꺼짐, `covariance_steering.py` 787–797행) | 저장소 설정으로 만든 DI 시험 인스턴스는 훈련과 같은 σ₀. 논문 서술의 시험 분포와 다름. 표 1의 DI 행이 어느 쪽으로 계산되었는지는 알 수 없음 |
| D6 | Lyapunov 데이터의 평형화 | 부록 F.2에 언급 없음. 부록 G는 COSMO.jl의 Ruiz 스케일링을 적음 | `configs/datasets/*/lyapunov_n40*.yaml`만 `equilibrate: true`(Ruiz 10회, `cli/generate.py` 137–138행). 다른 클래스는 평형화 없음 | 절대 허용오차 1e-3이 Lyapunov에서는 평형화된 문제, 다른 클래스에서는 원 스케일 잔차에 적용됨(클래스 간 허용오차의 상대 의미가 다름) |
| D7 | 손실의 합 범위 | 식 13: k = 1..K | `dco/loss.py`: 잔차를 k = 0..K(K + 1개)로 만들고 `losses.sum(dim=-1)`. 웜스타트가 없으면 k = 0 항은 상수(로그 2). 웜스타트가 있으면 k = 0 항(초기 잔차/초기 잔차를 분리한 분모)이 웜스타트 출력에 대한 기울기를 가짐 | 작음. 웜스타트가 초기 잔차를 줄이는 방향의 기울기가 하나 더 있음(분모는 detach) |
| D8 | 수반 CG의 설정과 "정확한" 기울기 | §2.3: 결과 기울기는 "정확". 부록 D: 학습 중 상대 CG 허용오차 1e-8 | 수반 풀이는 상대 1e-6(`config.py` 105행), 반복 사이 수반 웜스타트 사용(`adjoint_warm_start: true`), 순방향 CG의 초기값은 이전 x̃이고 x̃의 기울기는 0(`linsys.py` 243행) | 기울기는 CG 허용오차까지 정확(토이에서 수반 1e-6, 순방향 1e-8일 때 상대 오차 2e-7, `03` §3.1). 수반 웜스타트와 허용오차는 논문에 없음. 저장소 테스트는 웜스타트가 풀어낸 기울기를 바꾸지 않음을 확인(`test_cg_layer.py`) |
| D9 | 식 13의 y 부호 | 식 3과 식 13 모두 y를 쓰는데 식 3 고정점은 Px + q − Aᵀy = 0, 식 13은 +Aᵀy | 솔버 내부의 y는 `dual_res = P x + q − Aᵀ y`(`functional.py` 129행)이고, 결과로 돌려줄 때 부호를 바꾼 뒤 손실은 `P x + q + Aᵀ y_hist`(`loss.py` 113행) | 표기 문제. 부호를 바꾼 y를 쓰면 일치 |
| D10 | SCP 안의 학습 solver | 그림 1: 20 시나리오, SCP 반복 20회, 부분문제 20 s, 400 s | `problems/covariance_steering.py`의 `solve_subproblem`의 "cosmo" 경로는 `Solver(...policy=cfg.policy or AdaptiveRhoOSQP())`로 만들며 웜스타트를 받지 않음, `solve()` 인자는 설정에서 받는 `solver_kwargs`(기본 직접 선형 풀이). `SCP_Config.max_iters = 15`(222행). 학습된 체크포인트를 SCP 안에서 돌리는 스크립트와 20 시나리오 설정은 없음 | 그림 1과 부록 H를 저장소로 만들 수 없음. 데이터 생성용 SCP(Clarabel로 풀어 10개 부분문제를 저장)만 있음 |
| D11 | 없는 실험 스크립트 | 그림 2, 3, 4, 5, 6, 표 2, 표 3, 표 6, 부록 H | `dco/cli`는 `generate`, `train`, `benchmark` 셋뿐. OOD 크기(n = 250, 500, N = 250, 500, 1000) 설정, COSMO.jl(juliacall)과 Clarabel 시간 측정, 메모리 피크 측정, 배치 solver의 시험(`super_batch.py`는 있음) 스크립트가 없음. 테스트는 기울기 규칙(`test_cg_layer.py`, `test_psd_projection.py`)과 solver 동치를 다룸 | 이 저장소로 재현 가능한 것은 표 1 계열(설정 파일, `benchmark`)뿐. 연산자는 A, Aᵀ, (A∘A)ᵀ CSR을 모두 보관(`linsys.py` `ReducedSystemOperator`)하므로 50k에서의 메모리 측정(`03` V8)은 이 코드가 아닌 다른 구현일 수 있음 |
| D12 | 환경 요구 | 논문: RTX 4090 float64, i9-13900K | `requires-python >= 3.14`, README와 모든 설정이 CUDA(`device: cuda`) | CUDA 없는 기기에서는 `device`를 바꿔야 함. float64이므로 MPS는 쓸 수 없음 |
| D13 | ρ 범위가 세 곳에 있음 | [1e-3, 1e3] (표 4) | 설정 파일 [1e-3, 1e3], 정책 클래스 기본값 [1e-4, 1e4](`policy.py` 185–186행), 솔버 전역 클램프 [1e-6, 1e6](`functional.py` 16–17행) | 설정 파일 값이 쓰이므로 학습에는 영향 없음. 기준선(적응형)은 전역 클램프의 범위까지 움직일 수 있음 |
| D14 | 슈퍼배치의 전역 게이트와 수렴 판정 | 부록 I: 배치 수렴은 가장 느린 인스턴스로 정해짐 | `rho_change_gate`는 배치 전체에서 하나라도 5배 넘게 바뀌면 전체 갱신(`functional.py` 155–157행), 수렴도 `torch.all(converged)`(`solver.py` 387행). 훈련·검증 중 기준선 손실(`summary.json`의 `baseline_*`)은 5개 슈퍼배치에서 계산 | 표 1은 인스턴스별 풀이(`benchmark.py`)라 영향 없음. 표 2의 배치 반복 수가 최악 인스턴스로 정해진다는 서술과 맞음 |

### 2.3 코드 구조에서 확인한 것

- **선형계는 한 번도 조밀하게 만들지 않습니다.** 간접 모드에서 `ReducedSystemOperator.matvec`는 `A_T @ (rhos * (A @ x))`만 계산하고, Jacobi 전처리의 대각은 `diag(P + σI) + (A∘A)ᵀρ`로 구합니다(`linsys.py` `diagonal`). 구조 검출(`structure.py`)로 P가 0이거나 대각이면 곱셈을 건너뜁니다(논문에 없는 최적화).
- **직접 모드는 조밀 (n + m)² KKT를 만들어 LU를 합니다**(`solver.py` `construct_coefficient_matrix`). 이것이 그림 2의 "Direct"에 해당하는 경로이고 이 저장소의 기본 `solve()` 인자는 직접 모드입니다(`linsys_solve_method="direct"`). 훈련과 벤치마크는 설정으로 간접 모드를 지정합니다.
- **테스트가 있습니다.** `tests/test_cg_layer.py`는 암묵 기울기를 조밀 autograd, 유한차분, 펼침 autograd와 비교하고, `tests/test_psd_projection.py`는 DK를 autograd(분리된 고유값), 유한차분(간격 1e-2, 1e-6, 1e-10), 모든 고유값이 같은 경우와 비교합니다. 이 분석은 테스트를 실행하지 않았습니다.

## 3. 논문 서술의 코드 쪽 확인 요약

- "ρ와 α 모두 정책이 낸다, 비분리 원뿔은 블록 평균": 코드와 같음(`rectify_rhos`).
- "초기 예측이 바닐라 COSMO와 같다": 영 초기화 + 학습 가능한 초기 로짓으로 코드가 보장하고, `tests/test_solver.py`가 확인(`test_untrained_policy_reproduces_fixed_hyperparameter_cosmo`).
- "각 정책은 10 GPU시간 미만": 로그와 설정에서 시간 정보 없음(코드는 `summary.json`에 `training_time_s`만 기록).
- "COSMO.jl 대비 표 6": 저장소에는 COSMO.jl 호출 코드가 없고 기준선 두 가지(상수, 적응형)는 PyTorch 구현입니다.

## 4. 구현 가능성 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**논문 수치의 전체 재현은 이 맥에서 불가능합니다. 방법의 핵심(두 역전파 규칙과 펼침 학습)은 작은 크기에서 구현되어 있고 확인을 끝냈습니다. 공식 코드로 표 1의 일부 클래스를 CPU에서 학습·벤치하는 것은 조건부로 가능합니다.**

| 수준 | 판정 | 근거 |
|---|---|---|
| 두 역전파 규칙, 정책 학습의 축소판 | **가능, 완료** | `analysis/toy_dk_cg_checks.py`(수 초), `analysis/toy_du_maxcut.py`(약 40초). 둘 다 새로 짠 torch CPU float64 코드 |
| 공식 테스트 (`pytest`) 실행 | 조건부 | clarabel, jaxtyping 설치와 Python ≥ 3.14 환경(torch 포함)이 필요. 설치 전 승인 필요. CPU 설정으로 도는 작은 테스트 |
| 공식 코드로 표 1 계열 재현: Max Cut, Lyapunov, Robust KF, Robust MPC | 조건부 (CPU, 시간 미측정) | 설정 파일의 `device`를 바꿔야 함. 데이터 생성(Clarabel로 기준해 계산), 학습(클래스당 Adam 600–2000 갱신, K 100–400), 벤치마크 모두 CPU float64. 필요 설치와 코드 실행 승인 |
| 공식 코드로 Lovász ϑ, DI, Unicycle | 조건부, 더 무거움 | 변수 5050(Lovász, 100×100 PSD), 3904, 2943이고 PSD 고유분해 비용 때문에 CPU에서 훨씬 느릴 것(측정 안 함). 논문은 RTX 4090에서 Unicycle 에폭당 29 s |
| 표 2 (GPU 배치, 순차 COSMO.jl 대비 14×) | **불가** | 24 GB CUDA GPU와 Julia/COSMO.jl이 필요. MPS는 float64를 지원하지 않음 |
| 그림 2 (n = 50k 메모리) | **불가** (소규모는 가능) | 50k는 24 GB GPU 기준. CPU에서 n ≤ 10k로 비율(밀집 대 행렬 없음)만 확인 가능 |
| 그림 1 (SCP 20 시나리오, 학습 solver 포함) | 조건부, 코드 일부만 있음 | SCP 틀과 Clarabel 연동은 있으나 체크포인트를 SCP 안에서 돌리는 코드(D10)가 없음. 새로 구현 필요 |
| 표 3 (OOD) | 조건부 | OOD 크기 설정을 직접 만들어야 함(D11). 가장 큰 Max Cut(n = 500, 변수 125,250)은 CPU에서 느릴 것 |
| 표 6의 COSMO.jl, Clarabel 시간 | 불가 / 조건부 | Julia 없음(불가). Clarabel은 파이썬 패키지 설치 필요(조건부) |

- **노력 추정 (측정 전):** 토이는 끝났습니다. 공식 코드로 CPU 소형 클래스 네 개의 표 1을 재현하려면 설치와 설정 정리 약 1일, 데이터 생성·학습·벤치 약 1–3일(추정, 측정 안 함). 코드를 읽지 않고 처음부터 PyTorch CPU로 행렬 없는 층과 DK 층, 정책 학습을 구현하는 경우는 약 3–5일(추정).
- **GPU가 필요한 부분:** 표 2, 그림 2의 큰 크기, Unicycle과 Lovász의 빠른 학습입니다. 이 저장소 규칙상 GPU가 필요한 재현은 하지 않습니다.
- **로봇이 필요한 부분:** 없음. 이 논문은 solver 논문이고 실로봇 실험이 없습니다(하드웨어 시험은 한계 문단에서 미래 과제).

## 5. 재실행

```bash
git clone https://github.com/alexoshin/deep-conic-optimization repo
git -C repo checkout cf16834997c3b7d89e877df8cbb1f4edc1238fb9
python3 analysis/recompute.py
python3 analysis/toy_dk_cg_checks.py
python3 analysis/toy_du_maxcut.py
```

`repo/`를 읽는 스크립트는 없습니다. 위 clone은 문서의 파일·행 번호를 확인할 때만 쓰며, 실행하거나 설치하지 않습니다.
