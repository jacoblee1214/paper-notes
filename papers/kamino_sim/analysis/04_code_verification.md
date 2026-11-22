# 04. 공식 코드 대조와 구현 가능성

## 0. 대상과 방법

- 저장소: `newton-physics/newton` (공식 구현이 들어 있는 NVIDIA Newton, Apache-2.0). 커밋 `f39d8199f18d5117f9dd8073c1eb71c9a196f020` (2026-11-21, "Differentiable ANYmal-D standing backflip ..."). 별도 Kamino 저장소는 없고 `newton/_src/solvers/kamino`(파일 169개, Python 약 10.7만 줄)와 `newton/examples/kamino`, `docs/`에 있습니다. 저장소 전체는 크므로(API 크기 약 225 MB) `git clone --depth 1 --filter=blob:none --sparse`로 이 세 경로만 받았습니다(`repo/`, 20 MB). 이후 `git`으로 다른 경로를 내려받지 않았습니다. 논문은 v2(2026-10-20) 기준입니다. **코드 HEAD는 논문보다 약 한 달 뒤입니다.** 솔버 디렉터리에 214개 커밋이 쌓여 있어 코드가 논문 시점과 다를 수 있다는 점을 아래 D 항목에서 구분해 적었습니다.
- **정적 읽기만 했습니다.** 코드를 실행, import, 설치하지 않았습니다. `import warp`, `import newton`은 이 환경에서 모두 실패합니다(미설치). 자산(USD, ONNX 정책)은 받지 않았습니다.
- 읽은 파일: `kamino/README.md`, `config.py`(기본값 전체), `solver_kamino.py`(Config 부분), `_src/solvers/padmm/{kernels,solver,types}.py`(알고리즘 1 부분), `_src/solvers/metrics.py`, `_src/dynamics/dual.py`(전처리), `examples/rl/example_rl_drlegs.py`, `newton/examples/kamino/example_kamino_robot_dr_legs.py`, `example_kamino_basic_fourbar.py`, `utils/hoberman.py`, `docs/solvers/kamino.rst`. 나머지(선형대수 분해, 충돌 감지, SCC, LOX)는 훑어보기만 했습니다.
- 판정: **일치** = 논문과 코드가 같음, **차이** = 다르거나 논문에 없는 기능, **미확인** = 저장소에서 찾지 못함.

## 1. 라이선스와 설치 요구사항

- 코드: Apache-2.0(GitHub API와 파일 헤더 SPDX). 문서(`docs/`)는 CC-BY-4.0. 솔버 README는 "BETA 1"이라며 사용을 권하지 않고 외부 기여를 받지 않는다고 씁니다(D1).
- 의존성: Newton과 Warp. 솔버 문서는 개발에 "Newton 소스 설치와 최신 Warp(nightly 또는 소스)"를 요구합니다. 성능 설계는 CUDA 그래프, 타일 연산(`wp.tile_*`), 그래프 조건 노드를 전제하며 `use_graph_conditionals`를 끄면 CUDA 없이 펼친 루프를 씁니다. Warp에는 CPU 장치가 있고(`device.is_cpu` 분기가 `utils/device.py`에 있음) 코드에 CPU 경로가 완전히 막혀 있지는 않아 보이지만, 이 맥에서 실행해 보지 않아 **CPU 동작은 미확인**입니다.
- 가중치와 자산: DR Legs USD는 `newton.utils.download_asset("disneyresearch")`로 받는 자산이고, RL 재생용 ONNX 정책(`drlegs_walk.onnx`)도 그 자산 안에 있다고 코드가 가정합니다. 받지 않았습니다.

## 2. 논문 ↔ 코드 대조 (D 항목)

| 번호 | 항목 | 논문 | 코드 | 판정 | 영향 |
|---|---|---|---|---|---|
| D1 | 공개 상태와 기여 정책 | §1 (iv) "오픈소스", §4 "커뮤니티 손에 맡겨 계속 개선" | `kamino/README.md`: "`SolverKamino`는 BETA 1, 사용을 권하지 않음, 개발팀 여력 부족으로 외부 기여를 받지 않음, 더 안정한 BETA 2는 2026년 여름 예정". 문서 `docs/solvers/kamino.rst`도 experimental 표기. 이 글을 쓰는 시점(2026-11-22)에도 갱신되지 않음 | **차이** | "커뮤니티가 계속 개선"이라는 서술은 현재 정책(기여 비수용)과 맞지 않음. 외부 연구 기반으로 쓰기에는 API 안정성 보장 없음 |
| D2 | 솔버 구성 | PADMM만 서술 | `SolverKamino.Config.dynamics_solver` 선택지 `padmm`(기본), `scc`(구 DVI), `lox`, `dvi`(별칭). 문서는 SCC가 더 빠르지만 부등식을 덜 정확히 풀고 PADMM이 "더 안전하고 더 널리 검증된" 선택이라고 설명 | **차이 (논문에 없는 기능)** | 논문 처리량이 어느 솔버 기준인지는 PADMM으로 읽히나 코드에서는 SCC가 속도 쪽 경로. 처리량 개선의 일부가 PADMM 밖에서 나왔을 수 있음 |
| D3 | 기본 적분기 | v2 부록 A.1.2: 반암시적 오일러, 선택적으로 Moreau–Jean 중점. v1은 DR Legs RL에 Moreau–Jean 250 Hz | `Config.integrator = "euler"` (기본), `"moreau"` 선택 (`solver_kamino.py` L329). 문서도 PADMM 기본이 오일러 | **일치 (v2), 차이 (v1)** | v1 학습 설정(중점 적분기)은 v2와 다름. v2는 RL 적분기를 명시하지 않음 |
| D4 | 정지 기준 | 알고리즘 1, 9행: 세 잔차 rprimal, rdual, rcmpl < ε. v1은 ε = 1e-6, v2는 ε 미기재 | 기본 1e-6 (`PADMMSolverConfig` L584–608), `max_iterations = 200`(L659). 그러나 (a) **rdual의 정의가 다름**: 정지 검사의 r_d는 ADMM식 잔차 `η(x − x_p) + ρ(y − y_p)`(`kernels.py` L950) 또는 기본값 `dual_residual_mode="delassus"`의 `D(x − x_p)` 변화량이고, 부록 A.1.4의 NCP 이중 잔차(속도 위반)와 다른 양. (b) `iterations > 1`이어야 수렴 판정. (c) DR Legs와 4절 링크 예제는 허용오차를 **1e-4**로 올림(`example_kamino_robot_dr_legs.py` L142–145) | **차이** | 알고리즘 1의 "r_dual"은 같은 이름의 두 양(정지용 ADMM 잔차, 보고용 NCP 잔차)을 가리킴. 표 1의 200회와 1e-6은 기본값과 맞으나 예제 기본은 1e-4 |
| D5 | 벌칙 ρ와 관절 행 | 알고리즘 1: 모든 행에 같은 ρ와 η (D_{η,ρ} = D + (η + ρ)I), ρ 값 미기재 | `rho_0 = 1.0`, `eta = 1e-5`(L621, 627), `penalty_update_method="fixed"`. 그리고 `unpenalized_bilateral_rows = True`(기본, L774): **쌍방 관절 행에는 ρ를 적용하지 않음**(`joint_rho_scale = 0`, `kernels.py` L159–168). 행별 ρ 스케일 배열이 있음 | **차이 (논문에 없음)** | 쌍방 관절은 η 정규화까지만으로 "정확히" 풀림. 문서는 이것이 수렴을 돕지만 잉여 구속이 없어야 한다고 설명. 논문의 균일 ρ는 이 기본값과 다른 알고리즘. 토이 T1에서 ρ가 반복 수를 좌우함(`03` §7) |
| D6 | 잉여 운동학 구속 제거 | 서술 없음. §1은 "과구속 위상에서 대수적 악조건"을 문제로 듦 | `detect_redundant_kinematics = True`(기본, L741), 임계 상대 피벗 1e-5. 닫힌 루프 모델에서만 작동하며 수치적으로 잉여인 관절 행(Delassus를 특이하게 만드는 행)을 버리고(운동은 같다고 설명) 이전 스텝 집합을 재사용 | **차이 (논문에 없는 기능)** | 루프를 "최소 튜닝"으로 푼다는 주장의 상당 부분이 이 구속 제거에 의존할 수 있음. 논문 방법 절에 없어서 재구현자가 놓치기 쉬움 |
| D7 | 가속, 웜 스타트, 전처리 세부 | Nesterov 재시작 "결합 잔차가 줄지 않으면", 웜 스타트 "캐시한 λ로 초기화", 전처리 P = diag(D)^(−1/2) | (a) 재시작 조건은 `r_a < 0.999 · r_a_p`이면 지속, 아니면 재시작 (`kernels.py` L1258). 임계 0.999 (b) 웜 스타트에 **0.9 배율**(`warmstart_scale`, L716). 최소 노름 해로 수렴시키기 위한 것이라고 주석 (c) 전처리 P는 `clamp(1/√(\|d\| + ε), 1/50, 50)`로 자름 (`dual.py` L432–433) | **차이 (세부)** | 모두 합리적인 구현 세부이고 알고리즘 1의 핵심 갱신(아래 D15)은 맞음. 다만 논문 수식만으로는 재현 값이 달라질 수 있음 |
| D8 | DR Legs 시뮬레이션 설정 | §3.3: Δt = 1.25 ms에서 걸음 추종(PD). 부록 A.7: RL은 Δt = 4 ms, 데시메이션 5. 처리량: PADMM 100, CR 15 | 공개 예제(`example_kamino_robot_dr_legs.py`): PADMM의 목표 Δt = 0.01 s(L55, 프레임 50 Hz에서 서브스텝 2), SCC는 프레임/12. 애니메이션 모드는 P 이득 150(L92), 서 있기 모드는 kp = 10, kd = 2(L104–110), 아마추어 0.011, 감쇠 0.044(L96–97). 재생 예제(`example_rl_drlegs.py`): sim_dt = 0.004, 디시메이션 5, pd_kp = 15, pd_kd = 0.6, 아마추어 0.01. CR `maxiter`는 SCC 경로에서만 9(v1의 9, v2는 15) | **차이** | 공개 예제의 Δt(10 ms)와 이득은 논문의 어느 설정과도 같지 않음. 같은 로봇이 파일과 모드마다 다른 PD 이득을 씀(10/2, 150, 15/0.6) |
| D9 | 구속 안정화 기본값 | 4절 링크: α = 0.01(soft), 0.1(stiff). 다른 안정화 서술 없음 | 기본 α = β = γ = 0.01, 접촉 관통 마진 δ = 1e-6, `cull_speculative_contacts = True`(기본)(`config.py` L257–278, L494). 그리고 새 옵션(`*_recovery_time`, `contact_compliance`, `limit_compliance`)이 추가되어 컴플라이언트 접촉도 가능 | **차이 (논문 이후 기능)** | δ = 1e-6으로 투기적 접촉을 자르므로 연속 충돌 감지 없음(표 1 캡션과 맞음). 표 1 rC가 이 한계에 지배될 가능성(`03` K2). 코드에 컴플라이언스 옵션이 생겨 "컴플라이언스 없이"라는 서술은 기본 설정에 한정됨 |
| D10 | 마찰 지원 범위 | §4: 선형 3D 쿨롱만, "비틀림·구름 접촉 마찰과 **관절 정지 마찰은 아직 없음**" | README: "관절 쿨롱 마찰, **공간 마찰**을 갖는 접촉 지원". `solver_kamino.py` L1686이 `joint_friction`을 처리. LOX에 `contact_spatial_friction` 옵션(L1492) | **차이** | 코드(논문 이후)가 논문의 한계 절보다 앞서 있음. README가 논문 한계 서술과 다름 |
| D11 | DR Legs 관측과 과제 | 식 (28): g, ω, v̂xy, ω̂z, sin/cos 위상, 구동 관절 12개 위치, 직전 두 행동 → 47차원. 걸음 주기 T = 1 s, 접촉 기준 0.5 비율 | 재생 코드(`example_rl_drlegs.py`): 관측 **94차원**(루트 회전행렬 9, 경로 이탈 4, 지령 세 형태, 위상 4(sin/cos 2차 고조파까지), 선속도와 각속도, 높이 지령과 오차, 수동 관절 24개 포함 **관절 36개 위치**, 행동과 직전 행동). `contact_duration = 0.3` → 위상률 1/(2·0.3), 걸음 주기 0.6 s. 높이 지령 [0.16, 0.27], `action_scale = 0.4`. 학습 코드, 보상, PPO 설정은 이 저장소에 없음 | **차이** | 공개 정책의 관측과 걸음 주기가 v2 부록 A.7.1의 서술과 다름. v1은 94차원이라고 적었고 v2의 식 (28)이 47차원으로 바뀜. 재생기는 ONNX 정책 추론만 함 |
| D12 | 모델과 실험 자산 | 표 A.1, §3.3: BDX, Olaf, Iron Man, Hoberman 구, Ant 변형, Franka 조작, DR Legs | `newton/examples/kamino`: DR Legs, 4절 링크, 이종 기본 모델(진자, 카트폴, 상자 등), ANYmal-D, 예제 `earworm_flowers`, Hoberman 생성기(`utils/hoberman.py`, 해상도 2에서 "240 스트럿, 421 관절" 재현 — 표 A.1과 일치). BDX, Olaf, Iron Man은 프로젝트 페이지에 "proprietary". Ant 변형, Franka 큐브 쌓기 예제는 찾지 못함. RL 예제(`example_rl_bipedal.py`)는 저장소 밖 `walking-character-rl` 자산 경로를 가정(공개 여부 확인 못함) | **미확인 (대부분 비공개)** | 표 1, 그림 3, 표 A.5, A.7의 대부분은 공개 자산으로 재현할 수 없음. 재현 가능 후보는 DR Legs, 4절 링크, Hoberman, 상자류 |
| D13 | 평가 스크립트 | 표 1(잔차 지표), 처리량 측정, MJWarp·XPBD 비교 | 지표 계산 모듈은 있음(`_src/solvers/metrics.py`, `Config.compute_solution_metrics = False`가 기본). Dolan–Moré 성능 프로파일 유틸(`utils/profiles.py`)도 있음. 그러나 표 1을 만드는 스크립트, MJWarp·XPBD 비교 구동기, 처리량 측정 스크립트는 찾지 못함 | **미확인** | 표 1과 그림 3을 같은 조건으로 재현하려면 평가 하네스를 새로 짜야 함 |
| D14 | 물리 지표 정의 | 식 (20)–(26): rJ, rC, rkin = ‖J_J u⁺‖∞, rv, rprimal, rdual = ‖v̂ − P_K*(v̂)‖∞, rcmpl, 자연 사상 rvi | `metrics.py`: `r_kinematics := ‖J_cts_joints u⁺‖∞`, `r_cts_joints`, NCP 원/이중/상보 잔차, `r_vi_natmap = ‖λ − P_K(λ − (v + s))‖∞` (de Saxcé 항 s 포함) | **일치** | — |
| D15 | 알고리즘 1의 갱신식 | x ← −D⁻¹_{η,ρ}(v_f + s − ηx − ρy − z), y ← P_K(x − z/ρ), z ← z − ρ(x − y), s ← Γ(z) | `kernels.py` L587, L661: `v = −v_f − s + η x_p + ρ y_p + z_p`(같은 식, 부호 일치. 유도로 확인), 쌍 갱신 `z = z_p + ρ(y − x)`(L689), de Saxcé `z_n ← v_n + μ‖v_T‖`(L216), 가속 a 점화 `(1 + √(1 + 4a²))/2`(L1261) | **일치** (D5, D7의 변형을 제외하고) | — |
| D16 | 이종 세계 | §2.2: (num_worlds, max_dim) 2D 스레드 그리드, 세계별 실제 크기로 조기 탈출 | 커널 발사가 `dim=(num_worlds, max_of_max_total_cts)`이고 `if tid >= problem_dim[wid]: return` 패턴 | **일치** | — |
| D17 | 리셋과 순기구학 | §2.1: 관절 좌표만 주어지면 FK 솔버가 일관된 몸체 자세를 만듦(v1은 Gauss-Newton) | `ForwardKinematicsSolverConfig`: Newton 반복 30회, 라인 서치 20회, 블록 Jacobi 전처리 CG, 증분 해(`config.py` L1698–1797). **`Config.use_fk_solver = False`가 기본**(`solver_kamino.py` L233)이고 DR Legs 예제는 True로 켬 | **일치 (알고리즘), 차이 (기본값)** | 리셋 기능은 있으나 켜야 함. 병렬 환경 리셋에서의 자기 충돌과 특이 자세 문제는 v1이 "향후 과제"라 했고 v2 본문은 다루지 않음 |

## 3. 읽은 결과 요약

- **알고리즘 1과 지표는 코드와 맞습니다**(D14–D16). 핵심 PADMM 갱신식, de Saxcé 보정, Nesterov 점화식, 잔차 정의가 같습니다.
- **그러나 코드의 기본 PADMM은 논문의 알고리즘 1 그대로가 아닙니다**(D5–D7). 관절 행의 ρ 제거, 잉여 구속 제거, 웜 스타트 배율, 전처리 클램프, 정지 잔차의 정의가 다르고, 이것들은 논문 방법 절에 없습니다. 루프 시스템에서의 "최소 튜닝" 주장은 이 부분과 분리하기 어렵습니다.
- **논문 이후 코드가 크게 자랐습니다**(D2, D9, D10). SCC와 LOX 솔버, 컴플라이언스 옵션, 관절 마찰이 들어와 논문의 한계 서술과 어긋납니다.
- **공개 자산으로는 일부만 재현할 수 있습니다**(D12, D13). 학습 코드와 보상, 평가 하네스, BDX/Olaf/Iron Man 모델이 없습니다.
- **정책 위험.** README가 BETA 1과 기여 비수용을 못 박습니다(D1). 이 코드를 연구 기반으로 쓴다면 특정 커밋에 고정해야 합니다.

## 4. 구현 가능성 판정 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**구현할 수 있는지: 논문 수치의 재현은 불가능하고, 알고리즘의 축소판 구현은 가능하며(일부 완료), 공식 코드의 CPU 실행은 조건부입니다.** 맥에서 할 수 있는 것은 (a) 코드와 논문의 대조(완료), (b) 보고 수치 재계산(완료), (c) PADMM과 접촉 모델의 numpy/torch 재구현, (d) 설치된 MuJoCo(CPU)로 하는 4절 링크 기준선입니다. GPU 처리량, RL 학습, 실물 전이는 이 맥에서 할 수 없습니다.

| 수준 | 판정 | 근거 |
|---|---|---|
| 논문·코드 정적 대조 (이 문서) | **가능 (완료)** | 파일 읽기만 |
| 보고 수치 재계산과 일관성 검사 | **가능 (완료)** | `analysis/recompute.py` (`03`) |
| PADMM과 PGS의 작은 LCP 비교 (질량비 의존성) | **가능 (완료)** | `analysis/toy_padmm_stack.py`, 약 7초, numpy |
| PADMM + de Saxcé 평면 접촉(마찰 포함) 재구현과 해석해 검증(상자 미끄럼, 반발) | **가능 (미착수)** | numpy 또는 torch CPU. 노력 1–2주(추정, 측정 전) |
| 4절 링크를 MuJoCo(등식 구속, solref)로 돌려 에너지와 구속 잔차 측정 | **가능 (미착수)** | 설치된 mujoco 4.0(CPU). 표 A.2의 MJWarp 열과 같은 정식화. 노력 2–3일(추정) |
| 공식 코드로 기본 예제(상자, 4절 링크, DR Legs 1세계) 실행 | **조건부** | Newton과 Warp 새 설치 승인 필요, Warp CPU에서 타일 연산과 커널이 도는지 미확인. 실패해도 지금까지의 분석에 영향 없음 |
| 표 1(정확도 비교)의 Kamino 열 | **조건부** | 위와 같은 설치 + 지표 모듈 사용. MJWarp와 XPBD 열은 GPU 솔버라 불가. 평가 하네스를 새로 짜야 함(D13). 노력 1–2주 |
| 처리량(그림 3, A.6, 표 A.5) | **불가** | RTX Pro 6000/5090급 GPU와 CUDA 그래프 전제. 이 맥에서는 의미 있는 수치가 안 나옴 |
| DR Legs와 Ant RL 학습 | **불가** | GPU 필요(3.5 h는 RTX 5090 기준), Isaac Lab 연결, 학습 코드와 보상은 저장소에 없음(D11) |
| BDX, Olaf, Iron Man 시뮬 | **불가** | 모델 비공개(D12) |
| 실물 sim-to-real | **불가** | 로봇 하드웨어 필요 |

- **노력 추정.** (c)는 1–2주, (d)는 2–3일, 설치가 승인되어 Warp CPU에서 돈다면 공식 코드 예제 실행은 1–2일입니다(모두 측정 전 추정). 나머지는 이 맥에서 할 수 없는 일이라 시간 추정을 하지 않습니다.
- **맥에서 의미 있는 것의 범위.** 논문의 핵심 주장(PADMM의 정확도, 루프의 처리, 학습 처리량)은 정확도 쪽만 CPU 소규모로 검증 가능합니다. 처리량과 학습은 GPU 서버가 필요합니다.

## 5. 재실행

공식 저장소를 다시 얻으려면 다음 명령을 쓰되 실행하거나 설치하지는 마십시오(sparse 복제, 약 20 MB).

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/newton-physics/newton.git repo
git -C repo sparse-checkout set newton/_src/solvers/kamino newton/examples/kamino docs
```
