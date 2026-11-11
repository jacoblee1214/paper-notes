# 04. 코드 검증: 논문 대 공식 저장소

대상: `https://github.com/coenwerem/vnb-grasp` (Apache-2.0), 커밋 `23702bde3eedea35e1990c0e2fa20e756021f85b` (2026-09-29, "add IROS controller comparison experiments"). `repo/`에 얕게 클론했습니다(gitignore). 코드는 **읽기만 했고 실행하지 않았습니다.** 설치도 하지 않았습니다. 저장소 메타데이터는 `gh api`로 확인했습니다(stars 3, forks 0, 생성 2026-07-24, 마지막 push 2026-09-29). 아래 판정은 정적 읽기 기준입니다.

판정 표기: **일치** (코드와 논문이 같음), **불일치** (다름), **미확인** (읽기로는 판단 못 함).

## 1. 확인한 파일

| 파일 | 내용 | 행 수 |
|---|---|---|
| `examples/run_variational_belief_experiments.py` | 전체 스윕, 5개 방법, 교란, 집계 | 2696 |
| `vnb_grasp/belief/variational_belief.py` | GMM 믿음, Gumbel rsample, CVaR 그래디언트 | 662 |
| `config/iros26_experiments.yaml` | 실험 상수(주석은 논문 초안 번호를 씀) | 310 |
| `config/contact.yaml` | 마찰 사전분포(논문 미언급) | 12 |
| `arenas/zarm_realhand_l6_right_arena/*.xml` | 마찰 값과 주석 | – |
| `examples/generate_paper_tables.py` | 표 생성 | 424 |
| `REPRODUCE.md`, `README.md` | 재현 절차와 주장 | 90, 159 |
| `grasp_db/*.json` | 파지 데이터 | 7 파일 |

`vnb_grasp/belief/belief_mpc.py`, `differentiable_metrics.py`, `contact_belief.py`, `particle_filter.py`는 훑었지만 세부는 읽지 않았습니다.

## 2. 대조 항목 (D1–D20)

### 마찰과 시뮬레이터

**D1. µ_f 값 (불일치, 높음).** 논문 Table I과 §V-D의 µ_eff 범위는 µ_f = 0.35일 때 값과 모두 일치합니다(`03` §6a). 논문 각주 2는 시뮬레이션에 µ_f = 1.0을 썼다고 합니다. 코드는 객체 geom의 마찰을 µ_o로 설정하고(`run_variational_belief_experiments.py` 815–819행), 손끝 geom은 아레나 XML 기본 클래스 `l6_right_collision`의 `friction="1.0 ..."`을 씁니다. 즉 **코드는 µ_f = 1.0이고, 논문 표의 µ_eff 범위는 다른 마찰입니다.**

**D2. 혼합 규칙 (불일치, 가장 중요).** 논문 식 (22)와 XML 주석 둘 다 기하평균이라고 합니다. MuJoCo 공식 문서(`modeling.html`, Contact parameters, friction)는 우선순위가 같으면 **원소별 최댓값**이라고 적고 있습니다(WebFetch 2026-11-11 확인, 인용 "Otherwise the element-wise maximum of each friction coefficient over the two geoms is used."). 손끝 1.0과 µ_o ≤ 1.0이면 실효 마찰은 1.0입니다. 낮은 마찰 체제, 마찰 교란(0.05–0.15), 두 체제 대부분이 같은 접촉이 됩니다. **정적 추론이므로 실행으로 확인해야 합니다**. 확인 전까지는 Table III의 "마찰 체제 간 차이"를 해석하지 않는 것이 맞습니다.

**D9. 접촉 강성 (불일치).** 논문 §V-A: κ ~ N(1000, 250²) N/m. 코드: `sample_stiffness`는 LogNormal(8, 0.3)을 1–20 kN/m로 자르고, 두 봉우리 체제에서는 µ/µ_nom으로 비례 조정합니다(234–282행). 그런데 `position_arm_and_object`는 강성을 **로그만 하고 solref에는 적용하지 않습니다**(`_ = stiffness`, 820–821행, 주석 "logged but not applied to solref"). 따라서 **강성 불확실성은 시뮬레이터에 들어가지 않습니다.** 논문과 코드가 다른 분포를 쓰고, 그 중 하나도 시뮬레이션에 영향을 주지 않습니다.

### 방법과 목적

**D3. 위험 가중치 λc (불일치, 높음).** 논문 Table V: λc = 1.0(VNB), 0.5(Gauss-CVaR), 0(Gauss). 코드:
- `VariationalBeliefConfig.risk_weight`(기본 0.5)는 정의만 되어 있고, `vnb_grasp/` 안에서 목적함수에 쓰이지 않습니다(`grep`식 확인: variational_belief.py 60행 외에 없음, `rs_quality.py`의 별도 `risk_weight`는 다른 클래스).
- `run_variational_episode`는 `gauss`, `gauss_cvar`, `variational`에서 모두 호출됩니다. 목적은 `cvar_gradient`(CVaR)로 고정됩니다. Gauss는 `n_components=1`, `risk_weight=0.5`입니다. 주석에는 "Gauss (K=1, risk_weight=0)"라고 되어 있지만 실제 호출 값은 0.5입니다(1716–1723, 2578–2612행).

**따라서 Gauss와 Gauss-CVaR는 같은 코드 경로입니다.** 논문 §VI-A의 "CVaR 가중치가 행동을 바꾸지 않았다"는 실험 결과가 아니라 구조상 같은 실행입니다. 또 Gauss는 논문이 말하는 위험 중립 기준선(λc = 0)이 아닙니다. Table III에서 VNB와 Gauss의 차이는 **K = 8 대 K = 1**(다중모드 대 단일 가우시안)에서만 나옵니다. 위험 항의 효과는 분리되지 않았습니다.

**D4. λv (불일치).** 논문 §IV-C, §IV-E: 시뮬레이션에서 λv = 0. Table V와 config(`visual_cost.lambda_v`): 0.3. 시뮬레이션 에피소드에서 Cvis가 실제로 계산되는지는 **미확인**입니다.

**D14. CEM 목적 (부분 불일치).** 논문: "Gaussian belief, population 64, elite 0.2, 3 iterations". 코드 `run_cem_episode`(2186–2194행): 점수 = 평균 비용 × (1 + 0.2 × 평균 닫힘) − 0.1 × (활성 관절 수/11). 기대비용에 임의 항을 더한 휴리스틱이고, 논문은 이것을 설명하지 않습니다. 또 CEM의 점수는 CVaR가 아닙니다.

**D22. Gumbel τ (불일치).** Table V: τ 1.0 → 0.1 코사인 어닐링, 500 epoch. 코드 `rsample`: `tau=0.5` 고정, `F.gumbel_softmax(..., hard=False)`(166–172행). 학습 루프는 예제 목록에서 찾지 못했습니다. 샘플링된 혼합 분포의 문제는 `03` §6d에 정리했습니다. 이 점에서 **Eq. (8)과 코드는 같습니다**(일치). 문제는 식 자체입니다.

**D21. 신경 믿음 네트워크 (미확인).** `run_variational_episode`는 에피소드마다 `NeuralBeliefFilter(v_config)`를 새로 만듭니다(약 1740행). 저장소 파일 목록에는 가중치 파일(`.pt`, `.pth`)이 없고, `examples/`에는 학습 실행 스크립트로 보이는 파일이 없습니다. 로딩 코드 전체를 확인하지 않았으므로 "학습되지 않은 무작위 초기화"라고 단정하지는 않습니다. 확인할 때까지 논문의 학습 설정(Table V의 Adam, 500 epoch)은 재현할 수 없는 것으로 봅니다.

### 교란 시험과 평가

**D5. 교란 개수 (일치 / 설정 불일치).** 코드 상수: 횡 충격 16(4 힘 × 4 방향) + 토크 9 + 마찰 저하 3 = 28(1011–1015행). 논문과 일치합니다. 그러나 `config/iros26_experiments.yaml`은 다른 목록(횡 힘 5개에 18 N 포함, 질량 증가 3개 포함, 합계 35)을 적고 있고, 코드는 이를 읽지 않습니다. **설정 파일은 실행 코드와 맞지 않습니다.**

**D6. 생존 임계값 (불일치).** 논문 Table V: 드롭 0.015 m, 변위 0.04 m, 회전 0.3 rad. 코드 상수: `DISP_THRESHOLD = 0.06`, `ROT_THRESHOLD = 0.5`, `DROP_THRESHOLD = 0.025`(1028–1031행). 주석에 "relaxed"라고 되어 있습니다. 논문 본문도 "1 cm"(§V-D 리프트 기준)을 씁니다. 코드의 임계값이 더 느슨해서 생존율이 과대 평가될 가능성이 있습니다. 정도는 실행 없이 판단할 수 없습니다.

**D10. 마찰 체제 (불일치).** 코드 `FRICTION_REGIMES`는 네 개(nominal, adversarial U[0.15, 0.40], wide, bimodal)입니다. 논문은 세 개를 보고합니다. 논문 Table III에 adversarial이 없고, 코드 REPRODUCE의 480 에피소드는 네 체제를 포함합니다. 논문은 "three friction regimes"라고만 적습니다.

**D11. 성공 기준 (불일치, 높음).** 논문 §V-D: 리프트 후 1 cm 이내 유지 + 모든 횡 펄스 생존. 코드 `success = stable and has_min_epsilon`(1292, 2021행), `EPSILON_MIN = 0.001`(291행). 리프트·전단 시험은 success일 때만 돌립니다. 즉 **SR은 "ε ≥ 0.001"이라는 거의 항상 만족되는 기준**이고, 논문이 말하는 리프트 기준이 아닙니다. config의 `fc_epsilon_threshold: 0.1`은 쓰이지 않습니다.

**D12. ε 측정 마찰 (불일치).** `compute_gws`는 `analyze_gws(contacts, center, friction_coef=0.8)`(845행)로 **0.8 고정**입니다. 최종 ε와 Table III의 ε·Quality 열은 에피소드 마찰과 무관합니다. 논문은 이 점을 밝히지 않습니다. 이 때문에 Table III의 ε 열이 거의 같은 값(0.010–0.012)인 것도 설명됩니다.

**D15. 전단 펄스 수 (불일치).** 논문: 3, 5, 8, 12 N의 4 펄스(§V-D, Table II). 코드: 리프트·전단 성공은 `n_survived == 3`(1011행 부근의 `shear_success`). config는 [3, 6, 12]. 세 가지가 서로 다릅니다.

**D8. 믿음 예측 실패율 P̂bel (미확인, 설명 불일치).** Table III 캡션: "Gaussian CDF, τ_f = 5.8". 코드: `failure_prob_predicted`는 `fail_prob_pred = step_log[-1]["failure_prob"]`(1300행 부근, 입자 경로)에서 옵니다. 그 값이 어떻게 계산되는지(Gaussian CDF인지 몬테카를로인지)는 VNB 경로에서 확인하지 못했습니다. `generate_paper_tables.py`는 같은 키를 읽습니다.

**D7. P̂emp 정의 (불일치).** 논문 식 (24)는 "모든 교란을 통과한 에피소드"의 비율입니다. 코드: `failure_prob_empirical = 1.0 - pert_rate`(1353행), 즉 교란 단위 평균입니다. Table III의 P̂emp도 이것과 일치합니다(`03` §1).

**D13. 실패 처리 (불일치, 집계).** 시간 초과와 예외는 `success=False`, `failure_prob_empirical=1.0`, 모든 지표 0인 자리표로 저장됩니다(2640–2680행). 평균에 포함됩니다. 논문은 이 처리를 말하지 않습니다.

### 나머지 대조

**D16. 실험 설계 (일치).** 객체 2개(cube, graspit_box), β 4개, 시드 3개(42, 123, 456), T_max = 80, CEM 64/0.2/3, 후보 20개와 닫힘 {0.05, 0.10, 0.15, 0.20}, 믿음 차원 64, 샘플 256/512, PF 입자 100. Table V와 맞습니다. 객체 마찰 명목값(PLA 0.40, LDPE 0.35, 알루미늄 0.25, 테니스 0.65)은 Table I과 맞습니다.

**D17. CVaR 근사 (일치).** `cvar_gradient`: 정렬로 η = 분위수, 손실 `softplus(beta=5)`, `eta + mean(softplus)/(1-β)`, η는 detach. 식 (12)와 같습니다. 그러나 편향은 `03` §6c에 있습니다.

**D18. 재매개변수화 (일치, 주의).** Gumbel 가중치와 위치-척도 샘플의 가중합(식 8–9)은 `rsample`과 같습니다. 주의점은 `03` §6d의 샘플링 문제입니다.

**D19. 파지 데이터셋 (일치, 수 확인).** `grasp_db/`의 파일 7개의 `n_grasps`는 36, 36, 27, 36, 36, 36, 36이며 합이 **243**입니다. 논문 초록의 "243 force-closed grasps"와 수가 같습니다. 다만 물체 7개 중 논문 실험에 쓰인 것은 cube와 box뿐이고, "force-closed"인지는 필드 확인을 하지 못했습니다(미확인).

**D20. 실물 실험 (재현 불가).** 저장소 `REPRODUCE.md`는 실물 실험을 범위 밖으로 명시합니다. 코드가 없습니다.

**D23. 저장소 메타데이터 (확인).** Apache-2.0, stars 3, forks 0, 마지막 push 2026-09-29, 기본 브랜치 main, HEAD `23702bd`. 논문 README의 "Code, simulation assets, and dataset"이라는 주장 중 코드, 자산, 데이터셋(243 grasp JSON)은 있습니다.

**D24. 논문 본문 대 README 수치 (불일치).** README는 "79% vs 63% under nominal friction, 5 to 7 times faster"라고 씁니다. 79/63은 Table III의 **Robust**(VNB 대 CEM, nominal)이지 SR이 아닙니다. 본문의 "roughly an order of magnitude"와 "5–7배"도 `03` §4처럼 맞지 않습니다.

**D25. 논문 주석이 가리키는 버전 (관찰).** `config/iros26_experiments.yaml` 주석은 "Sec 3.3, eq. 7", "Sec 5.1", "Sec 5.2"처럼 논문 본 번호와 다른 초안 번호를 씁니다. 제목도 "Variational Neural Beliefs for Robust Dexterous Grasping under Multi-Modal Uncertainty"로 arXiv 제목과 다릅니다. 저장소가 논문 이전 초안을 기반으로 했을 가능성이 있습니다.

## 3. 저장소에 있는 것과 없는 것

| 항목 | 상태 |
|---|---|
| MuJoCo 아레나, ZArm 622 + RealHand L6 | 있음 |
| 객체 자산과 GraspIt 파지 DB(243) | 있음 |
| 5개 방법의 스윕 러너와 집계 | 있음 |
| 논문 Table 생성 스크립트 | 있음(표 형식, 열 이름이 논문과 다름) |
| 그림 렌더링 스크립트 | **없음**(REPRODUCE.md가 Figs. 1, 4, 5를 범위 밖으로 둠) |
| 신경 믿음 학습 스크립트와 가중치 | **없음**(미확인, `D21`) |
| 실물 실험 코드(FR3, RealHand, ICP) | **없음** |
| 실험 로그 또는 결과 JSON | 없음(재생성이 필요) |

## 4. 구현 가능성 판정

**판정: 조건부 가능, 중간 노력. 지금 상태로 논문 표를 재현하는 것은 어렵습니다.**

공식 시뮬레이션 재현은 기술적으로 가능해 보입니다. 파이썬, MuJoCo 3, PyTorch, NumPy, SciPy가 필요하고(README), JAX는 미분 가능 지표용으로만 씁니다. README와 REPRODUCE.md는 `MUJOCO_GL=egl`을 headless에 쓰라고 하지만, 이는 렌더링이 필요할 때입니다. CUDA 요구는 저장소에서 확인하지 못했습니다(논문 Table V는 RTX 4070을 적었습니다). REPRODUCE.md의 추정은 480 에피소드 4–8시간(PF가 대부분)입니다. 샘플 실행(5개 방법 × 1 에피소드)은 약 6분입니다.

**이 맥 (Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음, torch와 MuJoCo 미설치).**
- 설치 전에는 아무것도 실행할 수 없습니다. 설치는 사용자 승인 후에 합니다.
- MuJoCo 자체는 macOS arm64용으로 설치할 수 있을 것으로 보입니다(미확인). 헤드리스 EGL 설정은 맥에서 쓰지 않습니다.
- 샘플 실행(약 6분, 단일 스레드 가능)은 CPU로 가능해 보입니다. 480 스윕은 4–8시간이므로 야간 실행이 가능합니다.
- 실물 재현은 불가능합니다(FR3 팔, RealHand L6 손, RealSense D435i와 Orbbec Astra 두 대, ICP 파이프라인).

**재현 전에 먼저 고쳐야 할 것.** 아래는 코드가 논문과 다르거나 설정이 실행과 맞지 않는 항목입니다. 이것들을 정하지 않은 채 재현하면 어떤 숫자가 나와도 해석하기 어렵습니다.
1. 마찰 혼합 규칙(D2)과 µ_f(D1): 최댓값 규칙을 쓰면 체제 설계를 바꿔야 합니다.
2. λc 연결(D3): Gauss를 K = 1, λc = 0으로 고쳐야 위험 항의 효과를 분리할 수 있습니다.
3. 강성(D9): solref에 적용할지, 논문대로 N(1000, 250²)을 쓸지 결정해야 합니다.
4. 성공 기준과 임계값(D6, D11, D15): 논문 기준으로 되돌려야 Table III를 비교할 수 있습니다.

**재현 계획은 사용자 확인 후에 진행합니다** (CLAUDE.md 규칙). 이 판정은 제안입니다.

## 5. 권리 판정 (CC BY 4.0 여부)

- **arXiv abs 페이지:** "view license"가 CC BY 4.0(`creativecommons.org/licenses/by/4.0/`)을 가리킵니다(2026-11-11 확인).
- **PDF 1쪽의 고지:** "© 2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works." 그리고 "Accepted for publication at the 2026 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)."
- **판정:** PDF에 **충돌하는 권리 고지(IEEE)가 있으므로 CLAUDE.md의 기준에 따라 논문 그림은 공개 산출물에 넣지 않습니다.** arXiv 라이선스와 IEEE 고지가 다르고, IEEE 고지는 "재게시"와 "다른 저작물에의 재사용"을 제한합니다. 개인 리포트와 PPT에도 논문 그림을 넣지 않았습니다(텍스트 인용과 수치 재계산만 사용). 그래서 `report/figures/`에는 우리가 그린 그래프만 있습니다.
- **코드:** Apache-2.0(LICENSE 파일 확인). 공개 산출물에 코드 조각을 싣지 않았습니다.
- **그래도 확인할 것:** IEEE 고지는 IEEE 게재본 기준일 수 있습니다. 게재본과 arXiv 판이 같은지 확인하지 않았습니다. 필요하면 저자나 IEEE에 문의해야 합니다(확인하지 못함).

## 6. 검증 근거 요약

- 코드 확인: `run_variational_belief_experiments.py`, `variational_belief.py`, `arenas/*.xml`, `config/iros26_experiments.yaml`, `REPRODUCE.md`, `README.md`, `grasp_db/`.
- 외부 확인: MuJoCo 문서(modeling.html, 2026-11-11 WebFetch, 혼합 규칙 "element-wise maximum").
- 실행하지 않은 것: 모든 코드, 설치, MuJoCo 시뮬레이션.
