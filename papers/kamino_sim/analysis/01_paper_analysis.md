# 01. 논문 분석: Kamino (arXiv:2603.16536)

*Kamino: GPU-based Simulation of Rigid Multi-Body Physics with Kinematic Loops* (Vassilios Tsounis*, Guirec Maloisel*, Christian Schumacher*, Ruben Grandia, Agon Serifi, David Müller, Chris Amevor, Tobias Widmer, Moritz Bächer; Disney Research Zurich, NVIDIA Zurich). *공동 제1저자. 분석 대상은 **arXiv v2 (2026-10-20, 26쪽)**이고, 원문 첫 쪽에 "10th Conference on Robot Learning (CoRL 2026), Austin TX, USA"가 찍혀 있습니다. 쪽·절·표 번호는 v2 기준이며, v1을 가리킬 때는 "v1"이라고 적습니다.

## 0. 메타와 버전 차이

| 항목 | 내용 |
|---|---|
| 학회 | CoRL 2026. `conferences/corl2026/papers.json`에서 `in_official: true`, 발표 유형 spotlight(2026-11-11 수요일, 세션 S7 "Humanoid locomotion and whole-body control", 포스터 98). 근거 목록은 corl.org 공식 채택 목록(수집일 2026-11-14) |
| arXiv 주석 | 비어 있음(comments, journal-ref 없음). 학회 표기는 PDF 첫 쪽의 CoRL 2026 문구와 공식 목록으로 확인 |
| 코드 | 공식 구현은 NVIDIA Newton 저장소 안(`newton/_src/solvers/kamino`). 별도 저장소 없음 |
| 프로젝트 페이지 | https://kamino.disneyresearch.com/ (초록, 예제 링크, 사용법, 인용) |
| 보조 문헌 | 같은 저자들의 기술 보고서 arXiv:2504.19771 ("On Solving the Dynamics of Constrained Rigid Multi-Body Systems with Kinematic Loops", 2025-04-28)가 수식 전체 유도를 맡습니다(본문 [16]) |

### 0.1 v1 → v2 차이

| 항목 | v1 (2026-03-17, 14쪽) | v2 (2026-10-20, 26쪽) |
|---|---|---|
| 제목 | *Kamino: GPU-based Massively Parallel Simulation of Multi-Body Systems with Challenging Topologies* | *Kamino: GPU-based Simulation of Rigid Multi-Body Physics with Kinematic Loops* |
| 저자 | 같음. 공동 제1저자 표기 없음 | 앞 세 명이 공동 제1저자 |
| 비교 대상 | 없음(Newton 내 다른 솔버는 말로만 언급) | MJWarp, XPBD(같은 Newton의 GPU 솔버), MuJoCo CPU를 정확도와 처리량에서 직접 비교 |
| 물리 검증 | 없음 | 상자 미끄럼·공 반발·상자 더미·4절 링크(해석 기준), 로봇 궤적 잔차 표 1 |
| 조작 | 없음 | 두 손가락 팔 큐브 쌓기 100회 중 100회 성공(§3.1) |
| 학습 | DR Legs 31 h(RTX Pro 6000, 환경 스텝 약 3,600/s), Olaf는 영상, BDX는 Isaac Sim과의 sim-to-sim 전이 | DR Legs 10시드(생존율 98.5 ± 0.4%), 학습 3.5 h(RTX 5090), 이종 Ant 변형 2.8 h, BDX와 Olaf 실물 sim-to-real(영상) |
| 시스템 크기 | Olaf 36 몸체·루프 5, Iron Man 151 몸체·루프 45, BDX 21 몸체 | Olaf 27·3, Iron Man 67·13, BDX 15 (표 A.1) |
| PADMM 반복 | "10–30회면 충분(Δt = 1/240 s)", CR 예산 9 | 표 1에서 200회(Kamino), 처리량 측정은 PADMM 100·CR 15 고정 |
| 한계 서술 | 강체 전용, 미분 불가, 이중 공식만 지원 | 처리량이 MJWarp·XPBD보다 낮음, 마찰은 선형 3D 쿨롱만(비틀림·구름·관절 정지 마찰 없음), 실물 검증이 제한적 |
| 통합 | Newton 안 | Newton과 Isaac Lab 경유를 명시 |

- v1은 "공식 코드 없이 새 솔버를 소개하는 글"에 가깝고, v2는 비교 실험과 해석해 검증을 붙인 개정판입니다. 같은 이름의 수치가 버전 사이에서 바뀐 곳은 `03` K 항목에 모았습니다(DR Legs 학습 31 h → 3.5 h, Iron Man 모델 크기).
- v1 그림 4 캡션은 "BDX와 Iron Man은 트리 구조"라고 쓰는데 같은 쪽 표 1은 Iron Man 루프를 45개로 적습니다. v2에서는 이 모순이 없어졌습니다(루프 13개).

## 1. 풀려는 문제와 기여 주장

- **문제.** 대부분의 GPU 시뮬레이터(Isaac Gym, Brax, MJX, MJWarp)는 최소 좌표(reduced coordinates)의 트리 구조를 가정해 닫힌 기구 사슬(네 마디 링크 등)을 별도 등식 구속이나 mimic joint로 근사합니다. 이 방식은 트리와 구속 사이의 임의 분할(편향)과 파라미터 조정이 필요합니다(§1).
- **접촉.** PGS 같은 국소 반복법은 루프와 비침투 접촉이 겹치면 구속 드리프트·인공 컴플라이언스·발산이 생기고, MuJoCo식 볼록 완화나 Macklin의 비매끄러운 Newton은 컴플라이언스·정규화에 의존한다고 봅니다(§1).
- **주장.** 저자들은 다음 네 가지를 동시에 만족하는 "첫" 솔버라고 씁니다(§1, 끝 단락): (i) 임의 위상과 루프를 최소 튜닝으로 처리, (ii) 접촉 모델을 근사하지 않음, (iii) GPU 병렬로 RL 학습이 가능, (iv) 오픈소스. "to the best of our knowledge"가 붙지만 "첫"이라는 주장은 선행 PADMM 연구(Carpentier 외 [19, 20], 같은 저자의 [16])가 같은 공식화를 이미 다뤘다는 점에서(본문 §1은 "GPU 구현이 없었다"고 씀; 선행 구현의 플랫폼은 확인하지 않음) GPU 구현과 시스템 통합이 기여의 핵심입니다.

## 2. 방법

### 2.1 인터페이스 (§2.1)

- `SolverKamino`는 Newton의 `SolverBase`를 구현하고, 정적 `Model`과 GPU의 `State`, `Control`, `Contacts`를 받아 `solver.step(state_in, state_out, control, contacts, dt)`로 한 스텝 진행합니다. 이중 버퍼링과 CUDA 그래프 캡처를 전제합니다.
- 최대 좌표 표현에서는 리셋이 어렵다는 점을 따로 다룹니다. `solver.reset(state, world_mask, ...)`가 관절 좌표만 주어져도 순기구학(FK) 솔버로 루프 구속을 만족하는 몸체 자세를 만들어 줍니다(Schumacher 외 [12]).

### 2.2 최대 좌표 동역학 (§2.2, 부록 A.1)

- 몸체 i마다 SE(3) 자세와 6차원 트위스트를 둡니다. 임펄스-속도 뉴턴-오일러 식 M(u⁺ − u⁻) = Δt h + Jᵀλ (§2.2). 구속 세 종류를 한 시스템에 합칩니다. (i) 쌍방 관절 구속(암시적 PD, 아마추어, 점성 감쇠는 구속 안에 흡수), (ii) 단방향 관절 한계, (iii) 접촉(Signorini–Coulomb–Newton, de Saxcé 보정).
- 슈어 보수로 속도를 소거하면 식 (1)의 이중 문제가 됩니다: λ = argmin_{x∈K} ½xᵀDx + xᵀv_f + Γ(v⁺(x)). 여기서 D = JM⁻¹Jᵀ + R은 Delassus 행렬(식 11), v_f = J(u⁻ + ΔtM⁻¹h) + v*는 자유 속도(식 12), Γ는 de Saxcé 보정입니다(식 6: 접선 슬립 속도에 비례하는 법선 방향 속도 오프셋 μ‖v_T‖).
- 의미: 슬라이딩 접촉이 마찰 때문에 떨어지거나 파고들지 않게 합니다. 저자들은 MJWarp가 이 항이 없어 상자가 미끄러질 때 "울퉁불퉁한 바닥처럼 떤다"고 설명합니다(부록 A.4).
- 이종 세계: 모든 커널이 (num_worlds, max_dim) 2D 스레드 그리드를 쓰고, 세계별 실제 크기를 저장해 범위를 넘는 스레드는 일찍 종료합니다(§2.2 끝).

### 2.3 Proximal-ADMM (§2.3, 부록 A.2, 알고리즘 1)

- 알고리즘 1: D_{η,ρ} = D + (η + ρ)I를 한 번 분해(또는 희소 연산자 준비)하고, 각 반복에서 (1) s ← Γ(z), (2) x ← −D_{η,ρ}⁻¹(v_f + s − ηx − ρy − z), (3) y ← P_K(x − z/ρ), (4) z ← z − ρ(x − y), (5) Nesterov 가속 후 (rprimal, rdual, rcmpl) < ε이면 종료.
- 구현 요령(A.2.1): Nesterov 가속과 재시작(결합 잔차가 줄지 않으면 a = 1), 웜 스타트(접촉은 기하 쌍과 위치로 매칭), Jacobi 전처리 P = diag(D)^(−1/2)(단정밀도 때문에 중요하다고 설명), 반복 루프 전체를 CUDA 그래프로 캡처, 세계별 조기 종료 마스크.
- 선형 풀이(A.2.2): 작은 시스템은 블록 Cholesky(dense LLT, 야코비안은 dense 또는 BSR 희소), 큰 시스템은 matrix-free 델라수스 연산자와 웜 스타트된 켤레 잔차(CR). 보고된 예에서 CR 반복은 "최소 15회"(v1은 9회).
- 정밀도: Newton의 GPU 솔버는 모두 32비트이고, MuJoCo CPU만 64비트입니다(§3 각주 2). 정확도 비교가 정밀도 차이를 함께 담고 있다는 뜻입니다.

## 3. 실험과 수치 (v2)

### 3.1 물리 검증 (§3.1, 부록 A.4)

| 예제 | 기준 | 보고 결과 |
|---|---|---|
| 상자 미끄럼(힘을 정지 임계의 1.5배까지 선형 증가) | 두 번 적분한 해석해(Δt = 0.1 ms) | Kamino와 XPBD(반복 2 → 20)는 정확. MJWarp(stiff)는 과도한 고착, soft는 총 거리는 맞추나 관통과 울렁임 |
| 공 반발(e = 0.8) | 구간별 이차 해석해 | Kamino와 XPBD 일치. MJWarp는 반발을 직접 지원하지 않아 ke, kd 맞춤 필요, 관통 |
| 상자 더미(100 kg/1 kg 세 구성) | 질량과 무관한 해석 궤적 | Kamino 일관. XPBD는 반복 200에서도 오차가 큼(기본 2). MJWarp는 구속 연화로 관통 |
| 4절 링크(수동, 에너지 보존) | 에너지, 위치 구속 잔차 | 표 A.2(§03 V9). 구속 만족은 400 Hz에서 XPBD, 1000 Hz에서 MJWarp가 최선. Kamino는 감쇠가 안정화 가중 α에 단조라고 주장 |
| 조작(Franka Panda + 손, 큐브 쌓기) | 100회 무작위 초기 조건 | 100회 중 100회 성공 |

- 해석해 비교는 모두 **그림(곡선 겹침)** 으로만 제시되고 수치 오차(RMSE 등)는 없습니다. 여기서 맞다고 주장한 부분은 검증할 수치가 없어 `03`에서는 표 A.2와 표 1만 재계산했습니다.

### 3.2 로봇 궤적 잔차 (§3.1, 표 1, 부록 A.1.4)

- 지표: 구속 위반 rJ, rC(최대 노름), 관절 운동학 잔차 rkin = ‖J_J u⁺‖∞, 접촉 법선 속도 위반 rv, NCP 잔차 rprimal, rdual, rcmpl, 자연 사상 잔차 rvi(식 20–26). 시간에 걸쳐 **중앙값과 90분위**로 요약합니다(충돌 직후 프레임 제외 목적).
- 프로토콜(표 1 캡션): BDX, Olaf, DR Legs는 무작위 토크로 쓰러지게 하고, Iron Man은 접촉 없이 무구동. 구성 수준 지표(rJ, rC)는 솔버마다 독립 궤적, 나머지는 Kamino가 10 kHz로 만든 기준 궤적의 단일 스텝 문제를 모든 솔버에 같게 줌. XPBD 반복 20(Iron Man 200, 기본 설정은 발산), Kamino PADMM 200.
- 결과: 본문은 "best or near-best, 특히 속도 수준 지표(rkin, rvi)"라고 요약합니다. 우리의 셀별 재계산은 `03` V1.

### 3.3 처리량 (§3.2, 그림 3, 부록 A.5)

- RTX Pro 6000(그림 3, A.6, A.8), RTX 4090·5090(그림 A.7). 세계 수에 따라 약 수천에서 포화. 작은 시스템은 dense LLT, 큰 시스템(Iron Man)은 sparse CR이 낫다고 설명.
- BDX에서 Kamino(희소 LLT) 2.4 × 10⁵, XPBD 5.8 × 10⁵, MJWarp 6.1 × 10⁶ samples/s(§3.2). 즉 Kamino는 MJWarp보다 약 25배 느립니다.
- 처리량 측정은 조기 종료를 끄고 PADMM 100, CR 15 고정(A.5). DR Legs, Olaf, Iron Man에서 MJWarp와 XPBD의 처리량은 보고하지 않습니다.

### 3.4 복잡한 기구 (§3.3, 부록 A.6)

- DR Legs(미리 정한 걸음 + PD): MJWarp는 관절 트리를 어디에 뿌리내리느냐에 따라 달라집니다(골반 뿌리는 안정, 왼발 뿌리는 넘어짐). Kamino는 Δt = 1.25 ms에서 안정, XPBD는 Δt = 0.2 ms와 반속 재생이 필요.
- Iron Man(루프 13, 질량 0.012–49.2 kg): MJWarp는 루프 구속 파라미터 조정 + Δt = 0.2 ms, XPBD는 반복 100 + Δt = 0.2 ms, Kamino는 Δt = 5 ms에서 최소 튜닝.
- Hoberman 구(몸체 240, 루프 182): 세 솔버 모두 Δt = 2 ms 가능, MJWarp는 빠른 회전에서 튜닝 필요.

### 3.5 강화학습 (§3.4, 부록 A.7)

- 공통: 정책 50 Hz, 시뮬레이터 Δt = 4 ms, 데시메이션 5, 환경 4,096개, MLP 512×3(ELU), PPO. 학습 4,000 반복이 DR Legs 약 3.5 h, 이종 Ant 약 2.8 h(RTX 5090).
- DR Legs: 속도 지령 추종(v̂x, v̂y ∈ ±0.3 m/s, ω̂z ∈ ±0.8 rad/s), 걸음 주기 T = 1 s, 관측 식 (28), 보상 표 A.3(가중합), 종료는 10 s, 높이 < 0.12 m, 기울기 > 1 rad. 10시드 재학습에서 "모두 같은 걸음으로 수렴, 생존율 98.5 ± 0.4%, 보상 변동 2% 미만".
- 이종 Ant(3·4·5다리): 한 정책이 세 형태 모두를 제어. 관측에 형태 원-핫, 처리량은 표 A.5.
- BDX와 Olaf: 각 원 논문의 설정으로 학습하고 "PhysX에서 학습한 원래 컨트롤러와 비슷한 성능", 실물 sim-to-real은 영상으로만 제시.

## 4. 한계

논문이 쓴 한계(§4): (1) 처리량이 MJWarp와 XPBD보다 낮음(최대 좌표와 NCP 직접 풀이), (2) 선형 3D 쿨롱 마찰만 지원, (3) 실물 검증이 제한적.

우리가 덧붙이는 한계:
- **비교 설계.** 정확도 표는 한 궤적(또는 한 기준 궤적)의 중앙값과 p90이라 반복·불확실성이 없습니다. 비교 솔버의 파라미터 조정 수준(특히 MJWarp)은 저자가 정하며, 조정 공개 범위는 본문에 없습니다.
- **정밀도 혼재.** 세 GPU 솔버는 32비트, MuJoCo CPU는 64비트입니다.
- **처리량 비교의 빈칸.** BDX 외 모델에서 MJWarp와 XPBD의 처리량이 없고, 솔버마다 시간 간격이 6–25배 다른데도 "시뮬레이션 시간당" 비교가 없습니다(`03` V5).
- **학습 비교의 빈칸.** MJWarp나 XPBD로 같은 DR Legs 과제를 학습한 결과가 없습니다. "트리 근사 대비 학습 이득"은 보이지 않았습니다.
- **재현 자원.** BDX, Olaf, Iron Man 모델은 공개되지 않았고(프로젝트 페이지 "proprietary"), 정확도 표와 처리량 측정 스크립트도 저장소에 없습니다(`04` D12, D13).

## 5. 정량 주장 목록

| 번호 | 주장 | 위치 | 상태(우리 검증) |
|---|---|---|---|
| Q1 | Kamino는 표 1에서 best 또는 near-best 잔차를 낸다. 특히 속도 수준 지표 | §3.1, 표 1 | 부분 지지. 26개 셀 중 중앙값 19, p90 14에서 최선 (`03` V1) |
| Q2 | 해석해 네 예제 모두 Kamino가 튜닝 없이 근접. 조정 값은 4절 링크의 Baumgarte 가중뿐 | §3.1, 부록 A.4 | 확인 불가(그림만, 수치 오차 없음). A.2는 재계산(`03` V9) |
| Q3 | 조작 100회 100% 성공 | §3.1 | 산술 확인. Clopper–Pearson 하한 96.4% (V4) |
| Q4 | BDX 처리량 Kamino 2.4e5, XPBD 5.8e5, MJWarp 6.1e6 | §3.2 | 그림 A.6 판독과 일치. 비율 2.4배, 25배 (V5) |
| Q5 | Iron Man에서 sparse CR이 큰 시스템에 낫다 | §3.2, 그림 3 | 최대 처리량(약 1,800 세계)에서만. 4,000 세계 이상은 LLT 희소가 빠름(판독, V5) |
| Q6 | DR Legs Δt = 1.25 ms 안정, XPBD는 0.2 ms | §3.3 | 확인 불가(영상). 시간 간격 비 6.25배 |
| Q7 | Iron Man Δt = 5 ms 안정, MJWarp·XPBD는 0.2 ms | §3.3 | 확인 불가. 비 25배 |
| Q8 | DR Legs 10시드 생존율 98.5 ± 0.4%, 보상 변동 < 2% | §3.4 | 산술만. ± 정의와 에피소드 수 없음(V4) |
| Q9 | 이종 Ant 처리량 표 A.5, 이종이 5다리보다 빠르고 4다리보다 느림 | 부록 A.7.2 | 서술과 일치. 단순 혼합 모형 대비 14% 느림(V3) |
| Q10 | DR Legs 4,000 반복 학습이 RTX 5090에서 3.5 h | 부록 A.7 | 반복당 롤아웃 길이 미보고. 24라면 벤치 처리량의 약 2.6배가 필요해 불가능에 가까움(V6) |
| Q11 | 4절 링크: 1000 Hz와 stiff 구속이어야 1 mm 미만(링크 길이 약 6 cm) | 부록 A.4, 표 A.2 | 표와 일치(V9) |
| Q12 | 걸음 주기 T = 1 s는 2π√(L/g) ≈ 1 s (L = 0.265 m) | 부록 A.7.1 | 1.033 s로 확인(V7) |
| Q13 | 표 A.1의 루프 수 | 표 A.1 | 위상 항등식으로 모두 일치(V2) |
| Q14 | BDX, Olaf 실물 sim-to-real 성공 | §3.4 | 영상만, 수치 없음. 검증 불가 |
