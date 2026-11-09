# 04. 주장 검증: 공개 기반 코드 대조, 표 재계산, 물리 toy, 구현 가능성

논문의 공식 코드와 데이터는 공개되지 않았습니다(2026-11-09, arXiv 페이지·논문 본문·GitHub 검색으로 확인, `02` §5). 그래서 코드 대조 대신 (1) 논문이 기반으로 삼은 ACT·ALOHA 공개 저장소에서 논문 해석에 필요한 사실을 확인하고, (2) Table IV를 다시 계산하고, (3) 핵심 물리 주장을 numpy toy로 점검했습니다.

## 1. 기반 코드에서 확인한 사실

대조한 저장소: `tonyzhaozh/act` @ `742c753`(2024-01-28), `tonyzhaozh/aloha` @ `06369f0`(2024-04-19). 논문의 ALOHA Solo는 Trossen이 낸 별도 소프트웨어를 썼을 수 있으므로, 아래는 "원조 ALOHA·ACT 기준"입니다.

| # | 확인한 것 | 위치 | 논문 해석에 주는 의미 |
|---|---|---|---|
| D1 | follower 팔 서보는 Position P gain 800, **I gain 0** | `aloha/aloha_scripts/robot_utils.py:174-175` | 하중이 있으면 정상상태 오차가 남음. 접촉 중 δ가 곧 누르는 힘(`03` §1) |
| D2 | ACT 정책 입력은 **현재 qpos 한 프레임 + 카메라 이미지**. qvel·effort는 데이터에서 읽지만 쓰지 않음. `qpos_history`는 저장만 함 | `act/utils.py:37-38, 74`, `act/imitate_episodes.py:221, 240-249` | δ는 입력이 아니라 목표에만 있음. 한 프레임으로는 강성 판별이 안 됨(`03` §2) |
| D3 | 상태 차원 14 하드코딩(양팔 2 × 7), `nn.Linear(14, hidden_dim)` | `act/detr/models/detr_vae.py:58, 61, 69-70, 230, 258` | 한 팔 7 + effort 7 = 14가 구조 변경 없이 들어감. 논문의 "14차원 공유 임베딩"(Sec. IV-B)과 맞음(추정) |
| D4 | ALOHA 수집 스크립트는 **effort를 이미 기록**(`/observations/effort`, 14차원) | `aloha/aloha_scripts/record_episodes.py:130, 142, 160`, `real_env.py:75-80` | 기존 ALOHA 데이터에도 토크 대리값이 있음. 새 센서 없이 ACT + τ를 학습할 수 있음 |
| D5 | 기록되는 행동은 **leader(master) 관절각** | `aloha/aloha_scripts/real_env.py:140-149` | Base ACT 목표 = leader. 논문 Table II와 같음 |
| D6 | temporal ensembling k = 0.01, i = 0이 가장 오래된 예측, 켜면 매 스텝 질의 | `act/imitate_episodes.py:191-193, 255-257` | 평균 명령 나이 1.15 s(chunk 100, 50 Hz). 논문은 사용 여부를 적지 않음(`03` §8) |
| D7 | README 예시: kl 10, chunk 100, hidden 512, batch 8, d_ff 3,200, epoch 2,000, lr 1e-5 | `act/README.md:76-77` | 논문은 batch·epoch·lr만 다름 |
| D8 | 학습·추론 코드에 `.cuda()`가 8곳 하드코딩 | `act/imitate_episodes.py` | 이 맥(CUDA 없음)에서는 수정하거나 LeRobot 구현을 써야 함(§4) |

## 2. 주장별 검증

| # | 주장 (출처) | 검증 방법 | 결과 | 판정 |
|---|---|---|---|---|
| C1 | Table IV 수치 (Table IV) | 내부 일관성: Bottle 전체 = 두 모드 평균, 성공 수 정수, n_total 산술 | 4개 변형 모두 일관. n_total 80/40/80/120과 일치 | 맞음 |
| C2 | δ를 지우면 접촉 과제가 무너진다 (Abstract, Sec. VI-A) | Fisher·Holm·MH | Wipe 12/20 → 3/20(Holm p = 0.024). Bottle·Plug는 과제별 비유의. 합산 MH p = 2.8×10⁻⁴ | 맞음(Wipe가 주도) |
| C3 | 토크가 ACT-o의 접촉 행동을 회복한다 (Sec. VI-A) | 같음 | 세 과제 모두 유의(p ≤ 0.033), 합산 p = 3.4×10⁻¹⁰ | 맞음 |
| C4 | 토크가 Base ACT도 개선한다 (Abstract "improve the base ACT policy") | 같음 | 과제별 p = 0.065–0.33(Holm 0.19–0.36), 합산 p = 0.0034. 단단한 병만은 p = 0.0007 | 합산으로만 유의. 과제별로는 미확정 |
| C5 | ACT-o의 실패는 암묵적 힘 인지가 사라졌기 때문 (Sec. VII) | P 서보 정상상태 해석 + toy T1 | follower 목표 재생 힘은 시연의 0.25–0.46배이고 대부분 중력 처짐. 실패는 "감지 상실"보다 "구동 여유 상실"로 더 직접 설명됨 | 해석 보완 필요 |
| C6 | 토크가 시각적으로 같은 강성 상태를 구분하게 한다 (Sec. VI-A) | toy T2 + Youden J | 같은 높이에서 qpos 한 프레임 0.51, 토크 1.00, qpos 5프레임 0.96. J: Base 0, ACT + τ 0.4, ACT-o + τ 0.5 | 맞음. 다만 qpos 이력 대조군 없음 |
| C7 | 전류 기반 τ = k_t I가 중력·마찰 보상 없이 충분하다 (Sec. III-C, IV-B) | toy T3 | 정지 마찰 띠(가정 0.83 N) 아래 힘은 검출률 0.32–0.62, 2 N 이상은 ≈ 1. 중력 미보상 시 검출률 0.84 → 자세 회귀 후 1.00 | 수 N 이상 사건에는 맞음. 1 N 미만은 근거 약함 |
| C8 | 토크 변형이 Foam Stop 평형 F_x에 더 가깝게 수렴 (Sec. VI-B4, Fig. 4) | 그림 판독(±50) | 평균 차이는 토크 변형이 작지만(약 45–55 vs 100–150), Height 1에서는 ACT-o가 가장 가깝고 띠가 겹침. F_x는 중력 미보상 | 부분적. 정량 근거 약함 |
| C9 | 토크 변형이 Plug에서 더 정확한 "adjustment-tap"을 배운다 (Sec. VI-B3) | 재시도 이득 계산 | 이득: Base +10, ACT + τ +30, ACT-o + τ +40%p. 1회 기준 ACT + τ vs Base 4/10 vs 2/10(p = 0.63) | 재시도 규칙에 크게 의존 |
| C10 | 혼합(δ + τ)이 명시적(τ만)보다 낫다는 암시 (Fig. 1의 이중 체크) | Fisher·MH | 모든 과제 p ≥ 0.65, 합산 p = 1.0 | 근거 없음 |
| C11 | 동기: kinesthetic·고투명 인터페이스 데이터에서도 τ가 단서를 대신 (Sec. I, III-A) | 실험 설계 검토 | 실제 실험은 같은 leader-follower 데이터에서 목표만 바꿈. 다른 수집 방식은 시험하지 않음 | 시험되지 않음 |

검산 스크립트: `analysis/table_check.py`(C1–C4, C6의 J, C9, 하이퍼파라미터), `analysis/force_proxy_toy.py`(C5–C7). 결과는 각각 `table_check_results.json`, `force_proxy_toy_results.json`입니다.

## 3. 재현에 필요한데 논문에 없는 정보

| 항목 | 상태 |
|---|---|
| 과제별 시연 수, 시연자 수 | 없음 |
| 학습 시드 수, 변형마다 몇 번 학습했는지 | 없음 |
| effort 단위(원시 전류 단위인지 N·m인지)와 정규화 | "raw effort values"(Sec. IV-B)라고만 함. k_t 값 없음 |
| 행동 차원 처리(한 팔 7차원을 14 슬롯에 어떻게 넣었는지) | 없음 |
| temporal ensembling 사용 여부, chunk 실행 방식 | 없음 |
| 평가 순서(변형별 블록인지 섞었는지), 실패 판정 시간 한도 | "predefined time horizon"이라고만 함(Sec. V-D) |
| Board Wipe 성공의 "줄무늬" 판정 기준, 닦기 중 접촉력 | 없음 |
| Foam Stop의 J(θ) 계산, F_x 중력 기준값, 성공 판정 | 그림만 있음 |
| ALOHA Solo 서보 게인(P, I, D)과 소프트웨어 버전 | 없음(원조 ALOHA 기준 D1로 추정) |
| 하이퍼파라미터 선택 과정 | "적어도 한 변형이 성공 rollout을 내는지 눈으로 확인"(Sec. V-E) |

## 4. 구현할 수 있는지

### 4.1 공식 코드로 재현
**불가능합니다.** 공식 코드·데이터·체크포인트가 없습니다. 하드웨어도 필요합니다: ALOHA Solo(Interbotix ViperX-300s follower + WidowX-250s leader), 카메라 3대(BEV, 측면, 손목), 과제 소품(지우개·보드, 잠금식 접이 병, 멀티탭과 플러그, 검은 폼). 논문은 학습에 RTX 4090·A6000·A5000, 추론에 RTX 4090을 썼습니다(Sec. V-A).

### 4.2 이 맥(Apple M4 Pro, 통합 메모리 약 24 GiB, MPS, CUDA 없음)에서

| 단계 | 판단 |
|---|---|
| Table IV 재계산, 물리 toy(T1–T4), 그래프 | ✅ 완료(numpy·scipy, 1–2초) |
| 1-DoF 폐루프 toy 확장(시연 생성 → 간단한 회귀 정책 4종 → P 서보에서 재생) | ✅ 설치 없이 가능(numpy, 수 분) |
| 공개 ALOHA 데이터(LeRobot 허브)의 effort·action·qpos로 δ–effort 관계와 접촉 검출 분석 | 조건부. CPU로 충분하지만 데이터 다운로드 승인 필요 |
| 시뮬레이션(MuJoCo, gym-aloha 계열)에서 ACT 네 변형 학습 | 조건부. ACT는 작아서(hidden 512, ResNet18 백본) MPS로 학습 가능해 보이지만 torch·mujoco 설치 승인이 필요하고, 원조 ACT 코드는 `.cuda()` 하드코딩(D8)이라 수정하거나 LeRobot 구현을 써야 함. 8,000 epoch 전체 재현은 수 시간–하루 이상으로 예상(미측정) |
| 실물 재현 | 불가. ALOHA Solo 하드웨어 필요 |

장벽은 GPU보다 **하드웨어와 접촉 과제 데이터**입니다. ACT 자체는 작은 모델이라 GPU가 결정적이지 않습니다. 다만 시뮬레이터의 접촉·서보 모델이 실물 Dynamixel의 전류·마찰과 다르므로, 시뮬레이션 결과로 "전류 대리값이 충분하다"를 확인할 수는 없습니다.

### 4.3 아이디어만 다시 구현
**쉽습니다.** 기존 ACT·LeRobot 파이프라인에서 세 줄짜리 변경입니다.
- **ACT + τ:** proprio 입력에 effort를 이어 붙임(정규화 통계만 추가). ALOHA 데이터에는 effort가 이미 있음(D4).
- **ACT-o:** 목표를 `action` 대신 미래 `qpos`로 바꿈.
- **ACT-o + τ:** 둘 다.

어려운 부분은 구현이 아니라 **평가**입니다. 이 논문의 빈칸을 채우려면 (1) 닦기 중 실제 접촉력(외부 F/T나 effort 로그), (2) qpos 이력·qvel 입력 대조군, (3) 여러 시드, (4) 판별 과제의 Youden J, (5) 재시도 없는 1회 성공률을 함께 보고해야 합니다. 그리고 `03` §1의 해석대로라면, leader가 없는 데이터에서 Base ACT식 구동 여유를 되살리는 방법은 τ를 입력에 넣는 것보다 **목표에 넣는 것**(q_target = q_follower + K_p⁻¹(τ − τ_g(q)), DexForce식)이 더 직접적입니다.
