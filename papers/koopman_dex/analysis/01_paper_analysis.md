# 01. 논문 분석: Going with the Flow (K-UBM)

*Going with the Flow: Koopman Behavioral Models as Pseudo Planners for Visuo-Motor Dexterity* (Yunhai Han, Jiaqi Fu, Linhao Bai, Ziyu Xiao, Zhaodong Yang, Yogita Choudhary, Krishna Jha, Chuizheng Kong, Shreyas Kousik, Harish Ravichandar; Georgia Institute of Technology). arXiv:2602.07413, 분석 대상 **v4 (2026-11-20, 29쪽)**. 쪽과 표 번호는 v4 기준입니다.

## 0. 판본, 학회, 권리

### 0.1 판본과 차이

| 판본 | 날짜 | 쪽수 | 제목과 주요 내용 |
|---|---|---|---|
| v1 | 2026-02-07 | 18 | 제목 "… as **Implicit** Planners …". 시뮬 7과제, 실로봇 **2과제**(Kinova 팔 + PSYONIC 손, Uncover Pot과 Open Lid, 기준선 ACT 하나, 시드 하나; Table III에서 ACT 4/10, 4/10 대 K-UBM 10/10, 8/10). 초록 "matches or exceeds", 본문 "K-UBM achieves the **highest** mean success rate" |
| v2 | 2026-02-10 | 18 | v1과 본문 같음(제목 줄과 저자 표기만 다름) |
| v3 | 2026-06-09 | 29 | 제목 "… **Pseudo** Planners …". 실로봇을 Franka + LEAP 4과제로 교체, 기준선 Diffusion과 ACT, 시뮬 기준선에 UVA, KODex, KOROL 추가, 반응성 실험을 7과제로 확대(표 9). 본문 "**second highest** mean success rate"(UVA가 더 높음). 초록은 여전히 "matches or exceeds" |
| v4 | 2026-11-20 | 29 | CoRL 2026 최종본. 초록을 "performs comparably to state-of-the-art baselines … considerably smoother executions, fast inference, efficient learning"으로 바꿈. **§4.3 구조 절제(그림 4)가 새로 추가**됨(v3의 §4.3은 "Additional Experiments"). 부록 J 절제가 "두 개 실로봇 과제"에서 "네 개"로 바뀜. 표 3–6, 9, 10의 수치는 v3와 같음 |

- v1/v2 → v3: 실험 설정 전체(로봇, 과제, 기준선)가 바뀌었고 "highest"가 "second highest"가 됐습니다. v3 → v4: 수치 표는 그대로이고 초록의 주장 강도가 낮아졌습니다("matches or exceeds" → "performs comparably"). v3에 없던 구조 절제가 v4에서 처음 나옵니다.
- 이 문서의 수치는 v4 표와 그림에서 가져왔습니다. 그림 3과 4의 막대는 숫자가 적혀 있지 않아 픽셀 판독한 값(오차 약 1%p)이고 그렇다고 표시했습니다.

### 0.2 학회

arXiv 댓글: "Accepted by CoRL 2026. Website: k-ubm.github.io". v4 첫 쪽 하단에 "10th Conference on Robot Learning (CoRL 2026), Austin TX, USA". 이 저장소의 `conferences/corl2026/papers.json`(corl.org 공식 채택 목록 수집분)에 키 2602.07413, 포스터, 세션 "Poster post-p3", 11/11(수) 10:30–12:00, 포스터 번호 123, `in_official: true`로 있습니다. 학회 표기는 **CoRL 2026 포스터**입니다.

### 0.3 권리

arXiv abs 페이지의 라이선스 링크는 `arxiv.org/licenses/nonexclusive-distrib/1.0/`(arXiv 비독점 배포 라이선스, CC가 아님)입니다. 따라서 이 분석의 공개본에는 논문 그림을 넣지 않고 우리가 그린 그래프만 씁니다. PDF 본문에서 ©, 저작권, All rights reserved 문구는 찾지 못했습니다(`pdftotext`로 확인).

## 1. 문제와 주장

- **문제.** 다지 손(multi-fingered) 시각-운동 모델은 확산(diffusion)과 트랜스포머 정책에 기대는데, 데이터와 계산이 많이 들고(Abstract), 기술을 반응형 사상 o_t → a_t로 보며 고정 길이 액션 청킹을 쓴다는 점이 "시간적 일관성과 반응성의 경직된 교환"을 만든다고 봅니다(§1).
- **제안.** 기술을 로봇 행동 흐름(action flow)과 시각 특징 흐름(visual flow)이 함께 진화하는 결합 동역학계로 표현하는 **Unified Behavioral Models (UBM)**(§2)과 그 첫 사례인 **Koopman-UBM (K-UBM)**(§3). 초기 조건만으로 전체 기술 구간을 잠재 공간에서 선형 롤아웃해 계획처럼 쓰므로 "pseudo planner"라 부릅니다.
- **반응성.** 예측한 시각 흐름과 관측이 임계 이상 벌어지면 다시 계획(replan)하는 사건 기반 방식(§1, 부록 I, K).
- **주장의 강도(v4).** 시뮬 7과제와 실로봇 4과제에서 SOTA 기준선과 "comparable"이면서 더 매끄럽고, 빠르고, 가림에 강하다(Abstract, §1, §6).

## 2. 방법

### 2.1 UBM 정의 (§2)

데이터 D는 N개 시연 {(o_t, a_t)}, o_t = {q_t, I_t}(관절 상태, RGB), a_t = q_target(관절 목표). 반응형 정책은 π: o_{t−H:t} → a_{t:t+C}. UBM은 통합 행동 상태 ξ_t = [a_t, o_t]를 인코더로 z_t = f_enc(ξ_t)로 올리고 z_{t+1} = f_UBM(z_t), a_t = f_dec(z_t)로 읽습니다. 초기 조건 ξ_0에서 z_0를 만들고 f_UBM을 t_p ≤ T̄ 스텝 굴리면 전체 행동 궤적을 얻습니다. 세계 모델 x_{t+1} = f_WM(x_t, a_t)와 달리 임의 행동의 영향이 아니라 시연 행동과 환경 변화의 결합을 모델링한다고 주장합니다.

### 2.2 K-UBM (§3, 부록 B, F, G)

1. **시각 특징 φ_t** (두 종류). (i) Object Flow: SAM3로 객체 마스크를 얻고 CoTracker로 256점을 추적해 16×16×2 격자로 놓은 뒤, 스트라이드 2 합성곱 하나(2→8채널, 16→8)와 1×1 합성곱으로 2×8×8 = **128차원**으로 압축하는 오토인코더(부록 B). (ii) DynaMo: ResNet-18, 자기지도 동역학 일관성 손실(부록 B, 식 2–4, λ = 0.04).
2. **통합 상태** ξ_t = [a_t; φ_t] ∈ R^{dξ}. φ_t는 행동과 평균 ℓ2 노름이 같아지도록 상수배 조정합니다(§3.2).
3. **상태 포함 리프팅** z_t = [ξ_t; ψ_t], ψ_t = g_θ(ξ_t)(MLP). d_z = d_ξ + d_ψ. 디코더 없이 z의 앞 부분을 그대로 읽습니다("decoder-free", §3.2).
4. **선형 동역학** z_{t+1} = K z_t, K ∈ R^{d_z×d_z}.
5. **다단계 손실** L = E_t[ Σ_{l=1}^{H} ‖K^l z_t − z_{t+l}‖²₂ ] (§3.2). 안정화 요령 둘: K의 학습률을 인코더보다 낮게 + 기울기 클리핑, K를 **항등 행렬로 초기화**(§3.2, 부록 J).
6. **데이터 수정** (부록 G): 첫 행동이 시연마다 다르므로 a_0 = q_1, I_0 = I_1로 두는 보조 초기 스텝을 넣습니다.
7. **재계획** (부록 K): d_t = ‖centroid(관측 흐름) − centroid(예측 흐름)‖, |d_t − d_{t−1}|이 이전 점프 평균의 **3배**를 넘으면 트리거. "절대 픽셀 임계가 없다"고 씁니다.

표 7(시뮬 하이퍼파라미터): 은닉층 (128, 256), 리프팅 256, ReLU, 인코더 lr 5e-4, K lr 5e-5, 다단계 길이 15, 항등 초기화. 실로봇은 예측 구간만 50(부록 F).

표 2(KODex 다항 리프팅 차원)는 닫힌 식으로 재현됩니다(`03` §1.8).

## 3. 실험

### 3.1 시뮬레이션 (§4, 부록 C–E)

- **과제.** DexArt(SAPIEN; XArm6 + Allegro; Bucket, Laptop, Toilet; 학습 30·테스트 30 시연)와 Adroit(MuJoCo; Door, Tool, In-hand reorientation, Relocation; 학습 50·테스트 30). 시연은 사전학습된 RL 전문가에서 수집(§4.1).
- **기준선.** Diffusion Policy, ACT, UVA, KODex, KOROL(§4.1). 시각 특징을 모든 방법에 같게 주었다고 씁니다(UVA는 원시 영상). 기준선 구조는 부록 E에서 **과제별 최적을 테스트 성공률로** 고릅니다(표 3–6).
- **시드.** 5개(KODex는 해석해라 1개). 그림 3 오차 막대는 시드 간 표준편차.
- **결과(그림 3).** K-UBM의 7과제 평균은 Flow 약 73.7%, DynaMo 약 73.0%(막대 판독). UVA는 약 83.7%와 84.0%로 더 높습니다. 본문 표현은 "second highest mean success rate"(§4.2). Door에서 K-UBM은 약 46%와 54%인 반면 UVA는 약 90%입니다(그림 3, 판독).
- **추론 시간(표 1, Bucket, 스텝당 정책 쿼리 평균 ms, 시각 특징 추출 제외).** K-UBM 0.42(Flow)/0.47(DynaMo), KODex 0.40/0.37, KOROL 0.41/0.36, ACT 0.52/0.29, Diffusion 29.95/36.31, UVA 98.6.
- **구조 절제(그림 4, v4 신규).** 7과제 × 3시드, Flow 특징. L-L(상태 제외), NL-L+S(비선형 동역학), NL-S(MLP, LSTM, Transformer)와 비교해 K-UBM이 평균 최고(그림 눈금 판독으로 다른 변형은 약 27–50%, K-UBM 약 73%). 같은 K-UBM의 평균(그림 4 약 73%)이 5시드 그림 3의 값과 같은 범위입니다(`03` §1.4).

### 3.2 실로봇 (§5, 부록 C.3, M)

Franka Research 3 + LEAP 손(23차원 상태). 과제 4개: 꽃 꽂기, 칩 붓기, 화이트보드 지우기, 저녁 서빙. 과제당 **시연 20개**(키네스테틱 교시), 기준선 Diffusion(2,8,16)과 ACT(40). 표 10: 성공률(%) Diffusion 70/100/40/80 (평균 72.5), ACT 100/90/80/100 (92.5), K-UBM 90/70/100/80 (85.0). 값이 모두 10의 배수라 **과제당 10회 평가**로 읽힙니다(논문에 시행 수 명시 없음; jerk 셀이 "10 runs"). 팔 jerk 평균 44.69/28.42/4.98, 손 jerk 14.16/16.56/4.14 (rad/s³). 추론 시간 26.4/0.058/0.017 ms(그림 5). 본문: "ACT가 가장 높은 평균 성공률이고 K-UBM은 경쟁력이 있다(85.0% vs 92.5%)", 팔 jerk 6–9배, 손 jerk 3–4배 감소.

### 3.3 반응성 (부록 I, 표 8–9)

과제별로 롤아웃 중 한 번 객체 위치나 목표를 바꿈(표 8). 7과제 × 30회 = 210회. 성공 수(표 9): Diffusion 23/22/13/0/12/1/16, ACT 24/23/11/10/9/0/10, K-UBM 15/17/7/15/23/17/27. 평균 0.41, 0.41, 0.58(K-UBM 최고). 트리거 검출 RT: K-UBM 22/27/17/30/29/30/30 (평균 0.88). 실패 원인 두 가지를 적습니다: 2D 흐름 휴리스틱이 3D 변화에 둔감한 트리거 실패, 데이터에 없는 전이 행동이 필요한 재계획 실패.

### 3.4 기타

- **센서 열화(부록 L, 그림 13).** 이미지를 검게 만들면 Diffusion과 ACT의 성공률이 크게 떨어집니다(Door만 예외). **K-UBM은 이 실험에서 제외**("open-loop라 필요 없다").
- **Koopman 학습 요령(부록 J, 그림 11).** 같은 "best-performing random seed"로 4조건(identity init × separate LR) 비교. 별도 학습률이 없으면 학습 손실과 스펙트럼 반경이 불안정하고, 항등 초기화가 없으면 안정하지만 효율이 낮다고 보고.
- **시각 특징 예측(부록 H, 그림 7–9).** 실패 롤아웃이 성공 롤아웃보다 흐름 RMSE가 크다.

## 4. 한계와 근거의 간격

저자 서술(§6): 선형 동역학이 급격한 접촉 불연속을 뭉개고, 확률성이 큰 환경에서는 개루프 계획이 부족할 수 있으며, 시각 피드백에 의존하는 재계획이 가림과 외란이 겹치면 늦다.

이 분석에서 본 간격(자세한 수치는 `03`, `04`):

1. **"비교 가능"의 범위.** 시뮬 평균은 UVA에 약 10%p 뒤지고 Door에서 크게 뒤집니다. 실로봇은 시행이 40회뿐이라 K-UBM 대 ACT를 가르지 못합니다(Fisher p = 0.48).
2. **"효율적 학습" 문구.** v4 초록과 §1에 있으나 학습 시간, 메모리, 데이터 효율 비교는 본문과 부록에 없습니다(UVA는 "학습 시간이 길어서" 실로봇에서 제외했다고만 §5에 있음).
3. **"가림 강건" 문구.** 가림 실험은 반응형 정책만 대상이고 K-UBM에는 하지 않았습니다. K-UBM이 재계획할 때는 같은 시각 입력에 의존합니다(한계 3).
4. **기준선 선택 방식.** 기준선은 과제별로 테스트 성공률 기준 최적 구조를 골랐고(부록 E), K-UBM은 한 설정을 모든 과제에 썼습니다. 방향은 기준선에 유리한 쪽입니다.
5. **재현에 필요한 정보.** 공식 저장소에 K-UBM 본체와 시뮬 평가는 있으나 기준선, 실로봇, 반응성 실험 입력, 지터 계산은 없습니다(`04`).
