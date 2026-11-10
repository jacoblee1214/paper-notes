# DexTouch-WM 논문 분석 (arXiv:2609.20649)

**논문:** DexTouch-WM: Learning Action-Conditioned Tactile World Models from Human Touch for Dexterous Robot Manipulation
**저자:** Yan Qin, Yue Chen, Wenwei Lin, Shujia Liu, Chuqiao Lyu, Kailun Su, Weiyang Jin, Chenze Yu, Ping Luo, Wenbo Ding, Tianxing Chen, Renjing Xu (12명, 공동 제1저자 4명 ∗, 교신 3명 †)
**arXiv:** 2609.20649 · 최신 v2 (2026-09-18) · cs.RO · 본문 8쪽, 부록 없음
**학회 위치:** arXiv 댓글 "Accept to IROS 2026 Workshop RoBoWoMo (Lightning Talk)". **IROS 2026 본 트랙 논문이 아니라 워크숍 논문(라이트닝 토크)입니다.** 실험 규모와 보고 깊이는 본 트랙 기준보다 얇게 읽어야 합니다.
**라이선스:** arXiv 비독점 배포 1.0 (nonexclusive-distrib). CC BY가 아니므로 논문 그림은 개인용 리포트·PPT에만 쓰고 공개 산출물에는 넣지 않습니다(README 참고).

---

## 1. 한 문단 요약

사람 손에 건 압전저항 장갑(촉각 360 taxel/손, 그중 320개 사용)과 로봇 손(Wuji 20-DoF)에 같은 센서 배치를 달고, 사람 손 motion을 로봇 행동 공간으로 리타게팅(IK)합니다. 이렇게 하면 사람 데이터를 로봇용 행동조건 세계모델 학습에 그대로 쓸 수 있다는 것이 핵심 주장입니다. 모델은 사전학습 비디오 DiT(Wan2.2-TI2V-5B)에 경량 촉각 전문가(TactileDiT)를 Mixture-of-Transformers 방식으로 붙여 RGB와 양손 촉각을 함께 예측합니다. 실험의 핵심 질문은 "로봇 데이터를 5시간으로 고정하고 사람 데이터를 0→10→50→100시간으로 늘리면 로봇 도메인 예측이 좋아지는가"입니다(Table I). 결과는 좋아지는 방향이지만, 단일 실행·시드 미보고·검정 없음이라 "확실한 스케일링 법칙"이라고 하기는 어렵습니다. 정책 평가(Table II–III)와 데이터 생성(Table IV–V)은 정책 3개, 과제 4개 규모라 통계적으로 약합니다.

## 2. 문제 설정 (§III-A)

- 식 (1): p_θ(v_{1:T}, x_{1:T} | v_0, x_0, a_{1:T}, ℓ). 초기 RGB v_0, 촉각 x_0, 언어 ℓ, 미래 자세 a_{1:T}를 받아 미래 RGB와 촉각을 예측.
- 행동은 두 도메인 모두 같은 67차원 표현: 양 손목 motion, 양 손 20-DoF, 머리 카메라 motion (식 (6)). 67차원 구성 요소별 차원 수는 본문에 없어 추정하지 않았습니다.
- 사람 데이터: 25-keypoint 손 골격 + 손목 6-DoF(VIVE tracker) + 손 관절(MANUS) + 머리·손목 카메라. 장갑은 Moxian 촉각 장갑을 MANUS MetaGloves 아래 착용.

## 3. 방법

### 3.1 구조 (§III-B, Fig. 2)
- **시각 전문가:** Wan2.2-TI2V-5B DiT와 frozen video VAE. 첫 프레임은 깨끗한 잠재(시간 index 0)로 유지하고 이후 잠재만 노이즈를 넣어 예측.
- **촉각 전문가 (TactileDiT):** 가벼운 DiT, frozen Split-Hands 인코더·디코더. 두 전문가는 bi-modal joint attention으로 정보를 교환하고, AdaLN과 FFN은 모달리티별로 따로 씀(MoT [10]).
- **조건:** 언어는 frozen text encoder, 자세는 AdaLN(식 (3)). 비디오는 4프레임 chunk, 촉각은 원래 프레임율로 조건을 넣음(dual-rate).

### 3.2 해부학적 촉각 토큰화 (§III-B "Anatomy-aware")
- 손당 320 taxel = 엄지~소지 손끝 5개 × 4×4(=80) + 손바닥 15×16(=240) (Fig. 3). 80+240=320으로 맞음.
- 조각 격자(Grid)는 손끝과 손바닥을 한 캔버스에 이어 붙임. 논문은 이것이 "인공적인 이웃"을 만든다고 보고, 손끝 공유 인코더 1개와 손바닥 공유 인코더 1개 + 패드 정체성 임베딩(Split-Hands)을 씀.
- Split-Hands는 contact 가중 재구성으로 사전학습 후 동결. 가중을 둔 이유는 비접촉 taxel이 대부분이라 표현을 지배하기 때문.
- 근거: Fig. 4 (Grid 대비 재구성 손실 낮고 Contact-IoU 수렴 빠름). **Fig. 4는 곡선만 있고 최종 수치가 없습니다.**

### 3.3 잔차 촉각 잠재 (식 (2))
- z_t = E_x(x_t), r_t = z_t − z_0, x̂_t = D_x(z_0 + r̂_t). 초기 촉각 배경 압력을 빼고 상호작용으로 생긴 변화만 예측.
- 손실 식 (4): L = L_video^FM + λ_tac L_tac^FM. 촉각 항은 "접촉 또는 시간에 따라 변하는 토큰"에서만 계산(마스크). **λ_tac 값과 마스크 기준(임계값)은 본문에 없습니다.**

### 3.4 사람→로봇 행동 정렬 (§III-D)
- 식 (5): (e^h_t, J^h_t) → (e^r_t, q^r_t), 사람 손목 자세와 손 기하를 캘리브레이션된 카메라-세계 정합 후 IK로 WujiHand2 (20-DoF) 자세에 매핑.
- 행동은 전부 첫 프레임 기준 상대 손목 motion (식 (6)).
- **리타게팅 오차를 정량화한 결과는 없습니다.** 키포인트 오차, 접촉 위치 불일치, 물체 자세 오차 등 어느 것도 보고하지 않습니다. "사람 촉각이 로봇에서 전이된다"는 주장이 이 단계에 기대고 있어서 가장 큰 빈칸입니다.

### 3.5 학습
- 조건부 flow matching [11]. 학습 스텝, 배치, 학습률, GPU 종류, 시드 수 모두 본문에 없음.

## 4. 실험

### 4.1 데이터 (§IV-A)
- 사람: 본문 두 곳이 어긋납니다. Introduction과 §IV-C는 "50 tasks", §IV-A 첫 문단은 "54 tasks". 같은 데이터인지 확인할 수 없어 상충으로 기록합니다. 총 약 100시간.
- 로봇: 10 과제, Tianji 팔 + Wuji 손. 6과제 5시간(사전학습), 나머지 4과제는 다운스트림. 6과제 held-out 로봇 에피소드 수는 본문에 없습니다.
- **모호한 점:** "Human and robot task sets are disjoint"라고 하는데, 평가하는 held-out 로봇 에피소드 6과제가 사전학습 6과제와 같은지 다른지 명시하지 않습니다. 같은 과제라면 "교차 과제 전이"보다는 "사람 데이터가 같은 로봇 과제의 예측을 돕는다"에 가깝습니다.

### 4.2 평가 지표 (§IV-A)
- 시각: PSNR, SSIM, LPIPS, Aesthetic/Image Quality(WorldArena [17]), JEPA similarity, Subject consistency, Semantic alignment, Trajectory accuracy, Geometry error. **Trajectory accuracy와 Geometry error의 정의는 본문에 없습니다.**
- 촉각: PSNR_taxel, SSIM_taxel, LPIPS_heatmap, MSE, MAE, Contact-MAE, Contact-IoU, Contact-F1. **접촉의 임계값, Contact-MAE가 접촉 칸만 보는지 여부 등 정의가 없습니다.**

### 4.3 Table I: 설계와 사람 데이터 스케일링 (§IV-B, §IV-C)
- 열: V-only, Cross-Attn, Robot-only(= Split-Hands + AdaLN), +10h, +50h, +100h(= DexTouch-WM). 모두 로봇 5시간 고정.
- 단일 값만 있고 분산·시드·신뢰구간이 없습니다. 재계산 결과(03 §1)는 본문 수치와 맞습니다(Table I의 소수 셋째 자리까지 재현).
- 핵심 수치: PSNR 23.47→27.10 (+3.62 dB), Trajectory Accuracy 0.891→0.962, Geometry 0.131→0.100, PSNR_taxel 24.81→29.18, Contact-IoU 0.415→0.588, Contact-F1 0.551→0.706.
- 주의할 점: Aesthetic 0.400→0.414, Subject Consistency 0.751→0.751(변화 없음), Semantic Alignment는 +50h(0.866)가 +100h(0.860)보다 높음. 본문 "visual metrics saturate between +50h and +100h"는 PSNR(+0.22 dB)·Trajectory(0)에서는 맞지만 JEPA는 +0.007 더 오르고 Semantic은 떨어져 "포화"로 한 줄 요약하기는 어렵습니다.

### 4.4 Table II–III: 정책 평가 (§IV-D)
- 4과제 × 정책 3개(FTP-1, π0.5, X-VLA). 과제당 롤아웃 10개, 각 롤아웃을 사람 평가자 5명이 채점, 평균.
- 세 환경: 실제 로봇, WM-Robot(200 로봇 궤적으로 적응), WM-Mix(로봇 100 + 사람 100).
- Table III: Pearson r과 MMRV. **Pearson은 정책 3개로 계산되므로 자유도가 1입니다.** p<0.05가 되려면 |r|≥0.997이 필요하고, 본문의 최댓값 0.972(Place Shoes, WM-Mix)도 양측 p≈0.15입니다(03 §2). 평균 Pearson 0.844(WM-Mix) 대 0.646(WM-Robot)의 차이는 검정할 수 없습니다.
- MMRV는 공식(식 (7))대로 재계산하면 본문과 맞습니다(03 §2).
- 시뮬레이터 편향: 12개 셀에서 WM-Robot 점수가 실제보다 평균 +0.12, WM-Mix는 +0.18 높습니다(03 §2.4). 순위는 맞출 수 있어도 절대 점수는 과대평가됩니다. 본문은 이 편향을 언급하지 않습니다.
- 이 평가에서 WM-Mix는 사람 데이터를 로봇 적응 데이터의 절반으로 바꿉니다. 로봇 적응 데이터가 WM-Robot보다 100개 적으므로, 이득의 원인이 사람 데이터인지 과제 데이터량 차이인지 분리되지 않습니다. 그래도 WM-Mix의 과제 데이터에 사람 시연이 "과제 특화"로 들어갔다는 점은 본문에 "task-specific human demonstrations, disjoint from HumanTouch"라고 되어 있어 사전학습 사람 데이터와는 다릅니다.

### 4.5 Table IV–V: 합성 데이터로 정책 학습 (§IV-E)
- 실제 200 궤적 vs (실제 100 + 합성 100). **대조군은 실제 데이터가 두 배입니다.** 실제 100 + 합성 0 대조가 없어서 합성 데이터의 순효과를 알 수 없습니다.
- 4과제 평균: FTP-1은 실제 0.731, +WM-Mix 0.694(−0.037). π0.5는 0.625 → 0.494(−0.131). X-VLA는 0.500 → 0.400(−0.100). +WM-Robot은 FTP 0.688, π0.5 0.563, X-VLA 0.506.
- 본문은 "Human-adapted world models can supply useful demonstrations"라고 씁니다. 표를 보면 π0.5와 X-VLA에서는 WM-Mix가 오히려 떨어집니다. 이 문장은 FTP-1에서만 근거가 있습니다.
- Table V(정책 Pearson·MMRV)는 본문에서 따로 논의하지 않습니다. WM-Mix의 Place Phone Pearson은 0.277로 낮은데 MMRV는 0.067입니다. 상관과 순위 지표가 엇갈립니다.

### 4.6 Figure 5 (생성 시연)
- 정성 비교만 있습니다(GT와 두 모델의 RGB·촉각 프레임). 접촉 타이밍이 맞는다는 주장은 시각 확인에 의존합니다.

## 5. 논문이 답하는 것과 답하지 않는 것

| 질문 | 논문의 답 | 근거 수준 |
|---|---|---|
| 촉각을 함께 예측하면 시각 예측도 나아지나? | Cross-Attn은 정책 지표를 낮추고 AdaLN(Robot-only)이 회복 | 단일 실행, 차이 0.04(Traj. Acc.) |
| 사람 촉각 데이터가 로봇 예측을 돕나? | 100시간에서 시각·촉각·기하 지표 개선 | 단일 실행, 시드 없음 |
| 사람 데이터를 정책 평가에 쓸 수 있나? | 상관은 높아 보이나 정책 3개 | 자유도 1, 유의하지 않음 |
| 합성 데이터로 정책을 학습할 수 있나? | 정책마다 다름. FTP-1만 ≈, π0.5·X-VLA는 ↓ | 실제 데이터량이 다른 대조, 롤아웃 10개 |
| 리타게팅이 충분히 정확한가? | 보고 없음 | 없음 |
| 두 장갑 센서가 같은 신호를 내나? | "같은 센서 배치"만 말함. 보정 방법과 도메인 격차 측정 없음 | 없음 |

## 6. 논문 스스로 밝히는 한계

- 본문에 한계(Limitations) 절이 없습니다. 결론은 "scalable human interaction is a complementary axis"라고만 씁니다.
- 공개 코드·가중치·데이터 공개 문장도 없습니다(arXiv 페이지에도 프로젝트 링크 없음).

## 7. 저자 주장 대비 읽을 때의 주의

1. 헤드라인 수치인 "100h 대 5h"는 한 번 실행한 결과입니다. 1–2% 차이가 아니라 PSNR +3.6 dB, Contact-IoU +0.17이라 방향은 강하지만, 시드 간 분산이 얼마나 되는지는 본문에 없습니다.
2. 사람 데이터는 "같은 센서 배치"이지만 장갑 위에서 손가락 압력이 달라질 수 있습니다. 로봇 손끝의 강성·형상과 사람 손끝의 차이도 접촉 분포를 바꿉니다. 이 차이를 측정하지 않았습니다.
3. 정책 평가 결과를 "evaluator consistency"라고 부르지만 측정 단위가 정책 3개입니다.

## 8. 재현 가능성 (요약, 상세는 04 §4)

- 코드, 가중치, 데이터셋 모두 공개되지 않은 것으로 확인했습니다(04 §1).
- 학습 세부(스텝, 배치, λ_tac, 마스크 기준, Contact 임계값)가 없어 같은 구조를 만들어도 같은 수치가 나오지 않을 가능성이 큽니다.

## 9. 이 저장소의 관련 논문과의 연결 (상세는 02 §3)

- 미래 촉각 예측 측면: contact_grounded_policy(접촉 기반 정책), restacvla(촉각 잔차 잠재).
- 사람→로봇 리타게팅: flashdexretarget, gnr.
- 사람 영상→로봇 학습: dreamx.
- 세계행동모델: openwam, agentic_wam (Wan 계열 백본 사용).
- 촉각 정책 대안: gpt6_astra_tactile.
