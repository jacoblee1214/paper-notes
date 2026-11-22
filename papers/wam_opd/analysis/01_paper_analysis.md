# 01. 논문 분석: WAM-OPD

분석 대상은 *WAM-OPD: Joint Video-Action Supervision for World Action Model Post-Training with On-Policy Distillation* (arXiv:2608.22364 **v2**, 2026-10-01)입니다. 인용은 논문 절·표·그림 번호와 `paper/paper.txt` 기준입니다. 공식 코드(UCL-ERL/WAM-OPD, Apache-2.0)와의 대조는 `04`에 있습니다.

## 0. 메타와 라이선스

| 항목 | 내용 |
|---|---|
| 제목 | WAM-OPD: Joint Video-Action Supervision for World Action Model Post-Training with On-Policy Distillation |
| 저자 | Liuhaichen Yang, Zhengyang Zhong, Hanshang Zhu, Ningwei Bai, Qichen Yin, Zhi Han, Jiarui Qin, Zhuang Jiang, Chenchao Sheng, Hanbo Ma, Junkai Liu, Junkai Sun, Dongcheng Lyu, Bo Liu, Yi Dong, Zezhi Tang (교신) |
| 소속 | UCL, 상하이교통대, 맨체스터대, 중국과학원 난징소프트웨어기술연구원, 선전첨단기술대, 리버풀대 (쪽 1) |
| 분야 | cs.AI (주), cs.RO |
| 버전 | v1 2026-08-23 (1,220 KB), v2 2026-10-01 (3,635 KB). 최신 v2를 분석 (v1과의 차이는 §6) |
| 학회 | **학회 표기 없음 (arXiv).** arXiv 주석·journal-ref 없음, 연구실 공개 목록(ucl-erl.github.io/publication.html)이 "arXiv preprint"로 표기, CoRL 2026(`conferences/corl2026/papers.json` 755건)과 IROS 2026 목록(518건)에 없음, 제목 웹 검색에서 채택 정보 없음 (확인 2026-11-22) |
| 코드 | https://github.com/UCL-ERL/WAM-OPD (Apache-2.0, 커밋 `191b2e3`). 별도 프로젝트 페이지는 없음 |
| 라이선스 | arXiv abs 페이지의 라이선스 링크가 `http://creativecommons.org/licenses/by/4.0/`(**CC BY 4.0**, 확인 2026-11-22). PDF 첫·끝 쪽에 ©·All rights reserved·회사 고지가 없음(`pdftotext` 검사). 쪽 머리의 "ERL"은 연구실 로고 |

**그림 게시 판정:** 일반 규칙상 CC BY 4.0이라 가능하지만 그림마다 다릅니다. 그림 4(막대그래프)는 저자 자체 도표라서 출처를 달아 게시본에 넣습니다. 그림 1(모의 장면 + 실로봇 사진), 그림 2(작은 실로봇 카메라 프레임 포함), 그림 3(실로봇 사진 줄)은 실물·모의 프레임이 있어 게시본에서 뺍니다(개인용 리포트에만).

## 1. 한 문단 요약과 핵심 질문

월드 액션 모델(WAM)은 미래 영상과 행동을 함께 만듭니다. 질문은 "미리 학습한 WAM이 실행 중 마주치는 이력에서 더 강한 Teacher로부터 배울 수 있는가"입니다. 방법(WAM-OPD)은 Student가 환경에서 굴린 이력에 대해 Teacher가 영상 목표 Ô_T와 행동 목표 Â_T를 한 쌍으로 붙이고, Student는 자기 영상 예측을 stop-gradient로 행동 가지에 넣는 배포 계산 그대로 두 목표를 맞춥니다(§3.1–3.2, 식 (1)–(4)). 학습은 rank-8 LoRA(블록 0–29), 3 에폭, 학습률 2e-5, 유효 배치 4입니다(§3.3). Student는 Flash-WAM(1 영상 스텝, 1 행동 스텝), Teacher는 LingBot-VA(25/50 스텝)입니다(표 1).

보고된 결과: RoboTwin 2.0 12과제 평균 33.8% → 65.7%(+31.9 pp, 표 2), 실로봇 4과제 평균 51.4% → 64.6%(+13.2 pp, 표 3), 감독 구성 절제 3과제(그림 4)와 이력 출처 비교 3과제(표 4).

## 2. 방법

### 2.1 Student 이력과 Teacher 표지 (§3.1, 식 (1)–(3))

- 이력 h_{S,i} = (O_≤i^S, A_<i^S, ℓ). Student: Ô_i^S = f_θ^video(h), A_i^S = f_θ^action(h, sg(Ô_i^S)) (식 2).
- Teacher: Ô_i^T = f_T^video(h_{S,i}), Â_i^T = f_T^action(h_{S,i}, Ô_i^T) (식 3). Teacher 행동 목표는 **Teacher 자신의 영상**에 조건화되고 Student 행동은 Student 영상에 조건화됩니다. 논문은 이를 "Student 손실이 배포 계산과 정렬된다"고 설명합니다. 이 불일치(Teacher 계획 ≠ Student 계획)는 v1에서 한계로 명시했으나 v2 본문에는 한계로 적혀 있지 않습니다(§6).
- Teacher는 환경과 상호작용하지 않습니다.

### 2.2 손실 (§3.2, 식 (4))

- pseudo-Huber ρ_c(x, y) = sqrt((x−y)² + c²) − c, c = 1e-3. 영상은 영상 토큰 공간, 행동은 행동 공간.
- 속도 항: x_A(σ) = (1−σ)Â^T + σε, 목표 속도 v̂ = ε − Â^T. Student 속도 v^S를 **초기 행동 샘플링 상태 σ = 1(스케줄러 t = 1000)**에서만 평가합니다.
- L_M = m_v λ_v d_PH(Ô^S, Ô^T) + m_a [λ_a d_PH(A^S, Â^T) + λ_FM ||v^S − v̂||²]. (m_v, m_a) = (0,1) 행동 전용, (1,0) 영상 전용, (1,1) 결합. (λ_v, λ_a, λ_FM) = (1, 1, 0.2).
- c = 1e-3이면 오차가 1e-3보다 큰 대부분의 원소에서 ρ_c ≈ |x−y|로 L1 손실입니다(논문은 이 점을 말하지 않음).

### 2.3 최적화 절차 (§3.3, 알고리즘 1)

두 단계: (1) Student 롤아웃 K = 36개를 모으고 모든 이력에 Teacher 표지를 붙여 **고정 데이터셋**으로 저장, 24개(D_train)와 12개(D_cal)로 분할. (2) 환경을 다시 보지 않고 어댑터만 AdamW로 학습. 기본 가중치는 동결. 손실은 궤적 안에서 먼저 평균한 뒤 궤적을 동등하게 평균. 보정(calibration) 궤적에서 이른 시점과 늦은 시점 두 개씩을 앵커로 고정해 에폭마다 손실을 잽니다.

즉 알고리즘 1은 **롤아웃 한 번 + 오프라인 학습**입니다. DAgger식 반복(수집 → 학습 → 재수집)이 아니며, "on-policy"는 이력이 초기 Student가 만든 것이라는 뜻입니다. 논문 본문(§3.3, 알고리즘 1)에서 이 점은 구조로만 드러납니다. 모든 감독 모드는 같은 최적화 설정을 씁니다(§3.3).

## 3. 실험과 결과

### 3.1 설정 (§4.1)

- 시뮬레이션: RoboTwin 2.0 demo_clean 12과제, 과제마다 Student 롤아웃 36개(학습 24, 보정 12), **평가 60회/정책/과제**, "matched task seeds and the same noise bank"(§4.1 평가 규약).
- 실로봇: 4과제(Put Gripper Parts, Shake Bottle, Insert Screw Driver Tape, Put Tape Platform). LingBot-VA를 과제당 텔레오퍼레이션 40개로 적응시킨 뒤 증류해 Flash-WAM Student를 얻고 WAM-OPD를 적용. 정책당 과제당 **36 롤아웃**. 카메라 3대(손목 2, 머리 1), 물체 위치 변화 약 3 cm. 시뮬레이션과 같은 24/12 분할과 같은 최적화 예산.
- 지표: 성공률, 과제 평균은 동일 가중, 평균과 변화는 "반올림 전 성공 횟수로 계산"(§4.1 Metrics).

### 3.2 시뮬레이션 (§4.2, 표 2)

| 과제 | Teacher | Student | WAM-OPD | Δ(pp) |
|---|---|---|---|---|
| handover_mic | 80.0 | 1.7 | 35.0 | +33.3 |
| put_object_cabinet | 51.7 | 3.3 | 16.7 | +13.3 |
| place_fan | 96.7 | 21.7 | 50.0 | +28.3 |
| place_a2b_left | 90.0 | 11.7 | 53.3 | +41.7 |
| place_a2b_right | 90.0 | 16.7 | 65.0 | +48.3 |
| place_bread_basket | 88.3 | 25.0 | 60.0 | +35.0 |
| place_cans_plasticbox | 100.0 | 73.3 | 100.0 | +26.7 |
| stamp_seal | 96.7 | 16.7 | 81.7 | +65.0 |
| rotate_qrcode | 88.3 | 23.3 | 58.3 | +35.0 |
| place_container_plate | 98.3 | 93.3 | 98.3 | +5.0 |
| move_pillbottle_pad | 98.3 | 66.7 | 85.0 | +18.3 |
| place_object_stand | 96.7 | 51.7 | 85.0 | +33.3 |
| **평균** | **89.6** | **33.8** | **65.7** | **+31.9** |

12과제 모두 향상. 최대 이득 stamp_seal(+65.0). 본문: "larger Teacher–Student success gaps often coincide with larger WAM-OPD gains"(§4.2). Teacher 격차를 닫은 비율은 과제마다 0.28–1.0이며 WAM-OPD가 Teacher를 넘은 과제는 없습니다(`03` §2.3).

### 3.3 실로봇 (§4.2, 표 3)

| 과제 | Teacher | Student | WAM-OPD | Δ(pp) |
|---|---|---|---|---|
| Put Gripper Parts | 36.1 | 5.6 | 8.3 | +2.8 |
| Shake Bottle | 94.4 | 88.9 | 94.4 | +5.6 |
| Insert Screw Driver Tape | 66.7 | 52.8 | 83.3 | +30.6 |
| Put Tape Platform | 80.6 | 58.3 | 72.2 | +13.9 |
| **평균** | **69.4** | **51.4** | **64.6** | **+13.2** |

본문: 4과제 모두 "higher observed success"이고 Put Gripper Parts가 가장 어렵다(5.6 → 8.3).

### 3.4 감독 구성 (§4.3, 그림 4)

3과제(A2B Right, Bread Basket, Cans Plastic Box)에서 행동 전용 / 영상 전용 / 결합 성공률(%): 36.7 / 60.0 / 65.0, 58.3 / 51.7 / 60.0, 93.3 / 95.0 / 100.0. 결합이 최고 단일 설정을 5.0, 1.7, 5.0 pp 앞섭니다. 두 단일 설정 모두 Flash-WAM(16.7, 25.0, 73.3)보다 높습니다. 결합 값은 표 2와 같은 실행입니다(`03` §3).

### 3.5 이력 출처 (§4.4, 표 4)

| 과제 | Flash-WAM | Joint / Teacher 이력 | Joint / Student 이력 |
|---|---|---|---|
| place_a2b_right | 16.7 | 26.7 | 65.0 |
| place_bread_basket | 25.0 | 75.0 | 60.0 |
| place_cans_plasticbox | 73.3 | 100.0 | 100.0 |

"관계는 과제 의존적"이라고 결론짓습니다. Student 이력이 우세한 과제(A2B Right)와 Teacher 이력이 우세한 과제(Bread Basket)가 섞여 있습니다.

## 4. 저자가 말한 한계와 향후 과제 (§5)

v2에는 한계 절이 따로 없고, 결론이 "감독 구성은 과제에 따라 효과적인 신호가 다르다", "이력 출처 간 우열이 일관되지 않다"를 적습니다. 향후: Teacher 목표를 RL 보상과 결합, 다른 WAM 구조와 더 작은 Student, 흐름 매칭 VLA로의 확장(§5). 그림 3은 정량 비교가 아니라 과제 정의 예시라고 명시합니다(§4.1).

## 5. 주장과 증거의 틈 (우리 판단)

1. **절제 결론은 검정력이 부족합니다.** 결합과 최고 단일의 차이 5.0·1.7·5.0 pp는 3과제 모두 Fisher p ≥ 0.24이고 합산해도 3.9 pp, p = 0.48입니다. 60회로 5 pp 차이를 80% 검정력으로 보려면 조건당 약 1,470회가 필요합니다. 논문 문구는 "highest observed"로 조심스럽지만 초록 마지막 문장("demonstrate the value of paired video–action supervision")은 이 증거보다 강합니다(`03` §3).
2. **"최고 단일"은 사후 선택입니다.** 과제마다 더 높은 쪽을 고르면 결합에 유리한 편향이 생깁니다. 영상 전용 대 행동 전용은 A2B Right에서만 구별됩니다(p = 0.017, 보정 전).
3. **실로봇은 4과제 중 1과제만 개별적으로 확실합니다.** 개별 Fisher p는 Insert Screw Driver Tape만 0.011(Holm 보정 0.043)이고, Put Tape Platform 0.32, Shake Bottle 0.67, Put Gripper Parts 1.0입니다. 합산(144회)은 p = 0.031, 층화 검정 p = 0.006이지만 과제 4개뿐이고 짝 시드가 없습니다(`03` §2).
4. **평가 쌍 규모와 출처.** "60 matched trials"가 몇 개의 장면 시드와 몇 개의 잡음 은행인지 본문에 없습니다. 공개 저장소의 held-out은 장면 6 × 잡음 은행 2 = 12쌍이고, 60쌍에는 공개되지 않은 "extension48"이 필요합니다(`04` D4).
5. **체크포인트 선택이 서술에 없습니다.** 공개 파이프라인은 에폭 1–3을 선별 분할에서 골라 하나를 선택합니다. 논문은 "3 에폭"이라고만 적습니다(`04` D1).
6. **과제 선택.** 저장소는 "Teacher–Student 격차 자격(PASS)" 결정이 있는 과제만 바인딩하며 TASK_SPECS에는 12과제 밖에도 8과제가 있습니다. 12과제 선정 기준이 논문에 없습니다(`04` D2).
7. **Teacher가 공개 LingBot-VA 수치보다 낮습니다.** 같은 12과제의 공개 Easy 평균 96.5%에 대해 표 2의 Teacher는 89.6%이고, 6과제는 단측 이항 p < 0.01로 낮습니다(put_object_cabinet 51.7 대 85, handover_mic 80.0 대 94). Teacher 기준이 약해 보이는 과제일수록 "이득"의 상한이 낮아집니다(`03` §2.5).
8. **v1과 v2의 두 과제 값이 반대로 움직입니다.** 학습 궤적이 8+4에서 24+12로 3배 늘었는데 handover_mic 58.3% → 35.0%, put_object_cabinet 33.3% → 16.7%로 낮아졌습니다(Fisher p = 0.19, 0.23으로 구별되지는 않음). 평가 쌍 수와 어댑터가 모두 달라 원인은 분리되지 않으며 논문은 설명하지 않습니다(`03` §4).
9. **이득의 상당 부분은 Student의 낮은 출발점에서 옵니다.** 이득/격차 닫힘 비율은 평균 0.62이고 Student가 이미 높은 두 과제(place_container_plate 93.3, place_cans_plasticbox 73.3)는 천장에 닿았습니다. 격차와 이득의 Spearman은 0.76입니다(`03` §2.3).
10. **기준선 부재.** 같은 Teacher 예산의 SFT, 행동 전용 증류, 다른 OPD 변형, RL(SimpleVLA-RL·WAM-RL 등)과의 비교가 없습니다. Related Work가 이들을 언급하지만(§2) 실험에는 Student 전/후와 Teacher 참고만 있습니다.
11. **정의상 "OPD"의 범위.** 1회 수집 + 고정 데이터 오프라인 학습이므로 DAgger/OPD의 반복 구조는 없습니다(§2.3). 이득이 "on-policy 이력" 때문인지 단순한 Teacher 증류 때문인지는 표 4의 한 비교(3과제)만이 다룹니다.
12. **실로봇 Teacher 참고는 "참조"이지만 Student와 짝이 아닙니다.** Student와 WAM-OPD도 짝 시드가 아닌 독립 36회입니다(§4.1: 물체 위치 3 cm 변화).

## 6. 버전 차이 (v1 → v2)

| 항목 | v1 (2026-08-23, 11쪽) | v2 (2026-10-01, 13쪽) |
|---|---|---|
| 제목 | WAM-OPD: On-Policy Distillation for World Action Models | Joint Video–Action Supervision for … with On-Policy Distillation |
| 저자 | 4명 (Yang, Jiang, Sheng, Tang) | 16명 |
| 범위 | 2과제 "예비 능력 증명" (handover_mic, put_object_cabinet) | 12과제 + 실로봇 4과제 + 절제 3과제 |
| 학습 데이터 | Student 궤적 8, 보정 4, 160 표지 문맥, 120 최적화 스텝 | 36 궤적(학습 24, 보정 12). 문맥 수와 스텝 수 미기재 |
| 평가 | 장면 6 × 잡음 은행 2 = 12쌍 | 60회/정책/과제 (구성 미기재) |
| 결과 | handover_mic 0.0 → 58.3%, put_object_cabinet 16.7 → 33.3% | 1.7 → 35.0%, 3.3 → 16.7% |
| 손실 정의 | ρ_δ(e) = δ²(√(1+(e/δ)²) − 1) (식 9) | ρ_c = √((x−y)²+c²) − c (§3.2) |
| 한계 서술 | 별도 절(6절): 2과제, 고정 패키지의 부분 on-policy, Teacher 영상 조건 불일치 등 5가지 | 별도 한계 절 없음 |
| 체크포인트 선택 | 선별 분할에서 사전 선언 순서(성공, 의미 진행, 보정 손실)로 선택, 두 과제 모두 에폭 3 | 서술 없음 |
| 계획 표 | 표 2에 "TBD" 계획 5개(광범위 과제, 도메인 랜덤화, 행동 전용 대 결합, 고정 패키지 대 갱신 OPD, 교차 과제 유지) | 삭제. 광범위 과제와 행동 전용 대 결합은 수행, 도메인 랜덤화·갱신 OPD·교차 과제 유지는 v2에도 없음 |

v1의 손실 정의는 c = δ = 1e-3일 때 v2 식의 δ배(1e-3배) 척도입니다. 코드는 v2 형태입니다(`04` K5).

## 7. 이 분석에서 하지 않은 것

학습, 롤아웃, 평가는 하지 않았고 공식 코드는 정적으로만 읽었습니다. 모델 가중치·데이터셋·RoboTwin 자산은 받지 않았습니다. 표 값은 논문에서 옮겼고, 시행별 성공 기록(쌍체 구조)은 공개되지 않아 쌍체 검정은 가능한 범위(상·하한)로만 계산했습니다.
