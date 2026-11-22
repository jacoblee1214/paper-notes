# 01. 논문 분석: Beyond the Sink Interpretation: Radial Attention in Flow-Based VLA Policies

- 저자: Yuxi Zheng, Yueqian Lin, Yudong Liu, Zijia Tang, Junyu Pan, Yiheng Wang, Hai Helen Li, Yiran Chen (Duke University)
- arXiv: 2611.26385, cs.RO, 17쪽
- 버전: **v1 하나**(2026-11-19 제출). abs 페이지 submission history에 v1만 있음. 분석 대상은 v1
- 학회: arXiv 외 표기 없음 (`02_prior_work.md` §5)
- 원문: `paper/paper.pdf`(507 KB), `paper/paper.txt`(`pdftotext -layout`)

## 1. 한 문장 요약

flow 기반 VLA(π0.5 action expert, RoboCasa 5개 과제)에서 행동 청크 양 끝 토큰(0, 49)에 쏠리는 attention은 **RoPE가 만드는 위치 효과**이고, 이 끝점들은 언어모델의 "sink" 조건(작은 value, 출력에 거의 영향 없음)을 만족하지 않는다. 한편 prefix `<bos>`가 sink 조건에 맞고, 행동 토큰 사이의 차이(spread)는 상대 위치 오프셋 기반 routing이 유지한다.

## 2. 질문과 범위

- 질문 세 개: (1) 끝점 질량의 출처는 어디인가, (2) 끝점 value는 sink처럼 작은가, 그렇지 않다면 어떤 토큰이 sink 역할을 하며 행동 토큰을 구분하게 하는 것은 무엇인가 (Section 1, Abstract).
- 명시적으로 **검증하지 않는 것**: Wang et al.(AutoHorizon)의 horizon 규칙 자체. 논문은 "앵커를 어떻게 설명할지"에 대한 결과라고만 주장합니다 (Abstract, Section 1).
- 모델: π0.5 체크포인트 `pi05_pretrain_human300`, step 75,000 (RoboCasa365 공개 체크포인트, Section 3.1, ref [13]). 하나의 체크포인트.
- 과제: CloseFridge, OpenCabinet, OpenDrawer, PickPlaceCounterToCabinet, TurnOnMicrowave. 환경 seed 0–2(개방 루프), seed 0–49(폐루프). 10 denoising 단계, 높음·중간·마지막 노이즈 = 단계 0, 5, 9 (Section 3.1).
- 방법: 같은 관측과 같은 초기 노이즈를 공유하는 **쌍체 개입**. 마스킹(점수 제거 후 재정규화), value 0, 다른 관측(seed 99, 100, 101)의 value 방향 교체 (Section 3.3).

## 3. 방법 요점

| 식/절 | 내용 |
|---|---|
| 식 (1) | key j의 incoming mass $M_j$ = 모든 query와 head의 평균 attention |
| 식 (2) | 끝점 질량 $m_{edge} = \bar A_0 + \bar A_{49}$. head마다 **행동 key에 대해 다시 정규화**한 뒤 평균. prefix key는 분모에 없음. 균일 기준 2/50 = 0.040 |
| 식 (3) | $D_{act}$ = 최종 행동 청크의 RMSE (기준 실행과 비교) |
| 식 (4) | 세 가지 위치 변환: plus(정상 RoPE), ident(회전 없음), roll(위치 재배정 $P+((i+25) \bmod 50)$) |
| 식 (5)–(6) | 상대 오프셋 routing $T_{\ell,h}(\Delta)$를 다른 과제에서 추정하고, 각 행을 오프셋 프로파일로 재구성 (leave-one-task-out) |
| 식 (7) | gated-output 비율 $c_j$: 끝점이 attention 출력에 더하는 크기 (gate와 WO 투영 후) |
| 식 (8)–(9) | spread $S(Z)$와 spread ratio $r_{spread}$ |

## 4. 주요 결과 (본문 절과 표 번호)

### 4.1 끝점 질량은 위치에서 온다 (Section 4, 표 1, 표 2, 표 S1–S4)

- 레이어 0: 현재 층 RoPE만 제거하면 끝점 질량이 모든 노이즈에서 0.040(균일)으로 돌아감 (표 1).
- 레이어 12: 현재 층 회전을 빼도 질량이 남지만, **상위 층 0–11의 행동 토큰 RoPE를 빼면** 높음·중간 노이즈에서 0.040으로 감소 (표 2: 중간 0.164 → 0.040). 마지막 단계는 0.059–0.061에 머무름.
- prefix RoPE만 제거하면 거의 변화 없음 (레이어 12 중간 0.154 대 기준 0.165, 표 S4). 깊은 층의 집중은 행동 토큰 RoPE에서 옴.
- 끝점 두 곳만 RoPE를 빼면 집중이 사라지고 (중간 0.080), 중앙 24, 25에서 같은 조작을 하면 기준 근처 (0.154) 유지 (표 2).

### 4.2 끝점 value는 null sink가 아님 (Section 5, 표 3–5)

- value norm 비율: 끝점 0과 49의 행동 토큰 평균 대비 ℓ2 norm 비율 평균 1.034와 1.019 (표 3, 18층 평균, n = 1969).
- gated-output 비율 $c_j$: 1.569(위치 0), 1.489(위치 49) (표 3, n = 1955). 평균보다 크다.
- 폐루프 (표 4, 표 S5, 기준 113/250):

| 개입 | 성공 | 차이 (pp) | 95% 구간 | p |
|---|---|---|---|---|
| 끝점 마스크 | 121 | +3.2 | [−1.5, +7.9] | 0.24 |
| 끝점 value 0 | 120 | +2.8 | [−2.7, +8.3] | 0.39 |
| 끝점 교체 d99 | 104 | −3.6 | [−9.1, +1.9] | 0.25 |
| 끝점 교체 d100 | 95 | −7.2 | [−12.9, −1.5] | 0.020 (보정 전) |
| 끝점 교체 d101 | 105 | −3.2 | [−8.8, +2.4] | 0.33 |
| 중간 교체 (2,3) d99 | 42 | −28.4 | [−34.5, −22.3] | 2.3e−17 |
| 중간 교체 (2,4) d99 | 50 | −25.2 | [−31.5, −18.9] | 2.5e−13 |

- 같은 교체를 실행된 중간 토큰에 하면 손실이 크고, 끝점 교체 손실은 3–7 pp. 논문은 이를 "작은 비용"으로 해석 (Section 5.2).

### 4.3 prefix `<bos>`가 sink에 맞다 (Section 5.3–5.4, 표 5–6, 표 S7–S10)

- `<bos>`는 12/18 층에서 head 평균 질량이 image와 language보다 큼. 레이어 1에서 0.780 (image 0.012, language 0.112).
- value norm은 모든 층에서 image와 language 중앙값보다 작음 (레이어 1: 0.73 대 12.50, 6.19).
- value 0: 107 → 140/250, **+13.2 pp** [+7.0, +19.4], p < 1e−4 (표 6, 실행 A).
- 두 번째 실행 (λ 스케일, 기준 114): λ=0 **+6.0 pp** [+0.2, +11.8], p = 0.058; λ=0.5 +5.6 pp (표 S9, 본문 Section 5.4).
- value 교체 (방향, norm, 둘 다)는 Dact 중앙값 ≤ 0.003이고 성공 차이 0 (표 6).
- 메커니즘은 미해결로 명시됨 (Section 5.4, Appendix E). 점수 마스크의 국소 재구성 가설은 기각 (Appendix D: ‖δ‖ 평균 5.67 대 ‖B‖ 평균 2.96).

### 4.4 행동 토큰 routing은 상대 오프셋 구조를 따른다 (Section 6, 표 7, 표 8)

- 개방 루프 spread ratio: 상대 오프셋 routing 0.752 (과제별 0.645–0.845), 균일 0.024, 오프셋 셔플 0.194, key 치환 0.096 (표 7).
- 폐루프 (3개 과제, 과제당 50 에피소드, 150 중): 기준 58, 공유 오프셋 **64** (+4.0 pp, 95% [−2.9, +10.9], p = 0.35), 셔플 5, key 치환 6, 균일 6 (표 8).
- 과제 특이성: 학습에 쓰지 않은 OpenCabinet과 PickPlace에서는 공유 오프셋이 성공을 23 → 14, 34 → 21로 낮춤 (Section 6.2, Appendix F).

### 4.5 부록 실험

- Appendix G: SmolVLA (인과 행동 attention, BOS 없음) 프로브. 끝점 집중 신호는 나타나지 않았고, prefix 최고 주목 위치는 state 토큰. 폐루프 실험 없음.

## 5. 한계 (논문 자체 서술)

- 하나의 π0.5 체크포인트, 양방향 행동 attention, 공유 prefix 토큰, 5개 RoboCasa 과제 (Section 8).
- 실행 간 재현성 없음: GPU attention 감소 순서 때문에 같은 seed도 에피소드 라벨의 7–11%가 달라짐 (Appendix C).
- 끝점과 중간의 비교가 위치와 실행 여부를 분리하지 못함 (토큰 0은 실행, 49는 미실행) (Section 5.3, Section 8).
- BOS 이득은 이 체크포인트에서만 관찰. 두 번째 체크포인트에서는 λ와 행동 크기 관계가 나타나지 않음 (Section 8).
- 폐루프 routing은 3개 과제, 과제 특이적 (Section 6.2).

## 6. 논문 내부 주목점 (검증 세부는 `03`, `04`)

- 본문 Table 4의 실행이 어느 것인지 불명확 (Appendix B.1의 "먼저 실행된 같은 프로토콜"과 연결) — `04` D3.
- 끝점과 중간 비교의 실행 여부 혼합 — 논문도 인정 (`04` D5).
- BOS 두 실행의 기준 성공 수가 107과 114로 다름. "두 번 재현"이라는 표현이 독립 재현인지 불분명 — `03` V5, `04` D4.
- 표 8의 p = 0.35는 불일치쌍 수가 보고되지 않아 검증 불가 — `03` V6.
- 여러 비교에서 p 값이 보정되지 않음 (Section 5.2, Appendix B: "seven comparisons" 보정 안 함). d100의 p = 0.020은 Bonferroni 7배 하면 0.14.
- 초록의 "two paired runs"와 Section 9의 "The sink-like token is the prefix <bos>"는 한 체크포인트, 두 실행 근거. 결론의 강도가 근거 범위보다 넓음.

## 7. 계보 요약 (상세는 `02_prior_work.md`)

- 언어모델 sink 정의: Xiao et al. (StreamingLLM, ICLR 2024, [19]), Gu et al. (ICLR 2025, [8]: 작은 ℓ2 value, 출력 기여 작음).
- 위치 기반 설명: Barbero et al. (첫 토큰, [1]), Chen et al. (HoPE, [5]), Queipo-de Llano et al. (sink와 compression valley, [12]).
- 확산 트랜스포머 sink: Wu and Summa (ICML 2026, [18]).
- 행동 토큰 horizon 앵커: Wang et al. (ECCV 2026, [17], AutoHorizon).
