# 05. 주장·수치 검증 (회의적 리뷰어 관점)

- 대상: Wang et al., "GPT-6-Astra Understands Touch and Outperform Vision–Tactile–Language–Action Models", arXiv:2610.28457v1 (2026-10-20)
- 작성: 결과 검증 에이전트, 2026-10-23
- 검증 근거
  - 원문: `paper/paper.txt` 전문, `page-01.png`(Fig.1C를 3배 확대해 막대 라벨 판독), `page-03.png`(Fig.2), `page-11.png`(Fig.4)
  - 교차검토 대상: `01_paper_analysis.md`, `02_prior_work.md`, `03_derivations.md` §0.2
  - 이번에 직접 확인한 웹 출처(spot-check, 2026-10-23)
    - N0-VTLA (arXiv:2607.23782) HTML Table 1 (UniVTAC 8-task)
    - N0-TWAM (arXiv:2607.23783) HTML Table 2와 평가 프로토콜
    - GPT-as-Policy (arXiv:2609.38537) HTML Appendix H와 Appendix C Table 17
    - UniVTAC `envs/insert_hole.py` (github.com/univtac/UniVTAC, main 브랜치 raw)
- 재실행: `python3 analysis/stats_check.py --json analysis/stats_check_results.json`. numpy와 표준 라이브러리만 쓰고 scipy는 쓰지 않는다. 스크립트가 Tab.2 평균, Fig.1C, 출처 표가 서로 일치하는지 assert로 검사한다.
- 통계 가정. 이 문서의 p값과 CI는 모두 이 가정에 조건부다.
  1. 모든 셀은 n=100 독립 Bernoulli 시행으로 본다. Ours는 [§4.1]에 "100 fixed test episodes per task"로 명시되어 있다. N0-TWAM은 원 논문에 "Each simulation task is evaluated over 100 trials"로 명시되어 있다. N0-VTLA, π0.5, ACT+UniVTAC의 시행 수는 출처 표에 적혀 있지 않아 **미확인**이다.
  2. Ours와 baseline은 비짝(unpaired) 비교로 다룬다. baseline 수치는 다른 논문 표를 그대로 옮긴 값이고 seed가 같은지는 미확인이기 때문이다.
  3. Tab.3은 원문에 따르면 같은 100개 설정을 쓴 짝 비교다. 하지만 에피소드별 결과가 공개되지 않아 McNemar 검정은 상·하한만 계산했다.

---

## 0. 결론 요약

1. **수치 산술은 모두 맞다.** Tab.2의 5개 행 평균을 다시 계산해도 같다. Fig.1C의 30개 막대 라벨도 Tab.2와 모두 같다. Tab.3의 +68pp와 Tab.2/Tab.3의 Insert Hole 99%도 일치한다. baseline 20개 task 셀은 출처 논문 표와 같다. N0-VTLA와 N0-TWAM 행은 이번에 직접 대조했다.
2. **제목의 "Outperform VTLA Models"는 데이터와 모순된다.** 비교 대상 가운데 VTLA는 N0-VTLA 하나다. 5-task 평균은 N0-VTLA가 96.2, Ours가 93.8로 Ours가 낮다. 차이는 −2.4pp, SE는 1.34pp이고 양측 p=0.073이다. 방향을 정하면 오히려 "Ours가 낮다"는 쪽의 단측 p가 0.037이고, Jeffreys 사후확률로 계산한 P(Ours > N0-VTLA)는 0.042다. Pull-out Key에서는 Ours가 15pp 낮고, 이 차이는 Holm 보정 후에도 유의하다(p=0.0008).
3. **"comparable to SOTA"는 조건부로만 지지된다.** task를 고정하고 에피소드 수준에서 보면 ±5pp 동등성 TOST가 p=0.026으로 성립한다. ±3pp 기준(p=0.33)이나 ±2pp 기준에서는 성립하지 않는다. task를 표본으로 보는 수준에서는 df=4라 95% CI가 [−13.2, +8.4]pp로 넓어 동등성이든 우열이든 판정할 수 없다.
4. **저자가 내세운 우위는 모두 통계적으로 유의하지 않다.** Insert Hole에서 N0-VTLA를 앞선다는 주장은 Fisher p=0.21이다. Lift Can 최고치는 N0-TWAM 93 대비 p=0.77이다. 게다가 n=100에서는 기준 성공률이 95%일 때 Ours가 100%를 찍어도 검출력 80%에 도달하지 못한다. 실험 설계 단계에서 이미 우위를 입증할 수 없는 구조다.
5. **N0-TWAM 대비 평균 우위(+9.2pp)는 Lift Bottle 한 task에서 거의 다 나온다.** Lift Bottle은 93 vs 58로 +35pp다. Lift Bottle을 빼면 평균 차이는 +2.75pp이고, 나머지 4개 task는 하나도 유의하지 않다. task 수준 paired t는 p=0.23이다.
6. **촉각 ablation은 효과 크기 자체는 확실하다.** 99 vs 31, Fisher p=1.3e−27, 짝 McNemar p≤1.2e−19다. 그러나 w/o tactile 조건에서 Task Skill을 다시 증류했고 촉각 reference도 함께 뺐다. 따라서 이 결과는 "online 촉각 피드백의 효과"가 아니라 "촉각 파이프라인 전체의 효과"로 읽어야 한다. task도 하나뿐이다.
7. **내부 불일치는 세 가지다.** (a) "four methods"라고 쓰고 5개를 나열한다. (b) E0를 baseline으로 언급하지만 결과가 없다. (c) 제목("Outperform")과 초록("comparable")이 서로 다른 주장을 한다.
8. **분석 문서 정정이 필요하다.** 01은 Lift Bottle을 "유의하게 열세"라고 했지만 Fisher로는 p=0.065라 정정이 필요하다. 01의 Fig.2 판독(w/o tactile에서 "<5° 유지")과 01 (g)의 "비용은 출력 토큰이 지배" 가정도 고쳐야 한다. 02에는 인용 건수와 셀 수 집계 오류가 있고, Insert Hole을 "이진 선택"으로 본 것은 과장이다. 03은 GPT-as-Policy를 "미확인"으로 남겨 02와 충돌한다. 상세는 §5에 정리했다.

---

## 1. 내부 일관성 점검

| # | 점검 항목 | 결과 | 근거 |
|---|---|---|---|
| I1 | Tab.2 행 평균 재계산 | **5/5 일치**: ACT+UniVTAC 53.8, π0.5 43.0, N0-VTLA 96.2, N0-TWAM 84.6, Ours 93.8 | `stats_check.py` §1 |
| I2 | Fig.1C 막대 라벨과 Tab.2 비교 (5 task + Avg, 5개 방법) | **30/30 일치**. Fig.1C는 N0-TWAM을 N0-VTLA보다 먼저 놓아 Tab.2와 순서만 다르며, 결과에는 영향이 없다 | page-01.png 확대 판독 |
| I3 | Tab.2(Insert Hole Ours 99)와 Tab.3(Ours 99) | 일치 | Tab.2, Tab.3 |
| I4 | Tab.3의 "improvement of 68 percentage points" | 99−31=68, 일치 | §4.3 |
| I5 | 초록의 93.8%와 "31% to 99%" | 일치 | Abstract |
| I6 | Baselines 절: "four methods"라고 쓰고 E0, π0.5, ACT+UniVTAC, N0-VTLA, N0-TWAM 5개를 나열 | **불일치**. Tab.2, §4.2, Fig.1C에는 4개만 있다. 즉 "four"는 표와 맞고, E0만 결과 없이 들어가 있다 | §4.1 vs Tab.2 |
| I7 | E0를 vision-only baseline으로 언급 | **결과 없음**. 02에 따르면 출처 표에도 E0의 UniVTAC 수치가 없다. 인용 서지(제목, 학회)도 실제와 다르고 자기 인용이다 | §4.1, 02 §1 #23 |
| I8 | 제목 "Outperform"과 초록·결론 "comparable ... higher success on selected tasks" | **불일치**. 본문은 스스로 "approaches the 96.2%"라고 쓴다 [§4.2] | Title, Abstract, §4.2, §5 |
| I9 | Contribution 3 "100 seeds per task"와 §4.1 "100 fixed test episodes" | 일치 | §1, §4.1 |
| I10 | Tab.3 "Online observations: V+P vs V+P+T" | **표기가 차이를 축소한다**. 본문에 따르면 w/o tactile에서는 촉각 reference를 제거하고 Task Skill도 다시 만들었다. 표는 online 관측 차이만 보여 준다 | §4.3 |
| I11 | Tab.1의 Ours 행 "No parameter updates"와 Fig.1C의 "Ours (Training-free)" | **비대칭 표기**. ACT+UniVTAC 행에는 "50 demos"를 적었지만, Ours도 task당 50 demo(지식 구성 40, validation 10)를 쓰는데 그 사실을 Tab.1에 적지 않았다 | Tab.1, §4.1 |
| I12 | Fig.2(w/ tactile, step 191에서 성공)와 Fig.4(step 191에서 rightward probing 시작) | 서로 다른 에피소드다. 원문은 Fig.4를 "a single Insert Hole episode"라고만 해서 모순은 아니다. 01의 지적이 맞다 | page-03, page-11 |
| I13 | 명칭: Fig.2 캡션과 §4.2는 "GPT-as-Policy", 나머지는 "GPT-as-Tactile-Policy" | 혼용. 경미 | Fig.2, §4.2 |
| I14 | §4.1 "tasks cover ... tactile classification"(Grasp Classify) | 그런데 Grasp Classify의 촉각 ablation이 없다. 98%가 촉각 덕분인지 판단할 수 없다 | §4.1, §4.3 |

---

## 2. 주장별 판정표

판정 기준은 다음과 같다.
- **지지**: 원문 데이터로 주장이 성립한다.
- **부분지지**: 수치상으로는 맞지만 통계, 교란, 범위 때문에 표현이 과하다.
- **불충분**: 주장을 판정할 데이터가 없다.
- **모순**: 원문 데이터와 반대다.

| # | 주장 (원문 요지) | 위치 | 판정 | 근거 |
|---|---|---|---|---|
| C1 | "GPT-6-Astra **Understands** Touch" | Title, Abstract 첫 문단 | **불충분** | 관찰된 행동(변위가 크면 후퇴하고 작으면 전진, Fig.4)은 §3.2에 있는 Task Skill 예시 규칙과 구조가 같다. Task Skill을 뺀 조건, 반사실적 촉각 입력, 규칙에 없는 접촉 상황 실험이 없어서 "이해"와 "규칙 추종"을 구분할 수 없다 |
| C2 | "**Outperform** Vision–Tactile–Language–Action Models" | Title | **모순** | 비교 대상 중 VTLA는 N0-VTLA뿐이다. N0-TWAM은 world-action model이고, ACT+UniVTAC와 π0.5는 VTLA가 아니다. N0-VTLA 대비 평균은 −2.4pp(단측 p=0.037, 불리한 방향), task 우열은 2승 3패, Pull-out Key는 유의한 열세(Holm p=0.0008)다 |
| C3 | "93.8% mean success across five UniVTAC tasks" | Abstract | **지지**(산술) | 469/500. Wilson 95% CI는 [91.3, 95.6]이다. 다만 UniVTAC 8개 task 중 5개만 썼고, 선정 기준은 적혀 있지 않다 |
| C4 | "Overall performance is comparable to state-of-the-art policies" | Abstract, §1, §5 | **부분지지** | task 고정, 에피소드 수준 TOST로 보면 ±5pp에서는 동등(p=0.026), ±3pp에서는 동등하지 않음(p=0.33)이다. task를 표본으로 보면 판정할 수 없다(95% CI [−13.2, +8.4]). "comparable"은 "5pp 안쪽"이라는 뜻으로만 방어된다 |
| C5 | "higher success on selected tasks" | Abstract, §1 | **부분지지** | 수치상 Ours가 단독 1위인 task는 Lift Can 하나다. Insert Hole은 N0-TWAM과 공동 1위다. 1위인 baseline과 비교하면 어느 쪽도 유의하지 않다(LC vs N0-TWAM p=0.77, IH는 동률) |
| C6 | "SOTA-level performance without policy fine-tuning ... over 100 seeds per task following the benchmark protocol" | §1 Contribution 3 | **부분지지** | 평균 순위는 2/5이고 SOTA(N0-VTLA)보다 낮다. 100 trial 프로토콜은 N0-TWAM 원 논문과 같다. 하지만 baseline과 seed가 같은지는 미확인이고, baseline 수치는 직접 돌린 값이 아니라 옮겨 온 값이다 |
| C7 | "On Insert Hole, tactile feedback improves success from 31% to 99%" | Abstract, Tab.3, §4.3 | **부분지지** | 차이 자체는 매우 확실하다(Newcombe 95% CI [+57.4, +76.3]pp, McNemar p≤1.2e−19). 그러나 비교에서 바뀐 것이 셋이다. ① online 촉각, ② 촉각 reference, ③ 다시 증류한 Task Skill. "tactile feedback"만의 효과로 귀속할 수 없다. 단일 task, 단일 run이다 |
| C8 | "Mean 93.8% approaches the 96.2% of N0-VTLA and exceeds the 84.6% of N0-TWAM" | §4.2 | **지지** | N0-TWAM 대비 +9.2pp. 에피소드 수준에서 z=5.09, CMH p=1.7e−6이다. 단, 이 우위의 대부분은 Lift Bottle(+35pp)에서 나온다. task 수준 t는 p=0.23, sign test는 p=0.125다 |
| C9 | "On Insert Hole, 99%, matching N0-TWAM and outperforming N0-VTLA" | §4.2 | **부분지지** | N0-TWAM과 같다는 부분은 맞다(99 vs 99). N0-VTLA를 앞선다는 부분은 유의하지 않다(+4pp, Fisher p=0.21, CI [−1.3, +10.2]). 이를 입증하려면 arm당 약 285 에피소드가 필요하다 |
| C10 | "On Lift Can, 95%, the highest success rate among the compared methods" | §4.2 | **부분지지** | 수치상 최고는 맞다. 출처 표에 있지만 이 논문이 빠뜨린 InternVLA-A1(90)을 포함해도 최고다. 하지만 N0-TWAM 93과 통계적으로 구분되지 않는다(p=0.77, 검출에 arm당 약 2,213 필요) |
| C11 | "Across all five tasks, it matches or outperforms N0-TWAM, demonstrating consistent performance" | §4.2 | **부분지지** | 수치상으로는 5/5 성립한다(0, +2, +4, +35, +5). 개별 task에서 유의한 것은 Lift Bottle 하나다(나머지 p≥0.28). "consistent"라는 근거는 Ours의 성공률 범위가 84–99로 좁다는 정도다 |
| C12 | "matching or exceeding the compared methods on selected tasks" | §5 | **지지**(수치) | IH 공동 1위, LC 1위. 통계적 우위는 아니다 |
| C13 | Baselines "previously evaluated on UniVTAC" | §4.1 | **지지**(출처), **불충분**(공정성) | 20개 task 셀이 출처 표와 같다. 같은 출처 표에 있는 더 강한 행(InternVLA-A1의 IH 93과 LC 90, Xiaomi-Robotics-0의 IH 96)은 이 논문에 실리지 않았다. 이 행들을 넣어도 Ours의 task별 순위는 바뀌지 않는다. seed와 시행 수가 같은지는 미확인이다 |
| C14 | E0를 vision-only baseline으로 비교 | §4.1 | **모순** | 결과가 없다(I6, I7) |
| C15 | "Without additional robotics-specific pretraining or per-task policy fine-tuning" | Abstract, Tab.1 | **부분지지** | 파라미터를 갱신하지 않는 것은 설계상 참이다. 그러나 ① GPT-6-Astra의 사전학습 데이터가 공개되지 않아 "robotics-specific pretraining 없음"은 검증할 수 없다. ② task별로 데모 50개를 쓰고 Task Skill을 증류하며 10개 validation을 한다. 이 validation이 반복 튜닝이었는지는 미확인이고, 사실상 task별 적응 단계다 |
| C16 | Tab.1 "Knowledge-guided probing ✓" (Ours만) | Tab.1 | **불충분** | 저자가 직접 정의한 정성 열이다. 측정 지표가 없다 |
| C17 | "Execution analysis further demonstrates contact-guided probing and correction" | Abstract, §4.4 | **불충분** | 근거는 선택된 에피소드 1–2개(Fig.2, Fig.4)뿐이다. probing 빈도나 성공 기여 같은 집계 통계가 없다. "demonstrates"보다 "illustrates"가 맞다. 원문 §4.4 본문도 스스로 "illustrate"라고 쓴다 |
| C18 | Fig.2: "w/o tactile ... orientation and lateral adjustments remain limited" | §1, §4.3, Fig.2 | **부분지지**(n=1) | 그림 곡선을 보면 w/o tactile의 측면 변위는 약 0mm다. 축 변화는 step 50–90 사이에 약 10–12°까지 갔다가 약 5°로 돌아온다. 단일 에피소드라 일반화할 수 없다 |
| C19 | "These results establish ... an effective route to **efficient** contact-rich manipulation" | §5 | **불충분** | 추론 비용, 지연, 토큰을 하나도 보고하지 않았다. 같은 프로토콜인 GPT-as-Policy는 RoboDojo Direct 50 instance에서 1.13B 토큰을 썼다(arXiv:2609.38537 Tab.17). "efficient"는 학습 비용에만 해당한다 |
| C20 | Direct-control 프로토콜: xhigh, IK, 1–5 step, 에피소드별 세션 | §3.4, §4.1 | **지지**(계보) | GPT-as-Policy Appendix H와 같다: "gpt-6-astra with xhigh reasoning through the Responses API", repeat count 1–5(각 0.05s control step), "one model thread for the physical episode". 이번에 직접 확인했다. UniVTAC에서 control step 주기가 같은지는 미확인이다 |

---

## 3. 통계 표

### 3.1 Wilson 95% CI (n=100 가정, %)

| Method | Insert Hole | Lift Can | Grasp Classify | Lift Bottle | Pull-out Key |
|---|---|---|---|---|---|
| ACT + UniVTAC Encoder | 24 [16.7, 33.2] | 29 [21.0, 38.5] | 99 [94.6, 99.8] | 71 [61.5, 79.0] | 46 [36.6, 55.7] |
| π0.5 | 25 [17.5, 34.3] | 6 [2.8, 12.5] | 49 [39.4, 58.7] | 100 [96.3, 100.0] | 35 [26.4, 44.7] |
| N0-VTLA | 95 [88.8, 97.8] | 88 [80.2, 93.0] | 100 [96.3, 100.0] | 99 [94.6, 99.8] | 99 [94.6, 99.8] |
| N0-TWAM | 99 [94.6, 99.8] | 93 [86.3, 96.6] | 94 [87.5, 97.2] | 58 [48.2, 67.2] | 79 [70.0, 85.8] |
| **Ours** | 99 [94.6, 99.8] | 95 [88.8, 97.8] | 98 [93.0, 99.4] | 93 [86.3, 96.6] | 84 [75.6, 89.9] |
| Tab.3 w/o tactile | 31 [22.8, 40.6] | – | – | – | – |

### 3.2 Ours vs N0-VTLA, N0-TWAM (task별, 비짝)

| 비교 | Task | Ours | Base | 차이(pp) | Newcombe 95% CI | z | p (z) | p (Fisher) | Holm (Fisher, 5 task) |
|---|---|---|---|---|---|---|---|---|---|
| vs N0-VTLA | Insert Hole | 99 | 95 | +4 | [−1.3, +10.2] | +1.66 | 0.097 | 0.212 | 0.423 |
| | Lift Can | 95 | 88 | +7 | [−0.9, +15.3] | +1.77 | 0.076 | 0.126 | 0.379 |
| | Grasp Classify | 98 | 100 | −2 | [−7.0, +2.0] | −1.42 | 0.155 | 0.497 | 0.497 |
| | Lift Bottle | 93 | 99 | −6 | [−12.8, −0.3] | −2.17 | 0.030 | **0.065** | 0.259 |
| | Pull-out Key | 84 | 99 | −15 | [−23.5, −7.6] | −3.80 | 1.4e−4 | **1.5e−4** | **7.7e−4** |
| vs N0-TWAM | Insert Hole | 99 | 99 | 0 | [−4.5, +4.5] | 0 | 1 | 1 | 1 |
| | Lift Can | 95 | 93 | +2 | [−5.1, +9.3] | +0.60 | 0.55 | 0.77 | 1 |
| | Grasp Classify | 98 | 94 | +4 | [−1.9, +10.6] | +1.44 | 0.15 | 0.28 | 1 |
| | Lift Bottle | 93 | 58 | +35 | [+23.6, +45.4] | +5.75 | 8.7e−9 | **6.7e−9** | **3.3e−8** |
| | Pull-out Key | 84 | 79 | +5 | [−5.8, +15.7] | +0.91 | 0.36 | 0.47 | 1 |

참고로 ACT+UniVTAC, π0.5와의 비교도 스크립트 §3에 있다. 눈여겨볼 결과가 둘 있다. 첫째, Lift Bottle에서 **vision-only π0.5(100)가 Ours(93)보다 유의하게 높다**(Fisher p=0.014, Holm p=0.014). 촉각을 쓰는 방법이 시각 전용 VLA에 지는 task가 있다는 뜻인데, 원문은 이를 논의하지 않는다. 둘째, Grasp Classify에서는 ACT+UniVTAC(99)와 차이가 없다(p=1).

방법론 메모를 덧붙인다. 성공률이 0.93–1.00 경계에 몰려 있어 기대 실패 수가 5보다 작은 셀이 많다. 그래서 z 검정(Wald 또는 pooled)은 반보수적이고, Fisher exact를 기준으로 삼아야 한다. Lift Bottle은 z 검정으로는 p=0.030이지만 Fisher로는 0.065라 유의성 판정이 뒤집힌다.

### 3.3 5-task 평균 비교

| 방법 | Ours vs N0-VTLA (93.8 vs 96.2) | Ours vs N0-TWAM (93.8 vs 84.6) |
|---|---|---|
| (A) 에피소드 수준, task 고정: 차이, SE | −2.4pp, SE 1.34 | +9.2pp, SE 1.81 |
| 양측 p / 95% CI | 0.073 / [−5.0, +0.2] | 5e−7 / [+5.7, +12.7] |
| 단측 p (Ours < base) / 90% CI | **0.037** / [−4.6, −0.2] | – |
| (A′) TOST 동등성 p (±2 / ±3 / ±5pp) | 0.62 / 0.33 / **0.026** | 동등하지 않음 |
| (B) task 수준 paired t (df=4) | d=[+4, +7, −2, −6, −15], t=−0.62, p=0.57, 95% CI [−13.2, +8.4] | d=[0, +2, +4, +35, +5], t=+1.41, p=0.23, 95% CI [−8.9, +27.3] |
| sign test | 2/5 양(+), p=1.0 | 4/4 양(+), 동률 1, p=0.125 |
| (C) CMH (task 층화) | χ²=2.58, p=0.108, MH OR=0.60 | χ²=22.9, p=1.7e−6, MH OR=2.90 |
| (D) pooled Fisher 469/500 vs base/500 | p=0.11 | p=3.4e−6 |
| (E) Jeffreys Beta MC, 2×10⁵ draws | P(Ours > VTLA)=**0.042**, 95% CrI [−5.1, +0.3] | P(Ours > TWAM)≈1.000, CrI [+5.5, +12.7] |

(A)와 (B)는 서로 다른 질문에 답한다. (A)는 이 5개 task에 대한 조건부 추론이고 에피소드 수준 잡음만 반영한다. (B)는 task를 모집단에서 뽑은 표본으로 보는 추론이며, "다른 contact-rich task에서도 그럴까"라는 일반화 질문에 해당한다. 제목과 초록의 일반 주장은 (B)의 질문인데, df=4에서는 어느 방향으로도 결론이 나지 않는다. 또 N0-VTLA의 100% 셀은 binomial 분산이 0이라 (A)의 SE를 약간 과소추정한다. 이를 보정한 (E)도 같은 결론을 낸다.

### 3.4 Tab.3 촉각 ablation

| 항목 | 값 |
|---|---|
| 99 vs 31, 차이 Newcombe 95% CI | +68pp [+57.4, +76.3] |
| 비짝 Fisher | p=1.3e−27 |
| 짝 McNemar(exact). Ours 실패 1건이므로 불일치쌍 c∈{0,1}, b=68+c | c=0: p=6.8e−21, c=1: p=1.2e−19 |
| 해석 | 통계적으로는 의심할 여지가 없다. 남는 문제는 **인과 귀속**(Task Skill 재증류, reference 제거)과 **외적 타당성**(1개 task)이다 |

### 3.5 검출력(power): 이 실험이 원리적으로 판정할 수 있는 범위

조건은 α=0.05(양측), power=0.8, 정규근사다.

| 대상 | 필요 표본 |
|---|---|
| IH 99 vs 95 | arm당 285 |
| LC 95 vs 88 | arm당 248 |
| LC 95 vs 93 | arm당 2,213 |
| LB 93 vs 99 | arm당 167 |
| 평균 93.8 vs 96.2 (task 층화 분산) | **task당 245**, arm당 약 1,225 |
| 평균 93.8 vs 96.2 (pooled 비율 근사, 03 §0.2의 방식) | arm 전체 1,294 (task당 약 259) |
| n=100에서 기준 95%보다 **높은** 쪽의 MDE | **계산 불가**. Ours=100%여도 power가 0.8에 못 미친다 |
| n=100에서 기준 95% / 99%보다 낮은 쪽의 MDE | −12.4pp / −9.0pp |

결론적으로 상한에 가까운 baseline을 상대로 "outperform"을 입증하는 것은 n=100 설계에서 원리적으로 불가능하다. 우위를 주장하려면 task당 수백 에피소드가 필요하다.

### 3.6 UniVTAC 8-task 시나리오 (제외 task: Put Bottle, Insert HDMI, Insert Tube)

| Baseline | 8-task 평균 (출처) | 제외 3개 평균 | Ours가 이 baseline의 8-task 평균과 같아지려면 제외 3개에서 필요한 평균 |
|---|---|---|---|
| N0-VTLA | 83.1 | 61.3 | **≥65.3%** |
| N0-TWAM | 84.5 | 84.3 | **≥69.0%** |
| ACT+UniVTAC | 48.0 | 38.3 | 제외 task에서 0%여도 Ours가 앞선다 |
| π0.5 | 41.4 | 38.7 | 제외 task에서 0%여도 Ours가 앞선다 |

5-task 부분집합을 고르면 N0-VTLA는 크게 유리해지고(+13.1pp) N0-TWAM은 거의 영향이 없다. 따라서 "5-task 선택이 Ours에 유리하게 체리피킹됐다"는 증거는 **없다**. 다만 Ours가 제외된 3개 task에서 몇 %를 내는지 모르기 때문에(미확인) 8-task 순위는 판정할 수 없다.

---

## 4. 계산 코드 (핵심부 인라인)

전체 코드는 `analysis/stats_check.py`에 있다. 아래는 판정에 직접 쓰인 함수만 옮긴 것이다.

```python
import math, numpy as np
N = 100
def norm_cdf(x): return 0.5 * math.erfc(-x / math.sqrt(2))

def wilson(k, n=N, z=1.959964):
    p = k / n; den = 1 + z*z/n
    c = (p + z*z/(2*n)) / den
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return max(0, c-h), min(1, c+h)

def newcombe_diff_ci(k1, k2, n1=N, n2=N):           # Newcombe (1998) method 10
    p1, p2 = k1/n1, k2/n2; (l1, u1), (l2, u2) = wilson(k1, n1), wilson(k2, n2)
    d = p1 - p2
    return d - math.hypot(p1-l1, u2-p2), d + math.hypot(u1-p1, p2-l2)

def _lchoose(n, k): return math.lgamma(n+1) - math.lgamma(k+1) - math.lgamma(n-k+1)
def fisher_two_sided(k1, k2, n1=N, n2=N):          # sum of P(table) <= P(observed)
    K, M = k1+k2, n1+n2
    lp = {x: _lchoose(n1, x) + _lchoose(n2, K-x) - _lchoose(M, K)
          for x in range(max(0, K-n2), min(K, n1)+1)}
    return min(1.0, sum(math.exp(v) for v in lp.values() if v <= lp[k1] + 1e-9))

def stratified_mean_diff(ours, base, n=N):         # equal-weight task mean, tasks fixed
    d = np.mean(ours) - np.mean(base)
    var = sum(a/n*(1-a/n)/n + c/n*(1-c/n)/n for a, c in zip(ours, base)) / len(ours)**2
    se = math.sqrt(var) * 100; z = d / se
    return d, se, z, 2*(1-norm_cdf(abs(z))), norm_cdf(z)   # diff, SE, z, p_two, p(ours<base)

def tost(d, se, m): return max(1-norm_cdf((d+m)/se), norm_cdf((d-m)/se))

def cmh(ours, base, n=N):                          # Cochran–Mantel–Haenszel, cc=0.5
    dev = var = 0.0
    for a, c in zip(ours, base):
        b, d = n-a, n-c; t = a+b+c+d
        dev += a - (a+b)*(a+c)/t
        var += (a+b)*(c+d)*(a+c)*(b+d) / (t*t*(t-1))
    chi2 = (abs(dev)-0.5)**2 / var
    return chi2, math.erfc(math.sqrt(chi2/2))

ours = [99, 95, 98, 93, 84]; vtla = [95, 88, 100, 99, 99]; twam = [99, 93, 94, 58, 79]
print(stratified_mean_diff(ours, vtla))   # (-2.4, 1.34, -1.79, 0.073, 0.037)
print(fisher_two_sided(93, 99))           # 0.0649  (Lift Bottle; z-test gives 0.030)
print(cmh(ours, twam))                    # (22.9, 1.7e-06)
```

이 밖에 스크립트에 들어 있는 것은 다음과 같다.
- task 수준 paired t: 불완전 베타 함수로 t-CDF를 직접 구현했고, df=4 닫힌형과 소수 15자리까지 일치하는 것을 확인했다.
- 정확 sign test, Holm 보정, Jeffreys Beta Monte Carlo(seed 0)
- 검출력/MDE 계산, 8-task 시나리오
- 일치성 assert: Tab.2 평균, Fig.1C, 출처 표 셀

---

## 5. 분석 문서에 대한 정정 사항

### 5.1 `01_paper_analysis.md`

| # | 위치 | 원 서술 | 정정 |
|---|---|---|---|
| K1 | §3.3 표, L6 | Lift Bottle 93 vs 99: "z=−2.17, p≈0.03, **유의하게 열세**" | 기대 실패 수가 5보다 작아 z 검정은 부적절하다. **Fisher p=0.065, Holm 보정 p=0.26으로 유의하지 않다.** 여러 검정에서 일관되게 유의한 열세는 Pull-out Key 하나뿐이다(Holm p=0.0008). L6의 "열세(Pull-out Key, Lift Bottle)는 유의"도 같은 방식으로 고쳐야 한다 |
| K2 | §3.3 | Lift Can vs N0-VTLA "p≈0.08", Insert Hole "p≈0.10" | z 검정 값으로는 맞다. Fisher로는 각각 0.13과 0.21이다. 결론("유의하지 않음")은 그대로다 |
| K3 | §3.3 | 평균 "p≈0.07, 경계선" | 양측 p는 맞다. 다만 단측(Ours가 낮음) p=0.037, P(Ours>VTLA)=0.042, ±5pp TOST 동등 p=0.026을 함께 적어야 한다. task 수준 CI [−13.2, +8.4]도 병기해야 한다. "comparable"은 5pp 마진에서만 지지된다 |
| K4 | §3.3 마지막 문단 | "N0-TWAM과 모든 태스크에서 동등하거나 그 이상은 수치상 사실" | 맞다. 다만 유의한 우위는 Lift Bottle 하나이고, LB를 빼면 평균 차이는 +2.75pp라는 점을 덧붙여야 한다 |
| K5 | §3.4 | Fig.2: "w/o tactile 조건은 축 변화가 5° 미만에 머물고" | **그림과 다르다.** w/o tactile 곡선은 step 50–90 사이에 약 10–12°까지 올라갔다가("Recover") 약 5°로 끝난다. "<5°"는 step 165 keyframe에 붙은 라벨이다. w/ tactile의 측면 변위는 "Align"에서 최대 약 43mm, "Success" 시점에 약 33mm다. 그림 판독 오차는 ±3 정도다 |
| K6 | §3.5 | baseline 수치는 "기존 보고치 인용 가능성 높음(확실도: 중간)" | **확인됨으로 올린다.** 02의 대조와 이번 spot-check(arXiv:2607.23782 Tab.1, 2607.23783 Tab.2)에서 셀 단위로 일치했다 |
| K7 | §2.4 | 마커 격자 "대략 10×8" | Fig.4 quiver를 판독하면 약 9열×7행으로 보인다. 렌더 해상도가 낮아 확정할 수 없으니 미확인으로 두는 것이 맞다. 경미하다 |
| K8 | §5(g) | "비용은 출력(reasoning) 토큰이 지배", 호출당 출력 2k–20k | **유일한 외부 근거와 충돌한다.** GPT-as-Policy(같은 프로토콜) Tab.17에 따르면 RoboDojo Direct 50 instance의 총 1,132.3M 토큰 가운데 cached input이 1,107.3M(97.8%), output이 2.09M(0.18%)이다. instance당 output은 약 42k이고 cached input은 약 22.1M이다. 01과 같은 가상 단가를 적용하면 instance당 약 $14.6이고, 그중 cached input이 약 $11.1로 **입력이 지배**한다. 단, Tab.17의 "output"에 reasoning 토큰이 포함되는지는 미확인이다. 결론적으로 01의 토큰 구성 가정은 근거가 약하다. 대표값 $13k는 외부 근거로 지지되지 않는다. instance당 $2–15 수준의 외부 근거(Astranomy 약 $1.96, 02와 03 인용, 직접 미확인)를 쓰면 600 에피소드에 약 $1.2k–$9k다 |
| K9 | §5(g) | Fig.4 시점 간격으로 "평균 horizon ≈4" 추정 | keyframe 간격(3–5)은 horizon ≤5와 **모순되지 않을** 뿐이다. 두 keyframe 사이에 호출이 여러 번 있었을 수도 있으므로 horizon을 식별할 근거는 아니다. 확실도를 "낮음"으로 내려야 한다. 참고로 GPT-as-Policy의 control step은 0.05s이고, UniVTAC에서도 같은지는 미확인이다 |
| K10 | L4 | "시뮬레이션은 LLM 호출 동안 시간을 멈출 수 있지만" | 이 논문은 정지 여부를 서술하지 않는다(**미확인**). GPT-as-Policy도 Appendix H(direct)에서는 언급하지 않는다. Appendix G의 RoboDojo Hybrid에서만 "Simulation time pauses while Astra deliberates"라고 쓴다 |
| K11 | L1, Tab.1 해설 | 데이터 사용량 비교 | 덧붙일 점이 있다. Tab.1은 ACT 행에만 "50 demos"를 적고 Ours 행에는 task당 50 demo 사용을 적지 않았다(I11) |

### 5.2 `02_prior_work.md`

| # | 위치 | 원 서술 | 정정 |
|---|---|---|---|
| P1 | §0-1, §1 각주 | "References 22건 중 19건 일치", "인용 23건(서지 22개에 SayCan arXiv ID를 보충해 셈)" | **집계 오류다.** 원문 참고문헌은 **23건**이고 SayCan(Brohan et al.)도 그중 하나다. 정확한 집계는 23건 중 20건 일치, 3건 불일치 또는 부분 확인이다 |
| P2 | 요약, §2.3 | "Table 2의 baseline 25개 셀 모두 일치" | Tab.2의 baseline 셀은 4행×5 task = **20개**이고, 평균까지 넣으면 24개다. 일치한다는 결론은 맞다 |
| P3 | 요약 7, §5.3 | Insert Hole은 "사실상 이진 선택" | **과장이다.** 기울기 부호는 이진이다(`rotate ∈ {0, π}`, ±π/6, 소스에서 확인). 하지만 `base_offset = create_noise([0.02, 0.03, 0.0])`로 슬롯 위치에 연속 노이즈가 들어가고(분포 형태는 미확인), 성공하려면 target 기준 \|x\|,\|y\|<1cm, 축 정렬 dot>0.99(약 8.1° 이내), z<−4cm를 모두 만족해야 한다. 그래서 방향 선택 뒤에도 연속 정렬 문제가 남는다. Fig.4의 정책이 맨 처음 near-vertical 방향을 probing한 것도 눈여겨볼 만하다. 기울기가 ±30°뿐이라면 수직은 절대 정답이 될 수 없으므로, 이 행동은 "Task Skill이 ±30° 이진 사전지식을 담았다"는 02의 추정과 **긴장 관계**에 있다. 근거는 에피소드 하나라 약하다 |
| P4 | §5.3 | "w/o tactile 31%는 무작위 방향 선택의 기대치 약 50%보다 낮음" | 비교 기준이 성립하지 않는다. 방향을 맞혀도 위치·축 정렬과 삽입 조건이 남으므로 50%는 상한도 기준선도 아니다. 삭제하거나 "참고용"으로 낮춰야 한다 |
| P5 | §4 표 | Unexpected Robot Policy(2609.24170)를 "반대 증거"로 서술 | 설정이 다르다. one-shot 원시 데모, RoboDojo, 촉각 없음이다. 본 논문은 40개 데모를 요약한 Task Skill을 쓴다. "간접적인 상반 사례" 정도로 낮춰야 한다 |
| P6 | 작업 요약 | "인용 23건을 웹으로 검증" | 본문 스스로 SayCan arXiv ID와 Inner Monologue/VoxPoser 학회는 "지식 기반"이라고 적었다. "웹 검증과 일부 지식 기반 확인"으로 고쳐야 한다 |
| P7 | §2.3, §5.2 | baseline 선정에 대한 언급 | 빠진 내용이 있다. 출처 표(N0-VTLA Tab.1, N0-TWAM Tab.2)에는 StarVLA-α, InternVLA-A1(IH 93, LC 90, GC 86), Xiaomi-Robotics-0(IH 96, PK 80), GigaWorld-Policy, LingBot-VA, FastWAM, ACT(vision only), VITaL 행도 있는데 본 논문은 4개만 골랐다. 이 행들을 넣어도 Ours의 task별 순위는 바뀌지 않는다. 그래도 선정 기준이 적혀 있지 않다 |
| P8 | §5.2 | N0-VTLA의 UniVTAC rollout 수 미확인 | 이번 확인에서도 N0-VTLA HTML 앞부분 100k자에는 적혀 있지 않았다(미확인 유지). N0-TWAM은 "100 trials per simulation task"로 확인됐다. N0-TWAM의 기본 촉각 입력이 "clean" Taxim GelSight render이고 gel-rendered에서는 82.4라는 점도 확인했다 |

02에서 이번에 재확인해 **맞다고 확인된** 서술은 다음과 같다.
- baseline 셀 출처
- N0-VTLA 8-task 평균 83.1, N0-TWAM 84.5, π0.5 41.4
- GPT-as-Policy Appendix H의 프로토콜(xhigh, Responses API, schema tool call, repeat 1–5, 에피소드당 thread 하나)
- RoboDojo Direct 1,132.3M 토큰
- Insert Hole의 ±π/6 이진 기울기와 `check_success` 조건

### 5.3 `03_derivations.md`와 다른 문서 사이의 충돌

| # | 위치 | 문제 | 정정 |
|---|---|---|---|
| X1 | 03 G11, §미해결 4 | "GPT-as-Policy arXiv:2609.38537은 확인하지 못했다" | 02와 이번 확인에서 **존재가 확인됐다**: Galbot Team, "Systematically Exploring the Capabilities of GPT-6 Astra as Embodied Policies", Appendix H가 있다. 03을 고쳐야 한다 |
| X2 | 03 §0.2 | "arm당 약 1,293 episode" | 5개 task를 합친 pooled 비율 근사값이다(task당 약 259). task 층화 분산으로 계산하면 task당 245, arm당 약 1,225다. 단위를 분명히 적어야 한다 |
| X3 | 03 D2 구조 행 | "(원문과 SenseNova는 사고 중 시뮬 정지)" | 본 논문 원문에는 시뮬레이션 정지에 대한 서술이 없다. "원문" 부분을 "미확인"으로 고쳐야 한다(K10과 같은 문제) |
| X4 | 03 §0.2 표 | Fisher p값 | 이 문서와 일치한다: IH 0.21, LC 0.13, GC 0.50, LB 0.065, PK 0.0002, pooled 0.11, ablation 1.3e−27. 정정할 것은 없다. 다만 03이 Lift Bottle을 유의하지 않다고 본 판단(p=0.065)이 맞고, 01이 틀렸다(K1) |
| X5 | 02 §2.2 vs 03 §(리소스) | UniVTAC 센서: 02는 "데이터 수집은 GelSight Mini만 지원(README)", 03은 "실험은 ViTai GF225" | 논문과 repo의 차이일 수 있어 모순이라고 단정하지 않는다. 다만 재구현 에이전트가 어느 센서 설정을 쓸지 정해야 하므로 **미확인**으로 남긴다 |

---

## 6. 미해결 이슈

1. N0-VTLA, π0.5, ACT+UniVTAC의 UniVTAC 시행 수와 seed는 미확인이다. §3의 p값은 모두 n=100 가정에 조건부다. seed가 같다면 짝 검정(McNemar)으로 검정력이 높아질 수 있지만, 에피소드 단위 결과가 공개되지 않아 할 수 없다.
2. Ours가 제외된 3개 task(Put Bottle, Insert HDMI, Insert Tube)에서 얼마를 내는지 미확인이라 8-task 순위를 판정할 수 없다. 제외 task 평균이 65.3% 이상이면 N0-VTLA, 69.0% 이상이면 N0-TWAM과 같아진다.
3. GPT-as-Policy Tab.17의 "output" 토큰에 reasoning 토큰이 들어가는지 미확인이다. 그래서 K8의 비용 구조 정정에도 불확실성이 남는다. 본 논문에 맞는 토큰과 지연 수치는 여전히 없다.
4. UniVTAC의 `create_noise`가 어떤 분포인지(균등 ±인지 등), 다른 4개 task의 `check_success` 원문, step_lim이 control step 기준인지는 이번에 확인하지 못했다(insert_hole.py만 확인).
5. Astranomy의 호출당 18.56s와 episode당 $1.96, GPT-as-Policy의 호출당 39.86s(locomotion)는 02와 03의 보고를 옮긴 것이고 이번에 직접 확인하지 않았다.
6. Fig.2 곡선 수치(약 10–12°, 약 43/33mm)는 렌더 이미지(910×1287)를 눈으로 읽은 값이라 ±3 정도의 오차가 있다.
7. w/o tactile 조건의 Task Skill 원문, 10개 validation 데모의 용도, Task Skill을 몇 번 수정했는지는 공개되지 않았다. 그래서 C7과 C15의 판정을 더 진전시킬 수 없다.
