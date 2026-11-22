# 03. 수치 재계산과 일관성 검사

이 문서는 논문 Fig. 3와 본문·부록의 수치를 다시 계산하고, 표와 문장의 일치를 검사합니다. 재실행하지 않는 학습과 시뮬레이션은 없습니다. 계산은 `analysis/recompute.py`(numpy, scipy, 수 초), 결과는 `analysis/recompute_results.json`에 있습니다. 검사 라벨은 V1–V9입니다.

## 0. 자료와 한계

- **집계 파일:** 공식 저장소의 `dexmachina/eval/stats/dexmachina_main_stats_AUC.json`(MIT). 손 4개 × 물체 7개 = 28셀. 셀마다 평균(`value`, 0–1)과 seed 간 표준오차(`err`), seed 5개(`n_seeds`)가 있습니다. seed별 값은 없습니다.
- **Fig. 3 라벨:** 막대 위 숫자(소수 첫째 자리, %)를 300 dpi 확대본에서 눈으로 판독했습니다. 판독 오류가 있을 수 있으나, DexMachina 28셀은 집계 파일과 모두 맞았습니다(V1). 기준선 값은 이 판독에만 의존합니다.
- **기준선의 seed 자료는 없습니다.** 기준선 오차막대는 판독하지 않았습니다. 그래서 기준선과의 비교는 물체를 짝으로 본 검정(n = 7, ManipTrans는 4)만 합니다.
- **ADD-AUC 정의:** 집계 코드는 부품(top, bottom)별 ADD에서 1–9 cm 임계값 9개의 정확도를 구하고, 사다리꼴 적분으로 AUC를 만든 뒤 두 부품 평균을 씁니다(`repo/dexmachina/eval/group_results.py` 71–96행). AUC는 0–1 값이므로 ×100을 하면 Fig. 3의 값이 됩니다.

## 1. 집계 값과 그림 일치 (V1)

- 28셀 모두 `round(value × 100, 1)`이 Fig. 3 "Ours" 막대 라벨과 같습니다. 불일치는 0입니다.
- **지표 이름 불일치:** Fig. 3의 y축과 막대 라벨은 "Success Rate (%)"이지만 값은 AUC×100입니다(예: Inspire Ketchup-100, 0.7071 → 70.7). 본문 §5와 App. B.4는 ADD-AUC라고 씁니다. 따라서 "성공률 70.7%"라는 읽기는 틀립니다. 이 값은 1–9 cm 허용 오차에서 물체 자세 정확도를 평균한 것입니다.
- **seed 수:** 28셀 모두 n = 5이고, seed 집합은 {15, 24, 42, 66, 113}으로 셀 사이에 같습니다(V6).
- **표준오차 정의:** `err`는 `np.std(seed값) / sqrt(n)`이고 `np.std`의 기본 `ddof=0`입니다(`group_results.py` 236행). 표본 표준편차로 고치면 값이 sqrt(5/4) ≈ 1.118배 커집니다. 아래 표는 ddof=1로 고친 SEM과 t(4) 95% 구간을 씁니다.

| 손 | 물체 | 평균 | SEM (ddof=1) | 95% 구간 |
|---|---|---|---|---|
| Inspire | Ketchup-100 | 70.7 | 10.7 | [40.9, 100.5] |
| Inspire | Box-200 | 87.1 | 2.0 | [81.6, 92.5] |
| Inspire | Mixer-170 | 89.8 | 1.2 | [86.6, 93.0] |
| Inspire | Ketchup-300 | 32.3 | 15.5 | [−10.6, 75.3] |
| Inspire | Mixer-300 | 64.3 | 8.7 | [40.3, 88.4] |
| Inspire | Notebook-300 | 71.2 | 14.4 | [31.3, 111.2] |
| Inspire | Waffleiron-300 | 23.8 | 8.4 | [0.4, 47.2] |
| Allegro | Ketchup-100 | 91.4 | 0.04 | [91.3, 91.5] |
| Allegro | Box-200 | 88.6 | 1.1 | [85.7, 91.6] |
| Allegro | Mixer-170 | 77.3 | 14.1 | [38.2, 116.4] |
| Allegro | Ketchup-300 | 83.0 | 2.4 | [76.3, 89.7] |
| Allegro | Mixer-300 | 81.1 | 1.1 | [78.1, 84.2] |
| Allegro | Notebook-300 | 87.1 | 1.1 | [84.0, 90.1] |
| Allegro | Waffleiron-300 | 75.4 | 3.3 | [66.1, 84.6] |
| XHand | Ketchup-100 | 90.4 | 0.2 | [89.7, 91.0] |
| XHand | Box-200 | 86.5 | 1.6 | [82.1, 91.0] |
| XHand | Mixer-170 | 89.9 | 0.6 | [88.2, 91.5] |
| XHand | Ketchup-300 | 72.4 | 4.9 | [58.7, 86.0] |
| XHand | Mixer-300 | 66.2 | 7.8 | [44.7, 87.8] |
| XHand | Notebook-300 | 89.1 | 1.4 | [85.2, 92.9] |
| XHand | Waffleiron-300 | 80.3 | 1.4 | [76.4, 84.3] |
| Schunk | Ketchup-100 | 89.7 | 1.0 | [87.1, 92.4] |
| Schunk | Box-200 | 85.9 | 0.6 | [84.2, 87.6] |
| Schunk | Mixer-170 | 90.3 | 1.5 | [86.0, 94.6] |
| Schunk | Ketchup-300 | 70.0 | 5.4 | [55.0, 85.0] |
| Schunk | Mixer-300 | 66.8 | 3.7 | [56.7, 77.0] |
| Schunk | Notebook-300 | 69.6 | 10.1 | [41.7, 97.6] |
| Schunk | Waffleiron-300 | 41.4 | 11.7 | [8.9, 73.9] |

**해석 주의:**
- 여러 셀의 구간이 0 아래나 100 위로 나갑니다(예: Notebook-300, Inspire, 111.2). 정규 근사 구간이 경계를 넘는 것이므로 숫자로 읽기보다 폭을 보는 것이 맞습니다.
- Allegro Ketchup-100의 SEM은 0.04입니다. 5개 seed의 값이 거의 같다는 뜻입니다. 91% 근처에서 포화된 결과인지, seed 간 실제로 같은 결과인지는 seed별 값이 없어 구별할 수 없습니다. 이 셀은 분산 추정에 쓰기 어렵습니다.
- 셀당 seed가 5개뿐이라, SEM이 큰 셀(Inspire 300계열, Allegro Mixer-170)은 10–30 %p 차이를 구별하기 어렵습니다.

## 2. 손별 평균과 손 사이 비교 (V2)

물체 7개 셀 평균의 평균을 손별 성능으로 보고, 손 사이 차이는 물체를 짝으로 한 대응 t 검정(df = 6)으로 봅니다. 물체는 고정된 7개이므로 이 검정은 "이 7개 물체에서의 차이"입니다. 일반화 검정이 아닙니다.

| 손 | 평균 (AUC×100) | 95% 구간 (물체 7개) |
|---|---|---|
| Allegro | 83.4 | [77.9, 88.9] |
| XHand | 82.1 | [73.3, 90.9] |
| Schunk | 73.4 | [57.4, 89.4] |
| Inspire | 62.8 | [39.2, 86.4] |

| 비교 | 평균 차이 | t(6) | p |
|---|---|---|---|
| Allegro − XHand | +1.3 | 0.38 | 0.72 |
| Allegro − Schunk | +10.0 | 1.80 | 0.12 |
| XHand − Schunk | +8.7 | 1.52 | 0.18 |
| Schunk − Inspire | +10.6 | 1.91 | 0.11 |
| Allegro − Inspire | +20.7 | 2.31 | 0.06 |

**§5.3 주장 대조:** "크고 완전 구동된 손이 더 높은 성능을 준다"는 방향은 집계 값과 맞습니다(Allegro, XHand > Schunk > Inspire). 그러나 손 사이 차이는 대부분 p > 0.05입니다. 특히 Schunk가 Inspire보다 낫다는 문장(§5.3)은 평균 차이가 10.6 %p이고 p = 0.11입니다. "Ability"는 집계 파일에 없어 이 절에서 확인할 수 없습니다. "자유도가 크기보다 중요하다"는 주장도 자유도를 따로 재지 않았으므로 검사하지 못했습니다.

## 3. 제안 방법과 기준선 비교 (V3)

셀마다 `Ours − 기준선`을 구하고, 같은 물체끼리 대응 t 검정을 했습니다.

| 기준선 | 손 | 물체 수 | 평균 차이 | Ours가 더 높은 물체 | p (df) |
|---|---|---|---|---|---|
| Kinematic Only | 4 손 전부 | 7 | +51 ~ +71 | 7/7 (4손 모두) | < 0.001 |
| ObjDex 재구현 | Inspire | 7 | +16.4 | 5/7 | 0.069 (6) |
| | Allegro | 7 | +16.9 | 5/7 | 0.086 (6) |
| | XHand | 7 | +21.0 | 7/7 | 0.013 (6) |
| | Schunk | 7 | +31.1 | 7/7 | 0.017 (6) |
| Task + Aux | Inspire | 7 | +12.6 | 5/7 | 0.14 (6) |
| | Allegro | 7 | +15.6 | 5/7 | 0.068 (6) |
| | XHand | 7 | +22.2 | 6/7 | 0.008 (6) |
| | Schunk | 7 | +19.7 | 5/7 | 0.043 (6) |
| ManipTrans | 4 손 전부 | 4 | +36 ~ +49 | 4/4 (4손 모두) | 0.002 ~ 0.049 (3) |

- **Kinematic Only:** 가장 쉬운 비교입니다. Ours가 모든 셀에서 높고, 기준선이 3–23%라 큰 차이가 확실합니다.
- **ManipTrans:** 4개 물체(Ketchup-100, Box-200, Mixer-170, Ketchup-300)만 있습니다. 긴 과제 3개(Mixer-300, Notebook-300, Waffleiron-300)는 그림에 막대가 없습니다. 이 논문이 "긴 과제에서 커리큘럼이 중요하다"는 근거로 쓰는 비교가 긴 과제에서는 빠져 있습니다.
- **Task + Aux와 ObjDex:** 평균 차이는 모든 손에서 양수지만, Inspire와 Allegro는 p > 0.05입니다. 물체 단위로 보면 Ours가 낮은 셀이 9개 있습니다(표 V3-b).

**V3-b. Ours가 기준선보다 낮은 셀 (28셀 중 9셀, 비교 11건):**

| 손 | 물체 | 대비 | 차이 (%p) |
|---|---|---|---|
| Inspire | Ketchup-100 | Task + Aux | −17.1 |
| Inspire | Waffleiron-300 | Task + Aux | −5.9 |
| Inspire | Waffleiron-300 | ObjDex | −0.7 |
| Inspire | Mixer-300 | ObjDex | −2.3 |
| Allegro | Ketchup-100 | Task + Aux | −0.6 |
| Allegro | Ketchup-100 | ObjDex | −0.2 |
| Allegro | Mixer-170 | Task + Aux | −1.3 |
| Allegro | Mixer-300 | ObjDex | −5.2 |
| XHand | Ketchup-100 | Task + Aux | −1.1 |
| Schunk | Ketchup-100 | Task + Aux | −0.3 |
| Schunk | Notebook-300 | Task + Aux | −11.5 |

**§5.1 주장 대조:** 본문 굵은 글씨 "DexMachina consistently improves performance on all hands and tasks"는 이 표와 맞지 않습니다. 대부분 셀에서 우위이지만, 짧은 과제 Ketchup-100에서는 4개 손 모두 Task + Aux가 약간 또는 크게 앞섭니다. 세 손은 차이가 1 %p 안쪽(−0.3 ~ −1.1)이라 동률에 가깝고, Inspire는 −17.1입니다. 나머지 큰 역전은 Schunk Notebook-300(−11.5)입니다. 본문은 "rare exceptions"라고 씁니다(Fig. 3 캡션). 역전 셀이 9개(28셀의 32%)라면 "드문 예외"라는 표현은 과합니다.

**ManipTrans 재구현 비교 (§5.2):** 본문은 ManipTrans 커리큘럼이 "커리큘럼 없음과 비슷하다"고 씁니다. 그림에서는 ManipTrans 막대가 Task + Aux보다 낮습니다(예: XHand Ketchup-100 54.6 대 91.5). 다만 ManipTrans는 원 설정과 다르게 재구현한 것이고(App. B.3), 그 선택이 결과에 준 영향은 이 자료로 가릴 수 없습니다.

## 4. 부록 B.3 문장과 Fig. 3 값 (V4)

부록 B.3은 ObjDex 재구현이 "Ketchup-100에서 모든 손이 90%를 넘고(원 논문 41.2%), Mixer-170에서 네 손 중 세 손이 70%를 넘는다(원 논문 57.6%)"고 씁니다.

- Ketchup-100의 Task Rew (ObjDex) 막대: Inspire 68.8, Allegro 91.6, XHand 89.9, Schunk 88.6. **90을 넘는 손은 1개입니다.**
- Mixer-170의 Task Rew (ObjDex) 막대: Inspire 63.2, Allegro 74.1, XHand 70.7, Schunk 21.0. **70을 넘는 손은 2개입니다.**

본문이 가리키는 "재구현"이 Fig. 3의 "Task Rew (ObjDex)"인지 불분명합니다. Task + Aux 막대로 보면 Ketchup-100에서 90을 넘는 손은 Allegro 92.0과 XHand 91.5 두 개(Schunk 90.0은 기준과 같음)이고, Mixer-170에서 70을 넘는 손은 Inspire 71.8과 Allegro 78.6 두 개뿐입니다. 어느 막대로 봐도 부록 문장과 맞지 않습니다. 저자의 확인이 필요합니다.

## 5. 시연 이름과 프레임 수 (V5)

본문 Fig. 3의 이름(예: Ketchup-100)은 코드의 시작-끝 프레임 이름(ketchup30-130)에서 옮겨졌습니다. 끝 − 시작 값이 이름의 숫자와 일치합니다.

| 코드 이름 | 프레임 수 (끝 − 시작) | 논문 이름 |
|---|---|---|
| ketchup30-130 | 100 | Ketchup-100 |
| box30-230 | 200 | Box-200 |
| mixer30-200 | 170 | Mixer-170 |
| ketchup40-340, mixer40-340, notebook40-340, waffleiron40-340 | 300 | Ketchup-300, Mixer-300, Notebook-300, Waffleiron-300 |

코드의 에피소드 길이도 `end − start`로 정해집니다(`envs/constructors.py` 88행). 이는 App. A.1의 T = end − start와 맞습니다.

## 6. 기타 일관성 (V6–V9)

- **V6 seed:** 28셀 모두 n = 5, seed 집합 동일(§1).
- **V7 실험 규모:** 본문 B.1은 학습 env 12,000개와 L40s 또는 H100 한 장을 씁니다. 공식 예제 스크립트(`examples/train_rl.sh`)는 `-B 4096`이고 메인 결과 조건은 `group_cfgs/dexmachina_main.yaml`에 있습니다. 예제와 논문 설정은 다릅니다(04 D8). 이는 재현 설명의 차이이고 결과 모순은 아닙니다.
- **V8 행동 절제 (Fig. 8):** 3 시드입니다(B.1). 메인 실험과 다른 시드 수입니다. 절제 결과는 이 저장소에 없어 검사하지 못했습니다.
- **V9 실험 횟수:** 시드 5개 × 평가 에피소드 20개(B.4)입니다. 20개 에피소드는 seed 안에서 평균되므로 독립 단위가 아닙니다. 독립 단위는 seed 5개입니다.

## 7. 결론

1. 재계산 가능한 DexMachina 수치(28셀)는 집계 파일과 그림 모두에서 일치합니다(V1). 지표는 성공률이 아니라 AUC입니다.
2. 손 사이 차이는 대부분 통계적으로 확인되지 않습니다(V2). 손 설계 결론은 방향만 지지됩니다.
3. 기준선 대비 우위는 Kinematic과 ManipTrans에서는 뚜렷하고, ObjDex와 Task + Aux에서는 셀 단위로 9개가 뒤집힙니다(V3). "모든 손과 과제에서 일관된 개선"은 이 자료로 확인되지 않습니다.
4. 부록 B.3의 ObjDex 수치 문장은 그림 3과 맞지 않습니다(V4).
5. 기준선에 seed 수준 자료와 통계 검정이 없습니다. "significantly outperforms" 표현은 검정으로 뒷받침되지 않습니다.

## 8. 재계산이 하지 못한 것 (추가 자료가 필요한 검정)

- seed별 값이 있으면 기준선의 대응 검정(seed 단위, n = 5)과 셀별 Welch 검정이 가능합니다. 이 자료는 README의 Google Drive 링크에 있고 크기는 확인하지 못했습니다. 받으려면 승인이 필요합니다(크기를 먼저 확인해 보고).
- Ability와 DexRobot 결과는 그림 5에만 있어 판독이 필요합니다.
- Fig. 8 행동 절제 값은 판독하지 않았습니다.
