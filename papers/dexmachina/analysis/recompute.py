"""DexMachina (arXiv:2505.24853) 논문 Figure 3 수치를 다시 계산하고 통계를 확인한다.

입력 1: 공식 코드 저장소의 집계 파일 dexmachina/eval/stats/dexmachina_main_stats_AUC.json
        (repo/ 는 git clone한 공식 코드, .gitignore에 있음). 각 셀의 평균(value)과
        seed 간 표준오차(err)만 들어 있다. seed별 값은 없다.
입력 2: Figure 3 막대 위 숫자 라벨(수기 판독, 소수 첫째 자리, %). 기준선 방법들은 이 값만 쓸 수 있다.
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)

가정과 주의
- 집계 파일의 err = np.std(seed값) / sqrt(n) 이다 (ddof=0). group_results.py 207-237행 확인.
  표본 표준편차(ddof=1)로 바꾸면 sqrt(n/(n-1)) = sqrt(5/4) 배가 된다. 여기서는 ddof=1로 고쳐 쓴다.
- 셀당 seed 수 n = 5 (논문 B.1, 집계 파일의 n_seeds). 20 episode는 seed 안에서 평균되므로 독립 단위가 아니다.
- 그림 3 막대 값은 집계 파일의 value x 100 과 일치해야 한다(검사 V1).
- 기준선(Kinematic, Task Rew(ObjDex), Task+Aux, ManipTrans)은 seed 수준 자료가 없다.
  그래서 물체 7개(ManipTrans는 4개)를 짝지은 검정만 한다(물체 수준, 셀 평균 사용).
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
STATS = HERE.parent / "repo" / "dexmachina" / "eval" / "stats" / "dexmachina_main_stats_AUC.json"
OUT = HERE / "recompute_results.json"

HANDS = ["inspire_hand", "allegro_hand", "xhand", "schunk_hand"]
HAND_KO = {"inspire_hand": "Inspire", "allegro_hand": "Allegro", "xhand": "XHand", "schunk_hand": "Schunk"}
OBJ_ORDER = [
    "ketchup30-130-s01-u01",
    "box30-230-s01-u01",
    "mixer30-200-s01-u01",
    "ketchup40-340-s01-u02",
    "mixer40-340-s01-u01",
    "notebook40-340-s02-u02",
    "waffleiron40-340-s01-u01",
]
OBJ_LABEL = ["Ketchup-100", "Box-200", "Mixer-170", "Ketchup-300", "Mixer-300", "Notebook-300", "Waffleiron-300"]

# Figure 3 막대 라벨 (수기 판독). 순서: Kinematic Only, Task Rew (ObjDex), Task + Aux, Ours, ManipTrans
# None = 그림에 막대가 없음 (ManipTrans는 Ketchup-100, Box-200, Mixer-170, Ketchup-300에만 있음)
FIG3 = {
    "inspire_hand": [
        [13.6, 68.8, 87.8, 70.7, 42.6], [19.8, 37.0, 64.3, 87.1, 19.3], [23.0, 63.2, 71.8, 89.8, 37.2],
        [3.7, 22.6, 13.3, 32.3, 5.8], [7.4, 66.6, 55.6, 64.3, None], [6.5, 41.4, 28.8, 71.2, None],
        [6.0, 24.5, 29.7, 23.8, None]],
    "allegro_hand": [
        [11.7, 91.6, 92.0, 91.4, 82.2], [40.6, 53.0, 77.7, 88.6, 56.3], [20.1, 74.1, 78.6, 77.3, 39.9],
        [3.8, 28.1, 31.2, 83.0, 19.8], [8.3, 86.3, 72.4, 81.1, None], [5.6, 66.2, 60.1, 87.1, None],
        [5.6, 66.6, 62.9, 75.4, None]],
    "xhand": [
        [12.3, 89.9, 91.5, 90.4, 54.6], [19.6, 42.0, 65.5, 86.5, 29.8], [20.3, 70.7, 54.9, 89.9, 36.7],
        [3.4, 33.1, 31.0, 72.4, 22.8], [8.3, 59.4, 58.9, 66.2, None], [6.1, 73.2, 61.6, 89.0, None],
        [5.6, 59.7, 56.2, 80.3, None]],
    "schunk_hand": [
        [14.2, 88.6, 90.0, 89.7, 76.0], [24.2, 42.4, 51.7, 85.9, 54.0], [20.1, 21.0, 62.0, 90.3, 34.5],
        [3.9, 22.6, 25.0, 70.0, 26.6], [8.0, 41.6, 55.2, 66.8, None], [5.6, 69.4, 81.1, 69.6, None],
        [5.6, 10.4, 10.8, 41.4, None]],
}
BASE_NAMES = ["Kinematic", "ObjDex-re", "TaskAux", "Ours", "ManipTrans"]
T975 = {n: float(stats.t.ppf(0.975, n)) for n in (4, 6)}

out = {"source": "dexmachina/eval/stats/dexmachina_main_stats_AUC.json", "checks": {}, "cells": {},
       "fig3_labels": {h: [[x for x in row] for row in FIG3[h]] for h in HANDS}}

with open(STATS, encoding="utf-8") as f:
    raw = json.load(f)
cells = raw["avg_success_rates"]["method1"]

# V1: 집계 파일 value x100 과 그림 3 Ours 라벨 비교, err 재계산
mismatch = []
ns = set()
cell_rows = {}
for h in HANDS:
    for j, o in enumerate(OBJ_ORDER):
        c = cells[h][o]
        n = c["n_seeds"]
        ns.add(n)
        mean = 100 * c["value"]
        sem0 = 100 * c["err"]                       # np.std ddof=0 / sqrt(n)
        sem1 = sem0 * math.sqrt(n / (n - 1))        # ddof=1 로 고친 SEM
        tcrit = float(stats.t.ppf(0.975, n - 1))
        lo, hi = mean - tcrit * sem1, mean + tcrit * sem1
        fig = FIG3[h][j][3]
        if round(mean, 1) != fig:
            mismatch.append((h, OBJ_LABEL[j], round(mean, 2), fig))
        cell_rows[(h, j)] = dict(mean=mean, sem_ddof0=sem0, sem_ddof1=sem1, ci95=[lo, hi], n=n,
                                 fig_ours=fig)
        out["cells"][f"{h}|{OBJ_LABEL[j]}"] = dict(
            mean=round(mean, 2), sem_ddof0=round(sem0, 2), sem_ddof1=round(sem1, 2),
            ci95=[round(lo, 2), round(hi, 2)], n=n, fig_ours=fig)
out["checks"]["V1_json_vs_fig3_ours"] = dict(
    n_seeds_values=sorted(ns), n_cells=len(cell_rows), mismatches=mismatch)

# V2: 손(hand)별 평균 (물체 7개 평균), 물체를 짝으로 본 t 검정
hand_mean = {}
for h in HANDS:
    vals = np.array([cell_rows[(h, j)]["mean"] for j in range(7)])
    m = vals.mean()
    se = vals.std(ddof=1) / math.sqrt(7)
    hand_mean[h] = vals
    out.setdefault("hand_avg", {})[HAND_KO[h]] = dict(
        mean=round(m, 2), ci95=[round(m - T975[6] * se, 2), round(m + T975[6] * se, 2)])
pairs = [("allegro_hand", "xhand"), ("schunk_hand", "inspire_hand"), ("xhand", "schunk_hand"),
         ("allegro_hand", "schunk_hand"), ("allegro_hand", "inspire_hand")]
hand_tests = []
for a, b in pairs:
    d = hand_mean[a] - hand_mean[b]
    t, p = stats.ttest_rel(hand_mean[a], hand_mean[b])
    hand_tests.append(dict(pair=f"{HAND_KO[a]} - {HAND_KO[b]}", mean_diff=round(float(d.mean()), 2),
                           t=round(float(t), 2), p=round(float(p), 4), df=6))
out["checks"]["V2_hand_pairs_paired_over_7_objects"] = hand_tests

# V3: Ours vs 기준선 (그림 라벨, 물체 짝 t 검정, 방향과 승패)
base_idx = {"Kinematic": 0, "ObjDex-re": 1, "TaskAux": 2, "ManipTrans": 4}
v3 = []
exceptions = []
for base, bi in base_idx.items():
    for h in HANDS:
        pairs_o, ours, other, labels = [], [], [], []
        for j in range(7):
            b = FIG3[h][j][bi]
            if b is None:
                continue
            ours.append(FIG3[h][j][3]); other.append(b); labels.append(OBJ_LABEL[j])
        ours = np.array(ours, float); other = np.array(other, float)
        d = ours - other
        row = dict(baseline=base, hand=HAND_KO[h], n_objects=len(d),
                   mean_diff=round(float(d.mean()), 2), ours_wins=int((d > 0).sum()))
        if len(d) >= 3:
            t, p = stats.ttest_rel(ours, other)
            row.update(t=round(float(t), 2), p=round(float(p), 4), df=len(d) - 1)
        v3.append(row)
        for lab, dd in zip(labels, d):
            if dd < 0:
                exceptions.append(dict(baseline=base, hand=HAND_KO[h], object=lab, diff=round(float(dd), 1)))
out["checks"]["V3_ours_vs_baselines"] = v3
out["checks"]["V3_ours_below_baseline_cells"] = exceptions

# V4: 부록 B.3 문장 "Ketchup-100에서 ObjDex 재구현이 모든 손에서 90% 초과, Mixer-170에서 세 손에서 70% 초과"
kc = [FIG3[h][0][1] for h in HANDS]
mc = [FIG3[h][2][1] for h in HANDS]
out["checks"]["V4_appendix_B3_objdex_claim"] = dict(
    ketchup100_objdex_values=kc, n_over_90=int(sum(v > 90 for v in kc)),
    mixer170_objdex_values=mc, n_over_70=int(sum(v > 70 for v in mc)),
    note="부록 B.3 문장(모든 손 90% 초과, 세 손 70% 초과)과 그림 3 막대 값이 맞지 않는다.")

# V5: 본문 "다섯 개 중 두 번째 Ours 평균 순서" 확인에 쓸 손별 평균(Ours만)
out["checks"]["V5_note"] = ("seed 수준 자료가 없어 셀 신뢰구간은 집계 err에서 역산한 값이다. "
                            "기준선 막대의 오차막대는 그림에서 판독하지 않았다.")

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

# 요약 출력
print("V1 n_seeds:", sorted(ns), "cells:", len(cell_rows), "mismatch:", mismatch)
print("hand avg (Ours, 7 objects):")
for k, v in out["hand_avg"].items():
    print(" ", k, v)
for t in hand_tests:
    print(" V2", t)
for r in v3:
    print(" V3", r)
print(" V3 below-baseline cells:", len(exceptions))
for e in exceptions:
    print("   ", e)
print(" V4", out["checks"]["V4_appendix_B3_objdex_claim"])
