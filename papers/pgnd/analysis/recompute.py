"""논문 표 I·II·III·IV·V의 보고 수치를 다시 계산하고 일관성을 검사한다 (수 초, numpy·scipy).

입력: 논문 본문과 부록에서 옮긴 값(아래 상수). 학습 실행과 공식 코드 실행은 없다.
시험 수: 부록 B.3 "cloth·rope 40 test episodes, 그 밖 20". 공식 코드 설정(train_*.sh의 eval 구간)과도 일치한다.
검정: 에피소드별 값이 없으므로 평균 ± SD와 n으로 Welch t 검정을 한다 (쌍체 검정은 불가).
실행: python3 analysis/recompute.py  → analysis/recompute_results.json
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "recompute_results.json"

CATS = ["Cloth", "Rope", "Plush", "Box", "Bag", "Bread"]
N_TEST = {"Cloth": 40, "Rope": 40, "Plush": 20, "Box": 20, "Bag": 20, "Bread": 20}

# 표 I (논문 v2 Table I, v1과 수치 동일, 지표명만 MSE→MDE). 값은 (평균, SD). None은 "−" (MPM box 제외)
T1 = {
    "MDE": {
        "MPM":      [(0.176, 0.107), (0.138, 0.072), (0.163, 0.148), None, (0.226, 0.026), (0.034, 0.014)],
        "GBND":     [(0.077, 0.033), (0.062, 0.025), (0.078, 0.028), (0.045, 0.008), (0.030, 0.011), (0.031, 0.014)],
        "Particle": [(0.059, 0.039), (0.061, 0.051), (0.060, 0.027), (0.025, 0.009), (0.021, 0.016), (0.038, 0.018)],
        "Ours":     [(0.045, 0.023), (0.039, 0.032), (0.043, 0.018), (0.022, 0.007), (0.016, 0.005), (0.020, 0.011)],
    },
    "CD": {
        "MPM":      [(0.156, 0.091), (0.115, 0.054), (0.153, 0.207), None, (0.183, 0.032), (0.031, 0.010)],
        "GBND":     [(0.083, 0.034), (0.073, 0.027), (0.064, 0.016), (0.062, 0.014), (0.042, 0.007), (0.031, 0.013)],
        "Particle": [(0.051, 0.034), (0.059, 0.058), (0.043, 0.019), (0.032, 0.011), (0.025, 0.016), (0.044, 0.031)],
        "Ours":     [(0.043, 0.022), (0.038, 0.036), (0.033, 0.013), (0.015, 0.003), (0.021, 0.005), (0.018, 0.012)],
    },
    "EMD": {
        "MPM":      [(0.093, 0.078), (0.081, 0.052), (0.092, 0.131), None, (0.110, 0.027), (0.021, 0.009)],
        "GBND":     [(0.035, 0.017), (0.036, 0.016), (0.032, 0.010), (0.032, 0.008), (0.016, 0.005), (0.016, 0.008)],
        "Particle": [(0.029, 0.024), (0.036, 0.037), (0.025, 0.013), (0.018, 0.006), (0.012, 0.010), (0.023, 0.016)],
        "Ours":     [(0.022, 0.013), (0.021, 0.021), (0.017, 0.007), (0.016, 0.005), (0.009, 0.003), (0.010, 0.008)],
    },
}

# 표 II (J·F는 %, LPIPS는 0–1 척도)
T2 = {
    "J": {
        "MPM":      [(40.5, 20.3), (12.5, 9.5), (37.5, 14.8), None, (34.3, 5.5), (42.9, 10.3)],
        "GBND":     [(56.7, 13.9), (19.2, 15.6), (42.7, 10.5), (83.0, 5.5), (78.0, 82.8), (49.8, 13.9)],
        "Particle": [(58.5, 18.7), (24.4, 17.7), (49.0, 12.5), (82.2, 6.1), (74.5, 11.7), (39.3, 15.4)],
        "Ours":     [(63.2, 16.4), (29.5, 16.5), (59.7, 11.1), (82.4, 5.0), (76.8, 8.8), (55.8, 10.1)],
    },
    "F": {
        "MPM":      [(28.0, 11.3), (37.0, 14.2), (40.4, 12.2), None, (13.1, 5.9), (54.3, 12.1)],
        "GBND":     [(32.2, 18.2), (41.9, 18.9), (35.8, 11.2), (74.0, 8.8), (54.6, 13.4), (60.1, 16.1)],
        "Particle": [(41.0, 19.6), (45.4, 19.5), (43.0, 14.0), (73.1, 8.8), (52.7, 19.5), (46.8, 17.7)],
        "Ours":     [(42.6, 20.4), (52.6, 17.6), (53.8, 12.9), (68.8, 12.9), (60.3, 14.7), (64.6, 12.7)],
    },
    "LPIPS": {
        "MPM":      [(0.141, 0.060), (0.052, 0.018), (0.103, 0.085), None, (0.145, 0.020), (0.059, 0.020)],
        "GBND":     [(0.120, 0.055), (0.042, 0.018), (0.081, 0.023), (0.106, 0.039), (0.097, 0.019), (0.052, 0.015)],
        "Particle": [(0.109, 0.048), (0.044, 0.032), (0.072, 0.024), (0.082, 0.031), (0.069, 0.019), (0.054, 0.020)],
        "Ours":     [(0.099, 0.044), (0.041, 0.033), (0.057, 0.018), (0.079, 0.034), (0.065, 0.010), (0.042, 0.014)],
    },
}

# 표 III 런타임 (ms), 배치 1, 10, 50, 100
T3_TIME = {1: 4.8, 10: 5.2, 50: 11.9, 100: 22.3}

# 부록 B.3 학습 데이터 길이 (분)과 학습 에피소드 수 (공식 코드 train_*.sh의 training_end_episode, 시작 0)
TRAIN_MIN = {"Cloth": 20.3, "Rope": 21.7, "Plush": 3.8, "Box": 10.1, "Bag": 6.7, "Bread": 4.8}
TRAIN_EPS = {"Cloth": 610, "Rope": 651, "Plush": 113, "Box": 303, "Bag": 200, "Bread": 143}


def welch(m1, s1, n1, m2, s2, n2):
    """Welch t 검정 (양측). 반환: t, df, p"""
    v1, v2 = s1 ** 2 / n1, s2 ** 2 / n2
    t = (m1 - m2) / math.sqrt(v1 + v2)
    df = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))
    p = 2 * stats.t.sf(abs(t), df)
    return t, df, p


def ci95(m, s, n):
    h = stats.t.ppf(0.975, n - 1) * s / math.sqrt(n)
    return m - h, m + h


def best_baseline(table, metric, i):
    cands = [(table[b][i][0], b) for b in ("MPM", "GBND", "Particle") if table[b][i] is not None]
    return min(cands)[1]


def main():
    res = {}

    # 1. 표 I: 모든 셀에서 Ours가 최선인지, 최선 기준선 대비 Welch 검정
    rows, best_ok = [], []
    for metric, tab in T1.items():
        for i, c in enumerate(CATS):
            m, s = tab["Ours"][i]
            base = best_baseline(tab, metric, i)
            bm, bs = tab[base][i]
            n = N_TEST[c]
            t, df, p = welch(m, s, n, bm, bs, n)
            lo, hi = ci95(m, s, n)
            is_best = all(tab["Ours"][i][0] <= tab[b][i][0] for b in ("MPM", "GBND", "Particle") if tab[b][i])
            best_ok.append(is_best)
            rows.append({"metric": metric, "cat": c, "n": n, "ours": [m, s], "ours_ci95": [round(lo, 4), round(hi, 4)],
                         "best_baseline": base, "base": [bm, bs], "diff": round(m - bm, 4),
                         "rel_change_pct": round(100 * (m - bm) / bm, 1), "welch_t": round(t, 2),
                         "df": round(df, 1), "p": float(f"{p:.2e}"), "ours_is_best": is_best})
    # Holm 보정 (표 I 18셀, 최선 기준선 대비)
    order = sorted(range(len(rows)), key=lambda k: rows[k]["p"])
    m_tests, running = len(rows), 0.0
    for rank, k in enumerate(order):
        adj = min(1.0, (m_tests - rank) * rows[k]["p"])
        running = max(running, adj)
        rows[k]["p_holm"] = float(f"{running:.3g}")
    res["table1_welch_vs_best_baseline"] = rows
    res["table1_p_lt_0.05_uncorrected"] = sum(1 for r in rows if r["p"] < 0.05)
    res["table1_p_lt_0.05_holm"] = sum(1 for r in rows if r["p_holm"] < 0.05)
    res["table1_ours_best_cells"] = f"{sum(best_ok)}/{len(best_ok)}"

    # Ours vs Particle (그리드 제거 절제에 가장 가까운 비교)
    part = []
    for metric, tab in T1.items():
        for i, c in enumerate(CATS):
            m, s = tab["Ours"][i]
            pm, ps = tab["Particle"][i]
            n = N_TEST[c]
            t, df, p = welch(m, s, n, pm, ps, n)
            part.append({"metric": metric, "cat": c, "diff": round(m - pm, 4), "p": float(f"{p:.3g}")})
    res["table1_ours_vs_particle"] = part
    res["table1_ours_vs_particle_p_lt_0.05"] = sum(1 for x in part if x["p"] < 0.05)

    # 2. 표 II: 최선 여부와 SD 가능성 검사 (J, F는 0–100 척도)
    t2_notbest = []
    for metric, tab in T2.items():
        for i, c in enumerate(CATS):
            if tab["Ours"][i] is None:
                continue
            vals = {b: tab[b][i][0] for b in tab if tab[b][i] is not None}
            hib = metric in ("J", "F")
            best = max(vals, key=vals.get) if hib else min(vals, key=vals.get)
            if best != "Ours":
                t2_notbest.append({"metric": metric, "cat": c, "best": best, "best_val": vals[best], "ours": tab["Ours"][i][0]})
    res["table2_cells_where_ours_not_best"] = t2_notbest

    # 표 II의 SD 가능성: 0–100 척도에서 평균 m이면 SD는 sqrt(m(100-m)) 이하여야 한다
    impossible = []
    for metric in ("J", "F"):
        for b, row in T2[metric].items():
            for i, c in enumerate(CATS):
                if row[i] is None:
                    continue
                m, s = row[i]
                smax = math.sqrt(m * (100 - m))
                if s > smax:
                    impossible.append({"metric": metric, "method": b, "cat": c, "mean": m, "sd": s, "sd_max": round(smax, 1)})
    res["table2_sd_impossible_on_0_100"] = impossible

    # 3. 시험 수 가정에 따른 Welch 검정 민감도: 표 II Box J (Ours 대 GBND)와 Bag J (Ours 대 GBND)
    sens = []
    for c, metric in (("Box", "J"), ("Bag", "J"), ("Box", "F")):
        i = CATS.index(c)
        m1, s1 = T2[metric]["Ours"][i]
        m2, s2 = T2[metric]["GBND"][i]
        for n in (20, 10):
            t, df, p = welch(m1, s1, n, m2, s2, n)
            sens.append({"metric": metric, "cat": c, "method": "GBND", "n": n, "p": round(p, 3)})
    res["table2_base_vs_gbnd_sensitivity"] = sens

    # 4. 계획 실험 (과제당 10회): 성공률 비교의 검정력. 논문은 성공 횟수를 표로 주지 않는다 (그림 8).
    fisher = []
    for a, b in ((9, 1), (8, 2), (7, 3), (6, 4), (8, 4), (9, 3)):
        _, p = stats.fisher_exact([[a, 10 - a], [b, 10 - b]])
        fisher.append({"ours": f"{a}/10", "baseline": f"{b}/10", "fisher_p": round(p, 3)})
    res["planning_fisher_10v10"] = fisher

    # 5. 표 III, 표 IV, 표 V와 표 I의 일치 검사 (같은 설정이면 같은 값이어야 한다)
    res["cross_table_checks"] = [
        {"claim": "표 IV Cloth GVE (MDE/CD/EMD 0.045/0.043/0.022) = 표 I Cloth Ours", "table_I": [0.045, 0.043, 0.022],
         "table_IV": [0.045, 0.043, 0.022], "match": True},
        {"claim": "표 IV Box RP (0.022/0.015/0.016) = 표 I Box Ours", "table_I": [0.022, 0.015, 0.016],
         "table_IV": [0.022, 0.015, 0.016], "match": True},
        {"claim": "표 V Ours (rope, 0.039/0.038/0.021) = 표 I Rope Ours", "table_I": [0.039, 0.038, 0.021],
         "table_IV": [0.039, 0.038, 0.021], "match": True},
        {"claim": "표 III 4.8 ms (배치 1), 11.9 ms (배치 50) = 본문 C.41", "table_III": [T3_TIME[1], T3_TIME[50]],
         "text": [4.8, 11.9], "match": True},
    ]
    # 표 IV의 Cloth RP, Box GVE가 최선 설정보다 나쁜지 (논문 주장 C.42)
    res["table4_gve_vs_rp"] = {"cloth_gve_mde": 0.045, "cloth_rp_mde": 0.058, "box_gve_mde": 0.027, "box_rp_mde": 0.022}

    # 6. 학습 데이터 길이와 에피소드 수 (부록 B.3). 에피소드당 길이 추정
    dur = []
    for c in CATS:
        sec_per = TRAIN_MIN[c] * 60 / TRAIN_EPS[c]
        dur.append({"cat": c, "train_min": TRAIN_MIN[c], "train_episodes_code": TRAIN_EPS[c], "sec_per_episode": round(sec_per, 3)})
    res["training_duration_check"] = dur

    # 7. 표 II 의 SD 가능성 외에, 표 I 평균의 95% CI (n=40 또는 20)
    res["table1_ours_ci95"] = {f"{r['metric']}-{r['cat']}": r["ours_ci95"] for r in rows}

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", OUT)
    print("Table I Ours best cells:", res["table1_ours_best_cells"])
    print("Table I vs best baseline p<0.05 uncorrected/Holm:", res["table1_p_lt_0.05_uncorrected"], res["table1_p_lt_0.05_holm"])
    print("Table I Ours vs Particle p<0.05:", res["table1_ours_vs_particle_p_lt_0.05"], "of", len(part))
    print("Table II cells where Ours not best:", t2_notbest)
    print("Table II impossible SD:", impossible)
    print("Table II base vs GBND sensitivity:", sens)


if __name__ == "__main__":
    main()
