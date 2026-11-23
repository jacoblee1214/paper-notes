"""WT-UMI (arXiv:2606.13232 v1) 보고 수치 재계산과 검정.

표 1–5와 본문·부록 서술 수치를 옮겨 적고 (1) 정수 격자 검사 (2) 평균·비율 재계산 (3) 본문 서술 수치 검사
(4) Wilson, Fisher, CMH, Holm 검정 (5) 검정력 계산 (6) 힘 헤드 매개변수 하한 계산을 한다. 학습이나 실험은 하지 않는다.
시행 수는 논문 5.2절과 부록 8.6절에 있는 값(과제당 25회)을 쓴다. 표 5의 시행 수는 논문에 없어 정수 해만 찾는다.

실행: python3 analysis/recompute.py   -> analysis/recompute_results.json (numpy, scipy, statsmodels, 수 초)
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import StratifiedTable

OUT = Path(__file__).resolve().parent / "recompute_results.json"
N = 25
R = {}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = [0.0] * m
    run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (m - i) * ps[idx])
        adj[idx] = min(1.0, run)
    return adj


def grid(pct, n):
    """퍼센트(소수 둘째 자리까지 보고)가 k/n 반올림으로 나올 수 있는 k 목록."""
    return [k for k in range(n + 1) if abs(100 * k / n - pct) < 0.5 * 10 ** (-(1 if pct == round(pct, 1) else 2)) + 1e-9]


# ---------- 표 2 (5.2절, ViT-FMT, 과제당 N = 25) ----------
# 열: 성공%(Tel, Comb), 드리프트 mm, 힘 N, 병진 가속도 m/s2, 회전 가속도 rad/s2
T2 = {
    "T1": dict(succ=(100.0, 100.0), drift=(20.41, 18.12), force=(3.27, 4.77), trans=(4.08, 3.14), rot=(22.93, 20.29)),
    "T2": dict(succ=(100.0, 100.0), drift=(24.61, 21.04), force=(0.17, 0.52), trans=(6.03, 1.87), rot=(34.54, 12.63)),
    "T3": dict(succ=(60.0, 80.0), drift=(26.01, 25.00), force=(0.66, 0.96), trans=(3.49, 1.85), rot=(27.82, 14.05)),
}
MIN_TEL = [2.2, 2.2, 1.4]
MIN_COMB = [13.2, 15.7, 8.8]

# ---------- 표 3 = 표 4의 ViT-FMT 행 ----------
# 열: succ(w/o, w/), drift, force, trans, rot (각각 w/o, w/)
T4 = {
    "ViT-FMT": {
        "T1": dict(succ=(100, 100), drift=(18.12, 15.67), force=(4.77, 5.50), trans=(3.14, 2.98), rot=(20.29, 18.56)),
        "T2": dict(succ=(100, 100), drift=(21.04, 19.44), force=(0.52, 0.13), trans=(1.87, 1.95), rot=(12.63, 13.61)),
        "T3": dict(succ=(80, 92), drift=(25.00, 22.08), force=(0.96, 1.61), trans=(1.85, 1.29), rot=(14.05, 10.38)),
    },
    "ViT-DiT": {
        "T1": dict(succ=(52, 52), drift=(21.79, 18.00), force=(2.21, 2.57), trans=(2.92, 2.54), rot=(18.73, 14.97)),
        "T2": dict(succ=(100, 100), drift=(21.22, 19.61), force=(0.19, 0.76), trans=(4.67, 2.41), rot=(25.93, 15.17)),
        "T3": dict(succ=(52, 52), drift=(26.22, 22.20), force=(1.74, 0.93), trans=(2.33, 2.05), rot=(16.47, 14.79)),
    },
    "pi0.5": {
        "T1": dict(succ=(88, 92), drift=(21.51, 11.78), force=(2.53, 2.80), trans=(4.69, 4.82), rot=(29.33, 29.79)),
        "T2": dict(succ=(68, 76), drift=(19.78, 15.97), force=(0.50, 0.50), trans=(5.06, 5.03), rot=(31.86, 31.22)),
        "T3": dict(succ=(84, 76), drift=(20.18, 19.40), force=(2.57, 3.38), trans=(4.11, 3.53), rot=(27.26, 23.56)),
    },
    "Psi0": {
        "T1": dict(succ=(88, 92), drift=(15.80, 13.56), force=(3.14, 3.03), trans=(4.20, 3.27), rot=(27.06, 23.01)),
        "T2": dict(succ=(89, 96), drift=(20.48, 18.69), force=(2.50, 2.50), trans=(4.94, 5.44), rot=(32.12, 33.04)),
        "T3": dict(succ=(0, 0), drift=None, force=None, trans=None, rot=None),
    },
}
# 5.4절 본문의 값 (병진 m/s2, 회전 rad/s2)
TEXT54 = {"ViT-FMT": (2.07, 14.18), "pi0.5": (4.62, 28.45), "ViT-DiT": (2.90, 18.35), "Psi0": (3.38, 22.83)}

# ---------- 표 5 (부록 8.7, 데이터 재생) ----------
T5 = {
    "Raw Teleoperation": dict(succ=85.35, drift=15.59, force=3.30, trans=3.93, rot=26.42),
    "Correction": dict(succ=89.29, drift=17.88, force=3.23, trans=3.57, rot=25.51),
    "Correction + Admittance": dict(succ=96.15, drift=11.46, force=3.02, trans=4.06, rot=27.15),
}

# ---------- 표 1 ----------
T1 = {"Human": dict(rmse=1.05, lag=68, rate_meas=5.86, rate_pred=3.74), "Teleop": dict(rmse=2.07, lag=151, rate_meas=30.62, rate_pred=19.80)}


def pct_change(a, b):
    """a -> b 상대 변화 (%)."""
    return 100 * (b - a) / a


def main():
    # ===== V1: 정수 격자 (N = 25) =====
    vals = {"T2:Tel/T1": 100.0}
    cells = []
    for t, d in T2.items():
        for j, name in enumerate(("Tel", "Comb")):
            cells.append((f"표2 {t} {name}", d["succ"][j]))
    for b, tasks in T4.items():
        for t, d in tasks.items():
            for j, name in enumerate(("w/o", "w/")):
                cells.append((f"표4 {b} {t} {name}", float(d["succ"][j])))
    v1 = []
    for label, p in cells:
        ks = [k for k in range(N + 1) if abs(100 * k / N - p) <= 0.5]
        exact = [k for k in range(N + 1) if abs(100 * k / N - p) < 1e-9]
        v1.append(dict(cell=label, pct=p, k_exact=exact, k_within_half=ks))
    R["V1_grid"] = v1
    R["V1_bad"] = [c["cell"] for c in v1 if not c["k_exact"]]

    # ===== V2: 표 2 서술 수치 =====
    tel_avg = np.mean([d["succ"][0] for d in T2.values()])
    comb_avg = np.mean([d["succ"][1] for d in T2.values()])
    share = [100 * a / b for a, b in zip(MIN_TEL, MIN_COMB)]
    drift_rel_over_comb = np.mean([100 * (d["drift"][0] / d["drift"][1] - 1) for d in T2.values()])
    drift_red = np.mean([100 * (1 - d["drift"][1] / d["drift"][0]) for d in T2.values()])
    force_ratio = [d["force"][1] / d["force"][0] for d in T2.values()]
    trans_red = [100 * (1 - d["trans"][1] / d["trans"][0]) for d in T2.values()]
    rot_red = [100 * (1 - d["rot"][1] / d["rot"][0]) for d in T2.values()]
    R["V2"] = dict(
        tel_avg=round(float(tel_avg), 2), comb_avg=round(float(comb_avg), 2),
        tel_share_pct=[round(x, 1) for x in share], human_min=[round(b - a, 1) for a, b in zip(MIN_TEL, MIN_COMB)],
        human_to_tel_ratio=[round((b - a) / a, 2) for a, b in zip(MIN_TEL, MIN_COMB)],
        comb_over_tel_data=[round(b / a, 2) for a, b in zip(MIN_TEL, MIN_COMB)],
        drift_tel_larger_than_comb_rel_pct_mean=round(float(drift_rel_over_comb), 2),
        drift_comb_reduction_rel_to_tel_pct_mean=round(float(drift_red), 2),
        drift_per_task_diff_mm=[round(d["drift"][0] - d["drift"][1], 2) for d in T2.values()],
        force_ratio=[round(x, 2) for x in force_ratio], force_ratio_mean=round(float(np.mean(force_ratio)), 2),
        trans_reduction=[round(x, 1) for x in trans_red], trans_reduction_mean=round(float(np.mean(trans_red)), 1),
        rot_reduction=[round(x, 1) for x in rot_red], rot_reduction_mean=round(float(np.mean(rot_red)), 1),
    )

    # ===== V3: 표 2 성공률 검정 =====
    kt = {"T1": (25, 25), "T2": (25, 25), "T3": (15, 20)}
    v3 = {}
    for t, (a, b) in kt.items():
        v3[t] = dict(tel=f"{a}/25", comb=f"{b}/25", wilson_tel=wilson(a, 25), wilson_comb=wilson(b, 25), fisher_p=round(fisher(a, 25, b, 25), 3))
    ta = sum(a for a, _ in kt.values())
    tb = sum(b for _, b in kt.values())
    v3["pooled"] = dict(tel=f"{ta}/75", comb=f"{tb}/75", wilson_tel=wilson(ta, 75), wilson_comb=wilson(tb, 75), fisher_p=round(fisher(ta, 75, tb, 75), 3),
                        diff_pp=round(100 * (tb - ta) / 75, 1))
    # T3 차이의 신뢰구간 (Newcombe)
    def newcombe(k1, n1, k2, n2):
        l1, u1 = [x / 100 for x in wilson(k1, n1)]
        l2, u2 = [x / 100 for x in wilson(k2, n2)]
        p1, p2 = k1 / n1, k2 / n2
        d = p2 - p1
        lo = d - math.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2)
        hi = d + math.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2)
        return [round(100 * lo, 1), round(100 * hi, 1)]
    v3["T3_diff_newcombe_pp"] = newcombe(15, 25, 20, 25)
    # 검정력: 60% 대 80%, n = 25 (정확 검정 시뮬이 아니라 Fisher 정확 검정 기각역으로 계산)
    def power_fisher(p1, p2, n, alpha=0.05):
        pw = 0.0
        for k1 in range(n + 1):
            pk1 = stats.binom.pmf(k1, n, p1)
            for k2 in range(n + 1):
                if fisher(k1, n, k2, n) < alpha:
                    pw += pk1 * stats.binom.pmf(k2, n, p2)
        return pw
    v3["power_60_vs_80_n25"] = round(power_fisher(0.6, 0.8, 25), 3)
    def n_for_power(p1, p2, target=0.8, alpha=0.05):
        # 정규 근사 (두 비율, 양측)
        za, zb = 1.959964, 0.841621
        pb = (p1 + p2) / 2
        n = ((za * math.sqrt(2 * pb * (1 - pb)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) / (p2 - p1)) ** 2
        return math.ceil(n)
    v3["n_per_arm_60_vs_80"] = n_for_power(0.6, 0.8)
    v3["n_per_arm_80_vs_92"] = n_for_power(0.8, 0.92)
    v3["n_per_arm_88_vs_96"] = n_for_power(0.88, 0.96)
    R["V3"] = v3

    # ===== V4: 표 3 서술 수치 (ViT-FMT) =====
    f = T4["ViT-FMT"]
    drift_red = [100 * (1 - d["drift"][1] / d["drift"][0]) for d in f.values()]
    force_chg = [pct_change(d["force"][0], d["force"][1]) for d in f.values()]
    trans_red = [100 * (1 - d["trans"][1] / d["trans"][0]) for d in f.values()]
    rot_red = [100 * (1 - d["rot"][1] / d["rot"][0]) for d in f.values()]
    R["V4"] = dict(
        drift_red_per_task=[round(x, 1) for x in drift_red], drift_red_mean=round(float(np.mean(drift_red)), 1),
        force_change_per_task=[round(x, 1) for x in force_chg], force_change_mean=round(float(np.mean(force_chg)), 1),
        trans_red_per_task=[round(x, 1) for x in trans_red], trans_red_mean=round(float(np.mean(trans_red)), 1),
        rot_red_per_task=[round(x, 1) for x in rot_red], rot_red_mean=round(float(np.mean(rot_red)), 1),
        tasks_where_smoothness_worse=[t for t, d in f.items() if d["trans"][1] > d["trans"][0] or d["rot"][1] > d["rot"][0]],
        table3_equals_table2_comb=all(
            f[t][m][0] == T2[t][m][1] for t in f for m in ("drift", "force", "trans", "rot")) and all(f[t]["succ"][0] == T2[t]["succ"][1] for t in f),
    )

    # ===== V5: 표 4, 5.4절, 8.6절 =====
    v5 = {}
    # 백본별 평균 (표 4의 w/ 열과 w/o 열, 가용 과제만, 0 포함 변형)
    per = {}
    for b, tasks in T4.items():
        avail = [d for d in tasks.values() if d["trans"] is not None]
        row = {}
        for j, cond in enumerate(("wo", "w")):
            row[cond] = dict(
                trans=round(float(np.mean([d["trans"][j] for d in avail])), 2),
                rot=round(float(np.mean([d["rot"][j] for d in avail])), 2),
                drift=round(float(np.mean([d["drift"][j] for d in avail])), 2),
                succ=round(float(np.mean([d["succ"][j] for d in tasks.values()])), 2),
                n_tasks_metrics=len(avail),
            )
        # 0을 포함한 3과제 평균 (정의되지 않은 값을 0으로 넣었을 때)
        row["w_trans_zero_filled"] = round(float(sum(d["trans"][1] if d["trans"] else 0 for d in tasks.values()) / 3), 2)
        row["w_rot_zero_filled"] = round(float(sum(d["rot"][1] if d["rot"] else 0 for d in tasks.values()) / 3), 2)
        row["text54"] = TEXT54[b]
        per[b] = row
    v5["per_backbone"] = per
    # 5.4절 본문 값이 어느 열과 일치하는가
    match = {}
    for b, row in per.items():
        t, r = TEXT54[b]
        match[b] = dict(trans_vs_w=round(t - row["w"]["trans"], 2), trans_vs_wo=round(t - row["wo"]["trans"], 2),
                        rot_vs_w=round(r - row["w"]["rot"], 2), rot_vs_wo=round(r - row["wo"]["rot"], 2))
    v5["text54_minus_table4"] = match
    # 성공률 평균 (표 4, 8.6절의 "98.7% on average")
    succ_avg = {b: dict(both=round(float(np.mean([x for d in t.values() for x in d["succ"]])), 2),
                        wo=round(float(np.mean([d["succ"][0] for d in t.values()])), 2),
                        w=round(float(np.mean([d["succ"][1] for d in t.values()])), 2)) for b, t in T4.items()}
    v5["success_avg"] = succ_avg
    # 같은 과제 집합(T1, T2)으로 맞춘 가속도 비교
    t12 = {}
    for b, tasks in T4.items():
        t12[b] = dict(trans_w=round(float(np.mean([tasks[t]["trans"][1] for t in ("T1", "T2")])), 2),
                      rot_w=round(float(np.mean([tasks[t]["rot"][1] for t in ("T1", "T2")])), 2))
    v5["T1T2_only_w"] = t12
    # 8.6절: 드리프트 17%, 병진 14%, 회전 13%
    cells = [(b, t, d) for b, tasks in T4.items() for t, d in tasks.items() if d["drift"] is not None]
    dr = [100 * (1 - d["drift"][1] / d["drift"][0]) for _, _, d in cells]
    tr = [100 * (1 - d["trans"][1] / d["trans"][0]) for _, _, d in cells]
    ro = [100 * (1 - d["rot"][1] / d["rot"][0]) for _, _, d in cells]
    v5["admittance_reduction_cell_mean"] = dict(n_cells=len(cells), drift=round(float(np.mean(dr)), 1), trans=round(float(np.mean(tr)), 1), rot=round(float(np.mean(ro)), 1))
    # 백본 평균 후 평균
    bb = {}
    for b in T4:
        cs = [d for d in T4[b].values() if d["drift"] is not None]
        bb[b] = (np.mean([100 * (1 - d["drift"][1] / d["drift"][0]) for d in cs]), np.mean([100 * (1 - d["trans"][1] / d["trans"][0]) for d in cs]),
                 np.mean([100 * (1 - d["rot"][1] / d["rot"][0]) for d in cs]))
    v5["admittance_reduction_backbone_mean"] = dict(drift=round(float(np.mean([x[0] for x in bb.values()])), 1), trans=round(float(np.mean([x[1] for x in bb.values()])), 1),
                                                   rot=round(float(np.mean([x[2] for x in bb.values()])), 1), per_backbone={b: [round(float(y), 1) for y in x] for b, x in bb.items()})
    # 평균값(mm, m/s2) 기준 비율
    v5["admittance_reduction_ratio_of_means"] = dict(
        drift=round(100 * (1 - np.mean([d["drift"][1] for _, _, d in cells]) / np.mean([d["drift"][0] for _, _, d in cells])), 1),
        trans=round(100 * (1 - np.mean([d["trans"][1] for _, _, d in cells]) / np.mean([d["trans"][0] for _, _, d in cells])), 1),
        rot=round(100 * (1 - np.mean([d["rot"][1] for _, _, d in cells]) / np.mean([d["rot"][0] for _, _, d in cells])), 1))
    # 부호 세기 (11 셀)
    v5["sign_counts"] = dict(
        drift_lower=sum(d["drift"][1] < d["drift"][0] for _, _, d in cells),
        trans_lower=sum(d["trans"][1] < d["trans"][0] for _, _, d in cells),
        rot_lower=sum(d["rot"][1] < d["rot"][0] for _, _, d in cells),
        force_higher=sum(d["force"][1] > d["force"][0] for _, _, d in cells),
        force_equal=sum(d["force"][1] == d["force"][0] for _, _, d in cells),
        n=len(cells))
    nn = len(cells)
    v5["sign_p_drift_all_lower"] = float(2 * 0.5 ** nn)
    # 성공률 쌍별 Fisher (w/o vs w/), 정수 격자 반올림
    def k_of(p):
        return int(round(p * N / 100))
    fish = []
    for b, tasks in T4.items():
        for t, d in tasks.items():
            if d["succ"][0] == d["succ"][1] == 0:
                continue
            a, c = k_of(d["succ"][0]), k_of(d["succ"][1])
            fish.append(dict(backbone=b, task=t, wo=f"{a}/25", w=f"{c}/25", diff_pp=round(100 * (c - a) / 25, 1), p=round(fisher(a, 25, c, 25), 3)))
    adj = holm([x["p"] for x in fish])
    for x, ad in zip(fish, adj):
        x["holm"] = round(ad, 3)
    v5["admittance_success_fisher"] = fish
    # 풀링 (과제×백본 층화, CMH)
    tabs = []
    for x in fish:
        a, c = int(x["wo"].split("/")[0]), int(x["w"].split("/")[0])
        tabs.append(np.array([[c, 25 - c], [a, 25 - a]]))
    st = StratifiedTable(tabs)
    v5["admittance_success_cmh"] = dict(OR=round(float(st.oddsratio_pooled), 2), p=round(float(st.test_null_odds().pvalue), 3),
                                        ci=[round(float(x), 2) for x in st.oddsratio_pooled_confint()], n_strata=len(tabs),
                                        w_total=sum(int(x["w"].split("/")[0]) for x in fish), wo_total=sum(int(x["wo"].split("/")[0]) for x in fish), n_total=25 * len(fish))
    # 최소 검출 가능 차이 (n = 25, 80% power, 대략)
    v5["up_down_equal_success"] = dict(up=sum(x["diff_pp"] > 0 for x in fish), down=sum(x["diff_pp"] < 0 for x in fish), equal=sum(x["diff_pp"] == 0 for x in fish))
    # 8.6절의 "ViT-FMT 98.7%"
    v5["fmt_98_7_candidates"] = dict(both_conditions=succ_avg["ViT-FMT"]["both"], w=succ_avg["ViT-FMT"]["w"], wo=succ_avg["ViT-FMT"]["wo"],
                                     note="보고 98.7은 어느 평균과도 일치하지 않음")
    # 순위: 백본 간 성공률 (w/ 열) Fisher, ViT-FMT 대 나머지 (pooled 75 trials)
    pool = {}
    for b, tasks in T4.items():
        kw = sum(k_of(d["succ"][1]) for d in tasks.values())
        pool[b] = (kw, 75)
    r_pool = {}
    for b in ("ViT-DiT", "pi0.5", "Psi0"):
        r_pool[b] = dict(fmt=f"{pool['ViT-FMT'][0]}/75", other=f"{pool[b][0]}/75", p=float(f"{fisher(pool['ViT-FMT'][0], 75, pool[b][0], 75):.2g}"))
    v5["backbone_vs_fmt_w_pooled"] = r_pool
    v5["pi05_vs_psi0_vs_dit_note"] = "각 과제가 25회라는 가정(표 4)"
    R["V5"] = v5

    # ===== V6: 표 5 =====
    # 보고 성공률이 k/n 반올림으로 나오는 작은 n
    def denoms(p, nmax=300):
        out = []
        for n in range(2, nmax + 1):
            for k in range(n + 1):
                if abs(100 * k / n - p) <= 0.005 + 1e-9:
                    out.append((k, n))
        return out
    v6 = {}
    for name, d in T5.items():
        ds = denoms(d["succ"])
        v6[name] = dict(pct=d["succ"], smallest_k_n=ds[:3], n_le_50=[x for x in ds if x[1] <= 50])
    v6["drift_reduction_corr_to_corradm_pct"] = round(100 * (1 - T5["Correction + Admittance"]["drift"] / T5["Correction"]["drift"]), 1)
    v6["drift_raw_tel_to_correction_pct"] = round(pct_change(T5["Raw Teleoperation"]["drift"], T5["Correction"]["drift"]), 1)
    v6["drift_raw_tel_to_corradm_pct"] = round(pct_change(T5["Raw Teleoperation"]["drift"], T5["Correction + Admittance"]["drift"]), 1)
    v6["succ_diff_pp"] = dict(tel_to_corr=round(89.29 - 85.35, 2), corr_to_adm=round(96.15 - 89.29, 2), tel_to_adm=round(96.15 - 85.35, 2))
    v6["trans_change_adm_vs_corr_pct"] = round(pct_change(T5["Correction"]["trans"], T5["Correction + Admittance"]["trans"]), 1)
    v6["rot_change_adm_vs_corr_pct"] = round(pct_change(T5["Correction"]["rot"], T5["Correction + Admittance"]["rot"]), 1)
    # 표 5와 표 4/2의 ViT-FMT T1 비교(다른 평가라는 점만 확인)
    v6["fisher_if_n28_26_for_89_96"] = round(fisher(25, 28, 25, 26), 3) if False else None
    # 25/28 (89.29) 대 25/26 (96.15): n이 28, 26일 때의 비교 (추정 분모, 가정)
    v6["assumed_89_29_is_25_of_28_and_96_15_is_25_of_26"] = dict(fisher_p=round(float(stats.fisher_exact([[25, 3], [25, 1]])[1]), 3))
    R["V6"] = v6

    # ===== V7: 표 1 =====
    v7 = dict(
        human_rmse_over_rms_pct=round(100 * 1.05 / 5.56, 1),
        lag_ratio=round(151 / 68, 2),
        rate_pred_over_meas=dict(human=round(3.74 / 5.86, 3), teleop=round(19.80 / 30.62, 3)),
        rate_meas_teleop_over_human=round(30.62 / 5.86, 2),
        rate_pred_teleop_over_human=round(19.80 / 3.74, 2),
        rmse_ratio_teleop_over_human=round(2.07 / 1.05, 2),
        lag_in_20ms_steps=dict(human=round(68 / 20, 2), teleop=round(151 / 20, 2)),
        horizon_steps=20, horizon_ms=20 * 20,
    )
    R["V7"] = v7

    # ===== V8: 힘 헤드 매개변수 하한 (D = 320, 2층, 4헤드) =====
    D = 320
    def dec_layer_params(D, dff):
        sa = 4 * D * D + 4 * D  # 자기 어텐션 (in_proj 3D^2 + out D^2, 편향 포함)
        ca = 4 * D * D + 4 * D  # 교차 어텐션
        ff = 2 * D * dff + dff + D
        ln = 3 * 2 * D
        return sa + ca + ff + ln
    v8 = {}
    for dff in (D, 2 * D, 4 * D, 2048):
        v8[f"dff_{dff}"] = 2 * dec_layer_params(D, dff)
    v8["attention_only_2_layers"] = 2 * (2 * (4 * D * D + 4 * D))
    v8["queries_Ha_x_D"] = 20 * D
    v8["reported"] = 500_000
    v8["lower_bound_over_reported"] = round(v8["dff_%d" % D] / 500_000, 2)
    # 자기 어텐션을 뺀 교차 어텐션 전용 (참고)
    v8["cross_only_2_layers_dff_D"] = 2 * (4 * D * D + 4 * D + 2 * D * D + D + D + 4 * D)
    R["V8"] = v8

    # ===== V9: 게이트 임계값과 평균 접촉 힘 =====
    v9 = dict(gate_N=0.5,
              mean_contact_force_T2=dict(tel=0.17, comb=0.52, comb_adm_off=0.52, adm_on=0.13),
              mean_contact_force_T3=dict(tel=0.66, comb=0.96),
              T2_values_below_or_near_gate=[0.17, 0.52, 0.13],
              cap_N=20, human_force_rms_T1=5.56,
              step_test_N=[1, 4], steady_state_error_pct=3.5, steady_state_error_N_at_4=round(0.035 * 4, 3), at_1=round(0.035 * 1, 3))
    R["V9"] = v9

    # ===== V10: 전송 과제 자료량, 시간 =====
    v10 = dict(beam_min=round(56 * 15 / 60, 1), table_min=round(45 * 15 / 60, 1),
               beam_weight_N=8.72, beam_mass_kg=round(8.72 / 9.81, 2), table_weight_N=39, table_mass_kg=round(39 / 9.81, 2),
               policy_rate_hz=50, chunk_ms=20 * 20, pi05_rate_check_ms=round(1000 / 12, 1),
               tokens_h=2 * 20)
    R["V10"] = v10

    # ===== V11: 시행 수 25에서 구분 가능한 차이 (Fisher, 검정력 80%) =====
    mde = {}
    for base in (0.5, 0.6, 0.8, 0.88):
        for delta in np.arange(0.04, 0.5, 0.04):
            p2 = min(0.999, base + delta)
            if power_fisher(base, p2, 25) >= 0.8:
                mde[str(base)] = round(float(delta), 2)
                break
    mde["power_80_to_100_n25"] = round(power_fisher(0.8, 0.999, 25), 3)
    mde["power_80_to_92_n25"] = round(power_fisher(0.8, 0.92, 25), 3)
    R["V11_min_detectable_diff_n25"] = mde

    OUT.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: R[k] for k in ("V1_bad", "V2", "V3", "V4")}, ensure_ascii=False, indent=1))
    print(json.dumps({k: R[k] for k in ("V5", "V6", "V7", "V8", "V9", "V10", "V11_min_detectable_diff_n25")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
