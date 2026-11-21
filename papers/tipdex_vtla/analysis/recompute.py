"""TipDex-VTLA (arXiv:2611.25683 v1) 논문 표 수치 재계산과 검정.

입력: paper/paper.txt (pdftotext -layout)에서 옮긴 표 1–3의 성공 횟수.
  - 실세계 표 2·3: 과제당 20회 시행 (논문 4.1절). 성공률은 20회 중 횟수.
  - 시뮬레이션 표 1: 과제당 100회 롤아웃 (논문 4.1절). 성공률은 100회 중 횟수.
  - 기준선 행 중 시뮬레이션 공개 결과(LingBot-VA, VITaL, FTP-1, ACT+UniVTAC)는
    원 논문 설정을 그대로 가져온 값이라 시행 수를 알 수 없습니다(논문 Table 1 캡션).
    이 행들은 평균 재계산만 하고 검정에서 뺍니다.

실행: python3 analysis/recompute.py  -> analysis/recompute_results.json
"""
import json
import math
from pathlib import Path

from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"

# ---- 표 1: UniVTAC 8 과제, 과제별 성공률(%) (논문 Table 1) ----
SIM_TASKS = ["Lift Bottle", "Pull-out Key", "Lift Can", "Put Bottle",
             "Insert Hole", "Insert HDMI", "Insert Tube", "Grasp Classify"]
SIM_PAPER_AVG = {  # 논문 표에 적힌 Avg.
    "ACT (Vision Only)": 30.9, "π0.5 (Without Tactile)": 41.4, "LingBot-VA": 31.4,
    "ACT + UniVTAC": 48.0, "VITaL": 40.5, "FTP-1": 64.8, "T-Rex": 81.3,
    "N0-VTLA": 83.1, "TipDex-VTLA": 84.9,
}
SIM_PCT = {
    "ACT (Vision Only)": [42, 28, 20, 28, 19, 15, 45, 50],
    "π0.5 (Without Tactile)": [100, 35, 6, 34, 25, 8, 74, 49],
    "LingBot-VA": [0, 58, 0, 0, 42, 38, 96, 17],
    "ACT + UniVTAC": [71, 46, 29, 31, 24, 28, 56, 99],
    "VITaL": [72, 47, 8, 32, 25, 6, 34, 100],
    "FTP-1": [97, 48, 65, 47, 64, 18, 79, 100],
    "T-Rex": [99, 99, 75, 79, 100, 8, 95, 95],
    "N0-VTLA": [99, 99, 88, 60, 95, 25, 99, 100],
    "TipDex-VTLA": [100, 97, 87, 80, 89, 30, 96, 100],
}
SIM_N = 100
SIM_PUBLISHED_ONLY = ["LingBot-VA", "VITaL", "FTP-1", "ACT + UniVTAC"]

# ---- 표 2: 실세계 4 과제, 과제별 성공률(%), 조건당 20회 (논문 Table 2) ----
REAL_TASKS = ["Egg", "Grape", "Tissue", "Beans"]
REAL_N = 20
REAL_NORMAL = {
    "ACT": [0, 30, 20, 25], "π0.5": [20, 80, 40, 80], "T-Rex": [65, 80, 60, 80],
    "N0-VTLA": [60, 85, 70, 60], "TipDex-VTLA": [90, 95, 75, 90],
}
REAL_UNSEEN = {
    "ACT": [0, 0, 0, 0], "π0.5": [20, 30, 25, 20], "T-Rex": [20, 30, 30, 10],
    "N0-VTLA": [30, 20, 35, 40], "TipDex-VTLA": [70, 90, 60, 75],
}
REAL_PAPER_AVG = {
    "normal": {"ACT": 18.8, "π0.5": 55.0, "T-Rex": 71.3, "N0-VTLA": 68.8, "TipDex-VTLA": 87.5},
    "unseen": {"ACT": 0.0, "π0.5": 23.8, "T-Rex": 22.5, "N0-VTLA": 31.3, "TipDex-VTLA": 73.8},
}

# ---- 표 3: 구성요소 절제, 실세계 (논문 Table 3) ----
ABL_NORMAL = {"Full": [90, 95, 75, 90], "w/o VASE": [85, 95, 70, 90],
              "w/o Tactile Verification": [80, 85, 60, 90], "w/o Gate Head": [80, 85, 65, 85]}
ABL_UNSEEN = {"Full": [70, 90, 60, 75], "w/o VASE": [45, 40, 55, 30],
              "w/o Tactile Verification": [65, 80, 45, 60], "w/o Gate Head": [30, 45, 40, 45]}
ABL_PAPER_AVG = {"normal": {"Full": 87.5, "w/o VASE": 85.0, "w/o Tactile Verification": 78.8,
                            "w/o Gate Head": 78.8},
                 "unseen": {"Full": 73.8, "w/o VASE": 42.5, "w/o Tactile Verification": 62.5,
                            "w/o Gate Head": 40.0}}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * (c - h), 1), round(100 * (c + h), 1))


def pct_to_count(pcts, n):
    counts = []
    for v in pcts:
        k = round(v * n / 100)
        assert abs(k * 100 / n - v) < 1e-9, (v, n)
        counts.append(k)
    return counts


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def fisher(k1, n1, k2, n2):
    table = [[k1, n1 - k1], [k2, n2 - k2]]
    return float(stats.fisher_exact(table, alternative="two-sided")[1])


def main():
    res = {"generated_by": "analysis/recompute.py", "source": "arXiv:2611.25683 v1, Tables 1-3"}

    # 표 1 평균 재계산
    sim = {}
    for name, pcts in SIM_PCT.items():
        mean = sum(pcts) / len(pcts)
        sim[name] = {"mean_recomputed": round(mean, 3), "paper_avg": SIM_PAPER_AVG[name],
                     "match": round(mean, 1) == SIM_PAPER_AVG[name]}
    res["table1_mean_check"] = sim

    # 표 1 TipDex 대 N0-VTLA, 대 T-Rex: 과제별 100회, 전체 800회 pooled
    tip_k = pct_to_count(SIM_PCT["TipDex-VTLA"], SIM_N)
    comp = {}
    for base in ["N0-VTLA", "T-Rex", "π0.5 (Without Tactile)"]:
        bk = pct_to_count(SIM_PCT[base], SIM_N)
        k1, n1 = sum(tip_k), SIM_N * len(tip_k)
        k2, n2 = sum(bk), SIM_N * len(bk)
        comp[base] = {
            "tipdex_success": f"{k1}/{n1}", "baseline_success": f"{k2}/{n2}",
            "diff_pp": round(100 * (k1 / n1 - k2 / n2), 2),
            "fisher_p_pooled_800": round(fisher(k1, n1, k2, n2), 4),
            "caveat": "과제 간 독립 가정, 같은 시드·초기 상태 여부 불명. 지표로만 사용",
        }
    res["table1_pairwise"] = comp
    res["table1_tipdex_task_wilson95"] = {
        t: {"k": k, "n": SIM_N, "wilson95_pct": wilson(k, SIM_N)}
        for t, k in zip(SIM_TASKS, tip_k)}

    # 표 2 실세계: 평균 재계산, 조건당 80회 pooled
    real = {"normal": {}, "unseen": {}}
    for cond, table, label in [("normal", REAL_NORMAL, "N"), ("unseen", REAL_UNSEEN, "U")]:
        counts = {m: pct_to_count(v, REAL_N) for m, v in table.items()}
        for m, ks in counts.items():
            n_total = REAL_N * len(ks)
            k_total = sum(ks)
            mean = 100 * k_total / n_total
            real[cond][m] = {
                "k_per_task": ks, "k_total": f"{k_total}/{n_total}",
                "mean_pct_recomputed": round(mean, 2),
                "paper_avg": REAL_PAPER_AVG[cond][m],
                "match": round(mean, 1) == REAL_PAPER_AVG[cond][m],
                "wilson95_pct": wilson(k_total, n_total),
            }
        base_keys = [m for m in counts if m != "TipDex-VTLA"]
        tk = counts["TipDex-VTLA"]
        pvals, names = [], []
        for m in base_keys:
            bk = counts[m]
            p = fisher(sum(tk), 80, sum(bk), 80)
            pvals.append(p)
            names.append(m)
        adj = holm(pvals)
        real[cond]["tests_vs_tipdex_fisher"] = {
            m: {"p_raw": round(p, 6), "p_holm": round(a, 6),
                "diff_pp": round(100 * (sum(tk) - sum(counts[m])) / 80, 2)}
            for m, p, a in zip(names, pvals, adj)}
    res["table2"] = real

    # 표 2 N0-VTLA 대 TipDex 미조정 p (본문 "42.5 pp") 확인
    res["table2_text_claim_42_5pp"] = {
        "computed_diff_pp_unseen_vs_N0": round(
            100 * (sum(pct_to_count(REAL_UNSEEN["TipDex-VTLA"], 20)) -
                   sum(pct_to_count(REAL_UNSEEN["N0-VTLA"], 20))) / 80, 2),
        "paper_text": 42.5,
    }

    # 표 3 절제: 평균 재계산 + 전체 대비 검정 (조건당 80회)
    abl = {"normal": {}, "unseen": {}}
    for cond, table in [("normal", ABL_NORMAL), ("unseen", ABL_UNSEEN)]:
        full_k = pct_to_count(table["Full"], 20)
        for name, pcts in table.items():
            ks = pct_to_count(pcts, 20)
            mean = sum(ks) / 80 * 100
            entry = {"k_per_task": ks, "k_total": f"{sum(ks)}/80",
                     "mean_pct_recomputed": round(mean, 2),
                     "paper_avg": ABL_PAPER_AVG[cond][name],
                     "match": round(mean, 1) == ABL_PAPER_AVG[cond][name],
                     "wilson95_pct": wilson(sum(ks), 80)}
            if name != "Full":
                entry["fisher_p_vs_full"] = round(fisher(sum(full_k), 80, sum(ks), 80), 6)
                entry["diff_vs_full_pp"] = round(100 * (sum(ks) - sum(full_k)) / 80, 2)
            abl[cond][name] = entry
    # 11.3 pp 주장: 완전 모델 대 w/o Tactile Verification (미조정)
    uf = pct_to_count(ABL_UNSEEN["Full"], 20)
    ut = pct_to_count(ABL_UNSEEN["w/o Tactile Verification"], 20)
    abl["claim_11_3pp"] = {
        "diff_pp": round(100 * (sum(uf) - sum(ut)) / 80, 2),
        "fisher_p_two_sided": round(fisher(sum(uf), 80, sum(ut), 80), 4),
        "note": "논문은 11.3 pp 이득을 주장하지만 검정 결과를 제시하지 않음",
    }
    res["table3"] = abl

    # 검정력: 조건당 80회에서 차이 11.25 pp(59 vs 50)를 80% 검정력으로 찾는 데 필요한 n (근사, 양측 5%)
    p1, p2 = 59 / 80, 50 / 80
    pbar = (p1 + p2) / 2
    za, zb = 1.959964, 0.841621
    n_needed = ((za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
                / (p1 - p2) ** 2)
    res["power_11_3pp"] = {"n_per_arm_for_80pct_power": math.ceil(n_needed),
                           "trials_per_arm_in_paper": 80}

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
