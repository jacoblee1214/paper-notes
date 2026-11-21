"""ACT(ALOHA) 논문 Table I·II, 본문 수치 재계산 (CPU, 재학습 없음).

입력: 논문 표 I·II의 값(스크립트 안에 직접 적음, 2026-11-21 원문 확인). 출력: analysis/recompute_results.json
1) 실세계 과제: 25회 평가 기준 단계별 성공 횟수 복원(정수 여부), 단조성, Wilson 95% CI
2) 실세계 최종 단계: ACT 대 최고 기준선 Fisher 정확 검정(양측), Holm 보정(6과제)
3) 시뮬레이션: 3 시드 × 50회 = 150회 풀링 근사, 본문의 '59/49/29/20%p' 우위 대조, Wilson CI
4) 본문 문장과 표의 일치 검사 (CVAE 35.3%, 시간 표현, 80–90% 문구, 데모 분량)
5) 사용자 연구(5Hz 대 50Hz) 평균 시간 증가율
실행: python3 analysis/recompute.py
"""
import json
import math
from pathlib import Path

from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "recompute_results.json"

# ---- 논문 Table I (시뮬레이션): 값은 "scripted | human" 성공률(%) ----
SIM = {
    "Cube Transfer (sim)": {
        "stages": ["Touched", "Lifted", "Transfer"],
        "BC-ConvMLP": [(34, 3), (17, 1), (1, 0)],
        "BeT": [(60, 16), (51, 13), (27, 1)],
        "RT-1": [(44, 4), (33, 2), (2, 0)],
        "VINN": [(13, 17), (9, 11), (3, 0)],
        "ACT": [(97, 82), (90, 60), (86, 50)],
    },
    "Bimanual Insertion (sim)": {
        "stages": ["Grasp", "Contact", "Insert"],
        "BC-ConvMLP": [(5, 0), (1, 0), (1, 0)],
        "BeT": [(21, 0), (4, 0), (3, 0)],
        "RT-1": [(6, 0), (0, 0), (1, 0)],
        "VINN": [(6, 0), (1, 0), (1, 0)],
        "ACT": [(93, 76), (90, 66), (32, 20)],
    },
}

# ---- 논문 Table I (실세계, 25회 평가, 1 시드) ----
REAL1 = {
    "Slide Ziploc": {"stages": ["Grasp", "Pinch", "Open"],
                     "BC-ConvMLP": [0, 0, 0], "BeT": [8, 0, 0], "RT-1": [4, 0, 0],
                     "VINN": [28, 0, 0], "ACT": [92, 96, 88]},
    "Slot Battery": {"stages": ["Grasp", "Place", "Insert"],
                     "BC-ConvMLP": [0, 0, 0], "BeT": [4, 0, 0], "RT-1": [4, 0, 0],
                     "VINN": [20, 0, 0], "ACT": [100, 100, 96]},
}
# ---- 논문 Table II (실세계 3과제, BeT만 비교) ----
REAL2 = {
    "Open Cup": {"stages": ["Tip Over", "Grasp", "Open Lid"], "BeT": [12, 0, 0], "ACT": [100, 96, 84]},
    "Thread Velcro": {"stages": ["Lift", "Grasp", "Insert"], "BeT": [24, 0, 0], "ACT": [92, 40, 20]},
    "Prep Tape": {"stages": ["Grasp", "Cut", "Handover", "Hang"], "BeT": [8, 0, 0, 0], "ACT": [96, 92, 72, 64]},
    "Put On Shoe": {"stages": ["Lift", "Insert", "Support", "Secure"], "BeT": [12, 0, 0, 0], "ACT": [100, 92, 92, 92]},
}
N_REAL = 25
N_SIM = 150  # 3 시드 × 50회


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(100 * (c - h), 1), round(100 * (c + h), 1)


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj = [0.0] * len(pvals)
    running = 0.0
    m = len(pvals)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def main():
    res = {}

    # 1) 실세계: 정수 복원, 단조성, CI
    checks = {}
    for name, d in list(REAL1.items()) + list(REAL2.items()):
        for method in [k for k in d if k != "stages"]:
            vals = d[method]
            counts = [round(v * N_REAL / 100) for v in vals]
            exact = all(abs(c * 100 / N_REAL - v) < 1e-9 for c, v in zip(counts, vals))
            mono = all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1))
            checks[f"{name} / {method}"] = {"counts": counts, "integer_match": exact, "monotone": mono}
    res["real_count_checks"] = checks
    res["real_all_integer"] = all(c["integer_match"] for c in checks.values())
    res["real_all_monotone"] = all(c["monotone"] for c in checks.values())
    res["real_non_monotone"] = [k for k, c in checks.items() if not c["monotone"]]

    # 2) 실세계 최종 단계: ACT 대 최고 기준선 (Fisher, 양측), Holm 보정
    final_rows = []
    for name, d in list(REAL1.items()) + list(REAL2.items()):
        act_final = round(d["ACT"][-1] * N_REAL / 100)
        base = {m: round(d[m][-1] * N_REAL / 100) for m in d if m not in ("stages", "ACT")}
        best_m = max(base, key=base.get)
        k_best = base[best_m]
        table = [[act_final, N_REAL - act_final], [k_best, N_REAL - k_best]]
        p = fisher_exact(table, alternative="two-sided")[1]
        final_rows.append({"task": name, "ACT_final": f"{act_final}/{N_REAL}",
                           "ACT_CI95": wilson(act_final, N_REAL),
                           "best_baseline": best_m, "baseline_final": f"{k_best}/{N_REAL}",
                           "p_fisher_two_sided": p})
    adj = holm([r["p_fisher_two_sided"] for r in final_rows])
    for r, a in zip(final_rows, adj):
        r["p_holm"] = a
        r["p_fisher_two_sided"] = float(f"{r['p_fisher_two_sided']:.3g}")
        r["p_holm"] = float(f"{a:.3g}")
    res["real_final_vs_best_baseline"] = final_rows

    # Thread Velcro 최종 단계 5/25 대 BeT 0/25 (단측 검정도 참고)
    tv = fisher_exact([[5, 20], [0, 25]], alternative="greater")[1]
    res["thread_velcro_insert_two_sided_p"] = float(f"{fisher_exact([[5, 20], [0, 25]])[1]:.3g}")
    res["thread_velcro_insert_one_sided_p"] = round(tv, 4)

    # 3) 시뮬레이션: 풀링 근사 (150회 가정), CI, 우위(%p)
    sim_out = {}
    for task, d in SIM.items():
        for si, mode in enumerate(["scripted", "human"]):
            for st_i, st in enumerate(d["stages"]):
                act_pct = d["ACT"][st_i][si]
                best = max((d[m][st_i][si], m) for m in d if m not in ("stages", "ACT"))
                k_act = round(act_pct * N_SIM / 100)
                k_best = round(best[0] * N_SIM / 100)
                p = fisher_exact([[k_act, N_SIM - k_act], [k_best, N_SIM - k_best]])[1]
                sim_out[f"{task} | {mode} | {st}"] = {
                    "ACT": act_pct, "ACT_CI95": wilson(k_act, N_SIM),
                    "best_baseline": best[1], "baseline": best[0],
                    "gap_pp": act_pct - best[0], "p_fisher_pooled": float(f"{p:.3g}"),
                }
    res["sim_final_gap"] = {k: v for k, v in sim_out.items() if k.endswith("Transfer") or k.endswith("Insert")}
    res["sim_all_cells"] = sim_out

    # 본문 주장: 59, 49, 29, 20 %p (최종 단계 기준)
    claimed = {"Cube Transfer scripted": 59, "Cube Transfer human": 49, "Insertion scripted": 29, "Insertion human": 20}
    got = {
        "Cube Transfer scripted": sim_out["Cube Transfer (sim) | scripted | Transfer"]["gap_pp"],
        "Cube Transfer human": sim_out["Cube Transfer (sim) | human | Transfer"]["gap_pp"],
        "Insertion scripted": sim_out["Bimanual Insertion (sim) | scripted | Insert"]["gap_pp"],
        "Insertion human": sim_out["Bimanual Insertion (sim) | human | Insert"]["gap_pp"],
    }
    res["claimed_gap_pp"] = {k: {"claimed": claimed[k], "table": got[k], "match": claimed[k] == got[k]} for k in claimed}

    # 4) 본문 문장 대 표
    cvae_human_mean = (50 + 20) / 2  # Table I 시뮬 human 최종 단계 평균(ACT, 두 과제)
    res["cvae_text_check"] = {"text_percent": 35.3, "table_mean_percent": cvae_human_mean,
                              "difference_pp": round(35.3 - cvae_human_mean, 2)}
    # 5Hz 대 50Hz 사용자 연구 (본문 평균 시간, 초)
    t = {"zip_tie": (33, 20), "cups": (16, 10)}
    inc = {k: round(100 * (a - b) / b, 1) for k, (a, b) in t.items()}
    inc_mean = round((inc["zip_tie"] + inc["cups"]) / 2, 1)
    inc_total = round(100 * (sum(a for a, _ in t.values()) - sum(b for _, b in t.values())) /
                      sum(b for _, b in t.values()), 1)
    res["user_study_5hz_vs_50hz"] = {"per_task_increase_pct": inc, "mean_of_tasks_pct": inc_mean,
                                     "pooled_increase_pct": inc_total, "text_claim_pct": 62,
                                     "note": "p<0.001 (repeated measures)는 개별 시도 자료가 없어 재계산 불가"}
    # 데모 분량: 50회 × 8–14 s, 100회(Thread Velcro)
    lo, hi = 50 * 8 / 60, 50 * 14 / 60
    res["demo_minutes"] = {"50_demos_range_min": [round(lo, 1), round(hi, 1)],
                           "100_demos_range_min": [round(2 * lo, 1), round(2 * hi, 1)],
                           "text_claim_min": [10, 20],
                           "note": "본문 '과제당 10–20분'은 50회에 맞지 않음(50회는 6.7–11.7분)"}
    # 시뮬레이션 80M 파라미터, 학습 5시간 등은 재계산 불가
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))

    print("integer match:", res["real_all_integer"], "monotone:", res["real_all_monotone"], res["real_non_monotone"])
    for r in final_rows:
        print(r["task"], r["ACT_final"], r["ACT_CI95"], "vs", r["best_baseline"], r["baseline_final"],
              "p=%.2e" % r["p_fisher_two_sided"], "holm=%.2e" % r["p_holm"])
    print("thread velcro one-sided p", res["thread_velcro_insert_one_sided_p"])
    print("claimed gaps", res["claimed_gap_pp"])
    print("cvae", res["cvae_text_check"])
    print("user study", res["user_study_5hz_vs_50hz"])
    print("demo", res["demo_minutes"])
    print("sim final:")
    for k, v in res["sim_final_gap"].items():
        print(" ", k, v)


if __name__ == "__main__":
    main()
