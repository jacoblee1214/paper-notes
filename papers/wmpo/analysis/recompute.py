"""WMPO (arXiv:2511.09515 v1) 표 수치를 다시 계산하고 통계 검정을 한다.

입력: 논문 Table 1, Table 2, 실세계 실험(본문 4.6)의 값을 아래에 그대로 옮겼다(출처 주석).
출력: analysis/recompute_results.json 과 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)

가정(논문에 없는 것은 표시한다)
- Table 1 과 Table 2 의 분모는 본문 4.1·4.4 의 "128 initial states" 로 가정한다(n = 128 per cell).
- 실세계 n = 30 (본문 4.6, "average success rate over 30 trials").
- 각 셀은 독립 이항 표본으로 취급한다. 시드 분산과 과제 간 분산은 반영하지 않는다(논문이 시드 수를 보고하지 않음).
- 정수 검사: 보고값 × n / 100 이 정수에 가까운지 본다. 어긋나면 반올림 불일치 또는 분모 가정 오류다.
"""
import json
import math
from pathlib import Path

from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
RES = {}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def int_check(reported_pct, n):
    """보고된 백분율(소수 첫째 자리)이 n 중 정수 k로 설명되는지 본다."""
    ks = [k for k in range(n + 1) if round(100 * k / n, 1) == round(reported_pct, 1)]
    return ks


# --- Table 1 (논문 p.7). 값은 % 단위. 열: Coffee, StackThree, ThreePieceAssembly, Square.
T1 = {
    ("base", None): [43.8, 46.9, 19.5, 24.2],
    ("GRPO", 128): [38.3, 52.3, 17.2, 25.0],
    ("DPO", 128): [43.8, 53.9, 23.4, 28.1],
    ("Ours", 128): [61.7, 56.3, 37.5, 32.8],
    ("GRPO", 1280): [47.7, 54.7, 20.3, 25.8],
    ("DPO", 1280): [52.3, 57.0, 26.7, 33.6],
    ("Ours", 1280): [75.0, 64.1, 46.1, 45.3],
}
T1_MEAN = {("base", None): 33.6, ("GRPO", 128): 33.2, ("DPO", 128): 37.3, ("Ours", 128): 47.1,
           ("GRPO", 1280): 37.1, ("DPO", 1280): 42.4, ("Ours", 1280): 57.6}
TASKS = ["Coffee", "StackThree", "ThreePieceAssembly", "Square"]

# --- Table 2 (논문 p.9). 열: Pos. Dis., Bg. Dis., Tex. Dis.
T2 = {
    "Base": [14.1, 46.1, 10.9],
    "GRPO": [15.6, 47.7, 10.9],
    "DPO": [16.4, 34.4, 7.8],
    "Ours": [22.3, 50.0, 16.4],
}
T2_MEAN = {"Base": 23.7, "GRPO": 24.7, "DPO": 19.5, "Ours": 29.6}
T2_COLS = ["Pos", "Bg", "Tex"]

N_SIM = 128
N_RW = 30


def two_prop(k1, n1, k2, n2):
    """Fisher 정확 검정(양측)과 두 비율 차이의 Wald 95% CI."""
    _, p = stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])
    p1, p2 = k1 / n1, k2 / n2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    d = p1 - p2
    return {"diff_pts": round(100 * d, 2), "wald95_pts": [round(100 * (d - 1.96 * se), 2), round(100 * (d + 1.96 * se), 2)],
            "fisher_p": float(p)}


def main():
    out = {"assumptions": {"n_sim_cell": N_SIM, "n_realworld": N_RW}}

    # 1) 정수 검사 (Table 1 셀, Table 2 셀)
    bad = []
    for key, vals in T1.items():
        for task, v in zip(TASKS, vals):
            ks = int_check(v, N_SIM)
            if not ks:
                bad.append({"table": "T1", "method": key[0], "P": key[1], "task": task, "reported": v})
    for m, vals in T2.items():
        for col, v in zip(T2_COLS, vals):
            ks = int_check(v, N_SIM)
            if not ks:
                bad.append({"table": "T2", "method": m, "col": col, "reported": v})
    out["integer_check_fail_n128"] = bad

    # 2) 평균 재계산
    mean_chk = {}
    for key, vals in T1.items():
        re = round(sum(vals) / 4, 2)
        mean_chk[str(key)] = {"reported": T1_MEAN[key], "recomputed": re,
                              "ok": abs(round(re, 1) - T1_MEAN[key]) < 0.06}
    for m, vals in T2.items():
        re = round(sum(vals) / 3, 2)
        mean_chk["T2 " + m] = {"reported": T2_MEAN[m], "recomputed": re,
                               "ok": abs(round(re, 1) - T2_MEAN[m]) < 0.06}
    out["mean_check"] = mean_chk

    # 3) 셀 수를 복원한 뒤 Fisher 검정 (Table 1, P=128 과 1280의 Ours 대 DPO, Ours 대 GRPO)
    def counts(vals):
        return [int_check(v, N_SIM)[0] if int_check(v, N_SIM) else round(v * N_SIM / 100) for v in vals]

    t1_counts = {str(k): counts(v) for k, v in T1.items()}
    out["table1_counts_n128"] = t1_counts
    pooled = {}
    for P in (128, 1280):
        ours = t1_counts[str(("Ours", P))]
        for base_name in ("DPO", "GRPO"):
            other = t1_counts[str((base_name, P))]
            k1, n1 = sum(ours), 4 * N_SIM
            k2, n2 = sum(other), 4 * N_SIM
            pooled[f"P{P} Ours vs {base_name} (512 pooled episodes)"] = two_prop(k1, n1, k2, n2)
            # 과제별 셀
            per = []
            for i, t in enumerate(TASKS):
                r = two_prop(ours[i], N_SIM, other[i], N_SIM)
                per.append({"task": t, "ours": ours[i], "other": other[i], **r})
            pooled[f"P{P} Ours vs {base_name} per task"] = per
    out["table1_tests"] = pooled

    # 4) Table 2 셀 (Pos. Dis. Ours 22.3 은 n=128에서 정수가 아니므로 가장 가까운 정수로 근사)
    t2 = {}
    for m, vals in T2.items():
        t2[m] = [round(v * N_SIM / 100) for v in vals]
    out["table2_counts_approx"] = t2
    t2_tests = {}
    for col_i, col in enumerate(T2_COLS):
        t2_tests[f"{col}: Ours vs DPO"] = two_prop(t2["Ours"][col_i], N_SIM, t2["DPO"][col_i], N_SIM)
        t2_tests[f"{col}: Ours vs Base"] = two_prop(t2["Ours"][col_i], N_SIM, t2["Base"][col_i], N_SIM)
        t2_tests[f"{col}: Ours vs GRPO"] = two_prop(t2["Ours"][col_i], N_SIM, t2["GRPO"][col_i], N_SIM)
    out["table2_tests"] = t2_tests

    # 5) 실세계 (본문 4.6): base 53% (16/30), DPO 60% (18/30), WMPO 70% (21/30). n = 30 확인
    rw = {"Base": 16, "DPO": 18, "WMPO": 21}
    rw_int = {k: round(v / N_RW * 100, 1) for k, v in rw.items()}
    out["realworld_counts"] = {"counts": rw, "pct": rw_int,
                               "reported": {"Base": 53, "DPO": 60, "WMPO": 70}}
    out["realworld_tests"] = {
        "WMPO vs Base": two_prop(21, N_RW, 16, N_RW),
        "WMPO vs DPO": two_prop(21, N_RW, 18, N_RW),
        "DPO vs Base": two_prop(18, N_RW, 16, N_RW),
    }
    out["realworld_wilson95"] = {k: [round(100 * x, 1) for x in wilson(v, N_RW)] for k, v in rw.items()}

    # 6) 서로 다른 n 이 달라지면 차이 검정이 어떻게 변하는지: 보고 차이 WMPO-DPO=10점(실세계)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("정수 검사 실패 셀 (n=128):")
    for b in bad:
        print("  ", b)
    print("평균 재계산 불일치:", [k for k, v in mean_chk.items() if not v["ok"]] or "없음")
    for k, v in pooled.items():
        if isinstance(v, dict):
            print(f"{k}: diff {v['diff_pts']} pts, CI {v['wald95_pts']}, Fisher p {v['fisher_p']:.4g}")
    for k, v in t2_tests.items():
        print(f"T2 {k}: diff {v['diff_pts']} pts, Fisher p {v['fisher_p']:.4g}")
    for k, v in out["realworld_tests"].items():
        print(f"RW {k}: diff {v['diff_pts']} pts, CI {v['wald95_pts']}, Fisher p {v['fisher_p']:.4g}")
    print("결과 저장:", OUT)


if __name__ == "__main__":
    main()
