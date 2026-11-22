"""DiffWAM (arXiv:2611.19904 v1) 논문 표 수치를 다시 계산하고 통계 검정을 한다.

입력: paper/paper.txt(pdftotext -layout)의 표 12, 나머지 표 값은 아래에 옮겨 적었다(출처는 주석).
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)

가정(논문에 없는 것은 표시한다)
- "±" 값은 표본 표준편차(n-1)로 가정한다(Table 10 캡션과 일치).
- 시도 수: LIBERO 40과제 x 100 = 4000회, RoboTwin 전체 50과제 x 100 = 5000회(클린·랜덤 각각),
  클린-랜덤 실험 50과제 x 30 = 1500회(조건당), 실로봇 60회(과제당 20).
  FastWAM 클린-랜덤 실험의 시도 수는 논문에 없다. 1.50%는 45/3000 또는 75/5000과 맞고 1500회와는 맞지 않는다.
- 학습 실행 간 변동은 DiffWAM의 3회 실행(Table 10)에서만 추정할 수 있다. FastWAM은 실행 1회라 분산을 모른다.
  따라서 시도 단위 검정은 학습 실행 변동을 빠뜨린다. 이 점을 함께 적는다.
"""
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
PAPER_TXT = HERE.parent / "paper" / "paper.txt"
OUT = HERE / "recompute_results.json"
RES = {}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def newcombe(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lo, hi


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def prop_comp(name, k1, n1, k2, n2):
    d, lo, hi = newcombe(k1, n1, k2, n2)
    r = {"comparison": name, "k1": k1, "n1": n1, "k2": k2, "n2": n2,
         "diff_pp": round(100 * d, 2), "newcombe95_pp": [round(100 * lo, 2), round(100 * hi, 2)],
         "fisher_p": round(fisher(k1, n1, k2, n2), 4)}
    print(f"[{name}] {k1}/{n1} vs {k2}/{n2}: diff {100*d:.2f}pp, 95% [{100*lo:.2f}, {100*hi:.2f}], Fisher p={r['fisher_p']}")
    return r


def one_sample_t(mean, sd, n, mu0):
    t = (mean - mu0) / (sd / math.sqrt(n))
    p = 2 * stats.t.sf(abs(t), n - 1)
    return float(t), float(p)


# ---------- 1. 표 평균 검산 ----------
# Table 1 (LIBERO): 스위트별 성공률 (Spatial, Object, Goal, Long, 표의 Avg.)
T1 = {
    "OpenVLA": ([84.7, 88.4, 79.2, 53.7], 76.5),
    "OpenVLA-O": ([97.6, 98.4, 97.9, 94.5], 97.1),
    "MemoryVLA": ([98.4, 98.4, 96.4, 93.4], 96.65),
    "pi0.5": ([98.8, 98.2, 98.0, 92.4], 96.85),
    "GR00T-N1": ([94.4, 97.6, 93.0, 90.6], 93.9),
    "Motus": ([96.8, 99.8, 96.6, 97.6], 97.7),
    "LingBot-VA": ([98.5, 99.6, 97.2, 98.5], 98.45),
    "FastWAM": ([98.2, 99.8, 97.8, 92.4], 97.05),
    "ImageWAM": ([97.2, 99.2, 98.8, 98.4], 98.4),
    "WLA": ([99.0, 100.0, 97.8, 97.6], 98.6),
    "DiffWAM-S": ([98.0, 99.8, 99.0, 98.4], 98.8),
}
# Table 2 (RoboTwin 2.0 full data): (Clean, Random, 표의 Avg.)
T2 = {
    "pi0": (65.92, 58.80, 62.36), "X-VLA": (72.08, 72.84, 72.46), "pi0.5": (82.74, 76.76, 79.75),
    "ABot-M0": (86.60, 85.08, 85.84), "QwenVLA": (86.10, 87.20, 86.65), "Motus": (88.66, 87.02, 87.84),
    "LingBot-VA": (92.90, 91.50, 92.20), "FastWAM": (91.88, 90.32, 91.10), "X-WAM": (89.76, 90.68, 90.22),
    "WLA": (92.94, 90.02, 91.48), "DiffWAM-L": (93.28, 92.02, 92.65),
}
# Table 3 (clean-to-random): (Clean, Random, Avg.)
T3 = {
    "pi0": (46.4, 16.3, 31.35), "X-VLA": (68.0, 20.9, 44.45), "pi0.5": (70.7, 46.0, 58.35),
    "StarVLA": (46.5, 3.2, 24.85), "AHA-WAM": (64.3, 3.2, 33.75), "FastWAM": (70.2, 1.50, 35.85),
    "DiffWAM": (69.7, 5.27, 37.47),
}
# Table 5, 6, 7 (LIBERO 스위트별, Spatial/Object/Goal/Long, Avg.)
T5 = {"FastWAM": ([98.2, 99.8, 97.8, 92.4], 97.05), "+Diff loss": ([97.8, 100.0, 97.4, 95.8], 97.75),
      "+Diff Expert (DiffWAM-S)": ([98.0, 99.8, 99.0, 98.4], 98.8)}
T6 = {"M=5": ([99.6, 100.0, 97.2, 96.0], 98.2), "M=10": ([98.0, 99.8, 99.0, 98.4], 98.8),
      "M=15": ([98.2, 99.8, 97.4, 97.0], 98.1), "M=20": ([97.0, 100.0, 98.4, 98.0], 98.35)}
T7 = {"Only Diff Expert": ([99.2, 100.0, 95.8, 89.8], 96.2)}
# Table 8: 같은 체크포인트에서 1 epoch 추가 학습, 클린
T8 = {"FastWAM (continued)": 88.47, "DiffWAM (continued)": 93.13}
# Table 4
T4 = {"LIBERO": (97.05, 97.70, 98.80), "RoboTwin-Clean": (88.47, 87.72, 93.13), "RoboTwin-Full": (91.88, 93.28, 92.60)}
# Table 10: 독립 학습 3회 (Run1, Run2, Run3)
T10 = {"RoboTwin full": ([92.49, 92.65, 92.43], 92.52, 0.11),
       "LIBERO": ([98.60, 98.80, 98.60], 98.67, 0.12),
       "RoboTwin clean-only (Clean/Random 평균)": ([37.33, 37.40, 37.47], 37.40, 0.07)}


def check_means():
    out = {}
    bad = []
    for k, (vals, avg) in T1.items():
        calc = round(sum(vals) / 4, 2)
        if abs(calc - avg) > 0.006:
            bad.append(("T1", k, calc, avg))
    for k, (c, r, avg) in T2.items():
        calc = round((c + r) / 2, 2)
        if abs(calc - avg) > 0.006:
            bad.append(("T2", k, calc, avg))
    for k, (c, r, avg) in T3.items():
        calc = (c + r) / 2
        out[f"T3 {k}"] = {"calc": round(calc, 3), "reported": avg, "diff": round(calc - avg, 3)}
    for k, (vals, avg) in list(T5.items()) + list(T6.items()) + list(T7.items()):
        calc = round(sum(vals) / 4, 2)
        if abs(calc - avg) > 0.006:
            bad.append(("T5-7", k, calc, avg))
    for name, (runs, mean_rep, sd_rep) in T10.items():
        m = float(np.mean(runs))
        sd = float(np.std(runs, ddof=1))
        out[f"T10 {name}"] = {"mean": round(m, 3), "reported_mean": mean_rep, "sd": round(sd, 3),
                              "reported_sd": sd_rep, "max_run": max(runs)}
    print("평균 불일치 (허용 0.006):", bad if bad else "없음")
    return out, bad


# ---------- 2. Table 12 (과제별 RoboTwin 2.0) 파싱과 검산 ----------
def parse_table12():
    lines = PAPER_TXT.read_text(encoding="utf-8", errors="ignore").splitlines()
    rows = []
    row_re = re.compile(r"^\s*(.+?)\s+((?:\d+\s+){13}\d+)\s*$")
    start = next(i for i, l in enumerate(lines) if l.strip().startswith("Adjust Bottle"))
    for l in lines[start:start + 60]:
        m = row_re.match(l)
        if not m:
            continue
        name = m.group(1).strip()
        nums = [int(x) for x in m.group(2).split()]
        if len(nums) == 14:
            rows.append((name, nums))
        if name.startswith("Overall"):
            break
    methods = ["pi0.5", "Motus", "LingBot-VA", "GigaWorld-Policy", "X-WAM", "FastWAM", "DiffWAM-L"]
    tasks = rows
    overall = [82.74, 76.76, 88.66, 87.02, 92.90, 91.50, 86.36, 85.04, 89.76, 90.68, 91.88, 90.32, 93.28, 92.02]  # 표 12 Overall 행
    return methods, tasks, overall


def task_level():
    methods, tasks, overall = parse_table12()
    print(f"\n표 12 과제 행 {len(tasks)}개 파싱")
    arr = np.array([t[1] for t in tasks], dtype=float)  # (50, 14)
    if len(tasks) != 50:
        raise SystemExit(f"과제 행 수 {len(tasks)} != 50")
    res = {"n_tasks": len(tasks)}
    summary = {}
    for j, m in enumerate(methods):
        c = arr[:, 2 * j]
        r = arr[:, 2 * j + 1]
        summary[m] = {"clean_mean_over_tasks": round(c.mean(), 2), "random_mean_over_tasks": round(r.mean(), 2),
                      "avg_mean_over_tasks": round((c.mean() + r.mean()) / 2, 2)}
    res["task_means"] = summary
    print("과제 평균 (단순 평균) vs 표 Overall:")
    for j, m in enumerate(methods):
        print(f"  {m:16s} clean {summary[m]['clean_mean_over_tasks']:6.2f} (표 {overall[2*j]})  random {summary[m]['random_mean_over_tasks']:6.2f} (표 {overall[2*j+1]})")
    # DiffWAM-L 과제별 최고(동점 포함) 수
    dl = len(methods) - 1
    best_clean = sum(1 for t in tasks if t[1][2 * dl] == max(t[1][2 * j] for j in range(len(methods))))
    best_rand = sum(1 for t in tasks if t[1][2 * dl + 1] == max(t[1][2 * j + 1] for j in range(len(methods))))
    res["diffwam_L_best_or_tied"] = {"clean": best_clean, "random": best_rand, "of": len(tasks)}
    print(f"DiffWAM-L 과제별 최고 또는 동점: 클린 {best_clean}/50, 랜덤 {best_rand}/50")
    # FastWAM 대비 과제 쌍체 (100회 기준 반올림 값, 과제당 시도 수는 표 캡션에 없음)
    fw = methods.index("FastWAM"); dw = methods.index("DiffWAM-L")
    for lab, off in [("clean", 0), ("random", 1)]:
        d = arr[:, 2 * dw + off] - arr[:, 2 * fw + off]
        w = stats.wilcoxon(d[d != 0]) if np.any(d != 0) else None
        sign = stats.binomtest(int((d > 0).sum()), int((d != 0).sum()), 0.5).pvalue
        print(f"FastWAM 대비 DiffWAM-L {lab}: 이긴 과제 {(d>0).sum()}, 진 과제 {(d<0).sum()}, 같음 {(d==0).sum()}, 부호검정 p={sign:.3f}, 평균차 {d.mean():.2f}pp")
        res[f"pair_{lab}"] = {"wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "ties": int((d == 0).sum()),
                              "sign_test_p": round(float(sign), 4), "mean_diff_pp": round(float(d.mean()), 2),
                              "wilcoxon_p": round(float(w.pvalue), 4) if w else None}

    # 과제별 차이의 견고성: 상위 기여 과제, 중앙값, 절사평균
    dw_i = methods.index("DiffWAM-L"); fw_i = methods.index("FastWAM")
    dc = arr[:, 2 * dw_i] - arr[:, 2 * fw_i]
    dr = arr[:, 2 * dw_i + 1] - arr[:, 2 * fw_i + 1]
    names = [t[0] for t in tasks]
    top = np.argsort(dc)
    rob = {
        "clean_mean": round(float(dc.mean()), 2), "clean_median": round(float(np.median(dc)), 2),
        "clean_trimmed10": round(float(stats.trim_mean(dc, 0.1)), 2),
        "clean_mean_without_top1": round(float(np.delete(dc, np.argmax(dc)).mean()), 2),
        "clean_largest_gain": [names[i] + f" {dc[i]:+.0f}" for i in top[::-1][:3]],
        "clean_largest_loss": [names[i] + f" {dc[i]:+.0f}" for i in top[:3]],
        "random_mean": round(float(dr.mean()), 2), "random_median": round(float(np.median(dr)), 2),
        "random_trimmed10": round(float(stats.trim_mean(dr, 0.1)), 2),
        "random_largest_loss": [names[i] + f" {dr[i]:+.0f}" for i in np.argsort(dr)[:3]],
    }
    res["robustness"] = rob
    print("과제별 차이 견고성:", rob)
    return res


def main():
    RES["tables"], bad = check_means()
    RES["table12"] = task_level()

    print("\n--- 실로봇 (과제당 20회, 3과제 60회) ---")
    RES["real_world"] = [prop_comp("실로봇 DiffWAM-S 대 FastWAM", 53, 60, 44, 60)]
    print("과제별: Place Pens 18/20 대 16/20, Store Block 17/20 대 14/20, Stack Cups 18/20 대 14/20")
    RES["real_world_per_task_fisher"] = {
        "Place Pens": round(fisher(18, 20, 16, 20), 3),
        "Store Block": round(fisher(17, 20, 14, 20), 3),
        "Stack Cups": round(fisher(18, 20, 14, 20), 3)}
    print("과제별 Fisher p:", RES["real_world_per_task_fisher"])

    print("\n--- LIBERO (4000회 = 40과제 x 100) ---")
    # 시도 단위 검정: 학습 실행 변동을 빠뜨린다. 참고용.
    RES["libero_trial_level"] = [prop_comp("LIBERO DiffWAM-S 대 FastWAM (시도 단위)", 3952, 4000, 3882, 4000)]
    m, s, n = 98.67, 0.12, 3
    t, p = one_sample_t(m, s, n, 97.05)
    print(f"DiffWAM 3회 실행 평균 {m} (SD {s}) 대 FastWAM 단일 값 97.05: t={t:.1f}, df=2, p={p:.4f} (FastWAM 변동 무시)")
    RES["libero_run_level_one_sample"] = {"t": round(t, 2), "p": round(p, 4)}

    print("\n--- RoboTwin full (5000회 = 50과제 x 100, 클린 및 랜덤 각각) ---")
    # FastWAM의 n과 실행 수를 모른다. 평균 대비 시도 수준의 추정치만 준다.
    RES["robotwin_full_run_level_one_sample"] = {}
    m, s = 92.52, 0.11
    t, p = one_sample_t(m, s, 3, 91.10)
    print(f"DiffWAM-L 3회 평균 {m} (SD {s}) 대 FastWAM 91.10: t={t:.1f}, p={p:.4f} (FastWAM 변동 무시)")
    RES["robotwin_full_run_level_one_sample"] = {"t": round(t, 2), "p": round(p, 4)}

    # RoboTwin full: 시도 단위 검정 (5000회 조건당, 표 2의 소수 둘째 자리 값을 정수로 환산). 학습 변동 무시, 참고용.
    rt = {}
    for lab, a_, b_ in [("클린", (93.28, 91.88), None), ("랜덤", (92.02, 90.32), None)]:
        k1 = round(a_[0] * 50); k2 = round(a_[1] * 50)
        rt[lab] = prop_comp(f"RoboTwin full {lab} DiffWAM-L 대 FastWAM (시도 단위, 5000회)", k1, 5000, k2, 5000)
    RES["robotwin_full_trial_level"] = rt

    print("\n--- Clean-to-random (1500회 = 50과제 x 30, 조건당) ---")
    RES["clean_to_random"] = {
        "clean_k_over_n": "1045/1500 = {:.2f}%".format(100 * 1045 / 1500),
        "random_k_over_n": "79/1500 = {:.2f}%".format(100 * 79 / 1500),
        "avg_k_over_n": "1124/3000 = {:.2f}%".format(100 * 1124 / 3000),
    }
    print(RES["clean_to_random"])
    # FastWAM 시도 수 가정 민감도
    sens = []
    for n_fw, k_fw in [(3000, 45), (5000, 75)]:
        sens.append(prop_comp(f"랜덤 DiffWAM 79/1500 대 FastWAM {k_fw}/{n_fw} (가정)", 79, 1500, k_fw, n_fw))
    RES["clean_to_random_fastwam_sensitivity"] = sens
    print("  주: 1.50%는 1500회로는 정수가 되지 않는다(22.5회). FastWAM 시도 수 미기재.")
    print("  참고 Table 3 평균 37.47: 1124/3000 = 37.467 -> 반올림 37.47 일치")

    print("\n--- 절제 (Table 5, 7) 실행 변동 기준 ---")
    diff_loss = 97.75 - 97.05
    diff_exp = 98.80 - 97.75
    print(f"Diff loss 효과 +{diff_loss:.2f}, Diff Expert 추가 효과 +{diff_exp:.2f}. DiffWAM 3회 SD 0.12(LIBERO)와 비교하면 각각 약 {diff_loss/0.12:.1f}배, {diff_exp/0.12:.1f}배. 3회 SD는 표본이 작아 불확실하다.")
    RES["ablation_effects_pp"] = {"diff_loss": round(diff_loss, 2), "diff_expert_over_loss": round(diff_exp, 2)}

    print("\n--- 학습 효율 (그림 6 읽은 값) ---")
    for a, b in [(6.9, 9.6)]:
        print(f"95%까지 스텝 {a}k 대 {b}k: 감소 {100*(b-a)/b:.1f}%. 그림 읽기 오차 ±0.1k면 {100*(b-0.1-(a+0.1))/(b-0.1):.1f}%~{100*(b+0.1-(a-0.1))/(b+0.1):.1f}%")
    RES["efficiency"] = {"reduction_pct_point": round(100 * (9.6 - 6.9) / 9.6, 2),
                         "range_with_0p1k_reading_error": [round(100 * (9.5 - 7.0) / 9.5, 1), round(100 * (9.7 - 6.8) / 9.7, 1)]}

    print("\n--- 표 4 차이 검산 ---")
    RES["table4_diffs"] = {"S-L LIBERO": round(98.80 - 97.70, 2), "S-L Clean": round(93.13 - 87.72, 2), "L-S Full": round(93.28 - 92.60, 2)}
    print(RES["table4_diffs"])

    print("\n--- 표 10 SD 대 표 2·3 보고 평균 ---")
    print("DiffWAM-L 표 2 92.65 = 실행 2 (run 2). LIBERO 98.80 = 실행 2. 클린-랜덤 37.47 = 실행 3. 모두 세 실행 중 최댓값.")
    RES["headline_is_max_run"] = {"LIBERO": max(T10["LIBERO"][0]) == 98.80, "RoboTwin_full": max(T10["RoboTwin full"][0]) == 92.65,
                                  "clean_only": max(T10["RoboTwin clean-only (Clean/Random 평균)"][0]) == 37.47}
    print(RES["headline_is_max_run"])

    OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
