"""DexForce 그림 수치(Fig. 4, 7, 8, 9와 v1 Fig. 9) 재계산. 실행: python3 analysis/table_check.py

- C1: Fig. 7 성공 수 → 평균 76%, "near-zero"
- C2: Fig. 8 화살표의 상대 변화율 16개 재계산
- C3: Fisher 정확검정으로 * 표시 확인(v2), v1의 two-proportion z-test와 비교, Holm 보정
- C4: "force 관측은 해치지 않는다"(8개 비교의 부호 검정)와 "0/1은 RGB-only와 같거나 나쁘다" 점검
- C5: Fig. 9 k_f 범위를 힘 배율 G = k_p·k_f로 환산(k_p = 310), 1/k_p가 범위 안인지, 상한 0.025 절단
- C6: k_f 범위 너비 순위와 force 관측 이득 순위의 Spearman 상관(정확 순열 p)
- C7: 실험 규모(시연 수, rollout 수, k_f sweep에 든 재생 횟수 추정)
- C8: v1 Fig. 9(proprioception 절제, v2에서 삭제)
그림 수치는 300 dpi 확대 그림에서 막대 안의 숫자를 읽어 옮겼다. Fig. 9는 fig_digitize.py 결과를 쓴다.
"""
import itertools
import json
import math
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
N = 30  # 과제·조건마다 같은 초기 배치 30개 (Sec. IV)
TASKS = ["slide cube", "reorient smiley", "open AirPods", "grasp battery", "unscrew nut", "flip box"]
DEMOS = {"slide cube": 80, "reorient smiley": 60, "open AirPods": 150, "grasp battery": 110,
         "unscrew nut": 80, "flip box": 70}  # Fig. 4

# (성공, 부분 성공, 실패)
FIG7 = {  # 행동 절제: force-informed vs observed (모두 RGB, F/T 관측)
    "slide cube": {"force": (27, 0, 3), "observed": (1, 0, 29)},
    "reorient smiley": {"force": (25, 4, 1), "observed": (0, 2, 28)},
    "open AirPods": {"force": (17, 8, 5), "observed": (0, 0, 30)},
    "grasp battery": {"force": (22, 3, 5), "observed": (0, 0, 30)},
    "unscrew nut": {"force": (26, 4, 0), "observed": (0, 23, 7)},
    "flip box": {"force": (20, 3, 7), "observed": (0, 1, 29)},
}
FIG8 = {  # 관측 절제(모두 force-informed 행동 학습)와 v2 그림의 화살표 값
    "open AirPods": {"ft": (17, 8, 5), "rgb": (1, 2, 27), "bin": (1, 8, 21), "arrow": (1600, 0.0), "star": True},
    "unscrew nut": {"ft": (26, 4, 0), "rgb": (11, 9, 10), "bin": (16, 9, 5), "arrow": (136, 45.5), "star": True},
    "slide cube": {"ft": (27, 0, 3), "rgb": (17, 0, 13), "bin": (15, 0, 15), "arrow": (58.8, -11.8), "star": True},
    "grasp battery": {"ft": (22, 3, 5), "rgb": (17, 5, 8), "bin": (10, 3, 17), "arrow": (29.4, -41.2), "star": False},
    "reorient smiley": {"ft": (25, 4, 1), "rgb": (20, 8, 2), "bin": (14, 16, 0), "arrow": (25.0, -30.0), "star": False},
    "flip box": {"ft": (20, 3, 7), "rgb": (19, 5, 6), "bin": (18, 4, 8), "arrow": (5.26, -5.26), "star": False},
    "flip box - OOD": {"ft": (13, 12, 5), "rgb": (8, 13, 9), "bin": (8, 14, 8), "arrow": (62.5, 0.0), "star": False},
    "unscrew nut - OOD": {"ft": (15, 8, 7), "rgb": (11, 12, 7), "bin": (15, 7, 8), "arrow": (36.4, 36.4), "star": False},
}
V1_FIG9 = {  # v1 proprioception 절제 (v2에서 삭제)
    "open AirPods": {"RGB,F/T": (17, 8, 5), "RGB": (1, 2, 27), "proprio,RGB,F/T": (7, 8, 15), "proprio,RGB": (2, 10, 18)},
    "unscrew nut": {"RGB,F/T": (26, 4, 0), "RGB": (11, 9, 10), "proprio,RGB,F/T": (8, 10, 12), "proprio,RGB": (1, 0, 29)},
}
KP, KV, KF_USED, KF_CAP, KF_STEP = 310.0, 2.0, 0.0045, 0.025, 0.0001  # Sec. III, IV Q3 (v2)


def wilson(k: int, n: int = N, z: float = 1.959964) -> list:
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 3), round(c + h, 3)]


def fisher(k1: int, k2: int, n: int = N) -> float:
    return float(stats.fisher_exact([[k1, n - k1], [k2, n - k2]])[1])


def ztest(k1: int, k2: int, n: int = N) -> float:
    p1, p2, p = k1 / n, k2 / n, (k1 + k2) / (2 * n)
    se = math.sqrt(p * (1 - p) * 2 / n)
    return float(2 * stats.norm.sf(abs(p1 - p2) / se)) if se > 0 else 1.0


def holm(ps: dict) -> dict:
    items = sorted(ps.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = round(running, 5)
    return out


def spearman_exact(x: list, y: list) -> tuple:
    rho = stats.spearmanr(x, y).statistic
    n, ranks_y = len(x), stats.rankdata(y)
    rx = stats.rankdata(x)
    cnt = tot = 0
    for perm in itertools.permutations(ranks_y):
        tot += 1
        cnt += stats.spearmanr(rx, perm).statistic >= rho - 1e-12
    return float(rho), cnt / tot


def main() -> None:
    out: dict = {}

    # C1 Fig. 7
    fs = sum(FIG7[t]["force"][0] for t in TASKS)
    os_ = sum(FIG7[t]["observed"][0] for t in TASKS)
    out["C1_fig7"] = {
        "force_informed_success": f"{fs}/{6 * N}", "mean_success": round(fs / (6 * N), 4), "claim": "average 76%",
        "per_task_mean_of_rates": round(sum(FIG7[t]["force"][0] / N for t in TASKS) / 6, 4),
        "observed_success": f"{os_}/{6 * N}", "observed_partial": sum(FIG7[t]["observed"][1] for t in TASKS),
        "range": [min(FIG7[t]["force"][0] for t in TASKS) / N, max(FIG7[t]["force"][0] for t in TASKS) / N],
        "per_task": {t: {"force": FIG7[t]["force"][0], "ci": wilson(FIG7[t]["force"][0]),
                         "fisher_p_vs_observed": float(f"{fisher(FIG7[t]['force'][0], FIG7[t]['observed'][0]):.2e}")}
                     for t in TASKS},
        "fig7_force_equals_fig8_ft": all(FIG7[t]["force"] == FIG8[t]["ft"] for t in TASKS),
    }

    # C2 화살표, C3 검정
    c2, c3, ps_id = {}, {}, {}
    for t, v in FIG8.items():
        ft, rgb, b = v["ft"][0], v["rgb"][0], v["bin"][0]
        up = round(100 * (ft - rgb) / rgb, 2) if rgb else None
        down = round(100 * (b - rgb) / rgb, 2) if rgb else None
        c2[t] = {"ft_vs_rgb_pct": up, "paper": v["arrow"][0], "bin_vs_rgb_pct": down, "paper_bin": v["arrow"][1],
                 "match": abs(up - v["arrow"][0]) < 0.5 and abs(down - v["arrow"][1]) < 0.5}  # 그림은 반올림 값
        pf, pz = fisher(ft, rgb), ztest(ft, rgb)
        c3[t] = {"ft": ft, "rgb": rgb, "fisher_p": round(pf, 5), "z_p_v1": round(pz, 5),
                 "star_paper": v["star"], "star_fisher": pf < 0.05, "star_z": pz < 0.05,
                 "abs_diff": ft - rgb, "ci_ft": wilson(ft), "ci_rgb": wilson(rgb)}
        if "OOD" not in t:
            ps_id[t] = pf
    out["C2_arrows_all_match"] = all(r["match"] for r in c2.values())
    out["C2_arrows"] = c2
    out["C3_tests"] = c3
    out["C3_holm_id6"] = holm(ps_id)

    # C4 부호 검정과 0/1 비교
    diffs = [v["ft"][0] - v["rgb"][0] for v in FIG8.values()]
    npos = sum(d > 0 for d in diffs)
    bin_vs_rgb = {t: ("better" if v["bin"][0] > v["rgb"][0] else "equal" if v["bin"][0] == v["rgb"][0] else "worse")
                  for t, v in FIG8.items()}
    out["C4"] = {
        "ft_minus_rgb": diffs, "positive": npos, "of": len(diffs),
        "sign_test_one_sided_p": float(stats.binomtest(npos, len(diffs), 0.5, alternative="greater").pvalue),
        "ft_gt_bin_all": all(v["ft"][0] > v["bin"][0] for t, v in FIG8.items() if "OOD" not in t),
        "bin_vs_rgb": bin_vs_rgb,
        "claim": "RGB, 0/1 policies tend to do equivalently, or worse, than RGB-only (Sec. IV Q2)",
        "ood_success_ft": {t: f"{v['ft'][0]}/{N}" for t, v in FIG8.items() if "OOD" in t},
    }

    # C5 Fig. 9 → 힘 배율
    dig = json.loads((ROOT / "analysis" / "fig9_digitize_results.json").read_text())["bars"]
    c5 = {"kp": KP, "kf_used": KF_USED, "gain_at_kf_used": round(KP * KF_USED, 3),
          "kf_unit_gain": round(1 / KP, 5), "v1_Kf": round(1 / 220, 6), "v1_kf_times_kp": round(KP / 220, 3),
          "tasks": {}}
    for t, d in dig.items():
        lo, hi = d["min_mean"], d["max_mean"]
        c5["tasks"][t] = {
            "kf_range": [lo, hi], "gain_range": [round(KP * lo, 2), round(KP * hi, 2)],
            "tolerance_ratio_hi_over_lo": round(hi / lo, 1),
            "unit_gain_inside": lo < 1 / KP < hi,
            "kf_used_inside": lo < KF_USED < hi,
            "max_plus_sd_exceeds_cap": round(hi + d.get("max_sd", 0), 5) > KF_CAP,
            "min_plus_sd_above_unit_gain": round(lo + d.get("min_sd", 0), 5) > 1 / KP,
        }
    out["C5_fig9_gain"] = c5

    # C6 순위 상관
    order = ["unscrew nut", "open AirPods", "slide cube", "reorient smiley", "grasp battery", "flip box"]
    width = [dig[t]["max_mean"] - dig[t]["min_mean"] for t in order]
    rel = [(FIG8[t]["ft"][0] - FIG8[t]["rgb"][0]) / FIG8[t]["rgb"][0] for t in order]
    absd = [FIG8[t]["ft"][0] - FIG8[t]["rgb"][0] for t in order]
    rho_rel, p_rel = spearman_exact(width, [-r for r in rel])
    rho_abs, p_abs = spearman_exact(width, [-a for a in absd])
    out["C6_sensitivity_vs_benefit"] = {
        "order_by_width": order, "width": [round(w, 5) for w in width],
        "rel_gain": [round(r, 3) for r in rel], "abs_gain": absd,
        "spearman_width_vs_rel_gain": round(-rho_rel, 3), "exact_p_one_sided": round(p_rel, 4),
        "spearman_width_vs_abs_gain": round(-rho_abs, 3), "exact_p_one_sided_abs": round(p_abs, 4),
        "note": "너비가 좁을수록 이득이 큰지(음의 상관)를 부호를 뒤집어 양의 상관으로 검정. n = 6, 순열 720개",
    }

    # C7 규모
    sweeps = {t: round((dig[t]["max_mean"] - dig[t]["min_mean"]) / KF_STEP + 2) for t in dig}
    out["C7_scale"] = {
        "demos_total": sum(DEMOS.values()), "demos": DEMOS,
        "rollouts_fig7_fig8": 6 * N + 8 * 3 * N, "rollouts_v1_fig9_extra": 2 * 2 * N,
        "kf_sweep_replays_per_demo_est": sweeps, "kf_sweep_replays_total_est": 5 * sum(sweeps.values()),
        "note": "0.0045에서 0.0001씩 위아래로 실패할 때까지(상한 0.025), 과제당 시연 5개. 평균 범위 기준 추정",
    }

    # C8 v1 proprioception
    out["C8_v1_proprio"] = {t: {k: v[0] for k, v in d.items()} | {
        "fisher_p_ft_vs_proprio_ft": float(f"{fisher(d['RGB,F/T'][0], d['proprio,RGB,F/T'][0]):.2e}")}
        for t, d in V1_FIG9.items()}

    path = ROOT / "analysis" / "table_check_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for k, v in out.items():
        print(k, json.dumps(v, ensure_ascii=False)[:260])
    print("wrote", path)


if __name__ == "__main__":
    main()
