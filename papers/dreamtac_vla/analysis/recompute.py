"""DreamTacVLA (arXiv:2512.23864 v5) 표·본문 수치 재계산과 검정.

입력은 논문 표 1, 표 4, 그림 4(판독 값), 본문 서술을 옮긴 상수이고 출력은 recompute_results.json 이다.
검사 번호는 V1–V12 (03_analysis.md 와 같은 번호).

가정(결과마다 문장에 밝힌다)
  - 표 1: "100 trials per run, mean ± SD over 3 runs" → 실행 3회 x 100회 = 과제당 300회로 합산(시행 독립 가정).
          보수적 검정으로 한 번의 100회만 쓴 경우도 함께 낸다.
  - 표 4: "100 trials per task" → 100회. 소수점 값은 100회 한 번으로는 나올 수 없다(V4).
  - 그림 4: 막대 위 숫자를 읽은 값(그림 판독 값). 시행 수는 논문에 없다.

실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "recompute_results.json"

TASKS = ["Peg-in-Hole", "USB Insert", "Gear Assembly", "Tool Stabilize"]
# 표 1: 평균, SD (%)
T1 = {
    "ACT": [(35.2, 0.7), (62.6, 0.5), (22.4, 0.8), (19.3, 0.6)],
    "Diffusion Policy": [(35.5, 0.9), (56.3, 0.8), (33.1, 0.7), (30.4, 0.9)],
    "pi0": [(48.7, 1.0), (59.4, 0.9), (45.2, 1.1), (41.0, 0.8)],
    "ACT (w/ Tactile)": [(45.6, 0.3), (63.1, 0.4), (40.2, 0.2), (39.1, 0.7)],
    "HSA-Only": [(60.8, 0.9), (63.7, 0.8), (51.5, 1.0), (42.9, 0.7)],
    "Dream-Only": [(75.4, 0.8), (75.2, 0.7), (64.9, 0.6), (68.5, 0.9)],
    "Full (HSA & Dream)": [(95.0, 0.2), (85.7, 0.6), (81.1, 0.4), (74.6, 0.5)],
}
# 표 4 (IsaacSim, 100회)
T4 = {
    "ACT": [72.4, 78.1, 65.3, 61.7],
    "Diffusion Policy": [75.6, 80.2, 69.4, 66.1],
    "pi0": [83.5, 85.9, 77.2, 74.6],
    "HSA-Only": [89.1, 90.4, 84.6, 82.7],
    "Dream-Only": [93.8, 94.6, 88.9, 90.1],
    "Full (HSA & Dream)": [98.6, 97.9, 95.2, 94.7],
}
# 본문이 말한 평균과 차이 (4.2절)
TEXT_CLAIMS_REAL = {
    "Full avg": 84.1, "pi0 avg": 48.6, "Full - pi0": 35.5, "Peg gain over pi0": 46.3,
    "ACT avg": 34.9, "ACT+Tactile avg": 47.0, "HSA-only avg": 54.7, "Dream-only avg": 71.0,
    "Full - HSA-only": 29.4, "Full - Dream-only": 13.1,
}
TEXT_CLAIMS_SIM = {"Full avg": 96.6, "HSA-only avg": 86.7, "Dream-only avg": 91.9}
# 그림 4 판독 값
FIG4_SIZE = {"300M": 68.5, "600M": 79.2, "1B": 85.1}
FIG4_DATA = {"20%": 45.2, "40%": 62.8, "60%": 78.5, "80%": 83.1, "100%": 86.4}


def r1(x):
    return round(x + 1e-9, 1)


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def fisher_p(k1, n1, k2, n2):
    return float(fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def mean_feasible(x, m):
    """m회 실행(각 정수 %)의 평균이 소수 첫째 자리로 x가 되는 정수 합 S가 있는가."""
    for s in range(0, 100 * m + 1):
        if r1(s / m) == x:
            return True
    return False


def main():
    res = {}

    # ---- V1: 표 1 평균과 본문 수치 -------------------------------------------------
    avg = {k: r1(np.mean([m for m, _ in v])) for k, v in T1.items()}
    full_raw = float(np.mean([m for m, _ in T1["Full (HSA & Dream)"]]))
    pi0_raw = float(np.mean([m for m, _ in T1["pi0"]]))
    calc = {
        "Full avg": avg["Full (HSA & Dream)"], "pi0 avg": avg["pi0"], "Full - pi0": r1(full_raw - pi0_raw),
        "Peg gain over pi0": r1(T1["Full (HSA & Dream)"][0][0] - T1["pi0"][0][0]),
        "ACT avg": avg["ACT"], "ACT+Tactile avg": avg["ACT (w/ Tactile)"], "HSA-only avg": avg["HSA-Only"],
        "Dream-only avg": avg["Dream-Only"],
        "Full - HSA-only": r1(full_raw - np.mean([m for m, _ in T1["HSA-Only"]])),
        "Full - Dream-only": r1(full_raw - np.mean([m for m, _ in T1["Dream-Only"]])),
    }
    res["V1_real_text_vs_table"] = {k: {"text": TEXT_CLAIMS_REAL[k], "computed": calc[k],
                                        "match": abs(TEXT_CLAIMS_REAL[k] - calc[k]) < 0.051} for k in TEXT_CLAIMS_REAL}
    res["V1_real_row_means"] = avg
    avg4 = {k: r1(np.mean(v)) for k, v in T4.items()}
    calc4 = {"Full avg": avg4["Full (HSA & Dream)"], "HSA-only avg": avg4["HSA-Only"], "Dream-only avg": avg4["Dream-Only"]}
    res["V1_sim_text_vs_table"] = {k: {"text": TEXT_CLAIMS_SIM[k], "computed": calc4[k],
                                       "match": abs(TEXT_CLAIMS_SIM[k] - calc4[k]) < 0.051} for k in TEXT_CLAIMS_SIM}
    res["V1_sim_row_means"] = avg4
    # 순서 일관성: 모든 과제에서 ACT < pi0 < ... ordering 검사
    order_real = ["ACT", "Diffusion Policy", "pi0", "HSA-Only", "Dream-Only", "Full (HSA & Dream)"]
    viol = []
    for t in range(4):
        for a, b in zip(order_real[2:], order_real[3:]):
            if T1[a][t][0] >= T1[b][t][0]:
                viol.append((TASKS[t], a, b))
    res["V1_ordering_pi0_to_full_violations_real"] = viol
    res["V1_acttactile_vs_act_gain"] = {TASKS[t]: r1(T1["ACT (w/ Tactile)"][t][0] - T1["ACT"][t][0]) for t in range(4)}

    # ---- V2: 표 1 평균의 정수 가능성 (3회 x 100회) --------------------------------
    feas3 = {}
    bad3 = []
    for name, rows in T1.items():
        for t, (m, _) in enumerate(rows):
            ok = mean_feasible(m, 3)
            feas3[f"{name}|{TASKS[t]}"] = ok
            if not ok:
                bad3.append(f"{name}|{TASKS[t]}={m}")
    res["V2_mean_feasible_3runs_x100"] = {"n_cells": len(feas3), "n_infeasible": len(bad3), "infeasible": bad3}
    res["V2_cells_feasible_if_runs"] = {str(m): all(mean_feasible(x, m) for rows in T1.values() for x, _ in rows)
                                         for m in (1, 3, 5, 10)}

    # ---- V3: 표 1 SD의 정수 가능성 (3회 x 100회, 실행별 정수 %) -------------------
    grid = np.arange(0, 101)
    a, b, c = np.meshgrid(grid, grid, grid, indexing="ij")
    a = a.ravel(); b = b.ravel(); c = c.ravel()
    mean = (a + b + c) / 3.0
    sd1 = np.sqrt(((a - mean) ** 2 + (b - mean) ** 2 + (c - mean) ** 2) / 2.0)
    sd0 = np.sqrt(((a - mean) ** 2 + (b - mean) ** 2 + (c - mean) ** 2) / 3.0)
    mean_r = np.round(mean + 1e-9, 1)
    sd1_r = np.round(sd1 + 1e-9, 1)
    sd0_r = np.round(sd0 + 1e-9, 1)
    nz = sd1 > 0
    res["V3_min_nonzero_sd_of_3_integer_runs"] = {"sample_sd_ddof1": float(sd1[nz].min()), "population_sd_ddof0": float(sd0[nz].min())}
    sd_feas = {}
    n_mean_only = 0
    n_both = 0
    n_sd_below_min = 0
    for name, rows in T1.items():
        for t, (m, s) in enumerate(rows):
            msel = mean_r == m
            both1 = bool(np.any(msel & (sd1_r == s)))
            both0 = bool(np.any(msel & (sd0_r == s)))
            sd_feas[f"{name}|{TASKS[t]}"] = {"mean": m, "sd": s, "mean_feasible": bool(msel.any()),
                                              "joint_feasible_ddof1": both1, "joint_feasible_ddof0": both0}
            n_both += int(both1 or both0)
            n_mean_only += int(msel.any())
            n_sd_below_min += int(0 < s < 0.4714)
    res["V3_sd_joint_feasibility_3runs_x100"] = {
        "n_cells": 28, "n_mean_feasible": n_mean_only, "n_joint_feasible_either_sd_def": n_both,
        "n_cells_sd_below_0.47": n_sd_below_min, "cells": sd_feas}

    # ---- V4: 표 4 소수점 값의 정수 가능성 ----------------------------------------
    cells4 = [(k, TASKS[t], x) for k, v in T4.items() for t, x in enumerate(v)]
    is_int = [abs(x - round(x)) < 1e-9 for _, _, x in cells4]
    res["V4_table4_100_trials"] = {
        "n_cells": len(cells4), "n_integer_valued": int(sum(is_int)),
        "feasible_if_runs": {str(m): all(mean_feasible(x, m) for _, _, x in cells4) for m in (1, 3, 5, 10)},
        "n_feasible_3runs": int(sum(mean_feasible(x, 3) for _, _, x in cells4)),
    }

    # ---- V5: Wilson 구간과 이항 SD 대비 보고 SD ----------------------------------
    cis = {}
    ratio = []
    for name, rows in T1.items():
        for t, (m, s) in enumerate(rows):
            p = m / 100.0
            binom_sd100 = 100 * math.sqrt(p * (1 - p) / 100)
            binom_se300 = 100 * math.sqrt(p * (1 - p) / 300)
            k300 = round(p * 300)
            lo, hi = wilson(k300, 300)
            lo1, hi1 = wilson(round(p * 100), 100)
            cis[f"{name}|{TASKS[t]}"] = {"mean": m, "reported_sd": s, "binomial_sd_per_100": round(binom_sd100, 2), "binomial_se_n300": round(binom_se300, 2),
                                         "ratio_reported_to_binomial": round(s / binom_sd100, 2),
                                         "wilson95_n300": [round(lo, 1), round(hi, 1)],
                                         "wilson95_n100": [round(lo1, 1), round(hi1, 1)]}
            ratio.append(s / binom_sd100)
    res["V5_wilson_and_sd"] = cis
    res["V5_sd_ratio_summary"] = {"min": round(min(ratio), 3), "median": round(float(np.median(ratio)), 3), "max": round(max(ratio), 3)}

    # ---- V6: 실제 표 1 쌍 비교 (Fisher, Holm) -----------------------------------
    pairs = [("Full (HSA & Dream)", "pi0"), ("Full (HSA & Dream)", "Dream-Only"), ("Dream-Only", "HSA-Only"),
             ("HSA-Only", "pi0"), ("HSA-Only", "ACT (w/ Tactile)"), ("ACT (w/ Tactile)", "ACT"),
             ("Full (HSA & Dream)", "HSA-Only")]
    out6 = {}
    for n_runs, key in ((3, "n300_assumed_3x100"), (1, "n100_single_run")):
        n = 100 * n_runs
        rows = []
        for a_, b_ in pairs:
            for t in range(4):
                ka = round(T1[a_][t][0] / 100 * n)
                kb = round(T1[b_][t][0] / 100 * n)
                rows.append({"a": a_, "b": b_, "task": TASKS[t], "a_pct": T1[a_][t][0], "b_pct": T1[b_][t][0],
                             "ka": ka, "kb": kb, "n": n, "p": fisher_p(ka, n, kb, n)})
        adj = holm([r["p"] for r in rows])
        for r, q in zip(rows, adj):
            r["p_holm"] = q
        out6[key] = rows
    res["V6_pairwise_real"] = out6
    nonsig3 = [(r["a"], r["b"], r["task"], f'{r["p_holm"]:.3g}') for r in out6["n300_assumed_3x100"] if r["p_holm"] >= 0.05]
    nonsig1 = [(r["a"], r["b"], r["task"], f'{r["p_holm"]:.3g}') for r in out6["n100_single_run"] if r["p_holm"] >= 0.05]
    res["V6_nonsignificant_holm_n300"] = nonsig3
    res["V6_nonsignificant_holm_n100"] = nonsig1

    # 4과제 합산 (과제 평균 성공 횟수를 더함)
    pooled = {}
    for a_, b_ in pairs:
        n = 1200
        ka = sum(round(T1[a_][t][0] / 100 * 300) for t in range(4))
        kb = sum(round(T1[b_][t][0] / 100 * 300) for t in range(4))
        pooled[f"{a_} vs {b_}"] = {"ka": ka, "kb": kb, "n": n, "p": fisher_p(ka, n, kb, n)}
    res["V6_pooled_four_tasks_n1200"] = pooled

    # ---- V7: 시뮬 표 4 (n=100 가정) 쌍 비교 ---------------------------------------
    rows7 = []
    for a_, b_ in [("Full (HSA & Dream)", "Dream-Only"), ("Dream-Only", "HSA-Only"), ("Full (HSA & Dream)", "pi0"),
                   ("HSA-Only", "pi0")]:
        for t in range(4):
            ka, kb = round(T4[a_][t]), round(T4[b_][t])
            rows7.append({"a": a_, "b": b_, "task": TASKS[t], "ka": ka, "kb": kb, "p": fisher_p(ka, 100, kb, 100)})
    adj7 = holm([r["p"] for r in rows7])
    for r, q in zip(rows7, adj7):
        r["p_holm"] = q
    res["V7_pairwise_sim_n100_rounded_counts"] = rows7

    # ---- V8: 실제와 시뮬의 순서 일관성, 격차 ---------------------------------------
    names = ["pi0", "HSA-Only", "Dream-Only", "Full (HSA & Dream)"]
    gap = {n_: [r1(T4[n_][t] - T1[n_][t][0]) for t in range(4)] for n_ in names}
    res["V8_sim_minus_real"] = gap
    res["V8_full_minus_pi0_real_vs_sim"] = {
        "real": [r1(T1["Full (HSA & Dream)"][t][0] - T1["pi0"][t][0]) for t in range(4)],
        "sim": [r1(T4["Full (HSA & Dream)"][t] - T4["pi0"][t]) for t in range(4)]}
    # 표 4에서 ACT 대비 baseline 순서 (ACT < DP < pi0 < HSA < Dream < Full)가 모든 과제에서 유지되는지
    seq = ["ACT", "Diffusion Policy", "pi0", "HSA-Only", "Dream-Only", "Full (HSA & Dream)"]
    res["V8_sim_order_violations"] = [(TASKS[t], x, y) for t in range(4) for x, y in zip(seq, seq[1:]) if T4[x][t] >= T4[y][t]]

    # ---- V9: 그림 4 판독 값 ------------------------------------------------------
    sz = list(FIG4_SIZE.values())
    dt = list(FIG4_DATA.values())
    res["V9_fig4"] = {
        "size_gains": [r1(sz[1] - sz[0]), r1(sz[2] - sz[1])],
        "data_gains": [r1(dt[i + 1] - dt[i]) for i in range(4)],
        "text_claim_gain_300M_to_600M": 10.7,
        "text_claim_matches": r1(sz[1] - sz[0]) == 10.7,
        "table1_full_avg": avg["Full (HSA & Dream)"], "table1_dream_only_avg": avg["Dream-Only"],
        "fig4_300M": FIG4_SIZE["300M"], "fig4_100pct": FIG4_DATA["100%"], "fig4_1B": FIG4_SIZE["1B"],
        "note": "그림 4 점들이 어느 과제·프로토콜인지 본문에 없음. 300M(본 모델 ViT-L)은 68.5, 표 1 전체 모델 평균은 84.1",
    }

    # ---- V10: 어댑터 파라미터 수와 본문 수치 -----------------------------------------
    def adapter_params(d, h=512, depth=3, heads=8):
        ln = 2 * d
        mlp = d * h + h + (depth - 2) * (h * h + h) + h * d + d
        scale = 1
        attn = 4 * d * d + 4 * d
        query = d
        ln_pool = 2 * d
        return {"adapter_total": ln + mlp + scale, "mlp_only": mlp, "pool_total": attn + query + ln_pool,
                "sum": ln + mlp + scale + attn + query + ln_pool}
    vitl = adapter_params(1024)
    vitg = adapter_params(1408)
    res["V10_adapter_params"] = {
        "vit_l_d1024": vitl, "vit_g_d1408": vitg,
        "overhead_pct_of_300M_vitl": round(100 * vitl["sum"] / 300e6, 2),
        "paper_text": {"5.5M, 1.8% of 300M": [5.5e6, 1.8], "~1.5M per tactile encoder (B.1)": 1.5e6},
        "note": "5.5M는 어댑터 + 어텐션 풀링 합, 1.5M은 어댑터 MLP만(약 1.3M)과 가까움(추정)"}

    # ---- V11: HSA 패치 격자와 카메라 행렬 ------------------------------------------
    K = np.array([[647.0, 0.0, 653.0], [0.0, 644.0, 364.0], [0.0, 0.0, 1.0]])
    E_rot = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    t_vec = np.array([0.7, -0.49, 1.14])
    pts = {"(0.3,0,0.1)": np.array([0.3, 0.0, 0.1]), "(0.0,-0.4,0.05)": np.array([0.0, -0.4, 0.05])}
    proj = {}
    for nme, X in pts.items():
        Xc = E_rot @ X + t_vec
        uvw = K @ Xc
        proj[nme] = {"cam_xyz": Xc.round(3).tolist(), "uv": (uvw[:2] / uvw[2]).round(1).tolist()}
    res["V11_camera"] = {
        "det_rotation_block": float(np.linalg.det(E_rot)),
        "cx_cy": [653.0, 364.0], "table5_rgb_resolution": [640, 480],
        "cx_exceeds_width": 653.0 > 640, "example_projection_if_world_to_camera": proj,
        "patch_grid": {"224/16": 224 // 16, "tokens_16": (224 // 16) ** 2, "224/14": 224 / 14,
                       "appendix_C3_formula_divides_by": 14, "appendix_C3_text_patch_size": 16},
        "note": "C.3은 patch14, patch size 16, 격자 16x16을 함께 서술. 224/14=16, 224/16=14",
    }

    # ---- V12: 데이터 구성 --------------------------------------------------------
    sim_demos = 1000 * 4
    real_demos = 100 * 4
    res["V12_dataset_composition"] = {
        "sim_demos_1000_per_task_x4": sim_demos, "real_demos_100_per_task_x4": real_demos,
        "sim_share_by_demo_count": round(100 * sim_demos / (sim_demos + real_demos), 1),
        "paper_sim_share_pct": 80, "paper_total_tactile_frames": 2_000_000,
        "hf_dataset_episodes": {"usb": 50, "gearassemb": 50, "peginhole": 52},
        "note": "시연 수 기준으로는 sim 약 91%. 80/20은 프레임 수 기준일 수 있음(논문에 기준 없음, 추정)"}
    # 프레임 수: 에피소드 길이(저장소 constants) x 에피소드 수, 50 Hz
    eplen = {"peginhole": 350, "gearassemb": 700, "usb": 500}
    frames = {k: eplen[k] * n for k, n in (("usb", 50), ("gearassemb", 50), ("peginhole", 52))}
    res["V12_hf_frames_upper_bound"] = {"episode_len_in_repo_constants": eplen, "frames_at_max_len": frames,
                                         "total": sum(frames.values()),
                                         "seconds_per_episode_at_50Hz": {k: v / 50 for k, v in eplen.items()}}

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")

    # ---- 요약 출력 ---------------------------------------------------------------
    print("V1 본문 대 표 (실로봇):", {k: v["match"] for k, v in res["V1_real_text_vs_table"].items()})
    print("V1 본문 대 표 (시뮬):", {k: v["match"] for k, v in res["V1_sim_text_vs_table"].items()})
    print("V2 표 1 평균 정수 가능성(3x100): 불가능 셀", res["V2_mean_feasible_3runs_x100"]["n_infeasible"], "/ 28")
    print("V2 모든 셀이 가능한 실행 수:", res["V2_cells_feasible_if_runs"])
    print("V3 SD 최소(3회 정수 %):", res["V3_min_nonzero_sd_of_3_integer_runs"])
    s3 = res["V3_sd_joint_feasibility_3runs_x100"]
    print("V3 평균+SD 동시 가능:", s3["n_joint_feasible_either_sd_def"], "/ 28; SD<0.47 셀:", s3["n_cells_sd_below_0.47"])
    print("V4 표 4:", res["V4_table4_100_trials"])
    print("V5 SD/이항SD:", res["V5_sd_ratio_summary"])
    print("V6 Holm 후 비유의 (n300):", nonsig3)
    print("V6 Holm 후 비유의 (n100):", nonsig1)
    print("V8 순서 위반:", res["V8_sim_order_violations"], res["V1_ordering_pi0_to_full_violations_real"])
    print("V9:", res["V9_fig4"]["size_gains"], res["V9_fig4"]["data_gains"])
    print("V10:", res["V10_adapter_params"]["vit_l_d1024"], res["V10_adapter_params"]["overhead_pct_of_300M_vitl"])
    print("V11:", res["V11_camera"]["det_rotation_block"], res["V11_camera"]["example_projection_if_world_to_camera"])
    print("V12:", res["V12_dataset_composition"]["sim_share_by_demo_count"], res["V12_hf_frames_upper_bound"]["total"])


if __name__ == "__main__":
    main()
