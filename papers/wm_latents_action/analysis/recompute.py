"""논문 표 값을 공식 저장소의 원자료(results/*.csv)와 논문 표 자체에서 다시 계산한다.

실행: python3 analysis/recompute.py   (numpy, scipy; 수 초)
입력: repo/results/ (공식 저장소를 얕게 복제한 폴더, 실행하지 않고 CSV만 읽는다)
      표 값은 논문 v2 (arXiv:2606.07687v2)에서 옮겨 적은 것이다.
출력: analysis/recompute_results.json (키는 영문, R²는 r2)

검사 라벨: K = 일관성 검사, C = 주장 판정 입력, 숫자 뒤에 붙인다. (R, A, P는 쓰지 않는다)
"""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "repo" / "results"
OUT = ROOT / "analysis" / "recompute_results.json"


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def newcombe_diff(k1, n1, k2, n2):
    """두 비율 차이 p1 - p2의 Newcombe(Wilson 기반) 95% 구간."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lo, hi


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def read_csv(name):
    with open(REPO / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    res = {}

    # ---------- 1. 폐루프 표 (표 4, 표 20): 원자료 성공 수 ----------
    rows = read_csv("closed_loop_success_by_suite.csv")
    succ = {}
    for r in rows:
        key = (r["row"], r["eval"], r["suite"], r["perturbation"])
        succ[key] = (int(r["successes"]), int(r["episodes"]))

    def cell(row, suite, pert, ev="libero_pro"):
        return succ.get((row, ev, suite, pert))

    targets = ["baseline", "clip_diff_id", "clip_diff_id_seed1", "clip_batch_shuffled",
               "clip_random_init", "token_vjepa2_id_w0.01", "token_vjepa2_id_w0.05", "token_webdino_w0.01"]
    shifts = ["language", "position", "object", "task", "environment"]
    suites = ["libero_10", "libero_spatial", "libero_object", "libero_goal"]

    # 표 20의 "L10" 열과 "네 스위트 평균" 열을 원자료에서 다시 만든다
    table = {}
    for t in targets:
        l10 = {s: cell(t, "libero_10", s) for s in shifts}
        if any(v is None for v in l10.values()):
            continue
        l10_pct = {s: 100 * l10[s][0] / l10[s][1] for s in shifts}
        all_k = sum(cell(t, su, s)[0] for su in suites for s in shifts if cell(t, su, s))
        all_n = sum(cell(t, su, s)[1] for su in suites for s in shifts if cell(t, su, s))
        pct_all = 100 * all_k / all_n
        table[t] = {"l10_pct": {k: round(v, 1) for k, v in l10_pct.items()},
                    "pro_all_pct": round(pct_all, 2), "pro_all_n": all_n}
    res["closed_loop_table"] = table

    # LIBERO 표준 평가 (Spat, Obj, Goal, L10, Avg)
    std = {}
    for t in ["baseline", "clip_diff_id", "clip_batch_shuffled", "clip_random_init",
              "token_vjepa2_id_w0.01", "token_webdino_w0.01"]:
        vals = {}
        for su in suites:
            c = cell(t, su, "none", "libero_std")
            if c:
                vals[su] = 100 * c[0] / c[1]
        if len(vals) == 4:
            std[t] = {"per_suite_pct": {k: round(v, 1) for k, v in vals.items()},
                      "avg_pct": round(sum(vals.values()) / 4, 2)}
    res["libero_std_table"] = std

    # K1: 원자료 기반 값과 논문 표 4의 값 비교 (논문 값은 논문 표 4 / 표 20에서 옮김)
    paper_l10_pos = {"baseline": 8, "clip_diff_id": 24, "clip_diff_id_seed1": 18,
                     "clip_batch_shuffled": 15, "clip_random_init": 14,
                     "token_vjepa2_id_w0.01": 23, "token_vjepa2_id_w0.05": 8, "token_webdino_w0.01": 14}
    paper_pro_all = {"baseline": 68.7, "clip_diff_id": 69.0, "clip_diff_id_seed1": 65.1,
                     "clip_batch_shuffled": 65.8, "clip_random_init": 67.6,
                     "token_vjepa2_id_w0.01": 67.5, "token_vjepa2_id_w0.05": 65.8, "token_webdino_w0.01": 65.0}
    paper_std_avg = {"baseline": 93.0, "clip_diff_id": 93.8, "clip_batch_shuffled": 89.0,
                     "clip_random_init": 93.5, "token_vjepa2_id_w0.01": 90.2, "token_webdino_w0.01": 93.5}
    k1 = {}
    for t, v in paper_l10_pos.items():
        if t in table:
            k1[t] = {"l10_pos_csv": table[t]["l10_pct"]["position"], "l10_pos_paper": v,
                     "match": abs(table[t]["l10_pct"]["position"] - v) < 0.6}
    for t, v in paper_pro_all.items():
        if t in table:
            k1.setdefault(t, {})["pro_all_csv"] = table[t]["pro_all_pct"]
            k1[t]["pro_all_paper"] = v
            k1[t]["pro_all_match"] = abs(table[t]["pro_all_pct"] - v) < 0.06
    for t, v in paper_std_avg.items():
        if t in std:
            k1.setdefault(t, {})["std_avg_csv"] = std[t]["avg_pct"]
            k1[t]["std_avg_paper"] = v
            k1[t]["std_avg_match"] = abs(std[t]["avg_pct"] - v) < 0.06
    res["K1_closed_loop_match"] = k1

    # C1: L10 위치 이동 성공률의 검정 (대 baseline), 시도 100회
    base_pos = cell("baseline", "libero_10", "position")
    c1 = {}
    for t in ["clip_diff_id", "clip_diff_id_seed1", "token_vjepa2_id_w0.01",
              "clip_batch_shuffled", "clip_random_init", "token_vjepa2_id_w0.05", "token_webdino_w0.01"]:
        c = cell(t, "libero_10", "position")
        d, lo, hi = newcombe_diff(c[0], c[1], base_pos[0], base_pos[1])
        c1[t] = {"k": c[0], "n": c[1], "base_k": base_pos[0], "base_n": base_pos[1],
                 "diff_pp": round(100 * d, 1), "newcombe95_pp": [round(100 * lo, 1), round(100 * hi, 1)],
                 "fisher_p": round(fisher(c[0], c[1], base_pos[0], base_pos[1]), 4)}
    # 다중 비교 보정: 대상 행 7개 x 같은 열의 Holm (보수적으로 Bonferroni도 같이 적는다)
    ps = sorted(((v["fisher_p"], k) for k, v in c1.items()))
    m = len(ps)
    holm = {}
    running = 0.0
    for i, (p, k) in enumerate(ps):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        holm[k] = round(running, 4)
    for k in c1:
        c1[k]["holm_p_over_7_rows"] = holm[k]
        c1[k]["bonferroni_p_over_7_rows"] = round(min(1.0, 7 * c1[k]["fisher_p"]), 4)
    res["C1_l10_position_tests"] = c1

    # C1b: 표적과 대조군 사이의 차이 (L10 위치 이동), 쌍별 Fisher
    pairs = [("clip_diff_id", "clip_batch_shuffled"), ("clip_diff_id", "clip_random_init"),
             ("token_vjepa2_id_w0.01", "token_webdino_w0.01"), ("clip_diff_id", "clip_diff_id_seed1")]
    c1b = {}
    for a_, b_ in pairs:
        ca, cb = cell(a_, "libero_10", "position"), cell(b_, "libero_10", "position")
        d, lo, hi = newcombe_diff(ca[0], ca[1], cb[0], cb[1])
        c1b[f"{a_} vs {b_}"] = {"diff_pp": round(100 * d, 1), "newcombe95_pp": [round(100 * lo, 1), round(100 * hi, 1)],
                                "fisher_p": round(fisher(ca[0], ca[1], cb[0], cb[1]), 4)}
    res["C1b_target_vs_control"] = c1b

    # C1c: 두 DIFF + ID 시드를 합친 값(42/200) 대 baseline(8/100) (참고용: 시드를 풀링하면 시드 간 변동을 과소평가한다)
    ca = (cell("clip_diff_id", "libero_10", "position")[0] + cell("clip_diff_id_seed1", "libero_10", "position")[0],
          200)
    res["C1c_pooled_two_seeds"] = {"k": ca[0], "n": ca[1], "fisher_p_vs_baseline": round(fisher(ca[0], ca[1], 8, 100), 4)}

    # C2: DIFF + ID 대 baseline을 과제별 쌍체로 (L10 위치 이동, 과제 10개, 과제당 10회)
    bytask = read_csv("closed_loop_success_by_task.csv")
    per = defaultdict(dict)
    for r in bytask:
        if r["eval"] == "libero_pro" and r["suite"] == "libero_10" and r["perturbation"] == "position":
            per[r["row"]][int(r["task_id"])] = (int(r["successes"]), int(r["episodes"]))
    diffs = []
    for tid in sorted(per["baseline"]):
        a = per["clip_diff_id"][tid][0] - per["baseline"][tid][0]
        diffs.append(a)
    diffs = np.array(diffs)
    try:
        w = stats.wilcoxon(diffs, zero_method="wilcox")
        wp = float(w.pvalue)
    except ValueError:
        wp = None
    res["C2_task_paired"] = {"n_tasks": int(len(diffs)), "diff_by_task_successes": diffs.tolist(),
                             "mean_diff_successes_per_task": round(float(diffs.mean()), 2),
                             "wilcoxon_p": None if wp is None else round(wp, 4),
                             "tasks_positive": int((diffs > 0).sum()), "tasks_negative": int((diffs < 0).sum()),
                             "tasks_zero": int((diffs == 0).sum())}

    # 시도 수 확인: 과제별 성공 수 합이 표의 성공 수와 같은가 (K2)
    k2_ok = {}
    for row in ["baseline", "clip_diff_id"]:
        s = sum(v[0] for v in per[row].values())
        n = sum(v[1] for v in per[row].values())
        k2_ok[row] = {"sum_task_successes": s, "sum_task_episodes": n,
                      "suite_row_successes": cell(row, "libero_10", "position")[0]}
    res["K2_task_sum_check"] = k2_ok

    # ---------- 2. 일치 대조 (표 2) ----------
    mc = read_csv("matched_controls_per_cell.csv")
    cellvals = defaultdict(list)
    seedvals = defaultdict(list)
    for r in mc:
        key = (r["backbone"], r["condition"])
        cellvals[key].append(float(r["r2"]))
        seedvals[(r["backbone"], r["condition"], r["training_seed"])].append(float(r["r2"]))
    paper_t2 = {("vjepa2", "frozen"): 0.455, ("vjepa2", "true_id"): 0.863, ("vjepa2", "shuffled_id"): 0.184,
                ("vjepa2", "forward_dynamics"): 0.326,
                ("videomae", "frozen"): 0.513, ("videomae", "true_id"): 0.829, ("videomae", "shuffled_id"): 0.422,
                ("videomae", "forward_dynamics"): 0.454,
                ("webdino", "frozen"): 0.075, ("webdino", "true_id"): 0.156, ("webdino", "shuffled_id"): 0.063,
                ("webdino", "forward_dynamics"): -0.071}
    k3 = {}
    for key, paper in paper_t2.items():
        v = np.array(cellvals[key])
        k3["|".join(key)] = {"mean_csv": round(float(v.mean()), 4), "paper": paper,
                             "n": int(v.size), "sd_csv": round(float(v.std(ddof=1)), 4),
                             "match_0.001": abs(float(v.mean()) - paper) < 0.0015}
    res["K3_matched_control_means"] = k3

    # 학습 시드별 평균 (시드 2개): 시드 간 변동이 얼마나 큰가
    k4 = {}
    for backbone in ["vjepa2", "videomae", "webdino"]:
        for cond in ["true_id", "shuffled_id", "forward_dynamics"]:
            s0 = np.mean(seedvals[(backbone, cond, "0")])
            s1 = np.mean(seedvals[(backbone, cond, "1")])
            k4[f"{backbone}|{cond}"] = {"seed0_mean": round(float(s0), 4), "seed1_mean": round(float(s1), 4),
                                        "abs_seed_gap": round(abs(float(s0 - s1)), 4)}
    res["K4_training_seed_gap"] = k4

    # 200 에피소드 기준 쌍체 대비 (학습 시드 2개 x 프로브 시드 3개 = 6개 값, 의사 반복)
    c5 = {}
    for backbone in ["vjepa2", "videomae", "webdino"]:
        a = np.array(cellvals[(backbone, "true_id")])
        b = np.array(cellvals[(backbone, "shuffled_id")])
        t_res = stats.ttest_ind(a, b, equal_var=False)
        c5[backbone] = {"gap_abs_200ep": round(float(a.mean() - b.mean()), 4),
                        "welch_p_pseudo_replicated": float(t_res.pvalue),
                        "note": "프로브 시드 3개를 독립으로 취급하므로 과소평가된 p. 참고용"}
    res["C5_absolute_gap_200ep"] = c5

    # ---------- 3. 교차 벤치마크 순위 상관 (표 10) ----------
    names = ["V-JEPA 2 ViT-L", "V-JEPA 2.1 ViT-B", "VideoMAE V1 ViT-L", "Web-DINO ViT-L",
             "SigLIP 2 ViT-L", "LAPA", "SDXL VAE", "Cosmos-1 tokenizer"]
    libero = [0.84, 0.82, 0.75, 0.16, 0.17, 0.51, -0.41, -0.29]
    calvin = [0.86, 0.88, 0.80, 0.81, 0.77, 0.78, 0.09, 0.39]
    meta = [0.59, 0.40, 0.35, -0.44, -0.17, -0.35, -0.54, -0.31]
    rho_ll_m = stats.spearmanr(libero, meta)
    rho_ll_c = stats.spearmanr(libero, calvin)
    rho_c_m = stats.spearmanr(calvin, meta)
    res["K5_spearman_table10"] = {
        "libero_vs_metaworld": {"rho": round(float(rho_ll_m.statistic), 3), "paper": 0.88},
        "libero_vs_calvin": {"rho": round(float(rho_ll_c.statistic), 3), "paper": 0.83},
        "calvin_vs_metaworld": {"rho": round(float(rho_c_m.statistic), 3), "paper": 0.67},
        "n_backbones": len(names),
        "note": "표 10의 8개 백본 값으로 계산. 논문 ρ는 같은 8개 백본을 쓴 것으로 보인다",
    }

    # ---------- 4. 표 1과 표 14의 차이, 표 11의 차이 (산술 점검) ----------
    t1 = {  # (frozen, +ID, paper delta)
        "V-JEPA 2 ViT-L": (0.40, 0.84, 0.44), "V-JEPA 2.1 ViT-B": (0.44, 0.82, 0.38),
        "VideoMAE V1 ViT-L": (0.46, 0.75, 0.29), "Web-DINO ViT-L": (-0.01, 0.16, 0.17),
        "SigLIP 2 ViT-L": (0.05, 0.17, 0.12), "LAPA-LAQ-OpenX": (0.41, 0.51, 0.10),
        "DIFF": (0.43, 0.57, 0.15), "SDXL VAE": (-0.55, -0.41, 0.14), "Cosmos-1 tokenizer": (-0.36, -0.29, 0.07),
    }
    res["K6_table1_delta"] = {k: {"delta_rounded": round(v[1] - v[0], 2), "paper": v[2],
                                   "diff": round(v[1] - v[0] - v[2], 2)} for k, v in t1.items()}

    # K7: 표 11 열 차이 (블록별 +ID - frozen)
    frozen11 = [0.193, 0.315, 0.411, 0.426, 0.451, 0.452, 0.459, 0.456, 0.435, 0.454, 0.444, 0.488,
                0.474, 0.485, 0.514, 0.516, 0.512, 0.502, 0.516, 0.487, 0.460, 0.422, 0.394, 0.425]
    id11 = [0.084, 0.304, 0.364, 0.416, 0.421, 0.422, 0.462, 0.475, 0.429, 0.434, 0.474, 0.509,
            0.559, 0.558, 0.615, 0.595, 0.625, 0.643, 0.632, 0.658, 0.708, 0.721, 0.712, 0.687]
    delta11 = [-0.109, -0.011, -0.047, -0.011, -0.031, -0.030, 0.003, 0.020, -0.006, -0.021, 0.030, 0.021,
               0.084, 0.072, 0.101, 0.079, 0.113, 0.141, 0.116, 0.171, 0.248, 0.300, 0.317, 0.263]
    f = np.array(frozen11)
    i_ = np.array(id11)
    dd = np.array(delta11)
    res["K7_table11"] = {
        "max_abs_delta_mismatch": round(float(np.max(np.abs(i_ - f - dd))), 3),
        "frozen_max_block": int(np.argmax(f)), "frozen_max": float(f.max()),
        "id_max_block": int(np.argmax(i_)), "id_max": float(i_.max()),
        "last4_delta_range": [float(dd[-4:].min()), float(dd[-4:].max())],
        "blocks12_18_delta_range": [float(dd[12:19].min()), float(dd[12:19].max())],
        "frozen_drop_16_to_22": round(float(f[15] - f[22]), 3),
    }

    # K8: 표 19의 "r2 - constant" 열이 같은 상수 예측값 오프셋을 쓰는지
    offset = {"caip": (-1.480, -1.446), "vjepa2": (-0.006, 0.028), "videomae": (-0.186, -0.151),
              "webdino": (-1.432, -1.398), "caip_text": (-3.950, -3.916)}
    res["K8_table19_offset"] = {k: round(v[0] - v[1], 3) for k, v in offset.items()}

    # K9: 표 6 대 표 5 대 표 9: 같은 설정(DIFF, L2, λ=0.05)의 값이 다른지
    res["K9_l2_lambda005_values"] = {"table5_lambda_0.05": 0.572, "table6_L2_default": 0.578,
                                      "table9_lambda_0.05": 0.572, "section4_6_probe_r2": 0.5724,
                                      "appendixA_k1_value": 0.578,
                                      "L1_minus_L2_table6": round(0.629 - 0.578, 3)}

    # K10: 본문 수치 점검 (V-JEPA - VideoMAE, 표 1 기준)
    res["K10_text_checks"] = {
        "vjepa_minus_videomae_table1": round(0.84 - 0.75, 2), "text_says": 0.10,
        "vjepa_minus_videomae_table2_200ep": round(0.863 - 0.829, 3), "text_says_2": 0.03,
        "webdino_probe_gap_400_to_3200_siglip": round(0.26 - (-0.02), 2), "text_says_3": "~0.2",
        "webdino_gap_to_calvin_at_3200": round(0.81 - 0.37, 2), "text_says_4": 0.43,
    }

    # K11: 표 17과 본문의 p=0.02 차이, 표 2 대 ∆26 (쌍체 차이가 아닌 값이라는 주석을 확인)
    res["K11_table17_gap"] = {"vjepa_p0.02_minus_frozen": round(0.598 - 0.400, 3),
                              "vjepa_p0.02_minus_videomae_p0.02": round(0.598 - 0.480, 3)}

    # K12: 표 13의 유효 배치와 부록 M의 유효 배치(16) 차이
    res["K12_batch_protocol"] = {"table13_vjepa_effective_batch": 4, "table13_vjepa_updates": 30000,
                                 "appendixM_matched_effective_batch": 16, "appendixM_updates": 20000}

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", OUT)
    print(json.dumps({k: res[k] for k in ["K1_closed_loop_match", "C2_task_paired", "K5_spearman_table10"]},
                     ensure_ascii=False, indent=1)[:3000])


if __name__ == "__main__":
    main()
