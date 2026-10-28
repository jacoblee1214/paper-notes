"""Hydra-0 수치 점검 (CPU).

1) 논문 Table 2(paper/paper.txt)와 프로젝트 페이지 'Full numerical results'(paper/project_page.html)의 값 일치 여부
2) Average 행 재계산 (무가중 평균, SE 결합 sqrt(Σse²)/k)
3) 헤드라인 수치(90.4%, 60.2%) 재현과, 비교 기준을 바꿨을 때의 감소율
4) '같은 Cosmos 백본에서 5개 지표 모두 개선' 주장, '4-step이 대부분 칸에서 최고' 주장 확인
5) 4-step vs 50-step 교사(A14B) 차이의 z 점수
6) RoboLab: 페이지의 혼동행렬(FP 10, FN 11, 300 에피소드)로 일치율·κ·전체 성공률 재계산, Pearson r 신뢰구간
실행: python3 analysis/table_check.py → analysis/table_check_results.json
"""
import html
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["PSNR", "SSIM", "ObjEPE", "GripEPE", "FID", "FVD", "VLM"]
HIGHER = {"PSNR": True, "SSIM": True, "ObjEPE": False, "GripEPE": False, "FID": False, "FVD": False, "VLM": True}
MODELS = ["ATI", "Wan-Move", "Cosmos 2.5", "Ours (Cosmos 2.5 2B)", "Ours (Wan2.2 5B)",
          "Ours (Wan2.2 A14B)", "Ours (Wan2.2 A14B 4-step)"]
DATASETS = ["XVLA-Soft-Fold", "Deform360", "DROID", "MolmoAct2", "ABC-130k", "Average"]
NUM = r"(?:—|\d+\.\d+±\d+\.\d+|\d+\.\d+)"


def parse_paper():
    lines = (ROOT / "paper" / "paper.txt").read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("Table 2:"))
    rows = []
    for l in lines[start:start + 60]:
        for m in MODELS[::-1]:
            idx = l.find(m + " ")
            if idx >= 0 and (m != "Cosmos 2.5" or "Ours" not in l):
                vals = re.findall(NUM, l[idx + len(m):])
                if len(vals) == 7:
                    rows.append((m, vals))
                break
    assert len(rows) == 42, len(rows)
    table = {}
    for i, (m, vals) in enumerate(rows):
        ds = DATASETS[i // 7]
        rec = {}
        for k, v in zip(METRICS, vals):
            if v == "—":
                rec[k] = None
            elif "±" in v:
                a, b = v.split("±")
                rec[k] = (float(a), float(b))
            else:
                rec[k] = (float(v), None)
        table[(ds, m)] = rec
    return table


def parse_page():
    s = (ROOT / "paper" / "project_page.html").read_text()
    s = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", s, flags=re.S)
    t = html.unescape(re.sub(r"<[^>]+>", " | ", s))
    t = re.sub(r"(\s*\|\s*)+", " | ", t)
    t = t[t.find("Full numerical results"):]
    out = {}
    order = ["Average", "XVLA-Soft-Fold", "Deform360", "DROID", "MolmoAct2", "ABC-130k"]
    for ds in order:
        seg = t[t.find(ds + " —"):]
        for m in MODELS:
            j = seg.find(" | " + m + " | ")
            vals = re.findall(r"\| (—|\d+\.\d+) ", seg[j + len(m) + 4: j + len(m) + 400])[:7]
            out[(ds, m)] = [None if v == "—" else float(v) for v in vals]
    return out


def main():
    paper, page = parse_paper(), parse_page()
    res = {}

    mism = []
    for key, rec in paper.items():
        for k, pv in zip(METRICS, page[key]):
            a = rec[k][0] if rec[k] else None
            if (a is None) != (pv is None) or (a is not None and abs(a - pv) > 1e-9):
                mism.append({"cell": f"{key[0]}/{key[1]}/{k}", "paper": a, "page": pv})
    res["paper_vs_page_mismatches"] = mism

    avg_check = []
    for m in MODELS:
        for k in METRICS:
            vals = [paper[(d, m)][k] for d in DATASETS[:-1] if paper[(d, m)][k] is not None]
            mean = sum(v[0] for v in vals) / len(vals)
            rep = paper[("Average", m)][k]
            entry = {"cell": f"{m}/{k}", "reported": rep[0], "recomputed": round(mean, 3),
                     "abs_diff": round(abs(mean - rep[0]), 3)}
            if rep[1] is not None and all(v[1] is not None for v in vals):
                se = math.sqrt(sum(v[1] ** 2 for v in vals)) / len(vals)
                entry["reported_se"], entry["recomputed_se"] = rep[1], round(se, 3)
            avg_check.append(entry)
    tol = {"PSNR": 0.006, "SSIM": 0.0006, "ObjEPE": 0.006, "GripEPE": 0.006, "FID": 0.06, "FVD": 0.06, "VLM": 0.006}
    res["average_row_off_by_more_than_rounding"] = [e for e in avg_check
                                                    if e["abs_diff"] > tol[e["cell"].split("/")[-1]]]
    res["average_se_off"] = [e for e in avg_check if "reported_se" in e and abs(e["reported_se"] - e["recomputed_se"]) > 0.011]

    A = lambda m, k: paper[("Average", m)][k][0]
    red = lambda base, new: round(100 * (1 - new / base), 2)
    best = "Ours (Wan2.2 A14B 4-step)"
    res["headline"] = {
        "grip_EPE_reduction_best_vs_Cosmos2.5": red(A("Cosmos 2.5", "GripEPE"), A(best, "GripEPE")),
        "obj_EPE_reduction_best_vs_Cosmos2.5": red(A("Cosmos 2.5", "ObjEPE"), A(best, "ObjEPE")),
        "controlled_same_backbone_grip": red(A("Cosmos 2.5", "GripEPE"), A("Ours (Cosmos 2.5 2B)", "GripEPE")),
        "controlled_same_backbone_obj": red(A("Cosmos 2.5", "ObjEPE"), A("Ours (Cosmos 2.5 2B)", "ObjEPE")),
        "best_vs_zero_shot_ATI_grip": red(A("ATI", "GripEPE"), A(best, "GripEPE")),
        "best_vs_zero_shot_ATI_obj": red(A("ATI", "ObjEPE"), A(best, "ObjEPE")),
        "best_vs_zero_shot_WanMove_obj": red(A("Wan-Move", "ObjEPE"), A(best, "ObjEPE")),
        "Cosmos2.5_baseline_grip_vs_ATI_ratio": round(A("Cosmos 2.5", "GripEPE") / A("ATI", "GripEPE"), 2),
    }

    claim = []
    for d in DATASETS[:-1]:
        for k in ["PSNR", "SSIM", "GripEPE", "FID", "FVD", "ObjEPE"]:
            b, o = paper[(d, "Cosmos 2.5")][k], paper[(d, "Ours (Cosmos 2.5 2B)")][k]
            if b is None:
                continue
            ok = o[0] > b[0] if HIGHER[k] else o[0] < b[0]
            if not ok:
                claim.append(f"{d}/{k}: ours {o[0]} vs base {b[0]}")
    res["controlled_claim_violations"] = claim

    cells, wins = 0, 0
    for d in DATASETS[:-1]:
        for k in METRICS:
            col = {m: paper[(d, m)][k][0] for m in MODELS if paper[(d, m)][k] is not None}
            if not col:
                continue
            cells += 1
            top = max(col.values()) if HIGHER[k] else min(col.values())
            wins += col[best] == top
    res["4step_best_cells"] = f"{wins}/{cells}"

    zs = {}
    for k in METRICS:
        a, b = paper[("Average", best)][k], paper[("Average", "Ours (Wan2.2 A14B)")][k]
        if a[1] and b[1]:
            zs[k] = round((a[0] - b[0]) / math.sqrt(a[1] ** 2 + b[1] ** 2), 1)
    res["4step_minus_teacher_z_on_Average"] = zs

    N, FP, FN, ref_rate = 300, 10, 11, 0.267
    ref_pos = round(ref_rate * N)
    sim_pos = ref_pos - FN + FP
    po = (N - FP - FN) / N
    pr, ps = ref_pos / N, sim_pos / N
    pe = pr * ps + (1 - pr) * (1 - ps)
    kappa = (po - pe) / (1 - pe)

    def r_ci(r, n):
        z, se = math.atanh(r), 1 / math.sqrt(n - 3)
        return [round(math.tanh(z - 1.96 * se), 3), round(math.tanh(z + 1.96 * se), 3)]

    res["robolab"] = {
        "reference_successes": ref_pos, "simulated_successes": sim_pos,
        "agreement": round(po, 4), "cohen_kappa": round(kappa, 3),
        "sim_rate": round(ps, 4), "ref_rate": round(pr, 4),
        "always_fail_baseline_accuracy": round(1 - pr, 3),
        # MAE 5.7pp(반올림) × 30쌍 × 10회 = 쌍별 성공 횟수 차이의 합 T. 정수 T 중 반올림해서 5.7이 되는 값
        "abs_count_diff_sum_T_compatible_with_MAE": [T for T in range(0, 60) if round(T / 300 * 100, 1) == 5.7],
        "note": "에피소드 불일치 수 ≥ T (쌍 안에서 FP와 FN이 상쇄될 수 있으므로)",
        "pearson_r_0.96_CI_n30": r_ci(0.96, 30), "pearson_r_0.96_CI_n5": r_ci(0.96, 5),
        "sensitivity(TP/ref_pos)": round((ref_pos - FN) / ref_pos, 3),
        "precision(TP/sim_pos)": round((ref_pos - FN) / sim_pos, 3),
    }
    corpus = {"DROID": (223075, 313.7), "ABC-130k": (1048681, 1474.7), "MolmoAct2": (126335, 177.7),
              "EgoDex": (89380, 125.7), "Deform360": (60315, 84.8), "XVLA-Soft-Fold": (17772, 25.0), "H1-Fold-Clothes": (76, 0.1)}
    res["corpus_hours"] = {k: {"recomputed_h": round(w * 81 / 16 / 3600, 1), "reported_h": h,
                               "share_pct": round(100 * h / 2201.7, 1)} for k, (w, h) in corpus.items()}
    res["corpus_total_h"] = round(sum(h for _, h in corpus.values()), 1)
    (ROOT / "analysis" / "table_check_results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
