"""Qwen-RobotWorld (arXiv:2606.17030 v3) 표 2–5 수치 재계산.

논문이 보고한 합계(Overall, Total, Qual/Domain/Overall)를 구성 항목에서 다시 더하고,
순위와 1위와의 차이를 계산한다. 원자료(샘플별 점수, 시행 수)가 없으므로 신뢰구간은
계산하지 않는다. CPU만 쓰며 수 초에 끝난다.

실행: python3 analysis/recompute.py  → analysis/recompute_results.json
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ---- 표 2 EWMBench: SceneC, HSD, Dyn, nDTW, Diversity, BLEU, CLIP, Logics, 보고된 Overall
EWM = {
    "Veo3": ([0.8415, 0.2130, 0.1932, 0.1613, 0.0221, 0.2139, 0.8965, 0.9474], 3.49),
    "Wan2.6": ([0.6712, 0.2034, 0.0900, 0.1715, 0.0502, 0.1616, 0.8743, 1.0000], 3.22),
    "Kling26": ([0.8211, 0.3272, 0.1822, 0.3423, 0.0173, 0.2591, 0.9014, 1.0000], 3.85),
    "LTX-2": ([0.7850, 0.2076, 0.1283, 0.2443, 0.0120, 0.1425, 0.8869, 0.5000], 3.01),
    "Sora2": ([0.8526, 0.2807, 0.3494, 0.2754, 0.0314, 0.2466, 0.9100, 0.9474], 3.89),
    "Cosmos": ([0.7963, 0.2500, 0.2052, 0.2533, 0.0803, 0.1230, 0.8458, 0.7333], 3.29),
    "GigaWorld": ([0.8707, 0.3050, 0.0849, 0.2783, 0.0278, 0.2048, 0.8873, 0.9000], 3.56),
    "LVP": ([0.8795, 0.4248, 0.0433, 0.6226, 0.0093, 0.2179, 0.8995, 0.9524], 4.05),
    "Vidar": ([0.7341, 0.1877, 0.1520, 0.1769, 0.0653, 0.1607, 0.8821, 0.9411], 3.30),
    "Wow": ([0.8866, 0.2494, 0.0529, 0.2566, 0.0266, 0.1932, 0.9001, 0.9524], 3.52),
    "Ours": ([0.9142, 0.5660, 0.3429, 0.6708, 0.0114, 0.2079, 0.8834, 1.0000], 4.60),
}

# ---- 표 3 DreamGen Bench: (GR1-Env PA, IF), (GR1-Object PA, IF), (GR1-Behavior PA, IF), 보고된 Total
DREAM = {
    "Cosmos-sft": ([0.709, 0.655, 0.775, 0.720, 0.649, 0.621], 4.129),
    "LVP": ([0.810, 0.772, 0.745, 0.829, 0.713, 0.889], 4.758),
    "Vidar": ([0.445, 0.647, 0.478, 0.726, 0.394, 0.651], 3.341),
    "GigaWorld": ([0.621, 0.933, 0.500, 0.852, 0.426, 0.884], 4.216),
    "Wow": ([0.793, 0.826, 0.755, 0.849, 0.809, 0.696], 4.728),
    "Ours": ([0.828, 0.793, 0.840, 0.878, 0.781, 0.832], 4.952),
}

# ---- 표 4 PBench: VBench 8개(I2V-Bg, I2V-S, Aes, Img, Bg-Con, Mot, Sub-Con, O-Con), Qual, Domain, 보고된 Overall
PB = {
    "Veo3": ([0.975, 0.980, 0.526, 0.698, 0.938, 0.994, 0.927, 0.128], 0.771, 0.882, 0.827),
    "Wan2.6": ([0.856, 0.843, 0.514, 0.719, 0.906, 0.978, 0.843, 0.136], 0.724, 0.832, 0.778),
    "Sora2": ([0.981, 0.973, 0.487, 0.672, 0.961, 0.994, 0.954, 0.129], 0.769, 0.841, 0.805),
    "Kling26": ([0.982, 0.979, 0.521, 0.699, 0.920, 0.990, 0.927, 0.124], 0.768, 0.874, 0.821),
    "LTX-2": ([0.948, 0.955, 0.506, 0.622, 0.932, 0.986, 0.904, 0.118], 0.746, 0.845, 0.796),
    "Cosmos": ([0.974, 0.973, 0.470, 0.663, 0.940, 0.989, 0.931, 0.160], 0.763, 0.840, 0.802),
    "LVP": ([0.979, 0.981, 0.515, 0.679, 0.954, 0.991, 0.962, 0.116], 0.772, 0.812, 0.792),
    "GigaWorld": ([0.957, 0.944, 0.495, 0.641, 0.925, 0.984, 0.892, 0.128], 0.746, 0.841, 0.794),
    "Vidar": ([0.935, 0.922, 0.501, 0.573, 0.912, 0.982, 0.863, 0.120], 0.726, 0.810, 0.768),
    "Wow": ([0.967, 0.957, 0.517, 0.689, 0.941, 0.980, 0.929, 0.111], 0.761, 0.786, 0.774),
    "Ours": ([0.956, 0.943, 0.455, 0.649, 0.956, 0.990, 0.933, 0.124], 0.751, 0.857, 0.804),
}

# ---- 표 5 WorldModelBench: Instr, Frame, Temp, CS Overall, Newton, Mass, Fluid, Penetr, Grav, Phys Overall, Total
WMB = {
    "Veo3": (2.52, 0.98, 0.95, 1.93, [1.00, 0.89, 0.99, 0.91, 1.00], 4.80, 9.25),
    "Wan2.6": (2.50, 0.99, 0.95, 1.94, [1.00, 0.89, 0.99, 0.94, 1.00], 4.83, 9.27),
    "Sora2": (2.21, 0.96, 0.93, 1.88, [1.00, 0.91, 0.99, 0.95, 1.00], 4.84, 8.93),
    "Kling26": (1.59, 0.97, 1.00, 1.97, [1.00, 1.00, 1.00, 1.00, 1.00], 5.00, 8.55),
    "LTX-2": (1.97, 0.69, 0.62, 1.32, [0.99, 0.60, 1.00, 0.73, 1.00], 4.32, 7.61),
    "Cosmos": (2.14, 1.00, 0.94, 1.94, [1.00, 0.92, 1.00, 0.94, 1.00], 4.86, 8.94),
    "LVP": (2.01, 0.89, 0.91, 1.80, [1.00, 0.93, 0.99, 0.95, 1.00], 4.87, 8.67),
    "GigaWorld": (2.13, 0.59, 0.46, 1.05, [1.00, 0.48, 0.99, 0.69, 0.98], 4.13, 7.31),
    "Vidar": (1.62, 0.54, 0.45, 0.99, [1.00, 0.56, 1.00, 0.85, 1.00], 4.40, 7.01),
    "Wow": (2.05, 0.76, 0.65, 1.41, [1.00, 0.65, 0.99, 0.81, 1.00], 4.45, 7.91),
    "Ours": (2.33, 0.87, 0.85, 1.72, [1.00, 1.00, 1.00, 0.94, 1.00], 4.94, 8.99),
}


def rank_of(scores, name):
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])
    pos = [k for k, _ in ordered].index(name) + 1
    return pos, ordered[0][1] - scores[name]


def main():
    res = {}
    res["ewm"] = {}
    for m, (vals, rep) in EWM.items():
        s = round(sum(vals), 4)
        res["ewm"][m] = {"sum": s, "reported": rep, "match_2dp": round(s, 2) == rep}
    ewm_scores = {m: round(sum(v), 4) for m, (v, _) in EWM.items()}
    res["ewm_rank"] = {"ours": rank_of(ewm_scores, "Ours")[0],
                       "gap_to_second": round(sorted(ewm_scores.values())[-1] - sorted(ewm_scores.values())[-2], 4)}
    res["ewm_ours_vs_lvp_hsd_pct"] = round((EWM["Ours"][0][1] / EWM["LVP"][0][1] - 1) * 100, 1)

    res["dream"] = {}
    for m, (vals, rep) in DREAM.items():
        s = round(sum(vals), 4)
        res["dream"][m] = {"sum": s, "reported": rep, "match": abs(s - rep) < 1e-3}
    d_scores = {m: round(sum(v), 4) for m, (v, _) in DREAM.items()}
    res["dream_rank"] = rank_of(d_scores, "Ours")[0]
    res["dream_gigaworld_env_if_beats_ours"] = DREAM["GigaWorld"][0][1] > DREAM["Ours"][0][1]

    res["pb"] = {}
    for m, (v8, qual, dom, rep) in PB.items():
        q = round(sum(v8) / 8, 4)
        o = round((qual + dom) / 2, 4)
        res["pb"][m] = {"qual_from_8": q, "qual_reported": qual, "qual_match": abs(q - qual) < 6e-4,
                        "overall_from_qd": o, "overall_reported": rep, "overall_match": abs(o - rep) < 6e-4}
    open_src = [m for m in PB if m not in ("Veo3", "Wan2.6", "Sora2", "Kling26", "LTX-2", "Ours")]
    res["pb_ours_margin_over_best_open"] = {
        "best_open": max(open_src, key=lambda m: PB[m][3]),
        "margin": round(PB["Ours"][3] - max(PB[m][3] for m in open_src), 4)}
    res["pb_domain_rank"] = rank_of({m: v[2] for m, v in PB.items()}, "Ours")[0]
    res["pb_overall_rank"] = rank_of({m: v[3] for m, v in PB.items()}, "Ours")[0]

    res["wmb"] = {}
    for m, (i, f, t, cs, phys, pov, tot) in WMB.items():
        pvs = round(sum(phys), 2)
        res["wmb"][m] = {"cs_match": abs(f + t - cs) < 0.02, "phys_sum": pvs, "phys_reported": pov,
                         "phys_match": abs(pvs - pov) < 0.02, "total_check": round(i + cs + pov, 2),
                         "total_reported": tot, "total_match": abs(i + cs + pov - tot) < 0.02}
    w_scores = {m: v[6] for m, v in WMB.items()}
    res["wmb_ours_rank"] = rank_of(w_scores, "Ours")[0]
    res["wmb_ours_penetration"] = WMB["Ours"][4][3]

    out = HERE / "recompute_results.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
