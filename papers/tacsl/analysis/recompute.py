"""TacSL (arXiv:2408.06506 v2) 보고 수치 재계산과 검정.

논문 표의 수치만 사용한다(원문 PDF의 Table I, II, III, IV, IX와 본문 수치).
시뮬레이션이나 재현 실행은 하지 않는다. 실행 시간은 수 초다.
결과는 recompute_results.json에 저장한다.
"""
import json
import math
import os

import numpy as np
from scipy import stats

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recompute_results.json")
res = {}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# 1) 속도 표 (Table I, II, III)
tab1_taxim = 7.28
tab1_ours = {1: 140, 2: 259, 4: 456, 8: 740, 16: 1045, 32: 1288, 64: 1470,
             128: 1545, 256: 1609, 512: 1631}
res["T1_speedup_512_env"] = round(tab1_ours[512] / tab1_taxim, 1)      # 논문 "224x"
res["T1_speedup_1_env"] = round(tab1_ours[1] / tab1_taxim, 1)          # 단일 환경 대비
# Table III 역수 검산: 0.613 ms -> FPS, 0.649 us -> FPS
res["T3_img_fps_from_total"] = round(1 / 0.613e-3, 0)                  # 표 I의 1631과 비교
res["T3_ff_fps_from_total"] = round(1 / 0.649e-6, 0)                   # 표 II의 1541043과 비교
res["T3_img_tactile_fraction"] = round(0.467 / 0.613, 3)
res["T3_ff_tactile_fraction"] = round(0.461 / 0.649, 3)

base_ff_10 = 3596      # Table II, 기준선 [16] 1 환경, 10x10
base_ff_100 = 2246     # Table II, 기준선 [16] 1 환경, 100x100
ours_ff_10 = {1: 152, 4: 734, 16: 2923, 64: 10162, 256: 32892, 1024: 127932,
              4096: 452372, 16384: 1287686, 32768: 1541043}
ours_ff_100 = {1: 144, 4: 530, 16: 2131, 64: 8070, 256: 27106, 1024: 65213, 4096: 103493}
res["T2_ff10_speedup_32768_env"] = round(ours_ff_10[32768] / base_ff_10, 1)  # "428x"
res["T2_ff100_speedup_4096_env"] = round(ours_ff_100[4096] / base_ff_100, 1)  # "46x"
res["T2_ff10_1env_ratio"] = round(ours_ff_10[1] / base_ff_10, 3)            # 1 환경에서 느림
res["T2_ff100_1env_ratio"] = round(ours_ff_100[1] / base_ff_100, 3)
res["T2_ff10_first_env_beating_baseline"] = min(k for k, v in ours_ff_10.items() if v > base_ff_10)

# 2) Table IV: 삽입, 시행 수 가정 비교 (seed 3개 x 1024 초기 배치 = 3072)
tab4_ins = {  # 열: Priv, Red, Red+Img, Red+FF, Red+Wrist, Red+Img+FF, Red+Img+Wrist
    "BC":     [0.826, 0.087, 0.820, 0.804, 0.936, 0.570, 0.941],
    "DAgger": [0.958, 0.053, 0.908, 0.925, 0.968, 0.876, 0.925],
    "AAC":    [0.973, 0.000, 0.834, 0.930, 0.948, 0.932, 0.897],
}
cols = ["Privileged", "Reduced", "Red+Img", "Red+FF", "Red+Wrist", "Red+Img+FF", "Red+Img+Wrist"]
N4 = 3072
ins = {}
for i, c in enumerate(cols):
    row = {}
    for algo in ("DAgger", "AAC"):
        a, b = tab4_ins[algo][i], tab4_ins["BC"][i]
        ka, kb = round(a * N4), round(b * N4)
        tbl = np.array([[ka, N4 - ka], [kb, N4 - kb]])
        _, p = stats.fisher_exact(tbl)
        row[algo] = {"diff_vs_BC": round(a - b, 3), "fisher_p_n3072": float(f"{p:.3g}")}
    ins[c] = row
res["T4_insertion_vs_BC_n3072"] = ins
res["T4_note"] = "시행 수는 논문에 없음. 3 seed x 1024 배치로 가정. seed 간 분산은 보고되지 않아 실제 p는 더 클 수 있음"

# 3) Table IX: 사양서 512 시행 가정, 특수 대 일반 정책 Fisher 검정
tab9 = {
    ("Placement", "m8"): (0.824, 0.887), ("Placement", "m12"): (0.992, 0.996),
    ("Placement", "m16"): (1.00, 0.991), ("Insertion", "m8"): (0.758, 0.805),
    ("Insertion", "m12"): (0.828, 0.906), ("Insertion", "m16"): (0.883, 0.922),
}
t9 = {}
for (task, peg), (sp, ge) in tab9.items():
    n = 512
    ks, kg = round(sp * n), round(ge * n)
    tbl = np.array([[ks, n - ks], [kg, n - kg]])
    _, p = stats.fisher_exact(tbl)
    t9[f"{task}_{peg}"] = {
        "specialist": sp, "generalist": ge,
        "count_check_specialist": f"{ks}/512 -> {ks/n:.4f}",
        "count_check_generalist": f"{kg}/512 -> {kg/n:.4f}",
        "diff_gen_minus_spec": round(ge - sp, 3), "fisher_p": float(f"{p:.3g}"),
    }
res["T9_specialist_vs_generalist"] = t9

# 4) 실로봇 (본문 그림 12, 81 시행 가정; 삽입 본문)
real = {"Vanilla": (22, 81, 0.272), "ColorAug": (71, 81, 0.877),
        "Diff+ColorAug": (74, 81, 0.914), "Concat+ColorAug": (63, 81, 0.779)}
rr = {}
for name, (k, n, rep) in real.items():
    lo, hi = wilson(k, n)
    rr[name] = {"k": k, "n": n, "rate": round(k / n, 4), "reported": rep,
                "match_1dp": round(100 * k / n, 1) == round(100 * rep, 1),
                "wilson95": [round(lo, 3), round(hi, 3)]}
res["real_placement"] = rr
ins_k, ins_n = 67, 81
lo, hi = wilson(ins_k, ins_n)
res["real_insertion"] = {"k": ins_k, "n": ins_n, "rate": round(ins_k / ins_n, 4),
                           "reported": 0.827, "wilson95": [round(lo, 3), round(hi, 3)]}
pairs = [("ColorAug", "Vanilla"), ("Diff+ColorAug", "ColorAug"), ("Concat+ColorAug", "Diff+ColorAug"),
         ("Concat+ColorAug", "ColorAug")]
pt = {}
for a, b in pairs:
    ka, na = real[a][0], real[a][1]
    kb, nb = real[b][0], real[b][1]
    _, p = stats.fisher_exact(np.array([[ka, na - ka], [kb, nb - kb]]))
    pt[f"{a}_vs_{b}"] = {"diff": round(ka / na - kb / nb, 3), "fisher_p": float(f"{p:.3g}")}
res["real_pairwise"] = pt
res["real_concat_77_9_possible_counts"] = [k for k in range(0, 82) if round(100 * k / 81, 1) == 77.9]

# 저장과 출력
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print(json.dumps(res, ensure_ascii=False, indent=1))
