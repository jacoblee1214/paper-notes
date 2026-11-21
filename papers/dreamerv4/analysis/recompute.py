"""Table 7 (n=1000 episodes per agent) 재계산: Wilson CI, 두 비율 검정, Table 8 평균 시간 불확실성.
입력은 논문 표 값만 사용한다 (paper/paper.txt Table 7, Table 8, Table 2). CPU, 1초 미만."""
import json, math
from scipy.stats import norm

N = 1000
# Table 7 success rates (%), columns as printed in Appendix D
T7 = {
    "Dreamer4": {"Log":99.1,"Planks":98.9,"Crafting table":98.5,"Stick":98.7,"Wooden pickaxe":96.6,"Cobblestone":95.9,"Stone pickaxe":90.1,"Iron ore":66.7,"Furnace":58.1,"Iron ingot":39.5,"Iron pickaxe":29.0,"Diamond":0.7},
    "WM+BC": {"Log":99.6,"Planks":99.6,"Crafting table":99.1,"Stick":98.9,"Wooden pickaxe":97.3,"Cobblestone":97.2,"Stone pickaxe":89.4,"Iron ore":62.9,"Furnace":51.1,"Iron ingot":27.8,"Iron pickaxe":16.9,"Diamond":0.0},
    "VLA": {"Log":98.5,"Planks":98.3,"Crafting table":97.2,"Stick":97.7,"Wooden pickaxe":94.1,"Cobblestone":91.6,"Stone pickaxe":76.7,"Iron ore":46.3,"Furnace":42.4,"Iron ingot":22.5,"Iron pickaxe":11.2,"Diamond":0.0},
    "BC": {"Log":97.3,"Planks":95.7,"Crafting table":93.5,"Stick":95.0,"Wooden pickaxe":86.5,"Cobblestone":83.9,"Stone pickaxe":53.8,"Iron ore":26.5,"Furnace":16.2,"Iron ingot":4.3,"Iron pickaxe":0.6,"Diamond":0.0},
    "VPT_ft": {"Log":84.3,"Planks":65.3,"Crafting table":4.7,"Stick":52.6,"Wooden pickaxe":0.0,"Cobblestone":6.9,"Stone pickaxe":0.0,"Iron ore":0.1,"Furnace":0.0,"Iron ingot":0.1,"Iron pickaxe":0.0,"Diamond":0.0},
}

def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return c-h, c+h

def two_prop(k1, n1, k2, n2):
    p1, p2 = k1/n1, k2/n2
    p = (k1+k2)/(n1+n2)
    se = math.sqrt(p*(1-p)*(1/n1+1/n2))
    if se == 0:
        return None, 1.0
    z = (p1-p2)/se
    return z, 2*(1-norm.cdf(abs(z)))

out = {"n_episodes": N, "wilson95": {}, "pairwise": {}}
for agent, row in T7.items():
    out["wilson95"][agent] = {}
    for item, pct in row.items():
        k = round(pct*N/100)
        lo, hi = wilson(k, N)
        out["wilson95"][agent][item] = {"k": k, "pct": pct, "ci_pct": [round(100*lo,2), round(100*hi,2)]}

# Dreamer 4 vs main baselines, key items
for other in ["WM+BC", "VLA", "BC", "VPT_ft"]:
    for item in ["Stone pickaxe", "Iron pickaxe", "Diamond"]:
        k1 = round(T7["Dreamer4"][item]*N/100); k2 = round(T7[other][item]*N/100)
        z, p = two_prop(k1, N, k2, N)
        out["pairwise"][f"Dreamer4_vs_{other}|{item}"] = {"k1": k1, "k2": k2, "z": None if z is None else round(z,3), "p_two_sided": round(p,6)}

from scipy.stats import fisher_exact
# Diamond: Dreamer 4 0.7% (7/1000) vs 0/1000 for all others: Fisher-style exact via binomial
tbl = [[7, 993], [0, 1000]]
out["diamond_fisher_vs_0"] = {"p_two_sided": round(float(fisher_exact(tbl)[1]),5), "note": "0 카운트라 정규근사 z 검정(p=0.008)은 부적절, Fisher 사용"}

# Table 8 average time to diamond (min) over successful episodes: n_succ = 7 -> spread indicates huge uncertainty
out["diamond_time_note"] = {"reported_min": 20.7, "n_successes_est": 7, "note": "평균 20.7분은 성공 7회 기준이라 표준오차를 계산할 수 없음 (SD 미보고)"}

# Table 2 FVD cascade check: text says v-space complete architecture = 124, table has no such row
out["table2_text_check"] = {"text_full_model_fvd": 57, "text_naive_fvd": 306, "text_vspace_full_fvd": 124, "table_has_124": False}

# Table 1 Oasis large / Dreamer 4 success over 16 tasks (Fisher exact, 16 tasks each)
from scipy.stats import fisher_exact
out["table1_fisher"] = {"Dreamer4_vs_OasisLarge": round(float(fisher_exact([[14,2],[5,11]])[1]),5),
                        "Dreamer4_vs_Lucid": round(float(fisher_exact([[14,2],[0,16]])[1]),10)}
out["table1_binomial_ci"] = {}
for name, k in [("Dreamer4_14of16", 14), ("OasisLarge_5of16", 5)]:
    lo, hi = wilson(k, 16)
    out["table1_binomial_ci"][name] = [round(100*lo,1), round(100*hi,1)]

# Intro claim "100x less data": VPT offline+web hours vs Dreamer 4 offline (Table 3)
out["data_ratio"] = {"VPT_web_plus_offline_h": 270000+2500, "Dreamer4_h": 2500, "ratio": round((270000+2500)/2500,1), "note": "VPT BC는 web 라벨링 데이터를 쓰고 Dreamer 4는 쓰지 않음; 100x는 web 시간 기준으로 맞음"}

# Action-data ablation: text 53% PSNR at 10h and 85% at 100h (Fig 7) -> check PSNR 85% vs caption 'over 80%'
out["fig7_text_vs_caption"] = {"text_100h_psnr_pct": 85, "caption_threshold_pct": 80}

from pathlib import Path
with open(Path(__file__).with_name("recompute_results.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(json.dumps({k: out[k] for k in ["pairwise","diamond_fisher_vs_0","table1_binomial_ci","data_ratio"]}, ensure_ascii=False, indent=1))
print("Dreamer4 diamond CI:", out["wilson95"]["Dreamer4"]["Diamond"])
print("Dreamer4 iron pickaxe CI:", out["wilson95"]["Dreamer4"]["Iron pickaxe"])
print("WM+BC iron pickaxe CI:", out["wilson95"]["WM+BC"]["Iron pickaxe"])
