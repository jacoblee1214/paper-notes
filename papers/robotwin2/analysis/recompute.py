"""RoboTwin 2.0 (arXiv:2506.18088 v2) reported-number recomputation.

Input: .scratch/paper_v2.txt  (pdftotext -layout paper/paper.pdf .scratch/paper_v2.txt)
Output: analysis/recompute_results.json  (stdout summary)
Only numpy/scipy/stdlib. No network, no model runs.
"""
import json, math, os, re
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TXT = os.path.join(ROOT, ".scratch", "paper_v2.txt")
OUT = os.path.join(HERE, "recompute_results.json")
lines = open(TXT, encoding="utf-8").read().split("\n")
res = {}

# ---------- Table 8 (10-task subset, code generation success) vs Table 1 ----------
t8 = {  # task: [R1V, R1FB, R1MM, R2V, R2FB, R2MM]  (percent)
 "beat_block_hammer":[16,48,56,23,34,53], "handover_block":[2,41,45,17,50,27],
 "pick_diverse_bottles":[65,65,64,60,60,62], "pick_dual_bottles":[99,99,100,100,100,100],
 "place_container_plate":[66,79,91,84,84,82], "place_dual_shoes":[19,22,25,0,2,22],
 "place_empty_cup":[90,90,100,61,61,85], "place_shoe":[72,90,90,100,100,100],
 "stack_blocks_three":[1,2,4,76,76,82], "stack_blocks_two":[44,68,64,100,100,100]}
t1_reported = {"R1.0 Vanilla":47.4,"R1.0 + FB":60.4,"R1.0 + MM FB":63.9,
               "R2.0 Vanilla":62.1,"R2.0 + FB":66.7,"R2.0 + MM FB":71.3}
names = ["R1.0 Vanilla","R1.0 + FB","R1.0 + MM FB","R2.0 Vanilla","R2.0 + FB","R2.0 + MM FB"]
asr = {n: float(np.mean([v[i] for v in t8.values()])) for i, n in enumerate(names)}
res["table1_from_table8"] = {n: {"recomputed": round(asr[n],2), "reported": t1_reported[n],
                                 "match": abs(asr[n]-t1_reported[n])<0.06} for n in names}
# abstract "10.9% gain": candidate pairings
res["abstract_10.9_candidates"] = {
  "R2.0MMFB_minus_R1.0FB": round(71.3-60.4,1),
  "R2.0MMFB_minus_R1.0MMFB_pts": round(71.3-63.9,1),
  "R2.0Vanilla_minus_R1.0Vanilla_pts": round(62.1-47.4,1)}
# Multimodal feedback effect per task (FB -> MM FB), sign count
fb_mm = {"R1.0": [], "R2.0": []}
for t, v in t8.items():
    fb_mm["R1.0"].append(v[2]-v[1]); fb_mm["R2.0"].append(v[5]-v[4])
res["MMFB_minus_FB_per_task"] = {k: {"deltas": d, "n_decrease": sum(x<0 for x in d),
                                     "n_increase": sum(x>0 for x in d)} for k,d in fb_mm.items()}

# ---------- Table 9 (50 tasks, R2.0 + MMFB) average ----------
t9_text = "\n".join(lines[lines.index([l for l in lines if "Table 9: Per-task" in l][0]):])
t9_block = t9_text[:t9_text.index("Avg Success Rate")+200]
t9 = re.findall(r"([A-Z][A-Za-z ]+?)\s+(\d+)%", t9_block)
t9 = [(n.strip(), int(v)) for n, v in t9 if n.strip() not in ("Rate","Task","Avg Success Rate")]
res["table9_n_tasks"] = len(t9)
res["table9_mean_50tasks"] = round(float(np.mean([v for _,v in t9])),2)
res["table9_reported_avg"] = 43.34

# ---------- Table 10 (50 tasks, RDT/Pi0/ACT/DP/DP3 Easy/Hard) ----------
def block(start_pat, end_pat):
    s = next(i for i,l in enumerate(lines) if re.search(start_pat, l))
    e = next(i for i in range(s, len(lines)) if re.search(end_pat, lines[i]))
    return lines[s:e]
row_re = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 ]+?)\s+((?:\d+(?:\.\d+)?%\s*){10})$")
def parse_rows(bl):
    out = {}
    for l in bl:
        m = row_re.match(l)
        if m:
            vals = [float(x.rstrip("%")) for x in re.findall(r"\d+(?:\.\d+)?%", m.group(2))]
            out[m.group(1).strip()] = vals
    return out
t10 = parse_rows(block(r"Table 10: RoboTwin 2.0 Simulation", r"Average \(%\)"))
res["table10_n_tasks"] = len(t10)
arr = np.array(list(t10.values()))
cols = ["RDT_easy","RDT_hard","Pi0_easy","Pi0_hard","ACT_easy","ACT_hard","DP_easy","DP_hard","DP3_easy","DP3_hard"]
res["table10_mean_50"] = {c: round(float(arr[:,i].mean()),2) for i,c in enumerate(cols)}
res["table10_reported_avg"] = dict(zip(cols,[34.5,13.7,46.4,16.3,29.7,1.7,28.0,0.6,55.2,5.0]))
res["table10_drop_pts"] = {"RDT": round(float(arr[:,0].mean()-arr[:,1].mean()),2),
                           "Pi0": round(float(arr[:,2].mean()-arr[:,3].mean()),2)}
# Wilcoxon paired easy vs hard per model (task level, 50 tasks)
res["table10_wilcoxon_easy_vs_hard_p"] = {m: float(stats.wilcoxon(arr[:,2*i], arr[:,2*i+1]).pvalue)
                                          for i,m in enumerate(["RDT","Pi0","ACT","DP","DP3"])}

# ---------- Table 11 (50 tasks, embodiments) ----------
emb_cols = ["R1_Aloha","R1_ARX","R1_Franka","R1_Piper","R1_UR5","R2_Aloha","R2_ARX","R2_Franka","R2_Piper","R2_UR5"]
t11 = parse_rows(block(r"Table 11: Success Rates of Different", r"^\s*Average\s+65\.3%|Average\s+65\.3"))
res["table11_n_tasks"] = len(t11)
a11 = np.array(list(t11.values()))
m11 = a11.mean(axis=0)
res["table11_mean_50"] = {c: round(float(m11[i]),2) for i,c in enumerate(emb_cols)}
res["table11_reported_avg"] = dict(zip(emb_cols,[65.3,68.8,67.6,2.3,57.7,78.8,74.2,67.2,25.1,57.1]))
res["table2_reported"] = {"R1.0 Aloha":65.1,"R1.0 Piper":2.4,"R1.0 Franka":67.3,"R1.0 UR5":57.6,"R1.0 ARX":68.6,
                          "R1.0 avg":52.2,"R2.0 Aloha":78.8,"R2.0 Piper":25.1,"R2.0 Franka":67.2,"R2.0 UR5":57.1,"R2.0 ARX":74.2,"R2.0 avg":60.5}
res["text_aloha_gain_claim"] = "text 13.5% vs Table 2 +13.7%; Table 11 gives 78.8-65.3=13.5"
res["table2_avg_check"] = {"R1.0": round(np.mean([65.1,2.4,67.3,57.6,68.6]),2),
                           "R2.0": round(np.mean([78.8,25.1,67.2,57.1,74.2]),2)}

# ---------- Table 3 (8 tasks, 100 rollouts assumed) ----------
t3 = {"ACT":[0,0,4,0,0,0,6,2], "DP":[0,0,0,0,0,0,0,1],
      "RDT":[30,13,12,17,7,32,24,15], "Pi0":[41,12,21,2,6,46,29,23],
      "RDT+Clean":[8,12,13,9,9,21,21,24], "Pi0+Clean":[55,15,35,8,6,33,26,21],
      "RDT+Rand":[49,17,18,6,30,35,27,16], "Pi0+Rand":[62,7,22,22,18,50,31,21]}
t3_rep = {"ACT":2.0,"DP":0.0,"RDT":18.8,"Pi0":22.5,"RDT+Clean":14.6,"Pi0+Clean":24.9,"RDT+Rand":24.8,"Pi0+Rand":29.1}
res["table3_means"] = {k: {"recomputed": round(float(np.mean(v)),2), "reported": t3_rep[k]} for k,v in t3.items()}
res["table3_rel_gain"] = {"RDT_rand_vs_RDT_pct": round((24.8/18.8-1)*100,1), "Pi0_rand_vs_Pi0_pct": round((29.1/22.5-1)*100,1),
                          "RDT_clean_vs_RDT_pct": round((14.6/18.8-1)*100,1)}
res["table3_n_tasks"] = 8  # text says "five unseen tasks" (Sec 4.3)

# ---------- Table 4 (real world, 4 tasks x 4 settings, granularity 2% => n = 50 per cell) ----------
# order per task: [Seen-F, Seen-T, Unseen-F, Unseen-T] for each of: 10 real, 10 real+1k, zero-shot
t4 = {
 "StackBowls": {"10real":[22,12,10,12], "real1k":[64,58,50,56], "zs":[None,None,60,52]},
 "Handover":   {"10real":[40,16,36,0],  "real1k":[48,12,56,36], "zs":[None,None,56,20]},
 "PickBottle": {"10real":[20,8,4,8],    "real1k":[36,40,26,28], "zs":[None,None,10,32]},
 "ClickBell":  {"10real":[36,20,12,16], "real1k":[24,56,24,48], "zs":[None,None,20,14]}}
N = 50
def pooled(key_idx, cond):
    vals = [t4[t][cond][key_idx] for t in t4 if t4[t][cond][key_idx] is not None]
    k = sum(round(v*N/100) for v in vals); n = len(vals)*N
    return k, n
def wilson(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return round(100*(c-h),1), round(100*(c+h),1)
t4_out = {}
for label, idx in [("Seen-F",0),("Seen-T",1),("Unseen-F",2),("Unseen-T",3)]:
    row = {}
    for cond in ["10real","real1k","zs"]:
        k,n = pooled(idx, cond)
        if n: row[cond] = {"k":k,"n":n,"pct":round(100*k/n,1),"wilson95":wilson(k,n)}
    t4_out[label] = row
res["table4_pooled"] = t4_out
# headline ratios from abstract
u_t_10 = t4_out["Unseen-T"]["10real"]; u_t_fs = t4_out["Unseen-T"]["real1k"]; u_t_zs = t4_out["Unseen-T"]["zs"]
res["abstract_367_check"] = round((u_t_fs["pct"]/u_t_10["pct"]-1)*100,1)
res["abstract_228_check"] = round((u_t_zs["pct"]/u_t_10["pct"]-1)*100,1)
res["all_cells_mean_10real"] = round(float(np.mean([t4[t]["10real"][i] for t in t4 for i in range(4)])),2)
res["all_cells_mean_real1k"] = round(float(np.mean([t4[t]["real1k"][i] for t in t4 for i in range(4)])),2)
res["all_cells_rel_gain_pct"] = round((res["all_cells_mean_real1k"]/res["all_cells_mean_10real"]-1)*100,1)
res["text_24.4_check"] = round(np.mean([13.5,27.5,23.5,33.0]),2)
# Fisher and 2x2 tests for Unseen-T (pooled trials, task-clustered caveat)
tab = [[u_t_10["k"], u_t_10["n"]-u_t_10["k"]], [u_t_fs["k"], u_t_fs["n"]-u_t_fs["k"]]]
res["fisher_unseenT_10real_vs_real1k"] = {"OR": None, "p": float(stats.fisher_exact(tab)[1])}
tab2 = [[u_t_10["k"], u_t_10["n"]-u_t_10["k"]], [u_t_zs["k"], u_t_zs["n"]-u_t_zs["k"]]]
res["fisher_unseenT_10real_vs_zs"] = {"p": float(stats.fisher_exact(tab2)[1])}
tab3 = [[t4_out["Seen-F"]["10real"]["k"], 200-t4_out["Seen-F"]["10real"]["k"]],
        [t4_out["Seen-F"]["real1k"]["k"], 200-t4_out["Seen-F"]["real1k"]["k"]]]
res["fisher_SeenF_10real_vs_real1k"] = {"p": float(stats.fisher_exact(tab3)[1])}

# ---------- G.4 VLM confusion matrix ----------
TP,FP,TN,FN = 16,61,40,13
acc=(TP+TN)/(TP+FP+TN+FN); prec=TP/(TP+FP); rec=TP/(TP+FN); f1=2*prec*rec/(prec+rec)
res["vlm_detection"] = {"n":TP+FP+TN+FN, "acc":round(acc,3), "prec":round(prec,3), "rec":round(rec,3), "f1":round(f1,3),
                        "reported_acc":0.431, "reported_prec":0.208, "reported_rec":0.552, "reported_f1":0.302,
                        "localization_12_of_40": round(12/40,3)}


# ---------- Task-cluster bootstrap of the headline ratios (4 tasks, Unseen-Cluttered cell) ----------
# per-task counts out of 50 (n=50 inferred from 2% granularity; paper does not state the trial count)
rng = np.random.default_rng(0)
c10 = np.array([12,0,8,16])*50/100; cfs = np.array([56,36,28,48])*50/100; czs = np.array([52,20,32,14])*50/100
def boot_ratio(a, b, B=20000):
    idx = rng.integers(0, 4, size=(B, 4))
    ra = a[idx].mean(axis=1); rb = b[idx].mean(axis=1)
    keep = ra > 0  # drop resamples where the baseline mean is 0 (ratio undefined)
    r = rb[keep]/ra[keep]
    return [round(float(np.percentile(r,2.5)),2), round(float(np.percentile(r,97.5)),2), round(float(keep.mean()),4)]
res["cluster_bootstrap_ratio_1k_vs_10real_CI95"] = boot_ratio(c10, cfs)
res["cluster_bootstrap_ratio_zs_vs_10real_CI95"] = boot_ratio(c10, czs)
res["cluster_bootstrap_note"] = "resample 4 tasks with replacement; ratio of mean success rates; 20000 draws, seed 0"

json.dump(res, open(OUT,"w"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(res, ensure_ascii=False, indent=1, default=str)[:6000])
