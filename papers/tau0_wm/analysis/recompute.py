"""τ0-WM (arXiv:2606.01027 v2) 표·그림 수치 재계산과 일관성·유의성 검사.

입력은 논문 PDF에서 옮긴 값이다(Table I, II의 숫자, Fig. 4 막대 위의 숫자 라벨, 본문 수치).
논문은 주 결과(Fig. 4)와 Table I의 시행 수를 밝히지 않는다. Table II는 본문에 "20회 반복"이 있다.
그래서 시행 수가 필요한 검정은 가정값을 쓰고 결과 옆에 가정을 적는다.
  - Table II: n = 20 (본문 VII-B)
  - Fig. 4 성공률: n = 20 가정 (모든 값이 0.05의 배수)
  - Table I SFT: n = 20 가정 (0.05의 배수), Table I zero-shot: n = 32 가정 (0.22, 0.06, 0.56, 0.53이 정수 성공 수로 복원되는 가장 작은 n)
실행: python3 analysis/recompute.py  → analysis/recompute_results.json (수 초, numpy·scipy)
"""
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

OUT = Path(__file__).with_name("recompute_results.json")
RES = {}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def fisher(k1, n1, k2, n2):
    return float(fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def rnd(x, d=2):
    """반올림(0.5 올림, 부동소수 오차 방지)"""
    return math.floor(x * 10 ** d + 0.5 + 1e-9) / 10 ** d


# ---- V1 데이터 구성 (III절, 그림 1)
hours = {"robot": 17.8, "umi": 6.5, "ego": 3.0}
tot = sum(hours.values())
RES["V1_data_mix"] = {"sum_hours": round(tot, 2), "claimed": 27.3,
                    "share_pct": {k: round(100 * v / tot, 1) for k, v in hours.items()},
                    "fig1_labels_pct": {"robot": 65, "umi": 24, "ego": 11},
                    "labels_sum": 65 + 24 + 11}

# ---- Fig. 4 값 (막대 라벨)
tasks = ["Toolbox", "School Bag", "Badminton", "Faucet"]
SR = {"Fast-WAM": [.20, .30, .25, .05], "pi0.5": [.60, .60, .40, .15], "tau0-WM": [.55, .80, .80, .40]}
SR_AVG_LABEL = {"Fast-WAM": .20, "pi0.5": .44, "tau0-WM": .64}
PR = {"Fast-WAM": [.35, .40, .55, .30], "pi0.5": [.65, .58, .73, .45], "tau0-WM": [.63, .73, .93, .60]}
PR_AVG_LABEL = {"Fast-WAM": .40, "pi0.5": .60, "tau0-WM": .72}

# ---- V2 Fig. 4 평균
v2 = {}
for m in SR:
    a = float(np.mean(SR[m])); b = float(np.mean(PR[m]))
    v2[m] = {"sr_mean": round(a, 4), "sr_rounded": rnd(a), "sr_label": SR_AVG_LABEL[m], "sr_match": rnd(a) == SR_AVG_LABEL[m],
             "prog_mean": round(b, 4), "prog_rounded": rnd(b), "prog_label": PR_AVG_LABEL[m], "prog_match": rnd(b) == PR_AVG_LABEL[m]}
RES["V2_fig4_means"] = v2

# ---- V3 Table I, II 평균과 정수 복원
T1 = {"zs": {"Robot": [.22, .06, .14], "Robot+UMI+Ego": [.56, .53, .55]},
      "sft": {"Robot": [.85, .55, .70], "Robot+UMI+Ego": [.90, .75, .83]}}
T2 = {"w/o TTC": [.55, .30, .43], "w. CFG": [.25, .15, .20], "w. ACG": [.40, .35, .38], "w. RCS": [.65, .35, .50], "w. RCS+LAR": [.70, .50, .60]}
v3 = {"table1": {}, "table2": {}}
for blk, rows in T1.items():
    for k, (c, u, a) in rows.items():
        m = (c + u) / 2
        v3["table1"][f"{blk}/{k}"] = {"mean_of_two": m, "rounded": rnd(m), "label": a, "match": rnd(m) == a}
for k, (a, b, c) in T2.items():
    m = (a + b) / 2
    v3["table2"][k] = {"mean_of_two": m, "rounded": rnd(m), "label": c, "match": rnd(m) == c, "tie_at_half": abs(m * 100 - math.floor(m * 100) - .5) < 1e-6}


def feasible_n(vals, nmax=200):
    out = []
    for n in range(1, nmax + 1):
        if all(any(abs(rnd(k / n) - v) < 1e-9 for k in range(n + 1)) for v in vals):
            out.append(n)
    return out


v3["n_feasible"] = {
    "table1_zero_shot_cells(.22,.06,.56,.53)": feasible_n([.22, .06, .56, .53])[:8],
    "table1_sft_cells(.85,.55,.90,.75)": feasible_n([.85, .55, .90, .75])[:8],
    "table2_cells": feasible_n([.55, .30, .25, .15, .40, .35, .65, .70, .50])[:8],
    "fig4_sr_cells": feasible_n([x for m in SR.values() for x in m])[:8],
}
# zero-shot 평균 0.55: n=32에서 (18+17)/64 = 0.546875
v3["zero_shot_n32_pooled"] = {"Robot": (7 + 2) / 64, "Robot+UMI+Ego": (18 + 17) / 64}
RES["V3_table_means_and_integrality"] = v3

# ---- V4 진행도 >= 성공률 검사 (성공한 시행의 진행도를 1로 보는 정의일 때 항상 성립)
v4 = []
for m in SR:
    for t, s, p in zip(tasks, SR[m], PR[m]):
        v4.append({"method": m, "task": t, "sr": s, "progress": p, "progress_ge_sr": p >= s - 1e-9})
RES["V4_progress_vs_sr"] = {"cells": v4, "violations": [c for c in v4 if not c["progress_ge_sr"]]}
# 진행도 값이 1/80의 배수인가 (n=20, 단계 4개 가정)
RES["V4b_progress_grid"] = {m: [abs(p * 80 - round(p * 80)) < 0.5 for p in PR[m]] for m in PR}
RES["V4b_progress_values_x80"] = {m: [round(p * 80, 2) for p in PR[m]] for m in PR}

# ---- V5 Fig. 4 성공률 비교 (n=20 가정)
N = 20
cnt = {m: [round(x * N) for x in SR[m]] for m in SR}
v5 = {"assumed_n": N, "counts": cnt, "wilson": {m: [wilson(k, N) for k in cnt[m]] for m in SR}, "pairs": {}}
for a, b in [("tau0-WM", "pi0.5"), ("tau0-WM", "Fast-WAM"), ("pi0.5", "Fast-WAM")]:
    rows = []
    for i, t in enumerate(tasks):
        rows.append({"task": t, "k": [cnt[a][i], cnt[b][i]], "p_fisher": round(fisher(cnt[a][i], N, cnt[b][i], N), 4)})
    ka, kb = sum(cnt[a]), sum(cnt[b])
    ps = sorted(r["p_fisher"] for r in rows)
    holm, run = [], 0.0
    for i, p in enumerate(ps):
        run = max(run, min(1.0, p * (len(ps) - i)))
        holm.append(run)
    v5["pairs"][f"{a} vs {b}"] = {"per_task": rows, "pooled_k": [ka, kb], "pooled_n": 4 * N,
                                  "pooled_p_fisher": round(fisher(ka, 4 * N, kb, 4 * N), 5),
                                  "n_task_p_lt_0.05": sum(p < .05 for p in ps), "holm_adjusted_sorted": [round(h, 4) for h in holm]}
v5["tau0_wins_by_sr"] = sum(SR["tau0-WM"][i] > max(SR["pi0.5"][i], SR["Fast-WAM"][i]) for i in range(4))
v5["tau0_wins_by_progress"] = sum(PR["tau0-WM"][i] > max(PR["pi0.5"][i], PR["Fast-WAM"][i]) for i in range(4))
v5["pooled_wilson"] = {m: wilson(sum(cnt[m]), 4 * N) for m in SR}
v5["sr_gain_tau0_minus_pi05_pp"] = round(100 * (np.mean(SR["tau0-WM"]) - np.mean(SR["pi0.5"])), 2)
v5["relative_gain_tau0_over_pi05"] = round(float(np.mean(SR["tau0-WM"]) / np.mean(SR["pi0.5"]) - 1), 3)
RES["V5_fig4_tests"] = v5

# ---- V6 Table I (SFT n=20, zero-shot n=32 가정)
v6 = {}
for blk, n in (("zs", 32), ("sft", 20)):
    base, full = T1[blk]["Robot"], T1[blk]["Robot+UMI+Ego"]
    kb = [round(x * n) for x in base[:2]]; kf = [round(x * n) for x in full[:2]]
    v6[blk] = {"assumed_n": n, "robot_k": kb, "full_k": kf,
               "p_clean": round(fisher(kf[0], n, kb[0], n), 5), "p_clut": round(fisher(kf[1], n, kb[1], n), 5),
               "pooled": [sum(kf), 2 * n, sum(kb), 2 * n], "p_pooled": round(fisher(sum(kf), 2 * n, sum(kb), 2 * n), 6),
               "wilson_robot": [wilson(k, n) for k in kb], "wilson_full": [wilson(k, n) for k in kf]}
v6["zs_ratio"] = round(.55 / .14, 2)
v6["zs_alt_n100"] = {"p_clean": round(fisher(56, 100, 22, 100), 8), "p_clut": round(fisher(53, 100, 6, 100), 10)}
RES["V6_table1_tests"] = v6

# ---- V7 Table II (n=20)
n = 20
k = {name: [round(v[0] * n), round(v[1] * n)] for name, v in T2.items()}
v7 = {"assumed_n": n, "counts": k, "pairs": {}}
for a, b in [("w. RCS", "w/o TTC"), ("w. RCS+LAR", "w. RCS"), ("w. RCS+LAR", "w/o TTC"), ("w. RCS+LAR", "w. ACG"), ("w. RCS+LAR", "w. CFG"), ("w. CFG", "w/o TTC"), ("w. ACG", "w/o TTC")]:
    per = [round(fisher(k[a][i], n, k[b][i], n), 4) for i in range(2)]
    pooled = round(fisher(sum(k[a]), 2 * n, sum(k[b]), 2 * n), 4)
    v7["pairs"][f"{a} vs {b}"] = {"k": [sum(k[a]), sum(k[b])], "per_task_p": per, "pooled_p": pooled}
v7["wilson"] = {name: [wilson(x, n) for x in ks] for name, ks in k.items()}
p0 = np.mean(T2["w/o TTC"][:2])
v7["oracle_best_of_4_if_iid"] = {"p_single": round(float(p0), 3), "upper_bound": round(float(1 - (1 - p0) ** 4), 3),
                                 "per_task": [round(1 - (1 - .55) ** 4, 3), round(1 - (1 - .30) ** 4, 3)]}
v7["gain_pp_vs_wo_TTC"] = {name: round(100 * (np.mean(T2[name][:2]) - p0), 1) for name in T2}
# 한 칸 20회 대 20회에서 p<0.05가 되는 최소 차이 (기준 성공 횟수별)
mdd = {}
for b in range(0, 21):
    for a in range(b + 1, 21):
        if fisher(a, 20, b, 20) < .05:
            mdd[b] = (a - b) * 5
            break
v7["min_detectable_pp_n20"] = mdd
RES["V7_table2_tests"] = v7

# ---- V8 지연 산술 (부록 A, B)
lat = {"standard_ms": 220, "text_cache_ms": 180, "compile_ms": 140}
RES["V8_latency"] = {**lat, "cache_saving_pct": round(100 * (220 - 180) / 220, 1), "compile_saving_pct": round(100 * (180 - 140) / 180, 1),
                   "total_saving_pct": round(100 * (220 - 140) / 220, 1), "chunk_executed": 30, "queries_per_s_at_180ms": round(1000 / 180, 2),
                   "queries_per_s_at_140ms": round(1000 / 140, 2)}

# ---- V9 입력 크기 산술 (코드 설정 configs/*.yaml: img_size 192x256, 2뷰 가로 결합, 잠재 stride 16, 패치 1x2x2, chunk 9프레임)
H, W, V, frames = 192, 256, 2, 9
lat_frames = (frames - 1) // 4 + 1
tokens = lat_frames * (H // 16) * (V * W // 16) // (2 * 2)
params_video = json.loads((Path(__file__).with_name("param_count_results.json")).read_text())["video_backbone_params"]
n_mem, sim_steps, N_prop = 3, 5, 4          # 공개 코드 기본값: 기억 프레임 3, 시뮬레이터 5스텝, 후보 4 (논문 기본 후보 수)
tokens_sim = (lat_frames + n_mem) * (H // 16) * (V * W // 16) // 4
vam_tflop = 2 * params_video * tokens / 1e12
acvs_tflop = 2 * params_video * tokens_sim * sim_steps * N_prop / 1e12
RES["V9_compute_arith"] = {"latent_frames": lat_frames, "video_tokens_per_forward": tokens,
                         "video_fwd_tflop_est": round(vam_tflop, 2),
                         "sim_tokens_with_memory": tokens_sim, "acvs_all_candidates_tflop_est": round(acvs_tflop, 1),
                         "acvs_over_vam_ratio": round(acvs_tflop / vam_tflop, 1),
                         "note": "2*파라미터*토큰 근사. 어텐션 항, VAE, 텍스트 인코더, 행동·보상 가지 제외. 설정값은 공개 코드 yaml과 기본 인자 기준이고 논문에는 해상도가 없음"}

# ---- V10 RCS 점수와 시행 수: 논문 Table II의 RCS 이득이 우연의 범위인지
RES["V10_rcs_gain_se"] = {"diff_pp": 7.5, "se_pp_n40_pooled": round(100 * math.sqrt(.5 * .5 / 40 * 2), 1)}

OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print(json.dumps({k: RES[k] for k in ("V2_fig4_means", "V4_progress_vs_sr")}, ensure_ascii=False, indent=1, default=float)[:2500])
print(json.dumps(RES["V5_fig4_tests"]["pairs"], ensure_ascii=False, default=float)[:3000])
print(json.dumps(RES["V6_table1_tests"], ensure_ascii=False, default=float))
print(json.dumps(RES["V7_table2_tests"], ensure_ascii=False, default=float))
print(RES["V3_table_means_and_integrality"]["n_feasible"], RES["V9_compute_arith"])
