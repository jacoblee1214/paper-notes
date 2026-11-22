"""LaWAM (arXiv:2606.15768 v2) 표 수치 재계산. 표준 라이브러리만 사용, 수 초.

실행: python3 analysis/recompute.py  ->  analysis/recompute_results.json
표 5(RoboTwin 과제별)는 paper/paper.txt 에서 읽는다(paper.txt 가 없으면 이 부분을 건너뜀).
"""
import json
import math
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent / "paper" / "paper.txt"
PAPER_V1 = HERE.parent / "paper" / "paper_v1.txt"
OUT = {}


# ---------- 통계 도구 ----------
def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def fisher(k1, n1, k2, n2):
    """양측 Fisher 정확 검정 (확률이 관측값 이하인 표를 모두 더함)."""
    K = k1 + k2
    N = n1 + n2

    def pm(x):
        return math.comb(n1, x) * math.comb(n2, K - x) / math.comb(N, K)
    lo, hi = max(0, K - n2), min(n1, K)
    p0 = pm(k1)
    return min(1.0, sum(pm(x) for x in range(lo, hi + 1) if pm(x) <= p0 * (1 + 1e-9)))


def ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def ztest(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return 2 * (1 - ncdf(abs(p1 - p2) / se)) if se > 0 else 1.0


def sign_test(wins, losses):
    n = wins + losses
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def mde(n1, n2, p=0.5, z_a=1.959964, z_b=0.841621):
    """80% 검정력, 양측 5%에서 검출 가능한 최소 차이(퍼센트포인트, 근사)."""
    return 100 * (z_a + z_b) * math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj = [0.0] * len(ps)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


# ---------- V1. 표 1 (LIBERO) ----------
T1 = {  # Long, Goal, Object, Spatial, Average(보고), 지연 ms, 크기
    "OpenVLA-OFT": [94.5, 97.9, 98.4, 97.6, 97.1, None, "7B"],
    "pi0": [88.4, 94.4, 96.8, 98.0, 94.4, 220, "3.5B"],
    "pi0.5": [92.4, 98.0, 98.2, 98.8, 96.9, 220, "3.5B"],
    "GR00T-N1.6": [94.4, 97.5, 98.5, 97.7, 97.0, 259, "3.3B"],
    "LAPA": [55.4, 58.8, 74.6, 73.8, 65.7, None, "7B"],
    "UniVLA": [92.0, 95.6, 96.8, 96.5, 95.2, None, "7B"],
    "Mantis": [94.2, 94.4, 99.2, 98.8, 96.7, None, "5.8B"],
    "VLA-JEPA": [95.8, 97.2, 99.6, 96.2, 97.2, None, "3B"],
    "F1": [91.3, 95.4, 97.8, 98.2, 95.7, 399.0, "4B"],
    "Motus": [97.6, 96.6, 99.8, 96.8, 97.7, 3231, "8B"],
    "Cosmos-Policy": [97.6, 98.2, 100.0, 98.1, 98.5, 1413, "2.1B"],
    "LingBot-VA": [98.5, 97.2, 99.6, 98.5, 98.5, 4482, "5.5B"],
    "Fast-WAM": [95.2, 97.0, 100.0, 98.2, 97.6, 486, "6B"],
    "LaWAM": [97.0, 98.4, 99.6, 99.4, 98.6, 187, "2.3B"],
}
v1 = {}
for m, r in T1.items():
    mean4 = sum(r[:4]) / 4
    # 스위트당 500회(10과제 x 50)에서 정수 성공 수가 되는가?
    nonint = [x for x in r[:4] if abs(x * 5 - round(x * 5)) > 1e-6]
    v1[m] = {"avg_recomputed": round(mean4, 3), "avg_reported": r[4],
             "avg_match(0.05)": abs(mean4 - r[4]) <= 0.05 + 1e-9,
             "non_integer_at_500": nonint}
OUT["V1_table1"] = v1
la = T1["LaWAM"]
la_succ = [round(x * 5) for x in la[:4]]
OUT["V1_lawam_counts"] = {"per_suite_successes_of_500": la_succ, "total_of_2000": sum(la_succ),
                          "rate": sum(la_succ) / 20, "wilson95": wilson(sum(la_succ), 2000)}
cmp1 = {}
for m in ["LingBot-VA", "Cosmos-Policy", "Motus", "Fast-WAM", "VLA-JEPA", "OpenVLA-OFT", "pi0.5", "GR00T-N1.6"]:
    r = T1[m]
    k = round(r[4] * 20)  # 2000회 가정(원 논문 평가 횟수가 다르면 근사)
    cmp1[m] = {"avg": r[4], "fisher_p_vs_LaWAM_if_n2000": fisher(sum(la_succ), 2000, k, 2000)}
OUT["V1_lawam_vs_others_n2000_assumed"] = cmp1
OUT["V1_mde_pp_n2000_each"] = mde(2000, 2000, p=0.985)
OUT["V1_long_suite"] = {"LaWAM": la[0], "LingBot-VA": 98.5, "Cosmos": 97.6, "Motus": 97.6,
                        "note": "LIBERO-Long 최고는 LingBot-VA 98.5, LaWAM 97.0은 4위권"}
ranks_long = sorted(((r[0], m) for m, r in T1.items()), reverse=True)[:5]
OUT["V1_long_top5"] = ranks_long

# ---------- V2. 지연, 파라미터, 메모리, 학습 비용 ----------
lat = {m: T1[m][5] for m in T1 if T1[m][5]}
OUT["V2_latency_ratio_vs_lawam"] = {m: round(v / 187, 2) for m, v in lat.items()}
OUT["V2_params"] = {"lawm_over_wan5b": 0.23 / 5, "reduction_pct": round(100 * (1 - 0.23 / 5), 1)}
OUT["V2_memory_ratio_fastwam_over_lawam"] = round(23.68 / 6.04, 2)
cost = {"stage1_lawm": (400, 1024, 100_000), "stage2_pretrain": (3000, 1024, 200_000),
        "libero_post": (40, 256, 25_000), "robotwin_post": (1200, 1024, 100_000)}
OUT["V2_cost_implied"] = {k: {"gpu_hours": h, "samples_per_gpu_sec": round(b * s / (h * 3600), 1)}
                          for k, (h, b, s) in cost.items()}
OUT["V2_cost_total_gpu_hours"] = {"pretrain_stage1_plus_2": 3400, "plus_robotwin": 4600, "plus_libero": 3440}
OUT["V2_wallclock_hours"] = {"stage1_16gpu": 400 / 16, "stage2_64gpu": 3000 / 64, "libero_8gpu": 40 / 8}


# ---------- V3. LaWM 디코더 파라미터 수 (식으로 계산) ----------
def adaln_block(d, ffn=4):
    attn = 4 * d * d + 4 * d              # nn.MultiheadAttention: in_proj 3d^2+3d, out_proj d^2+d
    f = d * ffn * d + ffn * d + ffn * d * d + d
    mod = d * 6 * d + 6 * d
    return attn + f + mod


def decoder_params(layers, d=1024, inp=768, code=32):
    blocks = layers * adaln_block(d)
    proj = (inp * d + d) + (d * inp + inp)
    return blocks + proj + 2 * inp + (code * d + d)


OUT["V3_decoder_params"] = {"12_layers(release_cfg)": decoder_params(12),
                            "24_layers(paper_E2_text)": decoder_params(24),
                            "paper_claim": "230M"}


# 공개 LAM 체크포인트(HF lawam_lam, 2,900,637,794 바이트, fp32 가정) 크기와 대조
def lam_total(dec_layers, cross_block=4.2e6):
    d = 1024
    enc_layer = (4 * d * d + 4 * d) + (d * 4096 + 4096) + (4096 * d + d) + 4 * d
    enc = 24 * enc_layer + (768 * d + d) + (d * 32 + 32) + d + cross_block
    dino = 85.7e6
    sp = (32 * 14 * 1024 + 32 * 1024) + (32 * 1024 * 1024 + 32 * 1024) + (32 * 2048 * 1024 + 32 * 1024) \
        + (32 * 1024 * 14 + 32 * 14) + (32 * 1024 + 1024)
    return enc + dino + sp + decoder_params(dec_layers) + 2500


OUT["V3_lam_checkpoint_size_check"] = {
    "hf_file_bytes": 2900637794,
    "implied_params_fp32": 2900637794 / 4,
    "estimate_12_dec_layers": lam_total(12), "estimate_24_dec_layers": lam_total(24),
    "bytes_if_12": 4 * lam_total(12), "bytes_if_24": 4 * lam_total(24),
    "note": "인코더 24층 + 디코더 12층 + DINOv3-B + 상태 예측기 추정. 교차 어텐션 블록 크기는 4.2M로 가정(최대 12.6M까지 변해도 0.3% 안쪽)."
}

# ---------- V4. 표 5 (RoboTwin) ----------
NAMES = ["Fast-WAM", "GigaWorld", "LingBot-VA", "pi0.5", "Motus", "LaWAM"]


def parse_t5(path, start_pat, end_pat="Average"):
    if not path.exists():
        return None
    lines = path.read_text().splitlines()
    rows = {}
    on = False
    for ln in lines:
        if re.search(start_pat, ln):
            on = True
            continue
        if on:
            m = re.match(r"\s*([A-Za-z0-9 ]+?)\s{2,}((?:\d+\s+){11}\d+)\s*$", ln)
            if m:
                rows[m.group(1).strip()] = list(map(int, m.group(2).split()))
            elif ln.strip().startswith(end_pat) and rows:
                break
    return rows


t5 = parse_t5(PAPER, r"^Table 5: Per-task")
rt = {}
if t5:
    cols = {}
    for i, n in enumerate(NAMES):
        cols[n + "_clean"] = [r[2 * i] for r in t5.values()]
        cols[n + "_rand"] = [r[2 * i + 1] for r in t5.values()]
    rep = {"Fast-WAM": (91.98, 90.52), "GigaWorld": (86.36, 85.04), "LingBot-VA": (91.50, 90.92),
           "pi0.5": (82.74, 76.76), "Motus": (88.66, 87.02), "LaWAM": (92.64, 89.80)}
    tab = {}
    for n in NAMES:
        c, r_ = cols[n + "_clean"], cols[n + "_rand"]
        tab[n] = {"n_tasks": len(c), "clean_mean": round(sum(c) / len(c), 3), "rand_mean": round(sum(r_) / len(r_), 3),
                  "reported_clean": rep[n][0], "reported_rand": rep[n][1],
                  "clean_match": abs(sum(c) / len(c) - rep[n][0]) < 0.006,
                  "rand_match": abs(sum(r_) / len(r_) - rep[n][1]) < 0.006,
                  "overall_mean_clean_rand": round((sum(c) + sum(r_)) / (2 * len(c)), 3),
                  "clean_successes_of_5000": sum(c), "rand_successes_of_5000": sum(r_)}
    rt["per_method"] = tab
    rt["n_rows_parsed"] = len(t5)
    rt["rank_by_overall_mean"] = sorted(((v["overall_mean_clean_rand"], k) for k, v in tab.items()), reverse=True)
    la_c, la_r = cols["LaWAM_clean"], cols["LaWAM_rand"]
    paired = {}
    random.seed(0)
    for n in NAMES[:-1]:
        for cond, lv in (("clean", la_c), ("rand", la_r)):
            bv = cols[f"{n}_{cond}"]
            d = [a - b for a, b in zip(lv, bv)]
            w = sum(x > 0 for x in d)
            l = sum(x < 0 for x in d)
            boots = sorted(sum(random.choice(d) for _ in d) / len(d) for _ in range(5000))
            kA, kB = sum(lv), sum(bv)
            paired[f"{n}_{cond}"] = {"mean_diff_pp": round(sum(d) / len(d), 3), "task_wins": w, "task_losses": l,
                                     "ties": len(d) - w - l, "sign_test_p": sign_test(w, l),
                                     "boot95_ci_pp": [round(boots[125], 3), round(boots[4874], 3)],
                                     "pooled_z_p_n5000": ztest(kA, 5000, kB, 5000)}
    rt["lawam_vs_baselines"] = paired
    # 과제별 LaWAM이 가장 약한 곳
    worst = sorted(((c_ - m_, nm) for nm, (c_, m_) in
                    {k: (r[10], max(r[0], r[2], r[4], r[6], r[8])) for k, r in t5.items()}.items()))[:6]
    rt["lawam_clean_largest_deficit_vs_best_baseline"] = [(nm, d) for d, nm in worst]
    # 표 2 발췌 행과 일치 확인
    t2_rows = {"Move Can Pot": [95, 92, 76, 78, 93, 93, 51, 55, 34, 74, 98, 93],
               "Move Stapler Pad": [84, 63, 92, 82, 59, 71, 56, 42, 83, 85, 94, 87],
               "Open Laptop": [100, 100, 96, 98, 96, 90, 90, 96, 95, 91, 100, 100],
               "Pick Dual Bottles": [100, 94, 86, 86, 100, 100, 93, 63, 96, 90, 100, 95],
               "Place A2B Left": [94, 89, 94, 88, 96, 94, 87, 82, 88, 79, 98, 91],
               "Place Container Plate": [98, 100, 98, 96, 97, 98, 99, 95, 98, 99, 100, 100],
               "Place Dual Shoes": [88, 88, 96, 84, 94, 81, 75, 75, 93, 87, 98, 94],
               "Place Object Basket": [82, 90, 90, 92, 89, 85, 80, 76, 81, 87, 92, 90],
               "Place Object Scale": [83, 88, 88, 80, 98, 87, 86, 80, 88, 85, 95, 88],
               "Place Shoe": [97, 98, 98, 96, 99, 97, 92, 93, 99, 97, 100, 100],
               "Put Bottles Dustbin": [93, 82, 72, 70, 82, 87, 84, 79, 81, 79, 94, 92],
               "Put Object Cabinet": [94, 82, 74, 74, 87, 80, 80, 79, 88, 71, 90, 82],
               "Scan Object": [96, 86, 60, 64, 90, 89, 72, 65, 67, 66, 96, 90],
               "Stack Bowls Two": [90, 96, 96, 92, 98, 99, 95, 96, 98, 98, 100, 99]}
    rt["table2_excerpt_equals_table5"] = {k: (t5.get(k) == v) for k, v in t2_rows.items()}
    # v1 표 4 와 v2 표 5 비교
    t4 = parse_t5(PAPER_V1, r"^Table 4: Per-task")
    if t4:
        diff = {k: (t4[k], t5[k]) for k in t5 if k in t4 and t4[k] != t5[k]}
        rt["v1_vs_v2_per_task_diff"] = {"n_v1_rows": len(t4), "n_changed": len(diff), "changed": diff}
OUT["V4_robotwin"] = rt

# 초록 91.22 = (92.64 + 89.80) / 2
OUT["V4_abstract_9122"] = (92.64 + 89.80) / 2

# ---------- V5. 표 3, 8 (실로봇) ----------
T3 = {  # 과제: (n, {방법: 성공 수})
    "Pick-and-Place": (30, {"pi0.5": 26, "GR00T-N1.6": 25, "Fast-WAM": 17, "LingBot-VA": 23, "LaWAM": 28}),
    "Open Drawer": (30, {"pi0.5": 24, "GR00T-N1.6": 23, "Fast-WAM": 19, "LingBot-VA": 25, "LaWAM": 26}),
    "Fold Towel": (30, {"pi0.5": 25, "GR00T-N1.6": 14, "Fast-WAM": 21, "LingBot-VA": 0, "LaWAM": 27}),
    "Fold Towel-Light": (50, {"pi0.5": 23, "Fast-WAM": 7, "LaWAM": 34}),
    "Clean Table": (50, {"pi0.5": 14, "Fast-WAM": 3, "LaWAM": 21}),
    "Clean Table-Light": (50, {"pi0.5": 5, "Fast-WAM": 0, "LaWAM": 12}),
}
T3_pct = {"Pick-and-Place": {"pi0.5": 86.7, "GR00T-N1.6": 83.3, "Fast-WAM": 56.7, "LingBot-VA": 76.7, "LaWAM": 93.3},
          "Open Drawer": {"pi0.5": 80.0, "GR00T-N1.6": 76.7, "Fast-WAM": 63.3, "LingBot-VA": 83.3, "LaWAM": 86.7},
          "Fold Towel": {"pi0.5": 83.3, "GR00T-N1.6": 46.7, "Fast-WAM": 70.0, "LingBot-VA": 0.0, "LaWAM": 90.0},
          "Fold Towel-Light": {"pi0.5": 46, "Fast-WAM": 14, "LaWAM": 68},
          "Clean Table": {"pi0.5": 28, "Fast-WAM": 6, "LaWAM": 42},
          "Clean Table-Light": {"pi0.5": 10, "Fast-WAM": 0, "LaWAM": 24}}
real = {"pct_match": {}, "wilson": {}, "fisher_lawam_vs": {}}
for t, (n, d) in T3.items():
    for m, k in d.items():
        real["pct_match"][f"{t}|{m}"] = abs(100 * k / n - T3_pct[t][m]) <= 0.05
        real["wilson"][f"{t}|{m}"] = {"k": k, "n": n, "pct": round(100 * k / n, 2),
                                      "ci": [round(x, 1) for x in wilson(k, n)]}
    for m, k in d.items():
        if m != "LaWAM":
            real["fisher_lawam_vs"][f"{t}|{m}"] = fisher(d["LaWAM"], n, k, n)
# 표 8과 비교
T8 = {"Fold Towel-Light": {"pi0.5": (46, 33.0, 59.6), "Fast-WAM": (14, 7.0, 26.2), "LaWAM": (68, 54.2, 79.2)},
      "Clean Table": {"pi0.5": (28, 17.5, 41.7), "Fast-WAM": (6, 2.1, 16.2), "LaWAM": (42, 29.4, 55.8)},
      "Clean Table-Light": {"pi0.5": (10, 4.3, 21.4), "Fast-WAM": (0, 0.0, 7.1), "LaWAM": (24, 14.3, 37.4)}}
t8chk = {}
for t, d in T8.items():
    for m, (pct, lo, hi) in d.items():
        k = T3[t][1][m]
        a, b = wilson(k, 50)
        t8chk[f"{t}|{m}"] = {"reported": [lo, hi], "recomputed": [round(a, 1), round(b, 1)],
                             "match": abs(a - lo) <= 0.11 and abs(b - hi) <= 0.11}
real["table8_check"] = t8chk
vs_pi = [(k, p) for k, p in real["fisher_lawam_vs"].items() if k.endswith("|pi0.5")]
hp = holm([p for _, p in vs_pi])
real["lawam_vs_pi05_holm_over_6"] = {k: {"p": p, "holm": h} for (k, p), h in zip(vs_pi, hp)}
# 합산: 30회 3과제(90회), 50회 3조건(150회)
agg = {}
for m in ["pi0.5", "Fast-WAM", "LaWAM"]:
    k30 = sum(T3[t][1][m] for t in ["Pick-and-Place", "Open Drawer", "Fold Towel"])
    k50 = sum(T3[t][1][m] for t in ["Fold Towel-Light", "Clean Table", "Clean Table-Light"])
    agg[m] = {"k90": k30, "k150": k50}
agg["fisher_lawam_vs_pi05_n90"] = fisher(agg["LaWAM"]["k90"], 90, agg["pi0.5"]["k90"], 90)
agg["fisher_lawam_vs_pi05_n150"] = fisher(agg["LaWAM"]["k150"], 150, agg["pi0.5"]["k150"], 150)
agg["fisher_lawam_vs_pi05_n240"] = fisher(agg["LaWAM"]["k90"] + agg["LaWAM"]["k150"], 240,
                                          agg["pi0.5"]["k90"] + agg["pi0.5"]["k150"], 240)
real["aggregate"] = agg
real["mde_pp"] = {"n30_each": mde(30, 30, 0.8), "n50_each": mde(50, 50, 0.35)}
# 조명 유지율
real["light_retention"] = {"Fold Towel (Light/normal)": {m: round(100 * (T3[ 'Fold Towel-Light'][1][m] / 50) / (T3['Fold Towel'][1][m] / 30), 1)
                                                      for m in ["pi0.5", "Fast-WAM", "LaWAM"]},
                           "Clean Table (Light/normal)": {m: (round(100 * T3['Clean Table-Light'][1][m] / T3['Clean Table'][1][m], 1))
                                                          for m in ["pi0.5", "Fast-WAM", "LaWAM"]}}
# v1 표 3 평균열
v1avg = {"pi0.5": ((86.7, 80.0, 83.3), 83.3), "GR00T-N1.6": ((83.3, 76.7, 46.7), 68.9), "Fast-WAM": ((56.7, 63.3, 70.0), 63.3),
         "LingBot-VA": ((76.7, 83.3, 0.0), 53.3), "LaWAM": ((93.3, 86.7, 90.0), 90.0)}
real["v1_avg_col"] = {m: {"recomputed": round(sum(v) / 3, 2), "reported": r} for m, (v, r) in v1avg.items()}
OUT["V5_real_world"] = real

# ---------- V6. 표 4, 6 (LIBERO-10 절제) ----------
T4 = {"Full LaWAM": 97.0, "w/o Stage-2 Pretraining": 96.4, "w/o KI": 94.2, "w/o LA Distillation": 93.4,
      "w/o KI & LA Distillation": 93.2, "w/o Subgoal Conditioning": 94.6, "Future Prediction": 93.4,
      "LA Distillation Only": 95.2, "Base VLA": 91.0, "Direct Future Prediction": 94.4,
      "Capacity-Matched Adapter": 91.4}
ab = {"counts_of_500": {k: v * 5 for k, v in T4.items()},
      "integer_ok": {k: abs(v * 5 - round(v * 5)) < 1e-6 for k, v in T4.items()},
      "wilson95": {k: [round(x, 1) for x in wilson(round(v * 5), 500)] for k, v in T4.items()}}
pairs = [("Full LaWAM", "w/o Stage-2 Pretraining"), ("w/o Stage-2 Pretraining", "w/o KI"),
         ("w/o Stage-2 Pretraining", "w/o LA Distillation"), ("w/o Stage-2 Pretraining", "w/o KI & LA Distillation"),
         ("w/o Stage-2 Pretraining", "w/o Subgoal Conditioning"), ("w/o Stage-2 Pretraining", "Future Prediction"),
         ("w/o Stage-2 Pretraining", "LA Distillation Only"), ("w/o Stage-2 Pretraining", "Base VLA"),
         ("w/o Stage-2 Pretraining", "Direct Future Prediction"), ("w/o Stage-2 Pretraining", "Capacity-Matched Adapter"),
         ("Capacity-Matched Adapter", "Base VLA"), ("w/o LA Distillation", "w/o KI & LA Distillation"),
         ("w/o Subgoal Conditioning", "LA Distillation Only"), ("w/o Subgoal Conditioning", "w/o LA Distillation"),
         ("LA Distillation Only", "w/o Stage-2 Pretraining"), ("Full LaWAM", "Base VLA")]
ab["pairs"] = {}
ps = []
for a, b in pairs:
    ka, kb = round(T4[a] * 5), round(T4[b] * 5)
    p = fisher(ka, 500, kb, 500)
    ab["pairs"][f"{a} vs {b}"] = {"diff_pp": round(T4[a] - T4[b], 1), "fisher_p": p}
ab["mde_pp_500_vs_500_p0.95"] = mde(500, 500, 0.95)
ab["mde_pp_500_vs_500_p0.93"] = mde(500, 500, 0.93)
ab["ki_effect_with_LA"] = round(T4["w/o Stage-2 Pretraining"] - T4["w/o KI"], 1)
ab["ki_effect_without_LA"] = round(T4["w/o LA Distillation"] - T4["w/o KI & LA Distillation"], 1)
OUT["V6_ablation"] = ab

# ---------- 판본 비교용 (v1 표 1 == v2 표 1?) ----------
if PAPER_V1.exists() and PAPER.exists():
    def t1_rows(p):
        out = {}
        for ln in p.read_text().splitlines():
            m = re.match(r"\s*(OpenVLA-OFT|π0\.5|π0|GR00T-N1\.6|LAPA|UniVLA|Mantis|VLA-JEPA|F1|Motus|Cosmos-Policy|LingBot-VA|Fast-WAM|LaWAM)\b.*?\s((?:\d+\.?\d*\s+){4,6}\d+\.?\d*)\s*$", ln)
            if m and m.group(1) not in out:
                out[m.group(1)] = m.group(2).split()
        return out
    a, b = t1_rows(PAPER_V1), t1_rows(PAPER)
    OUT["V7_table1_v1_vs_v2"] = {"rows_v1": len(a), "rows_v2": len(b),
                                  "identical_numbers": {k: (a.get(k, [])[-5:] == b.get(k, [])[-5:]) for k in b}}

(HERE / "recompute_results.json").write_text(json.dumps(OUT, ensure_ascii=False, indent=1, default=str))
print("saved", HERE / "recompute_results.json")
