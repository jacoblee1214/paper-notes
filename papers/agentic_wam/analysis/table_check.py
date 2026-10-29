"""AW-0 (Agentic World Action Models) 표·그림 수치 재계산 (CPU, 수 초).

- V1: Table 11(50과제 × 6방법 × 2설정)을 paper/paper.txt에서 읽어 평균을 다시 내고 Table 1·Fig. 4와 대조
- V2: Table 11 값이 50회 시행 격자(2% 단위)에 있는지
- V3: 본문 수치 주장(§3.1–3.3, 부록 B.1, J.2)과 그림 평균(Fig. 5·6·7, Table 3·4)
- V4: Table 12(채널별 시뮬레이션 step 비율)가 Table 1 성공률과 내부적으로 맞는지
- V5: 실제 로봇 점수가 루브릭 점수 격자(48회 × 최소 배점)에 있는지
- S1: RoboTwin 과제 단위 비교(부호 검정, 과제별 Fisher)
- S2: 실제 로봇 비교의 표본 오차 상한(점수 ∈ [0,1]이면 분산 ≤ m(1−m))
- S3: 코드 전문가 효과 분해(w/o code → Subgoal → Code)
- L1: RoboTwin 공식 리더보드(paper/site/robotwin_leaderboard.json)의 π0.5와 논문 재현 π0.5 대조, AW-0 순위

실행: python3 analysis/table_check.py  → analysis/table_check_results.json
"""
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
TXT = (ROOT / "paper" / "paper.txt").read_text()
METHODS = ["pi05", "EgoScale", "DreamZero", "WLA", "AW-0", "AW-0 hybrid"]
REAL_TASKS = ["Arrange Fruits", "Pack Backpack", "Replace Lightbulb", "Fold Towels",
              "Fruits w/ Tong", "Deal Poker", "Flip Scoreboard", "Solve Cube"]

# Table 1 (본문)
TABLE1 = {"pi05": (54.9, 27.7, 41.3), "DreamZero": (33.6, 17.6, 25.6), "EgoScale": (26.2, 2.9, 14.5),
          "WLA": (64.8, 3.6, 34.2), "AW-0": (80.8, 20.6, 50.7), "AW-0 hybrid": (63.1, 29.2, 46.2)}
# Table 2 (실제 로봇, 과제 순서는 REAL_TASKS, 마지막은 Avg.)
TABLE2 = {"pi05": [70.8, 49.7, 63.8, 81.7, 7.3, 13.1, 26.3, 36.1, 43.6],
          "DreamZero": [53.3, 36.7, 67.4, 79.8, 22.6, 37.1, 28.4, 42.5, 46.0],
          "EgoScale": [37.5, 27.3, 60.2, 53.3, 17.6, 25.6, 40.8, 24.9, 35.9],
          "WLA": [39.6, 29.6, 56.0, 74.6, 14.2, 22.9, 29.7, 52.3, 39.9],
          "AW-0": [91.7, 64.6, 75.0, 84.4, 54.2, 58.3, 68.8, 72.1, 71.1]}
# 그림 값: Fig. 5·7은 막대 라벨(프로젝트 페이지 데이터와 같음), Table 3·4는 본문 표
ABLATIONS = {
    "V->C->A (Fig.5)": [31.3, 29.1, 77.1, 60.8, 25.0, 52.1, 45.8, 65.3],
    "V->A w/o code (Fig.5)": [27.1, 16.7, 47.9, 53.2, 10.4, 20.8, 39.6, 47.1],
    "Code w/o comments (Fig.7)": [80.4, 52.1, 68.6, 80.1, 45.8, 47.5, 62.5, 65.7],
    "Code w/o skills (Fig.7)": [90.2, 55.9, 65.5, 76.9, 48.9, 42.0, 63.7, 63.2],
    "Subgoal (Fig.7)": [88.3, 56.3, 72.9, 78.2, 58.3, 43.8, 52.1, 59.2],
    "VAE for code (Tab.3)": [83.3, 39.6, 66.7, 61.3, 41.7, 47.9, 50.0, 53.9],
    "AR code (Tab.4)": [81.3, 62.5, 79.2, 53.2, 20.8, 44.1, 39.3, 55.8],
}
REPORTED_ABL_AVG = {"V->C->A (Fig.5)": 48.3, "V->A w/o code (Fig.5)": 32.9, "Code w/o comments (Fig.7)": 62.8,
                    "Code w/o skills (Fig.7)": 63.3, "Subgoal (Fig.7)": 63.6, "VAE for code (Tab.3)": 55.6,
                    "AR code (Tab.4)": 54.5}
# 부록 E 루브릭의 배점 최대공약수(점수 1.0 기준). 한 시행의 점수는 이 값의 정수배
RUBRIC_GRID = [0.1, 0.05, 0.1, 0.1, 0.05, 0.1, 0.1, 0.1]
N_REAL = 48                     # 3 학습 시드 × 16 시행 (§3.2)


def check(name, ok, **kw):
    return {"claim": name, "ok": bool(ok), **kw}


# ---------------------------------------------------------------- Table 11
def parse_table11():
    block = TXT[TXT.index("Table 11: Per-task RoboTwin"):TXT.index("J.2   H YBRID")]
    rows, avg = {}, None
    for line in block.splitlines():
        s = line.strip()
        m = re.match(r"([A-Z][A-Za-z0-9 ]+?)\s+((?:\d+\s+){11}\d+)$", s)
        if m and not s.startswith("Task"):
            rows[m.group(1).strip()] = [int(x) for x in m.group(2).split()]
        m = re.match(r"Average\s+(.*)", s)
        if m:
            avg = [float(x) for x in m.group(1).split()]
    assert len(rows) == 50, len(rows)
    return rows, avg


def table11_checks(rows, avg):
    arr = np.array(list(rows.values()), dtype=float)          # [50, 12] = 6방법 × (clean, rand)
    mean = arr.mean(0)
    out = {"recomputed_avg": [round(x, 2) for x in mean], "reported_avg": avg,
           "max_abs_diff": round(float(np.abs(mean - np.array(avg)).max()), 3)}
    out["all_even (50-rollout grid)"] = bool((arr % 2 == 0).all())
    out["table1_match"] = {}
    for i, m in enumerate(METHODS):
        c, r = mean[2 * i], mean[2 * i + 1]
        t1 = TABLE1[m]
        out["table1_match"][m] = {"clean": [round(c, 2), t1[0]], "rand": [round(r, 2), t1[1]],
                                  "average": [round((c + r) / 2, 2), t1[2]],
                                  "ok": abs(c - t1[0]) < 0.06 and abs(r - t1[1]) < 0.06
                                  and abs((t1[0] + t1[1]) / 2 - t1[2]) < 0.06}
    return arr, out


# ---------------------------------------------------------------- 본문 주장
def text_claims(arr):
    c = []
    t = TABLE1
    c.append(check("§3.1 AW-0 clean − WLA clean = 16.0", abs(t["AW-0"][0] - t["WLA"][0] - 16.0) < 0.05))
    c.append(check("§3.1 hybrid retains 46% (29.2/63.1)", round(29.2 / 63.1 * 100) == 46, value=round(29.2 / 63.1 * 100, 1)))
    c.append(check("§3.1 default retains 25% (20.6/80.8)", round(20.6 / 80.8 * 100) == 25, value=round(20.6 / 80.8 * 100, 1)))
    c.append(check("§3.1 hybrid clean 17.7 below default", abs(80.8 - 63.1 - 17.7) < 0.05))
    c.append(check("§3.1 AW-0 average 50.7 is highest in Table 1",
                   max(v[2] for k, v in t.items() if k != "AW-0") < 50.7))
    for m, v in TABLE2.items():
        c.append(check(f"Table 2 Avg. {m}", abs(np.mean(v[:8]) - v[8]) < 0.06, value=round(float(np.mean(v[:8])), 2)))
    c.append(check("§3.2 +25.1 over strongest baseline (DreamZero 46.0)",
                   abs(71.1 - max(v[8] for k, v in TABLE2.items() if k != "AW-0") - 25.1) < 0.05))
    for k, v in ABLATIONS.items():
        c.append(check(f"avg {k} = {REPORTED_ABL_AVG[k]}", abs(np.mean(v) - REPORTED_ABL_AVG[k]) < 0.06,
                       value=round(float(np.mean(v)), 2)))
    ours = np.array(TABLE2["AW-0"][:8])
    wins = lambda k: int((ours > np.array(ABLATIONS[k])).sum())
    c.append(check("§3.3 ViT > VAE on all 8", wins("VAE for code (Tab.3)") == 8))
    c.append(check("§3.3 joint > AR on 7 of 8", wins("AR code (Tab.4)") == 7))
    c.append(check("§3.3 full code > w/o comments on all 8", wins("Code w/o comments (Fig.7)") == 8))
    c.append(check("§3.3 full code > w/o skills on all 8", wins("Code w/o skills (Fig.7)") == 8))
    c.append(check("§3.3 full code > subgoal on 7 of 8", wins("Subgoal (Fig.7)") == 7))
    c.append(check("§3.3 code improves over w/o code on all 8", wins("V->A w/o code (Fig.5)") == 8))
    c.append(check("Fig.5 C->V->A beats V->C->A on 7 of 8 (Replace Lightbulb 77.1 > 75.0, 본문은 명시 안 함)",
                   wins("V->C->A (Fig.5)") == 7))
    c.append(check("§3.3 latency 0.55/0.23 = 2.39x", abs(0.55 / 0.23 - 2.39) < 0.005, value=round(0.55 / 0.23, 3)))
    c.append(check("§3.3 +0.04 s = 21.1% over 0.19 s", abs(0.04 / 0.19 * 100 - 21.1) < 0.05))
    # 부록 B.1 Table 7
    c.append(check("B.1 YAM slices/episode 13.5", round(986314 / 73101, 1) == 13.5))
    c.append(check("B.1 ego slices/episode 6.2", round(1089059 / 175541, 1) == 6.2))
    c.append(check("B.1 total slices/episode 8.3", round(2075373 / 248642, 1) == 8.3, value=round(2075373 / 248642, 3)))
    c.append(check("B.1 totals add up", 73101 + 175541 == 248642 and abs(1919.30 + 993.91 - 2913.22) < 0.01
                   and 986314 + 1089059 == 2075373 and 279 + 327 == 606))
    c.append(check("A: visual tokens 3 views × (224/16 × 320/16)/4 = 210", 3 * (224 // 16) * (320 // 16) // 4 == 210))
    c.append(check("A: effective batch 6 × 48 × 9 = 2592", 6 * 48 * 9 == 2592))
    c.append(check("A: RoboTwin batch 2 nodes × 8 × 6 = 96", 2 * 8 * 6 == 96))
    c.append(check("A: 64-D = 2 × (9 + 22 + 1)", 2 * (9 + 22 + 1) == 64))
    c.append(check("A: 9 frames at stride 4 cover the 32-step action horizon", (9 - 1) * 4 == 32))
    return c


# ---------------------------------------------------------------- Table 12
def table12_checks():
    t12 = {"clean": {"sr": 0.631, "all": (8.5, 26720), "succ": (14.5, 11549), "fail": (6.3, 52687)},
           "random": {"sr": 0.292, "all": (5.6, 41716), "succ": (35.0, 8400), "fail": (3.7, 55483)}}
    out = {}
    for k, d in t12.items():
        p = d["sr"]
        steps = p * d["succ"][1] + (1 - p) * d["fail"][1]
        code = (p * d["succ"][1] * d["succ"][0] + (1 - p) * d["fail"][1] * d["fail"][0]) / steps
        # 성공률을 모를 때: All steps/ep에서 역산한 성공 비율
        p_inv = (d["fail"][1] - d["all"][1]) / (d["fail"][1] - d["succ"][1])
        out[k] = {"steps_per_ep_recomputed": round(steps), "reported": d["all"][1],
                  "code_share_recomputed": round(code, 2), "reported_share": d["all"][0],
                  "success_rate_implied_by_steps": round(p_inv * 100, 2), "table1_success": p * 100,
                  "fail_to_success_step_ratio": round(d["fail"][1] / d["succ"][1], 2)}
    return out


# ---------------------------------------------------------------- 루브릭 격자
def rubric_grid():
    rows = {"AW-0 (Tab.2)": TABLE2["AW-0"][:8]}
    for k in ("pi05", "DreamZero", "EgoScale", "WLA"):
        rows[f"{k} (Tab.2)"] = TABLE2[k][:8]
    rows.update(ABLATIONS)
    res, n_on, n_tot = {}, 0, 0
    tol = 0.05                                                 # 표기 반올림 ±0.05%p
    for k, vals in rows.items():
        r = []
        for v, g in zip(vals, RUBRIC_GRID):
            step = g / N_REAL * 100                            # 평균 점수의 최소 단위(%p)
            q = v / step
            on = abs(q - round(q)) * step <= tol + 1e-9
            r.append({"value": v, "grid_pp": round(step, 4), "k": round(q, 2), "on_grid": bool(on)})
            if g == 0.1:
                n_on += on
                n_tot += 1
        res[k] = r
    # 0.1 배점 과제는 격자 간격 0.2083%p에 허용폭 ±0.05 → 무작위 값이라면 약 48%만 격자에 걸림
    p_random = 2 * tol / (0.1 / N_REAL * 100)
    binom = stats.binomtest(n_on, n_tot, p_random, alternative="greater").pvalue
    hybrid = 90.6 / (0.1 / N_REAL * 100)
    return {"per_row": res, "grid_0.1_tasks_on_grid": f"{n_on}/{n_tot}", "p_if_random": round(p_random, 3),
            "binom_p_greater": float(binom),
            "hybrid_fold_90.6_k": round(hybrid, 2)}


# ---------------------------------------------------------------- 통계
def robotwin_stats(arr):
    """과제 단위 비교. 열: 0..11 = (pi05 c,r, Ego c,r, DZ c,r, WLA c,r, AW c,r, hyb c,r)."""
    col = {f"{m}|{s}": 2 * i + j for i, m in enumerate(METHODS) for j, s in enumerate(["clean", "rand"])}
    out = {}

    def compare(a, b):
        x, y = arr[:, col[a]], arr[:, col[b]]
        d = x - y
        w, l = int((d > 0).sum()), int((d < 0).sum())
        sign_p = stats.binomtest(w, w + l, 0.5).pvalue if w + l else 1.0
        wil = stats.wilcoxon(x, y).pvalue if (d != 0).any() else 1.0
        # 과제별 Fisher (50 vs 50 시행, 성공 횟수 = 성공률/2)
        pos = neg = 0
        for xi, yi in zip(x, y):
            k1, k2 = int(xi) // 2, int(yi) // 2
            if stats.fisher_exact([[k1, 50 - k1], [k2, 50 - k2]]).pvalue < 0.05:
                pos += int(xi > yi)
                neg += int(xi < yi)
        return {"mean_diff": round(float(d.mean()), 2), "wins": w, "losses": l, "ties": int((d == 0).sum()),
                "sign_p": float(sign_p), "wilcoxon_p": float(wil),
                "tasks_sig_better(Fisher p<.05)": pos, "tasks_sig_worse": neg}

    out["AW-0 vs WLA clean"] = compare("AW-0|clean", "WLA|clean")
    out["AW-0 vs pi05 clean"] = compare("AW-0|clean", "pi05|clean")
    out["AW-0 vs pi05 rand"] = compare("AW-0|rand", "pi05|rand")
    out["hybrid vs AW-0 rand"] = compare("AW-0 hybrid|rand", "AW-0|rand")
    out["hybrid vs pi05 rand"] = compare("AW-0 hybrid|rand", "pi05|rand")
    out["hybrid vs AW-0 clean"] = compare("AW-0 hybrid|clean", "AW-0|clean")
    # hybrid의 random 이득이 어느 과제에 몰려 있는가
    d = arr[:, col["AW-0 hybrid|rand"]] - arr[:, col["AW-0|rand"]]
    names = list(ROWS.keys())
    order = np.argsort(-d)
    out["hybrid-rand gain top5"] = [(names[i], int(d[i])) for i in order[:5]]
    out["hybrid-rand gain share of top5"] = round(float(d[order[:5]].sum() / d.sum()), 2)
    return out


def se_upper(m, n=N_REAL):
    """점수 ∈ [0,1], 평균 m이면 분산 ≤ m(1−m). 시드 간 변동은 포함하지 않는 하한 쪽 근사."""
    m = m / 100
    return math.sqrt(max(m * (1 - m), 1e-9) / n) * 100


def real_stats():
    ours = TABLE2["AW-0"][:8]
    out = {}

    def comp(name, a, b, n_a=N_REAL, n_b=N_REAL):
        per = []
        for x, y in zip(a, b):
            se = math.sqrt(se_upper(x, n_a) ** 2 + se_upper(y, n_b) ** 2)
            per.append(round((x - y) / se, 2))
        avg_se = math.sqrt(sum(se_upper(x, n_a) ** 2 for x in a) / 64 + sum(se_upper(y, n_b) ** 2 for y in b) / 64)
        return {"z_upper_per_task": per, "n_tasks_|z|>=1.96": int(sum(abs(z) >= 1.96 for z in per)),
                "avg_diff": round(float(np.mean(a) - np.mean(b)), 2), "avg_z_upper": round((np.mean(a) - np.mean(b)) / avg_se, 2)}

    out["AW-0 vs DreamZero"] = comp("dz", ours, TABLE2["DreamZero"][:8])
    out["AW-0 vs pi05"] = comp("pi", ours, TABLE2["pi05"][:8])
    for k in ("Subgoal (Fig.7)", "Code w/o comments (Fig.7)", "Code w/o skills (Fig.7)", "AR code (Tab.4)",
              "VAE for code (Tab.3)", "V->C->A (Fig.5)", "V->A w/o code (Fig.5)"):
        out[f"Code vs {k}"] = comp(k, ours, ABLATIONS[k])
    out["Subgoal vs w/o code"] = comp("sg", ABLATIONS["Subgoal (Fig.7)"], ABLATIONS["V->A w/o code (Fig.5)"])
    # 실제 로봇 하이브리드: Fold 하나, 시행 수 미기재 → 16회와 48회 두 경우
    out["hybrid Fold 90.6 vs 84.4"] = {f"n={n}": round((90.6 - 84.4) / math.sqrt(se_upper(90.6, n) ** 2 + se_upper(84.4, n) ** 2), 2)
                                       for n in (16, 48)}
    out["note"] = ("z_upper = 차이 / (분산 상한으로 만든 SE). 루브릭 점수는 이진보다 분산이 작을 수 있어 z는 커질 수 있고, "
                   "학습 시드 3개의 변동을 넣으면 작아진다. 시드별 값이 없어 둘 다 확인할 수 없다.")
    return out


def decomposition():
    w = np.mean(ABLATIONS["V->A w/o code (Fig.5)"])
    sg = np.mean(ABLATIONS["Subgoal (Fig.7)"])
    code = np.mean(TABLE2["AW-0"][:8])
    vae = np.mean(ABLATIONS["VAE for code (Tab.3)"])
    return {"w/o code": round(w, 2), "subgoal (NL plan + ViT)": round(sg, 2), "full code": round(code, 2),
            "gain_total": round(code - w, 2), "gain_plan_stream(subgoal - w/o)": round(sg - w, 2),
            "gain_code_over_NL(code - subgoal)": round(code - sg, 2),
            "share_code_over_NL": round((code - sg) / (code - w), 3),
            "code_with_VAE_input": round(vae, 2),
            "best_baseline_avg(DreamZero)": 46.0,
            "subgoal_beats_best_baseline": bool(sg > 46.0),
            "note": "w/o code 변형은 Qwen3-VL ViT 경로도 함께 빠진다(build.py: code_vision은 use_code_expert일 때만)."}


# ---------------------------------------------------------------- 리더보드
def leaderboard(arr):
    d = json.loads((ROOT / "paper" / "site" / "robotwin_leaderboard.json").read_text())
    tasks = d["tasks"]
    names = list(ROWS.keys())
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    idx = [ [norm(t) for t in tasks].index(norm(n)) for n in names]
    pi = next(e for e in d["entries"] if e["id"] == "cotrain:Pi_05")
    lb_c = np.array(pi["easy"])[idx]
    lb_r = np.array(pi["hard"])[idx]
    pc, pr = arr[:, 0], arr[:, 1]
    out = {"leaderboard_updated": d["board"]["updated"], "protocol": d["board"]["eval_protocol"],
           "train_data": d["board"]["train_data"],
           "pi05_leaderboard": {"clean": pi["easy_mean"], "rand": pi["hard_mean"], "listed_at": pi["listed_at"],
                                "avg": round((pi["easy_mean"] + pi["hard_mean"]) / 2, 2)},
           "pi05_paper": {"clean": round(float(pc.mean()), 2), "rand": round(float(pr.mean()), 2)},
           "per_task_corr_clean": round(float(np.corrcoef(lb_c, pc)[0, 1]), 3),
           "per_task_corr_rand": round(float(np.corrcoef(lb_r, pr)[0, 1]), 3),
           "tasks_paper_lower_clean": int((pc < lb_c).sum()), "tasks_paper_lower_rand": int((pr < lb_r).sum()),
           "wilcoxon_clean_p": float(stats.wilcoxon(pc, lb_c).pvalue),
           "wilcoxon_rand_p": float(stats.wilcoxon(pr, lb_r).pvalue)}
    aw = TABLE1["AW-0"]
    hy = TABLE1["AW-0 hybrid"]
    out["AW-0 vs leaderboard pi05"] = {"clean": round(aw[0] - pi["easy_mean"], 2), "rand": round(aw[1] - pi["hard_mean"], 2),
                                       "avg": round(aw[2] - out["pi05_leaderboard"]["avg"], 2),
                                       "hybrid_rand": round(hy[1] - pi["hard_mean"], 2)}
    # 논문 공개일(2026-10-21) 이전 등록 항목 중 순위 (리더보드 기본 순위 = (clean + rand)/2)
    cut = "26.10.21"
    prior = [e for e in d["entries"] if e["listed_at"] <= cut]
    key = lambda e: (e["easy_mean"] + e["hard_mean"]) / 2
    rank = lambda v, k: 1 + sum(1 for e in prior if k(e) > v)
    out["entries_listed_by_2026-10-21"] = len(prior)
    out["AW-0_rank_avg"] = rank(aw[2], key)
    out["AW-0_rank_clean"] = rank(aw[0], lambda e: e["easy_mean"])
    out["hybrid_rank_rand"] = rank(hy[1], lambda e: e["hard_mean"])
    out["entries_above_AW-0_clean"] = sorted([(e["name"], e["easy_mean"]) for e in prior if e["easy_mean"] > aw[0]],
                                             key=lambda x: -x[1])
    out["note"] = "리더보드는 과제당 100회, 논문은 50회. π0.5 리더보드 값은 RoboTwin 팀이 같은 학습 데이터(50 clean × 50과제)로 낸 co-train 결과"
    return out, {"lb_clean": lb_c.tolist(), "lb_rand": lb_r.tolist()}


if __name__ == "__main__":
    ROWS, AVG = parse_table11()
    arr, t11 = table11_checks(ROWS, AVG)
    claims = text_claims(arr)
    lb, lb_raw = leaderboard(arr)
    res = {"V1_V2_table11": t11, "V3_claims": claims, "V4_table12": table12_checks(), "V5_rubric_grid": rubric_grid(),
           "S1_robotwin": robotwin_stats(arr), "S2_real": real_stats(), "S3_decomposition": decomposition(),
           "L1_leaderboard": lb,
           "_table11": {k: v for k, v in ROWS.items()}, "_leaderboard_pi05_per_task": lb_raw}
    (ROOT / "analysis" / "table_check_results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=lambda o: o.item()))
    bad = [c["claim"] for c in claims if not c["ok"]]
    print("Table 11 avg max diff:", t11["max_abs_diff"], "| even grid:", t11["all_even (50-rollout grid)"])
    print("Table 1 match:", all(v["ok"] for v in t11["table1_match"].values()))
    print(f"claims OK {len(claims) - len(bad)}/{len(claims)}; failing: {bad}")
    print("Table 12:", json.dumps(res["V4_table12"], ensure_ascii=False))
    rg = res["V5_rubric_grid"]
    print("rubric grid (0.1 tasks):", rg["grid_0.1_tasks_on_grid"], "p_random", rg["p_if_random"], "binom p", round(rg["binom_p_greater"], 4))
    for k, v in res["S1_robotwin"].items():
        print("S1", k, v)
    for k, v in res["S2_real"].items():
        print("S2", k, v)
    print("S3", res["S3_decomposition"])
    print("L1", json.dumps(lb, ensure_ascii=False, indent=0))
