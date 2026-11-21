"""논문 표 수치를 다시 계산하고 일관성과 검정을 확인한다 (CPU, 수 초).

입력: paper/paper.txt (pdftotext -layout 결과). 표 11, 12, 13, 26은 텍스트에서 파싱하고,
      표 1, 2, 14, 18과 본문 수치는 아래 상수에 옮겨 적었다(출처 표기).
출력: analysis/recompute_results.json, 콘솔 요약.
검정: 시드 5개의 평균과 표준편차(보고값)로 Welch t 검정, Holm 보정. 시드 간 변동만 반영하며
      검증 클립(1,697 / 1,365)과 참가자(22명)의 표집 변동은 반영하지 않는다.
"""
import json
import math
import re
import statistics
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
TXT = (ROOT / "paper" / "paper.txt").read_text(encoding="utf-8").replace("−", "-").split("\n")
NUM = re.compile(r"(-?\d*\.\d+|-?\d+)(?:\s*±\s*(\d*\.\d+|\d+))?")
METHODS5 = ["GaitDynamics", "Fixed-PD", "Torque", "Muscle", "ImpedanceMimic"]
ACTS = ["walk", "run", "stand", "lateral", "flexion", "jump", "Mean"]
COLS14 = ["Ep", "q", "qdot", "px", "kinr", "FZ", "FZr", "FMLr", "FAPr", "CoP_ML", "CoP_AP", "tau", "tau_r", "taudot"]
COLS13 = ["Ep", "q", "qdot", "px", "kinr", "FZ", "FZr", "FMLr", "FAPr", "CoP_ML", "CoP_AP", "taudot"]
RESULTS = {}
CHECKS = []


def check(label, ok, detail):
    CHECKS.append({"id": label, "ok": bool(ok), "detail": detail})


def parse_rows(a, b, methods, col_win=70):
    out = []
    for line in TXT[a:b]:
        for m in methods:
            i = line.find(m)
            if 0 <= i < col_win:
                rest = line[i + len(m):].replace("∗", "").replace("†", "")
                toks = [(float(x.group(1)), float(x.group(2)) if x.group(2) else None) for x in NUM.finditer(rest)]
                if len(toks) >= 5:  # 각주 줄 제외
                    out.append((m, toks))
                break
    return out


def blocks(rows, methods_per_block, n_blocks=7):
    """활동별 블록을 순서대로 자른다. 블록 안 방법 이름으로 사전을 만든다."""
    res, i = {}, 0
    for k in range(n_blocks):
        blk = rows[i:i + methods_per_block[k]] if isinstance(methods_per_block, list) else rows[i:i + methods_per_block]
        i += len(blk)
        res[ACTS[k]] = {m: t for m, t in blk}
    return res


# 표 11 (seen): 줄 1788-1840, 표 12 (unseen): 줄 1843-1900 (1-indexed), 방법 5개 블록(GaitDynamics 포함), 평균 블록은 4개
T11 = blocks(parse_rows(1787, 1840, METHODS5), [5, 5, 5, 5, 5, 5, 5])
T12 = blocks(parse_rows(1842, 1900, METHODS5), [5, 5, 5, 5, 5, 5, 5])
T26 = blocks(parse_rows(3102, 3152, ["Fixed-PD", "Torque", "Muscle", "ImpedanceMimic"]), 4)
T13 = blocks(parse_rows(1983, 2005, ["Fixed-PD", "Torque", "ImpedanceMimic"]), 3)
T13 = blocks(parse_rows(1983, 2005, ["Fixed-PD", "Torque", "ImpedanceMimic"]), 3)
for name, T, cols in [("T11_seen", T11, COLS14), ("T12_unseen", T12, COLS14), ("T26_no_grf", T26, COLS14), ("T13_G1", T13, COLS13)]:
    pass

# ---- V1: 평균 행 = 여섯 활동 평균 (반올림 범위 안인지)
def mean_row_check(T, cols, label, tol_abs=None):
    worst = {}
    for meth in [m for m in T["Mean"] if m != "GaitDynamics"]:
        dev = {}
        for ci, c in enumerate(cols):
            vals = [T[a][meth][ci][0] for a in ACTS[:6] if meth in T[a] and len(T[a][meth]) > ci]
            if len(vals) < 6:
                continue
            dev[c] = round(statistics.mean(vals) - T["Mean"][meth][ci][0], 3)
        worst[meth] = dev
    # 반올림 기준: 소수 둘째 자리 값은 0.01, 정수 값은 0.5, 그 밖은 0.5 (정수 단위 열 q̇, CoP, τ̇ 등)
    bad = []
    for meth, dev in worst.items():
        for c, d in dev.items():
            if c in ("Ep", "qdot", "CoP_ML", "CoP_AP", "taudot") and abs(d) > 0.5 + 0.05:
                bad.append((meth, c, d))
            if c in ("q", "FZr", "FMLr", "FAPr", "kinr", "FZ", "tau", "tau_r", "px") and abs(d) > 0.011 + 0.0:
                bad.append((meth, c, d))
    check(f"V1-{label}", not bad, f"평균 행과 활동 평균의 차이가 반올림 범위를 넘는 항목: {bad}" if bad else "모든 열이 반올림 범위 안")
    RESULTS.setdefault("V1_mean_row_deviation", {})[label] = worst


mean_row_check(T11, COLS14, "T11_seen")
mean_row_check(T12, COLS14, "T12_unseen")
mean_row_check(T26, COLS14, "T26_no_grf")

# ---- V2: 표 1의 평균 열과 표 11, 12 평균 행의 대응
TABLE1 = {  # 표 1 평균 열: (Ep, q, FZr, FMLr)
    "seen": {"Fixed-PD": (93.0, 2.91, .73, .53), "Torque": (95.4, 4.48, .82, .62), "Muscle": (84.8, 7.18, .77, .62), "ImpedanceMimic": (96.9, 2.28, .80, .70)},
    "unseen": {"Fixed-PD": (91.5, 3.13, .70, .45), "Torque": (79.2, 5.01, .79, .53), "Muscle": (75.3, 8.65, .74, .62), "ImpedanceMimic": (98.1, 2.45, .76, .65)},
}
for split, T, cols in [("seen", T11, COLS14), ("unseen", T12, COLS14)]:
    bad = []
    for meth, vals in TABLE1[split].items():
        for lab, c, v in zip(["Ep", "q", "FZr", "FMLr"], ["Ep", "q", "FZr", "FMLr"], vals):
            rep = T["Mean"][meth][cols.index(c)][0]
            if abs(rep - v) > (0.051 if lab == "Ep" else 0.0051 if lab in ("FZr", "FMLr") else 0.0051 * 10 if lab == "q" else 0.01):
                bad.append((meth, lab, v, rep))
    check(f"V2-table1-{split}", not bad, f"표 1 ↔ 표 {'11' if split=='seen' else '12'} 평균 행 불일치: {bad}" if bad else f"표 1 평균 열과 일치")

# ---- V3: 평균 행, 시드 5개의 Welch 검정 (Holm 보정)
N_SEEDS = 5
TCRIT = stats.t.ppf(0.975, N_SEEDS - 1)
FLOOR = {"Ep": 0.05, "q": 0.005, "FZr": 0.005, "FMLr": 0.005, "FAPr": 0.005, "tau_r": 0.005, "taudot": 0.05}  # 반올림 자리의 절반
for split, T, cols in [("seen", T11, COLS14), ("unseen", T12, COLS14)]:
    metrics = ["Ep", "q", "FZr", "FMLr", "FAPr", "tau_r", "taudot"]
    pvals = []
    rows = []
    for c in metrics:
        ci = cols.index(c)
        im = T["Mean"]["ImpedanceMimic"][ci]
        for m in ["Fixed-PD", "Torque", "Muscle"]:
            b = T["Mean"][m][ci]
            # 표준편차가 ±.00으로 적힌 칸은 반올림 하한(마지막 자리의 절반)으로 바닥을 깐 값도 함께 검정한다
            floor = FLOOR[c]
            im_sd_f, b_sd_f = max(im[1], floor), max(b[1], floor)
            t, p = stats.ttest_ind_from_stats(im[0], im[1], N_SEEDS, b[0], b[1], N_SEEDS, equal_var=False)
            _, p_floor = stats.ttest_ind_from_stats(im[0], im_sd_f, N_SEEDS, b[0], b_sd_f, N_SEEDS, equal_var=False)
            pvals.append(p_floor)
            rows.append({"metric": c, "baseline": m, "im_mean": im[0], "im_sd": im[1], "base_mean": b[0], "base_sd": b[1], "welch_t": float(t), "p_raw": float(p),
                         "p_sd_floor": float(p_floor), "sd_floor": floor,
                         "im_ci95": [im[0] - TCRIT * im[1] / math.sqrt(N_SEEDS), im[0] + TCRIT * im[1] / math.sqrt(N_SEEDS)]})
    # Holm
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m_ = len(pvals); adj = [0] * m_; run = 0
    for rank, i in enumerate(order):
        val = min(1.0, (m_ - rank) * pvals[i]); run = max(run, val); adj[i] = run
    for r, a in zip(rows, adj):
        r["p_holm"] = float(a)  # 바닥을 깐 sd 기준 Holm 보정 p
    RESULTS.setdefault("V3_welch_mean_row", {})[split] = rows
    n_sig = sum(1 for r in rows if r["p_holm"] < 0.05)
    nonsig = [(r["metric"], r["baseline"], round(r["p_holm"], 3)) for r in rows if r["p_holm"] >= 0.05]
    check(f"V3-welch-{split}", True, f"{split}: 21 비교 중 Holm 보정 p<0.05는 {n_sig}개(sd 바닥 적용). 보정 후 유의하지 않은 비교: {nonsig}")

# ---- V4: 표 14 (G1, 활동 평균, 시드 5개). 같은 PD 방식끼리 비교
T14 = {  # (Ep mean, sd, q mean, sd)
    "explicit": {"Fixed-PD": (56.9, 3.6), "Fixed-PD 2x": (63.3, 5.5), "ImpedanceMimic": (82.1, 4.3), "ImpedanceMimic 2x": (83.5, 3.6)},
    "stable": {"Fixed-PD": (74.5, 2.6), "Fixed-PD 2x": (71.1, 4.9), "ImpedanceMimic": (75.9, 5.0), "ImpedanceMimic 2x": (79.7, 3.5)},
    "implicit": {"Fixed-PD": (60.6, 4.0), "Fixed-PD 2x": (65.8, 2.1), "ImpedanceMimic": (75.5, 4.3), "ImpedanceMimic 2x": (77.4, 1.9)},
}
g1 = []
for scheme, d in T14.items():
    im_m, im_s = d["ImpedanceMimic"]
    for base in ["Fixed-PD", "Fixed-PD 2x"]:
        bm, bs = d[base]
        t, p = stats.ttest_ind_from_stats(im_m, im_s, N_SEEDS, bm, bs, N_SEEDS, equal_var=False)
        g1.append({"scheme": scheme, "im": "ImpedanceMimic", "baseline": base, "diff": im_m - bm, "welch_p": float(p)})
RESULTS["V4_G1_survival_welch"] = g1
stable_p = [g for g in g1 if g["scheme"] == "stable" and g["baseline"] == "Fixed-PD"][0]
check("V4-G1-stable-margin", stable_p["welch_p"] > 0.05,
      f"G1 안정 PD 생존율 차이 {stable_p['diff']:.1f}점, Welch p={stable_p['welch_p']:.2f} (시드 5개). 본문 '1.4점 이상 우위'는 통계적으로 구분되지 않음.")

# ---- V5: 표 2 (人 참가자, 생존 Ep, q, FZr, FMLr; 시드 평균 기준). 'every integration scheme' 주장 확인
T2 = {  # seen: Ep, q, FZr, FMLr ; unseen Ep, q, FZr, FMLr
    ("explicit", "Fixed-PD"): ((93.0, 2.91, .73, .53), (91.5, 3.13, .70, .45)),
    ("explicit", "Fixed-PD 2x"): ((96.2, 2.53, .72, .50), (95.9, 2.80, .68, .44)),
    ("explicit", "ImpedanceMimic"): ((96.9, 2.28, .80, .70), (98.1, 2.45, .76, .65)),
    ("explicit", "ImpedanceMimic 2x"): ((96.9, 2.30, .79, .68), (97.8, 2.47, .75, .63)),
    ("stable", "Fixed-PD"): ((93.5, 2.44, .75, .65), (92.2, 2.70, .72, .59)),
    ("stable", "Fixed-PD 2x"): ((96.6, 2.10, .75, .67), (96.2, 2.35, .71, .61)),
    ("stable", "ImpedanceMimic"): ((93.8, 2.22, .79, .70), (94.9, 2.45, .74, .64)),
    ("stable", "ImpedanceMimic 2x"): ((96.3, 2.07, .77, .68), (97.4, 2.27, .73, .63)),
    ("implicit", "Fixed-PD"): ((95.7, 2.52, .72, .61), (94.6, 2.72, .70, .55)),
    ("implicit", "Fixed-PD 2x"): ((95.8, 2.21, .67, .56), (96.5, 2.47, .65, .50)),
    ("implicit", "ImpedanceMimic"): ((97.2, 2.22, .77, .68), (97.8, 2.38, .75, .62)),
    ("implicit", "ImpedanceMimic 2x"): ((96.8, 2.16, .72, .62), (96.8, 2.33, .70, .56)),
}
v5 = []
for scheme in ["explicit", "stable", "implicit"]:
    for split_i, split in enumerate(["seen", "unseen"]):
        im_nom = T2[(scheme, "ImpedanceMimic")][split_i]
        im_2x = T2[(scheme, "ImpedanceMimic 2x")][split_i]
        fx_nom = T2[(scheme, "Fixed-PD")][split_i]
        fx_2x = T2[(scheme, "Fixed-PD 2x")][split_i]
        for name, a, b in [("matched nominal", im_nom, fx_nom), ("matched 2x", im_2x, fx_2x),
                           ("IM nominal vs Fixed 2x", im_nom, fx_2x)]:
            wins = {"Ep": a[0] > b[0], "q": a[1] < b[1], "FZr": a[2] > b[2], "FMLr": a[3] > b[3]}
            v5.append({"scheme": scheme, "split": split, "pair": name, "ImpedanceMimic_wins": wins})
RESULTS["V5_table2_claims"] = v5
matched = [x for x in v5 if x["pair"].startswith("matched")]
lost_matched = [(x["scheme"], x["split"], x["pair"], k) for x in matched for k, w in x["ImpedanceMimic_wins"].items() if not w]
check("V5-table2-matched", len(lost_matched) == 1 and lost_matched[0][:3] == ("stable", "seen", "matched 2x"),
      f"짝 맞춘 비교(같은 우선 gain)에서 IM이 지는 칸: {lost_matched} (논문 '96.3 대 96.6, 한 칸' 주장과 일치하는지 확인)")
cross = [(x["scheme"], x["split"], k) for x in v5 if x["pair"] == "IM nominal vs Fixed 2x" for k, w in x["ImpedanceMimic_wins"].items() if not w]
RESULTS["V5_cross_prior_losses"] = cross

# ---- V6: 표 18 계산 확인 (3개 실행을 GPU 한 개에 묶음)
t18 = [("Fixed-PD", 4.1, 40, 55), ("Torque", 4.0, 40, 54), ("Muscle", 5.5, 40, 73), ("PHC", 10.6, 40, 141), ("SimPoE", 13.3, 40, 177), ("ImpedanceMimic", 4.0, 40, 53)]
t18_out = []
for name, h, runs, rep in t18:
    calc = h * runs / 3
    t18_out.append({"method": name, "calc_gpu_h": round(calc, 1), "reported": rep, "per_run_gpu_h": round(h / 3, 2)})
RESULTS["V6_table18_gpu_hours"] = t18_out
tot = sum(r[3] for r in t18)
check("V6-table18-total", abs(tot - 550) < 10, f"여섯 방법 합 {tot} GPU-h (본문 '~550'). IM 실행당 {4.0/3:.2f} GPU-h (본문 '1.3').")

# ---- V7: 본문 §4.2 손실 수치
loss_fx = 98.1 - 56.9
loss_tq = 98.1 - 44.0
own_fx = 91.5 - 56.9
own_tq = 79.2 - 44.0
RESULTS["V7_robot_loss"] = {"IM_from_98.1": 98.1 - 82.1, "FixedPD_from_98.1": loss_fx, "Torque_from_98.1": loss_tq,
                             "FixedPD_from_own_human": own_fx, "Torque_from_own_human": own_tq, "IM_from_own_human": 98.1 - 82.1}
check("V7-robot-loss-wording", False, f"본문 '98.1에서 82.1로, Fixed-PD와 Torque는 약 35점 손실'은 기준이 섞임. 98.1 기준 손실은 {loss_fx:.1f}과 {loss_tq:.1f}점, 각자 인간 기준 손실은 {own_fx:.1f}과 {own_tq:.1f}점(부록 A.3.3과 일치).")

# ---- V8: 데이터 구성 (표 3, 4, 5)
clips_train = {"walk": 3298, "run": 806, "stand": 1591, "lateral": 369, "flexion": 464, "jump": 268}
nc = (clips_train["lateral"] + clips_train["flexion"] + clips_train["jump"]) / sum(clips_train.values())
check("V8-noncyclic-share", abs(nc * 100 - 16) < 0.5, f"비주기 활동(lateral, flexion, jump) 학습 클립 비율 {nc*100:.1f}% (본문 16%)")
rec = [49, 6, 10, 7, 15, 12, 16, 17, 12]; ret = [49, 6, 10, 7, 12, 11, 8, 17, 9]
check("V8-subject-retention", sum(ret) == 129, f"기록 {sum(rec)}명 중 {sum(ret)}명 유지, 제외 {sum(rec)-sum(ret)}명. Tan et al. (2022)는 16명 중 8명만 유지(50%)")
RESULTS["V8_retention"] = {"recorded": sum(rec), "retained": sum(ret), "per_dataset_retained_pct": [round(100 * r / c) for r, c in zip(ret, rec)]}

out = ROOT / "analysis" / "recompute_results.json"
out.write_text(json.dumps({"checks": CHECKS, "results": RESULTS}, ensure_ascii=False, indent=1), encoding="utf-8")
for c in CHECKS:
    print(("OK  " if c["ok"] else "FLAG") + " " + c["id"] + " :: " + c["detail"])
print("wrote", out)
