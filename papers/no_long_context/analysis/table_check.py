"""표 2–17과 Appendix B 비용 모델 재계산 (CPU, 단일 스레드). 실행: python3 analysis/table_check.py

- C1: Eq. 1의 r 값과 "3.6×·5.6×·7×", r = 0.20
- C2: 명목 비용(Nominal retail cost) = 토큰 × 공개 단가 재계산
- C3: EORC(Eq. 2–4) 재계산. K = 1000인지 1024인지 칸마다 판정
- C4: Fig. 5·6의 cost savings와 본문 범위(2.2–8.8×, 2.4–5.0×, 6.5–54.6×, 4.4–233.3×)
- C5: 실제(평탄) 단가에서 압축이 비용을 늘린 설정 수, 캐시 입력 비중
- C6: 정확도가 k/n으로 나오는지(표본 수 정합성)
- C7: "22개 설정 중 2%p 넘게 잃은 것이 5개 이하"와 정확도 범위
- C8: 차이의 유의성(비대응 두 비율 z), best-of-4 선택의 귀무 기대값(시뮬레이션)
- C9: Table 2(SAC)와 Table 9–10(오픈 모델)
- C10: Table 1(문학 작품 단어 ↔ 토큰 1.33배)
- C11: 긴 입력 기준선의 입력 길이와 컨텍스트 창(200K/272K) 비교
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PRICE = {"opus": (5.0, 0.5, 25.0), "gpt": (4.0, 0.4, 20.0)}   # 새 입력 / 캐시 입력 / 출력, $/M (Table 3 캡션)
WMAX = {"opus": 200_000, "gpt": 272_000}                         # Appendix B
R_MEAS = {"GLM 5.2": 0.2772, "Kimi K2.6": 0.1777, "Qwen 3 Coder Next": 0.1439}  # Eq. 1
R = 0.20

# ── 장기 과제 표 (Table 3–8). 행: (W, acc_opus, acc_gpt, tok_opus, tok_gpt, nom_opus, nom_gpt, eorc_opus, eorc_gpt)
# tok = (새 입력, 캐시 입력, 출력) M 토큰. Table 8은 (입력, 출력)만 있어 캐시 0으로 둠
LH = {
    "SWE-bench Verified": (500, [
        (None, 70.0, 84.4, (20.0, 420.0, 4.4), (16.4, 100.2, 3.1), 420.00, 167.68, 420.00, 167.68),
        (48, 73.6, 88.8, (20.0, 334.7, 4.5), (17.6, 94.2, 3.9), 379.85, 186.08, 130.20, 55.82),
        (32, 72.0, 85.0, (23.0, 255.9, 4.7), (17.2, 73.4, 4.3), 360.45, 184.16, 97.82, 46.04),
        (24, 74.8, 85.0, (22.0, 166.5, 4.1), (17.7, 60.4, 4.7), 295.75, 188.96, 69.71, 42.52),
        (16, 68.8, 84.6, (28.3, 127.3, 4.3), (19.7, 45.8, 5.3), 312.65, 203.12, 62.53, 40.62)]),
    "SWE-bench Multilingual": (300, [
        (None, 66.3, 79.3, (11.9, 384.1, 3.0), (7.4, 66.2, 1.6), 326.55, 88.08, 326.55, 88.08),
        (48, 68.0, 80.0, (12.8, 245.5, 2.9), (8.8, 65.6, 1.9), 259.25, 99.44, 87.56, 29.48),
        (32, 67.0, 78.7, (14.0, 147.0, 2.8), (10.0, 49.1, 2.3), 213.50, 105.64, 57.23, 26.16),
        (24, 66.0, 78.7, (15.6, 91.0, 2.7), (10.8, 40.3, 2.6), 191.00, 111.32, 44.54, 24.85),
        (16, 64.3, 77.7, (17.0, 65.4, 2.7), (12.4, 31.5, 3.1), 185.20, 124.20, 37.04, 24.84)]),
    "Terminal-Bench 2.1": (89, [
        (None, 47.2, 87.6, (3.9, 81.6, 1.1), (3.6, 30.4, 1.1), 87.80, 48.56, 87.80, 48.56),
        (48, 50.6, 88.8, (6.2, 77.5, 1.8), (3.7, 16.7, 1.3), 114.75, 47.48, 39.33, 14.24),
        (32, 52.8, 86.5, (6.7, 47.1, 1.7), (4.0, 14.1, 1.6), 99.55, 53.64, 27.02, 13.41),
        (24, 50.6, 85.4, (6.5, 34.0, 1.7), (6.8, 18.0, 2.8), 92.00, 90.40, 21.68, 20.34),
        (16, 48.3, 85.4, (7.6, 23.0, 1.6), (4.2, 10.3, 2.0), 89.50, 60.92, 17.90, 12.18)]),
    "FRAMES": (824, [
        (None, 81.0, 90.2, (27.4, 606.5, 2.7), (21.9, 177.6, 2.4), 507.75, 206.64, 507.75, 206.64),
        (48, 79.7, 88.7, (30.2, 307.6, 2.6), (27.6, 171.1, 3.5), 369.80, 248.84, 126.76, 74.65),
        (32, 79.4, 88.0, (28.7, 196.6, 2.4), (30.0, 147.7, 4.2), 301.80, 263.08, 81.90, 65.77),
        (24, 77.3, 86.9, (36.8, 173.2, 2.9), (29.4, 125.4, 4.6), 343.10, 259.76, 80.87, 58.45),
        (16, 79.6, 87.1, (39.2, 108.2, 3.1), (28.9, 105.2, 5.7), 327.60, 271.68, 65.52, 54.34)]),
    "BFCLv3": (200, [
        (None, 62.0, 39.5, (3.6, 18.9, 0.3), (1.8, 15.0, 0.1), 34.95, 15.20, 34.95, 15.20),
        (24, 64.0, 39.0, (4.6, 12.6, 0.3), (2.2, 9.5, 0.2), 36.80, 16.60, 8.64, 3.74),
        (16, 61.5, 39.0, (4.5, 8.9, 0.3), (2.0, 8.1, 0.2), 34.45, 15.24, 6.89, 3.05),
        (12, 62.5, 39.5, (4.9, 7.5, 0.4), (2.2, 6.9, 0.3), 38.25, 17.56, 7.65, 3.51)]),
    "tau2-bench": (184, [
        (None, 79.3, 73.9, (14.5, 0.0, 0.3), (11.4, 0.0, 0.2), 80.00, 49.60, 80.00, 49.60),
        (16, 79.3, 74.5, (15.1, 0.0, 0.3), (15.3, 0.0, 0.2), 83.00, 65.20, 16.60, 13.04),
        (12, 78.2, 77.2, (14.3, 0.0, 0.3), (15.2, 0.0, 0.3), 79.00, 66.80, 15.80, 13.36),
        (8, 78.8, 73.9, (13.2, 0.0, 0.3), (14.4, 0.0, 0.3), 73.50, 63.60, 14.70, 12.72)]),
}
# 장기 과제 기준선 총 길이 평균(K, Table 3–7)
LH_BASE_LEN = {"SWE-bench Verified": (48, 22), "SWE-bench Multilingual": (35, 23), "Terminal-Bench 2.1": (34, 28),
               "FRAMES": (25, 17), "BFCLv3": (15.2, 11.2)}

# ── 긴 입력 표 (Table 11–17). 정확도는 첫 지표(avg/recall/mean 등)
LI = {
    "LongBench v2": (503, [
        (None, 68.0, 70.8, (17.6, 263.4, 2.2), (9.1, 47.0, 1.2), 274.70, 79.20, 274.70, 79.20),
        (16, 67.6, 72.6, (20.4, 61.4, 2.1), (9.6, 49.5, 1.6), 185.20, 90.20, 37.04, 18.04)]),
    "LOONG": (1600, [
        (None, 70.1, 79.3, (272.0, 0.0, 0.3), (272.0, 0.0, 0.1), 1367.50, 1090.00, 1367.50, 1090.00),
        (48, 79.8, 84.6, (57.6, 232.1, 5.2), (23.7, 85.1, 6.2), 534.05, 252.84, 181.11, 75.85),
        (32, 80.2, 84.0, (54.6, 174.5, 5.5), (23.5, 69.7, 6.4), 497.75, 249.88, 134.18, 62.47),
        (16, 79.3, 82.7, (55.4, 115.2, 6.7), (31.1, 61.0, 13.7), 502.10, 422.80, 100.42, 84.56)]),
    "AA-LCR": (100, [
        (None, 65.0, 82.0, (10.4, 0.0, 0.0), (9.4, 0.0, 0.0), 52.00, 37.60, 52.00, 37.60),
        (32, 80.0, 83.0, (2.7, 17.2, 0.3), (0.9, 3.0, 0.3), 29.60, 10.80, 7.98, 2.70),
        (16, 84.0, 83.0, (2.9, 9.0, 0.3), (1.1, 3.0, 0.3), 26.50, 11.60, 5.30, 2.32)]),
    "NIAH": (125, [
        (None, 60.0, 80.0, (14.8, 0.0, 0.0), (13.5, 0.0, 0.1), 74.00, 56.00, 74.00, 56.00),
        (48, 100.0, 100.0, (0.8, 0.0, 0.0), (0.2, 0.0, 0.0), 4.00, 0.80, 1.36, 0.24)]),
    "RULER": (None, [
        (None, 62.5, 65.5, (175.5, 0.0, 0.1), (176.9, 0.0, 0.9), 880.00, 725.60, 880.00, 725.60),
        (16, 95.7, 96.4, (14.8, 17.7, 0.8), (7.5, 12.7, 0.5), 102.85, 45.08, 20.57, 9.02)]),
    "Counting-Stars": (None, [
        (None, 98.5, 99.3, (9.5, 0.0, 0.3), (9.5, 0.0, 0.3), 55.00, 44.00, 55.00, 44.00),
        (16, 99.7, 99.7, (1.6, 0.2, 0.1), (0.8, 0.5, 0.1), 10.60, 5.40, 2.12, 1.08)]),
    "BABILong": (550, [
        (None, 67.1, 70.7, (72.1, 0.0, 0.0), (69.4, 0.0, 0.3), 360.50, 283.60, 360.50, 283.60),
        (16, 92.0, 94.0, (8.3, 6.4, 0.5), (2.3, 2.3, 0.3), 57.20, 16.12, 11.44, 3.22)]),
}
# exact-match 지표(정수 개수 검사용)
EXACT = {"Counting-Stars": (192, [65.1, 76.6, 81.2, 81.2]), "LOONG perfect": (1600, [38.9, 44.4, 52.6, 53.0, 53.7, 52.1, 53.4, 51.9])}

# Table 2 (SAC). 정확도와 피크 컨텍스트(median/max, K)
T2 = {
    ("opus", "SWE-bench Verified", 500): dict(base=70.0, best_eocc=74.8, sac=70.2, peak=((32, 200), (19, 22), (8, 16))),
    ("opus", "Terminal-Bench 2.1", 89): dict(base=47.2, best_eocc=52.8, sac=51.8, peak=((27, 130), (24, 31), (9, 19))),
    ("gpt", "SWE-bench Verified", 500): dict(base=84.4, best_eocc=88.8, sac=82.4, peak=((18, 173), (22, 35), (7, 15))),
    ("gpt", "Terminal-Bench 2.1", 89): dict(base=87.6, best_eocc=88.8, sac=84.3, peak=((17, 133), (23, 40), (8, 37))),
}
# Table 9–10 (오픈 모델): base, 48K/40K, 32K/24K, 16K/12K
T9 = {"Qwen 3.6 27B": (68.6, 69.4, 68.2, 65.0), "Qwen 3 Coder Next": (53.0, 49.2, 40.4, 3.6),
      "Gemma 4 31B": (65.2, 65.0, 65.8, 58.0), "Step 3.7 Flash": (70.0, 66.8, 64.6, 56.6),
      "MiniMax M2.7": (72.6, 73.0, 73.2, 67.8)}
T10 = {"Qwen 3.6 27B": (43.8, 48.3, 49.4, 43.8), "Qwen 3 Coder Next": (29.2, 34.8, 34.8, 30.3),
       "Gemma 4 31B": (40.4, 34.8, 29.2, 33.7), "Step 3.7 Flash": (41.6, 32.6, 34.8, 28.1),
       "MiniMax M2.7": (40.4, 37.1, 38.2, 40.4)}
# Table 1 (문학 작품): 단어, 토큰 (K)
T1 = [("The Old Man and the Sea", 27, 37), ("The Great Gatsby", 48, 64), ("Qur'an", 156, 207),
      ("KJ New Testament", 181, 241), ("Great Expectations", 184, 245), ("The Lord of the Rings", 481, 640),
      ("War and Peace", 566, 753), ("KJ Old Testament", 603, 802)]


def nominal(tok, model):
    p = PRICE[model]
    return round(tok[0] * p[0] + tok[1] * p[1] + tok[2] * p[2], 4)


def factor(w_k, model, kilo):
    """Eq. 2의 단가 배수. w_k는 총 컨텍스트 한도(K), kilo는 1000 또는 1024. 기준선(None)은 1."""
    if w_k is None:
        return 1.0
    w = w_k * kilo
    lo = 16 * kilo
    return R + (1 - R) * (max(w, lo) - lo) / (WMAX[model] - lo)


def check_table(tables):
    rows = []
    for bench, (n, data) in tables.items():
        for w, ao, ag, to, tg, no, ng, eo, eg in data:
            for model, tok, nom_rep, eorc_rep in (("opus", to, no, eo), ("gpt", tg, ng, eg)):
                nom = nominal(tok, model)
                cand = {k: nom_rep * factor(w, model, k) for k in (1000, 1024)}
                best = min(cand, key=lambda k: abs(cand[k] - eorc_rep))
                rows.append(dict(bench=bench, W=w, model=model, nominal_reported=nom_rep, nominal_recomputed=nom,
                                 nominal_err=round(nom - nom_rep, 3), eorc_reported=eorc_rep,
                                 eorc_k1000=round(cand[1000], 3), eorc_k1024=round(cand[1024], 3),
                                 eorc_best_kilo=best if w not in (None,) and w > 16 else "clamp/base",
                                 eorc_err=round(cand[best] - eorc_rep, 3)))
    return rows


def savings(tables):
    out = {}
    for bench, (n, data) in tables.items():
        base = data[0]
        for w, *_rest in data[1:]:
            row = [r for r in data if r[0] == w][0]
            out[f"{bench} {w}K"] = {"opus": round(base[7] / row[7], 2), "gpt": round(base[8] / row[8], 2),
                                    "opus_nominal_ratio": round(base[5] / row[5], 3),
                                    "gpt_nominal_ratio": round(base[6] / row[6], 3)}
    return out


def k_over_n(acc, n):
    k = round(acc / 100 * n)
    return k, abs(100 * k / n - acc) <= 0.05 + 1e-9


def ztest(p1, p2, n1, n2):
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    if se == 0:
        return float("inf"), 0.0
    z = (p2 - p1) / se
    p = math.erfc(abs(z) / math.sqrt(2))
    return round(z, 2), round(p, 4)


def best_of_k_null(p, n, k=4, sims=20000, seed=0):
    """귀무(모든 설정의 참 정확도가 같음)에서 기준선 대비 '최고 설정' 차이의 분포. 각 설정과 기준선을 독립 시행으로 가정."""
    rng = np.random.default_rng(seed)
    base = rng.binomial(n, p, sims) / n
    cfg = rng.binomial(n, p, (sims, k)) / n
    best = (cfg.max(axis=1) - base) * 100
    return round(float(best.mean()), 2), round(float(np.quantile(best, 0.95)), 2)


def main():
    out = {}
    out["C1_r"] = {"measured_r": R_MEAS, "inverse": {m: round(1 / r, 2) for m, r in R_MEAS.items()},
                   "mean_r": round(float(np.mean(list(R_MEAS.values()))), 4), "used_r": R,
                   "note": "1/r = 3.6·5.6·7.0배. Qwen 3 Coder Next는 Gated DeltaNet:full attention 3:1 하이브리드인데 가장 큰 7배"}

    rows = check_table(LH) + check_table(LI)
    out["C2_C3_cost_rows"] = rows
    nom_errs = [abs(r["nominal_err"]) for r in rows]
    eorc_errs = [abs(r["eorc_err"]) for r in rows]
    kilo_use = {}
    for r in rows:
        if r["eorc_best_kilo"] in (1000, 1024):
            kilo_use.setdefault(f'{r["bench"]}/{r["model"]}', set()).add(r["eorc_best_kilo"])
    out["C2_C3_summary"] = {"rows": len(rows), "max_nominal_abs_err": max(nom_errs), "max_eorc_abs_err": max(eorc_errs),
                            "kilo_convention_by_table_model": {k: sorted(v) for k, v in kilo_use.items()},
                            "note": "명목 비용은 모두 재현. EORC는 칸마다 K=1000 또는 1024 중 하나로만 맞음(표·모델마다 섞임)"}

    sv = savings(LH)
    svi = savings(LI)
    o_lh = [v["opus"] for v in sv.values()]
    g_lh = [v["gpt"] for v in sv.values()]
    o_li = [v["opus"] for v in svi.values()]
    g_li = [v["gpt"] for v in svi.values()]
    out["C4_savings"] = {"long_horizon": sv, "long_input": svi,
                         "ranges": {"lh_opus": [min(o_lh), max(o_lh)], "lh_gpt": [min(g_lh), max(g_lh)],
                                    "li_opus": [min(o_li), max(o_li)], "li_gpt": [min(g_li), max(g_li)]},
                         "claimed": {"lh_opus": [2.2, 8.8], "lh_gpt": [2.4, 5.0], "li_opus": [6.5, 54.6], "li_gpt": [4.4, 233.3]}}

    inc = {"opus": [], "gpt": []}
    for k, v in sv.items():
        for m in ("opus", "gpt"):
            if v[f"{m}_nominal_ratio"] < 1:
                inc[m].append(k)
    cache_share = {}
    for bench, (n, data) in LH.items():
        b = data[0]
        cache_share[bench] = {"opus": round(b[3][1] * PRICE["opus"][1] / b[5], 3),
                              "gpt": round(b[4][1] * PRICE["gpt"][1] / b[6], 3)}
    out_tok = {}
    for bench, (n, data) in LH.items():
        out_tok[bench] = {"opus": [r[3][2] for r in data], "gpt": [r[4][2] for r in data]}
    out["C5_flat_price"] = {"configs_where_nominal_cost_rises": {m: len(v) for m, v in inc.items()},
                            "of_total": len(sv), "list": inc, "baseline_cached_cost_share": cache_share,
                            "output_tokens_M_base_then_configs": out_tok,
                            "note": "공개 단가(평탄)로는 GPT 22개 중 21개, Opus 22개 중 7개에서 압축이 비용을 늘림"}

    gran = []
    for bench, (n, data) in list(LH.items()) + list(LI.items()):
        if n is None:
            continue
        for r in data:
            for m, acc in (("opus", r[1]), ("gpt", r[2])):
                k, ok = k_over_n(acc, n)
                if not ok:
                    gran.append(dict(bench=bench, W=r[0], model=m, acc=acc, n=n, nearest=f"{k}/{n}={100*k/n:.2f}"))
    for (m, bench, n), d in T2.items():
        k, ok = k_over_n(d["sac"], n)
        if not ok:
            gran.append(dict(bench=bench, W="SAC", model=m, acc=d["sac"], n=n, nearest=f"{k}/{n}={100*k/n:.2f}"))
    for name, (n, vals) in EXACT.items():
        for a in vals:
            k, ok = k_over_n(a, n)
            if not ok:
                gran.append(dict(bench=name, acc=a, n=n, nearest=f"{k}/{n}={100*k/n:.3f}"))
    out["C6_granularity_mismatches"] = gran

    deltas = {"opus": [], "gpt": []}
    for bench, (n, data) in LH.items():
        for r in data[1:]:
            deltas["opus"].append((bench, r[0], round(r[1] - data[0][1], 1)))
            deltas["gpt"].append((bench, r[0], round(r[2] - data[0][2], 1)))
    out["C7_lh_deltas"] = {m: {"range": [min(d[2] for d in v), max(d[2] for d in v)],
                               "n_configs": len(v), "n_lose_more_than_2pp": sum(d[2] < -2.0 for d in v),
                               "losses_gt_2pp": [d for d in v if d[2] < -2.0]} for m, v in deltas.items()}

    sig = []
    for bench, (n, data) in list(LH.items()) + list(LI.items()):
        if n is None:
            n = {"RULER": None, "Counting-Stars": 192}.get(bench)
        if n is None:
            continue
        b = data[0]
        for r in data[1:]:
            for m, i in (("opus", 1), ("gpt", 2)):
                z, p = ztest(b[i] / 100, r[i] / 100, n, n)
                sig.append(dict(bench=bench, W=r[0], model=m, delta=round(r[i] - b[i], 1), z=z, p=p))
    lh_names = set(LH)
    out["C8_significance"] = {
        "rows": sig,
        "lh_significant_at_0.05": [s for s in sig if s["bench"] in lh_names and s["p"] < 0.05],
        "li_significant_at_0.05": [(s["bench"], s["model"], s["delta"]) for s in sig if s["bench"] not in lh_names and s["p"] < 0.05],
        "best_of_4_null_swe_verified": {"opus": best_of_k_null(0.70, 500), "gpt": best_of_k_null(0.844, 500)},
        "best_of_4_null_tb": {"opus": best_of_k_null(0.472, 89), "gpt": best_of_k_null(0.876, 89)},
        "note": "best_of_4_null = (최고 설정 − 기준선)의 귀무 평균, 95% 분위 (%p). 독립 시행 가정이라 대응 설계보다 큼"}

    t2 = {}
    for (m, bench, n), d in T2.items():
        t2[f"{m} {bench}"] = {"sac_minus_base": round(d["sac"] - d["base"], 1),
                              "best_eocc_minus_base": round(d["best_eocc"] - d["base"], 1),
                              "sac_z_p_vs_base": ztest(d["base"] / 100, d["sac"] / 100, n, n),
                              "peak_max_reduction_x": round(d["peak"][0][1] / d["peak"][2][1], 1)}
    out["C9_table2_sac"] = {"rows": t2, "gpt_tb_75_of_89": round(75 / 89 * 100, 2),
                            "opus_tb_sac_51.8_nearest": [f"{k}/89={100*k/89:.2f}" for k in (46, 47)],
                            "token_spend_or_cost_reported_for_sac": False}
    ow = {}
    for name, (t9, t10) in {k: (T9[k], T10[k]) for k in T9}.items():
        ow[name] = {"swe_delta_16K": round(t9[3] - t9[0], 1), "tb_delta_16K": round(t10[3] - t10[0], 1),
                    "swe_best": round(max(t9[1:]) - t9[0], 1), "tb_best": round(max(t10[1:]) - t10[0], 1)}
    out["C9b_open_weight"] = ow

    out["C10_table1"] = {name: {"words_K": w, "tokens_K": t, "ratio": round(t / w, 3)} for name, w, t in T1}

    out["C11_long_input_windows"] = {
        "opus_window_K": 200, "gpt_standard_window_K": 272, "gpt_max_window_K": 1050,
        "input_lengths_K": {"NIAH": [16, 500], "RULER": [256, 500], "BABILong": [64, 1000],
                            "Counting-Stars": [4, 128], "AA-LCR_avg": 100, "LongBench v2_words": [8, 2000]},
        "RULER_share_over_opus_window": 1.0,
        "note": "RULER는 모든 입력이, NIAH·BABILong은 일부 입력이 Opus 창(200K)을 넘음. 기준선이 넘친 입력을 어떻게 처리했는지는 논문에 없음"}

    mean_in = {}
    for bench, (n, data) in LI.items():
        if n:
            b = data[0]
            mean_in[bench] = {"opus_K": round((b[3][0] + b[3][1]) * 1000 / n, 1), "gpt_K": round((b[4][0] + b[4][1]) * 1000 / n, 1)}
    out["C11_long_input_windows"]["baseline_mean_input_per_task_K"] = mean_in
    out["C11_long_input_windows"]["identical_token_totals_across_models"] = [
        b for b, (n, d) in LI.items() if d[0][3][:2] == d[0][4][:2]]

    tot = {}
    for grp, tables in (("long_horizon", LH), ("long_input", LI)):
        for m, i in (("opus", 5), ("gpt", 6)):
            tot[f"{grp}_{m}"] = round(sum(r[i] for n, d in tables.values() for r in d), 2)
    tot["all"] = round(sum(tot.values()), 2)
    per_task = {}
    for bench, (n, data) in list(LH.items()) + list(LI.items()):
        if n:
            per_task[bench] = {"opus_base": round(data[0][5] / n, 3), "gpt_base": round(data[0][6] / n, 3),
                               "opus_cheapest_cfg": round(min(r[5] for r in data[1:]) / n, 3),
                               "gpt_cheapest_cfg": round(min(r[6] for r in data[1:]) / n, 3)}
    out["C12_api_cost"] = {"total_nominal_usd_tables_3_8_11_17": tot, "per_task_usd": per_task,
                           "note": "Table 2(SAC)와 오픈 모델 실험 비용은 표에 없어 빠짐. 단일 실행 기준"}

    # C13: 프롬프트 캐시가 없다면(캐시 입력도 새 입력 단가) 평탄 단가에서 압축이 싼가
    nc = {}
    for bench, (n, data) in LH.items():
        for m, ti in (("opus", 3), ("gpt", 4)):
            pr = PRICE[m]
            cost = [(r[ti][0] + r[ti][1]) * pr[0] + r[ti][2] * pr[2] for r in data]
            for r, c in zip(data[1:], cost[1:]):
                nc[f"{bench} {r[0]}K {m}"] = round(cost[0] / c, 2)
    out["C13_no_cache_flat_price_savings"] = {"rows": nc, "n_cheaper": sum(v > 1 for v in nc.values()), "n": len(nc),
                                              "range": [min(nc.values()), max(nc.values())],
                                              "note": "캐시 입력을 새 입력 단가로 매긴 가상 조건. 하네스가 프롬프트 캐시를 못 쓰면 이쪽에 가까움"}

    path = ROOT / "analysis" / "table_check_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=list))
    s = out["C2_C3_summary"]
    print("C1", out["C1_r"]["inverse"], out["C1_r"]["mean_r"])
    print("C2/C3", s["max_nominal_abs_err"], s["max_eorc_abs_err"], s["kilo_convention_by_table_model"])
    print("C4", out["C4_savings"]["ranges"])
    print("C5", out["C5_flat_price"]["configs_where_nominal_cost_rises"])
    print("C6", gran)
    print("C7", {m: (v["range"], v["n_lose_more_than_2pp"]) for m, v in out["C7_lh_deltas"].items()})
    print("C8 lh sig", [(s["bench"], s["W"], s["model"], s["delta"], s["p"]) for s in out["C8_significance"]["lh_significant_at_0.05"]])
    print("C8 null", out["C8_significance"]["best_of_4_null_swe_verified"], out["C8_significance"]["best_of_4_null_tb"])
    print("C9", out["C9_table2_sac"]["rows"])
    print("C11", out["C11_long_input_windows"]["baseline_mean_input_per_task_K"], out["C11_long_input_windows"]["identical_token_totals_across_models"])
    print("C12", tot)
    print("C13", out["C13_no_cache_flat_price_savings"]["n_cheaper"], out["C13_no_cache_flat_price_savings"]["range"])
    print("wrote", path)


if __name__ == "__main__":
    main()
