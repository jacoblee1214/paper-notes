"""Self Forcing (arXiv:2506.08009 v2) 보고 수치 재계산. CPU, 표준 라이브러리만 사용.

검사 항목
1. VBench Total = a*Quality + (1-a)*Semantic 의 가중치 a 추정 (Table 1, Table 2 전 행)
2. Table 1 과 Table 2 에 같은 행이 있으면 값이 같은지
3. 사용자 선호도 (Figure 4): n = 1003 (App. E) 에서 정수 승수를 역산하고 이항검정, Wilson 95% CI
4. 지연 시간 비율 (Table 1), FPS 와 16 FPS 재생률 비교
"""
import json, math
from pathlib import Path
from statistics import mean

# Table 1: (Total, Quality, Semantic), FPS, latency
T1 = {
    "LTX-Video": (80.00, 82.30, 70.79),
    "Wan2.1-1.3B": (84.26, 85.30, 80.09),
    "SkyReels-V2": (82.67, 84.70, 74.53),
    "MAGI-1": (79.18, 82.04, 67.74),
    "CausVid": (81.20, 84.05, 69.80),
    "Self Forcing chunk": (84.31, 85.07, 81.28),
    "NOVA": (80.12, 80.39, 79.05),
    "Pyramid Flow": (81.72, 84.74, 69.62),
    "Self Forcing frame": (84.26, 85.25, 80.30),
}
T1_PERF = {  # FPS, latency(s)
    "LTX-Video": (8.98, 13.5), "Wan2.1-1.3B": (0.78, 103), "SkyReels-V2": (0.49, 112),
    "MAGI-1": (0.19, 282), "CausVid": (17.0, 0.69), "Self Forcing chunk": (17.0, 0.69),
    "NOVA": (0.88, 4.1), "Pyramid Flow": (6.7, 2.5), "Self Forcing frame": (8.9, 0.45),
}
# Table 2 (chunk-wise, frame-wise): (Total, Quality, Semantic)
T2_CHUNK = {
    "DF (50x2)": (82.95, 83.66, 80.09), "TF (50x2)": (83.58, 84.34, 80.52),
    "DF+DMD": (82.76, 83.49, 79.85), "TF+DMD": (82.32, 82.73, 80.67),
    "SF DMD": (84.31, 85.07, 81.28), "SF SiD": (84.07, 85.52, 78.24), "SF GAN": (83.88, 85.06, 79.16),
}
T2_FRAME = {
    "DF (50x2)": (77.24, 79.72, 67.33), "TF (50x2)": (80.34, 81.34, 76.34),
    "DF+DMD": (80.56, 81.02, 78.71), "TF+DMD": (78.12, 79.62, 72.11),
    "SF DMD": (84.26, 85.25, 80.30), "SF SiD": (83.54, 84.71, 78.86), "SF GAN": (83.27, 84.57, 78.08),
}

out = {}

# 1. Total = a*Q + (1-a)*S
rows = list(T1.items()) + [("T2chunk " + k, v) for k, v in T2_CHUNK.items()] + [("T2frame " + k, v) for k, v in T2_FRAME.items()]
a_est = []
max_res = 0.0
for name, (tot, q, s) in rows:
    if abs(q - s) < 1e-9:
        continue
    a = (tot - s) / (q - s)
    a_est.append(a)
    pred = 0.8 * q + 0.2 * s
    max_res = max(max_res, abs(pred - tot))
out["vbench_weight_a_range"] = [round(min(a_est), 4), round(max(a_est), 4)]
out["vbench_total_0.8Q_0.2S_max_abs_residual"] = round(max_res, 3)
out["vbench_rows_checked"] = len(a_est)

# 2. Table 1 vs Table 2 same rows
out["T1_vs_T2_SF_chunk_match"] = T1["Self Forcing chunk"] == T2_CHUNK["SF DMD"]
out["T1_vs_T2_SF_frame_match"] = T1["Self Forcing frame"] == T2_FRAME["SF DMD"]

# 3. Margin of Self Forcing over Wan2.1 (Total)
out["total_margin_chunk_vs_wan"] = round(84.31 - 84.26, 2)

# 4. Latency and FPS checks
out["latency_ratio_wan_over_sf_chunk"] = round(103 / 0.69, 1)
out["latency_ratio_wan_over_sf_frame"] = round(103 / 0.45, 1)
out["fps_margin_chunk_over_16fps"] = round(17.0 / 16.0, 3)
out["frame_wise_fps_vs_16fps_ratio"] = round(8.9 / 16.0, 3)

# 5. User study, n = 1003 (App. E), one rater per prompt
N = 1003
def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

def binom_two_sided_p(k, n):
    """정확한 양측 이항검정 (p = 0.5)."""
    from math import comb
    tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)

pref = {"CausVid": 66.1, "Wan2.1": 62.7, "SkyReels-V2": 57.9, "MAGI-1": 54.2}
pref_out = {}
for name, pct in pref.items():
    cand = [(abs(round(k / N * 100, 1) - pct), k) for k in range(N + 1)]
    _, k = min(cand)
    lo, hi = wilson(k, N)
    pref_out[name] = {
        "reported_pct": pct, "k_wins_inferred": k, "pct_check": round(k / N * 100, 2),
        "wilson95": [round(lo * 100, 1), round(hi * 100, 1)],
        "exact_two_sided_p_vs_50": float(f"{binom_two_sided_p(k, N):.3g}"),
    }
out["user_study"] = pref_out

HERE = Path(__file__).resolve().parent
json.dump(out, open(HERE / "recompute_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps(out, ensure_ascii=False, indent=2))
