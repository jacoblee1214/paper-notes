"""Recompute reported numbers of Wan (arXiv:2503.20314 v2) from the printed tables.

Inputs: numbers transcribed by hand from v2 (Tables 2-8, Sec. 4.1.4, 4.3.2, 4.4, 4.2.2).
Outputs: recompute_results.json next to this file. numpy and scipy only. No model runs.
"""
import json, math, os
import numpy as np
from scipy import stats

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recompute_results.json")
R = {}

# ---- Table 4 (VBench): Total = (4*Quality + Semantic)/5 ? ----
vbench = {
    "MiniMax-Video-01": (84.85, 77.65, 83.41),
    "Hunyuan": (85.09, 75.82, 83.24),
    "Gen-3 (2024-07)": (84.11, 75.17, 82.32),
    "CogVideoX1.5-5B": (82.78, 79.76, 82.17),
    "Kling (2024-07)": (83.39, 75.68, 81.85),
    "Sora": (85.51, 79.35, 84.28),
    "Wan 1.3B": (84.92, 80.10, 83.96),
    "Wan 14B (2025-02-24)": (86.67, 84.44, 86.22),
}
rows = {}
for k, (q, s, t) in vbench.items():
    calc = (4 * q + s) / 5
    rows[k] = {"calc": round(calc, 3), "reported": t, "diff": round(calc - t, 3)}
R["vbench_total_check"] = rows
R["vbench_max_abs_diff"] = max(abs(v["diff"]) for v in rows.values())

# ---- Table 2 (Wan-Bench): 14 dimensions, weights unknown ----
dims = ["LargeMotion", "HumanArtifacts", "PixelStab", "IDCons", "PhysPlaus", "Smooth",
        "CompImgQual", "SceneQual", "Stylization", "SingleObj", "MultiObj", "SpatialPos",
        "CameraCtrl", "ActionFollow"]
wb = {
    "CN-TopB":   [0.405,0.712,0.977,0.940,0.836,0.765,0.621,0.369,0.623,0.987,0.840,0.518,0.465,0.917],
    "Hunyuan":   [0.413,0.734,0.983,0.935,0.898,0.890,0.605,0.373,0.386,0.912,0.850,0.464,0.406,0.735],
    "Mochi":     [0.420,0.622,0.981,0.930,0.728,0.530,0.530,0.368,0.403,0.949,0.693,0.512,0.605,0.907],
    "CN-TopA":   [0.284,0.833,0.974,0.936,0.759,0.880,0.668,0.386,0.346,0.942,0.880,0.434,0.529,0.783],
    "Sora":      [0.482,0.786,0.952,0.925,0.933,0.930,0.665,0.388,0.606,0.932,0.882,0.458,0.380,0.721],
    "Wan 1.3B":  [0.468,0.707,0.976,0.938,0.912,0.790,0.596,0.385,0.430,0.930,0.859,0.476,0.483,0.844],
    "Wan 14B":   [0.415,0.691,0.972,0.946,0.939,0.910,0.640,0.386,0.328,0.952,0.860,0.590,0.527,0.860],
}
weighted_reported = {"CN-TopB":0.690,"Hunyuan":0.673,"Mochi":0.639,"CN-TopA":0.693,"Sora":0.700,"Wan 1.3B":0.689,"Wan 14B":0.724}
unw = {k: float(np.mean(v)) for k, v in wb.items()}
R["wanbench_unweighted_mean"] = {k: round(v, 4) for k, v in sorted(unw.items(), key=lambda x: -x[1])}
R["wanbench_reported_weighted"] = weighted_reported
R["wanbench_rank_unweighted"] = [k for k, _ in sorted(unw.items(), key=lambda x: -x[1])]
R["wanbench_rank_reported"] = [k for k, _ in sorted(weighted_reported.items(), key=lambda x: -x[1])]
# Sensitivity of Wan14B - Sora (reported 0.724 vs 0.700) for N=1035 prompts, unknown per-prompt SD s.
N_wb = 1035
diff = 0.724 - 0.700
sens = {}
for s in [0.10, 0.15, 0.20, 0.25, 0.30]:
    se = s * math.sqrt(2 / N_wb)
    z = diff / se
    sens[str(s)] = {"se_diff": round(se, 4), "z": round(z, 2), "p_two_sided": float(2 * (1 - stats.norm.cdf(z)))}
R["wanbench_sens_wan14_vs_sora"] = {"diff": diff, "N": N_wb, "by_sd": sens,
    "s_needed_for_p05": round(diff / (1.96 * math.sqrt(2 / N_wb)), 3)}

# ---- Table 3 (T2V win-rate gap) and Table 7 (I2V) ----
# Table 7 has a negative cell (-4.2%), so the printed values are a net gap (win% - loss%), not a plain win share.
# Under multinomial sampling with ties, Var(g) approx (1 - g^2)/N  (upper bound on variance when ties are small: p_w+p_l<=1).
def gap_ci(g, n):
    se = math.sqrt(max(1 - g * g, 1e-9) / n)
    return {"g": g, "n": n, "ci95_low": round(g - 1.96 * se, 4), "ci95_high": round(g + 1.96 * se, 4),
            "win_implied_if_no_ties": round((1 + g) / 2, 4)}
t3 = {
  "Visual Quality": ([30.6,15.9,27.8,48.1], 5710),
  "Motion Quality": ([16.1,9.7,14.9,40.3], 5785),
  "Matching":       ([46.0,57.9,56.7,69.1], 5578),
  "Overall Ranking":([44.0,44.0,48.9,67.6], 5560),
}
R["table3_ci"] = {}
for dim, (vals, n) in t3.items():
    R["table3_ci"][dim] = {opp: gap_ci(v / 100, n) for opp, v in zip(["CN-TopA","CN-TopB","CN-TopC","Runway"], vals)}
t7 = {
  "Visual Quality": ([29.2,60.8,24.6,55.6], 896),
  "Motion Quality": ([21.7,21.7,32.5,67.0], 890),
  "Matching":       ([-4.2,35.0,51.7,72.2], 896),
  "Overall Ranking":([10.8,47.5,50.8,81.6], 890),
}
R["table7_ci"] = {}
for dim, (vals, n) in t7.items():
    R["table7_ci"][dim] = {opp: gap_ci(v / 100, n) for opp, v in zip(["CN-TopA","CN-TopB","CN-TopC","CN-TopD"], vals)}
R["table7_matching_vs_topA_ci_covers_zero"] = (R["table7_ci"]["Matching"]["CN-TopA"]["ci95_low"] <= 0 <= R["table7_ci"]["Matching"]["CN-TopA"]["ci95_high"])
# If the printed value were a plain win share (not gap), Table 3 Overall Ranking vs CN-TopA would be 44%: <50% -> Wan would not lead.
R["note_t3_overall_vs_topA_as_share"] = 0.44

# ---- Table 5 (VAE FID) and Table 6 (text encoder FID): no seeds, no CI reported ----
R["table5_vae_minus_vaed"] = {"10k": round(42.60 - 44.21, 2), "15k": round(40.55 - 41.16, 2)}
R["table5_label_mismatch"] = "text says 100k/150k; table rows say 10k/15k"
R["table6_umt5_minus_qwen2nd"] = round(43.01 - 42.91, 2)

# ---- Table 8 (ArcFace, personalization) ----
R["table8_wan_minus_topA"] = round(0.5526 - 0.5655, 4)

# ---- DiT parameter count from configs (wan/configs/*.py), linear weights only ----
def dit_params(d, ffn, L, text_dim=4096, freq=256):
    attn = 4 * d * d          # q,k,v,o
    cross = 4 * d * d         # q,k,v,o (text projected to d first)
    ffn_p = 2 * d * ffn
    per_layer = attn + cross + ffn_p
    shared = freq * d + d * d + text_dim * d + d * d + d * 6 * d  # time/text embeds + time_projection
    return per_layer * L + shared
R["dit_params_14B_billion"] = round(dit_params(5120, 13824, 40) / 1e9, 3)
R["dit_params_1_3B_billion"] = round(dit_params(1536, 8960, 30) / 1e9, 3)

# ---- Token counts: L = (1+T/4) x H/16 x W/16 ; 81 frames ----
def tokens(w, h, frames=81):
    t = (frames - 1) // 4 + 1
    return t * (h // 16) * (w // 16), (t, h // 16, w // 16)
R["tokens_720p_81f"] = tokens(1280, 720)[0]
R["tokens_480p_81f"] = tokens(832, 480)[0]
R["frames_5s_16fps"] = 5 * 16 + 1

# ---- Memory sanity (bytes/param): weights only, rough ----
R["weights_1_3B_bf16_GB"] = round(1.3e9 * 2 / 2**30, 2)
R["umt5_xxl_5p3B_bf16_GB"] = round(5.3e9 * 2 / 2**30, 2)
R["umt5_xxl_5p3B_fp32_GB"] = round(5.3e9 * 4 / 2**30, 2)

# ---- Forward FLOPs per DiT pass (rough): attention 4*L^2*d per layer, linear 2*P*L ----
def fwd_flops(L, d, layers, params):
    attn = 4 * L * L * d * layers
    lin = 2 * params * L
    return attn, lin
f14 = fwd_flops(75600, 5120, 40, 14.28e9)
f13 = fwd_flops(32760, 1536, 30, 1.42e9)
R["flops_14B_720p_fwd"] = {"attn": f14[0], "linear": f14[1], "attn_share": round(f14[0] / sum(f14), 3)}
R["flops_13B_480p_fwd"] = {"attn": f13[0], "linear": f13[1], "attn_share": round(f13[0] / sum(f13), 3)}
# 50 sampling steps x 2 (CFG cond + uncond) = 100 forward passes for a 5 s clip (no caching)
R["flops_14B_720p_clip"] = sum(f14) * 100
R["flops_13B_480p_clip"] = sum(f13) * 100
R["weights_14B_bf16_GiB"] = round(14.28e9 * 2 / 2**30, 2)
R["weights_13B_bf16_GiB"] = round(1.42e9 * 2 / 2**30, 2)
R["umt5_5p3B_fp32_GiB_check"] = round(5.3e9 * 4 / 2**30, 2)

# ---- Speedups (Sec. 4.4): reported separately; not composable from the text ----
R["speedups_reported"] = {"diffusion_cache_14B": 1.62, "fp8_gemm_dit": 1.13, "fa8_attention": 1.27}

with open(OUT, "w") as f:
    json.dump(R, f, ensure_ascii=False, indent=1)
print(json.dumps({k: R[k] for k in ["vbench_max_abs_diff","dit_params_14B_billion","dit_params_1_3B_billion","tokens_720p_81f","tokens_480p_81f","table7_matching_vs_topA_ci_covers_zero","wanbench_rank_unweighted","wanbench_rank_reported","wanbench_sens_wan14_vs_sora"]}, ensure_ascii=False, indent=1))
