"""DUST 표·본문 수치 재계산과 연산량 환산 (CPU, 수 초).

- V1: 본문 주장(§3.2, §4.1, §4.2, TL;DR)을 부록 Table 1·Figure 4 표·Table 8에서 다시 계산
- V2: Table 1의 ∞ 행(power law 극한)을 다섯 칸으로 재적합, 한 칸씩 뺀 민감도
- V3: Table 2(EGGROLL이 Dust@64를 따라잡는 population 배수) 재계산
- V4: 공식 코드(code_official/dust.py)의 draw 할당이 부록 Table 3·4와 같은지
- R1: 학습 step 수, 토큰/파라미터, 균등 예측 손실 대비 위치 (실험이 놓인 학습 구간)
- R2: 한 update의 연산량을 forward 환산으로 계산 → backprop(≈ forward 3회) 대비 배수
- R3: Table 8 cosine 법칙으로 학습 population(16k)에서의 층별 cosine과 필요한 population

실행: python3 analysis/table_check.py  → analysis/table_check_results.json
"""
import ast
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = ["100k", "1M", "10M", "20M"]
TOKENS = {"100k": 1e5, "1M": 1e6, "10M": 1e7, "20M": 2e7}
POPS = [64, 256, 1024, 4096, 16384]

# 부록 Table 1 (3 시드 평균 test loss)
DUST = {"100k": [7.240, 7.165, 7.151, 7.164, 7.170], "1M": [6.290, 6.067, 5.934, 5.911, 5.916],
        "10M": [5.397, 5.231, 5.111, 5.072, 5.049], "20M": [5.265, 5.139, 4.960, 4.912, 4.802]}
DUST_INF = {"100k": 7.161, "1M": 5.892, "10M": 5.013, "20M": 4.431}
EGG = {"100k": [7.961, 7.605, 7.386, 7.302, 7.263], "1M": [7.428, 7.200, 7.046, 6.892, 6.708],
       "10M": [7.056, 6.906, 6.692, 6.352, 6.033], "20M": [6.987, 6.811, 6.517, 6.200, 5.865]}
BP = {"100k": 7.216, "1M": 5.959, "10M": 4.989, "20M": 4.633}
# 부록 Table 2 (Dust@64를 따라잡는 데 필요한 EGGROLL population / 64)
TABLE2 = {"power_all": {"1M": 23000, "10M": 16000, "20M": 8100},
          "line_top3": {"1M": 8200, "10M": 3700, "20M": 3300},
          "line_top2": {"1M": 6000, "10M": 4100, "20M": 3100}}
# Figure 4 표 (10M tokens, 모델 크기별)
SIZES = ["2.0M", "7.3M", "38M", "243M"]
FIG4 = {64: [5.705, 5.556, 5.558, 5.719], 256: [5.486, 5.362, 5.358, 5.419], 1024: [5.265, 5.161, 5.158, 5.214],
        4096: [5.189, 5.095, 5.053, 5.124], 16384: [5.171, 5.065, 5.036, 5.086], "bp": [5.180, 5.066, 5.015, 5.048]}
# 부록 Table 8 (100M token 체크포인트의 cosine 법칙 적합)
TABLE8 = {"MLP out": (0.8779, 31623, 0.0081), "MLP in": (0.8577, 48417, 0.0065), "Attn out": (0.8729, 21627, 0.0093),
          "Attn Q": (1.0, 76736, 0.0356), "Attn K": (1.0, 616595, 0.0191), "Attn V": (0.8776, 173780, 0.0269),
          "Value gate": (0.6657, 47863, 0.0367), "Value emb": (1.0, 495450, 0.0247), "Token emb": (0.7318, 19498, 0.0083),
          "LM head": (1.0, 759, 0.0032)}
# 모델 (부록 A, model.py)
D, L, V, T, BATCH, PARAMS = 512, 8, 4096, 2048, 8, 37.7e6


def check(name, ok, **kw):
    return {"claim": name, "ok": bool(ok), **kw}


def power(P, A, a, E):
    return A * np.power(P, -a) + E


def fit_power(P, y):
    P, y = np.asarray(P, float), np.asarray(y, float)
    best = None
    for a0 in (0.1, 0.3, 0.6, 1.0):
        try:
            p, _ = curve_fit(power, P, y, p0=(y[0] - y[-1] + 0.1, a0, y[-1] - 0.05), maxfev=200000)
        except RuntimeError:
            continue
        r = float(np.sum((power(P, *p) - y) ** 2))
        if best is None or r < best[1]:
            best = (p, r)
    return best[0]


def claims():
    c = []
    c.append(check("§3.2 100k: Dust below backprop from a few hundred draws", DUST["100k"][0] > BP["100k"] > DUST["100k"][1],
                   values=[DUST["100k"][:2], BP["100k"]]))
    c.append(check("§3.2 1M: Dust below backprop from 1k draws", DUST["1M"][1] > BP["1M"] > DUST["1M"][2]))
    c.append(check("§3.2 10M: fitted limit just above backprop", 0 < DUST_INF["10M"] - BP["10M"] < 0.05,
                   gap=round(DUST_INF["10M"] - BP["10M"], 3)))
    c.append(check("§3.2 20M: limit 4.431 below backprop 4.633, measured 16k still above", DUST_INF["20M"] < BP["20M"] < DUST["20M"][-1],
                   gap_16k=round(DUST["20M"][-1] - BP["20M"], 3)))
    g = {b: round(EGG[b][-1] - DUST[b][0], 3) for b in BUDGETS}
    c.append(check("§3.2 EGGROLL@16k vs Dust@64: within 0.02 at 100k, 0.4–0.6 above elsewhere",
                   abs(g["100k"]) < 0.025 and all(0.4 <= g[b] <= 0.65 for b in ("1M", "10M", "20M")), gaps=g))
    # §4.1
    f = {k: np.array(v) for k, v in FIG4.items()}
    c.append(check("§4.1 from 256 up loss falls 2M→7M→38M", all(f[p][0] > f[p][1] > f[p][2] for p in POPS[1:])))
    c.append(check("§4.1 243M ≈ 2M at 64 draws and better at every other population",
                   abs(f[64][3] - f[64][0]) < 0.02 and all(f[p][3] < f[p][0] for p in POPS[1:]),
                   diff_64=round(f[64][3] - f[64][0], 3)))
    gain = (f[1024] - f[16384])
    c.append(check("§4.1 past 1k draws, 38M/243M gain ~30% more than 2M/7M",
                   abs(gain[2:].mean() / gain[:2].mean() - 1.3) < 0.08, ratio=round(float(gain[2:].mean() / gain[:2].mean()), 3),
                   gains=[round(x, 3) for x in gain]))
    gap = {p: [round(x, 3) for x in (f[p] - f["bp"])] for p in POPS}
    c.append(check("§4.1 gap to backprop grows with size but barely (본문)", True, gap_by_pop=gap,
                   note="64 draws에서 2M 0.525 → 243M 0.671. 16k에서는 −0.009 → +0.038"))
    c.append(check("§4.1 243M is worse than 38M at every population (본문 'only slightly worse')",
                   all(f[p][3] > f[p][2] for p in POPS), note="backprop도 243M(5.048)이 38M(5.015)보다 나쁨"))
    # §4.2 / Table 8
    rm = [v[2] for v in TABLE8.values()]
    c.append(check("§4.2 fits RMSE below 0.06, 0.0032–0.0367 at 100M", max(rm) < 0.06 and min(rm) == 0.0032 and max(rm) == 0.0367))
    # 부록 A
    c.append(check("부록 A: 37.7M parameters (아래 R2의 근사 계산과 비교)", True))
    return c


def refits():
    out = {}
    for b in BUDGETS:
        p = fit_power(POPS, DUST[b])
        loo = []
        for i in range(5):
            P = [x for j, x in enumerate(POPS) if j != i]
            y = [x for j, x in enumerate(DUST[b]) if j != i]
            loo.append(round(float(fit_power(P, y)[2]), 3))
        out[b] = {"reported_limit": DUST_INF[b], "refit_limit": round(float(p[2]), 3), "alpha": round(float(p[1]), 3),
                  "limits_leave_one_out(64,256,1k,4k,16k)": loo, "backprop": BP[b]}
    out["note"] = ("시드 평균으로 적합(논문도 Fig. 2에서 같은 방식). 20M의 16k 칸은 자기만의 상수로 따로 튜닝되었고"
                   "(부록 F), 손실 보상 draw도 14,336개라 사다리의 다른 칸과 조건이 다름")
    return out


def eggroll_extrapolation():
    out = {}
    for b in ("1M", "10M", "20M"):
        target = DUST[b][0]
        A, a, E = fit_power(POPS, EGG[b])
        mult_pow = ((target - E) / A) ** (-1 / a) / 64 if target > E else math.inf
        lp = np.log(POPS)
        res = {"power_all": mult_pow}
        for name, k in (("line_top3", 3), ("line_top2", 2)):
            s, i = np.polyfit(lp[-k:], EGG[b][-k:], 1)
            res[name] = math.exp((target - i) / s) / 64
        out[b] = {k: {"recomputed": round(v, -2) if v < 1e9 else "inf", "reported": TABLE2[k][b]} for k, v in res.items()}
    return out


def code_allocation():
    """dust.py의 recipe()를 정적으로 읽어 K = 16,384의 할당을 다시 만든다."""
    src = (ROOT / "code_official" / "dust.py").read_text()
    base_w = ast.literal_eval(re.search(r"override\.get\('writers', (\[[^\]]+\])\)", src).group(1))
    base_h = ast.literal_eval(re.search(r"override\.get\('hubs', (\[[^\]]+\])\)", src).group(1))
    local = ast.literal_eval(re.search(r"for v in (\[[^\]]+\])\]", src).group(1))
    emb = int(re.search(r"embedding, local = (\d+) \* scale", src).group(1))
    ov = re.search(r"16384: dict\(writers=(\[[^\]]+\]), hubs=(\[[^\]]+\]), gamma=([\d.]+)\)", src)
    s = 16384 // 256
    alloc = {"writers": [v * s for v in base_w], "hubs": [v * s for v in base_h], "embedding": emb * s,
             "local(q,v,gate,k,ve)": [v * s for v in local], "head": 4 * 16384,
             "10M@16k writers": [v * s for v in ast.literal_eval(ov.group(1))],
             "10M@16k hubs": [v * s for v in ast.literal_eval(ov.group(2))], "10M@16k gamma": float(ov.group(3))}
    paper = {"writers": [2432, 2048, 1024, 640, 384, 256, 128, 128], "hubs": [384, 384, 256, 128, 128, 128, 128, 128],
             "embedding": 640, "local(q,v,gate,k,ve)": [128, 1280, 384, 768, 512], "head": 65536,
             "10M@16k writers": [2176, 1792, 896, 512, 384, 256, 128, 128],
             "10M@16k hubs": [768, 640, 512, 256, 256, 256, 256, 256], "10M@16k gamma": 0.99}
    K = 2 * sum(alloc["writers"]) + sum(alloc["hubs"]) + alloc["embedding"]
    K10 = 2 * sum(alloc["10M@16k writers"]) + sum(alloc["10M@16k hubs"]) + alloc["embedding"]
    return {"match": {k: alloc[k] == paper[k] for k in paper}, "K_1M": K, "K_10M@16k": K10, "alloc": alloc}


def regime():
    out = {}
    for b in BUDGETS:
        steps = TOKENS[b] / (BATCH * T)
        out[b] = {"steps": round(steps, 1), "tokens_per_param": round(TOKENS[b] / PARAMS, 4),
                  "backprop_loss": BP[b], "uniform_loss_lnV": round(math.log(V), 3),
                  "frac_of_ln V removed by backprop": round(1 - BP[b] / math.log(V), 3)}
    out["chinchilla_tokens_for_37.7M(≈20 tok/param)"] = 20 * PARAMS
    out["243M model at 10M tokens: tokens_per_param"] = round(1e7 / 243e6, 4)
    return out


def compute_cost(alloc, window_ctx=(768, 768, 768, 1024)):
    """한 update의 연산량(MAC/token)을 forward 1회 기준으로 환산.

    블록: q,k,v,o 4d² + MLP 8d² + attention 2·ctx·d. ctx는 causal 평균 문맥 길이(창 1024 블록은 768, 전체 2048은 1024).
    head: d·V. draw가 블록 l에서 시작하면 블록 l..8과 head를 다시 계산한다(부록 Table 3 'its block onward').
    attention 내부 draw는 그 블록의 attention만(2·ctx·d), head draw는 캐시된 logit의 256열만(무시).
    """
    ctx = [window_ctx[i % 4] for i in range(L)]
    block = [12 * D * D + 2 * c * D for c in ctx]
    head = D * V
    fwd = sum(block) + head
    tail = lambda l: (sum(block[l:]) + head) / fwd             # 블록 l(0-index)부터 다시
    w = alloc["writers"]
    h = alloc["hubs"]
    direct = sum((2 * w[l] + h[l]) * tail(l) for l in range(L)) + alloc["embedding"] * 1.0
    local = sum(alloc["local(q,v,gate,k,ve)"]) * sum(2 * c * D for c in ctx) / L / fwd * L / L
    scalars = 8 * 2
    total = 1 + direct + local + scalars
    bp = 3.0
    out = {"forward_MAC_per_token(M)": round(fwd / 1e6, 1), "head_share_of_forward": round(head / fwd, 3),
           "direct_draws_forward_equiv": round(direct, 0), "local_draws_forward_equiv": round(local, 1),
           "residual_scalars_forward_equiv": scalars, "total_forward_equiv_per_update": round(total, 0),
           "per_K_draw": round(direct / 16384, 3), "x_backprop_step(3 forwards)": round(total / bp, 0)}
    out["by_population"] = {}
    for K in POPS[1:]:
        tK = 1 + direct * K / 16384 + local * K / 16384 + scalars
        out["by_population"][K] = {"forward_equiv": round(tK), "x_backprop": round(tK / bp), "vs_EGGROLL_same_pop": round(tK / K, 3)}
    # 같은 연산량이면 backprop이 볼 수 있는 토큰 수
    out["compute_matched_backprop_tokens_for_Dust16k@1M"] = f"{total / bp * 1e6:.2e}"
    return out


def cosine_at(K):
    out = {}
    for k, (cm, c, _) in TABLE8.items():
        cos = cm / math.sqrt(1 + c / K)
        out[k] = {"cos@16k": round(cos, 3), "c_max": cm, "K_for_90%_of_cmax": round(c * (1 / 0.81 - 1)),
                  "K_for_cos_0.8": (round(c / ((cm / 0.8) ** 2 - 1)) if cm > 0.8 else "도달 불가")}
    return out


def param_count():
    """부록 A 구조로 파라미터 수 근사(37.7M과 대조)."""
    per_block = 4 * D * D + 2 * D * 4 * D
    ve = (L // 2) * V * D                       # value embedding: 블록 절반
    gates = (L // 2) * 32 * 8
    return {"blocks": L * per_block, "token_emb": V * D, "head": V * D, "value_emb": ve, "gates": gates,
            "total(M)": round((L * per_block + 2 * V * D + ve + gates + 2 * L) / 1e6, 2), "reported(M)": 37.7}


if __name__ == "__main__":
    ca = code_allocation()
    res = {"V1_claims": claims(), "V2_refits": refits(), "V3_eggroll_extrapolation": eggroll_extrapolation(),
           "V4_code_allocation": ca, "R1_regime": regime(), "R2_compute": compute_cost(ca["alloc"]),
           "R3_cosine_16k": cosine_at(16384), "param_count": param_count()}
    (ROOT / "analysis" / "table_check_results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    bad = [c["claim"] for c in res["V1_claims"] if not c["ok"]]
    print(f"claims OK {len(res['V1_claims']) - len(bad)}/{len(res['V1_claims'])}; failing: {bad}")
    for k in ("V2_refits", "V3_eggroll_extrapolation", "R1_regime", "R2_compute", "R3_cosine_16k", "param_count"):
        print(k, json.dumps(res[k], ensure_ascii=False))
    print("V4", ca["match"], ca["K_1M"], ca["K_10M@16k"])
    for c in res["V1_claims"]:
        print(" -", c)
