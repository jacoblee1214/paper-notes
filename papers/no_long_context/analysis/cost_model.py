"""'7배' 비용 비율이 구조에 따라 얼마나 달라지는지 보는 numpy 장난감 모델 (CPU, 단일 스레드).
실행: python3 analysis/cost_model.py  →  analysis/cost_model_results.json

M1. 메모리 한계 서빙: 논문 §2.5 측정 규약(요청 길이를 [0, W]에서 균일하게 뽑고 1,024 토큰 생성)을 따르면
    요청 하나의 평균 컨텍스트는 약 W/2. KV 풀이 가득 찬 decode 단계 시간은 거의 일정하므로 처리량 ∝ 동시 요청 수
    ∝ 1 / (s0 + k·W/2). s0 = 요청당 고정 상태(SSM·DeltaNet), k = 토큰당 KV 바이트.
    → T(16K)/T(256K) = (s0 + k·128K) / (s0 + k·8K).  순수 어텐션 16배, 순수 SSM 1배.
M2. 관측값(3.6·5.6·7배)과 M1 상한의 차이 → 16K에서는 메모리가 아닌 다른 한계(연산, 동시성 상한)가 먼저 걸림.
M3. 배치 1 지연(논문 Fig. 2 방식): (가중치 읽기 + s0 + k·L) / (가중치 읽기 + s0 + k·16K).
M4. EORC 절감 배수가 r에 얼마나 민감한지 (r = 1이면 평탄 단가 = 명목 비용).
M5. 같은 규칙으로 기준선도 값 매기기: 논문은 FHA 기준선에 배수 1(창 전체를 균일하게 쓰는 서버 값)을 매기지만,
    실제 FHA 문맥 대부분은 짧음. 요청마다 Eq. 2를 W = 2c(c = 그 요청의 컨텍스트)로 적용하는 몬테카를로 추정.

구조 수치 출처
- Nemotron 3.5 Super VL: ../nemotron35_super_vl/analysis/param_check_results.json (P4_state_memory). 이 repo에서 config로 검증한 값.
- Qwen3-Next-80B-A3B(공개 config, 이 repo에서 재확인하지 않음): 48층 = 12 × (Gated DeltaNet 3 + gated attention 1),
  attention 16 Q / 2 KV head, head dim 256; DeltaNet 32 V head × 128 × 128 상태. 논문의 Qwen 3 Coder Next가
  같은 구조라는 것은 논문 §2.3의 "same hybrid Gated DeltaNet/full-attention pattern" 문장에 기댄 가정.
- Kimi K2.6: Kimi K2(DeepSeek-V3 계열 MLA, 61층, 512+64 차원 latent)와 같다고 가정. 논문은 FP8 KV라고만 씀.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KI, MI, GB = 1024, 1024 ** 2, 1e9
OBS = {"GLM 5.2": 1 / 0.2772, "Kimi K2.6": 1 / 0.1777, "Qwen 3 Coder Next": 1 / 0.1439}

# ── 구조별 (s0 = 요청당 고정 상태 바이트, k = 토큰당 KV 바이트)
NEMO = json.loads((ROOT.parent / "nemotron35_super_vl/analysis/param_check_results.json").read_text())["P4_state_memory"] \
    if (ROOT.parent / "nemotron35_super_vl/analysis/param_check_results.json").exists() else None
nemo_kv_fp8 = 4096
nemo_kv_bf16 = 8192
nemo_ssm_fp16 = (80 + 2.34) * MI
nemo_ssm_fp32 = (160 + 2.34) * MI
qn_kv_bf16 = 2 * 12 * 2 * 256 * 2                        # K,V × attention 12층 × KV head 2 × dim 256 × 2B = 24,576 B
qn_state_fp32 = 36 * 32 * 128 * 128 * 4                   # DeltaNet 36층 상태 (FP32) ≈ 72 MiB
qn_state_bf16 = qn_state_fp32 // 2
qn_conv = 36 * 3 * (2 * 16 * 128 + 32 * 128) * 2          # conv kernel 4 → 3칸, BF16
kimi_kv_fp8 = 61 * (512 + 64) * 1                         # MLA latent, FP8

ARCH = {
    "순수 어텐션(GQA·MLA 무관)": (0.0, 1.0),
    "Kimi K2.6 가정(MLA, FP8 KV)": (0.0, kimi_kv_fp8),
    "Qwen3-Next 구조 가정(DeltaNet 상태 FP32)": (qn_state_fp32 + qn_conv, qn_kv_bf16),
    "Qwen3-Next 구조 가정(DeltaNet 상태 BF16)": (qn_state_bf16 + qn_conv, qn_kv_bf16),
    "Nemotron 3.5 Super VL (FP8 KV, SSM FP16)": (nemo_ssm_fp16, nemo_kv_fp8),
    "Nemotron 3.5 Super VL (FP8 KV, SSM FP32)": (nemo_ssm_fp32, nemo_kv_fp8),
    "Nemotron 3.5 Super VL (BF16 KV, SSM FP16)": (nemo_ssm_fp16, nemo_kv_bf16),
    "순수 SSM": (1.0, 0.0),
}


def mem_ratio(s0, k, lo=16 * KI, hi=256 * KI, out=1024):
    mean_lo, mean_hi = lo / 2 + out / 2, hi / 2 + out / 2   # 요청 길이 균일 + 생성 중 평균 512 토큰
    return (s0 + k * mean_hi) / (s0 + k * mean_lo)


def eq2_factor(w_tokens, wmax, r=0.20, lo=16_000):
    """논문 Eq. 2 배수 (K=1000). w_tokens 배열 가능. Wmax 넘는 값은 선형 외삽(요청 단위 가격용)."""
    w = np.maximum(np.asarray(w_tokens, float), lo)
    return r + (1 - r) * (w - lo) / (wmax - lo)


# ── M4/M5용 표 값 (table_check.py와 같은 출처: Table 3–7)
PRICE = {"opus": (5.0, 0.5, 25.0), "gpt": (4.0, 0.4, 20.0)}
WMAX = {"opus": 200_000, "gpt": 272_000}
# (W, θ, tok_opus(I,H,O), tok_gpt, 기준선 총 길이 mean/median/max는 별도)
LH = {
    "SWE-bench Verified": (500, [(None, None, (20.0, 420.0, 4.4), (16.4, 100.2, 3.1)),
                                 (48, 36, (20.0, 334.7, 4.5), (17.6, 94.2, 3.9)),
                                 (32, 24, (23.0, 255.9, 4.7), (17.2, 73.4, 4.3)),
                                 (24, 18, (22.0, 166.5, 4.1), (17.7, 60.4, 4.7)),
                                 (16, 12, (28.3, 127.3, 4.3), (19.7, 45.8, 5.3))]),
    "SWE-bench Multilingual": (300, [(None, None, (11.9, 384.1, 3.0), (7.4, 66.2, 1.6)),
                                     (48, 36, (12.8, 245.5, 2.9), (8.8, 65.6, 1.9)),
                                     (32, 24, (14.0, 147.0, 2.8), (10.0, 49.1, 2.3)),
                                     (24, 18, (15.6, 91.0, 2.7), (10.8, 40.3, 2.6)),
                                     (16, 12, (17.0, 65.4, 2.7), (12.4, 31.5, 3.1))]),
    "Terminal-Bench 2.1": (89, [(None, None, (3.9, 81.6, 1.1), (3.6, 30.4, 1.1)),
                                (48, 36, (6.2, 77.5, 1.8), (3.7, 16.7, 1.3)),
                                (32, 24, (6.7, 47.1, 1.7), (4.0, 14.1, 1.6)),
                                (24, 18, (6.5, 34.0, 1.7), (6.8, 18.0, 2.8)),
                                (16, 12, (7.6, 23.0, 1.6), (4.2, 10.3, 2.0))]),
    "FRAMES": (824, [(None, None, (27.4, 606.5, 2.7), (21.9, 177.6, 2.4)),
                     (48, 36, (30.2, 307.6, 2.6), (27.6, 171.1, 3.5)),
                     (32, 24, (28.7, 196.6, 2.4), (30.0, 147.7, 4.2)),
                     (24, 18, (36.8, 173.2, 2.9), (29.4, 125.4, 4.6)),
                     (16, 12, (39.2, 108.2, 3.1), (28.9, 105.2, 5.7))]),
}
# 기준선 "Total length" (K): mean, median, max  (Table 3–6)
BASE_LEN = {"SWE-bench Verified": {"opus": (48, 32, 200), "gpt": (22, 18, 173)},
            "SWE-bench Multilingual": {"opus": (35, 31, 95), "gpt": (23, 21, 63)},
            "Terminal-Bench 2.1": {"opus": (34, 27, 130), "gpt": (28, 17, 133)},
            "FRAMES": {"opus": (25, 14, 92), "gpt": (17, 10, 140)}}


def eorc(tok, model, factors):
    p = PRICE[model]
    return sum(t * c * f for t, c, f in zip(tok, p, factors))


def paper_factor(w_k, model, r):
    if w_k is None:
        return 1.0
    return float(eq2_factor(w_k * 1000, WMAX[model], r))


def simulate(n_tasks, mean, median, cap, model, H_target, theta=None, brief=3_000, c0=3_000, rng=None, heavy=False):
    """FHA 궤적을 로그정규 최종 길이(K)로 뽑고, 컨텍스트가 δ씩 자란다고 보고 토큰 종류별 평균 배수를 구한다.
    δ는 기준선 캐시 입력 합계(H_target, 토큰)를 맞추도록 정한다. theta가 있으면 θ에서 brief 크기로 압축."""
    mu, sig = np.log(median * 1000), np.sqrt(max(2 * np.log(mean / median), 1e-6))
    if heavy:   # 꼬리를 max에도 맞춤: max를 1 − 1/(2·500) 분위(z≈3.09)로 보고 σ를 더 큰 쪽으로
        sig = max(sig, np.log(cap / median) / 3.09)
    L = np.minimum(rng.lognormal(mu, sig, n_tasks), cap * 1000)
    L = np.maximum(L, c0 + 1)
    # δ 보정: Σ_t c_t ≈ (L² − c0²) / (2δ)
    delta = float(np.mean((L ** 2 - c0 ** 2) / 2) / H_target)
    w = WMAX[model]
    acc = {"H_w": 0.0, "H_f": 0.0, "S_w": 0.0, "S_f": 0.0, "comp": 0}
    for Li in L:
        steps = max(int((Li - c0) / delta), 1)
        if theta is None:
            c = c0 + delta * np.arange(steps)
        else:
            c = np.empty(steps)
            cur = c0
            for s in range(steps):
                if cur >= theta * 1000:
                    cur = brief
                    acc["comp"] += 1
                c[s] = cur
                cur += delta
        g = eq2_factor(2 * c, w)
        acc["H_w"] += c.sum()
        acc["H_f"] += (c * g).sum()           # 캐시 입력: 컨텍스트 길이만큼 다시 읽음
        acc["S_w"] += steps
        acc["S_f"] += g.sum()                 # 새 입력·출력: 단계마다 비슷한 양
    return {"phi_cached": acc["H_f"] / acc["H_w"], "phi_step": acc["S_f"] / acc["S_w"],
            "delta_tokens": round(delta), "mean_compactions": acc["comp"] / n_tasks,
            "share_tasks_at_cap": float(np.mean(L >= cap * 1000 - 1))}


def main():
    out = {}
    rows = {}
    for name, (s0, k) in ARCH.items():
        ratio = mem_ratio(s0, k)
        rows[name] = {"s0_MiB": round(s0 / MI, 2), "kv_B_per_token": k,
                      "s0_over_k_tokens": None if k == 0 else round(s0 / k), "mem_bound_ratio_16K_vs_256K": round(ratio, 2)}
    out["M1_memory_bound_ratio"] = rows
    if NEMO:
        out["M1_nemotron_repo_values"] = {k: NEMO[k] for k in list(NEMO)[:12]}
    curve_x = np.array([0, 1, 2, 4, 8, 16, 21, 32, 42, 64, 128, 256]) * 1000
    out["M1_curve"] = {"s0_over_k_tokens": curve_x.tolist(),
                       "ratio": [round(mem_ratio(x, 1.0), 2) for x in curve_x]}

    impl = {}
    for m, ob in OBS.items():
        key = {"Kimi K2.6": "Kimi K2.6 가정(MLA, FP8 KV)", "Qwen 3 Coder Next": "Qwen3-Next 구조 가정(DeltaNet 상태 FP32)"}.get(m)
        ub = rows[key]["mem_bound_ratio_16K_vs_256K"] if key else None
        impl[m] = {"observed": round(ob, 2), "mem_bound_upper": ub,
                   "fraction_of_mem_bound_concurrency_used_at_16K": None if ub is None else round(ob / ub, 2)}
    out["M2_observed_vs_upper"] = {"rows": impl, "note": "GLM 5.2 구조는 확인하지 못해 비워 둠. 비율 < 1이면 16K 지점이 메모리가 아닌 다른 한계에 걸림"}

    # M3 배치 1 지연 (Nemotron Super VL, repo 값: NVFP4 decode 가중치 읽기 9.76 GB/token)
    wread = 9.76 * GB
    b1 = {}
    for L in (16, 64, 128, 256, 1024):
        b1[f"{L}K"] = round((wread + nemo_ssm_fp16 + nemo_kv_fp8 * L * KI) / (wread + nemo_ssm_fp16 + nemo_kv_fp8 * 16 * KI), 3)
    out["M3_batch1_latency_ratio_super_vl"] = {"vs_16K": b1, "assume": "대역폭 한계, 가중치 9.76 GB/token(NVFP4), FP8 KV, SSM FP16"}

    # M4 r 민감도
    r_list = [0.1439, 0.1777, 0.20, 0.2772, round(1 / 5.20, 4), round(1 / 3.47, 4), 0.5, 1.0]
    sens = {}
    for bench, (n, data) in LH.items():
        base = data[0]
        for w, th, to, tg in data[1:]:
            sens[f"{bench} {w}K"] = {m: [round(eorc(base[2 if m == 'opus' else 3], m, [1, 1, 1]) /
                                           eorc(tok, m, [paper_factor(w, m, r)] * 3), 2) for r in r_list]
                                     for m, tok in (("opus", to), ("gpt", tg))}
    out["M4_r_sensitivity"] = {"r_values": r_list, "savings": sens,
                               "note": "r=1이면 평탄 단가(명목 비용 비율). 0.19·0.29는 Super VL 메모리 한계 모델의 1/5.2·1/3.47"}

    # M5 같은 규칙(요청 단위 Eq. 2)으로 기준선과 압축 설정 모두 값 매기기
    rng = np.random.default_rng(0)
    m5 = {}
    for bench, (n, data) in LH.items():
        for mi, m in ((2, "opus"), (3, "gpt")):
            mean, med, mx = BASE_LEN[bench][m]
            H_base = data[0][mi][1] * 1e6 / n
            res = {}
            for heavy in (False, True):
                sims = {w: simulate(4000, mean, med, mx, m, H_base, theta=th, rng=np.random.default_rng(1), heavy=heavy)
                        for w, th, *_ in data}
                res[heavy] = sims
            base_tok = data[0][mi]
            base_paper = eorc(base_tok, m, [1, 1, 1])
            for w, th, *toks in data[1:]:
                tok = toks[mi - 2]
                paper = eorc(tok, m, [paper_factor(w, m, 0.2)] * 3)
                same = {}
                for heavy, sims in res.items():
                    s0, s = sims[None], sims[w]
                    bm = eorc(base_tok, m, [s0["phi_step"], s0["phi_cached"], s0["phi_step"]])
                    cm = eorc(tok, m, [s["phi_step"], s["phi_cached"], s["phi_step"]])
                    same[heavy] = (round(bm / cm, 2), round(s0["phi_cached"], 3), round(s["phi_cached"], 3),
                                   round(s["mean_compactions"], 2), s["delta_tokens"])
                m5[f"{bench} {w}K {m}"] = {
                    "paper_savings": round(base_paper / paper, 2),
                    "same_rule_savings": same[False][0], "same_rule_savings_heavy_tail": same[True][0],
                    "baseline_factor_cached": same[False][1], "baseline_factor_cached_heavy": same[True][1],
                    "config_factor_cached": same[False][2], "config_factor_paper": round(paper_factor(w, m, 0.2), 3),
                    "break_even_baseline_factor": round(paper / base_paper, 3),
                    "sim_mean_compactions": same[False][3], "delta_tokens": same[False][4]}
    out["M5_same_rule"] = {"rows": m5,
                           "assume": "로그정규(기준선 median·mean에 맞춤, max에서 자름), 시작 3K, 요약 3K, δ는 기준선 캐시 합계로 보정, "
                                     "압축 후에도 단계 수 같음. 압축 설정의 토큰 합계는 논문 값 그대로 사용"}
    vals = [v["same_rule_savings"] for v in m5.values()]
    hv = [v["same_rule_savings_heavy_tail"] for v in m5.values()]
    pv = [v["paper_savings"] for v in m5.values()]
    by_model = {m: {"paper": [v["paper_savings"] for k, v in m5.items() if k.endswith(m)],
                    "same": [v["same_rule_savings"] for k, v in m5.items() if k.endswith(m)],
                    "heavy": [v["same_rule_savings_heavy_tail"] for k, v in m5.items() if k.endswith(m)]} for m in ("opus", "gpt")}
    out["M5_summary"] = {"paper_range": [min(pv), max(pv)], "same_rule_range": [min(vals), max(vals)],
                         "same_rule_heavy_range": [min(hv), max(hv)],
                         "median_ratio_same_over_paper": round(float(np.median(np.array(vals) / np.array(pv))), 2),
                         "by_model_ranges": {m: {k: [min(v), max(v)] for k, v in d.items()} for m, d in by_model.items()},
                         "n_below_1": {"same": int(sum(v < 1 for v in vals)), "heavy": int(sum(v < 1 for v in hv))},
                         "break_even_note": "기준선의 실제 평균 배수가 break_even보다 크면 압축이 싸다. 압축 설정은 논문 값 f(W)(상한)로 둠"}

    path = ROOT / "analysis" / "cost_model_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for k, v in rows.items():
        print(f"M1 {k:44s} s0/k={v['s0_over_k_tokens']}  ratio={v['mem_bound_ratio_16K_vs_256K']}")
    print("M2", impl)
    print("M3", b1)
    print("M4 SWE-V 48K", sens["SWE-bench Verified 48K"], "\n   r=", r_list)
    for k, v in m5.items():
        print("M5", k, v["paper_savings"], v["same_rule_savings"], v["same_rule_savings_heavy_tail"],
              v["baseline_factor_cached"], v["baseline_factor_cached_heavy"], v["break_even_baseline_factor"],
              v["sim_mean_compactions"], v["delta_tokens"])
    print("M5 summary", out["M5_summary"])
    print("wrote", path)


if __name__ == "__main__":
    main()
