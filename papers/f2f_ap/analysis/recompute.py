"""F2F-AP (arXiv:2604.02408 v4) 표 1·2의 수치 재계산과 검정.

실행: python3 analysis/recompute.py  -> analysis/recompute_results.json (수 초, numpy/scipy/statsmodels)

재계산 항목
  V1  표 1 성공률 퍼센트와 Wilson 95% 구간
  V2  과제별 F2F-AP 대 각 기준선 Fisher 정확 검정 (양측), Holm 보정(28개)
  V3  과제를 층으로 한 Mantel-Haenszel 합산 (F2F-AP 대 각 기준선), 단순 합산 비율
  V4  본문 "성공률 2배 이상" 주장 점검 (최고 기준선 대비 배수)
  V5  실행 시간(ET) 비교: 평균만 주어져 검정은 불가, 비율과 순위만
  V6  표 2 절제: 전체 모델 대 각 절제 Fisher, 지연 추세
  V7  잠재 지연 산술 H = ceil(총지연/dt), 추론 주기와의 관계
  V8  버전 간 값 변화 (v1/v2 그림 값 대 v4 표)
  V9  검정력: 15/20 대 12/20을 구분하는 데 필요한 시도 수
  K1–K6 내부 일관성 (분모, 표 1과 표 2의 같은 값, 데이터 양 산술, 대조 손실 마스크 기대 빈도, 예측기 파라미터 수)
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import StratifiedTable

OUT = Path(__file__).with_name("recompute_results.json")

TASKS = ["Hammer", "Banana", "Carrot", "Pull-Back Car", "Chip Can", "Bottle", "Length Meas."]
# (성공, 시도, ET 초 또는 None)  표 1 (v4, 본문 p.8)
T1 = {
    "Syn. UMI": [(5, 20, 5.7), (5, 20, 3.0), (3, 10, 2.8), (0, 20, None), (3, 20, 4.1), (2, 20, 9.2), (4, 10, 2.6)],
    "Asyn. UMI": [(7, 20, 6.2), (10, 20, 2.9), (5, 10, 2.4), (1, 20, 1.4), (2, 20, 4.2), (2, 20, 9.1), (5, 10, 2.4)],
    "RTC": [(12, 20, 4.1), (12, 20, 2.4), (6, 20, 2.0), (0, 20, None), (3, 20, 3.9), (3, 20, 8.2), (5, 20, 2.2)],
    "VLASH": [(6, 20, 3.3), (13, 20, 2.5), (3, 10, 1.4), (3, 20, 1.9), (1, 20, 3.0), (2, 20, 7.6), (3, 10, 1.4)],
    "F2F-AP": [(15, 20, 3.4), (13, 20, 2.5), (6, 10, 1.6), (8, 20, 2.5), (9, 20, 3.5), (11, 20, 7.9), (7, 10, 1.9)],
}
BASELINES = ["Syn. UMI", "Asyn. UMI", "RTC", "VLASH"]

# 표 2 (병 잡기 과제, 20회)
T2 = {
    "Fewer Flow Points": (10, 20, 7.8), "Diff. Line Color": (11, 20, 7.8),
    "Low Contrastive Wt.": (7, 20, 8.5), "w/o Contrastive Loss": (8, 20, 8.5),
    "w/o Future State": (3, 20, 8.3), "w/o Heatmap": (5, 20, 9.1),
    "MF-UMI": (1, 20, 8.7), "RTC-MF-UMI": (0, 20, None), "Low Freq. Inference": (3, 20, 10.2),
    "300ms Latency": (12, 20, 7.5), "500ms Latency": (10, 20, 8.5), "600ms Latency": (4, 20, 11.2),
    "Full F2F-AP": (11, 20, 7.9),
}

# v1/v2 그림 (서로 같은 값, 그림 6-8과 표 I)과 v4 표의 차이 (v1/v2 값, v4 값)
V1_VS_V4 = [
    ("Banana, VLASH", (7, 20, 2.0), (13, 20, 2.5)),
    ("Carrot, VLASH", (3, 10, 1.7), (3, 10, 1.4)),
    ("Pull-Back Car, F2F-AP", (8, 20, 2.3), (8, 20, 2.5)),
    ("Length Meas., Asyn. UMI", (3, 10, 2.2), (5, 10, 2.4)),
]
LAT_V1 = {"300ms": (6, 10, 7.4), "400ms": (6, 10, 7.9), "500ms": (5, 10, 8.5), "600ms": (2, 10, 11.2)}
LAT_V4 = {"300ms": (12, 20, 7.5), "500ms": (10, 20, 8.5), "600ms": (4, 20, 11.2)}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, run)
    return adj.tolist()


def main():
    res = {}

    # V1
    v1 = {}
    for m, rows in T1.items():
        v1[m] = []
        for t, (k, n, et) in zip(TASKS, rows):
            lo, hi = wilson(k, n)
            v1[m].append({"task": t, "k": k, "n": n, "sr": round(100 * k / n, 1),
                          "wilson_lo": round(100 * lo, 1), "wilson_hi": round(100 * hi, 1), "et": et})
    res["V1_table1_wilson"] = v1

    # V2
    v2, flat = {}, []
    for b in BASELINES:
        v2[b] = []
        for i, t in enumerate(TASKS):
            kf, nf, _ = T1["F2F-AP"][i]
            kb, nb, _ = T1[b][i]
            p = fisher(kf, nf, kb, nb)
            v2[b].append({"task": t, "f2f": f"{kf}/{nf}", "base": f"{kb}/{nb}", "p": p})
            flat.append(p)
    adj = holm(flat)
    j = 0
    for b in BASELINES:
        for row in v2[b]:
            row["p_holm28"] = adj[j]
            j += 1
    for b in BASELINES:
        ph = holm([r["p"] for r in v2[b]])
        for r, a in zip(v2[b], ph):
            r["p_holm7_within_baseline"] = a
    res["V2_fisher_per_task"] = v2
    res["V2_summary"] = {
        "n_tests": len(flat),
        "raw_p_lt_0.05": int(sum(p < 0.05 for p in flat)),
        "holm_lt_0.05": int(sum(a < 0.05 for a in adj)),
        "holm7_within_baseline_lt_0.05": int(sum(r["p_holm7_within_baseline"] < 0.05 for b in BASELINES for r in v2[b])),
    }

    # V3
    v3 = {}
    for m, rows in T1.items():
        K = sum(r[0] for r in rows)
        N = sum(r[1] for r in rows)
        lo, hi = wilson(K, N)
        v3[m] = {"k": K, "n": N, "sr": round(100 * K / N, 1), "wilson": [round(100 * lo, 1), round(100 * hi, 1)]}
    mh = {}
    for b in BASELINES:
        tables = []
        for i in range(len(TASKS)):
            kf, nf, _ = T1["F2F-AP"][i]
            kb, nb, _ = T1[b][i]
            tables.append(np.array([[kf, nf - kf], [kb, nb - kb]], dtype=float))
        # 성공이 0/0인 층(둘 다 0)은 정보가 없다. 층이 모두 퇴화하면 건너뜀
        st = StratifiedTable(tables)
        ci = st.oddsratio_pooled_confint()
        mh[b] = {"OR_pooled": float(st.oddsratio_pooled), "ci95": [float(ci[0]), float(ci[1])],
                 "p_mh": float(st.test_null_odds(correction=True).pvalue),
                 "breslow_day_p": float(st.test_equal_odds(adjust=True).pvalue)}
    res["V3_pooled"] = v3
    res["V3_mantel_haenszel"] = mh

    # V4 최고 기준선 대비 배수
    v4 = []
    for i, t in enumerate(TASKS):
        kf, nf, _ = T1["F2F-AP"][i]
        best = max(BASELINES, key=lambda b: T1[b][i][0] / T1[b][i][1])
        kb, nb, _ = T1[best][i]
        ratio = (kf / nf) / (kb / nb) if kb else None
        v4.append({"task": t, "f2f": round(kf / nf, 3), "best_base": best, "best_sr": round(kb / nb, 3),
                   "ratio": None if ratio is None else round(ratio, 2)})
    res["V4_best_baseline_ratio"] = v4
    claim_tasks = [0, 3, 5]  # 본문: 해머, 풀백 카, 병
    claim = {}
    for i in claim_tasks:
        kf, nf, _ = T1["F2F-AP"][i]
        claim[TASKS[i]] = {b: round((kf / nf) / (T1[b][i][0] / T1[b][i][1]), 2) if T1[b][i][0] else None
                           for b in BASELINES}
    res["V4_twofold_claim_by_baseline"] = claim
    # v1 기준선 3종(RTC 없음)과 같은 비교
    res["V4_twofold_claim_v1_baselines_hammer"] = round((15 / 20) / (7 / 20), 2)

    # V5 ET
    v5 = {}
    for b in ["Syn. UMI", "Asyn. UMI", "RTC", "VLASH"]:
        rows = []
        for i, t in enumerate(TASKS):
            ef, eb = T1["F2F-AP"][i][2], T1[b][i][2]
            if ef is not None and eb is not None:
                rows.append({"task": t, "f2f": ef, "base": eb, "ratio_f2f_over_base": round(ef / eb, 2)})
        v5[b] = {"rows": rows, "f2f_faster": sum(r["ratio_f2f_over_base"] < 1 for r in rows),
                 "f2f_slower": sum(r["ratio_f2f_over_base"] > 1 for r in rows),
                 "n_compared": len(rows)}
    res["V5_et"] = v5

    # V6 절제
    full = T2["Full F2F-AP"]
    v6 = {}
    for name, (k, n, et) in T2.items():
        if name == "Full F2F-AP":
            continue
        lo, hi = wilson(k, n)
        v6[name] = {"k": k, "n": n, "sr": round(100 * k / n, 1), "wilson": [round(100 * lo, 1), round(100 * hi, 1)],
                    "fisher_vs_full": fisher(full[0], full[1], k, n), "et": et}
    res["V6_ablation_vs_full"] = v6
    # 지연 추세: 300, 375(전체), 500, 600 ms (전체의 실제 총지연을 375 ms로 둔 가정)
    xs = np.array([300, 375, 500, 600], dtype=float)
    ks = np.array([12, 11, 10, 4])
    ns = np.array([20, 20, 20, 20])
    pbar = ks.sum() / ns.sum()
    xbar = (ns * xs).sum() / ns.sum()
    num = (ks - ns * pbar) @ (xs - xbar)
    den = math.sqrt(pbar * (1 - pbar) * (ns * (xs - xbar) ** 2).sum())
    z = num / den
    res["V6_latency_trend"] = {"x_ms": xs.tolist(), "k": ks.tolist(), "cochran_armitage_z": float(z),
                              "p_two_sided": float(2 * stats.norm.sf(abs(z))),
                              "note": "전체 모델의 총지연을 375 ms(논문 측정 합)로 가정한 추세 검정. 논문은 '300ms Latency'가 무엇을 바꾼 것인지 정의하지 않는다."}
    res["V6_600_vs_300"] = fisher(12, 20, 4, 20)

    # V7 지연 산술
    do, di, dc, dt = 0.125, 0.200, 0.050, 0.1
    res["V7_latency"] = {
        "total_upper_s": do + di + dc, "H_upper": math.ceil((do + di + dc) / dt - 1e-9),
        "H_if_dc_zero": math.ceil((do + di) / dt - 1e-9),
        "H_range_total_s": "0.3 < total <= 0.4 이면 H = 4",
        "H_for_ablation_300ms": math.ceil(0.300 / dt - 1e-9), "H_for_500ms": math.ceil(0.5 / dt - 1e-9),
        "H_for_600ms": math.ceil(0.6 / dt - 1e-9),
        "paper_inference_hz": 5, "paper_inference_period_s": 0.2,
        "code_f2f_steps_per_inference": 4, "code_f2f_cycle_s": 4 * dt, "code_f2f_hz": 1 / (4 * dt),
        "code_baseline_steps_per_inference": 2, "code_baseline_hz": 1 / (2 * dt),
        "pullback_window_s": 0.05 / 0.25, "pullback_window_note": "유효 파지 영역 5 cm / 평균 속도 25 cm/s = 0.2 s",
    }

    # V8 버전 간 변화
    v8 = []
    for name, (k1, n1, e1), (k4, n4, e4) in V1_VS_V4:
        v8.append({"cell": name, "v1_v2": f"{k1}/{n1}, {e1}s", "v4": f"{k4}/{n4}, {e4}s",
                   "success_changed": (k1, n1) != (k4, n4)})
    res["V8_changed_cells"] = v8
    res["V8_banana_f2f_vs_vlash"] = {"v1_v2": fisher(13, 20, 7, 20), "v4": fisher(13, 20, 13, 20)}
    lat = []
    for key in ("300ms", "500ms", "600ms"):
        k1, n1, e1 = LAT_V1[key]
        k4, n4, e4 = LAT_V4[key]
        lat.append({"latency": key, "v1": f"{k1}/{n1}, {e1}s", "v4": f"{k4}/{n4}, {e4}s",
                    "exactly_doubled": (k4 == 2 * k1 and n4 == 2 * n1)})
    res["V8_latency_rows"] = lat
    # v4 값이 독립 재실험 20회라면 v1 성공 수의 정확한 2배가 우연히 3개 모두 나올 확률 (p = k1/n1 가정)
    prob = 1.0
    for key in ("300ms", "500ms", "600ms"):
        k1, n1, _ = LAT_V1[key]
        prob *= stats.binom.pmf(2 * k1, 20, k1 / n1)
    res["V8_prob_all_three_doubled_if_independent_rerun"] = prob
    res["V8_latency_wilson_if_n10"] = {k: [round(100 * x, 1) for x in wilson(v[0], v[1])] for k, v in LAT_V1.items()}

    # V9 검정력
    def power(p1, p2, n, sims=20000, seed=0):
        rng = np.random.default_rng(seed)
        a = rng.binomial(n, p1, sims)
        b = rng.binomial(n, p2, sims)
        cnt = 0
        for x, y in zip(a, b):
            if fisher(int(x), n, int(y), n) < 0.05:
                cnt += 1
        return cnt / sims

    # 계산량을 줄이려고 sims를 줄임
    v9 = {}
    for n in (20, 40, 80, 160):
        v9[str(n)] = power(0.75, 0.60, n, sims=1500)
    res["V9_power_75_vs_60"] = v9
    res["V9_note"] = "참 성공률 75% 대 60%일 때 Fisher 양측 5% 검정력 (모의 1,500회)."

    # K1 분모
    denoms = {m: [r[1] for r in rows] for m, rows in T1.items()}
    res["K1_denominators"] = denoms
    res["K1_note"] = "Carrot과 Length Meas.에서 RTC만 20회, 나머지는 10회다 (가능한 해석: RTC가 나중에 추가되며 20회로 수행)."
    # 정수 가능성: 표시 퍼센트는 없으므로 ET만 확인

    # K2 표 1과 표 2의 같은 값
    res["K2_same_cells"] = {
        "Bottle F2F-AP table1": "11/20, 7.9s", "Full F2F-AP table2": "11/20, 7.9s", "equal": True,
        "Bottle Asyn. UMI (v1 Fig.8)": "2/20, 9.1s", "table1 Bottle Asyn. UMI": "2/20, 9.1s",
        "MF-UMI v1": "1/20, 8.7s", "MF-UMI table2": "1/20, 8.7s",
    }

    # K3 데이터 양
    res["K3_data_amount"] = {
        "minutes_total": 10, "traj_range": [100, 200],
        "sec_per_traj_range": [10 * 60 / 200, 10 * 60 / 100],
        "bottle_robot_et_range_s": [7.6, 9.2],
        "bottle_100traj_at_et_min": 100 * 7.6 / 60,
        "gripper_free_extra_dataset": "B.3은 같은 크기의 장애물 없는 영상을 따로 모은다고 하므로 총량은 약 2배",
    }

    # K4 대조 손실 마스크가 실제로 닿는 빈도 (공식 코드: 배치 32 무작위, 같은 에피소드 & |step diff| < 10 원시 프레임)
    def mask_stats(n_ep, ep_len_frames, thr=10, batch=32, sims=20000, seed=1):
        rng = np.random.default_rng(seed)
        masked = []
        for _ in range(sims):
            ep = rng.integers(0, n_ep, batch)
            st = rng.integers(0, ep_len_frames, batch)
            same = ep[:, None] == ep[None, :]
            close = np.abs(st[:, None] - st[None, :]) < thr
            m = same & close
            np.fill_diagonal(m, False)
            masked.append(m.sum())
        masked = np.array(masked)
        return {"mean_masked_entries_per_batch": float(masked.mean()),
                "frac_batches_with_any_masked": float((masked > 0).mean())}

    res["K4_mask_effect"] = {
        "ep100_len240": mask_stats(100, 240), "ep200_len240": mask_stats(200, 240),
        "ep100_len480": mask_stats(100, 480),
        "note": "공식 코드의 temporal mask(기본 ignore_time_diff_threshold=10, 원시 프레임)는 무작위 32개 배치에서 거의 작동하지 않는다. 100–200개 에피소드, 에피소드당 240–480프레임 가정(영상 약 4–8초).",
    }

    # K5 예측기 파라미터 수 (공식 코드 구조 기준 해석적 계산)
    d, heads, depth, mult = 384, 8, 8, 4
    dim_head = d // heads
    attn = d * d + d * (2 * dim_head) + d * d  # to_q, to_kv(다중 질의, 키·값 한 헤드), out
    ff = 2 * d * d * mult
    ln = 2 * d  # 감마(가중치)만 학습, 베타는 버퍼 => d
    layer = attn + ff + d * 2  # 정규화 두 개(감마만): 2d
    transformer = layer * depth
    patch = 16 * 16 * 3 * d + d
    img_dec = d * (3 * 16 * 16) + 3 * 16 * 16
    r, T = 2, 6
    K2 = (2 * r + 1) ** 2
    qk = 2 * (d * d + d)
    pre_conv_in = d + (T - 1) * K2
    pre_conv = pre_conv_in * 256 * 9 + 256 + 2 * 256
    up1 = 256 * 128 * 16 + 128 + 2 * 128
    up2 = 128 * 64 * 16 + 64 + 2 * 64
    hm_head = 64 + 1
    reg_in = d + (T - 1) * K2 + (T - 1) * 2
    reg = reg_in * 512 + 512 + 2 * 512 + 512 * 256 + 256 + 256 * 4 + 4
    total = transformer + patch + img_dec + qk + pre_conv + up1 + up2 + hm_head + reg + d + d  # mask_token, ...
    res["K5_predictor_params"] = {"total_millions": total / 1e6, "fp32_MB": total * 4 / 1e6, "fp16_MB": total * 2 / 1e6,
                                  "repo_ckpt_MB": 30.658434,
                                  "note": "공식 체크포인트 크기 30.7 MB는 파라미터 약 15M을 16비트로 저장한 크기와 맞는다(32비트면 약 60 MB). 정적 계산이며 체크포인트를 열지 않았다."}

    # K6 정책 쪽 크기 (참고): ViT-B/16 ~86M + U-Net
    res["K6_note"] = "정책 인코더는 CLIP ViT-B/16(약 86M, 학습), U-Net은 down_dims 256/512/1024. 예측기는 동결."

    # K7 추론 1회 연산량 대략 (예측기 2회 통과 + 정책 인코더 2프레임). U-Net은 제외. 정적 어림 계산
    tokens = 6 * 14 * 14
    lin_flops = 2 * (attn + ff) * depth * tokens
    att_flops = depth * 4 * tokens * tokens * d
    conv_flops = 2 * (pre_conv_in * 256 * 9) * 14 * 14
    pred_pass = lin_flops + att_flops + conv_flops
    vit_b = 17.6e9  # ViT-B/16 224x224 한 장 (일반적으로 알려진 값)
    res["K7_inference_flops_rough"] = {
        "predictor_one_pass_GFLOPs": pred_pass / 1e9, "predictor_two_passes_GFLOPs": 2 * pred_pass / 1e9,
        "policy_encoder_two_frames_GFLOPs": 2 * vit_b / 1e9,
        "sum_without_unet_GFLOPs": (2 * pred_pass + 2 * vit_b) / 1e9,
        "note": "예측기는 추론마다 관측 지평선 2개에 대해 두 번 통과한다(predict_action의 for 루프). U-Net 16 DDIM 스텝은 제외. 어림 계산이며 측정이 아니다.",
    }

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    # 요약 출력
    print("V3 pooled:", {m: (v["k"], v["n"], v["sr"]) for m, v in v3.items()})
    print("V2:", res["V2_summary"])
    for b in BASELINES:
        print(b, [(r["task"][:6], round(r["p"], 3)) for r in v2[b]])
    print("MH:", {b: (round(v["OR_pooled"], 2), [round(x, 2) for x in v["ci95"]], "%.2e" % v["p_mh"], round(v["breslow_day_p"], 3)) for b, v in mh.items()})
    print("holm7:", {b: [(r["task"][:5], round(r["p_holm7_within_baseline"], 3)) for r in v2[b] if r["p_holm7_within_baseline"] < 0.05] for b in BASELINES})
    print("fisher MF/RTC-MF:", {k: "%.2e" % v["fisher_vs_full"] for k, v in v6.items() if k in ("MF-UMI", "RTC-MF-UMI")})
    print("V4 twofold:", claim)
    print("V5:", {b: (v["f2f_faster"], v["f2f_slower"], v["n_compared"]) for b, v in v5.items()})
    print("V6:", {k: (v["k"], round(v["fisher_vs_full"], 3)) for k, v in v6.items()})
    print("V6 trend:", res["V6_latency_trend"]["cochran_armitage_z"], res["V6_latency_trend"]["p_two_sided"], res["V6_600_vs_300"])
    print("V8:", res["V8_banana_f2f_vs_vlash"], res["V8_prob_all_three_doubled_if_independent_rerun"])
    print("V9:", v9)
    print("K4:", res["K4_mask_effect"])
    print("K5:", res["K5_predictor_params"])
    print("K7:", res["K7_inference_flops_rough"])


if __name__ == "__main__":
    main()
