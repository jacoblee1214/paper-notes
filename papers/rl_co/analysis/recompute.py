"""논문 표 1–6, 그림 4·5b·6의 수치를 다시 계산하고 일관성을 검사한다 (numpy, scipy, 수 초).

실행: python3 analysis/recompute.py  → analysis/recompute_results.json
입력은 논문 v5 PDF의 표 값과 그림에서 읽은 값을 옮긴 것이다. 원자료(trial 로그)는 없다.

가정(논문 App. D.1과 본문에서 옮김):
- 한 체크포인트당 2개 세션. 세션당 시도 수는 PnP 32, Push 30, OD 10, CD 10.
- 표 1의 ±는 두 세션 값의 모집단 표준편차(ddof=0) = |r1 - r2| / 2 이다. 그래서 r = 평균 ± ± 값으로 세션별 성공률을 되살린다.
- 평균 Avg는 과제별 세션 값의 단순 평균이다.
- 그림 4의 Unseen States는 4 물체 x 5 자세 = 20 시도(App. D.5)로 본다. Unseen Objects의 시도 수는 적혀 있지 않아 64(2 세션 x 32)로 가정한다(값이 1/64 배수).
- 그림 6은 눈으로 읽었다(±0.025). 시도 수 20으로 본다(OD, 10 시도 x 2 세션).
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
N_SESS = {"PnP": 32, "Push": 30, "OD": 10, "CD": 10}
TASKS = ["PnP", "Push", "OD", "CD"]

# 표 1 (v5). 값 = (평균, ±). 순서: PnP, Push, OD, CD, Avg
T1 = {
    "OpenVLA": {
        "Real-only": [(6.3, 0.0), (20.0, 13.3), (0.0, 0.0), (10.0, 10.0), (9.1, 5.8)],
        "SFT": [(23.4, 4.7), (51.7, 5.0), (0.0, 0.0), (85.0, 5.0), (40.0, 3.7)],
        "RL-Co": [(57.8, 10.9), (68.3, 11.7), (35.0, 15.0), (95.0, 5.0), (64.0, 0.7)],
    },
    "pi05": {
        "Real-only": [(71.9, 9.4), (0.0, 0.0), (0.0, 0.0), (35.0, 15.0), (26.7, 1.4)],
        "SFT": [(68.8, 9.4), (10.0, 3.3), (10.0, 0.0), (95.0, 5.0), (45.9, 4.4)],
        "RL-Co": [(81.3, 9.4), (18.3, 1.7), (65.0, 5.0), (100.0, 0.0), (66.1, 4.0)],
    },
}
# v4(이전 버전) 표 1에서 바뀐 값: 비교용
T1_V4_DIFF = {
    "OpenVLA Real-only Avg": {"v4": (16.5, 13.3), "v5": (9.1, 5.8)},
    "OpenVLA RL-Co PnP": {"v4": (58.8, 10.0), "v5": (57.8, 10.9)},
    "pi05 RL-Co Push": {"v4": (18.4, 1.7), "v5": (18.3, 1.7)},
    "pi05 RL-Co Avg": {"v4": (66.2, 4.0), "v5": (66.1, 4.0)},
}

# 그림 4 (π0.5, PnP 분포 이동). 막대 위 값 = 분포 이동 후 성공률(%)
FIG4 = {
    "Unseen Objects": {"n": 64, "Real-only": 25.0, "SFT": 31.3, "RL-Co": 56.3},
    "Unseen States": {"n": 20, "Real-only": 40.0, "SFT": 55.0, "RL-Co": 70.0},
}
# 표 2, 5, 6, 그림 5b, 표 4 (값과 시도 수)
T2 = {"Heavy DR zero-shot": 10.9, "Cosmos augmentation": 67.2, "RL-Co (PnP)": 81.3}
T5 = {"One-stage": (0.0, None), "RialTo-style": (75.0, 32), "RL-Co (PnP)": (81.3, 32)}
T6 = {"SFT MimicGen": (10.0, 20), "SFT strong rollouts": (20.0, 20), "RL-Co (OD)": (65.0, 20)}
FIG5B = {"no real either stage": 6.3, "no Stage II real": 12.5, "no Stage I real": 40.3, "full RL-Co": 81.3}
TABLE4 = {"SFT co-training": 32.5, "RL-Co": 42.5}  # 시도 수 미기재
# 그림 6(OD, π0.5, 시도 20 가정), 읽은 값
FIG6 = {
    "Real-only": {20: 0.00, 50: 0.05, 100: 0.10, 150: 0.25, 200: 0.50},
    "SFT": {20: 0.05, 50: 0.25, 100: 0.65, 150: 0.70, 200: 0.70},
    "RL-Co": {20: 0.65, 50: 0.55, 100: 0.75, 150: 0.85, 200: 0.85},
}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def fisher(k1, n1, k2, n2):
    table = [[k1, n1 - k1], [k2, n2 - k2]]
    _, p = stats.fisher_exact(table, alternative="two-sided")
    return float(p)


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adj = [None] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def reconstruct(mean, sd, n):
    """세션별 성공 수 (k1, k2)를 전수 검색한다. 평균과 모집단 SD를 보고된 자릿수로 반올림했을 때 일치하는 쌍만 남긴다.

    반환: 첫 번째 후보(k1 <= k2 순), 후보가 있는지 여부, 후보 개수.
    """
    cands = []
    for k1 in range(n + 1):
        for k2 in range(k1, n + 1):
            r1, r2 = 100 * k1 / n, 100 * k2 / n
            m, s = (r1 + r2) / 2, abs(r1 - r2) / 2
            if abs(m - mean) <= 0.0501 and abs(s - sd) <= 0.0501:
                cands.append((k1, k2))
    if not cands:
        return [None, None], False, 0
    return list(cands[0]), True, len(cands)


def main():
    res = {"meta": {"paper": "arXiv:2602.12628v5 (2026-10-22)", "computed": "2026-11-15"}}

    # 1. 표 1 재구성과 Avg 검사
    recon = {}
    consistency = []
    for vla, rows in T1.items():
        for method, cells in rows.items():
            sess = {}
            for t_i, t in enumerate(TASKS):
                mean, sd = cells[t_i]
                ks, ok, ncand = reconstruct(mean, sd, N_SESS[t])
                sess[t] = {"k": ks, "n_per_session": N_SESS[t], "integer_ok": ok, "n_candidates": ncand}
                if not ok:
                    consistency.append(f"표1 {vla} {method} {t}: 정수 성공 수 쌍으로 재현 안 됨 ({mean}±{sd})")
            # Avg 검사: 세션별 과제 평균 -> 평균, 모집단 SD
            r_s = np.array([[sess[t]["k"][s] * 100 / N_SESS[t] for t in TASKS] for s in range(2)])
            assert all(sess[t]["integer_ok"] for t in TASKS), (vla, method)
            avg_s = r_s.mean(axis=1)
            avg_mean, avg_sd = float(avg_s.mean()), float(abs(avg_s[0] - avg_s[1]) / 2)
            rep_mean, rep_sd = cells[4]
            avg_ok = abs(avg_mean - rep_mean) <= 0.06 and abs(avg_sd - rep_sd) <= 0.06
            recon[f"{vla}|{method}"] = {
                "sessions": sess,
                "avg_recomputed": [round(avg_mean, 2), round(avg_sd, 2)],
                "avg_reported": [rep_mean, rep_sd],
                "avg_ok": avg_ok,
            }
            if not avg_ok:
                consistency.append(f"표1 {vla} {method} Avg: 재계산 {avg_mean:.2f}±{avg_sd:.2f} vs 보고 {rep_mean}±{rep_sd}")
    res["table1_reconstruction"] = recon
    res["v4_to_v5_changes"] = T1_V4_DIFF

    # 2. 과제별 2 세션 합산 (trial 수 2n) 과 Wilson, Fisher
    pairs = [("RL-Co", "SFT"), ("RL-Co", "Real-only"), ("SFT", "Real-only")]
    tests = {}
    pvals_main = []
    keys_main = []
    for vla in T1:
        for t in TASKS + ["Pooled"]:
            counts = {}
            for method in T1[vla]:
                if t == "Pooled":
                    k = sum(sum(recon[f"{vla}|{method}"]["sessions"][x]["k"]) for x in TASKS)
                    n = sum(2 * N_SESS[x] for x in TASKS)
                else:
                    k = sum(recon[f"{vla}|{method}"]["sessions"][t]["k"])
                    n = 2 * N_SESS[t]
                counts[method] = (k, n, wilson(k, n))
            res.setdefault("pooled_counts", {})[f"{vla}|{t}"] = {m: {"k": v[0], "n": v[1], "wilson95": v[2]} for m, v in counts.items()}
            for a, b in pairs:
                ka, na, _ = counts[a]
                kb, nb, _ = counts[b]
                p = fisher(ka, na, kb, nb)
                key = f"{vla}|{t}|{a} vs {b}"
                tests[key] = {"k": [ka, kb], "n": [na, nb], "fisher_p": round(p, 6)}
                if a == "RL-Co" and b == "SFT" and t != "Pooled":
                    pvals_main.append(p)
                    keys_main.append(key)
    for key, adj in zip(keys_main, holm(pvals_main)):
        tests[key]["holm_p_over_8"] = round(adj, 6)
    res["tests"] = tests

    # 3. 그림 4 (분포 이동). 막대 값을 시도 수로 바꾼다
    fig4 = {}
    for cond, d in FIG4.items():
        n = d["n"]
        row = {}
        for method in ["Real-only", "SFT", "RL-Co"]:
            k = round(d[method] * n / 100)
            row[method] = {"k": k, "n": n, "check": round(k * 100 / n, 1), "wilson95": wilson(k, n)}
        fig4[cond] = row
        fig4[cond]["p_RL_vs_SFT"] = round(fisher(row["RL-Co"]["k"], n, row["SFT"]["k"], n), 4)
        fig4[cond]["p_RL_vs_Real"] = round(fisher(row["RL-Co"]["k"], n, row["Real-only"]["k"], n), 4)
    res["fig4_generalization"] = fig4

    # 4. 표 2, 5, 6, 그림 5b, 표 4의 산술 검사
    misc = {}
    k_rialto, n5 = round(75.0 * 32 / 100), 32
    misc["table5_RialTo_vs_RLCo_p"] = round(fisher(k_rialto, n5, 26, 32), 4)
    misc["table6"] = {}
    for name, (v, n) in T6.items():
        k = round(v * n / 100)
        misc["table6"][name] = {"k": k, "n": n, "check": round(k * 100 / n, 1)}
    misc["table6"]["p_RLCo_vs_SFT_MimicGen"] = round(fisher(13, 20, 2, 20), 6)
    misc["table6"]["p_RLCo_vs_SFT_strong_rollouts"] = round(fisher(13, 20, 4, 20), 4)
    # 그림 5b: 2 세션 x 32 시도면 성공률은 1/64 (1.5625%) 배수여야 한다
    fig5b = {}
    for name, v in FIG5B.items():
        k64 = v * 64 / 100
        fig5b[name] = {"value": v, "k_of_64": round(k64, 2), "on_grid_1_64": abs(k64 - round(k64)) < 0.05}
    misc["fig5b_grid_check"] = fig5b
    misc["table2_heavyDR_sessions"] = [round(1 / 32 * 100, 2), round(6 / 32 * 100, 2)]
    misc["table2_heavyDR_mean_sd_from_sessions"] = [round((1 / 32 + 6 / 32) * 50, 2), round((6 / 32 - 1 / 32) * 50, 2)]
    # 표 2: PnP, 시도 32 x 2 세션 = 64 (값 1/64 배수로 확인)
    misc["table2_tests"] = {
        "RL-Co 52/64 vs Cosmos 43/64": round(fisher(52, 64, 43, 64), 4),
        "RL-Co 52/64 vs heavy DR 7/64": round(fisher(52, 64, 7, 64), 6),
        "RL-Co 52/64 vs real-only 46/64 (표 1, pi05 PnP)": round(fisher(52, 64, 46, 64), 4),
    }
    # 그림 6: OD, pi05, 시도 20 가정. 그래프에서 읽은 값을 개수로 환산
    fig6_counts = {m: {d: round(v * 20) for d, v in FIG6[m].items()} for m in FIG6}
    misc["fig6_counts"] = fig6_counts
    misc["fig6_tests"] = {
        "RL-Co@20 (13) vs real-only@200 (10)": round(fisher(13, 20, 10, 20), 4),
        "RL-Co@20 (13) vs SFT@200 (14)": round(fisher(13, 20, 14, 20), 4),
        "RL-Co@50 (11) vs SFT@200 (14)": round(fisher(11, 20, 14, 20), 4),
    }
    misc["table4_note"] = "시도 수 미기재. 2.5 단위라 n이 40의 배수일 가능성(가정 n=40)"
    misc["table4_p_if_n40"] = round(fisher(13, 40, 17, 40), 4)
    res["misc"] = misc
    consistency.append("그림 5b: 40.3은 2 세션 x 32 시도(1/64 단위)로 나오지 않음")
    consistency.append("그림 6: SFT co-training 20 demos 값 약 0.05(1/20)와 표 1·6의 10%(2/20)가 다름")

    # 5. 검정력: 같은 시도 수에서 차이를 80%로 잡으려면 팔당 몇 시도인가 (정규 근사, 양측 0.05)
    def n_per_arm(p1, p2, alpha=0.05, power=0.8):
        za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
        pbar = (p1 + p2) / 2
        num = (za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        return math.ceil(num / (p1 - p2) ** 2)

    res["power"] = {
        "OD pi05 65 vs 10 (%)": n_per_arm(0.65, 0.10),
        "PnP pi05 81.3 vs 68.8 (%)": n_per_arm(0.813, 0.688),
        "Push pi05 18.3 vs 10.0 (%)": n_per_arm(0.183, 0.10),
        "OpenVLA avg 64 vs 40 (%)": n_per_arm(0.64, 0.40),
    }

    # 6. 논문 주장의 숫자 확인
    claims = {
        "abstract +24 OpenVLA (RL-Co avg - SFT avg, pp)": round(64.0 - 40.0, 1),
        "abstract +24 OpenVLA (RL-Co avg - real-only avg, pp)": round(64.0 - 9.1, 1),
        "abstract +20 pi05 (RL-Co avg - SFT avg, pp)": round(66.1 - 45.9, 1),
        "abstract +20 pi05 (v4: 66.2 - 45.9, pp)": round(66.2 - 45.9, 1),
    }
    res["claims"] = claims
    res["consistency_flags"] = consistency
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    # 요약 출력
    print("consistency flags:")
    for c in consistency:
        print(" -", c)
    print("Avg checks:", {k: v["avg_ok"] for k, v in recon.items()})
    print("recon avg:", {k: v["avg_recomputed"] for k, v in recon.items()})
    for k in ["OpenVLA|Pooled", "pi05|Pooled"]:
        print(k, res["pooled_counts"][k])
    for key in ["pi05|OD|RL-Co vs SFT", "pi05|OD|RL-Co vs Real-only", "pi05|PnP|RL-Co vs SFT",
                "pi05|Push|RL-Co vs SFT", "OpenVLA|PnP|RL-Co vs SFT", "OpenVLA|OD|RL-Co vs SFT",
                "OpenVLA|Pooled|RL-Co vs SFT", "pi05|Pooled|RL-Co vs SFT"]:
        print(key, tests[key])
    print("fig4:", {c: (fig4[c]["p_RL_vs_SFT"], fig4[c]["p_RL_vs_Real"]) for c in fig4})
    print("misc:", misc["table6"], misc["table4_p_if_n40"], misc["table5_RialTo_vs_RLCo_p"])
    print("power:", res["power"])


if __name__ == "__main__":
    main()
