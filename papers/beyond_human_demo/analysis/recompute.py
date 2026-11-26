"""GLIDE (arXiv:2609.24996 v1) 표 1·2의 수치 재계산과 일관성 검사. 결과는 recompute_results.json.

실행: python3 analysis/recompute.py   (numpy, scipy; 수 초)

입력은 논문 표 1·2 (본문 4.2, 4.3절)를 옮긴 값과, 공개 데이터셋 카드의 메타데이터(에피소드 수, 프레임 수, fps)이다.
모든 시행 수는 논문에 "10회"로 적혀 있다(4.1절 Metrics, 4.2절 M = 10, 표 2 캡션).
검사 ID는 V1–V8 (논문 수치 재계산)이다.
"""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
N = 10  # 모든 셀의 시행 수 (논문 4.1, 4.2, 표 2 캡션)

TASKS = ["tomato", "marker", "wine"]
# ---- 표 1 (데이터 수집) : 방법별 성공 횟수 k/10, 보조 지표 (tomato: 평균 남은 토마토, marker·wine: 분당 처리량)
T1 = {
    "OpenTV":  {"tomato": (0, 0.3), "marker": (1, 0.5), "wine": (0, 0.0)},
    "Manual":  {"tomato": (2, 5.4), "marker": (6, 2.8), "wine": (6, 0.6)},
    "G(0)":    {"tomato": (6, 7.4), "marker": (7, 2.1), "wine": (0, 0.0)},
    "G(1)":    {"tomato": (6, 7.1), "marker": (6, 2.6), "wine": (0, 0.0)},
    "G*":      {"tomato": (7, 6.7), "marker": (9, 3.9), "wine": (9, 0.9)},
}
# ---- 표 2 (정책 배치): 행 순서는 논문 표의 위에서 아래. 데이터 열의 아이콘은 텍스트에 없어
#      4.3절 본문(Vanilla 0%·20%, success-only 0→60 등)으로 복원했다.
T2_ROWS = ["Vanilla/unguarded", "Vanilla/guarded", "Success-only/unguarded", "Success-only/guarded",
           "Mixed/unguarded", "Mixed/guarded"]
T2 = {
    "tomato": [(0, 0.1), (2, 2.4), (0, 3.0), (6, 6.2), (0, 3.5), (7, 6.3)],
    "marker": [(0, 0.0), (2, 1.1), (2, 1.0), (5, 2.3), (4, 2.0), (6, 2.9)],
    "wine":   [(0, 0.0), (0, 0.0), (5, 0.6), (4, 0.4), (4, 0.5), (6, 0.5)],
}
# ---- 공개 데이터셋 카드 (huggingface 데이터셋 yuchensong/glide_data, README 표, 메타데이터만 읽음)
HF = {
    "tomato_mixed": dict(eps=30, frames=13170, fps=30), "tomato_success": dict(eps=19, frames=8508, fps=30),
    "marker_mixed": dict(eps=30, frames=12769, fps=30), "marker_success": dict(eps=22, frames=9216, fps=30),
    "wine_mixed": dict(eps=60, frames=162158, fps=45), "wine_success": dict(eps=30, frames=83240, fps=45),
}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def cmh(tables):
    """층화 2x2 (a, b, c, d)의 Mantel-Haenszel 공통 오즈비와 연속성 보정 카이제곱."""
    num = den = 0.0
    sa = ea = va = 0.0
    for a, b, c, d in tables:
        n = a + b + c + d
        num += a * d / n
        den += b * c / n
        sa += a
        ea += (a + b) * (a + c) / n
        va += (a + b) * (c + d) * (a + c) * (b + d) / (n * n * (n - 1))
    chi = (abs(sa - ea) - 0.5) ** 2 / va
    return dict(or_mh=num / den if den else float("inf"), chi2=chi, p=float(stats.chi2.sf(chi, 1)))


def cochran_armitage(ks, ns, scores):
    ks, ns, s = np.array(ks, float), np.array(ns, float), np.array(scores, float)
    p = ks.sum() / ns.sum()
    t = (s * (ks - ns * p)).sum()
    var = p * (1 - p) * ((ns * s * s).sum() - (ns * s).sum() ** 2 / ns.sum())
    z = t / np.sqrt(var)
    return float(z), float(2 * stats.norm.sf(abs(z)))


def main():
    out = {}

    # ---- V1: 표 1·2의 모든 성공률이 10회 정수 횟수로 설명되는지, Wilson 95% 구간
    v1 = {"table1": {}, "table2": {}}
    for m, row in T1.items():
        v1["table1"][m] = {t: dict(k=k, n=N, rate=k / N, wilson=[round(x, 3) for x in wilson(k, N)]) for t, (k, _) in row.items()}
    for t in TASKS:
        v1["table2"][t] = {r: dict(k=k, n=N, rate=k / N, wilson=[round(x, 3) for x in wilson(k, N)])
                           for r, (k, _) in zip(T2_ROWS, T2[t])}
    out["V1_wilson"] = v1

    # ---- V2: 표 1 핵심 비교 (Fisher 양측), 과제별과 Holm(3과제)
    cmp1 = {}
    for a, b in [("G*", "OpenTV"), ("G*", "Manual"), ("Manual", "OpenTV"), ("G*", "G(0)"), ("G(0)", "Manual")]:
        ps = [fisher(T1[a][t][0], N, T1[b][t][0], N) for t in TASKS]
        cmp1[f"{a} vs {b}"] = dict(counts={t: [T1[a][t][0], T1[b][t][0]] for t in TASKS},
                                   p=dict(zip(TASKS, [round(p, 4) for p in ps])),
                                   p_holm3=dict(zip(TASKS, [round(p, 4) for p in holm(ps)])))
    # 세 과제를 합친 비교(30 대 30). 과제 이질성은 CMH로 따로 본다
    for a, b in [("G*", "OpenTV"), ("G*", "Manual")]:
        ka = sum(T1[a][t][0] for t in TASKS)
        kb = sum(T1[b][t][0] for t in TASKS)
        cmp1[f"{a} vs {b}"]["pooled"] = dict(k=[ka, kb], n=[30, 30], p_fisher=round(fisher(ka, 30, kb, 30), 5),
                                              cmh=cmh([(T1[a][t][0], N - T1[a][t][0], T1[b][t][0], N - T1[b][t][0]) for t in TASKS]))
    out["V2_table1_tests"] = cmp1

    # ---- V3: 표 2 비교 (Fisher). 같은 정책으로 guarded/unguarded는 쌍(같은 과제 순서)이라 짝 검정이 맞지만 시행별 결과가 없어 독립 Fisher로 대신한다.
    t2 = {}
    for t in TASKS:
        k = [x[0] for x in T2[t]]
        t2[t] = {
            "guard effect Vanilla": [k[0], k[1], round(fisher(k[1], N, k[0], N), 4)],
            "guard effect Success-only": [k[2], k[3], round(fisher(k[3], N, k[2], N), 4)],
            "guard effect Mixed": [k[4], k[5], round(fisher(k[5], N, k[4], N), 4)],
            "Mixed vs Success-only (guarded)": [k[3], k[5], round(fisher(k[5], N, k[3], N), 4)],
            "Mixed vs Success-only (unguarded)": [k[2], k[4], round(fisher(k[4], N, k[2], N), 4)],
            "Mixed/guarded vs Vanilla/guarded": [k[1], k[5], round(fisher(k[5], N, k[1], N), 4)],
        }
    pooled = {}
    for name, (i0, i1) in {"guard effect Vanilla": (0, 1), "guard effect Success-only": (2, 3), "guard effect Mixed": (4, 5)}.items():
        k0 = sum(T2[t][i0][0] for t in TASKS)
        k1 = sum(T2[t][i1][0] for t in TASKS)
        pooled[name] = dict(k=[k0, k1], n=[30, 30], p_fisher=round(fisher(k1, 30, k0, 30), 5),
                            cmh=cmh([(T2[t][i1][0], N - T2[t][i1][0], T2[t][i0][0], N - T2[t][i0][0]) for t in TASKS]))
    k_s = sum(T2[t][3][0] for t in TASKS)
    k_m = sum(T2[t][5][0] for t in TASKS)
    pooled["Mixed vs Success-only (guarded)"] = dict(k=[k_s, k_m], n=[30, 30], p_fisher=round(fisher(k_m, 30, k_s, 30), 5),
                                                    cmh=cmh([(T2[t][5][0], N - T2[t][5][0], T2[t][3][0], N - T2[t][3][0]) for t in TASKS]))
    out["V3_table2_tests"] = dict(per_task=t2, pooled=pooled)

    # ---- V4: 본문 합계와 표의 일치 (19, 22개 성공, 30 에피소드, 위치 비교)
    out["V4_episode_sums"] = dict(
        tomato_success_sum=sum(T1[m]["tomato"][0] for m in ["G(0)", "G(1)", "G*"]),
        marker_success_sum=sum(T1[m]["marker"][0] for m in ["G(0)", "G(1)", "G*"]),
        text_claim=dict(tomato=19, marker=22, episodes=30),
        wine_pool=dict(episodes=60, successes=30, success_rate=0.5, vs_table1_G_star="9/10",
                       p_fisher_9of10_vs_30of60=round(fisher(9, 10, 30, 60), 4)),
        wine_wilson_30of60=[round(x, 3) for x in wilson(30, 60)],
    )

    # ---- V5: 토마토 평균 남은 개수의 정수 가능 범위 (성공 = 8개 모두 유지 가정; 이 가정은 논문에 없다)
    v5 = {}
    rows = [(f"T1 {m}", T1[m]["tomato"][0], T1[m]["tomato"][1]) for m in T1]
    rows += [(f"T2 {r}", T2["tomato"][i][0], T2["tomato"][i][1]) for i, r in enumerate(T2_ROWS)]
    for name, k, avg in rows:
        total = round(avg * N)  # 소수 첫째 자리 반올림이라 합은 ±0.5 범위
        lo, hi = 8 * k, 8 * k + 7 * (N - k)
        ok = (lo - 0.5) <= avg * N <= (hi + 0.5)
        fail_mean = (avg * N - 8 * k) / (N - k) if k < N else None
        v5[name] = dict(k=k, avg_left=avg, total_left_approx=total, feasible_range=[lo, hi], inside=bool(ok),
                        implied_mean_left_in_failed_trials=None if fail_mean is None else round(fail_mean, 2))
    out["V5_tomato_feasibility_success_means_all_8_kept"] = v5

    # ---- V6: 처리량 = 성공 횟수 / 경과 시간(분) 이라 가정할 때 10회 시행의 함의 경과 시간 (반올림 구간 포함)
    v6 = {}
    def implied(k, thr):
        if k == 0 or thr == 0:
            return None
        return [round(k / (thr + 0.05), 2), round(k / (thr - 0.05), 2)] if thr > 0.05 else None
    for m in T1:
        for t in ("marker", "wine"):
            k, thr = T1[m][t]
            v6[f"T1 {m} {t}"] = dict(k=k, thr=thr, implied_total_min_10trials=implied(k, thr))
    for i, r in enumerate(T2_ROWS):
        for t in ("marker", "wine"):
            k, thr = T2[t][i]
            v6[f"T2 {r} {t}"] = dict(k=k, thr=thr, implied_total_min_10trials=implied(k, thr))
    out["V6_throughput_implied_minutes"] = v6

    # ---- V7: 데이터셋 카드와 논문의 에피소드·시간 대조
    v7 = {}
    for name, d in HF.items():
        sec = d["frames"] / d["fps"]
        v7[name] = dict(eps=d["eps"], minutes=round(sec / 60, 2), mean_ep_seconds=round(sec / d["eps"], 2))
    for t in TASKS:
        m, s = HF[f"{t}_mixed"], HF[f"{t}_success"]
        fail_eps = m["eps"] - s["eps"]
        fail_sec = m["frames"] / m["fps"] - s["frames"] / s["fps"]
        v7[f"{t}_failed_subset"] = dict(eps=fail_eps, mean_ep_seconds=round(fail_sec / fail_eps, 2))
    # 마커: 세 라운드 처리량에서 함의되는 경과 시간 합 대 카드의 30에피소드 기록 시간
    mk = {"G(0)": 7, "G(1)": 6, "G*": 9}
    thr = {"G(0)": 2.1, "G(1)": 2.6, "G*": 3.9}
    lo = sum(mk[m] / (thr[m] + 0.05) for m in mk)
    hi = sum(mk[m] / (thr[m] - 0.05) for m in mk)
    v7["marker_rounds_implied_minutes_sum"] = [round(lo, 2), round(hi, 2)]
    v7["marker_card_recorded_minutes_30eps"] = round(HF["marker_mixed"]["frames"] / HF["marker_mixed"]["fps"] / 60, 2)
    v7["wine_G_star_implied_s_per_trial"] = round(9 / 0.9 * 60 / 10, 1)
    v7["wine_card_mean_ep_s_all60"] = round(HF["wine_mixed"]["frames"] / HF["wine_mixed"]["fps"] / 60, 2) * 60 / HF["wine_mixed"]["eps"]
    out["V7_dataset_card_crosscheck"] = v7

    # ---- V8: 시행 수와 검정력 (Manual 60% 대 G* 90%, 세 과제의 Manual 대 G* 관측 차이)
    def power(p0, p1, n, alpha=0.05):
        k0 = np.arange(n + 1)
        k1 = np.arange(n + 1)
        pm0 = stats.binom.pmf(k0, n, p0)
        pm1 = stats.binom.pmf(k1, n, p1)
        pw = 0.0
        for a in k0:
            for b in k1:
                if fisher(int(b), n, int(a), n) < alpha:
                    pw += pm0[a] * pm1[b]
        return float(pw)

    v8 = {}
    for p0, p1 in [(0.6, 0.9), (0.2, 0.7)]:
        v8[f"{p0}_vs_{p1}"] = {str(n): round(power(p0, p1, n), 3) for n in (10, 20, 30, 40)}
    out["V8_power_fisher_exact"] = v8

    (ROOT / "analysis" / "recompute_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote recompute_results.json")


if __name__ == "__main__":
    main()
