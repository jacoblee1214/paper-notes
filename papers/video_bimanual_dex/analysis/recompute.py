"""논문 Table I-III 수치 재계산 (표준 라이브러리만 사용, CPU 즉시 실행)."""
import json, math
from pathlib import Path

OUT = Path(__file__).resolve().parent / "recompute_results.json"
from math import comb

def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

def fisher_two_sided(a, b, c, d):
    # 2x2 table [[a,b],[c,d]], two-sided exact p
    n1, n2, m = a + b, c + d, a + c
    n = n1 + n2
    def p_of(x):
        return comb(n1, x) * comb(n2, m - x) / comb(n, m)
    lo, hi = max(0, m - n2), min(n1, m)
    p0 = p_of(a)
    return min(1.0, sum(p_of(x) for x in range(lo, hi + 1) if p_of(x) <= p0 * (1 + 1e-9)))

# Table I: 평균 성공률 (%), 태스크당 150 에피소드 (3 영상 x 50)
N1 = 150
T1 = {
    "H2S2R":          [0.0, 3.0, 0.0, 7.3],
    "ViViDex":        [0.0, 6.0, 0.0, 0.0],
    "w/o HP & Conf.": [16.0, 50.7, 6.7, 98.7],
    "w/o HP":         [41.3, 64.7, 16.7, 99.3],
    "PriDex":         [98.0, 44.7, 40.0, 46.0],
}
tasks = ["Pour", "Sweep", "Unscrew", "Clean"]
out = {"table1": {}, "table1_checks": {}, "table2": {}, "table3": {}, "pairwise": {}}
for m, v in T1.items():
    counts = [round(x / 100 * N1, 2) for x in v]
    out["table1"][m] = {"mean_recomputed": round(sum(v) / 4, 3), "per_task_count_if_150": counts}
print("== Table I 평균 재계산 ==")
for m, v in T1.items():
    print(f"{m:16s} mean={sum(v)/4:.3f}  counts(150)={[round(x/100*N1,2) for x in v]}")

# 분수로 맞지 않는 값 확인: 150의 배수가 아닌 백분율
print("\n== 150 에피소드에서 정수가 되지 않는 칸 ==")
for m, v in T1.items():
    for t, x in zip(tasks, v):
        k = x / 100 * N1
        if abs(k - round(k)) > 0.05:
            print(f"  {m} {t} {x}% -> {k:.2f} episodes (정수 아님)")

# Table I: PriDex vs w/o HP 태스크별 Fisher 정확검정 (pooled 150 vs 150)
print("\n== PriDex vs w/o HP (태스크별, Fisher exact, 150 vs 150) ==")
pd_, wo = T1["PriDex"], T1["w/o HP"]
for t, a, b in zip(tasks, pd_, wo):
    ka, kb = round(a / 100 * N1), round(b / 100 * N1)
    p = fisher_two_sided(ka, N1 - ka, kb, N1 - kb)
    lo, hi = wilson(ka, N1)
    lo2, hi2 = wilson(kb, N1)
    print(f"  {t:8s} PriDex {ka}/150 CI[{lo*100:.1f},{hi*100:.1f}]  w/o HP {kb}/150 CI[{lo2*100:.1f},{hi2*100:.1f}]  p={p:.4f}")
    out["pairwise"][t] = {"PriDex": ka, "wo_HP": kb, "fisher_p": round(p, 5)}

# 평균 차이: 4개 태스크 평균 (unweighted) 의 부트스트랩 대신 태스크별 pooled 정확 합계 기준 Wilson 근사
print("\n== 전체 pooled (600 에피소드) PriDex vs w/o HP ==")
kP = sum(round(x / 100 * N1) for x in pd_)
kW = sum(round(x / 100 * N1) for x in wo)
p = fisher_two_sided(kP, 600 - kP, kW, 600 - kW)
print(f"  PriDex {kP}/600 = {kP/6:.2f}%  w/o HP {kW}/600 = {kW/6:.2f}%  Fisher p={p:.4f}")
out["pooled_all_tasks"] = {"PriDex": kP, "wo_HP": kW, "fisher_p": round(p, 5)}

# Table II: distilled policy, 150 closed-loop trials per task
T2 = {"w/o HP": [18.0, 72.0, 0.7, 6.0], "PriDex": [24.0, 80.0, 5.3, 16.0]}
print("\n== Table II 평균 재계산 ==")
for m, v in T2.items():
    print(f"{m:8s} mean={sum(v)/4:.3f}  counts(150)={[round(x/100*N1,2) for x in v]}")
print("== Table II 태스크별 Fisher (150 vs 150) ==")
for t, a, b in zip(tasks, T2["PriDex"], T2["w/o HP"]):
    ka, kb = round(a / 100 * N1), round(b / 100 * N1)
    p = fisher_two_sided(ka, N1 - ka, kb, N1 - kb)
    print(f"  {t:8s} PriDex {ka}/150 vs w/o HP {kb}/150  p={p:.4f}")
    out["table2"][t] = {"PriDex": ka, "wo_HP": kb, "fisher_p": round(p, 5)}
kP2 = sum(round(x / 100 * N1) for x in T2["PriDex"])
kW2 = sum(round(x / 100 * N1) for x in T2["w/o HP"])
p2 = fisher_two_sided(kP2, 600 - kP2, kW2, 600 - kW2)
print(f"  pooled PriDex {kP2}/600 ({kP2/6:.2f}%) vs w/o HP {kW2}/600 ({kW2/6:.2f}%)  p={p2:.4f}")

# Table III: 실로봇 10 trials
T3 = {"Pour": (2, 5), "Sweep": (3, 8), "Clean": (4, 6)}
print("\n== Table III 실로봇 (10 trials/task) ==")
ro, sr = 0, 0
for t, (a, b) in T3.items():
    ro += a; sr += b
    lo, hi = wilson(a, 10); lo2, hi2 = wilson(b, 10)
    p = fisher_two_sided(a, 10 - a, b, 10 - b)
    print(f"  {t:6s} real-only {a}/10 CI[{lo*100:.0f},{hi*100:.0f}]  sim+real {b}/10 CI[{lo2*100:.0f},{hi2*100:.0f}]  Fisher p={p:.3f}")
p3 = fisher_two_sided(ro, 30 - ro, sr, 30 - sr)
lo, hi = wilson(ro, 30); lo2, hi2 = wilson(sr, 30)
print(f"  pooled real-only {ro}/30 ({ro/30*100:.1f}%) CI[{lo*100:.0f},{hi*100:.0f}]  sim+real {sr}/30 ({sr/30*100:.1f}%) CI[{lo2*100:.0f},{hi2*100:.0f}]  Fisher p={p3:.4f}")
out["table3"] = {"real_only": ro, "sim_real": sr, "fisher_p_pooled": round(p3, 5)}

with open(OUT, "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("\nsaved recompute_results.json")

# 확인 추가: 신뢰도 적응 효과 (w/o HP vs w/o HP & Conf.), 태스크별 Fisher + 4개 검정 Bonferroni
print("\n== 신뢰도 적응 효과: w/o HP vs w/o HP & Conf. (태스크별, Bonferroni x4) ==")
conf_adapt = {}
for t, a, b in zip(tasks, T1["w/o HP"], T1["w/o HP & Conf."]):
    ka, kb = round(a / 100 * N1), round(b / 100 * N1)
    p = fisher_two_sided(ka, N1 - ka, kb, N1 - kb)
    conf_adapt[t] = {"w_o_HP": ka, "w_o_HP_and_Conf": kb, "fisher_p": round(p, 5), "bonf_p": round(min(1.0, 4 * p), 5)}
    print(f"  {t:8s} w/o HP {ka}/150 vs no-conf {kb}/150  p={p:.4f}  Bonf={min(1,4*p):.4f}")
kA = sum(round(x / 100 * N1) for x in T1["w/o HP"]); kB = sum(round(x / 100 * N1) for x in T1["w/o HP & Conf."])
pc = fisher_two_sided(kA, 600 - kA, kB, 600 - kB)
print(f"  pooled {kA}/600 ({kA/6:.2f}%) vs {kB}/600 ({kB/6:.2f}%)  p={pc:.4g}")
out["confidence_adaptation"] = {"per_task": conf_adapt, "pooled": {"w_o_HP": kA, "no_conf": kB, "fisher_p": pc}}

print("\n== 태스크별 Bonferroni x4 (Table II, PriDex vs w/o HP) ==")
for t in tasks:
    p = out["table2"][t]["fisher_p"]
    print(f"  {t:8s} p={p:.4f}  Bonf={min(1,4*p):.4f}")
with open(OUT, "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
