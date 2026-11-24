"""FM-VLA (arXiv:2607.18231 v1) 논문 표·그림 수치를 다시 계산하고 검정한다.

입력: 논문 표 1, 표 2, 그림 3(막대 위 라벨), 부록 B~E의 값을 아래에 그대로 옮겨 적었다(출처는 주석).
      외부 데이터는 쓰지 않는다. 학습이나 재현은 하지 않는다.
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)

가정(논문에 없는 것은 표시한다)
- 성공률의 분모는 부록 B.2의 시행 수다(과제당 방법마다 18회). 표 1의 괄호 안 성공 수(k/18)를 그대로 쓴다.
- 여러 과제를 하나의 풀로 합치는 검정은 시행이 독립이라고 가정한다. 같은 모델·같은 로봇의 시행이 묶여 있을 수 있어 p값은 낙관적이다.
- 그림 3의 막대 값은 막대 위 라벨을 쓴다(그림 판독 값). 성공 수는 18회 기준으로 역산한다.
- 그림 4의 에피소드 길이는 그림 축에서 읽은 대략 값이다(그림 판독 값).
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
RES = {}
N = 18

# ---------- 표 1 (성공 수 k/18, 보고된 %, 보고된 평균) ----------
TASKS = ["Cups", "Buttons", "Wipe"]
T1 = {
    "pi05": ([13, 2, 0], [72.2, 11.1, 0.0], 27.8),
    "TA-VLA": ([9, 2, 1], [50.0, 11.1, 5.6], 22.2),
    "pi-MEM": ([14, 6, 9], [77.8, 33.3, 50.0], 53.7),
    "force_only": ([10, 0, 4], [55.6, 0.0, 22.2], 25.9),
    "state_only": ([18, 2, 2], [100.0, 11.1, 11.1], 40.7),
    "FM-VLA(GRU)": ([10, 7, 1], [55.6, 38.9, 5.6], 33.3),
    "FM-VLA(Q-Former)": ([18, 3, 10], [100.0, 16.7, 55.6], 57.4),
    "FM-VLA": ([18, 13, 14], [100.0, 72.2, 77.8], 83.3),
}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def newcombe_diff(k1, n1, k2, n2, z=1.959964):
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1, z)
    l2, u2 = wilson(k2, n2, z)
    d = p1 - p2
    return d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]], alternative="two-sided")[1])


def sg(x):
    return float(f"{x:.3g}")


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def r1(x):
    return round(float(x), 1)


# ---------- V1-V3: 표 1 산술 ----------
v1 = []
for m, (ks, pcts, avg) in T1.items():
    rates = [r1(100 * k / N) for k in ks]
    pooled = 100 * sum(ks) / (N * 3)
    mean_of_reported = sum(pcts) / 3
    v1.append({
        "method": m, "k": ks, "rates_recomputed": rates, "rates_reported": pcts,
        "rates_match": rates == pcts,
        "avg_reported": avg, "avg_pooled": r1(pooled), "avg_mean_of_reported_pct": r1(mean_of_reported),
        "avg_match": r1(pooled) == avg,
        "pooled_k": sum(ks), "pooled_n": N * 3,
        "wilson95_pooled_pct": [r1(100 * x) for x in wilson(sum(ks), N * 3)],
    })
RES["V1_V2_table1_arith"] = v1

# 보고된 퍼센트가 18회 정수 격자에서 가능한가
feas = {}
for pct in sorted({p for ks, ps, a in T1.values() for p in ps}):
    feas[str(pct)] = [k for k in range(N + 1) if r1(100 * k / N) == pct]
RES["V3_percent_integer_feasibility_n18"] = feas

# ---------- V8: Fisher 검정 (FM-VLA 대 각 방법) ----------
cmp_rows = []
for m, (ks, _, _) in T1.items():
    if m == "FM-VLA":
        continue
    fk = T1["FM-VLA"][0]
    for j, t in enumerate(TASKS):
        d, lo, hi = newcombe_diff(fk[j], N, ks[j], N)
        cmp_rows.append({"vs": m, "scope": t, "k1": fk[j], "k2": ks[j], "n": N,
                         "diff_pp": r1(100 * d), "newcombe95_pp": [r1(100 * lo), r1(100 * hi)],
                         "fisher_p": sg(fisher(fk[j], N, ks[j], N))})
    k1, k2 = sum(fk), sum(ks)
    d, lo, hi = newcombe_diff(k1, 3 * N, k2, 3 * N)
    cmp_rows.append({"vs": m, "scope": "pooled(54)", "k1": k1, "k2": k2, "n": 3 * N,
                     "diff_pp": r1(100 * d), "newcombe95_pp": [r1(100 * lo), r1(100 * hi)],
                     "fisher_p": sg(fisher(k1, 3 * N, k2, 3 * N))})
pt = [r for r in cmp_rows if r["scope"] != "pooled(54)"]
adj = holm([r["fisher_p"] for r in pt])
for r, a in zip(pt, adj):
    r["holm_p_21_tests"] = sg(a)
pp = [r for r in cmp_rows if r["scope"] == "pooled(54)"]
adj = holm([r["fisher_p"] for r in pp])
for r, a in zip(pp, adj):
    r["holm_p_7_tests"] = sg(a)
RES["V4_V6_fm_vla_vs_each"] = cmp_rows

# 기준선끼리 (π-MEM 대 π0.5, 대 TA-VLA)
base = []
for a, b in [("pi-MEM", "pi05"), ("pi-MEM", "TA-VLA"), ("state_only", "pi05"), ("force_only", "pi05")]:
    for j, t in enumerate(TASKS):
        ka, kb = T1[a][0][j], T1[b][0][j]
        base.append({"a": a, "b": b, "scope": t, "ka": ka, "kb": kb, "fisher_p": sg(fisher(ka, N, kb, N))})
    ka, kb = sum(T1[a][0]), sum(T1[b][0])
    base.append({"a": a, "b": b, "scope": "pooled(54)", "ka": ka, "kb": kb, "fisher_p": sg(fisher(ka, 54, kb, 54))})
RES["V8_baselines_among_themselves"] = base

# ---------- V9: "80% 이상" 주장 ----------
lo, hi = wilson(45, 54)
RES["V9_over80"] = {"k": 45, "n": 54, "rate_pct": r1(100 * 45 / 54), "wilson95_pct": [r1(100 * lo), r1(100 * hi)],
                    "binom_p_H0_rate_le_0.8": round(float(stats.binomtest(45, 54, 0.8, alternative="greater").pvalue), 3)}

# ---------- V10: 횟수 세기 과제의 우연 수준 ----------
chance = {}
for m in ["pi05", "pi-MEM", "FM-VLA", "state_only", "force_only"]:
    ks = T1[m][0]
    chance[m] = {t: round(float(stats.binomtest(ks[j], N, 1 / 3, alternative="two-sided").pvalue), 3)
                 for j, t in enumerate(TASKS) if t != "Cups"}
RES["V12_vs_chance_1_of_3"] = {
    "note": "N을 무작위로 고르는 정책의 성공 확률 1/3과 비교(양측 이항검정). 실제로는 정지·접촉 실패도 있어 이 값은 상한 기준이 아니라 참고 수준이다.",
    "p": chance}

# ---------- V11: Cups 과제의 하한 ----------
RES["V13_cups_lower_bound"] = {
    "note": "앞 컵에 블록이 있는 시행은 9회(부록 B.2). 기억 없는 pi0.5가 13/18 성공이면 뒤 컵 시행에서 최소 13-9=4회 성공해야 한다.",
    "pi05_successes": 13, "front_trials": 9, "min_back_successes": 13 - 9,
    "pi_MEM_min_back": 14 - 9, "TA-VLA_min_back": 9 - 9,
    "state_only_back": 18 - 9, "force_only_min_back": max(0, 10 - 9),
}

# ---------- V4-V7: 그림 3, 지연, 토큰, 시간 단위 ----------
fig3_pct = {4: 11.1, 8: 77.8, 16: 55.6, 32: 33.3}
fig3 = {}
for K, pct in fig3_pct.items():
    ks = [k for k in range(N + 1) if r1(100 * k / N) == pct]
    fig3[K] = {"label_pct": pct, "k_of_18": ks, "wilson95_pct": [r1(100 * x) for x in wilson(ks[0], N)],
               "total_action_expert_suffix_tokens": 30 + K + 1}
RES["V11_fig3_token_ablation"] = {
    "table1_wipe_k": 14, "k8_matches_table1": fig3[8]["k_of_18"] == [14],
    "values": fig3,
    "pairs": {
        "8_vs_16": sg(fisher(14, N, 10, N)),
        "8_vs_32": sg(fisher(14, N, 6, N)),
        "8_vs_4": sg(fisher(14, N, 2, N)),
        "16_vs_32": sg(fisher(10, N, 6, N)),
    },
    "note": "토큰 수 = 30(잡음 행동) + K + 1(상태). 본문은 '사전학습 행동 전문가가 최대 50 토큰'을 근거로 16과 32의 하락을 설명한다. K=16은 47 토큰으로 50 이하다.",
}
lat = {"pi05": (60.7, 0.3), "pi-MEM_K5": (99.8, 0.4), "pi-MEM_K16": (190.0, 1.0), "FM-VLA": (64.0, 0.4)}
RES["V10_latency"] = {
    "delta_vs_base_ms": {k: r1(v[0] - 60.7) for k, v in lat.items() if k != "pi05"},
    "delta_reported_ms": {"pi-MEM_K5": 39.1, "pi-MEM_K16": 129.3, "FM-VLA": 3.3},
    "ratio_vs_base": {k: round(v[0] / 60.7, 3) for k, v in lat.items() if k != "pi05"},
    "note": "표 2의 Δ는 모두 일치. 그림 3의 지연(64.0, 64.0, 65.0, 65.0 ms)과 표 2의 64.0은 K=8에서 일치.",
}
RES["K1_time_units"] = {
    "pre_pad_max_len_frames": 1000, "rate_hz_E1": 30, "seconds_at_30hz": round(1000 / 30, 1),
    "seconds_at_100hz": 10.0,
    "note": "3.2.1은 최대 10 s, E.1은 1000 프레임(약 10 s)과 30 Hz를 함께 적는다. 30 Hz에서 1000 프레임은 약 33.3 s이고 100 Hz에서 10 s다.",
}
RES["K6_state_window"] = {"taps": 10, "stride": 3, "rate_hz": 30,
                          "span_s": round(27 / 30, 2), "ds": 7 * 2 + 2, "ds_reported": 16}

# ---------- V16: π-MEM 시각 기억 구간 ----------
def span(K, stride=45, hz=30):
    return (K - 1) * stride / hz

RES["V16_pimem_memory_span"] = {
    "K5_span_s": span(5), "K16_span_s": span(16),
    "episode_length_s_figure_read": {"Cups": 20, "Buttons": 8, "Wipe": 10},
    "note": "K=5, stride 45 프레임(30 Hz)이면 기억은 최근 6.0 s. K=16이면 22.5 s. 그림 4의 시간 축(그림 판독 값)은 Cups 약 20 s, Buttons 약 8 s, Wipe 약 10 s. 성공률은 K=5만 보고, K=16은 지연만 보고(표 2).",
}

# ---------- V17: 압축 비율 ----------
comp_rows = {}
for name, secs in [("Buttons_8s", 8), ("Wipe_10s", 10), ("Cups_20s", 20)]:
    frames = secs * 30
    comp_rows[name] = {"input_values": frames * 6, "latent_values": 8 * 96, "ratio": round(frames * 6 / (8 * 96), 2)}
RES["V17_compression_ratio_30hz"] = comp_rows

# ---------- V12: KL 항의 크기 ----------
def kl_dim(mu, sigma):
    return 0.5 * (mu ** 2 + sigma ** 2 - 1 - math.log(sigma ** 2))

kl_rows = []
for mu, sg in [(0.0, 1.0), (1.0, 0.3), (1.0, 0.1), (1.0, 0.03), (1.0, 0.01), (2.0, 0.01)]:
    kl = kl_dim(mu, sg)
    kl_rows.append({"mu": mu, "sigma": sg, "kl_nats_per_dim": round(kl, 3),
                    "free_bits_active_(kl<0.5)": kl < 0.5,
                    "beta_times_max(kl,0.5)": round(1e-3 * max(kl, 0.5), 5)})
RES["V14_kl_term_scale"] = {
    "beta": 1e-3, "free_bits_lambda": 0.5,
    "note": "식 4의 KL 항 = β · 평균 max(KL, λ). 차원당 KL이 0.5 nats 미만이면 기울기가 꺼지고, 그 이상에서도 계수가 1e-3이다. 재구성 항(분위수 정규화된 값의 제곱오차)에 비해 작아 보이나 실제 크기는 원자료가 없어 계산하지 못했다.",
    "rows": kl_rows,
    "floor_value_of_kl_term_if_all_dims_below_lambda": 1e-3 * 0.5,
}

# ---------- V13: 검정력 ----------
def n_per_arm(p1, p2, alpha=0.05, power=0.8):
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pb = (p1 + p2) / 2
    num = (za * math.sqrt(2 * pb * (1 - pb)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(num / (p1 - p2) ** 2)


def power_mc(p1, p2, n, sims=20000, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.binomial(n, p1, sims)
    b = rng.binomial(n, p2, sims)
    hit = 0
    cache = {}
    for x, y in zip(a, b):
        key = (x, y)
        if key not in cache:
            cache[key] = fisher(int(x), n, int(y), n) < 0.05
        hit += cache[key]
    return hit / sims


RES["V7_power"] = {
    "n_per_arm_80pct_power_normal_approx": {
        "Buttons_72.2_vs_33.3(pi-MEM)": n_per_arm(13 / 18, 6 / 18),
        "Wipe_77.8_vs_50.0(pi-MEM)": n_per_arm(14 / 18, 9 / 18),
        "Cups_100_vs_77.8(pi-MEM)_p_clipped_0.99": n_per_arm(0.99, 14 / 18),
        "pooled_83.3_vs_53.7": n_per_arm(45 / 54, 29 / 54),
    },
    "fisher_power_n18_true_p_equal_observed": {
        "Buttons_13/18_vs_6/18": round(power_mc(13 / 18, 6 / 18, 18), 3),
        "Wipe_14/18_vs_9/18": round(power_mc(14 / 18, 9 / 18, 18), 3),
        "Cups_18/18_vs_14/18_(p=.97_vs_.78)": round(power_mc(0.97, 14 / 18, 18), 3),
    },
}

# ---------- V14: 토큰 수를 평가 시행으로 고른 효과(선택 편향) ----------
rng = np.random.default_rng(1)
p_common = (14 + 10 + 6) / 54  # K=8,16,32가 모두 같은 참 성공률이라는 귀무
sims = 200000
mx = rng.binomial(N, p_common, (sims, 3)).max(axis=1)
RES["V15_selection_on_eval"] = {
    "null": "K=8,16,32의 참 성공률이 같다(공통 p = 30/54 = 0.556)고 둔다.",
    "mean_of_best_of_3_pct": r1(100 * mx.mean() / N),
    "p_best_of_3_at_least_14of18": round(float((mx >= 14).mean()), 3),
    "observed_best_pct": r1(100 * 14 / N),
    "note": "세 설정 중 최고값을 고르면 참 값이 같아도 평균 약 65%가 나온다(위 값). 관측 최고 14/18은 이 귀무에서 드물지 않은 편이다(확률은 위 값). 토큰 수를 평가 시행으로 골랐다면 최고값은 낙관적일 수 있다.",
}

# ---------- 학습 예산 ----------
RES["V18_training_budget"] = {
    "stage2_samples_fm_vla_and_baselines": 50_000 * 32,
    "stage1_vae_samples": 100_000 * 512,
    "demos": {"Cups": 200, "Buttons": 350, "Wipe": 200, "total": 750},
    "buttons_demos_per_N_if_uniform": round(350 / 3, 1),
    "note": "FM-VLA만 Stage 1 힘 VAE 사전학습(10만 스텝, 배치 512)을 추가로 받는다. 데이터는 같은 시연의 wrench이다.",
}

OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=1), encoding="utf-8")

# ---------- 요약 출력 ----------
print("== V1 표 1 산술 ==")
for r in v1:
    print(f"{r['method']:18s} rates_match={r['rates_match']} avg {r['avg_reported']} pooled {r['avg_pooled']} (match={r['avg_match']}) wilson95={r['wilson95_pooled_pct']}")
print("== V8 FM-VLA 대 각 방법 (Fisher) ==")
for r in cmp_rows:
    print(f"{r['vs']:18s} {r['scope']:11s} {r['k1']}/{r['n']} vs {r['k2']}/{r['n']} diff {r['diff_pp']:6.1f} CI {r['newcombe95_pp']} p={r['fisher_p']}")
print("== V9 ==", RES["V9_over80"])
print("== V4 ==", RES["V11_fig3_token_ablation"]["pairs"], RES["V11_fig3_token_ablation"]["k8_matches_table1"])
print("== V13 ==", RES["V7_power"])
print("== V14 ==", RES["V15_selection_on_eval"])
print("== V10 ==", RES["V12_vs_chance_1_of_3"]["p"])
print("wrote", OUT)
