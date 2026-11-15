"""DreamZero(arXiv:2602.15922v1) 수치 재계산: 표 2·3·4, 그림 8·9·10의 평균, 검정과 신뢰구간.

입력: 논문 본문 표(표 2·3·4)와 그림 막대 라벨. 그림 값은 렌더링한 PDF 그림에서 읽었다(분석 폴더 paper/ 기준, 쪽 13–15).
가정은 출력 JSON의 "assumptions"에 적는다. 실행: python3 analysis/table_check.py  (scipy, numpy, 약 1초)
출력: analysis/table_check_results.json
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).with_name("table_check_results.json")
res = {"assumptions": [], "checks": {}}


def assume(text):
    res["assumptions"].append(text)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


# ---------- 1. 표 2: 교차 embodiment 전이 (± = 표준오차) ----------
# 그림 9의 DreamZero 10개 과제 값에서 Pulling Cart(50.0)를 뺀 9개 평균이 38.3이다(표 2 주석 10: YAM 설정에서 제외한 과제와 같은 9개).
fig9_dz = {"Untie": 28.1, "Hat": 85.7, "Pen": 35.7, "Straw": 46.9, "Cube": 23.9,
           "Paint": 28.6, "Ironing": 28.6, "Shake": 59.2, "FoldMap": 8.2, "PullCart": 50.0}
fig9_pi05_pre = {"Untie": 9.4, "Hat": 42.9, "Pen": 14.3, "Straw": 18.8, "Cube": 6.2,
                 "Paint": 17.9, "Ironing": 14.3, "Shake": 25.0, "FoldMap": 0.0, "PullCart": 14.3}
fig9_gr_pre = {"Untie": 0.1, "Hat": 12.5, "Pen": 0.0, "Straw": 6.2, "Cube": 0.0,
               "Paint": 6.2, "Ironing": 0.0, "Shake": 18.8, "FoldMap": 0.0, "PullCart": 0.0}
fig9_gr_scr = {"Untie": 0, "Hat": 0, "Pen": 0, "Straw": 0, "Cube": 0,
               "Paint": 0, "Ironing": 7.1, "Shake": 0, "FoldMap": 0, "PullCart": 0}
nine = [k for k in fig9_dz if k != "PullCart"]
v9 = np.array([fig9_dz[k] for k in nine])
sd9 = v9.std(ddof=1)
res["checks"]["table2_fig9_recompute"] = {
    "dz_mean_10_tasks": round(np.mean(list(fig9_dz.values())), 2),
    "dz_mean_9_tasks_excl_pullcart": round(v9.mean(), 2),
    "sd_9_tasks": round(sd9, 2),
    "se_9_tasks_sd_over_sqrt9": round(sd9 / 3, 2),
    "paper_table2_se_dz": 7.6,
    "pi05_pre_mean_10": round(np.mean(list(fig9_pi05_pre.values())), 2),
    "gr00t_pre_mean_10": round(np.mean(list(fig9_gr_pre.values())), 2),
    "gr00t_scratch_mean_10": round(np.mean(list(fig9_gr_scr.values())), 2),
    "note_gr00t_pre_fig_avg_label": 5,
}
assume("표 2의 ± 값은 9개 과제 사이의 표준오차(SD/√9)다. 그림 9 값에서 SD 22.8, SE 7.6이 나와 일치한다.")

t2 = {"DreamZero": (38.3, 7.6), "+Human2Robot": (54.3, 10.4), "+Robot2Robot": (55.4, 9.5)}
base_m, base_se = t2["DreamZero"]
rel = {k: round(100 * (m / base_m - 1), 1) for k, (m, _) in t2.items() if k != "DreamZero"}
res["checks"]["table2_relative_gain_pct"] = rel
res["checks"]["table2_abstract_claim"] = "초록은 '42% 이상'이라고 쓰지만 Human2Robot은 41.8%다(54.3/38.3)."


def two_sample_se_diff(se1, se2, rho=0.0, n=9):
    # 과제 단위 대응(같은 과제 집합)일 때 상관 rho를 반영한 차이의 SE. rho=0이면 독립 가정.
    s1, s2 = se1 * math.sqrt(n), se2 * math.sqrt(n)
    return math.sqrt(max(s1 * s1 + s2 * s2 - 2 * rho * s1 * s2, 0)) / math.sqrt(n)


t2_tests = {}
for k in ("+Human2Robot", "+Robot2Robot"):
    m, se = t2[k]
    d = m - base_m
    row = {"diff": round(d, 1)}
    for rho in (0.0, 0.5, 0.8):
        s = two_sample_se_diff(se, base_se, rho)
        tt = d / s
        p = 2 * stats.t.sf(abs(tt), df=8 + 8 - 2 if rho == 0 else 16)
        row[f"rho={rho}"] = {"se_diff": round(s, 2), "t": round(tt, 2), "p_two_sided": round(p, 3)}
    t2_tests[k] = row
res["checks"]["table2_tests_unpaired_and_paired"] = t2_tests
assume("표 2 검정은 과제 단위 상관 rho를 0, 0.5, 0.8로 둔 민감도다. 논문은 과제를 짝지었는지 밝히지 않았다.")

# ---------- 2. 표 3: DreamZero-Flash (테이블 버싱, 과제당 10 롤아웃) ----------
n3 = 10
t3 = {"4-step (350ms)": (83, 6.1), "1-step (150ms)": (52, 10.2), "Flash 1-step (150ms)": (74, 10.1)}
ci3 = {}
for k, (m, se) in t3.items():
    tcrit = stats.t.ppf(0.975, n3 - 1)
    ci3[k] = {"mean": m, "sd_implied": round(se * math.sqrt(n3), 1),
              "ci95": [round(m - tcrit * se, 1), round(m + tcrit * se, 1)]}
res["checks"]["table3_ci"] = ci3
d_flash_vs_1 = 74 - 52
se_d1 = math.sqrt(10.1 ** 2 + 10.2 ** 2)
d_flash_vs_4 = 74 - 83
se_d4 = math.sqrt(10.1 ** 2 + 6.1 ** 2)
df = n3 - 1 + n3 - 1
res["checks"]["table3_tests"] = {
    "flash_vs_1step": {"diff": d_flash_vs_1, "se": round(se_d1, 2), "t": round(d_flash_vs_1 / se_d1, 2),
                       "p_two_sided": round(2 * stats.t.sf(abs(d_flash_vs_1 / se_d1), df), 3)},
    "flash_vs_4step": {"diff": d_flash_vs_4, "se": round(se_d4, 2), "ci95_diff": [round(d_flash_vs_4 - stats.t.ppf(0.975, df) * se_d4, 1), round(d_flash_vs_4 + stats.t.ppf(0.975, df) * se_d4, 1)],
                       "t": round(d_flash_vs_4 / se_d4, 2), "p_two_sided": round(2 * stats.t.sf(abs(d_flash_vs_4 / se_d4), df), 3),
                       "note": "차이 -9의 95% 신뢰구간이 -32에서 +14까지다. '거의 같다'를 검정하려면 여유 폭을 정하고 더 많은 롤아웃이 필요하다."},
    "speed_ratio_350_over_150": round(350 / 150, 2),
}
assume("표 3은 과제당 10 롤아웃(본문 4.2절)이고 ± 값은 표준오차다. SD는 SE×√10으로 되돌렸다.")

# 선형 모형 t(N) = a + b*N (N = DiT 단계 수). 350 ms(4단계), 150 ms(1단계, Flash) 두 점으로 적합.
b = (350 - 150) / 3
a = 150 - b
res["checks"]["table3_linear_latency_fit"] = {"per_dit_step_ms": round(b, 1), "fixed_ms": round(a, 1),
    "note": "고정 항에는 VAE, 텍스트·이미지 인코더, 실제 관측을 KV 캐시에 넣는 추가 DiT 순전파(알고리즘 2)가 들어간다. 두 점 적합이라 기울기와 절편은 근사값이다."}

# ---------- 3. 표 4: 소거 실험 (PnP Easy, 롤아웃 수 미보고) ----------
t4 = {"Repetitive 14B AR": (33, 4.2), "Diverse 14B AR": (50, 6.3), "5B AR (Diverse)": (21, 4.2),
      "14B BD (Diverse)": (50, 14.4)}
res["checks"]["table4_note"] = "표 4의 PnP Easy 롤아웃 수가 없다. n = 24(3 과제 × 8)로 가정하고 n = 12, 48로 민감도를 봤다."
t4_tests = {}
for name_a, name_b in [("Diverse 14B AR", "Repetitive 14B AR"), ("Diverse 14B AR", "5B AR (Diverse)"), ("14B BD (Diverse)", "Diverse 14B AR")]:
    ma, sa = t4[name_a]
    mb, sb = t4[name_b]
    row = {}
    for n in (12, 24, 48):
        sa_sd, sb_sd = sa * math.sqrt(n), sb * math.sqrt(n)
        se = math.sqrt(sa ** 2 + sb ** 2)
        tt = (ma - mb) / se
        p = 2 * stats.t.sf(abs(tt), 2 * n - 2)
        row[f"n={n}"] = {"diff": ma - mb, "se": round(se, 2), "t": round(tt, 2), "p_two_sided": round(p, 4)}
    t4_tests[f"{name_a} vs {name_b}"] = row
res["checks"]["table4_tests"] = t4_tests
assume("표 4의 ± 표준오차 기준 검정. 롤아웃 수 n은 가정이다. 같은 학습 조건의 반복 실행은 1회라서 seed 분산은 빠져 있다.")

# ---------- 4. 그림 8 평균 재계산 (과제 수 가중) ----------
cat = {"PnPEasy": (3, {"DZ": 93.8, "pi05_pre": 52.1, "gr_pre": 17.6}),
       "PnPHard": (4, {"DZ": 48.4, "pi05_pre": 22.7, "gr_pre": 4.7}),
       "ContactRich": (3, {"DZ": 49.0, "pi05_pre": 9.2, "gr_pre": 4.2})}
w = {}
for who in ("DZ", "pi05_pre", "gr_pre"):
    n_tot = sum(c[0] for c in cat.values())
    weighted = sum(c[0] * c[1][who] for c in cat.values()) / n_tot
    unweighted = np.mean([c[1][who] for c in cat.values()])
    w[who] = {"task_weighted": round(weighted, 2), "category_unweighted": round(float(unweighted), 2)}
res["checks"]["fig8_seen_average_recompute"] = {
    "note": "본문 62.2 = 과제 수(3·4·3)로 가중한 평균. 범주 평균을 그냥 나누면 63.7이 된다. 본문에 가중 방식이 없다.",
    "values": w}

# ---------- 5. 그림 10 평균 재계산 (후학습, 3 과제) ----------
fig10 = {"GR00T scratch": [2.5, 27, 0], "GR00T pretrained": [65, 56, 39],
         "pi0.5 scratch": [1.5, 0, 0], "pi0.5 pretrained": [92.5, 71, 76], "DreamZero": [92.5, 96, 83]}
res["checks"]["fig10_average_recompute"] = {k: {"mean3": round(np.mean(v), 2)} for k, v in fig10.items()}
res["checks"]["fig10_average_recompute"]["note"] = "그림 라벨 AVG: GR00T 9.8/53.3, pi0.5 0.5/79.8, DreamZero 90.5. 본문 '10% 앞섬'은 90.5 - 79.8 = 10.7 pp와 맞다."

# ---------- 6. 성공률 (DROID, 롤아웃 80개) ----------
droid = {"unseen DZ": (18, 80), "unseen GR00T pre": (10, 80), "unseen pi0.5 pre": (6, 80),
         "seen DZ": (60, 80)}
ci_droid = {k: {"k": kk, "n": nn, "rate": round(100 * kk / nn, 1), "wilson95": wilson(kk, nn)} for k, (kk, nn) in droid.items()}
fe = {}
for lbl, (kA, nA), (kB, nB) in [("unseen DZ vs pi0.5 pre", droid["unseen DZ"], droid["unseen pi0.5 pre"]),
                                 ("unseen DZ vs GR00T pre", droid["unseen DZ"], droid["unseen GR00T pre"])]:
    _, p = stats.fisher_exact([[kA, nA - kA], [kB, nB - kB]])
    fe[lbl] = round(p, 4)
res["checks"]["droid_success"] = {"ci": ci_droid, "fisher_p": fe,
    "seen_pi05_or_gr00t_42pct_check": "분모 80이면 42%는 정수 개수가 아니다(33/80 = 41.3, 34/80 = 42.5). 표기 방식 확인이 필요하다."}
assume("DROID 성공률은 롤아웃 80개(본문 4.1절 'total of 80 evaluation rollouts')로 가정했다. 22.5, 12.5, 7.5, 75는 각각 18, 10, 6, 60으로 정확히 떨어진다.")

# ---------- 7. 조합 검사: 표 1 누적 배율과 38× ----------
res["checks"]["table1_arithmetic"] = {
    "gb200_final_speedup_16.6_x_flash_2.33": round(16.6 * 2.33, 1),
    "naive_5.7s_over_38": round(5.7 / 38 * 1000, 0),
    "naive_5.7s_over_16.6": round(5.7 / 16.6 * 1000, 0),
    "cfg_parallel_expected_from_47pct_latency_cut": round(1 / (1 - 0.47), 2),
    "dit_caching_16_to_4_ideal": 4.0,
    "dit_caching_observed_ratio_5.4_over_1.8": round(5.4 / 1.8, 2),
    "note": "표 1의 행은 누적 배율이다. 16.6 × 2.33 = 38.7이고 본문은 38×라고 한다. 누적 16.6×는 약 343 ms로 표 3의 4단계 350 ms와 맞는다."}

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote", OUT)
print(json.dumps(res["checks"]["table2_tests_unpaired_and_paired"], ensure_ascii=False, indent=1)[:900])
print(json.dumps(res["checks"]["table2_fig9_recompute"], ensure_ascii=False))
print(json.dumps(res["checks"]["table3_tests"], ensure_ascii=False)[:600])
print(json.dumps(res["checks"]["fig8_seen_average_recompute"], ensure_ascii=False)[:500])
print(json.dumps(res["checks"]["droid_success"]["fisher_p"], ensure_ascii=False))
print(json.dumps(res["checks"]["table4_tests"], ensure_ascii=False)[:700])
