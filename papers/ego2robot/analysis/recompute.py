"""Ego2Robot(arXiv:2608.02580 v1) 표 값 재계산과 일관성 검사.

입력: paper/paper.txt (pdftotext -layout)에서 표 5(과제별 결과)를 읽고, 표 1·2·4와 그림 3·4의 값은 아래에 옮겨 둔다.
출력: analysis/recompute_results.json, 표준 출력 요약. numpy, scipy만 쓰고 1초대에 끝난다.
검사 라벨: V1–V14 (03_analysis.md의 절과 같은 번호)
"""
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = {}

MODELS = ["Robot-only", "Ego2R(1:3)", "Ego2R(3:1)", "Ego2R(1:1)"]

# ---- 표 1 (논문 p.7). 열: Clean, Rand, Visual, Scene, Embody, Task, EBench ----
T1_COLS = ["Clean", "Rand", "Visual", "Scene", "Embody", "Task", "EBench"]
T1 = {
    "Robot-only": [62.2, 50.9, 61.4, 52.9, 23.8, 46.2, 39.6],
    "Ego2R(1:3)": [61.4, 51.0, 61.2, 52.5, 21.9, 49.5, 47.4],
    "Ego2R(3:1)": [64.1, 49.2, 62.7, 54.3, 28.2, 51.6, 51.7],
    "Ego2R(1:1)": [68.1, 53.5, 67.3, 56.9, 27.2, 54.1, 49.8],
}
T1_DELTA = {  # 표에 적힌 변화량
    "Ego2R(1:3)": [-0.8, 0.1, -0.2, -0.4, -1.9, 3.3, 7.8],
    "Ego2R(3:1)": [1.9, -1.7, 1.3, 1.4, 4.4, 5.4, 12.1],
    "Ego2R(1:1)": [5.9, 2.6, 5.9, 4.0, 3.4, 7.9, 10.2],
}
# ---- 표 2 (논문 p.7). 열: BG Light Color | Height Clutter Camera | ARX UR5 Franka | Obj Lang ----
T2_COLS = ["BG", "Light", "Color", "Height", "Clutter", "Camera", "ARX", "UR5", "Franka", "Obj", "Lang"]
T2 = {
    "Robot-only": [66.6, 58.2, 59.4, 60.1, 48.3, 50.4, 44.1, 20.2, 7.0, 29.3, 63.1],
    "Ego2R(1:3)": [65.0, 58.3, 60.3, 58.6, 49.3, 49.6, 43.7, 17.6, 4.5, 36.8, 62.2],
    "Ego2R(3:1)": [65.5, 60.9, 61.8, 62.0, 49.2, 51.6, 47.6, 31.4, 5.6, 40.0, 63.1],
    "Ego2R(1:1)": [70.3, 65.8, 65.8, 62.4, 52.0, 56.3, 51.2, 25.0, 5.3, 39.6, 68.5],
}
T2_DELTA = {
    "Ego2R(1:3)": [-1.6, 0.1, 0.9, -1.5, 1.0, -0.8, -0.4, -2.6, -2.5, 7.5, -0.9],
    "Ego2R(3:1)": [-1.1, 2.7, 2.4, 1.9, 0.9, 1.2, 3.5, 11.2, -1.4, 10.7, 0.0],
    "Ego2R(1:1)": [3.7, 7.6, 6.4, 2.3, 3.7, 5.9, 7.1, 4.8, -1.7, 10.3, 5.4],
}
GROUPS = {"Visual": [0, 1, 2], "Scene": [3, 4, 5], "Embody": [6, 7, 8], "Task": [9, 10]}
N_EP = 50          # 과제당 에피소드 (4절, B.3)
N_TASK = 50
N_CELL = N_EP * N_TASK  # 한 셀의 에피소드 수 = 2,500


def r1(x):
    return round(x + 1e-9, 1)


# ---------------------------------------------------------------- 표 5 파싱
def parse_table5():
    txt = (ROOT / "paper" / "paper.txt").read_text(encoding="utf-8")
    start = txt.index("Table 5: Per-task results")
    rows = {}
    for line in txt[start:].splitlines():
        m = re.match(r"^([a-z0-9][a-z0-9 ]*?)\s{2,}((?:\d+\.\d\s+){9}\d+\.\d)\s*$", line.strip() + " ")
        if m:
            rows[m.group(1).strip()] = [float(v) for v in m.group(2).split()]
        m2 = re.match(r"^Average\s+((?:\d+\.\d\s+){9}\d+\.\d)", line.strip() + " ")
        if m2:
            rows["Average"] = [float(v) for v in m2.group(1).split()]
    return rows


T5 = parse_table5()
avg_row = T5.pop("Average")
tasks = list(T5.keys())
M5 = np.array([T5[t] for t in tasks])  # 50 x 10: Pi0.5 C/R, Robot C/R, 1:3 C/R, 3:1 C/R, 1:1 C/R
assert M5.shape == (50, 10), M5.shape
T5_NAMES = ["Pi0.5", "Robot-only", "Ego2R(1:3)", "Ego2R(3:1)", "Ego2R(1:1)"]
clean = {n: M5[:, 2 * i] for i, n in enumerate(T5_NAMES)}
rand = {n: M5[:, 2 * i + 1] for i, n in enumerate(T5_NAMES)}

# ------------------------------------------------ V1. 표 5 평균 대 보고된 평균 행, 표 1 Clean/Rand
v1 = {}
for i, n in enumerate(T5_NAMES):
    c, r = clean[n].mean(), rand[n].mean()
    v1[n] = {"clean_mean": round(c, 3), "clean_reported_avg_row": avg_row[2 * i],
             "rand_mean": round(r, 3), "rand_reported_avg_row": avg_row[2 * i + 1]}
    if n in T1:
        v1[n]["table1_clean"] = T1[n][0]
        v1[n]["table1_rand"] = T1[n][1]
    v1[n]["max_abs_diff_vs_avg_row"] = round(max(abs(c - avg_row[2 * i]), abs(r - avg_row[2 * i + 1])), 3)
OUT["V1_table5_means"] = v1

# ------------------------------------------------ V2. 표 5 개별 값의 정수·짝수 가능성
# 과제당 50 에피소드의 이진 성공이면 성공률은 2의 배수(%)여야 한다
allvals = M5.flatten()
odd_mask = (np.round(allvals) % 2 == 1)
non_int = int(np.sum(np.abs(allvals - np.round(allvals)) > 1e-9))
per_col_odd = {}
for i, n in enumerate(T5_NAMES):
    per_col_odd[n] = {"clean_odd": int(np.sum(np.round(clean[n]) % 2 == 1)),
                      "rand_odd": int(np.sum(np.round(rand[n]) % 2 == 1))}
OUT["V2_parity"] = {"n_values": int(allvals.size), "n_non_integer": non_int, "n_odd": int(odd_mask.sum()),
                    "share_odd": round(float(odd_mask.mean()), 3), "per_model": per_col_odd,
                    "pi05_all_even": bool(np.all(np.round(M5[:, 0:2]) % 2 == 0)),
                    "note": "50에피소드 이진 성공이면 모든 값이 짝수여야 한다. 홀수가 있으면 분모가 50이 아니다(예: 100)."}
# 분모 100이면 모든 정수가 가능
OUT["V2_parity"]["pi05_odd_count"] = int(np.sum(np.round(M5[:, 0:2]) % 2 == 1))

# ------------------------------------------------ V3. 표 1 변화량 = 표값 − Robot-only
v3 = []
for n in MODELS[1:]:
    for j, col in enumerate(T1_COLS):
        d = r1(T1[n][j] - T1["Robot-only"][j])
        if abs(d - T1_DELTA[n][j]) > 0.051:
            v3.append({"model": n, "col": col, "computed": d, "reported": T1_DELTA[n][j]})
OUT["V3_table1_delta_mismatch"] = v3
v3b = []
for n in MODELS[1:]:
    for j, col in enumerate(T2_COLS):
        d = r1(T2[n][j] - T2["Robot-only"][j])
        if abs(d - T2_DELTA[n][j]) > 0.051:
            v3b.append({"model": n, "col": col, "computed": d, "reported": T2_DELTA[n][j]})
OUT["V3_table2_delta_mismatch"] = v3b

# ------------------------------------------------ V4. 표 2 → 표 1 차원 평균 (B.3 정의)
v4 = []
for n in MODELS:
    for gi, (g, idx) in enumerate(GROUPS.items()):
        mean = float(np.mean([T2[n][k] for k in idx]))
        rep = T1[n][2 + gi]
        v4.append({"model": n, "group": g, "computed": round(mean, 2), "reported": rep,
                   "ok_to_0.1": abs(mean - rep) <= 0.051 + 1e-9})
OUT["V4_group_means"] = v4
OUT["V4_group_mismatch"] = [x for x in v4 if not x["ok_to_0.1"]]

# ------------------------------------------------ V5. 본문 문장 검사
t1_best = {c: max(MODELS, key=lambda m: T1[m][j]) for j, c in enumerate(T1_COLS)}
lead_11 = [c for c, m in t1_best.items() if m == "Ego2R(1:1)"]
d11 = {c: r1(T2["Ego2R(1:1)"][j] - T2["Robot-only"][j]) for j, c in enumerate(T2_COLS)}
pos_counts = {n: int(sum(1 for j in range(11) if T2[n][j] > T2["Robot-only"][j])) for n in MODELS[1:]}
pos_counts_t1 = {n: int(sum(1 for j in range(7) if T1[n][j] > T1["Robot-only"][j])) for n in MODELS[1:]}
OUT["V5_text_claims"] = {
    "table1_best_per_column": t1_best,
    "1:1_leads_in": lead_11, "1:1_leads_count": len(lead_11),
    "table2_delta_1:1": d11,
    "table2_n_positive_of_11": pos_counts,
    "table1_n_positive_of_7": pos_counts_t1,
    "text_bg_light_color_1:1": [d11["BG"], d11["Light"], d11["Color"]],
    "text_camera_1:1": d11["Camera"],
    "text_obj_3:1": r1(T2["Ego2R(3:1)"][9] - T2["Robot-only"][9]),
    "text_lang_1:1_value": T2["Ego2R(1:1)"][10],
    "franka_values": [T2[m][8] for m in MODELS],
    "franka_text": "Franka remains below 7%: 로봇 전용 7.0은 7 미만이 아니다(같다)",
}

# ------------------------------------------------ V6. 단일 실행 표본오차와 유의성 (표 1·2, 2,500 에피소드 가정)
def two_prop(p1, p2, n=N_CELL):
    p1, p2 = p1 / 100, p2 / 100
    pp = (p1 + p2) / 2
    se = math.sqrt(2 * pp * (1 - pp) / n)
    z = (p2 - p1) / se
    return z, 2 * (1 - stats.norm.cdf(abs(z))), se * 100


v6 = {"table1": {}, "table2": {}}
for n in MODELS[1:]:
    v6["table1"][n] = {}
    for j, c in enumerate(T1_COLS):
        if c == "EBench":
            continue
        # Visual/Scene/Embody/Task는 평균 셀이라 단순 근사(셀당 2,500보다 크지만 같은 분모 가정)
        z, p, se = two_prop(T1["Robot-only"][j], T1[n][j], N_CELL * (2 if c in ("Task",) else 3 if c in ("Visual", "Scene", "Embody") else 1))
        v6["table1"][n][c] = {"delta": r1(T1[n][j] - T1["Robot-only"][j]), "z": round(z, 2), "p": round(p, 4), "se_diff_pp": round(se, 2)}
    v6["table2"][n] = {}
    for j, c in enumerate(T2_COLS):
        z, p, se = two_prop(T2["Robot-only"][j], T2[n][j])
        v6["table2"][n][c] = {"delta": r1(T2[n][j] - T2["Robot-only"][j]), "z": round(z, 2), "p": round(p, 4)}
OUT["V6_two_proportion_z"] = v6
# 다중비교 보정(표 2 33개 비교, Benjamini–Hochberg와 Bonferroni)
ps, labels = [], []
for n in MODELS[1:]:
    for c in T2_COLS:
        ps.append(v6["table2"][n][c]["p"]); labels.append((n, c))
ps_arr = np.array(ps)
order = np.argsort(ps_arr)
m = len(ps_arr)
bh = np.empty(m)
prev = 1.0
for rank, idx in reversed(list(enumerate(order, start=1))):
    prev = min(prev, ps_arr[idx] * m / rank)
    bh[idx] = prev
OUT["V6_table2_multiple_comparison"] = {
    "n_tests": m,
    "p_lt_0.05_raw": int((ps_arr < 0.05).sum()),
    "p_lt_0.05_bonferroni": int((ps_arr * m < 0.05).sum()),
    "p_lt_0.05_BH": int((bh < 0.05).sum()),
    "survivors_BH": [f"{labels[i][0]}|{labels[i][1]}" for i in range(m) if bh[i] < 0.05],
    "survivors_raw": [f"{labels[i][0]}|{labels[i][1]}" for i in range(m) if ps_arr[i] < 0.05],
}
# '>5%p 이면 초록' 규칙의 위양성: 같은 분모에서 5pp 차이를 검출하는 최소 크기
for base in (20, 50, 65):
    pp = base / 100
    se = math.sqrt(2 * pp * (1 - pp) / N_CELL) * 100
    OUT.setdefault("V6_se_diff_by_base", {})[str(base)] = round(se, 2)

# ------------------------------------------------ V7. 표 5 기반 과제 단위 짝지은 비교(Clean, Rand)
def paired(a, b, rng):
    d = b - a
    t = stats.ttest_rel(b, a)
    try:
        w = stats.wilcoxon(d)
        wp = float(w.pvalue)
    except ValueError:
        wp = float("nan")
    boots = [rng.choice(d, size=d.size, replace=True).mean() for _ in range(10000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean_diff": round(float(d.mean()), 2), "sd_diff": round(float(d.std(ddof=1)), 2),
            "se_mean_diff": round(float(d.std(ddof=1) / math.sqrt(d.size)), 2),
            "t": round(float(t.statistic), 2), "p_t": round(float(t.pvalue), 4), "p_wilcoxon": round(wp, 4),
            "boot95": [round(float(lo), 2), round(float(hi), 2)],
            "n_tasks_up": int((d > 0).sum()), "n_tasks_down": int((d < 0).sum()), "n_tie": int((d == 0).sum())}


rng = np.random.default_rng(0)
v7 = {}
for n in MODELS[1:]:
    v7[n] = {"clean": paired(clean["Robot-only"], clean[n], rng), "rand": paired(rand["Robot-only"], rand[n], rng)}
OUT["V7_paired_over_tasks"] = v7

# ------------------------------------------------ V8. 과제 단위 이항 표본오차(50 에피소드)로 본 평균의 SE
def se_avg(p_arr, n_ep=N_EP):
    p = p_arr / 100
    return float(np.sqrt(np.sum(p * (1 - p) / n_ep)) / len(p) * 100)


v8 = {}
for n in T5_NAMES:
    v8[n] = {"clean_se": round(se_avg(clean[n]), 2), "rand_se": round(se_avg(rand[n]), 2)}
OUT["V8_binomial_se_of_average"] = v8
# 두 모델 차이의 z (독립 에피소드 가정)
v8z = {}
for n in MODELS[1:]:
    for key, d in (("clean", clean), ("rand", rand)):
        se = math.sqrt(se_avg(d["Robot-only"]) ** 2 + se_avg(d[n]) ** 2)
        diff = d[n].mean() - d["Robot-only"].mean()
        z = diff / se
        v8z[f"{n}|{key}"] = {"diff": round(float(diff), 2), "se": round(se, 2), "z": round(float(z), 2),
                             "p": round(float(2 * (1 - stats.norm.cdf(abs(z)))), 4)}
OUT["V8_diff_z_binomial"] = v8z
# 분모가 과제당 100 에피소드라면(V2의 홀수 값이 시사) SE는 1/sqrt(2)배
v8z100 = {}
for n in MODELS[1:]:
    for key, d in (("clean", clean), ("rand", rand)):
        se = math.sqrt(se_avg(d["Robot-only"], 100) ** 2 + se_avg(d[n], 100) ** 2)
        diff = d[n].mean() - d["Robot-only"].mean()
        z = diff / se
        v8z100[f"{n}|{key}"] = {"diff": round(float(diff), 2), "se": round(se, 2), "z": round(float(z), 2),
                                "p": round(float(2 * (1 - stats.norm.cdf(abs(z)))), 4)}
OUT["V8_diff_z_binomial_n100"] = v8z100

# ------------------------------------------------ V9. Clean 대 Rand 과제별 역전(무작위화가 더 쉬운 과제 수)
v9 = {}
for n in T5_NAMES:
    d = rand[n] - clean[n]
    v9[n] = {"n_tasks_rand_gt_clean": int((d > 0).sum()), "n_tasks_rand_ge_clean": int((d >= 0).sum()),
             "max_rand_minus_clean": float(d.max()), "task_of_max": tasks[int(np.argmax(d))]}
OUT["V9_rand_gt_clean"] = v9

# ------------------------------------------------ V10. 혼합 비율 대 효과(용량-반응)와 단조성
ego_share = {"Robot-only": 0.0, "Ego2R(1:3)": 0.25, "Ego2R(1:1)": 0.5, "Ego2R(3:1)": 0.75}
OUT["V10_dose_response"] = {c: {f"{ego_share[n]:.2f}": T1[n][j] for n in sorted(MODELS, key=lambda m: ego_share[m])}
                            for j, c in enumerate(T1_COLS)}
# 같은 비율 세 번(1:3, 1:1, 3:1) 중 1:1이 최댓값인 열: 비단조
OUT["V10_nonmonotone_cols"] = [c for j, c in enumerate(T1_COLS)
                              if T1["Ego2R(1:1)"][j] > max(T1["Ego2R(1:3)"][j], T1["Ego2R(3:1)"][j])]

# ------------------------------------------------ V11. 데이터 양과 노출 비율
hours = {"ANT": 7, "EgoDex": 732, "ViTRA": 249, "EgoVerse": 954}
robot_h = {"DROID": 511, "AgibotWorld": 2404, "InternData": 3650}
steps, bs, gpus = 200_000, 12, 8
frames = steps * bs * gpus
ego_total = sum(hours.values())
v11 = {
    "ego_sources_sum_h": ego_total, "paper_says_approx_h": 1940,
    "robot_sum_h": sum(robot_h.values()), "paper_says_h": 6565,
    "frames_per_run": frames,
    "amplification_18561_over_ego": round(18561 / ego_total, 3), "amplification_vs_1940": round(18561 / 1940, 3),
    "ideal_15x_hours": ego_total * 15, "kept_fraction_of_15x": round(18561 / (ego_total * 15), 3),
    "per_morph_hours": round(18561 / 15, 1),
    "ablation_hours_if_linear": {k: round(18561 / 15 * k, 0) for k in (1, 5, 10, 15)},
    "ablation_hours_reported": {"1": 1237, "5": 6187, "10": 12374, "15": 18561},
    "raw_plus_15": 18561 + 1940, "raw_plus_15_reported": 20501,
}
v11["exposure"] = {}
for fps in (10, 15, 30):
    robot_frames = sum(robot_h.values()) * 3600 * fps
    ego_frames = 18561 * 3600 * fps
    v11["exposure"][str(fps)] = {
        "robot_only_epochs": round(frames / robot_frames, 4),
        "1:1_robot_epochs": round(0.5 * frames / robot_frames, 4),
        "1:1_ego_epochs": round(0.5 * frames / ego_frames, 4),
        "3:1_ego_epochs": round(0.75 * frames / ego_frames, 4),
        "1:3_ego_epochs": round(0.25 * frames / ego_frames, 4),
        "1:1_ego_hours_seen": round(0.5 * frames / (3600 * fps), 0),
    }
OUT["V11_data_and_exposure"] = v11

# ------------------------------------------------ V12. 그림 3 절제(값은 그림 3에서 읽음)
fig3 = {"Ego(raw)": 28.1, "Ego2R(1)": 31.7, "Ego2R(5)": 32.3, "Ego2R(10)": 33.1, "Ego2R(15)": 33.5, "Ego2R(15)+Ego": 37.3}
v12 = {"values": fig3}
def z_pair(a, b):
    z, p, se = two_prop(a, b)
    return {"diff": r1(b - a), "z": round(z, 2), "p": round(p, 4)}
v12["pipeline_value_1_vs_raw"] = z_pair(28.1, 31.7)
v12["morph_1_to_15"] = z_pair(31.7, 33.5)
v12["morph_5_to_15"] = z_pair(32.3, 33.5)
v12["raw_added"] = z_pair(33.5, 37.3)
v12["text_pipeline_gain_check"] = r1(31.7 - 28.1)
steps_m = [31.7, 32.3, 33.1, 33.5]
v12["increments"] = [r1(steps_m[i + 1] - steps_m[i]) for i in range(3)]
# 형태학수에 대한 선형 추세 기울기 (1,5,10,15)
x = np.array([1, 5, 10, 15]); y = np.array(steps_m)
slope = np.polyfit(x, y, 1)[0]
v12["slope_pp_per_morphology"] = round(float(slope), 3)
OUT["V12_ablation"] = v12

# ------------------------------------------------ V13. 실로봇(그림 4 값) 점수 가능성
step_w = {"Put Fruits": 100 / 3, "Put Blocks": 25.0, "Fold Towel": 50.0, "Sweep Trash": 25.0, "Insert Screw": 25.0}
n_steps = {"Put Fruits": 3, "Put Blocks": 4, "Fold Towel": 2, "Sweep Trash": 4, "Insert Screw": 4}
step_w_table4 = dict(step_w)
step_w_table4["Put Fruits"] = 33.3
fig4 = {  # 그림 4 막대 위 숫자 (Scratch, Robot-only, Mix, Mix+Play)
    "Put Fruits": [3, 23, 25, 31], "Put Blocks": [12, 35, 38, 49], "Fold Towel": [0, 25, 25, 30],
    "Sweep Trash": [0, 1, 5, 14], "Insert Screw": [0, 2, 6, 15],
}
CONF = ["Scratch", "Robot-only", "Mix", "Mix+Play"]
v13 = {"cells": {}, "infeasible": []}
for task, vals in fig4.items():
    for cname, v in zip(CONF, vals):
        w = step_w_table4[task]
        reach = [k * w / 20 for k in range(0, 20 * n_steps[task] + 1)]
        # 20 시행, 각 시행 점수는 w의 정수배(0..n_steps). 평균 = w * (총 단계 수) / 20
        feas = [r for r in reach if round(r) == v]
        v13["cells"][f"{task}|{cname}"] = {"value": v, "feasible": bool(feas), "nearest_means": [round(min(reach, key=lambda r: abs(r - v)), 2)]}
        if not feas:
            v13["infeasible"].append({"cell": f"{task}|{cname}", "value": v, "nearest_attainable": round(min(reach, key=lambda r: abs(r - v)), 2)})
OUT["V13_real_robot_feasibility"] = v13
diff_play_robot = {t: fig4[t][3] - fig4[t][1] for t in fig4}
diff_play_mix = {t: fig4[t][3] - fig4[t][2] for t in fig4}
diff_mix_robot = {t: fig4[t][2] - fig4[t][1] for t in fig4}
def pooled(diffs, sd_trial):
    d = np.array(list(diffs.values()), float)
    se = sd_trial * math.sqrt(2 / (20 * len(d)))
    z = d.mean() / se
    return {"mean_diff": round(float(d.mean()), 2), "se_with_sd%d" % sd_trial: round(se, 2), "z": round(z, 2),
            "p": round(float(2 * (1 - stats.norm.cdf(abs(z)))), 4)}
OUT["V13_real_robot_diffs"] = {
    "play_minus_robot": diff_play_robot, "play_minus_mix": diff_play_mix, "mix_minus_robot": diff_mix_robot,
    "pooled_play_vs_robot": {str(sd): pooled(diff_play_robot, sd) for sd in (30, 40, 50)},
    "pooled_mix_vs_robot": {str(sd): pooled(diff_mix_robot, sd) for sd in (30, 40, 50)},
    "per_task_se_diff": {str(sd): round(sd * math.sqrt(2 / 20), 1) for sd in (30, 40, 50)},
    "sign_test_p_5of5_one_sided": round(0.5 ** 5, 4),
    "text_check": {"put_blocks_gain_text": 14, "computed": diff_play_robot["Put Blocks"],
                   "insert_screw_gain_text": 13, "computed2": diff_play_robot["Insert Screw"]},
    "largest_gain_task": max(diff_play_robot, key=diff_play_robot.get),
    "mean_play": round(float(np.mean([fig4[t][3] for t in fig4])), 1),
    "mean_robot": round(float(np.mean([fig4[t][1] for t in fig4])), 1),
    "mean_mix": round(float(np.mean([fig4[t][2] for t in fig4])), 1),
    "mean_scratch": round(float(np.mean([fig4[t][0] for t in fig4])), 1),
}
OUT["V13_table4_step_sums"] = {t: round(step_w_table4[t] * n_steps[t], 1) for t in step_w_table4}

# ------------------------------------------------ V14. 수치 내부 일관성(기타)
v14 = {}
v14["episodes_eval_cell"] = N_CELL
v14["real_robot_ego_play_minutes_per_episode_s"] = round(35 * 60 / 675, 2)
v14["finetune_demos_robotwin"] = 50 * 50
v14["ebench_finetune_demos"] = 7 * 400
v14["speed_align_factors"] = {"60%": round(1 / 0.6, 2), "45%": round(1 / 0.45, 2), "25%": round(1 / 0.25, 2)}
v14["pretrain_batch"] = 12 * 8
v14["frames_200k_x_96"] = 200_000 * 96
# 도달 반경 대비 표 3 (논문)과 코드 ARM_REACH의 비교는 04에서 한다. 표 3의 FR3 = Panda 1.272 일치 여부
v14["table3_panda_equals_fr3_reach"] = True
OUT["V14_misc"] = v14

# ---- 평균 Pi0.5 대비
OUT["Pi05_vs_robot_only"] = {"clean": [avg_row[0], avg_row[2]], "rand": [avg_row[1], avg_row[3]],
                            "tasks_pi05_better_clean": int((clean["Pi0.5"] > clean["Robot-only"]).sum()),
                            "tasks_pi05_better_rand": int((rand["Pi0.5"] > rand["Robot-only"]).sum())}
# 과제 수준 상관(Clean vs Rand)
OUT["task_corr_clean_rand"] = {n: round(float(np.corrcoef(clean[n], rand[n])[0, 1]), 3) for n in T5_NAMES}
# 1:1 이 가장 좋은 과제 수
best_clean = np.argmax(np.stack([clean[n] for n in MODELS]), axis=0)
best_rand = np.argmax(np.stack([rand[n] for n in MODELS]), axis=0)
OUT["tasks_best_by_model"] = {"clean": {MODELS[i]: int((best_clean == i).sum()) for i in range(4)},
                              "rand": {MODELS[i]: int((best_rand == i).sum()) for i in range(4)}}
# 가장 크게 오르내린 과제
d11_clean = clean["Ego2R(1:1)"] - clean["Robot-only"]
d11_rand = rand["Ego2R(1:1)"] - rand["Robot-only"]
top_up = np.argsort(-d11_rand)[:5]; top_dn = np.argsort(d11_rand)[:5]
OUT["rand_1:1_vs_robot_extremes"] = {"up": [(tasks[i], float(d11_rand[i])) for i in top_up],
                                    "down": [(tasks[i], float(d11_rand[i])) for i in top_dn]}
# 상위 5개 과제를 빼면 Rand +2.6은 어떻게 되나
mask = np.ones(50, bool); mask[top_up] = False
OUT["rand_1:1_gain_without_top5_tasks"] = round(float(d11_rand[mask].mean()), 2)
OUT["rand_1:1_gain_median_task"] = float(np.median(d11_rand))

(ROOT / "analysis" / "recompute_results.json").write_text(json.dumps(OUT, indent=1, ensure_ascii=False), encoding="utf-8")

# ------------------------------------------------ 요약 출력
print("V1 max diff vs avg row:", {n: v1[n]["max_abs_diff_vs_avg_row"] for n in v1})
print("V2 odd share", OUT["V2_parity"]["n_odd"], "/", OUT["V2_parity"]["n_values"], OUT["V2_parity"]["per_model"], "pi05 odd", OUT["V2_parity"]["pi05_odd_count"])
print("V3 mismatch T1", v3, "T2", v3b)
print("V4 mismatch", OUT["V4_group_mismatch"])
print("V5", OUT["V5_text_claims"]["1:1_leads_in"], OUT["V5_text_claims"]["table2_n_positive_of_11"], OUT["V5_text_claims"]["table1_n_positive_of_7"])
print("V6 mc", OUT["V6_table2_multiple_comparison"])
for n in MODELS[1:]:
    print("V7", n, v7[n]["clean"], v7[n]["rand"])
print("V8", v8)
print("V8z", {k: (v["diff"], v["z"], v["p"]) for k, v in v8z.items()})
print("V8z100", {k: (v["diff"], v["z"], v["p"]) for k, v in v8z100.items()})
print("V9", {n: v9[n]["n_tasks_rand_gt_clean"] for n in v9})
print("V10 nonmono", OUT["V10_nonmonotone_cols"])
print("V11", {k: v11[k] for k in v11 if k != "exposure"}, v11["exposure"])
print("V12", v12)
print("V13 infeasible", v13["infeasible"], OUT["V13_real_robot_diffs"])
print("extra", OUT["Pi05_vs_robot_only"], OUT["task_corr_clean_rand"], OUT["tasks_best_by_model"])
print("rand extremes", OUT["rand_1:1_vs_robot_extremes"], OUT["rand_1:1_gain_without_top5_tasks"], OUT["rand_1:1_gain_median_task"])
