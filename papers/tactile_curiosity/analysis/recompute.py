"""그림에서 읽은 값(figure_values.json)으로 논문의 주장을 다시 계산한다. 결과는 recompute_results.json.

논문 v3에는 결과 표가 없고 그림 2, 3, 4, 6만 있다. 그래서 다음을 가정한다(문장마다 같은 가정을 적는다).
  - 그림 2, 3의 오차 막대는 SEM이고 시드 5개(본문 4절 첫 문단, 그림 캡션).
  - 그림 4, 6의 오차는 표준오차이고 과제당 시드 10개(그림 4, 6 캡션).
  - 시드별 값은 서로 독립이고 평균 근처에서 대략 정규(검정은 Welch t, 자유도는 Welch-Satterthwaite).
  - 오차 막대가 마커 아래로 숨어 읽히지 않은 값(se = None)은 마커 반지름(약 4.5 pt)에 해당하는 값을 se의 상한으로 둔다
    (행마다 축 길이가 달라 0.020, 0.034, 0.014). 이는 유의성을 보수적으로 만드는 선택이다.
이 가정들 때문에 p 값은 "읽은 값에서 대략 이 정도"로만 쓴다.

실행: python3 analysis/extract_figure_values.py 를 먼저 돌려 figure_values.json을 만든 뒤
      python3 analysis/recompute.py
"""
import json
import math
from itertools import combinations
from pathlib import Path

from scipy import stats

HERE = Path(__file__).resolve().parent
FV = json.loads((HERE / "figure_values.json").read_text(encoding="utf-8"))
OUT = HERE / "recompute_results.json"
SE_HIDDEN = {"Ketchup in tray": 0.020, "Soup in tray": 0.034, "Butter in tray": 0.034,
             "Book to back caddy": 0.034, "Stove and frying pan": 0.014}
R = {}


def welch(m1, se1, n1, m2, se2, n2):
    """평균과 표준오차로 Welch t 검정(양측). 시드 수 n은 가정."""
    v1, v2 = se1 ** 2, se2 ** 2
    se = math.sqrt(v1 + v2)
    t = (m1 - m2) / se
    df = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))
    p = 2 * stats.t.sf(abs(t), df)
    return {"diff": round(m1 - m2, 4), "t": round(t, 3), "df": round(df, 1), "p": float(f"{p:.3g}")}


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    m = len(ps)
    adj = [None] * m
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        adj[i] = run
    return adj


# ---------------- V1: 그림 2b (단일 물체, 시드 5개 가정) ----------------
b = FV["fig2b"]["bars"]
n = 5
v1 = {"n_assumed": n, "drq_dashed": FV["fig2b"]["drq_dashed"], "implied_sd": {}, "vs_RE": {}, "pairs": {}, "vs_DrQ_line": {}}
for k, v in b.items():
    v1["implied_sd"][k] = round(v["sem"] * math.sqrt(n), 1)
for k in ("T", "TV", "TVL", "V", "L", "VL"):
    v1["vs_RE"][k] = welch(b[k]["mean"], b[k]["sem"], n, b["RE"]["mean"], b["RE"]["sem"], n)
for a, c in (("T", "TV"), ("TV", "TVL"), ("T", "TVL")):
    v1["pairs"][f"{a} vs {c}"] = welch(b[a]["mean"], b[a]["sem"], n, b[c]["mean"], b[c]["sem"], n)
drq = FV["fig2b"]["drq_dashed"]
for k in ("T", "TV", "TVL"):
    t = (b[k]["mean"] - drq) / b[k]["sem"]
    v1["vs_DrQ_line"][k] = {"diff": round(b[k]["mean"] - drq, 1), "z_vs_line_only": round(t, 2),
                            "p_one_sample_t_df4": float(f"{2 * stats.t.sf(abs(t), n - 1):.3g}")}
# 시드 결과가 바닥(RE 평균 약 104)과 높은 값 H 두 가지뿐이라는 가정에서, 평균과 SD를 맞추는 성공 시드 수 k
floor = b["RE"]["mean"]
bim = {}
for k in ("T", "TV", "TVL"):
    m, sd_obs = b[k]["mean"], v1["implied_sd"][k]
    cand = []
    for ksucc in range(1, 5):
        H = floor + (m - floor) * n / ksucc
        vals = [floor] * (n - ksucc) + [H] * ksucc
        mu = sum(vals) / n
        sd = math.sqrt(sum((x - mu) ** 2 for x in vals) / (n - 1))
        cand.append({"k_success_seeds": ksucc, "H": round(H, 0), "sd": round(sd, 0)})
    bim[k] = {"sd_observed": sd_obs, "two_point_cases": cand}
v1["bimodal_check"] = bim
R["V1_fig2b"] = v1

# ---------------- V2: 그림 3b, 3c (다중 물체, 시드 5개 가정) ----------------
f3b, f3c = FV["fig3b"], FV["fig3c"]
v2 = {"n_assumed": 5,
      "b_T_vs_TV": welch(f3b["T"]["mean"], f3b["T"]["sem"], 5, f3b["TV"]["mean"], f3b["TV"]["sem"], 5),
      "b_T_vs_RE": welch(f3b["T"]["mean"], f3b["T"]["sem"], 5, f3b["RE"]["mean"], f3b["RE"]["sem"], 5),
      "b_TV_vs_RE": welch(f3b["TV"]["mean"], f3b["TV"]["sem"], 5, f3b["RE"]["mean"], f3b["RE"]["sem"], 5),
      "b_ratio_T_over_TV": round(f3b["T"]["mean"] / f3b["TV"]["mean"], 2),
      "c_T_vs_TV": welch(f3c["T"]["mean"], f3c["T"]["sem"], 5, f3c["TV"]["mean"], f3c["TV"]["sem"], 5),
      "c_TV_read": f3c["TV"], "c_T_read": f3c["T"],
      "text_claim_TV_effective_objects": "약 2 (본문 4.1절)", }
R["V2_fig3"] = v2

# ---------------- V3: 그림 2a, 3a 곡선의 끝 구간 ----------------
R["V3_grasp_fraction"] = FV["grasp_fraction_curves"]

# ---------------- V4: 그림 6 (과제당 시드 10개 가정) ----------------
f6 = FV["fig6_final_return"]
rows = list(f6.keys())


def pick(row, name):
    lst = f6[row].get(name)
    if lst is None:
        return None
    if name == "DSRL-PPO or DSRL-SAC":
        return None
    m = lst[0]
    return m["mean"], (m["se"] if m["se"] is not None else SE_HIDDEN[row])


def sac_ppo(row):
    lst = f6[row]["DSRL-PPO or DSRL-SAC"]
    lst = sorted(lst, key=lambda x: x["mean"])
    ppo, sac = lst[0], lst[-1]  # PPO는 평균이 0에 가까운 점, SAC는 나머지 점
    return ((sac["mean"], sac["se"] if sac["se"] is not None else SE_HIDDEN[row]),
            (ppo["mean"], ppo["se"] if ppo["se"] is not None else SE_HIDDEN[row]))


n6 = 10
v4 = {"n_assumed": n6, "se_hidden_upper_bound": SE_HIDDEN, "rows": {}, "holm_family": "TacEx 세 가지 x 5과제 vs DSRL-SAC (15개)"}
fam, fam_keys = [], []
for row in rows:
    sac, ppo = sac_ppo(row)
    info = {"DSRL-SAC": sac, "DSRL-PPO(점이 0 근처)": ppo}
    for name in ("Tactile", "Tactile+vision", "Tactile+latent"):
        m, se = pick(row, name)
        w = welch(m, se, n6, sac[0], sac[1], n6)
        info[f"{name} vs SAC"] = w
        fam.append(w["p"])
        fam_keys.append((row, name))
    # 촉각 목표가 없는 보너스 중 가장 높은 것과 비교
    non = {k: pick(row, k) for k in ("Latent", "Image", "State")}
    best_name = max(non, key=lambda k: non[k][0])
    for name in ("Tactile", "Tactile+vision"):
        m, se = pick(row, name)
        info[f"{name} vs best non-tactile-target ({best_name})"] = welch(m, se, n6, non[best_name][0], non[best_name][1], n6)
    info["non-tactile-target means"] = {k: v[0] for k, v in non.items()}
    info["summed taxels"] = {"info_gain": pick(row, "Summed taxels: info gain"), "reward": pick(row, "Summed taxels: reward")}
    v4["rows"][row] = info
adj = holm(fam)
v4["holm_adjusted_p"] = {f"{r_} / {nm}": round(a, 3) for (r_, nm), a in zip(fam_keys, adj)}
# TV 대 SAC를 과제 단위로 모은 짝지은 비교(과제 5개)
diffs = []
for row in rows:
    sac, _ = sac_ppo(row)
    m, _ = pick(row, "Tactile+vision")
    diffs.append(m - sac[0])
t, p = stats.ttest_1samp(diffs, 0.0)
v4["task_level_TV_minus_SAC"] = {"diffs": [round(d, 3) for d in diffs], "mean": round(sum(diffs) / 5, 3), "t": round(float(t), 2), "p_df4": float(f"{p:.3g}")}

# 본문 "Tactile+vision achieves the highest reported mean final return on all five tasks" 확인
top = {}
for row in rows:
    means = {}
    for name in ("Tactile", "Tactile+vision", "Tactile+latent", "Latent", "Image", "State", "DPPO",
                 "Summed taxels: info gain", "Summed taxels: reward"):
        means[name] = f6[row][name][0]["mean"]
    sac, _ = sac_ppo(row)
    means["DSRL-SAC"] = sac[0]
    best = max(means, key=means.get)
    second = sorted(means.values())[-2]
    top[row] = {"best": best, "TV": means["Tactile+vision"], "runner_up_value": second,
                "margin_TV_over_runner_up": round(means["Tactile+vision"] - sorted([v for k, v in means.items() if k != "Tactile+vision"])[-1], 3)}
v4["tv_is_highest_check"] = top
R["V4_fig6"] = v4

# ---------------- V5: 그림 4 끝점, 8과제 (오차 읽지 않음) ----------------
f4 = FV["fig4_final"]
v5 = {"tasks": {}, "counts": {}}
for tid, v in f4.items():
    sac = v["DSRL-SAC"]["final"]
    row = {"SAC": sac}
    for k, nm in (("TacEx tactile", "T"), ("TacEx vision+tactile", "TV"), ("TacEx latent+tactile", "TL")):
        row[nm] = v[k]["final"]
        row[f"{nm}-SAC"] = round(v[k]["final"] - sac, 3)
    row["segments"] = v["DSRL-SAC"]["segments"]
    row["last_step_SAC"] = v["DSRL-SAC"]["last_step"]
    row["last_step_TacEx"] = v["TacEx tactile"]["last_step"]
    v5["tasks"][f"{tid} {v['task']}"] = row
for nm in ("T", "TV", "TL"):
    ds = [r[f"{nm}-SAC"] for r in v5["tasks"].values()]
    v5["counts"][nm] = {"gap_gt_0.10": sum(d > 0.10 for d in ds), "gap_gt_0.05": sum(d > 0.05 for d in ds),
                       "gap_abs_le_0.03": sum(abs(d) <= 0.03 for d in ds), "negative_lt_-0.03": sum(d < -0.03 for d in ds),
                       "mean_gap": round(sum(ds) / len(ds), 3)}
# 베이스라인이 이미 0.9 이상인 과제
v5["baseline_ge_0.9"] = [k for k, r in v5["tasks"].items() if r["SAC"] >= 0.9]
R["V5_fig4_endpoints"] = v5

# ---------------- V6: 그림 4 대 그림 6 일관성 (5개 과제) ----------------
map_task = {"58": "Ketchup in tray", "55": "Soup in tray", "56": "Butter in tray", "77": "Book to back caddy", "21": "Stove and frying pan"}
cons = {}
maxdiff = 0.0
for tid, row in map_task.items():
    v = f4[tid]
    sac6, _ = sac_ppo(row)
    c = {"SAC fig4": v["DSRL-SAC"]["final"], "SAC fig6": sac6[0],
         "T fig4": v["TacEx tactile"]["final"], "T fig6": pick(row, "Tactile")[0],
         "TV fig4": v["TacEx vision+tactile"]["final"], "TV fig6": pick(row, "Tactile+vision")[0],
         "TL fig4": v["TacEx latent+tactile"]["final"], "TL fig6": pick(row, "Tactile+latent")[0]}
    for nm in ("SAC", "T", "TV", "TL"):
        maxdiff = max(maxdiff, abs(c[f"{nm} fig4"] - c[f"{nm} fig6"]))
    cons[f"{tid} {row}"] = c
R["V6_fig4_vs_fig6"] = {"rows": cons, "max_abs_diff": round(maxdiff, 3),
                        "note": "그림 4 끝점(마지막 평가)과 그림 6 최종 반환(마지막 10% 구간 시간 가중 평균)은 정의가 달라 완전히 같을 필요는 없다"}

# ---------------- V7: 표 1 산술 (UTD, 단계 수, 평가 간격) ----------------
grad_updates = 500_000
utd = 20
query_freq = 20
queries = grad_updates / utd
R["V7_table1_arithmetic"] = {
    "gradient_updates_(max_steps)": grad_updates, "utd_per_query": utd, "queries": queries,
    "env_steps_upper_bound_(queries x query_freq)": queries * query_freq,
    "task69_updates": 1_000_000, "task69_queries": 1_000_000 / utd,
    "eval_interval_updates": 20_000, "points_in_5e5_curve": grad_updates // 20_000 + 1,
    "points_read_from_pdf_segments": f4["58"]["DSRL-SAC"]["segments"] + 1,
    "points_task69_read_from_pdf": f4["69"]["DSRL-SAC"]["segments"] + 1,
    "note": "저장소 README는 max_steps가 경사 갱신 횟수이고 multi_grad_step은 저장된 조종 전이 한 개당 갱신 횟수라고 적는다. "
            "조종 전이 한 개는 최대 query_freq = 20 환경 스텝이므로 표 1의 'UTD 20 경사 스텝/환경 스텝'은 전이 단위로 읽어야 계산이 맞는다."}

# ---------------- V8: 효과 크기 (그림 6에서 SAC 대비 이득의 상한 해석) ----------------
R["V8_ablation_fig5_notes"] = {
    "task58_fig5_endpoints_read": "그림 5는 곡선만 있어 끝점을 눈으로 읽음: SAC 0.16, Tac-SAC 약 0.22, TacEx(tactile) 약 0.38, MaxinfoSac(state, 촉각 관측 없음) 약 0.03, TacEx(state) 약 0.32",
    "decomposition_task58": {"observation_only (Tac-SAC - SAC)": 0.06, "exploration_with_tactile_bonus (TacEx tactile - Tac-SAC)": 0.16},
}


# ---------------- V9: 시드 수 (80% 검정력, 양측 0.05, 정규 근사) ----------------
def seeds_needed(m1, sd1, m2, sd2, power=0.80, alpha=0.05):
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return math.ceil((za + zb) ** 2 * (sd1 ** 2 + sd2 ** 2) / (m1 - m2) ** 2)


sd2b = {k: b[k]["sem"] * math.sqrt(5) for k in b}
sd3b = {k: f3b[k]["sem"] * math.sqrt(5) for k in f3b}
R["V9_seeds_needed_per_arm"] = {
    "assumption": "읽은 SD(SEM x sqrt(5)) 사용, 정규 근사, 80% 검정력, 양측 0.05, 두 팔 시드 수 같음 (추정)",
    "fig2b_T_vs_RE": seeds_needed(b["T"]["mean"], sd2b["T"], b["RE"]["mean"], sd2b["RE"]),
    "fig2b_TV_vs_RE": seeds_needed(b["TV"]["mean"], sd2b["TV"], b["RE"]["mean"], sd2b["RE"]),
    "fig2b_T_vs_TV": seeds_needed(b["T"]["mean"], sd2b["T"], b["TV"]["mean"], sd2b["TV"]),
    "fig3b_T_vs_TV": seeds_needed(f3b["T"]["mean"], sd3b["T"], f3b["TV"]["mean"], sd3b["TV"]),
    "fig3b_T_vs_RE": seeds_needed(f3b["T"]["mean"], sd3b["T"], f3b["RE"]["mean"], sd3b["RE"]),
}

OUT.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: R[k] for k in ("V1_fig2b", "V2_fig3")}, ensure_ascii=False, indent=1))
print(json.dumps(R["V4_fig6"], ensure_ascii=False, indent=1)[:6000])
print(json.dumps(R["V5_fig4_endpoints"], ensure_ascii=False, indent=1))
print(json.dumps(R["V6_fig4_vs_fig6"], ensure_ascii=False, indent=1))
