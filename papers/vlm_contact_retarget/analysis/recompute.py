"""REACT (arXiv:2610.28523 v1) 보고 수치 재계산과 일관성 검사 (CPU, numpy/scipy, 수 초).

입력: paper/paper.txt (pdftotext -layout 결과). 표 행은 원문 텍스트에서 읽고, 읽은 값은 results에 함께 저장한다.
출력: analysis/recompute_results.json
실행: python3 analysis/recompute.py
검사 번호 V1–V16은 03_analysis.md의 절과 같다.
"""
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
TEXT = (ROOT / "paper" / "paper.txt").read_text(encoding="utf-8")
LINES = [l.replace("−", "-") for l in TEXT.splitlines()]
NUM = re.compile(r"-?\d+(?:\.\d+)?")
OUT = {}


def table_rows(caption, labels, window=40):
    """caption 줄을 찾은 뒤 window 줄 안에서 각 label로 시작하는 줄의 숫자를 읽는다."""
    start = next(i for i, l in enumerate(LINES) if caption in l)
    rows = {}
    for lab in labels:
        for l in LINES[start:start + window]:
            s = l.strip()
            rest = s[len(lab):]
            if s.startswith(lab) and rest.strip()[:1] in "0123456789-+":
                rows[lab] = [float(x) for x in NUM.findall(rest)]
                break
        else:
            raise SystemExit(f"row not found: {caption} / {lab}")
    return rows


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def r(x, k=3):
    return round(float(x), k)


# ---------------------------------------------------------------- V1 표 1, 9, 10 일관성
t1 = table_rows("Table 1: Contact-aware retargeting",
                ["GMR", "OmniRetarget", "Ours w/o Stage 2 co-opt", "Ours w/o VLM filter", "Ours"], 12)
# 열: jitter(m,s) pen(m,s) recall(m,s) prec(m,s) user(m,sem); Stage 2 co-opt 행은 user 없음
T1 = {}
for k, v in t1.items():
    T1[k] = dict(jit=v[0:2], pen=v[2:4], rec=v[4:6], prec=v[6:8], user=v[8:10] if len(v) >= 10 else None)

t9 = table_rows("Table 9: Retargeting Recall vs. contact threshold", ["GMR", "OmniRetarget", "Ours w/o Stage 2 co-opt",
                                                                       "Ours w/o VLM filtering", "Ours"], 12)
t10 = table_rows("Table 10: Per-action retargeting Recall and Precision",
                 ["GMR", "OmniRetarget", "Ours w/o Stage 2 co-opt", "Ours w/o VLM filtering", "Ours"], 20)
# T10 열: hug R,P / hand-shake R,P / pat R,P / high-five R,P / link R,P / total R,P
v1 = {"table9_eps10_equals_table1_recall": {}, "table10_total_equals_table1": {}, "table10_mean_of_actions": {}}
name9 = {"Ours w/o VLM filtering": "Ours w/o VLM filter"}
for m, vals in t9.items():
    v1["table9_eps10_equals_table1_recall"][m] = dict(table9=vals[2], table1=T1[name9.get(m, m)]["rec"][0])
for m, vals in t10.items():
    tot_r, tot_p = vals[10], vals[11]
    Rs = vals[0:10:2]
    Ps = vals[1:10:2]
    t1m = T1[name9.get(m, m)]
    v1["table10_total_equals_table1"][m] = dict(t10_R=tot_r, t1_R=t1m["rec"][0], t10_P=tot_p, t1_P=t1m["prec"][0])
    v1["table10_mean_of_actions"][m] = dict(R_mean=r(np.mean(Rs), 2), R_total=tot_r, diff_R=r(tot_r - np.mean(Rs), 2),
                                            P_mean=r(np.mean(Ps), 2), P_total=tot_p, diff_P=r(tot_p - np.mean(Ps), 2))
v1["note"] = ("재현율은 행동당 GT 클립 16개로 가중이 같으므로 총합 = 행동별 평균이어야 한다(OmniRetarget은 58개라 예외). "
              "Ours 59.08 대 59.4, w/o VLM 57.78 대 58.0처럼 0.2–0.35 어긋난다. 정밀도는 예측이 있는 클립만 평균하므로 어긋나도 된다.")
v1["events"] = dict(pred=1740, gt=203, ratio=r(1740 / 203, 2), gt_per_clip=r(203 / 80, 2))
OUT["V1_table_consistency"] = v1

# ---------------------------------------------------------------- V2 본문 문장 대 표 1
OUT["V2_text_vs_table1"] = dict(
    stage2_recall_jump=(T1["Ours w/o Stage 2 co-opt"]["rec"][0], T1["Ours w/o VLM filter"]["rec"][0]),
    vlm_precision=(T1["Ours w/o VLM filter"]["prec"][0], T1["Ours"]["prec"][0]),
    vlm_recall=(T1["Ours w/o VLM filter"]["rec"][0], T1["Ours"]["rec"][0]),
    user_rank_gain=r(T1["Ours w/o VLM filter"]["user"][0] - T1["Ours"]["user"][0], 2),
    omni_infeasible_clips=dict(feasible=766, total=1100, infeasible_frac=r(1 - 766 / 1100, 3)),
    omni_gt_clips=dict(feasible=58, total=80, infeasible_frac=r(1 - 58 / 80, 3)),
    stage1_vs_omni=dict(pen=(T1["Ours w/o Stage 2 co-opt"]["pen"][0], T1["OmniRetarget"]["pen"][0]),
                        prec=(T1["Ours w/o Stage 2 co-opt"]["prec"][0], T1["OmniRetarget"]["prec"][0]),
                        jitter=(T1["Ours w/o Stage 2 co-opt"]["jit"][0], T1["GMR"]["jit"][0])),
)

# ---------------------------------------------------------------- V3 표 11 (사용자 연구)
t11 = table_rows("Table 11: User study: per-question mean ranks", ["GMR", "OmniRetarget", "Ours w/o VLM filtering", "Ours"], 12)
U = {m: [(v[0], v[1]), (v[2], v[3]), (v[4], v[5])] for m, v in t11.items()}
v3 = {"rank_sum_per_question": [r(sum(U[m][q][0] for m in U), 2) for q in range(3)],
      "table1_user_equals_table11_q3": {("Ours w/o VLM filter" if m == "Ours w/o VLM filtering" else m):
                                        (T1[("Ours w/o VLM filter" if m == "Ours w/o VLM filtering" else m)]["user"], U[m][2])
                                        for m in U}}
sd = {m: [r(U[m][q][1] * math.sqrt(28), 3) for q in range(3)] for m in U}
v3["implied_participant_sd"] = sd
gaps = []
for q in range(3):
    for a, b in [("GMR", "OmniRetarget"), ("OmniRetarget", "Ours w/o VLM filtering"), ("Ours w/o VLM filtering", "Ours")]:
        d = U[a][q][0] - U[b][q][0]
        se = math.sqrt(U[a][q][1] ** 2 + U[b][q][1] ** 2)
        gaps.append(dict(q=q + 1, a=a, b=b, gap=r(d, 2), z_independent=r(d / se, 1)))
v3["adjacent_gaps"] = gaps
v3["note"] = ("한 참가자의 순위 합은 항상 10이므로 방법 사이 순위는 음의 상관이다. 독립 가정 z는 짝 비교의 보수적 값이 아니라 대략치이다. "
              "참가자 40명 중 28명(70%)만 80시행을 끝냈다.")
OUT["V3_user_study"] = v3

# ---------------------------------------------------------------- V4 표 1 요약통계 검정
def welch(m1, s1, n1, m2, s2, n2):
    t, p = stats.ttest_ind_from_stats(m1, s1, n1, m2, s2, n2, equal_var=False)
    return dict(diff=r(m1 - m2, 2), t=r(t, 2), p=float(f"{p:.3g}"))


v4 = {}
a, b, c = T1["Ours"], T1["Ours w/o VLM filter"], T1["OmniRetarget"]
v4["recall_ours_vs_noVLM_n80"] = welch(a["rec"][0], a["rec"][1], 80, b["rec"][0], b["rec"][1], 80)
v4["precision_ours_vs_noVLM_n80_upper"] = welch(a["prec"][0], a["prec"][1], 80, b["prec"][0], b["prec"][1], 80)
v4["recall_ours_vs_omni_n80_58"] = welch(a["rec"][0], a["rec"][1], 80, c["rec"][0], c["rec"][1], 58)
v4["precision_ours_vs_omni_n58_upper"] = welch(a["prec"][0], a["prec"][1], 58, c["prec"][0], c["prec"][1], 58)
v4["jitter_ours_vs_noVLM_n1100"] = welch(a["jit"][0], a["jit"][1], 1100, b["jit"][0], b["jit"][1], 1100)
v4["penetration_ours_vs_noVLM_n1100"] = welch(a["pen"][0], a["pen"][1], 1100, b["pen"][0], b["pen"][1], 1100)
v4["jitter_ours_vs_omni_n1100_766"] = welch(a["jit"][0], a["jit"][1], 1100, c["jit"][0], c["jit"][1], 766)
v4["penetration_ours_vs_omni_n1100_766"] = welch(a["pen"][0], a["pen"][1], 1100, c["pen"][0], c["pen"][1], 766)
v4["note"] = ("같은 클립을 쓰는 비교라 짝 검정이 더 강하지만 클립별 값이 없어 독립(Welch)로 근사했다. 정밀도의 n은 상한이다"
              "(예측이 없는 클립 제외). 재현율 +1.4%p(VLM)는 독립 근사로 구분되지 않는다.")
OUT["V4_table1_tests"] = v4

# ---------------------------------------------------------------- V5 표 2 교사
t2 = table_rows("Table 2: Teacher tracking results", ["Ours", "(-) mesh contact reward", "(-) VLM contact filtering"], 8)
# 열: HF mpjpe, contact m, s, hug mpjpe, m, s, link mpjpe, m, s
names = ["high-five", "hug", "link-arms"]
T2 = {}
for k, v in t2.items():
    T2[k] = {names[i]: dict(mpjpe=v[3 * i], contact=v[3 * i + 1], sd=v[3 * i + 2]) for i in range(3)}
v5 = {"deltas_mesh_reward": {}, "deltas_vlm_filter": {}, "mpjpe_vlm_filter_delta_cm": {}}
for n in names:
    v5["deltas_mesh_reward"][n] = r(T2["Ours"][n]["contact"] - T2["(-) mesh contact reward"][n]["contact"], 1)
    v5["deltas_vlm_filter"][n] = r(T2["Ours"][n]["contact"] - T2["(-) VLM contact filtering"][n]["contact"], 1)
    v5["mpjpe_vlm_filter_delta_cm"][n] = r(T2["(-) VLM contact filtering"][n]["mpjpe"] - T2["Ours"][n]["mpjpe"], 1)
v5["sd_range"] = (min(x["sd"] for m in T2.values() for x in m.values()), max(x["sd"] for m in T2.values() for x in m.values()))
v5["z_vs_sd_link_vlm"] = r(v5["deltas_vlm_filter"]["link-arms"] / math.hypot(0.1, 0.4), 1)
v5["note"] = ("±는 '3 seeds'. C.5에 따르면 교사 체크포인트는 하나이고 평가 시드만 바뀌므로 학습 변동은 포함하지 않는다. "
              "MPJPE는 한 값만 있고 변동이 없다. 교사 표에는 3개 행동만 있고 handshake, pat-on-back은 없다.")
OUT["V5_teacher_table2"] = v5

# ---------------------------------------------------------------- V6 표 3 학생 절제
t3 = table_rows("Table 3: Student policy results", ["Ours", "(-) future generation", "generation → regression", "(-) pose prediction"], 8)
acts = ["hug", "high-five", "hand-shake", "link-arms", "pat-on-back"]
T3 = {k: dict(zip(acts, v[:5])) for k, v in t3.items()}
v6 = {"ours_minus_variant": {}, "wins_of_ours": {}}
for var in ["(-) future generation", "generation → regression", "(-) pose prediction"]:
    d = {a_: r(T3["Ours"][a_] - T3[var][a_], 1) for a_ in acts}
    v6["ours_minus_variant"][var] = d
    v6["wins_of_ours"][var] = sum(1 for x in d.values() if x > 0)
v6["text_claims"] = dict(
    pose_removal_drops_hug_hs_link_pat=[v6["ours_minus_variant"]["(-) pose prediction"][k] for k in ["hug", "hand-shake", "link-arms", "pat-on-back"]],
    future_gen_adds_hs_link_pat=[v6["ours_minus_variant"]["(-) future generation"][k] for k in ["hand-shake", "link-arms", "pat-on-back"]],
    gen_vs_reg_current_past=({a_: r(T3["(-) future generation"][a_] - T3["generation → regression"][a_], 1) for a_ in acts}),
)
v6["ours_best_actions"] = [a_ for a_ in acts if T3["Ours"][a_] >= max(T3[m][a_] for m in T3)]
v6["note"] = ("초록과 4.3절의 '5개 중 4개 향상'은 (-) pose prediction과 generation→regression 기준에서는 맞다. "
              "(-) future generation 기준에서는 3개뿐이고 hug는 -8.7%p다. 전체 모델이 최고인 행동은 3개(hand-shake, link-arms, pat-on-back)다. "
              "표 3은 변동(±) 없이 방 배치 5개 평균이며, 체크포인트는 변형당 하나다.")
OUT["V6_student_table3"] = v6

# ---------------------------------------------------------------- V7 표 6 (부록 C.4) 짝 t 검정
T6 = {
    "hug": dict(po=[7.4, 5.7, 7.7], other=[2.3, 4.2, 1.6], po_mean=6.9, other_mean=2.7, share=0.26, paper_t=3.1),
    "pat-on-back": dict(po=[12.4, 12.1, 6.2], other=[0.1, -1.6, 4.6], po_mean=10.2, other_mean=1.0, share=0.22, paper_t=2.4),
    "link-arms": dict(po=[35.9, 34.1, 31.4], other=[10.9, -6.1, 7.6], po_mean=33.8, other_mean=4.1, share=0.70, paper_t=5.6),
}
norm = TEXT.replace("−", "-")
v7 = {}
for k, d in T6.items():
    for arr in (d["po"], d["other"]):
        s = " / ".join(f"{x:+.1f}" for x in arr)
        assert s in "\n".join(LINES), s  # 원문 표 6에 있는 시드별 값인지 확인
    diff = np.array(d["po"]) - np.array(d["other"])
    t, p = stats.ttest_1samp(diff, 0.0)
    v7[k] = dict(means_po=r(np.mean(d["po"]), 2), means_other=r(np.mean(d["other"]), 2), paper_means=(d["po_mean"], d["other_mean"]),
                 seed_diffs=[r(x, 1) for x in diff], t_df2=r(t, 2), p_two_sided=r(p, 3), paper_t=d["paper_t"],
                 share_weighted_gain=r(d["share"] * np.mean(d["po"]) + (1 - d["share"]) * np.mean(d["other"]), 1))
v7["table3_gain_vs_regression"] = {k: r(T3["Ours"][k] - T3["generation → regression"][k], 1) for k in ["hug", "pat-on-back", "link-arms"]}
v7["note"] = ("t(2)는 평가 시드 3개이고 체크포인트는 변형당 하나다. 시드 간 변동은 평가 잡음이지 학습 변동이 아니다. "
              "구간 PO와 기타의 차이는 모든 시드에서 양수이나, p < 0.05는 link-arms 하나뿐이다. 임계값(0.8, 0.4)은 분포를 본 뒤 정했다(C.4).")
OUT["V7_table6_paired"] = v7

# ---------------------------------------------------------------- V8 표 7 교사 대 학생, 훈련 분할 대 보류 분할
t7 = table_rows("Table 7: Teacher vs. student contact recall", ["Hug", "High-five", "Hand-shake", "Link-arms", "Pat-on-back", "Mean"], 12)
T7 = {k: dict(T15=v[0], S15=v[1], rec15=v[2], T30=v[3], S30=v[4], rec30=v[5]) for k, v in t7.items()}
key_map = {"Hug": "hug", "High-five": "high-five", "Hand-shake": "hand-shake", "Link-arms": "link-arms", "Pat-on-back": "pat-on-back"}
v8 = {"computed_ratio15": {}, "computed_ratio30": {}}
r15, r30 = [], []
for k in key_map:
    x15 = T7[k]["S15"] / T7[k]["T15"] * 100
    x30 = T7[k]["S30"] / T7[k]["T30"] * 100
    r15.append(x15); r30.append(x30)
    v8["computed_ratio15"][k] = r(x15, 1); v8["computed_ratio30"][k] = r(x30, 1)
v8["mean_of_ratios"] = (r(np.mean(r15), 1), r(np.mean(r30), 1))
v8["ratio_of_means"] = (r(T7["Mean"]["S15"] / T7["Mean"]["T15"] * 100, 1), r(T7["Mean"]["S30"] / T7["Mean"]["T30"] * 100, 1))
v8["column_means_check"] = {c: (r(np.mean([T7[k][c] for k in key_map]), 2), T7["Mean"][c]) for c in ["T15", "S15", "T30", "S30"]}
v8["student_train_split_vs_heldout_30cm"] = {key_map[k]: dict(train_split_table7=T7[k]["S30"], heldout_table3=T3["Ours"][key_map[k]],
                                                              gap=r(T7[k]["S30"] - T3["Ours"][key_map[k]], 1)) for k in key_map}
heldout_ratio = {key_map[k]: r(T3["Ours"][key_map[k]] / T7[k]["T30"] * 100, 1) for k in key_map}
v8["heldout_student_over_teacher_train_30cm"] = dict(per_action=heldout_ratio, mean=r(np.mean(list(heldout_ratio.values())), 1))
v8["note"] = ("서론과 4.3절의 '교사의 75%'는 부록 C.5 표 7로, 학생의 훈련 분할에서 계산한 값이다. 보류 분할(표 3)을 같은 교사값으로 나누면 "
              "평균 약 54%이다(교사는 보류 모션을 학습했으므로 근사). 학생의 훈련 대 보류 격차는 high-five에서 49%p로 가장 크다.")
OUT["V8_teacher_student"] = v8

# ---------------------------------------------------------------- V9 표 5 (HMR2.0 대 E2E)
t5 = table_rows("Table 5: End-to-end vs. external human-pose estimation", ["Ext. human pose (HMR2.0 [48])", "Ours (E2E)"], 8)
cols5 = ["hug", "hand-shake", "pat-on-back", "link-arms", "high-five"]
ext = dict(zip(cols5, t5["Ext. human pose (HMR2.0 [48])"][1:6] if len(t5["Ext. human pose (HMR2.0 [48])"]) > 5 else t5["Ext. human pose (HMR2.0 [48])"]))
e2e = dict(zip(cols5, t5["Ours (E2E)"]))
v9 = {"ext_hmr2": ext, "e2e_earlier_setting": e2e,
      "table3_ours_current": {k: T3["Ours"][k] for k in cols5},
      "e2e_minus_current": {k: r(e2e[k] - T3["Ours"][k], 1) for k in cols5},
      "table7_student_30cm_train_split": {key_map[k]: T7[k]["S30"] for k in key_map},
      "duplicate_value_78.6": "표 5 hand-shake E2E 30 cm = 78.6, 표 7 hand-shake 학생 15 cm = 78.6 (서로 다른 설정·임계값)",
      "note": ("표 5의 E2E 열(이전 설정)은 현재 표 3과 pat-on-back 23.8 대 51.1, link-arms 30.1 대 67.8로 크게 다르다. "
               "HMR2.0 기준선은 이전 설정이라 현재 모델과 직접 비교는 아니다. 실로봇 결과도 이전 설정(3층 CNN)이다.")}
OUT["V9_table5"] = v9

# ---------------------------------------------------------------- V10 표 4 실로봇
real = {"high-five": 5, "hug": 8, "hand-shake": 5, "link-arms": 5, "pat-on-back": 8}
v10 = {"sum": sum(real.values()), "reported_total": 31, "wilson95": {}}
for k, x in real.items():
    lo, hi = wilson(x, 10)
    v10["wilson95"][k] = (x, 10, r(x / 10, 2), r(lo, 3), r(hi, 3))
lo, hi = wilson(31, 50)
v10["wilson95"]["total"] = (31, 50, r(31 / 50, 3), r(lo, 3), r(hi, 3))
tbl = [[16, 4], [15, 15]]
v10["fisher_hug_pat_vs_others"] = dict(table=tbl, p=r(stats.fisher_exact(tbl)[1], 3))
v10["fisher_8of10_vs_5of10"] = r(stats.fisher_exact([[8, 2], [5, 5]])[1], 3)
sim = [T3["Ours"][k] for k in real]
rho, p = stats.spearmanr(sim, list(real.values()))
v10["spearman_sim_heldout_vs_real"] = dict(rho=r(rho, 2), p=r(p, 3), n=5)
v10["tests_vs_50pct"] = {k: r(stats.binomtest(x, 10, 0.5).pvalue, 3) for k, x in real.items()}
v10["note"] = ("성공 기준이 행동마다 다르다(handshake, high-five는 손이 약 15 cm 안, 나머지는 접근과 완료). 이전 설정의 학생이다. "
              "시험당 10회라 구간이 매우 넓고(5/10는 [24, 76]%), 8/10 대 5/10 두 칸만 비교하면 Fisher p = 0.35이고, hug+pat 16/20 대 나머지 15/30으로 묶으면 p = 0.04이나 묶음이 사후 선택이다.")
OUT["V10_real_robot_table4"] = v10

# ---------------------------------------------------------------- V11 표 8 (라벨 품질)
t8 = {}
rows8 = [("Hug", [0.060, 0.241, 0.127, 0.081, 0.201, 0.181]), ("Hand-shake", [0.275, 0.785, 0.495, 0.442, 0.661, 0.674]),
         ("Pat-on-back", [0.107, 0.824, 0.212, 0.227, 0.760, 0.387]), ("High-five", [0.260, 0.866, 0.417, 0.336, 0.660, 0.626]),
         ("Link-arms", [0.257, 0.600, 0.374, 0.339, 0.549, 0.453]), ("Total", [0.192, 0.663, 0.330, 0.285, 0.567, 0.468])]
cap_i = next(i for i, l in enumerate(LINES) if "Table 8: VLM contact-label quality" in l)
blk = "\n".join(LINES[cap_i:cap_i + 25])
for name, vals in rows8:
    for x in vals:
        assert f"{x:.3f}" in blk, (name, x)  # 원문 텍스트에 있는 값인지 확인
v11 = {"rows": {}, "bound_checks": {}}


def bound(P, R):
    return min(2 * P / (1 + P), 2 * R / (1 + R))


for name, v in rows8:
    for tag, (P, R, F) in {"distance": v[0:3], "vlm": v[3:6]}.items():
        hm = 2 * P * R / (P + R)
        v11["bound_checks"][f"{name}_{tag}"] = dict(P=P, R=R, F1_reported=F, harmonic_of_means=r(hm, 3),
                                                   max_possible_mean_F1=r(bound(P, R), 3), infeasible=F > bound(P, R) + 1e-9)
v11["n_infeasible"] = sum(1 for x in v11["bound_checks"].values() if x["infeasible"])
v11["relative_gains"] = dict(precision=r(0.285 / 0.192 - 1, 3), f1=r(0.468 / 0.330 - 1, 3), recall_drop=r(0.663 - 0.567, 3))
v11["per_action_precision_gain_x"] = {n: r(v[3] / v[0], 2) for n, v in rows8[:5]}
v11["per_action_recall_drop"] = {n: r(v[1] - v[4], 3) for n, v in rows8[:5]}
v11["total_vs_mean_of_actions"] = {c: (r(np.mean([v[i] for _, v in rows8[:5]]), 3), rows8[5][1][i])
                                   for c, i in dict(P_d=0, R_d=1, F_d=2, P_v=3, R_v=4, F_v=5).items()}
v11["note"] = ("클립별 F1 = 2PR/(P+R)이고 P, R이 [0,1]이면 평균 F1은 2P/(1+P)(P는 평균 정밀도)(옌센 부등식)를 넘을 수 없다. "
               "표 8의 12개 행 중 일부가 이 상한을 넘는다(예: 총 VLM 0.468 > 0.444, high-five VLM 0.626 > 0.503). "
               "정밀도를 예측이 있는 클립만 평균했거나 F1을 다른 방식(사건 수준)으로 계산했다면 가능하지만 본문은 설명하지 않는다.")
OUT["V11_table8_label_quality"] = v11

# ---------------------------------------------------------------- V12 데이터 산술
OUT["V12_dataset_arithmetic"] = dict(
    master_pool=dict(per_action=[282, 278, 275, 252, 268], sum=282 + 278 + 275 + 252 + 268, reported=1355),
    teacher_pool=dict(per_action=[244, 233, 196, 205, 222], sum=244 + 233 + 196 + 205 + 222, reported=1100),
    dropped=dict(n=1355 - 1100, reported=255, frac=r((1355 - 1100) / 1355, 4), reported_frac=0.188),
    per_action_dropped={a_: m - t for a_, m, t in zip(acts, [282, 278, 275, 252, 268], [244, 233, 196, 205, 222])},
    gt_clips=dict(n=80, per_action=16, frac_of_teacher_pool=r(80 / 1100, 3)),
    user_study=dict(participants=40, analysed=28, frac=r(28 / 40, 2)),
    student_pool=dict(note="학생 풀의 클립 수(두 평가자 3점, FOV 필터)는 논문에 숫자로 없다. 9:1 분할만 기재"),
)

# ---------------------------------------------------------------- V13 보상 함수 형태 (식 5–7)
def phi(F, tau=1.0):
    return min(math.exp(F - tau), 1.0)


def dist_rew(d, sig=50.0):
    return math.exp(-sig * d * d)


def force_rew(F, label=1, Fmin=5.0, Fmax=50.0, cu=0.5, co=0.05, cw=0.005):
    if label == 1:
        if F < Fmin:
            return math.exp(-cu * (Fmin - F))
        if F > Fmax:
            return math.exp(-co * (F - Fmax))
        return 1.0
    return -cw * F if F > Fmin else 0.0


OUT["V13_reward_shapes"] = dict(
    gate_phi={str(F): r(phi(F), 3) for F in [0, 0.5, 1, 2, 5]},
    dist_reward_by_cm={str(c): r(dist_rew(c / 100), 4) for c in [0, 2, 5, 10, 15, 20, 30]},
    force_reward_label1={str(F): r(force_rew(F), 3) for F in [0, 1, 2.5, 5, 20, 50, 80, 120]},
    force_penalty_label0={str(F): r(force_rew(F, 0), 3) for F in [0, 5, 20, 50, 100]},
    aggregate_weights=dict(dir=3, dist=5, force=0.5),
    note=("σ_d = 50 m⁻²이면 15 cm에서 거리 보상이 0.32, 30 cm에서 0.011이다. 학생을 30 cm로 평가하는 '접촉 재현율'은 보상이 거의 0인 거리까지 허용한다. "
          "힘 게이트 φ는 F ≥ 1 N이면 1이고 τ = 1 N은 단위가 맞는 힘 임계값이다(식 5의 exp(F-τ)는 단위 없는 형태). "
          "힘 보상의 하한 5 N, 상한 50 N은 시뮬 접촉력이며 실로봇 측정 힘과의 대응은 보고되지 않았다."),
)

# ---------------------------------------------------------------- V14 손실 가중, 해상도, 게이트 산술
OUT["V14_training_arithmetic"] = dict(
    depth_pixels=192 * 108, five_frames_dim=5, short_proprio_dim=5 * 108, long_proprio_steps=100, long_seconds=100 / 50,
    action_dim=dict(homie_cmds=4, residual_arm=14, total=18),
    gate_pixels_15pct=r(0.15 * 192 * 108, 0),
    lambda_note=("C.3은 'combined pose loss enters the total loss with weight 0.5, i.e. λgen = λpose = 0.5'라고 쓴다. "
                 "DDIM 예측과 단일 단계 추정이 '같은 비율'로 합쳐진다면 각 0.25인지 0.5인지 문장만으로 정할 수 없다. 합이면 1.0이다."),
    gate_note=("부분 관측 구간이 link-arms에서 점유율 0.4 미만(파트너가 시야를 벗어남)인데, 학습은 파트너 픽셀 15% 이하인 프레임의 기울기를 끈다. "
               "점유율은 로봇 몸 픽셀을 포함하므로 두 기준은 같은 양이 아니다. 두 기준의 겹침은 보고되지 않았다."),
    kl_ramp="λKL 1e-4 -> 1e-2, epoch 500–2000",
    teacher_ppo=dict(envs=8192, steps_per_iter=24, batch=8192 * 24, lr=1e-5),
    student_envs=128,
)

# ---------------------------------------------------------------- V15 부록 C.4와 그림 3
OUT["V15_fig3_vs_table6"] = {k: dict(fig3_po=T6[k]["po_mean"], fig3_other=T6[k]["other_mean"]) for k in T6}

# ---------------------------------------------------------------- V16 표 9 임계값 민감도
ths = [3, 5, 10, 15, 20, 25, 30]
v16 = {"table9": {m: dict(zip(ths, v)) for m, v in t9.items()}}
v16["ours_minus_noVLM_by_eps"] = {e: r(t9["Ours"][i] - t9["Ours w/o VLM filtering"][i], 1) for i, e in enumerate(ths)}
v16["ours_minus_omni_by_eps"] = {e: r(t9["Ours"][i] - t9["OmniRetarget"][i], 1) for i, e in enumerate(ths)}
v16["note"] = ("VLM 필터의 재현율 이득은 ε = 10 cm에서 +1.4, 15 cm +0.7, 20 cm +4.8, 25 cm +3.0이고 30 cm에서는 -1.3이다. "
               "VLM 필터의 주 효과는 재현율이 아니라 정밀도(+2.6%p)와 사용자 선호이다. OmniRetarget은 가능한 58개 클립만 채점되어 큰 ε에서 유리하다(E.4).")
OUT["V16_threshold_sensitivity"] = v16

# 입력 읽기 검증용 원자료
OUT["parsed_tables"] = dict(table1=T1, table2=T2, table3=T3, table7=T7, table5=dict(ext=ext, e2e=e2e))

(ROOT / "analysis" / "recompute_results.json").write_text(json.dumps(OUT, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote analysis/recompute_results.json")
for k in ["V1_table_consistency", "V4_table1_tests", "V6_student_table3", "V7_table6_paired", "V8_teacher_student", "V10_real_robot_table4", "V11_table8_label_quality"]:
    print("==", k)
    print(json.dumps(OUT[k], ensure_ascii=False)[:2500])
