"""RFW-S (arXiv:2611.23056 v2) 표·본문 수치 재계산과 일관성 검사 (numpy, scipy).
실행: python3 -I analysis/recompute.py
입력: repo/paper/analysis_data/*.csv (공식 저장소의 파생 CSV, 해시는 source_registry.json과 대조)
결과: analysis/recompute_results.json

원칙: 시뮬레이션을 다시 돌리지 않는다. 공개된 파생 CSV에서 논문 표·본문 숫자를 다시 만든다.
논문 쪽 값은 paper/paper.txt(pdftotext -layout)에서 옮겼고, 옮긴 자리는 주석에 표 번호를 적었다.
"""
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "repo" / "paper" / "analysis_data"
OUT = HERE / "recompute_results.json"
res = {}
checks = []  # (id, description, paper value, recomputed value, pass?)


def rd(name):
    with open(DATA / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check(cid, desc, paper, recomputed, tol=0.0):
    ok = abs(float(paper) - float(recomputed)) <= tol
    checks.append({"id": cid, "desc": desc, "paper": paper, "recomputed": recomputed, "pass": bool(ok)})
    return ok


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def clopper(k, n, alpha=0.05):
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lo), float(hi))


# ------------------------------------------------------------ 0. 파일 해시 대조
reg = json.loads((DATA / "source_registry.json").read_text(encoding="utf-8"))
hash_ok = {}
for fn, h in reg["files"].items():
    digest = hashlib.sha256((DATA / fn).read_bytes()).hexdigest()
    hash_ok[fn] = digest == h
res["hash_check"] = {"files": len(hash_ok), "matching": sum(hash_ok.values()),
                     "mismatch": [k for k, v in hash_ok.items() if not v]}

# ------------------------------------------------------------ 1. 표 2, 그림 4, 5.1 (전체 풀 센서스)
vf = rd("verification_by_family.csv")
tot = Counter()
fam = {}
for r in vf:
    for k in ["category_n", "pool_n", "physics_pass_n", "runtime_pass_n", "intersection_n"]:
        tot[k] += int(r[k])
    fam[r["family"]] = {k: int(r[k]) for k in ["category_n", "pool_n", "physics_pass_n", "runtime_pass_n", "intersection_n"]}
res["census_total"] = dict(tot)
check("T2a", "합계 categories", 485, tot["category_n"])
check("T2b", "합계 pool", 100000, tot["pool_n"])
check("T2c", "합계 physical", 83618, tot["physics_pass_n"])
check("T2d", "합계 RC", 99586, tot["runtime_pass_n"])
check("T2e", "합계 both", 83365, tot["intersection_n"])
N = tot["pool_n"]
both = tot["intersection_n"]
phys_only = tot["physics_pass_n"] - both
rc_only = tot["runtime_pass_n"] - both
neither = N - both - phys_only - rc_only
res["fig4_overlap"] = {"both": both, "phys_only": phys_only, "rc_only": rc_only, "neither": neither}
check("F4a", "그림4 physical만 통과 253", 253, phys_only)
check("F4b", "그림4 RC만 통과 16221", 16221, rc_only)
check("F4c", "그림4 둘 다 실패 161", 161, neither)
res["retention_pct"] = {
    "physical": round(100 * tot["physics_pass_n"] / N, 2),
    "rc": round(100 * tot["runtime_pass_n"] / N, 2),
    "both": round(100 * both / N, 2),
}
check("S51a", "물리 검증 통과율 83.62%", 83.62, res["retention_pct"]["physical"], 0.005)
check("S51b", "RC 통과율 99.59%", 99.59, res["retention_pct"]["rc"], 0.005)
check("S51c", "교집합 83.37%", 83.37, res["retention_pct"]["both"], 0.005)
fam_ret = {f: round(100 * v["intersection_n"] / v["pool_n"], 2) for f, v in fam.items()}
res["family_both_pct"] = fam_ret
check("S51d", "soft solids 교집합 비율 60.92%", 60.92, fam_ret["solid"], 0.005)
check("S51e", "cloth sheets 교집합 비율 98.46%", 98.46, fam_ret["cloth_sheet"], 0.005)
# 가족별 실패 = 모형 한계 + 수치 실패 + 판단 보류 (표 10과 대조)
ps = rd("production_states_by_family.csv")
states = defaultdict(lambda: defaultdict(int))
for r in ps:
    states[r["agent_status"]][r["family"]] += int(r["n"])
st_tot = {k: sum(v.values()) for k, v in states.items()}
res["production_states_total"] = st_tot
check("T9a", "first-pass accept 83,292", 83292, st_tot["accepted_first_pass"])
check("T9b", "corrected accept 326", 326, st_tot["corrected_then_accepted"])
check("T9c", "model-limit reject 16,317", 16317, st_tot["rejected_model_limit"])
check("T9d", "numerical reject 2", 2, st_tot["rejected_numerical_failure"])
check("T9e", "human escalation 63", 63, st_tot["uncertain_human_escalation"])
check("T9f", "row 합 100,000", 100000, sum(st_tot.values()))
soft_lim = states["rejected_model_limit"]["solid"]
res["soft_share_of_model_limit_pct"] = round(100 * soft_lim / st_tot["rejected_model_limit"], 2)
check("S51f", "soft solids 모형 한계 비율 82.69%", 82.69, res["soft_share_of_model_limit_pct"], 0.005)
# 물리 검증 실패 수 = 모형 한계 + 수치 + 보류 (가족별 교차 검사)
fam_key = {"solid": "solid", "cloth_sheet": "cloth_sheet", "shell": "shell", "garment": "garment", "bag": "bag"}
x_bad = []
for f, v in fam.items():
    fail_phys = v["pool_n"] - v["physics_pass_n"]
    s = sum(states[k][f] for k in ["rejected_model_limit", "rejected_numerical_failure", "uncertain_human_escalation"])
    if fail_phys != s:
        x_bad.append((f, fail_phys, s))
res["phys_fail_vs_states_mismatch"] = x_bad  # 빈 리스트면 일치
check("X1", "가족별 물리 실패 = 모형한계+수치+보류 (불일치 가족 수)", 0, len(x_bad))

# 범주 수준 (coverage_by_type): 낮은 유지 범주, 생성 할당 206/207
cov = rd("coverage_by_type.csv")
pool_counts = Counter(int(r["pool_n"]) for r in cov)
res["pool_per_category_counts"] = dict(pool_counts)
check("S31a", "206인 범주 수 395", 395, pool_counts[206])
check("S31b", "207인 범주 수 90", 90, pool_counts[207])
low = sorted(cov, key=lambda r: int(r["intersection_n"]))[:5]
res["lowest_intersection_categories"] = [(r["object_type"], int(r["intersection_n"]), int(r["pool_n"])) for r in low]
gummy = [r for r in cov if r["object_type"].lower().startswith("gummy")]
res["gummy_rows"] = [(r["object_type"], r["intersection_n"], r["pool_n"]) for r in gummy]
mx = max(int(r["intersection_n"]) for r in cov)
mn = min(int(r["intersection_n"]) for r in cov)
res["intersection_range"] = [mn, mx]

# ------------------------------------------------------------ 2. 표 6, 표 18, 5.3 (교차 분석, 독립 검증)
ct = rd("calibration_independent_transitions.csv")
for r in ct:
    n = int(r["n"])
    cells = {k: int(r[k]) for k in ["cal_pass_held_pass", "cal_pass_held_fail", "cal_fail_held_pass", "cal_fail_held_fail", "missing"]}
    assert sum(cells.values()) == n, r
    res[f"crosstab_{r['experiment']}"] = cells
check("T6a", "30 targets 둘 다 통과 682", 682, int(ct[0]["cal_pass_held_pass"]))
check("T6b", "30 targets 교정만 통과 49", 49, int(ct[0]["cal_pass_held_fail"]))
check("T6c", "30 targets 독립만 통과 3", 3, int(ct[0]["cal_fail_held_pass"]))
check("T6d", "300 bags 교정만 통과 325", 325, int(ct[1]["cal_pass_held_fail"]))
rate_bag = 325 / 8400
lo, hi = wilson(325, 8400)
res["bag_cal_only_rate_pct"] = {"point": round(100 * rate_bag, 3), "wilson95": [round(100 * lo, 3), round(100 * hi, 3)]}
check("T6e", "325/8400 = 3.87%", 3.87, 100 * rate_bag, 0.005)

# 독립 단위는 자산(300 bags)이다. 8,400개 실행은 같은 자산의 28개 셀(7 방법 x 4 예산)이다.
flt = rd("filter_long.csv")
per_bag = defaultdict(lambda: [0, 0])  # bag -> [cal_only_count, runs]
for r in flt:
    b = r["sample_id"]
    per_bag[b][1] += 1
    if r["status_success"] == "False":
        per_bag[b][0] += 1
bags_with_reject = sum(1 for v in per_bag.values() if v[0] > 0)
res["filter_bags"] = {"n_bags": len(per_bag), "bags_with_any_reject": bags_with_reject,
                      "runs_per_bag": sorted(Counter(v[1] for v in per_bag.values()).items())}
check("T6f", "325 rejections 가 26개 bag에 걸침", 26, bags_with_reject)
bag_lo, bag_hi = wilson(bags_with_reject, len(per_bag))
res["bag_level_reject_rate_pct"] = {"point": round(100 * bags_with_reject / len(per_bag), 2),
                                    "wilson95": [round(100 * bag_lo, 2), round(100 * bag_hi, 2)]}
# 방법별 거부 수 이분산 검정 (독립 검증 통과 수, 7개 방법)
by_m = defaultdict(lambda: [0, 0])
for r in flt:
    by_m[r["method"]][1] += 1
    by_m[r["method"]][0] += int(r["status_success"] == "True")
mt = np.array([[v[0], v[1] - v[0]] for v in by_m.values()])
chi2, p_chi, dof, _ = stats.chi2_contingency(mt)
res["filter_method_heterogeneity"] = {"chi2": round(float(chi2), 3), "dof": int(dof), "p": round(float(p_chi), 4),
                                      "pass_by_method": {k: v[0] for k, v in by_m.items()}}

rl = rd("recovery_long.csv")
by_rm = defaultdict(lambda: [0, 0])
for r in rl:
    by_rm[r["method"]][1] += 1
    by_rm[r["method"]][0] += int(r["status_success"] == "True")
mt2 = np.array([[v[0], v[1] - v[0]] for v in by_rm.values()])
chi2b, p_b, dof_b, _ = stats.chi2_contingency(mt2)
res["recovery_method_heterogeneity"] = {"chi2": round(float(chi2b), 3), "dof": int(dof_b), "p": round(float(p_b), 4),
                                        "pass_by_method": {k: v[0] for k, v in by_rm.items()}}

# 임계값 일관성: 교정 통과는 cal ≤ 0.05, 독립 통과는 heldout ≤ 0.08 (표 6, 식 4)
def fnum(x):
    return None if x in ("", None) else float(x)

cal_viol = sum(1 for r in flt if fnum(r["calibration_error"]) is not None and fnum(r["calibration_error"]) > 0.05 and r["status_success"] != "")
ind_viol = sum(1 for r in flt if r["status_success"] == "True" and fnum(r["heldout_error"]) is not None and fnum(r["heldout_error"]) > 0.08)
ind_fail_ok = sum(1 for r in flt if r["status_success"] == "False" and fnum(r["heldout_error"]) is not None and fnum(r["heldout_error"]) <= 0.08)
res["threshold_consistency_filter"] = {"cal_error_above_0.05_rows": cal_viol,
                                       "passed_indep_but_error_above_0.08": ind_viol,
                                       "failed_indep_but_error_le_0.08": ind_fail_ok}
check("K1", "독립 통과인데 오차 > 0.08 인 행 (0이어야 함)", 0, ind_viol)

# ------------------------------------------------------------ 3. 표 15–19 (재현 가능한 평균과 쌍별 차이)
rbm = rd("recovery_by_method_budget.csv")
fbm = rd("filter_by_method_budget.csv")
def pooled(rows, method):
    num = sum(float(r["heldout_error_mean"]) * int(r["heldout_error_n"]) for r in rows if r["method"] == method)
    den = sum(int(r["heldout_error_n"]) for r in rows if r["method"] == method)
    return num / den, den
pooled_tab = {}
for m in sorted(set(r["method"] for r in rbm)):
    pr, nr = pooled(rbm, m)
    pf, nf = pooled(fbm, m)
    pooled_tab[m] = {"recovery_err": round(pr, 4), "recovery_valid_n": nr, "expanded_err": round(pf, 4), "expanded_valid_n": nf}
res["pooled_errors_table15"] = pooled_tab
check("T15a", "Fixed profile recovery error 0.0521", 0.0521, pooled_tab["fixed_profile"]["recovery_err"], 0.00006)
check("T15b", "Fixed profile recovery valid n 115", 115, pooled_tab["fixed_profile"]["recovery_valid_n"])
check("T15c", "Full agent expanded error 0.0117 (표 15, 반올림)", 0.0117, pooled_tab["full_numeric_agent"]["expanded_err"], 0.00006)

# 표 19: 쌍별 차이 (full numeric agent - fixed profile), 자산 단위 부트스트랩, seed 20260926, 5000회
def paired_boot(long_rows, budget, seed=20260926, B=5000):
    fixed = {}
    full = {}
    for r in long_rows:
        if r["budget_allowed"] != str(budget):
            continue
        if r["heldout_error"] == "":
            continue
        key = r["sample_id"] if "sample_id" in r else r["target_handle"]
        if r["method"] == "fixed_profile" or r["method"] == "fixed":
            fixed[key] = float(r["heldout_error"])
        if r["method"] in ("full_numeric_agent", "numeric_agent", "agent_numeric"):
            full[key] = float(r["heldout_error"])
    keys = sorted(set(fixed) & set(full))
    d = np.array([full[k] - fixed[k] for k in keys])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    means = d[idx].mean(axis=1)
    return {"pairs": len(d), "diff": round(float(d.mean()), 4),
            "ci95": [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]}

methods_present = sorted(set(r["method"] for r in rl))
res["methods_in_recovery_long"] = methods_present
for b in [1, 2, 4, 8]:
    pb = paired_boot(rl, b)
    res[f"table19_recovery_b{b}_recomputed"] = pb
for b in [1, 2, 4, 8]:
    res[f"table19_filter_b{b}_recomputed"] = paired_boot(flt, b)
check("T19a", "Filter 예산 8 쌍별 차이 -0.0036 (재계산 평균)", -0.0036,
      res["table19_filter_b8_recomputed"]["diff"], 0.00006)

# 쌍 수 (표 19의 Pairs 열): 두 방법 모두 값이 있는 자산 수
def pairs_count(long_rows, budget):
    fixed, full = set(), set()
    for r in long_rows:
        if r["budget_allowed"] != str(budget) or r["heldout_error"] == "":
            continue
        k = r["sample_id"] if "sample_id" in r else r["target_handle"]
        if r["method"] == "fixed_profile":
            fixed.add(k)
        if r["method"] == "full_numeric_agent":
            full.add(k)
    return len(fixed & full)
res["pairs_recovery_by_budget"] = {b: pairs_count(rl, b) for b in [1, 2, 4, 8]}
check("T19b", "Recovery 예산 1 쌍 수 27", 27, res["pairs_recovery_by_budget"][1])
check("T19c", "Recovery 예산 8 쌍 수 28", 28, res["pairs_recovery_by_budget"][8])

# ------------------------------------------------------------ 4. 표 7, 표 20, 5.4 (그리퍼 적용)
app = rd("application_long.csv")
meth_name = {"fixed": "Fixed command", "reactive_slip": "Slip feedback", "deligrasp_prior": "Force prior (local)",
             "privileged_oracle": "Privileged reference"}
res["application_total"] = {"rows": len(app), "task_success": sum(r["task_success"] == "True" for r in app),
                            "intact": sum(r["intact_success"] == "True" for r in app),
                            "damage_flag": sum(r["damage_failure"] == "True" for r in app)}
check("Q1", "작업 성공 합 3,079", 3079, res["application_total"]["task_success"])
check("Q2", "무손상 성공 합 0", 0, res["application_total"]["intact"])
check("Q3", "손상 플래그 합 4,000", 4000, res["application_total"]["damage_flag"])
dps = [float(r["damage_proxy"]) for r in app]
res["damage_proxy_min_max"] = [round(min(dps), 4), round(max(dps), 4)]
check("Q4", "최소 손상 점수 0.2653 (≥0.25 여야 전부 플래그)", 0.2653, min(dps), 0.00006)

held = {m: {} for m in meth_name}
cal = {m: [0, 0] for m in meth_name}
for r in app:
    m = r["method"]
    if r["split"] == "hidden_test":
        held[m][r["benchmark_instance_id"]] = r["task_success"] == "True"
    else:
        cal[m][1] += 1
        cal[m][0] += int(r["task_success"] == "True")
table7 = {}
for m in meth_name:
    k = sum(held[m].values())
    n = len(held[m])
    lo, hi = wilson(k, n)
    table7[meth_name[m]] = {"held_out": f"{k}/{n}", "held_out_pct": round(100 * k / n, 2),
                            "wilson95_pct": [round(100 * lo, 1), round(100 * hi, 1)],
                            "cal": f"{cal[m][0]}/{cal[m][1]}"}
res["table7_recomputed"] = table7
check("T7a", "Fixed command held-out 72.1%", 72.1, table7["Fixed command"]["held_out_pct"], 0.05)
check("T7b", "Force prior held-out 58.8%", 58.8, table7["Force prior (local)"]["held_out_pct"], 0.05)

# 같은 340개 held-out 인스턴스 쌍별 비교 (McNemar 정확 검정)
def mcnemar(a, b):
    keys = sorted(set(held[a]) & set(held[b]))
    n01 = sum(1 for k in keys if held[a][k] and not held[b][k])
    n10 = sum(1 for k in keys if (not held[a][k]) and held[b][k])
    n = n01 + n10
    p = 1.0 if n == 0 else min(1.0, 2 * stats.binom.cdf(min(n01, n10), n, 0.5))
    return {"a_only": n01, "b_only": n10, "p_exact": round(float(p), 4), "n": len(keys)}
res["mcnemar_heldout"] = {
    "fixed_vs_slip": mcnemar("fixed", "reactive_slip"),
    "fixed_vs_force": mcnemar("fixed", "deligrasp_prior"),
    "fixed_vs_reference": mcnemar("fixed", "privileged_oracle"),
    "slip_vs_force": mcnemar("reactive_slip", "deligrasp_prior"),
}
# 전체 4,000 기록을 같은 인스턴스로 쪼갠 뒤 방법 간 차이 (교정 전 풀링 수치의 함정 확인)
pooled_counts = {meth_name[m]: sum(r["task_success"] == "True" for r in app if r["method"] == m) for m in meth_name}
res["pooled_task_success_counts"] = pooled_counts
check("T7c", "풀링 성공 수 고정 명령 840", 840, pooled_counts["Fixed command"])
check("T7d", "풀링 성공 수 슬립 피드백 879", 879, pooled_counts["Slip feedback"])
# 조건(fragility)별 평균 손상 점수 (본문 미보고, 참고용)
cond = rd("application_by_condition.csv")
res["damage_mean_by_condition"] = {r["condition"]: round(float(r["damage_proxy_mean"]), 4) for r in cond}

# ------------------------------------------------------------ 5. 표 14, 그림 12d (전문가 평가)
hrc = rd("human_rating_counts.csv")
dims = defaultdict(dict)
for r in hrc:
    dims[r["dimension"]].setdefault("n", int(r["n"]))
    dims[r["dimension"]][int(r["score"])] = int(r["count"])
t14 = {}
for d, v in dims.items():
    n = v["n"]
    ge4 = sum(c for s, c in v.items() if s in (4, 5))
    s5 = v.get(5, 0)
    lo, hi = wilson(ge4, n)
    t14[d] = {"rated": n, "ge4": ge4, "ge4_pct": round(100 * ge4 / n, 1), "wilson95_pct": [round(100 * lo, 1), round(100 * hi, 1)],
              "score5_count_fig12d": s5}
res["table14_recomputed"] = t14
check("T14a", "Settling 148/150", 148, t14["q1_settle"]["ge4"])
check("T14b", "Deformation 146/150", 146, t14["q2_deformation"]["ge4"])
check("T14c", "Recovery 72/75", 72, t14["q3_recovery"]["ge4"])
check("T14d", "Contact 143/150", 143, t14["q4_contact"]["ge4"])
check("T14e", "Overall 141/150", 141, t14["q5_overall"]["ge4"])
check("F12d", "그림 12d 막대 값 = 점수 5 개수 (Settle 142)", 142, t14["q1_settle"]["score5_count_fig12d"])

# ------------------------------------------------------------ 6. 표 5, 표 13 (정적 구조 비교, 4개 객체)
sc = rd("structure_cases.csv")
res["structure_status_counts"] = dict(Counter(r["status"] for r in sc))
res["structure_variant_status"] = {f"{k[0]}|{k[1]}": v for k, v in Counter((r["variant"], r["status"]) for r in sc).items()}
rows = defaultdict(dict)
for r in sc:
    if r["status"] != "simulated_for_review":  # 저장소에만 있는 미실행 변형(surface_fix_flexible_garment)은 제외
        continue
    rows[(r["sample_id"], r["probe"])][r["variant"]] = (float(r["height_ratio"]), float(r["final_max_speed"]))
variants = ["shape_preserving", "softer_material", "flexible_cloth"]
probe_tab = {}
for probe in ["drop", "compression"]:
    for v in variants:
        hs = [rows[(s, p)][v][0] for (s, p) in rows if p == probe and v in rows[(s, p)]]
        ss = [rows[(s, p)][v][1] for (s, p) in rows if p == probe and v in rows[(s, p)]]
        probe_tab[f"{probe}:{v}"] = {"n": len(hs), "height_mean": round(float(np.mean(hs)), 4),
                                     "speed_mean": round(float(np.mean(ss)), 5)}
res["table5_recomputed"] = probe_tab
check("T5a", "Drop shape-preserving 높이 0.499", 0.499, probe_tab["drop:shape_preserving"]["height_mean"], 0.0006)
check("T5b", "Drop flexible 높이 0.312", 0.312, probe_tab["drop:flexible_cloth"]["height_mean"], 0.0006)
check("T5c", "Compression flexible 높이 0.303", 0.303, probe_tab["compression:flexible_cloth"]["height_mean"], 0.0006)
check("T5d", "Drop flexible 속도 0.14820", 0.14820, probe_tab["drop:flexible_cloth"]["speed_mean"], 0.00006)
# 방향 일관성: 4개 객체에서 flexible이 shape-preserving보다 높이 비율이 낮은 객체 수
objs = sorted({s for (s, p) in rows})
signs = {}
for probe in ["drop", "compression"]:
    signs[probe] = sum(1 for (s, p) in rows if p == probe and rows[(s, p)]["flexible_cloth"][0] < rows[(s, p)]["shape_preserving"][0])
res["flexible_lower_height_objects"] = signs
# 부호 검정 (4/4, 양측) : 2 * (1/2)^4
res["sign_test_p_4_of_4_two_sided"] = round(2 * 0.5 ** 4, 4)
# 속도 평균의 반올림 범위: 표 13은 소수 셋째 자리이므로 평균은 ±0.0005 범위 안에서만 맞는다
rounding_note = {}
for probe in ["drop", "compression"]:
    for v in variants:
        ss = [rows[(s, p)][v][1] for (s, p) in rows if p == probe and v in rows[(s, p)]]
        rounding_note[f"{probe}:{v}"] = {"speed_mean_from_rounded_values": round(float(np.mean(ss)), 5)}
res["speed_mean_note"] = rounding_note

# ------------------------------------------------------------ 7. 표 11 (라우팅), 그림 12c
rt = rd("routing_execution.csv")
res["routing_executed_solver"] = dict(Counter(r["executed_solver"] for r in rt))
res["routing_physical_status"] = dict(Counter(r["physical_status"] for r in rt))
check("T11a", "PBD cloth 193", 193, res["routing_executed_solver"].get("pbd_cloth", 0))
check("T11b", "Elastoplastic MPM 100", 100, res["routing_executed_solver"].get("mpm_elastoplastic", 0))

# ------------------------------------------------------------ 8. 감사 (2,000 bag 안정성)
audit = rd("audit_long.csv")
res["audit_status"] = dict(Counter(r["status_success"] for r in audit))
check("Q5", "안정성 감사 통과 1,998", 1998, res["audit_status"].get("True", 0))
p_audit = wilson(1998, 2000)
res["audit_pass_wilson95_pct"] = [round(100 * p_audit[0], 2), round(100 * p_audit[1], 2)]

# ------------------------------------------------------------ 9. 정리
res["checks"] = checks
res["checks_pass"] = sum(c["pass"] for c in checks)
res["checks_total"] = len(checks)
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"checks {res['checks_pass']}/{res['checks_total']} pass")
for c in checks:
    if not c["pass"]:
        print("FAIL", c)
print("written", OUT)
