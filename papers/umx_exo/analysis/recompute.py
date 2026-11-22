"""논문 표·본문 수치 재계산과 일관성 검사 (numpy, scipy만, 수 초).

대상: UME (arXiv:2606.14218 v1). 표 1(과제별 성공률, 방법당 20회), 표 2(냉장고 단계별), 그림 5(분당 시연 수),
표 3(ACT 학습 설정), 표 4·5(부품 비용 합계).

검사 번호는 V1.. (재계산), K1.. (일관성)로 붙이고 결과는 recompute_results.json 에 저장한다.
성공 횟수는 본문이 직접 적은 것(UME 18/20, 17/20, 19/20, 19/20; 표 2의 20 x 비율)과 표 1의 비율 x 20 이다.
시행은 방법마다 독립이라고 가정한다(같은 초기 조건을 공유하는지 논문에 없다). 따라서 Fisher 정확 검정을 쓴다.

실행: python3 analysis/recompute.py   → analysis/recompute_results.json
"""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
N = 20
TASKS = ["box_pushing", "box_flipping", "gpu_picking", "fridge_drink"]
# 표 1 비율 (논문 그대로)
T1 = {
    "No-torque": [0.50, 0.0, 0.75, 0.90],
    "UMI": [0.40, 0.0, 0.0, 0.0],
    "UME": [0.90, 0.85, 0.95, 0.95],
}
# 본문이 적은 UME 성공 횟수 (§3.1–3.4)
TEXT_UME = [18, 17, 19, 19]
# 표 2 (냉장고 단계별): Turn Around, Open fridge, Take out drink, Close fridge, Transport drink, Place drink
T2 = {
    "No-torque": [1.00, 0.95, 0.90, 0.90, 0.90, 0.90],
    "UMI": [1.00, 0.95, 0.00, 0.00, 0.00, 0.00],
    "UME": [1.00, 1.00, 0.95, 0.95, 0.95, 0.95],
}
STAGES = ["turn_around", "open_fridge", "take_out", "close_fridge", "transport", "place"]


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(c - h), float(c + h)


def clopper(k, n, alpha=0.05):
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def mh_exact_pooled(strata):
    """층별(과제별) 2x2 표의 Mantel-Haenszel 공통 오즈비와 정확 조건부 검정 대신 순열 검정(방법 라벨 섞기)."""
    num = den = 0.0
    for a, b, c, d in strata:        # a=UME 성공, b=UME 실패, c=기준 성공, d=기준 실패
        n = a + b + c + d
        num += a * d / n
        den += b * c / n
    return num / den if den > 0 else float("inf")


def perm_pooled(ume, base, reps=20000, seed=0):
    """과제 안에서 UME/기준선 라벨을 섞어 층별 성공 수 차이 합의 분포를 만든다 (층화 순열 검정)."""
    rng = np.random.default_rng(seed)
    obs = sum(u - b for u, b in zip(ume, base))
    cnt = 0
    for _ in range(reps):
        s = 0
        for u, b in zip(ume, base):
            pool = np.array([1] * (u + b) + [0] * (2 * N - u - b))
            rng.shuffle(pool)
            s += pool[:N].sum() - pool[N:].sum()
        if abs(s) >= abs(obs):
            cnt += 1
    return (cnt + 1) / (reps + 1)


def power_two_prop(p1, p2, n, reps=4000, seed=0):
    rng = np.random.default_rng(seed)
    hit = 0
    for _ in range(reps):
        a = rng.binomial(n, p1)
        b = rng.binomial(n, p2)
        if fisher(a, n, b, n) < 0.05:
            hit += 1
    return hit / reps


def n_needed(p1, p2, target=0.8, nmax=400):
    for n in range(10, nmax, 5):
        if power_two_prop(p1, p2, n, reps=600, seed=n) >= target:
            return n
    return None


def main():
    res = {}
    # ---- V1: 표 1 정수화, 본문 대조 ----
    counts = {m: [int(round(p * N)) for p in v] for m, v in T1.items()}
    res["V1_counts"] = counts
    res["V1_integer_feasible"] = {m: [abs(p * N - round(p * N)) < 1e-9 for p in v] for m, v in T1.items()}
    res["V1_text_match_ume"] = counts["UME"] == TEXT_UME
    # ---- V2: Wilson, Clopper–Pearson ----
    res["V2_ci"] = {m: [{"k": k, "n": N, "p": k / N, "wilson": wilson(k, N), "cp": clopper(k, N)} for k in counts[m]]
                    for m in counts}
    # ---- V3: 쌍별 Fisher + Holm ----
    pairs = [("UME", "No-torque"), ("UME", "UMI"), ("No-torque", "UMI")]
    pv = {}
    flat = []
    keys = []
    for a, b in pairs:
        for t, name in enumerate(TASKS):
            p = fisher(counts[a][t], N, counts[b][t], N)
            pv[f"{a}_vs_{b}:{name}"] = p
            flat.append(p)
            keys.append(f"{a}_vs_{b}:{name}")
    adj = holm(flat)
    res["V3_fisher"] = {k: {"p": pv[k], "p_holm12": adj[i]} for i, k in enumerate(keys)}
    # UME 대 두 기준선 8개만 Holm
    ume_keys = [k for k in keys if k.startswith("UME_vs")]
    adj8 = holm([pv[k] for k in ume_keys])
    for k, a8 in zip(ume_keys, adj8):
        res["V3_fisher"][k]["p_holm8"] = a8
    # ---- V4: 풀링 ----
    tot = {m: int(sum(counts[m])) for m in counts}
    res["V4_pooled_counts"] = {m: f"{tot[m]}/{4 * N}" for m in tot}
    res["V4_pooled_rate"] = {m: tot[m] / (4 * N) for m in tot}
    res["V4_pooled_wilson"] = {m: wilson(tot[m], 4 * N) for m in tot}
    res["V4_fisher_pooled_UME_vs_Notorque"] = fisher(tot["UME"], 80, tot["No-torque"], 80)
    res["V4_fisher_pooled_UME_vs_UMI"] = fisher(tot["UME"], 80, tot["UMI"], 80)
    strata_nt = [(counts["UME"][t], N - counts["UME"][t], counts["No-torque"][t], N - counts["No-torque"][t]) for t in range(4)]
    # 0 칸이 있으므로 Haldane 0.5 보정 MH
    strata_nt_h = [tuple(x + 0.5 for x in s) for s in strata_nt]
    res["V4_MH_OR_UME_vs_Notorque_haldane"] = mh_exact_pooled(strata_nt_h)
    res["V4_stratified_perm_p_UME_vs_Notorque"] = perm_pooled(counts["UME"], counts["No-torque"])
    # 힘 매개 두 과제(밀기, 뒤집기)와 나머지 두 과제 나눔
    for name, idx in [("force_tasks(push,flip)", [0, 1]), ("space_tasks(gpu,fridge)", [2, 3])]:
        u = sum(counts["UME"][i] for i in idx)
        b = sum(counts["No-torque"][i] for i in idx)
        res[f"V4_{name}_UME_vs_Notorque"] = {"ume": f"{u}/40", "notorque": f"{b}/40", "fisher_p": fisher(u, 40, b, 40)}
    # ---- V5: 평균 대조 ----
    res["V5_mean_rates"] = {m: float(np.mean(T1[m])) for m in T1}
    # ---- V6: 표 2 일관성 ----
    c2 = {m: [int(round(p * N)) for p in v] for m, v in T2.items()}
    res["V6_stage_counts"] = c2
    res["V6_monotone_nonincreasing"] = {m: all(c2[m][i] >= c2[m][i + 1] for i in range(5)) for m in c2}
    res["V6_final_equals_table1_fridge"] = {m: c2[m][-1] == counts[m][3] for m in c2}
    cond = {}
    for m in c2:
        row = []
        for i in range(1, 6):
            row.append(None if c2[m][i - 1] == 0 else c2[m][i] / c2[m][i - 1])
        cond[m] = row
    res["V6_conditional_stage_rate"] = cond
    res["V6_text_claims"] = {
        "UMI_fails_open_fridge_once": c2["UMI"][1] == 19,
        "UMI_never_takes_out_drink": c2["UMI"][2] == 0,
        "No-torque_open_fridge_fail": N - c2["No-torque"][1],
        "No-torque_takeout_fail": c2["No-torque"][1] - c2["No-torque"][2],
        "UME_takeout_fail": c2["UME"][1] - c2["UME"][2],
    }
    # ---- V7: 그림 5 (분당 시연 수) ----
    dpm = {"UME_wo_torque": 6, "UME": 20, "Human": 28}
    res["V7_dpm_ratio_ume_over_notorque"] = dpm["UME"] / dpm["UME_wo_torque"]
    res["V7_dpm_ume_over_human"] = dpm["UME"] / dpm["Human"]
    res["V7_dpm_cycle_seconds"] = {k: 60.0 / v for k, v in dpm.items()}
    # 막대 값이 정수 반올림이라면 가능한 범위
    lo = (dpm["UME"] - 0.5) / (dpm["UME_wo_torque"] + 0.5)
    hi = (dpm["UME"] + 0.5) / (dpm["UME_wo_torque"] - 0.5)
    res["V7_ratio_range_if_rounded"] = [lo, hi]
    res["V7_human_frac_range_if_rounded"] = [(dpm["UME"] - 0.5) / (dpm["Human"] + 0.5), (dpm["UME"] + 0.5) / (dpm["Human"] - 0.5)]
    # 사용자 수: 그림 11은 사용자 3명을 보여 준다. 평균의 표준오차 계산은 불가(분산 미보고).
    res["V7_note"] = "n_users=3 (그림 11에 3명), 분산·검정 없음, 과제는 박스 뒤집기 하나, 에피소드 1회 주기에 리셋 포함"
    # ---- V8: 부품 비용 ----
    bom_ume = [(117, 8), (88, 8), (3.5, 6), (13, 4), (52, 1), (41, 1), (2, 14), (9, 1), (3, 6), (5, 6), (7, 1), (2, 1)]
    bom_robot = [(5200, 1), (900, 4), (386, 1), (37.5, 3), (52, 1), (45, 1), (7.5, 14), (1.6, 16), (7, 1)]
    res["V8_ume_bom_sum"] = float(sum(p * c for p, c in bom_ume))
    res["V8_robot_bom_sum"] = float(sum(p * c for p, c in bom_robot))
    res["V8_actuator_share_of_ume"] = float((117 * 8 + 88 * 8) / sum(p * c for p, c in bom_ume))
    res["V8_base_cost_swerve_vs_pcw25"] = {"pcw25_x4": 900 * 4, "mk4_x4_if_300": 300 * 4}
    res["V8_ume_to_robot_arm_cost_ratio"] = float(sum(p * c for p, c in bom_ume) / 5200)
    # ---- V9: 학습량 산술 (표 3, 부록 D) ----
    steps, batch = 40000, 200
    res["V9_samples_seen"] = steps * batch
    res["V9_steps_per_hour_8h"] = steps / 8.0
    res["V9_samples_per_second_8h"] = steps * batch / (8 * 3600)
    fridge_frames = {hz: 157 * 90 * hz for hz in (10, 30)}   # 90초 에피소드(본문 §3.4 약 1분 30초), 가정한 프레임율
    res["V9_fridge_frames_assumed"] = fridge_frames
    res["V9_fridge_epochs_assumed"] = {hz: steps * batch / f for hz, f in fridge_frames.items()}
    res["V9_note"] = "프레임율과 에피소드 길이는 가정. 90초는 본문 §3.4. 다른 과제 길이는 논문에 없음."
    # ---- V10: 트랜스포머 파라미터 산술 (논문 표 3 설정 대 공식 코드 기본값) ----
    def mha(d):
        return 4 * d * d + 4 * d

    def ffn(d, f):
        return 2 * d * f + f + d

    def tparams(d, f, le, ld):
        enc = le * (mha(d) + ffn(d, f) + 4 * d)
        dec = ld * (2 * mha(d) + ffn(d, f) + 6 * d)
        return enc + dec

    res["V10_transformer_params_paper"] = tparams(512, 3200, 4, 7)
    res["V10_transformer_params_code_default"] = tparams(256, 2048, 4, 6)
    res["V10_ratio_paper_over_code"] = res["V10_transformer_params_paper"] / res["V10_transformer_params_code_default"]
    res["V10_note"] = "인코더 4층(공통), 디코더 7층 대 6층, d_model 512 대 256, FFN 3200 대 2048. 임베딩·헤드·ResNet18(약 11.2M)은 제외한 산술 추정."
    # ---- V11: 검정력 ----
    res["V11_power_n20"] = {
        "0.95_vs_0.90": power_two_prop(0.95, 0.90, 20),
        "0.90_vs_0.50": power_two_prop(0.90, 0.50, 20),
        "0.95_vs_0.75": power_two_prop(0.95, 0.75, 20),
    }
    res["V11_n_per_arm_for_80pct_power"] = {
        "0.95_vs_0.75": n_needed(0.95, 0.75),
        "0.95_vs_0.90": n_needed(0.95, 0.90, nmax=1200),
    }
    # 관측된 격차 중 n=20에서 검출 가능한 최소 격차 (기준 0.9)
    res["V11_note"] = "검정력은 Fisher 정확 검정(양측, alpha=0.05)의 시뮬레이션, 4000회(필요 n은 600회)"
    # ---- K: 일관성 (수치 대조) ----
    k = {}
    k["K1_table1_vs_text_ume_counts"] = counts["UME"] == TEXT_UME
    k["K2_table2_final_equals_table1"] = all(res["V6_final_equals_table1_fridge"].values())
    k["K3_percentages_integer_feasible_n20"] = all(all(v) for v in res["V1_integer_feasible"].values())
    k["K4_dpm_ratio_3.3x_ok"] = abs(res["V7_dpm_ratio_ume_over_notorque"] - 3.3) < 0.05
    k["K5_dpm_71pct_ok"] = abs(res["V7_dpm_ume_over_human"] - 0.71) < 0.005
    k["K6_bom_ume_1900_ok"] = abs(res["V8_ume_bom_sum"] - 1900) < 1e-6
    k["K7_bom_robot_9533.1_ok"] = abs(res["V8_robot_bom_sum"] - 9533.1) < 1e-6
    res["K"] = k
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    print("V1 counts:", counts, "text match:", res["V1_text_match_ume"])
    for key, v in res["V3_fisher"].items():
        print(f"{key:40s} p={v['p']:.4g}  holm12={v['p_holm12']:.3g}", f"holm8={v.get('p_holm8', float('nan')):.3g}")
    print("pooled:", res["V4_pooled_counts"], "UME vs NT p=", res["V4_fisher_pooled_UME_vs_Notorque"],
          "UME vs UMI p=", res["V4_fisher_pooled_UME_vs_UMI"], "perm p=", res["V4_stratified_perm_p_UME_vs_Notorque"],
          "MH OR", res["V4_MH_OR_UME_vs_Notorque_haldane"])
    print("force/space:", res["V4_force_tasks(push,flip)_UME_vs_Notorque"], res["V4_space_tasks(gpu,fridge)_UME_vs_Notorque"])
    print("V6:", res["V6_monotone_nonincreasing"], res["V6_conditional_stage_rate"])
    print("V7:", res["V7_dpm_ratio_ume_over_notorque"], res["V7_dpm_ume_over_human"], res["V7_dpm_cycle_seconds"],
          res["V7_ratio_range_if_rounded"], res["V7_human_frac_range_if_rounded"])
    print("V8:", res["V8_ume_bom_sum"], res["V8_robot_bom_sum"], res["V8_actuator_share_of_ume"])
    print("V9:", res["V9_samples_per_second_8h"], res["V9_fridge_epochs_assumed"])
    print("V10:", res["V10_transformer_params_paper"], res["V10_transformer_params_code_default"], res["V10_ratio_paper_over_code"])
    print("V11:", res["V11_power_n20"], res["V11_n_per_arm_for_80pct_power"])
    print("K:", k)


if __name__ == "__main__":
    main()
