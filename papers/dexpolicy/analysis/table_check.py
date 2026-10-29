"""Table I–VIII과 본문 수치 재계산 (CPU). 실행: python3 analysis/table_check.py

- C1: Table I의 Overall이 다섯 물체 평균인지, 8개 물체 평균(Sec. V-B)이 나머지 3개 ≈ 0과 맞는지
- C2: 본문에 공개된 seed별 값(FPO 머스터드·토마토·머그, GRPO 머스터드)으로 Table I 평균·표본 SD 재현
- C3: Table II의 물체별 차이 평균이 본문의 다섯 물체 평균 차이와 맞는지
- C4: Table III 평균, 물체별·합산 Wilson 95% 구간과 Fisher 정확검정 (시행 단위, 모델은 조건당 1개)
- C5: seed 수준 평균±SD로 본 Welch t 근사, 물체별 증감 개수
- C6: GRPO 연속 학습 step 수(8,192의 배수), Adam 업데이트 수와 log σ 도달 가능 범위
- C7: Table V 통과 판정, Table VI·VII·VIII 내부 일관성
- C8: 하드웨어 "조건당 모델 1개"의 의미: 시뮬레이션 seed 분포로 본 단일 모델 추첨
"""
import json
import math
from itertools import product
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OBJ5 = ("mustard", "mug", "banana", "sugar", "tomato")

# Table I: 평균, 표본 SD (3 seed), Overall
T1 = {
    "PPO": {"base": ([84.0, 25.3, 8.7, 0.0, 42.0], [5.3, 9.5, 13.3, 0.0, 14.7], (32.0, 5.3)),
            "dex": ([94.7, 40.7, 2.3, 0.0, 40.7], [9.2, 17.1, 2.1, 0.0, 9.1], (35.7, 3.9))},
    "GRPO": {"base": ([30.8, 25.3, 3.0, 4.2, 7.0], [53.4, 15.0, 3.9, 4.3, 3.1], (14.1, 8.5)),
             "dex": ([34.5, 41.7, 3.2, 88.0, 59.8], [56.7, 3.3, 2.8, 12.3, 4.5], (45.4, 14.4))},
    "FPO": {"base": ([51.7, 25.3, 16.3, 91.3, 62.3], [46.0, 34.8, 8.0, 13.3, 54.0], (49.4, 21.5)),
            "dex": ([100.0, 31.0, 17.7, 99.0, 92.7], [0.0, 53.7, 16.6, 1.7, 4.0], (68.1, 11.4))},
}
EIGHT = {"PPO": (20.0, 22.3), "FPO": (30.9, 42.5)}  # Sec. V-B, 나머지 3개는 ≤ 0.34%

# 본문에 공개된 seed별 값 (seed 0/1/2)
SEEDS = {
    "FPO mustard": ([88, 67, 0], [100, 100, 100], "Sec. V-B"),
    "FPO tomato": ([95, 0, 92], [95, 88, 95], "Sec. V-B"),
    "FPO mug": ([65, 0, 11], [0, 0, 93], "Sec. V-B"),
    "GRPO mustard": ([0, 0, 92.5], [100, 0.5, 3.0], "Sec. V-C"),
}

# Table II: +DexPolicy − baseline (%p)
T2 = {"seeds12_sigma0": [-44.5, 23.5, 0.2, 80.5, 49.8],
      "all_sigma003": [6.3, 13.3, 0.3, 72.0, 43.0],
      "all_sigma010": [-6.7, 2.7, 0.7, -32.0, -1.7]}
T2_TEXT = {"seeds12_sigma0": (15.3, 37.2), "all_sigma003": (14.8, 41.8), "all_sigma010": (16.1, 8.7)}

# Table III: 20회/조건, (tomato, sugar, mustard) × (Target, Lift)
T3 = {
    "PPO": {"base": ((10, 35), (0, 5), (15, 75)), "dex": ((45, 70), (0, 10), (85, 95))},
    "GRPO": {"base": ((5, 35), (10, 75), (15, 30)), "dex": ((55, 90), (90, 100), (45, 95))},
    "FPO": {"base": ((25, 30), (20, 45), (30, 60)), "dex": ((80, 95), (75, 85), (100, 100))},
}
T3_TEXT = {"PPO": ((8.3, 43.3), (38.3, 58.3)), "GRPO": ((10.0, 63.3), (46.7, 95.0)), "FPO": ((25.0, 85.0), (45.0, 93.3))}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def main() -> None:
    out: dict = {}

    # C1
    c1 = {}
    for m, d in T1.items():
        for cond, (means, _, (ov, _)) in d.items():
            c1[f"{m}/{cond}"] = {"mean_of_5": round(float(np.mean(means)), 2), "reported": ov}
    for m, (b, x) in EIGHT.items():
        for cond, rep in (("base", b), ("dex", x)):
            s5 = 5 * np.mean(T1[m][cond][0])
            c1[f"{m}/{cond}/eight_object"] = {
                "if_other3_zero": round(s5 / 8, 2), "if_other3_at_0.34": round((s5 + 3 * 0.34) / 8, 2), "reported": rep}
    out["C1_overall_and_eight_object_means"] = c1

    # C2
    c2 = {}
    for k, (b, x, src) in SEEDS.items():
        c2[k] = {"source": src,
                 "base_mean_sd": [round(float(np.mean(b)), 2), round(float(np.std(b, ddof=1)), 2)],
                 "dex_mean_sd": [round(float(np.mean(x)), 2), round(float(np.std(x, ddof=1)), 2)],
                 "paired_up_tie_down": [int(sum(xx > bb for bb, xx in zip(b, x))),
                                        int(sum(xx == bb for bb, xx in zip(b, x))),
                                        int(sum(xx < bb for bb, xx in zip(b, x)))]}
    c2["GRPO mustard seeds1_2"] = {"base": float(np.mean([0, 92.5])), "dex": float(np.mean([0.5, 3.0])),
                                   "text": "46.25% → 1.75% (Sec. V-C)"}
    # FPO Overall SD가 공개 seed 값과 양립하는지: 알려진 세 물체 합 + 나머지 두 물체(바나나, 설탕) 평균을 seed마다 같다고 둔 근사
    for cond, idx in (("base", 0), ("dex", 1)):
        known = np.array([SEEDS["FPO mustard"][idx], SEEDS["FPO tomato"][idx], SEEDS["FPO mug"][idx]]).sum(axis=0)
        rest = T1["FPO"][cond][0][2] + T1["FPO"][cond][0][3]
        per_seed = (known + rest) / 5
        c2[f"FPO/{cond}/overall_if_banana_sugar_constant"] = {
            "per_seed": [round(v, 1) for v in per_seed], "sd": round(float(np.std(per_seed, ddof=1)), 1),
            "reported_sd": T1["FPO"][cond][2][1]}
    out["C2_seed_level_values"] = c2

    # C3
    out["C3_table2_vs_text"] = {
        k: {"mean_diff_table2": round(float(np.mean(v)), 2),
            "text_diff": round(T2_TEXT[k][1] - T2_TEXT[k][0], 2), "text": T2_TEXT[k]} for k, v in T2.items()}
    out["C3_table2_vs_text"]["note"] = "seeds12_sigma0 행은 본문 'Excluding seed 0 ... 15.3% → 37.2%'와 대조"

    # C4
    c4 = {}
    for m, d in T3.items():
        row = {}
        for metric_i, metric in enumerate(("Target", "Lift")):
            kb = [round(v[metric_i] * 20 / 100) for v in d["base"]]
            kd = [round(v[metric_i] * 20 / 100) for v in d["dex"]]
            per_obj = []
            for name, a, b in zip(("tomato", "sugar", "mustard"), kb, kd):
                _, p = stats.fisher_exact([[a, 20 - a], [b, 20 - b]])
                per_obj.append({"object": name, "base": f"{a}/20", "dex": f"{b}/20", "fisher_p": float(f"{p:.2g}")})
            A, B = sum(kb), sum(kd)
            _, p = stats.fisher_exact([[A, 60 - A], [B, 60 - B]])
            row[metric] = {"mean_base_dex": [round(100 * A / 60, 1), round(100 * B / 60, 1)],
                           "reported": T3_TEXT[m][metric_i], "pooled": f"{A}/60 vs {B}/60",
                           "wilson_base": wilson(A, 60), "wilson_dex": wilson(B, 60),
                           "pooled_fisher_p": float(f"{p:.2g}"), "per_object": per_obj}
        c4[m] = row
    c4["target_le_lift_all_rows"] = all(t <= l for d in T3.values() for c in d.values() for (t, l) in c)
    out["C4_hardware"] = c4

    # C5
    c5 = {}
    for m, d in T1.items():
        (mb, sb), (mx, sx) = d["base"][2], d["dex"][2]
        se = math.sqrt(sb ** 2 / 3 + sx ** 2 / 3)
        t = (mx - mb) / se
        df = se ** 4 / ((sb ** 2 / 3) ** 2 / 2 + (sx ** 2 / 3) ** 2 / 2)
        p = 2 * stats.t.sf(abs(t), df)
        diffs = np.array(d["dex"][0]) - np.array(d["base"][0])
        c5[m] = {"diff_pp": round(mx - mb, 1), "welch_t": round(t, 2), "df": round(df, 1), "p_two_sided": round(p, 3),
                 "objects_up_tie_down": [int((diffs > 0.05).sum()), int((abs(diffs) <= 0.05).sum()), int((diffs < -0.05).sum())],
                 "per_object_diff": [round(v, 1) for v in diffs]}
    c5["note"] = ("seed별 Overall이 없어 비대응 Welch 근사. 대응(같은 seed) 검정은 계산 불가. "
                  "본문: PPO는 8개 물체 평균이 1/3 seed에서만 개선, FPO는 2/3 (Sec. V-B)")
    out["C5_seed_level_tests"] = c5

    # C6
    roll_grpo = 512 * 16
    adam_steps_5m = 5_000_000 / 256 * 5
    c6 = {
        "grpo_rollout": roll_grpo,
        "phase_steps_over_rollout": {str(s): s / roll_grpo for s in (507_904, 253_952)},
        "total_continuation": 507_904 + 253_952, "tomato_total": 507_904 + 2 * 253_952,
        "ppo_updates_per_5m": round(5_000_000 / 4096), "adam_steps_5m": round(adam_steps_5m),
        "max_log_std_change_lr1e-5": round(adam_steps_5m * 1e-5, 3),
        "observed_baseline_log_change": round(math.log(0.1855 / math.exp(-1.6)), 3),
        "needed_to_reach_0.04": round(math.log(0.04 / math.exp(-1.6)), 3),
        "init_std_exp(-1.6)": round(math.exp(-1.6), 4),
        "grpo_phase_adam_steps": [round(507_904 / 256 * 5), round(253_952 / 256 * 5)],
        "grpo_max_log_std_change": round(507_904 / 256 * 5 * 1e-4 + 253_952 / 256 * 5 * 5e-5, 3),
        "note": "Adam 한 step의 매개변수 변화는 대략 lr 이하(기울기 부호가 일정할 때 lr에 가까움). 근사적 상한",
    }
    out["C6_budget_and_adam_bound"] = c6

    # C7
    t5 = {"Uniform linear": (91.9, 86.2, 110.6, "No"), "Continuous front-loaded": (93.8, 92.5, 95.8, "No"),
          "No 2M jump": (100.3, 95.5, 108.3, "Yes"), "No final drop": (94.3, 91.9, 78.2, "No")}
    out["C7_table5_gate"] = {k: {"all_means_ge_95": all(v >= 95 for v in r[:3]), "reported_pass": r[3]}
                             for k, r in t5.items()}
    t7 = {"reward_auc_0_10M": (36.57, 47.79, 30.7), "reward_auc_5_10M": (53.09, 66.44, 25.2),
          "reward_last20": (59.51, 68.28, 14.7), "contact_auc": (0.620, 0.704, 13.5), "lift_auc": (0.0654, 0.0731, 11.7)}
    out["C7_table7_rel_change"] = {k: {"recomputed": round(100 * (b - a) / a, 1), "reported": r} for k, (a, b, r) in t7.items()}
    out["C7_text_10M_reeval"] = {"mean_reward": round(100 * (69.26 - 62.30) / 62.30, 1),
                                 "paired_gain_mean": round(np.mean([6.69, 5.67, 8.52]), 2),
                                 "lift": round(100 * (0.1095 - 0.1066) / 0.1066, 1)}

    # C8: 조건당 모델 1개 → 시뮬레이션 seed 분포에서 하나를 뽑는 것과 같다
    b, x, _ = SEEDS["FPO mug"]
    pairs = list(product(b, x))
    out["C8_single_model_lottery"] = {
        "example": "FPO mug(시뮬레이션 seed 값)에서 baseline과 DexPolicy 모델을 하나씩 뽑을 때",
        "P(dex>base)": round(sum(xx > bb for bb, xx in pairs) / 9, 2),
        "P(dex<base)": round(sum(xx < bb for bb, xx in pairs) / 9, 2),
        "P(tie)": round(sum(xx == bb for bb, xx in pairs) / 9, 2),
        "GRPO mustard P(dex>base)": round(sum(xx > bb for bb, xx in product(*SEEDS["GRPO mustard"][:2])) / 9, 2),
        "note": "하드웨어 정책은 별도 embodiment로 학습해서 이 분포와 같지 않다. 단일 모델 비교가 seed 운에 얼마나 열려 있는지 보는 예시",
    }

    path = ROOT / "analysis" / "table_check_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for k, v in out.items():
        print(k, json.dumps(v, ensure_ascii=False)[:400])
    print("wrote", path)


if __name__ == "__main__":
    main()
