"""Table 1·2와 Fig. 7(판독값) 재계산, 부록 보상·학습 설정의 산술 (CPU). 실행: python3 analysis/table_check.py

- C1: "약 70%" 평균, 원격조작 "약 30%(집게)" 재계산 (Sec. 5.2)
- C2: 칸별(n = 10)·도구별(n = 60 또는 40) Wilson 구간과 Fisher 정확검정
- C3: 단계별 평가를 곱했을 때의 기대 연쇄 성공률과 Table 2 비교 (조건부 해석 두 가지)
- C4: Fig. 7 판독값의 경향 검정(Cochran–Armitage)과 끝점 Fisher 검정, Table 1과의 정합
- C5: 부록 A.2 보상 스케일 — 각 항이 절반이 되는 오차, 항별 최대 기여
- C6: 부록 A.4 PPO 배치 산술
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
N = 10  # Table 1 캡션: "Each evaluation contains 10 trials"

# Table 1: (instance 1, instance 2) 성공률
PHASES = {
    "tongs": ["grasp", "open", "close"],
    "pliers": ["grasp", "open", "close"],
    "clothespins": ["grasp", "open", "close"],
    "syringes": ["grasp", "use"],
}
T1 = {
    "teleop": {
        "tongs": [(0.3, 0.3), (0.1, 0.2), (0.3, 0.3)],
        "pliers": [(0.2, 0.1), (0.1, 0.1), (0.1, 0.0)],
        "clothespins": [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
        "syringes": [(0.0, 0.0), (0.0, 0.0)],
    },
    "openloop": {
        "tongs": [(0.5, 0.6), (0.6, 0.7), (0.4, 0.6)],
        "pliers": [(0.4, 0.4), (0.3, 0.4), (0.3, 0.3)],
        "clothespins": [(0.3, 0.2), (0.2, 0.4), (0.4, 0.3)],
        "syringes": [(0.1, 0.0), (0.3, 0.3)],
    },
    "ours": {
        "tongs": [(0.8, 0.8), (0.8, 0.8), (0.7, 0.8)],
        "pliers": [(0.7, 0.6), (0.7, 0.7), (0.7, 0.7)],
        "clothespins": [(0.8, 0.7), (0.8, 0.7), (0.6, 0.7)],
        "syringes": [(0.7, 0.5), (0.6, 0.6)],
    },
}
# Table 2: 도구 사용 과제 성공 (wrist는 원격조작, 손가락은 정책)
T2 = {"tongs": (7, 10), "pliers": (5, 10), "clothespins": (6, 10), "syringes": (5, 10)}
# Fig. 7 판독값 (300 dpi 렌더에서 격자선과 정확히 겹침: 0.1 단위)
FIG7 = {
    "tongs_grasp_vs_trajectories": {"x": ["2k", "4k", "8k", "16k"], "sr": [0.2, 0.4, 0.7, 0.8]},
    "syringe_use_vs_keyframes": {"x": ["1k", "2k", "4k", "8k"], "sr": [0.0, 0.3, 0.3, 0.6]},
    "pliers_mani_vs_force_rand": {"x": ["0%", "25%", "50%", "100%"], "sr": [0.1, 0.2, 0.4, 0.7]},
}


def wilson(k: int, n: int, z: float = 1.959964) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def fisher(k1: int, n1: int, k2: int, n2: int) -> float:
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def counts(method: str, tool: str) -> list[int]:
    return [round(v * N) for pair in T1[method][tool] for v in pair]


def cochran_armitage(ks: list[int], n: int, scores=None) -> float:
    """선형 경향 검정 양측 p (각 점 시행 수 n 동일)."""
    k = np.array(ks, float)
    t = np.arange(len(ks), dtype=float) if scores is None else np.array(scores, float)
    nn = np.full_like(k, n)
    p = k.sum() / nn.sum()
    tbar = (nn * t).sum() / nn.sum()
    num = (t * (k - nn * p)).sum()
    var = p * (1 - p) * (nn * (t - tbar) ** 2).sum()
    if var == 0:
        return 1.0
    z = num / math.sqrt(var)
    return float(2 * stats.norm.sf(abs(z)))


def main() -> None:
    out: dict = {}

    # C1 평균
    cells = {m: [v for tool in T1[m] for pair in T1[m][tool] for v in pair] for m in T1}
    grasp = {m: [v for tool in T1[m] for v in T1[m][tool][0]] for m in T1}
    mani = {m: [v for tool in T1[m] for pair in T1[m][tool][1:] for v in pair] for m in T1}
    out["C1_means"] = {
        m: {"all_22_cells": round(float(np.mean(cells[m])), 3),
            "grasp_8_cells": round(float(np.mean(grasp[m])), 3),
            "manipulation_14_cells": round(float(np.mean(mani[m])), 3)}
        for m in T1
    }
    out["C1_teleop_tongs_mean"] = round(float(np.mean(cells["teleop"][:6])), 3)
    out["C1_claim"] = "approximately 70% success rates for both grasping and in-hand manipulation; teleop around 30% on tongs (Sec. 5.2)"

    # C2 칸별·도구별 검정
    c2 = {}
    for tool, phases in PHASES.items():
        o, b, t = counts("ours", tool), counts("openloop", tool), counts("teleop", tool)
        no = len(o) * N
        row = {
            "ours_pooled": [sum(o), no, wilson(sum(o), no)],
            "openloop_pooled": [sum(b), no, wilson(sum(b), no)],
            "teleop_pooled": [sum(t), no, wilson(sum(t), no)],
            "p_ours_vs_openloop": float(f"{fisher(sum(o), no, sum(b), no):.2e}"),
            "p_ours_vs_teleop": float(f"{fisher(sum(o), no, sum(t), no):.2e}"),
        }
        cellp = [fisher(ko, N, kb, N) for ko, kb in zip(o, b)]
        row["cells_sig_vs_openloop_p<0.05"] = f"{sum(p < 0.05 for p in cellp)}/{len(cellp)}"
        row["cell_p_vs_openloop_min"] = round(min(cellp), 3)
        c2[tool] = row
    ko = sum(sum(counts("ours", t)) for t in PHASES)
    kb = sum(sum(counts("openloop", t)) for t in PHASES)
    out["C2_tests"] = c2
    out["C2_all_cells_ours_vs_openloop"] = {"ours": [ko, 220], "openloop": [kb, 220],
                                            "p": float(f"{fisher(ko, 220, kb, 220):.2e}")}
    out["C2_cell_ci_example"] = {"0.8 (8/10)": wilson(8, 10), "0.5 (5/10)": wilson(5, 10),
                                 "0.7 (7/10)": wilson(7, 10)}

    # C3 단계 연쇄
    c3 = {}
    for tool, phases in PHASES.items():
        m = [float(np.mean(pair)) for pair in T1["ours"][tool]]  # 단계별 두 인스턴스 평균
        full = float(np.prod(m))
        cond = float(np.prod(m[1:]))  # 잡기 성공을 조건으로
        k, n = T2[tool]
        c3[tool] = {
            "phase_sr": [round(x, 3) for x in m],
            "product_all_phases": round(full, 3),
            "product_after_grasp": round(cond, 3),
            "table2": f"{k}/{n}",
            "P(X>=k | all phases indep.)": round(float(stats.binom.sf(k - 1, n, full)), 3),
            "P(X>=k | after grasp)": round(float(stats.binom.sf(k - 1, n, cond)), 3),
            "end_to_end_if_table2_is_conditional": round(k / n * m[0], 3),
        }
    # 네 과제 합 (Poisson-binomial 근사: 독립 이항 합의 정확 분포)
    def sum_tail(ps, k_obs):
        dist = np.array([1.0])
        for p in ps:
            d = stats.binom.pmf(np.arange(11), 10, p)
            dist = np.convolve(dist, d)
        return float(dist[k_obs:].sum())
    k_tot = sum(k for k, _ in T2.values())
    c3["sum"] = {
        "table2_total": f"{k_tot}/40",
        "expected_all_phases": round(10 * sum(v["product_all_phases"] for v in c3.values()), 1),
        "expected_after_grasp": round(10 * sum(v["product_after_grasp"] for k_, v in c3.items() if k_ != "sum"), 1),
        "P(total>=obs | all phases)": round(sum_tail([c3[t]["product_all_phases"] for t in PHASES], k_tot), 4),
        "P(total>=obs | after grasp)": round(sum_tail([c3[t]["product_after_grasp"] for t in PHASES], k_tot), 4),
    }
    out["C3_phase_chain_vs_table2"] = c3

    # C4 Fig. 7
    c4 = {}
    for key, d in FIG7.items():
        ks = [round(v * N) for v in d["sr"]]
        c4[key] = {
            "counts_of_10": ks,
            "trend_p_cochran_armitage": round(cochran_armitage(ks, N), 4),
            "endpoint_fisher_p": round(fisher(ks[-1], N, ks[0], N), 4),
            "adjacent_last_two_fisher_p": round(fisher(ks[-1], N, ks[-2], N), 3),
            "max_point_ci": wilson(ks[-1], N),
        }
    c4["matches_table1"] = {
        "tongs_grasp_16k=0.8": T1["ours"]["tongs"][0],
        "syringe_use_8k=0.6": T1["ours"]["syringes"][1],
        "pliers_open/close_100%=0.7": [T1["ours"]["pliers"][1], T1["ours"]["pliers"][2]],
    }
    out["C4_fig7"] = c4

    # C5 보상 스케일 (부록 A.2)
    beta = {"b11_revolute_per_rad": 10.0, "b11_prismatic_per_m": 50.0, "b12_tool_pos_per_m": 100.0,
            "b13_tool_rot_per_rad": 5.0, "b21_hand_joint_L1_per_rad": 10.0, "b22_palm_pos_per_m": 50.0,
            "b23_palm_rot_per_rad": 5.0}
    half = {k: round(math.log(2) / v, 4) for k, v in beta.items()}
    w1, w2, lam1, lam2 = 0.1, 0.15, 0.3, 10.0
    out["C5_reward_scales"] = {
        "half_reward_error": half,
        "half_reward_hand_L1_per_joint_avg_16dof_rad": round(math.log(2) / 10.0 / 16, 5),
        "max_dense_terms_inhand": {"r_tool": 1 + lam1, "w1*r_hand": w1 * 1.0,
                                   "w2*r_contact_per_contact": w2},
        "max_dense_terms_grasp": {"w1*r_hand(with wrist)": w1 * (1 + lam2)},
        "contacts_to_match_r_tool_max": round((1 + lam1) / w2, 1),
        "success_bonus": 100, "fail_penalty": -50, "global_scale": 0.1,
        "note": "r_contact는 접촉 쌍 개수라 상한이 없다. 접촉 9개면 r_tool 최대(1.3)와 같다",
    }

    # C6 PPO 배치
    envs, horizon, minibatch, epochs = 4096, 8, 8192, 5
    out["C6_ppo_batch"] = {
        "samples_per_update": envs * horizon,
        "minibatches_per_epoch": envs * horizon // minibatch,
        "gradient_steps_per_update": epochs * envs * horizon // minibatch,
    }

    path = ROOT / "analysis" / "table_check_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for k, v in out.items():
        print(k, json.dumps(v, ensure_ascii=False)[:400])
    print("wrote", path)


if __name__ == "__main__":
    main()
