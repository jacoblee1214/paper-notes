"""AdvDex (arXiv:2608.14028) 보고 수치 재계산.

입력은 논문 표 값만 씁니다(paper/paper.txt의 Table 1–4, Figure 1 문단). 실행과 학습은 하지 않습니다.
가정: 실로봇 표(Table 3, 4)는 시도당 성공 여부이며 과제당 20회입니다(§4.3 Protocol, 20 trials per task).
Table 3·4 값이 모두 5의 배수인 점과 맞습니다. 출력: recompute_results.json (수 초).
"""
import json
import math
import os

from scipy.stats import fisher_exact

HERE = os.path.dirname(os.path.abspath(__file__))
N = 20  # 과제당 시도 수 (논문 §4.3)

# Table 3: 실로봇 성공률 (%). 열 순서: 단일 파지, 다중 파지, 물 붓기, 밀기, 쌓기, 미학습 물체, 미학습 환경
COLS = ["grasp_single", "grasp_multi", "pour", "push", "stack", "unseen_obj", "unseen_env"]
SEEN = COLS[:5]
T3 = {
    "pi05": [75, 60, 55, 85, 60, 35, 45],
    "vita": [70, 65, 40, 90, 70, 40, 35],
    "wo_pretrain": [35, 20, 10, 50, 25, 5, 0],
    "wo_omnishare": [75, 55, 45, 85, 65, 25, 20],
    "wo_adv": [70, 60, 40, 80, 50, 15, 30],
    "ours": [80, 70, 55, 90, 70, 50, 60],
}
# Table 4: 사람 데이터만 있는 과제 (co-training), 열: 상자 인형, 버튼, 병 옮기기, 도구 사용
T4_COLS = ["box_doll", "press_button", "move_bottle", "tool_use"]
T4 = {
    "pi05": [30, 50, 30, 0],
    "vita": [40, 45, 25, 20],
    "wo_adv_pre": [50, 35, 35, 15],
    "wo_adv_post": [25, 30, 15, 5],
    "ours": [60, 70, 45, 30],
}
# Figure 1 막대 값 (논문에 적힌 값)
FIG1 = {"real_avg": {"pi05": 59.3, "vita": 58.6, "ours": 67.9}, "transfer_avg": {"pi05": 27.5, "vita": 32.5, "ours": 51.3}}


def to_counts(pct_list, n=N):
    out = []
    for p in pct_list:
        k = round(p * n / 100)
        assert abs(k * 100 / n - p) < 1e-9, f"{p}% is not k/{n}"
        out.append(k)
    return out


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def newcombe_diff(k1, n1, k2, n2, z=1.959964):
    """Newcombe (1998) 방법 10: 두 비율 차이의 점수 구간 (Wilson 구간 결합)."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1, z)
    l2, u2 = wilson(k2, n2, z)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, (lo, hi)


def fisher(k1, n1, k2, n2):
    _, p = fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])
    return p


def main():
    res = {"assumption_trials_per_cell": N, "table3": {}, "table4": {}, "checks": {}}

    counts = {m: to_counts(v) for m, v in T3.items()}
    counts4 = {m: to_counts(v) for m, v in T4.items()}

    # 1. 열별 Wilson 구간
    res["table3"]["wilson"] = {
        m: {c: {"k": k, "p": round(k / N, 3), "wilson95": [round(x, 3) for x in wilson(k, N)]}
            for c, k in zip(COLS, counts[m])}
        for m in counts
    }

    # 2. 주요 비교: 전체 AdvDex 대 기준선, 열별 Fisher 정확 검정과 차이의 Newcombe 구간
    comps = {"ours_vs_pi05": "pi05", "ours_vs_vita": "vita", "ours_vs_wo_adv": "wo_adv", "ours_vs_wo_omnishare": "wo_omnishare"}
    res["table3"]["pairwise"] = {}
    for name, base in comps.items():
        rows = {}
        for c, ko, kb in zip(COLS, counts["ours"], counts[base]):
            d, ci = newcombe_diff(ko, N, kb, N)
            rows[c] = {"diff_pp": round(100 * d, 1), "newcombe95_pp": [round(100 * ci[0], 1), round(100 * ci[1], 1)],
                       "fisher_p": round(fisher(ko, N, kb, N), 4)}
        # 5개 seen 과제를 합친 검정 (과제당 20회, 합 100회)
        ko = sum(counts["ours"][:5]); kb = sum(counts[base][:5])
        d, ci = newcombe_diff(ko, 5 * N, kb, 5 * N)
        rows["seen_pooled"] = {"ours_k": ko, "base_k": kb, "n": 5 * N, "diff_pp": round(100 * d, 1),
                               "newcombe95_pp": [round(100 * ci[0], 1), round(100 * ci[1], 1)],
                               "fisher_p": round(fisher(ko, 5 * N, kb, 5 * N), 4)}
        # 7개 열 전체 (unseen 2열 포함, 같은 가중)
        ko = sum(counts["ours"]); kb = sum(counts[base])
        rows["all7_pooled"] = {"ours_k": ko, "base_k": kb, "n": 7 * N, "fisher_p": round(fisher(ko, 7 * N, kb, 7 * N), 4)}
        res["table3"]["pairwise"][name] = rows

    # 3. 평균 재계산 (Figure 1 문단 값과 대조)
    avg7 = {m: round(sum(v) / 7, 2) for m, v in T3.items()}
    avg5 = {m: round(sum(v[:5]) / 5, 2) for m, v in T3.items()}
    tr = {m: round(sum(v) / 4, 2) for m, v in T4.items()}
    res["checks"]["fig1_real_avg_7cols"] = {m: {"recomputed": avg7[m], "paper": FIG1["real_avg"][m]} for m in FIG1["real_avg"]}
    res["checks"]["avg_seen5"] = avg5
    res["checks"]["fig1_transfer_avg_table4"] = {m: {"recomputed": tr[m], "paper": FIG1["transfer_avg"][m]} for m in FIG1["transfer_avg"]}

    # 4. 본문 주장 검사: "Ours matches or outperforms baselines on all five seen tasks"
    res["checks"]["ours_ge_all_seen"] = all(
        counts["ours"][i] >= max(counts["pi05"][i], counts["vita"][i]) for i in range(5))
    res["checks"]["ours_ties_seen"] = [COLS[i] for i in range(5)
                                       if counts["ours"][i] == max(counts["pi05"][i], counts["vita"][i])]

    # 5. 4.3 본문 "unseen 열에서 제거 시 더 큰 하락" 검사 (drop = ours - ablation, %p)
    drops = {}
    for abl in ["wo_omnishare", "wo_adv"]:
        drops[abl] = {c: round(T3["ours"][i] - T3[abl][i], 1) for i, c in enumerate(COLS)}
    res["checks"]["ablation_drops_pp"] = drops
    res["checks"]["unseen_drop_claim"] = {
        "unseen_obj": {"wo_omnishare": drops["wo_omnishare"]["unseen_obj"], "wo_adv": drops["wo_adv"]["unseen_obj"]},
        "unseen_env": {"wo_omnishare": drops["wo_omnishare"]["unseen_env"], "wo_adv": drops["wo_adv"]["unseen_env"]},
        "seen_mean_drop": {"wo_omnishare": round(sum(drops["wo_omnishare"][c] for c in SEEN) / 5, 1),
                           "wo_adv": round(sum(drops["wo_adv"][c] for c in SEEN) / 5, 1)},
    }

    # 6. Table 4 소표본: 과제별 "Ours 대 w/o Adv(Post)" Fisher, 전체 풀링
    t4 = {}
    for base in ["pi05", "vita", "wo_adv_pre", "wo_adv_post"]:
        ko = sum(counts4["ours"]); kb = sum(counts4[base])
        t4[base] = {"ours_k": ko, "base_k": kb, "n": 4 * N, "fisher_p": round(fisher(ko, 4 * N, kb, 4 * N), 4),
                    "per_task_fisher": {c: round(fisher(a, N, b, N), 3) for c, a, b in zip(T4_COLS, counts4["ours"], counts4[base])}}
    res["table4"] = t4

    # 7. 단일 시도 한계: 한 정책당 시드 1개 가정 시 시도 샘플링 불확실성만 반영됨
    res["checks"]["note"] = "각 방법은 학습된 정책 1개 (논문 §4.3). 학습 시드 변동은 반영되지 않음."

    # 8. 반올림/잘못된 값 검사: 모든 값이 20회 중 정수 회인지 확인 (to_counts에서 assert)
    res["checks"]["all_values_multiples_of_5"] = True

    # 9. 필요한 시도 수 (두 비율 p1 대 p2 검정력 0.8, 양측 0.05, 정규 근사)
    def n_needed(p1, p2, alpha_z=1.959964, power_z=0.841621):
        pbar = (p1 + p2) / 2
        num = (alpha_z * math.sqrt(2 * pbar * (1 - pbar)) + power_z * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        return math.ceil(num / (p1 - p2) ** 2)
    res["checks"]["n_per_arm_for_80pct_power"] = {
        "p70_vs_p60_proxy_for_seen_gap": n_needed(0.70, 0.60),
        "ours_vs_pi05_unseen_obj": n_needed(0.50, 0.35),
        "ours_vs_vita_unseen_env": n_needed(0.60, 0.35),
    }

    out = os.path.join(HERE, "recompute_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: res["table3"]["pairwise"][k]["seen_pooled"] for k in res["table3"]["pairwise"]}, indent=1))
    print(json.dumps(res["checks"], ensure_ascii=False, indent=1))
    print(json.dumps(res["table4"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
