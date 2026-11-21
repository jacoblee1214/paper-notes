"""3D-ViTac 논문 표 수치 재계산 (CPU, 수 초).

입력: 논문 Table 1(실로봇 4과제), Table 2(카메라 수 절제), Table 3(단일 팔).
표의 값은 pdf 렌더링 이미지(paper/paper_v2.pdf, p.7, p.8, p.22)에서 읽었다.
각 셀은 시도 수 n=20 에피소드 성공률이라고 본다(Table 1, 3 캡션). Table 2는 캡션에 n이
없어 20으로 가정한다. 단계 성공률의 분모는 표에 없어 n=20으로 역산했다.

출력: analysis/recompute_results.json
"""
import json
import math
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
N = 20  # 에피소드 수 (Table 1, 3 캡션; Table 2는 가정)

# ---- Table 1: 단계 4개 (마지막 열이 Whole Task) ----
# 순서: [단계1, 단계2, 단계3, Whole]
T1 = {
    "egg": {  # Egg Steaming (30 demos): Open Egg Tray, Grasp Egg, Place Egg, Whole
        "RGB Only": [0.95, 0.75, 0.60, 0.50],
        "RGB w/ Tactile Image": [0.95, 0.90, 0.80, 0.70],
        "PC. Only": [1.00, 0.75, 0.65, 0.55],
        "PC. w/ Tactile Image": [1.00, 0.85, 0.70, 0.70],
        "PC. w/ Tactile Points (Ours)": [1.00, 0.90, 0.90, 0.85],
    },
    "fruit": {  # Fruit Preparation (30 demos): Grasp and Place Plate, Open Plastic Bag, Grasp Grapes, Whole
        "RGB Only": [0.95, 0.75, 0.45, 0.45],
        "RGB w/ Tactile Image": [0.90, 0.80, 0.70, 0.70],
        "PC. Only": [1.00, 0.85, 0.50, 0.45],
        "PC. w/ Tactile Image": [0.95, 0.90, 0.75, 0.65],
        "PC. w/ Tactile Points (Ours)": [0.95, 0.85, 0.80, 0.80],
    },
    "hex": {  # Hex Key Collection (30 demos): Right Hand Grasp, Left Hand Grasp, In-Hand Adj., Whole
        "RGB Only": [0.90, 0.85, 0.55, 0.45],
        "RGB w/ Tactile Image": [0.95, 0.90, 0.70, 0.60],
        "PC. Only": [1.00, 0.90, 0.65, 0.65],
        "PC. w/ Tactile Image": [0.95, 0.85, 0.60, 0.50],
        "PC. w/ Tactile Points (Ours)": [1.00, 0.95, 0.95, 0.90],
    },
    "sandwich": {  # Sandwich Serving (50 demos): Grasp Serving Spoon, Tilt Pot, Get Fried Egg, Whole
        "RGB Only": [1.00, 0.95, 0.70, 0.60],
        "RGB w/ Tactile Image": [1.00, 1.00, 0.85, 0.75],
        "PC. Only": [1.00, 1.00, 0.75, 0.65],
        "PC. w/ Tactile Image": [1.00, 0.90, 0.75, 0.65],
        "PC. w/ Tactile Points (Ours)": [1.00, 1.00, 0.90, 0.85],
    },
}
T1_STEP_NAMES = {
    "egg": ["Open Egg Tray", "Grasp Egg", "Place Egg", "Whole Task"],
    "fruit": ["Grasp and Place Plate", "Open Plastic Bag", "Grasp Grapes", "Whole Task"],
    "hex": ["Right Hand Grasp", "Left Hand Grasp", "In-Hand Adjustment", "Whole Task"],
    "sandwich": ["Grasp Serving Spoon", "Tilt Pot", "Get Fried Egg", "Whole Task"],
}

# ---- Table 2: 카메라 수 절제 (egg: Open, Grasp, Place, Whole / hex: RH, LH, Adj, Whole) ----
T2 = {
    "egg": {
        "Tactile Points Only": [0.70, 0.55, 0.10, 0.10],
        "Single Cam PC. Only": [0.90, 0.65, 0.60, 0.50],
        "Multi Cam PC. Only": [1.00, 0.75, 0.65, 0.55],
        "Single Cam PC. w/ Tactile Points": [0.95, 0.90, 0.85, 0.80],
        "Multi Cam PC. w/ Tactile Points": [1.00, 0.90, 0.90, 0.85],
    },
    "hex": {
        "Tactile Points Only": [0.50, 0.30, 0.30, 0.15],
        "Single Cam PC. Only": [0.80, 0.60, 0.45, 0.30],
        "Multi Cam PC. Only": [1.00, 0.90, 0.65, 0.65],
        "Single Cam PC. w/ Tactile Points": [0.85, 0.80, 0.75, 0.70],
        "Multi Cam PC. w/ Tactile Points": [1.00, 0.95, 0.95, 0.90],
    },
}

# ---- Table 3: 단일 팔 (Rotation, Relocation / Grasp, Insertion); None = 표에서 '/' ----
T3 = {
    "Light Bulb": {
        "RGB Only": [0.85, 0.85], "RGB w/ Tactile Image": [0.95, 0.95],
        "PC. Only": [0.85, 0.85], "PC. w/ Tactile Points (Ours)": [1.00, 1.00],
        "Ours (New Set of Sensors)": [1.00, 1.00], "Ours (Changing Sensor Location)": [0.95, 0.90],
    },
    "Peg": {
        "RGB Only": [1.00, 0.30], "RGB w/ Tactile Image": [1.00, 0.65],
        "PC. Only": [1.00, 0.65], "PC. w/ Tactile Points (Ours)": [1.00, 0.85],
    },
}

BASELINES = ["RGB Only", "RGB w/ Tactile Image", "PC. Only", "PC. w/ Tactile Image"]
OURS = "PC. w/ Tactile Points (Ours)"


def k_of(p, n=N):
    k = round(p * n)
    assert abs(k / n - p) < 1e-9, (p, n)
    return int(k)


def wilson(k, n=N, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def clopper(k, n=N, a=0.05):
    lo = stats.beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


def fisher(k1, k2, n1=N, n2=N):
    tab = [[k1, n1 - k1], [k2, n2 - k2]]
    return float(stats.fisher_exact(tab, alternative="two-sided")[1])


def holm(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i])
        running = max(running, val)
        adj[i] = running
    return adj


def sample_size_two_prop(p1, p2, alpha=0.05, power=0.8):
    """두 독립 비율 비교의 1군당 n (정규근사, 양측)."""
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    num = za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    return math.ceil((num / (p1 - p2)) ** 2)


def main():
    out = {"n_episodes": N, "assumption": "Table 2 n=20 가정; 단계 분모 n=20 역산"}

    # 1) 셀별 성공 횟수와 CI (Table 1)
    cells = {}
    for task, rows in T1.items():
        for name, vals in rows.items():
            for j, v in enumerate(vals):
                k = k_of(v)
                w = wilson(k)
                cells[f"T1|{task}|{name}|{T1_STEP_NAMES[task][j]}"] = {
                    "k": k, "p": v, "wilson95": [round(w[0], 3), round(w[1], 3)]}
    out["table1_cells"] = cells

    # 2) Whole Task: Ours 대 기준선, 과제별 Fisher, Holm 보정 (16개 검정)
    tests = []
    for task, rows in T1.items():
        k_ours = k_of(rows[OURS][3])
        for b in BASELINES:
            k_b = k_of(rows[b][3])
            tests.append({"task": task, "baseline": b, "k_ours": k_ours, "k_base": k_b,
                          "diff": round(rows[OURS][3] - rows[b][3], 2),
                          "p_fisher": fisher(k_ours, k_b)})
    adj = holm([t["p_fisher"] for t in tests])
    for t, a in zip(tests, adj):
        t["p_holm"] = round(a, 4)
        t["p_fisher"] = round(t["p_fisher"], 4)
    out["table1_whole_vs_baseline"] = tests

    # 3) 과제 4개 합친 80 에피소드 (단순 합산, 과제 난이도 차이 무시한 참고값)
    pooled = {}
    for name in [OURS] + BASELINES:
        ks = sum(k_of(T1[t][name][3]) for t in T1)
        pooled[name] = {"k": ks, "n": 4 * N, "p": round(ks / (4 * N), 3)}
    out["table1_pooled_whole"] = pooled
    out["pooled_fisher_ours_vs_PCTac"] = round(fisher(pooled[OURS]["k"], pooled["PC. w/ Tactile Image"]["k"],
                                                      4 * N, 4 * N), 5)
    out["pooled_fisher_ours_vs_PC"] = round(fisher(pooled[OURS]["k"], pooled["PC. Only"]["k"],
                                                   4 * N, 4 * N), 5)
    out["pooled_fisher_ours_vs_RGB"] = round(fisher(pooled[OURS]["k"], pooled["RGB Only"]["k"],
                                                    4 * N, 4 * N), 6)

    # 3-1) 과제를 층(strata)으로 둔 Cochran-Mantel-Haenszel 검정 (과제 간 변동을 반영)
    from statsmodels.stats.contingency_tables import StratifiedTable
    cmh = {}
    for b in BASELINES:
        tables = []
        for task, rows in T1.items():
            ko, kb = k_of(rows[OURS][3]), k_of(rows[b][3])
            tables.append(np.array([[ko, N - ko], [kb, N - kb]]))
        st = StratifiedTable(tables)
        res = st.test_null_odds(correction=False)
        cmh[b] = {"OR_mh": round(float(st.oddsratio_pooled), 2), "p_cmh": round(float(res.pvalue), 4)}
    out["table1_whole_cmh_ours_vs_baseline"] = cmh

    # 3-2) n=20 설계의 실제 검정력 (Fisher 정확검정, 시뮬레이션 2000회)
    sim_rng = np.random.default_rng(1)
    pw = {}
    for p1, p2 in [(0.85, 0.50), (0.85, 0.65), (0.85, 0.70), (0.90, 0.65)]:
        hits = 0
        for _ in range(2000):
            a, b_ = sim_rng.binomial(N, p1), sim_rng.binomial(N, p2)
            if fisher(a, b_) < 0.05:
                hits += 1
        pw[f"{p1} vs {p2}"] = round(hits / 2000, 3)
    out["power_n20_fisher_sim"] = pw

    # 4) 일관성 검사: Whole Task <= 각 단계 성공률 (순차 과제라면 전체 성공은 어느 단계보다 높을 수 없음)
    viol = []
    for task, rows in T1.items():
        for name, vals in rows.items():
            for j in range(3):
                if vals[3] > vals[j] + 1e-9:
                    viol.append((task, name, T1_STEP_NAMES[task][j], vals[3], vals[j]))
    for group, rows in T2.items():
        for name, vals in rows.items():
            for j in range(len(vals) - 1):
                if vals[-1] > vals[j] + 1e-9:
                    viol.append((f"T2-{group}", name, j, vals[-1], vals[j]))
    out["whole_le_step_violations"] = viol

    # 5) Table 2: 카메라 절제에서 촉각 이득(촉각 포함 - 제외)과 단일/다중 카메라 차이
    gain = {}
    for group in T2:
        for cam in ["Single Cam", "Multi Cam"]:
            base = T2[group][f"{cam} PC. Only"][3]
            tac = T2[group][f"{cam} PC. w/ Tactile Points"][3]
            gain[f"{group}|{cam}"] = {
                "pc_only": base, "pc_tac": tac, "gain": round(tac - base, 2),
                "p_fisher": round(fisher(k_of(tac), k_of(base)), 4)}
    # 이득의 차이 (단일 - 다중) 부트스트랩 신뢰구간 (이항 재표본, 10000회)
    rng = np.random.default_rng(0)
    did = {}
    for group in T2:
        pc_s, tac_s = k_of(T2[group]["Single Cam PC. Only"][3]), k_of(T2[group]["Single Cam PC. w/ Tactile Points"][3])
        pc_m, tac_m = k_of(T2[group]["Multi Cam PC. Only"][3]), k_of(T2[group]["Multi Cam PC. w/ Tactile Points"][3])
        bs = (rng.binomial(N, tac_s / N, 10000) - rng.binomial(N, pc_s / N, 10000)
              - (rng.binomial(N, tac_m / N, 10000) - rng.binomial(N, pc_m / N, 10000))) / N
        did[group] = {"diff_in_gain_single_minus_multi": round((tac_s - pc_s - (tac_m - pc_m)) / N, 3),
                      "boot95": [round(float(np.quantile(bs, 0.025)), 3), round(float(np.quantile(bs, 0.975)), 3)]}
    out["table2_gain"] = gain
    out["table2_diff_in_gain"] = did

    # 6) Table 3: 펙 삽입과 조명 회수
    t3 = {}
    peg_ours = k_of(T3["Peg"][OURS][1])
    for name, vals in T3["Peg"].items():
        if name == OURS:
            continue
        t3[f"Peg Insertion | Ours vs {name}"] = {
            "k_ours": peg_ours, "k_base": k_of(vals[1]), "p_fisher": round(fisher(peg_ours, k_of(vals[1])), 4)}
    t3["Peg Insertion | Ours wilson95"] = [round(x, 3) for x in wilson(peg_ours)]
    out["table3"] = t3

    # 7) 검정력: 이 표 규모(n=20)로 찾을 수 있는 차이, 그리고 필요한 시도 수
    power = {}
    for p1, p2 in [(0.85, 0.50), (0.85, 0.70), (0.85, 0.65), (0.90, 0.65), (0.65, 0.45), (0.95, 0.85)]:
        power[f"{p1} vs {p2}"] = sample_size_two_prop(p1, p2)
    out["sample_size_per_arm_80pct_power"] = power

    # 8) 촉각 센서 균일 하중 실험 면적 점검 (부록 A.2.1)
    load_area = 176.7  # mm^2, 원형 접촉면 (논문 값)
    unit_area = 3.0    # mm^2 per sensing point (논문 본문)
    block_area = 4 * unit_area  # 2x2 블록
    out["sensor_load_check"] = {
        "load_area_mm2": load_area,
        "block_area_mm2_2x2": block_area,
        "blocks_covered_by_load": round(load_area / block_area, 1),
        "implied_circle_diameter_mm": round(2 * math.sqrt(load_area / math.pi), 1),
    }

    # 9) 논문 본문 수치 대조 (반올림 확인)
    out["text_checks"] = {
        "tactile_units_total": 4 * 256,
        "ntac_bimanual": 256 * 4,
        "ntac_single": 256 * 2,
        "sensor_pads_per_bimanual_system": 4,
    }

    path = os.path.join(HERE, "recompute_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # 요약 출력
    print("Whole Task: Ours vs baselines (Fisher p, Holm p)")
    for t in tests:
        print(f"  {t['task']:9s} vs {t['baseline']:22s} diff={t['diff']:+.2f} p={t['p_fisher']:.4f} holm={t['p_holm']:.4f}")
    print("pooled 80 episodes:", pooled)
    print("pooled Fisher ours vs PC w/ TacImg:", out["pooled_fisher_ours_vs_PCTac"],
          " vs PC only:", out["pooled_fisher_ours_vs_PC"], " vs RGB:", out["pooled_fisher_ours_vs_RGB"])
    print("violations:", viol)
    print("gain:", gain)
    print("diff in gain:", did)
    print("table3:", t3)
    print("sample size:", power)
    print("sensor:", out["sensor_load_check"])


if __name__ == "__main__":
    main()
