"""SimplerEnv (arXiv:2405.05941 v1, CoRL 2024 / PMLR 270) 논문 표 수치를 다시 계산한다.

입력: 논문 Table I, IV, V, VI, VII, XIV의 값을 아래에 옮겨 적었다(출처는 주석). 외부 데이터는 쓰지 않는다.
      공식 저장소의 metrics.py 값(REAL_PERF, SIMPLER_PERF)은 repo/ 를 읽어 옮겼다.
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)

가정(논문에 없는 것은 표시한다)
- 실로봇 비율의 분모는 부록 B의 시도 수다. Google: Pick 25x3=75, Move Near 60, Open 27, Close 27, Apple 27.
  WidowX: 과제당 24 (부록 B), 단 Eggplant는 표 V 값이 1/30 격자에만 맞아 30을 가정한다(D 항목).
- 시뮬 분모는 부록 B와 공식 스크립트에서 얻는다: Google Pick 25x3x4(arm 색 4개)=300, Move Near 60x4=240,
  Open/Close/Apple 27x4=108. WidowX 24 에피소드 x 3 시드 = 72.
- 몬테카를로는 실로봇 쪽만 이항 재표본하고 시뮬 값은 고정한다(시뮬 원자료는 없다).
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
RES = {}
rng = np.random.default_rng(20241105)


def mmrv(sim, real, strict_repo=False):
    """논문 식 (1)-(2). strict_repo=True면 공식 metrics.py의 '>' 비교를 쓴다."""
    sim, real = np.asarray(sim, float), np.asarray(real, float)
    n = len(sim)
    tot = 0.0
    for i in range(n):
        best = 0.0
        for j in range(n):
            if strict_repo:
                bad = (sim[i] > sim[j]) != (real[i] > real[j])
            else:
                bad = (sim[i] < sim[j]) != (real[i] < real[j])
            if bad:
                best = max(best, abs(real[i] - real[j]))
        tot += best
    return tot / n


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.corrcoef(x, y)[0, 1])


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def grid_ok(x, n, tol=0.0006):
    """x가 k/n의 반올림인지(소수 셋째 자리 기준)."""
    k = round(x * n)
    return abs(x - k / n) <= tol + 1e-9, k


# ---------------- Table IV (Google Robot) ----------------
POL = ["RT-1 (Converged)", "RT-1 (15%)", "RT-1-X", "RT-2-X", "Octo-Base", "RT-1 (Begin)"]
COLS = ["Pick H", "Pick V", "Pick S", "Pick Avg", "Move Near", "Open", "Close", "Open/Close Avg", "Apple Avg"]
N_REAL = [25, 25, 25, 75, 60, 27, 27, 54, 27]
N_SIM = [100, 100, 100, 300, 240, 108, 108, 216, 108]  # 시뮬 = 실로봇 시도 x 4 (arm 색)
REAL = np.array([  # Table IV, Real Eval
    [0.960, 0.880, 0.720, 0.853, 0.633, 0.815, 0.926, 0.870, 0.185],
    [1.000, 0.960, 0.800, 0.920, 0.583, 0.704, 0.889, 0.796, 0.185],
    [0.880, 0.560, 0.840, 0.760, 0.450, 0.519, 0.741, 0.630, 0.407],
    [0.920, 0.800, 1.000, 0.907, 0.733, 0.333, 0.630, 0.481, 0.074],
    [0.440, 0.200, 0.240, 0.293, 0.350, 0.148, 0.519, 0.333, 0.000],
    [0.200, 0.000, 0.200, 0.133, 0.017, 0.000, 0.000, 0.0002, 0.000],  # 0.0002 은 표 오타로 보인다(D 항목)
])
SIM_VM = np.array([  # Table IV, SIMPLER Visual Matching
    [0.960, 0.900, 0.710, 0.857, 0.442, 0.601, 0.861, 0.730, 0.065],
    [0.860, 0.790, 0.480, 0.710, 0.354, 0.463, 0.667, 0.565, 0.130],
    [0.820, 0.330, 0.550, 0.567, 0.317, 0.296, 0.891, 0.597, 0.213],
    [0.740, 0.740, 0.880, 0.787, 0.779, 0.157, 0.343, 0.250, 0.037],
    [0.210, 0.210, 0.090, 0.170, 0.042, 0.009, 0.444, 0.227, 0.000],
    [0.050, 0.000, 0.030, 0.027, 0.050, 0.000, 0.278, 0.139, 0.000],
])
SIM_VA = np.array([  # Table IV, SIMPLER Variant Aggregation
    [0.969, 0.760, 0.964, 0.898, 0.500, 0.270, 0.376, 0.323, 0.026],
    [0.920, 0.704, 0.813, 0.813, 0.446, 0.212, 0.323, 0.267, 0.021],
    [0.569, 0.204, 0.698, 0.490, 0.323, 0.069, 0.519, 0.294, 0.101],
    [0.822, 0.754, 0.893, 0.823, 0.792, 0.333, 0.372, 0.353, 0.206],
    [0.005, 0.000, 0.013, 0.006, 0.031, 0.000, 0.021, 0.011, 0.000],
    [0.022, 0.013, 0.031, 0.022, 0.040, 0.005, 0.132, 0.069, 0.000],
])
PRINT_MMRV_VM = [0.027, 0.027, 0.053, 0.031, 0.111, 0.000, 0.123, 0.055, 0.000]
PRINT_R_VM = [0.981, 0.964, 0.942, 0.976, 0.855, 0.983, 0.768, 0.915, 0.969]
PRINT_MMRV_VA = [0.093, 0.133, 0.140, 0.084, 0.111, 0.303, 0.3213, 0.321, 0.148]
PRINT_R_VA = [0.947, 0.937, 0.933, 0.960, 0.887, 0.629, 0.613, 0.737, 0.235]

# ---------------- Table V (WidowX + Bridge): 열 = (grasp, success) x 4 과제 ----------------
WPOL = ["RT-1-X", "Octo-Base", "Octo-Small"]
WCOLS = ["Spoon grasp", "Spoon succ", "Carrot grasp", "Carrot succ", "Stack grasp", "Stack succ", "Eggplant grasp", "Eggplant succ"]
W_REAL = np.array([
    [0.042, 0.000, 0.167, 0.000, 0.000, 0.000, 0.033, 0.000],
    [0.500, 0.333, 0.500, 0.250, 0.292, 0.000, 0.400, 0.233],
    [0.542, 0.417, 0.208, 0.083, 0.583, 0.125, 0.700, 0.433],
])
W_SIM = np.array([
    [0.167, 0.000, 0.208, 0.042, 0.083, 0.000, 0.000, 0.000],
    [0.347, 0.125, 0.528, 0.083, 0.319, 0.000, 0.667, 0.431],
    [0.778, 0.472, 0.278, 0.097, 0.403, 0.042, 0.875, 0.569],
])
W_PRINT_MMRV = [0, 0, 0, 0.111, 0, 0, 0, 0]
W_PRINT_R = [0.778, 0.827, 0.995, 0.575, 0.964, 1.000, 0.995, 0.990]
W_N_REAL = [24, 24, 24, 24, 24, 24, 30, 30]
W_N_SIM = [72] * 8

# ---------------- 공식 저장소 metrics.py 값 (repo/simpler_env/utils/metrics.py) ----------------
REPO_REAL = {  # 과제 평균 (Google: Pick, MN, Open, Close, Apple)
    "pick": [0.853, 0.920, 0.760, 0.907, 0.633, 0.583, 0.450, 0.733, 0.133, 0.017, 0.293, 0.350],
}
REPO_REAL_WIDOWX_EGG = {"rt-1-x": 0.000, "octo-base": 0.250, "octo-small": 0.400}
PAPER_REAL_WIDOWX_EGG = {"rt-1-x": 0.000, "octo-base": 0.233, "octo-small": 0.433}
REPO_SIM_WIDOWX_EGG = {"rt-1-x": 0.000, "octo-base": 0.431, "octo-small": 0.569}


TABLE_IV = {}


def sec_table_iv():
    global TABLE_IV
    out = TABLE_IV = {"grid_real": [], "grid_sim": [], "mmrv_check": [], "pearson_check": [], "va_check": []}
    for c, name in enumerate(COLS):
        bad_real = []
        for p in range(6):
            ok, k = grid_ok(REAL[p, c], N_REAL[c])
            if not ok:
                bad_real.append((POL[p], REAL[p, c], k))
        out["grid_real"].append({"col": name, "n": N_REAL[c], "off_grid": bad_real})
        bad_sim = []
        for p in range(6):
            ok, k = grid_ok(SIM_VM[p, c], N_SIM[c], tol=0.0006)
            if not ok:
                bad_sim.append((POL[p], SIM_VM[p, c], k))
        out["grid_sim"].append({"col": name, "n": N_SIM[c], "off_grid_vm": bad_sim})
    for c, name in enumerate(COLS):
        m = mmrv(SIM_VM[:, c], REAL[:, c])
        ms = mmrv(SIM_VM[:, c], REAL[:, c], strict_repo=True)
        r = pearson(SIM_VM[:, c], REAL[:, c])
        out["mmrv_check"].append({"col": name, "mmrv_paper_rule": round(m, 4), "mmrv_repo_rule": round(ms, 4), "printed": PRINT_MMRV_VM[c]})
        out["pearson_check"].append({"col": name, "r": round(r, 4), "printed": PRINT_R_VM[c]})
        mva = mmrv(SIM_VA[:, c], REAL[:, c])
        rva = pearson(SIM_VA[:, c], REAL[:, c])
        out["va_check"].append({"col": name, "mmrv_va": round(mva, 4), "printed_va": PRINT_MMRV_VA[c], "r_va": round(rva, 4), "printed_r_va": PRINT_R_VA[c]})
    # Table I 의 열 평균 재계산: Pick = Pick Avg 열, Move Near, Drawer = mean(Open/Close Avg, Apple)
    m_vm = [PRINT_MMRV_VM[3], PRINT_MMRV_VM[4], (PRINT_MMRV_VM[7] + PRINT_MMRV_VM[8]) / 2]
    RES["table_I_vismatch_mmrv_avg"] = round(sum(m_vm) / 3, 4)  # 표 I 0.056
    RES["table_I_drawer_mmrv_from_IV"] = round((PRINT_MMRV_VM[7] + PRINT_MMRV_VM[8]) / 2, 4)  # 표 I 0.027
    # 실로봇 Open/Close Avg 는 (Open+Close)/2 인지 확인
    diffs = [round(REAL[p, 7] - (REAL[p, 5] + REAL[p, 6]) / 2, 4) for p in range(6)]
    RES["oc_avg_check_real"] = diffs
    sim_diffs = [round(SIM_VM[p, 7] - (SIM_VM[p, 5] + SIM_VM[p, 6]) / 2, 4) for p in range(6)]
    RES["oc_avg_check_sim"] = sim_diffs


def sec_real_uncertainty():
    """실로봇 Pick Avg (n=75), Move Near (n=60): Wilson 95% 구간, 정책 간 Fisher 검정."""
    rows = []
    for p in range(6):
        k = round(REAL[p, 3] * 75)
        lo, hi = wilson(k, 75)
        rows.append({"policy": POL[p], "k_of_75": k, "rate": round(REAL[p, 3], 3), "wilson95": [round(lo, 3), round(hi, 3)]})
    RES["pick_avg_real_wilson"] = rows
    pairs = [(0, 1), (0, 3), (1, 2), (1, 3), (2, 0), (3, 0), (0, 4)]
    fis = []
    for a, b in pairs:
        ka, kb = round(REAL[a, 3] * 75), round(REAL[b, 3] * 75)
        p_val = stats.fisher_exact([[ka, 75 - ka], [kb, 75 - kb]])[1]
        fis.append({"pair": f"{POL[a]} vs {POL[b]}", "k": [ka, kb], "p_fisher": round(float(p_val), 3)})
    RES["pick_avg_real_fisher"] = fis
    # 순위 위반 크기와 표본 잡음 비교: 실로봇 두 정책 차이의 표준오차
    viol = []
    for a in range(6):
        for b in range(a + 1, 6):
            pa, pb = REAL[a, 3], REAL[b, 3]
            se = math.sqrt(pa * (1 - pa) / 75 + pb * (1 - pb) / 75)
            viol.append({"pair": f"{POL[a]} / {POL[b]}", "real_diff": round(abs(pa - pb), 3), "se_diff": round(se, 3),
                         "ratio": round(abs(pa - pb) / se, 2)})
    RES["pick_avg_pair_ratio"] = viol


def sec_mc_mmrv(n_draw=20000):
    """실로봇 쪽만 이항 재표본: 각 열의 MMRV 분포(시뮬 고정)."""
    out = []
    for c in [3, 4, 7, 8]:
        n = N_REAL[c]
        ks = np.round(REAL[:, c] * n).astype(int)
        vals = []
        for _ in range(n_draw):
            rr = rng.binomial(n, ks / n) / n
            vals.append(mmrv(SIM_VM[:, c], rr))
        vals = np.array(vals)
        out.append({"col": COLS[c], "mmrv_point": round(mmrv(SIM_VM[:, c], REAL[:, c]), 3),
                    "mc_median": round(float(np.median(vals)), 3),
                    "mc_95": [round(float(np.quantile(vals, 0.025)), 3), round(float(np.quantile(vals, 0.975)), 3)]})
    RES["mc_mmrv_real_noise"] = out


def sec_pearson_ci():
    """N=6 Pearson r의 Fisher z 95% 구간 (표본 6은 구간이 넓다)."""
    res = []
    for c in [3, 4, 7, 8]:
        r = pearson(SIM_VM[:, c], REAL[:, c])
        z = math.atanh(min(r, 0.99999))
        se = 1 / math.sqrt(6 - 3)
        res.append({"col": COLS[c], "r": round(r, 3), "fisher95": [round(math.tanh(z - 1.96 * se), 3), round(math.tanh(z + 1.96 * se), 3)]})
    r = pearson(W_SIM[:, 6], W_REAL[:, 6])
    RES["pearson_ci_google"] = res
    RES["widowx_n3_note"] = "N=3 정책이면 Fisher 구간이 [-1, 1]에 가깝다"


def sec_widowx():
    mm = [round(mmrv(W_SIM[:, c], W_REAL[:, c]), 3) for c in range(8)]
    pr = [round(pearson(W_SIM[:, c], W_REAL[:, c]), 3) for c in range(8)]
    RES["widowx_mmrv"] = {"calc": mm, "printed": W_PRINT_MMRV}
    RES["widowx_pearson"] = {"calc": pr, "printed": W_PRINT_R}
    grid = []
    for c, name in enumerate(WCOLS):
        for p, pol in enumerate(WPOL):
            ok, k = grid_ok(W_REAL[p, c], W_N_REAL[c])
            if not ok:
                grid.append(f"real {name} {pol} {W_REAL[p, c]} not k/{W_N_REAL[c]}")
    RES["widowx_real_grid_offs"] = grid
    # 에그플랜트 실로봇 값: 24 가정 대 30 가정
    egg24 = [(pol, W_REAL[p, 7], round(W_REAL[p, 7] * 24, 2)) for p, pol in enumerate(WPOL)]
    egg30 = [(pol, W_REAL[p, 7], round(W_REAL[p, 7] * 30, 2)) for p, pol in enumerate(WPOL)]
    RES["egg_success_times_24"] = egg24
    RES["egg_success_times_30"] = egg30
    # 공식 repo REAL_PERF 가 쓰는 에그플랜트 값으로 MMRV/Pearson 을 다시 계산
    rr_repo = [REPO_REAL_WIDOWX_EGG[p.lower()] for p in WPOL]
    rr_paper = [PAPER_REAL_WIDOWX_EGG[p.lower()] for p in WPOL]
    ss = [REPO_SIM_WIDOWX_EGG[p.lower()] for p in WPOL]
    RES["egg_success_repo_vs_paper"] = {
        "repo_real": rr_repo, "paper_real": rr_paper, "sim": ss,
        "mmrv_repo_real": round(mmrv(ss, rr_repo), 3), "mmrv_paper_real": round(mmrv(ss, rr_paper), 3),
        "r_repo_real": round(pearson(ss, rr_repo), 3), "r_paper_real": round(pearson(ss, rr_paper), 3),
    }
    # 표 V 성공률 열 MMRV 의 실로봇 쪽 잡음: 과제 24 가정 이항 재표본
    ks = np.round(W_REAL[:, 3] * 24).astype(int)
    vals = [mmrv(W_SIM[:, 3], rng.binomial(24, ks / 24) / 24) for _ in range(20000)]
    RES["widowx_carrot_succ_mc"] = {"mc_95": [round(float(np.quantile(vals, 0.025)), 3), round(float(np.quantile(vals, 0.975)), 3)]}


def sec_widowx_pairs():
    """WidowX 성공률 열: 정책 쌍 Fisher 검정 (실로봇 시도 24, 에그플랜트 30 가정)."""
    out = []
    for c in [1, 3, 5, 7]:
        n = W_N_REAL[c]
        for a, b in [(1, 2)]:
            ka, kb = round(W_REAL[a, c] * n), round(W_REAL[b, c] * n)
            pv = stats.fisher_exact([[ka, n - ka], [kb, n - kb]])[1]
            out.append({"col": WCOLS[c], "pair": f"{WPOL[a]} vs {WPOL[b]}", "k": [ka, kb], "n": n, "p_fisher": round(float(pv), 3)})
    RES["widowx_pairs_octo"] = out


def sec_repo_vs_paper_google():
    """공식 repo REAL_PERF / SIMPLER_PERF 의 Google 과제 평균 값을 논문 표 IV 의 Avg 열과 대조."""
    keys = ["rt-1-converged", "rt-1-15pct", "rt-1-x", "rt-2-x", "octo-base", "rt-1-begin"]
    repo_real_pick = [0.853, 0.920, 0.760, 0.907, 0.293, 0.133]
    paper_real_pick = [REAL[p, 3] for p in [0, 1, 2, 3, 4, 5]]
    repo_sim_pick = [0.857, 0.710, 0.567, 0.787, 0.170, 0.027]
    paper_sim_pick = [SIM_VM[p, 3] for p in [0, 1, 2, 3, 4, 5]]
    RES["repo_vs_paper_pick_avg"] = {
        "real_equal": all(abs(a - b) < 1e-9 for a, b in zip(repo_real_pick, paper_real_pick)),
        "sim_equal": all(abs(a - b) < 1e-9 for a, b in zip(repo_sim_pick, paper_sim_pick)),
    }


def sec_table_vi_vii():
    """표 VII(시뮬 원자료)에서 표 VI의 시뮬 |Δ|를 다시 만든다. RT-1 w/o Aug, Pick과 Move Near."""
    base = {"Pick": 0.920, "MN": 0.467}
    var = {
        "Background": [(0.933, 0.907), (0.533, 0.567)],
        "Lighting": [(0.960, 0.960), (0.483, 0.600)],
        "Distractors": [(0.880, 0.901), (None, 0.600)],  # MN 은 변형 하나만 (base 가 이미 distractor)
        "Table Texture": [(0.867, 0.747), (0.550, 0.200)],
        "Camera Pose": [(0.053, 0.280), (0.117, 0.433)],
    }
    printed_pick = {"Background": 0.013, "Lighting": 0.040, "Distractors": 0.027, "Table Texture": 0.113, "Camera Pose": 0.753}
    printed_mn = {"Background": 0.083, "Lighting": 0.075, "Distractors": 0.133, "Table Texture": 0.175, "Camera Pose": 0.192}
    rows = []
    for f in var:
        dp = [abs(v - base["Pick"]) for v in var[f][0] if v is not None]
        dm = [abs(v - base["MN"]) for v in var[f][1] if v is not None]
        rows.append({"factor": f, "pick_calc": round(float(np.mean(dp)), 3), "pick_printed": printed_pick[f],
                     "mn_calc": round(float(np.mean(dm)), 3), "mn_printed": printed_mn[f]})
    RES["table_vii_to_vi"] = rows
    # Table VI 의 MMRV/r 을 5 요인에 대해 다시 계산 (Avg 블록만 재현 가능: 실로봇 과제별 |Δ|는 논문에 없다): 시뮬 |Δ| (Pick, Avg) 대 실로봇 |Δ| (Real TableTop 열)
    real_wo = [0.028, 0.083, 0.111, 0.389, 0.458]
    pick_wo = [0.013, 0.040, 0.027, 0.113, 0.753]
    avg_wo = [0.048, 0.057, 0.080, 0.144, 0.473]
    real_w = [0.167, 0.042, 0.083, 0.167, 0.375]
    pick_w = [0.153, 0.033, 0.033, 0.220, 0.613]
    avg_w = [0.123, 0.075, 0.059, 0.189, 0.394]
    RES["table_vi_recalc"] = {
        "wo_aug_avg_mmrv": round(mmrv(avg_wo, real_wo), 3), "wo_aug_avg_r": round(pearson(avg_wo, real_wo), 3),
        "wo_aug_pick_mmrv": round(mmrv(pick_wo, real_wo), 3), "wo_aug_pick_r": round(pearson(pick_wo, real_wo), 3),
        "w_aug_avg_mmrv": round(mmrv(avg_w, real_w), 3), "w_aug_avg_r": round(pearson(avg_w, real_w), 3),
        "w_aug_pick_mmrv": round(mmrv(pick_w, real_w), 3), "w_aug_pick_r": round(pearson(pick_w, real_w), 3),
        "printed": {"wo_aug_avg": [0.000, 0.831], "wo_aug_pick": [0.000, 0.779], "w_aug_avg": [0.041, 0.970], "w_aug_pick": [0.041, 0.984]},
    }
    # 본문 4% 대 25% (실로봇 테이블 질감)와 표 VI 실로봇 열 0.389 의 관계
    RES["table_texture_text_vs_vi"] = {"text_solid_pct": 4, "text_patterned_pct": 25, "text_mean": 14.5, "table_vi_real_wo_aug": 38.9}
    # 본문 시뮬 2%(단색)와 24%(패턴): 표 VII 의 RT-1 w/o Aug 테이블 질감
    solid = [(0.867 - 0.920), (0.550 - 0.467)]
    patt = [(0.747 - 0.920), (0.200 - 0.467)]
    RES["sim_table_texture_from_VII"] = {"solid_mean_pp": round(100 * float(np.mean(solid)), 1), "patterned_mean_pp": round(100 * float(np.mean(patt)), 1)}


def sec_kruskal():
    """표 XIV(Google, Visual Matching) 재현: 이진 시행 결과에 대한 Kruskal-Wallis, 정책마다 p<0.05 개수."""
    printed = [0, 0, 2, 3, 3, 1, 2, 2, 0]
    calc = []
    for c in range(9):
        cnt = 0
        for p in range(6):
            k_r = round(REAL[p, c] * N_REAL[c])
            k_s = round(SIM_VM[p, c] * N_REAL[c])  # 공식 calc_metrics_evaluation_videos.py: 시뮬도 실로봇과 같은 n으로 반올림
            a = np.array([1] * k_r + [0] * (N_REAL[c] - k_r), float)
            b = np.array([1] * k_s + [0] * (N_REAL[c] - k_s), float)
            if np.all(a == a[0]) and np.all(b == b[0]) and a[0] == b[0]:
                continue
            try:
                pval = stats.kruskal(a, b).pvalue
            except ValueError:
                pval = 1.0
            if pval < 0.05:
                cnt += 1
        calc.append(cnt)
    RES["kruskal_google_vm"] = {"calc": calc, "printed": printed, "note": "공식 저장소와 같이 시뮬 n = 실로봇 n (정책별 반올림), 이진 시행"}


def sec_repo_ties():
    """공식 코드(>)와 논문 식(<)의 차이는 동률에서만 생긴다. 표 IV 열에서 동률 쌍 개수."""
    ties = 0
    for c in range(9):
        for a in range(6):
            for b in range(a + 1, 6):
                if SIM_VM[a, c] == SIM_VM[b, c] or REAL[a, c] == REAL[b, c]:
                    ties += 1
    RES["tie_pairs_table_iv_vm"] = ties


def sec_trials():
    g_real = 75 + 60 + 27 + 27 + 27  # Pick, MN, Open, Close, Apple per policy (Apple 은 별도 과제)
    RES["real_trials"] = {"google_per_policy": g_real, "google_6_policies": 6 * g_real,
                          "widowx_per_policy_24egg": 24 * 3 + 24, "widowx_per_policy_30egg": 24 * 3 + 30,
                          "widowx_total_3_policies_30egg": 3 * (24 * 3 + 30)}


if __name__ == "__main__":
    sec_table_iv(); RES["table_iv"] = TABLE_IV; sec_real_uncertainty(); sec_mc_mmrv(); sec_pearson_ci(); sec_widowx(); sec_widowx_pairs()
    sec_repo_vs_paper_google(); sec_table_vi_vii(); sec_kruskal(); sec_repo_ties(); sec_trials()
    OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    for k, v in RES.items():
        print(k, json.dumps(v, ensure_ascii=False, default=str)[:600])
