"""H2RBench (arXiv:2609.24778 v3) 논문 표와 그림의 수치를 다시 계산한다.

입력: 논문 표의 값을 아래에 그대로 옮겨 적었다(출처는 주석). 그림에서 읽은 점은 digitized_points.json에 있다
      (digitize_figures.py가 만든다). 실세계 시도 횟수는 논문에 없다.
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)

가정(논문에 없는 것은 표시한다)
- 한 번의 평가는 30회 롤아웃이다(Table 8의 SR 값이 k/30 꼴이고 README 예시가 --rollouts 30이다). "90회, 3 시드"는 30 x 3이다.
- Table 3의 ± 는 3개 시드 사이의 표준편차이고 표본(n-1)인지 모집단(n)인지는 밝히지 않았다. 둘 다 시험한다.
- Fig. 5와 Fig. 15의 "Success Rate" 축은 실제로는 진행도(Prog.)이고, 시뮬 값은 Table 7의 Prog.@MaxH(단일 값)와 일치한다.
- 그림에서 읽은 값의 오차는 벡터 그림이 약 0.1, 래스터 그림(Fig. 15a)이 약 0.5 퍼센트포인트다.
"""
import itertools
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
OUT = HERE / "recompute_results.json"
RES = {}

METHODS = ["Phantom", "AMPLIFY", "Point Policy", "GHOST"]
TASKS = ["Pick-and-Place", "Stacking", "Insertion", "Sequential"]
STAGES = {"Pick-and-Place": 3, "Stacking": 4, "Insertion": 4, "Sequential": 3}  # Table 6
NROLL = 30  # 평가 1회의 롤아웃 수(가정)

# Table 3 = Table 9: (Prog 평균, Prog SD, SR 평균, SR SD), 3 시드
T3 = {
    "Phantom": {"Pick-and-Place": (77.0, 1.4, 56.7, 2.7), "Stacking": (75.6, 2.7, 25.6, 6.8),
                "Insertion": (39.2, 3.6, 7.8, 4.2), "Sequential": (37.8, 2.4, 13.3, 4.7)},
    "AMPLIFY": {"Pick-and-Place": (74.4, 2.7, 58.9, 4.2), "Stacking": (85.3, 3.7, 61.1, 9.6),
                "Insertion": (35.0, 6.2, 2.2, 1.6), "Sequential": (8.1, 3.4, 0.0, 0.0)},
    "Point Policy": {"Pick-and-Place": (82.6, 7.4, 62.2, 11.7), "Stacking": (79.7, 7.7, 61.1, 17.1),
                     "Insertion": (37.8, 6.5, 6.7, 11.5), "Sequential": (17.4, 5.6, 0.0, 0.0)},
    "GHOST": {"Pick-and-Place": (91.1, 4.9, 82.2, 6.9), "Stacking": (88.9, 5.9, 76.7, 8.8),
              "Insertion": (64.7, 1.3, 34.5, 6.9), "Sequential": (63.3, 8.4, 38.9, 15.0)},
}
# Table 7: Prog.@MaxH, Best Prog., # Human (최고값의 사람 데모 수)
T7 = {
    "Phantom": {"Pick-and-Place": (76.7, 76.7, 100), "Stacking": (79.2, 79.2, 300),
                "Insertion": (40.8, 46.7, 100), "Sequential": (36.7, 36.7, 300)},
    "Point Policy": {"Pick-and-Place": (88.9, 88.9, 100), "Stacking": (88.3, 93.3, 100),
                     "Insertion": (45.0, 45.0, 300), "Sequential": (16.7, 16.7, 300)},
    "AMPLIFY": {"Pick-and-Place": (71.1, 72.2, 40), "Stacking": (94.2, 99.2, 200),
                "Insertion": (41.7, 41.7, 100), "Sequential": (12.2, 15.6, 100)},
    "GHOST": {"Pick-and-Place": (96.7, 96.7, 100), "Stacking": (90.0, 95.0, 200),
              "Insertion": (65.0, 65.8, 100), "Sequential": (71.1, 71.1, 300)},
}
# Fig. 3(B) 숫자 라벨(로그 오즈 이득), 열 순서는 TASKS
FIG3B = {"Phantom": [0.36, 0.73, -0.28, 0.53], "AMPLIFY": [0.29, 0.94, -0.53, 0.34],
         "Point Policy": [0.13, -0.37, -0.49, -0.10], "GHOST": [0.17, -0.05, 0.23, -0.49]}
# Table 8 (Stacking, 분포 밖 배치): Prog, SR1..SR4
T8 = {"Phantom": (48.3, [100.0, 73.3, 20.0, 0.0]), "AMPLIFY": (41.7, [100.0, 60.0, 6.7, 0.0]),
      "Point Policy": (78.3, [100.0, 100.0, 66.7, 46.7]), "GHOST": (58.3, [100.0, 86.7, 30.0, 16.7])}
# Table 10: 추가 과제, Diffusion Policy(로봇 100회)와 XSkill
T10 = {"Diffusion Policy": {"Stacking": (95.0, 80.0), "Insertion": (41.7, 0.0), "Sequential": (82.2, 63.3)},
       "XSkill": {"Stacking": (28.3, 0.0), "Insertion": (10.0, 0.0), "Sequential": (10.0, 0.0)}}
# Table 11: AMPLIFY 잠재 공간의 W2 거리 (낮음/중간/높음 사람 데모)
T11 = {"Stacking": [10.91, 9.62, 9.05], "Pick-and-Place": [6.85, 7.27, 8.17],
       "Insertion": [6.08, 10.01, 10.97], "Sequential": [8.91, 8.81, 8.84]}
# Table 12-14 보고값
REP = {"max": dict(r=0.894, rho=0.851, mmrv=0.060), "robot": dict(r=0.971, rho=0.940, mmrv=0.014)}
T13 = {"robot": {"Pick-and-Place": 0.000, "Stacking": 0.042, "Insertion": 0.000, "Sequential": 0.015, "mean": 0.014},
       "max": {"Pick-and-Place": 0.000, "Stacking": 0.171, "Insertion": 0.037, "Sequential": 0.033, "mean": 0.060}}
T14 = {("robot", "Full"): (16, 0.971, 0.940, 0.014), ("robot", "-AMPLIFY"): (12, 0.984, 0.970, 0.003),
       ("robot", "-Stacking"): (12, 0.971, 0.986, 0.005), ("max", "Full"): (16, 0.894, 0.851, 0.060),
       ("max", "-AMPLIFY"): (12, 0.924, 0.902, 0.007), ("max", "-Stacking"): (12, 0.887, 0.890, 0.023)}

DIG = json.loads((HERE / "digitized_points.json").read_text(encoding="utf-8"))
FIG4 = DIG["fig4_prog"]            # method -> task -> 4개 사람 데모 수준의 Prog.
FIG5 = DIG["fig5_max_human"]       # method -> task -> [sim, real]
FIG15A = DIG["fig15a_robot_only"]  # method -> task -> [sim, real]


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def logit(p):
    return math.log(p / (1 - p))


def fisher_exact(k1, n1, k2, n2):
    return stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]


# ---------------------------------------------------------------- K1 표 3과 표 9가 같은가
def k1_table3_equals_table9():
    p = HERE.parent / "paper" / "paper.txt"
    if not p.exists():
        return {"skipped": "paper/paper.txt 없음"}
    lines = p.read_text(encoding="utf-8").splitlines()
    rows = {}
    for ln in lines:
        for m in METHODS:
            if ln.strip().startswith(m) and "±" in ln:
                nums = [float(x) for x in re.findall(r"\d+\.\d", ln)]
                rows.setdefault(m, []).append(nums)
    same = {m: (len(v) == 2 and v[0] == v[1]) for m, v in rows.items()}
    return {"rows_found": {m: len(v) for m, v in rows.items()}, "identical": same}


# ---------------------------------------------------------------- K2 평균과 SD의 정수 실현 가능성
def triple_grid(smax):
    a = np.arange(smax + 1)
    j1, j2, j3 = np.meshgrid(a, a, a, indexing="ij")
    keep = (j1 <= j2) & (j2 <= j3)
    return j1[keep], j2[keep], j3[keep]


def k2_feasibility():
    """Prog는 시드당 j/(30*S), SR는 k/30 꼴이라는 가정에서 보고된 평균과 SD가 정수로 만들어지는지 본다."""
    out = {}
    bad_mean, bad_sd = [], []
    for m in METHODS:
        for t in TASKS:
            pm, ps, sm, ss = T3[m][t]
            S = STAGES[t]
            for name, mean, sd, den in [("Prog", pm, ps, 30 * S), ("SR", sm, ss, 30)]:
                j1, j2, j3 = triple_grid(den)
                rates = np.stack([j1, j2, j3]) / den * 100
                mu = rates.mean(0)
                ok_mean = np.round(mu, 1) == round(mean, 1)
                ok_mean |= np.abs(mu - mean) < 0.0501   # 반올림 경계 허용
                s1 = rates.std(0, ddof=1)
                s0 = rates.std(0, ddof=0)
                ok1 = ok_mean & (np.abs(s1 - sd) < 0.0501)
                ok0 = ok_mean & (np.abs(s0 - sd) < 0.0501)
                key = f"{m}|{t}|{name}"
                out[key] = dict(mean_ok=bool(ok_mean.any()), sd_sample_ok=bool(ok1.any()),
                                sd_pop_ok=bool(ok0.any()), sd_any_ok=bool(ok1.any() or ok0.any()))
                if not ok_mean.any():
                    bad_mean.append(key)
                elif not (ok1.any() or ok0.any()):
                    bad_sd.append(key)
    n = len(out)
    return dict(cells=n, mean_infeasible=bad_mean, sd_infeasible_given_mean=bad_sd,
                sd_sample_feasible=sum(v["sd_sample_ok"] for v in out.values()),
                sd_pop_feasible=sum(v["sd_pop_ok"] for v in out.values()), detail=out)


# ---------------------------------------------------------------- K3 Prog와 SR의 범위 관계
def k3_prog_sr_bounds():
    viol = []
    rows = {}
    for m in METHODS:
        for t in TASKS:
            pm, _, sm, _ = T3[m][t]
            S = STAGES[t]
            lo, hi = sm, sm + (100 - sm) * (S - 1) / S
            rows[f"{m}|{t}"] = dict(prog=pm, sr=sm, lower=round(lo, 1), upper=round(hi, 1))
            if not (lo - 0.05 <= pm <= hi + 0.05):
                viol.append(f"{m}|{t}")
    return dict(violations=viol, rows=rows)


# ---------------------------------------------------------------- K4 Fig. 4 끝점과 Table 3
def k4_fig4_vs_table3():
    diffs = {}
    mx = 0
    for m in METHODS:
        for t in TASKS:
            d = FIG4[m][t][3] - T3[m][t][0]
            diffs[f"{m}|{t}"] = round(d, 2)
            mx = max(mx, abs(d))
    return dict(max_abs_diff=round(mx, 2), diffs=diffs)


# ---------------------------------------------------------------- K5 로그 오즈 이득 다시 계산
def k5_delta_logit():
    rows = {}
    mx = 0
    flips = []
    for m in METHODS:
        for ti, t in enumerate(TASKS):
            p0, p1 = FIG4[m][t][0] / 100, FIG4[m][t][3] / 100
            d_prog = logit(p1) - logit(p0)
            sr = T3[m][t][2] / 100
            d_sr = None
            rows[f"{m}|{t}"] = dict(fig3b=FIG3B[m][ti], from_prog=round(d_prog, 2), robot_only_prog=FIG4[m][t][0],
                                    maxh_prog=FIG4[m][t][3], maxh_sr=T3[m][t][2])
            mx = max(mx, abs(d_prog - FIG3B[m][ti]))
    # 본문 정의는 성공률이지만, 같은 식을 최대 사람 데모 SR에 적용하면 로봇 전용 SR이 없으므로 계산할 수 없다.
    # 대신 SR=0인 칸(AMPLIFY 순차, Point Policy 순차)에서 SR 기반 Delta가 정의되지 않음을 보인다.
    undefined_if_sr = [f"{m}|{t}" for m in METHODS for t in TASKS if T3[m][t][2] == 0.0]
    return dict(max_abs_diff_vs_fig3b=round(mx, 3), rows=rows, undefined_if_defined_on_sr=undefined_if_sr)


# ---------------------------------------------------------------- K6 Table 7 대 Table 3, Fig. 5의 x
def k6_table7():
    rows = {}
    zs = []
    for m in METHODS:
        for t in TASKS:
            t7 = T7[m][t][0]
            t3, sd = T3[m][t][0], T3[m][t][1]
            z = (t7 - t3) / sd if sd > 0 else float("nan")
            zs.append(z)
            fx = FIG5[m][t][0]
            rows[f"{m}|{t}"] = dict(table7=t7, table3_mean=t3, seed_sd=sd, z=round(z, 2), fig5_x=fx,
                                    fig5_x_minus_table7=round(fx - t7, 2))
    zs = np.array(zs)
    best_vs_max = {f"{m}|{t}": round(T7[m][t][1] - T7[m][t][0], 1) for m in METHODS for t in TASKS}
    # Best Prog는 사람 데모 수준(3-5개)에서 최대를 고른 값이다.
    return dict(rows=rows, mean_z=round(float(np.nanmean(zs)), 2), sd_z=round(float(np.nanstd(zs, ddof=1)), 2),
                n_abs_z_gt_1=int(np.nansum(np.abs(zs) > 1)), n_abs_z_gt_2=int(np.nansum(np.abs(zs) > 2)),
                max_abs_fig5_x_minus_table7=max(abs(v["fig5_x_minus_table7"]) for v in rows.values()),
                best_minus_maxh=best_vs_max)


# ---------------------------------------------------------------- 상관과 MMRV
def mmrv(s, r):
    """식 (3): 과제 하나에서 s, r은 방법별 [0,1] 값."""
    s, r = np.asarray(s, float), np.asarray(r, float)
    n = len(s)
    tot = 0.0
    for i in range(n):
        best = 0.0
        for j in range(n):
            if j == i:
                continue
            if (s[i] < s[j]) != (r[i] < r[j]):
                best = max(best, abs(r[i] - r[j]))
        tot += best
    return tot / n


def pairs(points, subset=None):
    xs, ys, tag = [], [], []
    for m in METHODS:
        for t in TASKS:
            if subset == "-AMPLIFY" and m == "AMPLIFY":
                continue
            if subset == "-Stacking" and t == "Stacking":
                continue
            xs.append(points[m][t][0])
            ys.append(points[m][t][1])
            tag.append((m, t))
    return np.array(xs), np.array(ys), tag


def metrics(points, subset=None):
    x, y, tag = pairs(points, subset)
    r = stats.pearsonr(x, y)[0]
    rho = stats.spearmanr(x, y)[0]
    per_task = {}
    for t in TASKS:
        idx = [i for i, (m, tt) in enumerate(tag) if tt == t]
        if len(idx) >= 2:
            per_task[t] = mmrv(x[idx] / 100, y[idx] / 100)
    mm = float(np.mean(list(per_task.values())))
    return dict(n=len(x), r=float(r), rho=float(rho), mmrv=mm, per_task_mmrv=per_task)


def fisher_ci(r, n, z=1.959964):
    zr = math.atanh(r)
    se = 1 / math.sqrt(n - 3)
    return math.tanh(zr - z * se), math.tanh(zr + z * se)


def within_task(points):
    """과제 평균을 뺀 뒤의 상관(12 자유도)과 과제별 Kendall tau."""
    x, y, tag = pairs(points)
    xc, yc = x.copy(), y.copy()
    taus = {}
    for t in TASKS:
        idx = [i for i, (m, tt) in enumerate(tag) if tt == t]
        xc[idx] -= x[idx].mean()
        yc[idx] -= y[idx].mean()
        taus[t] = float(stats.kendalltau(x[idx], y[idx])[0])
    r = stats.pearsonr(xc, yc)[0]
    # 자유도: 16 - 4(과제 평균) - 0 = 12 (상관 추정에 쓰는 자유도는 n-2-3 = 11이 더 엄밀)
    df = len(x) - len(TASKS) - 1
    t_stat = r * math.sqrt(df / (1 - r * r))
    p = 2 * stats.t.sf(abs(t_stat), df)
    return dict(r_within=float(r), df=df, p=float(p), kendall_per_task=taus,
                kendall_mean=float(np.mean(list(taus.values()))),
                between_task_share_x=float(np.var([x[[i for i, (m, tt) in enumerate(tag) if tt == t]].mean() for t in TASKS]) /
                                           np.var(x)))


def k7_sim_vs_real():
    out = {}
    for name, pts in [("max", FIG5), ("robot", FIG15A)]:
        met = metrics(pts)
        x, y, _ = pairs(pts)
        lo, hi = fisher_ci(met["r"], met["n"])
        p_r = stats.pearsonr(x, y)[1]
        out[name] = dict(recomputed=met, reported=REP[name], r_ci95=[lo, hi], r_p=float(p_r),
                         within=within_task(pts))
        out[name]["r2"] = met["r"] ** 2
        # 부분 집합(표 14)
        sub = {}
        for sname in ["-AMPLIFY", "-Stacking"]:
            sub[sname] = metrics(pts, sname)
        out[name]["subsets"] = sub
    # 표 13 평균과 표 12의 일관성
    for k in ["robot", "max"]:
        vals = [T13[k][t] for t in TASKS]
        out[k + "_table13_mean_check"] = dict(mean=sum(vals) / 4, reported_mean=T13[k]["mean"])
    return out


def k8_robot_only_sim_vs_fig4():
    rows = {}
    for m in METHODS:
        for t in TASKS:
            a = FIG15A[m][t][0]
            b = FIG4[m][t][0]
            rows[f"{m}|{t}"] = dict(fig15a_sim=a, fig4_robot_only=b, diff=round(a - b, 1))
    d = np.array([v["diff"] for v in rows.values()])
    return dict(rows=rows, mean_abs=float(np.abs(d).mean()), max_abs=float(np.abs(d).max()),
                n_abs_gt_5=int((np.abs(d) > 5).sum()))


def k9_delta_prediction():
    """시뮬이 사람 데모의 효과(최대 사람 - 로봇 전용)를 예측하는가. 16쌍."""
    out = {}
    for label, simmax, simrob in [("fig5_vs_fig15a", FIG5, FIG15A)]:
        dx, dy = [], []
        for m in METHODS:
            for t in TASKS:
                dx.append(simmax[m][t][0] - simrob[m][t][0])
                dy.append(simmax[m][t][1] - simrob[m][t][1])
        dx, dy = np.array(dx), np.array(dy)
        r, p = stats.pearsonr(dx, dy)
        rho, prho = stats.spearmanr(dx, dy)
        sign_agree = int(np.sum(np.sign(dx) == np.sign(dy)))
        out[label] = dict(r=float(r), p=float(p), rho=float(rho), p_rho=float(prho),
                          sign_agree=sign_agree, n=len(dx), dx=dx.round(1).tolist(), dy=dy.round(1).tolist())
    # 시뮬 효과를 3-시드 평균(Fig. 4)으로 쓸 때
    dx, dy = [], []
    for m in METHODS:
        for t in TASKS:
            dx.append(FIG4[m][t][3] - FIG4[m][t][0])
            dy.append(FIG5[m][t][1] - FIG15A[m][t][1])
    r, p = stats.pearsonr(dx, dy)
    rho, prho = stats.spearmanr(dx, dy)
    out["fig4_3seed_vs_real"] = dict(r=float(r), p=float(p), rho=float(rho), p_rho=float(prho),
                                     sign_agree=int(np.sum(np.sign(dx) == np.sign(dy))), n=len(dx))
    return out


def k10_alt_sim_x():
    """시뮬 값으로 다른 정의를 쓰면 상관이 어떻게 달라지는가(실세계 y는 그림에서 읽은 값 그대로)."""
    out = {}
    x_t3 = np.array([T3[m][t][0] for m in METHODS for t in TASKS])
    x_t7 = np.array([FIG5[m][t][0] for m in METHODS for t in TASKS])
    y = np.array([FIG5[m][t][1] for m in METHODS for t in TASKS])
    sr = np.array([T3[m][t][2] for m in METHODS for t in TASKS])
    for name, x in [("table3_prog_mean", x_t3), ("table7_prog_single", x_t7), ("table3_sr_final", sr)]:
        out[name] = dict(r=float(stats.pearsonr(x, y)[0]), rho=float(stats.spearmanr(x, y)[0]))
    return out


def k11_monte_carlo(n_draw=4000, seed=0):
    """읽기 오차와 표본 잡음을 넣었을 때 r, rho, MMRV의 흔들림.
    실세계 y: 평가가 30회라고 가정한 이항 잡음(진행도를 비율로 취급한 근사). 시뮬 x: Table 3의 시드 SD / sqrt(1)(단일 시행)."""
    rng = np.random.default_rng(seed)
    base = {}
    for name, pts in [("max", FIG5), ("robot", FIG15A)]:
        rs, rhos, mms = [], [], []
        for _ in range(n_draw):
            pert = {}
            for m in METHODS:
                pert[m] = {}
                for t in TASKS:
                    sx, ry = pts[m][t]
                    sd_x = max(T3[m][t][1], 1.0)        # 시뮬 단일 시행의 시드 변동(표 3 SD, 하한 1)
                    p = min(max(ry / 100, 0.01), 0.99)
                    n_eff = NROLL * STAGES[t]
                    sd_y = 100 * math.sqrt(p * (1 - p) / n_eff)
                    pert[m][t] = [sx + rng.normal(0, sd_x), ry + rng.normal(0, sd_y)]
            mt = metrics(pert)
            rs.append(mt["r"]); rhos.append(mt["rho"]); mms.append(mt["mmrv"])
        q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 50)), float(np.percentile(a, 97.5))]
        base[name] = dict(r=q(rs), rho=q(rhos), mmrv=q(mms))
    return base


# ---------------------------------------------------------------- 사람 데모의 효과 대 시드 잡음
def k12_effect_vs_noise():
    rows = {}
    for m in METHODS:
        for t in TASKS:
            d = FIG4[m][t][3] - FIG4[m][t][0]
            sd = T3[m][t][1]                       # 최대 사람 데모 설정의 시드 SD(로봇 전용 SD는 표로 주어지지 않음)
            z = d / (sd * math.sqrt(2 / 3)) if sd > 0 else float("nan")  # 평균 차이의 표준오차(두 평균, SD 같다고 가정)
            rows[f"{m}|{t}"] = dict(delta_prog=round(d, 1), seed_sd=sd, z=round(z, 2))
    zs = np.array([v["z"] for v in rows.values()])
    return dict(rows=rows, n_abs_z_gt_2=int((np.abs(zs) > 2).sum()), n_total=len(zs))


def k13_welch():
    out = {}
    for t in TASKS:
        for metric, idx in [("Prog", 0), ("SR", 2)]:
            ranked = sorted(METHODS, key=lambda m: -T3[m][t][idx])
            a, b = ranked[0], ranked[1]
            m1, s1 = T3[a][t][idx], T3[a][t][idx + 1]
            m2, s2 = T3[b][t][idx], T3[b][t][idx + 1]
            res = stats.ttest_ind_from_stats(m1, s1, 3, m2, s2, 3, equal_var=False)
            out[f"{t}|{metric}"] = dict(best=a, second=b, diff=round(m1 - m2, 1), t=float(res.statistic),
                                        p=float(res.pvalue))
    return out


def k14_table8():
    out = {}
    for m, (prog, srs) in T8.items():
        calc = sum(srs) / 4
        out[m] = dict(prog=prog, mean_of_stage_sr=round(calc, 2), diff=round(prog - calc, 2),
                      counts_of_30=[round(s * NROLL / 100, 2) for s in srs])
    # 학습 분포 안(Fig. 4, 사람 100회) 대비 하락
    drop = {}
    for m in METHODS:
        i = FIG4[m]["Stacking"][1]
        drop[m] = dict(in_dist_prog_at_100=i, ood_prog=T8[m][0], drop=round(i - T8[m][0], 1))
    # Fisher: Point Policy 대 GHOST의 SR4와 SR3 (n=30)
    f4 = fisher_exact(round(46.7 * .3), 30, round(16.7 * .3), 30)
    f3 = fisher_exact(round(66.7 * .3), 30, round(30.0 * .3), 30)
    # Prog 평균은 단계별 SR의 평균이므로 Prog 차이를 4 x 30 = 120 시행의 비율 차이로 근사 검정
    return dict(rows=out, drop_vs_in_distribution=drop, fisher_sr4_pp_vs_ghost=float(f4),
                fisher_sr3_pp_vs_ghost=float(f3))


def k15_table11():
    # AMPLIFY: 작업별 W2 변화(높음-낮음)와 로그 오즈 이득의 순위 일치
    dW = {t: T11[t][2] - T11[t][0] for t in TASKS}
    gains = {t: FIG3B["AMPLIFY"][i] for i, t in enumerate(TASKS)}
    rho, p = stats.spearmanr([-dW[t] for t in TASKS], [gains[t] for t in TASKS])
    # 단조성
    mono = {t: ("감소" if T11[t][0] > T11[t][1] > T11[t][2] else "증가" if T11[t][0] < T11[t][1] < T11[t][2] else "비단조")
            for t in TASKS}
    return dict(delta_w2=dW, gains=gains, rho_neg_dw2_vs_gain=float(rho), p_two_sided=float(p),
                p_exact_n4_two_sided=2 / 24, monotonicity=mono)


def k16_xskill_loss():
    L = 0.5 * math.log(32) + math.log((16 + 1) / 1)
    return dict(L=L, reported=4.57)


def k17_dp_vs_h2r():
    rows = {}
    for t in ["Stacking", "Insertion", "Sequential"]:
        dp_prog, dp_sr = T10["Diffusion Policy"][t]
        best_prog = max((T3[m][t][0], m) for m in METHODS)
        best_sr = max((T3[m][t][2], m) for m in METHODS)
        rows[t] = dict(dp_prog=dp_prog, dp_sr=dp_sr, best_h2r_prog=best_prog, best_h2r_sr=best_sr,
                       h2r_methods_below_dp_prog=sum(1 for m in METHODS if T3[m][t][0] < dp_prog),
                       h2r_methods_below_dp_sr=sum(1 for m in METHODS if T3[m][t][2] < dp_sr))
    # 같은 과제의 로봇 전용 곡선(Fig. 4, 사람 0) 대비
    return rows


def k18_wilson_sr():
    """SR을 90회 이항으로 보고 Wilson 구간(시드 간 상관과 같은 초기 조건 재사용으로 실제 불확실성은 더 클 수 있다)."""
    out = {}
    for m in METHODS:
        for t in TASKS:
            sm = T3[m][t][2]
            k = round(sm * 90 / 100)
            lo, hi = wilson(k, 90)
            out[f"{m}|{t}"] = dict(sr=sm, k_of_90=k, wilson95=[round(100 * lo, 1), round(100 * hi, 1)])
    return out


def k19_pvalue_corr():
    r = 0.894
    t = r * math.sqrt(14 / (1 - r * r))
    return dict(t=t, p=2 * stats.t.sf(t, 14), ci95=fisher_ci(r, 16))


def k20_scaling():
    """Fig. 4(3 시드 평균)의 사람 데모 수준 4개에서 진행도 추세: 단조성, 범위, 시드 SD와의 비교."""
    rows = {}
    for m in METHODS:
        for t in TASKS:
            v = FIG4[m][t]
            rho = stats.spearmanr(range(4), v)[0]
            rng = max(v) - min(v)
            rows[f"{m}|{t}"] = dict(values=v, spearman_with_level=round(float(rho), 2), range=round(rng, 1),
                                    seed_sd_at_max=T3[m][t][1], range_over_sd=round(rng / T3[m][t][1], 1),
                                    monotone_up=bool(all(v[i] <= v[i + 1] for i in range(3))),
                                    monotone_down=bool(all(v[i] >= v[i + 1] for i in range(3))))
    return dict(rows=rows,
                n_monotone=sum(1 for r in rows.values() if r["monotone_up"] or r["monotone_down"]),
                n_range_lt_2sd=sum(1 for r in rows.values() if r["range"] < 2 * r["seed_sd_at_max"]))


def main():
    RES["K1_table3_equals_table9"] = k1_table3_equals_table9()
    RES["K2_feasibility"] = k2_feasibility()
    RES["K3_prog_sr_bounds"] = k3_prog_sr_bounds()
    RES["K4_fig4_vs_table3"] = k4_fig4_vs_table3()
    RES["K5_delta_logit"] = k5_delta_logit()
    RES["K6_table7"] = k6_table7()
    RES["K7_sim_vs_real"] = k7_sim_vs_real()
    RES["K8_robot_only_sim_vs_fig4"] = k8_robot_only_sim_vs_fig4()
    RES["K9_delta_prediction"] = k9_delta_prediction()
    RES["K10_alt_sim_x"] = k10_alt_sim_x()
    RES["K11_monte_carlo"] = k11_monte_carlo()
    RES["K12_effect_vs_noise"] = k12_effect_vs_noise()
    RES["K13_welch_top2"] = k13_welch()
    RES["K14_table8"] = k14_table8()
    RES["K15_table11"] = k15_table11()
    RES["K16_xskill_loss"] = k16_xskill_loss()
    RES["K17_dp_vs_h2r"] = k17_dp_vs_h2r()
    RES["K18_wilson_sr"] = k18_wilson_sr()
    RES["K19_corr_pvalue"] = k19_pvalue_corr()
    RES["K20_scaling"] = k20_scaling()
    OUT.write_text(json.dumps(RES, indent=1, ensure_ascii=False, default=float), encoding="utf-8")

    def show(k, v):
        print(f"[{k}]", json.dumps(v, ensure_ascii=False, default=float)[:1400])

    for k in ["K1_table3_equals_table9", "K4_fig4_vs_table3"]:
        show(k, RES[k])
    k2 = RES["K2_feasibility"]
    print("[K2] cells", k2["cells"], "mean infeasible", k2["mean_infeasible"], "sd infeasible", k2["sd_infeasible_given_mean"],
          "sd sample ok", k2["sd_sample_feasible"], "sd pop ok", k2["sd_pop_feasible"])
    show("K3 violations", RES["K3_prog_sr_bounds"]["violations"])
    k5 = RES["K5_delta_logit"]
    print("[K5] max |recomputed - Fig3B|", k5["max_abs_diff_vs_fig3b"], "undefined if SR:", k5["undefined_if_defined_on_sr"])
    k6 = RES["K6_table7"]
    print("[K6] mean z", k6["mean_z"], "sd z", k6["sd_z"], "|z|>1", k6["n_abs_z_gt_1"], "|z|>2", k6["n_abs_z_gt_2"],
          "max |fig5 x - table7|", k6["max_abs_fig5_x_minus_table7"])
    for name in ["max", "robot"]:
        d = RES["K7_sim_vs_real"][name]
        print(f"[K7 {name}] recomputed", {k: round(v, 3) for k, v in d["recomputed"].items() if k in ("r", "rho", "mmrv")},
              "reported", d["reported"], "r CI", [round(x, 3) for x in d["r_ci95"]])
        print("    per-task MMRV", {t: round(v, 3) for t, v in d["recomputed"]["per_task_mmrv"].items()})
        w = d["within"]
        print("    within-task r", round(w["r_within"], 3), "p", round(w["p"], 3), "kendall mean", round(w["kendall_mean"], 3),
              {t: round(v, 2) for t, v in w["kendall_per_task"].items()}, "between-task share of x var", round(w["between_task_share_x"], 2))
        print("    subsets", {k: {kk: round(vv, 3) for kk, vv in v.items() if kk in ("r", "rho", "mmrv")} for k, v in d["subsets"].items()})
    show("K8", {k: RES["K8_robot_only_sim_vs_fig4"][k] for k in ["mean_abs", "max_abs", "n_abs_gt_5"]})
    show("K9", RES["K9_delta_prediction"])
    show("K10", RES["K10_alt_sim_x"])
    show("K11 monte carlo", RES["K11_monte_carlo"])
    show("K12", {k: RES["K12_effect_vs_noise"][k] for k in ["n_abs_z_gt_2", "n_total"]})
    show("K13", RES["K13_welch_top2"])
    show("K14", RES["K14_table8"])
    show("K15", RES["K15_table11"])
    show("K16", RES["K16_xskill_loss"])
    show("K17", RES["K17_dp_vs_h2r"])
    show("K19", RES["K19_corr_pvalue"])
    k20 = RES["K20_scaling"]
    print("[K20] monotone cells", k20["n_monotone"], "range < 2 SD cells", k20["n_range_lt_2sd"])


if __name__ == "__main__":
    main()
