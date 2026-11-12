"""FlashSAC 논문 수치 재계산: 공개 학습 곡선(repo/results/*.csv)에서 최종 성능, 같은 환경 step 비교, 유의성, 소거 실험 지표를 다시 계산한다.

실행: python3 analysis/recompute_results.py  → analysis/recompute_results.json
입력: 공식 저장소를 얕게 복제한 폴더 `repo/` (git clone --depth 1 https://github.com/Holiday-Robot/FlashSAC repo)
주의: 공개 CSV에는 벽시계 시간 열이 없다. 벽시계(compute time) 주장은 이 스크립트로 검증할 수 없다.
정규화 상수는 논문 부록 Table 1–8의 Normalize Score 열을 그대로 옮겼다. 정규화 식은 논문에 명시되지 않아
정규화 점수 = 반환 / Normalize Score 로 가정했다(MuJoCo는 random 점수 보정 없음).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "repo" / "results"
OUT = ROOT / "analysis" / "recompute_results.json"
RNG = np.random.default_rng(0)
N_BOOT = 10000

# 논문 부록 Table 1–8의 Normalize Score (논문 값 그대로)
NORM_GPU = {
    "Isaac-Repose-Cube-Shadow-Direct-v0": 10000, "Isaac-Repose-Cube-Allegro-Direct-v0": 6000,
    "Isaac-Velocity-Flat-G1-v0": 40, "Isaac-Velocity-Rough-G1-v0": 40,
    "Isaac-Velocity-Flat-H1-v0": 40, "Isaac-Velocity-Rough-H1-v0": 40,
    "Isaac-Lift-Cube-Franka-v0": 160, "Isaac-Open-Drawer-Franka-v0": 100,
    "Isaac-Velocity-Flat-Anymal-C-v0": 30, "Isaac-Velocity-Rough-Anymal-C-v0": 30,
    "Isaac-Velocity-Flat-Anymal-D-v0": 30, "Isaac-Velocity-Rough-Anymal-D-v0": 30,
    "G1JoystickRoughTerrain": 40, "G1JoystickFlatTerrain": 40,
    "T1JoystickRoughTerrain": 40, "T1JoystickFlatTerrain": 40,
    "go2-walk_easy": 25, "go2-walk": 25, "panda-grasp": 2.5,
}
SUCCESS_GPU = {"PickSingleYCB-v1", "PegInsertionSide-v1", "LiftPegUpright-v1", "PokeCube-v1", "PullCube-v1", "RollBall-v1"}
NORM_CPU = {
    "halfcheetah-v4": 18165, "hopper-v4": 4075, "walker2d-v4": 7397, "ant-v4": 10133, "humanoid-v4": 10281,
    "h1-walk-v0": 700, "h1-stand-v0": 800, "h1-run-v0": 700, "h1-reach-v0": 12000, "h1-maze-v0": 1200,
    "h1-hurdle-v0": 700, "h1-crawl-v0": 700, "h1-sit-simple-v0": 750, "h1-sit-hard-v0": 750,
    "h1-balance-simple-v0": 800, "h1-balance-hard-v0": 800, "h1-stair-v0": 700, "h1-slide-v0": 700, "h1-pole-v0": 700,
}
MUJOCO_CPU = {"halfcheetah-v4", "hopper-v4", "walker2d-v4", "ant-v4", "humanoid-v4"}
VISUAL_NON_DMC = {"door-v0-sparse", "hammer-v0-sparse", "pen-v0-sparse"}  # Adroit 계열, 논문 정규화 표에 없음


def canon(name):
    return name.replace("_", "-").lower()


def load(rel):
    d = pd.read_csv(RES / rel, index_col=0)
    d = d.dropna(subset=["value"])
    d["env"] = d["env_name"].map(canon)
    return d


def norm_gpu(env):
    """(지표, 정규화 상수). 지표는 avg_success 또는 avg_return."""
    if env in {canon(e) for e in SUCCESS_GPU}:
        return "avg_success", 1.0
    if env in {canon(e) for e in NORM_GPU}:
        return "avg_return", [v for k, v in NORM_GPU.items() if canon(k) == env][0]
    return "avg_return", None


def norm_cpu(env):
    """CPU 과제: MyoSuite는 성공률, MuJoCo·HBench는 표 값, 나머지 CPU 과제는 DMC(이론 최댓값 1000)."""
    if env.startswith("myo-"):
        return "avg_success", 1.0
    if env in MUJOCO_CPU or env in NORM_CPU:
        return "avg_return", NORM_CPU[env]
    return "avg_return", 1000.0


def norm_visual(env):
    if env in VISUAL_NON_DMC:
        return "avg_return", None
    return "avg_return", 1000.0


def per_seed_curves(d, env, metric):
    """{seed: (steps, values)} 정렬된 곡선."""
    out = {}
    s = d[(d.env == env) & (d.metric == metric)]
    for seed, g in s.groupby("seed"):
        g = g.sort_values("env_step")
        out[int(seed)] = (g.env_step.to_numpy(float), g.value.to_numpy(float))
    return out


def value_at(curve, t):
    steps, vals = curve
    return float(np.interp(t, steps, vals))


def seed_values(d, env, metric, norm, at=None):
    """시드별 값(정규화 적용). at=None이면 각 시드의 마지막 값, 아니면 env step=at에서 보간."""
    cur = per_seed_curves(d, env, metric)
    vals = []
    for seed in sorted(cur):
        c = cur[seed]
        v = c[1][-1] if at is None else value_at(c, at)
        vals.append(v / (norm if norm else 1.0))
    return np.array(vals)


def bootstrap_diff_ci(a, b):
    """두 시드 집합 평균 차이의 95% 백분위 부트스트랩 구간."""
    if len(a) < 2 or len(b) < 2:
        return [None, None]
    ia = RNG.integers(0, len(a), (N_BOOT, len(a)))
    ib = RNG.integers(0, len(b), (N_BOOT, len(b)))
    diff = a[ia].mean(1) - b[ib].mean(1)
    return [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


def compare(a, b):
    """FlashSAC(a) 대 기준선(b): 차이, 부트스트랩 CI, Welch p, Mann-Whitney p."""
    res = {"n_a": int(len(a)), "n_b": int(len(b)),
           "mean_a": float(a.mean()) if len(a) else None, "mean_b": float(b.mean()) if len(b) else None}
    if len(a) >= 2 and len(b) >= 2:
        res["diff"] = float(a.mean() - b.mean())
        res["diff_ci95"] = bootstrap_diff_ci(a, b)
        res["p_welch"] = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
        res["p_mwu"] = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    return res


def fmt(x, nd=3):
    return None if x is None else round(float(x), nd)


out = {}

# ---------- 1. GPU 시뮬레이터 (state-based): FlashSAC 50M 대 PPO 200M, FastTD3 50M ----------
fs = load("stateRL_gpusim/FlashSAC.csv")
ppo = load("stateRL_gpusim/baseline/PPO.csv")
ftd = load("stateRL_gpusim/baseline/FastTD3.csv")
gpu_envs = sorted(set(fs.env) & set(ppo.env) & set(ftd.env))
gpu_rows = []
for env in gpu_envs:
    metric, norm = norm_gpu(env)
    if norm is None:
        continue
    a_final = seed_values(fs, env, metric, norm)
    p_final = seed_values(ppo, env, metric, norm)
    f_final = seed_values(ftd, env, metric, norm)
    p_at50 = seed_values(ppo, env, metric, norm, at=50_000_896)
    f50 = compare(a_final, p_final)
    gpu_rows.append({
        "env": env, "metric": metric, "norm": norm,
        "flashsac_final": fmt(a_final.mean()), "flashsac_sd": fmt(a_final.std(ddof=1)),
        "ppo200M_final": fmt(p_final.mean()), "ppo200M_sd": fmt(p_final.std(ddof=1)),
        "ppo_at_50M": fmt(p_at50.mean()),
        "fasttd3_final": fmt(f_final.mean()),
        "n_seeds": [int(len(a_final)), int(len(p_final)), int(len(f_final))],
        "flashsac_vs_ppo200M": compare(a_final, p_final),
        "flashsac_vs_fasttd3": compare(a_final, f_final),
        "flashsac_vs_ppo_at_50M": compare(a_final, p_at50),
        "fasttd3_fail_seeds": int(np.sum(f_final < 0.5 * a_final.mean())),
    })

for key in ["flashsac_vs_ppo200M", "flashsac_vs_fasttd3", "flashsac_vs_ppo_at_50M"]:
    ps = [r[key].get("p_welch") for r in gpu_rows if r[key].get("p_welch") is not None]
    adj = holm(ps)
    j = 0
    for r in gpu_rows:
        if r[key].get("p_welch") is not None:
            r[key]["p_welch_holm"] = adj[j]
            j += 1

hi_dim = {"Isaac-Repose-Cube-Shadow-Direct-v0", "Isaac-Repose-Cube-Allegro-Direct-v0", "Isaac-Velocity-Flat-G1-v0",
          "Isaac-Velocity-Rough-G1-v0", "Isaac-Velocity-Flat-H1-v0", "Isaac-Velocity-Rough-H1-v0",
          "G1JoystickRoughTerrain", "G1JoystickFlatTerrain", "T1JoystickRoughTerrain", "T1JoystickFlatTerrain"}
hi_dim = {canon(x) for x in hi_dim}
summary_gpu = {}
for label, sel in [("all", gpu_rows), ("high_dim_10", [r for r in gpu_rows if r["env"] in hi_dim]),
                   ("low_dim", [r for r in gpu_rows if r["env"] not in hi_dim])]:
    summary_gpu[label] = {
        "n_tasks": len(sel),
        "flashsac_wins_vs_ppo200M_mean": int(sum(r["flashsac_final"] > r["ppo200M_final"] for r in sel)),
        "flashsac_wins_vs_ppo_at_50M": int(sum(r["flashsac_final"] > r["ppo_at_50M"] for r in sel)),
        "flashsac_wins_vs_fasttd3": int(sum(r["flashsac_final"] > r["fasttd3_final"] for r in sel)),
        "sig_holm05_vs_ppo200M": int(sum(1 for r in sel if (r["flashsac_vs_ppo200M"].get("p_welch_holm") or 1) < 0.05)),
        "sig_holm05_vs_ppo_at_50M": int(sum(1 for r in sel if (r["flashsac_vs_ppo_at_50M"].get("p_welch_holm") or 1) < 0.05)),
        "sig_holm05_vs_fasttd3": int(sum(1 for r in sel if (r["flashsac_vs_fasttd3"].get("p_welch_holm") or 1) < 0.05)),
        "fasttd3_tasks_with_failed_seed": int(sum(1 for r in sel if r["fasttd3_fail_seeds"] > 0)),
        "fasttd3_fail_seed_total": int(sum(r["fasttd3_fail_seeds"] for r in sel)),
    }
out["gpu_state"] = {"rows": gpu_rows, "summary": summary_gpu,
                    "ppo_env_steps_max": float(ppo.env_step.max()), "flashsac_env_steps_max": float(fs.env_step.max())}

# 제외된 과제 (정규화 상수가 논문 표에 없음)
out["gpu_tasks_without_norm"] = sorted(set(fs.env) - {r["env"] for r in gpu_rows})

# ---------- 2. CPU 시뮬레이터 (state-based): 같은 환경 step 1M에서 비교 ----------
fsc = load("stateRL_cpusim/FlashSAC.csv")
base_cpu = {
    "PPO": load("stateRL_cpusim/baseline/PPO.csv"),
    "SimbaV2": load("stateRL_cpusim/baseline/SimbaV2.csv"),
    "XQC": load("stateRL_cpusim/baseline/XQC.csv"),
    "TD-MPC2": load("stateRL_cpusim/baseline/TD-MPC2.csv"),
    "MR.Q": load("stateRL_cpusim/baseline/MRQ.csv"),
}
STEP_CPU = 1_000_000
cpu_rows = []
for env in sorted(set(fsc.env)):
    metric, norm = norm_cpu(env)
    if norm is None:
        continue
    a = seed_values(fsc, env, metric, norm, at=STEP_CPU)
    row = {"env": env, "norm": norm, "metric": metric, "flashsac_1M": fmt(a.mean()), "n_seeds": int(len(a))}
    for name, d in base_cpu.items():
        if env not in set(d.env):
            continue
        b = seed_values(d, env, metric, norm, at=STEP_CPU)
        if len(b) == 0:
            continue
        row[name + "_1M"] = fmt(b.mean())
        row["vs_" + name] = compare(a, b)
    cpu_rows.append(row)

cpu_summary = {}
for name in base_cpu:
    sel = [r for r in cpu_rows if ("vs_" + name) in r]
    ps = [r["vs_" + name].get("p_welch") for r in sel if r["vs_" + name].get("p_welch") is not None]
    adj = holm(ps) if ps else []
    sig = sum(1 for p in adj if p < 0.05)
    cpu_summary[name] = {
        "n_tasks": len(sel),
        "flashsac_higher_mean": int(sum(r["flashsac_1M"] > r[name + "_1M"] for r in sel)),
        "sig_holm05": int(sig),
        "n_tasks_with_test": len(ps),
        "mean_norm_flashsac": fmt(np.mean([r["flashsac_1M"] for r in sel])) if sel else None,
        "mean_norm_baseline": fmt(np.mean([r[name + "_1M"] for r in sel])) if sel else None,
    }
out["cpu_state"] = {"rows": cpu_rows, "summary": cpu_summary, "step": STEP_CPU,
                    "baseline_env_steps_max": {k: float(v.env_step.max()) for k, v in base_cpu.items()}}

# ---------- 3. 시각 기반 (DMC): 1M 환경 step, 정규화 1000 ----------
fsv = load("visualRL/FlashSAC.csv")
vis_base = {"DrQ-v2": load("visualRL/baseline/DrQ-v2.csv"), "MR.Q": load("visualRL/baseline/MRQ.csv"),
            "XQC": load("visualRL/baseline/XQC.csv"), "DrM": load("visualRL/baseline/DrM.csv")}
vis_rows = []
for env in sorted(set(fsv.env)):
    _, norm = norm_visual(env)
    if norm is None:  # Adroit 계열 과제는 정규화 상수가 없어 제외하고 개수만 보고한다
        continue
    a = seed_values(fsv, env, "avg_return", norm, at=STEP_CPU)
    row = {"env": env, "flashsac_1M": fmt(a.mean()), "n_seeds": int(len(a))}
    for name, d in vis_base.items():
        if env in set(d.env):
            b = seed_values(d, env, "avg_return", norm, at=STEP_CPU)
            if len(b):
                row[name + "_1M"] = fmt(b.mean())
                row["vs_" + name] = compare(a, b)
    vis_rows.append(row)
vis_summary = {}
for name in vis_base:
    sel = [r for r in vis_rows if ("vs_" + name) in r]
    ps = [r["vs_" + name].get("p_welch") for r in sel if r["vs_" + name].get("p_welch") is not None]
    adj = holm(ps) if ps else []
    vis_summary[name] = {"n_tasks": len(sel), "flashsac_higher_mean": int(sum(r["flashsac_1M"] > r[name + "_1M"] for r in sel)),
                         "sig_holm05": int(sum(p < 0.05 for p in adj))}
out["visual"] = {"rows": vis_rows, "step": STEP_CPU,

                 "baseline_env_steps_max": {k: float(v.env_step.max()) for k, v in vis_base.items()}, "summary": vis_summary}

# ---------- 4. 실물 전 단계 sim-to-real: 시드 1개, 환경 step 단위 ----------
s2 = {}
for tag, rel in [("flashsac_flat", "sim2real/flashsac_g1flat.csv"), ("ppo_flat", "sim2real/ppo_g1flat.csv"),
                 ("flashsac_rough", "sim2real/flashsac_g1rough.csv"), ("ppo_rough", "sim2real/ppo_g1rough.csv")]:
    d = pd.read_csv(RES / rel, index_col=0)
    seeds = sorted(d.seed.unique())
    s = d[d.metric == "avg_terrain_level"].sort_values("env_step")
    steps = s.env_step.to_numpy(float)
    vals = s.value.to_numpy(float)
    sm = pd.Series(vals).rolling(50, min_periods=1).mean().to_numpy()
    plateau = float(sm[-500:].mean())

    def first_step(mask):
        idx = np.flatnonzero(mask)
        return float(steps[idx[0]]) if idx.size else None

    s2[tag] = {"seeds": [int(x) for x in seeds], "env_steps_max": float(steps.max()),
               "terrain_level_first_value": fmt(vals[0]),
               "plateau_last500_mean": fmt(plateau),
               "env_step_first_ge_3.0": first_step(sm >= 3.0),
               "env_step_first_80pct_plateau": first_step(sm >= 0.8 * plateau),
               "rolling_window_points": 50}
out["sim2real"] = s2

# ---------- 5. 소거 실험 (IsaacLab 4과제, 시드 5개) ----------
ABL_ENVS = [canon(e) for e in ["Isaac-Repose-Cube-Shadow-Direct-v0", "Isaac-Repose-Cube-Allegro-Direct-v0",
                               "Isaac-Velocity-Flat-G1-v0", "Isaac-Velocity-Rough-G1-v0"]]
base_fs = fs[fs.env.isin(ABL_ENVS)]


def abl_final(rel, metric, normalize=True):
    d = load(rel)
    per_env = {}
    for env in sorted(set(d.env)):
        m, norm = norm_gpu(env)
        if metric in ("avg_return",) and normalize:
            v = seed_values(d, env, "avg_return", norm)
        else:
            cur = per_seed_curves(d, env, metric)
            v = np.array([c[1][-1] for c in cur.values()])
        per_env[env] = v
    return per_env


arch_files = {
    "a_MLP": "ablation_architecture/a_MLP.csv", "b_residual": "ablation_architecture/b_residual.csv",
    "c_batchnorm": "ablation_architecture/c_batchnorm.csv", "d_rmsnorm": "ablation_architecture/d_rmsnorm.csv",
    "e_distcritic": "ablation_architecture/e_distcritic.csv", "f_flashSAC": "ablation_architecture/f_flashSAC.csv",
}
STAB = ["critic/condition_number", "critic/max_eigval", "critic/min_eigval", "srank", "dormant_neurons",
        "feature_magnitude", "gradient_param_norm", "weight_param_norm"]
arch = {}
for name, rel in arch_files.items():
    d = load(rel)
    entry = {"return_final_norm": {}, "stability_final": {}}
    rets = abl_final(rel, "avg_return")
    entry["return_final_norm"] = {e: fmt(v.mean()) for e, v in rets.items()}
    entry["return_mean_over_envs"] = fmt(np.mean([v.mean() for v in rets.values()]))
    for m in STAB:
        vals = []
        for env in sorted(set(d.env)):
            cur = per_seed_curves(d, env, m)
            vals.extend([c[1][-1] for c in cur.values()])
        if vals:
            entry["stability_final"][m] = fmt(np.mean(vals), 4)
    arch[name] = entry
out["ablation_architecture"] = arch

# 조건수가 a→f로 단조 감소하는지 (환경 평균의 마지막 값)
cond = [arch[k]["stability_final"].get("critic/condition_number") for k in arch_files]
out["condition_number_monotone_decrease_mean"] = bool(all(cond[i] >= cond[i + 1] for i in range(len(cond) - 1)))
# 평균은 꼬리가 길어 중앙값과 로그 평균도 같이 본다
cond_med, cond_logmean = [], []
for name in arch_files:
    vals = []
    for env in sorted(set(load(arch_files[name]).env)):
        cur = per_seed_curves(load(arch_files[name]), env, "critic/condition_number")
        vals.extend([c[1][-1] for c in cur.values()])
    vals = np.asarray(vals)
    cond_med.append(float(np.median(vals)))
    cond_logmean.append(float(np.exp(np.mean(np.log(vals)))))
out["condition_number_final_median"] = dict(zip(arch_files, [fmt(x, 0) for x in cond_med]))
out["condition_number_final_geomean"] = dict(zip(arch_files, [fmt(x, 0) for x in cond_logmean]))
out["condition_number_monotone_decrease_median"] = bool(all(cond_med[i] >= cond_med[i + 1] for i in range(len(cond_med) - 1)))
out["condition_number_monotone_decrease_geomean"] = bool(all(cond_logmean[i] >= cond_logmean[i + 1] for i in range(len(cond_logmean) - 1)))
out["condition_number_lowest_variant_median"] = list(arch_files)[int(np.argmin(cond_med))]

scaling = {
    "buffer": ["buffer_0.1M", "buffer_1M", "buffer_50M"], "batch": ["batch_0.5k", "batch_1k", "batch_4k", "batch_8k"],
    "width": ["width_critic64_actor32", "width_critic128_actor64", "width_critic512_actor256", "width_critic1024_actor512"],
    "depth": ["depth_1block", "depth_3block", "depth_4block"], "utd": ["utd05", "utd1", "utd4", "utd8"],
    "exploration": ["target_std_0.05", "target_std_0.1", "target_std_0.2", "target_std_0.25", "no_noise_repeat"],
}
abl_out = {}
base_vals = {env: seed_values(base_fs, env, "avg_return", norm_gpu(env)[1]) for env in ABL_ENVS
             if norm_gpu(env)[1] is not None}
for group, names in scaling.items():
    abl_out[group] = {}
    for n in names:
        d = load(f"ablation_scaling/{n}.csv") if group != "exploration" else load(f"ablation_exploration/{n}.csv")
        per = {}
        for env in ABL_ENVS:
            m, norm = norm_gpu(env)
            if env not in set(d.env):
                continue
            v = seed_values(d, env, "avg_return", norm)
            per[env] = {"mean": fmt(v.mean()), "vs_flashsac": compare(v, base_vals[env])}
        abl_out[group][n] = per
out["ablation_scaling_exploration"] = abl_out
out["ablation_default_mean_norm"] = fmt(np.mean([v.mean() for v in base_vals.values()]))

# 탐색 소거에서 기본값(0.15)은 공개 stateRL_gpusim/FlashSAC 같은 과제, 시드 5개다.
out["notes"] = {
    "baseline_for_ablations": "stateRL_gpusim/FlashSAC.csv (같은 4과제, 같은 시드 5개)",
    "normalization": "반환 / Normalize Score (논문 부록 표 값). 정규화 식은 논문에 명시되지 않음",
    "equal_step_comparison": "환경 step 보간(np.interp) 값으로 비교",
    "wall_clock": "공개 CSV에 벽시계 열이 없어 시간 기반 주장은 검증 불가",
}

OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
print("wrote", OUT)
print("GPU summary", json.dumps(summary_gpu, ensure_ascii=False))
print("CPU summary", json.dumps(cpu_summary, ensure_ascii=False))
print("cond monotone", out["condition_number_monotone_decrease_mean"], cond)
print("sim2real", json.dumps(s2, ensure_ascii=False))
