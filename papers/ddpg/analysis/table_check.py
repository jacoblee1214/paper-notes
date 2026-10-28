"""Table 1·본문·부록 수치 재계산 (CPU). 실행: python3 analysis/table_check.py

- C1: Table 1 요약 통계 (26개 MuJoCo 과제, 정규화 점수: 무작위 0, iLQG 1)
- C2: 계획기(iLQG)를 넘는 과제 수 (평균 기준, 최고 기준)
- C3: 표적망 효과 = DDPG(lowd) vs DPG+BN(cntrl), 픽셀 vs 저차원 (과제 단위 부호·Wilcoxon 검정)
- C4: 5개 시드의 퍼짐 하한 (R_best − R_av로부터 표본 표준편차 하한)
- C5: Table 2(32개 환경)와 Table 1(26개)의 목록 차이
- C6: "≈130,000 parameters" 확인 (Table 2의 관측·행동 차원, 400-300 구조)
- C7: 픽셀 망 "≈430,000 parameters"를 만족하는 합성곱 설정 후보 (논문에 커널·stride 없음)
- C8: "DQN보다 20배 적은 step" 주장의 단위
- C9: OU 잡음(θ = 0.15, σ = 0.2)의 정상 표준편차와 상관 시간, dt에 따라
"""
import itertools
import json
import math
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

# Table 1: R_av,lowd, R_best,lowd, R_av,pix, R_best,pix, R_av,cntrl, R_best,cntrl
TABLE1 = {
    "blockworld1": (1.156, 1.511, 0.466, 1.299, -0.080, 1.260),
    "blockworld3da": (0.340, 0.705, 0.889, 2.225, -0.139, 0.658),
    "canada": (0.303, 1.735, 0.176, 0.688, 0.125, 1.157),
    "canada2d": (0.400, 0.978, -0.285, 0.119, -0.045, 0.701),
    "cart": (0.938, 1.336, 1.096, 1.258, 0.343, 1.216),
    "cartpole": (0.844, 1.115, 0.482, 1.138, 0.244, 0.755),
    "cartpoleBalance": (0.951, 1.000, 0.335, 0.996, -0.468, 0.528),
    "cartpoleParallelDouble": (0.549, 0.900, 0.188, 0.323, 0.197, 0.572),
    "cartpoleSerialDouble": (0.272, 0.719, 0.195, 0.642, 0.143, 0.701),
    "cartpoleSerialTriple": (0.736, 0.946, 0.412, 0.427, 0.583, 0.942),
    "cheetah": (0.903, 1.206, 0.457, 0.792, -0.008, 0.425),
    "fixedReacher": (0.849, 1.021, 0.693, 0.981, 0.259, 0.927),
    "fixedReacherDouble": (0.924, 0.996, 0.872, 0.943, 0.290, 0.995),
    "fixedReacherSingle": (0.954, 1.000, 0.827, 0.995, 0.620, 0.999),
    "gripper": (0.655, 0.972, 0.406, 0.790, 0.461, 0.816),
    "gripperRandom": (0.618, 0.937, 0.082, 0.791, 0.557, 0.808),
    "hardCheetah": (1.311, 1.990, 1.204, 1.431, -0.031, 1.411),
    "hopper": (0.676, 0.936, 0.112, 0.924, 0.078, 0.917),
    "hyq": (0.416, 0.722, 0.234, 0.672, 0.198, 0.618),
    "movingGripper": (0.474, 0.936, 0.480, 0.644, 0.416, 0.805),
    "pendulum": (0.946, 1.021, 0.663, 1.055, 0.099, 0.951),
    "reacher": (0.720, 0.987, 0.194, 0.878, 0.231, 0.953),
    "reacher3daFixedTarget": (0.585, 0.943, 0.453, 0.922, 0.204, 0.631),
    "reacher3daRandomTarget": (0.467, 0.739, 0.374, 0.735, -0.046, 0.158),
    "reacherSingle": (0.981, 1.102, 1.000, 1.083, 1.010, 1.083),
    "walker2d": (0.705, 1.573, 0.944, 1.476, 0.393, 1.397),
}
TORCS = (-393.385, 1840.036, -401.911, 1876.284, -911.034, 1961.600)
COLS = ("av_lowd", "best_lowd", "av_pix", "best_pix", "av_cntrl", "best_cntrl")

# Table 2: dim(s), dim(a), dim(o)
TABLE2 = {
    "blockworld1": (18, 5, 43), "blockworld3da": (31, 9, 102), "canada": (22, 7, 62), "canada2d": (14, 3, 29),
    "cart": (2, 1, 3), "cartpole": (4, 1, 14), "cartpoleBalance": (4, 1, 14), "cartpoleParallelDouble": (6, 1, 16),
    "cartpoleParallelTriple": (8, 1, 23), "cartpoleSerialDouble": (6, 1, 14), "cartpoleSerialTriple": (8, 1, 23),
    "cheetah": (18, 6, 17), "fixedReacher": (10, 3, 23), "fixedReacherDouble": (8, 2, 18),
    "fixedReacherSingle": (6, 1, 13), "gripper": (18, 5, 43), "gripperRandom": (18, 5, 43),
    "hardCheetah": (18, 6, 17), "hardCheetahNice": (18, 6, 17), "hopper": (14, 4, 14), "hyq": (37, 12, 37),
    "hyqKick": (37, 12, 37), "movingGripper": (22, 7, 49), "movingGripperRandom": (22, 7, 49),
    "pendulum": (2, 1, 3), "reacher": (10, 3, 23), "reacher3daFixedTarget": (20, 7, 61),
    "reacher3daRandomTarget": (20, 7, 61), "reacherDouble": (6, 1, 13), "reacherObstacle": (18, 5, 38),
    "reacherSingle": (6, 1, 13), "walker2d": (18, 6, 41),
}


def col(name):
    i = COLS.index(name)
    return [v[i] for v in TABLE1.values()]


def summary(xs):
    xs = sorted(xs)
    n = len(xs)
    med = xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])
    return {"mean": round(sum(xs) / n, 3), "median": round(med, 3), "min": xs[0], "max": xs[-1]}


def paired(a, b):
    """과제 단위 비교: a > b인 과제 수, 평균 차이, Wilcoxon 부호 순위 검정(양측)."""
    diff = [x - y for x, y in zip(a, b)]
    w = stats.wilcoxon(a, b)
    return {"a_gt_b": sum(d > 0 for d in diff), "n": len(diff), "mean_diff": round(sum(diff) / len(diff), 3),
            "wilcoxon_p": float(f"{w.pvalue:.2e}")}


def mlp_params(obs, act, h1=400, h2=300):
    actor = (obs * h1 + h1) + (h1 * h2 + h2) + (h2 * act + act)
    critic = (obs * h1 + h1) + ((h1 + act) * h2 + h2) + (h2 + 1)  # 행동은 두 번째 은닉층에 합류
    return actor, critic


def conv_out(n, k, s, pad):
    return (n + 2 * ((k - 1) // 2) - k) // s + 1 if pad == "same" else (n - k) // s + 1


def pixel_params(k, s, pad, act=6, h=200, c=32, n=64, cin=9):
    size, convp, ch = n, 0, cin
    for _ in range(3):
        size = conv_out(size, k, s, pad)
        if size < 1:
            return None
        convp += ch * c * k * k + c
        ch = c
    flat = c * size * size
    actor = convp + flat * h + h + h * h + h + h * act + act
    critic = convp + flat * h + h + (h + act) * h + h + h + 1  # 행동은 완전연결층부터
    return {"k": k, "s": s, "pad": pad, "out": size, "actor": actor, "critic": critic}


def ou_stats(theta, sigma, dt):
    """x' = x + θ(0 − x)dt + σ√dt ε 의 정상 표준편차, lag-1 자기상관, 상관 시간(1/e 감쇠까지의 step)."""
    rho = 1 - theta * dt
    var = sigma ** 2 * dt / (1 - rho ** 2)
    return {"dt": dt, "stationary_std": round(math.sqrt(var), 4), "lag1_autocorr": round(rho, 4),
            "corr_time_steps": round(-1 / math.log(rho), 1), "half_life_steps": round(math.log(0.5) / math.log(rho), 1)}


def main() -> None:
    out: dict = {}
    n = len(TABLE1)

    out["C1_summary"] = {"n_tasks": n, **{c: summary(col(c)) for c in COLS}}

    out["C2_beats_planner"] = {
        c: sorted(k for k, v in TABLE1.items() if v[COLS.index(c)] > 1.0) for c in COLS
    }
    out["C2_counts"] = {c: len(v) for c, v in out["C2_beats_planner"].items()}

    out["C3_paired"] = {
        "lowd_vs_cntrl_av (표적망 + BN vs BN만)": paired(col("av_lowd"), col("av_cntrl")),
        "lowd_vs_cntrl_best": paired(col("best_lowd"), col("best_cntrl")),
        "pix_vs_lowd_av": paired(col("av_pix"), col("av_lowd")),
        "tasks_where_pix_av_ge_lowd_av": sorted(k for k, v in TABLE1.items() if v[2] >= v[0]),
        "tasks_where_cntrl_av_ge_lowd_av": sorted(k for k, v in TABLE1.items() if v[4] >= v[0]),
    }

    # 5개 값의 평균이 m, 최댓값이 M이면 표본 표준편차 ≥ sqrt(1.25/4)·(M − m) ≈ 0.559·(M − m)
    # (한 값이 M, 나머지 넷이 m − (M − m)/4로 같을 때 최소). R_best가 같은 5개 최종 점수의 최댓값일 때만 성립.
    lb = {k: round(0.559 * (v[1] - v[0]), 3) for k, v in TABLE1.items()}
    out["C4_seed_spread_lower_bound_lowd"] = {
        "std_lower_bound": dict(sorted(lb.items(), key=lambda kv: -kv[1])[:8]),
        "tasks_with_lb_over_0.3": sum(x > 0.3 for x in lb.values()),
        "median_lb": summary(list(lb.values()))["median"],
        "assumption": "R_best가 R_av를 만든 같은 5개 최종 점수 중 최댓값일 때. 학습 중 최고 체크포인트라면 하한이 아님",
    }

    out["C5_env_lists"] = {
        "table2_n": len(TABLE2), "table1_n": n,
        "in_table2_not_table1": sorted(set(TABLE2) - set(TABLE1)),
        "in_table1_not_table2": sorted(set(TABLE1) - set(TABLE2)),
    }

    counts = {k: mlp_params(TABLE2[k][2], TABLE2[k][1]) for k in TABLE1}
    actors = [a for a, _ in counts.values()]
    critics = [c for _, c in counts.values()]
    out["C6_lowd_params_400_300"] = {
        "actor_range": [min(actors), max(actors)], "critic_range": [min(critics), max(critics)],
        "cheetah": counts["cheetah"], "pendulum": counts["pendulum"],
        "reading": "약 13만은 actor·critic 각각의 크기 (두 망 합계가 아님). BN 파라미터는 수천 개 수준",
    }

    # 세 층이 같은 커널·stride를 쓴다고 가정하고 가장 가까운 설정을 찾는다 (논문에 커널·stride·padding 없음)
    cands = [pixel_params(k, s, p) for k, s, p in itertools.product(range(2, 9), (1, 2, 3, 4), ("valid", "same"))]
    cands = sorted((c for c in cands if c), key=lambda c: abs(c["actor"] - 430_000))
    out["C7_pixel_net_closest_uniform_configs"] = {
        "closest": cands[:4],
        "best_rel_error": round(abs(cands[0]["actor"] - 430_000) / 430_000, 3),
        "reading": "같은 커널·stride 세 층으로는 430k(±7%)를 만들 수 없음. 층마다 다른 설정이었을 것으로 보이며, 논문만으로는 픽셀 망을 복원할 수 없음",
    }

    # DQN (Mnih et al., 2015): 5천만 프레임, 프레임 4개마다 행동 1번 (내 사전 지식, 원문 확인 필요)
    dqn_frames, dqn_skip, ddpg_steps, ddpg_repeat = 50e6, 4, 2.5e6, 3
    out["C8_dqn_comparison"] = {
        "claimed_ratio": 20,
        "dqn_frames_over_ddpg_steps": round(dqn_frames / ddpg_steps, 1),
        "agent_steps_ratio": round(dqn_frames / dqn_skip / ddpg_steps, 1),
        "sim_frames_ratio_pixel_tasks": round(dqn_frames / (ddpg_steps * ddpg_repeat), 1),
    }

    out["C9_ou_noise"] = {f"dt={dt}": ou_stats(0.15, 0.2, dt) for dt in (1.0, 0.1, 0.01)}
    out["C9_note"] = "논문은 θ, σ만 주고 dt를 주지 않음. rllab은 dt = 1, OpenAI baselines·SB3는 dt = 0.01이 기본"

    out["C10_torcs"] = dict(zip(COLS, TORCS))

    path = ROOT / "analysis" / "table_check_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for k, v in out.items():
        print(k, json.dumps(v, ensure_ascii=False)[:400])
    print("wrote", path)


if __name__ == "__main__":
    main()
