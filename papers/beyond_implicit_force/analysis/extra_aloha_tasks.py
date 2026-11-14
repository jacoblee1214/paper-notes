"""공개 ALOHA 과제 6개에서 δ = action − qpos 와 effort의 관계, 중력 회귀 잔차, 접촉 후보 검출을 같은 규칙으로 본다.

대상(LeRobot 허브, 영상 제외, 과제마다 parquet 한 개):
  effort 열이 있는 과제 5개: aloha_static_screw_driver(기존 검증과 같은 파일), aloha_static_coffee,
  aloha_static_fork_pick_up, aloha_static_vinh_cup, aloha_static_pingpong_test.
  effort 열이 없는 과제(battery, candy, tape, thread_velcro, ziploc_slide)는 받았지만 δ–effort 검증에 쓸 수 없어 원자료를 지웠다.
  각 파일의 sha256은 허브 트리 API의 LFS oid와 대조했다(data/aloha_tasks/*.tree.json).

규칙은 extra_aloha_effort.py와 같다.
  - 준정적 프레임: 관절 속도 ≤ 해당 관절의 중앙값.
  - 중력 모델: 같은 팔 6관절 sin/cos 선형 최소제곱, 그리퍼 제외.
  - 과제마다 에피소드 앞 70%로 학습, 나머지로 검증.
  - 잔차 r = effort − 중력 추정. 접촉 후보 = 준정적 MAD 기준 k·σ 초과 (k = 2, 3, 4).

접촉 검출 지표(라벨 없음, 대리 지표):
  - 그리퍼가 물체를 쥔 프레임의 대리 라벨 = 준정적 프레임에서 |δ_gripper| 가 중앙값 + 3·MAD를 넘는 프레임.
    (명령은 닫힘인데 그리퍼가 막혀 있으면 δ가 커진다고 본다. 확인된 접촉 라벨이 아니다.)
  - 점수 = 같은 팔 6관절 잔차의 |r_j|/σ_j 최댓값. 이 점수의 AUC를 대리 라벨에 대해 잰다.

불확실성: 에피소드 단위 부트스트랩 200회(시드 0).
"""
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pyarrow
import pyarrow.parquet as pq
import scipy
from scipy.stats import rankdata, spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = HERE / "extra_aloha_tasks_results.json"
TASKS = [
    ("aloha_static_screw_driver", ROOT / "data" / "aloha_static_screw_driver_file-000.parquet"),
    ("aloha_static_coffee", ROOT / "data" / "aloha_tasks" / "aloha_static_coffee.file-000.parquet"),
    ("aloha_static_fork_pick_up", ROOT / "data" / "aloha_tasks" / "aloha_static_fork_pick_up.file-000.parquet"),
    ("aloha_static_vinh_cup", ROOT / "data" / "aloha_tasks" / "aloha_static_vinh_cup.file-000.parquet"),
    ("aloha_static_pingpong_test", ROOT / "data" / "aloha_tasks" / "aloha_static_pingpong_test.file-000.parquet"),
]
JOINTS = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate", "gripper"]
NAMES = [f"{arm}_{j}" for arm in ("left", "right") for j in JOINTS]
GRIP = {6, 13}
N_BOOT = 200
SEED = 0
K_LIST = (2.0, 3.0, 4.0)
TRAIN_FRAC = 0.7


def auc(pos, neg):
    allv = np.r_[pos, neg]
    ranks = rankdata(allv)
    n1, n0 = len(pos), len(neg)
    u = ranks[:n1].sum() - n1 * (n1 + 1) / 2
    return float(u / (n1 * n0))


def r2(y, yhat):
    return float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))


def design(q):
    cols = [np.ones(q.shape[0])]
    for k in range(q.shape[1]):
        cols += [np.sin(q[:, k]), np.cos(q[:, k])]
    return np.stack(cols, axis=1)


def load(path: Path):
    tab = pq.read_table(path)
    ep = tab["episode_index"].to_numpy()
    st = np.stack(tab["observation.state"].to_numpy(zero_copy_only=False)).astype(float)
    ef = np.stack(tab["observation.effort"].to_numpy(zero_copy_only=False)).astype(float)
    ac = np.stack(tab["action"].to_numpy(zero_copy_only=False)).astype(float)
    return ep, st, ef, ac


def analyse(name: str, path: Path, rng: np.random.Generator) -> dict:
    ep, st, ef, ac = load(path)
    if st.shape[1] != 14 or ef.shape[1] != 14 or ac.shape[1] != 14:
        return {"task": name, "excluded": f"dims state {st.shape[1]}, effort {ef.shape[1]}, action {ac.shape[1]} (14 required)"}
    delta = ac - st
    first = np.r_[True, ep[1:] != ep[:-1]]
    valid = ~first
    vel = np.zeros_like(st)
    vel[1:] = np.abs(st[1:] - st[:-1])
    qs = np.zeros_like(st, dtype=bool)
    for j in range(14):
        qs[:, j] = valid & (vel[:, j] <= np.median(vel[valid, j]))
    eps = np.unique(ep)
    cut = eps[int(np.ceil(TRAIN_FRAC * len(eps))) - 1]
    train = ep <= cut
    test = ~train

    ghat = np.zeros_like(ef)
    gravity = {}
    for off in (0, 7):
        idx = list(range(off, off + 6))
        X = design(st[:, idx])
        for j in idx:
            if j in GRIP:
                continue
            m_tr = train & qs[:, j]
            m_te = test & qs[:, j]
            coef, *_ = np.linalg.lstsq(X[m_tr], ef[m_tr, j], rcond=None)
            ghat[:, j] = X @ coef
            gravity[NAMES[j]] = {"R2_test_qs": r2(ef[m_te, j], ghat[m_te, j]) if m_te.sum() > 10 else float("nan")}
    for j in GRIP:
        ghat[:, j] = np.median(ef[qs[:, j] & train, j])
    resid = ef - ghat

    def boot_rho(x, y, mask):
        base = float(spearmanr(x[mask], y[mask]).statistic)
        eps_m = [e for e in eps if mask[ep == e].any()]
        idx_by = {e: np.where((ep == e) & mask)[0] for e in eps_m}
        vals = []
        for _ in range(N_BOOT):
            pick = rng.choice(eps_m, size=len(eps_m), replace=True)
            sel = np.concatenate([idx_by[e] for e in pick])
            vals.append(float(spearmanr(x[sel], y[sel]).statistic))
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return base, float(lo), float(hi)

    per_joint = {}
    for j, nm in enumerate(NAMES):
        rho, lo, hi = boot_rho(delta[:, j], ef[:, j], qs[:, j])
        per_joint[nm] = {"spearman_qs": rho, "spearman_qs_ci95": [lo, hi],
                         "effort_std": float(ef[:, j].std()), "delta_std_rad": float(delta[:, j].std())}

    # 접촉 후보 (잔차 기준)
    sig = {}
    med = {}
    for j in range(14):
        if j in GRIP:
            continue
        med[j] = float(np.median(resid[qs[:, j], j]))
        sig[j] = float(1.4826 * np.median(np.abs(resid[qs[:, j], j] - med[j])))
    flags = {}
    for k in K_LIST:
        entry = {}
        for j in range(14):
            if j in GRIP:
                continue
            fl = np.abs(resid[:, j] - med[j]) > k * sig[j]
            pos = np.abs(delta[valid & fl, j])
            neg = np.abs(delta[valid & ~fl, j])
            entry[NAMES[j]] = {"flag_frac": float(fl[valid].mean()),
                               "AUC_abs_delta_qs": float(auc(np.abs(delta[qs[:, j] & fl, j]), np.abs(delta[qs[:, j] & ~fl, j])))
                               if (qs[:, j] & fl).sum() > 5 and (qs[:, j] & ~fl).sum() > 5 else float("nan")}
        flags[f"k{int(k)}"] = entry

    # 그리퍼 막힘 대리 라벨과 같은 팔 잔차 점수
    proxy = {}
    for arm, off in (("left", 0), ("right", 7)):
        g = off + 6
        dg = delta[:, g]
        q_g = qs[:, g]
        gm = np.median(dg[q_g])
        gmad = 1.4826 * np.median(np.abs(dg[q_g] - gm))
        blocked = valid & (np.abs(dg - gm) > 3 * gmad) & q_g
        arm_joints = list(range(off, off + 6))
        score = np.max(np.stack([np.abs(resid[:, j] - med[j]) / sig[j] for j in arm_joints if j not in GRIP], axis=1), axis=1)
        # 점수 비교는 준정적 프레임 안에서 한다 (대리 라벨도 준정적에서 정의)
        m = q_g
        pos = score[m & blocked]
        neg = score[m & ~blocked]
        proxy[arm] = {
            "blocked_frac_qs": float(blocked[m].mean()),
            "n_blocked": int((m & blocked).sum()),
            "AUC_arm_resid_score_vs_gripper_blocked": float(auc(pos, neg)) if pos.size > 5 and neg.size > 5 else float("nan"),
            "median_score_blocked": float(np.median(pos)) if pos.size else float("nan"),
            "median_score_free": float(np.median(neg)) if neg.size else float("nan"),
            "spearman_absdelta_gripper_vs_effort_gripper_qs": float(spearmanr(np.abs(dg[q_g]), np.abs(ef[q_g, g])).statistic),
            "median_abs_effort_gripper_blocked": float(np.median(np.abs(ef[m & blocked, g]))) if (m & blocked).sum() else float("nan"),
            "median_abs_effort_gripper_free": float(np.median(np.abs(ef[m & ~blocked, g]))) if (m & ~blocked).sum() else float("nan"),
        }

    return {
        "task": name,
        "source_file_bytes": path.stat().st_size,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "episodes": int(eps.size),
        "frames": int(ep.size),
        "train_episodes": f"{int(eps.min())}-{int(cut)}",
        "test_episodes": f"{int(cut) + 1}-{int(eps.max())}",
        "gravity_test_R2_by_joint": gravity,
        "per_joint": per_joint,
        "contact_candidates": flags,
        "gripper_blocked_proxy": proxy,
    }


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    per_task = []
    for name, path in TASKS:
        rec = analyse(name, path, rng)
        per_task.append(rec)
        if "excluded" in rec:
            print(name, "excluded", rec["excluded"], flush=True)
        else:
            rhos = [rec["per_joint"][n]["spearman_qs"] for n in NAMES if n not in ("left_gripper", "right_gripper")]
            print(f"{name}: episodes {rec['episodes']} median rho(arm, qs) {np.median(rhos):.3f}", flush=True)

    ok = [r for r in per_task if "excluded" not in r]
    summary = {"n_tasks_used": len(ok), "n_tasks_excluded": len(per_task) - len(ok)}
    arm_names = [n for n in NAMES if "gripper" not in n]
    rho_mat = np.array([[r["per_joint"][n]["spearman_qs"] for n in arm_names] for r in ok])
    summary["arm_joint_spearman_qs_median_over_tasks"] = dict(zip(arm_names, np.round(np.median(rho_mat, axis=0), 3).tolist()))
    summary["arm_joint_spearman_qs_min_over_tasks"] = dict(zip(arm_names, np.round(rho_mat.min(axis=0), 3).tolist()))
    summary["arm_joint_spearman_qs_max_over_tasks"] = dict(zip(arm_names, np.round(rho_mat.max(axis=0), 3).tolist()))
    summary["tasks_with_positive_rho_frac_of_arm_joints"] = {r["task"]: float(np.mean([r["per_joint"][n]["spearman_qs"] > 0 for n in arm_names])) for r in ok}
    grav = [v["R2_test_qs"] for r in ok for v in r["gravity_test_R2_by_joint"].values()]
    summary["gravity_test_R2_median_over_all_joints_tasks"] = float(np.nanmedian(grav))
    summary["gravity_test_R2_frac_below_0_3"] = float(np.nanmean(np.array(grav) < 0.3))
    for k in ("k2", "k3", "k4"):
        fr = [r["contact_candidates"][k][n]["flag_frac"] for r in ok for n in arm_names]
        summary[f"contact_flag_frac_{k}_median"] = float(np.median(fr))
    auc_p = [r["gripper_blocked_proxy"][a]["AUC_arm_resid_score_vs_gripper_blocked"] for r in ok for a in ("left", "right")]
    summary["gripper_blocked_proxy_AUC_per_arm"] = [float(x) for x in auc_p]
    summary["gripper_blocked_proxy_AUC_median"] = float(np.nanmedian(auc_p))

    results = {
        "script": "analysis/extra_aloha_tasks.py",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                     "pyarrow": pyarrow.__version__},
        "settings": {"qs_rule": "관절 속도 ≤ 해당 관절 중앙값", "gravity": "같은 팔 6관절 sin/cos (그리퍼 제외)",
                     "train": f"에피소드 앞 {int(TRAIN_FRAC * 100)}%", "k_sigma": list(K_LIST), "n_boot": N_BOOT,
                     "seed": SEED, "gripper_blocked_rule": "준정적 |δ_gripper − median| > 3·MAD",
                     "contact_label": "라벨 없음. 그리퍼 막힘 대리 라벨만 사용"},
        "summary": summary,
        "per_task": per_task,
        "wall_time_s": round(time.time() - t0, 1),
    }
    RESULTS.write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
