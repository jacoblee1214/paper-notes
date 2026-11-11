"""추가 검증 (beyond_implicit_force): 강성 판별 정책에 반사실 토크를 넣는 개입 실험.

실행: .venv/bin/python analysis/extra_counterfactual_tau.py
     (먼저 analysis/extra_mujoco_wipe.py를 실행해 .scratch/mujoco_wipe/의 정책 파일을 만들어야 한다.)
결과: analysis/extra_counterfactual_tau_results.json

질문: 토크 입력을 쓰는 정책이 그 토크를 실제로 따르는가? 같은 짝(seed, 시도 번호)에서
  (a) 무른 물체 시도에 단단한 물체의 토크 시간열을 주입하면 물러나는가,
  (b) 단단한 물체 시도에 무른 물체의 토크 시간열을 주입하면 누르는가,
  (c) 토크를 0으로 고정하면 두 물체의 결과가 어떻게 되는가.
주입은 시간 맞춤(open loop)이고, 팔의 움직임은 정책이 폐루프로 결정한다. 접촉 시작 시각은 두 물체가 같은 속도로
내려오므로 거의 같다. 정책 세 개(시드 3)와 시도 30쌍을 쓴다. 평가 시드는 학습 시드와 다르다.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "4"

import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.stats import fisher_exact

import extra_mujoco_wipe as E

torch.set_num_threads(4)

RESULTS = Path(__file__).resolve().parent / "extra_counterfactual_tau_results.json"
POLICIES = ["ACT + τ", "ACT-o + τ"]
SEEDS = [0, 1, 2]
N_PAIRS = 30


def load_policy(variant, seed):
    name = f"stiff_{variant.replace(' ', '').replace('+', 'p')}_s{seed}.pt"
    ck = torch.load(E.SCRATCH / name, weights_only=False)
    din = ck["state"]["net.0.weight"].shape[1]
    net = E.MLP(din, E.H * 2)
    net.load_state_dict(ck["state"])
    net.eval()
    return net, ck["norm"]


def rollout(arm, net, norm, variant, obj_k, mode, rng, tau_trace=None):
    """mode: 'obs' 실제 토크, 'zero' 토크 0 고정, 'inject' tau_trace 주입. 각 제어 스텝의 관측 토크를 기록한다."""
    spec = E.VARIANTS[variant]
    x0 = rng.uniform(-0.02, 0.02)                # 학습 시연과 같은 시작 위치 분포
    arm.reset(E.ik(np.array([x0, 0.25 - E.BASE_Z])), obj_k=obj_k)
    T = int(E.STIFF_T / E.DT_CTRL)
    qh = np.stack([arm.d.qpos[:2].copy()] * 5)
    Fs, taus = [], []
    chunk = None
    for t in range(T):
        tau_obs = arm.tau() + rng.normal(0, E.TAU_NOISE, 2)
        if t % E.REPLAN == 0:
            q_now = arm.d.qpos[:2].copy() + rng.normal(0, E.Q_NOISE, 2)
            if mode == "zero":
                tau_in = np.zeros(2)
            elif mode == "inject":
                tau_in = tau_trace[t]
            else:
                tau_in = tau_obs
            qh = np.concatenate([[q_now], qh[:4]])
            chunk = E.predict_chunk(net, norm, E.obs_vec(spec["feat"], qh, tau_in), q_now)
        taus.append(tau_obs)
        Fs.append(arm.tip_force())
        arm.step(chunk[t % E.REPLAN])
    return dict(F=np.array(Fs), tau=np.array(taus), tip_z=float(arm.tip_xz()[1]))


def outcome(ep, soft):
    """soft: 눌림 성공 / 물러남. hard: 물러남 성공 / 계속 누름(끝에도 접촉 유지, 최대 힘 > 5.5)."""
    F = ep["F"]
    end_mean = float(F[-int(1.0 / E.DT_CTRL):].mean())
    if soft:
        if E.stiff_success(ep, True):
            return "pressed"
        if end_mean < 1.0 and F.max() >= 2.0:
            return "retreated"
        return "other"
    if E.stiff_success(ep, False):
        return "retreated"
    if end_mean >= 1.0 and F.max() > 5.5:
        return "kept_pressing"
    return "other"


def rate(outs, key):
    return float(np.mean([o == key for o in outs])) if outs else None


def onset_check(arm, n=60):
    """접촉 시작(F ≥ 1.8 N 첫 시점)의 토크 크기와 5스텝 관절 이동량이 무른/단단한 물체에서 얼마나 다른지."""
    rng = np.random.default_rng(11)
    rows = {True: [], False: []}
    for i in range(n):
        soft = i % 2 == 0
        d = E.demo_stiff(arm, rng, soft=soft)
        idx = np.flatnonzero(d["F"] > 1.8)
        if idx.size == 0:
            continue
        t = idx[0]
        rows[soft].append((float(np.linalg.norm(d["tau"][t])),
                           float(np.linalg.norm(d["q"][t] - d["q"][max(t - 5, 0)]))))
    res = {}
    for soft, name in [(True, "soft"), (False, "hard")]:
        a = np.array(rows[soft])
        res[name] = {"n": int(len(a)), "tau_norm_mean_Nm": round(float(a[:, 0].mean()), 3),
                     "tau_norm_sd_Nm": round(float(a[:, 0].std()), 3),
                     "dq_5steps_mean_rad": round(float(a[:, 1].mean()), 4),
                     "dq_5steps_sd_rad": round(float(a[:, 1].std()), 4)}
    return res


def main():
    t_all = time.perf_counter()
    arm = E.Arm(True)
    out = {"env": {"torch": torch.__version__, "mujoco": mujoco.__version__, "device": "cpu", "threads": 4,
                   "n_pairs": N_PAIRS, "policy_seeds": SEEDS, "policies": POLICIES,
                   "eval_seed_base": 100000}, "onset_check": onset_check(arm), "results": {}}
    for variant in POLICIES:
        out["results"][variant] = {}
        per = {c: {"soft": [], "hard": []} for c in
               ["baseline", "inject_hard_tau_into_soft", "inject_soft_tau_into_hard", "tau_zero"]}
        for seed in SEEDS:
            net, norm = load_policy(variant, seed)
            for i in range(N_PAIRS):
                rng_k = np.random.default_rng(100000 + 1000 * seed + i)
                k_soft = rng_k.uniform(*E.K_SOFT)
                k_hard = rng_k.uniform(*E.K_HARD)
                ob = lambda tag: np.random.default_rng(100000 + 1000 * seed + 10 * i + tag)
                ep_hard = rollout(arm, net, norm, variant, k_hard, "obs", ob(1))
                ep_soft = rollout(arm, net, norm, variant, k_soft, "obs", ob(2))
                per["baseline"]["hard"].append(outcome(ep_hard, False))
                per["baseline"]["soft"].append(outcome(ep_soft, True))
                tr_hard = ep_hard["tau"]
                tr_soft = ep_soft["tau"]
                per["inject_hard_tau_into_soft"]["soft"].append(
                    outcome(rollout(arm, net, norm, variant, k_soft, "inject", ob(3), tau_trace=tr_hard), True))
                per["inject_soft_tau_into_hard"]["hard"].append(
                    outcome(rollout(arm, net, norm, variant, k_hard, "inject", ob(4), tau_trace=tr_soft), False))
                per["tau_zero"]["soft"].append(outcome(rollout(arm, net, norm, variant, k_soft, "zero", ob(5)), True))
                per["tau_zero"]["hard"].append(outcome(rollout(arm, net, norm, variant, k_hard, "zero", ob(6)), False))
        # 집계: 소프트 물체의 물러남 비율, 단단한 물체의 누름 유지 비율, 눌림/물러남 성공률
        summ = {}
        for c, d in per.items():
            s, h = d["soft"], d["hard"]
            summ[c] = {
                "soft_n": len(s), "soft_pressed": rate(s, "pressed"), "soft_retreated": rate(s, "retreated"),
                "hard_n": len(h), "hard_retreated": rate(h, "retreated"), "hard_kept_pressing": rate(h, "kept_pressing"),
            }
        b, si, hi, z = (summ[k] for k in ["baseline", "inject_hard_tau_into_soft",
                                          "inject_soft_tau_into_hard", "tau_zero"])
        bs = per["baseline"]["soft"]
        ss_inj = per["inject_hard_tau_into_soft"]["soft"]
        bh = per["baseline"]["hard"]
        hh_inj = per["inject_soft_tau_into_hard"]["hard"]
        a_tab = [[sum(o == "retreated" for o in bs), sum(o != "retreated" for o in bs)],
                 [sum(o == "retreated" for o in ss_inj), sum(o != "retreated" for o in ss_inj)]]
        b_tab = [[sum(o == "retreated" for o in bh), sum(o != "retreated" for o in bh)],
                 [sum(o == "retreated" for o in hh_inj), sum(o != "retreated" for o in hh_inj)]]
        summ["fisher_soft_retreat_baseline_vs_hard_tau_p"] = float(fisher_exact(a_tab)[1])
        summ["fisher_hard_retreat_baseline_vs_soft_tau_p"] = float(fisher_exact(b_tab)[1])
        out["results"][variant] = summ
        print(variant, json.dumps(summ, ensure_ascii=False), flush=True)
    out["wall_time_s"] = round(time.perf_counter() - t_all, 1)
    RESULTS.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print("wall", out["wall_time_s"], "s", flush=True)


if __name__ == "__main__":
    main()
