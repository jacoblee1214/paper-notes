"""식 (2)의 수치 확인과 위치 손실 대 토크 손실의 소규모 CPU 장난감 비교 (K1-K3).

K1: 심플렉틱 오일러에서 q_{t+1} - q̂_{t+1} = Δt² M(q_t)^-1 (τ_t - τ̂_t) 가 기계 정밀도로 성립하는지 (논문 식 2, 부록 D)
K2: 4자유도 팔의 질량행렬 M(q)의 이방성 — 같은 크기의 토크 오차가 방향에 따라 위치 오차를 얼마나 다르게 만드는가
K3: 작은 MLP(용량 제한)를 토크 손실 / 위치 손실로 각각 학습했을 때 1스텝 위치·토크 오차 비교
    (관측 위치에 인코더 잡음을 넣고, 토크 손실의 라벨은 잡음 낀 위치의 유한차분 역동역학으로 만든다)

모델 파일: 저자 저장소의 MJCF(pamy_no_muscles.xml, racket.xml)를 데이터로 읽을 수 있으면 사용하고
(--assets <xml 폴더>), 없으면 자체 4링크 사슬을 쓴다. 저장소의 코드는 실행·import하지 않는다.
실행: python3 toy_position_vs_torque_loss.py [--assets <xml 폴더>] [--seeds 3]   (mujoco, torch, numpy; CPU 2분 이내)
"""
import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
DT = 0.002
DEG = np.pi / 180.0
Q_LO = np.array([-90, -75, -85, -85]) * DEG  # 논문 표 3의 관절 한계
Q_HI = np.array([90, 85, 85, 85]) * DEG

FALLBACK_XML = """
<mujoco><option timestep="0.002" gravity="0 0 -9.81"/>
<worldbody><body pos="0 0 1.2"><joint name="j1" axis="1 0 0"/>
<inertial pos="0.01 0 0" mass="1.37" diaginertia="0.0025 0.0015 0.0015"/>
<body pos="0 0 0"><joint name="j2" axis="1 0 0"/><inertial pos="0 0.2 0" mass="0.87" diaginertia="0.018 0.018 0.0002"/>
<body pos="0 0.4 0"><joint name="j3" axis="1 0 0"/><inertial pos="0.004 0.02 0" mass="0.29" diaginertia="0.0002 0.00016 0.00009"/>
<body pos="0 0.05 0"><joint name="j4" axis="1 0 0"/><inertial pos="0.11 0 0" mass="0.5" diaginertia="0.00095 0.00095 0.00003"/>
</body></body></body></body></worldbody>
<actuator><motor joint="j1"/><motor joint="j2"/><motor joint="j3"/><motor joint="j4"/></actuator></mujoco>
"""


def build_model(assets):
    if assets:
        spec = mujoco.MjSpec.from_file(str(Path(assets) / "pamy_no_muscles.xml"))
        racket = mujoco.MjSpec.from_file(str(Path(assets) / "racket.xml"))
        j4 = spec.body("J4")
        frame = j4.add_frame()
        frame.attach_body(racket.body("racket"), "", "")
        spec.add_exclude(bodyname1="robot_arm", bodyname2="racket")
        for d in range(1, 5):
            spec.add_exclude(bodyname1=f"J{d}", bodyname2="racket")
        spec.option.timestep = DT
        return spec.compile(), "authors_mjcf(racket)"
    m = mujoco.MjModel.from_xml_string(FALLBACK_XML)
    return m, "fallback_4link"


def mass_matrix(m, d):
    M = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, M)
    return M


def k1_eq2(m, rng, n=300):
    d = mujoco.MjData(m)
    errs, rel = [], []
    for _ in range(n):
        q = rng.uniform(Q_LO, Q_HI) * 0.7
        v = rng.uniform(-2, 2, 4)
        tau = rng.uniform(-3, 3, 4)
        dtau = rng.normal(0, 0.5, 4)
        # 실제(τ)로 한 스텝: q_{t+1}
        d.qpos[:], d.qvel[:], d.ctrl[:] = q, v, tau
        mujoco.mj_forward(m, d)
        M = mass_matrix(m, d)
        mujoco.mj_step(m, d)
        q_next = d.qpos.copy()
        # 데이터셋 방식 라벨: 역동역학 (v_t = (q_t-q_{t-1})/dt, a_t = 중앙차분)
        q_prev = q - DT * v
        a = (q_next - 2 * q + q_prev) / DT**2
        d.qpos[:], d.qvel[:], d.qacc[:] = q, v, a
        mujoco.mj_inverse(m, d)
        tau_label = d.qfrc_inverse.copy()
        # 예측 토크 τ̂ = τ_label + δ 로 한 스텝
        d.qpos[:], d.qvel[:], d.ctrl[:] = q, v, tau_label + dtau
        mujoco.mj_step(m, d)
        q_hat = d.qpos.copy()
        lhs = q_next - q_hat
        rhs = DT**2 * np.linalg.solve(M, -dtau)  # τ - τ̂ = -δ
        errs.append(np.max(np.abs(lhs - rhs)))
        rel.append(np.linalg.norm(lhs - rhs) / (np.linalg.norm(rhs) + 1e-30))
        # 라벨 자체가 원래 토크(액추에이터 ctrl)와 같은지
    return {"n": n, "max_abs_resid_rad": float(np.max(errs)), "max_rel_resid": float(np.max(rel)),
            "label_equals_applied_torque": None}


def k1b_label(m, rng, n=100):
    d = mujoco.MjData(m)
    worst = 0.0
    for _ in range(n):
        q = rng.uniform(Q_LO, Q_HI) * 0.7
        v = rng.uniform(-2, 2, 4)
        tau = rng.uniform(-3, 3, 4)
        d.qpos[:], d.qvel[:], d.ctrl[:] = q, v, tau
        mujoco.mj_step(m, d)
        q_next = d.qpos.copy()
        q_prev = q - DT * v
        a = (q_next - 2 * q + q_prev) / DT**2
        d.qpos[:], d.qvel[:], d.qacc[:] = q, v, a
        mujoco.mj_inverse(m, d)
        worst = max(worst, float(np.max(np.abs(d.qfrc_inverse - tau))))
    return worst


def k2_anisotropy(m, rng, n=2000):
    d = mujoco.MjData(m)
    conds, offd, ratios = [], [], []
    diag_mean = []
    for _ in range(n):
        q = rng.uniform(Q_LO, Q_HI)
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        M = mass_matrix(m, d)
        ev = np.linalg.eigvalsh(M)
        conds.append(ev[-1] / ev[0])
        s = np.sqrt(np.diag(M))
        C = np.abs(M / np.outer(s, s)) - np.eye(4)
        offd.append(C.max())
        diag_mean.append(np.diag(M))
        # 같은 크기(‖δτ‖=1)의 무작위 방향 토크 오차 → 위치 오차 크기의 변동
        Minv = np.linalg.inv(M)
        dirs = rng.normal(size=(200, 4))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        mag = np.linalg.norm(dirs @ Minv.T, axis=1) * DT**2
        ratios.append(mag.max() / mag.min())
    diag_mean = np.array(diag_mean)
    return {
        "n_configs": n,
        "cond_M_median": float(np.median(conds)), "cond_M_p10": float(np.percentile(conds, 10)), "cond_M_p90": float(np.percentile(conds, 90)),
        "max_normalized_offdiag_median": float(np.median(offd)), "max_normalized_offdiag_p90": float(np.percentile(offd, 90)),
        "equal_norm_torque_err_to_position_err_ratio_max_over_min_median": float(np.median(ratios)),
        "M_diag_median": np.median(diag_mean, axis=0).tolist(),
        "cond_M_list": [float(c) for c in conds],
        "ratio_list": [float(r) for r in ratios],
    }


def sample_states(m, rng, n, noise_deg):
    """무작위 상태에서 한 스텝 데이터를 만든다. 반환: 깨끗한/잡음 낀 q_{t-1}, q_t, q_{t+1}, 제어 이력, 진짜 토크."""
    d = mujoco.MjData(m)
    q = rng.uniform(Q_LO, Q_HI, size=(n, 4)) * 0.8
    v = rng.uniform(-2.5, 2.5, size=(n, 4))
    u = rng.uniform(-1, 1, size=(n, 4))
    du = rng.normal(0, 0.02, size=(n, 4))  # u_t - u_{t-1}
    Minv = np.zeros((n, 4, 4))
    bias = np.zeros((n, 4))
    qn = np.zeros((n, 4))
    # 숨은 액추에이터(비선형 + 이력): τ* = A tanh(a u) + H du/dt - 마찰 항 (관절별 상수)
    A = np.array([4.0, 6.0, 3.0, 1.5])
    Hh = np.array([60.0, 90.0, 40.0, 15.0])
    Fc = np.array([0.4, 0.8, 0.5, 0.15])
    tau = A * np.tanh(1.7 * u) + Hh * du - Fc * np.tanh(v / 0.05)
    for i in range(n):
        d.qpos[:], d.qvel[:], d.qacc[:] = q[i], v[i], 0
        mujoco.mj_inverse(m, d)
        bias[i] = d.qfrc_inverse  # 중력+코리올리 (qacc=0)
        mujoco.mj_forward(m, d)
        Minv[i] = np.linalg.inv(mass_matrix(m, d))
        acc = Minv[i] @ (tau[i] - bias[i])
        v1 = v[i] + DT * acc
        qn[i] = q[i] + DT * v1
    qp = q - DT * v
    s = noise_deg * DEG
    noisy = lambda x: x + rng.normal(0, s, x.shape)
    return dict(q=q, v=v, u=u, du=du, tau=tau, Minv=Minv, bias=bias, qn=qn, qp=qp, nq=noisy(q), nqn=noisy(qn), nqp=noisy(qp))


def train_eval(tr, te, loss_kind, seed, hidden=24, steps=2500):
    torch.manual_seed(seed)

    def feats(D):
        # 입력: 위치, 위치 변화(속도 대용), 제어, 제어 변화 -- (관측된 잡음 낀 위치 사용)
        dq = (D["nq"] - D["nqp"]) / DT
        return np.concatenate([D["nq"], dq / 3.0, D["u"], D["du"] * 20.0], axis=1)

    Xtr = torch.tensor(feats(tr), dtype=torch.float32)
    Xte = torch.tensor(feats(te), dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, 4))
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    T = lambda a: torch.tensor(a, dtype=torch.float32)

    # 측정된(잡음 낀) 값으로 정의되는 양들
    nq, nqp, nqn = T(tr["nq"]), T(tr["nqp"]), T(tr["nqn"])
    v_meas = (nq - nqp) / DT
    Minv_tr, bias_tr = T(tr["Minv"]), T(tr["bias"])  # 질량행렬·편향은 모델에서 계산(잡음 없는 q로; 단순화)
    # 토크 라벨: 유한차분 역동역학 (잡음 낀 위치)
    a_meas = (nqn - 2 * nq + nqp) / DT**2
    M_tr = torch.linalg.inv(Minv_tr)
    tau_label = (M_tr @ a_meas.unsqueeze(-1)).squeeze(-1) + bias_tr
    mu, sd = tau_label.mean(0), tau_label.std(0)

    for it in range(steps):
        idx = torch.randint(0, Xtr.shape[0], (512,))
        pred = net(Xtr[idx])
        if loss_kind == "torque":
            loss = (((pred - (tau_label[idx] - mu) / sd)) ** 2).mean()
        else:
            tau_hat = pred * sd + mu
            acc = (Minv_tr[idx] @ (tau_hat - bias_tr[idx]).unsqueeze(-1)).squeeze(-1)
            q_hat = nq[idx] + DT * (v_meas[idx] + DT * acc)
            loss = (((q_hat - nqn[idx]) / (DT**2 * 1.0)) ** 2).mean() * 1e-4  # 스케일만 정리
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = net(Xte) * sd + mu
        Minv_te, bias_te = T(te["Minv"]), T(te["bias"])
        nq_te, nqp_te = T(te["nq"]), T(te["nqp"])
        v_te = (nq_te - nqp_te) / DT
        acc = (Minv_te @ (pred - bias_te).unsqueeze(-1)).squeeze(-1)
        q_hat = nq_te + DT * (v_te + DT * acc)
        err_meas = (q_hat - T(te["nqn"])).abs().mean().item() / DEG  # 논문 방식: 측정 위치와 비교
        err_clean = (q_hat - T(te["qn"])).abs().mean().item() / DEG  # 잡음 없는 정답과 비교
        tau_err = (pred - T(te["tau"])).abs().mean().item()
    return err_meas, err_clean, tau_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--noise_deg", type=float, default=0.0005)
    ap.add_argument("--ntrain", type=int, default=20000)
    args = ap.parse_args()
    t0 = time.time()
    m, kind = build_model(args.assets)
    assert m.nv == 4, m.nv
    rng = np.random.default_rng(0)
    out = {"model": kind, "nv": int(m.nv), "mass_total_kg": float(m.body_mass.sum())}
    out["K1_eq2"] = k1_eq2(m, rng)
    out["K1_label_vs_applied_torque_max_abs_Nm"] = k1b_label(m, rng)
    out["K2_anisotropy"] = k2_anisotropy(m, rng)
    sweep = []
    for noise in (args.noise_deg, args.noise_deg * 10):
        tr = sample_states(m, rng, args.ntrain, noise)
        te = sample_states(m, rng, 4000, noise)
        for hidden in (8, 24):
            res = {"torque": [], "position": []}
            for seed in range(args.seeds):
                for k in ("torque", "position"):
                    res[k].append(train_eval(tr, te, k, seed, hidden=hidden))
            row = {"noise_deg": noise, "hidden": hidden}
            for k, v in res.items():
                a_ = np.array(v)
                row[k] = {"pos_err_vs_measured_deg": [float(a_[:, 0].mean()), float(a_[:, 0].std())],
                          "pos_err_vs_clean_deg": [float(a_[:, 1].mean()), float(a_[:, 1].std())],
                          "torque_abs_err_Nm": [float(a_[:, 2].mean()), float(a_[:, 2].std())]}
            row["ratio_position_over_torque_pos_err_measured"] = row["position"]["pos_err_vs_measured_deg"][0] / row["torque"]["pos_err_vs_measured_deg"][0]
            row["ratio_position_over_torque_torque_err"] = row["position"]["torque_abs_err_Nm"][0] / row["torque"]["torque_abs_err_Nm"][0]
            sweep.append(row)
    out["K3_toy_training"] = {"ntrain": args.ntrain, "seeds": args.seeds, "sweep": sweep}
    out["seconds"] = round(time.time() - t0, 1)
    (HERE / "toy_position_vs_torque_loss_results.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "K3_toy_training"}, indent=1))
    for r in sweep:
        print("K3", r["noise_deg"], r["hidden"], "pos_err(meas) 토크손실 %.5f 위치손실 %.5f 비 %.3f | 토크오차 비 %.2f" % (
            r["torque"]["pos_err_vs_measured_deg"][0], r["position"]["pos_err_vs_measured_deg"][0],
            r["ratio_position_over_torque_pos_err_measured"], r["ratio_position_over_torque_torque_err"]))


if __name__ == "__main__":
    main()
