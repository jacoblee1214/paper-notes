"""추가 검증 (stiffness_copilot): MuJoCo 평면 2링크 팔의 임피던스 제어와 벽 접촉에서 강성 설정 비교.

실행: .venv/bin/python analysis/extra_copilot_mujoco.py
결과: analysis/extra_copilot_mujoco_results.json

- 팔: MuJoCo 2링크(수평면, 중력 없음). 관절 토크 = Jᵀ F, 제어 주기 1 kHz. 팁 질량 1 kg(축별 감쇠 D = 1.4√k와 맞춤).
- 벽: 법선 n(θ) = (sin θ, cos θ)인 평면. 접촉 스프링 k_env = 5000 N/m, 쿨롱 마찰 μ. 벽 힘은 팁에 외력으로 가한다.
  팔 모델과 벽의 접촉은 MuJoCo 접촉 모델을 쓰지 않고 스프링-마찰을 파이썬에서 계산한다(접촉 강성 보정 문제를 피하려고).
- 조작자: 1 s 동안 벽 표면 2 cm 위에서 −δ까지 내려가고, 이후 접선으로 0.05 m/s로 1 s 이동한다.
- 임피던스: F = K(x_d − x) + D(ẋ_d − ẋ), K = k_n n̂n̂ᵀ + k_t t̂t̂ᵀ, D의 축별 값은 1.4√k.
- 예측기: 작은 MLP(입력 4, 은닉 64×2)가 벽 양 끝점의 잡음 섞인 관측에서 (cos 2θ, sin 2θ)를 예측한다.
  20 Hz로 갱신하고 지수 평활(0.2)을 거쳐 K의 방향에 쓴다.
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
from torch import nn

torch.set_num_threads(4)

RESULTS = Path(__file__).resolve().parent / "extra_copilot_mujoco_results.json"
L1 = L2 = 0.30
BASE = np.array([0.0, 0.30])
DT = 0.001
T_END = 3.0
N_STEPS = int(T_END / DT)
K_ENV = 5000.0
MU = 0.4
MLP_UPDATE = 50                      # 20 Hz
TRIALS = 20

XML = f"""
<mujoco model="planar2link">
  <compiler angle="radian"/>
  <option timestep="{DT}" gravity="0 0 0" integrator="implicitfast"/>
  <default><joint damping="0.2" armature="0.01"/></default>
  <worldbody>
    <body name="base" pos="{BASE[0]} {BASE[1]} 0">
      <body name="l1">
        <joint name="j1" type="hinge" axis="0 0 1"/>
        <geom type="capsule" fromto="0 0 0 {L1} 0 0" size="0.012" mass="0.05" contype="0" conaffinity="0"/>
        <body name="l2" pos="{L1} 0 0">
          <joint name="j2" type="hinge" axis="0 0 1"/>
          <geom type="capsule" fromto="0 0 0 {L2} 0 0" size="0.010" mass="0.05" contype="0" conaffinity="0"/>
          <body name="tip" pos="{L2} 0 0">
            <geom name="tipg" type="sphere" size="0.01" mass="1.0" contype="0" conaffinity="0"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def ik(r):
    """기저 기준 상대 위치 r → 관절각 (팔꿈치 양의 해)."""
    c2 = np.clip((r @ r - L1 * L1 - L2 * L2) / (2 * L1 * L2), -1.0, 1.0)
    q2 = np.arccos(c2)
    q1 = np.arctan2(r[1], r[0]) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
    return np.array([q1, q2])


class Estimator(nn.Module):
    """벽 양 끝점 4개 → (cos 2θ, sin 2θ)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 2))

    def forward(self, x):
        return self.net(x)


def observe_endpoints(theta, sigma, rng):
    t = np.array([np.cos(theta), -np.sin(theta)])
    p1 = 0.1 * t + rng.normal(0, sigma, 2)
    p2 = -0.1 * t + rng.normal(0, sigma, 2)
    return np.concatenate([p1, p2]) / 0.1


def train_estimator(seed=0, steps=3000):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = 20000
    thetas = rng.uniform(-np.radians(40), np.radians(40), n)
    sigmas = rng.uniform(0.0, 0.006, n)
    X = np.stack([observe_endpoints(th, s, rng) for th, s in zip(thetas, sigmas)]).astype(np.float32)
    Y = np.stack([np.cos(2 * thetas), np.sin(2 * thetas)], 1).astype(np.float32)
    net = Estimator()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    for i in range(steps):
        idx = torch.randint(0, n, (256,))
        loss = ((net(Xt[idx]) - Yt[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    return net, float(loss.item())


def theta_from_pred(pred):
    return 0.5 * np.arctan2(pred[1], pred[0])


def sim_trial(model, data, bid_tip, theta, delta, mode, k_n, k_t, rng=None, sigma=0.0, e_deg=0.0,
              est=None):
    """mode: 'iso' (k_n = k_t), 'dir' (법선 k_n, 접선 k_t), 'dir_mlp' (법선 방향을 예측기로 추정)."""
    mujoco.mj_resetData(model, data)
    n_true = np.array([np.sin(theta), np.cos(theta)])
    t_true = np.array([np.cos(theta), -np.sin(theta)])
    x0 = 0.02 * n_true
    data.qpos[:2] = ik(x0 - BASE)
    mujoco.mj_forward(model, data)
    # 초기 법선 추정: 예측기 모드는 관측에서 시작, 나머지는 고정값
    if mode == "dir_mlp":
        th_hat = theta_from_pred(est(torch.tensor(observe_endpoints(theta, sigma, rng).astype(np.float32)[None]))
                                 .detach().numpy()[0])
    else:
        th_hat = theta + np.radians(e_deg)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    Fn_log, u_log, ud_log = np.zeros(N_STEPS), np.zeros(N_STEPS), np.zeros(N_STEPS)
    for k in range(N_STEPS):
        t = k * DT
        mujoco.mj_jacBody(model, data, jacp, jacr, bid_tip)
        J = jacp[:2, :2]
        x = data.xpos[bid_tip][:2].copy()
        v = J @ data.qvel[:2]
        # 조작자 목표
        dd = 0.02 - (0.02 + delta) * min(t, 1.0)
        dd_dot = -(0.02 + delta) if t < 1.0 else 0.0
        uu = 0.05 * max(t - 1.0, 0.0)
        uu_dot = 0.05 if t > 1.0 else 0.0
        x_d = dd * n_true + uu * t_true
        v_d = dd_dot * n_true + uu_dot * t_true
        # 추정 법선 갱신(20 Hz)
        if mode == "dir_mlp" and k % MLP_UPDATE == 0 and k > 0:
            obs = observe_endpoints(theta, sigma, rng).astype(np.float32)
            pred = est(torch.tensor(obs[None])).detach().numpy()[0]
            if k == MLP_UPDATE:
                c_s = pred
            else:
                c_s = 0.8 * c_s + 0.2 * pred
            th_hat = theta_from_pred(c_s)
        n_hat = np.array([np.sin(th_hat), np.cos(th_hat)])
        t_hat = np.array([np.cos(th_hat), -np.sin(th_hat)])
        if mode == "iso":
            K = k_n * np.eye(2)
            D = 1.4 * np.sqrt(k_n) * np.eye(2)
        else:
            K = k_n * np.outer(n_hat, n_hat) + k_t * np.outer(t_hat, t_hat)
            D = 1.4 * np.sqrt(k_n) * np.outer(n_hat, n_hat) + 1.4 * np.sqrt(k_t) * np.outer(t_hat, t_hat)
        F_cmd = K @ (x_d - x) + D @ (v_d - v)
        tau = J.T @ F_cmd
        data.qfrc_applied[:2] = tau
        # 벽: 스프링 + 쿨롬 마찰 (팁에 외력)
        dn = x @ n_true
        pen = max(0.0, -dn)
        Fn = K_ENV * pen
        vt = v @ t_true
        Ft = -MU * Fn * np.tanh(vt / 0.005)
        F_env = Fn * n_true + Ft * t_true
        data.xfrc_applied[bid_tip, :2] = F_env
        Fn_log[k] = Fn
        u_log[k] = x @ t_true
        ud_log[k] = uu
        mujoco.mj_step(model, data)
    t_ss = np.arange(N_STEPS) * DT
    win = (t_ss >= 2.0) & (t_ss < 3.0)
    return dict(F_peak=float(Fn_log.max()), F_ss=float(Fn_log[win].mean()),
                track_rms_mm=float(np.sqrt(np.mean((ud_log[win] - u_log[win]) ** 2)) * 1e3),
                theta_hat_err_deg=float(np.degrees(np.angle(np.exp(2j * (th_hat - theta))) / 2)))


def stats(rows, key):
    a = np.array([r[key] for r in rows])
    if key == "theta_hat_err_deg":
        a = np.abs(a)                        # 부호 평균은 0에 가까워 잡음을 가리므로 절댓값을 쓴다
    return {"mean": round(float(a.mean()), 4), "sd": round(float(a.std(ddof=1)), 4) if len(a) > 1 else 0.0}


def main():
    t_all = time.perf_counter()
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    bid_tip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tip")
    out = {"env": {"mujoco": mujoco.__version__, "torch": torch.__version__, "numpy": np.__version__,
                   "device": "cpu", "threads": 4, "dt_s": DT, "episode_s": T_END, "mu": MU, "k_env": K_ENV,
                   "tip_mass_kg": 1.0, "wall": "spring + Coulomb friction applied as external force on tip"}}

    # (A) θ = 0, δ 세 값, 법선 정확: 세 설정 (03 §4.2–4.3의 폐루프 대응)
    rows_a = {}
    for delta in (0.005, 0.01, 0.02):
        for name, mode, kn, kt in [("고정 고강성 3000/3000", "iso", 3000.0, 3000.0),
                                   ("고정 저강성 300/300", "iso", 300.0, 300.0),
                                   ("방향별 법선 300 / 접선 3000", "dir", 300.0, 3000.0)]:
            r = sim_trial(model, data, bid_tip, 0.0, delta, mode, kn, kt)
            rows_a.setdefault(f"delta_{int(delta * 1000)}mm", {})[name] = {
                k: round(v, 4) for k, v in r.items() if k != "theta_hat_err_deg"}
            print("A", delta, name, rows_a[f"delta_{int(delta * 1000)}mm"][name], flush=True)
    out["A_theta0_delta_sweep"] = rows_a
    out["A_analytic_F_ss_N"] = {f"delta_{d}mm": {
        "iso_300": round(5000 * 300 * d / (300 + 5000), 3), "iso_3000": round(5000 * 3000 * d / (3000 + 5000), 3),
        "dir_normal_300": round(5000 * 300 * d / (300 + 5000), 3)} for d in (5, 10, 20)}

    # (B) 법선 무작위(±30°), δ = 1 cm, 시험 20회 (짝지음), 예측기 학습
    est, est_loss = train_estimator(seed=0)
    rng_test = np.random.default_rng(7)
    conds = [("고정 고강성 3000/3000", "iso", 3000.0, 3000.0, 0.0, None),
             ("고정 저강성 300/300", "iso", 300.0, 300.0, 0.0, None),
             ("방향별, 법선 정확", "dir", 300.0, 3000.0, 0.0, None),
             ("방향별, 법선 고정 오차 +20°", "dir", 300.0, 3000.0, 20.0, None),
             ("방향별, 예측기, 끝점 잡음 0 mm", "dir_mlp", 300.0, 3000.0, 0.0, 0.0),
             ("방향별, 예측기, 끝점 잡음 2 mm", "dir_mlp", 300.0, 3000.0, 0.0, 0.002),
             ("방향별, 예측기, 끝점 잡음 5 mm", "dir_mlp", 300.0, 3000.0, 0.0, 0.005)]
    thetas = rng_test.uniform(-np.radians(30), np.radians(30), TRIALS)
    rows_b = {}
    for name, mode, kn, kt, e_deg, sig in conds:
        rs = []
        for i, th in enumerate(thetas):
            rng = np.random.default_rng(500 + i)
            rs.append(sim_trial(model, data, bid_tip, th, 0.01, mode, kn, kt, rng=rng, sigma=sig or 0.0,
                                e_deg=e_deg, est=est))
        rows_b[name] = {k: stats(rs, k) for k in ("F_peak", "F_ss", "track_rms_mm", "theta_hat_err_deg")}
        rows_b[name]["theta_hat_err_deg"]["note"] = "절댓값 평균"
        print("B", name, {k: v["mean"] for k, v in rows_b[name].items()}, flush=True)
    out["B_random_normal_delta1cm"] = rows_b
    out["B_theta_range_deg"] = 30
    out["B_trials"] = TRIALS
    out["B_estimator"] = {"layers": "4-64-64-2 tanh", "train_steps": 3000, "final_batch_mse": round(est_loss, 5)}
    out["wall_time_s"] = round(time.perf_counter() - t_all, 1)
    RESULTS.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print("wall", out["wall_time_s"], "s", flush=True)


if __name__ == "__main__":
    main()
