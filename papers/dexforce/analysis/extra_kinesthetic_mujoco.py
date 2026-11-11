"""추가 검증 (dexforce): MuJoCo 모의 kinesthetic 시연 → 힘 목표 재생에서 힘 배율 공식과 허용 창 확인.

실행: .venv/bin/python analysis/extra_kinesthetic_mujoco.py
결과: analysis/extra_kinesthetic_mujoco_results.json

- 팁: MuJoCo 슬라이드 관절(m = 0.05 kg), 위치 서보(k_p = 310 N/m, k_v = 2 N·s/m)로 구동. 시연은 가상 leader를
  적분 제어해 접촉력이 f_d(3 N)에 이르게 하고, 관측 위치 x_o(엔코더)와 접촉력 f를 기록한다.
- 물체: 벽(누름) 또는 두 면(집기). 접촉은 강성 k_e의 한쪽 스프링으로 계산해 팁에 외력으로 가한다
  (접촉 강성을 MuJoCo 접촉 파라미터로 맞추지 않으려고).
- 재생: 목표 x_f(t) = x_o(t) + k_f f(t)를 같은 시간열로 재생한다. 정상상태 힘 F를 시연 힘 f_d로 나눈 G를 잰다.
- 허용 창: 정상상태 힘이 [N_lo, N_hi] = [0.5, 5] N에 드는 k_f 범위를 k_f 격자에서 구한다.
  접촉 면 위치 오차 e를 재생 때 벽 위치에 더해 창 하한이 얼마나 움직이는지도 본다.
- 한계: 1 kHz보다 큰 dt(0.5 ms)를 쓰며, 끌기(마찰로 블록을 미는) 과제는 포함하지 않았다.
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

torch.set_num_threads(4)

RESULTS = Path(__file__).resolve().parent / "extra_kinesthetic_mujoco_results.json"
DT = 5e-4
KP, KV = 310.0, 2.0
M_TIP = 0.05
F_D = 3.0
N_LO, N_HI = 0.5, 5.0
DEMO_S, REPLAY_S = 2.5, 2.5
G_LEAD = 0.05                                  # 가상 leader 적분 이득 [m/(N·s)]
K_E_LIST = [1e3, 1e4, 1e5]
KF_GRID = np.linspace(0.0, 0.012, 61)

PRESS_XML = f"""
<mujoco model="press">
  <option timestep="{DT}" gravity="0 0 0" integrator="implicitfast"/>
  <worldbody>
    <body name="tip" pos="-0.02 0 0">
      <joint name="jt" type="slide" axis="1 0 0"/>
      <geom type="sphere" size="0.005" mass="{M_TIP}" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <actuator><position name="a" joint="jt" kp="{KP}" kv="{KV}"/></actuator>
</mujoco>
"""

PINCH_XML = f"""
<mujoco model="pinch">
  <option timestep="{DT}" gravity="0 0 0" integrator="implicitfast"/>
  <worldbody>
    <body name="objb" pos="0 0 0">
      <joint name="jo" type="slide" axis="1 0 0"/>
      <geom type="box" size="0.01 0.01 0.01" mass="0.05" contype="0" conaffinity="0"/>
    </body>
    <body name="tipA" pos="-0.03 0 0">
      <joint name="ja" type="slide" axis="1 0 0"/>
      <geom type="sphere" size="0.005" mass="{M_TIP}" contype="0" conaffinity="0"/>
    </body>
    <body name="tipB" pos="0.03 0 0">
      <joint name="jb" type="slide" axis="1 0 0"/>
      <geom type="sphere" size="0.005" mass="{M_TIP}" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <actuator>
    <position name="aa" joint="ja" kp="{KP}" kv="{KV}"/>
    <position name="ab" joint="jb" kp="{KP}" kv="{KV}"/>
  </actuator>
</mujoco>
"""


def g_formula(kf, ke):
    return (1 / ke + kf) / (1 / KP + 1 / ke)


def kf_window_formula(ke, f_d):
    """G(k_f) ∈ [N_lo/f_d, N_hi/f_d]을 k_f로 풀면 닫힌 구간이다."""
    lo = (N_LO / f_d) * (1 / KP + 1 / ke) - 1 / ke
    hi = (N_HI / f_d) * (1 / KP + 1 / ke) - 1 / ke
    return float(lo), float(hi)


# ---------- 누름(1 자유도 + 벽) ----------

def press_demo(ke, offset=0.0):
    m = mujoco.MjModel.from_xml_string(PRESS_XML)
    d = mujoco.MjData(m)
    n = int(DEMO_S / DT)
    xL = -0.02
    d.qpos[0] = xL                             # 팁을 leader와 같은 위치에서 시작
    d.ctrl[0] = xL
    mujoco.mj_forward(m, d)
    xo, f = np.zeros(n), np.zeros(n)
    for i in range(n):
        x = d.qpos[0]
        pen = max(0.0, x - offset)
        F = ke * pen
        xo[i], f[i] = x, F
        d.qfrc_applied[0] = -F
        # 가상 leader: 접촉 뒤에는 f를 F_D로 맞추도록 적분, 접촉 전에는 천천히 전진
        if F > 0.02 or pen > 0:
            xL += G_LEAD * (F_D - F) * DT
        else:
            xL += 0.02 * DT
        d.ctrl[0] = xL
        mujoco.mj_step(m, d)
    return xo, f, xL


def press_replay(ke, kf, xo, f, offset=0.0):
    m = mujoco.MjModel.from_xml_string(PRESS_XML)
    d = mujoco.MjData(m)
    d.qpos[0] = xo[0]
    d.ctrl[0] = xo[0]
    mujoco.mj_forward(m, d)
    n = int(REPLAY_S / DT)
    idx = np.minimum(np.arange(n), len(xo) - 1)
    Fs = np.zeros(n)
    for i in range(n):
        d.ctrl[0] = xo[idx[i]] + kf * f[idx[i]]
        x = d.qpos[0]
        F = ke * max(0.0, x - offset)
        Fs[i] = F
        d.qfrc_applied[0] = -F
        mujoco.mj_step(m, d)
    tail = Fs[-int(0.5 / DT):]
    return float(tail.mean())


# ---------- 집기(두 면) ----------

def pinch_demo(ke):
    m = mujoco.MjModel.from_xml_string(PINCH_XML)
    d = mujoco.MjData(m)
    n = int(DEMO_S / DT)
    xLA, xLB = -0.03, 0.03
    d.joint("ja").qpos[0] = xLA
    d.joint("jb").qpos[0] = xLB
    mujoco.mj_forward(m, d)
    xo_A, xo_B, fA, fB = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    yo = np.zeros(n)
    d.ctrl[:] = [xLA, xLB]
    w2 = 0.02                                  # 물체 반폭 (면 위치 y ± 0.02)
    for i in range(n):
        y = d.joint("jo").qpos[0]
        xa = d.joint("ja").qpos[0]
        xb = d.joint("jb").qpos[0]
        pA = max(0.0, xa - (y - w2))
        pB = max(0.0, (y + w2) - xb)
        FA, FB = ke * pA, ke * pB
        xo_A[i], xo_B[i], fA[i], fB[i], yo[i] = xa, xb, FA, FB, y
        d.joint("ja").qfrc_applied[0] = -FA
        d.joint("jb").qfrc_applied[0] = FB
        d.joint("jo").qfrc_applied[0] = FA - FB
        if FA > 0.02 or pA > 0:
            xLA += G_LEAD * (F_D - FA) * DT
        else:
            xLA += 0.02 * DT
        if FB > 0.02 or pB > 0:
            xLB -= G_LEAD * (F_D - FB) * DT
        else:
            xLB -= 0.02 * DT
        d.ctrl[:] = [xLA, xLB]
        mujoco.mj_step(m, d)
    return dict(xA=xo_A, xB=xo_B, fA=fA, fB=fB, y=yo)


def pinch_replay(ke, kf, demo):
    m = mujoco.MjModel.from_xml_string(PINCH_XML)
    d = mujoco.MjData(m)
    n = int(REPLAY_S / DT)
    idx = np.minimum(np.arange(n), len(demo["xA"]) - 1)
    d.joint("ja").qpos[0] = demo["xA"][0]
    d.joint("jb").qpos[0] = demo["xB"][0]
    d.joint("jo").qpos[0] = demo["y"][0]
    mujoco.mj_forward(m, d)
    d.ctrl[:] = [demo["xA"][0], demo["xB"][0]]
    w2 = 0.02
    FA_s, FB_s, ys = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        j = idx[i]
        d.ctrl[:] = [demo["xA"][j] + kf * demo["fA"][j], demo["xB"][j] - kf * demo["fB"][j]]
        xa = d.joint("ja").qpos[0]
        xb = d.joint("jb").qpos[0]
        y = d.joint("jo").qpos[0]
        pA = max(0.0, xa - (y - w2))
        pB = max(0.0, (y + w2) - xb)
        FA, FB = ke * pA, ke * pB
        FA_s[i], FB_s[i], ys[i] = FA, FB, y
        d.joint("ja").qfrc_applied[0] = -FA
        d.joint("jb").qfrc_applied[0] = FB
        d.joint("jo").qfrc_applied[0] = FA - FB
        mujoco.mj_step(m, d)
    tail = slice(-int(0.5 / DT), None)
    return float(FA_s[tail].mean()), float(FB_s[tail].mean()), float(np.abs(ys[tail]).max())


def main():
    t_all = time.perf_counter()
    out = {"env": {"mujoco": mujoco.__version__, "torch": torch.__version__, "numpy": np.__version__,
                   "device": "cpu", "threads": 4, "dt_s": DT, "k_p": KP, "k_v": KV, "f_demo_N": F_D,
                   "N_lo_N": N_LO, "N_hi_N": N_HI, "tip_mass_kg": M_TIP,
                   "contact": "one-sided spring k_e applied as external joint force (not MuJoCo contact)"}}

    # (1) 힘 배율 G: 재생 정상상태 / 시연 정상상태 vs 식
    rows = []
    for ke in K_E_LIST:
        xo, f, _ = press_demo(ke)
        fd = float(f[-int(0.5 / DT):].mean())
        for kf_name, kf in [("0", 0.0), ("1/k_p", 1 / KP), ("0.0045", 0.0045), ("0.01", 0.01)]:
            F = press_replay(ke, kf, xo, f)
            rows.append(dict(k_e=ke, k_f_name=kf_name, k_f=kf, demo_force_N=round(fd, 3),
                             replay_force_N=round(F, 4), G_measured=round(F / fd, 4),
                             G_formula=round(g_formula(kf, ke), 4)))
            print("G", rows[-1], flush=True)
    out["G_table"] = rows

    # (2) 허용 창: 누름, k_e별 k_f 격자, 그리고 접촉 면 위치 오차 e
    win = []
    for ke, off in [(1e3, 0.0), (1e4, 0.0), (1e5, 0.0), (1e4, 0.001), (1e4, 0.002)]:
        xo, f, _ = press_demo(ke)                  # 시연은 벽 위치 기준(오차 0)
        fd = float(f[-int(0.5 / DT):].mean())
        ok = []
        for kf in KF_GRID:
            F = press_replay(ke, kf, xo, f, offset=off)   # 재생 때만 벽 위치가 off만큼 어긋난다
            ok.append(N_LO <= F <= N_HI)
        ok = np.array(ok)
        idx = np.flatnonzero(ok)
        lo_f, hi_f = kf_window_formula(ke, fd)
        win.append(dict(k_e=ke, wall_offset_mm=off * 1e3, demo_force_N=round(fd, 3),
                        window_sim=(None if idx.size == 0 else [round(float(KF_GRID[idx[0]]), 5),
                                                                round(float(KF_GRID[idx[-1]]), 5)]),
                        contiguous=bool(idx.size == 0 or (idx[-1] - idx[0] + 1 == idx.size)),
                        window_formula_nominal=[round(lo_f, 5), round(hi_f, 5)],
                        k_f0_success=bool(ok[0]), n_grid=len(KF_GRID)))
        print("W", win[-1], flush=True)
    out["press_window"] = win

    # (3) 집기: 두 면, k_e별 k_f 격자와 k_f = 0 기준선
    pinch_rows = []
    for ke in [1e3, 1e4]:
        demo = pinch_demo(ke)
        fdA = float(demo["fA"][-int(0.5 / DT):].mean())
        fdB = float(demo["fB"][-int(0.5 / DT):].mean())
        ok = []
        for kf in KF_GRID:
            FA, FB, ymax = pinch_replay(ke, kf, demo)
            ok.append(bool(N_LO <= FA <= N_HI and N_LO <= FB <= N_HI and ymax < 0.005))
        ok = np.array(ok)
        idx = np.flatnonzero(ok)
        lo_f, hi_f = kf_window_formula(ke, fdA)
        pinch_rows.append(dict(k_e=ke, demo_force_A_N=round(fdA, 3), demo_force_B_N=round(fdB, 3),
                               window_sim=(None if idx.size == 0 else [round(float(KF_GRID[idx[0]]), 5),
                                                                       round(float(KF_GRID[idx[-1]]), 5)]),
                               window_formula_nominal=[round(lo_f, 5), round(hi_f, 5)],
                               k_f0_success=bool(ok[0]), n_grid=len(KF_GRID)))
        print("P", pinch_rows[-1], flush=True)
    out["pinch_window"] = pinch_rows
    out["wall_time_s"] = round(time.perf_counter() - t_all, 1)
    RESULTS.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print("wall", out["wall_time_s"], "s", flush=True)


if __name__ == "__main__":
    main()
