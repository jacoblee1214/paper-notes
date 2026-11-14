"""추가 검증 (dexforce): 마찰 레일 위 큐브 밀기에서 힘 목표 재생의 k_f와 마찰 허용 범위.

실행: .venv/bin/python analysis/extra_slide_cube_mujoco.py
결과: analysis/extra_slide_cube_mujoco_results.json

- 과제: 손가락 하나가 레일(슬라이드 관절 하나) 위의 정사각 큐브를 목표 구역(중심 D = 6 cm)까지 민다.
  끝판은 없다. 레일 마찰은 관절 마찰(frictionloss = μ·m·g)로 준다.
- 손가락: 구 하나를 슬라이드 관절에 붙이고 위치 서보(k_p = 310 N/m, k_v = 2 N·s/m)로 구동한다.
  접촉 힘은 MuJoCo 접촉 기록(mj_contactForce)에서 손가락–큐브 법선 힘으로 읽는다.
- 시연: 가상 사람이 리더를 목표 근처까지 min-jerk로 움직인다. 시연마다 이동 시간 T와 멈춤 위치 여유 Δ를
  다르게 뽑아서 힘 프로파일이 달라지게 했다. 손가락 측정 위치 x_o와 접촉 힘 f를 기록한다. 시연 마찰 μ = 0.5.
- 재생: 목표 x_f(t) = x_o(t) + k_f f(t)를 같은 시각으로 재생한다. k_f = 0은 위치만 쓰는 재생이다.
  재생 마찰 μ를 바꾸고, 큐브 초기 위치를 ±2 mm 바꿔서 시험한다.
- 성공: 재생 끝의 큐브 위치가 목표 ±3 mm 안에 있을 때. 너무 약하면 정지(stall), 너무 세면 넘침(overshoot).
- 측정: 성공률, 정지 거리, 넘침 거리, 접촉 상실 시간, 힘 추적 RMSE(시연 힘 대비).
- 한계: 한 손가락, 1 자유도 레일, 손가락–큐브 접촉 마찰 0.5 고정, 접촉 강성은 MuJoCo 기본값.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "4"

import json
import time
from pathlib import Path

import mujoco
import numpy as np

RESULTS = Path(__file__).resolve().parent / "extra_slide_cube_mujoco_results.json"
DT = 1e-3
G = 9.81
M_CUBE = 0.3
HALF = 0.02
M_TIP = 0.05
R_TIP = 0.008
KP, KV = 310.0, 2.0
D_GOAL = 0.06
X_TIP0 = -0.03                        # 시연 시작 시 손가락 위치 (큐브 면과 2 mm 떨어짐)
MU_DEMO = 0.5
MU_TIP = 0.5
SEEDS = [0, 1, 2]
N_DEMO = 5
DEMO_S = 2.5
DELTA_RANGE = (0.0, 0.003)            # 리더가 목표 접촉 위치를 넘어 더 가는 양 [m]
T_RANGE = (1.0, 1.4)                  # 리더 이동 시간 [s] (min-jerk)
GOAL_TOL = 0.003
MU_LIST = [0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
KF_LIST = [0.0, 0.001, 0.002, 1 / KP, 0.005, 0.008]
PLACEMENTS = [-0.002, 0.0, 0.002]     # 재생 때 큐브 시작 위치 오차 [m]


def build(mu_rail):
    xml = f"""
<mujoco model="slide">
  <option timestep="{DT}" gravity="0 0 {-G}" integrator="implicitfast"/>
  <worldbody>
    <body name="cube" pos="0 0 {HALF}">
      <joint name="jc" type="slide" axis="1 0 0" frictionloss="{mu_rail * M_CUBE * G}"/>
      <geom name="cube_g" type="box" size="{HALF} {HALF} {HALF}" mass="{M_CUBE}" friction="{MU_TIP} 0.005 0.0001"/>
    </body>
    <body name="tip" pos="0 0 {HALF}">
      <joint name="jt" type="slide" axis="1 0 0"/>
      <geom name="tip_g" type="sphere" size="{R_TIP}" mass="{M_TIP}" friction="{MU_TIP} 0.005 0.0001"/>
    </body>
  </worldbody>
  <actuator><position name="a" joint="jt" kp="{KP}" kv="{KV}"/></actuator>
</mujoco>
"""
    m = mujoco.MjModel.from_xml_string(xml)
    return m


def tip_cube_force(m, d):
    """손가락–큐브 접촉의 법선 힘 크기 [N]."""
    gt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "tip_g")
    gc = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cube_g")
    f6 = np.zeros(6)
    tot = 0.0
    for i in range(d.ncon):
        g1, g2 = d.contact.geom[i]
        if {g1, g2} == {gt, gc}:
            mujoco.mj_contactForce(m, d, i, f6)
            tot += abs(f6[0])
    return tot


def min_jerk(s):
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def demo(delta, t_move):
    """리더를 목표 접촉 위치 + Δ까지 t_move 동안 min-jerk로 옮기는 시연 한 번.
    반환: 손가락 위치, 힘, 큐브 위치."""
    m = build(MU_DEMO)
    d = mujoco.MjData(m)
    n = int(DEMO_S / DT)
    jt0 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "jt")
    d.qpos[m.jnt_qposadr[jt0]] = X_TIP0
    d.ctrl[0] = X_TIP0
    mujoco.mj_forward(m, d)
    x_end = (D_GOAL - HALF - R_TIP) + delta              # 목표에서 큐브 중심 기준 접촉 위치(D − 0.028) + Δ
    xo, fs, xc = np.zeros(n), np.zeros(n), np.zeros(n)
    jt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "jt")
    jc = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "jc")
    qa_t = m.jnt_qposadr[jt]
    qa_c = m.jnt_qposadr[jc]
    for i in range(n):
        t = i * DT
        xL = X_TIP0 + (x_end - X_TIP0) * min_jerk(t / t_move)
        d.ctrl[0] = xL
        xo[i] = d.qpos[qa_t]
        fs[i] = tip_cube_force(m, d)
        xc[i] = d.qpos[qa_c]
        mujoco.mj_step(m, d)
    return dict(xo=xo, f=fs, xc=xc, delta=delta, t_move=t_move)


def replay(mu_rail, kf, dm, cube_dx):
    m = build(mu_rail)
    d = mujoco.MjData(m)
    jt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "jt")
    jc = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "jc")
    qa_t = m.jnt_qposadr[jt]
    qa_c = m.jnt_qposadr[jc]
    d.qpos[qa_c] = cube_dx
    d.qpos[qa_t] = dm["xo"][0]
    d.ctrl[0] = dm["xo"][0]
    mujoco.mj_forward(m, d)
    n = len(dm["xo"])
    fs, xc, xt = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        d.ctrl[0] = dm["xo"][i] + kf * dm["f"][i]
        xt[i] = d.qpos[qa_t]
        fs[i] = tip_cube_force(m, d)
        xc[i] = d.qpos[qa_c]
        mujoco.mj_step(m, d)
    tail = slice(-int(0.5 / DT), None)
    xc_final = float(xc[-1])
    f_hold = float(fs[tail].mean())
    err = xc_final - D_GOAL
    reached = abs(err) <= GOAL_TOL
    # 큐브가 움직이기 시작한 뒤 목표 구역에 처음 들어가기 전까지 접촉이 없는 시간
    moving = np.flatnonzero(xc > 1e-3)
    t0 = moving[0] if moving.size else n
    in_zone = np.abs(xc - D_GOAL) <= GOAL_TOL
    first_in = int(np.argmax(in_zone)) if in_zone.any() else n
    loss = (fs == 0.0)
    loss[:t0] = False
    loss[first_in:] = False
    return dict(
        success=bool(reached),
        reached=bool(reached),
        stall_mm=float(max(0.0, -err) * 1e3),
        overshoot_mm=float(max(0.0, err) * 1e3),
        f_hold_N=f_hold,
        contact_loss_s=float(loss.sum() * DT),
        force_rmse_N=float(np.sqrt(np.mean((fs - dm["f"]) ** 2))),
    )


def main():
    t_all = time.perf_counter()
    rng_all = {s: np.random.default_rng(s) for s in SEEDS}
    demos = {}
    demo_rows = []
    for s in SEEDS:
        demos[s] = []
        for k in range(N_DEMO):
            delta = float(rng_all[s].uniform(DELTA_RANGE[0], DELTA_RANGE[1]))
            t_move = float(rng_all[s].uniform(T_RANGE[0], T_RANGE[1]))
            dm = demo(delta, t_move)
            demos[s].append(dm)
            demo_rows.append(dict(seed=s, demo=k, delta_mm=round(delta * 1e3, 2), t_move_s=round(t_move, 2),
                                  demo_hold_force_N=round(float(dm["f"][-int(0.5 / DT):].mean()), 3),
                                  demo_final_cube_mm=round(float(dm["xc"][-1]) * 1e3, 2)))
            print("demo", demo_rows[-1], flush=True)

    rows = []
    for s in SEEDS:
        for k, dm in enumerate(demos[s]):
            for dx in PLACEMENTS:
                for mu in MU_LIST:
                    for kf in KF_LIST:
                        r = replay(mu, kf, dm, dx)
                        r.update(seed=s, demo=k, cube_dx_mm=dx * 1e3, mu=mu, kf=kf)
                        rows.append(r)
            print("seed", s, "demo", k, "done", round(time.perf_counter() - t_all, 1), "s", flush=True)

    def agg(sel):
        succ = [r["success"] for r in sel]
        return dict(n=len(sel), success_rate=round(float(np.mean(succ)), 3),
                    success_by_seed=[round(float(np.mean([r["success"] for r in sel if r["seed"] == s])), 3)
                                     for s in SEEDS],
                    reached_rate=round(float(np.mean([r["reached"] for r in sel])), 3),
                    stall_mm_median=round(float(np.median([r["stall_mm"] for r in sel])), 2),
                    overshoot_mm_median=round(float(np.median([r["overshoot_mm"] for r in sel])), 2),
                    f_hold_N_median=round(float(np.median([r["f_hold_N"] for r in sel])), 3),
                    contact_loss_s_mean=round(float(np.mean([r["contact_loss_s"] for r in sel])), 3),
                    force_rmse_N_mean=round(float(np.mean([r["force_rmse_N"] for r in sel])), 3))

    grid = []
    for kf in KF_LIST:
        for mu in MU_LIST:
            sel = [r for r in rows if r["kf"] == kf and r["mu"] == mu]
            grid.append(dict(kf=round(kf, 5), mu=mu, **agg(sel)))
    # 큐브 위치 오차에 따른 민감도 (μ = 0.5 시연 마찰과 같은 조건)
    placement = []
    for kf in KF_LIST:
        for dx in PLACEMENTS:
            sel = [r for r in rows if r["kf"] == kf and r["mu"] == MU_DEMO and r["cube_dx_mm"] == dx * 1e3]
            placement.append(dict(kf=round(kf, 5), cube_dx_mm=dx * 1e3, **agg(sel)))
    # 허용 μ 범위: 각 k_f에서 성공률 ≥ 0.5 인 μ의 최댓값 (위치 오차 평균)
    mu_range = []
    for kf in KF_LIST:
        ok = [mu for mu in MU_LIST if np.mean([r["success"] for r in rows if r["kf"] == kf and r["mu"] == mu]) >= 0.5]
        mu_range.append(dict(kf=round(kf, 5), mu_max_success_ge_0p5=(max(ok) if ok else None),
                             mu_list_success=ok))

    out = {
        "env": {"mujoco": mujoco.__version__, "numpy": np.__version__, "device": "cpu",
                "dt_s": DT, "k_p": KP, "k_v": KV, "tip_mass_kg": M_TIP, "tip_radius_m": R_TIP,
                "cube_mass_kg": M_CUBE, "cube_half_m": HALF, "goal_m": D_GOAL, "goal_tol_m": GOAL_TOL,
                "demo_mu_rail": MU_DEMO, "tip_cube_mu": MU_TIP,
                "seeds": SEEDS, "demos_per_seed": N_DEMO, "placements_mm": [p * 1e3 for p in PLACEMENTS],
                "rail_friction": "joint frictionloss = mu * m * g", "end_stop": "none"},
        "demos": demo_rows,
        "grid_by_kf_mu": grid,
        "placement_at_demo_mu": placement,
        "mu_range_by_kf": mu_range,
        "n_episodes": len(rows),
        "wall_time_s": None,
    }
    out["wall_time_s"] = round(time.perf_counter() - t_all, 1)
    RESULTS.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print("wall", out["wall_time_s"], "s", "episodes", len(rows), flush=True)


if __name__ == "__main__":
    main()
