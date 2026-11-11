"""추가 검증 (beyond_implicit_force): MuJoCo 한 팔 위치 서보에서 닦기·강성 판별 과제와 소형 정책 비교.

실행: .venv/bin/python analysis/extra_mujoco_wipe.py   (결과: analysis/extra_mujoco_wipe_results.json)

- 로봇: 평면 2링크 팔(수평면, 관절마다 P 위치 서보 kp = 70 N·m/rad, 중력 없음).
  사람 시연은 leader의 데카르트 목표를 IK로 관절 목표로 바꿔 서보에 넣는다(원격조작 구조).
- 과제 1 (닦기): 표면(z = 0)을 일정한 수직력 6 N으로 누르며 0.3 m 옆으로 민다.
- 과제 2 (강성 판별): 무른 물체(k = 150–220 N/m)는 6 N까지 누르고, 단단한 물체(k = 2000–4000 N/m)는
  2 N에서 물러난다. 관찰은 관절각(qpos), 관절 토크 대리값(서보 힘), 또는 qpos 이력뿐이다.
- 정책: 은닉층 256 두 개의 MLP가 H = 20 스텝 행동 chunk를 예측하고 5 스텝마다 재계획한다.
  비전 입력은 없다. 행동 공간은 leader 목표(Base) 또는 follower 목표(ACT-o)이다.
- 변형: Base, ACT + τ, ACT-o, ACT-o + τ, ACT-o + qpos 이력(5프레임, 토크 대신 시간 정보).
- 학습 시드 3개, 시험 시도 30회씩(과제 2는 무른 15회·단단한 15회). 학습된 정책은 .scratch에 저장한다
  (다른 스크립트가 불러 쓴다).
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

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / ".scratch" / "mujoco_wipe"
RESULTS = Path(__file__).resolve().parent / "extra_mujoco_wipe_results.json"

L1 = L2 = 0.30
BASE_Z = 0.30                   # 팔 밑면 높이 (월드 z)
DT_CTRL, SUB = 0.02, 10         # 50 Hz 제어, 물리 dt 0.002
H, REPLAN = 20, 5               # chunk 길이, 재계획 주기 (스텝)
KJ, KV = 70.0, 2.0              # 관절 P 서보 (N·m/rad), 속도항
G_INT = 0.02                    # 사람의 힘 적분 이득 [m/(N·s)]
V_DESC = 0.10                   # 접근 속도 [m/s]
F_TGT, F_TOUCH = 6.0, 2.0       # 닦기 목표 힘, 단단한 물체에서 물러나는 문턱 [N]
WIPE_T, STIFF_T = 6.0, 4.0      # 에피소드 길이 [s]
SWEEP_X = (-0.15, 0.15)         # 닦기 구간 (팁 x, 월드)
V_SWEEP = 0.10
TAU_NOISE, Q_NOISE = 0.10, 1e-4
K_SOFT, K_HARD = (150.0, 220.0), (2000.0, 4000.0)
MU_RANGE = (0.4, 0.8)

VARIANTS = {
    "Base ACT": dict(target="leader", feat="q"),
    "ACT + τ": dict(target="leader", feat="qt"),
    "ACT-o": dict(target="follower", feat="q"),
    "ACT-o + τ": dict(target="follower", feat="qt"),
    "ACT-o + qhist": dict(target="follower", feat="qhist"),
}


def make_xml(with_obj):
    obj = """
    <body name="obj" pos="0 0 0.10">
      <joint name="objz" type="slide" axis="0 0 1" range="-0.05 0" stiffness="200" springref="0"
             damping="15" limited="true" armature="0"/>
      <geom name="objg" type="box" size="0.04 0.04 0.04" mass="0.01" friction="0.6 0.005 0.0001"/>
    </body>""" if with_obj else ""
    return f"""
<mujoco model="wipe_arm">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <default><joint armature="0.02" damping="0.5"/></default>
  <worldbody>
    <geom name="table" type="plane" size="1 1 0.1" friction="0.6 0.005 0.0001"/>
    <body name="base" pos="0 0 {BASE_Z}">
      <body name="l1">
        <joint name="j1" type="hinge" axis="0 -1 0" range="-6.3 6.3"/>
        <geom type="capsule" fromto="0 0 0 {L1} 0 0" size="0.02" mass="0.3" contype="0" conaffinity="0"/>
        <body name="l2" pos="{L1} 0 0">
          <joint name="j2" type="hinge" axis="0 -1 0" range="0.05 2.95"/>
          <geom type="capsule" fromto="0 0 0 {L2} 0 0" size="0.018" mass="0.2" contype="0" conaffinity="0"/>
          <body name="tipbody" pos="{L2} 0 0">
            <geom name="tip" type="sphere" size="0.015" mass="0.05" friction="0.6 0.005 0.0001"/>
          </body>
        </body>
      </body>
    </body>{obj}
  </worldbody>
  <actuator>
    <position name="a1" joint="j1" kp="{KJ}" kv="{KV}"/>
    <position name="a2" joint="j2" kp="{KJ}" kv="{KV}"/>
  </actuator>
</mujoco>
"""


def fk(q):
    """관절각 → 팔 밑면 기준 (x, z). MuJoCo 모델과 같은 방향 규약(hinge 축 -y)."""
    return np.array([L1 * np.cos(q[0]) + L2 * np.cos(q[0] + q[1]),
                     L1 * np.sin(q[0]) + L2 * np.sin(q[0] + q[1])])


def ik(p):
    x, z = p
    c2 = np.clip((x * x + z * z - L1 * L1 - L2 * L2) / (2 * L1 * L2), -1.0, 1.0)
    q2 = np.arccos(c2)
    q1 = np.arctan2(z, x) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
    return np.array([q1, q2])


class Arm:
    def __init__(self, with_obj):
        self.m = mujoco.MjModel.from_xml_string(make_xml(with_obj))
        self.d = mujoco.MjData(self.m)
        self.with_obj = with_obj
        self.tip = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "tip")
        self.table = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "table")
        self.jobj = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "objz") if with_obj else -1
        self.objg = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "objg") if with_obj else -1

    def reset(self, q0, obj_k=None, mu=0.6):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:2] = q0
        if self.with_obj:
            self.d.qpos[2] = 0.0
            self.m.jnt_stiffness[self.jobj] = obj_k
            self.m.geom_friction[self.objg, 0] = mu
        self.m.geom_friction[self.table, 0] = mu
        self.m.geom_friction[self.tip, 0] = mu
        self.d.ctrl[:] = q0
        mujoco.mj_forward(self.m, self.d)

    def tip_xz(self):
        p = self.d.geom_xpos[self.tip]
        return np.array([p[0], p[2] - BASE_Z])

    def tip_force(self):
        f6 = np.zeros(6)
        fn = 0.0
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if self.tip in (c.geom1, c.geom2):
                mujoco.mj_contactForce(self.m, self.d, i, f6)
                fn += abs(f6[0])
        return fn

    def tau(self):
        return self.d.actuator_force[:2].copy()

    def step(self, q_cmd):
        self.d.ctrl[:] = q_cmd
        for _ in range(SUB):
            mujoco.mj_step(self.m, self.d)


# ---------- 시연자(사람 원격조작 모사) ----------

def demo_wipe(arm, rng):
    mu = rng.uniform(*MU_RANGE)
    x0 = rng.uniform(-0.17, -0.13)                       # 시작 위치 무작위(상태 공간을 넓혀 정책 외삽을 줄임)
    x_end = rng.uniform(0.13, 0.17)
    x, z = x0, 0.10 - BASE_Z
    arm.reset(ik(np.array([x, z])), mu=mu)
    n = int(WIPE_T / DT_CTRL)
    contact, sweep0 = None, None
    qs, ctrls, taus, Fs, zs = [], [], [], [], []
    for t in range(n):
        F = arm.tip_force()
        qs.append(arm.d.qpos[:2].copy())
        taus.append(arm.tau() + rng.normal(0, TAU_NOISE, 2))
        Fs.append(F)
        zs.append(arm.tip_xz()[1])
        if contact is None:
            z -= V_DESC * DT_CTRL
            if F > 0.3:
                contact = t
        else:
            z += G_INT * (F - F_TGT) * DT_CTRL
            if t >= contact + int(0.2 / DT_CTRL):
                if sweep0 is None:
                    sweep0 = t
                x = min(x + V_SWEEP * DT_CTRL, x_end)
        cmd = ik(np.array([x, z + rng.normal(0, 3e-4)]))
        ctrls.append(cmd)
        arm.step(cmd)
    return dict(q=np.array(qs), ctrl=np.array(ctrls), tau=np.array(taus), F=np.array(Fs),
                z=np.array(zs), mu=mu, contact=contact)


def demo_stiff(arm, rng, soft):
    k = rng.uniform(*K_SOFT) if soft else rng.uniform(*K_HARD)
    x0 = rng.uniform(-0.02, 0.02)
    arm.reset(ik(np.array([x0, 0.25 - BASE_Z])), obj_k=k)
    z = 0.25 - BASE_Z
    f_t = rng.uniform(5.5, 6.5)
    f_touch = rng.uniform(1.8, 2.2)
    phase = "approach"
    n = int(STIFF_T / DT_CTRL)
    qs, ctrls, taus, Fs = [], [], [], []
    for t in range(n):
        F = arm.tip_force()
        qs.append(arm.d.qpos[:2].copy())
        taus.append(arm.tau() + rng.normal(0, TAU_NOISE, 2))
        Fs.append(F)
        if phase == "approach":
            z -= V_DESC * DT_CTRL
            if F > 0.3:
                phase = "press"
        elif phase == "press":
            if soft:
                z += G_INT * (F - f_t) * DT_CTRL
            else:
                z -= 0.02 * DT_CTRL          # 단단한 물체: 문턱 근처에서 천천히 누른다
                if F >= f_touch:
                    phase = "retreat"
        else:
            z = min(z + 0.10 * DT_CTRL, 0.25 - BASE_Z)
        cmd = ik(np.array([x0, z + rng.normal(0, 3e-4)]))
        ctrls.append(cmd)
        arm.step(cmd)
    return dict(q=np.array(qs), ctrl=np.array(ctrls), tau=np.array(taus), F=np.array(Fs),
                k=k, soft=soft, tip_z=float(arm.tip_xz()[1]))


# ---------- 관찰, 데이터셋, 정책 ----------

def obs_vec(feat, qh, tau):
    """qh: (5, 2) 최근 관절각(첫 행이 현재). tau: (2,) 서보 토크."""
    if feat == "q":
        return qh[0]
    if feat == "qt":
        return np.concatenate([qh[0], tau])
    return qh.reshape(-1)                       # qhist


def build_dataset(demos, task, variant):
    spec = VARIANTS[variant]
    X, Y = [], []
    for d in demos:
        T = len(d["q"])
        src = d["ctrl"] if spec["target"] == "leader" else d["q"]
        for t in range(T - H):
            qh = np.stack([d["q"][max(t - j, 0)] for j in range(5)])
            X.append(obs_vec(spec["feat"], qh, d["tau"][t]))
            chunk = src[t:t + H] - d["q"][t]     # 현재 관절각 기준 상대 목표
            Y.append(chunk.reshape(-1))
    return np.array(X, np.float32), np.array(Y, np.float32)


class MLP(nn.Module):
    def __init__(self, din, dout, hid=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(),
                                 nn.Linear(hid, dout))

    def forward(self, x):
        return self.net(x)


def train_policy(X, Y, seed, dev="cpu", epochs=60, bs=256):
    torch.manual_seed(seed)
    xm, xs = X.mean(0), np.maximum(X.std(0), 0.2)     # 표준편차 하한: 작은 분산 입력의 과증폭 방지
    ym, ys = Y.mean(0), Y.std(0) + 1e-6
    Xt = torch.tensor((X - xm) / xs).to(dev)
    Yt = torch.tensor((Y - ym) / ys).to(dev)
    net = MLP(X.shape[1], Y.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            loss = ((net(Xt[idx]) - Yt[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    net = net.cpu().eval()                      # 시험(한 샘플 추론)은 CPU에서 한다
    norm = dict(xm=xm, xs=xs, ym=ym, ys=ys)
    return net, norm


def predict_chunk(net, norm, feat_vec, q_now):
    with torch.no_grad():
        x = torch.tensor(((feat_vec - norm["xm"]) / norm["xs"])[None], dtype=torch.float32)
        y = net(x).numpy()[0] * norm["ys"] + norm["ym"]
    return y.reshape(H, 2) + q_now                # 절대 관절 목표


def run_policy(arm, net, norm, variant, task, rng, soft=None):
    spec = VARIANTS[variant]
    if task == "wipe":
        mu = rng.uniform(*MU_RANGE)
        x = rng.uniform(-0.17, -0.13)
        arm.reset(ik(np.array([x, 0.10 - BASE_Z])), mu=mu)
        T = int(WIPE_T / DT_CTRL)
    else:
        k = rng.uniform(*K_SOFT) if soft else rng.uniform(*K_HARD)
        arm.reset(ik(np.array([rng.uniform(-0.02, 0.02), 0.25 - BASE_Z])), obj_k=k)
        T = int(STIFF_T / DT_CTRL)
    qh = np.stack([arm.d.qpos[:2].copy()] * 5)
    Fs, qs, zs = [], [], []
    chunk = None
    for t in range(T):
        if t % REPLAN == 0:
            q_now = arm.d.qpos[:2].copy() + rng.normal(0, Q_NOISE, 2)
            tau = arm.tau() + rng.normal(0, TAU_NOISE, 2)
            qh = np.concatenate([[q_now], qh[:4]])
            chunk = predict_chunk(net, norm, obs_vec(spec["feat"], qh, tau), q_now)
        Fs.append(arm.tip_force())
        qs.append(arm.d.qpos[:2].copy())
        zs.append(arm.tip_xz()[1])
        arm.step(chunk[t % REPLAN])
    return dict(F=np.array(Fs), q=np.array(qs), z=np.array(zs), tip_z=float(arm.tip_xz()[1]))


def wipe_success(ep):
    """접촉은 팁 높이(표면 위 2.5 mm 이내)로 판정한다. 표면 접촉력은 접촉 스텝에서 평균한다.
    (스틱슬립으로 팁이 순간적으로 떨어져 접촉 힘이 0이 되는 스텝이 있어서, 접촉 비율을 힘 > 0으로 잡지 않는다.)"""
    F, z = ep["F"], ep["z"]
    cont = np.flatnonzero(F > 0.3)
    if cont.size == 0:
        return False, 0.0
    t0 = cont[0] + int(0.2 / DT_CTRL)
    t1 = min(t0 + int(3.0 / DT_CTRL), len(F))
    if t1 - t0 < 10:
        return False, 0.0
    win_z, win_F = z[t0:t1], F[t0:t1]
    xs = ep["q"][t0:t1]
    travel = float(np.ptp([fk(q)[0] for q in xs]))
    on = win_z < -BASE_Z + 0.0175
    fmean = float(win_F[on].mean()) if on.any() else 0.0
    ok = (on.mean() >= 0.9) and (3.0 <= fmean <= 10.0) and travel >= 0.2
    return bool(ok), fmean


def stiff_success(ep, soft):
    F = ep["F"]
    if soft:
        last = F[-int(1.0 / DT_CTRL):]
        ok = (last.min() > 0.5) and (3.0 <= last.mean() <= 9.0)
    else:
        ok = (F.max() <= 5.5) and (ep["tip_z"] + BASE_Z >= 0.185)     # 최대 힘 < 무른 목표 6 N, 물러남(월드 z)
    return bool(ok)


def bench_device():
    """소형 MLP 한 스텝 시간을 CPU와 MPS에서 재고, 빠르게 나온 장치를 고른다(차이가 작으면 CPU)."""
    res = {"cpu_ms": None, "mps_ms": None}
    x = torch.randn(256, 40)
    for dev in ["cpu", "mps"]:
        if dev == "mps" and not torch.backends.mps.is_available():
            continue
        net = MLP(40, H * 2).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
        xx = x.to(dev)
        for i in range(60):
            if i == 10:
                t0 = time.perf_counter()
            loss = (net(xx) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if dev == "mps":
            torch.mps.synchronize()
        res[f"{dev}_ms"] = round((time.perf_counter() - t0) / 50 * 1e3, 3)
    dev = "mps" if (res["mps_ms"] is not None and res["mps_ms"] < 0.7 * res["cpu_ms"]) else "cpu"
    return dev, res                             # 학습 장치. 시험 추론은 항상 CPU


def main():
    t_all = time.perf_counter()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    dev, bench = bench_device()
    print("device", dev, bench, flush=True)
    rng = np.random.default_rng(0)

    # 정확도 점검: 팔 FK(모델)와 해석식 FK가 같은지
    arm_w = Arm(False)
    arm_w.reset(np.array([0.4, 0.8]))
    fk_err = float(np.abs(arm_w.tip_xz() - fk(np.array([0.4, 0.8]))).max())

    out = {"env": {"torch": torch.__version__, "mujoco": mujoco.__version__, "numpy": np.__version__,
                   "device_train": dev, "device_infer": "cpu", "device_bench_ms": bench, "threads": 4,
                   "seeds_train": [0, 1, 2],
                   "dt_phys_s": 0.002, "control_hz": 50, "chunk_H": H, "replan_steps": REPLAN},
           "fk_check_max_err_m": fk_err, "model": {"mlp": "40-256-256-out, GELU, AdamW 1e-3",
                                                  "epochs": 60, "batch": 256}}
    # 시연
    t0 = time.perf_counter()
    arm_s = Arm(True)
    demos_w = [demo_wipe(arm_w, rng) for _ in range(100)]
    demos_s = [demo_stiff(arm_s, rng, soft=(i % 2 == 0)) for i in range(100)]
    out["demo"] = {
        "wipe_n": len(demos_w), "stiff_n": len(demos_s),
        "wipe_success_rate": round(float(np.mean([wipe_success(d)[0] for d in demos_w])), 3),
        "stiff_soft_success": None, "stiff_hard_success": None,
        "demo_time_s": None,
    }
    out["demo"]["stiff_soft_success"] = round(float(np.mean([stiff_success(d, True) for d in demos_s if d["soft"]])), 3)
    out["demo"]["stiff_hard_success"] = round(float(np.mean([stiff_success(d, False) for d in demos_s if not d["soft"]])), 3)
    out["demo"]["demo_time_s"] = round(time.perf_counter() - t0, 2)
    print("demos done", out["demo"], flush=True)

    # 학습 + 시험
    test_rng = np.random.default_rng(12345)
    test_seeds = list(range(30))
    res = {}
    for task, demos in (("wipe", demos_w), ("stiff", demos_s)):
        res[task] = {}
        for variant in VARIANTS:
            X, Y = build_dataset(demos, task, variant)
            per_seed = []
            for seed in [0, 1, 2]:
                tt = time.perf_counter()
                net, norm = train_policy(X, Y, seed, dev=dev)
                train_s = time.perf_counter() - tt
                ev_rng = np.random.default_rng(1000 + seed)
                if task == "wipe":
                    eps = [run_policy(arm_w, net, norm, variant, "wipe", ev_rng) for _ in range(30)]
                    succ = [wipe_success(e) for e in eps]
                    per_seed.append(dict(seed=seed, success=float(np.mean([s for s, _ in succ])),
                                         mean_normal_force_N=float(np.mean([f for _, f in succ])),
                                         train_s=round(train_s, 2)))
                else:
                    soft_s = [stiff_success(run_policy(arm_s, net, norm, variant, "stiff", ev_rng, soft=True), True)
                              for _ in range(15)]
                    hard_s = [stiff_success(run_policy(arm_s, net, norm, variant, "stiff", ev_rng, soft=False), False)
                              for _ in range(15)]
                    sr, hr = float(np.mean(soft_s)), float(np.mean(hard_s))
                    per_seed.append(dict(seed=seed, soft_press=sr, hard_retreat=hr, youden_J=sr + hr - 1.0,
                                         train_s=round(train_s, 2)))
                torch.save({"state": net.state_dict(), "norm": norm, "variant": variant, "task": task,
                            "seed": seed}, SCRATCH / f"{task}_{variant.replace(' ', '').replace('+', 'p')}_s{seed}.pt")
                print(task, variant, seed, per_seed[-1], flush=True)
            if task == "wipe":
                res[task][variant] = dict(
                    success_mean=round(float(np.mean([p["success"] for p in per_seed])), 3),
                    success_by_seed=[round(p["success"], 3) for p in per_seed],
                    mean_normal_force_N=round(float(np.mean([p["mean_normal_force_N"] for p in per_seed])), 2),
                )
            else:
                sp = [p["soft_press"] for p in per_seed]
                hr = [p["hard_retreat"] for p in per_seed]
                jj = [p["youden_J"] for p in per_seed]
                res[task][variant] = dict(
                    soft_press_mean=round(float(np.mean(sp)), 3), hard_retreat_mean=round(float(np.mean(hr)), 3),
                    youden_J_mean=round(float(np.mean(jj)), 3), youden_J_by_seed=[round(v, 3) for v in jj],
                )
    out["results"] = res
    out["test"] = {"wipe_episodes_per_seed": 30, "stiff_episodes_per_seed": "15 soft + 15 hard"}
    out["wall_time_s"] = round(time.perf_counter() - t_all, 1)
    RESULTS.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps(out["results"], ensure_ascii=False, indent=2), flush=True)
    print("wall", out["wall_time_s"], "s", flush=True)


if __name__ == "__main__":
    main()
