"""목표 정렬 보조 손실만 바꾼 소규모 비교: 연속 대조(α = 2.5) 대 InfoNCE 대 없음.

범위와 가정(논문에 없는 설계 가정 포함, 논문 값에 맞추지 않음):
- 손: 공개 Allegro Hand v3 MJCF (google-deepmind/mujoco_menagerie, wonik_allegro, BSD-3-Clause).
  data/menagerie/wonik_allegro/ 에 sparse clone으로 받는다. 손은 고정된 배경으로만 쓰고 손가락 제어와 접촉은 없다.
- 물체: 정육면체(외형 대칭, 단색)와 비대칭 상자(면마다 색이 다름). 둘 다 mujoco.Renderer로 그린다.
  자세는 기구학적으로 설정한다(mj_forward만 호출, 물리 적분 없음).
- 교사: 진짜 자세 오차를 아는 비례 제어기. 명령은 세계 좌표 회전 벡터(이득 0.25, 최대 0.06 rad/스텝)이고
  스텝마다 0.01 rad 표준편차 소음이 더해진다. 물체 자세는 명령으로 기구학적으로 갱신된다.
- 학생: 현재 영상과 목표 영상을 각각 독립 CNN으로 인코딩(분리 인코더)하고, 정책 헤드가 교사 명령을 행동 복제한다.
  재구성 헤드와 GRU는 없다. 보조 손실은 세 가지(없음, 연속 대조, InfoNCE)이며 계수는 모두 0.1.
- InfoNCE: 같은 배치 안에서 (현재, 목표) 쌍의 측지 각이 0.4 rad 아래이면 양성. 온도 0.1은 이 비교의 선택값이다
  (논문은 온도를 주지 않음).
- 데이터: 교사 궤적 400개(물체마다 같은 자세 집합, 고정 시드). 학습 쌍은 같은 궤적의 미래 상태(40%),
  에피소드 목표(30%), 무작위 상태(30%)에서 뽑는다. DAgger는 쓰지 않는다.
- 평가: 학습에 쓰지 않은 무작위 목표 16 에피소드, 에피소드당 200 스텝. 도달은 측지 각 < 0.4 rad가 3스텝 연속일 때이며,
  도달하면 새 목표를 뽑는다. CS = 에피소드당 도달 횟수, TTS = 첫 도달 스텝(도달한 에피소드 평균).
- 시드 5개. 시드는 가중치 초기화와 학습 쌍 추출을 바꾸며, 데이터와 평가 에피소드는 모든 팔과 시드가 같다.
- 진단: 무작위 자세 쌍에서 임베딩 코사인과 각도의 상관, 그리고 대칭 등가 상태(정육면체에서는 영상이 같다)에서의
  학생 명령 크기를 잰다.

실행: .venv/bin/python -I analysis/extra_goal_alignment.py  (torch, mujoco, scipy, matplotlib; CPU 6 프로세스, 프로세스당 스레드 2)
출력: analysis/extra_goal_alignment_results.json, report/figures/own_goal_alignment_*.png
"""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import mujoco
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
HAND = ROOT / "data" / "menagerie" / "wonik_allegro" / "right_hand.xml"
SMOKE = os.environ.get("GA_SMOKE") == "1"
CACHE = ROOT / "data" / ("goal_alignment_smoke" if SMOKE else "goal_alignment_cache")
OUT = ROOT / ".scratch" / "goal_alignment_smoke.json" if SMOKE else Path(__file__).resolve().parent / "extra_goal_alignment_results.json"
FIG_DIR = ROOT / ".scratch" / "goal_alignment_figs" if SMOKE else ROOT / "report" / "figures"

TAU = 0.4
ALPHA = 2.5
LAM = 0.1
TEMP = 0.1
GAIN = 0.25
MAXSTEP = 0.06
NOISE = 0.01
IMG = 32
N_EP = 20 if SMOKE else 400
T_EP = 60
DATA_SEED = 0
STEPS = 30 if SMOKE else 2000
BATCH = 128
LR = 1e-3
SEEDS = [0, 1] if SMOKE else [0, 1, 2, 3, 4]
ARMS = ["none", "infonce", "cont"]
KINDS = ["cube", "asym"]
N_EVAL = 2 if SMOKE else 16
EVAL_STEPS = 60 if SMOKE else 200
EVAL_SEED = 2026
DIAG_SEED = 7
DIAG_PAIRS = 100 if SMOKE else 400
DIAG_SYM = 50 if SMOKE else 200
OBJ_CENTER = np.array([0.02, -0.05, 0.05])
ASYM_HALF = (0.04, 0.025, 0.015)
WORKERS = 6


# ---------- 쿼터니언 (w, x, y, z) ----------

def qmul(a, b):
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)


def qconj(q):
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def qnorm(q):
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def qexp(v):
    ang = np.linalg.norm(v, axis=-1, keepdims=True)
    half = ang / 2
    s = np.where(ang > 1e-12, np.sin(half) / np.maximum(ang, 1e-12), 0.5)
    return np.concatenate([np.cos(half), v * s], axis=-1)


def rotvec(q):
    q = np.where(q[..., :1] < 0, -q, q)
    v = q[..., 1:]
    s = np.linalg.norm(v, axis=-1, keepdims=True)
    ang = 2 * np.arctan2(s, q[..., :1])
    return v * (ang / np.maximum(s, 1e-12))


def angle(a, b):
    return 2 * np.arccos(np.clip(np.abs(np.sum(a * b, axis=-1)), 0, 1))


def haar(n, rng):
    return qnorm(rng.normal(size=(n, 4)))


def octahedral_group():
    """정육면체 회전군 O (24원소), 생성원으로부터 닫힘 계산."""
    gens = [np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]),
            np.array([np.cos(np.pi / 3), *(np.sin(np.pi / 3) * np.ones(3) / np.sqrt(3))])]
    elems = [np.array([1.0, 0, 0, 0])]
    frontier = list(elems)
    while frontier:
        new = []
        for a in frontier:
            for g in gens:
                b = qmul(a, g)
                if not any(abs(np.dot(b, e)) > 1 - 1e-9 for e in elems):
                    elems.append(b)
                    new.append(b)
        frontier = new
    assert len(elems) == 24
    return np.stack(elems)


# ---------- 교사와 기구학 전이 ----------

def teacher_label(qc, qg):
    """교사 명령을 MAXSTEP으로 나눈 값(단위 공 안). 진짜 자세 오차를 쓴다."""
    v = GAIN * rotvec(qmul(qg, qconj(qc)))
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v * np.minimum(1.0, MAXSTEP / np.maximum(n, 1e-12)) / MAXSTEP


def advance(q, cmd, noise):
    return qnorm(qmul(qexp(cmd + noise), q))


# ---------- 렌더링 (mujoco.Renderer) ----------

def make_model(kind):
    spec = mujoco.MjSpec.from_file(str(HAND))
    spec.worldbody.add_light(pos=[0, 0, 0.5], dir=[0, 0, -1])
    spec.worldbody.add_light(pos=[0.3, 0.3, 0.5], dir=[-1, -1, -1])
    body = spec.worldbody.add_body(name="obj", pos=OBJ_CENTER.tolist())
    body.add_freejoint()
    box = mujoco.mjtGeom.mjGEOM_BOX
    if kind == "cube":
        body.add_geom(name="core", type=box, size=[0.03, 0.03, 0.03], rgba=[0.9, 0.5, 0.2, 1])
    else:
        hx, hy, hz = ASYM_HALF
        t = 0.0015
        body.add_geom(name="core", type=box, size=[hx, hy, hz], rgba=[0.2, 0.2, 0.2, 1])
        faces = [
            ([hx + t, 0, 0], [t, hy, hz], [1, 0, 0]),
            ([-hx - t, 0, 0], [t, hy, hz], [0, 1, 0]),
            ([0, hy + t, 0], [hx, t, hz], [0, 0, 1]),
            ([0, -hy - t, 0], [hx, t, hz], [1, 1, 0]),
            ([0, 0, hz + t], [hx, hy, t], [1, 0, 1]),
            ([0, 0, -hz - t], [hx, hy, t], [0, 1, 1]),
        ]
        for k, (pos, size, rgb) in enumerate(faces):
            body.add_geom(name=f"face{k}", type=box, pos=pos, size=size, rgba=rgb + [1])
    return spec.compile()


class Scene:
    def __init__(self, kind):
        self.m = make_model(kind)
        self.d = mujoco.MjData(self.m)
        self.qadr = self.m.jnt_qposadr[self.m.body_jntadr[self.m.body("obj").id]]
        self.r = mujoco.Renderer(self.m, height=IMG, width=IMG)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = OBJ_CENTER
        self.cam.distance = 0.14
        self.cam.azimuth = 90
        self.cam.elevation = -30

    def render(self, q):
        a = self.qadr
        self.d.qpos[a:a + 3] = OBJ_CENTER
        self.d.qpos[a + 3:a + 7] = q
        mujoco.mj_forward(self.m, self.d)
        self.r.update_scene(self.d, camera=self.cam)
        return self.r.render().copy()

    def render_many(self, qs):
        return np.stack([self.render(q) for q in qs])


def load_pool(kind):
    """학습용 상태 pool: 교사 궤적 상태 (N_EP x (T_EP+1)) 와 에피소드 목표 N_EP개를 렌더링한다. 결과는 캐시."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{kind}_{N_EP}x{T_EP}_{IMG}.npz"
    if path.exists():
        z = np.load(path)
        return z["Q"], z["imgs"]
    rng = np.random.default_rng(DATA_SEED)
    goals = haar(N_EP, rng)
    states = np.zeros((N_EP, T_EP + 1, 4))
    for e in range(N_EP):
        states[e, 0] = haar(1, rng)[0]
        for t in range(T_EP):
            a = teacher_label(states[e, t], goals[e])
            states[e, t + 1] = advance(states[e, t], MAXSTEP * a, rng.normal(size=3) * NOISE)
    Q = np.concatenate([states.reshape(-1, 4), goals])
    imgs = Scene(kind).render_many(Q)
    np.savez(path, Q=Q, imgs=imgs)
    return Q, imgs


def sample_pairs(rng, B):
    Tp = T_EP + 1
    e = rng.integers(0, N_EP, B)
    t = rng.integers(0, T_EP, B)
    cur = e * Tp + t
    t2 = t + 1 + (rng.random(B) * (T_EP - t)).astype(int)
    hind = e * Tp + t2
    epi = N_EP * Tp + e
    rnd = rng.integers(0, N_EP * Tp, B)
    u = rng.random(B)
    goal = np.where(u < 0.4, hind, np.where(u < 0.7, epi, rnd))
    return cur, goal


# ---------- 학생 모델 ----------

def to_tensor(imgs):
    return torch.from_numpy(np.ascontiguousarray(imgs)).permute(0, 3, 1, 2).float() / 255.0 - 0.5


class Encoder(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 1), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 2 * 2, dim),
        )

    def forward(self, x):
        return self.net(x)


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_obj = Encoder()
        self.enc_goal = Encoder()
        self.head = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 3), nn.Tanh())

    def forward(self, xo, xg):
        zo = self.enc_obj(xo)
        zg = self.enc_goal(xg)
        return self.head(torch.cat([zo, zg], 1)), zo, zg


def infonce(zo, zg, qc, qg):
    d = 2 * np.arccos(np.clip(np.abs(qc @ qg.T), 0, 1))
    pos = torch.from_numpy(d < TAU)
    has = pos.any(1)
    if not has.any():
        return zo.sum() * 0
    logits = F.normalize(zo, dim=1) @ F.normalize(zg, dim=1).T / TEMP
    lse_all = torch.logsumexp(logits, 1)
    lse_pos = torch.logsumexp(logits.masked_fill(~pos, -1e9), 1)
    return (lse_all - lse_pos)[has].mean()


def train_run(arm, seed, Q, imgs):
    torch.manual_seed(seed)
    rng = np.random.default_rng(100 + seed)
    model = Student()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    for _ in range(STEPS):
        cur, goal = sample_pairs(rng, BATCH)
        qc, qg = Q[cur], Q[goal]
        a_t = torch.from_numpy(teacher_label(qc, qg)).float()
        a, zo, zg = model(to_tensor(imgs[cur]), to_tensor(imgs[goal]))
        loss = F.mse_loss(a, a_t)
        if arm == "cont":
            s_star = torch.from_numpy(np.exp(-ALPHA * angle(qc, qg))).float()
            loss = loss + LAM * F.mse_loss(F.cosine_similarity(zo, zg, dim=1), s_star)
        elif arm == "infonce":
            loss = loss + LAM * infonce(zo, zg, qc, qg)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model


@torch.no_grad()
def predict(model, imgs_c, imgs_g):
    a, _, _ = model(to_tensor(imgs_c), to_tensor(imgs_g))
    a = a.numpy()
    n = np.linalg.norm(a, axis=1, keepdims=True)
    return a / np.maximum(n, 1.0)


# ---------- 평가 ----------

def closed_loop(sc, mode, model=None):
    """mode: 'teacher' (특권 교사), 'zero' (명령 0), 'student'. 에피소드마다 고정 시드."""
    cs_list, tts = [], []
    for k in range(N_EVAL):
        rg = np.random.default_rng(EVAL_SEED + k)
        rn = np.random.default_rng(EVAL_SEED + 10_000 + k)
        q = haar(1, rg)[0]
        g = haar(1, rg)[0]
        img_g = sc.render(g) if mode == "student" else None
        streak, cs, first = 0, 0, None
        for t in range(EVAL_STEPS):
            streak = streak + 1 if angle(q, g) < TAU else 0
            if streak == 3:
                cs += 1
                first = t if first is None else first
                g = haar(1, rg)[0]
                streak = 0
                if mode == "student":
                    img_g = sc.render(g)
            if mode == "teacher":
                a = teacher_label(q, g)
            elif mode == "zero":
                a = np.zeros(3)
            else:
                a = predict(model, sc.render(q)[None], img_g[None])[0]
            q = advance(q, MAXSTEP * a, rn.normal(size=3) * NOISE)
        cs_list.append(cs)
        if first is not None:
            tts.append(first)
    return {
        "cs": float(np.mean(cs_list)),
        "tts": float(np.mean(tts)) if tts else None,
        "frac_eps_success": float(np.mean([c > 0 for c in cs_list])),
    }


def diagnostics(sc, model, G):
    rng = np.random.default_rng(DIAG_SEED)
    qa, qb = haar(DIAG_PAIRS, rng), haar(DIAG_PAIRS, rng)
    d = angle(qa, qb)
    s_star = np.exp(-ALPHA * d)
    with torch.no_grad():
        zo = model.enc_obj(to_tensor(sc.render_many(qa)))
        zg = model.enc_goal(to_tensor(sc.render_many(qb)))
        cos = F.cosine_similarity(zo, zg, dim=1).numpy()
    scatter = (d, cos)
    out = {
        "pearson_cos_vs_angle": float(stats.pearsonr(cos, d)[0]),
        "spearman_cos_vs_angle": float(stats.spearmanr(cos, d)[0]),
        "pearson_cos_vs_sstar": float(stats.pearsonr(cos, s_star)[0]),
        "mean_abs_cos_minus_sstar": float(np.mean(np.abs(cos - s_star))),
    }
    # 대칭 등가 상태: q_cur = q_goal ⊗ g (g는 O의 원소, 각도 > TAU). 정육면체에서는 영상이 같다.
    qg = haar(DIAG_SYM, rng)
    angG = 2 * np.arccos(np.clip(np.abs(G[:, 0]), 0, 1))
    valid = np.where(angG > TAU)[0]
    qc = qmul(qg, G[rng.choice(valid, DIAG_SYM)])
    ic, ig = sc.render_many(qc), sc.render_many(qg)
    a_s = predict(model, ic, ig)
    with torch.no_grad():
        zc = model.enc_obj(to_tensor(ic)).numpy()
        zgg = model.enc_goal(to_tensor(ig)).numpy()
    cos_sym = np.sum(zc * zgg, 1) / (np.linalg.norm(zc, axis=1) * np.linalg.norm(zgg, axis=1))
    stu_norm = MAXSTEP * np.linalg.norm(a_s, axis=1)
    tea_norm = MAXSTEP * np.linalg.norm(teacher_label(qc, qg), axis=1)
    d_sym = angle(qc, qg)
    out.update({
        "sym_mean_pixel_absdiff": float(np.mean(np.abs(ic.astype(float) - ig.astype(float)))),
        "sym_frac_identical_image": float(np.mean(np.all(ic == ig, axis=(1, 2, 3)))),
        "sym_student_cmd_rad": float(np.mean(stu_norm)),
        "sym_teacher_cmd_rad": float(np.mean(tea_norm)),
        "sym_stall_frac": float(np.mean(stu_norm < 0.2 * MAXSTEP)),
        "sym_mean_cos": float(np.mean(cos_sym)),
        "sym_mean_sstar": float(np.mean(np.exp(-ALPHA * d_sym))),
    })
    return out, scatter


# ---------- 작업 단위 ----------

def run_job(job):
    kind, arm, seed = job
    torch.set_num_threads(2)
    Q, imgs = load_pool(kind)
    sc = Scene(kind)
    model = train_run(arm, seed, Q, imgs)
    res = {"kind": kind, "arm": arm, "seed": seed}
    res.update(closed_loop(sc, "student", model))
    res["diag"], (d, cos) = diagnostics(sc, model, octahedral_group())
    if seed == SEEDS[0]:
        res["scatter"] = (d, cos)  # 그림용. JSON에는 쓰지 않는다
    return res


def welch_and_paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    w = stats.ttest_ind(a, b, equal_var=False)
    p = stats.ttest_rel(a, b)
    return {"diff_mean": float(a.mean() - b.mean()), "welch_t": float(w.statistic), "welch_p": float(w.pvalue),
            "paired_t": float(p.statistic), "paired_p": float(p.pvalue)}


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj


def git_commit_of_hand():
    try:
        return subprocess.run(["git", "-C", str(ROOT / "data" / "menagerie"), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def make_figures(results, scatter):
    import matplotlib
    matplotlib.use("Agg")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for name in ("Apple SD Gothic Neo", "Noto Sans CJK KR", "Noto Sans KR"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    BLUE, ORANGE, GREY = "#1f5fae", "#d9822b", "#7a7a75"
    names = {"none": "없음", "infonce": "InfoNCE", "cont": "연속 대조"}
    kind_names = {"cube": "정육면체 (대칭 외형)", "asym": "비대칭 상자"}

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9), sharey=True)
    for ax, kind in zip(axes, KINDS):
        R = results["arms_runs"][kind]
        for i, arm in enumerate(ARMS):
            vals = [r["cs"] for r in R[arm]["runs"]]
            ax.bar(i, np.mean(vals), color=[GREY, ORANGE, BLUE][i], alpha=0.55, width=0.6)
            ax.errorbar(i, np.mean(vals), yerr=np.std(vals, ddof=1), color="black", capsize=4)
            ax.scatter(np.full(len(vals), i) + np.linspace(-0.12, 0.12, len(vals)), vals, color="black", s=12, zorder=3)
        ax.axhline(results["references"][kind]["teacher"]["cs"], color="green", ls="--", lw=1, label="교사(특권)")
        ax.axhline(results["references"][kind]["zero"]["cs"], color="red", ls=":", lw=1, label="명령 0")
        ax.set_xticks(range(3), [names[a] for a in ARMS])
        ax.set_title(kind_names[kind], fontsize=10)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("CS (에피소드당 도달 횟수)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("보조 손실만 바꾼 비교: 막대 = 5 시드 평균, 점 = 시드, 오차 막대 = sd", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "own_goal_alignment_cs.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(9, 5.6), sharex=True, sharey=True)
    xs = np.linspace(0, np.pi, 100)
    for row, kind in enumerate(KINDS):
        for col, arm in enumerate(ARMS):
            ax = axes[row, col]
            d, cos = scatter[(kind, arm)]
            ax.scatter(d, cos, s=6, alpha=0.5, color="black")
            ax.plot(xs, np.exp(-ALPHA * xs), color=BLUE, ls="--", lw=1.2, label="목표 exp(-αθ)")
            ax.axvline(TAU, color=ORANGE, lw=0.8)
            ax.set_title(f"{kind_names[kind].split(' ')[0]} / {names[arm]}", fontsize=9)
            ax.grid(alpha=0.25)
            if row == 1:
                ax.set_xlabel("측지 각 θ (rad)")
            if col == 0:
                ax.set_ylabel("임베딩 코사인")
    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.suptitle("무작위 자세 쌍(400개, 시드 0): 임베딩 코사인 대 측지 각. 주황선 = τ = 0.4", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "own_goal_alignment_embed.png", dpi=160)
    plt.close(fig)


def main():
    t0 = time.time()
    for kind in KINDS:
        load_pool(kind)
        print(f"pool ready: {kind} ({time.time() - t0:.0f}s)", flush=True)
    refs = {}
    for kind in KINDS:
        sc = Scene(kind)
        refs[kind] = {"teacher": closed_loop(sc, "teacher"), "zero": closed_loop(sc, "zero")}
    print("references done", flush=True)

    jobs = [(k, a, s) for k in KINDS for a in ARMS for s in SEEDS]
    ctx = get_context("spawn")
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=ctx) as ex:
        runs = list(ex.map(run_job, jobs))
    print(f"student runs done ({time.time() - t0:.0f}s)", flush=True)

    arms = {k: {a: {"runs": []} for a in ARMS} for k in KINDS}
    scatter = {}
    for r in runs:
        if "scatter" in r:
            scatter[(r["kind"], r["arm"])] = r.pop("scatter")
        arms[r["kind"]][r["arm"]]["runs"].append({
            "seed": r["seed"], "cs": r["cs"], "tts": r["tts"], "frac_eps_success": r["frac_eps_success"],
            "diag": r["diag"],
        })
    summary = {}
    tests = {}
    for kind in KINDS:
        summary[kind] = {}
        for arm in ARMS:
            R = arms[kind][arm]["runs"]
            cs = [x["cs"] for x in R]
            ttsv = [x["tts"] for x in R if x["tts"] is not None]
            diag_keys = R[0]["diag"].keys()
            summary[kind][arm] = {
                "cs_mean": float(np.mean(cs)), "cs_sd": float(np.std(cs, ddof=1)),
                "tts_mean": float(np.mean(ttsv)) if ttsv else None,
                "tts_sd": float(np.std(ttsv, ddof=1)) if len(ttsv) > 1 else None,
                "frac_eps_success_mean": float(np.mean([x["frac_eps_success"] for x in R])),
                "diag_mean": {k: float(np.mean([x["diag"][k] for x in R])) for k in diag_keys},
                "diag_sd": {k: float(np.std([x["diag"][k] for x in R], ddof=1)) for k in diag_keys},
                "cs_values": cs,
            }
        pairs = [("cont", "infonce"), ("cont", "none"), ("infonce", "none")]
        raw = []
        for a, b in pairs:
            ca = [x["cs"] for x in arms[kind][a]["runs"]]
            cb = [x["cs"] for x in arms[kind][b]["runs"]]
            raw.append(welch_and_paired(ca, cb))
        adj = holm([t["welch_p"] for t in raw])
        tests[kind] = {}
        for (a, b), t, pa in zip(pairs, raw, adj):
            t["welch_p_holm"] = float(pa)
            tests[kind][f"{a}_minus_{b}"] = t
        pooled = np.sqrt((summary[kind]["cont"]["cs_sd"] ** 2 + summary[kind]["infonce"]["cs_sd"] ** 2
                          + summary[kind]["none"]["cs_sd"] ** 2) / 3)
        n = len(SEEDS)
        df = 2 * (n - 1)
        mde = (stats.t.ppf(0.975, df) + stats.t.ppf(0.8, df)) * pooled * np.sqrt(2 / n)
        tests[kind]["min_detectable_cs_diff_80pct"] = float(mde)

    out = {
        "config": {
            "tau_rad": TAU, "alpha": ALPHA, "lambda_aux": LAM, "infonce_temperature": TEMP,
            "gain": GAIN, "max_step_rad": MAXSTEP, "noise_rad_per_step": NOISE, "image_px": IMG,
            "train_episodes": N_EP, "episode_len": T_EP, "steps": STEPS, "batch": BATCH, "lr": LR,
            "seeds": SEEDS, "eval_episodes": N_EVAL, "eval_steps": EVAL_STEPS,
            "encoder": "separate 4-conv CNN per image (16-32-32-32 ch), 128-d output; policy MLP 256-256-3 (tanh)",
        },
        "hand": {"source": "google-deepmind/mujoco_menagerie", "folder": "wonik_allegro",
                 "commit": git_commit_of_hand(), "license": "BSD-3-Clause (SimLab), LICENSE in folder"},
        "references": refs,
        "summary": summary,
        "tests": tests,
        "arms_runs": arms,
        "runtime_s": round(time.time() - t0, 1),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT} ({time.time() - t0:.0f}s)", flush=True)
    make_figures(out, scatter)
    print("figures written", flush=True)


if __name__ == "__main__":
    main()
