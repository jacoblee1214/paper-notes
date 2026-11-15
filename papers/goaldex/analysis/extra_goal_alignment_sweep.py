"""목표 정렬 보조 손실의 강도를 인코더 기울기 노름으로 맞춘 뒤, InfoNCE 온도와 연속 대조 α를 바꾸는 민감도 비교 (비대칭 상자, 5 시드).

기반: analysis/extra_goal_alignment.py의 환경, 교사, 학생 구조, 평가 함수를 그대로 가져온다. 이 파일은 학습 루프와 계수 규칙만 바꾼다.

강도 맞춤: 보조 손실 계수 w를 인코더 기울기 노름 비로 정한다.
    w = λ_m · EMA(‖∇_enc L_act‖) / EMA(‖∇_enc L_aux‖),  25 스텝마다 갱신, EMA 계수 0.9, λ_m = 1.
    그러면 보조 항의 기울기 노름은 행동 항의 λ_m배가 된다. 측정은 학습 중인 같은 배치에서 한다.
sweep: InfoNCE 온도 {0.05, 0.1, 0.3}, 연속 대조 α {1.0, 2.5, 5.0}, 기준은 보조 손실 없음.
가정: λ_m = 1은 이 비교의 선택값이다. 논문은 계수나 기울기 노름 규칙을 주지 않는다.

실행: python3 -I analysis/extra_goal_alignment_sweep.py  (torch와 mujoco가 있는 환경)
출력: analysis/extra_goal_alignment_sweep_results.json (집계와 시드별 CS, 소음과 데이터는 기반 스크립트와 같다)
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import extra_goal_alignment as G  # noqa: E402  (환경, 교사, 학생, 평가, 데이터 생성)

LAM_M = 1.0
REFRESH = 25
EMA = 0.9
TEMPS = [0.05, 0.1, 0.3]
ALPHAS = [1.0, 2.5, 5.0]
KIND = "asym"
OUT = HERE / "extra_goal_alignment_sweep_results.json"
THREADS = 4
WORKERS = 1  # 6 프로세스로 돌렸을 때 다른 작업과 겹쳐 맥이 과부하되어, 한 프로세스로 차례대로 돌린다


def enc_params(model):
    return list(model.enc_obj.parameters()) + list(model.enc_goal.parameters())


def grad_norm(loss, params):
    gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return float(torch.sqrt(sum((g ** 2).sum() for g in gs if g is not None)).item())


def infonce_t(zo, zg, qc, qg, temp):
    d = 2 * np.arccos(np.clip(np.abs(qc @ qg.T), 0, 1))
    pos = torch.from_numpy(d < G.TAU)
    has = pos.any(1)
    if not has.any():
        return zo.sum() * 0
    logits = F.normalize(zo, dim=1) @ F.normalize(zg, dim=1).T / temp
    lse_all = torch.logsumexp(logits, 1)
    lse_pos = torch.logsumexp(logits.masked_fill(~pos, -1e9), 1)
    return (lse_all - lse_pos)[has].mean()


def train_matched(arm, temp, alpha, seed, Q, imgs):
    torch.manual_seed(seed)
    rng = np.random.default_rng(100 + seed)
    model = G.Student()
    opt = torch.optim.Adam(model.parameters(), lr=G.LR)
    params = enc_params(model)
    ema_a = ema_x = None
    w = 0.0
    w_log = []
    model.train()
    for step in range(G.STEPS):
        cur, goal = G.sample_pairs(rng, G.BATCH)
        qc, qg = Q[cur], Q[goal]
        a_t = torch.from_numpy(G.teacher_label(qc, qg)).float()
        a, zo, zg = model(G.to_tensor(imgs[cur]), G.to_tensor(imgs[goal]))
        l_act = F.mse_loss(a, a_t)
        if arm == "none":
            loss = l_act
        else:
            if arm == "cont":
                s_star = torch.from_numpy(np.exp(-alpha * G.angle(qc, qg))).float()
                l_aux = F.mse_loss(F.cosine_similarity(zo, zg, dim=1), s_star)
            else:
                l_aux = infonce_t(zo, zg, qc, qg, temp)
            if step % REFRESH == 0:
                ga = grad_norm(l_act, params)
                gx = grad_norm(l_aux, params)
                ema_a = ga if ema_a is None else EMA * ema_a + (1 - EMA) * ga
                ema_x = gx if ema_x is None else EMA * ema_x + (1 - EMA) * gx
                w = LAM_M * ema_a / max(ema_x, 1e-12)
                w_log.append([step, ga, gx, w])
            loss = l_act + w * l_aux
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model, w_log


def run_job(job):
    arm, temp, alpha, seed = job
    torch.set_num_threads(THREADS)
    t0 = time.time()
    Q, imgs = G.load_pool(KIND)
    model, w_log = train_matched(arm, temp, alpha, seed, Q, imgs)
    sc = G.Scene(KIND)
    res = G.closed_loop(sc, "student", model)
    return {"arm": arm, "temp": temp, "alpha": alpha, "seed": seed,
            "cs": res["cs"], "tts": res["tts"], "frac_eps_success": res["frac_eps_success"],
            "w_first": w_log[0][3] if w_log else None, "w_last": w_log[-1][3] if w_log else None,
            "gact_last": w_log[-1][1] if w_log else None, "gaux_last": w_log[-1][2] if w_log else None,
            "seconds": round(time.time() - t0, 1)}


def run_job_logged(job):
    r = run_job(job)
    print(f"{label(r)} seed {r['seed']}: CS {r['cs']:.3f} ({r['seconds']:.0f}s)", flush=True)
    return r


def label(r):
    if r["arm"] == "none":
        return "none"
    if r["arm"] == "infonce":
        return f"infonce_t{r['temp']}"
    return f"cont_a{r['alpha']}"


def welch(a, b):
    w = stats.ttest_ind(a, b, equal_var=False)
    return float(w.statistic), float(w.pvalue)


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    t0 = time.time()
    G.load_pool(KIND)
    print("pool ready", flush=True)
    jobs = [("none", None, None, s) for s in G.SEEDS]
    jobs += [("infonce", t, None, s) for t in TEMPS for s in G.SEEDS]
    jobs += [("cont", None, a, s) for a in ALPHAS for s in G.SEEDS]
    ctx = get_context("spawn")
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=ctx) as ex:
        runs = list(ex.map(run_job_logged, jobs))
    print(f"runs done ({time.time() - t0:.0f}s)", flush=True)

    groups = {}
    for r in runs:
        groups.setdefault(label(r), []).append(r)
    summary = {}
    for k, rs in groups.items():
        cs = [x["cs"] for x in sorted(rs, key=lambda x: x["seed"])]
        summary[k] = {"cs_mean": float(np.mean(cs)), "cs_sd": float(np.std(cs, ddof=1)),
                      "cs_values": cs,
                      "tts_mean": float(np.mean([x["tts"] for x in rs if x["tts"] is not None]))
                      if any(x["tts"] is not None for x in rs) else None,
                      "w_last_mean": float(np.mean([x["w_last"] for x in rs])) if rs[0]["arm"] != "none" else None,
                      "gact_last_mean": float(np.mean([x["gact_last"] for x in rs])) if rs[0]["arm"] != "none" else None,
                      "gaux_last_mean": float(np.mean([x["gaux_last"] for x in rs])) if rs[0]["arm"] != "none" else None}

    cont_keys = [f"cont_a{a}" for a in ALPHAS]
    inf_keys = [f"infonce_t{t}" for t in TEMPS]
    pairs = [(c, i) for c in cont_keys for i in inf_keys]
    raw = []
    for c, i in pairs:
        a = [x["cs"] for x in sorted(groups[c], key=lambda x: x["seed"])]
        b = [x["cs"] for x in sorted(groups[i], key=lambda x: x["seed"])]
        t, p = welch(a, b)
        raw.append({"pair": f"{c}_minus_{i}", "diff_mean": float(np.mean(a) - np.mean(b)),
                    "welch_t": t, "welch_p": p})
    adj = holm([x["welch_p"] for x in raw])
    for x, pa in zip(raw, adj):
        x["welch_p_holm"] = float(pa)
    base = [x["cs"] for x in sorted(groups["none"], key=lambda x: x["seed"])]
    for k in cont_keys + inf_keys:
        cs = [x["cs"] for x in sorted(groups[k], key=lambda x: x["seed"])]
        t, p = welch(cs, base)
        summary[k]["vs_none_welch_p"] = p
        summary[k]["vs_none_diff"] = float(np.mean(cs) - np.mean(base))

    out = {"config": {"matching": "coefficient = LAM_M * EMA(|grad_enc L_act|) / EMA(|grad_enc L_aux|)",
                      "lam_m": LAM_M, "refresh_steps": REFRESH, "ema": EMA,
                      "temps": TEMPS, "alphas": ALPHAS, "kind": KIND, "seeds": G.SEEDS,
                      "steps": G.STEPS, "batch": G.BATCH, "lr": G.LR, "eval_episodes": G.N_EVAL,
                      "eval_steps": G.EVAL_STEPS, "tau_rad": G.TAU, "image_px": G.IMG,
                      "base_module": "extra_goal_alignment.py"},
           "summary": summary, "pairwise_cont_minus_infonce": raw, "runs": runs,
           "runtime_s": round(time.time() - t0, 1)}
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
