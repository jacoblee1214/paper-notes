"""추가 검증 1 (restacvla): 행동 손실만으로 학습한 게이트가 드문 예상 밖 충돌에 반응하는가. 실행: python3 analysis/extra_gate_rare.py

gate_toy.py의 1차원 접근-접촉-해제 에피소드와 CMP(분위수 구간 평균)를 그대로 쓴다.
게이트 g = Sigmoid(MLP(입력))를 numpy로 학습하고, 정책은 p = w0 + w1·v + w2·g·‖z‖ 로 둔다.
  행동 목표 a*: 기대 접촉 중 +1(밀기), 예상 밖 충돌 중 -1(후퇴), 그 밖 0.
  학습 손실: 행동 MSE만(논문 Sec. III-C). 보조 감독 변형은 여기에 게이트 라벨(접촉 또는 충돌)의 BCE를 더한다.
  게이트 입력: σ만(논문 Eq. 3), σ와 원시 예측 오차(변형).
학습 충돌 비율 p ∈ {1, 2, 5, 10, 20}%를 바꿔 충돌 시점의 게이트 열림, 정책의 후퇴 반응을 잰다.
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import gate_toy as gt  # noqa: E402

SIGMA_N = 0.05
P_TRAIN = [0.01, 0.02, 0.05, 0.10, 0.20]
N_TRAIN, N_TEST, N_SEED, H, ITERS, LR = 200, 200, 3, 4, 800, 0.03
INPUTS = {"σ (논문 Eq. 3)": ["sigma"], "σ + 예측 오차": ["sigma", "err"]}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_split(rng, n, p):
    v, z, contact, collide, ton = gt.episodes(rng, n, p, SIGMA_N)
    return {"v": v, "z": z, "contact": contact, "collide": collide, "ton": ton}


def gate_inputs(cmp, d, inputs):
    s_, err = gt.apply_cmp(cmp, d["v"], d["z"])
    cols = [np.log(s_).ravel()]
    if "err" in inputs:
        cols.append(np.log(err + 1e-12).ravel())
    X = np.stack(cols, -1)
    tact = np.linalg.norm(d["z"], axis=-1).ravel()
    v = d["v"].ravel()
    a_star = np.where(d["collide"], -1.0, np.where(d["contact"], 1.0, 0.0)).ravel()
    y = (d["contact"] | d["collide"]).ravel().astype(float)
    return X, v, tact, a_star, y


def train(X, v, s, a, y, aux, seed, lam_aux=1.0):
    rng = np.random.default_rng(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    P = {"W1": rng.normal(0, 0.5, (X.shape[1], H)), "b1": np.zeros(H), "u": rng.normal(0, 0.5, H),
         "c": np.zeros(1), "p": np.zeros(4)}
    M = {k: np.zeros_like(x) for k, x in P.items()}
    V = {k: np.zeros_like(x) for k, x in P.items()}
    N = len(a)
    for it in range(1, ITERS + 1):
        h = np.tanh(Xs @ P["W1"] + P["b1"])
        g = sigmoid(h @ P["u"] + P["c"][0])
        pol = P["p"][0] + P["p"][1] * v + P["p"][2] * g * s + P["p"][3] * g * s * v
        dpol = 2 * (pol - a) / N
        G = {"p": np.array([dpol.sum(), (dpol * v).sum(), (dpol * g * s).sum(), (dpol * g * s * v).sum()])}
        do = dpol * (P["p"][2] + P["p"][3] * v) * s * g * (1 - g)
        if aux:
            do = do + lam_aux * (g - y) / N
        G["u"] = h.T @ do
        G["c"] = np.array([do.sum()])
        dpre = np.outer(do, P["u"]) * (1 - h ** 2)
        G["W1"] = Xs.T @ dpre
        G["b1"] = dpre.sum(0)
        for k in P:
            M[k] = 0.9 * M[k] + 0.1 * G[k]
            V[k] = 0.999 * V[k] + 0.001 * G[k] ** 2
            mh = M[k] / (1 - 0.9 ** it)
            vh = V[k] / (1 - 0.999 ** it)
            P[k] = P[k] - LR * mh / (np.sqrt(vh) + 1e-8)
    return mu, sd, P


def predict(model, X, v, s):
    mu, sd, P = model
    h = np.tanh(((X - mu) / sd) @ P["W1"] + P["b1"])
    g = sigmoid(h @ P["u"] + P["c"][0])
    pol = P["p"][0] + P["p"][1] * v + P["p"][2] * g * s + P["p"][3] * g * s * v
    return g, pol


def evaluate(model, X, v, s, a, y, d):
    g, pol = predict(model, X, v, s)
    col = d["collide"].ravel()
    con = d["contact"].ravel()
    free = ~(col | con)
    pol_open = predict_open(model, v, s)
    return {
        "collision_gate_recall(g>0.5)": round(float((g[col] > 0.5).mean()), 3),
        "collision_gate_mean_g": round(float(g[col].mean()), 3),
        "contact_gate_recall(g>0.5)": round(float((g[con] > 0.5).mean()), 3),
        "free_space_false_open": round(float((g[free] > 0.5).mean()), 3),
        "collision_policy_mean_action": round(float(pol[col].mean()), 3),
        "collision_retreat_fraction(a<-0.5)": round(float((pol[col] < -0.5).mean()), 3),
        "oracle_gate_open_collision_action": round(float(pol_open[col].mean()), 3),
        "behavior_mse_all": round(float(((pol - a) ** 2).mean()), 4),
    }


def predict_open(model, v, s):
    """게이트를 완전히 열었을 때(g = 1)의 정책 출력: 게이트 학습이 없었을 때의 상한."""
    _, _, P = model
    return P["p"][0] + P["p"][1] * v + P["p"][2] * s + P["p"][3] * s * v


def main():
    results = {}
    runs = {}
    for label, inputs in INPUTS.items():
        for aux in (False, True):
            key = f"{label} | {'보조 감독 있음' if aux else '행동 손실만'}"
            results[key] = {}
            for p in P_TRAIN:
                rows = []
                for seed in range(N_SEED):
                    rng = np.random.default_rng(100 + seed)
                    tr = make_split(rng, N_TRAIN, p)
                    te = make_split(rng, N_TEST, 0.5)
                    cmp = gt.fit_cmp(tr["v"], tr["z"])
                    Xtr, vtr, str_, atr, ytr = gate_inputs(cmp, tr, inputs)
                    Xte, vte, ste, ate, yte = gate_inputs(cmp, te, inputs)
                    model = train(Xtr, vtr, str_, atr, ytr, aux, seed=seed)
                    m = evaluate(model, Xte, vte, ste, ate, yte, te)
                    m["train_collision_steps"] = int(tr["collide"].sum())
                    rows.append(m)
                keys = [k for k in rows[0] if k != "train_collision_steps"]
                agg = {k: round(float(np.mean([r[k] for r in rows])), 3) for k in keys}
                agg["train_collision_steps_mean"] = round(float(np.mean([r["train_collision_steps"] for r in rows])), 1)
                agg["collision_gate_recall_sd"] = round(float(np.std([r["collision_gate_recall(g>0.5)"] for r in rows])), 3)
                results[key][f"p={p}"] = agg
                print(key, f"p={p}", {k: agg[k] for k in ("collision_gate_recall(g>0.5)", "collision_retreat_fraction(a<-0.5)",
                                                             "oracle_gate_open_collision_action", "contact_gate_recall(g>0.5)")})
    out = {"setup": {"sigma_n": SIGMA_N, "train_episodes": N_TRAIN, "test_episodes": N_TEST, "test_collision_rate": 0.5,
                     "seeds": N_SEED, "gate_hidden": H, "adam_iters": ITERS, "lr": LR,
                     "policy": "p = w0 + w1·v + w2·g·‖z‖ + w3·g·‖z‖·v (상호작용항: 충돌 후퇴를 표현하려고 추가), 학습은 행동 MSE(+선택적 게이트 BCE)",
                     "expert": "기대 접촉 +1, 예상 밖 충돌 -1, 그 밖 0",
                     "cmp": "gate_toy.fit_cmp (학습 세트로 적합)"},
           "results": results}
    path = ROOT / "analysis" / "extra_gate_rare_results.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    plot(results)
    print("wrote", path)


def plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                              "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
    if font:
        font_manager.fontManager.addfont(font)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})
    xs = [p * 100 for p in P_TRAIN]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), sharey=True)
    colors = {"σ (논문 Eq. 3)": "#2a78d6", "σ + 예측 오차": "#eb6834"}
    for ax, metric, title in zip(axes, ["collision_gate_recall(g>0.5)", "contact_gate_recall(g>0.5)"],
                                 ["예상 밖 충돌 구간 게이트 열림", "기대 접촉 구간 게이트 열림"]):
        for key, data in results.items():
            label, mode = key.split(" | ")
            c = colors[label]
            ls = "-" if "보조" in mode else "--"
            y = [data[f"p={p}"][metric] for p in P_TRAIN]
            ax.plot(xs, y, ls, marker="o", color=c, ms=4, lw=1.5,
                    label=f"{label}, {'보조 감독' if '보조' in mode else '행동 손실만'}")
        ax.set_xlabel("학습 데이터 충돌 비율 [%]")
        ax.set_title(title, fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(color="#e4e4e7", lw=0.6)
    axes[0].set_ylabel("게이트 열림 비율 (g > 0.5)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, frameon=False, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(ROOT / "report" / "figures" / "extra_gate_rare.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
