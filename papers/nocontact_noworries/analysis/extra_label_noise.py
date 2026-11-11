"""추가 검증 1 (nocontact_noworries): 학습 시 라벨 잡음 주입이 추론 시 접촉 신호 오류에 대한 민감도를 줄이는가.
실행: python3 analysis/extra_label_noise.py   (numpy, CPU 1스레드, 수십 초)

1-DoF 회전 toy: 관절이 일정 속도로 접촉면을 향해 움직이고, 접촉이 시작되면 멈춰야 한다(control_toy.py와 같은 과제).
  프레임 단위 접촉 검출 d_t는 제어 toy와 같이 FPR·FNR로 오류가 난다.
  정책은 최근 W = 4 프레임의 d로 로지스틱 회귀를 학습한 "멈춤 확률" P(c_t = 1 | window)이다.
  멈춤 규칙: 처음으로 P > 0.5가 된 프레임에서 멈춘다.
  성공: 접촉 시작(20프레임 뒤) 후 0–3 프레임 안에 멈추고, 그 전에는 멈추지 않음.
학습 조건 (추론 조건은 공통: FPR 0–5%, FNR 0–20%):
  clean  : 깨끗한 라벨과 깨끗한 입력
  label  : 학습 라벨 c_t를 확률 r로 뒤집음 (라벨 잡음, FPR = FNR = r)
  input  : 학습 입력 d를 확률 r로 뒤집음 (입력 잡음, 라벨은 깨끗함)
r ∈ {0.02, 0.05, 0.10}.
질문: 학습 잡음(r)이 추론 민감도 dS/dFPR을 줄이는가. 학습 잡음이 있으면 정책이 더 많은 연속 증거를 요구해 지연이 늘어나는지도 본다.
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
N_PRE, D_MAX, W = 20, 3, 4
T = N_PRE + 10          # 에피소드 길이 (접촉 후 10프레임 관찰)
N_TRAIN, N_TEST = 4000, 20000
FPRS = [0.0, 0.005, 0.01, 0.02, 0.05]
FNRS = [0.0, 0.1, 0.2]
TRAIN_RATES = [0.02, 0.05, 0.10]
RNG_SEED = 1


def flips(rng, x, rate):
    return np.where(rng.random(x.shape) < rate, 1 - x, x)


def detect(rng, c, fpr, fnr):
    """프레임마다 독립 오류. c: (n, T) 0/1 참 접촉."""
    u = rng.random(c.shape)
    return np.where(c == 1, (u >= fnr).astype(int), (u < fpr).astype(int))


def window(d, W=W):
    """각 프레임에서 최근 W개 값 (앞은 0 패딩). (n, T) -> (n, T, W)"""
    n, t = d.shape
    pad = np.concatenate([np.zeros((n, W - 1), dtype=d.dtype), d], axis=1)
    return np.stack([pad[:, W - 1 - k: W - 1 - k + t] for k in range(W)], axis=-1)


def logistic_fit(X, y, l2=1e-3, iters=25):
    X = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (p - y) + l2 * w
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(len(w))
        w -= np.linalg.solve(H, g)
    return w


def logistic_prob(w, X):
    X = np.column_stack([np.ones(len(X)), X])
    return 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))


def stop_metrics(p, c_onset):
    """p: (n, T) 멈춤 확률. 처음 p>0.5인 프레임을 멈춤 시점으로 본다."""
    n = p.shape[0]
    above = p > 0.5
    has = above.any(1)
    stop = np.where(has, above.argmax(1), T)
    ok = has & (stop >= c_onset) & (stop <= c_onset + D_MAX)
    early = has & (stop < c_onset)
    return {"success": round(float(ok.mean()), 4), "early_stop": round(float(early.mean()), 4),
            "missed_or_late": round(float(1 - ok.mean() - early.mean()), 4)}


def make_episodes(rng, n):
    c = np.zeros((n, T), dtype=int)
    onset = N_PRE + rng.integers(0, 2, size=n)           # 접촉 시작 20 또는 21
    for i in range(n):
        c[i, onset[i]:] = 1
    return c, onset


def train_policy(rng, mode, r):
    c, _ = make_episodes(rng, N_TRAIN)
    if mode == "clean":
        d = detect(rng, c, 0.0, 0.0)
        y = c
    elif mode == "label":
        d = detect(rng, c, 0.0, 0.0)
        y = flips(rng, c, r)
    else:                                                  # input
        d = flips(rng, detect(rng, c, 0.0, 0.0), r)
        y = c
    X = window(d).reshape(-1, W)
    return logistic_fit(X, y.reshape(-1).astype(float))


def evaluate(w, rng):
    c, onset = make_episodes(rng, N_TEST)
    out = {}
    for fpr in FPRS:
        for fnr in FNRS:
            d = detect(rng, c, fpr, fnr)
            p = logistic_prob(w, window(d).reshape(-1, W)).reshape(c.shape)
            out[f"fpr={fpr},fnr={fnr}"] = stop_metrics(p, onset)
    return out


def sensitivity(res):
    """fnr=0.1 고정, FPR 0 → 0.02 구간의 성공률 감소 기울기 (성공률 %p per 1%FPR)."""
    s0 = res["fpr=0.0,fnr=0.1"]["success"]
    s2 = res["fpr=0.02,fnr=0.1"]["success"]
    return round(float((s0 - s2) / 2.0 * 100), 2)


def main():
    rng = np.random.default_rng(RNG_SEED)
    results = {}
    weights = {}
    configs = [("clean", 0.0)] + [(m, r) for m in ("label", "input") for r in TRAIN_RATES]
    for mode, r in configs:
        key = "clean" if mode == "clean" else f"{mode} 잡음 r={r}"
        w = train_policy(rng, mode, r)
        weights[key] = [round(float(x), 3) for x in w]
        res = evaluate(w, rng)
        results[key] = {"metrics": res, "sensitivity_success_pp_per_1pct_fpr": sensitivity(res)}
        print(key, "sens", results[key]["sensitivity_success_pp_per_1pct_fpr"],
              "| FPR0.01 FNR0.1:", res["fpr=0.01,fnr=0.1"], flush=True)
    out = {"setup": {"N_PRE": N_PRE, "D_MAX_frames": D_MAX, "window": W, "T": T, "N_train": N_TRAIN, "N_test": N_TEST,
                     "policy": "로지스틱 회귀, 최근 4프레임 d, 멈춤 = 처음 P>0.5", "seed": RNG_SEED,
                     "note": "제어 toy(control_toy.py)는 K 디바운스 규칙을 쓰고, 여기서는 학습된 확률 규칙을 쓴다"},
           "weights_bias_then_W_lags": weights, "results": results}
    path = ROOT / "analysis" / "extra_label_noise_results.json"
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
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    styles = {"clean": ("#0b0b0b", "-"), "label": ("#2a78d6", "--"), "input": ("#eb6834", ":")}
    for key, data in results.items():
        mode = key.split(" ")[0]
        color, ls = styles[mode]
        ys = [data["metrics"][f"fpr={f},fnr=0.1"]["success"] for f in FPRS]
        ax.plot([f * 100 for f in FPRS], ys, ls, marker="o", ms=3.5, color=color, lw=1.4, label=key)
    ax.set_xlabel("추론 시 접촉 검출 위양성률 FPR [%] (FNR = 10%)")
    ax.set_ylabel("멈춤 성공률")
    ax.set_ylim(0, 1.02)
    ax.grid(color="#e4e4e7", lw=0.6)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.set_title("학습 잡음 주입 대 추론 민감도 (1-DoF toy)", fontsize=9)
    fig.tight_layout()
    fig.savefig(ROOT / "report" / "figures" / "extra_label_noise.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
