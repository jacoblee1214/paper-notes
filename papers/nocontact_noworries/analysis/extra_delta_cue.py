"""추가 검증 2 (nocontact_noworries): 물리 단서 δ·c가 속도 교란을 걸러내는가. 실행: python3 analysis/extra_delta_cue.py

03 §5의 1차 추종 모델을 그대로 쓴다: q_{t+1} = q_t + α(q^com_t − q_t) − β c_t, δ = β/α.
관절 추종 오차 e_t = q^com_t − q_t 는 δ·c_t(접촉 신호)와 속도 항 (1−α)/α·v_t(명령 속도)의 합이다.
  물리 단서(P1): 명령 속도 v_t는 기록되므로 속도 항을 빼면 r_t = e_t − (1−α)/α·v_t ≈ δ·c_t + 잡음.

정책 입력 (로지스틱 회귀, 최근 W = 4 프레임 창, 멈춤 = 처음 P>0.5):
  M0: 시각 접촉 검출 d (FPR 1%, FNR 10%) 만
  M1: d + 원시 추종 오차 e (속도 항이 섞임)
  M2: d + 물리 잔차 r (속도 항을 뺌)
  단서만: e 단독, r 단독 (AUC 참고용)
학습 시연: 접촉 때 사람이 감속한다고 가정(v = 0.02·(1 − 0.8c)). 그러면 속도가 접촉과 상관한다 (03 §6.2의 κ).
시험 두 가지:
  matched  : 학습과 같은 시연 분포
  disturbed: 접촉과 무관한 속도 교란(v = 0.02 + U(−0.01, 0.01)), 감속 없음
접촉 추정 증분 = M1 또는 M2의 AUC − M0의 AUC.
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import extra_label_noise as base  # noqa: E402  (같은 과제 정의·창·로지스틱·멈춤 규칙을 재사용)

T, W = base.T, base.W
ALPHA, BETA = 0.2, 0.01            # δ = β/α = 0.05 rad (03 §5의 가정)
DELTA = BETA / ALPHA
SPEED_TERM = (1 - ALPHA) / ALPHA   # 정상상태에서 e = (1−α)/α · v
ENC_NOISE = 0.005                  # 관절 측정 잡음 [rad]
FPR, FNR = 0.01, 0.10              # 시각 검출 오류 (추론과 학습 공통)
N_TRAIN, N_TEST = 4000, 6000


def simulate(rng, c, v):
    """c: (n,T) 접촉, v: (n,T) 명령 속도. 반환 e_meas, r (물리 잔차)."""
    n = c.shape[0]
    qcom = np.cumsum(v, axis=1)
    q = np.zeros(n)
    e = np.zeros((n, T))
    for t in range(T):
        e[:, t] = qcom[:, t] - q
        q = q + ALPHA * (qcom[:, t] - q) - BETA * c[:, t]
    e_meas = e + rng.normal(0, ENC_NOISE, e.shape)
    v_known = np.concatenate([np.zeros((n, 1)), np.diff(qcom, axis=1)], axis=1)
    r = e_meas - SPEED_TERM * v_known
    return e_meas, r


def gen(rng, n, disturbed):
    c, onset = base.make_episodes(rng, n)
    if disturbed:
        off = rng.uniform(-0.01, 0.01, size=(n, 1))
        v = 0.02 + off + rng.normal(0, 0.001, (n, T))
    else:
        v = 0.02 * (1 - 0.8 * c) + rng.normal(0, 0.001, (n, T))
    e, r = simulate(rng, c, v)
    d = base.detect(rng, c, FPR, FNR)
    return {"c": c, "onset": onset, "d": d, "e": e, "r": r, "v": v}


def feats(data, kind):
    cols = []
    if kind in ("M0", "M1", "M2", "d_only"):
        cols.append(base.window(data["d"]))
    if kind == "M1":
        cols.append(base.window(data["e"]))
    if kind == "M2":
        cols.append(base.window(data["r"]))
    if kind == "e_only":
        cols.append(base.window(data["e"]))
    if kind == "r_only":
        cols.append(base.window(data["r"]))
    X = np.concatenate(cols, axis=-1)
    return X.reshape(-1, X.shape[-1])


def auc(score, y):
    ranks = rankdata(score)
    npos = y.sum()
    nneg = len(y) - npos
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def run_kind(kind, train, tests):
    X = feats(train, kind)
    y = train["c"].reshape(-1).astype(float)
    w = base.logistic_fit(X, y)
    out = {}
    for name, data in tests.items():
        p = base.logistic_prob(w, feats(data, kind)).reshape(data["c"].shape)
        out[name] = {"AUC": round(auc(p.ravel(), data["c"].ravel()), 4),
                     "stop": base.stop_metrics(p, data["onset"])}
    return out, [round(float(x), 3) for x in w]


def main():
    rng = np.random.default_rng(7)
    train = gen(rng, N_TRAIN, disturbed=False)
    tests = {"matched": gen(rng, N_TEST, disturbed=False), "disturbed": gen(rng, N_TEST, disturbed=True)}
    kinds = ["M0", "M1", "M2", "e_only", "r_only"]
    res, weights = {}, {}
    for k in kinds:
        res[k], weights[k] = run_kind(k, train, tests)
        print(k, {t: res[k][t]["AUC"] for t in tests}, {t: res[k][t]["stop"]["success"] for t in tests}, flush=True)

    inc = {}
    for m in ("M1", "M2"):
        inc[m] = {t: round(res[m][t]["AUC"] - res["M0"][t]["AUC"], 4) for t in tests}
    # 속도 교란 아래의 AUC 저하 (matched 대비)
    drop = {m: round(res[m]["matched"]["AUC"] - res[m]["disturbed"]["AUC"], 4) for m in kinds}
    speed_corr = {"train_corr(v, c)": round(float(np.corrcoef(train["v"].ravel(), train["c"].ravel())[0, 1]), 3),
                  "disturbed_corr(v, c)": round(float(np.corrcoef(tests["disturbed"]["v"].ravel(),
                                                                  tests["disturbed"]["c"].ravel())[0, 1]), 3)}
    out = {"setup": {"alpha": ALPHA, "beta": BETA, "delta_rad": DELTA, "speed_term_coef": SPEED_TERM,
                     "encoder_noise_rad": ENC_NOISE, "vision_FPR": FPR, "vision_FNR": FNR, "window": W,
                     "T": T, "N_train": N_TRAIN, "N_test": N_TEST,
                     "train_policy": "접촉 중 감속 v = 0.02(1−0.8c) (속도–접촉 상관 가정)",
                     "disturbed_policy": "v = 0.02 + U(−0.01, 0.01), 접촉과 무관, 감속 없음",
                     "seed": 7},
           "speed_contact_correlation": speed_corr,
           "results": res, "increment_AUC_over_M0": inc, "AUC_drop_matched_to_disturbed": drop,
           "logistic_weights_order": "bias, then window lags d(t..t-3), [e or r](t..t-3)"}
    path = ROOT / "analysis" / "extra_delta_cue_results.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    plot(out)
    print("increment", inc)
    print("drop", drop)
    print("wrote", path)


def plot(out):
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
    names = {"M0": "시각만(M0)", "M1": "시각+원시 e", "M2": "시각+잔차 r(물리 단서)"}
    keys = ["M0", "M1", "M2"]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    x = np.arange(len(keys))
    w = 0.36
    m_vals = [out["results"][k]["matched"]["AUC"] for k in keys]
    d_vals = [out["results"][k]["disturbed"]["AUC"] for k in keys]
    b1 = ax.bar(x - w / 2, m_vals, w, color="#2a78d6", label="학습과 같은 시연")
    b2 = ax.bar(x + w / 2, d_vals, w, color="#eb6834", label="속도 교란(접촉과 무관)")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008, f"{b.get_height():.2f}",
                    ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([names[k] for k in keys], fontsize=8)
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("프레임 단위 AUC")
    ax.grid(axis="y", color="#e4e4e7", lw=0.6)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.set_title("물리 잔차 단서가 속도 교란을 걸러내는가 (toy)", fontsize=9)
    fig.tight_layout()
    fig.savefig(ROOT / "report" / "figures" / "extra_delta_cue.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
