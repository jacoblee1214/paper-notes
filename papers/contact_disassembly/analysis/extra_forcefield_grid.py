"""추가 검증 1 (contact_disassembly): 힘장 척도·릿지 λ·표본 수 격자에서 척도 무시 조건과 비단조성 지도.
실행: python3 analysis/extra_forcefield_grid.py

concat_toy.py의 설정(잠재 z ∈ R², 시각 V 64차원, 힘장 F 20차원, 릿지 분류, 평가 4000, 시드 40)을 그대로 쓴다.
격자: 원단위 힘 척도 α ∈ {1, 0.1, 0.01, 0.001} × 릿지 λ ∈ {0.1, 1, 10} × 표본 수 n ∈ {30, 60, 120, 250, 500, 1000}.
조건: V(시각만), V+F 원단위(α 곱함), V+F 표준화(α와 무관).
  척도 무시: |acc(V+F 원단위) − acc(V)| < 0.01 (표준오차 약 0.005의 2배 정도를 기준으로 사전 정의).
  비단조성: n과 α를 고정하고 λ를 바꿀 때 순위가 뒤집히는 칸, 또는 α에 대한 이득이 단조가 아닌 칸을 센다.
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import concat_toy as ct  # noqa: E402  (같은 생성 모델과 릿지 분류기)

ALPHAS = [1.0, 0.1, 0.01, 0.001]
LAMS = [0.1, 1.0, 10.0]
NS = [30, 60, 120, 250, 500, 1000]
SEEDS = ct.SEEDS


def main():
    t0 = time.time()
    acc = {}  # (lam, n, alpha) -> dict(cond -> list over seeds)
    for lam in LAMS:
        for n in NS:
            for a in ALPHAS:
                acc[(lam, n, a)] = {"V": [], "V_raw": [], "V+F_raw": [], "V+F_std": []}
    for seed in range(SEEDS):
        for a in ALPHAS:
            # 같은 시드면 α에 관계없이 z, V, T 잡음이 같고 F만 α배 되도록 난수 스트림을 다시 만든다 (짝지은 비교)
            rng_t = np.random.default_rng(7000 + seed)
            test, y_te = ct.sample(ct.N_TEST, rng_t, alpha=a)
            for n in NS:
                rng_tr = np.random.default_rng(9000 + 1000 * seed + n)
                train, y_tr = ct.sample(n, rng_tr, alpha=a)
                for lam in LAMS:
                    cell = acc[(lam, n, a)]
                    cell["V"].append(ct.acc(ct.fit_predict(train["V"], y_tr, test["V"], True, lam=lam), y_te))
                    cell["V_raw"].append(ct.acc(ct.fit_predict(train["V"], y_tr, test["V"], False, lam=lam), y_te))
                    Xtr = np.hstack([train["V"], train["F"]]); Xte = np.hstack([test["V"], test["F"]])
                    cell["V+F_raw"].append(ct.acc(ct.fit_predict(Xtr, y_tr, Xte, False, lam=lam), y_te))
                    cell["V+F_std"].append(ct.acc(ct.fit_predict(Xtr, y_tr, Xte, True, lam=lam), y_te))
    # 표 만들기 (시드 평균, 짝지은 차이의 표준오차)
    table = {}
    for (lam, n, a), cell in acc.items():
        v = np.array(cell["V"]); vr = np.array(cell["V_raw"]); r = np.array(cell["V+F_raw"]); s = np.array(cell["V+F_std"])
        d = r - vr   # 같은 (비표준화) 설정에서 힘장 추가의 효과: 척도 영향만 남는다
        table[f"lam={lam}|n={n}|alpha={a}"] = {
            "V_std": round(float(v.mean()), 4), "V_raw": round(float(vr.mean()), 4),
            "V+F_raw": round(float(r.mean()), 4), "V+F_std": round(float(s.mean()), 4),
            "gain_raw_minus_V": round(float(d.mean()), 4), "se_gain_raw": round(float(d.std(ddof=1) / np.sqrt(len(d))), 4),
            "gain_std_minus_V": round(float((s - v).mean()), 4),
            "scale_ignored": bool(abs(d.mean()) < 0.01),
        }
    # 척도 무시 칸의 비율: 원단위 조건 중 α ≤ 0.01
    small = [t for k, t in table.items() if "alpha=0.01" in k or "alpha=0.001" in k]
    big = [t for k, t in table.items() if "alpha=1.0" in k or "alpha=0.1" in k]
    # 비단조성: 고정된 (λ, n)에서 α에 따른 원단위 이득 순서
    nonmono = []
    for lam in LAMS:
        for n in NS:
            g = [table[f"lam={lam}|n={n}|alpha={a}"]["gain_raw_minus_V"] for a in ALPHAS]
            # α가 1 → 0.001로 줄 때 이득이 단조 감소하지 않으면 비단조
            diffs = np.diff(g)
            if not (np.all(diffs <= 0.005) or np.all(diffs >= -0.005)):
                nonmono.append({"lam": lam, "n": n, "gain_by_alpha(1,0.1,0.01,0.001)": [round(x, 3) for x in g]})
    # λ에 따른 순위 뒤집힘: 같은 (n, α)에서 원단위와 V의 부호가 λ에 따라 바뀌는 칸
    sign_flip = []
    for n in NS:
        for a in ALPHAS:
            signs = [np.sign(table[f"lam={lam}|n={n}|alpha={a}"]["gain_raw_minus_V"]) for lam in LAMS]
            if len(set(signs)) > 1:
                sign_flip.append({"n": n, "alpha": a, "gain_sign_by_lam(0.1,1,10)": [int(x) for x in signs]})
    summary = {
        "cells_total": len(table),
        "scale_ignored_cells_alpha_le_0.01": int(sum(t["scale_ignored"] for t in small)),
        "cells_alpha_le_0.01": len(small),
        "cells_alpha_ge_0.1_where_scale_ignored": int(sum(t["scale_ignored"] for t in big)),
        "nonmonotone_alpha_profiles(lam,n)": nonmono,
        "gain_sign_flip_across_lambda": sign_flip,
        "standardized_gain_range": [round(min(t["gain_std_minus_V"] for t in table.values()), 4),
                                    round(max(t["gain_std_minus_V"] for t in table.values()), 4)],
    }
    out = {"setup": {"seeds": SEEDS, "n_test": ct.N_TEST, "alphas": ALPHAS, "lams": LAMS, "ns": NS,
                     "base": "concat_toy.sample / fit_predict (ridge, 라벨 ±1, 무잡음 평가)",
                     "scale_ignored_rule": "|mean(V+F원단위) − mean(V, 같은 비표준화 설정)| < 0.01"},
           "summary": summary, "table": table, "runtime_s": round(time.time() - t0, 1)}
    path = ROOT / "analysis" / "extra_forcefield_grid_results.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    plot(table)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("nonmonotone_alpha_profiles(lam,n)",
                                                                       "gain_sign_flip_across_lambda")},
                     ensure_ascii=False, indent=1))
    print("nonmono count", len(nonmono), "sign flips", len(sign_flip))
    print("runtime", out["runtime_s"], "wrote", path)


def plot(table):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                              "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
    if font:
        font_manager.fontManager.addfont(font)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({"font.size": 8, "axes.unicode_minus": False})
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9), sharey=True)
    vmax = 0.06
    for ax, lam in zip(axes, LAMS):
        M = np.zeros((len(NS), len(ALPHAS)))
        for i, n in enumerate(NS):
            for j, a in enumerate(ALPHAS):
                M[i, j] = table[f"lam={lam}|n={n}|alpha={a}"]["gain_raw_minus_V"]
        im = ax.imshow(M, cmap="RdBu", vmin=-vmax, vmax=vmax, origin="lower", aspect="auto")
        for i in range(len(NS)):
            for j in range(len(ALPHAS)):
                ax.text(j, i, f"{M[i, j]*100:+.1f}", ha="center", va="center", fontsize=6.2)
        ax.set_xticks(range(len(ALPHAS)))
        ax.set_xticklabels([f"{a:g}" for a in ALPHAS], fontsize=7)
        ax.set_yticks(range(len(NS)))
        ax.set_yticklabels([str(n) for n in NS], fontsize=7)
        ax.set_xlabel("힘 척도 α")
        ax.set_title(f"λ = {lam:g}: V+F원단위 - V (비표준화, %p)", fontsize=8)
    axes[0].set_ylabel("표본 수 n")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="정확도 차이 (파랑: 이득, 빨강: 손해)")
    fig.savefig(ROOT / "report" / "figures" / "extra_forcefield_grid.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
