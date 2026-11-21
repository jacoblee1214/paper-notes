"""우리가 그린 그래프 세 장 (report/figures/own_*.png). 논문 그림은 쓰지 않는다.

입력: analysis/recompute_results.json (표 15 재계산), analysis/toy_interaction_results.json (토이 T1),
      표 5의 값 (recompute.py의 T5 상수와 같은 값).
실행: python3 analysis/plot_figures.py
"""
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[1]
ANA = ROOT / "analysis"
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def korean_font():
    for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.formatter.use_mathtext"] = False
INK, ACC, GREY, ORANGE, GREEN = "#1b1b1a", "#1f5fae", "#9a9890", "#d9822b", "#4a9d5b"


def toy_rho():
    d = json.loads((ANA / "toy_interaction_results.json").read_text(encoding="utf-8"))
    hs = np.array(d["hs"])
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    style = {"additive": (GREY, "가법 힘"), "gain": (ORANGE, "게인 고장"),
             "saturating": (GREEN, "포화"), "damping": (ACC, "감쇠 고장")}
    from matplotlib.ticker import FuncFormatter
    for k, (c, label) in style.items():
        vals = np.array([row["rho"] for row in d["rho"][k]])
        slope = d["loglog_slope"][k]
        text = f"{label}, 기울기 {slope:.2f}" if slope and abs(slope) > 0.05 else f"{label}, 거의 일정"
        ax.plot(hs, vals, "o-", color=c, label=text)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(hs)
    ax.set_xticklabels([f"{h:g}" for h in hs])
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"1e{int(round(np.log10(v)))}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_ylim(1e-14, 10)
    ax.set_xlabel("전이 길이 h (진자 단위)")
    ax.set_ylabel("상호작용 비율 ρ (토이)")
    ax.set_title("토이 진자: 메커니즘별 ρ와 전이 길이 (CPU 토이, 논문 값과 다른 설정)", fontsize=10)
    ax.grid(alpha=.3)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "own_rho_duration.png", dpi=200)
    plt.close(fig)


def planning_ci():
    d = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))
    rows = [r for r in d["table15"] if r["encoder"] == "MAE-B"]
    labels = [r["env"] for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    for offset, key, c, lab in ((-0.25, "k_pred", GREY, "학습 예측기"), (0.0, "k_oracle", "#6c6c6c", "오라클 (결과 없음)"),
                                 (0.25, "k_cf", ACC, "반사실 세계모델")):
        pts, lo, hi = [], [], []
        for r in rows:
            k = r[key]
            p = 100 * k / r["n"]
            ci = r["wilson_cf"] if key == "k_cf" else r["wilson_pred"]
            if key == "k_oracle":
                # 오라클 구간은 같은 Wilson 식으로 다시 계산
                from math import sqrt
                z, n = 1.96, r["n"]
                den = 1 + z * z / n
                c0 = (k / n + z * z / (2 * n)) / den
                h0 = z * sqrt((k / n) * (1 - k / n) / n + z * z / (4 * n * n)) / den
                ci = [100 * (c0 - h0), 100 * (c0 + h0)]
            pts.append(p)
            lo.append(p - ci[0])
            hi.append(ci[1] - p)
        ax.errorbar(x + offset, pts, yerr=[lo, hi], fmt="o", color=c, capsize=3, label=lab, ms=5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(n={r['n']})" for l, r in zip(labels, rows)], fontsize=8)
    ax.set_ylabel("한 번 계획 성공률 (%)")
    ax.set_ylim(0, 85)
    ax.set_title("MAE-B: 한 번 계획 성공률과 Wilson 95% 구간 (표 15에서 재계산)", fontsize=10)
    ax.grid(axis="y", alpha=.3)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "own_planning_ci.png", dpi=200)
    plt.close(fig)


def cf_r2():
    # 표 5, MAE-B (8k 설정). (오라클, 예측기, 특징 공간 전달, 읽기 후 전달)
    env = ["HandReach", "Ball-in-Cup", "FetchReach", "Liquid\ntransport", "BlueROV2"]
    oracle = [49.1, 51.1, 50.0, 50.6, 51.8]
    pred = [38.1, 41.6, 44.8, 42.1, 31.1]
    feat = [82.4, 92.0, 94.0, 82.3, 47.7]
    feat_ci = [(80, 85), (91, 93), (93, 95), (78, 86), (44, 51)]
    ro = [85.6, 94.2, 95.7, 94.8, 73.6]
    ro_ci = [(83, 87), (93, 95), (94, 97), (92, 96), (71, 77)]
    x = np.arange(len(env))
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.bar(x - 0.27, pred, 0.27, color=GREY, label="학습 예측기 (결과 없음)")
    ax.bar(x, feat, 0.27, color="#8fb3e0", yerr=[[f - a for f, (a, _) in zip(feat, feat_ci)],
                                                  [b - f for f, (_, b) in zip(feat, feat_ci)]], capsize=3,
           label="특징 공간 전달")
    ax.bar(x + 0.27, ro, 0.27, color=ACC, yerr=[[r - a for r, (a, _) in zip(ro, ro_ci)],
                                                [b - r for r, (_, b) in zip(ro, ro_ci)]], capsize=3,
           label="읽기 후 전달")
    ax.scatter(x, oracle, marker="_", s=400, color=INK, zorder=5, label="오라클 (결과 없음)")
    ax.set_xticks(x)
    ax.set_xticklabels(env, fontsize=8)
    ax.set_ylabel("물리 반사실 R² (%)")
    ax.set_ylim(0, 105)
    ax.set_title("MAE-B, 8k 사실 에피소드 (표 5 값, 구간은 논문 값)", fontsize=10)
    ax.grid(axis="y", alpha=.3)
    ax.legend(fontsize=7.5, frameon=False, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "own_cf_r2.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    toy_rho()
    planning_ci()
    cf_r2()
    print("ok")
