"""우리가 만든 그래프 네 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_sim_tradeoff.png : 표 2의 ID 위치 오차 대 OOD-Geometry 생존율 (R_safe 훑기)과 부록 K 표 8의 OOD 생존율 비교
own_real_ci.png      : 표 4 실로봇 생존율과 Wilson 95% 구간 (범주당 15회 가정)
own_estimator_bins.png: 표 5 안전 추정기 구간별 사람 판정 구성 (500개 상태)
own_toy_checks.png   : 토이 T2 (잠재 차원별 셸 두께)와 토이 T1 (timeout=1 규칙의 목표 편향)
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; recompute.py, toy_*.py 결과 json 필요)
한글 폰트는 Linux Noto CJK, macOS Apple SD Gothic Neo 순서로 찾는다.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

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
INK, ACC, GREY, ORANGE = "#1b1b1a", "#1f5fae", "#9a9890", "#c4651a"


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def plot_tradeoff():
    # 표 2: (R, ID ep cm, OOD-Geometry SR %)
    cmp_rows = [(1.0, 15.6, 82.5), (1.5, 6.0, 62.4), (2.0, 4.5, 46.9), (2.5, 4.1, 34.1), (3.0, 3.8, 26.3), (100, 3.5, 5.1)]
    base = {"UMI-on-Legs": (4.9, 4.7), "Latent Shielding": (4.4, 5.6), "Neural CBF": (5.4, 7.1)}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    ax.plot([r[2] for r in cmp_rows], [r[1] for r in cmp_rows], "-o", color=ACC, label="CMP (R_safe 훑기)")
    for R, ep, sr in cmp_rows:
        ax.annotate(f"R={R:g}", (sr, ep), textcoords="offset points", xytext=(5, 4), fontsize=7.5, color=ACC)
    for (name, (ep, sr)), m in zip(base.items(), ["s", "^", "D"]):
        ax.scatter([sr], [ep], marker=m, color=GREY, s=40, label=name, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(3.2, 18)
    ax.set_xlim(0, 96)
    ax.set_yticks([4, 6, 10, 15])
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("OOD-Geometry 생존율 (%)")
    ax.set_ylabel("ID 위치 오차 e_p (cm, 로그)")
    ax.set_title("표 2: 정확도와 생존율의 교환", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc="upper left")
    ax = axes[1]
    names = ["UMI-on-Legs", "명령 클리핑", "후방 증강", "CMP (R=2)"]
    geo = [4.7, 90.2, 87.0, 46.9]
    rev = [19.0, 33.0, 30.2, 65.2]
    x = np.arange(4)
    ax.bar(x - 0.18, geo, 0.36, color=[GREY, ORANGE, ORANGE, ACC], label="OOD-Geometry")
    ax.bar(x + 0.18, rev, 0.36, color=[GREY, ORANGE, ORANGE, ACC], alpha=0.5, label="OOD-Rev (밝은 막대)")
    for i, (g, r) in enumerate(zip(geo, rev)):
        ax.text(i - 0.18, g + 1.5, f"{g:g}", ha="center", fontsize=7.5)
        ax.text(i + 0.18, r + 1.5, f"{r:g}", ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_ylabel("생존율 (%)")
    ax.set_title("부록 K 표 8 (ID e_r 0.105–0.130 rad)", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "own_sim_tradeoff.png", dpi=170)
    plt.close(fig)


def plot_real():
    counts = {"UMI-on-Legs": (12, 0, 0), "Latent Shielding": (11, 5, 3), "Neural CBF": (12, 9, 6), "CMP": (15, 14, 13)}
    cats = ["ID", "중등도 OOD", "극단 OOD"]
    cols = {"UMI-on-Legs": GREY, "Latent Shielding": "#c9c6bd", "Neural CBF": "#8fb0d9", "CMP": ACC}
    fig, ax = plt.subplots(figsize=(8.2, 3.7))
    w = 0.2
    for j, (m, v) in enumerate(counts.items()):
        pct = np.array([100 * k / 15 for k in v])
        ci = np.array([wilson(k, 15) for k in v])
        x = np.arange(3) + (j - 1.5) * w
        ax.bar(x, pct, w, color=cols[m], label=m)
        ax.errorbar(x, pct, yerr=[pct - ci[:, 0], ci[:, 1] - pct], fmt="none", ecolor=INK, capsize=2, lw=0.9)
        for xi, k in zip(x, v):
            ax.text(xi, 2, f"{k}", ha="center", color="white" if (m in ("CMP", "UMI-on-Legs") and k > 0) else INK, fontsize=7.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(cats)
    ax.set_ylabel("생존율 (%) / 막대 안은 성공 횟수 (분모 15)")
    ax.set_ylim(0, 108)
    ax.set_title("표 4: 실로봇 생존율과 Wilson 95% 구간 (범주당 5 과제 x 3회 = 15회 가정)", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(FIG / "own_real_ci.png", dpi=170)
    plt.close(fig)


def plot_bins():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))["table5"]["counts_500"]
    labels = ["W_max < 0.6", "0.6 ≤ W_max ≤ 0.8", "W_max > 0.8"]
    s, u = np.array(rec["Salvageable"]), np.array(rec["Unsalvageable"])
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    x = np.arange(3)
    ax.bar(x, s, 0.55, color=ACC, label="사람 판정: 구제 가능")
    ax.bar(x, u, 0.55, bottom=s, color=ORANGE, label="사람 판정: 구제 불가")
    for i in range(3):
        tot = s[i] + u[i]
        ax.text(i, tot + 6, f"{tot}개 중 불가 {100 * u[i] / tot:.0f}%", ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("롤아웃 상태 수 (표 5, 500개)")
    ax.set_ylim(0, 340)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "own_estimator_bins.png", dpi=170)
    plt.close(fig)


def plot_toys():
    shell = json.loads((ANA / "toy_shell_geometry_results.json").read_text(encoding="utf-8"))
    lam = json.loads((ANA / "toy_lambda_bound_results.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6))
    rng = np.random.default_rng(1)
    ax = axes[0]
    eps = 0.1
    for w, c in ((0.9, ACC), (0.8, ORANGE), (0.6, GREY)):
        R = 1 / (w + eps) ** 3
        n = R * np.linalg.norm(rng.standard_normal((100000, 6)), axis=1)
        ax.hist(n, bins=np.linspace(0, 12, 120), density=True, alpha=0.6, color=c, label=f"W={w} (R={R:.2f})")
    ax.axvline(2.0, color=INK, ls="--", lw=1)
    ax.text(2.1, ax.get_ylim()[1] * 0.55, "R_safe=2.0", fontsize=8)
    ax.set_xlabel("||z|| (d=6, 사전분포 표본)")
    ax.set_title("T2: d=6에서 노름 분포가 겹침", fontsize=9.5)
    ax.legend(fontsize=7.5)
    ax = axes[1]
    ds = ["d=2", "d=6", "d=32", "d=256"]
    inv = [shell["by_dimension"][d]["P(order inverted)"]["W 0.9 vs 0.8"] for d in ds]
    inv2 = [shell["by_dimension"][d]["P(order inverted)"]["W 0.8 vs 0.6"] for d in ds]
    x = np.arange(4)
    ax.bar(x - 0.18, [100 * v for v in inv], 0.36, color=ACC, label="W 0.9 대 0.8")
    ax.bar(x + 0.18, [100 * v for v in inv2], 0.36, color=ORANGE, label="W 0.8 대 0.6")
    ax.set_xticks(x)
    ax.set_xticklabels(ds)
    ax.set_ylabel("노름 순서가 뒤집힐 확률 (%)")
    ax.set_title("T2: 차원별 순서 뒤집힘", fontsize=9.5)
    ax.legend(fontsize=7.5)
    ax.grid(axis="y", alpha=0.25)
    ax = axes[2]
    for key, c in (("0.8", ACC), ("0.95", ORANGE)):
        rows = lam["b_timeout1_bias"][key]["rows"]
        ks = [r["k"] for r in rows if r["k"] <= 40]
        v = [np.log10(r["mean(H_k - H_inf)_safe"]) for r in rows if r["k"] <= 40]
        ax.plot(ks, v, "-o", color=c, label=f"lambda={key}")
    ax.set_xlabel("타임아웃까지 남은 스텝 k")
    ax.set_ylabel("log10 (목표 편향, timeout=1 대 없음)")
    ax.set_title("T1: timeout=1 규칙의 위쪽 편향", fontsize=9.5)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_checks.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    plot_tradeoff()
    plot_real()
    plot_bins()
    plot_toys()
    print("wrote 4 figures to", FIG)
