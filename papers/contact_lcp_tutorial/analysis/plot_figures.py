"""Own plots (not from the paper) for the report and the blog post.

Reads analysis/formulation_checks_results.json and writes report/figures/own_*.png.
Run: python3 analysis/plot_figures.py   (matplotlib, numpy; a few seconds)
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "report", "figures")
os.makedirs(FIG, exist_ok=True)
with open(os.path.join(ROOT, "analysis", "formulation_checks_results.json"), encoding="utf-8") as fh:
    R = json.load(fh)

from matplotlib import font_manager  # noqa: E402

# Korean font: Linux Noto CJK first, then macOS Apple SD Gothic Neo (same order as the other folders)
KOREAN_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]
_font_name = next(
    (font_manager.FontProperties(fname=p).get_name() for p in KOREAN_FONT_CANDIDATES if os.path.exists(p)),
    "DejaVu Sans",
)
for _p in KOREAN_FONT_CANDIDATES:
    if os.path.exists(_p):
        font_manager.fontManager.addfont(_p)
plt.rcParams.update({
    "font.family": _font_name,
    "axes.unicode_minus": False,
    "font.size": 10,
})
INK = "#1b1b1a"
BLUE = "#1f5fae"
ORANGE = "#c9642b"
GREY = "#8a8880"


def fig_lift_off():
    """V4: sliding contact, relaxed (no de Saxce) vs exact."""
    d = R["V4_lift_off_relaxation_vs_exact"]
    mu = d["mu"]
    vt = np.linspace(0, 1.5, 200)
    fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=160)
    ax.plot(vt, mu * vt, color=ORANGE, lw=2, label=r"완화 (Γ 제거): $v_{n}^{+} = \mu |v_t^{+}|$")
    ax.plot(vt, 0 * vt, color=BLUE, lw=2, label=r"정확 (Γ 포함): $v_{n}^{+} = 0$")
    rel = d["relaxed"]
    ax.scatter([abs(rel["v_plus"][1])], [rel["v_n_plus"]], color=ORANGE, zorder=5, s=36)
    ax.annotate(f"토이 V4: v_t+ = {abs(rel['v_plus'][1]):.2f}, v_n+ = {rel['v_n_plus']:.2f}",
                xy=(abs(rel["v_plus"][1]), rel["v_n_plus"]), xytext=(0.05, 0.42),
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.8), fontsize=8.5, color=INK)
    ax.set_xlabel(r"끝 속도의 접선 크기 $|v_t^{+}|$ (m/s)")
    ax.set_ylabel(r"끝 속도의 법선 $v_n^{+}$ (m/s)")
    ax.set_title("슬라이딩 접촉: 볼록 완화가 만드는 들림 (lift-off)", fontsize=10.5, color=INK)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "own_lift_off.png"))
    plt.close(fig)


def fig_pyramid():
    """V3: anisotropy of the polyhedral (pyramid) friction set."""
    d = R["V3_pyramid_anisotropy"]
    mu = d["mu"]
    fig, axes = plt.subplots(1, 4, figsize=(8.6, 2.6), dpi=160, subplot_kw={"aspect": "equal"})
    for ax, K in zip(axes, [3, 4, 6, 8]):
        ang = 2 * np.pi * np.arange(K + 1) / K
        ax.plot(mu * np.cos(ang), mu * np.sin(ang), color=ORANGE, lw=1.6)
        th = np.linspace(0, 2 * np.pi, 360)
        ax.plot(mu * np.cos(th), mu * np.sin(th), color=GREY, lw=0.8, ls="--")
        r_min = d["by_K"][str(K)]["min_radius_numeric"]
        ax.set_title(f"K = {K}: min/max = {d['by_K'][str(K)]['anisotropy_min_over_max']:.2f}",
                     fontsize=8.5, color=INK)
        ax.set_xlim(-0.75, 0.75)
        ax.set_ylim(-0.75, 0.75)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.plot([0, r_min], [0, 0], color=BLUE, lw=1.2)
    fig.suptitle("피라미드 근사의 마찰 한계 (실선: 다면체, 점선: 원뿔, 파란선: 최소 반지름)",
                 fontsize=9.5, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "own_pyramid_anisotropy.png"))
    plt.close(fig)


def fig_regularization():
    """V5: static friction under regularization R; f_t = W/(1+R) for every pressing force P."""
    d = R["V5_regularization_static_friction_loss"]
    W = d["W"]
    Rs = np.linspace(0, 1.5, 100)
    fig, ax = plt.subplots(figsize=(6.2, 3.4), dpi=160)
    ax.plot(Rs, W / (1 + Rs), color=ORANGE, lw=2, label=r"$f_t^{*}=W/(1+R)$ (내부 최적해)")
    ax.axhline(W, color=BLUE, lw=1.6, label=r"버틸 수 있어야 하는 접선 하중 $W$")
    for k, v in d["by_R"].items():
        ax.scatter([float(k)], [v["f_t_numeric"]], color=INK, s=20, zorder=5)
    ax.set_xlabel("정규화 R (스케일된 값)")
    ax.set_ylabel(r"접선 힘 $f_t$ (N)")
    ax.set_title("정규화가 있으면 정적 마찰이 사라진다 (토이 V5)", fontsize=10.5, color=INK)
    ax.set_ylim(0, 1.2)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "own_regularization_slip.png"))
    plt.close(fig)


def fig_penalty():
    """V8: penalty penetration m g / k versus stiffness; complementarity gives 0 at rest."""
    d = R["V8_penalty_penetration_scaling"]
    ks = np.array([float(k) for k in d["by_k"]])
    pen = np.array([v["penetration_m"] for v in d["by_k"].values()]) * 1e3
    fig, ax = plt.subplots(figsize=(6.0, 3.3), dpi=160)
    ax.loglog(ks, pen, "o-", color=ORANGE, lw=1.8, label=r"벌점(스프링) 모형: $d = mg/k$")
    ax.axhline(0.0, color=BLUE, lw=1.6)
    ax.set_xlabel(r"접촉 강성 $k$ (N/m)")
    ax.set_ylabel("정지 시 관통 깊이 (mm)")
    ax.set_title("벌점 모형은 강성에 따라 관통하고, 상보성은 정지 시 0 (토이 V8)",
                 fontsize=9.5, color=INK)
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "own_penalty_penetration.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig_lift_off()
    fig_pyramid()
    fig_regularization()
    fig_penalty()
    print("wrote figures to", FIG)
