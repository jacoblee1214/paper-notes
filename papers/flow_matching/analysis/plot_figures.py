"""우리가 그린 그래프 한 장을 report/figures/에 만든다(논문 그림은 쓰지 않는다. 라이선스가 CC BY가 아님).

  own_nfe_fid.png : 표 1 왼쪽의 NFE(로그 축)와 FID, 세 데이터셋, 다섯 방법 (논문 표 값을 옮김)
실행: python3 analysis/plot_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def korean_font():
    for name in ("Noto Sans CJK KR", "Apple SD Gothic Neo", "AppleSDGothicNeo"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()

# 표 1 왼쪽: (NFE, FID)
DATA = {
    "CIFAR-10": {"DDPM": (274, 7.48), "SM": (242, 19.94), "ScoreFlow": (428, 20.78), "FM-Dif": (183, 8.06), "FM-OT": (142, 6.35)},
    "ImageNet-32": {"DDPM": (262, 6.99), "SM": (178, 5.68), "ScoreFlow": (195, 14.14), "FM-Dif": (193, 6.37), "FM-OT": (122, 5.02)},
    "ImageNet-64": {"DDPM": (264, 17.36), "SM": (441, 19.74), "ScoreFlow": (601, 24.95), "FM-Dif": (187, 16.88), "FM-OT": (138, 14.45)},
}
STYLE = {"DDPM": "#888888", "SM": "#b0b0b0", "ScoreFlow": "#c9a227", "FM-Dif": "#4c8bc2", "FM-OT": "#1f5fae"}

fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), dpi=200, sharey=False)
for ax, (name, pts) in zip(axes, DATA.items()):
    for m, (nfe, fid) in pts.items():
        ax.scatter(nfe, fid, s=38 if m == "FM-OT" else 22, color=STYLE[m],
                   edgecolor="black" if m == "FM-OT" else "none", linewidth=0.6, zorder=3)
        ax.annotate(m, (nfe, fid), xytext=(-4, 4) if nfe > 400 else (4, 3), textcoords="offset points",
                    fontsize=6.5, ha="right" if nfe > 400 else "left")
    ax.set_xscale("log")
    ax.set_xlim(90, 800)
    ax.set_xticks([100, 200, 400, 800])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(["100", "200", "400", "800"]))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("NFE (로그 축, dopri5 1e-5)", fontsize=7.5)
    ax.set_title(name, fontsize=8.5)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, linewidth=0.5)
axes[0].set_ylabel("FID (낮을수록 좋음)", fontsize=7.5)
fig.suptitle("표 1 값: 샘플링 비용과 품질 (단일 실행, 시드 정보 없음)", fontsize=8.5, y=0.99)
fig.tight_layout()
fig.savefig(FIG / "own_nfe_fid.png")
print("wrote", FIG / "own_nfe_fid.png")
