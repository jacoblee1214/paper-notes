"""논문 표 값으로 우리가 그린 그래프 세 장 (report/figures/own_*.png).

  own_rgb_gain.png   : 표 2 RGB 품질 (PSNR), 세 변형 × 두 시점. 시드 1개, 창별 SD 없음
  own_depth_source.png: 표 A3 깊이 소스별 RGB PSNR과 AbsRel (50k 스텝)
  own_pointworld.png : 표 A5 PointWorld 대 DepthWorld, 2 s와 8 s (중앙값, mm)

한글 폰트: macOS Apple SD Gothic Neo → Linux Noto CJK 순서로 찾는다.
실행: python3 analysis/plot_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
CANDIDATES = [
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]
for path in CANDIDATES:
    if Path(path).exists():
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False
INK, ACC, GREY = "#1b1b1a", "#1f5fae", "#9a978f"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)


def rgb_gain():
    labels = ["RGB\n(Ctrl-World 기준)", "RGB+D", "RGB+D\nPM-DPT"]
    ext = [22.63, 24.09, 24.11]
    wri = [16.98, 17.96, 18.00]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), sharey=False)
    for ax, vals, title in ((axes[0], ext, "외부 시점 PSNR (dB)"), (axes[1], wri, "손목 시점 PSNR (dB)")):
        bars = ax.bar(labels, vals, color=[GREY, ACC, ACC], width=0.6)
        lo = min(vals) - 1.5
        ax.set_ylim(lo, max(vals) + 0.9)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:.2f}", ha="center", fontsize=9, color=INK)
        ax.set_title(title, fontsize=10, color=INK)
        style(ax)
    fig.suptitle("표 2: RGB 품질, 깊이 공동 예측의 효과 (단일 실행, 창별 SD 없음)", fontsize=10, color=INK)
    fig.text(0.01, 0.01, "축 시작값은 0이 아님. 차이는 외부 +1.46 (RGB+D), +1.48 (PM-DPT) dB. 출처: arXiv:2610.08780 표 2.",
             fontsize=7.5, color=GREY)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(FIG / "own_rgb_gain.png", dpi=200)
    plt.close(fig)


def depth_source():
    labels = ["S2M2\n(논문 기본)", "ZED NEURAL", "ZED ULTRA"]
    ext_psnr, wri_psnr = [24.07, 24.26, 22.11], [17.68, 17.77, 16.28]
    ext_abs, wri_abs = [0.074, 0.079, 0.194], [0.266, 0.254, 0.461]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    x = range(len(labels))
    w = 0.38
    ax = axes[0]
    ax.bar([i - w / 2 for i in x], ext_psnr, w, color=ACC, label="외부")
    ax.bar([i + w / 2 for i in x], wri_psnr, w, color=GREY, label="손목")
    for i in x:
        ax.text(i - w / 2, ext_psnr[i] + 0.08, f"{ext_psnr[i]:.2f}", ha="center", fontsize=8, color=INK)
        ax.text(i + w / 2, wri_psnr[i] + 0.08, f"{wri_psnr[i]:.2f}", ha="center", fontsize=8, color=INK)
    ax.set_ylim(14, 26)
    ax.set_xticks(list(x), labels)
    ax.set_title("RGB PSNR (dB), 높을수록 좋음", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    ax = axes[1]
    ax.bar([i - w / 2 for i in x], ext_abs, w, color=ACC, label="외부")
    ax.bar([i + w / 2 for i in x], wri_abs, w, color=GREY, label="손목")
    for i in x:
        ax.text(i - w / 2, ext_abs[i] + 0.008, f"{ext_abs[i]:.3f}", ha="center", fontsize=8, color=INK)
        ax.text(i + w / 2, wri_abs[i] + 0.008, f"{wri_abs[i]:.3f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(list(x), labels)
    ax.set_title("깊이 AbsRel, 낮을수록 좋음 (각자 학습 소스 기준)", fontsize=10, color=INK)
    style(ax)
    fig.suptitle("표 A3: 학습 깊이 소스에 따른 결과 (50k 스텝, K=10 롤아웃)", fontsize=10, color=INK)
    fig.text(0.01, 0.01, "깊이 지표는 각 모델이 학습한 소스 기준이라 행끼리 절대 비교가 아님. 출처: arXiv:2610.08780 표 A3.",
             fontsize=7.5, color=GREY)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(FIG / "own_depth_source.png", dpi=200)
    plt.close(fig)


def pointworld():
    horizons = [2, 8]
    series = {
        "추적 점 ℓ2, 움직이는 점 (PointWorld)": ([23, 56], ACC, "-"),
        "추적 점 ℓ2, 움직이는 점 (DepthWorld)": ([31, 46], ACC, "--"),
        "Chamfer, 전체 장면 (PointWorld)": ([8, 18], GREY, "-"),
        "Chamfer, 전체 장면 (DepthWorld)": ([8, 9], GREY, "--"),
    }
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for name, (vals, color, ls) in series.items():
        ax.plot(horizons, vals, ls, marker="o", color=color, label=name, lw=1.8)
        for h, v in zip(horizons, vals):
            ax.text(h + 0.12, v, f"{v}", fontsize=8, color=INK, va="center")
    ax.set_xticks(horizons, ["2 s", "8 s"])
    ax.set_ylabel("중앙값 (mm), 낮을수록 좋음", fontsize=9, color=INK)
    ax.set_xlim(1.6, 8.8)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    style(ax)
    ax.set_title("표 A5: PointWorld 대 DepthWorld (공정한 대각선 비교)", fontsize=10, color=INK)
    fig.text(0.01, 0.01, "DepthWorld는 VAE 왕복 GT 기준, PointWorld는 원본 GT 기준. 출처: arXiv:2610.08780 표 A5.",
             fontsize=7.5, color=GREY)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "own_pointworld.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    rgb_gain()
    depth_source()
    pointworld()
    for name in ("own_rgb_gain.png", "own_depth_source.png", "own_pointworld.png"):
        print("wrote", FIG / name)
