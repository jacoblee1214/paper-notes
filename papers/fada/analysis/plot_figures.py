"""재계산 결과를 그래프 두 장으로 그린다 (우리가 만든 그림만 사용).

실행: python3 analysis/plot_figures.py  → report/figures/own_effects.png, own_success.png
폰트: 한글 폰트를 Linux(Noto CJK) → macOS(Apple SD Gothic Neo) 순서로 찾는다.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def pick_korean_font():
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo", "AppleSDGothicNeo"]:
        if cand in names:
            return cand
    return None


KR = pick_korean_font()
if KR:
    plt.rcParams["font.family"] = KR
plt.rcParams["axes.unicode_minus"] = False


def diff_ci(m1, s1, n1, m2, s2, n2, level=0.95):
    """평균 차이 m1 - m2 와 Welch 신뢰구간."""
    v1, v2 = s1 ** 2 / n1, s2 ** 2 / n2
    se = (v1 + v2) ** 0.5
    df = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))
    q = stats.t.ppf(0.5 + level / 2, df)
    d = m1 - m2
    return d, d - q * se, d + q * se


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return c - h, c + h


def effects_figure():
    rows = [
        # (라벨, FADA 평균, sd, n, 비교군 평균, sd, n, 구분)
        ("실로봇 G1 부하 (Ev)", 0.797, 0.018, 5, 1.289, 0.711, 5, "hw"),
        ("실로봇 G1 Kungfu (Empjpe)", 0.86, 0.084, 5, 1.18, 0.103, 5, "hw"),
        ("실로봇 T1 부하 (Ev)", 0.866, 0.040, 5, 1.043, 0.025, 5, "hw"),
        ("시뮬 G1 Slope", 0.800, 0.236, 20, 0.946, 0.358, 20, "sim"),
        ("시뮬 G1 Kungfu", 0.714, 0.048, 20, 0.961, 0.055, 20, "sim"),
        ("시뮬 T1 부하", 0.885, 0.135, 20, 1.042, 0.147, 20, "sim"),
        ("시뮬 T1 Slope", 0.914, 0.193, 20, 1.034, 0.334, 20, "sim"),
        ("시뮬 T1 Falcon", 0.347, 0.043, 20, 0.880, 0.094, 20, "sim"),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.3), dpi=200)
    for i, (lab, m1, s1, n1, m2, s2, n2, kind) in enumerate(rows):
        d, lo, hi = diff_ci(m1, s1, n1, m2, s2, n2)
        y = len(rows) - i
        color = "#1f5fae" if kind == "hw" else "#7a7a72"
        ax.errorbar(d, y, xerr=[[d - lo], [hi - d]], fmt="o", color=color, ecolor=color, capsize=3)
    ax.axvline(0, color="#333", lw=0.8, ls="--")
    ax.set_yticks([len(rows) - i for i in range(len(rows))])
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel("FADA - FADA-zs 평균 차이 (정규화 오차, 음수일수록 개선), 95% Welch 구간", fontsize=8)
    ax.set_title("논문 표 1·2 재계산: 파란색 n=5 (실로봇), 회색 n=20 (시뮬, 시도 수 가정)", fontsize=8.5)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", color="#e5e3dc", lw=0.6)
    fig.tight_layout()
    fig.savefig(OUT / "own_effects.png")
    plt.close(fig)


def success_figure():
    groups = [
        ("실로봇 Slope", [(1, 5, "FADA-zs"), (4, 5, "FADA"), (0, 5, "TF-DAgger")]),
        ("실로봇 Basket", [(1, 5, "FADA-zs"), (5, 5, "FADA"), (0, 5, "TF-DAgger")]),
        ("그림 8 제로샷 (MuJoCo, G1)", [(10, 10, "FADA"), (7, 10, "-A"), (4, 10, "-B"), (1, 10, "-A·B")]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), dpi=200, gridspec_kw={"width_ratios": [1, 1, 1.3]})
    for ax, (title, vals) in zip(axes, groups):
        for j, (k, n, lab) in enumerate(vals):
            p = k / n
            lo, hi = wilson(k, n)
            ax.errorbar(p * 100, j, xerr=[[(p - lo) * 100], [(hi - p) * 100]], fmt="o", color="#1f5fae", capsize=3)
            ax.text(104, j, f"{k}/{n}", va="center", fontsize=7)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels([v[2] for v in vals], fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlim(-3, 120)
        ax.set_xticks([0, 50, 100])
        ax.tick_params(axis="x", labelsize=7)
        ax.set_title(title, fontsize=7.8)
        ax.grid(axis="x", color="#e5e3dc", lw=0.6)
    axes[0].set_xlabel("성공률 (%), Wilson 95% 구간", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "own_success.png")
    plt.close(fig)


if __name__ == "__main__":
    effects_figure()
    success_figure()
    print("saved", OUT / "own_effects.png", OUT / "own_success.png", "font:", KR)
