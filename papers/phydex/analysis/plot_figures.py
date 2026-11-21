"""논문 수치로 직접 그린 그래프 (논문 그림을 옮기지 않는다). 실행: python3 analysis/plot_figures.py

입력: analysis/recompute.py의 값과 같은 논문 수치 (Table 2–4, Figure 4·7).
출력: report/figures/own_*.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from recompute import FIG4_WITH_GT, FIG4_WITHOUT, FIG7, TABLE2, TABLE3, TABLE4, N, POLICIES, wilson

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
FONT = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                          "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})
INK2, GRID = "#55555a", "#e4e4e7"
GRAY, ORANGE, TEAL_L, TEAL, RED = "#b5b5ba", "#d9822b", "#7fb3ad", "#1b6f6a", "#b5503c"
COL = {"GR00T N1.7": GRAY, "DynamicVLA": TEAL_L, "ForceVLA": TEAL}


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print("wrote", OUT / name)


def fig_alpha():
    """Table 2: 정적 단일 목적지에서 α(무작위화 강도)에 따른 성공률, Wilson 95% 구간."""
    alphas = [0.0, 0.5, 1.0]
    fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=200)
    for i, p in enumerate(POLICIES):
        xs, ys, lo, hi = [], [], [], []
        for a in alphas:
            k = sum(round(TABLE2[a][o][i] * N / 100) for o in TABLE2[a])
            n = 3 * N
            l, h = wilson(k, n)
            xs.append(a)
            ys.append(100 * k / n)
            lo.append(max(0.0, 100 * (k / n - l)))
            hi.append(max(0.0, 100 * (h - k / n)))
        ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=3, color=COL[p], label=p, linewidth=1.6)
    ax.set_xlabel("α (무작위화 폭 배율; 0 = 없음, 1 = 전체 PhyDEX)")
    ax.set_ylabel("성공률 (%)")
    ax.set_xticks(alphas)
    ax.set_title("정적 단일 목적지 3 객체 평균 (150 rollouts/점, Wilson 95%)", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    save(fig, "own_alpha_curve.png")


def fig_gt():
    """Figure 4 재그림: 실제 값(사전·사후)과 n=50 가정의 구간. GR00T 사후 12.5%는 50회에서 불가능."""
    names = ["DynamicVLA", "GR00T N1.7", "ForceVLA"]
    fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=200)
    x = range(len(names))
    w = 0.36
    without = [FIG4_WITHOUT[p] for p in names]
    with_gt = [FIG4_WITH_GT[p] for p in names]
    b1 = ax.bar([i - w / 2 for i in x], without, w, color=GRAY, label="물리 정보 없음")
    b2 = ax.bar([i + w / 2 for i in x], with_gt, w, color=ORANGE, label="실제 물리량 입력(GT)")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6, f"{b.get_height():g}%",
                    ha="center", fontsize=8)
    # ForceVLA 구간 (n=50 가정)
    for i, p in enumerate(names):
        if p == "ForceVLA":
            from recompute import pct_to_count
            k = pct_to_count(with_gt[i], N)
            l, h = wilson(k, N)
            ax.errorbar(i + w / 2, with_gt[i], yerr=[[max(0.0, with_gt[i] - 100 * l)], [max(0.0, 100 * h - with_gt[i])]],
                        color=INK2, capsize=3, linewidth=0.9)
    ax.text(1 + w / 2, with_gt[1] + 3.5, "n 미보고\n(50회에서 12.5% 불가)", ha="center", fontsize=6.5, color=RED)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylabel("성공률 (%)")
    ax.set_ylim(0, 42)
    ax.set_title("균일 객체·정적 단일 목적지 (Fig. 4 수치, 점 추정만 있음)", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style(ax)
    save(fig, "own_gt_effect.png")


def fig_sim_real():
    """같은 두 정책의 성공률을 세 환경에서 비교: F/T 입력이 있는 ForceVLA의 이득이 환경마다 다르다."""
    groups = [
        ("시뮬 (α=1,\n정적 단일)", [3, 5], 150),  # GR00T, ForceVLA 성공 수 (n=150)
        ("휴머노이드\n(Table 3)", [9, 13], 150),
        ("실물 (Table 4)", [16, 77], 150),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
    w = 0.36
    for gi, (label, ks, n) in enumerate(groups):
        for j, (p, k) in enumerate(zip(["GR00T N1.7", "ForceVLA"], ks)):
            l, h = wilson(k, n)
            pct = 100 * k / n
            xpos = gi + (j - 0.5) * w
            ax.bar(xpos, pct, w, color=COL[p], label=p if gi == 0 else None)
            ax.errorbar(xpos, pct, yerr=[[max(0.0, pct - 100 * l)], [max(0.0, 100 * h - pct)]], color=INK2, capsize=2, linewidth=0.8)
            ax.text(xpos, 100 * h + 1.2, f"{pct:.1f}%", ha="center", fontsize=7.5)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("성공률 (%)")
    ax.set_ylim(0, 65)
    ax.set_title("GR00T N1.7 대 ForceVLA (각 150회, Wilson 95%)", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style(ax)
    save(fig, "own_sim_real_contrast.png")


def fig_failure():
    """Figure 7 재그림: 정책별 실패 모드 비율 (논문 값)."""
    modes = ["파지 오류", "접근 오류", "던지기 오류", "무효 도달"]
    cols = [ORANGE, "#c7b8e0", "#e6a4a0", "#a9c4e0"]
    fig, ax = plt.subplots(figsize=(6.4, 2.8), dpi=200)
    for yi, p in enumerate(POLICIES):
        left = 0
        for mi, v in enumerate(FIG7[p]):
            ax.barh(yi, v, left=left, color=cols[mi], label=modes[mi] if yi == 0 else None, height=0.6)
            if v >= 4:
                ax.text(left + v / 2, yi, f"{v:g}", ha="center", va="center", fontsize=7.5)
            left += v
    ax.set_yticks(range(len(POLICIES)))
    ax.set_yticklabels(POLICIES)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("실패 중 비율 (%)")
    ax.legend(frameon=False, fontsize=7.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    ax.set_title("실패 모드 분포 (Fig. 7 수치, 정책별)", fontsize=9)
    style(ax)
    ax.grid(False)
    save(fig, "own_failure_modes.png")


def fig_reach_throw():
    """본문 21.24%와 Fig. 7·Table 1 개수로 재구성한 '던지기 단계 도달' 비율 비교."""
    from recompute import TABLE1, pct_to_count
    fig, ax = plt.subplots(figsize=(6.2, 3.3), dpi=200)
    rec = []
    for i, p in enumerate(POLICIES):
        succ = sum(pct_to_count(v[i], N) for v in TABLE1.values())
        fail = 9 * N - succ
        thr_err = round(FIG7[p][2] / 100 * fail)
        rec.append(100 * (thr_err + succ) / (9 * N))
    labels = ["GR00T N1.7", "DynamicVLA", "ForceVLA", "세 정책 평균"]
    vals = rec + [sum(rec) / 3]
    bars = ax.bar(labels, vals, color=[GRAY, TEAL_L, TEAL, "#8a8a90"], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}%", ha="center", fontsize=8)
    ax.axhline(21.24, color=ORANGE, linestyle="--", linewidth=1.2)
    ax.text(3.35, 21.24 + 0.6, "본문 21.24%", color=ORANGE, fontsize=8, ha="right")
    ax.set_ylabel("던지기 단계 도달 비율 (%)")
    ax.set_ylim(0, 25)
    ax.set_title("재구성 (Fig. 7 · Table 1 개수 기반, n = 450/정책)", fontsize=9)
    style(ax)
    save(fig, "own_reach_throw.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_alpha()
    fig_gt()
    fig_sim_real()
    fig_failure()
    fig_reach_throw()
