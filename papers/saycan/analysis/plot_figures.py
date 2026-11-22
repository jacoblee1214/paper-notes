"""우리가 그린 그래프 두 장 (논문 그림 아님). 실행: python3 analysis/plot_figures.py → report/figures/own_*.png

표 값은 analysis/recompute.py와 같은 상수를 쓴다. 한글 폰트는 macOS 후보(AppleSDGothicNeo)와 Linux 후보(Noto CJK)를 차례로 찾는다.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
N = 101


def korean_font():
    for p in ["/System/Library/Fonts/AppleSDGothicNeo.ttc",
              "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
        if Path(p).exists():
            font_manager.fontManager.addfont(p)
            return font_manager.FontProperties(fname=p).get_name()
    return "DejaVu Sans"


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return 100 * (c - h), 100 * (c + h)


def fig_overall():
    rows = [  # (라벨, 성공 수, 종류)
        ("PaLM-SayCan 계획 (모의)", 85, "plan"),
        ("PaLM-SayCan 계획 (실제 주방, 보고값 81%)", 82, "plan"),
        ("가치함수 없음 (No VF) 계획", 68, "plan"),
        ("생성 결과만 (Gen.) 계획", 75, "plan"),
        ("FLAN-SayCan 계획", 71, "plan"),
        ("PaLM-SayCan 실행 (모의)", 75, "exec"),
        ("PaLM-SayCan 실행 (실제 주방, 보고값 60%)", 61, "exec"),
        ("FLAN-SayCan 실행", 62, "exec"),
        ("BC USE 실행", 9, "exec"),
    ]
    f = korean_font()
    plt.rcParams["font.family"] = f
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for i, (lab, k, kind) in enumerate(rows[::-1]):
        pct = 100 * k / N
        lo, hi = wilson(k, N)
        color = "#1f5fae" if kind == "plan" else "#b8542a"
        ax.errorbar(pct, i, xerr=[[pct - lo], [hi - pct]], fmt="o", color=color, capsize=3)
        ax.text(hi + 1.2, i, f"{pct:.0f}%  (k={k}/{N})", va="center", fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=8.5)
    ax.set_xlim(0, 110)
    ax.set_xlabel("성공률 (%) · 막대는 Wilson 95% 구간")
    ax.axvline(50, color="#dddddd", lw=0.8, zorder=0)
    ax.grid(axis="x", color="#eeeeee", lw=0.6)
    ax.set_title("표 2·3 기반 전체 성공률 (101개 지시문, 비쌍체 구간)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "own_overall_ci.png", dpi=170)
    plt.close(fig)


def fig_family():
    fams = ["자연어 단일", "명사", "동사", "구조화", "구현체", "크라우드", "장기"]
    n_f = [15, 15, 15, 15, 11, 15, 15]
    series = {
        "PaLM-SayCan 모의 계획": [100, 67, 100, 93, 64, 87, 73],
        "PaLM-SayCan 실제 주방 계획": [93, 60, 93, 93, 64, 73, 73],
        "가치함수 없음 계획": [73, 53, 87, 93, 18, 67, 67],
        "FLAN-SayCan 모의 계획": [67, 60, 80, 100, 64, 73, 47],
    }
    colors = ["#1f5fae", "#7fa6d6", "#b8542a", "#6b9a4b"]
    f = korean_font()
    plt.rcParams["font.family"] = f
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    w = 0.2
    for j, (lab, vals) in enumerate(series.items()):
        xs = [i + (j - 1.5) * w for i in range(len(fams))]
        ax.bar(xs, vals, width=w, color=colors[j], label=lab)
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels([f"{a}\n(n={b})" for a, b in zip(fams, n_f)], fontsize=8)
    ax.set_ylabel("계획 성공률 (%)")
    ax.set_ylim(0, 115)
    ax.legend(fontsize=7.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False)
    ax.set_title("표 2·3의 가족별 계획 성공률 (숫자는 논문 표 값 그대로)", fontsize=10)
    ax.grid(axis="y", color="#eeeeee", lw=0.6)
    fig.tight_layout()
    fig.savefig(OUT / "own_family_plan.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_overall()
    fig_family()
    print("wrote", sorted(p.name for p in OUT.glob("own_*.png")))
