"""우리가 만든 그래프 세 장을 report/figures/ 에 저장한다. 논문 그림은 쓰지 않는다.

  own_paired_effects.png   폐루프 쌍체 차이와 95% 구간 (재계산 결과 기준)
  own_endpoint_mass.png    RoPE 조건별 끝점 질량 (표 2 값), 균일 기준 0.040 표시
  own_bos_runs.png         BOS 편집의 실행별 성공 수 (표 6, 표 S9, S10)

입력: analysis/recompute_results.json (표 값은 recompute.py에 옮겨 적은 것)
실행: python3 analysis/plot_figures.py   (matplotlib, CPU, 수 초)
색: 시리즈 색은 #1f5fae 하나만 쓰고, 기준선은 회색 점선. dataviz 검사(단일 색 통과) 기준.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))

ACCENT = "#1f5fae"
GRID = "#dcdad3"
INK = "#1b1b1a"
REF = "#55544f"

# 한글 폰트: Linux Noto CJK 다음 macOS Apple SD Gothic Neo 순서로 찾는다
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]
for cand in FONT_CANDIDATES:
    if Path(cand).exists():
        font_manager.fontManager.addfont(cand)
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
matplotlib.rcParams["axes.unicode_minus"] = False


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def paired_effects():
    rows = RES["paired_closed_loop"]
    labels = [r["name"] for r in rows]
    y = list(range(len(rows)))[::-1]
    fig, ax = plt.subplots(figsize=(7.4, 0.38 * len(rows) + 1.3))
    for yi, r in zip(y, rows):
        lo, hi = r["ci_pp_recomputed"]
        d = r["diff_pp_recomputed"]
        ax.plot([lo, hi], [yi, yi], color=ACCENT, lw=1.6)
        ax.plot(d, yi, "o", color=ACCENT, ms=4.5)
    ax.axvline(0, color=REF, lw=0.9, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("기준 대비 성공률 차이 (pp), 쌍체 Wald 95% 구간")
    ax.set_title("폐루프 쌍체 차이 (재계산, 기준은 같은 실행)", fontsize=10, color=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "own_paired_effects.png", dpi=170)
    plt.close(fig)


def endpoint_mass():
    # 표 2 (중간 노이즈 mid, 높은 노이즈 high). 값은 논문 표 2에서 옮김. 균일 기준 0.040.
    conds = ["기준", "전 행동 토큰\nRoPE 제거", "끝점(0, 49)\nRoPE 제거", "중앙(24, 25)\nRoPE 제거", "cyclic shift"]
    high = [0.065, 0.040, 0.033, 0.065, 0.035]
    mid = [0.164, 0.040, 0.080, 0.154, 0.036]
    x = list(range(len(conds)))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.bar([i - w / 2 for i in x], high, width=w, color=ACCENT, alpha=0.45, label="high noise")
    ax.bar([i + w / 2 for i in x], mid, width=w, color=ACCENT, label="mid noise")
    for i, (h, m) in enumerate(zip(high, mid)):
        ax.text(i - w / 2, h + 0.003, f"{h:.3f}", ha="center", fontsize=6.8, color=INK)
        ax.text(i + w / 2, m + 0.003, f"{m:.3f}", ha="center", fontsize=6.8, color=INK)
    ax.axhline(0.040, color=REF, lw=0.9, ls="--", label="균일 기준 0.040")
    ax.set_xticks(x)
    ax.set_xticklabels(conds, fontsize=7.5)
    ax.set_ylabel("끝점 질량 (식 2)")
    ax.set_ylim(0, 0.19)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.set_title("upstream action RoPE 제거에 따른 12층 끝점 질량 (표 2)", fontsize=10, color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_endpoint_mass.png", dpi=170)
    plt.close(fig)


def bos_runs():
    # 표 6 (실행 A, 기준 107), 표 S9와 S10 (실행 B, 기준 114). 각 칸은 250 에피소드 중 성공 수.
    rows = [
        ("실행 A · 기준", 107), ("실행 A · BOS 점수 마스크", 116),
        ("실행 A · BOS 값 0", 140), ("실행 A · BOS 값 교체", 107),
        ("실행 B · 기준 (λ=1)", 114), ("실행 B · λ=0.5", 128), ("실행 B · λ=0", 129),
    ]
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    y = list(range(len(rows)))[::-1]
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    ax.axvline(107, color=REF, lw=0.8, ls="--")
    for yi, v in zip(y, vals):
        ax.hlines(yi, 107, v, color=ACCENT, lw=1.2, alpha=0.5)
        ax.plot(v, yi, "o", color=ACCENT, ms=6)
        ax.text(v + 1.5, yi, f"{v}", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlim(90, 150)
    ax.set_xlabel("성공 에피소드 수 (250 중)")
    ax.set_title("BOS 편집의 실행별 성공 수: 기준이 실행마다 107과 114로 다르다", fontsize=9.5, color=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "own_bos_runs.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    paired_effects()
    endpoint_mass()
    bos_runs()
    print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
