"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림을 다시 읽어 그린 것).

own_offline_returns.png : 그림 2b와 3b의 오프라인 반환 평균 ± SEM과 Welch p(시드 5개 가정, 그림 판독 값)
own_fig4_endpoints.png  : 그림 4의 8개 과제 끝점(그림 판독 값): DSRL-SAC 대비 TacEx 세 가지
own_fig6_forest.png     : 그림 6의 최종 반환 평균 ± 표준오차(시드 10개 가정, 그림 판독 값) 방법별
입력: analysis/figure_values.json, analysis/recompute_results.json
실행: python3 analysis/plot_figures.py   (matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ANA = ROOT / "analysis"
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
FV = json.loads((ANA / "figure_values.json").read_text(encoding="utf-8"))
RR = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))


def korean_font():
    for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
INK, ACC, GREY, RED = "#1b1b1a", "#1f5fae", "#9a9890", "#c0392b"
COL = {"T": "#1f5fae", "TV": "#d81b60", "TVL": "#e69f00", "V": "#56b4e9", "L": "#999999", "VL": "#009e73", "RE": "#555555"}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK)


def fig_offline():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), gridspec_kw={"width_ratios": [7, 4]})
    b = FV["fig2b"]["bars"]
    ax = axes[0]
    keys = ["T", "TV", "TVL", "V", "L", "VL", "RE"]
    for i, k in enumerate(keys):
        ax.errorbar(i, b[k]["mean"], yerr=b[k]["sem"], fmt="o", color=COL[k], capsize=4, ms=7)
    ax.axhline(FV["fig2b"]["drq_dashed"], color=INK, ls="--", lw=1)
    ax.text(6.4, FV["fig2b"]["drq_dashed"] + 8, "DrQ 점선 (SEM 없음)", ha="right", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys)
    ax.set_ylabel("오프라인 반환 (평균 ± SEM)")
    ax.set_title("그림 2b 단일 물체 (시드 5개 가정)", fontsize=10.5)
    v1 = RR["V1_fig2b"]["vs_RE"]
    for i, k in enumerate(keys[:-1]):
        p = v1[k]["p"]
        ax.text(i, -62, f"p={p:.2g}", ha="center", fontsize=8, color=RED if p < 0.05 else GREY)
    ax.set_ylim(-100, 450)
    style(ax)
    b3 = FV["fig3b"]
    ax = axes[1]
    keys3 = ["T", "TV", "V", "RE"]
    for i, k in enumerate(keys3):
        ax.errorbar(i, b3[k]["mean"], yerr=b3[k]["sem"], fmt="o", color=COL[k], capsize=4, ms=7)
    ax.set_xticks(range(len(keys3)))
    ax.set_xticklabels(keys3)
    ax.set_title("그림 3b 다중 물체 (시드 5개 가정)", fontsize=10.5)
    ax.text(0.5, -62, f"T 대 TV p={RR['V2_fig3']['b_T_vs_TV']['p']:.2g}", ha="center", fontsize=8, color=GREY)
    ax.set_ylim(-100, 700)
    style(ax)
    fig.suptitle("오프라인 반환: 막대 높이를 점으로 다시 그림 (그림 판독 값). 아래 p는 무작위 탐색(RE) 대비 Welch", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "own_offline_returns.png", dpi=170)
    plt.close(fig)


def fig_fig4():
    rows = RR["V5_fig4_endpoints"]["tasks"]
    names = list(rows.keys())
    short = {"58": "58 케첩", "69": "69 푸딩 접시 왼쪽", "61": "61 푸딩 트레이", "77": "77 책 캐디", "55": "55 수프", "21": "21 스토브+팬",
             "56": "56 버터", "20": "20 스토브"}
    labels = [short[n.split()[0]] for n in names]
    x = np.arange(len(names))
    w = 0.2
    fig, ax = plt.subplots(figsize=(9.6, 3.9))
    series = [("SAC", "DSRL-SAC", "#333333"), ("T", "TacEx 촉각", "#1f5fae"), ("TV", "TacEx 시각+촉각", "#d81b60"), ("TL", "TacEx 잠재+촉각", "#e69f00")]
    for j, (k, lab, c) in enumerate(series):
        ax.bar(x + (j - 1.5) * w, [rows[n][k] for n in names], w, label=lab, color=c, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=8.5)
    ax.set_ylabel("마지막 평가의 평균 성공률 (그림 판독)")
    ax.set_title("그림 4 끝점: DSRL-SAC 대비 이득은 세 과제에 모여 있음 (오차는 읽지 않음, 시드 10개)", fontsize=10)
    ax.legend(fontsize=8.5, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    ax.set_ylim(0, 1.25)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "own_fig4_endpoints.png", dpi=170)
    plt.close(fig)


def fig_fig6():
    rows = RR["V4_fig6"]["rows"]
    f6 = FV["fig6_final_return"]
    order = [("Tactile", "촉각", "#1f5fae"), ("Tactile+vision", "시각+촉각", "#d81b60"), ("Tactile+latent", "잠재+촉각", "#e69f00"),
             ("State", "상태(촉각 목표 없음)", "#4d4d4d"), ("Image", "영상(촉각 목표 없음)", "#56b4e9"), ("Latent", "잠재(촉각 목표 없음)", "#999999"),
             ("DSRL-SAC", "DSRL-SAC", "#000000")]
    fig, axes = plt.subplots(1, 5, figsize=(11.2, 3.4))
    for ax, (row, info) in zip(axes, rows.items()):
        for i, (key, lab, c) in enumerate(order):
            if key == "DSRL-SAC":
                m, se = info["DSRL-SAC"]
            else:
                e = f6[row][key][0]
                m, se = e["mean"], e["se"] if e["se"] is not None else {"Ketchup in tray": 0.020, "Stove and frying pan": 0.014}.get(row, 0.034)
            ax.errorbar(m, i, xerr=se, fmt="o", color=c, capsize=3, ms=6)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([o[1] for o in order] if ax is axes[0] else [], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(row, fontsize=9.5)
        ax.set_xlim(left=-0.02)
        style(ax)
    fig.suptitle("그림 6 최종 반환 (평균 ± 표준오차, 시드 10개 가정, 그림 판독 값; 숨은 오차 막대는 마커 반지름을 상한으로 표시)", fontsize=9.5, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "own_fig6_forest.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    fig_offline()
    fig_fig4()
    fig_fig6()
    print("wrote", [p.name for p in FIG.glob("own_*.png")])
