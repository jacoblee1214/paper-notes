"""논문 표 값으로 그래프 두 장을 그린다(논문 그림은 쓰지 않는다).

출력: report/figures/own_real_ci.png (Table 3, Wilson 95% 구간), report/figures/own_transfer_ci.png (Table 4).
실행: python3 analysis/plot_figures.py
한글 폰트: Linux Noto CJK → macOS Apple SD Gothic Neo 순서로 찾는다.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from recompute import N, COLS, T3, T4, T4_COLS, wilson

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(os.path.dirname(HERE), "report", "figures")


def set_korean_font():
    for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


LABELS = {"pi05": "π0.5", "vita": "VITRA", "wo_pretrain": "사전학습 없음", "wo_omnishare": "OmniShare 없음",
          "wo_adv": "적대 학습 없음", "ours": "AdvDex"}
COL_KO = {"grasp_single": "단일 파지", "grasp_multi": "다중 파지", "pour": "물 붓기", "push": "큐브 밀기",
          "stack": "병 쌓기", "unseen_obj": "미학습 물체", "unseen_env": "미학습 환경"}


def plot_real():
    methods = ["ours", "pi05", "vita", "wo_adv", "wo_omnishare"]
    width = 0.16
    fig, ax = plt.subplots(figsize=(9.2, 4.2), dpi=160)
    for j, m in enumerate(methods):
        xs, ys, lo, hi = [], [], [], []
        for i, c in enumerate(COLS):
            k = round(T3[m][i] * N / 100)
            l, u = wilson(k, N)
            x = i + (j - 2) * width
            xs.append(x); ys.append(100 * k / N); lo.append(100 * (k / N - l)); hi.append(100 * (u - k / N))
        ax.bar(xs, ys, width=width, label=LABELS[m], alpha=0.9)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor="#333", capsize=2, lw=0.8)
    ax.set_xticks(range(len(COLS)))
    ax.set_xticklabels([COL_KO[c] for c in COLS], fontsize=8.5)
    ax.set_ylabel("성공률 (%)")
    ax.set_ylim(0, 115)
    ax.axvline(4.5, color="#999", ls="--", lw=0.8)
    ax.legend(fontsize=7.5, ncol=5, loc="upper center", frameon=False)
    ax.set_title("Table 3 재그림: 실로봇 성공률 (과제당 20회, Wilson 95% 구간)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "own_real_ci.png"))
    plt.close(fig)


def plot_transfer():
    methods = ["ours", "vita", "pi05", "wo_adv_pre", "wo_adv_post"]
    names = {"ours": "AdvDex", "vita": "VITRA", "pi05": "π0.5", "wo_adv_pre": "적대 없음(사전학습)", "wo_adv_post": "적대 없음(후학습)"}
    width = 0.16
    fig, ax = plt.subplots(figsize=(8.0, 3.8), dpi=160)
    for j, m in enumerate(methods):
        xs, ys, lo, hi = [], [], [], []
        for i in range(len(T4_COLS)):
            k = round(T4[m][i] * N / 100)
            l, u = wilson(k, N)
            x = i + (j - 2) * width
            xs.append(x); ys.append(100 * k / N); lo.append(100 * (k / N - l)); hi.append(100 * (u - k / N))
        ax.bar(xs, ys, width=width, label=names[m], alpha=0.9)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor="#333", capsize=2, lw=0.8)
    ax.set_xticks(range(len(T4_COLS)))
    ax.set_xticklabels(["상자 인형", "버튼 누르기", "병 옮기기", "도구 사용"], fontsize=8.5)
    ax.set_ylabel("성공률 (%)")
    ax.set_ylim(0, 115)
    ax.legend(fontsize=7.5, ncol=3, loc="upper center", frameon=False)
    ax.set_title("Table 4 재그림: 사람 전용 과제 공동학습 (과제당 20회 가정, Wilson 95% 구간)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "own_transfer_ci.png"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    set_korean_font()
    plot_real()
    plot_transfer()
    print("wrote", os.path.join(FIG, "own_real_ci.png"), os.path.join(FIG, "own_transfer_ci.png"))
