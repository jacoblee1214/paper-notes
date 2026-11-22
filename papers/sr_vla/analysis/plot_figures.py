"""우리가 그린 그래프 두 장 (논문 그림은 쓰지 않음). 실행: python3 analysis/plot_figures.py

own_stall.png       : 표 III의 질의당 정지(ms)와 속도비 (논문 수치 그대로)
own_success_ci.png  : 스위트별 성공률 차이와 95% 구간 (시행 수 500 가정, recompute.py와 같은 값)
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "report", "figures")

# 한글 폰트: Linux 후보를 먼저 찾고, 없으면 macOS 후보를 쓴다
for cand in ["Noto Sans CJK KR", "Noto Sans KR", "AppleSDGothicNeo", "Apple SD Gothic Neo"]:
    if any(cand in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

COL = {"base": "#8a8a85", "other": "#c7c4bb", "srvla": "#1f5fae"}


def stall_fig():
    names = ["π0.5\n동기", "OFT\n동기", "VLASH", "VLA-Cache\n(OFT)", "SR-VLA\n(π0.5)", "SR-VLA\n(OFT)"]
    stall = [128.11, 135.18, 60.25, 121.88, 47.95, 106.19]
    speed = [1.00, 1.52, 2.13, 1.68, 2.67, 1.93]
    colors = [COL["base"], COL["other"], COL["other"], COL["other"], COL["srvla"], COL["srvla"]]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    bars = ax.bar(names, stall, color=colors, width=0.62)
    for b, s, v in zip(bars, stall, speed):
        ax.text(b.get_x() + b.get_width() / 2, s + 2.5, f"{s:.1f} ms\n{v:.2f}×", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("질의당 정지 Stall (ms)", fontsize=9)
    ax.set_ylim(0, 165)
    ax.set_title("표 III: 질의당 정지와 속도비 (속도비 기준: π0.5 동기)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.text(0.01, 0.01, "출처: SR-VLA (arXiv:2611.22356) 표 III 수치. 평균만 있고 구간이 없습니다.", fontsize=6.5, color="#55544f")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = os.path.join(FIG, "own_stall.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def success_fig():
    with open(os.path.join(HERE, "recompute_results.json"), encoding="utf-8") as f:
        res = json.load(f)
    tests = res["suite_tests"]
    rows = []  # (라벨, 차이, 구간)
    for key, short in [("srvla_pi05_vs_pi05_sync", "π0.5"), ("srvla_oft_vs_oft_sync", "OFT")]:
        for s in ["Spatial", "Object", "Goal", "Long"]:
            x = tests[key][s]
            rows.append((f"{short} {s}", x["diff_pp"], x["ci95_pp"]))
    rows = rows[::-1]  # 위에서부터 읽히도록
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
    ys = list(range(len(rows)))
    for y, (_, d, (lo, hi)) in zip(ys, rows):
        ax.plot([lo, hi], [y, y], color="#55544f", lw=1.2)
        ax.plot([d], [y], "o", color=COL["srvla"], ms=4.5)
    ax.axvline(0, color="#999", lw=0.8, ls="--")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("SR-VLA 대 기준의 성공률 차이 (pp), 95% 구간 (시행 500 가정, 비쌍 근사)", fontsize=8)
    ax.set_title("스위트별 성공률 차이: 대부분 구간이 0을 포함", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.text(0.01, 0.01, "시행 수는 논문에 없어 500으로 가정했습니다. 원문 표 III 수치로 계산한 값입니다.", fontsize=6.5, color="#55544f")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = os.path.join(FIG, "own_success_ci.png")
    fig.savefig(out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    print("wrote", stall_fig())
    print("wrote", success_fig())
