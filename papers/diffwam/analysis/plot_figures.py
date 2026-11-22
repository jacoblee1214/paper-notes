"""DiffWAM 분석의 그래프 세 장을 만든다 (우리가 그린 그래프, 논문 그림 아님).

  report/figures/own_runs.png        : 표 10 독립 학습 3회 값과 본문 보고 값(최댓값) 대비
  report/figures/own_real_ci.png     : 실로봇 성공률과 Wilson 95% 구간 (과제당 20회)
  report/figures/own_task_diff.png   : 표 12, 과제별 DiffWAM-L 대 FastWAM 클린 차이

입력: analysis/recompute_results.json과 paper/paper.txt (표 12). 실행: python3 analysis/plot_figures.py
한글 폰트: Noto Sans CJK KR → Apple SD Gothic Neo 순서로 찾는다.
"""
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
RES_PATH = ROOT / "analysis" / "recompute_results.json"

for cand in ["Noto Sans CJK KR", "Apple SD Gothic Neo", "AppleGothic"]:
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False
INK, ACC, GREY, ORANGE = "#1b1b1a", "#1f5fae", "#8a8880", "#c0602a"


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def own_runs():
    runs = {"LIBERO (4000회)": ([98.60, 98.80, 98.60], 98.80, 98.67),
            "RoboTwin full 평균 (Clean·Random)": ([92.49, 92.65, 92.43], 92.65, 92.52),
            "RoboTwin clean-only (Clean·Random 평균)": ([37.33, 37.40, 37.47], 37.47, 37.40)}
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.1))
    for ax, (name, (vals, headline, mean)) in zip(axes, runs.items()):
        for i, v in enumerate(vals):
            ax.scatter([i], [v], s=46, color=ACC, zorder=3)
        ax.axhline(headline, color=ORANGE, lw=1.2, ls="--", label=f"본문 보고 {headline}")
        ax.axhline(mean, color=GREY, lw=1.2, ls=":", label=f"3회 평균 {mean}")
        ax.set_xticks([0, 1, 2], ["1", "2", "3"])
        ax.set_xlabel("학습 실행 번호")
        ax.set_title(name, fontsize=9.5, color=INK)
        lo, hi = min(vals), max(vals)
        pad = max(0.15, (hi - lo) * 0.8)
        ax.set_ylim(lo - pad * 1.5, hi + pad * 1.5)
        ax.grid(axis="y", color="#e5e3dc", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7, frameon=False, loc="lower center")
    axes[0].set_ylabel("성공률 (%)")
    fig.suptitle("표 10: 독립 학습 3회 값 (본문 보고 값은 세 회 중 최댓값)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "own_runs.png", dpi=200)
    plt.close(fig)


def own_real_ci():
    tasks = ["Place Pens", "Store Block", "Stack Cups", "전체 (3과제)"]
    dw = [(18, 20), (17, 20), (18, 20), (53, 60)]
    fw = [(16, 20), (14, 20), (14, 20), (44, 60)]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = list(range(len(tasks)))
    for off, data, col, lab in [(-0.16, dw, ACC, "DiffWAM-S"), (0.16, fw, GREY, "FastWAM")]:
        pts, lo_err, hi_err = [], [], []
        for k, n in data:
            p = 100 * k / n
            lo, hi = wilson(k, n)
            pts.append(p); lo_err.append(p - lo); hi_err.append(hi - p)
        ax.errorbar([i + off for i in x], pts, yerr=[lo_err, hi_err], fmt="o", color=col, capsize=4, label=lab)
    ax.set_xticks(x, tasks, fontsize=8.5)
    ax.set_ylabel("성공률 (%)")
    ax.set_ylim(40, 105)
    ax.grid(axis="y", color="#e5e3dc", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_title("실로봇 성공률과 Wilson 95% 구간 (과제당 20회, 본문 Fig. 5 수치)", fontsize=9.5, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "own_real_ci.png", dpi=200)
    plt.close(fig)


def own_task_diff():
    lines = (ROOT / "paper" / "paper.txt").read_text(encoding="utf-8", errors="ignore").splitlines()
    row_re = re.compile(r"^\s*(.+?)\s+((?:\d+\s+){13}\d+)\s*$")
    start = next(i for i, l in enumerate(lines) if l.strip().startswith("Adjust Bottle"))
    rows = []
    for l in lines[start:start + 80]:
        m = row_re.match(l)
        if m and len(m.group(2).split()) == 14:
            rows.append([int(v) for v in m.group(2).split()])
        if len(rows) == 50:
            break
    # 열 순서: pi0.5, Motus, LingBot-VA, GigaWorld-Policy, X-WAM, FastWAM, DiffWAM-L (clean, random 쌍)
    d = [r[12] - r[10] for r in rows]
    order = sorted(range(len(d)), key=lambda i: d[i])
    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    ax.bar(range(len(order)), [d[i] for i in order],
           color=[ACC if d[i] >= 0 else ORANGE for i in order], width=0.75)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks([])
    ax.set_ylabel("클린 차이 (%p)")
    ax.set_xlabel("과제 50개 (차이 크기 순, 과제 이름 생략)")
    ax.grid(axis="y", color="#e5e3dc", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    w = sum(1 for v in d if v > 0); l = sum(1 for v in d if v < 0); t = len(d) - w - l
    ax.set_title(f"표 12 과제별 클린 차이 (DiffWAM-L - FastWAM): 이김 {w}, 짐 {l}, 같음 {t}, 평균 {sum(d)/len(d):+.2f}%p",
                 fontsize=9.5, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "own_task_diff.png", dpi=200)
    plt.close(fig)
    return {"wins": w, "losses": l, "ties": t, "mean": round(sum(d) / len(d), 2)}


if __name__ == "__main__":
    own_runs()
    own_real_ci()
    info = own_task_diff()
    res = json.loads(RES_PATH.read_text(encoding="utf-8"))
    res["task_diff_clean_plot"] = info
    RES_PATH.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", sorted(p.name for p in FIG.glob("own_*.png")), info)
