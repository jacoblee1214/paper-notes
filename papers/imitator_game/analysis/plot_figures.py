"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_real_levels.png : 실제 로봇 P+FT 성공률을 레벨별로, 모델마다 Wilson 95% 구간 (표 5, 레벨당 75회 가정)
own_sim_pourkettle.png : 시뮬 unseen P+FT 과제 x 레벨 성공률 (15개 변형, 3개 규모 평균) - PourKettle L2가 0
own_human_auto.png  : 자동 SR 대 사람 SR (변형 x 설정 60점, 프로젝트 페이지 공개 수치)와 상관
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; data/ 가 있으면 뒤의 둘도 그린다)
한글 폰트는 Linux Noto CJK, macOS Apple SD Gothic Neo 순서로 찾는다.
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def korean_font():
    for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
INK, ACC, GREY, RED = "#1b1b1a", "#1f5fae", "#9a9890", "#b3412c"
COL = {"pi0.5": "#7a6f5d", "XSkill": ACC, "ACT/DINOv2": "#3b8f6b", "DP/DINOv2": "#c28a2c"}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


T5 = {
    "pi0.5": [.44, .35, .37, .28],
    "XSkill": [.53, .57, .56, .29],
    "ACT/DINOv2": [.37, .37, .33, .33],
    "DP/DINOv2": [.33, .37, .29, .25],
}


def plot_real_levels():
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    levels = ["L0", "L1", "L2", "L3"]
    x = np.arange(4)
    for i, (m, v) in enumerate(T5.items()):
        ks = [round(p * 75) for p in v]
        lo = [wilson(k, 75)[0] for k in ks]
        hi = [wilson(k, 75)[1] for k in ks]
        xs = x + (i - 1.5) * 0.12
        ax.errorbar(xs, v, yerr=[np.array(v) - lo, np.array(hi) - v], fmt="o-", color=COL[m], lw=1.2, ms=4, capsize=2, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_ylabel("실제 로봇 P+FT 성공률 (사람 판정)")
    ax.set_ylim(0.1, 0.75)
    ax.set_title("표 5를 다시 그림: 레벨당 75회 가정의 Wilson 95% 구간", fontsize=10)
    ax.legend(fontsize=8, ncol=4, loc="upper center", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_real_levels.png", dpi=170)
    plt.close(fig)


def plot_pourkettle():
    p = DATA / "site_sim_detail.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    models = [m["key"] for m in d["models"]]
    tasks = [t["key"] for t in d["tasks"] if not t["seen"]]
    M = np.zeros((len(tasks), 4))
    for i, t in enumerate(tasks):
        for j, L in enumerate(d["levels"]):
            M[i, j] = np.mean([d["pft"][sc][t][L][k][0] for sc in ("15", "30", "45") for k in models])
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4)); ax.set_xticklabels(d["levels"])
    ax.set_yticks(range(len(tasks))); ax.set_yticklabels([t.replace("TwoRobot", "") for t in tasks], fontsize=8)
    for i in range(len(tasks)):
        for j in range(4):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if M[i, j] > 0.55 else INK)
    ax.set_title("시뮬 unseen 과제 P+FT 성공률 (15개 변형, 3개 규모 평균)", fontsize=9.5)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(FIG / "own_sim_pourkettle.png", dpi=170)
    plt.close(fig)


def plot_human_auto():
    p = DATA / "site_figdata.js"
    if not p.exists():
        return
    s = p.read_text()
    pts = re.findall(r'\{ model: "([^"]+)", fam: "([^"]+)", setting: "([^"]+)", x: ([\d.]+), y: ([\d.]+)', s)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    cols = {"Seen": ACC, "ZS": RED, "Scr": GREY, "P+FT": "#3b8f6b"}
    for st in ("Seen", "Scr", "P+FT", "ZS"):
        xs = [float(x) for m, f, s_, x, y in pts if s_ == st]
        ys = [float(y) for m, f, s_, x, y in pts if s_ == st]
        ax.scatter(xs, ys, s=22, color=cols[st], label=st, alpha=0.85)
    ax.plot([0, 1], [0, 1], "--", color=GREY, lw=0.8)
    xs = np.array([float(x) for *_, x, y in pts]); ys = np.array([float(y) for *_, x, y in pts])
    r = np.corrcoef(xs, ys)[0, 1]
    ax.set_xlabel("자동 SR (시뮬)"); ax.set_ylabel("사람 SR (Arena, 같은 롤아웃)")
    ax.set_title(f"변형 x 설정 60점, Pearson r = {r:.3f}", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_human_auto.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    plot_real_levels()
    plot_pourkettle()
    plot_human_auto()
    print("wrote figures in", FIG)
