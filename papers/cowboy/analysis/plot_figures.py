"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_fig5_scenes.png : 그림 5의 장면별 성공률 24개와 평균 (본문 91%, 표 3 91.1%와 비교), 75회로 불가능한 값 표시
own_consistency.png : (왼) 표 3 실로봇 열의 성공 + 충돌 합, (오) 표 2 교사 대 학생 성공률 차이와 95% 구간(범주당 2000 환경 가정)
own_toy_gate.png    : 토이 T1, 목표 3 m 한 경우의 팔 신장비와 조작성 시계열, 그리고 30 목표 평균 조작성
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
먼저 recompute.py 와 toy_gate_reaching.py 를 돌려 결과 json 을 만들어 둔다.
"""
import json
from math import sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

import toy_gate_reaching as toy  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ANA = ROOT / "analysis"
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def korean_font():
    for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
INK, ACC, GREY, RED = "#1b1b1a", "#1f5fae", "#9a9890", "#c0392b"


def plot_fig5():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))["V3_fig5"]
    vals = rec["values"]
    bad = {i for i, _ in rec["infeasible_at_75"]}
    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    x = np.arange(1, 25)
    colors = [RED if i in bad else ACC for i in x]
    ax.bar(x, vals, color=colors, width=0.7)
    ax.axhline(rec["mean_of_scenes"], color=INK, lw=1.2, ls="-", label=f"장면 평균 {rec['mean_of_scenes']:.1f}%")
    ax.axhline(91.1, color=GREY, lw=1.2, ls="--", label="표 3 'Ours' 91.1%")
    p = rec["mean_of_scenes"] / 100
    h = 1.96 * sqrt(p * (1 - p) / 75) * 100
    ax.axhspan(rec["mean_of_scenes"] - h, rec["mean_of_scenes"] + h, color=GREY, alpha=0.15, label="75회 이항 95% 변동(장면 하나)")
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.4, str(v), ha="center", fontsize=7, color=RED if xi in bad else INK)
    ax.set_ylim(70, 102)
    ax.set_xticks(x)
    ax.set_xlabel("장면 번호 (그림 5, 왼쪽 위부터 행 순서)")
    ax.set_ylabel("장면별 성공률 (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8, framealpha=0.9)
    ax.set_title("그림 5 장면별 성공률(붉은색 = 장면당 75회로는 나올 수 없는 값)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_fig5_scenes.png", dpi=170)
    plt.close(fig)


def plot_consistency():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))
    v5 = rec["V5_success_plus_collision"]
    v6 = rec["V6_teacher_vs_student"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    ax = axes[0]
    names = v5["real_names"]
    succ = [46.7, 71.1, 0.0, 91.1]
    coll = [8.2, 31.7, 11.3, 3.9]
    x = np.arange(4)
    ax.bar(x, succ, color=ACC, label="성공 (충돌 없이 10 cm)")
    ax.bar(x, coll, bottom=succ, color=GREY, label="충돌 환경")
    ax.axhline(100, color=RED, lw=1.2)
    for xi, s, c in zip(x, succ, coll):
        ax.text(xi, s + c + 1.5, f"{s + c:.1f}", ha="center", fontsize=8, color=RED if s + c > 100 else INK)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
    ax.set_ylim(0, 118)
    ax.set_ylabel("%")
    ax.set_title("표 3 실로봇: 성공 + 충돌 (100 초과 불가)", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="upper left")
    ax = axes[1]
    rows = v6["rows"]
    cats = [r["cat"] for r in rows]
    gap = [r["gap"] for r in rows]
    n = v6["assumed_envs_per_category"]
    err = []
    for r in rows:
        p1, p2 = r["teacher"] / 100, r["student"] / 100
        se = sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n) * 100
        err.append(1.96 * se)
    ax.bar(np.arange(4), gap, color=ACC, yerr=err, capsize=3, ecolor=INK)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(cats, fontsize=8)
    ax.set_ylabel("교사 - 학생 성공률 (%p)")
    ax.set_title(f"표 2 교사 대 학생 (범주당 {n} 환경 가정, 95% 구간)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "own_consistency.png", dpi=170)
    plt.close(fig)


def plot_toy():
    res = json.loads((ANA / "toy_gate_reaching_results.json").read_text(encoding="utf-8"))
    target = np.array([3.0, 0.0])
    q0 = np.array([0.0, 0.0, 0.6, 1.4])
    runs = {
        "가중 IK (기저 가중 3)": toy.run_ik(q0, target, 3.0),
        "게이트 어트랙터": toy.run_fabric(q0, target),
    }
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4))
    t = np.arange(toy.STEPS + 1) * toy.DT
    cols = {"가중 IK (기저 가중 3)": GREY, "게이트 어트랙터": ACC}
    for name, traj in runs.items():
        traj = np.array(traj)
        ee = np.array([toy.fk(q) for q in traj])
        ext = np.linalg.norm(ee - traj[:, :2], axis=1) / toy.REACH
        axes[0].plot(t, ext, color=cols[name], label=name)
        axes[1].plot(t, traj[:, 0], color=cols[name], label=name)
    axes[0].set_xlim(0, 8)
    axes[1].set_xlim(0, 8)
    axes[0].set_ylabel("팔 신장비 |ee - 기저| / 0.8 m")
    axes[0].set_xlabel("시간 (s)")
    axes[0].legend(fontsize=7.5)
    axes[0].set_title("목표 3 m: 팔 신장", fontsize=9.5)
    axes[1].set_ylabel("기저 x (m)")
    axes[1].set_xlabel("시간 (s)")
    axes[1].set_title("기저 전진", fontsize=9.5)
    names = ["weighted_IK_wb1", "weighted_IK_wb3", "weighted_IK_wb10", "gated_fabric"]
    labels = ["IK w=1", "IK w=3", "IK w=10", "게이트"]
    means = [res[n]["manip_mean"]["mean"] for n in names]
    axes[2].bar(np.arange(4), means, color=[GREY, GREY, GREY, ACC])
    axes[2].set_xticks(np.arange(4))
    axes[2].set_xticklabels(labels, fontsize=8)
    axes[2].set_ylabel("팔 조작성 평균 (최대 0.16)")
    axes[2].set_title("30개 목표 평균 조작성", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_gate.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    plot_fig5()
    plot_consistency()
    plot_toy()
    print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
