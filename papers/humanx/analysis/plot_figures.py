"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_gsr_ablation.png : Table 1 GSR (기준선 3개와 XMimic 절제 5단계, 작업 3개)
own_real_ci.png      : Table 2 실제 로봇 성공률과 Wilson 95% 구간 (시도 10회·50회)
own_xgen_example.png : 공개 XGen 예제의 접촉 전 포물선 맞춤과 접촉 구간 물체-손목 거리
실행: python3 plot_figures.py   (recompute_results.json, xgen_example_check_results.json 필요; numpy, matplotlib)
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


def korean_font():
    for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
INK, ACC, GREY, SOFT = "#1b1b1a", "#1f5fae", "#9a9890", "#8fb0d9"

GSR = {  # Table 1 (v2) GSR %, 순서: SkillMimic, OmniRetarget, HDMI, Base, +IP, +IVT, +Data Aug, +Tea-Stu
    "Basketball": [0.0, 0.0, 2.4, 4.9, 13.5, 10.9, 60.6, 64.7],
    "Badminton": [30.9, 20.4, 25.3, 41.6, 60.5, 67.2, 84.4, 90.6],
    "Cargo": [0.0, 0.0, 1.8, 50.9, 94.5, 94.3, None, 96.3],
}
NAMES = ["SkillMimic", "OmniRetarget", "HDMI", "Base", "+IP", "+IVT", "+Data Aug", "+Tea-Stu"]


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def plot_gsr():
    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    w = 0.1
    cols = [GREY, GREY, GREY, "#c7d7ec", SOFT, "#5f8fc7", "#2f6db8", ACC]
    hatch = ["", "//", "..", "", "", "", "", ""]
    for j, name in enumerate(NAMES):
        vals = [GSR[t][j] for t in GSR]
        xs = np.arange(3) + (j - 3.5) * w
        for x, v in zip(xs, vals):
            if v is None:
                ax.text(x, 2, "n/a", ha="center", fontsize=6.5, color=GREY, rotation=90)
            else:
                ax.bar(x, v, width=w * 0.92, color=cols[j], hatch=hatch[j], edgecolor="white", label=name if x == xs[0] else None)
    ax.set_xticks(range(3))
    ax.set_xticklabels(list(GSR))
    ax.set_ylabel("GSR (%)")
    ax.set_ylim(0, 105)
    ax.legend(ncol=4, fontsize=7.5, frameon=False, loc="upper left")
    ax.set_title("Table 1의 GSR: 기준선 3개와 절제 5단계 (평가 횟수 n은 논문에 없음)", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_gsr_ablation.png", dpi=170)
    plt.close(fig)


def plot_real():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))
    t2 = {k: v for k, v in rec["V5_table2_wilson"].items() if v["in_v2"]}
    names = list(t2)
    order = sorted(names, key=lambda n: (("NEP" in n), -t2[n]["rate"]))
    fig, ax = plt.subplots(figsize=(9.0, 3.9))
    y = np.arange(len(order))[::-1]
    for yi, n in zip(y, order):
        v = t2[n]
        pct = 100 * v["rate"]
        lo, hi = wilson(v["k"], v["n"])
        c = ACC if "MoCap" in n else GREY
        ax.errorbar(pct, yi, xerr=[[max(0.0, pct - lo)], [max(0.0, hi - pct)]], fmt="o", color=c, ecolor=c, capsize=3, ms=6)
        ax.text(101.5, yi, f'{v["k"]}/{v["n"]}', va="center", fontsize=8.5)
    ax.axvline(80, color=INK, lw=0.8, ls="--")
    ax.text(80.6, -0.75, "80%", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=8.5)
    ax.set_xlim(30, 112)
    ax.set_xlabel("성공률 (%), 막대는 Wilson 95% 구간 (파랑: MoCap 50회, 회색: NEP 10회)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Table 2 실제 로봇: 9개 중 어느 구간도 하한이 80%를 넘지 않음", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_real_ci.png", dpi=170)
    plt.close(fig)


def plot_xgen():
    r = json.loads((ANA / "xgen_example_check_results.json").read_text(encoding="utf-8"))
    s = r["series"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    ax = axes[0]
    ax.plot(s["t_pre_s"], s["z_pre_m"], "o", ms=4, color=ACC, label="예제 데이터 z")
    ax.plot(s["t_pre_s"], s["z_fit_m"], "-", color=INK, lw=1, label="2차 곡선 맞춤")
    ax.set_xlabel("시간 (s), 접촉 시작 전")
    ax.set_ylabel("공 높이 z (m)")
    a = r["parabola"]["fit_accel_xyz"]
    ax.set_title(f"농구 예제, 접촉 전: 가속도 z = {a['z']:.4f} m/s², 잔차 0.0 mm", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    d = np.array(s["dist_m"])
    ax.plot(np.arange(len(d)), d, color=GREY)
    ax.axvspan(s["contact_start"], s["release"], color=SOFT, alpha=0.45)
    ax.set_xlabel("프레임 (음영: 접촉 구간)")
    ax.set_ylabel("물체-손목 중점 거리 (m)")
    ad = r["anchor_distance"]
    ax.set_title(f"상자 예제, 접촉 구간 거리 평균 {ad['mean_dist_m']*100:.1f} cm, 표준편차 {ad['std_dist_mm']:.1f} mm", fontsize=9.5)
    for a_ in axes:
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_xgen_example.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    plot_gsr()
    plot_real()
    plot_xgen()
    print("wrote", [p.name for p in FIG.glob("own_*.png")])
