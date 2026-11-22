"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_success_ci.png  : 표 1 성공률과 Wilson 95% 구간 (과제 4개 x 방법 3개, 방법당 20회)
own_stage_dpm.png   : 표 2 냉장고 단계별 성공 횟수와 그림 5 분당 시연 수(값은 논문, 주기는 60/DPM 계산)
own_toy_feedback.png: 토이 T1, 되돌림 이득 사슬(팔로워 토크 대 손에 전달되는 토크)과 어깨 야코비안 사상의 증폭 분포
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 먼저 recompute.py와 toy_retarget_torque.py를 돌릴 것)
한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서로 찾는다.
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
INK, ACC, GREY, ORG = "#1b1b1a", "#1f5fae", "#9a9890", "#c4691f"


def plot_success():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))
    tasks = ["박스 밀기", "박스 뒤집기", "GPU 집기", "냉장고 음료"]
    methods = [("UMI", GREY), ("No-torque", "#8fb0d9"), ("UME", ACC)]
    fig, ax = plt.subplots(figsize=(9.4, 4.0))
    w = 0.26
    x = np.arange(4)
    for mi, (m, col) in enumerate(methods):
        ci = rec["V2_ci"][m]
        pct = np.array([c["p"] * 100 for c in ci])
        lo = np.array([c["wilson"][0] * 100 for c in ci])
        hi = np.array([c["wilson"][1] * 100 for c in ci])
        pos = x + (mi - 1) * w
        ax.bar(pos, pct, width=w * 0.92, color=col, label=m)
        ax.errorbar(pos, pct, yerr=[pct - lo, hi - pct], fmt="none", ecolor=INK, capsize=2.5, lw=0.9)
        for xp, c in zip(pos, ci):
            ax.text(xp, 2, f"{c['k']}/20", ha="center", va="bottom", fontsize=7, color="white" if c["p"] > 0.15 else INK,
                    rotation=90)
    for i, name in enumerate(["box_pushing", "box_flipping", "gpu_picking", "fridge_drink"]):
        p = rec["V3_fisher"][f"UME_vs_No-torque:{name}"]["p"]
        txt = f"p={p:.2g}" if p < 0.99 else "p=1"
        ax.text(i, 106, "UME 대 No-torque\n" + txt, ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylim(0, 122)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("성공률 (%) · 방법당 20회 · Wilson 95%")
    ax.legend(loc="upper center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.12))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_success_ci.png", dpi=170)
    plt.close(fig)


def plot_stage_dpm():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    stages = ["방향 전환", "냉장고 열기", "음료 꺼내기", "냉장고 닫기", "운반", "내려놓기"]
    for m, col, mk in [("UME", ACC, "o"), ("No-torque", "#5f86b8", "s"), ("UMI", GREY, "^")]:
        ax.plot(range(6), rec["V6_stage_counts"][m], marker=mk, color=col, label=m, lw=1.6, ms=5)
    ax.set_xticks(range(6))
    ax.set_xticklabels(stages, fontsize=7.5, rotation=20, ha="right")
    ax.set_ylim(-1, 21)
    ax.set_ylabel("단계까지 성공한 횟수 (20회 중)")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("냉장고 음료 꺼내기 단계별 (표 2)", fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    dpm = [6, 20, 28]
    cyc = rec["V7_dpm_cycle_seconds"]
    labels = ["UME w/o\ntorque", "UME", "사람"]
    ax.bar(range(3), dpm, color=[GREY, ACC, "#6e9d6b"], width=0.6)
    for i, (v, k) in enumerate(zip(dpm, ["UME_wo_torque", "UME", "Human"])):
        ax.text(i, v + 0.6, f"{v} DPM\n주기 {cyc[k]:.1f} s", ha="center", fontsize=8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 36)
    ax.set_ylabel("분당 시연 수 (그림 5, 사용자 3명 평균)")
    ax.set_title("박스 뒤집기 수집 처리량: 리셋 포함 주기 60/DPM", fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_stage_dpm.png", dpi=170)
    plt.close(fig)


def plot_toy():
    t = json.loads((ANA / "toy_retarget_torque_results.json").read_text(encoding="utf-8"))
    a = t["A_feedback_chain"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    ax = axes[0]
    xs = np.array([r["tau_follower"] for r in a["shoulder"]])
    ax.plot(xs, xs, color=GREY, ls="--", lw=1, label="투명한 전달(1:1)")
    ax.plot(xs, [r["felt"] for r in a["shoulder"]], marker="o", color=ACC, ms=4, label="어깨·팔꿈치 (kp 240, 클립 4 Nm)")
    xw = np.array([r["tau_follower"] for r in a["wrist"]])
    ax.plot(xw, [r["felt"] for r in a["wrist"]], marker="s", color=ORG, ms=4, label="손목·그리퍼 (kp 25, 클립 1 Nm)")
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 8)
    ax.set_xlabel("팔로워 관절 접촉 토크 (Nm)")
    ax.set_ylabel("조작자 손에 전달되는 토크 (Nm)")
    ax.set_title("되돌림 이득 사슬 (정적 가정, 코드 상수)", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    b1, b2 = t["B_map_box1.0"], t["B_map_box1.5"]
    for b, col, lab in [(b1, ACC, "외골격 표본 ±1.0 rad"), (b2, ORG, "±1.5 rad")]:
        pn = b["per_neutral"]
        med = [p["amp_median"] for p in pn]
        p99 = [p["amp_p99"] for p in pn]
        ax.scatter(med, p99, color=col, s=24, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("증폭 ‖M‖₂ 중앙값 (중립 자세별)")
    ax.set_ylabel("증폭 99분위수 (로그)")
    ax.set_title("어깨 사상 M = J_exo^T J_robot^-T: 중립 자세 12개", fontsize=9.5)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_feedback.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    plot_success()
    plot_stage_dpm()
    plot_toy()
    print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
