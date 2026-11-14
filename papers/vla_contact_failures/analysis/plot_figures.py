"""우리가 그린 그래프 네 장을 report/figures/own_*.png로 만든다(논문 그림이 아니다. 공개본에도 넣을 수 있다).

입력: analysis/table_check_results.json, analysis/toy_schedule_results.json, analysis/toy_power_results.json
실행: python3 analysis/plot_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def korean_font():
    cands = ["/System/Library/Fonts/AppleSDGothicNeo.ttc",
             "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
             "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"]
    for c in cands:
        if Path(c).exists():
            font_manager.fontManager.addfont(c)
            return font_manager.FontProperties(fname=c).get_name()
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
C_BLUE, C_ORANGE, C_GREY, C_GREEN, C_RED = "#1f5fae", "#d9822b", "#7a7a75", "#3b8a4f", "#b23b3b"


def save(fig, name):
    fig.savefig(FIG / name, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / name)


def fig_schedule():
    x = np.linspace(0.001, 0.999, 800)
    beta_pdf = stats.beta.pdf(x, 1.5, 1.0)
    logit = np.log(x / (1 - x))
    ln_pdf = stats.norm.pdf(logit + 1.5) / (x * (1 - x))  # 논문 식 (2)에서 s=1
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(x, beta_pdf, color=C_GREY, lw=2, label="Beta(1.5, 1) (기존)")
    ax.plot(x, ln_pdf, color=C_BLUE, lw=2, label="Logit-Normal, m=1.5, s=1 (제안)")
    ax.axvspan(0, 0.2, color=C_ORANGE, alpha=0.15)
    ax.text(0.235, 3.45, "τ<0.2 (음영)\nBeta 9%, LN 55%", ha="left", fontsize=8.5, color="#7a4a10")
    ax.set_xlabel("노이즈 수준 τ (1 = 순수 노이즈)")
    ax.set_ylabel("확률 밀도")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 3.8)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_title("샘플링 분포 (논문 식과 Beta(1.5, 1)로 계산)", fontsize=10)
    save(fig, "own_schedule.png")


def fig_success(t):
    tasks = ["plug", "usb", "button", "board", "key"]
    labels = ["플러그", "USB", "버튼", "지우개", "열쇠"]
    series = [("pi05", "π0.5", C_GREY), ("pi05_LN", "π0.5+LN", C_ORANGE),
              ("FACT", "FACT", C_BLUE), ("ForceVLA", "ForceVLA", C_GREEN), ("TA-VLA", "TA-VLA", C_RED)]
    rows = {(r["method"], r["task"]): r for r in t["main_tests"]}
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    width = 0.16
    xs = np.arange(len(tasks))
    base_vals = t["main_counts"]["pi05"]
    for k, (key, name, col) in enumerate(series):
        vals = t["main_counts"][key]
        lo, hi = [], []
        for j, task in enumerate(tasks):
            from math import sqrt
            n = 40
            p = vals[j] / n
            z = 1.959964
            d = 1 + z * z / n
            c = (p + z * z / (2 * n)) / d
            h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
            lo.append((p - (c - h)) * 100)
            hi.append(((c + h) - p) * 100)
        vals_pct = [v / 40 * 100 for v in vals]
        ax.bar(xs + (k - 2) * width, vals_pct, width, color=col, label=name,
               yerr=[lo, hi], capsize=2, error_kw={"lw": 0.7, "ecolor": "#333"})
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("성공률 (%)")
    ax.set_ylim(0, 115)
    ax.legend(ncol=5, frameon=False, fontsize=7.8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.set_title("과제별 성공률과 Wilson 95% 구간 (과제당 40회, 논문 표 1 값)", fontsize=9.5)
    save(fig, "own_success_ci.png")


def fig_ablation(t):
    rows = t["ablation_tests_not_in_paper"]
    names = []
    vals = []
    p_raw = []
    p_h = []
    for r in rows:
        names.append(f'{r["ablation"]} / {r["task"]}')
        vals.append(r["delta_pp"])
        p_raw.append(r["p_FACT_greater"])
        p_h.append(r["p_holm_9"])
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    y = np.arange(len(names))[::-1]
    colors = [C_BLUE if ph <= 0.05 else (C_ORANGE if pr <= 0.05 else C_GREY) for ph, pr in zip(p_h, p_raw)]
    ax.barh(y, vals, color=colors, height=0.6)
    for yi, v, pr, ph in zip(y, vals, p_raw, p_h):
        fp = lambda q: "<.001" if q < 0.001 else f"{q:.3f}"
        ax.text(v + (1.2 if v >= 0 else -1.2), yi, f"p={fp(pr)}, Holm={fp(ph)}", va="center",
                ha="left" if v >= 0 else "right", fontsize=7)
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7.8)
    ax.set_xlabel("FACT 대비 성공률 차이 (pp, 음수 = 제거 시 하락)")
    ax.set_xlim(-95, 30)
    ax.set_title("구성요소 제거 (논문에 검정 없음, 재계산). 파랑: Holm 보정 후 p<.05", fontsize=9)
    save(fig, "own_ablation.png")


def fig_toy(p):
    labels = ["귀무, 비교 1개\n(p≤.05)", "귀무, 비교 20개\n(보정 없음, 하나 이상)",
              "귀무, 비교 20개\n(Bonferroni .05/20)", "참 차이 20pp\n검정력(40회)"]
    vals = [p["type1_single_comparison"], p["fwer_any_of_20_uncorrected"],
            p["fwer_any_of_20_holm_approx_bonferroni"], p["power_true_diff_20pp"]]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    cols = [C_GREY, C_RED, C_GREEN, C_BLUE]
    ax.bar(range(4), vals, color=cols, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=8.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=7.6)
    ax.set_ylim(0, 0.62)
    ax.set_ylabel("비율 (2000회 반복)")
    ax.set_title(f"토이: 40회 두 팔 Fisher 검정. 유의한 결과의 평균 추정 차이 "
                 f"{p['mean_observed_diff_given_significant_pp']} pp (참 20 pp)", fontsize=8.8)
    save(fig, "own_toy_power.png")


def main():
    t = json.loads((ROOT / "analysis" / "table_check_results.json").read_text())
    p = json.loads((ROOT / "analysis" / "toy_power_results.json").read_text())
    fig_schedule()
    fig_success(t)
    fig_ablation(t)
    fig_toy(p)


if __name__ == "__main__":
    main()
