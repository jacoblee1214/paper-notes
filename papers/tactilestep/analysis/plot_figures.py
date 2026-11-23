"""우리가 만든 그래프 세 장을 report/figures/ 에 저장한다. 논문 그림 조각은 여기서 만들지 않는다.

  own_hw_ci.png        표 3 실로봇 결과(충격력, 소음, 접촉 면적비)와 95% 구간 (시행 20회, ±는 SD로 가정)
  own_gain_split.png   기준선 대비 개선 중 촉각 관측 없이(w/o tac. obs.) 얻은 몫, 그리고 충격력과 소음의 관계
  own_tactile_toy.png  토이: 식 7 접촉 면적비의 임계값 의존, 힘 의존, 25 Hz 표본화 최댓값

입력: analysis/recompute_results.json, analysis/toy_tactile_sim_results.json
실행: python3 analysis/plot_figures.py   (matplotlib, CPU, 수 초)
"""
import json
import math
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
RC = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))
TOY = json.loads((ROOT / "analysis" / "toy_tactile_sim_results.json").read_text(encoding="utf-8"))

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
INK, MUTED, ACC, ACC2, WARN = "#1b1b1a", "#6b6a64", "#1f5fae", "#d9822b", "#a33a3a"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})

# 표 3 (논문 값). 재계산 스크립트와 같은 값이다.
T3 = {
    "계단 오름": {"F": ((499.9, 40.7), (375.2, 47.3), (349.8, 32.5)), "L": ((101.2, 3.7), (85.1, 3.4), (72.2, 2.8)),
              "A": ((0.470, 0.013), (0.421, 0.028), (0.482, 0.016))},
    "계단 내림": {"F": ((359.3, 23.2), (353.3, 38.6), (344.1, 20.1)), "L": ((97.2, 1.2), (84.9, 3.9), (67.1, 1.4)),
              "A": ((0.483, 0.020), (0.417, 0.044), (0.598, 0.027))},
    "단 오름": {"F": ((695.0, 49.4), (371.5, 50.5), (355.7, 39.8)), "L": ((110.1, 6.0), (98.2, 5.3), (83.0, 4.1))},
    "단 내림": {"F": ((608.9, 38.6), (445.7, 54.1), (404.6, 43.2)), "L": ((90.8, 5.2), (87.5, 4.4), (75.9, 3.6))},
    "평지": {"F": ((201.3, 23.3), (202.1, 32.4), (191.3, 17.7)), "L": ((90.6, 2.5), (85.9, 2.9), (66.5, 1.6)),
           "A": ((0.453, 0.011), (0.362, 0.035), (0.510, 0.014))},
    "경사": {"F": ((255.0, 34.9), (204.7, 42.1), (202.0, 31.0)), "L": ((83.4, 1.0), (77.1, 2.0), (69.9, 1.1)),
           "A": ((0.486, 0.026), (0.490, 0.033), (0.499, 0.030))},
}
POLS = ["Hiking 기준선", "촉각 관측 없음", "TactileStep"]
COLS = ["#b9b7ae", "#8fa9c9", ACC]
N = 20
tcrit = stats.t.ppf(0.975, N - 1)


def ci(sd):
    return tcrit * sd / math.sqrt(N)


# ---------------------------------------------------------------- 그래프 1
fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.5), dpi=200)
for ax, (met, title, unit) in zip(axes, [("F", "충격력 F_impact", "N"), ("L", "최대 A-가중 소음", "dB"),
                                          ("A", "접촉 면적비 A_c (일부 지형)", "")]):
    terr = [t for t in T3 if met in T3[t]]
    x = np.arange(len(terr))
    for k in range(3):
        m = [T3[t][met][k][0] for t in terr]
        e = [ci(T3[t][met][k][1]) for t in terr]
        ax.errorbar(x + (k - 1) * 0.22, m, yerr=e, fmt="o", ms=4, color=COLS[k], ecolor=COLS[k], capsize=2,
                    label=POLS[k])
    ax.set_xticks(x)
    ax.set_xticklabels(terr, rotation=25, ha="right")
    ax.set_title(f"{title} {('[' + unit + ']') if unit else ''}", fontsize=9.5, color=INK)
    ax.grid(axis="y", color="#e5e3dc", lw=0.6)
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle("표 3 실로봇 결과 (평균과 95% 구간, 시행 20회와 ±=SD 가정)", fontsize=10, color=INK, y=1.02)
fig.tight_layout()
fig.savefig(FIG / "own_hw_ci.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 그래프 2
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.0, 3.7), dpi=200, gridspec_kw={"width_ratios": [1.15, 1]})
sp = RC["table3_gain_split"]
terr = ["Stair up", "Stair down", "Platform up", "Platform down", "Flat", "Slope"]
kor = ["계단 오름", "계단 내림", "단 오름", "단 내림", "평지", "경사"]
x = np.arange(6)
sf = [sp[t]["F"]["share_without_obs_pct"] for t in terr]
sl = [sp[t]["L"]["share_without_obs_pct"] for t in terr]
a1.bar(x - 0.2, sf, 0.4, color=ACC2, label="충격력")
a1.bar(x + 0.2, sl, 0.4, color=ACC, label="소음")
a1.axhline(100, color=MUTED, lw=0.8, ls=":")
a1.axhline(0, color=INK, lw=0.6)
a1.set_xticks(x)
a1.set_xticklabels(kor, rotation=20, ha="right")
a1.set_ylabel("기준선 대비 개선 중 촉각 관측 없이\n얻은 몫 (%)")
a1.set_title("개선의 얼마가 촉각 관측 없이도 생기나", fontsize=9.5)
a1.legend(frameon=False, fontsize=8)
Fs, Ls, lab = [], [], []
for t, k in zip(terr, kor):
    for j, pol in enumerate(("base", "wo", "ts")):
        Fs.append(T3[k]["F"][j][0])
        Ls.append(T3[k]["L"][j][0])
        lab.append(j)
Fs, Ls, lab = np.array(Fs), np.array(Ls), np.array(lab)
for j in range(3):
    a2.scatter(Fs[lab == j], Ls[lab == j], s=26, color=COLS[j], label=POLS[j], edgecolor="white", lw=0.5)
r, p = stats.pearsonr(Fs, Ls)
a2.set_xlabel("충격력 [N]")
a2.set_ylabel("최대 소음 [dB]")
a2.set_title(f"18개 조건의 힘과 소음 (Pearson r = {r:.2f})", fontsize=9.5)
a2.legend(frameon=False, fontsize=8)
a2.grid(color="#e5e3dc", lw=0.6)
fig.tight_layout()
fig.savefig(FIG / "own_gain_split.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 그래프 3 (토이)
fig, (b1, b2, b3) = plt.subplots(1, 3, figsize=(10.6, 3.4), dpi=200)
e1 = TOY["M1_edge_overhang"]
true = [r["true_contact_frac"] for r in e1]
b1.plot(true, true, color=MUTED, ls=":", label="참 접촉 비율")
for key, c, lab_ in [("A_tac_eps0.2", WARN, "임계 0.2 N"), ("A_tac_eps1.0", ACC, "임계 1 N"), ("A_tac_eps3.0", ACC2, "임계 3 N")]:
    b1.plot(true, [r[key] for r in e1], "o-", ms=3.5, color=c, label=lab_)
b1.set_xlabel("참 접촉 비율 (발이 모서리에 걸린 정도)")
b1.set_ylabel("식 7 접촉 면적비 A_tac")
b1.set_title("임계 eps_f 에 따른 면적비", fontsize=9.5)
b1.legend(frameon=False, fontsize=8)
e3 = TOY["M3_force_dependence"]
b2.plot([r["F_N"] for r in e3], [r["A_tac"] for r in e3], "o-", color=ACC)
b2.axhline(e3[0]["true_contact_frac"], color=MUTED, ls=":")
b2.set_xlabel("총 수직력 F [N]")
b2.set_ylabel("A_tac (참 비율 0.7)")
b2.set_title("같은 접촉, 다른 힘", fontsize=9.5)
e5 = TOY["M5_sampling"]
w = [r["pulse_width_ms"] for r in e5]
b3.plot(w, [r["mean_peak_ratio_25Hz"] for r in e5], "o-", color=WARN, label="25 Hz")
b3.plot(w, [r["mean_peak_ratio_100Hz"] for r in e5], "o-", color=ACC, label="100 Hz")
b3.set_xlabel("충격 펄스 폭 [ms] (반사인 가정)")
b3.set_ylabel("표본 최댓값 / 참 최댓값")
b3.set_title("표본화가 최댓값을 낮추는 정도", fontsize=9.5)
b3.legend(frameon=False, fontsize=8)
for ax in (b1, b2, b3):
    ax.grid(color="#e5e3dc", lw=0.6)
fig.tight_layout()
fig.savefig(FIG / "own_tactile_toy.png", bbox_inches="tight")
plt.close(fig)
print("wrote 3 figures to", FIG)
