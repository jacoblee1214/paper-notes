#!/usr/bin/env python3
"""그래프 세 장을 그린다 (report/figures/own_*.png). 실행: python3 plot_figures.py
recompute.py의 입력 표를 불러 쓰므로 먼저 recompute.py를 실행한다(같은 폴더)."""
import os, sys, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import importlib.util
spec = importlib.util.spec_from_file_location("rc", os.path.join(HERE, "recompute.py"))
rc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)

for cand in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
             "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
             "/System/Library/Fonts/AppleSDGothicNeo.ttc"]:
    if os.path.exists(cand):
        fm.fontManager.addfont(cand)
        plt.rcParams["font.family"] = fm.FontProperties(fname=cand).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False
OUT = os.path.join(HERE, "..", "report", "figures")
os.makedirs(OUT, exist_ok=True)
POL = ["ACT", "DP", "pi0.5", "Psi0", "Cosmos", "GR00T", "FastWAM"]
LAB = {"ACT": "ACT", "DP": "DP", "pi0.5": "π0.5", "Psi0": "Ψ0", "Cosmos": "Cosmos 3", "GR00T": "GR00T N1.7", "FastWAM": "FastWAM"}

# 1. 사건 깔때기 (Fig.5 값, 표준 모드)
fig, ax = plt.subplots(figsize=(8.4, 4.2))
names = ["후보 접촉", "올바른 첫 접촉", "올바른 도구 들기", "도구-대상 접촉", "성공"]
cols = ["#9aa5b1", "#6c8ebf", "#2f6db5", "#e08a2c", "#c0392b"]
w = 0.15
for i, n in enumerate(names):
    ax.bar([k + (i - 2) * w for k in range(len(POL))], [rc.FIG5[p]["S"][i] for p in POL], w, label=n, color=cols[i])
ax.set_xticks(range(len(POL))); ax.set_xticklabels([LAB[p] for p in POL], fontsize=9)
ax.set_ylabel("에피소드 비율 (%)"); ax.set_ylim(0, 105); ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.12), frameon=False)
ax.set_title("표준 모드의 사건 비율 (논문 Fig. 5 값, L1·L2 6칸 평균)", fontsize=10, pad=26)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "own_funnel.png"), dpi=170); plt.close(fig)

# 2. GR00T / FastWAM: L1, L2 시나리오별 S vs D, Wilson 95%
fig, axes = plt.subplots(1, 2, figsize=(9, 3.9), sharey=True)
for ax, pol in zip(axes, ["GR00T", "FastWAM"]):
    xs, labels = [], []
    j = 0
    for lv in (1, 2):
        for s in rc.SC:
            for m, c, off in (("S", "#2f6db5", -0.18), ("D", "#e08a2c", 0.18)):
                k = rc.cell(pol, m, lv, s); lo, hi = rc.wilson(k, 100)
                ax.errorbar(j + off, k, yerr=[[k - lo * 100], [hi * 100 - k]], fmt="o", color=c, ms=4, capsize=2, label=("표준(S)" if m == "S" else "디코이(D)") if j == 0 else None)
            labels.append(f"L{lv}\n{s[:4]}")
            j += 1
    ax.set_xticks(range(j)); ax.set_xticklabels(labels, fontsize=8); ax.set_title(LAB[pol], fontsize=10); ax.grid(axis="y", alpha=.3)
axes[0].set_ylabel("성공률 (%) 와 Wilson 95% 구간 (n=100)"); axes[0].legend(fontsize=8, frameon=False)
fig.suptitle("표 III에서 다시 그린 L1·L2 성공률: 같은 시나리오의 S와 D는 대부분 구간이 겹칩니다", fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "own_l1l2_ci.png"), dpi=170); plt.close(fig)

# 3. 실로봇 (표 IV) Wilson 95%
fig, ax = plt.subplots(figsize=(8.2, 3.8))
conds = ["Move-S", "Move-D", "Ret-S", "Ret-D"]
T4 = {"ACT": [1, 0, 2, 1], "GR00T": [6, 4, 4, 4], "FastWAM": [5, 4, 4, 5]}
cc = {"ACT": "#9aa5b1", "GR00T": "#2f6db5", "FastWAM": "#e08a2c"}
for pi, (p, v) in enumerate(T4.items()):
    for ci, k in enumerate(v):
        lo, hi = rc.wilson(k, 10)
        ax.errorbar(ci + (pi - 1) * 0.2, k * 10, yerr=[[k * 10 - lo * 100], [hi * 100 - k * 10]], fmt="o", color=cc[p], capsize=3, label=p if ci == 0 else None)
ax.set_xticks(range(4)); ax.set_xticklabels(conds); ax.set_ylabel("성공률 (%), 시행 10회, Wilson 95%"); ax.set_ylim(-2, 100); ax.grid(axis="y", alpha=.3)
ax.legend(frameon=False, fontsize=8)
ax.set_title("실로봇 L1 (표 IV 값). 구간이 넓어 GR00T와 FastWAM은 구별되지 않습니다", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "own_real_ci.png"), dpi=170); plt.close(fig)
print("ok")
