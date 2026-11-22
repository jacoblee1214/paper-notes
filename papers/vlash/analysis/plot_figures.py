"""우리가 만든 그래프 네 장 (report/figures/own_*.png). matplotlib. recompute와 토이 결과 json을 읽는다.
실행: python3 analysis/plot_figures.py
"""
import json
import platform
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
for cand in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/System/Library/Fonts/AppleSDGothicNeo.ttc",
             "/System/Library/Fonts/Supplemental/AppleGothic.ttf"):
    if Path(cand).exists():
        font_manager.fontManager.addfont(cand)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False
C = {"Sync": "#c0504d", "Naive Async": "#e69f00", "RTC": "#7fb069", "VLASH": "#1f5fae"}

R = json.load(open(HERE / "recompute_results.json"))
dig = json.load(open(HERE / "fig_digitized.json"))

# 1. 시도 수 역산
fig, ax = plt.subplots(figsize=(7.2, 4.6))
x = np.linspace(0.05, 0.95, 100)
ax.fill_between(x * 100, 60 / (1 + 2 * x), 60 / (1 + x), color="#dddddd", label="가능한 범위 (20 롤아웃, 최대 3회)")
for m, c in C.items():
    pts = [(r["errmid"], r["errmid"] / 100 * (1 - r["errmid"] / 100) / (r["hw"] / 100) ** 2) for r in dig["fig7_pi05"] if r["method"] == m]
    ax.scatter(*zip(*pts), color=c, s=28, label=f"{m}")
ax.set_xlabel("보고된 성공률 (%) — 그림 7에서 읽음")
ax.set_ylabel("에러바에서 역산한 시도 수 N")
ax.set_title("그림 7(π0.5): 막대마다 분모가 22–56으로 다르다")
ax.legend(fontsize=8)
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(OUT / "own_fig7_denominators.png", dpi=170)
plt.close(fig)

# 2. 시도 단위 지표 토이
T = json.load(open(HERE / "toy_attempts_results.json"))
fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
labs = {1.0: "이질적 (농도 1)", 4.0: "중간 (농도 4)", 1000.0: "균일"}
for conc, mk in ((1.0, "o"), (4.0, "s"), (1000.0, "^")):
    rows = [t for t in T if t["conc"] == conc]
    axs[0].plot([t["mean_q"] for t in rows], [t["attempt_level_true"] for t in rows], mk + "-", label=labs[conc])
    axs[1].plot([t["mean_q"] for t in rows], [t["coverage_1.96se"] for t in rows], mk + "-", label=labs[conc])
axs[0].plot([0.2, 0.9], [0.2, 0.9], "k:", lw=1)
axs[0].set_xlabel("시도당 평균 성공 확률")
axs[0].set_ylabel("보고 SR(성공/시도)의 기대값")
axs[0].set_title("난이도가 이질적이면 SR이 낮아짐")
axs[1].axhline(0.95, color="k", ls=":", lw=1)
axs[1].set_ylim(0.75, 1.0)
axs[1].set_xlabel("시도당 평균 성공 확률")
axs[1].set_ylabel("±1.96 SE 구간의 실제 포함 비율")
axs[1].set_title("이항 SE 구간은 95%에 못 미침")
axs[0].legend(fontsize=8)
for a in axs:
    a.grid(alpha=.3)
fig.tight_layout()
fig.savefig(OUT / "own_toy_attempts.png", dpi=170)
plt.close(fig)

# 3. 상태 대용 토이
S = json.load(open(HERE / "toy_state_proxy_results.json"))
fig, ax = plt.subplots(figsize=(6.6, 4.2))
for sp, mk in ((1, "o"), (2, "s"), (4, "^")):
    rows = [s for s in S if s["speed"] == sp]
    ax.plot([s["alpha"] for s in rows], [s["proxy_removes_pct_of_naive_error"] for s in rows], mk + "-", label=f"속도 x{sp}")
ax.axhline(0, color="k", lw=1)
ax.set_xscale("log")
ax.set_xlabel("추종 이득 alpha (작을수록 느리게 따라감)")
ax.set_ylabel("마지막 명령을 미래 상태로 쓸 때 줄어드는 오차 (%)")
ax.set_title("지연 4스텝: 추종이 느리면 대용이 오히려 나쁨")
ax.legend(fontsize=8)
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(OUT / "own_state_proxy.png", dpi=170)
plt.close(fig)

# 4. 반응 지연 속도 향상의 기준 의존
fig, ax = plt.subplots(figsize=(6.6, 4.2))
Ks = np.array([5, 10, 15, 25, 35, 50])
Lg = {"H100": 23.2, "RTX 5090": 29.4, "RTX 4090": 32.3}
for g, L in Lg.items():
    ax.plot(Ks, (Ks * 20 + 2 * L) / (2 * L), "o-", label=g)
ax.axvline(25, color="k", ls=":", lw=1)
ax.set_xlabel("동기 방식의 실행 청크 길이 K (50 Hz)")
ax.set_ylabel("최대 반응 지연 속도 향상 (동기 / VLASH)")
ax.set_title("표 II의 8.7–11.8배는 K = 25 선택에 비례")
ax.legend(fontsize=8)
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(OUT / "own_latency_k.png", dpi=170)
plt.close(fig)
print("wrote 4 figures to", OUT)
