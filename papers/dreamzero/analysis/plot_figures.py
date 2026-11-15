"""우리가 그린 그래프 네 장 (report/figures/own_*.png). 공개 가능(논문 그림이 아니라 재계산·토이 결과).
실행: python3 analysis/plot_figures.py  (matplotlib, numpy; 한글 폰트는 Apple SD Gothic Neo → Noto CJK KR 순서)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
AN = ROOT / "analysis"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
for name in ("Apple SD Gothic Neo", "Noto Sans CJK KR", "Noto Sans KR"):
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False
GREY, ACC, LIGHT = "#55544f", "#1f5fae", "#9bb7dc"

tc = json.loads((AN / "table_check_results.json").read_text(encoding="utf-8"))["checks"]
lat = json.loads((AN / "latency_results.json").read_text(encoding="utf-8"))
toy1 = json.loads((AN / "toy_dit_cache_results.json").read_text(encoding="utf-8"))
toy2 = json.loads((AN / "toy_decoupled_noise_results.json").read_text(encoding="utf-8"))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=180)
    plt.close(fig)
    print("wrote", OUT / name)


# 1. 표 3 (4단계 대 1단계 대 Flash)와 DROID 성공률의 95% 구간
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
ax = axes[0]
labels = list(tc["table3_ci"].keys())
means = [tc["table3_ci"][k]["mean"] for k in labels]
lo = [m - tc["table3_ci"][k]["ci95"][0] for k, m in zip(labels, means)]
hi = [tc["table3_ci"][k]["ci95"][1] - m for k, m in zip(labels, means)]
xs = range(len(labels))
ax.bar(xs, means, color=[LIGHT, LIGHT, ACC], width=0.6)
ax.errorbar(xs, means, yerr=[lo, hi], fmt="none", ecolor=GREY, capsize=4)
for x, m in zip(xs, means):
    ax.text(x, m + 3, f"{m}", ha="center", fontsize=8, color=GREY)
ax.set_xticks(list(xs)); ax.set_xticklabels(["4단계\n350 ms", "1단계\n150 ms", "Flash 1단계\n150 ms"], fontsize=8)
ax.set_ylim(0, 105); ax.set_ylabel("과제 진행도 (%)", fontsize=9)
ax.set_title("표 3: 테이블 버싱 (n = 10, 95% t 구간)", fontsize=9)
ax = axes[1]
d = tc["droid_success"]["ci"]
keys = ["unseen DZ", "unseen GR00T pre", "unseen pi0.5 pre"]
names = ["DreamZero", "GR00T N1.6\n(사전학습)", "π0.5\n(사전학습)"]
rates = [d[k]["rate"] for k in keys]
wl = [r - d[k]["wilson95"][0] for k, r in zip(keys, rates)]
wh = [d[k]["wilson95"][1] - r for k, r in zip(keys, rates)]
xs = range(len(keys))
ax.bar(xs, rates, color=[ACC, LIGHT, LIGHT], width=0.6)
ax.errorbar(xs, rates, yerr=[wl, wh], fmt="none", ecolor=GREY, capsize=4)
for x, r in zip(xs, rates):
    ax.text(x, r + 3, f"{r}", ha="center", fontsize=8, color=GREY)
ax.set_xticks(list(xs)); ax.set_xticklabels(names, fontsize=8)
ax.set_ylim(0, 60); ax.set_ylabel("성공률 (%)", fontsize=9)
ax.set_title("DROID 미관측 과제 성공률 (n = 80, Wilson 95%)", fontsize=9)
for a in axes:
    a.spines[["top", "right"]].set_visible(False)
    a.tick_params(labelsize=8)
save(fig, "own_ci_success.png")

# 2. 필요한 실효 처리량 대 장비 피크
fig, ax = plt.subplots(figsize=(8.6, 3.9))
cases = lat["per_gpu_effective_TFs_needed"]
short = {
    "논문 Table 3: 4단계(캐시), 350 ms": "4단계 350 ms\n(논문 Table 3)",
    "논문 Table 3: Flash 1단계, 150 ms": "Flash 1단계\n150 ms",
    "논문 목표: 7 Hz 청크 주기 143 ms, 4단계": "4단계 143 ms\n(7 Hz 목표)",
    "공개 코드 기본 mask(8 DiT 단계), README GB200 0.6 s": "8단계 0.6 s\n(공개 README)",
}
ks = list(cases.keys())
vals = [cases[k]["needed_TF_per_s"] for k in ks]
ax.barh(range(len(ks)), vals, color=[ACC if "Flash" not in k else "#2f8f5b" for k in ks], height=0.55)
for i, v in enumerate(vals):
    ax.text(v + 25, i, f"{v:,.0f} TF/s", va="center", fontsize=8, color=GREY)
from matplotlib.lines import Line2D
handles = []
for x, lbl, c in [(2250, "B200 BF16 dense 2.25 PF/s", "#b05a2a"), (4500, "B200 FP8 dense 4.5 PF/s", "#8a6d1f"), (9000, "B200 FP4 dense 9 PF/s", "#5a7d2f")]:
    ax.axvline(x, color=c, ls="--", lw=1)
    handles.append(Line2D([0], [0], color=c, ls="--", lw=1, label=lbl))
ax.legend(handles=handles, fontsize=7, frameon=False, loc="lower right")
ax.set_yticks(range(len(ks))); ax.set_yticklabels([short[k] for k in ks], fontsize=8)
ax.set_xlabel("GPU 하나당 필요한 실효 처리량 (TF/s, 추정)", fontsize=9)
ax.set_xlim(0, 10000)
ax.set_title("14B DiT 청크 추론: 필요 처리량 대 사양 (가정: 480P, 최대 문맥, 순전파당 127 TF)", fontsize=9)
ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=8)
save(fig, "own_latency_needed.png")

# 3. 토이 1: DiT 호출 수 대 끝점 오차
fig, ax = plt.subplots(figsize=(7.4, 3.8))
tags = ["A", "B", "C", "D", "E"]
for (name, v), tag in zip(toy1["results"].items(), tags):
    col = "#b05a2a" if "논문" in name else ACC
    ax.scatter(v["dit_calls"], v["rms_endpoint_error"], s=60, color=col, zorder=3)
    ax.annotate(tag, (v["dit_calls"], v["rms_endpoint_error"]), textcoords="offset points", xytext=(7, 5), fontsize=9, color=GREY)
ax.text(0.98, 0.97, "\n".join(f"{t}: {n}" for t, n in zip(tags, toy1["results"].keys())), transform=ax.transAxes,
        ha="right", va="top", fontsize=6.8, color=GREY, linespacing=1.5)
ax.set_xlabel("DiT 호출 수 (샘플 한 번당)", fontsize=9)
ax.set_ylabel("끝점 RMS 오차 (참조 대비)", fontsize=9)
ax.set_xlim(2, 18); ax.set_ylim(0, 0.5)
ax.set_title("토이 1: 속도 재사용 캐시와 끝점 오차 (정확한 속도장, 2차원)", fontsize=9)
ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=8)
save(fig, "own_toy_cache.png")

# 4. 토이 2: coupled 대 decoupled
fig, ax = plt.subplots(figsize=(7.4, 3.6))
keys = ["1step_video_0.125", "4step_video_0.125", "1step_video_clean"]
lbl = ["1단계, 영상 t=0.125", "4단계, 영상 t=0.125", "1단계, 영상 깨끗(t=1)"]
import numpy as np
w = 0.26
names = {"coupled": "coupled", "decoupled": "decoupled (Flash형)", "indep": "대조군 indep (독립 U)"}
for j, (scheme, col) in enumerate([("coupled", LIGHT), ("decoupled", ACC), ("indep", "#c9b27c")]):
    m = [toy2["runs"][scheme][k]["mean"] for k in keys]
    s = [toy2["runs"][scheme][k]["se"] for k in keys]
    xs = np.arange(len(keys)) + (j - 1) * w
    ax.bar(xs, m, width=w, color=col, label=names[scheme])
    ax.errorbar(xs, m, yerr=s, fmt="none", ecolor=GREY, capsize=3)
ax.set_xticks(range(len(keys))); ax.set_xticklabels(lbl, fontsize=8)
ax.set_ylabel("행동 MSE (시드 5개 평균 ± SE)", fontsize=9)
ax.set_ylim(0, 0.9)
ax.set_title("토이 2: 영상과 행동의 잡음 스케줄 (장난감 세계, 방향만 확인)", fontsize=9)
ax.legend(fontsize=8, frameon=False)
ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=8)
save(fig, "own_toy_decoupled.png")
