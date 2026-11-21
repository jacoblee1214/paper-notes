"""우리가 그린 그래프 세 장 (report/figures/own_*.png). 논문 그림은 쓰지 않는다.

실행: python3 analysis/plot_figures.py
입력: 논문 표 1·그림 7·표 2의 값(하드코딩), analysis/recompute_results.json의 Poisson 구간.
한글 폰트: Linux Noto CJK → macOS Apple SD Gothic Neo 순서로 찾는다.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]
font_path = next((p for p in FONT_CANDIDATES if Path(p).exists()), None)
if font_path:
    font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 9

INK, GREY, ACC, SOFT = "#1b1b1a", "#8a8880", "#1f5fae", "#c9d7ec"
res = json.loads((HERE / "recompute_results.json").read_text(encoding="utf-8"))


def style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK)


# 그림 1: 표 1 실행 R@1 (3개 시드 SD) 와 본문에 적힌 SemPhys–MDM 95% 구간 대 시드 SD 구간
t1 = res["table1_as_read"]
names = sorted(t1, key=lambda k: t1[k]["r1"])
fig, ax = plt.subplots(figsize=(7.2, 3.6))
cols = [ACC if n == "SemPhys" else SOFT for n in names]
ax.barh(names, [t1[n]["r1"] for n in names], color=cols, edgecolor=GREY, linewidth=0.4)
ax.errorbar([t1[n]["r1"] for n in names], range(len(names)),
            xerr=[t1[n]["sd_r1"] for n in names], fmt="none", ecolor=INK, capsize=3, lw=0.8)
for i, n in enumerate(names):
    ax.text(t1[n]["r1"] + 1.2, i, f'{t1[n]["r1"]:.2f}', va="center", fontsize=8, color=INK)
ax.set_xlim(0, 100)
ax.set_xlabel("실행 후 R@1 (%), 표 1 (막대 끝: 3개 생성 시드 평균, 오차막대: SD)")
ax.set_title("실행 R@1 순위 (논문 표 1 다시 그림)", loc="left", color=INK)
style(ax)
fig.tight_layout()
fig.savefig(OUT / "own_r1_table.png", dpi=200)
plt.close(fig)

# 그림 2: 실패 사건 수 (Poisson 95%, 13,082 upright tasks)
fr = res["fail_event_counts_mean_over_seeds"]
fnames = list(fr)
fig, ax = plt.subplots(figsize=(7.2, 3.0))
for i, n in enumerate(fnames):
    e = fr[n]["expected_events"]
    lo, hi = fr[n]["exact_poisson95_events"]
    c = ACC if n == "SemPhys" else GREY
    ax.errorbar([e], [i], xerr=[[e - lo], [hi - e]], fmt="o", color=c, capsize=3)
    ax.text(hi * 1.12, i, f"약 {e:.0f}건 ({fr[n]['fail_per_thousand']}‰)" if e >= 10 else f"약 {e:.1f}건 ({fr[n]['fail_per_thousand']}‰)", va="center", fontsize=8, color=INK)
ax.set_yticks(range(len(fnames)), fnames)
ax.set_xscale("log")
ax.set_xlim(0.4, 2000)
ax.set_xlabel("실패 사건 수 (기대값 = 보고된 ‰ × 13,082, 정확 Poisson 95% 구간)")
ax.set_title("실패율이 사건 몇 건에서 나오는가", loc="left", color=INK)
style(ax)
fig.tight_layout()
fig.savefig(OUT / "own_fail_counts.png", dpi=200)
plt.close(fig)

# 그림 3: 절제·메커니즘 실험의 R@1 (보고된 평균, SD가 있으면 오차막대)
abl = [
    ("전체 SemPhys (그림 7, 3시드)", 79.30, 0.16),
    ("앵커 없음 (그림 7, 3시드)", 74.61, 6.35),
    ("생성 참조에만 의미 손실 (그림 7)", 74.21, 0.40),
    ("확산 지도 (matched control, 4.3절)", 73.31, None),
    ("창 단독 (그림 8)", 64.88, 1.09),
    ("끝점 패딩 (그림 8)", 68.07, 1.47),
    ("순서 있는 전후 문맥 H=32 (그림 8)", 79.31, 0.16),
    ("H=64 (표 2)", 80.11, 0.32),
    ("H=96 (표 2)", 79.81, 0.20),
    ("GT 궤적 재구성 (그림 7)", 56.93, None),
    ("선호도 학습 (4.3절)", 48.54, None),
]
fig, ax = plt.subplots(figsize=(7.2, 4.2))
y = list(range(len(abl)))[::-1]
for yi, (lab, v, sd) in zip(y, abl):
    c = ACC if lab.startswith("전체") else GREY
    if sd is None:
        ax.plot([v], [yi], marker="D", color=c, ms=5)
    else:
        ax.errorbar([v], [yi], xerr=[sd], fmt="o", color=c, capsize=3)
    ax.text(max(v, 0) + 1.2 + (sd or 0), yi, f"{v:.2f}" + ("" if sd is None else f" ± {sd:.2f}"),
            va="center", fontsize=7.5, color=INK)
ax.set_yticks(y, [a[0] for a in abl], fontsize=7.8)
ax.set_xlim(40, 92)
ax.set_xlabel("실행 R@1 (%). 마름모: SD 없음 (논문에 보고되지 않음)")
ax.set_title("절제와 메커니즘 실험의 실행 R@1 (보고값 그대로)", loc="left", color=INK)
style(ax)
fig.tight_layout()
fig.savefig(OUT / "own_ablations.png", dpi=200)
plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("own_*.png")))
