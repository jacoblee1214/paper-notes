"""우리가 논문 수치로 그린 그래프 두 장 (논문 그림이 아니다. 공개 가능).

실행: python3 analysis/plot_figures.py → report/figures/own_acceptance.png, own_ladder.png
입력은 논문 본문·표·부록의 보고값뿐이다.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

for cand in ["Noto Sans CJK KR", "Apple SD Gothic Neo", "AppleGothic"]:
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False
INK, MUTE, ACC, WARN, SOFT = "#1b1b1a", "#55544f", "#1f5fae", "#b5651d", "#9db8d9"

# 1. 제안 723개의 결과 구성 (부록 E). 최소 변형 목표 대 도로 여유 목표
labels = ["최소 변형 목표", "도로 여유 목표"]
accept = [383, 507]
clr = [331, 206]
head = [9, 10]
fig, ax = plt.subplots(figsize=(7.2, 2.6), dpi=200)
y = range(len(labels))
ax.barh(y, [a / 723 * 100 for a in accept], color=ACC, label="수락")
ax.barh(y, [c / 723 * 100 for c in clr], left=[a / 723 * 100 for a in accept], color=WARN, label="도로 여유 거부")
ax.barh(y, [h / 723 * 100 for h in head], left=[(a + c) / 723 * 100 for a, c in zip(accept, clr)],
        color=SOFT, label="헤딩 거부")
for i, a in enumerate(accept):
    ax.text(a / 723 * 50, i, f"{a}/723 ({a / 723 * 100:.1f}%)", va="center", ha="center", color="white", fontsize=8)
    ax.text(a / 723 * 100 + clr[i] / 723 * 50, i, f"{clr[i]}", va="center", ha="center", color="white", fontsize=8)
ax.set_yticks(list(y), labels, fontsize=9)
ax.set_xlim(0, 100)
ax.set_xlabel("제안 비율 (%), 241 참조 x 3 시드 = 723", fontsize=8.5, color=MUTE)
ax.invert_yaxis()
ax.legend(ncol=3, fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.32))
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.set_title("검증 통과 비율: 목표 선택에 따른 물리 수정 제안의 결과 (논문 부록 E 수치로 그림)", fontsize=8.5, color=INK, loc="left")
fig.tight_layout()
fig.savefig(FIG / "own_acceptance.png")
plt.close(fig)

# 2. 품질과 추가 롤아웃 비용 (표 3, 표 4). 롤아웃은 로그 축
rows = [("EAS\n(선택만)", 90.4, 0.0), ("고정 규칙", 90.6, 1.0), ("PRS", 90.7, 1.0),
        ("FCRS\n(학생)", 91.0, 1.0), ("CAST\n(탐색 교사)", 91.2, 9.81)]
fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=200)
xs = [max(r[2], 0.6) for r in rows]
ys = [r[1] for r in rows]
ax.scatter(xs, ys, s=60, color=[MUTE, MUTE, MUTE, ACC, WARN], zorder=3)
offsets = {"EAS\n(선택만)": (6, 6), "고정 규칙": (12, -12), "PRS": (12, 0), "FCRS\n(학생)": (12, 12), "CAST\n(탐색 교사)": (6, 6)}
for (name, pdms, roll), x, y0 in zip(rows, xs, ys):
    txt = f"{name}\n{pdms:.1f}" if "\n" in name else f"{name} {pdms:.1f}"
    ax.annotate(txt, (x, y0), textcoords="offset points", xytext=offsets[name], fontsize=7.5, color=INK)
ax.axhline(89.0, color=MUTE, lw=0.8, ls="--")
ax.text(0.62, 89.07, "WorldDrive 기준 89.0 (논문 표 1)", fontsize=7.5, color=MUTE, va="bottom")
ax.set_xscale("log")
ax.set_xlim(0.5, 14)
ax.set_xticks([0.6, 1, 10], ["0 (EAS)", "1", "10"], fontsize=8)
ax.set_xlabel("수정 한 번당 추가 롤아웃 수 (로그 축, EAS는 0을 0.6에 표시)", fontsize=8.5, color=MUTE)
ax.set_ylabel("NAVSIM v1 PDMS (navtest)", fontsize=8.5, color=MUTE)
ax.set_ylim(88.6, 91.6)
ax.tick_params(labelsize=8)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.set_title("품질과 비용: 학생은 롤아웃 1회로 교사 이득의 73.2%를 얻음 (논문 표 4)", fontsize=8.5, color=INK, loc="left")
fig.tight_layout()
fig.savefig(FIG / "own_ladder.png")
plt.close(fig)
print("wrote", FIG / "own_acceptance.png", FIG / "own_ladder.png")
