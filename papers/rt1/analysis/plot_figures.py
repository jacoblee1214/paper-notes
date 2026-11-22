"""보고 수치로 우리가 그린 그래프 세 장 (matplotlib). 논문 그림은 쓰지 않는다.

실행: python3 analysis/plot_figures.py  -> report/figures/own_*.png
입력: 논문 표 값(하드코딩). 구간(Wilson 95%)은 분모를 가정한 값이며 캡션에 그렇게 적는다.
"""
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

for cand in ("Apple SD Gothic Neo", "Noto Sans CJK KR", "NanumGothic"):
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

INK = "#1b1b1a"
ACC = "#1f5fae"
GREY = "#8c8a83"
ORANGE = "#c8702a"


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


# ------------------------------------------------ 그림 1: 표 2 (분모 가정: 일반화 21, 방해물 30, 배경 17)
CATS = [("미관측 과제\n(n=21 가정)", 21, [76, 52, 19, 43]),
        ("방해물\n(n=30 가정)", 30, [83, 43, 47, 23]),
        ("배경\n(n=17 가정)", 17, [59, 35, 41, 35])]
MODELS = [("RT-1", ACC), ("Gato", GREY), ("BC-Z", ORANGE), ("BC-Z XL", "#7a5ea8")]
fig, ax = plt.subplots(figsize=(10, 4.6), dpi=160)
w = 0.2
for j, (name, col) in enumerate(MODELS):
    for i, (_, n, vals) in enumerate(CATS):
        k = round(vals[j] * n / 100)  # 가정 분모에서 가장 가까운 정수 횟수
        p = 100 * k / n
        lo, hi = wilson(k, n)
        x = i + (j - 1.5) * w
        ax.bar(x, p, width=w, color=col, label=name if i == 0 else None,
               yerr=[[p - lo], [hi - p]], capsize=2, error_kw=dict(lw=0.8, ecolor=INK))
        ax.text(x, hi + 2, f"{vals[j]}", ha="center", va="bottom", fontsize=7.5, color=INK)
ax.set_xticks(range(len(CATS)))
ax.set_xticklabels([c[0] for c in CATS], fontsize=9)
ax.set_ylabel("성공률 (%)")
ax.set_ylim(0, 110)
ax.set_title("표 2: 보고 값(막대 위 숫자)과 가정 분모의 Wilson 95% 구간", fontsize=10.5, color=INK)
ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16))
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "own_table2_ci.png")
plt.close(fig)

# ------------------------------------------------ 그림 2: 표 13 어블레이션 (전체 RT-1 대비 변화, %p)
ABL = [  # (이름, 본 과제, 미관측, 방해물, 배경)
    ("모델 크기 축소", 89, 62, 77, 53),
    ("ImageNet 사전학습 제거", 84, 43, 60, 41),
    ("연속 행동", 68, 43, 37, 35),
    ("자기회귀 행동", 85, 71, 67, 65),
    ("관측 이력 제거", 82, 62, 50, 59),
    ("Transformer 제거", 86, 62, 67, 59),
]
FULL = (97, 76, 83, 59)
fig, ax = plt.subplots(figsize=(10, 4.8), dpi=160)
cols = [ACC, GREY, ORANGE, "#7a5ea8"]
labels = ["본 과제", "미관측 과제", "방해물", "배경"]
ys = list(range(len(ABL)))[::-1]
for c_idx in range(4):
    for r, row in zip(ys, ABL):
        d = row[1 + c_idx] - FULL[c_idx]
        ax.scatter(d, r + (c_idx - 1.5) * 0.14, s=38, color=cols[c_idx],
                   label=labels[c_idx] if r == ys[0] else None, zorder=3)
ax.axvline(0, color=INK, lw=0.8)
ax.set_yticks(ys)
ax.set_yticklabels([a[0] for a in ABL], fontsize=9)
ax.set_xlabel("전체 RT-1 대비 변화 (%p, 표 13 보고 값 차이)")
ax.set_title("표 13: 어블레이션별 변화 (분모 미보고, 점 추정만 표시)", fontsize=10.5, color=INK)
ax.grid(axis="x", color="#e4e2dc", lw=0.6, zorder=0)
ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.14))
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "own_ablation_delta.png")
plt.close(fig)

# ------------------------------------------------ 그림 3: 표 7 데이터 양·다양성 (일반화 평균)
DATA_SIZE = [("100% 데이터\n100% 과제", 100, 73, ACC),
             ("51% 데이터", 51, 50, GREY),
             ("37% 데이터", 37, 46, GREY),
             ("22% 데이터", 22, 29, GREY)]
NARROW = (97, 75, 54)  # 데이터 97%, 과제 75%, 일반화 평균 54
fig, ax = plt.subplots(figsize=(10, 4.4), dpi=160)
xs = [d[1] for d in DATA_SIZE]
ys_ = [d[2] for d in DATA_SIZE]
ax.plot(xs, ys_, marker="o", color=GREY, lw=1.4, label="데이터 양 축소 (과제 100% 유지)")
ax.scatter([100], [73], s=70, color=ACC, zorder=3)
ax.scatter([NARROW[0]], [NARROW[2]], s=70, color=ORANGE, zorder=3, label="과제 축소 (데이터 97% 유지, 과제 75%)")
for lab, x, y, _ in DATA_SIZE:
    ax.annotate(f"{y}", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8.5, color=INK)
ax.annotate(f"{NARROW[2]}", (NARROW[0], NARROW[2]), textcoords="offset points", xytext=(6, 6), fontsize=8.5, color=INK)
ax.set_xlabel("학습 데이터 비율 (%)")
ax.set_ylabel("일반화 평균 (%)")
ax.set_xlim(0, 105)
ax.set_ylim(20, 85)
ax.set_title("표 7: 데이터 양과 다양성의 영향 (일반화 평균 = 세 일반화 열의 평균)", fontsize=10.5, color=INK)
ax.legend(frameon=False, fontsize=8.5, loc="lower right")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "own_data_diversity.png")
plt.close(fig)
print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
