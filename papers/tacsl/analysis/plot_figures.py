"""TacSL 보고 수치로 그리는 자체 그래프 세 장 (논문 그림을 쓰지 않음).

출력: report/figures/own_speed_vs_envs.png, own_insertion_vs_bc.png, own_real_placement.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "report", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
from matplotlib import font_manager as fm

KO_FONT_CANDIDATES = [  # Linux Noto CJK -> macOS Apple SD Gothic Neo 순서
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]
for cand in KO_FONT_CANDIDATES:
    if os.path.exists(cand):
        fm.fontManager.addfont(cand)
        plt.rcParams["font.family"] = fm.FontProperties(fname=cand).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

# 1) 환경 수 대 처리량 (Table I, II). 기준선은 단일 환경 CPU.
fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=160)
envs_img = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
fps_img = [140, 259, 456, 740, 1045, 1288, 1470, 1545, 1609, 1631]
envs_ff = [1, 4, 16, 64, 256, 1024, 4096, 16384, 32768]
fps_ff = [152, 734, 2923, 10162, 32892, 127932, 452372, 1287686, 1541043]
ax.loglog(envs_img, fps_img, "o-", color="#1f6f8b", label="이미지 (표 I)")
ax.loglog(envs_ff, fps_ff, "s-", color="#c25b2a", label="힘장 10×10 (표 II)")
ax.axhline(7.28, color="#1f6f8b", ls=":", lw=1)
ax.text(1.1, 8.5, "기준선 Taxim 이미지 7.28 FPS (1 환경)", fontsize=7, color="#1f6f8b")
ax.axhline(3596, color="#c25b2a", ls=":", lw=1)
ax.text(1.1, 4300, "기준선 [16] 힘장 3596 FPS (1 환경)", fontsize=7, color="#c25b2a")
ax.set_xlabel("병렬 환경 수 (로그)")
ax.set_ylabel("처리량 FPS (로그)")
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.set_title("보고 수치의 환경 수별 처리량 (CPU 기준선은 1 환경)", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "own_speed_vs_envs.png"))
plt.close(fig)

# 2) 삽입 과제 입력별 DAgger, AAC 대 BC (표 IV). 시행 수 가정 없이 차이만 그림.
inputs = ["특권\n상태", "축소\n상태", "축소+\n촉각영상", "축소+\n힘장", "축소+\n손목", "축소+\n영상+힘장", "축소+\n영상+손목"]
bc = np.array([0.826, 0.087, 0.820, 0.804, 0.936, 0.570, 0.941])
dag = np.array([0.958, 0.053, 0.908, 0.925, 0.968, 0.876, 0.925])
aac = np.array([0.973, 0.000, 0.834, 0.930, 0.948, 0.932, 0.897])
x = np.arange(len(inputs))
w = 0.26
fig, ax = plt.subplots(figsize=(6.6, 3.8), dpi=160)
ax.bar(x - w, bc, w, label="BC (오프라인)", color="#9aa5ad")
ax.bar(x, dag, w, label="DAgger", color="#1f6f8b")
ax.bar(x + w, aac, w, label="AAC (RL)", color="#c25b2a")
ax.set_xticks(x)
ax.set_xticklabels(inputs, fontsize=7)
ax.set_ylabel("성공률 (표 IV)")
ax.set_ylim(0, 1.05)
ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
ax.set_title("삽입 과제 입력 조합별 성공률 (표 IV)", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "own_insertion_vs_bc.png"))
plt.close(fig)

# 3) 실로봇 배치 성공률과 Wilson 95% 구간 (본문 Fig 12 수치, 81 시행)
names = ["Vanilla", "ColorAug", "Diff+ColorAug", "Concat+ColorAug"]
k = np.array([22, 71, 74, 63])
n = 81
z = 1.96
p = k / n
d = 1 + z * z / n
c = (p + z * z / (2 * n)) / d
h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
fig, ax = plt.subplots(figsize=(5.6, 3.0), dpi=160)
xs = np.arange(len(names))
ax.errorbar(xs, p, yerr=[p - (c - h), (c + h) - p], fmt="o", color="#1f6f8b", capsize=4)
for i, (pp, kk) in enumerate(zip(p, k)):
    ax.text(i + 0.08, pp + 0.02, f"{kk}/81", fontsize=7)
ax.set_xticks(xs)
ax.set_xticklabels(names, fontsize=8)
ax.set_ylim(0, 1.05)
ax.set_ylabel("성공률 (Wilson 95%)")
ax.set_title("실로봇 배치 과제 81회: 시행 수가 작아 구간이 넓음", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "own_real_placement.png"))
plt.close(fig)
print("ok", sorted(os.listdir(FIG)))
