"""리포트용 그래프 네 장. 먼저 table_check.py와 noise_toy.py를 실행해 결과 JSON을 만든다.
실행: python3 analysis/plot_figures.py

1) noise_paths.png   : Fig. 9 스케줄 17개를 코드 식 그대로 index(u) 좌표에 그린 경로, 성공률 대 행동 최대 step
2) significance.png  : Section 4 비교의 차이와 95% 구간 (RoboTwin 과제 50 × 100 episode 이항 근사)
3) benchmark_gap.png : 부록 D 벤치마크별 OpenWAM-α와 1위의 차이, 순위
4) memory.png        : 체크포인트 구성(bf16 GB)과 이 맥(24 GiB 통합 메모리)
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FONT = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                          "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
                          "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e4e7"
BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#9a9a9f"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
TC = json.loads((HERE / "table_check_results.json").read_text())
NT = json.loads((HERE / "noise_toy_results.json").read_text())


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2)


def u_of(s):
    s = np.clip(np.asarray(s, float), 0, 1)
    return 1 - s / (5 - 4 * s)


FAM = {"VS action": (ORANGE, "o", "분산 이동, 행동 선행"), "VS video": (BLUE, "o", "분산 이동, 영상 선행"),
       "LO action": (ORANGE, "s", "선형 지연, 행동 선행"), "LO video": (BLUE, "s", "선형 지연, 영상 선행")}


def fam_of(nm):
    if nm == "sync":
        return None
    f, lead = nm.split()[:2]
    return f"{f} {lead.split('-')[0]}"


# ---------------------------------------------------------------- 1) noise paths
fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.6), gridspec_kw={"width_ratios": [1, 1.15]})
ax = axs[0]
for nm, pth in NT["paths_sigma"].items():
    p = np.array(pth)
    uv, ua = u_of(p[:, 0]), u_of(p[:, 1])
    f = fam_of(nm)
    if f is None:
        ax.plot(uv, ua, "-o", color=INK, lw=2, ms=3.5, zorder=5, label="sync (대각선)")
        continue
    if nm.endswith(("α=32", "o=0.8")):
        c, mk, lab = FAM[f]
        ax.plot(uv, ua, "-", marker=mk, color=c, lw=1.4, ms=3.5, alpha=0.95, label=f"{lab} ({nm.split()[-1]})")
    else:
        ax.plot(uv, ua, "-", color=FAM[f][0], lw=0.7, alpha=0.35)
ax.set_xlabel("영상 진행 u_v (index 좌표, 0 = 순수 noise)")
ax.set_ylabel("행동 진행 u_a")
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_title("(a) Fig. 9 경로 17개, 점 = 10 step의 평가 위치\n학습 표본은 이 좌표에서 평면 전체에 균등", fontsize=9, color=INK)
ax.legend(fontsize=6.8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
style(ax)

ax = axs[1]
ps = NT["path_stats"]
se = 1.96 * TC["C6_significance_robotwin"]["se_pp_at_92pct"]["average"]
for nm, s in ps.items():
    f = fam_of(nm)
    c, mk = (INK, "D") if f is None else FAM[f][:2]
    ax.errorbar(s["max_action_du"], s["fig9"], yerr=se, fmt=mk, color=c, ms=5, capsize=2, lw=0.8,
                mfc=c if (f is None or f.startswith("VS")) else "white")
    if nm in ("sync", "VS action-lead α=32", "VS action-lead α=4", "LO video-lead o=0.8", "LO action-lead o=0.8"):
        ax.annotate(nm, (s["max_action_du"], s["fig9"]), xytext=(5, 3), textcoords="offset points", fontsize=7, color=INK2)
rho = NT["spearman_path_stat_vs_fig9"]["max_action_du"]["rho"]
ax.set_xlabel("행동 스트림의 가장 큰 한 step (u 좌표)")
ax.set_ylabel("Fig. 9 성공률 (%)")
ax.set_title(f"(b) 성공률은 행동 step 크기와 함께 떨어진다 (Spearman ρ = {rho:.2f}, n = 17)\n"
             "막대 = 이항 95% 구간 (±0.53%p), 학습 seed 분산 미포함", fontsize=9, color=INK)
style(ax)
fig.tight_layout()
fig.savefig(OUT / "noise_paths.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- 2) significance
tests = TC["C6_significance_robotwin"]["tests"]
pick = [("T1 Dual JSA vs Single Vanilla (avg)", "Table 1: Dual JSA - Single Vanilla"),
        ("T2 ASV vs Isolated (avg)", "Table 2: Action Sees Video - Isolated"),
        ("Fig7 pretrained 5B vs scratch 5B (avg)", "Fig. 6/7: 사전학습 DiT - 무작위 초기화"),
        ("Fig6 14B vs 5B (avg)", "Fig. 6: 14B - 5B"),
        ("T1 Tri JSA vs Dual JSA (avg)", "Table 1: Tri - Dual JSA"),
        ("T2 ASV vs Mutual (avg)", "Table 2: ASV - Mutual (scratch)"),
        ("Fig9 sync vs best async LO action o=0.6 (avg)", "Fig. 9: sync - 최선 async"),
        ("Fig9 sync vs VS action α=32 (avg)", "Fig. 9: sync - 행동 선행 α=32"),
        ("Fig10 OOD Robot Only vs Scratch", "Fig. 10 OOD: Robot only - scratch"),
        ("Fig10 OOD co-train vs Robot Only", "Fig. 10 OOD: co-train - Robot only"),
        ("Fig10 OOD co-train vs 2단계", "Fig. 10 OOD: co-train - 2단계"),
        ("Fig10 ID co-train vs Scratch", "Fig. 10 ID: co-train - scratch"),
        ("Fig11 Pretrained Full Mutual-ASV", "Fig. 11 Full: Mutual - ASV"),
        ("Fig11 C2R ID Mutual-ASV", "Fig. 11 C2R ID: Mutual - ASV"),
        ("Fig11 C2R OOD Mutual-ASV", "Fig. 11 C2R OOD: Mutual - ASV"),
        ("T16 ABot-M0.5 vs OpenWAM (Full avg)", "Table 16: ABot-M0.5 - OpenWAM-α")]
fig, ax = plt.subplots(figsize=(7.6, 5.0))
for i, (k, lab) in enumerate(pick[::-1]):
    t = tests[k]
    sig = t["p"] < 0.05
    c = BLUE if sig else MUTED
    ax.errorbar(t["diff_pp"], i, xerr=1.96 * t["se_pp"], fmt="o", color=c, ms=4.5, capsize=2.5, lw=1.1)
    ax.text(t["diff_pp"] + 1.96 * t["se_pp"] + 0.25, i, f"{t['diff_pp']:+.2f}", va="center", fontsize=7.2, color=INK2)
ax.axvline(0, color=INK2, lw=0.8)
ax.set_yticks(range(len(pick)))
ax.set_yticklabels([lab for _, lab in pick[::-1]], fontsize=7.8)
ax.set_xlabel("차이 (%p), 막대 = 95% 구간. 파랑 p < 0.05, 회색 유의하지 않음")
ax.set_title("Section 4 비교: 큰 효과(구조·마스크·사전학습 OOD)와 잡음 안의 차이", fontsize=9.5, color=INK)
style(ax)
fig.tight_layout()
fig.savefig(OUT / "significance.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- 3) benchmark gap
D = TC["C7_appendixD"]
rows = [("LIBERO", D["LIBERO_rank"], "WAM"), ("VLABench (SR)", D["VLABench_SR_rank"], "VLA"),
        ("RoboTwin Full", D["RoboTwin_Full_rank"], "WAM"), ("RoboCasa-GR1", D["RoboCasa_GR1_rank"], "VLA"),
        ("RoboTwin C2R", D["RoboTwin_C2R_rank"], "VLA"), ("RoboDojo (SR)", D["RoboDojo_rank"], "VLA"),
        ("RoboCasa365", D["RoboCasa365_rank"], "VLA"), ("LIBERO-Plus", D["LIBERO_Plus_rank"], "VLA"),
        ("EBench (SR)", D["EBench_rank"], "-")]
fig, ax = plt.subplots(figsize=(7.4, 3.8))
for i, (lab, rk, kind) in enumerate(rows[::-1]):
    g = rk["gap_to_best"]
    c = AQUA if g == 0 else (MUTED if g > -1 else ORANGE)
    ax.barh(i, g if g != 0 else 0.15, color=c, height=0.6)
    txt = f"{rk['rank']}/{rk['of']}위" + ("" if g == 0 else f" · 1위 {rk['best']} {rk['best_value']}")
    ax.text(min(g, 0) - 0.4, i, txt, va="center", ha="right", fontsize=7.2, color=INK2)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r_[0] for r_ in rows[::-1]], fontsize=8)
ax.set_xlim(-26, 1.5)
ax.axvline(0, color=INK2, lw=0.8)
ax.set_xlabel("OpenWAM-α - 표의 1위 (%p). 초록 = 1위, 회색 = 1%p 안, 주황 = 1%p 이상 뒤짐")
ax.set_title("부록 D 표 9개에서 OpenWAM-α의 위치 (논문 표 그대로, 재계산 일치)", fontsize=9.5, color=INK)
style(ax)
fig.tight_layout()
fig.savefig(OUT / "benchmark_gap.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- 4) memory
P = TC["C3_params"]["by_group_B"]
parts = [("Wan2.2 DiT 5.0B", P["video_backbone.dit"], BLUE), ("ActionDiT 1.0B", P["action_backbone"], AQUA),
         ("Wan2.2-VAE 0.7B (고정)", P["video_backbone.vae"], MUTED), ("umT5-xxl 5.7B (고정)", P["video_backbone.text_encoder"], ORANGE)]
fig, ax = plt.subplots(figsize=(7.4, 2.5))
left = 0
for lab, b, c in parts:
    gb = b * 2
    ax.barh(0, gb, left=left, color=c, height=0.5)
    if gb > 3:
        ax.text(left + gb / 2, 0, f"{lab}\n{gb:.1f} GB", ha="center", va="center", fontsize=7, color="white")
    else:
        ax.text(left + gb / 2, 0.3 if "Action" in lab else -0.3, f"{lab} {gb:.1f} GB", ha="center",
                va="bottom" if "Action" in lab else "top", fontsize=7, color=INK)
    left += gb
mac = 24 * 1.073741824
ax.axvline(mac, color=INK, lw=1.2, ls="--")
ax.text(mac + 0.2, 0.36, "이 맥 통합 메모리 24 GiB = 25.8 GB\n(OS·활성값 몫 포함)", fontsize=7.2, color=INK, va="bottom")
ax.axvline(TC["C3_params"]["inference_without_text_encoder_GB_bf16"], color=AQUA, lw=1, ls=":")
ax.text(13.6, 0.42, "텍스트 인코더를 빼면 13.4 GB", fontsize=7.2, color=INK2)
ax.set_xlim(0, 30)
ax.set_ylim(-0.7, 0.8)
ax.set_yticks([])
ax.set_xlabel("bf16 가중치 크기 (GB). 공개 체크포인트 24.81 GB = 12.41B 파라미터")
ax.set_title("공개 체크포인트 구성과 이 맥의 메모리", fontsize=9.5, color=INK)
style(ax)
fig.tight_layout()
fig.savefig(OUT / "memory.png", dpi=200)
plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("*.png")))
