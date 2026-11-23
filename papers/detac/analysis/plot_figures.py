"""우리가 만든 그래프 세 장을 report/figures/ 에 저장한다(논문 그림은 쓰지 않는다).

  own_t1_psnr_fid.png  표 1 값(Unseen pose, Unseen object)의 FID 대 PSNR
  own_a1_wilson.png    표 A1 성공 수(30회/직물)와 Wilson 95% 구간, Fisher p
  own_bench.png        부록 A.1 구조(랜덤 가중치)의 한 스텝 시간(맥, 가정 해상도)

입력: analysis/recompute_results.json, analysis/toy_unet_bench_results.json
실행: python3 analysis/plot_figures.py   (matplotlib, 수 초)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RC = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))
BN = json.loads((ROOT / "analysis" / "toy_unet_bench_results.json").read_text(encoding="utf-8"))

for cand in ["/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
             "/System/Library/Fonts/AppleSDGothicNeo.ttc"]:
    if Path(cand).exists():
        font_manager.fontManager.addfont(cand)
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
matplotlib.rcParams["axes.unicode_minus"] = False
INK, MUTED, ACC, ACC2, WARN = "#1b1b1a", "#6b6a64", "#1f5fae", "#d9822b", "#a33a3a"

# 표 1 (FID, PSNR)
T1 = {
    "TELEA": ((237.95, 31.33), (232.66, 31.29)), "TacDiff": ((158.94, 31.52), (155.63, 31.40)),
    "LaMa": ((358.16, 29.09), (360.27, 29.16)), "SD 1.5": ((121.70, 25.44), (114.29, 24.16)),
    "U-Net": ((108.35, 38.56), (111.28, 39.03)), "Pix2Pix": ((112.53, 38.53), (109.66, 38.98)),
    "dual-head": ((104.27, 37.99), (105.18, 38.37)), "TacDiff enl.": ((172.01, 30.93), (180.96, 30.84)),
    "LaMa enl.": ((330.13, 30.23), (330.57, 30.27)), "DeTac w/o Lsur": ((25.78, 34.64), (22.59, 35.05)),
    "DeTac w/o Lcon": ((22.65, 37.77), (19.44, 38.28)), "DeTac Full": ((20.44, 39.45), (17.06, 39.07)),
}


def fig1():
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    for k, (ax, title) in enumerate(zip(axs, ("Unseen pose (49장)", "Unseen object (100장)"))):
        for name, v in T1.items():
            fid, psnr = v[k]
            col = ACC if name.startswith("DeTac") else (ACC2 if name in ("U-Net", "Pix2Pix", "dual-head") else MUTED)
            ax.scatter(fid, psnr, color=col, s=36, zorder=3)
            off = {"U-Net": (-28, 7), "Pix2Pix": (6, 6), "dual-head": (-4, -11)}.get(name, (3, 3))
            ax.annotate(name, (fid, psnr), fontsize=7, xytext=off, textcoords="offset points", color=INK)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("FID (낮을수록 좋음)")
        ax.grid(alpha=.25)
    axs[0].set_ylabel("PSNR (dB)")
    fig.suptitle("표 1 값: FID와 PSNR (파랑 DeTac 계열, 주황 같은 쌍 데이터의 결정론적 기준선)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_t1_psnr_fid.png", dpi=170)
    plt.close(fig)


def fig2():
    A = RC["V8_tableA1"]
    labs = ["Fabric A", "Fabric B", "Fabric C", "전체(90회)"]
    fig, ax = plt.subplots(figsize=(7.2, 4))
    for off, key, col, nm in ((-0.18, "raw", MUTED, "Raw mixed image"), (0.18, "deTac", ACC, "DeTac marker-removed")):
        cnt = A["counts"][key] + [A["counts"][key + "_overall"]]
        ns = [30, 30, 30, 90]
        for i, (k, n, w) in enumerate(zip(cnt, ns, A["wilson"][key])):
            p = 100 * k / n
            ax.errorbar(i + off, p, yerr=[[p - w[0]], [w[1] - p]], fmt="o", color=col, capsize=4, label=nm if i == 0 else None)
            ax.annotate(f"{k}/{n}", (i + off, p), xytext=(6, -3), textcoords="offset points", fontsize=8, color=INK)
    for i, f in enumerate(("A", "B", "C", "overall")):
        ax.text(i, 3, f"Fisher p = {A['fisher_p'][f]:.2g}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(4)); ax.set_xticklabels(labs)
    ax.set_ylim(0, 105); ax.set_ylabel("성공률 (%)  Wilson 95% 구간")
    ax.grid(alpha=.25); ax.legend(fontsize=8, loc="upper center")
    ax.set_title("표 A1: 직물당 30회(가정: 독립 시행)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "own_a1_wilson.png", dpi=170); plt.close(fig)


def fig3():
    keys = ["cpu_128", "cpu_256", "mps_128", "mps_256"]
    labels = ["CPU 128", "CPU 256", "MPS 128", "MPS 256"]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
    axs[0].bar(labels, [BN["bench"][k]["train_300_epochs_hours"] for k in keys], color=[MUTED, MUTED, ACC, ACC])
    axs[0].axhline(14, color=WARN, ls="--"); axs[0].text(0.02, 14.8, "논문 보고 RTX 4090 14 h", color=WARN, fontsize=8)
    axs[0].set_ylabel("300 epoch 학습 시간 (h, 추정)"); axs[0].set_title("학습 (배치 2, 139,800 스텝)", fontsize=10)
    axs[1].bar(labels, [BN["bench"][k]["sample_1000_steps_s"] for k in keys], color=[MUTED, MUTED, ACC, ACC])
    axs[1].set_ylabel("1000 스텝 샘플링 (s, 한 장)"); axs[1].set_title("추론(T = 1000 가정)", fontsize=10)
    for a in axs:
        a.grid(alpha=.25, axis="y"); a.tick_params(labelsize=8)
    fig.suptitle(f"부록 A.1 구조 {BN['params_M']} M 파라미터, 랜덤 가중치. 해상도(128, 256)는 가정", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "own_bench.png", dpi=170); plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3()
    print("wrote 3 figures")
