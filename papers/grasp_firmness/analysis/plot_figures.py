"""우리가 만든 그래프 세 장을 report/figures/에 저장한다.

own_table1_check.png : 표 1의 편향 대 (과잉 - 부족), MAE 대 (과잉 + 부족) 점검
own_fig2_toy.png     : 그림 2 판독 값과 Hertz 토이(젤 있는 그리퍼 대 강체 그리퍼)
own_real_success.png : 실물 파지 성공률 Wilson 구간(시행 수 가정)과 그림 6d 판독 오차
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 입력은 recompute_results.json, toy_hertz_reduction_results.json, fig6d_readings_raw.json)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ANA = ROOT / "analysis"
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def korean_font():
    for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
INK, ACC, GREY, RED = "#1b1b1a", "#1f5fae", "#9a9890", "#c0392b"
LABELS = {"naive": "Naive", "class_prior": "Class prior", "direct": "Direct", "slip_shear": "Slip shear",
          "slip_area": "Slip area", "slip_gbm": "Slip GBM", "slip_images": "Slip images", "ours": "Ours",
          "no_prior": "no prior", "wrong_prior": "wrong prior", "coulomb_only": "Coulomb-only", "ours_margin": "Ours+margin"}


def table1():
    src = (ANA / "recompute.py").read_text(encoding="utf-8")
    ns = {}
    start = src.index("T1 = {"); end = src.index("R = {}")
    exec(src[start:end], ns)
    T1 = ns["T1"]
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 4.2))
    for k, v in T1.items():
        c = RED if k in ("ours", "slip_images") else ACC
        ax[0].scatter(v["exc"]-v["dfc"], v["bias"], color=c, s=26)
        ax[1].scatter(v["exc"]+v["dfc"], v["mae"], color=c, s=26)
        if k in ("ours", "slip_images"):
            ax[0].annotate(LABELS[k], (v["exc"]-v["dfc"], v["bias"]), textcoords="offset points", xytext=(6, -10), fontsize=8, color=RED)
            ax[1].annotate(LABELS[k], (v["exc"]+v["dfc"], v["mae"]), textcoords="offset points", xytext=(6, -10), fontsize=8, color=RED)
    for a, lim, xl, yl in ((ax[0], (-0.7, 2.1), "과잉 - 부족 [N]", "표 1 편향 [N]"), (ax[1], (0.4, 2.4), "과잉 + 부족 [N]", "표 1 MAE [N]")):
        a.plot(lim, lim, color=GREY, lw=1, ls="--")
        a.set_xlim(lim); a.set_ylim(lim); a.set_xlabel(xl); a.set_ylabel(yl)
        a.spines[["top", "right"]].set_visible(False)
    ax[0].set_title("편향 = 과잉 - 부족 (표 1 점추정)", fontsize=10)
    ax[1].set_title("MAE = 과잉 + 부족 (표 1 점추정)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "own_table1_check.png", dpi=170); plt.close(fig)


def fig2_toy():
    r = json.load(open(ANA / "toy_hertz_reduction_results.json"))
    # 그림 2 판독 값 (E=50, 1000 kPa)
    E = np.array([50, 1000])
    read_l, read_h = np.array([18.7, 2.4]), np.array([8.7, 1.2])
    gel = r["fit_to_fig2"]["share0.5_R40mm"]; rig = r["fit_to_fig2_rigid_gripper"]["rigid_gripper_share0.5_R40mm"]
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.plot(E, read_l, "o-", color=INK, label="그림 2 판독 (m=0.05 kg)")
    ax.plot(E, read_h, "s-", color=INK, mfc="white", label="그림 2 판독 (m=0.50 kg)")
    ax.plot(E, [18.7, gel["pred_m0.05_E1000"]], "o--", color=RED, label="토이: 젤 E=245 kPa (m=0.05)")
    ax.plot(E, [gel["pred_m0.50_E50"], gel["pred_m0.50_E1000"]], "s--", color=RED, mfc="white", label="토이: 젤 E=245 kPa (m=0.50)")
    ax.plot(E, [18.7, rig["pred_m0.05_E1000"]], "o:", color=ACC, label="토이: 강체 그리퍼 (m=0.05)")
    ax.plot(E, [rig["pred_m0.50_E50"], rig["pred_m0.50_E1000"]], "s:", color=ACC, mfc="white", label="토이: 강체 그리퍼 (m=0.50)")
    ax.set_xscale("log"); ax.set_xlabel("물체 탄성률 E [kPa]"); ax.set_ylabel("힘 감소율 [%]")
    ax.set_title("E=50 kPa에서 τ를 맞춘 뒤 E=1000 kPa 예측", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "own_fig2_toy.png", dpi=170); plt.close(fig)


def wilson(k, n, z=1.959964):
    p = k/n; den = 1+z*z/n
    c = (p+z*z/(2*n))/den; h = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return c-h, c+h


def real():
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 4))
    names = ["Class prior", "Ours"]
    for i, (nm, k) in enumerate(zip(names, (63, 84))):
        for off, n, col in ((-0.12, 87, ACC), (0.12, 29, RED)):
            kk = k if n == 87 else round(k*29/87)
            lo, hi = wilson(kk, n)
            ax[0].errorbar(i+off, 100*kk/n, yerr=[[100*(kk/n-lo)], [100*(hi-kk/n)]], fmt="o", color=col, capsize=3,
                           label=(f"시행 87회 가정" if n == 87 else "물체 29개 가정") if i == 0 else None)
    ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(names); ax[0].set_ylim(40, 102); ax[0].set_xlim(-0.5, 1.5)
    ax[0].set_ylabel("성공률 [%] (Wilson 95%)"); ax[0].legend(fontsize=8, frameon=False)
    ax[0].set_title("그림 6e 성공률 0.72와 0.97 (분모는 가정)", fontsize=10)
    raw = json.load(open(ANA / "fig6d_readings_raw.json"))
    idx = [i for i in range(25) if i not in (16, 17, 23)]
    d = np.array([raw[i][0]-raw[i][1] for i in idx]); c = np.array([raw[i][2]-raw[i][1] for i in idx])
    ax[1].hist(c, bins=np.arange(-0.6, 1.8, 0.1), color=GREY, alpha=0.7, label="Class prior")
    ax[1].hist(d, bins=np.arange(-0.6, 1.8, 0.1), color=RED, alpha=0.8, label="Ours")
    ax[1].axvline(0, color=INK, lw=1)
    ax[1].set_xlabel("(적용 힘 - 최적 힘) [N], 그림 6d 판독 (막대 22개)"); ax[1].set_ylabel("물체 수"); ax[1].legend(fontsize=8, frameon=False)
    ax[1].set_title("그림 6d 판독 값의 오차 분포", fontsize=10)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "own_real_success.png", dpi=170); plt.close(fig)


if __name__ == "__main__":
    table1(); fig2_toy(); real()
    print("wrote 3 figures")
