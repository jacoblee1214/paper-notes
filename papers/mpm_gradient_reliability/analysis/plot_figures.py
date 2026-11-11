"""toy 결과 JSON을 그래프로 그린다. 실행: python3 analysis/plot_figures.py → report/figures/toy_*.png (CPU, 몇 초)."""
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, NullFormatter

HERE = Path(__file__).resolve().parent


def ascii_log_ticks(fig):
    """로그 눈금의 음수 기호(U+2212)가 한글 폰트에 없어 깨진다. ASCII 하이픈으로 찍는다."""
    for a in fig.axes:
        for axis in (a.xaxis, a.yaxis):
            if axis.get_scale() in ("log", "symlog"):
                axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0e}"))
                axis.set_minor_formatter(NullFormatter())


FIG = HERE.parent / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
names = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
for fam in ("Apple SD Gothic Neo", "Noto Sans CJK KR", "NanumGothic"):
    if fam in names:
        plt.rcParams["font.family"] = fam
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"  # 지수 눈금의 음수 기호가 한글 폰트에 없어서 분리

fd = json.loads((HERE / "fd_chaos_results.json").read_text())
red = json.loads((HERE / "reduction_order_results.json").read_text())
svd = json.loads((HERE / "svd_vjp_results.json").read_text())

# 1) FD 신뢰도: (a) 혼돈 사상의 U 곡선, (b) 항복 클램프의 임계 사건
fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.3))
for r, c in zip(fd["chaos"]["rows"][1:], ("#9ab7d3", "#4f7fb3", "#1f5fae", "#0b3c74")):
    hs = np.array([float(h) for h in r["fd_relerr_by_h"]])
    y = np.array(list(r["fd_relerr_by_h"].values()))
    ax[0].loglog(hs, np.maximum(y, 1e-16), "o-", color=c, ms=3, lw=1.2, label=f"T={r['T']}")
ax[0].set_xlabel("FD 간격 h (a 기준 상대)")
ax[0].set_ylabel("|FD - 참값| / |참값|")
ax[0].set_title("(a) 혼돈 사상: 절단 대 반올림", fontsize=9)
ax[0].legend(frameon=False, fontsize=7)
hc = np.logspace(-1, -9, 9)
for r, c in zip(fd["clamp"]["rows"][::2], ("#c96b2c", "#7a3e12", "#2e7d4f")):
    y = np.array([x["relerr_vs_ad"] for x in r["fd_by_h"]])
    ax[1].loglog(hc, np.maximum(y, 1e-16), "s-", color=c, ms=3, lw=1.2,
                 label=f"n={r['n_steps']} (소성 사건 {r['n_plastic_events']})")
ax[1].set_xlabel("FD 간격 h (Y 기준 절대)")
ax[1].set_ylabel("|FD - AD| / |AD|")
ax[1].set_title("(b) 항복 클램프: 임계가 창에 들 때 어긋남", fontsize=9)
ax[1].legend(frameon=False, fontsize=7)
fig.tight_layout()
ascii_log_ticks(fig)
fig.savefig(FIG / "toy_fd_reliability.png", dpi=200)
plt.close(fig)

# 2) 누적 순서 → 기울기 평균·산포와 부호 뒤집힘
rows_a = [r for r in red["rows"] if r["loss"] == "A"]
rows_b = [r for r in red["rows"] if r["loss"] == "B"]
T = np.array([r["T"] for r in rows_b])
ref = np.array([r["ref_f64_fixed_order"] for r in rows_b])
mean = np.array([r["f32_random_mean"] for r in rows_b])
sd = np.array([r["f32_random_sd"] for r in rows_b])
fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.3))
ax[0].errorbar(T, mean, yerr=sd, fmt="o", color="#1f5fae", ms=3, capsize=2, label="f32 무작위 순서: 평균 ± sd")
ax[0].plot(T, ref, "k*", ms=7, label="f64 고정 순서 기준")
ax[0].axhline(0, color="#999", lw=0.6)
ax[0].set_yscale("symlog", linthresh=1e-2)
ax[0].set_yticks([-1e9, -1e6, -1e3, -1, 0, 1, 1e3, 1e6, 1e9])
ax[0].set_xlabel("롤아웃 길이 T (스텝)")
ax[0].set_ylabel("∂L/∂a (손실 B, symlog)")
ax[0].set_title("(a) 기울기 평균과 산포", fontsize=9)
ax[0].legend(frameon=False, fontsize=7)
rel = np.array([r["f32_random_sd"] / abs(r["ref_f64_fixed_order"]) for r in rows_a])
ax[1].semilogy([r["T"] for r in rows_a], rel, "o-", color="#c96b2c", ms=3, label="상대 sd = sd/|기준| (손실 A)")
ax[1].semilogy(T, [r["sign_flip_fraction_vs_ref"] + 1e-4 for r in rows_b], "s--", color="#2e7d4f", ms=3,
               label="부호 뒤집힘 비율 + 1e-4 (손실 B)")
ax[1].set_xlabel("롤아웃 길이 T (스텝)")
ax[1].set_title("(b) 산포 성장과 부호 뒤집힘", fontsize=9)
ax[1].legend(frameon=False, fontsize=7)
fig.tight_layout()
ascii_log_ticks(fig)
fig.savefig(FIG / "toy_reduction_sign.png", dpi=200)
plt.close(fig)

# 3) SVD VJP: 특이값 간격에 따른 상대오차
gaps = np.array([r["gap"] for r in svd["sweeps"]["f64_naive"]])
fig, ax = plt.subplots(figsize=(4.8, 3.3))
for key, lab, c, mk in (("f32_naive", "f32, 일반 VJP", "#c96b2c", "o-"),
                        ("f64_naive", "f64, 일반 VJP", "#4f7fb3", "o-"),
                        ("f32_safe", "f32, 게이지 안전 식", "#2e7d4f", "s--")):
    y = np.array([r["median_relerr"] for r in svd["sweeps"][key]])
    ok = np.isfinite(y)
    ax.loglog(gaps[ok], y[ok], mk, color=c, ms=3, label=lab)
ax.loglog(gaps, 0.3 * 1.19e-7 / gaps, ":", color="#888", lw=0.8, label="0.3 eps32/g (참고선)")
ax.invert_xaxis()
ax.set_xlabel("특이값 간격 g = 1 - s2")
ax.set_ylabel("중앙 상대오차 (f64 FD 기준)")
ax.set_title("SVD VJP: 간격에 따른 오차", fontsize=9)
ax.legend(frameon=False, fontsize=7)
fig.tight_layout()
ascii_log_ticks(fig)
fig.savefig(FIG / "toy_svd_gap.png", dpi=200)
plt.close(fig)
print("wrote", sorted(p.name for p in FIG.glob("toy_*.png")))
