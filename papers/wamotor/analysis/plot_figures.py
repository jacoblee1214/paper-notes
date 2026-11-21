"""자체 그림 3장: 재계산 결과와 토이 결과를 그린다. 논문 그림은 쓰지 않는다.

실행: python3 analysis/plot_figures.py → report/figures/own_*.png
한글 폰트는 Linux(Noto CJK) → macOS(Apple SD Gothic Neo) 순서로 찾는다.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

for cand in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
             "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
             "/System/Library/Fonts/AppleSDGothicNeo.ttc"]:
    if Path(cand).exists():
        font_manager.fontManager.addfont(cand)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

rec = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))
toy = json.loads((ROOT / "analysis" / "toy_speculative_results.json").read_text(encoding="utf-8"))

# 1) LIBERO-Plus: 지연 대 평균 성공률 (재계산 값)
import importlib.util
spec = importlib.util.spec_from_file_location("rc", ROOT / "analysis" / "recompute.py")
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)
fig, ax = plt.subplots(figsize=(7, 4.4))
colors = {"Joint-WAM": "#1f5fae", "Fast-WAM": "#b5472b", "Faster-WAM": "#3c8c5a"}
for m in rc.MODELS:
    for i, meth in enumerate(rc.METHODS):
        row = rc.LIBERO_PLUS[m][i]
        marker = "*" if meth == "w/ours" else "o"
        ax.scatter(row[8], row[7], color=colors[m], marker=marker, s=90 if marker == "*" else 35,
                   alpha=0.9 if marker == "*" else 0.5)
        if meth in ("Vanilla", "w/ours", "FlowCast"):
            ax.annotate(f"{m} {meth}", (row[8], row[7]), fontsize=7, xytext=(4, 3), textcoords="offset points")
ax.set_xlabel("지연 (ms, LIBERO-Plus 평가 로그)")
ax.set_ylabel("평균 성공률 (%)")
ax.set_title("LIBERO-Plus: 지연과 성공률 (별 = WAMotor, 원 = 기준선)", fontsize=10)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / "own_latency_success.png", dpi=170)
plt.close(fig)

# 2) 차이와 95% 검출 한계 (독립 근사)
tests = rec["difference_tests_independent_approx"]
keys = [k for k in tests if "vs Vanilla" in k or "vs BAC" in k or "vs FlowCast" in k]
fig, ax = plt.subplots(figsize=(7.4, 5.2))
ys = list(range(len(keys)))[::-1]
for y, k in zip(ys, keys):
    d = tests[k]["diff_pp"]
    mde = tests[k]["mde_95_pp"]
    c = "#1f5fae" if abs(d) >= mde else "#9a9a94"
    ax.plot([-mde, mde], [y, y], color="#dcdad3", lw=6, solid_capstyle="butt")
    ax.scatter([d], [y], color=c, s=40, zorder=3)
ax.axvline(0, color="#333", lw=0.8)
ax.set_yticks(ys)
ax.set_yticklabels([k.replace(" ours", " 차이: ours") for k in keys], fontsize=7)
ax.set_xlabel("성공률 차이 (percentage point, 회색 띠 = 95% 검출 한계)")
ax.set_title("WAMotor 차이와 검출 한계 (시행 독립 근사, 과제 군집 무시)", fontsize=10)
ax.grid(alpha=0.25, axis="x")
fig.tight_layout()
fig.savefig(FIG / "own_diff_mde.png", dpi=170)
plt.close(fig)

# 3) 토이: 오차 대 모델 호출 수 (Pareto 비교)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for mode, c, lab in [("uniform", "#9a9a94", "균일 MSE 임계"), ("tsav", "#1f5fae", "TSAV(시간 간격 가중)")]:
    pts = sorted([(r["nfe_mean"], r["final_err_mean"], r["eps"]) for r in toy["runs"] if r["mode"] == mode])
    ax.plot([p[0] for p in pts], [max(p[1], 1e-6) for p in pts], "o-", color=c, label=lab)
    for x, y, e in pts:
        ax.annotate(f"{e:g}", (x, max(y, 1e-6)), fontsize=6.5, xytext=(3, 2), textcoords="offset points")
ax.set_yscale("log")
ax.set_xlabel("평균 모델 호출 수 (Euler 기준 10)")
ax.set_ylabel("최종 상태 오차 (평균, 로그)")
ax.set_title("토이 흐름 모델: 호출 수 대 오차 (ε 값 표시, 20 시드)", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / "own_toy_frontier.png", dpi=170)
plt.close(fig)
print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
