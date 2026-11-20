"""Own figures (not paper figures): LIBERO suite CIs and the leakage toy bars.
Reads analysis/recompute_results.json and analysis/toy_leakage_results.json; writes report/figures/own_*.png."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(os.path.dirname(HERE), "report", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": ["Apple SD Gothic Neo", "Noto Sans CJK KR", "DejaVu Sans"],
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})

rc = json.load(open(os.path.join(HERE, "recompute_results.json"), encoding="utf-8"))["table1_vla_jepa_wilson"]
names = ["Spatial", "Object", "Goal", "LIBERO-10", "Avg(2000 ep)"]
rates = [rc[n]["rate"] for n in names]
lo = [rc[n]["rate"] - rc[n]["wilson95"][0] for n in names]
hi = [rc[n]["wilson95"][1] - rc[n]["rate"] for n in names]
fig, ax = plt.subplots(figsize=(6.2, 3.0), dpi=200)
ax.errorbar(range(len(names)), rates, yerr=[lo, hi], fmt="o", color="#1f5fae", capsize=4)
ax.axhline(97.1, color="#888", lw=0.8, ls="--")
ax.text(0.5, 97.3, "점선: OpenVLA-OFT 평균 97.1", fontsize=7.5, color="#555", ha="center", va="bottom")
ax.set_xticks(range(len(names)), names)
ax.set_ylabel("성공률 (%)")
ax.set_ylim(88, 101)
ax.set_title("VLA-JEPA LIBERO 성공률과 Wilson 95% 구간 (재계산)", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "own_libero_ci.png"))

tz = json.load(open(os.path.join(HERE, "toy_leakage_results.json"), encoding="utf-8"))["r2_future_from_input"]
labels = ["현재 프레임만", "인과 인코더\n과거 특징", "양방향 인코더\n과거 특징"]
vals = [tz["current_frame_only"], tz["causal_history_features"], tz["bidi_history_features"]]
fig2, ax2 = plt.subplots(figsize=(6.2, 3.0), dpi=200)
bars = ax2.bar(labels, vals, color=["#bbb", "#7aa7d9", "#1f5fae"], width=0.55)
for b, v in zip(bars, vals):
    ax2.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
ax2.set_ylim(0, 1.05)
ax2.set_ylabel("미래 상태 예측 R² (테스트)")
ax2.set_title("누수 토이: 입력 특징이 담는 미래 정보 (토이, 실제 모델 아님)", fontsize=9.5)
fig2.tight_layout()
fig2.savefig(os.path.join(FIG, "own_leakage_toy.png"))
print("wrote", os.path.join(FIG, "own_libero_ci.png"), os.path.join(FIG, "own_leakage_toy.png"))
