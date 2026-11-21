"""own plots (not paper figures). Data: paper Table 1 and Table A2 (typed), Wilson 95% intervals.
Outputs: report/figures/own_frontier.png, report/figures/own_share_noise.png"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(os.path.dirname(HERE), "report", "figures")
for cand in ["/System/Library/Fonts/AppleSDGothicNeo.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    if os.path.exists(cand):
        font_manager.fontManager.addfont(cand)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

def wilson(k, n=100, z=1.96):
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d
    h = z*((p*(1-p)/n + z*z/(4*n*n))**0.5)/d
    return 100*(c-h), 100*(c+h)

# Table 1 latency (ms, mean) and success (k of 100) for uncached schedules, Optional-IDM s=1
pts = [("1 step", 81, 88), ("2 steps", 94, 95), ("4 steps", 119, 95), ("10 steps", 202, 95)]
cache = [("cache tau=2 (10 steps, 8 cached)", 158, 97), ("cache tau=8 (10 steps, 8 cached)", 116, 63),
         ("attenuated tau=8", 118, 81), ("cache tau=4 (4 steps, 2 cached)", 104, 96)]
fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=160)
xs = [p[1] for p in pts]; ys = [p[2] for p in pts]
ax.plot(xs, ys, "-o", color="#1f5fae", label="uncached (IDM, s=1)")
for name, x, k in pts:
    lo, hi = wilson(k)
    ax.errorbar(x, k, yerr=[[k-lo], [hi-k]], fmt="none", ecolor="#1f5fae", alpha=.6, capsize=3)
    ax.annotate(name, (x, k), textcoords="offset points", xytext=(6, -12), fontsize=8)
for name, x, k in cache:
    lo, hi = wilson(k)
    ax.errorbar(x, k, yerr=[[k-lo], [hi-k]], fmt="s", color="#c4572a", capsize=3, label=None)
    ax.annotate(name, (x, k), textcoords="offset points", xytext=(6, 4), fontsize=7, color="#c4572a")
ax.set_xlabel("평균 재계획 호출 시간 (ms, RTX A6000, 기본 컴파일)")
ax.set_ylabel("성공률 (%) · 100 LIBERO-Goal 상태")
ax.set_ylim(55, 102); ax.set_xlim(70, 225)
ax.grid(alpha=.25)
ax.set_title("Table 1 재계산: 2단계 무캐시가 캐시 대비 우위", fontsize=10)
fig.text(0.01, 0.01, "출처: arXiv:2611.19709 Table 1. 구간은 Wilson 95%. 그림은 우리가 그림.", fontsize=6.5, color="#55544f")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(os.path.join(FIG, "own_frontier.png"))

# Table A2: success vs cached share at s in {1,3,5}, fixed vs fresh noise (50 states)
fixed = [(0, 96), (30, 94), (40, 94), (50, 98), (60, 96), (70, 88), (80, 54)]
fresh = [(40, 96), (60, 82), (70, 62), (80, 26)]
fixed_orig = [(0, 94), (50, 94), (70, 90), (80, 72)]
fresh_s3 = [(75, 50)]
fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=160)
ax.plot([a for a, _ in fixed], [b for _, b in fixed], "-o", color="#1f5fae", label="IDM s=1, 고정 잠재 노이즈")
ax.plot([a for a, _ in fresh], [b for _, b in fresh], "--x", color="#c4572a", label="IDM s=1, 매 호출 노이즈 재샘플")
ax.plot([a for a, _ in fixed_orig], [b for _, b in fixed_orig], "-s", color="#55544f", alpha=.7, label="원본 체크포인트 s=1, 고정")
ax.scatter([75], [50], marker="^", color="#c4572a", label="IDM s=3, 75% (재샘플)")
ax.axvspan(60, 100, color="#f2d7cb", alpha=.35, lw=0)
ax.set_xlabel("캐시된 흐름 경로 비율 (%) = 1 - sigma_m")
ax.set_ylabel("성공률 (%) · 같은 50개 상태")
ax.set_ylim(0, 105); ax.set_xlim(-3, 88); ax.grid(alpha=.25)
ax.legend(fontsize=7, loc="lower left")
ax.set_title("캐시 비율과 노이즈 재샘플의 효과 (Table A2 재배치)", fontsize=10)
fig.text(0.01, 0.01, "출처: arXiv:2611.19709 Table A2. 비율은 식 (1)로 우리가 다시 계산. 그림은 우리가 그림.", fontsize=6.5, color="#55544f")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(os.path.join(FIG, "own_share_noise.png"))
print("ok")
