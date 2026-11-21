"""우리가 만든 그래프 네 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_domain_curves.png : 도메인별 과제 평균 학습 곡선 (공식 저장소 결과 파일, 과제 단위 95% t 구간)
own_paired_final.png  : 과제별 최종 값 산점도 (TD-MPC2 대 기준선, 0–100 척도)
own_scaling.png       : 논문 표 1의 파라미터 대 점수와 1M–48M 직선 적합, 317M 예측
own_collapse.png      : 시드별 학습 곡선, TD-MPC의 붕괴 (Walker Stand, Walker Walk, Dog Walk, Humanoid Run)
실행: python3 -I analysis/plot_figures.py   (numpy, scipy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
"""
import csv
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from scipy import stats  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "repo" / "results"
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
GRID = 100_000
ALGS = ["tdmpc2", "tdmpc", "sac", "dreamerv3"]
LABEL = {"tdmpc2": "TD-MPC2", "tdmpc": "TD-MPC", "sac": "SAC", "dreamerv3": "DreamerV3"}
COL = {"tdmpc2": "#1f5fae", "tdmpc": "#c2561c", "sac": "#7a7a72", "dreamerv3": "#2e8b57"}
DOMAINS = ["DMControl", "Meta-World", "ManiSkill2", "MyoSuite"]
DOM_COL = {"DMControl": "#1f5fae", "Meta-World": "#c2561c", "ManiSkill2": "#7a7a72", "MyoSuite": "#2e8b57"}


def korean_font():
    for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 9
INK = "#1b1b1a"


def domain_of(task):
    if task.startswith("mw-"):
        return "Meta-World"
    if task.startswith("myo-"):
        return "MyoSuite"
    if task in ("pick-cube", "pick-ycb", "stack-cube", "lift-cube", "turn-faucet"):
        return "ManiSkill2"
    return "DMControl"


def scale_of(task):
    return 1.0 if domain_of(task) == "DMControl" else 100.0


def load(alg, task):
    out = {}
    with open(RES / alg / f"{task}.csv") as f:
        rd = csv.DictReader(f)
        vcol = [c for c in rd.fieldnames if c not in ("step", "seed")][0]
        for r in rd:
            v = float(r[vcol]) * (scale_of(task) if vcol == "success" else 1.0)
            out.setdefault(int(r["seed"]), {})[int(round(float(r["step"])))] = v
    return out


def value_at(d, s):
    ks = [k for k in d if k <= s]
    return d[max(ks)] if ks else np.nan


def tasks():
    return sorted(p.stem for p in (RES / "tdmpc2").glob("*.csv"))


def common_end(task):
    ends = []
    for a in ALGS:
        d = load(a, task)
        ends.append((max(max(v) for v in d.values()) // GRID) * GRID)
    return min(ends)


def plot_domain_curves():
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.1))
    for ax, dom in zip(axes, DOMAINS):
        ts = [t for t in tasks() if domain_of(t) == dom]
        ends = {t: common_end(t) for t in ts}
        x_end = min(ends.values())
        grid = np.arange(0, x_end + 1, GRID)
        for a in ALGS:
            per_task = []
            for t in ts:
                d = load(a, t)
                # 시드 평균 후 과제별 곡선
                m = np.array([np.nanmean([value_at(d[s], g) for s in d]) for g in grid])
                per_task.append(m)
            M = np.vstack(per_task)
            mean = M.mean(axis=0)
            n = M.shape[0]
            se = M.std(axis=0, ddof=1) / np.sqrt(n)
            tcrit = stats.t.ppf(0.975, df=n - 1)
            ax.plot(grid / 1e6, mean, color=COL[a], lw=1.6, label=LABEL[a])
            top = 1000.0 if dom == "DMControl" else 100.0
            lo, hi = np.clip(mean - tcrit * se, 0, top), np.clip(mean + tcrit * se, 0, top)
            ax.fill_between(grid / 1e6, lo, hi, color=COL[a], alpha=0.12, lw=0)
        ax.set_title(f"{dom} ({len(ts)}과제)", fontsize=9.5, color=INK)
        ax.set_xlabel("환경 단계 (백만)")
        ax.set_ylabel("DMControl 반환" if dom == "DMControl" else "성공률 (%)")
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle("도메인별 과제 평균 학습 곡선 (공식 결과 파일 재그림, 음영은 과제 단위 95% t 구간)", fontsize=9.5, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "own_domain_curves.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def final_value(alg, task):
    end = common_end(task)
    d = load(alg, task)
    return np.mean([value_at(d[s], end) for s in d])


def plot_paired_final():
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharex=True, sharey=True)
    ts = tasks()
    for ax, base in zip(axes, ["tdmpc", "sac", "dreamerv3"]):
        for dom, mk in zip(DOMAINS, ["o", "s", "^", "D"]):
            sel = [t for t in ts if domain_of(t) == dom]
            k = 0.1 if dom == "DMControl" else 1.0
            x = [final_value("tdmpc2", t) * k for t in sel]
            y = [final_value(base, t) * k for t in sel]
            ax.scatter(x, y, s=18, marker=mk, alpha=0.75, color=DOM_COL[dom], label=dom, edgecolor="white", lw=0.4)
        ax.plot([0, 100], [0, 100], color=INK, lw=0.7, ls="--")
        ax.set_xlim(-3, 103)
        ax.set_ylim(-3, 103)
        ax.set_title(f"TD-MPC2 대 {LABEL[base]}", fontsize=9.5)
        ax.set_xlabel("TD-MPC2 최종 (0–100 척도)")
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("기준선 최종 (0–100 척도)")
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.suptitle("과제별 최종 값 (DMControl은 1000점 만점을 100점으로 환산, 과제 평균 3시드)", fontsize=9.5, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "own_paired_final.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_scaling():
    params = np.array([1, 5, 19, 48, 317], dtype=float)
    score = np.array([16.0, 49.5, 57.1, 68.0, 70.6])
    x = np.log10(params)
    b1, b0 = np.polyfit(x[:4], score[:4], 1)
    xs = np.linspace(0, np.log10(400), 100)
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    ax.plot(10 ** xs, b0 + b1 * xs, ls="--", color=COL["sac"], lw=1.1, label="1M–48M 직선 적합 (4점)")
    ax.scatter(params[:4], score[:4], color=COL["tdmpc2"], s=28, zorder=3, label="표 1 값 (1M–48M)")
    ax.scatter(params[4:], score[4:], color="#c2561c", s=34, zorder=3, marker="s", label="표 1 값 (317M)")
    for p, s_ in zip(params, score):
        ax.annotate(f"{s_:.1f}", (p, s_), textcoords="offset points", xytext=(4, -10), fontsize=7.5)
    ax.annotate("317M 예측 약 94.7", (317, b0 + b1 * np.log10(317)), textcoords="offset points", xytext=(-95, 6), fontsize=7.5, color=COL["sac"])
    ax.set_xscale("log")
    ax.set_xlabel("파라미터 수 (M, 로그 척도)")
    ax.set_ylabel("정규화 점수 (80과제)")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.set_title("모델 크기와 점수 (논문 표 1 값, 재계산)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "own_scaling.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_collapse():
    picks = [("walker-stand", "Walker Stand"), ("walker-walk", "Walker Walk"),
             ("dog-walk", "Dog Walk"), ("humanoid-run", "Humanoid Run")]
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 2.9))
    for ax, (t, title) in zip(axes, picks):
        for a in ["tdmpc2", "tdmpc"]:
            d = load(a, t)
            for i, s in enumerate(sorted(d)):
                xs = sorted(k for k in d[s] if k % GRID == 0)
                ax.plot([k / 1e6 for k in xs], [d[s][k] for k in xs], color=COL[a],
                        alpha=0.85, lw=1.0, ls=["-", "--", ":"][i % 3],
                        label=LABEL[a] if i == 0 else None)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("환경 단계 (백만)")
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("에피소드 반환")
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.suptitle("시드별 학습 곡선 (선 모양은 시드 1, 2, 3)", fontsize=9.5, y=1.03)
    fig.tight_layout()
    fig.savefig(FIG / "own_collapse.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_domain_curves()
    plot_paired_final()
    plot_scaling()
    plot_collapse()
    print("saved", sorted(p.name for p in FIG.glob("own_*.png")))
