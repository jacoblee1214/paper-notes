"""우리가 만든 그래프 네 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_q1_designs.png : PPP 성공률 대 B (무작위 3집합, 직교 16, 전체)와 B 같고 고유 쌍 수 같은 설계들의 퍼짐
own_ft.png         : 희소 사전학습 뒤 파인튜닝 (B=4, B=6): 시연 수에 따른 사전학습 기준 OOD 성공률, 처음부터 학습과 비교
own_q3.png         : 48과제 2S-PP: Cov6–Cov16의 OOD/ID 성공률과 조건부 위반율, 오라클 단계 정책 24%, 우연 수준 25%
own_theory.png     : 식 5의 첫째+둘째 항(설계별 계산값)과 OOD 성공률
입력: data/page/* (analysis/fetch_page_data.py), analysis/recompute_results.json
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "data" / "page"
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def korean_font():
    for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
INK, ACC, GREY, ORG, GRN = "#1b1b1a", "#1f5fae", "#9a9890", "#c8643c", "#4a8f5c"


def rcsv(name):
    with open(PAGE / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e5e3dc", lw=0.6)
    ax.set_axisbelow(True)


def plot_q1():
    d = json.load(open(PAGE / "results_data.json", encoding="utf-8"))["experiments"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    rnd = defaultdict(list)
    for k, v in d.items():
        if k.startswith("random_coverage"):
            rnd[v["n_train"]].append(v["ood_rate"])
    Bs = sorted(rnd)
    for B in Bs:
        ax.scatter([B] * len(rnd[B]), rnd[B], color=GREY, s=18, zorder=2)
    ax.plot(Bs, [np.mean(rnd[B]) for B in Bs], color=ACC, lw=1.6, marker="o", ms=4, label="무작위 커버리지 3집합 평균")
    ax.scatter([16], [d["orthogonal"]["ood_rate"]], marker="*", s=140, color=ORG, zorder=3, label="직교 16")
    ax.axhline(d["full"]["id_rate"], color=GRN, lw=1, ls="--", label="전체 64 (전체 성공률)")
    ax.scatter([4], [d["diag_b4"]["ood_rate"]], color=INK, s=20, zorder=3, label="대각 4")
    ax.set_xlabel("학습 과제 수 B")
    ax.set_xticks([4, 6, 8, 10, 12, 16, 20, 24])
    ax.set_ylabel("OOD 성공률 (%)")
    ax.set_title("PPP: B에 따른 OOD 성공률 (설계별 점)", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    style(ax)
    ax = axes[1]
    # 같은 B, 같은 고유 쌍 수(모두 B개)인 설계: B=6,7,8,10
    groups = {6: [], 7: [], 8: [], 10: []}
    for k, v in d.items():
        if k == "full":
            continue
        tr = [t.rsplit("_press_", 1) for t in v["train_tasks"]]
        pr = [(a.split("_into_", 1)[0], a.split("_into_", 1)[1], b) for a, b in tr]
        B = v["n_train"]
        if B in groups and len({(p[0], p[1]) for p in pr}) == B and len({(p[0], p[2]) for p in pr}) == B and len({(p[1], p[2]) for p in pr}) == B:
            groups[B].append((k, v["ood_rate"]))
    for i, (B, items) in enumerate(groups.items()):
        ys = [y for _, y in items]
        ax.scatter([i] * len(ys), ys, color=ORG, s=22, zorder=2)
        ax.plot([i, i], [min(ys), max(ys)], color=ORG, lw=1)
        ax.text(i, max(ys) + 2.5, f"{max(ys) - min(ys):.0f}p", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"B={B}\n({len(v)}개)" for B, v in groups.items()], fontsize=8)
    ax.set_ylabel("OOD 성공률 (%)")
    ax.set_ylim(0, 82)
    ax.set_title("쌍이 모두 고유한(최대 쌍 커버리지) 설계끼리의 퍼짐", fontsize=10)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "own_q1_designs.png", dpi=170)
    plt.close(fig)


def plot_ft():
    rows = rcsv("q2_data_ft_success_vs_demos.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), sharey=True)
    for ax, key, title in ((axes[0], "diag_B4", "사전학습 B=4 (대각)"), (axes[1], "diag_03", "사전학습 B=6")):
        pre = [r for r in rows if r["pretrain"] == key and r["group"] == "pretrain (no FT)"][0]
        for group, color, label in (("full", ACC, "전체 64과제로 파인튜닝"), ("orth", ORG, "직교 16으로 파인튜닝")):
            pts = [r for r in rows if r["pretrain"] == key and r["group"] == group]
            xs = [0] + [int(r["n_demos"]) for r in pts]
            ys = [float(pre["ood_rate"])] + [float(r["ood_rate"]) for r in pts]
            es = [0] + [float(r["ood_se"]) for r in pts]
            ax.errorbar(xs, ys, yerr=es, color=color, marker="o", ms=4, lw=1.4, capsize=2, label=label)
        sc = [r for r in rows if r["pretrain"] == "scratch_vs_" + key]
        ax.errorbar([int(r["n_demos"]) for r in sc], [float(r["ood_rate"]) for r in sc], yerr=[float(r["ood_se"]) for r in sc],
                    color=GREY, marker="s", ms=4, lw=1.2, capsize=2, label="처음부터 학습 (같은 시연 수)")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("파인튜닝 시연 수")
        style(ax)
    axes[0].set_ylabel("OOD 성공률 (사전학습 기준, %)")
    axes[0].legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.text(0.5, -0.02, "OOD = 사전학습에 없던 과제. 전체 64과제 파인튜닝에서는 이 과제들이 파인튜닝 데이터(과제당 1회 이상)에 들어 있다. 막대 = 시드 SE (시드 3개).",
             ha="center", fontsize=7.5, color="#55544f")
    fig.tight_layout()
    fig.savefig(FIG / "own_ft.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_q3():
    rows = rcsv("q3_48tasks_data_l4_2s_no_cont0_runs.csv")
    cond = [("cov6", "Cov6"), ("cov9", "Cov9"), ("cov12", "Cov12"), ("orthogonal", "Cov16")]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    ax = axes[0]
    w = 0.36
    for j, (split, color, label) in enumerate((("id", GRN, "ID"), ("ood", ORG, "OOD"))):
        for i, (c, _) in enumerate(cond):
            v = [float(r["pc_success"]) for r in rows if r["condition"] == c and r["split"] == split]
            ax.bar(i + (j - 0.5) * w, np.mean(v), w, color=color, alpha=0.85, label=label if i == 0 else None)
            ax.errorbar(i + (j - 0.5) * w, np.mean(v), yerr=np.std(v, ddof=1) / np.sqrt(len(v)), color=INK, capsize=2, lw=1)
            ax.scatter([i + (j - 0.5) * w] * len(v), v, color=INK, s=5, zorder=3)
    full = [float(r["pc_success"]) for r in rows if r["condition"] == "full" and r["split"] == "id"]
    ax.axhline(np.mean(full), color=ACC, ls="--", lw=1, label="Full (ID)")
    ax.set_xticks(range(4))
    ax.set_xticklabels([c for _, c in cond])
    ax.set_ylabel("성공률 (%)")
    ax.set_title("2S-PP (48과제, 시드 5): 성공률, 점 = 시드", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    style(ax)
    ax = axes[1]
    for j, (split, color, label) in enumerate((("id", GRN, "ID"), ("ood", ORG, "OOD"))):
        for i, (c, _) in enumerate(cond):
            v = [float(r["pc_same_container_given_place_obj0"]) for r in rows if r["condition"] == c and r["split"] == split]
            ax.bar(i + (j - 0.5) * w, np.mean(v), w, color=color, alpha=0.85)
            ax.errorbar(i + (j - 0.5) * w, np.mean(v), yerr=np.std(v, ddof=1) / np.sqrt(len(v)), color=INK, capsize=2, lw=1)
    ax.axhline(25, color=GREY, ls=":", lw=1.2)
    ax.text(3.45, 25.6, "c2를 보지 않을 때의 기댓값 25%", ha="right", fontsize=8, color="#55544f")
    ax.axhline(24.4, color=ACC, ls="--", lw=1)
    ax.text(3.45, 21.2, "오라클 단계 정책(활성 지시만): 24.1–24.5%", ha="right", fontsize=8, color=ACC)
    ax.set_xticks(range(4))
    ax.set_xticklabels([c for _, c in cond])
    ax.set_ylabel("같은 컨테이너 위반율 (%)")
    ax.set_ylim(0, 30)
    ax.set_title("조건부 위반율 (논문 정의: 첫 배치가 일어난 롤아웃 중)", fontsize=10)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "own_q3.png", dpi=170)
    plt.close(fig)


def plot_theory():
    rec = json.load(open(ROOT / "analysis" / "recompute_results.json", encoding="utf-8"))
    d = json.load(open(PAGE / "results_data.json", encoding="utf-8"))
    import itertools
    from collections import defaultdict as dd

    def parse(t):
        l, c = t.rsplit("_press_", 1)
        o, r = l.split("_into_", 1)
        return o, r, c

    allt = [parse(t) for t in d["meta"]["all_tasks"]]
    lv = [sorted({t[i] for t in allt}) for i in range(3)]

    def terms(train):
        tr = [parse(t) for t in train]
        out = []
        for i in range(3):
            t1 = t2 = 0.0
            for v in lv[i]:
                pv = sum(1 for t in tr if t[i] == v) / len(tr)
                t1 += abs(1 / 4 - pv)
                if pv > 0:
                    pc = dd(float)
                    for t in tr:
                        if t[i] == v:
                            pc[tuple(x for j, x in enumerate(t) if j != i)] += 1 / (pv * len(tr))
                    t2 += pv * sum(abs(pc.get(k, 0) - 1 / 16) for k in itertools.product(*[lv[j] for j in range(3) if j != i]))
            out.append(t1 + t2)
        return float(np.mean(out))

    xs, ys, Bs = [], [], []
    for k, v in d["experiments"].items():
        if k == "full" or v["ood_rate"] is None:
            continue
        xs.append(terms(v["train_tasks"]))
        ys.append(v["ood_rate"])
        Bs.append(v["n_train"])
    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    sc = ax.scatter(xs, ys, c=Bs, cmap="viridis", s=26, zorder=2)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("B")
    ax.set_xlabel("식 5의 첫째+둘째 항 (인자 평균, 계산값)")
    ax.set_ylabel("OOD 성공률 (%)")
    ax.set_title("PPP 설계 52개: 식 5의 항 대 OOD 성공률", fontsize=10)
    ax.invert_xaxis()
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "own_theory.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    plot_q1()
    plot_ft()
    plot_q3()
    plot_theory()
    print("wrote 4 figures to", FIG)
