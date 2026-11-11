"""추가 검증 2 (stiffness_copilot): 라벨 추정기의 민감도 지도. 실행: python3 analysis/extra_label_sens.py

label_toy.py의 식 (1)–(2) 라벨(섹터별 f̃f̃ᵀ+εI의 로그 평균 → exp)을 그대로 쓰고, 세 하이퍼파라미터를 바꾼다.
  m: 섹터 수 {50, 100, 200, 400, 800}   (기준 m = 200)
  ε: 정칙화 {1e-3, 1e-2, 1e-1, 1, 10}     (기준 ε = 0.1)
  크기 통계: 섹터 내 95퍼센타일 대 최댓값
입력 접촉력은 접선 쪽으로 치우친 분포(label_toy의 skew=1)이고, 시드 5개로 반복한다.
측정: (a) 방향 안정성 = 주축 방향이 기준 설정(m=200, ε=0.1, 95퍼센타일)에서 벗어난 각(같은 시드 안에서 비교),
      (b) 이방성 비율 λmax/λmin, (c) 크기 = sqrt(det K)의 기준 대비 비.
질문: 방향은 안정하고 크기만 민감한가.
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import label_toy as lt  # noqa: E402

MS = [50, 100, 200, 400, 800]
EPSS = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
STATS = ["p95", "max"]
REF = (200, 0.1, "p95")
SEEDS = range(5)
N_FORCE = 4000


def reps_for(F, centers, stat):
    mags = np.linalg.norm(F, axis=1)
    dirs = F / mags[:, None]
    idx = np.argmax(dirs @ centers.T, axis=1)
    reps = []
    for j in range(len(centers)):
        sel = idx == j
        if not sel.any():
            continue
        mbar = dirs[sel].mean(0)
        u = mbar / np.linalg.norm(mbar)
        mag = np.percentile(mags[sel], 95) if stat == "p95" else mags[sel].max()
        reps.append((u, mag))
    return reps


def label(F, m, eps, stat):
    K, _ = lt.paper_label(reps_for(F, lt.fib_sphere(m), stat), eps)
    w, V = np.linalg.eigh((K + K.T) / 2)
    top = V[:, np.argmax(w)]
    return K, top, float(w.max() / w.min()), float(np.sqrt(np.linalg.det(K)))


def angle_deg(a, b):
    return float(np.degrees(np.arccos(min(1.0, abs(float(a @ b))))))


def main():
    per_cfg = {}
    ref_dirs, ref_scale = {}, {}
    for seed in SEEDS:
        lt.rng = np.random.default_rng(100 + seed)
        F = lt.sample_forces(N_FORCE, 1)
        _, top_ref, _, s_ref = label(F, REF[0], REF[1], REF[2])
        ref_dirs[seed], ref_scale[seed] = top_ref, s_ref
    # 시드별 설정 측정
    grid = {}
    for stat in STATS:
        for m in MS:
            for eps in EPSS:
                angs, anis, scl = [], [], []
                for seed in SEEDS:
                    lt.rng = np.random.default_rng(100 + seed)
                    F = lt.sample_forces(N_FORCE, 1)
                    _, top, an, sc = label(F, m, eps, stat)
                    angs.append(angle_deg(top, ref_dirs[seed]))
                    anis.append(an)
                    scl.append(sc / ref_scale[seed])
                key = f"{stat}|m={m}|eps={eps:g}"
                grid[key] = {"dir_dev_deg_mean": round(float(np.mean(angs)), 2),
                             "dir_dev_deg_max": round(float(np.max(angs)), 2),
                             "aniso_mean": round(float(np.mean(anis)), 2),
                             "scale_ratio_sqrtdet_mean": round(float(np.mean(scl)), 3),
                             "scale_ratio_sqrtdet_min": round(float(np.min(scl)), 3),
                             "scale_ratio_sqrtdet_max": round(float(np.max(scl)), 3)}
                per_cfg[key] = grid[key]
                print(key, grid[key], flush=True)

    # 요약: 방향 편차가 크기 민감도와 비교해 얼마나 작은가
    dev_all = [v["dir_dev_deg_max"] for v in grid.values()]
    scale_span = [v["scale_ratio_sqrtdet_max"] / v["scale_ratio_sqrtdet_min"] for v in grid.values()]
    aniso_span = [v["aniso_mean"] for k, v in grid.items() if k.startswith("p95")]
    summary = {
        "max_dir_dev_deg_over_grid": round(float(np.max(dev_all)), 2),
        "median_dir_dev_deg_over_grid": round(float(np.median([v["dir_dev_deg_mean"] for v in grid.values()])), 2),
        "scale_ratio_range_over_grid(max/min)": [round(float(np.min([v["scale_ratio_sqrtdet_min"] for v in grid.values()])), 3),
                                                 round(float(np.max([v["scale_ratio_sqrtdet_max"] for v in grid.values()])), 3)],
        "aniso_range_p95_over_eps": [round(float(min(aniso_span)), 2), round(float(max(aniso_span)), 2)],
        "reference": {"m": REF[0], "eps": REF[1], "stat": REF[2]},
    }
    # 방향 편차 설정별 큰 값을 찾는 조건 (ε 극단)
    out = {"setup": {"force_samples": N_FORCE, "seeds": list(SEEDS), "force_model": "label_toy.sample_forces(skew=1)",
                     "m_grid": MS, "eps_grid": EPSS, "stat_grid": STATS, "reference": summary["reference"]},
           "summary": summary, "grid": grid}
    path = ROOT / "analysis" / "extra_label_sens_results.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    plot(grid)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("wrote", path)


def plot(grid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                              "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
    if font:
        font_manager.fontManager.addfont(font)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({"font.size": 8.5, "axes.unicode_minus": False})
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.8))
    mats = {"dir_dev_deg_max": np.zeros((len(MS), len(EPSS))),
            "aniso_mean": np.zeros((len(MS), len(EPSS))),
            "scale_ratio_sqrtdet_mean": np.zeros((len(MS), len(EPSS)))}
    for i, m in enumerate(MS):
        for j, e in enumerate(EPSS):
            v = grid[f"p95|m={m}|eps={e:g}"]
            for k in mats:
                mats[k][i, j] = v[k]
    titles = {"dir_dev_deg_max": "주축 방향 편차 최댓값 [°]", "aniso_mean": "이방성 λmax/λmin",
              "scale_ratio_sqrtdet_mean": "크기 sqrt(det K) / 기준"}
    cmaps = {"dir_dev_deg_max": "Blues", "aniso_mean": "Oranges", "scale_ratio_sqrtdet_mean": "Greens"}
    for ax, k in zip(axes, mats):
        im = ax.imshow(mats[k], cmap=cmaps[k], origin="lower", aspect="auto")
        for i in range(len(MS)):
            for j in range(len(EPSS)):
                ax.text(j, i, f"{mats[k][i, j]:.2g}", ha="center", va="center", fontsize=6.2)
        ax.set_xticks(range(len(EPSS)))
        ax.set_xticklabels([f"{e:g}" for e in EPSS], fontsize=6.5)
        ax.set_yticks(range(len(MS)))
        ax.set_yticklabels([str(m) for m in MS], fontsize=6.5)
        ax.set_xlabel("ε")
        ax.set_title(titles[k], fontsize=7.5)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    axes[0].set_ylabel("섹터 수 m")
    fig.tight_layout()
    fig.savefig(ROOT / "report" / "figures" / "extra_label_sens.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
