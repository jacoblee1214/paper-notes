"""추가 검증 1 (stiffness_copilot): 방향별 강성의 이득이 유지되는 오차 영역 지도. 실행: python3 analysis/extra_overshoot_map.py

contact_toy.py의 2D 임피던스 접촉 모델(질량 1 kg, 벽 강성 5000 N/m, 쿨롱 마찰 μ, 접선 목표 속도 0.05 m/s)을 그대로 쓴다.
격자: 과오버슛 δ ∈ {0.5, 1, 2, 3} cm × 법선 추정 오차 e ∈ {0, 15, 30, 45, 60}° × 마찰 μ ∈ {0, 0.2, 0.4, 0.6}.
비교 설계 세 가지: 고정 저강성(300/300), 고정 고강성(3000/3000), 방향별(법선 300 / 접선 3000).
허용 기준(사전 정의): 접선 추적 RMS ≤ 2 mm 이고 피크 접촉력 ≤ 20 N (contact_toy.F_LIMIT).
질문: 방향별 강성이 허용 기준을 만족하는 (δ, e, μ) 영역이 고정 저·고강성의 합집합보다 얼마나 넓은가, 어디서 깨지는가.
시드 5개, 시뮬레이션 격자 약 80점 × 설계 3개. 실행 약 1분.
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
import contact_toy as ct  # noqa: E402

DELTAS = [0.005, 0.01, 0.02, 0.03]
ERRS = [0, 15, 30, 45, 60]
MUS = [0.0, 0.2, 0.4, 0.6]
DESIGNS = {"고정 저강성 300/300": (300.0, 300.0), "고정 고강성 3000/3000": (3000.0, 3000.0),
           "방향별 300/3000": (300.0, 3000.0)}
TRACK_LIMIT_MM, FPEAK_LIMIT = 2.0, ct.F_LIMIT
SEEDS = 5


def main():
    rows = []
    for delta in DELTAS:
        for mu in MUS:
            for e in ERRS:
                kn = [v[0] for v in DESIGNS.values()]
                kt = [v[1] for v in DESIGNS.values()]
                out = ct.simulate(kn, kt, delta=delta, theta_err=np.deg2rad(e), seeds=SEEDS, mu=mu)
                row = {"delta_cm": delta * 100, "mu": mu, "err_deg": e}
                for i, name in enumerate(DESIGNS):
                    ok = bool(out["track_rms_mm_mean"][i] <= TRACK_LIMIT_MM and out["Fpeak_N_mean"][i] <= FPEAK_LIMIT)
                    row[name] = {"track_mm": round(float(out["track_rms_mm_mean"][i]), 2),
                                 "Fpeak_N": round(float(out["Fpeak_N_mean"][i]), 2),
                                 "Fss_N": round(float(out["Fss_N_mean"][i]), 2), "pass": ok}
                rows.append(row)

    names = list(DESIGNS)
    n = len(rows)
    count = {nm: sum(r[nm]["pass"] for r in rows) for nm in names}
    union_fixed = sum(1 for r in rows if r[names[0]]["pass"] or r[names[1]]["pass"])
    directional_only = [ {"delta_cm": r["delta_cm"], "mu": r["mu"], "err_deg": r["err_deg"]}
                         for r in rows if r[names[2]]["pass"] and not (r[names[0]]["pass"] or r[names[1]]["pass"])]
    dir_fail = [ {"delta_cm": r["delta_cm"], "mu": r["mu"], "err_deg": r["err_deg"],
                  "track_mm": r[names[2]]["track_mm"], "Fpeak_N": r[names[2]]["Fpeak_N"]}
                for r in rows if not r[names[2]]["pass"]]
    # μ=0에서 법선 오차 해석식(contact_toy E)과의 대조: 접촉력은 δ·K_eff·k/(K_eff+k)
    summary = {"grid_points": n, "pass_count": count,
               "pass_count_fixed_low_or_high": union_fixed,
               "directional_only_pass_count": len(directional_only),
               "directional_only_pass_points": directional_only,
               "directional_fail_points": dir_fail}
    out = {"setup": {"tracking_limit_mm": TRACK_LIMIT_MM, "F_peak_limit_N": FPEAK_LIMIT, "seeds": SEEDS,
                     "base_model": "contact_toy.simulate (m=1 kg, k_env=5000 N/m, v=0.05 m/s, damping 1.4√k)",
                     "grid": {"delta_cm": [d * 100 for d in DELTAS], "err_deg": ERRS, "mu": MUS}},
           "summary": summary, "rows": rows}
    path = ROOT / "analysis" / "extra_overshoot_map_results.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    plot(rows, names)
    print(json.dumps({k: v for k, v in summary.items() if k != "directional_only_pass_points"}, ensure_ascii=False, indent=1))
    print("wrote", path)


def plot(rows, names):
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
    # μ = 0.2 한 면만 그린다: 세 설계의 통과 여부(격자 칸)
    mu_show = 0.2
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), sharey=True)
    for ax, nm in zip(axes, names):
        grid = np.zeros((len(DELTAS), len(ERRS)))
        for r in rows:
            if abs(r["mu"] - mu_show) < 1e-9:
                i = [d * 100 for d in DELTAS].index(r["delta_cm"])
                j = ERRS.index(r["err_deg"])
                grid[i, j] = 1.0 if r[nm]["pass"] else 0.0
        ax.imshow(grid, cmap="Greens", vmin=0, vmax=1.3, origin="lower", aspect="auto")
        for i in range(len(DELTAS)):
            for j in range(len(ERRS)):
                ax.text(j, i, "통과" if grid[i, j] else "실패", ha="center", va="center", fontsize=6.5)
        ax.set_xticks(range(len(ERRS)))
        ax.set_xticklabels([f"{e}°" for e in ERRS], fontsize=7)
        ax.set_yticks(range(len(DELTAS)))
        ax.set_yticklabels([f"{d*100:g} cm" for d in DELTAS], fontsize=7)
        ax.set_title(nm, fontsize=8)
        ax.set_xlabel("법선 추정 오차 e")
    axes[0].set_ylabel("과오버슛 δ")
    fig.suptitle(f"허용 기준 통과 영역 (μ = {mu_show}; 추적 ≤ 2 mm, 피크 ≤ 20 N)", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(ROOT / "report" / "figures" / "extra_overshoot_map.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
