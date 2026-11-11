"""곡선에서 육안으로 읽은 값(fig_reading.json)으로 유의성 상한을 계산한다.

시드별 데이터가 없으므로 검정은 '읽은 평균과 띠 반폭(≈1 SD)'으로 한 근사다. 결과는 근사치로만 쓴다.
실행: python3 analysis/reading_stats.py → analysis/reading_stats_results.json
"""
import json
import math
from pathlib import Path

from scipy import stats

HERE = Path(__file__).resolve().parent
READ = json.loads((HERE / "fig_reading.json").read_text(encoding="utf-8"))
N = 5  # 논문 시드 수


def welch(m1, s1, m2, s2, n=N):
    res = stats.ttest_ind_from_stats(m1, s1, n, m2, s2, n, equal_var=False)
    se = math.sqrt(s1**2 / n + s2**2 / n)
    df = (s1**2 / n + s2**2 / n) ** 2 / ((s1**2 / n) ** 2 / (n - 1) + (s2**2 / n) ** 2 / (n - 1))
    return {"diff": m1 - m2, "se": se, "t": float(res.statistic), "df": df, "p": float(res.pvalue)}


def mde(sd, n=N, alpha=0.05, power=0.8):
    """두 독립 집단(각 n, SD 같음)에서 검정력 power로 찾을 최소 차이(근사, 정규 근사 아님: t 분포 사용)."""
    df = 2 * n - 2
    t_a = stats.t.ppf(1 - alpha / 2, df)
    t_b = stats.t.ppf(power, df)
    return (t_a + t_b) * sd * math.sqrt(2 / n)


def main():
    f3 = READ["fig3_pickcube_octo"]["final_success_at_1p25e7"]
    w3 = READ["fig3_pickcube_octo"]["band_halfwidth_final"]
    out = {
        "note": "Welch 근사, 띠 반폭을 SD로 취급. 실제 시드별 데이터가 필요하다.",
        "fig3_final_RPD-MSE_vs_PPO": welch(f3["RPD-MSE (Octo)"], w3["RPD-MSE (Octo)"], f3["PPO"], w3["PPO"]),
        "fig3_final_RPD-MSE_vs_PPO_mde_sd0.05": mde(0.05),
        "fig3_final_RPD-MSE_vs_PPO_mde_sd0.02": mde(0.02),
    }
    f7 = READ["fig7_pushcube_camera_sparse"]
    out["fig7_final_RPD-MSE-Octo_vs_PPO"] = welch(
        f7["final_success_at_1p5"]["RPD-MSE (Octo)"], f7["band_halfwidth_final"]["RPD-MSE (Octo)"],
        f7["final_success_at_1p5"]["PPO"], f7["band_halfwidth_final"]["PPO"])
    out["mde_table"] = {str(sd): {"n5": mde(sd, 5), "n10": mde(sd, 10)} for sd in [0.02, 0.05, 0.10]}
    (HERE / "reading_stats_results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    for k in ["fig3_final_RPD-MSE_vs_PPO", "fig7_final_RPD-MSE-Octo_vs_PPO"]:
        r = out[k]
        print(k, {kk: round(v, 4) for kk, v in r.items()})
    print("MDE (n=5, sd=0.05):", round(out["fig3_final_RPD-MSE_vs_PPO_mde_sd0.05"], 3))
    print("MDE table:", {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in out["mde_table"].items()})


if __name__ == "__main__":
    main()
