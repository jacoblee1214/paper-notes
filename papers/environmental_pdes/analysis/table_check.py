"""논문 수치 재계산 (CPU, 즉시 끝남). 입력은 논문 본문·Table I의 요약 통계이고, 출처를 옆에 적었다.

가정: "평균 ± 값"의 값은 표준편차(SD), 독립 실행 n = 5 (Fig. 4·5 캡션 "five independent runs").
논문이 PF의 SD를 주지 않은 항목은 SD를 0.05 / 0.1 / 0.2 / 0.3로 바꿔 본 민감도를 쓴다.
실행: python3 analysis/table_check.py → analysis/table_check_results.json
"""
import json
from pathlib import Path

from scipy import stats

HERE = Path(__file__).resolve().parent
N = 5


def welch(m1, s1, m2, s2, n=N):
    t, p = stats.ttest_ind_from_stats(m1, s1, n, m2, s2, n, equal_var=False)
    return float(t), float(p)


def min_detectable_diff(sd, n=N, alpha=0.05, power=0.8):
    # Welch 근사: 두 집단 같은 SD, 자유도 2n−2
    df = 2 * n - 2
    return float((stats.t.ppf(1 - alpha / 2, df) + stats.t.ppf(power, df)) * sd * (2 / n) ** 0.5)


out = {}

# 1) SWE, 500 입자 (본문 V-A, "0.80 ± 0.03 vs 1.07 ± 0.04")
t, p = welch(0.80, 0.03, 1.07, 0.04)
out["swe_500_rmse_rbpf_vs_pf"] = dict(t=t, p=p, reduction_pct=round(100 * (1.07 - 0.80) / 1.07, 1),
                                     paper_says="roughly 25%")
# 1b) 최종 오차 (본문: 0.70 ± 0.15 vs 0.82 ± 0.02)
t, p = welch(0.70, 0.15, 0.82, 0.02)
out["swe_500_final_rbpf_vs_pf"] = dict(t=t, p=p)
# 1c) 100 입자 RBPF가 500 입자 PF보다 낫다는 주장 (0.84 ± 0.04 vs 1.07 ± 0.04)
t, p = welch(0.84, 0.04, 1.07, 0.04)
out["swe_rbpf100_vs_pf500"] = dict(t=t, p=p)

# 2) 대류–확산, 500 입자 (본문 V-B: RBPF 1.23 ± 0.35, PF 2.32; PF SD 미보고 → 민감도)
sens = {}
for sd_pf in (0.05, 0.1, 0.2, 0.3):
    t, p = welch(1.23, 0.35, 2.32, sd_pf)
    sens[f"sd_pf={sd_pf}"] = dict(t=round(t, 2), p=float(f"{p:.3g}"))
out["adv_500_rbpf_vs_pf_sensitivity"] = dict(reduction_pct=round(100 * (2.32 - 1.23) / 2.32, 1), by_pf_sd=sens)

# 3) 대류–확산, 400 입자: RBPF 3.1 ± 1.17 (본문 V-B), PF는 2.3–2.4 대역
sens = {}
for sd_pf in (0.05, 0.1, 0.2, 0.3):
    t, p = welch(3.1, 1.17, 2.35, sd_pf)
    sens[f"sd_pf={sd_pf}"] = dict(t=round(t, 2), p=round(p, 3))
out["adv_400_rbpf_worse_than_pf"] = dict(note="본문: RBPF 3.1 ± 1.17 대 PF ~2.35 (대역 2.3–2.4 중앙값 사용)",
                                       by_pf_sd=sens)

# 4) Table I (SWE, 대류–확산): M별 비교. 열: RMSE, 최종 오차
#    PDF 텍스트 추출이 깨진 SD는 None으로 두고 검정에서 뺀다. 대류–확산 M=1 RMSE의 SD 1.63은 본문 V-B("2.63 ± 1.63 m")에서 가져왔다.
#    깨진 값: "2.63 ± 1.6.38", "2.27 ± 1.2.06", "1.61 ± 0.3.9", "2.11 ± 0.132" (원문 PDF를 눈으로 봐도 판독 불가)
table1 = {
    "swe": {1: ((1.59, 0.55), (1.35, 0.49)), 2: ((0.94, 0.10), (0.90, 0.26)), 3: ((0.82, 0.06), (0.69, 0.08))},
    "adv": {1: ((2.63, 1.63), (2.11, None)), 2: ((2.27, None), (1.61, None)), 3: ((1.51, 0.79), (1.62, 0.64))},
}
tbl = {}
for task, rows in table1.items():
    for a, b in ((1, 3), (2, 3), (1, 2)):
        rec = {}
        for key, idx in (("rmse", 0), ("final", 1)):
            (ma, sa), (mb, sb) = rows[a][idx], rows[b][idx]
            if sa is None or sb is None:
                rec[f"{key}_p"] = "SD 판독 불가"
            else:
                _, pv = welch(ma, sa, mb, sb)
                rec[f"{key}_p"] = float(f"{pv:.3g}")
        rec["rmse_ratio"] = round(rows[a][0][0] / rows[b][0][0], 2)
        tbl[f"{task}_M{a}_vs_M{b}"] = rec
out["table1_feature_pairs"] = tbl
out["table1_swe_M3_vs_text_500particles"] = dict(
    note="Table I의 SWE M=3(0.82 ± 0.06)와 본문 500 입자 RBPF(0.80 ± 0.03). 입자 수와 실행 설정이 표에 없어 같은 조건인지 모른다.")
out["table1_adv_M3_vs_text_500particles"] = dict(
    note="Table I의 대류–확산 M=3(1.51 ± 0.79)와 Fig. 5·본문 500 입자 RBPF(1.23 ± 0.35)가 다르다. 입자 수 또는 실행 설정이 다른 것으로 보인다.")

# 5) 검출력: n = 5에서 SD별 최소 검출 차이 (α = 0.05, 검정력 0.8)
out["min_detectable_difference_n5"] = {f"sd={s}": round(min_detectable_diff(s), 3) for s in (0.03, 0.05, 0.1, 0.35)}

# 6) 본문 수치 검산
out["ratio_checks"] = dict(
    swe_rmse_100_to_500_RBPF_gain_pct=round(100 * (0.84 - 0.80) / 0.84, 1),
    table1_swe_M1_over_M3=round(1.59 / 0.82, 2),
    adv_rbpf_pf_500_ratio=round(1.23 / 2.32, 3),
)

(HERE / "table_check_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
for k, v in out.items():
    print(k, v)
