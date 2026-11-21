"""FTP-1 (arXiv:2606.13102 v4) 논문 표 수치를 다시 계산하고 통계 검정을 한다.

입력: 논문 본문과 표의 값을 아래에 그대로 옮겨 적었다(출처는 주석). 외부 데이터는 쓰지 않는다.
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)

가정(논문에 없는 것은 표시한다)
- 표 1(시뮬)은 과제당 100회 롤아웃(공식 eval 기본값 TOTAL_NUM=100과 일치), 표 2–3(실로봇)은 과제당 20회다.
  퍼센트 값은 시도 수로 나누면 정수 성공 수가 되므로, 성공 수 = 퍼센트 × 시도 수로 복원한다.
- 표 1의 베이스라인 중 UniVTAC-ACT*, VITaL*은 원 벤치마크 값이라 시도 수가 100이라고 가정한다(논문에 명시 없음).
- 표 4(matched VLA)는 과제별 값이 없어 전체 600회 가정의 근사 계산이다.
- "Avg" 열은 과제 평균으로 가정한다(표 2 Average는 6과제 단순 평균인지 확인한다).
- 과제별 계층화 검정은 Cochran–Mantel–Haenszel(CMH, 연속성 보정 없음)을 쓴다. 과제마다 난이도가 다르므로 풀링 Fisher보다 적합하다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
RES = {}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def newcombe_diff(k1, n1, k2, n2, z=1.959964):
    """Newcombe (1998) 방법 10: 두 비율 차이의 Wilson 기반 구간 (k1/n1 - k2/n2)."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1, z)
    l2, u2 = wilson(k2, n2, z)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lo, hi


def cmh(tables):
    """층화 2x2 표 목록 [(a, n1, c, n2), ...]에 대한 CMH 검정 (연속성 보정 없음).
    a = 방법1 성공, n1 = 방법1 시도, c = 방법2 성공, n2 = 방법2 시도.
    반환: (chi2, p값, 층화 Mantel-Haenszel 공통 오즈비)."""
    sum_a = sum_e = sum_v = 0.0
    num_or = den_or = 0.0
    for a, n1, c, n2 in tables:
        b, d = n1 - a, n2 - c
        n = n1 + n2
        m1 = a + c  # 전체 성공
        m0 = n - m1
        e = n1 * m1 / n
        v = n1 * n2 * m1 * m0 / (n * n * (n - 1))
        sum_a += a
        sum_e += e
        sum_v += v
        num_or += a * d / n
        den_or += b * c / n
    chi2 = (abs(sum_a - sum_e)) ** 2 / sum_v
    p = 1 - stats.chi2.cdf(chi2, 1)
    orr = num_or / den_or if den_or > 0 else float("inf")
    return chi2, p, orr


def fisher(k1, n1, k2, n2):
    t = [[k1, n1 - k1], [k2, n2 - k2]]
    return stats.fisher_exact(t)[1]


def fmt_p(p):
    return "<1e-4" if p < 1e-4 else f"{p:.4f}"


def compare(name, ftp, base, n_per_task, label_base, label_a="FTP-1"):
    """두 방법 비교(a 대 b): 풀링 Fisher, CMH(과제 층화), 차이의 Newcombe 구간."""
    k1 = sum(ftp)
    k2 = sum(base)
    n = len(ftp) * n_per_task
    d, lo, hi = newcombe_diff(k1, n, k2, n)
    chi2, p_cmh, orr = cmh([(f, n_per_task, b, n_per_task) for f, b in zip(ftp, base)])
    out = {
        "comparison": f"{name}: {label_a} vs {label_base}",
        "ftp_successes": k1,
        "base_successes": k2,
        "trials_per_arm": n,
        "ftp_rate": round(k1 / n, 4),
        "base_rate": round(k2 / n, 4),
        "diff_pp": round(100 * d, 2),
        "diff_newcombe_95ci_pp": [round(100 * lo, 2), round(100 * hi, 2)],
        "p_fisher_pooled": float(fisher(k1, n, k2, n)),
        "cmh_chi2": round(chi2, 3),
        "p_cmh_stratified_by_task": float(p_cmh),
        "cmh_common_or": round(orr, 3),
    }
    print(
        f"{out['comparison']:<30} {k1}/{n} vs {k2}/{n}  diff {out['diff_pp']:+.1f}pp "
        f"CI[{out['diff_newcombe_95ci_pp'][0]:+.1f},{out['diff_newcombe_95ci_pp'][1]:+.1f}]  "
        f"p(Fisher)={fmt_p(out['p_fisher_pooled'])}  p(CMH)={fmt_p(out['p_cmh_stratified_by_task'])}  [{label_base}]"
    )
    return out


# ---------------------------------------------------------------------------
# 표 1: UniVTAC 시뮬레이션 (과제당 100회). 열 순서: Lift Bottle, Pull-out Key, Lift Can,
# Put Bottle, Insert Hole, Insert Tube
T1 = {
    "VITaL*": [72, 47, 8, 32, 25, 34],
    "UniVTAC-ACT*": [71, 46, 29, 31, 25, 56],
    "pi0.5": [97, 38, 72, 16, 31, 41],
    "Tactile-VLA": [97, 32, 15, 10, 41, 56],
    "FTP-pi0.5": [77, 30, 26, 19, 47, 72],
    "FTP-1": [97, 48, 65, 47, 64, 79],
}
T1_REPORTED_AVG = {"VITaL*": (36.33, 34.5), "UniVTAC-ACT*": (43.00, 39.5), "pi0.5": (49.16, 31.5),
                   "Tactile-VLA": (41.83, 34.75), "FTP-pi0.5": (45.16, 42.0), "FTP-1": (66.66, 59.5)}

# 표 10 (NTP-1, 같은 UniVTAC 열 순서)
T10_NTP1 = [88, 38, 66, 32, 31, 45]

# 표 2: 실로봇 seen (과제당 20회). 열: Draw Balloon, Fix Hand (Tear), Fix Hand (Finish),
# Twist Cap, Flip Book, Wipe Dish
T2 = {
    "pi0.5": [35, 70, 35, 40, 65, 30],
    "Tactile-VLA": [20, 80, 25, 10, 45, 35],
    "FTP-pi0.5": [25, 65, 25, 20, 70, 45],
    "FTP-1": [45, 80, 40, 65, 85, 60],
}
# 표 2 값은 % 이고 과제당 20회이므로 성공 수 = % / 5 로 복원한다.
T2_COUNTS = {k: [x // 5 for x in v] for k, v in T2.items()}
T2_REPORTED_AVG = {"pi0.5": 45.3, "Tactile-VLA": 35.8, "FTP-pi0.5": 41.6, "FTP-1": 62.5}

# 표 3: 비가시 센서 (과제당 20회). 열: Insert Hanoi (Flexiv Xense), Insert USB (Flexiv Xense),
# Wipe Board (TactileUMI)
T3 = {
    "pi0.5": [5, 0, 4],
    "Tactile-VLA": [0, 2, 3],
    "FTP-pi0.5": [1, 2, 6],
    "FTP-1": [11, 6, 11],
}
T3_REPORTED_AVG = {"pi0.5": 15.0, "Tactile-VLA": 8.3, "FTP-pi0.5": 15.0, "FTP-1": 46.6}

# 표 4 (matched VLA pretraining, UniVTAC 전체 평균만 보고됨)
T4_REPORTED = {"pi0.5 matched": 57.0, "FTP-1": 66.7}


def main():
    print("== 1. 평균 재계산 (보고값과 비교) ==")
    RES["table1_means"] = {}
    for m, v in T1.items():
        all6 = sum(v) / 6
        wo = (v[1] + v[3] + v[4] + v[5]) / 4  # Lift Bottle(0), Lift Can(2) 제외
        rep_all, rep_wo = T1_REPORTED_AVG[m]
        RES["table1_means"][m] = {
            "avg_all_recomputed": round(all6, 2), "avg_all_reported": rep_all,
            "avg_wo_lift_recomputed": round(wo, 2), "avg_wo_lift_reported": rep_wo,
        }
        print(f"T1 {m:<14} all {all6:6.2f} (rep {rep_all:6.2f})   w/o lift {wo:6.2f} (rep {rep_wo:6.2f})")

    print("\n표 10 NTP-1: all", round(sum(T10_NTP1) / 6, 2), "(rep 50.00), w/o lift", round((T10_NTP1[1] + T10_NTP1[3] + T10_NTP1[4] + T10_NTP1[5]) / 4, 2), "(rep 36.5)")
    RES["table10_NTP1_means"] = {"all": round(sum(T10_NTP1) / 6, 2), "wo_lift": round((38 + 32 + 31 + 45) / 4, 2)}

    print("\n== 2. 실로봇 seen 평균 (표 2) ==")
    RES["table2_means"] = {}
    for m, v in T2.items():
        mean = sum(v) / 6  # 표 2 값은 이미 %
        rep = T2_REPORTED_AVG[m]
        RES["table2_means"][m] = {"recomputed": round(mean, 3), "reported": rep, "successes_of_120": sum(T2_COUNTS[m])}
        flag = "" if abs(round(mean, 1) - rep) < 0.05 else "  <-- 불일치"
        print(f"T2 {m:<12} recomputed {mean:6.2f}  reported {rep:5.1f}  ({sum(T2_COUNTS[m])}/120){flag}")
    print("  (각 값은 20회의 배수이므로 6과제 평균은 5/6 단위여야 한다. 45.3은 가능한 값이 아니다: 45.0 또는 45.83)")

    print("\n== 3. 비가시 센서 평균 (표 3) ==")
    RES["table3_means"] = {}
    for m, v in T3.items():
        mean = sum(v) / 3 * 5  # 1회 = 5%
        rep = T3_REPORTED_AVG[m]
        RES["table3_means"][m] = {"recomputed": round(mean, 3), "reported": rep, "successes_of_60": sum(v)}
        print(f"T3 {m:<12} recomputed {mean:6.2f}  reported {rep:5.1f}  ({sum(v)}/60)")

    print("\n== 4. 비교 검정 ==")
    RES["tests"] = []
    # 표 1: 과제당 100회
    for base in ["pi0.5", "FTP-pi0.5", "Tactile-VLA"]:
        RES["tests"].append(compare("T1 sim (100/task)", T1["FTP-1"], T1[base], 100, base))
    # 표 1 non-lift 4과제만 (Pull-out Key, Put Bottle, Insert Hole, Insert Tube)
    idx = [1, 3, 4, 5]
    RES["tests"].append(compare("T1 sim non-lift (100/task)", [T1["FTP-1"][i] for i in idx],
                                [T1["pi0.5"][i] for i in idx], 100, "pi0.5"))
    RES["tests"].append(compare("T1 sim non-lift (100/task)", [T1["FTP-1"][i] for i in idx],
                                [T1["FTP-pi0.5"][i] for i in idx], 100, "FTP-pi0.5"))
    # 논문 본문 'NTP-1' 비교(표 10, 같은 과제 6개)
    RES["tests"].append(compare("T10 sim (100/task)", T1["FTP-1"], T10_NTP1, 100, "NTP-1"))
    RES["tests"].append(compare("T10 sim non-lift (100/task)", [T1["FTP-1"][i] for i in idx],
                                [T10_NTP1[i] for i in idx], 100, "NTP-1"))
    # 표 2: 과제당 20회
    for base in ["pi0.5", "FTP-pi0.5", "Tactile-VLA"]:
        RES["tests"].append(compare("T2 real seen (20/task)", T2_COUNTS["FTP-1"], T2_COUNTS[base], 20, base))
    # 표 2 내부 비교: 촉각 입력이 해치는가 (논문 본문 주장)
    RES["tests"].append(compare("T2 real seen (20/task)", T2_COUNTS["FTP-pi0.5"], T2_COUNTS["pi0.5"], 20, "pi0.5", "FTP-pi0.5"))
    RES["tests"].append(compare("T2 real seen (20/task)", T2_COUNTS["Tactile-VLA"], T2_COUNTS["pi0.5"], 20, "pi0.5", "Tactile-VLA"))
    # 표 10 내부 비교: NTP-1 대 FTP-pi0.5, 비-lift 4과제
    RES["tests"].append(compare("T10 sim non-lift (100/task)", [T10_NTP1[i] for i in idx],
                                [T1["FTP-pi0.5"][i] for i in idx], 100, "FTP-pi0.5", "NTP-1"))
    RES["tests"].append(compare("T10 sim (100/task)", T10_NTP1, T1["FTP-pi0.5"], 100, "FTP-pi0.5", "NTP-1"))
    # 표 3: 과제당 20회
    for base in ["pi0.5", "FTP-pi0.5", "Tactile-VLA"]:
        RES["tests"].append(compare("T3 real unseen (20/task)", T3["FTP-1"], T3[base], 20, base))

    print("\n== 5. 표 4 (matched VLA, 전체 600회 가정, 과제별 값 없음: 근사) ==")
    k_m = round(0.570 * 600)
    k_f = round(0.667 * 600)
    d, lo, hi = newcombe_diff(k_f, 600, k_m, 600)
    print(f"  FTP-1 {k_f}/600 vs matched {k_m}/600  diff {100*d:+.1f}pp CI[{100*lo:+.1f},{100*hi:+.1f}] (근사)")
    RES["table4_approx"] = {"ftp1_600": k_f, "matched_600": k_m, "diff_pp": round(100 * d, 2),
                            "ci_pp": [round(100 * lo, 2), round(100 * hi, 2)], "note": "과제별 값 없음, 전체 600 가정"}

    print("\n== 6. 상대 이득과 '+%' 표기 ==")
    rel_real = (62.5 - 45.3) / 45.3 * 100
    rel_unseen = (46.6 - 15.0) / 15.0 * 100
    rel_sim = (66.66 - 49.16) / 49.16 * 100
    print(f"  실로봇 seen: 차이 17.2 pp, 상대 +{rel_real:.0f}% (논문은 '+17.2%'로 표기)")
    print(f"  비가시 센서: 차이 31.6 pp, 상대 +{rel_unseen:.0f}% (FTP-pi0.5 15.0 기준 상대 +{(46.6-15)/15*100:.0f}%)")
    print(f"  시뮬 전체: 차이 17.5 pp, 상대 +{rel_sim:.0f}%")
    RES["relative_gains_pct"] = {"real_seen": round(rel_real, 1), "unseen": round(rel_unseen, 1),
                                 "sim_all": round(rel_sim, 1)}

    print("\n== 7. 표 2 과제별 분해 (FTP-1 - pi0.5, 과제 6개) ==")
    diffs = [(a - b) for a, b in zip(T2["FTP-1"], T2["pi0.5"])]  # 이미 %p
    names = ["Draw Balloon", "Fix Hand (Tear)", "Fix Hand (Finish)", "Twist Cap", "Flip Book", "Wipe Dish"]
    for nm, dv in zip(names, diffs):
        print(f"  {nm:<18} {dv:+.0f} pp")
    RES["table2_per_task_diff_pp_vs_pi0p5"] = dict(zip(names, diffs))
    print(f"  과제 중 FTP-1이 낮은 과제 수: {sum(1 for x in diffs if x < 0)} / 6")

    print("\n== 8. 검출력 근사 (20회 대 20회, 양측 0.05, 검정력 0.8, 정규근사) ==")
    z = 1.959964 + 0.841621
    mde_task = z * math.sqrt(2 * 0.25 / 20)
    mde_pool = z * math.sqrt(2 * 0.25 / 120)
    print(f"  과제 1개(20회 대 20회, p=0.5 가정) 최소 검출 차이 약 {100*mde_task:.0f} pp")
    print(f"  6과제 풀링(120회 대 120회) 약 {100*mde_pool:.0f} pp")
    RES["mde_pp"] = {"single_task_20v20": round(100 * mde_task, 1), "pooled_120v120": round(100 * mde_pool, 1)}

    print("\n== 9. 표 12 (F.3) 상대 변화 ==")
    force_t3, force_trunk, force_tok = 0.273, 0.314, 0.424
    print(f"  FeelAnyForce 선형 MAE: T3 {force_t3} -> trunk {force_trunk} ({100*(force_trunk/force_t3-1):+.0f}%), "
          f"tokenizer {force_tok} ({100*(force_tok/force_t3-1):+.0f}%) (낮을수록 좋음)")
    print(f"  T1 force MAE: T3 0.106 -> trunk 0.107 ({100*(0.107/0.106-1):+.1f}%), tokenizer 0.116 ({100*(0.116/0.106-1):+.0f}%)")
    print(f"  T2 slip F1: T3 0.725 -> trunk 0.780 (+0.055), tokenizer 0.786 (+0.061), 참조 0.922; 표준편차 없음")
    RES["table12"] = {"feelanyforce_mae_change_pct": {"trunk": round(100 * (force_trunk / force_t3 - 1), 1),
                                                      "tokenizer": round(100 * (force_tok / force_t3 - 1), 1)}}

    print("\n== 10. 파라미터 수 추정 (표 B.5 사양) ==")
    for name, w, depth, mlp, heads, hd in [("paper B.5 (w1024, mlp4096)", 1024, 18, 4096, 8, 256),
                                           ("code gemma_small (w512, mlp2048)", 512, 18, 2048, 8, 256)]:
        inner = heads * hd
        attn = w * inner * 2 + w * 256 * 2  # q, o (inner dim) + k, v (num_kv_heads=1, head_dim=256)
        mlp_p = 3 * w * mlp  # gated (GeGLU) MLP: gate, up, down
        per_layer = attn + mlp_p
        total = depth * per_layer
        print(f"  {name}: 블록 약 {total/1e6:.0f}M 파라미터 (임베딩·정규화 제외, 근사)")
        RES.setdefault("tactile_expert_params_est_M", {})[name] = round(total / 1e6, 1)

    OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=2))
    print(f"\n결과 저장: {OUT}")


if __name__ == "__main__":
    main()
