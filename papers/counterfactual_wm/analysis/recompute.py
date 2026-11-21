"""03 재계산: 논문 표 값을 시도 횟수로 바꿔 구간과 검정을 다시 계산하고, 표·본문 사이 일관성을 검사한다.

입력: 논문 표 값을 아래에 옮겨 적은 상수(출처 표 번호 표시). 원자료(에피소드별 결과)는 공개되어 있지 않아
쌍대(paired) 검정은 할 수 없다. 대신 비쌍대(unpaired) 두 비율 검정과 Wilson 구간을 계산하고, 논문의 쌍대
구간과 비교한다. 실행: python3 analysis/recompute.py (수 초, 표준 라이브러리와 numpy만 사용).
"""
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent / "recompute_results.json"


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def two_prop_z(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else float("nan")
    p = math.erfc(abs(z) / math.sqrt(2)) if se > 0 else float("nan")
    se_unpaired = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return dict(diff_pp=round(100 * (p1 - p2), 2), z=round(z, 3), p_two_sided=p,
                unpaired_ci95_pp=[round(100 * (p1 - p2 - 1.96 * se_unpaired), 1),
                                  round(100 * (p1 - p2 + 1.96 * se_unpaired), 1)])


# 표 15 (one-shot planning). 값: (predictor, no outcome, CF WM, 논문 쌍대 구간 CF-predictor, CF-oracle), n
T15 = {
    ("MAE-B", "HandReach"): (11, 15, 37, (27, 20, 33), (22, 16, 28), 300),
    ("MAE-B", "Ball-in-Cup"): (10, 15, 60, (50, 45, 56), (45, 38, 51), 300),
    ("MAE-B", "FetchReach"): (11, 14, 69, (58, 51, 64), (55, 48, 61), 300),
    ("MAE-B", "BlueROV2"): (18, 23, 50, (32, 25, 38), (27, 21, 34), 300),
    ("MAE-B", "Liquid transport"): (13, 19, 62, (49, 39, 59), (43, 32, 55), 100),
    ("DINOv2-S", "HandReach"): (13, 15, 35, (23, 16, 29), (20, 13, 27), 300),
    ("DINOv2-S", "Ball-in-Cup"): (9, 15, 46, (37, 31, 43), (31, 24, 37), 300),
    ("DINOv2-S", "FetchReach"): (12, 14, 61, (49, 42, 55), (46, 39, 53), 300),
    ("DINOv2-S", "BlueROV2"): (23, 23, 48, (25, 19, 31), (25, 19, 32), 300),
    ("DINOv2-S", "Liquid transport"): (25, 19, 64, (39, 25, 52), (45, 33, 57), 100),
}


def check_table15():
    rows, flags = [], []
    for (enc, env), (pred, orc, cf, paired_pred, paired_orc, n) in T15.items():
        kp, ko, kc = round(pred * n / 100), round(orc * n / 100), round(cf * n / 100)
        row = dict(encoder=enc, env=env, n=n, k_pred=kp, k_oracle=ko, k_cf=kc,
                   wilson_cf=[round(100 * x, 1) for x in wilson(kc, n)],
                   wilson_pred=[round(100 * x, 1) for x in wilson(kp, n)],
                   cf_vs_pred=two_prop_z(kc, n, kp, n), cf_vs_oracle=two_prop_z(kc, n, ko, n),
                   paper_paired_cf_vs_pred=paired_pred, paper_paired_cf_vs_oracle=paired_orc)
        # 표의 점추정 차이와 본문 숫자가 반올림 범위(±1) 안인지
        if abs((cf - pred) - paired_pred[0]) > 1.5 or abs((cf - orc) - paired_orc[0]) > 1.5:
            flags.append(f"{enc} {env}: point difference differs from reported paired difference by > 1.5 pp")
        # 쌍대 구간이 비쌍대 구간보다 좁은지(상관이 양이면 정상)
        row["paper_interval_narrower_than_unpaired"] = (paired_pred[2] - paired_pred[1]) < (
            row["cf_vs_pred"]["unpaired_ci95_pp"][1] - row["cf_vs_pred"]["unpaired_ci95_pp"][0])
        rows.append(row)
    return rows, flags


# 표 16 (same-event retry, 3rd attempt, disturbance) 과 표 17 (3-step replanning, persistent fault)
T17_PERSIST = {"MAE-B HandReach": (4, 11, 7, (4, 11)), "MAE-B Ball-in-Cup": (9, 22, 13, (8, 19)),
               "MAE-B FetchReach": (12, 51, 39, (33, 46)), "MAE-B BlueROV2": (25, 50, 25, (18, 31)),
               "DINOv2-S HandReach": (6, 18, 12, (7, 17)), "DINOv2-S Ball-in-Cup": (6, 20, 14, (9, 19)),
               "DINOv2-S FetchReach": (11, 40, 29, (23, 36)), "DINOv2-S BlueROV2": (24, 47, 23, (17, 30))}
T17_FRESH = {"MAE-B HandReach": (6, 5, -2), "MAE-B Ball-in-Cup": (7, 7, 0), "MAE-B FetchReach": (11, 7, -4),
             "MAE-B BlueROV2": (12, 13, 1), "DINOv2-S HandReach": (6, 7, 1), "DINOv2-S Ball-in-Cup": (7, 7, 0),
             "DINOv2-S FetchReach": (10, 8, -2), "DINOv2-S BlueROV2": (9, 15, 6)}
T16_3RD = {"MAE-B HandReach": (13, 60, 47), "MAE-B Ball-in-Cup": (11, 86, 75), "MAE-B FetchReach": (12, 97, 85),
           "MAE-B BlueROV2": (21, 71, 50), "MAE-B Liquid transport": (21, 87, 66),
           "DINOv2-S HandReach": (12, 51, 39), "DINOv2-S Ball-in-Cup": (9, 62, 53), "DINOv2-S FetchReach": (14, 92, 78),
           "DINOv2-S BlueROV2": (24, 63, 39), "DINOv2-S Liquid transport": (22, 84, 62)}


def check_tables_16_17():
    out, flags = {}, []
    for k, (p, c, d, ci) in T17_PERSIST.items():
        out.setdefault("t17_persistent", {})[k] = dict(pred=p, cf=c, diff_reported=d,
                                                        n=300, cf_vs_pred=two_prop_z(round(c * 3), 300, round(p * 3), 300))
        if abs((c - p) - d) > 1.5:
            flags.append(f"T17 persistent {k}: {c}-{p} != {d}")
    fresh_ranges = []
    for k, (p, c, d) in T17_FRESH.items():
        fresh_ranges.append(d)
        if abs((c - p) - d) > 1.5:
            flags.append(f"T17 fresh {k}: {c}-{p} != {d}")
    out["t17_fresh_delta_range"] = [min(fresh_ranges), max(fresh_ranges)]
    for k, (p, c, d) in T16_3RD.items():
        if abs((c - p) - d) > 1.5:
            flags.append(f"T16 3rd {k}: {c}-{p} != {d}")
    out["t16_third_cf_range"] = [min(v[1] for v in T16_3RD.values()), max(v[1] for v in T16_3RD.values())]
    out["t16_third_pred_range"] = [min(v[0] for v in T16_3RD.values()), max(v[0] for v in T16_3RD.values())]
    persist_pred = [v[0] for v in T17_PERSIST.values()]
    persist_cf = [v[1] for v in T17_PERSIST.values()]
    out["t17_persistent_pred_range"] = [min(persist_pred), max(persist_pred)]
    out["t17_persistent_cf_range"] = [min(persist_cf), max(persist_cf)]
    return out, flags


# 표 5 (MAE-B, DINOv2-S, DINOv2-B; 8k 설정). 값: (oracle, predictor, feature, feature CI, readout, readout CI)
T5 = {
    ("MAE-B", "HandReach"): (49.1, 38.1, 82.4, (80, 85), 85.6, (83, 87)),
    ("MAE-B", "Ball-in-Cup"): (51.1, 41.6, 92.0, (91, 93), 94.2, (93, 95)),
    ("MAE-B", "FetchReach"): (50.0, 44.8, 94.0, (93, 95), 95.7, (94, 97)),
    ("MAE-B", "Liquid transport"): (50.6, 42.1, 82.3, (78, 86), 94.8, (92, 96)),
    ("MAE-B", "BlueROV2 8k"): (51.8, 31.1, 47.7, (44, 51), 73.6, (71, 77)),
    ("MAE-B", "BlueROV2 128k"): (51.8, 47.4, 72.2, (71, 74), 93.5, (93, 94)),
    ("DINOv2-S", "HandReach"): (49.1, 39.9, 82.3, (81, 84), 86.1, (84, 88)),
    ("DINOv2-S", "Ball-in-Cup"): (51.1, 42.4, 89.5, (88, 91), 91.6, (90, 93)),
    ("DINOv2-S", "FetchReach"): (50.0, 45.6, 94.1, (94, 94), 95.7, (95, 96)),
    ("DINOv2-S", "Liquid transport"): (50.6, 43.6, 81.7, (78, 85), 93.8, (92, 95)),
    ("DINOv2-S", "BlueROV2 8k"): (51.8, 33.5, 53.0, (49, 57), 74.2, (69, 79)),
    ("DINOv2-B", "HandReach"): (49.1, 38.1, 83.3, (82, 85), 86.6, (85, 88)),
    ("DINOv2-B", "Ball-in-Cup"): (51.1, 39.1, 83.3, (82, 84), 87.6, (85, 90)),
    ("DINOv2-B", "FetchReach"): (50.0, 44.3, 94.0, (93, 95), 95.4, (94, 96)),
    ("DINOv2-B", "Liquid transport"): (50.6, 42.4, 73.4, (70, 77), 91.1, (89, 93)),
    ("DINOv2-B", "BlueROV2 8k"): (51.8, 30.7, 46.1, (43, 50), 70.7, (68, 73)),
    ("DINOv2-B", "BlueROV2 128k"): (51.8, 45.7, 65.9, (64, 68), 90.1, (89, 91)),
}


def check_table5():
    flags, pts = [], []
    for k, (o, p, f, fci, r, rci) in T5.items():
        for name, v, ci in (("feature", f, fci), ("readout", r, rci)):
            if not (ci[0] <= v <= ci[1]):
                flags.append(f"T5 {k} {name} {v} outside its interval {ci}")
        pts.append(dict(enc=k[0], env=k[1], readout_minus_feature=round(r - f, 1), readout_minus_pred=round(r - p, 1)))
    # 본문: 처음 세 환경에서 readout 추가 이득 1.4–4.3 점, PassWater 12.1–17.7 점 (MAE-B, DINOv2-S, DINOv2-B)
    g3 = [x["readout_minus_feature"] for x in pts if x["env"] in ("HandReach", "Ball-in-Cup", "FetchReach")]
    gl = [x["readout_minus_feature"] for x in pts if x["env"] == "Liquid transport"]
    return dict(rows=pts, gain_first_three_range=[min(g3), max(g3)], gain_liquid_range=[min(gl), max(gl)]), flags


def consistency():
    """본문 범위 문장과 표의 최소·최대를 비교한다."""
    items = {}
    # 초록·서론: 기준선 38–46 대 개선 86–96 (네 환경, MAE-B와 DINOv2)
    pred_4env = [v[1] for k, v in T5.items() if k[1] in ("HandReach", "Ball-in-Cup", "FetchReach", "Liquid transport")]
    ro_4env = [v[4] for k, v in T5.items() if k[1] in ("HandReach", "Ball-in-Cup", "FetchReach", "Liquid transport")]
    items["abstract_baseline_38_46_vs_table"] = [min(pred_4env), max(pred_4env)]
    items["abstract_readout_86_96_vs_table"] = [min(ro_4env), max(ro_4env)]
    # 표 15 범위: 본문 "35–69%", "9–25%", "14–23%"
    cf = [v[2] for v in T15.values()]
    pr = [v[0] for v in T15.values()]
    orc = [v[1] for v in T15.values()]
    items["planning_cf_range_vs_text_35_69"] = [min(cf), max(cf)]
    items["planning_pred_range_vs_text_9_25"] = [min(pr), max(pr)]
    items["planning_oracle_range_vs_text_14_23"] = [min(orc), max(orc)]
    return items


def table8_vs_table5():
    """표 8의 6,553 에피소드 값(73.7)과 표 5의 8k 값(73.6)은 같은 설정인데 0.1 차이가 난다."""
    return dict(mae_bluerov_8k_table5=73.6, mae_bluerov_6553_table8=73.7,
                note="같은 설정으로 보이나 0.1 차이. 시드·반올림 차이로 보이며 원자료 없이 확정 불가")


def main():
    rows, f15 = check_table15()
    t1617, f1617 = check_tables_16_17()
    t5, f5 = check_table5()
    res = dict(table15=rows, tables16_17=t1617, table5=t5, consistency=consistency(),
               table8_vs_5=table8_vs_table5(), flags=f15 + f1617 + f5)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
    print(json.dumps(dict(flags=res["flags"], consistency=res["consistency"],
                          gain_first_three=t5["gain_first_three_range"], gain_liquid=t5["gain_liquid_range"],
                          t17_fresh=t1617["t17_fresh_delta_range"]), ensure_ascii=False, default=float))
    for r in rows:
        print(r["encoder"], r["env"], "CF-pred p=%.2g" % r["cf_vs_pred"]["p_two_sided"],
              "CF-oracle p=%.2g" % r["cf_vs_oracle"]["p_two_sided"], r["wilson_cf"])


if __name__ == "__main__":
    main()
