"""Controller-Shaped Grasping (arXiv:2609.25887 v1) 보고 수치 재계산. 결과는 recompute_results.json.

실행: python3 analysis/recompute.py   (numpy, scipy 필요, 수 초)

논문 표(표 1–5, 7–12)와 본문 수치를 아래에 옮겨 놓고 다음을 확인한다.
  V1  행 합계(안정 + 느슨 + 낙하 = 시행 수)와 표 사이 같은 조건의 값
  V2  표 8의 Wilson 95% 구간과 Fisher 정확 검정(양측), 본문 p 값, Holm 보정
  V3  추가 시드(부록 E.3)의 평균 ± 표준편차가 본문 서술과 맞는지 (시드 4개, 표본 표준편차)
  V4  같은 비교를 시행 단위(한 시드 20회, 5개 시드 풀링 100회)와 시드 단위(Welch)로 본 p 값
  V5  표 11의 릴리스 시간 감소율과 시드별 평균
  V6  실험 2(동역학 손실)의 시행 단위 검정과 시드 풀링
  V7  정수 가능성: 퍼센트가 시행 수 20, 10에서 나올 수 있는지, 12.5% = 1/8
  V8  안정 판정 수와 접촉 자격(CQ) 에피소드 수의 관계
  V9  교사 근접 대역(2.41 ± 0.7)의 기하와 p10–p90의 정규 가정 표준편차
  V10 롤아웃 주파수 산술 (청크 길이, 재질의 주기, ≥10프레임의 시간)
  K1  표본 수 가정 아래 검출력 (몬테카를로)

가정은 문장 안에 밝힌다. 시드별 시행 수는 논문 서술대로 시드당 20회로 둔다(부록 E.3).
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
OUT = HERE / "recompute_results.json"
RES = {}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]], alternative="two-sided")[1])


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj = [None] * len(ps)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


# ---- 논문 값 ----
# (조건, 안정, 느슨, 낙하, n)  표 1, 표 3
MAIN = {
    "ACT 반사 데이터": (19, 1, 0, 20),
    "ACT 수동 데이터": (3, 11, 6, 20),
    "ACT 수동 + 중재기": (10, 4, 6, 20),
    "π0.5 반사 데이터": (19, 1, 0, 20),
    "π0.5 수동 데이터": (1, 14, 5, 20),
    "π0.5 수동 + 중재기": (11, 5, 4, 20),
    "종이컵 반사 데이터": (8, 2, 0, 10),
    "종이컵 수동 데이터": (3, 5, 2, 10),
}
# 부록 E.3 시드별 (시드 3407, 2025, 7, 123): 안정, 느슨, 낙하 (표 9, 10)
SEEDS = [3407, 2025, 7, 123]
T9 = {  # ACT
    "반사": [(19, 1, 0, 19, 15, 67.9), (20, 0, 0, 20, 16, 72.1), (18, 2, 0, 18, 14, 65.8), (19, 1, 0, 19, 15, 69.0)],
    "수동": [(3, 12, 5, 1, 0, 10.2), (2, 12, 6, 0, 0, 0.0), (4, 9, 7, 1, 0, 13.8), (4, 10, 6, 2, 0, 11.4)],
    "중재기": [(10, 4, 6, 9, 6, 37.8), (9, 3, 7, 11, 8, 41.1), (10, 5, 5, 10, 7, 39.0), (10, 4, 6, 10, 8, 39.7)],
}
T10 = {  # π0.5
    "반사": [(18, 2, 0, 17, 11, 67.3), (20, 0, 0, 19, 13, 72.1), (20, 0, 0, 18, 12, 69.4), (19, 1, 0, 18, 12, 71.0)],
    "수동": [(2, 14, 4, 2, 0, 11.2), (1, 13, 6, 1, 0, 0.0), (0, 14, 6, 0, 0, 0.0), (2, 13, 5, 2, 0, 7.4)],
    "중재기": [(12, 3, 5, 11, 8, 57.8), (10, 6, 4, 9, 6, 51.4), (11, 4, 5, 11, 8, 58.6), (9, 7, 4, 8, 6, 53.2)],
}
# 표 11: (백본, 변형, ≥10f 런, CQ-onset 근접 %, 릴리스 시간 s, 감소 %) 시드순
T11 = {
    ("ACT", "baseline"): [(14, 70.4, 16.1, None), (16, 74.1, 15.0, None), (15, 71.8, 16.4, None), (15, 73.0, 15.4, None)],
    ("ACT", "dyn"): [(17, 81.0, 13.8, 14.3), (18, 84.3, 12.9, 14.0), (18, 82.1, 13.2, 19.5), (18, 83.0, 13.5, 12.3)],
    ("π0.5", "baseline"): [(10, 77.4, 29.3, None), (12, 79.1, 27.9, None), (12, 78.0, 28.7, None), (13, 79.2, 27.4, None)],
    ("π0.5", "dyn"): [(15, 84.0, 22.4, 23.5), (16, 85.8, 22.0, 21.1), (16, 85.0, 21.5, 25.1), (17, 87.6, 20.9, 23.7)],
}
# 표 12: 외란 실험 (유지 횟수/20), 시드 3407, 2025, 7, 123
T12 = {"policy-only": [10, 12, 11, 9], "policy+arbiter": [20, 19, 20, 20]}
# 본문 서술 (평균 ± SD, 시드 4개)
TEXT_SEED = {
    "ACT 반사": (95.0, 4.1), "ACT 수동": (16.3, 4.8), "ACT 중재기": (48.8, 2.5),
    "π0.5 반사": (96.3, 4.8), "π0.5 수동": (6.3, 4.8), "π0.5 중재기": (52.5, 6.5),
    "외란 policy-only": (52.5, 6.5), "외란 중재기": (98.8, 2.5),
}
TEXT_EXP2 = {("ACT", "baseline"): (72.3, 1.6), ("ACT", "dyn"): (82.6, 1.4),
             ("π0.5", "baseline"): (78.4, 0.9), ("π0.5", "dyn"): (85.6, 1.5)}
# 표 8 (비교, A 횟수, A n, B 횟수, B n, 논문 p, 논문 A 구간, 논문 B 구간)
T8 = [
    ("ACT 반사 대 수동 (안정)", 19, 20, 3, 20, 4.1e-7, (76.4, 99.1), (5.2, 36.0)),
    ("ACT 반사 대 수동+중재기 (안정)", 19, 20, 10, 20, 3.3e-3, (76.4, 99.1), (29.9, 70.1)),
    ("ACT 반사 대 접촉 선별 수동 (안정)", 19, 20, 6, 20, 3.9e-5, (76.4, 99.1), (14.5, 51.9)),
    ("π0.5 반사 대 수동 (안정)", 19, 20, 1, 20, 5.8e-9, (76.4, 99.1), (0.9, 23.6)),
    ("π0.5 반사 대 수동+중재기 (안정)", 19, 20, 11, 20, 8.4e-3, (76.4, 99.1), (34.2, 74.2)),
    ("외란 policy-only 대 중재기 (실패)", 9, 20, 0, 20, 1.23e-3, (25.8, 65.8), (0.0, 16.1)),
]


def v1():
    out = {"행합계": {}}
    for name, (s, l, d, n) in MAIN.items():
        out["행합계"][name] = {"합": s + l + d, "n": n, "일치": s + l + d == n}
    for tab, nm in ((T9, "표 9 ACT"), (T10, "표 10 π0.5")):
        for k, rows in tab.items():
            for sd, r in zip(SEEDS, rows):
                out["행합계"][f"{nm} {k} 시드 {sd}"] = {"합": sum(r[:3]), "n": 20, "일치": sum(r[:3]) == 20}
    out["모두 일치"] = all(v["일치"] for v in out["행합계"].values())
    # 표 2, 7, E.4.1 사이 같은 조건
    out["표2_표7_ACT반사"] = {"표2": [19, 15, 68.7], "표7": [19, 15, 68.7], "일치": True}
    out["표4_표2_≥10f"] = {"ACT 기준선": [15, 15], "π0.5 기준선": [12, 12], "일치": True}
    # 표 11 시드별 ≥10f 평균이 주 실행과 가까운지
    for key in (("ACT", "baseline"), ("π0.5", "baseline")):
        v = [r[0] for r in T11[key]]
        out[f"표11 {key[0]} 기준선 ≥10f 시드 4개 평균"] = float(np.mean(v))
    RES["V1"] = out


def v2():
    rows = []
    ps = []
    for name, ka, na, kb, nb, p_paper, ci_a, ci_b in T8:
        p = fisher(ka, na, kb, nb)
        wa, wb = wilson(ka, na), wilson(kb, nb)
        rows.append({
            "비교": name, "A": f"{ka}/{na}", "B": f"{kb}/{nb}",
            "p_재계산": p, "p_논문": p_paper, "p_상대차": abs(p - p_paper) / p_paper,
            "wilson_A": [round(wa[0], 1), round(wa[1], 1)], "wilson_A_논문": list(ci_a),
            "wilson_B": [round(wb[0], 1), round(wb[1], 1)], "wilson_B_논문": list(ci_b),
        })
        ps.append(p)
    # 종이컵 (부록 E.4.2)
    p_cup = fisher(8, 10, 3, 10)
    cup = {"p_재계산": p_cup, "p_논문": 0.0698, "wilson_반사": [round(x, 1) for x in wilson(8, 10)],
           "wilson_수동": [round(x, 1) for x in wilson(3, 10)], "논문_반사": [49.0, 94.3], "논문_수동": [10.8, 60.3]}
    # Holm: 표 8의 6개 + 종이컵 1개
    allp = ps + [p_cup]
    adj = holm(allp)
    for r, a in zip(rows, adj[:6]):
        r["holm_p"] = a
    cup["holm_p"] = adj[6]
    RES["V2"] = {"표8": rows, "종이컵": cup}
    # 추가 비교: 안정 판정이 논문에서 검정되지 않은 쌍
    extra = {
        "ACT 수동+중재기 대 수동 (안정 10/20 대 3/20)": fisher(10, 20, 3, 20),
        "π0.5 수동+중재기 대 수동 (11/20 대 1/20)": fisher(11, 20, 1, 20),
        "ACT 중재기 낙하 6/20 대 수동 낙하 6/20": fisher(6, 20, 6, 20),
        "ACT 반사 대 접촉 선별 수동 (CQ 19/20 대 5/20)": fisher(19, 20, 5, 20),
        "종이컵 안정 8/10 대 3/10 (한쪽 검정)": float(stats.fisher_exact([[8, 2], [3, 7]], alternative="greater")[1]),
    }
    RES["V2"]["추가"] = extra


def _sd(x):
    return float(np.std(x, ddof=1))


def v3():
    out = {}
    pct = lambda vals: [100.0 * v / 20 for v in vals]
    series = {
        "ACT 반사": [r[0] for r in T9["반사"]], "ACT 수동": [r[0] for r in T9["수동"]], "ACT 중재기": [r[0] for r in T9["중재기"]],
        "π0.5 반사": [r[0] for r in T10["반사"]], "π0.5 수동": [r[0] for r in T10["수동"]], "π0.5 중재기": [r[0] for r in T10["중재기"]],
        "외란 policy-only": T12["policy-only"], "외란 중재기": T12["policy+arbiter"],
    }
    for k, v in series.items():
        p = pct(v)
        m, s = float(np.mean(p)), _sd(p)
        tm, ts = TEXT_SEED[k]
        out[k] = {"횟수": v, "평균": round(m, 2), "SD(표본)": round(s, 2), "SD(모집단)": round(float(np.std(p)), 2),
                  "논문": [tm, ts], "일치": abs(m - tm) < 0.06 and abs(s - ts) < 0.06}
    # 실험 2
    for key, rows in T11.items():
        near = [r[1] for r in rows]
        tm, ts = TEXT_EXP2[key]
        m, s = float(np.mean(near)), _sd(near)
        out[f"실험2 {key[0]} {key[1]} CQ-onset"] = {"평균": round(m, 2), "SD": round(s, 2), "논문": [tm, ts],
                                                 "일치": abs(m - tm) < 0.06 and abs(s - ts) < 0.06}
    # 주 실행(시드 42) 포함 5개 시드 평균
    main = {"ACT 반사": 19, "ACT 수동": 3, "ACT 중재기": 10, "π0.5 반사": 19, "π0.5 수동": 1, "π0.5 중재기": 11,
            "외란 policy-only": 11, "외란 중재기": 20}
    out["5개 시드 평균(시드 42 포함)"] = {
        k: {"평균": round(float(np.mean(pct([main[k]] + series[k]))), 2), "SD": round(_sd(pct([main[k]] + series[k])), 2)}
        for k in main}
    RES["V3"] = out


def v4():
    main = {"ACT 반사": 19, "ACT 수동": 3, "ACT 중재기": 10, "π0.5 반사": 19, "π0.5 수동": 1, "π0.5 중재기": 11}
    ser = {"ACT 반사": [r[0] for r in T9["반사"]], "ACT 수동": [r[0] for r in T9["수동"]], "ACT 중재기": [r[0] for r in T9["중재기"]],
           "π0.5 반사": [r[0] for r in T10["반사"]], "π0.5 수동": [r[0] for r in T10["수동"]], "π0.5 중재기": [r[0] for r in T10["중재기"]]}
    exp3 = {"외란 policy-only(유지)": [11] + T12["policy-only"], "외란 중재기(유지)": [20] + T12["policy+arbiter"]}
    cnt = {k: [main[k]] + ser[k] for k in main}
    cnt.update(exp3)
    pairs = [("ACT 반사", "ACT 수동"), ("ACT 반사", "ACT 중재기"), ("π0.5 반사", "π0.5 수동"),
             ("π0.5 반사", "π0.5 중재기"), ("외란 policy-only(유지)", "외란 중재기(유지)")]
    out = []
    for a, b in pairs:
        ka, kb = sum(cnt[a]), sum(cnt[b])
        n = 20 * len(cnt[a])
        pa = np.array(cnt[a]) / 20 * 100
        pb = np.array(cnt[b]) / 20 * 100
        w = stats.ttest_ind(pa, pb, equal_var=False)
        out.append({
            "비교": f"{a} 대 {b}", "풀링 횟수": [ka, kb], "n": n,
            "시행 단위 Fisher p(풀링 100회, 독립 가정)": fisher(ka, n, kb, n),
            "시드 단위 Welch p(시드 5개)": float(w.pvalue),
            "시드 평균(%)": [round(float(pa.mean()), 1), round(float(pb.mean()), 1)],
        })
    RES["V4"] = out


def v5():
    out = []
    for bb, base_t, dyn_t, red in (("ACT", 15.7, 13.4, 14.6), ("π0.5", 28.1, 21.7, 22.8)):
        out.append({"백본": bb, "주 실행": [base_t, dyn_t], "논문 감소": red, "재계산": round(100 * (base_t - dyn_t) / base_t, 2)})
    seeds = []
    for bb in ("ACT", "π0.5"):
        base = T11[(bb, "baseline")]
        dyn = T11[(bb, "dyn")]
        for sd, b, d in zip(SEEDS, base, dyn):
            calc = 100 * (b[2] - d[2]) / b[2]
            seeds.append({"백본": bb, "시드": sd, "기준선 s": b[2], "동역학 s": d[2], "논문": d[3], "재계산": round(calc, 2),
                          "일치(±0.06)": abs(calc - d[3]) < 0.06})
    RES["V5"] = {"주 실행": out, "시드별": seeds, "모두 일치": all(s["일치(±0.06)"] for s in seeds)}


def v6():
    # 주 실행 (≥10f 런): ACT 15/20 → 18/20, π0.5 12/20 → 16/20
    out = {}
    out["ACT 주 실행 Fisher(15/20 대 18/20)"] = fisher(15, 20, 18, 20)
    out["π0.5 주 실행 Fisher(12/20 대 16/20)"] = fisher(12, 20, 16, 20)
    for bb, mb, md in (("ACT", 15, 18), ("π0.5", 12, 16)):
        b = [mb] + [r[0] for r in T11[(bb, "baseline")]]
        d = [md] + [r[0] for r in T11[(bb, "dyn")]]
        kb, kd, n = sum(b), sum(d), 20 * len(b)
        w = stats.ttest_ind(np.array(d) / 20, np.array(b) / 20, equal_var=False)
        out[f"{bb} 5개 시드 풀링"] = {"기준선": f"{kb}/{n}", "동역학": f"{kd}/{n}", "Fisher p(풀링 독립 가정)": fisher(kb, n, kd, n),
                                 "시드 단위 Welch p": float(w.pvalue)}
        # CQ-onset 점유율 시드 단위
        nb = [r[1] for r in T11[(bb, "baseline")]]
        nd = [r[1] for r in T11[(bb, "dyn")]]
        out[f"{bb} CQ-onset 근접(시드 4개, Welch)"] = {"기준선": round(float(np.mean(nb)), 2), "동역학": round(float(np.mean(nd)), 2),
                                                  "p": float(stats.ttest_ind(nd, nb, equal_var=False).pvalue)}
        tb = [r[2] for r in T11[(bb, "baseline")]]
        td = [r[2] for r in T11[(bb, "dyn")]]
        out[f"{bb} 릴리스 시간(시드 4개, Welch)"] = {"기준선": round(float(np.mean(tb)), 2), "동역학": round(float(np.mean(td)), 2),
                                              "p": float(stats.ttest_ind(td, tb, equal_var=False).pvalue)}
    # 부록 F: 전 구간 CQ 근접 점유율
    out["부록 F 전 구간 점유율"] = {"ACT": [68.7, 54.8, round(54.8 - 68.7, 1)], "π0.5": [70.8, 61.2, round(61.2 - 70.8, 1)]}
    RES["V6"] = out


def v7():
    out = {}
    for n in (20, 10):
        out[f"n={n}에서 가능한 퍼센트(%)"] = sorted({round(100 * k / n, 1) for k in range(n + 1)})
    # 보고된 퍼센트가 20회, 10회로 정수 횟수가 되는가
    rep20 = [95, 15, 55, 30, 5, 70, 25, 50, 20, 100, 45, 0]
    rep10 = [80, 20, 0, 30, 50]
    out["n=20에서 불가능한 보고 값"] = [x for x in rep20 if abs(x * 20 / 100 - round(x * 20 / 100)) > 1e-9]
    out["n=10에서 불가능한 보고 값"] = [x for x in rep10 if abs(x * 10 / 100 - round(x * 10 / 100)) > 1e-9]
    # 12.5% = 1/8 (CQ 프레임 8개, 부록 E.4.3)
    out["12.5% = 1/8 (CQ 프레임 8개 중 1개)"] = 1 / 8 == 0.125
    # 표 11 CQ-onset, ≥10f: 시드별 ≥10f 런은 /20
    out["표 9 ≥10f 런 퍼센트(최대 CQ 에피소드 이내)"] = all(r[4] <= r[3] for tab in (T9, T10) for rows in tab.values() for r in rows)
    out["표 9·10 ≥10f ≤ CQ ≤ 20"] = all(r[4] <= r[3] <= 20 for tab in (T9, T10) for rows in tab.values() for r in rows)
    RES["V7"] = out


def v8():
    out = []
    # (표, 조건, 안정, CQ 에피소드)
    rows = [
        ("표 1·2 ACT", "반사", 19, 19), ("표 1·2 ACT", "수동", 3, 1), ("표 1·2 ACT", "수동+중재기", 10, 10),
        ("표 7 ACT", "접촉 선별 수동", 6, 5),
        ("E.4.1 π0.5", "반사", 19, 18), ("E.4.1 π0.5", "수동", 1, 1), ("E.4.1 π0.5", "수동+중재기", 11, 11),
    ]
    for tab, cond, st, cq in rows:
        out.append({"표": tab, "조건": cond, "안정": st, "CQ 에피소드": cq, "안정이 CQ보다 많음": st > cq, "차이": st - cq})
    for tab, nm in ((T9, "ACT"), (T10, "π0.5")):
        for k, rws in tab.items():
            for sd, r in zip(SEEDS, rws):
                out.append({"표": f"E.3 {nm}", "조건": f"{k} 시드 {sd}", "안정": r[0], "CQ 에피소드": r[3],
                            "안정이 CQ보다 많음": r[0] > r[3], "차이": r[0] - r[3]})
    n_more = sum(1 for o in out if o["안정이 CQ보다 많음"])
    RES["V8"] = {"행": out, "안정 > CQ인 행 수": n_more, "전체": len(out)}


def v9():
    med, half = 2.41, 0.7
    lo, hi = med - half, med + half
    p10, p90 = 1.98, 2.87
    # 정규 가정에서 p10–p90 반폭 = 1.2816 SD (중앙값 대칭 가정)
    sd_from_p = ((p90 - p10) / 2) / 1.2816
    out = {
        "대역 하한/상한": [round(lo, 2), round(hi, 2)], "논문 그림 5": [1.71, 3.11],
        "p10–p90": [p10, p90], "p10–p90 반폭": round((p90 - p10) / 2, 3),
        "정규 가정 SD(p10–p90에서)": round(sd_from_p, 3),
        "±0.7이 이 SD의 몇 배": round(half / sd_from_p, 2),
        "대역 하한 − CQ 임계(1.5)": round(lo - 1.5, 2),
        "중앙값 / CQ 임계": round(med / 1.5, 2),
        "코드 닫힘 정지 임계 1.3 대비 중앙값 비": round(med / 1.3, 2),
        "대역이 p10–p90보다 넓은 비율": round((hi - lo) / (p90 - p10), 2),
    }
    RES["V9"] = out


def v10():
    out = {}
    # π0.5: 청크 10 스텝, 20 Hz → 0.5 s. 재질의 1.7–1.8 Hz → 주기 0.556–0.588 s
    chunk_s = 10 / 20
    for f in (1.7, 1.8):
        out[f"π0.5 재질의 {f} Hz 주기(s)"] = round(1 / f, 3)
        out[f"π0.5 {f} Hz에서 추론+오버헤드(ms, 청크 0.5 s 제외)"] = round((1 / f - chunk_s) * 1000, 1)
    out["π0.5 청크 길이(s)"] = chunk_s
    out["반사 25 Hz 주기(ms)"] = 40
    out["반사 주기 대 π0.5 재질의 주기"] = round((1 / 1.75) / 0.04, 1)
    out["ACT 15 Hz 주기(ms)"] = round(1000 / 15, 1)
    out["반사 25 Hz 대 ACT 15 Hz 비"] = round(25 / 15, 2)
    # ≥10프레임의 시간 (프레임율을 서술하지 않으므로 후보별로)
    out["≥10프레임의 시간(s), 프레임율별"] = {str(hz): round(10 / hz, 2) for hz in (10, 15, 20, 25)}
    # 코드 정적 값: 닫힘 속도
    out["코드 닫힘 속도 mm/s (close_step_um 150 × 25 Hz)"] = round(150e-3 * 25, 3)
    out["30 mm 닫힘에 걸리는 시간 s"] = round(30 / (150e-3 * 25), 2)
    out["프레임 100 mm → 20 mm 전체 닫힘 s"] = round(80 / (150e-3 * 25), 2)
    RES["V10"] = out


def k1():
    rng = np.random.default_rng(0)

    def power(p1, p0, n, trials=4000):
        hit = 0
        for _ in range(trials):
            a = rng.binomial(n, p1)
            b = rng.binomial(n, p0)
            if fisher(a, n, b, n) < 0.05:
                hit += 1
        return hit / trials

    out = {
        "종이컵 80% 대 30%, n=10 (참값 가정)": power(0.8, 0.3, 10),
        "종이컵 80% 대 30%, n=20": power(0.8, 0.3, 20),
        "ACT 동역학 ≥10f 15/20 대 18/20 (75% 대 90%), n=20": power(0.90, 0.75, 20),
        "π0.5 동역학 ≥10f 12/20 대 16/20 (60% 대 80%), n=20": power(0.80, 0.60, 20),
        "ACT 동역학, 풀링 n=100 (75% 대 90%)": power(0.90, 0.75, 100, 1500),
    }
    # 필요한 시행 수: 80% 검출력, 종이컵
    need = None
    for n in range(10, 80, 2):
        if power(0.8, 0.3, n, 1500) >= 0.8:
            need = n
            break
    out["종이컵 80% 대 30%에서 검출력 0.8에 필요한 팔당 시행 수(약)"] = need
    RES["K1"] = out


def main():
    v1(); v2(); v3(); v4(); v5(); v6(); v7(); v8(); v9(); v10(); k1()
    OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
