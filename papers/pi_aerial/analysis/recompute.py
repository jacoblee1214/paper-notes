"""AirVLA (arXiv:2603.25038 v1) 논문 표 수치를 다시 계산하고 통계 검정을 한다.

입력: 논문 표 I, II, III의 값을 아래에 그대로 옮겨 적었다(출처는 주석). 외부 데이터는 쓰지 않는다.
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)

가정(논문에 없는 것은 표시한다)
- 표 I, II는 방법·과제마다 20회 시도다(표 캡션 "twenty trials per task").
- Place, Hover, Pick(표 II)은 앞 단계 성공 횟수를 분모로 하는 조건부 비율이다(본문 IV-D).
- 표 III은 객체당 10회, 게이트 영역당 5회다(본문 IV-D).
- 반올림은 소수 첫째 자리(논문 표기)로 보고, 정수 시도 수로 만들 수 있는 (k, n)을 찾는다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
RES = {}
N = 20


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(100 * (c - h), 1), round(100 * (c + h), 1)


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def feasible(pct, nmax):
    """pct(소수 첫째 자리)가 k/n (n <= nmax)로 만들어지는 (k, n) 목록."""
    out = []
    for n in range(1, nmax + 1):
        for k in range(0, n + 1):
            if round(100 * k / n, 1) == pct:
                out.append((k, n))
    return out


# ---- 표 I (논문 IV-D, 표 I) : 단계별 보고 비율 -------------------------------------------
# 값: (pick, place, gate_ns, hover_ns, gate_s, hover_s), None은 논문의 "-"
T1 = {
    "naive": (50.0, 0.0, 50.0, 60.0, 45.0, 100.0),
    "rtc": (85.0, 23.5, 80.0, 81.2, 95.0, 100.0),
    "guide": (100.0, 50.0, None, None, None, None),
    "act": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "dp": (10.0, 0.0, 15.0, 0.0, 0.0, 0.0),
}
# ---- 표 II : 합성 데이터 설정별 (gate, hover, pick, place) ------------------------------
T2 = {
    ("nosyn", "naive"): (35.0, 85.7, 42.9, 0.0),
    ("nosyn", "rtc"): (80.0, 100.0, 81.2, 15.4),
    ("nosyn", "guide"): (70.0, 100.0, 100.0, 35.7),
    ("nosyn", "act"): (0.0, 0.0, 0.0, 0.0),
    ("nosyn", "dp"): (5.0, 0.0, 0.0, 0.0),
    ("syn", "naive"): (70.0, 85.7, 25.0, 0.0),
    ("syn", "rtc"): (95.0, 94.7, 83.3, 20.0),
    ("syn", "guide"): (85.0, 100.0, 94.1, 62.5),
    ("syn", "act"): (25.0, 0.0, 0.0, 0.0),
    ("syn", "dp"): (0.0, 0.0, 0.0, 0.0),
}
# ---- 표 III : 객체 변형(객체당 10회), 게이트 영역(영역당 5회) ---------------------------------
T3_OBJ = {"chips": (10.0, 0.0), "sandwich": (70.0, 57.1), "box": (30.0, 33.3)}
T3_GATE = {"front": (0.0, None), "left": (0.0, None), "right": (40.0, 100.0)}


def chain_counts(pcts, n0=N):
    """보고 비율 사슬(앞 단계 성공 수가 다음 분모)을 정수 (k, n)로 복원한다.

    반환: 복원된 [(k, n), ...] 와 문제 목록. 분모가 0이면 비율은 '0.0'으로 적힌 것으로 본다.
    가능한 해가 여러 개이면(드묾) 앞 단계 분모와 정확히 일치하는 해를 우선한다.
    """
    counts, issues = [], []
    prev = n0
    for i, pct in enumerate(pcts):
        if prev == 0:
            counts.append((0, 0))
            if pct != 0.0:
                issues.append(f"단계 {i + 1}: 분모 0인데 {pct} 보고")
            continue
        sols = [(k, n) for k, n in feasible(pct, prev) if n == prev]
        if not sols:
            alt = feasible(pct, prev)
            issues.append(
                f"단계 {i + 1}: 분모 {prev}로 {pct}%를 만들 수 없음"
                + (f" (n<={prev}로는 {alt[:3]} 가능)" if alt else " (어떤 n<=분모로도 불가)")
            )
            k = None
            counts.append((None, prev))
            prev = 0 if k is None else k
            # 복원 불가 시 다음 단계는 가장 가까운 정수로 가정하지 않고 중단한다
            return counts, issues
        k, n = sols[0]
        counts.append((k, n))
        prev = k
    return counts, issues


def main():
    # ---------------- 표 I ----------------
    t1 = {}
    t1_issues = []
    for m, row in T1.items():
        pick, place, gns, hns, gs, hs = row
        rec = {}
        c, iss = chain_counts([pick, place])
        rec["pick_place"] = c
        t1_issues += [f"I/{m}/pick-place: {x}" for x in iss]
        if gns is not None:
            c, iss = chain_counts([gns, hns])
            rec["nav_nonsyn"] = c
            t1_issues += [f"I/{m}/nav-nonsyn: {x}" for x in iss]
            c, iss = chain_counts([gs, hs])
            rec["nav_syn"] = c
            t1_issues += [f"I/{m}/nav-syn: {x}" for x in iss]
        t1[m] = rec
    RES["table1_counts"] = {m: {k: v for k, v in r.items()} for m, r in t1.items()}
    RES["table1_issues"] = t1_issues

    # ---------------- 표 II ----------------
    t2 = {}
    t2_issues = []
    for (setting, m), row in T2.items():
        c, iss = chain_counts(list(row))
        t2[f"{setting}/{m}"] = c
        t2_issues += [f"II/{setting}/{m}: {x}" for x in iss]
    RES["table2_counts"] = t2
    RES["table2_issues"] = t2_issues

    # 표 II naive/nosyn pick 42.9의 가능한 해석
    RES["table2_nosyn_naive_pick_429_feasible_kn_le20"] = feasible(42.9, 20)

    # ---------------- 시도 수 합계 (460) ----------------
    tot = {
        "table1_pick_place": 5 * N,
        "table1_nav_nonsyn": 4 * N,
        "table1_nav_syn": 4 * N,
        "table2": 10 * N,
    }
    tot["main_total"] = sum(tot.values())
    tot["ood_objects"] = 3 * 10
    tot["ood_gate"] = 3 * 5
    tot["grand_total_with_ood"] = tot["main_total"] + tot["ood_objects"] + tot["ood_gate"]
    RES["trial_counts"] = tot

    # ---------------- 끝에서 끝(end-to-end) 성공률 ----------------
    def e2e(counts_chain, last):
        k, n = counts_chain[last]
        return k  # 마지막 단계 성공 횟수 = 전체 N 중 끝까지 성공

    e = {}
    e["pickplace_naive_place"] = e2e(t1["naive"]["pick_place"], 1)
    e["pickplace_rtc_place"] = e2e(t1["rtc"]["pick_place"], 1)
    e["pickplace_guide_place"] = e2e(t1["guide"]["pick_place"], 1)
    e["nav_nonsyn_naive_hover"] = e2e(t1["naive"]["nav_nonsyn"], 1)
    e["nav_nonsyn_rtc_hover"] = e2e(t1["rtc"]["nav_nonsyn"], 1)
    e["nav_syn_naive_hover"] = e2e(t1["naive"]["nav_syn"], 1)
    e["nav_syn_rtc_hover"] = e2e(t1["rtc"]["nav_syn"], 1)
    # 표 II 정수 복원이 되는 행의 끝에서 끝
    for key, c in t2.items():
        if all(x[0] is not None for x in c):
            e["comp_" + key.replace("/", "_") + "_place"] = c[3][0]
    RES["end_to_end_successes_out_of_20"] = e
    RES["end_to_end_pct_and_wilson"] = {
        k: {"k": v, "pct": 100 * v / N, "wilson95": wilson(v, N)} for k, v in e.items()
    }

    # ---------------- 초록·본문 주장 대조 ----------------
    claims = {}
    # (a) 초록 "23%에서 50%": 23.5는 4/17, 종단은 4/20
    claims["abstract_23_to_50"] = {
        "reported_conditional": {"rtc": "4/17=23.5%", "guide": "10/20=50%"},
        "end_to_end": {"rtc": "4/20=20%", "guide": "10/20=50%"},
        "fisher_conditional_p": fisher(4, 17, 10, 20),
        "fisher_end_to_end_p": fisher(4, 20, 10, 20),
    }
    # (b) 초록 "62% 전체 성공" (표 II synthetic guide place 62.5)
    sg = t2["syn/guide"]
    claims["abstract_62_overall"] = {
        "chain": sg,
        "conditional_place": f"{sg[3][0]}/{sg[3][1]}={100 * sg[3][0] / sg[3][1]:.1f}%",
        "end_to_end": f"{sg[3][0]}/20={100 * sg[3][0] / 20:.1f}%",
        "wilson95_end_to_end": wilson(sg[3][0], 20),
        "wilson95_conditional": wilson(sg[3][0], sg[3][1]),
    }
    # (c) 초록 "81%에서 100%" 합성 데이터
    claims["abstract_81_to_100_nav"] = {
        "hover_given_gate": {"nonsyn": "13/16=81.2%", "syn": "19/19=100%"},
        "gate": {"nonsyn": "16/20=80%", "syn": "19/20=95%"},
        "end_to_end": {"nonsyn": "13/20=65%", "syn": "19/20=95%"},
        "fisher_hover_given_gate_p": fisher(13, 16, 19, 19),
        "fisher_gate_p": fisher(16, 20, 19, 20),
        "fisher_end_to_end_p": fisher(13, 20, 19, 20),
        "wilson95_hover_nonsyn_13of16": wilson(13, 16),
        "wilson95_hover_syn_19of19": wilson(19, 19),
    }
    # 기여 절 "합성 데이터로 항법 성공 20% 증가", "가이던스로 10%-40%"
    claims["contrib_guidance_10_40"] = {
        "tableI_pick": 100.0 - 85.0,
        "tableI_place_cond": 50.0 - 23.5,
        "tableII_nosyn_pick": 100.0 - 81.2,
        "tableII_nosyn_place": 35.7 - 15.4,
        "tableII_syn_pick": 94.1 - 83.3,
        "tableII_syn_place": 62.5 - 20.0,
        "tableI_place_end_to_end_pp": 50.0 - 20.0,
        "tableII_nosyn_place_end_to_end_pp": 100 * 5 / 20 - 100 * 2 / 20,
        "tableII_syn_place_end_to_end_pp": 100 * 10 / 20 - 100 * 3 / 20,
    }
    RES["claims_vs_tables"] = claims

    # ---------------- 방법 간 검정 (Fisher, 종단 k/20) ----------------
    tests = {}

    def add(name, k1, n1, k2, n2):
        tests[name] = {"k1": k1, "n1": n1, "k2": k2, "n2": n2,
                       "p1": round(k1 / n1, 3), "p2": round(k2 / n2, 3),
                       "fisher_p": round(fisher(k1, n1, k2, n2), 4)}

    add("pickplace_end2end: guide vs rtc (10/20 vs 4/20)", 10, 20, 4, 20)
    add("pickplace_end2end: rtc vs naive (4/20 vs 0/20)", 4, 20, 0, 20)
    add("pickplace_end2end: guide vs naive (10/20 vs 0/20)", 10, 20, 0, 20)
    add("pick only: guide vs rtc (20/20 vs 17/20)", 20, 20, 17, 20)
    add("pick only: rtc vs naive (17/20 vs 10/20)", 17, 20, 10, 20)
    add("compositional place e2e synth: guide vs rtc (10/20 vs 3/20)", 10, 20, 3, 20)
    add("compositional place e2e nosyn: guide vs rtc (5/20 vs 2/20)", 5, 20, 2, 20)
    add("place e2e pooled 3 settings: guide vs rtc (25/60 vs 9/60)", 25, 60, 9, 60)
    add("nav gate nonsyn: rtc vs naive (16/20 vs 10/20)", 16, 20, 10, 20)
    add("nav gate syn: rtc vs naive (19/20 vs 9/20)", 19, 20, 9, 20)
    add("nav gate: syn vs nonsyn, rtc (19/20 vs 16/20)", 19, 20, 16, 20)
    add("nav gate: syn vs nonsyn, naive (9/20 vs 10/20)", 9, 20, 10, 20)
    add("nav e2e: syn vs nonsyn, rtc (19/20 vs 13/20)", 19, 20, 13, 20)
    add("nav e2e: syn vs nonsyn, naive (9/20 vs 6/20)", 9, 20, 6, 20)
    add("compositional gate, nosyn: guide vs rtc (14/20 vs 16/20) [동일 절차여야 함]", 14, 20, 16, 20)
    add("compositional gate, syn: guide vs rtc (17/20 vs 19/20) [동일 절차여야 함]", 17, 20, 19, 20)
    add("compositional gate: syn vs nosyn, rtc (19/20 vs 16/20)", 19, 20, 16, 20)
    add("compositional place e2e guide: syn vs nosyn (10/20 vs 5/20) [합성 데이터는 항법 전용]", 10, 20, 5, 20)
    add("compositional place|pick guide: syn vs nosyn (10/16 vs 5/14)", 10, 16, 5, 14)
    add("place e2e guide: single task vs compositional syn (10/20 vs 10/20)", 10, 20, 10, 20)
    add("OOD pick: sandwich vs chips (7/10 vs 1/10)", 7, 10, 1, 10)
    add("OOD gate right vs in-distribution rtc nonsyn (2/5 vs 16/20)", 2, 5, 16, 20)
    RES["tests"] = tests

    # ---------------- 표 III 복원 ----------------
    t3 = {}
    for o, (pk, pl) in T3_OBJ.items():
        kp = round(pk / 100 * 10)
        sols = feasible(pl, kp) if kp > 0 else []
        t3[o] = {"pick_k_of_10": kp, "place_given_pick": pl, "place_feasible_kn": [s for s in sols if s[1] == kp]}
    gate_k = {g: round(v[0] / 100 * 5) for g, v in T3_GATE.items()}
    t3["gate_k_of_5"] = gate_k
    t3["gate_pooled"] = {"k": sum(gate_k.values()), "n": 15, "pct": round(100 * sum(gate_k.values()) / 15, 1),
                         "wilson95": wilson(sum(gate_k.values()), 15)}
    obj_e2e = {"chips": 0, "sandwich": 4, "box": 1}
    t3["object_end_to_end_k_of_10"] = obj_e2e
    t3["object_end_to_end_wilson95"] = {o: wilson(k, 10) for o, k in obj_e2e.items()}
    t3["sandwich_57_is"] = "4/7 조건부; 종단은 4/10 = 40%"
    t3["pooled_objects_end_to_end"] = {"k": 5, "n": 30, "pct": round(100 * 5 / 30, 1), "wilson95": wilson(5, 30)}
    RES["table3"] = t3

    # ---------------- 표본 크기 / 검정력 ----------------
    def n_per_arm(p1, p2, alpha=0.05, power=0.8):
        za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
        pbar = (p1 + p2) / 2
        num = (za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        return math.ceil(num / (p1 - p2) ** 2)

    RES["sample_size_per_arm_80pct_power"] = {
        "20% vs 50% (pick-and-place e2e)": n_per_arm(0.2, 0.5),
        "65% vs 95% (nav e2e)": n_per_arm(0.65, 0.95),
        "15% vs 50% (compositional place e2e synth)": n_per_arm(0.15, 0.5),
        "80% vs 95% (gate only)": n_per_arm(0.8, 0.95),
    }
    # 20회 시도의 Wilson 구간 폭
    RES["wilson_width_n20"] = {"10/20": wilson(10, 20), "4/20": wilson(4, 20), "19/20": wilson(19, 20), "0/20": wilson(0, 20)}

    # ---------------- 데이터 규모 일관성 ----------------
    demos = 270
    hours = 10.0
    sec_per_demo = hours * 3600 / demos
    frames = hours * 3600 * 10  # 10 Hz
    samples = 30000 * 32
    RES["data_scale"] = {
        "teleop_demos": demos,
        "hours_claimed": hours,
        "sec_per_demo": round(sec_per_demo, 1),
        "frames_at_10Hz": int(frames),
        "samples_seen": samples,
        "epochs_over_frames": round(samples / frames, 2),
        "manip_120_plus_nav_150": 120 + 150,
        "nav_synth_fraction": round(50 / 200, 3),
        "note": "10시간이 270개 전체인지 조작 120개만인지 부록 문장이 모호하다. 후자면 epoch 수는 더 커진다.",
    }

    # ---------------- 추론 주기 ----------------
    RES["timing"] = {
        "naive_open_loop_seconds": 50 / 10,
        "naive_inference_hz": 10 / 50,
        "rtc_chunk_H": 25,
        "rtc_horizon_seconds_at_10Hz": 25 / 10,
        "camera_hz": 5,
        "actions_per_image": 10 / 5,
    }

    OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=1, default=lambda o: list(o)), encoding="utf-8")

    # ---------------- 표준출력 요약 ----------------
    print("460 =", tot)
    print("표 I 문제:", t1_issues or "없음")
    print("표 II 문제:", t2_issues or "없음")
    print("42.9 가능한 (k,n<=20):", RES["table2_nosyn_naive_pick_429_feasible_kn_le20"])
    for k, v in claims.items():
        print(k, json.dumps(v, ensure_ascii=False))
    for k, v in tests.items():
        print(f"{k}: p={v['fisher_p']}")
    print("표 III:", json.dumps(t3, ensure_ascii=False))
    print("표본 크기:", RES["sample_size_per_arm_80pct_power"])
    print("데이터 규모:", RES["data_scale"])


if __name__ == "__main__":
    main()
