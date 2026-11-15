"""RL Token 보고 수치 재계산 (scipy, 수 초, CPU).

입력: 논문 그림 4·5·6·7·8·9를 눈으로 읽은 값(±1%p 또는 ±2 성공/10분 수준의 읽기 오차).
본문에 명시된 값(임계 구간 50 에피소드, 중앙값 146/228/66, 5시간 데이터, 25 샘플/초, UTD 5)은 그대로 씁니다.
그림에서 읽은 값은 모두 `DIGITIZED`에 적었고, 원자료 표가 없으므로 재계산은 근사입니다.

실행: python3 analysis/table_check.py  → analysis/table_check_results.json
"""
import json
import math
from pathlib import Path

from scipy import stats

OUT = Path(__file__).resolve().parent

# 그림 5 (성공률), 읽은 값과 막대 오차 범위. 단위: 비율
SUCCESS_CRIT = {  # 임계 구간, 과제당 50 에피소드(본문 §VI-A)
    "screw": {"base": (0.20, 0.10, 0.37), "rlt": (0.65, 0.49, 0.81)},
    "zip": {"base": (0.35, 0.19, 0.51), "rlt": (1.00, 0.88, 1.00)},
    "ethernet": {"base": (0.98, 0.89, 1.00), "rlt": (1.00, 0.93, 1.00)},
    "charger": {"base": (0.80, 0.67, 0.89), "rlt": (1.00, 0.93, 1.00)},
}
SUCCESS_FULL = {  # 전체 과제, 에피소드 수 미보고
    "screw": {"base": (0.07, 0.02, 0.21), "rlt": (0.47, 0.30, 0.64)},
    "zip": {"base": (0.27, 0.14, 0.45), "rlt": (0.87, 0.71, 0.95)},
}
# 그림 4 (처리량, 10분당 성공 수). 중앙, 오차 하한, 상한
THRU_CRIT = {
    "screw": {"base": (6.7, 3.2, 12.8), "rlt": (39, 29, 48)},
    "zip": {"base": (17, 9, 25), "rlt": (120, 108, 133)},
    "ethernet": {"base": (148, 140, 156), "rlt": (400, 370, 430)},
    "charger": {"base": (135, 118, 155), "rlt": (600, 562, 640)},
}
THRU_FULL = {
    "screw": {"base": (1.7, 0.3, 5.5), "rlt": (13.5, 9.0, 19.0)},
    "zip": {"base": (2.8, 1.5, 4.7), "rlt": (13.0, 10.6, 15.5)},
}
# 그림 6 (이더넷 기준선 비교): 성공률(중앙, 하한, 상한), 처리량(중앙, 하한, 상한)
BASELINES_ETH = {
    "DAgger": {"succ": (1.00, 0.88, 1.00), "thru": (270, 240, 300)},
    "HIL-SERL": {"succ": (0.00, 0.00, 0.00), "thru": (0, 0, 0)},
    "PLD": {"succ": (0.40, 0.25, 0.58), "thru": (45, 27, 67)},
    "DSRL": {"succ": (0.98, 0.83, 0.99), "thru": (210, 180, 245)},
    "RLT": {"succ": (1.00, 0.93, 1.00), "thru": (400, 370, 430)},
}
# 본문에 적힌 값 (그림 9 중앙값, 스텝 at 50 Hz)
MEDIAN_STEPS = {"teleop": 146, "base": 228, "rlt": 66}
N_CRIT = 50
HZ = 50.0


def wilson(p, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def fisher(p1, p2, n):
    k1, k2 = round(p1 * n), round(p2 * n)
    odds, p = stats.fisher_exact([[k1, n - k1], [k2, n - k2]])
    return dict(k_base=k1, k_rlt=k2, n=n, p_value=float(p))


def main():
    res = {"assumptions": {"n_critical_per_task": N_CRIT,
                           "reading_precision_pp": 1.0,
                           "note": "그림에서 읽은 값. 원자료 표 없음. 재계산은 근사."}}

    # 1. 임계 구간 성공률: Wilson 구간과 그림의 오차 범위 비교, Fisher 검정 (n=50 가정)
    rows = {}
    for task, d in SUCCESS_CRIT.items():
        pb, lo_b, hi_b = d["base"]
        pr, lo_r, hi_r = d["rlt"]
        wb, wr = wilson(pb, N_CRIT), wilson(pr, N_CRIT)
        rows[task] = dict(
            base=dict(p=pb, bar=[lo_b, hi_b], wilson95=[round(wb[0], 3), round(wb[1], 3)],
                      count_out_of_50=round(pb * N_CRIT, 1)),
            rlt=dict(p=pr, bar=[lo_r, hi_r], wilson95=[round(wr[0], 3), round(wr[1], 3)],
                     count_out_of_50=round(pr * N_CRIT, 1)),
            fisher=fisher(pb, pr, N_CRIT),
        )
    res["success_critical_n50"] = rows

    # 2. 전체 과제 성공률: 에피소드 수 미보고 → n에 따른 민감도
    sens = {}
    for task, d in SUCCESS_FULL.items():
        sens[task] = {}
        for n in (20, 30, 50):
            pb, pr = d["base"][0], d["rlt"][0]
            sens[task][str(n)] = fisher(pb, pr, n)
    res["success_full_sensitivity"] = sens

    # 3. 처리량 비와 델타 방법 구간 (막대 오차를 대칭 표준오차로 근사)
    thru = {}
    for setting, table in (("critical", THRU_CRIT), ("full", THRU_FULL)):
        thru[setting] = {}
        for task, d in table.items():
            b, r = d["base"], d["rlt"]
            ratio = r[0] / b[0]
            se_b = (b[2] - b[1]) / (2 * 1.96)
            se_r = (r[2] - r[1]) / (2 * 1.96)
            se_log = math.sqrt((se_r / r[0]) ** 2 + (se_b / b[0]) ** 2)
            thru[setting][task] = dict(
                ratio=round(ratio, 2),
                ratio_95=[round(ratio * math.exp(-1.96 * se_log), 2),
                          round(ratio * math.exp(1.96 * se_log), 2)],
                base=b[0], rlt=r[0])
    res["throughput_ratio"] = thru

    # 4. 본문 주장과 그림 값의 대조
    sp_success = {}
    for task, d in SUCCESS_FULL.items():
        sp_success[task] = dict(pp_gain=round((d["rlt"][0] - d["base"][0]) * 100, 1),
                                relative_gain=round(d["rlt"][0] / d["base"][0], 1))
    res["claim_checks"] = {
        "full_task_gain_pp (본문 +40 나사, +60 지퍼타이)": sp_success,
        "median_speed_ratio_base_over_rlt": round(MEDIAN_STEPS["base"] / MEDIAN_STEPS["rlt"], 2),
        "median_speed_ratio_teleop_over_rlt": round(MEDIAN_STEPS["teleop"] / MEDIAN_STEPS["rlt"], 2),
        "ethernet_throughput_ratio_fig4": round(THRU_CRIT["ethernet"]["rlt"][0] / THRU_CRIT["ethernet"]["base"][0], 2),
        "min_task_speed_claim": "본문은 '최대 3배'. 그림 9 중앙값 비 3.45와 일치. 처리량 비는 6–7배(성공률 증가 포함)",
        "base_ethernet_median_below_stated_critical_range": MEDIAN_STEPS["base"] < 250,
    }

    # 5. 처리량 ≈ 성공률 × 600초 / 에피소드 시간 (중앙값으로 근사)
    cons = {}
    for name, succ, steps in (("base_ethernet", 0.98, MEDIAN_STEPS["base"]),
                              ("rlt_ethernet", 1.00, MEDIAN_STEPS["rlt"])):
        sec = steps / HZ
        pred = succ * 600 / sec
        obs = THRU_CRIT["ethernet"]["base" if name.startswith("base") else "rlt"][0]
        cons[name] = dict(episode_sec_median=round(sec, 2), predicted_throughput=round(pred, 0),
                          observed=obs, obs_over_pred=round(obs / pred, 2))
    res["consistency_throughput"] = cons

    # 6. 실로봇 샘플 산술 (본문 수치 + 25 샘플/초 + UTD 5)
    samples_per_s = 25.0  # 부록 B: 2스텝 스트라이드, 50 Hz
    arith = {}
    for label, hours in (("15분", 0.25), ("1시간", 1.0), ("5시간", 5.0)):
        secs = hours * 3600
        trans = secs * samples_per_s
        arith[label] = dict(
            transitions=int(trans),
            updates_if_UTD_per_transition=int(trans * 5),
            updates_if_UTD_per_10step_chunk=int(secs * (HZ / 10) * 5),
        )
    res["sample_arithmetic"] = arith
    # 에피소드 수와 임계 구간 길이 (본문: 400–1000 에피소드, 임계 구간 5–20초)
    ep = {}
    for n_ep in (400, 1000):
        for dur in (5, 10, 20):
            ep[f"{n_ep}ep_x_{dur}s_hours"] = round(n_ep * dur / 3600, 2)
    res["episode_budget_hours"] = ep
    res["ethernet_5min_rlt_successes"] = round(262 / 10 * 5, 0)  # 그림 7: 5분 시점 처리량 262/10분 가정

    # 7. 기준선 (그림 6): PLD 성공률 Wilson, 처리량 비
    res["baselines_eth_wilson"] = {
        k: dict(succ=v["succ"][0], wilson95=[round(x, 3) for x in wilson(v["succ"][0], N_CRIT)],
                thru=v["thru"][0]) for k, v in BASELINES_ETH.items()
    }

    (OUT / "table_check_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: res[k] for k in ("claim_checks", "consistency_throughput", "sample_arithmetic", "episode_budget_hours")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
