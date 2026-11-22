"""논문 그림 7(v2)의 성공률을 20회 시행의 성공 수로 환산하고 구간과 검정을 다시 계산한다.

실행: python3 analysis/recompute.py   → analysis/recompute_results.json
입력: 그림 7 막대 높이를 눈으로 판독한 값(5% 단위, 시행 20회). 논문은 표 대신 그림만 제공한다.
      판독 오차는 ±1회(5%p)로 본다. 판독값은 아래 READ에 그대로 적었다.
"""
import json
from pathlib import Path
from math import sqrt, ceil

from scipy.stats import fisher_exact, norm

OUT = Path(__file__).resolve().parent / "recompute_results.json"
N = 20

# (과제, 입력 조건, 모델, 사전학습 여부) -> 성공 수 (판독 %/5)
READ = {
    ("USB", "tactile+vision", "ACT", False): 90, ("USB", "tactile+vision", "ACT", True): 95,
    ("USB", "tactile+vision", "DP", False): 70, ("USB", "tactile+vision", "DP", True): 75,
    ("USB", "vision", "ACT", False): 20, ("USB", "vision", "ACT", True): 85,
    ("USB", "vision", "DP", False): 45, ("USB", "vision", "DP", True): 75,
    ("USB", "tactile", "ACT", False): 70, ("USB", "tactile", "ACT", True): 45,
    ("USB", "tactile", "DP", False): 0, ("USB", "tactile", "DP", True): 0,  # 막대 없음, 본문상 0%
    ("RECT", "tactile+vision", "ACT", False): 70, ("RECT", "tactile+vision", "ACT", True): 80,
    ("RECT", "tactile+vision", "DP", False): 35, ("RECT", "tactile+vision", "DP", True): 55,
    ("RECT", "vision", "ACT", False): 10, ("RECT", "vision", "ACT", True): 75,
    ("RECT", "vision", "DP", False): 35, ("RECT", "vision", "DP", True): 60,
    ("CUBE", "tactile+vision", "ACT", False): 30, ("CUBE", "tactile+vision", "ACT", True): 95,
    ("CUBE", "tactile+vision", "DP", False): 55, ("CUBE", "tactile+vision", "DP", True): 70,
    ("CUBE", "vision", "ACT", False): 95, ("CUBE", "vision", "ACT", True): 100,
    ("CUBE", "vision", "DP", False): 75, ("CUBE", "vision", "DP", True): 95,
}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def n_per_arm(p1, p2, alpha=0.05, power=0.8):
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    pbar = (p1 + p2) / 2
    num = za * sqrt(2 * pbar * (1 - pbar)) + zb * sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    return ceil((num / abs(p1 - p2)) ** 2)


def main():
    res = {"n_trials": N, "reads": {}, "pretrain_effect": [], "checks": {}}
    for key, s in READ.items():
        k = s // 5
        lo, hi = wilson(k, N)
        res["reads"]["|".join(map(str, key))] = {"successes": k, "rate": k / N,
                                                 "wilson95": [round(lo, 3), round(hi, 3)]}

    # 사전학습 효과: 같은 과제·입력·모델에서 사전학습 대 미사전학습 (Fisher, 양측)
    pairs = []
    for task in ["USB", "RECT", "CUBE"]:
        for inp in ["tactile+vision", "vision", "tactile"]:
            for mdl in ["ACT", "DP"]:
                a, b = READ.get((task, inp, mdl, False)), READ.get((task, inp, mdl, True))
                if a is None or b is None or (task != "USB" and inp == "tactile"):
                    continue
                if task == "USB" and inp == "tactile" and mdl == "DP":
                    continue
                ka, kb = a // 5, b // 5
                _, p = fisher_exact([[kb, N - kb], [ka, N - ka]])
                pairs.append({"task": task, "input": inp, "model": mdl,
                              "not_pre": ka, "pre": kb, "diff_pp": (kb - ka) * 5, "p_fisher": p})
    adj = holm([x["p_fisher"] for x in pairs])
    for x, a in zip(pairs, adj):
        x["p_holm"] = round(a, 4)
        x["p_fisher"] = round(x["p_fisher"], 4)
    res["pretrain_effect"] = pairs

    # 시각 전용 사전학습 대 촉각+시각 미사전학습 (논문의 "on par" 주장)
    def cmp(t1, t2):
        k1, k2 = READ[t1] // 5, READ[t2] // 5
        _, p = fisher_exact([[k1, N - k1], [k2, N - k2]])
        return {"a": k1, "b": k2, "p_fisher": round(p, 4)}

    res["checks"]["USB_ACT_vis_pre_vs_tv_nopre"] = cmp(("USB", "vision", "ACT", True), ("USB", "tactile+vision", "ACT", False))
    res["checks"]["USB_ACT_vis_pre_vs_tv_pre"] = cmp(("USB", "vision", "ACT", True), ("USB", "tactile+vision", "ACT", True))
    res["checks"]["USB_DP_vis_pre_vs_tv_pre"] = cmp(("USB", "vision", "DP", True), ("USB", "tactile+vision", "DP", True))

    # 논문 본문 검사: "실패 감소" 비율 (본문 "50%와 20%")
    fail = {}
    for mdl in ["ACT", "DP"]:
        f0 = N - READ[("USB", "tactile+vision", mdl, False)] // 5
        f1 = N - READ[("USB", "tactile+vision", mdl, True)] // 5
        fail[mdl] = {"fail_before": f0, "fail_after": f1, "relative_reduction": round(1 - f1 / f0, 3)}
    res["checks"]["usb_failure_reduction"] = fail

    # 표본 크기: 본문 주요 대비를 80% 검정력으로 잡으려면 군당 몇 회인가
    res["checks"]["n_per_arm_for_0.70_vs_0.85"] = n_per_arm(0.70, 0.85)
    res["checks"]["n_per_arm_for_0.90_vs_0.95"] = n_per_arm(0.90, 0.95)
    res["checks"]["n_per_arm_for_0.45_vs_0.75"] = n_per_arm(0.45, 0.75)

    # 체계적 점검: 본문 수치가 그림과 맞는지 (각 항목 True/False)
    res["checks"]["text_claims"] = {
        "abstract_USB_vision_20_to_85": READ[("USB", "vision", "ACT", False)] == 20 and READ[("USB", "vision", "ACT", True)] == 85,
        "USB_tv_5pp_increase_both": (READ[("USB", "tactile+vision", "ACT", True)] - READ[("USB", "tactile+vision", "ACT", False)] == 5)
                                    and (READ[("USB", "tactile+vision", "DP", True)] - READ[("USB", "tactile+vision", "DP", False)] == 5),
        "cube_tv_30_to_95_ACT": READ[("CUBE", "tactile+vision", "ACT", False)] == 30 and READ[("CUBE", "tactile+vision", "ACT", True)] == 95,
        "DP_vision_45_to_75": READ[("USB", "vision", "DP", False)] == 45 and READ[("USB", "vision", "DP", True)] == 75,
        "tactile_only_ACT_pre_lower": READ[("USB", "tactile", "ACT", True)] < READ[("USB", "tactile", "ACT", False)],
    }

    # 공식 코드의 clip_loss는 logit_scale = 1.0(온도 1)이다. 코사인 유사도가 [-1, 1]이면
    # 정답 항목의 softmax 확률 상한은 e / (e + (n-1)/e)이고, 교차엔트로피 하한은 그 로그의 음수다.
    from math import e, log
    res["checks"]["loss_floor_tau1"] = {
        str(n): round(-log(e / (e + (n - 1) / e)), 4) for n in (5, 7)
    }

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: res["checks"][k] for k in res["checks"]}, ensure_ascii=False, indent=1)[:2500])
    for x in pairs:
        print(x["task"], x["input"], x["model"], x["not_pre"], "->", x["pre"], "diff", x["diff_pp"], "p", x["p_fisher"], "holm", x["p_holm"])


if __name__ == "__main__":
    main()
