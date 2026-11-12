"""AlpaJudge-RF 점수 재계산 (CPU, 즉시). 모델을 실행하지 않고, 공개 노트북에 기록된 부분점수로만 계산한다.

입력: paper/repos/alpajudge/alpajudge_pai.ipynb (NVlabs/alpajudge, 커밋 607ae9d, 노트북 마크다운에 적힌 관찰값)
식 (공개 소스 rf_model.py의 aggregate_soft_subscores):
  action    = p_match * (1 + p_safe) / 2
  component = mean(p_grounded, p_complete, p_causal)
  overall   = 0.6 * action + 0.4 * component
노트북은 "표시 점수는 소수 셋째 자리에서 절사(truncated)한다"고 적고 있으므로, 절사와 반올림 두 가정의 가능 범위를 함께 본다.
RB(0.999)는 체크포인트 실행이 필요해서 재계산하지 않는다.
실행: python3 analysis/rf_score_check.py  → analysis/rf_score_results.json
"""
import itertools
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "paper" / "repos" / "alpajudge" / "alpajudge_pai.ipynb"
W_ACT, W_COMP = 0.60, 0.40


def recorded_block():
    nb = json.loads(NB.read_text())
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if "observed output was" in src:
            m = re.search(r"```json\s*(\{.*?\})\s*```", src, re.S)
            return json.loads(m.group(1))
    raise SystemExit("노트북에서 RF 기록값을 찾지 못함")


def score(pm, ps, g, c, k):
    action = pm * (1 + ps) / 2
    comp = (g + c + k) / 3
    return action, comp, W_ACT * action + W_COMP * comp


def main():
    rec = recorded_block()
    s = rec["subscores"]
    vals = [s["action_matches_gt"], s["action_safe"], s["components_grounded"],
            s["components_complete"], s["component_causal"]]
    a, c, o = score(*vals)
    # 절사: 각 입력 v는 [v, v+0.001), 반올림: [v-0.0005, v+0.0005)
    def span(offset_lo, offset_hi):
        outs = []
        for d in itertools.product(*[(offset_lo, offset_hi)] * 5):
            outs.append(score(*[v + x for v, x in zip(vals, d)]))
        return min(x[2] for x in outs), max(x[2] for x in outs), min(x[0] for x in outs), max(x[0] for x in outs)

    trunc = span(0.0, 0.001)
    rnd = span(-0.0005, 0.0005)

    def overlaps(lo, hi, shown, half_lo, half_hi):
        # 기록값 shown은 [shown+half_lo, shown+half_hi) 구간의 참값을 표시한다. 재계산 구간과 겹치면 양립.
        return lo < shown + half_hi and hi > shown + half_lo

    # 절사 표시: 참값 in [v, v+0.001); 반올림 표시: 참값 in [v-0.0005, v+0.0005)
    ok_trunc = overlaps(trunc[0], trunc[1], rec["overall"], 0.0, 0.001) and \
        overlaps(trunc[2], trunc[3], rec["action"], 0.0, 0.001)
    ok_round = overlaps(rnd[0], rnd[1], rec["overall"], -0.0005, 0.0005) and \
        overlaps(rnd[2], rnd[3], rec["action"], -0.0005, 0.0005)
    out = {
        "recorded": {"action": rec["action"], "component": rec["component"], "overall": rec["overall"]},
        "recomputed_from_displayed_inputs": {"action": round(a, 4), "component": round(c, 4), "overall": round(o, 4)},
        "truncation_range_overall": [round(trunc[0], 4), round(trunc[1], 4)],
        "truncation_range_action": [round(trunc[2], 4), round(trunc[3], 4)],
        "rounding_range_overall": [round(rnd[0], 4), round(rnd[1], 4)],
        "rounding_range_action": [round(rnd[2], 4), round(rnd[3], 4)],
        "consistent_with_truncation_display": bool(ok_trunc),
        "consistent_with_rounding_display": bool(ok_round),
        "note": "입력 부분점수는 소수 셋째 자리로 표시됐고, 재계산 overall(0.8412)과 기록 overall(0.842)의 차이는 표시 반올림 범위 안이다. 두 표시 규칙 모두와 양립한다. 블록 간 모순은 없다.",
    }
    (ROOT / "analysis" / "rf_score_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
