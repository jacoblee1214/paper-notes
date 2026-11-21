"""FAKTUAL (arXiv:2603.11634 v2) 논문 Table 5, Table 6 수치를 다시 계산하고 일관성을 검사한다.

입력: paper/paper.txt (pdftotext -layout으로 만든 v2 본문). 표 행을 정규식으로 읽는다.
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)

가정(논문에 없는 것은 표시한다)
- 각 셀의 값은 "20개 체크포인트 중 최댓값"이다(그림 3, 5, 6 캡션). 즉 시드당 성공률은 최댓값이라 위로 치우칠 수 있다.
- 시드당 롤아웃 수는 50이다. 셀 하나의 시드 3개는 150회 롤아웃으로 볼 수 있지만, 최댓값 선택 때문에 이항 구간은 근사다.
- 짝 비교(같은 시드 번호끼리)는 논문이 말하지 않는다. 그래서 독립 표본 Welch t 검정을 쓴다. 시드 3개라 검정력이 낮다.
- 실제 로봇 과제(그림 6)는 표 수치가 없어 재계산하지 않는다.
"""
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TXT = ROOT / "paper" / "paper.txt"
OUT = HERE / "recompute_results.json"

ROW = re.compile(
    r"^\s*(?P<task>[A-Z][A-Za-z ]*\(?[A-Z]*\)?)?\s*(?P<method>FAKTUAL|random|full dataset)\s+"
    r"(?P<demos>\d+)\s+(?P<s1>0\.\d+)\s+(?P<s2>0\.\d+)\s+(?P<s3>0\.\d+)\s+"
    r"(?P<mean>0\.\d+)\s+(?P<mn>0\.\d+)\s+(?P<mx>0\.\d+)\s*$"
)


def parse_tables(lines):
    """표 5(Metaworld)와 표 6(RoboMimic)의 행을 읽는다. 과제 이름은 빈 칸이면 앞 행을 잇는다."""
    t5_start = next(i for i, l in enumerate(lines) if l.startswith("Table 5: Success rates"))
    t6_start = next(i for i, l in enumerate(lines) if l.startswith("Table 6: Success rates"))
    t6_end = next(i for i, l in enumerate(lines) if i > t6_start and l.strip().startswith("I   Proofs"))
    out = {"table5": [], "table6": []}
    for key, a, b in (("table5", t5_start, t6_start), ("table6", t6_start, t6_end)):
        task = None
        for l in lines[a:b]:
            m = ROW.match(l)
            if not m:
                continue
            if m.group("task"):
                task = m.group("task").strip()
            out[key].append({
                "task": task, "method": m.group("method"), "demos": int(m.group("demos")),
                "seeds": [float(m.group(s)) for s in ("s1", "s2", "s3")],
                "mean": float(m.group("mean")), "min": float(m.group("mn")), "max": float(m.group("mx")),
            })
    return out


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def main():
    lines = TXT.read_text(encoding="utf-8").splitlines()
    tabs = parse_tables(lines)
    res = {"row_counts": {k: len(v) for k, v in tabs.items()}}

    # K1: 평균, 최솟값, 최댓값이 시드 값과 맞는지 확인
    mism = []
    for key, rows in tabs.items():
        for r in rows:
            s = np.array(r["seeds"])
            mean_calc = s.mean()
            # 소수 둘째 자리 반올림 값과 비교한다. 표 값은 원자료 반올림이므로 0.01 허용
            if abs(mean_calc - r["mean"]) > 0.0051:
                mism.append({"table": key, "task": r["task"], "method": r["method"], "demos": r["demos"],
                             "seeds": r["seeds"], "reported_mean": r["mean"], "calc_mean": round(mean_calc, 4)})
            if abs(s.min() - r["min"]) > 1e-9 or abs(s.max() - r["max"]) > 1e-9:
                mism.append({"table": key, "task": r["task"], "method": r["method"], "demos": r["demos"],
                             "issue": "min/max mismatch", "seeds": r["seeds"], "min": r["min"], "max": r["max"]})
    res["K1_mean_minmax_mismatches"] = mism

    # K2: 전체 데이터 행의 데모 수와 부록 G.1.2의 학습 데모 수 (PH 180, MH 270) 비교
    full = [(r["task"], r["demos"]) for r in tabs["table6"] if r["method"] == "full dataset"]
    res["K2_full_dataset_rows"] = full
    res["K2_appendix_G12_train_counts"] = {"PH": 180, "MH": 270}

    # K3: 짝 없는 Welch 검정, 셀별 차이와 Wilson 근사 구간(시드 3개 x 50회 = 150회)
    cells = []
    for key, rows in tabs.items():
        by = {}
        for r in rows:
            by.setdefault((r["task"], r["demos"]), {})[r["method"]] = r
        for (task, demos), d in by.items():
            if "FAKTUAL" not in d or "random" not in d:
                continue
            f, rd = d["FAKTUAL"], d["random"]
            fs, rs = np.array(f["seeds"]), np.array(rd["seeds"])
            t, p = stats.ttest_ind(fs, rs, equal_var=False)
            k_f = int(round(fs.mean() * 150))
            k_r = int(round(rs.mean() * 150))
            lo_f, hi_f = wilson(k_f, 150)
            lo_r, hi_r = wilson(k_r, 150)
            cells.append({
                "table": key, "task": task, "demos": demos,
                "faktual_mean": round(fs.mean(), 4), "random_mean": round(rs.mean(), 4),
                "diff_F_minus_R": round(fs.mean() - rs.mean(), 4),
                "welch_p": round(float(p), 4),
                "faktual_wilson_150": [round(lo_f, 3), round(hi_f, 3)],
                "random_wilson_150": [round(lo_r, 3), round(hi_r, 3)],
            })
    res["K3_cells"] = cells

    # K4: 셀 단위 부호 검정. FAKTUAL 평균이 높은 셀 수
    wins = sum(1 for c in cells if c["diff_F_minus_R"] > 0)
    ties = sum(1 for c in cells if c["diff_F_minus_R"] == 0)
    losses = sum(1 for c in cells if c["diff_F_minus_R"] < 0)
    n_nonzero = wins + losses
    sign_p = stats.binomtest(wins, n_nonzero, 0.5).pvalue if n_nonzero else None
    res["K4_sign_test"] = {"wins": wins, "ties": ties, "losses": losses, "two_sided_p": sign_p}

    # K5: 유의한 셀 수 (Welch p < 0.05)
    sig = [c for c in cells if c["welch_p"] < 0.05]
    res["K5_welch_p_lt_0.05"] = {"count": len(sig), "cells": [
        (c["task"], c["demos"], c["diff_F_minus_R"], c["welch_p"]) for c in sig]}

    # K6: 과제별 평균 차이 (데모 수 전체 평균)
    per_task = {}
    for c in cells:
        per_task.setdefault(c["task"], []).append(c["diff_F_minus_R"])
    res["K6_per_task_mean_diff"] = {k: round(float(np.mean(v)), 4) for k, v in per_task.items()}

    # K7: 그림 4 Pearson r (0.74–1.00)은 엔트로피 값이 없어 재계산할 수 없다. 표 수치만으로는 확인 불가.
    res["K7_pearson_claim"] = "entropy values not reported; cannot recompute (unverified)"

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    # 요약 출력
    print(f"rows parsed: table5={len(tabs['table5'])}, table6={len(tabs['table6'])}")
    print(f"K1 mismatches: {len(mism)}")
    for m in mism:
        print("   ", m)
    print("K2 full-dataset rows (MH should be 270 train):", sorted(set(full)))
    print(f"K3 cells: {len(cells)}")
    print(f"K4 sign test FAKTUAL>random wins={wins} ties={ties} losses={losses} p={sign_p}")
    print(f"K5 Welch p<0.05 cells: {len(sig)}")
    for c in sig:
        print("   ", c)
    print("K6 per-task mean diff:", res["K6_per_task_mean_diff"])
    neg = [(c["task"], c["demos"], c["diff_F_minus_R"]) for c in cells if c["diff_F_minus_R"] < 0]
    print("negative cells:", neg)


if __name__ == "__main__":
    main()
