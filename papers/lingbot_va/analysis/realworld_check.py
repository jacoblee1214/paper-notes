"""부록 실세계 표(S2–S7)를 논문 텍스트에서 직접 읽어 성공 규칙을 검사한다 (CPU, 1초 미만).

규칙(부록 A): 모든 단계를 끝내야 성공이며, 총점 = 최대점일 때만 성공이다.
각 시도 행: 시도 번호, 성공 표기, 단계별 점수 ..., 진행 점수. 단계 점수 합 = 진행 점수를 함께 확인한다.
입력: paper/paper.txt   출력: analysis/realworld_results.json
"""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
L = (ROOT / "paper" / "paper.txt").read_text(encoding="utf-8").splitlines()
SECS = {"Make Breakfast": 1505, "Pick Screws": 1564, "Fold Clothes": 1621,
        "Unpack Delivery": 1678, "Insert Tubes": 1735, "Fold Pants": 1792}
MAXS = {"Make Breakfast": 10, "Pick Screws": 5, "Fold Clothes": 6, "Unpack Delivery": 5,
        "Insert Tubes": 6, "Fold Pants": 3}
row = re.compile(r"^\s*(\d{1,2})\s+((?:\d+(?:\.\d+)?\s+)+\d+(?:\.\d+)?)\s*$")
out = {}
for name, start, who in [(n, s_, w) for n, s_ in SECS.items() for w in ("Ours", "π0.5")]:
    block = L[start:start + 60]
    # Ours 블록: "Ours" 줄 이후 Avg 이전까지 (첫 20행)
    o_idx = next(i for i, l in enumerate(block) if l.strip() == who)
    rows = []
    for l in block[o_idx + 1:]:
        if l.strip().startswith("Avg"):
            break
        m = row.match(l)
        if m:
            vals = [float(v) for v in m.group(2).split()]
            rows.append((int(m.group(1)), vals))
    assert len(rows) == 20, (name, len(rows))
    succ = np.array([r[1][0] for r in rows])
    prog = np.array([r[1][-1] for r in rows])
    steps_ok = all(abs(sum(r[1][1:-1]) - r[1][-1]) < 1e-9 for r in rows)
    rule = (prog == MAXS[name]).astype(int)
    mism = [r[0] for r, s, g in zip(rows, succ, rule) if s != g]
    out[f"{name} / {who}"] = {
        "n": len(rows), "stated_success": int(succ.sum()),
        "rule_success(progress==max)": int(rule.sum()),
        "trials_marked_success_but_not_max": [r[0] for r, s, g in zip(rows, succ, rule) if s == 1 and g == 0],
        "steps_sum_equals_progress": steps_ok,
        "PS_recomputed_pct": round(float(prog.mean() / MAXS[name] * 100), 1),
    }
(ROOT / "analysis" / "realworld_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(out, indent=1, ensure_ascii=False))
