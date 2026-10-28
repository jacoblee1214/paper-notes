"""공식 코드 두 곳의 동작을 CPU에서 확인한다 (외부 API 호출 없음). 실행: python3 analysis/code_logic_toy.py

E: reproduce/llm_answer_evaluator.py `create_fallback_evaluation`(229–268행)의 정확도 추출 규칙을
   그대로 옮겨, JSON 파싱이 실패한 판정 응답이 어떻게 채점되는지 본다.
G: modalprocessors.py의 그래프 구성 규칙(모달 앵커 이름 = "<이름> (image)", 앵커 내부 개체 → 앵커
   belongs_to 간선, 텍스트 그래프와는 개체 이름이 문자열로 같을 때만 병합)을 작은 무작위 그래프로 흉내 내,
   텍스트 개체에서 1-hop으로 그림 앵커에 닿는 비율이 이름 일치 확률 p와 앵커당 내부 개체 수 m에 따라
   어떻게 변하는지 본다. LightRAG local/mix 모드는 검색된 개체의 1-hop 간선만 펼친다(operate.py _get_node_data).
"""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def fallback_accuracy(response: str):
    """llm_answer_evaluator.py 229–268행과 같은 규칙."""
    acc = None
    for pattern in [r'"accuracy":\s*(\d+)', r"accuracy.*?(\d+)", r"correct.*?(\d+)", r"accurate.*?(\d+)"]:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            try:
                v = int(match.group(1))
                if v in [0, 1]:
                    acc = v
                    break
            except ValueError:
                continue
    if acc is None:
        positive = ["correct", "accurate", "right", "yes", "true", "match"]
        negative = ["incorrect", "wrong", "false", "no", "mismatch", "contradiction"]
        low = response.lower()
        if any(k in low for k in positive):
            acc = 1
        elif any(k in low for k in negative):
            acc = 0
    if acc is None:
        acc = 0
    return acc


CASES = [
    ("The generated answer is incorrect.", 0),
    ("Incorrect: the expected value is DKK 26,778 million, not 11,503 million.", 0),
    ("The answer does not match the expected answer (mismatch in the year).", 0),
    ("Not accurate; the model chose the VAE panel.", 0),
    ("This is wrong.", 0),
    ("The generated answer is correct.", 1),
    ('{"accuracy": 1, "reasoning": "matches"', 1),          # 닫는 괄호 없는 JSON
    ('{"accuracy": 0, "reasoning": "contradicts"', 0),
    ("Accuracy: 0 because the panel is wrong", 0),
]


def bridge_sim(p_match: float, m: int, n_anchor: int = 2000, seed: int = 0) -> dict:
    """앵커 하나당 내부 개체 m개. 각 내부 개체가 텍스트 그래프 개체와 이름이 정확히 같을 확률 p.
    텍스트 쪽 질의가 텍스트 개체를 찾았을 때 1-hop(belongs_to)으로 앵커에 닿으려면 내부 개체 하나 이상이
    병합되어 있어야 한다. 앵커 이름에는 '(image)' 접미사가 붙어 텍스트 개체와 병합되지 않는다."""
    rng = np.random.default_rng(seed)
    merged = rng.random((n_anchor, m)) < p_match
    reach = merged.any(1)
    return {"p_match": p_match, "m": m, "reachable_frac": round(float(reach.mean()), 3),
            "closed_form": round(1 - (1 - p_match) ** m, 3)}


def main() -> None:
    out = {"E_fallback": [], "G_bridge": []}
    wrong = 0
    for text, truth in CASES:
        got = fallback_accuracy(text)
        wrong += got != truth
        out["E_fallback"].append({"response": text, "intended": truth, "scored": got})
    out["E_summary"] = {"cases": len(CASES), "misscored": wrong,
                        "cause": "양성 키워드를 먼저 부분 문자열로 검사: 'incorrect'⊃'correct', 'mismatch'⊃'match', 'not accurate'⊃'accurate'"}
    for p in (0.05, 0.1, 0.2, 0.4):
        for m in (2, 5, 10):
            out["G_bridge"].append(bridge_sim(p, m))
    path = ROOT / "analysis" / "code_logic_toy_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out["E_summary"], ensure_ascii=False))
    for r in out["E_fallback"]:
        print(r["scored"], r["intended"], r["response"][:70])
    for r in out["G_bridge"]:
        print(r)
    print("wrote", path)


if __name__ == "__main__":
    main()
