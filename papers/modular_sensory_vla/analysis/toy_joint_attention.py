"""토이: 공동 교차 모달 self-attention에서 "동결된 행동 스트림"이 새 센서 스트림의 영향을 받는 정도.

논문 3.2절의 1단계는 행동 전문가 파라미터를 동결하고 무작위 초기화된 센서 스트림만 학습한다. 그런데 3.1절의 공동
어텐션은 모든 스트림의 K, V를 한 소프트맥스에 넣으므로, 행동 스트림의 출력은 파라미터가 동결되어도 센서 토큰의
K, V에 따라 달라진다. 이 토이는 한 층, 한 헤드의 어텐션에서 (a) 행동 토큰만 있을 때와 (b) 무작위 센서 토큰을
붙였을 때 행동 스트림 출력의 상대 변화를 잰다. (c) 센서 쪽 V 투영을 0으로 초기화해도 소프트맥스 재정규화로 행동 값이 희석되어
변화가 남는지 함께 본다(행동 쿼리가 센서 키를 보지 못하게 막는 마스크만 변화를 0으로 만든다).
무작위 가중치와 정규 입력만 쓰므로, 실제 학습된 가중치에서의 크기를 말해 주지는 않는다. 구조적 성질만 본다.

실행: python3 analysis/toy_joint_attention.py   (numpy, 1초 미만) → analysis/toy_joint_attention_results.json
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_joint_attention_results.json"
D, NA = 64, 16  # 은닉 차원, 행동 토큰 수 (논문은 두 값을 적지 않으므로 임의 값)


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def run(ns, seed, v_zero_init=False):
    rng = np.random.default_rng(seed)
    s = 1 / np.sqrt(D)
    Wq, Wk, Wv = (rng.normal(0, s, (D, D)) for _ in range(3))  # 동결된 행동 스트림
    Sk, Sv = rng.normal(0, s, (D, D)), rng.normal(0, s, (D, D))  # 무작위 초기화 센서 스트림
    if v_zero_init:
        Sv = np.zeros((D, D))
    xa = rng.normal(0, 1, (NA, D))
    xs = rng.normal(0, 1, (ns, D))
    q, ka, va = xa @ Wq, xa @ Wk, xa @ Wv
    base = softmax(q @ ka.T / np.sqrt(D)) @ va
    k = np.concatenate([ka, xs @ Sk])
    v = np.concatenate([va, xs @ Sv])
    att = softmax(q @ k.T / np.sqrt(D))
    joint = att @ v
    rel = np.linalg.norm(joint - base) / np.linalg.norm(base)
    mass = att[:, NA:].sum(-1).mean()
    return rel, mass


res = {"setup": {"D": D, "n_action_tokens": NA, "seeds": 200, "note": "무작위 가중치, 구조적 성질만"}, "rows": []}
for ns in (1, 4, 16, 64):
    rels, masses = zip(*[run(ns, s) for s in range(200)])
    rz, mz = zip(*[run(ns, s, v_zero_init=True) for s in range(200)])
    res["rows"].append({
        "n_sensor_tokens": ns,
        "attention_mass_on_sensor_tokens": round(float(np.mean(masses)), 3),
        "relative_change_of_action_output": round(float(np.mean(rels)), 3),
        "relative_change_with_zero_init_sensor_V": round(float(np.mean(rz)), 3),
        "attention_mass_with_zero_init_sensor_V": round(float(np.mean(mz)), 3),
    })
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
for r in res["rows"]:
    print(r)
