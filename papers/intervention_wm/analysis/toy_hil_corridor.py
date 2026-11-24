"""토이 2: 개입 예측 머리를 actor 쪽에만 쓰는 설계가 개입 부담을 줄이는 조건 (표 형태 numpy, 1분 안팎).

복도 x = 0..L. 목표 x = G에서 행동 0(잡기)을 하면 성공 보상 10. x >= L이면 떨어뜨림(종료, 보상 없음).
스텝 보상 -0.01, 감가 0.97. BC 사전 정책은 오른쪽으로 치우쳐 있어 G를 지나 L로 가기도 한다.
운영자 규칙: x >= Z인 상태에서 확률 q로 개입해 행동 -1을 준다(개입 라벨 i = 1). 논문 부록 D의 규칙
("떨어뜨리기 직전이나 작업 공간 이탈 직전에만 누른다")을 한 줄로 줄인 모형이다.

변형 (논문 식 5와 같은 형태: J = Q - a_d*d - a_i*p, 여기서 d, p는 표 형태로 센 종료 확률, 다음 상태 개입 확률):
  noWM   : a_d = 0, a_i = 0                       (모델 없음)
  aI0    : a_d = 1.0, a_i = 0                     (논문 alpha_i = 0)
  whirl  : a_d = 1.0, a_i = 0.4                   (논문 기본값)
  bonus  : a_d = 1.0, a_i = 0, 보상에 -0.4 * p     (논문 부록 C가 기각하는 대안: 보상 보너스)
이 토이는 논문 실험의 재현이 아니다. 개입이 떨어뜨림을 막으면 종료 라벨이 희소해진다는 논리가
성립하는지, 보상 보너스와 actor 항이 어떻게 갈리는지를 작은 모형에서 볼 뿐이다.
실행: python3 analysis/toy_hil_corridor.py  → analysis/toy_hil_corridor_results.json
"""
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
L, G, X0, H = 12, 8, 2, 40
GAMMA, R_SUCC, R_STEP = 0.97, 10.0, -0.01
ACTS = (-1, 0, 1)
BC = np.log(np.array([0.1, 0.1, 0.8]))
EPS, N_SEEDS = 800, 20
VARIANTS = {"noWM": (0.0, 0.0, 0.0), "aI0": (1.0, 0.0, 0.0), "whirl": (1.0, 0.4, 0.0), "bonus": (1.0, 0.0, 0.4)}


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def step(x, a_idx, Z, q, rng, operator=True):
    """한 스텝. 반환: 다음 x, 보상, 종료(떨어뜨림), 성공, 실행 행동 인덱스, 다음 상태 개입 여부."""
    a = a_idx
    if x == G and a == 1:
        return x, R_SUCC, False, True, a, False
    nx = max(0, x + ACTS[a])
    if nx >= L:
        return nx, R_STEP, True, False, a, False
    return nx, R_STEP, False, False, a, False


def run(seed, Z, q, a_d, a_i, beta):
    rng = np.random.default_rng(seed)
    th = np.tile(BC, (L + 1, 1)).copy()
    Q = np.zeros((L + 1, 3))
    dh = np.zeros((L + 1, 3))   # 종료 확률 머리
    ph = np.zeros((L + 1, 3))   # 다음 상태 개입 확률 머리
    fr, drops_train = [], 0
    ep_u, ep_T = [], []
    for ep in range(EPS):
        x, u, T = X0, 0, 0
        for t in range(H):
            pi = softmax(th[x])
            a = rng.choice(3, p=pi)
            # 개입: 이전 스텝 라벨이 아니라 현재 상태에서의 운영자 제어
            intervened = x >= Z and rng.random() < q
            if intervened:
                a = 0  # 행동 -1
                u += 1
            nx, r, drop, succ, a, _ = step(x, a, Z, q, rng)
            T += 1
            nint = float(nx >= Z and nx < L)  # 다음 상태가 개입 구역인가 (확률 q로 실제 개입; 라벨 평균은 q로 수렴)
            lab_i = nint * q
            done = drop or succ
            dh[x, a] += 0.1 * (float(drop) - dh[x, a])
            ph[x, a] += 0.1 * (lab_i - ph[x, a])
            rr = r - beta * ph[x, a]
            v_next = 0.0 if done else float(softmax(th[nx]) @ Q[nx])
            Q[x, a] += 0.2 * (rr + GAMMA * v_next - Q[x, a])
            # actor: 기대 J의 소프트맥스 기울기
            J = Q[x] - a_d * dh[x] - a_i * ph[x]
            pi_x = softmax(th[x])
            th[x] += 0.02 * pi_x * (J - pi_x @ J)
            if done:
                drops_train += int(drop)
                break
            x = nx
        ep_u.append(u)
        ep_T.append(T)
    ep_u, ep_T = np.array(ep_u), np.array(ep_T)
    roll = np.array([ep_u[max(0, k - 4):k + 1].sum() / ep_T[max(0, k - 4):k + 1].sum() for k in range(EPS)])
    # 자율 평가(개입 없음), 500회
    outc = {"succ": 0, "drop": 0, "timeout": 0}
    for _ in range(500):
        x = X0
        for t in range(H):
            a = rng.choice(3, p=softmax(th[x]))
            x, r, drop, succ, a, _ = step(x, a, Z, q, rng)
            if succ:
                outc["succ"] += 1
                break
            if drop:
                outc["drop"] += 1
                break
        else:
            outc["timeout"] += 1
    return roll, drops_train, ep_u.sum(), ep_T.sum(), outc


def summarize(Z, q):
    res = {}
    for name, (a_d, a_i, beta) in VARIANTS.items():
        rolls, drops, us, Ts, outs = [], [], [], [], []
        for s in range(N_SEEDS):
            roll, dr, u, T, o = run(1000 + s, Z, q, a_d, a_i, beta)
            rolls.append(roll)
            drops.append(dr)
            us.append(u)
            Ts.append(T)
            outs.append(o)
        rolls = np.array(rolls)
        first = [int(np.argmax(r <= 0.01)) if (r <= 0.01).any() else None for r in rolls]
        res[name] = {
            "operator_step_share_total": round(float(np.sum(us) / np.sum(Ts)), 3),
            "rolling_final_mean": round(float(rolls[:, -1].mean()), 3),
            "rolling_curve_every10": [round(float(x), 4) for x in rolls.mean(axis=0)[::10]],
            "rolling_at_ep200": round(float(rolls[:, 200].mean()), 3),
            "episodes_to_rolling_le_0.01_median": (float(np.median([f for f in first if f is not None])) if any(f is not None for f in first) else None),
            "n_seeds_reaching_le_0.01": int(sum(f is not None for f in first)),
            "train_drops_per_seed": round(float(np.mean(drops)), 1),
            "eval_success": round(float(np.mean([o["succ"] for o in outs]) / 500), 3),
            "eval_drop": round(float(np.mean([o["drop"] for o in outs]) / 500), 3),
            "eval_timeout": round(float(np.mean([o["timeout"] for o in outs]) / 500), 3),
        }
    return res


if __name__ == "__main__":
    t0 = time.time()
    out = {"setup": {"L": L, "G": G, "x0": X0, "H": H, "episodes": EPS, "seeds": N_SEEDS, "gamma": GAMMA}}
    for key, (Z, q) in {"Z10_q0.85": (10, 0.85), "Z10_q0.3": (10, 0.3), "Z9_q0.85": (9, 0.85)}.items():
        out[key] = {"Z": Z, "q": q, "variants": summarize(Z, q)}
        print(key, json.dumps(out[key]["variants"], ensure_ascii=False))
    out["seconds"] = round(time.time() - t0, 1)
    (ROOT / "analysis" / "toy_hil_corridor_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("seconds", out["seconds"])
