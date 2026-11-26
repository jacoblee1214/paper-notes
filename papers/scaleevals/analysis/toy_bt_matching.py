"""토이 1: 소프트 승리 Bradley–Terry(BT)가 평균 작업 진행도(TP)와 얼마나 다른 정보를 주는지, 그리고 Hessian 표준오차가 맞는지.

논문 3.3절의 식을 그대로 구현한다: 목표 y = (Pi + eps)/(Pi + Pj + 2 eps), eps = 0.01, P는 퍼센트,
가중 로지스틱 손실 + L2, 앵커 π0.5 = 1539, Elo 척도 400/ln 10.

가정(논문에 없음, 아래 값을 바꿔 가며 민감도를 본다):
  - 롤아웃별 TP는 Beta 분포(평균 = 표 1의 TP, 집중도 kappa). kappa가 작을수록 0과 1 근처로 몰림.
  - 매칭은 정책 쌍마다 무작위 일대일(헝가리안 유사도 대신), 쌍당 223개(표 8 Pairs 합 4466의 절반 2233을 10쌍으로 나눈 값).
  - 가중치 w = 1, 정규화 lambda = 1e-3 (논문 값 미기재).
이 토이는 논문 데이터를 재현하지 않는다. "비교가 롤아웃을 재사용하는 구조에서 Hessian SE가 표본 변동을 얼마나 반영하는가"와
"BT 격차가 평균 TP 로그 비와 얼마나 가까운가"만 본다.

실행: python3 analysis/toy_bt_matching.py   (numpy, 수 초)
출력: analysis/toy_bt_matching_results.json
"""
import json
import math
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_bt_matching_results.json"
POL = ["DreamZero", "pi0.5", "GR00T N1.7", "MolmoAct2", "GR00T N1.6"]
TP = np.array([38.5, 30.7, 23.2, 17.3, 15.5]) / 100
REP_BT = np.array([1580, 1539, 1500, 1456, 1453.0])
ELO = 400 / math.log(10)
ANCHOR = 1  # pi0.5
EPS, LAM, M_PER_PAIR, N = 0.01, 1e-3, 223, 350
PAIRS = [(i, j) for i in range(5) for j in range(i + 1, 5)]


def draw(rng, kappa):
    return np.stack([rng.beta(m * kappa, (1 - m) * kappa, N) for m in TP])  # (5, N), 0~1


def make_matches(rng, tp):
    I, J, Pi, Pj = [], [], [], []
    for i, j in PAIRS:
        a = rng.choice(tp.shape[1], M_PER_PAIR, replace=False)
        b = rng.choice(tp.shape[1], M_PER_PAIR, replace=False)
        I += [i] * M_PER_PAIR
        J += [j] * M_PER_PAIR
        Pi.append(100 * tp[i, a])
        Pj.append(100 * tp[j, b])
    Pi, Pj = np.concatenate(Pi), np.concatenate(Pj)
    y = (Pi + EPS) / (Pi + Pj + 2 * EPS)
    return np.array(I), np.array(J), y


def fit(I, J, y):
    """앵커 θ[ANCHOR] = 0 고정, 뉴턴법. 반환: Elo(앵커 1539), Hessian SE(Elo)"""
    free = [k for k in range(5) if k != ANCHOR]
    th = np.zeros(5)
    for _ in range(50):
        d = th[I] - th[J]
        p = 1 / (1 + np.exp(-d))
        g = np.zeros(5)
        np.add.at(g, I, -(y - p))
        np.add.at(g, J, (y - p))
        g += 2 * LAM * th
        H = np.zeros((5, 5))
        w = p * (1 - p)
        np.add.at(H, (I, I), w)
        np.add.at(H, (J, J), w)
        np.add.at(H, (I, J), -w)
        np.add.at(H, (J, I), -w)
        H += 2 * LAM * np.eye(5)
        Hf = H[np.ix_(free, free)]
        step = np.linalg.solve(Hf, g[free])
        th[free] -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    cov = np.linalg.inv(Hf)
    se = np.zeros(5)
    se[free] = np.sqrt(np.diag(cov))
    return 1539 + ELO * th, ELO * se


def run(kappa, reps=200, boots=200, seed=1):
    rng = np.random.default_rng(seed)
    elos, ses = [], []
    for _ in range(reps):
        tp = draw(rng, kappa)
        I, J, y = make_matches(rng, tp)
        e, s = fit(I, J, y)
        elos.append(e)
        ses.append(s)
    elos, ses = np.array(elos), np.array(ses)
    # 한 데이터셋에서 롤아웃 단위 부트스트랩
    tp0 = draw(rng, kappa)
    bs = []
    for _ in range(boots):
        idx = rng.integers(0, N, (5, N))
        tpb = np.take_along_axis(tp0, idx, axis=1)
        I, J, y = make_matches(rng, tpb)
        bs.append(fit(I, J, y)[0])
    bs = np.array(bs)
    gap_pred = [ELO * math.log(TP[i] / TP[i + 1]) for i in range(4)]
    gaps = elos[:, :-1] - elos[:, 1:]
    return {
        "kappa": kappa,
        "per_rollout_sd_TP_pp(mean over policies)": round(float(np.mean([100 * math.sqrt(m * (1 - m) / (kappa + 1)) for m in TP])), 1),
        "mean_elo": elos.mean(0).round(1).tolist(),
        "mean_adjacent_gap": gaps.mean(0).round(1).tolist(),
        "gap_if_ELO_ln_TP_ratio": [round(x, 1) for x in gap_pred],
        "mean_hessian_se": ses.mean(0).round(1).tolist(),
        "empirical_sd_over_datasets": elos.std(0, ddof=1).round(1).tolist(),
        "sd_ratio_empirical_over_hessian": (elos.std(0, ddof=1)[[0, 2, 3, 4]] / ses.mean(0)[[0, 2, 3, 4]]).round(2).tolist(),
        "bootstrap_sd_one_dataset(rollout resampling)": bs.std(0, ddof=1).round(1).tolist(),
        "P(MolmoAct2 > GR00T N1.6)": float(np.mean(elos[:, 3] > elos[:, 4])),
        "P(order equals reported order)": float(np.mean(np.all(np.diff(elos[:, [0, 1, 2, 3, 4]], axis=1) < 0, axis=1))),
        "sd_of_MA2_minus_N16_gap": float(gaps[:, 3].std(ddof=1)),
        "match_reuse": "롤아웃당 평균 재사용 횟수 = %.2f" % (M_PER_PAIR * 4 / N),
    }


def main():
    res = {"assumptions": {"matches_per_pair": M_PER_PAIR, "rollouts_per_policy": N, "lambda": LAM, "eps": EPS,
                           "matching": "무작위 일대일", "TP_distribution": "Beta(mean=표 1 TP, concentration=kappa)"},
           "runs": [run(k) for k in (0.7, 1.5, 3.0)],
           "reported": {"BT": REP_BT.tolist(), "hessian_SE": [16, 0, 17, 16, 16]}}
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    for r in res["runs"]:
        print({k: r[k] for k in ("kappa", "mean_adjacent_gap", "mean_hessian_se", "empirical_sd_over_datasets", "bootstrap_sd_one_dataset(rollout resampling)",
                                 "P(MolmoAct2 > GR00T N1.6)", "per_rollout_sd_TP_pp(mean over policies)")})


if __name__ == "__main__":
    main()
