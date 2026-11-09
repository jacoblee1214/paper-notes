"""latent action 정렬(identifiability) toy (CPU, numpy, 단일 스레드). 실행: python3 analysis/alignment_toy.py

질문: 재구성 손실만으로 사람·로봇의 latent action이 같은 좌표에 정렬되는가?

모형(선형): 행동 a ∈ R^2, 영상 공간의 움직임 Δo_e = R_e a + 잡음 (m = 8).
 - 로봇: R_r (직교 열 2개). 사람: R_h = R_r Q(θ) + η R_⊥  (θ = 시점 차이로 이미지 평면에서 움직임이 돌아 보이는 각도,
   η = 손이 그리퍼와 다르게 보이는 정도).
 - 인코더 z = E_e Δo (공유 또는 embodiment별), 디코더 Δo ≈ D_e z (embodiment별: 첫 프레임을 보는 디코더는 embodiment를 안다).
 - DreamX처럼 로봇 클립에만 행동 보조 손실 λ‖H z − a‖² (H 공유).
지표:
 - probe R²: embodiment마다 z → a 선형 회귀 (LARYBench식, 탐침을 따로 학습)
 - 전이 오차: 사람 z_h를 로봇 행동 head로 풀었을 때 ‖H z_h − a‖² / ‖a‖² (DreamX 식 5의 action transfer)
조건:
 S  공유 인코더 (가중치 공유가 유일한 정렬 압력)
 I  embodiment별 인코더 (강한 비선형 인코더가 embodiment를 알아보는 경우의 근사)
 P  embodiment별 인코더 + 짝지은 사람·로봇 클립 5%에 정렬 손실 ‖z_h − z_r‖² (DreamX는 쓰지 않는 신호)
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
M, D, K = 8, 2, 2
N = 1000
THETAS = [0, 15, 30, 60, 90, 180]
SEEDS = range(5)
ETA = 0.3


def rot(deg):
    t = np.deg2rad(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def renderers(theta, rng):
    Qm, _ = np.linalg.qr(rng.normal(size=(M, M)))
    Rr = Qm[:, :2]
    Rperp = Qm[:, 2:4]
    Rh = Rr @ rot(theta) + ETA * Rperp
    Rh = Rh / np.linalg.norm(Rh, axis=0, keepdims=True)
    return Rr, Rh


def train(Rr, Rh, mode, seed, steps=2500, lr=0.02, lam=1.0, wd=1e-3, pair_frac=0.05, noise=0.05, pair_w=5.0):
    rng = np.random.default_rng(1000 + seed)
    a_r, a_h = rng.normal(size=(N, D)), rng.normal(size=(N, D))
    do_r = a_r @ Rr.T + noise * rng.normal(size=(N, M))
    do_h = a_h @ Rh.T + noise * rng.normal(size=(N, M))
    npair = int(pair_frac * N)
    a_p = rng.normal(size=(npair, D))
    dp_r = a_p @ Rr.T + noise * rng.normal(size=(npair, M))
    dp_h = a_p @ Rh.T + noise * rng.normal(size=(npair, M))

    s = 0.1
    P = {"Er": s * rng.normal(size=(K, M)), "Dr": s * rng.normal(size=(M, K)), "Dh": s * rng.normal(size=(M, K)),
         "H": s * rng.normal(size=(D, K))}
    P["Eh"] = P["Er"] if mode == "S" else s * rng.normal(size=(K, M))
    mom = {k: np.zeros_like(v) for k, v in P.items()}
    vel = {k: np.zeros_like(v) for k, v in P.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8

    for t in range(1, steps + 1):
        G = {k: np.zeros_like(v) for k, v in P.items()}
        zr = do_r @ P["Er"].T
        zh = do_h @ P["Eh"].T
        # 재구성 (embodiment별 디코더)
        er = zr @ P["Dr"].T - do_r
        eh = zh @ P["Dh"].T - do_h
        G["Dr"] += 2 * er.T @ zr / N
        G["Dh"] += 2 * eh.T @ zh / N
        gzr = 2 * er @ P["Dr"] / N
        gzh = 2 * eh @ P["Dh"] / N
        # 로봇 행동 보조 손실
        ea = zr @ P["H"].T - a_r
        G["H"] += 2 * lam * ea.T @ zr / N
        gzr += 2 * lam * ea @ P["H"] / N
        G["Er"] += gzr.T @ do_r
        G["Eh"] += gzh.T @ do_h
        # 짝 정렬 손실
        if mode == "P":
            zpr = dp_r @ P["Er"].T
            zph = dp_h @ P["Eh"].T
            dz = zph - zpr
            G["Eh"] += 2 * pair_w * dz.T @ dp_h / npair
            G["Er"] -= 2 * pair_w * dz.T @ dp_r / npair
        for k in G:
            G[k] += 2 * wd * P[k]
        if mode == "S":
            G["Er"] = G["Er"] + G["Eh"]
        for k in ("Er", "Dr", "Dh", "H") + (() if mode == "S" else ("Eh",)):
            mom[k] = b1 * mom[k] + (1 - b1) * G[k]
            vel[k] = b2 * vel[k] + (1 - b2) * G[k] ** 2
            P[k] -= lr * (mom[k] / (1 - b1 ** t)) / (np.sqrt(vel[k] / (1 - b2 ** t)) + eps)
        if mode == "S":
            P["Eh"] = P["Er"]

    # 평가
    rng_t = np.random.default_rng(99)
    a = rng_t.normal(size=(2000, D))
    zr = (a @ Rr.T) @ P["Er"].T
    zh = (a @ Rh.T) @ P["Eh"].T

    def probe_r2(z, y):
        X = np.c_[z, np.ones(len(z))]
        w, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ w
        return float(1 - (res ** 2).sum() / ((y - y.mean(0)) ** 2).sum())

    recon_r = 1 - np.sum((zr @ P["Dr"].T - a @ Rr.T) ** 2) / np.sum((a @ Rr.T) ** 2)
    return {
        "probe_r2_robot": round(probe_r2(zr, a), 4),
        "probe_r2_human": round(probe_r2(zh, a), 4),
        "recon_r2_robot": round(float(recon_r), 4),
        "robot_action_err": round(float(np.sum((zr @ P["H"].T - a) ** 2) / np.sum(a ** 2)), 4),
        "transfer_action_err": round(float(np.sum((zh @ P["H"].T - a) ** 2) / np.sum(a ** 2)), 4),
    }


def main() -> None:
    res = {}
    for mode in ("S", "I", "P"):
        res[mode] = {}
        for th in THETAS:
            runs = []
            for sd in SEEDS:
                Rr, Rh = renderers(th, np.random.default_rng(sd))
                runs.append(train(Rr, Rh, mode, sd))
            agg = {k: {"mean": round(float(np.mean([r[k] for r in runs])), 4),
                       "min": round(float(np.min([r[k] for r in runs])), 4),
                       "max": round(float(np.max([r[k] for r in runs])), 4)} for k in runs[0]}
            res[mode][str(th)] = agg
            print(mode, th, {k: v["mean"] for k, v in agg.items()})
    # 이론 기준선: 공유 선형 인코더는 이미지 공간 회전을 보정할 수 없으므로 전이 오차 ≈ ‖Q(θ) − I‖²/2 = 2(1 − cos θ) (η 영향 제외)
    theory = {str(th): round(2 * (1 - np.cos(np.deg2rad(th))), 3) for th in THETAS}
    out = {
        "setup": {"m": M, "d": D, "k": K, "N_per_embodiment": N, "eta_human_nuisance": ETA, "seeds": len(SEEDS),
                  "steps": 2500, "optimizer": "Adam lr 0.02, weight decay 1e-3", "robot_action_aux_lambda": 1.0,
                  "pair_frac_for_P": 0.05, "pair_loss_weight_for_P": 5.0},
        "results": res,
        "theory_shared_encoder_rotation_err": theory,
        "reading": [
            "probe R²는 세 조건, 모든 θ에서 거의 1이다. 탐침을 embodiment마다 따로 학습하는 LARYBench식 평가는 정렬 여부를 재지 못한다",
            "공유 인코더(S)는 θ = 0에서 잘 전이되지만 θ가 커지면 전이 오차가 커진다. 정렬은 같은 행동이 이미지에서 비슷하게 움직여 보일 때만 생긴다",
            "embodiment별 인코더(I)는 θ와 무관하게 전이가 임의적이다. 재구성 손실은 embodiment마다 다른 좌표계를 허용한다",
            "짝지은 클립 5%(P)를 쓰면 θ와 무관하게 정렬된다. DreamX는 이 신호 없이 정렬된다고 주장하므로, 정렬은 가중치 공유와 시점 유사성에서 나온다고 보는 것이 자연스럽다",
        ],
    }
    path = ROOT / "analysis" / "alignment_toy_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("wrote", path)


if __name__ == "__main__":
    main()
