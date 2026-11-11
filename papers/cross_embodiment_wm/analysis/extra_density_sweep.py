"""추가 검증 1: 입자 밀도만 바꿨을 때 sum·mean 집계의 교차 밀도 일반화.

particle_toy.py의 2D 팔(링크 0.14·0.12 m), 접촉 반경·물체 입자·게이트를 그대로 쓴다. 바꾸는 것은 입자 간격 h뿐이다.
  h ∈ {0.010, 0.015, 0.025, 0.040, 0.060} m  →  입자 밀도 1/h가 6배 차이 (팔 형상은 같음)
참 세계 두 가지 (둘 다 가정, 논문 모형 아님)
  M (mean truth): particle_toy.true_step (차수 불변 평균 + 게이트)  — 기존 toy의 참 세계
  S (sum truth) : κ_s · Σ_j w_ij δ_j  (차수에 비례). κ_s는 h = 0.025에서 sum과 mean의 중앙값 크기가 같도록 한 번 정함
학습 예측기: 스칼라 κ 하나 (최소제곱). 집계는 sum 또는 mean.
측정
  (1) 학습 밀도 h_tr × 시험 밀도 h_te 행렬의 상대 MSE  (Σ(κf − t)² / Σt²)
  (2) 전 밀도를 합쳐 학습한 단일 κ의 밀도별 상대 MSE (논문의 "여러 손을 섞어 학습"에 해당)
  (3) 입자 수가 밀도에 비례함을 확인 (평균 이웃 수 = 접촉 가중합의 0이 아닌 입자 수)
에피소드: 손당 20개, 길이 40 스텝, 시드 고정. 재현 가능성: numpy 단일 스레드.
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
import time
from pathlib import Path

import numpy as np

import particle_toy as pt

OUT_JSON = Path(__file__).with_name("extra_density_sweep_results.json")
LENGTHS = (0.14, 0.12)
HS = [0.010, 0.015, 0.025, 0.040, 0.060]
N_EP, T = 20, 40


def truth_sum(obj, hand, dh, kappa_s):
    return kappa_s * pt.sum_features(obj, hand, dh)


def build(h, rng, truth):
    """에피소드 집합. truth='mean' 또는 'sum'. 반환: [(obj(M,2), hand(N,2), dh(N,2), dobj(M,2))]"""
    data = []
    for _ in range(N_EP):
        qs, P = pt.sample_hand(rng, LENGTHS, h, T)
        ang = np.pi * (3 - np.sqrt(5))
        idx = np.arange(pt.M_OBJ)
        rr = 0.05 * np.sqrt((idx + 0.5) / pt.M_OBJ)
        base_obj = np.stack([rr * np.cos(idx * ang), rr * np.sin(idx * ang)], -1)
        center = P[0, -1] + rng.uniform(-0.03, 0.03, 2) + np.array([0.02, 0.0])
        obj = (base_obj + center)[None]
        for t in range(T - 1):
            dh = P[t + 1] - P[t]
            if truth == "mean":
                dobj = pt.true_step(obj, P[t][None], dh[None])[0]
            else:
                dobj = truth_sum(obj, P[t][None], dh[None], KAPPA_S)[0]
            data.append((obj[0].copy(), P[t].copy(), dh.copy(), dobj.copy()))
            obj = obj + dobj[None]
    return data


def feats_targs(data, kind):
    F, Tt = [], []
    for obj, hand, dh, dobj in data:
        o, hd, dd = obj[None], hand[None], dh[None]
        f = pt.sum_features(o, hd, dd) if kind == "sum" else pt.mean_features(o, hd, dd)
        F.append(f[0].ravel())
        Tt.append(dobj.ravel())
    return np.concatenate(F), np.concatenate(Tt)


def fit_k(F, Tt):
    return float(F @ Tt / max(F @ F, 1e-12))


def rel_mse(k, F, Tt):
    return float(np.sum((k * F - Tt) ** 2) / np.sum(Tt ** 2))


def median_neighbors(data):
    """접촉 가중합이 0이 아닌(반경 안) 물체-손 입자 쌍의 평균 개수, 물체 입자당."""
    vals = []
    for obj, hand, _, _ in data[:50]:
        w = pt.contact_weights(obj[None], hand[None])[0]
        vals.append(float((w > 0).sum(1).mean()))
    return float(np.mean(vals))


KAPPA_S = 1.0


def main():
    global KAPPA_S
    t0 = time.time()
    out = {"setup": {"lengths_m": list(LENGTHS), "spacings_m": HS, "n_episodes_per_hand": N_EP, "T": T,
                     "R_contact_m": pt.R_CONTACT, "gate": pt.GATE, "M_obj": pt.M_OBJ, "seed": 0}}

    data_mean = {}
    for i, h in enumerate(HS):
        rng = np.random.default_rng(100 + i)
        data_mean[h] = build(h, rng, "mean")
    # κ_s: h = 0.025에서 sum과 mean의 중앙값 크기를 맞춤
    Fs, Ft = feats_targs(data_mean[0.025], "sum")
    Fm, _ = feats_targs(data_mean[0.025], "mean")
    KAPPA_S = float(np.sqrt(np.mean(Fm ** 2)) / max(np.sqrt(np.mean(Fs ** 2)), 1e-12))
    out["setup"]["kappa_s_truth"] = round(KAPPA_S, 5)

    data_sum = {}
    for i, h in enumerate(HS):
        rng = np.random.default_rng(100 + i)
        data_sum[h] = build(h, rng, "sum")

    out["neighbors_per_object_particle"] = {str(h): round(median_neighbors(data_mean[h]), 2) for h in HS}
    out["particles_per_hand"] = {str(h): int(data_mean[h][0][1].shape[0]) for h in HS}

    for truth, D in (("mean_truth", data_mean), ("sum_truth", data_sum)):
        res = {}
        feats = {h: {k: feats_targs(D[h], k) for k in ("sum", "mean")} for h in HS}
        for kind in ("sum", "mean"):
            M = np.zeros((len(HS), len(HS)))
            for a, htr in enumerate(HS):
                Ftr, Ttr = feats[htr][kind]
                k = fit_k(Ftr, Ttr)
                for b, hte in enumerate(HS):
                    Fte, Tte = feats[hte][kind]
                    M[a, b] = rel_mse(k, Fte, Tte)
            res[f"{kind}_train_by_test_rel_mse"] = {str(htr): [round(float(v), 4) for v in M[a]] for a, htr in enumerate(HS)}
            # 같은 밀도끼리(대각) 평균 대비 교차 밀도 최대 배율
            off = M[~np.eye(len(HS), dtype=bool)]
            diag = np.diag(M)
            res[f"{kind}_cross_over_same_median"] = round(float(np.median(off) / np.median(diag)), 2)
            # 전 밀도 합동 학습 κ
            Fall = np.concatenate([feats[h][kind][0] for h in HS])
            Tall = np.concatenate([feats[h][kind][1] for h in HS])
            kall = fit_k(Fall, Tall)
            res[f"{kind}_pooled_kappa"] = round(kall, 4)
            res[f"{kind}_pooled_rel_mse_by_test"] = {str(h): round(rel_mse(kall, *feats[h][kind]), 4) for h in HS}
            res[f"{kind}_pooled_kappa_by_hand_fit"] = {str(h): round(fit_k(*feats[h][kind]), 4) for h in HS}
        out[truth] = res

    # 밀도 1/h와 sum 특징 크기의 비례 관계: 기대 로그-로그 기울기 1
    sz = [float(np.sqrt(np.mean(feats_targs(data_sum[h], "sum")[0] ** 2))) for h in HS]
    out["sum_feature_scale_vs_density_loglog_slope"] = round(float(np.polyfit(np.log([1 / h for h in HS]), np.log(sz), 1)[0]), 3)
    mz = [float(np.sqrt(np.mean(feats_targs(data_sum[h], "mean")[0] ** 2))) for h in HS]
    out["mean_feature_scale_vs_density_loglog_slope"] = round(float(np.polyfit(np.log([1 / h for h in HS]), np.log(mz), 1)[0]), 3)
    out["runtime_s"] = round(time.time() - t0, 1)
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
