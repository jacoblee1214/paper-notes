"""GNR의 가설 toy: 실현 가능한 해가 조건 c에 매끄럽게 의존하는 저차원 manifold 근처에 있으면,
풀린 시연에서 배운 조건부 생성기가 시연마다 처음부터 찾는 표본 기반 최적화보다 적은 표본으로 푼다.

설정 (물리 시뮬레이션이 아니라 기하만 남긴 장난감):
- 조건 c ∈ R^2 (사람 시연의 요약: 물체 크기·접근 각 같은 것). 학습 범위 [-1, 1]^2, OOD는 c1 ∈ [1, 1.6].
- IK 기준 제어 ū(c) = W·φ(c) ∈ R^D (D = 24). 실현 가능한 잔차 Δ*(c) = A·h(c) + 모드(두 가지 파지)·B.
- 성공: 어떤 모드에 대해 ‖Δ − Δ*_m(c)‖ < ε. 표본 하나를 평가하는 것이 시뮬레이션 롤아웃 한 번에 해당.
- MPC: Δ = 0(IK)에서 시작하는 annealed CEM(반복 M × 표본 ns). 처음 성공에서 멈춤.
- 생성기 대용(GNR proxy): 학습 조건의 k-최근접 이웃에서 잔차를 가져와 국소 선형 보정 + 잡음으로 S개를 뽑음.
  flow matching이 아니라 가장 단순한 비모수 조건부 생성기다. 결론은 "manifold가 매끄럽고 학습 범위 안이면
  이웃의 해가 좋은 제안이 된다"는 정도로만 읽는다.
- 잔차(ū + Δ) 대 절대(이웃의 u = ū(c_j) + Δ_j 그대로) 비교로 부록 E.4의 잔차 예측 이점을 본다.
실행: python3 analysis/manifold_toy.py  (CPU 단일 스레드, 수십 초)
"""
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D, EPS = 24, 0.45
rng0 = np.random.default_rng(0)
W = rng0.normal(size=(D, 6)) * 1.2          # IK 기준은 c에 따라 크게 변함
A = rng0.normal(size=(D, 3)) * 0.9          # 실현 가능 잔차의 저차원 구조 (3차원 잠재)
B = rng0.normal(size=D) * 0.6               # 두 번째 파지 모드의 오프셋


def phi(c):
    return np.stack([c[:, 0], c[:, 1], c[:, 0] * c[:, 1], np.sin(2 * c[:, 0]), np.cos(1.5 * c[:, 1]), c[:, 0] ** 2], 1)


def h(c):
    return np.stack([np.sin(1.7 * c[:, 0]) + 0.3 * c[:, 1], np.cos(1.3 * c[:, 1]) * c[:, 0], 0.5 * c[:, 0] * c[:, 1]], 1)


def ubar(c):
    return phi(c) @ W.T


def targets(c):
    base = h(c) @ A.T
    return np.stack([base, base + B], 1)  # (n, 2, D)


def feasible(c, delta):
    """c (n,2), delta (n,k,D) -> (n,k) bool."""
    t = targets(c)  # (n,2,D)
    d = np.linalg.norm(delta[:, :, None, :] - t[:, None, :, :], axis=-1).min(-1)
    return d < EPS


def mpc(c, ns, M, rng, sigma0=1.2, decay=0.6, elite=0.1):
    """annealed CEM on Δ from 0. 반환: 성공 여부, 쓴 표본 수, 최종 해. 비용(표본)은 처음 성공에서 멈춘 값."""
    n = len(c)
    mu, sig = np.zeros((n, D)), np.full(n, sigma0)
    t = targets(c)
    done = np.zeros(n, bool)
    used = np.zeros(n, int)
    sol = np.zeros((n, D))
    for it in range(M):
        z = mu[:, None, :] + sig[:, None, None] * rng.normal(size=(n, ns, D))
        dist = np.linalg.norm(z[:, :, None, :] - t[:, None], axis=-1).min(-1)  # 비용 = 가장 가까운 모드까지 거리
        ok = dist < EPS
        first = np.where(ok.any(1), ok.argmax(1), -1)
        newly = (~done) & (first >= 0)
        used[~done] += np.where(newly[~done], first[~done] + 1, ns)
        sol[newly] = z[newly, first[newly]]
        done |= newly
        k = max(1, int(elite * ns))
        idx = np.argsort(dist, 1)[:, :k]
        mu = np.take_along_axis(z, idx[:, :, None], 1).mean(1)
        sig = sig * decay
    return done, used, sol


def gnr_proxy(c_tr, d_tr, c_te, S, rng, k=12, sigma=0.06, residual=True):
    """비모수 조건부 생성기: 대상 조건의 k-최근접 해를 모드별로 묶고(첫 이웃과의 거리로 2분할),
    모드마다 국소 선형 회귀 y ≈ y_0 + J (c - c_0)로 대상 조건의 평균을 예측한 뒤, 모드를 크기 비례로 골라
    등방 잡음 sigma를 더한다. residual=False면 같은 일을 절대 제어 u = ū + Δ에 대해 한다."""
    y_tr = d_tr if residual else ubar(c_tr) + d_tr
    dd = np.linalg.norm(c_te[:, None, :] - c_tr[None], axis=-1)
    nn = np.argsort(dd, 1)[:, :k]
    n = len(c_te)
    out = np.empty((n, S, D))
    ub = ubar(c_te)
    for i in range(n):
        idx = nn[i]
        Y, C = y_tr[idx], c_tr[idx]
        # 모드 분할: 잔차 공간에서 첫 이웃과 가까운 쪽 / 먼 쪽 (모드 간격 ≈ ‖B‖ ≫ 모드 안 변화)
        ref = d_tr[idx[0]]
        far = np.linalg.norm(d_tr[idx] - ref, axis=1) > 0.5 * np.linalg.norm(B)
        means, sizes = [], []
        for grp in (~far, far):
            if grp.sum() == 0:
                continue
            Cg, Yg = C[grp], Y[grp]
            X = np.hstack([np.ones((len(Cg), 1)), Cg - c_te[i]])
            coef, *_ = np.linalg.lstsq(X, Yg, rcond=None) if len(Cg) >= 3 else (np.vstack([Yg.mean(0), np.zeros((2, D))]), None)
            means.append(coef[0])  # 대상 조건에서의 예측값
            sizes.append(grp.sum())
        means, p = np.array(means), np.array(sizes, float) / sum(sizes)
        m = rng.choice(len(means), size=S, p=p)
        y = means[m] + sigma * rng.normal(size=(S, D))
        out[i] = y if residual else y - ub[i]
    return out


def pass_curve(succ):
    """succ (n,S) -> pass@s for s in Ss."""
    Ss = [1, 4, 16, 64, 256]
    return {s: round(100 * float(succ[:, :s].any(1).mean()), 1) for s in Ss if s <= succ.shape[1]}


def main() -> None:
    rng = np.random.default_rng(1)
    out = {"setup": {"D": D, "eps": EPS, "latent_dim": 3, "modes": 2,
                     "mpc_ns": 300, "mpc_M": 10, "mpc_budget": 3000}}
    # 시드 데이터: 학습 범위 조건에서 MPC로 푼 것만
    n_pool = 1600
    c_pool = rng.uniform(-1, 1, size=(n_pool, 2))
    ok, used, sol = mpc(c_pool, 300, 10, rng)
    out["seed_mpc"] = {"pool": n_pool, "mpc_success_pct": round(100 * ok.mean(), 1),
                       "mean_samples_used_success": round(float(used[ok].mean()), 1)}
    # 보간 OOD: 학습 시드에서 원판 영역(중심 (0.3,-0.2), 반지름 0.35)을 뺀다
    center, rad = np.array([0.3, -0.2]), 0.35
    keep = ok & (np.linalg.norm(c_pool - center, axis=1) > rad)
    c_ok, d_ok = c_pool[keep], sol[keep]
    out["seed_mpc"]["seeds_outside_holdout_disk"] = int(keep.sum())

    c_in = rng.uniform(-1, 1, size=(600, 2))
    c_in = c_in[np.linalg.norm(c_in - center, axis=1) > rad][:400]
    ang, rr = rng.uniform(0, 2 * np.pi, 400), rad * np.sqrt(rng.uniform(0, 1, 400))
    c_hole = center + np.stack([rr * np.cos(ang), rr * np.sin(ang)], 1)
    c_ood = np.stack([rng.uniform(1.0, 1.6, 400), rng.uniform(-1, 1, 400)], 1)
    sets = {"in_dist": c_in, "interp_ood": c_hole, "extrap_ood": c_ood}
    mpc_ok = {}
    for name, cc in sets.items():
        okm, usedm, _ = mpc(cc, 300, 10, rng)
        mpc_ok[name] = okm
        out[f"mpc_{name}"] = {"success_pct": round(100 * okm.mean(), 1), "mean_samples": round(float(usedm.mean()), 1)}
    okm, usedm, _ = mpc(c_in, 32, 6, rng)
    out["mpc_in_dist_budget192"] = {"success_pct": round(100 * okm.mean(), 1)}

    # GNR proxy: 데이터 크기 N × 표본 S, 잔차 vs 절대, 세 시험 집합
    res = {}
    for N in (25, 100, 400, len(c_ok)):
        idx = rng.permutation(len(c_ok))[:N]
        row = {}
        for name, cc in sets.items():
            for mode, residual in (("residual", True), ("absolute", False)):
                succ = feasible(cc, gnr_proxy(c_ok[idx], d_ok[idx], cc, 256, rng, residual=residual))
                row[f"{name}/{mode}"] = pass_curve(succ)
                if mode == "residual":
                    m = mpc_ok[name]
                    row[f"{name}/residual_split_S256"] = {
                        "on_mpc_success": round(100 * float(succ[m].any(1).mean()), 1) if m.any() else None,
                        "on_mpc_failure": round(100 * float(succ[~m].any(1).mean()), 1) if (~m).any() else None,
                        "n_mpc_failure": int((~m).sum())}
        res[str(N)] = row
    out["gnr_proxy"] = res
    # 모드 평균(회귀) 대 생성: 두 모드의 평균을 내면 어느 모드에도 속하지 않음
    t = targets(c_in)
    mean_pred = t.mean(1)
    out["regression_to_mean"] = {"success_pct": round(100 * float(feasible(c_in, mean_pred[:, None, :]).mean()), 1),
                                 "half_mode_gap_over_eps": round(float(np.linalg.norm(B) / 2 / EPS), 2)}

    path = ROOT / "analysis" / "manifold_toy_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False)[:3000])
    print("wrote", path)


if __name__ == "__main__":
    main()
