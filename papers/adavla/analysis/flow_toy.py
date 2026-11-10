"""AdaVLA 곡률 기준 toy (CPU, numpy 1스레드). 실행: python3 analysis/flow_toy.py

설정: 2차원 가우시안 혼합 목표 x1 ~ Σ π_k N(μ_k, σ_k² I), 출발 x0 ~ N(0, I), 선형 경로 x_τ = τ x1 + (1−τ) x0.
정확한 주변 속도(marginal velocity), 유도:
  D_k(τ) = τ² σ_k² + (1−τ)²
  w_k(x,τ) ∝ π_k N(x; τ μ_k, D_k I)
  m_k(x,τ) = μ_k + (τ σ_k² − (1−τ)) / D_k · (x − τ μ_k)     # E[x1−x0 | x, k]
  v(x,τ) = Σ_k w_k m_k
(유도는 analysis/03_analysis.md §4 참조, 아래 check_velocity()로 수치 검증)

비교 방법
- Euler n단계 (NFE = n)
- 고정 RK2(중점) n단계 (NFE = 2n)
- AdaVLA-as-written: 논문 식 5, 6, 8, 9를 그대로. ε=1e-8, s=0.1, 반복 최대 10회. 한계 도달 시 Euler 점프로 마무리하고 표시.
  구현 가정: 반복마다 v_τ와 v_mid, NFE 2.
- AdaVLA-consistent (절제): 같은 분기 규칙이지만 높은 곡률 분기에서 실제 중점 τ+ŝ/2에서 속도를 한 번 더 구하고 그 값으로 ŝ만큼 갱신. 이 분기 NFE는 3(v_τ, v_mid(곡률용), v_half).

출력: analysis/flow_toy_results.json
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "flow_toy_results.json"
SEED = 0
N = 4000
EPS = 1e-8
S = 0.1
MAX_IT = 10
TOL = 0.1
CTH_GRID = [0.01, 0.03, 0.1, 0.15, 0.3, 0.5, 1.0]


class GMM:
    def __init__(self, mus, sig, pi):
        self.mu = np.asarray(mus, float)          # (K, d)
        self.sig = np.asarray(sig, float)         # (K,)
        self.pi = np.asarray(pi, float)           # (K,)
        self.K, self.d = self.mu.shape

    def v(self, x, tau):
        """x: (N, d), tau: scalar or (N,). 정확한 주변 속도."""
        tau = np.broadcast_to(np.asarray(tau, float), (x.shape[0],))[:, None, None]  # (N,1,1)
        mu = self.mu[None]                                   # (1,K,d)
        s2 = (self.sig ** 2)[None, :, None]                  # (1,K,1)
        D = tau ** 2 * s2 + (1 - tau) ** 2                   # (N,K,1)
        diff = x[:, None, :] - tau * mu                      # (N,K,d)
        logw = np.log(self.pi)[None, :, None] - 0.5 * (diff ** 2).sum(-1, keepdims=True) / D \
            - 0.5 * self.d * np.log(D)
        logw = logw - logw.max(axis=1, keepdims=True)
        w = np.exp(logw)
        w = w / w.sum(axis=1, keepdims=True)                 # (N,K,1)
        m = mu + (tau * s2 - (1 - tau)) / D * diff           # (N,K,d)
        return (w * m).sum(axis=1)


def reference(g, x0, steps=2000):
    x = x0.copy()
    h = 1.0 / steps
    t = 0.0
    for _ in range(steps):
        k1 = g.v(x, t)
        k2 = g.v(x + 0.5 * h * k1, t + 0.5 * h)
        k3 = g.v(x + 0.5 * h * k2, t + 0.5 * h)
        k4 = g.v(x + h * k3, t + h)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return x


def euler(g, x0, n):
    x = x0.copy()
    h = 1.0 / n
    for i in range(n):
        x = x + h * g.v(x, i * h)
    return x, n


def midpoint(g, x0, n):
    x = x0.copy()
    h = 1.0 / n
    for i in range(n):
        t = i * h
        xm = x + 0.5 * h * g.v(x, t)
        x = x + h * g.v(xm, t + 0.5 * h)
    return x, 2 * n


def adavla(g, x0, cth, consistent=False):
    """Eq.5,6,8,9 를 마스크로 벡터화. 반환: 끝점, NFE(샘플별), C_first, 한계 도달 여부, 분기 기록."""
    n = x0.shape[0]
    x = x0.copy()
    tau = np.zeros(n)
    nfe = np.zeros(n)
    done = np.zeros(n, bool)
    exhausted = np.zeros(n, bool)
    jumped = np.zeros(n, bool)          # 낮은 곡률 점프로 끝난 샘플
    c_first = np.full(n, np.nan)
    for it in range(MAX_IT):
        act = ~done
        if not act.any():
            break
        idx = np.where(act)[0]
        xa, ta = x[idx], tau[idx]
        va = g.v(xa, ta)
        h = (1 - ta) / 2
        xm = xa + va * h[:, None]
        vm = g.v(xm, ta + h)
        nfe[idx] += 2
        C = np.linalg.norm(va - vm, axis=1) / (np.linalg.norm(vm, axis=1) + EPS)
        if it == 0:
            c_first[idx] = C
        low = C < cth
        # 낮은 곡률: 직접 점프
        li = idx[low]
        x[li] = xa[low] + vm[low] * (1 - ta[low])[:, None]
        done[li] = True
        jumped[li] = True
        # 높은 곡률: 제한된 스텝
        hi = ~low
        hi_idx = idx[hi]
        if hi_idx.size:
            th = ta[hi]
            shat = np.minimum((1 - th) / 2, np.maximum(S, cth / C[hi] * 2 * S))
            if consistent:
                xh = xa[hi] + va[hi] * (shat / 2)[:, None]
                vh = g.v(xh, th + shat / 2)
                nfe[hi_idx] += 1
                x[hi_idx] = xa[hi] + vh * shat[:, None]
            else:
                x[hi_idx] = xa[hi] + vm[hi] * shat[:, None]
            tau[hi_idx] = th + shat
            fin = tau[hi_idx] >= 1 - 1e-9
            done[hi_idx[fin]] = True
    # 한계 도달: Euler 점프로 마무리
    rest = ~done
    if rest.any():
        idx = np.where(rest)[0]
        ta = tau[idx]
        vm = g.v(x[idx], ta + (1 - ta) / 2)
        x[idx] = x[idx] + vm * (1 - ta)[:, None]
        nfe[idx] += 1
        exhausted[idx] = True
    return x, nfe, c_first, exhausted, jumped


def metrics(end, ref, nfe, centers, extra=None):
    err = np.linalg.norm(end - ref, axis=1)
    mode_ref = np.argmin(((ref[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
    mode_out = np.argmin(((end[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
    r = {
        "nfe_mean": round(float(nfe.mean()), 3),
        "err_mean": float(f"{err.mean():.4g}"),
        "err_p90": float(f"{np.percentile(err, 90):.4g}"),
        "frac_err_gt_tol": round(float((err > TOL).mean()), 4),
        "mode_mismatch": round(float((mode_ref != mode_out).mean()), 4),
    }
    if extra:
        r.update(extra)
    return r, err


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def check_velocity(g):
    """정확 속도의 수치 검증: 유한차분으로 d/dτ E[x_τ]의 적분 일관성 대신, 주변 밀도로 직접 계산."""
    rng = np.random.default_rng(123)
    tau = 0.4
    x = rng.normal(size=(5, g.d)) * 1.5
    # 몬테카를로: x_τ = τ x1 + (1-τ) x0 을 x 근방으로 제한하는 가중 추정 (커널 밀도)
    M = 400000
    k = rng.choice(g.K, size=M, p=g.pi)
    x1 = g.mu[k] + g.sig[k, None] * rng.normal(size=(M, g.d))
    x0 = rng.normal(size=(M, g.d))
    xt = tau * x1 + (1 - tau) * x0
    diff_target = x1 - x0
    res = []
    h = 0.1
    for xi in x:
        wgt = np.exp(-0.5 * ((xt - xi) ** 2).sum(1) / h ** 2)
        est = (wgt[:, None] * diff_target).sum(0) / wgt.sum()
        res.append(est)
    res = np.array(res)
    exact = g.v(x, tau)
    return float(np.max(np.abs(res - exact)))


def run_config(name, g, x0, centers, ref):
    out = {}
    e1_end, _ = euler(g, x0, 1)
    e1_err = np.linalg.norm(e1_end - ref, axis=1)
    _, _, c_ref, _, _ = adavla(g, x0, 0.15)
    out["diag_Cfirst_vs_euler1"] = {
        "spearman_Cfirst_vs_euler1_err": round(spearman(c_ref, e1_err), 4),
        "frac_lowC_0.15_and_euler1_err_gt_tol": round(float(((c_ref < 0.15) & (e1_err > TOL)).mean()), 4),
        "frac_highC_0.15_and_euler1_err_le_tol": round(float(((c_ref >= 0.15) & (e1_err <= TOL)).mean()), 4),
    }
    # 낮은 곡률인데 1단계 오차가 큰 샘플 하나 (무엇이 일어나는지 보이기)
    bad = np.where((c_ref < 0.15) & (e1_err > TOL))[0]
    if bad.size:
        i = int(bad[np.argmax(e1_err[bad])])
        out["example_lowC_highErr"] = {
            "x0": [round(float(v), 4) for v in x0[i]], "C_first": round(float(c_ref[i]), 4),
            "euler1_err": round(float(e1_err[i]), 4), "ref_endpoint": [round(float(v), 4) for v in ref[i]],
        }
    for n in [1, 2, 3, 5, 10, 20]:
        end, nfe = euler(g, x0, n)
        r, err = metrics(end, ref, np.full(x0.shape[0], nfe), centers)
        out[f"euler_{n}"] = r
    for n in [1, 2, 5]:
        end, nfe = midpoint(g, x0, n)
        r, _ = metrics(end, ref, np.full(x0.shape[0], nfe), centers)
        out[f"rk2_fixed_{n}"] = r
    for cth in CTH_GRID:
        end, nfe, c1, exh, jmp = adavla(g, x0, cth)
        extra = {"exhausted_frac": round(float(exh.mean()), 4), "jump_any_iter_frac": round(float(jmp.mean()), 4)}
        r, err = metrics(end, ref, nfe, centers, extra)
        # 첫 반복 곡률 기준 분기가 낮은 곡률인 샘플 중 오차가 큰 비율
        low_first = c1 < cth
        extra2 = {
            "lowC_highErr_count": int(((low_first) & (err > TOL)).sum()),
            "lowC_highErr_frac_of_low": round(float(((low_first) & (err > TOL)).sum() / max(1, low_first.sum())), 4),
            "spearman_Cfirst_err": round(spearman(c1, err), 4),
        }
        r.update(extra2)
        out[f"adavla_as_written_{cth}"] = r
        end2, nfe2, _, exh2, _ = adavla(g, x0, cth, consistent=True)
        r2, _ = metrics(end2, ref, nfe2, centers, {"exhausted_frac": round(float(exh2.mean()), 4)})
        out[f"adavla_consistent_{cth}"] = r2
    return out


def main():
    rng = np.random.default_rng(SEED)
    x0 = rng.normal(size=(N, 2))
    configs = {}
    configs["A_single_mu2_sig0.3"] = GMM([[2.0, 0.0]], [0.3], [1.0])
    for a in [1.0, 2.0, 4.0]:
        configs[f"B_two_modes_a{a:g}"] = GMM([[-a, 0.0], [a, 0.0]], [0.2, 0.2], [0.5, 0.5])
    ring = [[3, 0], [0, 3], [-3, 0], [0, -3]]
    configs["C_ring4_r3"] = GMM(ring, [0.25] * 4, [0.25] * 4)
    for c in [0.1, 10.0]:
        configs[f"D_scale{c:g}_a2"] = GMM(np.array([[-2.0, 0.0], [2.0, 0.0]]) * c, [0.2 * c, 0.2 * c], [0.5, 0.5])

    results = {"seed": SEED, "N": N, "eps": EPS, "s": S, "max_iter": MAX_IT, "tol": TOL, "cth_grid": CTH_GRID}
    vcheck = check_velocity(configs["B_two_modes_a2"])
    results["velocity_check_max_abs_diff_MC_kernel"] = float(f"{vcheck:.3g}")

    # C(τ) 프로파일: 기준 궤적에서 한 반복의 곡률 평균
    prof = {}
    g = configs["B_two_modes_a2"]
    ref_x = x0.copy()
    h = 1.0 / 2000
    t = 0.0
    snaps = {}
    for step in range(2000):
        if abs(t - 0.0) < 1e-12 or any(abs(t - q) < h / 2 for q in [0.25, 0.5, 0.75]):
            key = round(t, 2)
            va = g.v(ref_x, t)
            hh = (1 - t) / 2
            vm = g.v(ref_x + va * hh, t + hh)
            Cm = np.linalg.norm(va - vm, axis=1) / (np.linalg.norm(vm, axis=1) + EPS)
            snaps[key] = {"mean_C": float(f"{Cm.mean():.4g}"), "median_C": float(f"{np.median(Cm):.4g}")}
        k1 = g.v(ref_x, t)
        k2 = g.v(ref_x + 0.5 * h * k1, t + 0.5 * h)
        k3 = g.v(ref_x + 0.5 * h * k2, t + 0.5 * h)
        k4 = g.v(ref_x + h * k3, t + h)
        ref_x = ref_x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    results["C_profile_on_reference_path_B_a2"] = snaps

    for name, g in configs.items():
        ref = reference(g, x0)
        centers = g.mu
        results[name] = run_config(name, g, x0, centers, ref)
        print("done", name, flush=True)

    # 스케일 시험: 같은 Cth = 0.15에서 분기 비율 변화
    scale = {}
    for c in [0.1, 1.0, 10.0]:
        g = GMM(np.array([[-2.0, 0.0], [2.0, 0.0]]) * c, [0.2 * c, 0.2 * c], [0.5, 0.5])
        ref = reference(g, x0)
        end, nfe, c1, exh, jmp = adavla(g, x0, 0.15)
        err = np.linalg.norm(end - ref, axis=1)
        scale[str(c)] = {"first_iter_low_frac": round(float((c1 < 0.15).mean()), 4),
                         "jump_any_iter_frac": round(float(jmp.mean()), 4), "nfe_mean": round(float(nfe.mean()), 3),
                         "err_mean": float(f"{err.mean():.4g}"), "median_Cfirst": float(f"{np.median(c1):.4g}")}
    results["scale_test_cth0.15"] = scale

    # 요약 키
    b2 = results["B_two_modes_a2"]
    results["summary"] = {
        "B_a2_euler1_err_mean": b2["euler_1"]["err_mean"],
        "B_a2_euler1_mode_mismatch": b2["euler_1"]["mode_mismatch"],
        "B_a2_euler10_err_mean": b2["euler_10"]["err_mean"],
        "B_a2_adavla_cth0.15": {k: b2["adavla_as_written_0.15"][k] for k in ["nfe_mean", "err_mean", "mode_mismatch", "lowC_highErr_count", "spearman_Cfirst_err"]},
        "B_a2_consistent_cth0.15": {k: b2["adavla_consistent_0.15"][k] for k in ["nfe_mean", "err_mean", "mode_mismatch"]},
    }

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
