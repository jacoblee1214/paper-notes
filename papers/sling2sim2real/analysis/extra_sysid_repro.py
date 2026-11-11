"""추가 검증 1–2: 1자유도 toy의 보정 재현성(검증 1)과 일 항 가중치 스윕(검증 2) (CPU numpy, 1스레드).

sysid_toy.py의 모형·손실·적합 파이프라인(DE → 히스토리 NMS → 공분산 CEM → Nelder–Mead)을 그대로 import해 쓴다.
실행 방법(두 부분을 따로 돌린다. 각 5분 이내):
  python3 -I analysis/extra_sysid_repro.py repro   # (a) 같은 관측에 파이프라인 시드 50개
  python3 -I analysis/extra_sysid_repro.py sweep   # (b) 일 항 가중치 0.1–10배, 관측 10개씩, 힘 전용 기준선 포함
결과: analysis/extra_sysid_repro_results.json (두 부분의 결과를 같은 파일에 합친다)

(a) 재현성: 관측 한 세트를 고정하고 적합 시드(DE 초기 표본, CEM 표본)만 50개 바꾼다. θ̂의 산포와
    목표 1.925 m에서의 착탄 오차(cm), 그리고 DE 단계만의 최소 손실이 전체 파이프라인과 같은 골짜기에 있는지를 본다.
(b) 가중치: L_work = |W_o − W_s| / (σ_W / f), f ∈ {0.1, 0.3, 1, 3, 10}. f = 1은 기본 설정(σ_W = 0.015 J).
    관측은 10회 분(CRN: 모든 f에서 같은 관측과 같은 시드). 기준선: 힘 항만 쓴 적합(L_force).
주의: toy 한 개(발사 모형의 α = 35°, 가정된 질량·감쇠 구조)의 결과이며, 논문 모형이 아니다.
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sysid_toy as S  # noqa: E402

OUT = HERE / "extra_sysid_repro_results.json"
GOAL = S.GOALS[1]  # 1.925 m
NAMES = S.NAMES


def make_obs(rng):
    X, F, V = S.id_record(S.TRUE[None])
    return (X[0] + rng.normal(0, S.SIG_X, X.shape[1]), F[0] + rng.normal(0, S.SIG_F, F.shape[1]),
            V[0] + rng.normal(0, S.SIG_V, V.shape[1]))


def landing_err_cm(th):
    d = S.solve_pull(np.asarray(th)[None], np.array([GOAL]))
    return float(100 * (S.landing(S.TRUE[None], d)[0] - GOAL))


def theta_err(th):
    return {n: float(100 * abs(th[j] / S.TRUE[j] - 1)) for j, n in enumerate(NAMES)}


def part_repro():
    t0 = time.time()
    obs = make_obs(np.random.default_rng(20260711))
    rows = []
    for s in range(50):
        th_cma, th_de, Lbest, Lde, _ = S.fit(obs, "full", np.random.default_rng(1000 + s))
        rows.append(dict(seed=s, theta=th_cma.tolist(), theta_de=th_de.tolist(), loss=Lbest, loss_de=Lde,
                         landing_err_cm=landing_err_cm(th_cma), theta_err_pct=theta_err(th_cma)))
    th = np.array([r["theta"] for r in rows])
    le = np.array([r["landing_err_cm"] for r in rows])
    L = np.array([r["loss"] for r in rows])
    ok = L <= L.min() * 1.01 + 1e-9
    summary = dict(
        n_seeds=50,
        loss_min=float(L.min()), loss_median=float(np.median(L)), loss_max=float(L.max()),
        n_seeds_same_basin_within_1pct_loss=int(ok.sum()),
        theta_cv_pct_in_basin={n: float(100 * np.std(th[ok, j]) / np.mean(th[ok, j])) for j, n in enumerate(NAMES)},
        theta_median_in_basin={n: float(np.median(th[ok, j])) for j, n in enumerate(NAMES)},
        theta_true=S.TRUE.tolist(),
        landing_err_cm_mean=float(le.mean()), landing_err_cm_sd=float(le.std(ddof=1)),
        landing_err_cm_median=float(np.median(le)), landing_err_cm_range=[float(le.min()), float(le.max())],
        landing_err_cm_p90_abs=float(np.percentile(np.abs(le), 90)),
        wall_s=round(time.time() - t0, 1),
    )
    return dict(summary=summary, rows=rows)


def part_sweep():
    t0 = time.time()
    rng = np.random.default_rng(20260711)
    obs_list = [make_obs(rng) for _ in range(10)]
    base_sig_w = S.SIG_W
    res = {"factors": [0.1, 0.3, 1.0, 3.0, 10.0], "n_reps": 10, "sig_w_base": base_sig_w, "per_factor": {}}
    for f in res["factors"]:
        S.SIG_W = base_sig_w / f
        th_list, le_list = [], []
        for r in range(10):
            th_cma, _, _, _, _ = S.fit(obs_list[r], "full", np.random.default_rng(3000 + 17 * r))
            th_list.append(th_cma)
            le_list.append(landing_err_cm(th_cma))
        th = np.array(th_list)
        rel = np.abs(th / S.TRUE - 1) * 100
        res["per_factor"][str(f)] = dict(
            theta_median_rel_err_pct={n: float(np.median(rel[:, j])) for j, n in enumerate(NAMES)},
            landing_abs_err_cm_median=float(np.median(np.abs(le_list))),
            landing_abs_err_cm_mean=float(np.mean(np.abs(le_list))),
            landing_abs_err_cm_p90=float(np.percentile(np.abs(le_list), 90)),
        )
    S.SIG_W = base_sig_w
    # 힘 항만 쓴 기준선 (같은 관측, 같은 시드)
    th_list, le_list = [], []
    for r in range(10):
        th_cma, _, _, _, _ = S.fit(obs_list[r], "force", np.random.default_rng(3000 + 17 * r))
        th_list.append(th_cma)
        le_list.append(landing_err_cm(th_cma))
    th = np.array(th_list)
    rel = np.abs(th / S.TRUE - 1) * 100
    res["force_only_baseline"] = dict(
        theta_median_rel_err_pct={n: float(np.median(rel[:, j])) for j, n in enumerate(NAMES)},
        landing_abs_err_cm_median=float(np.median(np.abs(le_list))),
        landing_abs_err_cm_mean=float(np.mean(np.abs(le_list))),
        landing_abs_err_cm_p90=float(np.percentile(np.abs(le_list), 90)),
    )
    res["wall_s"] = round(time.time() - t0, 1)
    return res


def main():
    part = sys.argv[1] if len(sys.argv) > 1 else "repro"
    merged = json.loads(OUT.read_text()) if OUT.exists() else {}
    if part == "repro":
        merged["repro_de_seeds"] = part_repro()
    elif part == "sweep":
        merged["work_weight_sweep"] = part_sweep()
    else:
        raise SystemExit("usage: repro | sweep")
    OUT.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
    print(json.dumps(merged.get("repro_de_seeds", {}).get("summary") if part == "repro" else merged["work_weight_sweep"],
                     ensure_ascii=False, indent=1)[:3000])


if __name__ == "__main__":
    main()
