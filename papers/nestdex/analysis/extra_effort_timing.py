"""추가 검증 2: 공개 노력 곡선으로 본 시간 해상 성공 판별과 블록 순서 점검 (numpy, 수 초).

자료: paper/inner-policy-effort.json (프로젝트 사이트의 공개 시행 기록, 30회, 조건 3개, 0.04 s 간격, −2–6 s,
      0 s = 닫힘 명령 95% 시점). 새 자료를 받지 않는다.
질문 1: 닫힌 뒤 몇 초의 노력값만 있으면 성공을 예측할 수 있는가?
  - 시각 t마다 "t 이전 0.5 s 창의 평균 노력"(인과적, 미래 정보 없음)을 쓴다.
  - 성공 대 실패의 AUC(Mann–Whitney)와 시행 재표본 bootstrap 95% CI(2000회, 시드 고정).
  - 판정 시각: AUC의 95% CI 하한이 0.8을 처음 넘는 t (보고용 기준, 가정).
질문 2: 블록 순서 효과(시행 순서·위치)가 노력값과 성공에 있는가?
  - 조건 안에서 노력(유지 중앙값)과 에피소드 번호의 Spearman 상관.
  - 조건 평균을 뺀 잔차와 에피소드 번호의 Spearman 상관(조건 간 차이를 제외한 흐름).
  - 조건 안 성공률을 에피소드 번호 앞·뒤 절반으로 나눠 비교(Fisher 정확 검정).
한계: 시행 30회, 조건별 10회라 검정력이 낮다. 노력 곡선은 손 전체 RMS이며 팔·클러치 상태는 없다.
실행: python3 analysis/extra_effort_timing.py -> analysis/extra_effort_timing_results.json
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
TIMES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
WIN = 0.5
N_BOOT = 2000


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def causal_window_mean(e, t, times, win=WIN):
    m = (times > t - win - 1e-9) & (times <= t + 1e-9)
    return float(np.nanmean(e[m])) if m.any() else float("nan")


def main():
    d = json.loads((ROOT / "paper" / "inner-policy-effort.json").read_text())
    times = np.array(d["time"])
    trials = []
    for c in d["conditions"]:
        for tr in c["trials"]:
            e = np.array([np.nan if v is None else v for v in tr["effort"]], dtype=float)
            trials.append(dict(cond=c["id"], ep=int(tr["episode"]), y=1 if tr["outcome"] == "success" else 0,
                               effort=e, hold=float(tr["holdMedian"])))
    y = np.array([t["y"] for t in trials])
    rng = np.random.default_rng(20261111)
    out = {"source": d["source"], "n_trials": len(trials), "n_success": int(y.sum()),
           "window_s": WIN, "n_boot": N_BOOT, "time_resolved": {}}

    # ---- 질문 1: 시간 해상 AUC ----
    first_ok = None
    for t in TIMES:
        v = np.array([causal_window_mean(tr["effort"], t, times) for tr in trials])
        a = auc(v[y == 1], v[y == 0])
        boots = []
        idx_all = np.arange(len(trials))
        for _ in range(N_BOOT):
            bi = rng.choice(idx_all, len(idx_all), replace=True)
            yb, vb = y[bi], v[bi]
            if yb.sum() == 0 or (1 - yb).sum() == 0:
                continue
            boots.append(auc(vb[yb == 1], vb[yb == 0]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out["time_resolved"][f"t={t}"] = dict(auc=a, ci95=[float(lo), float(hi)],
                                               median_effort_success=float(np.nanmedian(v[y == 1])),
                                               median_effort_failure=float(np.nanmedian(v[y == 0])))
        if first_ok is None and lo >= 0.8:
            first_ok = t
    out["first_time_ci_lower_ge_0.8"] = first_ok

    # 참고: 노력 중앙값(0.5–5.5 s, 원 논문 방식, 비인과)의 AUC
    hold = np.array([t["hold"] for t in trials])
    out["reference_noncausal_hold_median_auc"] = auc(hold[y == 1], hold[y == 0])

    # ---- 질문 2: 블록 순서 ----
    order = {}
    resid = np.zeros(len(trials))
    cond_ids = sorted(set(t["cond"] for t in trials))
    for cid in cond_ids:
        idx = [i for i, t in enumerate(trials) if t["cond"] == cid]
        ep = np.array([trials[i]["ep"] for i in idx])
        h = np.array([trials[i]["hold"] for i in idx])
        yy = y[idx]
        rho, p = stats.spearmanr(ep, h)
        mid = np.median(ep)
        early, late = yy[ep <= mid], yy[ep > mid]
        fisher_p = float(stats.fisher_exact([[late.sum(), len(late) - late.sum()],
                                             [early.sum(), len(early) - early.sum()]])[1])
        order[cid] = dict(n=len(idx), spearman_episode_vs_hold_rho=float(rho), spearman_p=float(p),
                          success_early_half=f"{int(early.sum())}/{len(early)}",
                          success_late_half=f"{int(late.sum())}/{len(late)}", fisher_p_early_vs_late=fisher_p)
        resid[idx] = h - h.mean()
    ep_all = np.array([t["ep"] for t in trials])
    rho_r, p_r = stats.spearmanr(ep_all, resid)
    out["block_order"] = dict(per_condition=order,
                              pooled_residual_vs_episode_spearman=dict(rho=float(rho_r), p=float(p_r)))

    (HERE / "extra_effort_timing_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("time  AUC  CI95")
    for t in TIMES:
        r = out["time_resolved"][f"t={t}"]
        print(f"{t:4} {r['auc']:.3f} [{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}]")
    print("first t with CI lower >= 0.8:", first_ok, "| noncausal hold-median AUC:",
          round(out["reference_noncausal_hold_median_auc"], 3))
    for cid, v in order.items():
        print(cid, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()})
    print("pooled residual vs episode:", out["block_order"]["pooled_residual_vs_episode_spearman"])


if __name__ == "__main__":
    main()
