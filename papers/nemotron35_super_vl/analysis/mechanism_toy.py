"""보고서의 두 메커니즘을 numpy로 확인한다 (CPU, 단일 스레드, 수십 초). 실행: python3 analysis/mechanism_toy.py

1) ReplaySSM(§5.4): Mamba-2 decode 상태 S_new = α S_old + kᵀv 를 체크포인트 + 입력 버퍼로 재구성
   - 닫힌 식의 지수 확인: 본문은 ∏_{j>i+1} α_j, 같은 문단의 갱신식에서 유도하면 ∏_{j>i} α_j
   - FP16 상태를 매 토큰 반올림(RN·SR)할 때와 L 토큰마다 한 번만 반올림할 때의 오차
   - 공개 config 크기(헤드 128 × 64, 상태 128, 그룹 8)로 decode 한 단계의 상태 메모리 트래픽
2) MOPD proximal teacher(§3.6): 교사를 α π_T + (1−α) π_S 로 바꾸면 토큰 advantage ≥ log(1−α)
결과: analysis/mechanism_toy_results.json
"""
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(0)
N_STATE, P_HEAD, N_HEADS, N_GROUPS = 128, 64, 128, 8   # 공개 config (mamba_*), 상태 = (N, P) per head


def decays(n):
    """α_t = exp(-dt·A). -log α를 1e-4–5e-2에서 로그 균등으로 뽑는다 (느린 헤드와 빠른 헤드 혼합)."""
    return np.exp(-np.exp(rng.uniform(np.log(1e-4), np.log(5e-2), n)))


def inputs(n):
    return rng.normal(size=(n, N_STATE)) / np.sqrt(N_STATE), rng.normal(size=(n, P_HEAD))


# 1-a) 닫힌 식
def closed_form_check(n=16):
    a, (k, v) = decays(n), inputs(n)
    s0 = rng.normal(size=(N_STATE, P_HEAD))
    s = s0.copy()
    for t in range(n):
        s = a[t] * s + np.outer(k[t], v[t])
    def form(shift):
        out = np.prod(a) * s0
        for i in range(n):
            out = out + np.prod(a[i + 1 + shift:]) * np.outer(k[i], v[i])
        return out
    rel = lambda x: float(np.linalg.norm(x - s) / np.linalg.norm(s))
    return {"window": n, "rel_err_prod_j_gt_i": rel(form(0)), "rel_err_prod_j_gt_i_plus_1 (본문 표기)": rel(form(1))}


# 1-b) FP16 상태 반올림 오차
def to_fp16(x, stochastic):
    x16 = x.astype(np.float16)
    if not stochastic:
        return x16.astype(np.float64)
    up = x16.astype(np.float64) < x
    lo = np.where(up, x16, np.nextafter(x16, np.float16(-np.inf)))
    hi = np.where(up, np.nextafter(x16, np.float16(np.inf)), x16)
    exact = x16.astype(np.float64) == x
    lo, hi = lo.astype(np.float64), hi.astype(np.float64)
    p = np.where(exact, 0.0, (x - lo) / np.where(hi > lo, hi - lo, 1.0))
    return np.where(exact, x, np.where(rng.random(x.shape) < p, hi, lo))


def quant_error(neglog, n=4096, heads=4, L=16):
    """neglog = (lo, hi): 헤드별 -log α 범위. 느린 헤드는 감쇠량이 FP16 반 ulp(약 2.4e-4)보다 작다."""
    names = ["FP16 RN 매 토큰", "FP16 SR 매 토큰", f"Replay L={L} + FP16 RN", f"Replay L={L} + FP16 SR"]
    final, outerr = {k: [] for k in names}, {k: [] for k in names}
    for _ in range(heads):
        a, (k, v) = np.full(n, np.exp(-rng.uniform(*neglog))), inputs(n)   # 헤드마다 α 하나 (Mamba-2는 헤드 단위 스칼라 감쇠)
        q = rng.normal(size=(n, N_STATE)) / np.sqrt(N_STATE)
        ref = np.zeros((N_STATE, P_HEAD))
        st = {kk: np.zeros((N_STATE, P_HEAD)) for kk in names}
        errs = {kk: [] for kk in names}
        for t in range(n):
            kv = np.outer(k[t], v[t])
            ref = a[t] * ref + kv
            for kk in names:
                if kk.startswith("FP16"):
                    st[kk] = to_fp16(a[t] * st[kk] + kv, "SR" in kk)
                else:   # 버퍼 안에서는 정확(FP64)하게 누적, L 토큰마다 체크포인트를 FP16으로 접음
                    st[kk] = a[t] * st[kk] + kv
                    if (t + 1) % L == 0:
                        st[kk] = to_fp16(st[kk], "SR" in kk)
                o_ref = q[t] @ ref
                errs[kk].append(np.linalg.norm(q[t] @ st[kk] - o_ref) / (np.linalg.norm(o_ref) + 1e-12))
        for kk in names:
            final[kk].append(np.linalg.norm(st[kk] - ref) / np.linalg.norm(ref))
            outerr[kk].append(np.mean(errs[kk]))
    return {"neglog_alpha": list(neglog), "tokens": n, "heads": heads,
            "final_state_rel_err": {kk: float(np.mean(v)) for kk, v in final.items()},
            "mean_output_rel_err": {kk: float(np.mean(v)) for kk, v in outerr.items()}}


# 1-c) decode 한 단계, 요청 하나, Mamba 층 하나의 상태 관련 트래픽 (원소 수)
def traffic(L=16, drafts=(1, 3, 5, 7)):
    state = N_HEADS * P_HEAD * N_STATE                      # 1,048,576
    per_tok = N_HEADS * P_HEAD + N_GROUPS * N_STATE + N_HEADS  # x(=v) + B(=k) + dt(=α) = 9,344
    rows = {}
    for T in drafts:
        base = (1 + T) * state                              # 체크포인트 읽기 1 + 후보 상태 T개 쓰기
        replay = state + (L / 2) * per_tok + T * per_tok + state / L   # 체크포인트 읽기 + 버퍼 읽기·쓰기 + 접기 분할상환
        rows[f"T={T}"] = {"baseline_state_units": round(base / state, 3), "replay_state_units": round(replay / state, 3),
                          "reduction_x": round(base / replay, 2)}
    return {"state_elems_per_layer": state, "buffer_elems_per_token": per_tok,
            "buffer_token_over_state": round(per_tok / state, 4), "L": L, "by_draft_len": rows,
            "paper_claim": "T+1 → 약 1+1/L (§5.4)", "measured_T24": "출력 TPS +4.7–6.4%, 사용자당 +6.4–12.5%"}


# 2) proximal teacher 하한
def mopd_bound(V=4096, trials=200, alpha=0.2):
    mins, kl_full, kl_mix, zero_frac = [], [], [], []
    for _ in range(trials):
        pT, pS = (np.maximum(rng.dirichlet(np.full(V, 0.1)), 1e-12) for _ in range(2))
        pT, pS = pT / pT.sum(), pS / pS.sum()
        mix = alpha * pT + (1 - alpha) * pS
        A = np.log(mix) - np.log(pS)
        mins.append(A.min())
        kl_full.append(np.sum(pS * (np.log(pS) - np.log(pT))))
        kl_mix.append(np.sum(pS * (np.log(pS) - np.log(mix))))
    return {"alpha": alpha, "log(1-alpha)": float(np.log(1 - alpha)), "min_advantage_over_trials": float(np.min(mins)),
            "bound_holds": bool(np.min(mins) >= np.log(1 - alpha) - 1e-12),
            "median_KL(pS||pT)": float(np.median(kl_full)), "median_KL(pS||mix)": float(np.median(kl_mix)),
            "note": "단계 1(α=1)은 하한이 없음. 단계 2는 음의 신호를 −0.223으로 자르고 평균 신호 크기도 크게 줄인다"}


def main():
    out = {"replay_closed_form": closed_form_check(), "replay_fp16_error": {"fast_heads": quant_error((1e-2, 5e-2)), "slow_heads": quant_error((2e-5, 2e-4))},
           "replay_traffic": traffic(), "mopd_proximal_teacher": mopd_bound()}
    path = ROOT / "analysis" / "mechanism_toy_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
