"""토이 C: 초기화 동일성(O2.1, O2.2)과 shifted causal mask를 numpy로 확인한다 (CPU, 수 초).

논문 Figure 3과 식 (3)의 규칙을 repo의 src/openpi/models/pi0.py `make_reactive_attn_mask`,
`make_cross_attn_mask`(static reading, 실행 안 함)와 같은 인덱싱으로 옮겼다.
단순화: 토큰 수 작고 차원 16, 헤드 1개, LayerNorm은 파라미터 없음, 힘 메모리 GRU는 난수 가중치의 단일층.
목적: (1) 같은 노이즈일 때 반응 스트림이 기본 스트림과 같아지는지, (2) 노이즈가 다르면 같은지,
(3) 기본 스트림이 반응 토큰에 영향을 받지 않는지(누출 없음), (4) 마스크를 잘못 주면 깨지는지(음성 대조),
(5) 0 초기화 출력 투영을 조금 키우면 차이가 어떻게 커지는지.
출력: analysis/toy_init_equiv_results.json
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_init_equiv_results.json"
D, H, P, L, FD = 16, 10, 6, 3, 6  # 차원, 행동 청크, 프리픽스 토큰 수, 힘 지연 오프셋(논문 L=3, code cross_attention_latency=3), 힘 차원
NEG = -1e30


def ln(x):
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return (x - m) / np.sqrt(v + 1e-6)


def make_expert(rng):
    s = 1 / np.sqrt(D)
    return {k: rng.normal(0, s, (D, D)) for k in ("Wq", "Wk", "Wv", "Wo")} | {
        "W1": rng.normal(0, s, (D, 2 * D)), "W2": rng.normal(0, s / np.sqrt(2), (2 * D, D))}


def reactive_mask(variant="shifted"):
    """make_reactive_attn_mask 와 같은 인덱싱. 행 = 쿼리, 열 = 키. 토큰 순서: [prefix P, base H, reactive H]."""
    T = P + 2 * H
    m = np.ones((T, T), dtype=bool)  # ar_mask가 전부 0이므로 cumsum 조건은 모두 참
    m[:, -H:] = False  # 어떤 쿼리도 반응 토큰을 보지 못함 (기본 스트림 격리)
    m[0:-2 * H, -2 * H:] = False  # 프리픽스는 행동 토큰을 보지 못함
    for i in range(H):
        row = T - i - 1
        allowed = np.zeros(T, dtype=bool)
        allowed[: T - 2 * H] = True
        if variant == "shifted":
            allowed[T - H - i : T - i] = True  # 기본 a_{j+1:} 와 반응 r_{0:j} (j = H-1-i)
        elif variant == "unshifted_full_base":  # 음성 대조: 기본 토큰 전체를 본다
            allowed[P : P + H] = True
            allowed[T - H : T - i] = True
        m[row] = allowed
    return m


def cross_mask(variant="latency"):
    """make_cross_attn_mask 와 같음: 쿼리 i는 힘 토큰 k <= i-L 만 본다. use_force_history=False면 k==0만."""
    M = np.zeros((H, H), dtype=bool)
    for i in range(H):
        for k in range(H):
            ok = k <= i - L
            if variant == "single_first_token":
                ok = ok and k == 0
            M[i, k] = ok
    return M


def attend(q, k, v, mask):
    s = q @ k.T / np.sqrt(D)
    s = np.where(mask, s, NEG)
    rowmax = s.max(-1, keepdims=True)
    e = np.exp(s - rowmax) * mask
    den = e.sum(-1, keepdims=True)
    p = np.where(den > 0, e / np.where(den > 0, den, 1), 0.0)  # 허용 키가 없으면 0 (code의 nan_to_num과 같음)
    return p @ v


def force_memory(rng_force_w, F):
    """단일층 causal GRU 대용: m_j = tanh(F_j U + m_{j-1} V). 청크 시작에서 상태를 0으로 둔다."""
    U, V, W = rng_force_w
    m = np.zeros(D // 2)
    out = []
    for j in range(F.shape[0]):
        m = np.tanh(F[j] @ U + m @ V)
        out.append(m)
    return np.stack(out) @ W  # (H, D)


def forward(params, x_prefix, x_base, x_react, F, mask, cmask, Wcross, cross_on=True, n_layers=2):
    """반응 스트림에만 힘 교차어텐션을 넣는다. 기본 토큰은 같은 가중치를 쓰지만 교차어텐션은 없다."""
    hp, ha, hr = x_prefix.copy(), x_base.copy(), x_react.copy()
    mem = F  # 힘 메모리 (H, D), force_memory()로 미리 계산
    for l in range(n_layers):
        Ep, Ea, Er = params["prefix"][l], params["base"][l], params["react"][l]
        xp, xa, xr = ln(hp), ln(ha), ln(hr)
        def qkv(E, x, key):
            return x @ E[key]
        Q = np.concatenate([qkv(Ep, xp, "Wq"), qkv(Ea, xa, "Wq"), qkv(Er, xr, "Wq")])
        K = np.concatenate([qkv(Ep, xp, "Wk"), qkv(Ea, xa, "Wk"), qkv(Er, xr, "Wk")])
        Vv = np.concatenate([qkv(Ep, xp, "Wv"), qkv(Ea, xa, "Wv"), qkv(Er, xr, "Wv")])
        A = attend(Q, K, Vv, mask)
        o_p = A[:P] @ Ep["Wo"]
        o_a = A[P : P + H] @ Ea["Wo"]
        o_r = A[P + H :] @ Er["Wo"]
        hp = hp + o_p
        ha = ha + o_a
        hr = hr + o_r
        if cross_on:
            xr2 = ln(hr)
            Qc = xr2 @ Wcross["q"]
            Kc = mem @ Wcross["k"]
            Vc = mem @ Wcross["v"]
            C = attend(Qc, Kc, Vc, cmask) @ Wcross["o"]
            hr = hr + C
        hp = hp + np.tanh(ln(hp) @ Ep["W1"]) @ Ep["W2"]
        ha = ha + np.tanh(ln(ha) @ Ea["W1"]) @ Ea["W2"]
        hr = hr + np.tanh(ln(hr) @ Er["W1"]) @ Er["W2"]
    return hp, ha, hr


def make_inputs(rng):
    x_prefix = rng.normal(size=(P, D))
    F = rng.normal(size=(H, FD))
    return x_prefix, F


def main():
    rng = np.random.default_rng(0)
    results = {"setup": {"D": D, "H": H, "P": P, "latency_L": L, "layers": 2, "heads": 1}}
    base_params = {"prefix": [], "base": [], "react": []}
    for _ in range(2):
        Ea = make_expert(rng)
        Ep = make_expert(rng)
        base_params["prefix"].append(Ep)
        base_params["base"].append(Ea)
        base_params["react"].append({k: v.copy() for k, v in Ea.items()})  # 논문 O2.1: 기본 전문가 가중치 복사
    Wcross_zero = {"q": rng.normal(0, 0.3, (D, D)), "k": rng.normal(0, 0.3, (D, D)),
                   "v": rng.normal(0, 0.3, (D, D)), "o": np.zeros((D, D))}  # O2.2: 출력 투영 0 초기화
    force_w = (rng.normal(0, 0.5, (FD, D // 2)), rng.normal(0, 0.5, (D // 2, D // 2)), rng.normal(0, 0.5, (D // 2, D)))

    x_prefix, F = make_inputs(rng)
    mem = force_memory(force_w, F)
    x_a = rng.normal(size=(H, D))
    x_r_same = x_a.copy()
    x_r_diff = rng.normal(size=(H, D))
    M = reactive_mask("shifted")
    C = cross_mask("latency")

    # (0) 마스크 확인: r_j 가 보는 토큰 집합
    def allowed_of(row):
        return [int(i) for i in np.where(M[row])[0]]
    j = 4
    row = P + H + j
    expected = list(range(0, P)) + list(range(P + j + 1, P + H)) + list(range(P + H, P + H + j + 1))
    results["mask_check_r4_allowed_equals_paper_rule"] = allowed_of(row) == expected
    results["mask_check_a4_allowed"] = allowed_of(P + 4) == list(range(0, P + H))

    # (1) 같은 노이즈, 0 초기화: 반응 스트림 = 기본 스트림 ?
    hp, ha, hr = forward(base_params, x_prefix, x_a, x_r_same, mem, M, C, Wcross_zero)
    results["init_same_noise_max_abs_diff"] = float(np.max(np.abs(ha - hr)))
    results["init_same_noise_prefix_max_abs_diff_vs_no_react"] = float(np.max(np.abs(
        hp - forward(base_params, x_prefix, x_a, x_r_same, mem, M, C, Wcross_zero, cross_on=False)[0])))

    # (2) 다른 노이즈(논문 식 (1)의 ε, ε_r 독립, repo의 policy.py도 별도 난수): 같은 샘플이 아님
    _, ha2, hr2 = forward(base_params, x_prefix, x_a, x_r_diff, mem, M, C, Wcross_zero)
    results["init_indep_noise_max_abs_diff"] = float(np.max(np.abs(ha2 - hr2)))
    results["init_indep_noise_mean_abs_diff"] = float(np.mean(np.abs(ha2 - hr2)))

    # (3) 누출: 반응 입력(노이즈)을 바꿔도 기본 스트림은 같아야 한다
    _, ha3, _ = forward(base_params, x_prefix, x_a, x_r_diff * 3 + 1, mem, M, C, Wcross_zero)
    results["leak_base_change_when_reactive_input_changes"] = float(np.max(np.abs(ha2 - ha3)))

    # (4) 음성 대조: 마스크를 기본 토큰 전체로 열면 같은 노이즈에서도 깨진다
    M_bad = reactive_mask("unshifted_full_base")
    _, ha4, hr4 = forward(base_params, x_prefix, x_a, x_r_same, mem, M_bad, C, Wcross_zero)
    results["negative_control_full_base_mask_same_noise_diff"] = float(np.max(np.abs(ha4 - hr4)))

    # (5) 교차어텐션 출력 투영을 eps로 키우면 차이가 eps에 따라 커진다 (학습 후 비0 잔차의 감도)
    sens = {}
    for eps in (1e-3, 1e-2, 1e-1):
        W = dict(Wcross_zero)
        W["o"] = eps * rng.normal(0, 1 / np.sqrt(D), (D, D))
        _, ha5, hr5 = forward(base_params, x_prefix, x_a, x_r_same, mem, M, C, W)
        sens[str(eps)] = float(np.max(np.abs(ha5 - hr5)))
    results["nonzero_cross_output_sensitivity_max_abs_diff"] = sens

    # (6) 힘 지연: 쿼리 i가 보는 최신 힘 프레임 인덱스 (i - L), i < L 이면 힘 없음
    latest = {i: (i - L if i - L >= 0 else None) for i in range(H)}
    results["latest_force_index_used_by_action_i"] = latest
    results["single_first_token_variant_visible_force_index"] = {
        i: (0 if i - L >= 0 else None) for i in range(H)}

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
