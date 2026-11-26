"""식 (4)의 모달리티 가중 보너스와 공식 코드의 보너스 계산을 같은 입력으로 비교하는 numpy 토이 (수 초).

공식 코드(저장소 https://github.com/klemensiten/tacex 커밋 d268cdb0d769fc9868fe37af462f81ef59c4412c, `repo/`로 받아 정적으로 읽음)의
계산을 그대로 옮겼다. 저장소 코드는 실행하지 않았다.

  - jaxrl2/networks/ensemble_model.py  DeterministicEnsemble.get_info_gain
      ep_std = 앙상블 헤드 사이 표준편차, al_std = 1e-3 (결정적 앙상블이라 고정)
      ratio = (ep_std / al_std)^2 * 가중치,  보너스 = mean_dims(log(1 + ratio)),  그다음 실행 평균·표준편차로 표준화
  - jaxrl2/agents/pixel_maxinfosac/ensemble_utils.py  disagreement_weights
      선택한 모달리티의 차원에는 scale = (전체 출력 차원) / (선택 차원 합), 나머지는 0
      (모달리티마다 임의의 실수 ω를 줄 수 없고 선택 여부만 정한다)

논문 식 (4)는 ω_v ||σ_v|| + ω_ι ||σ_ι|| + ω_z ||σ_z|| (모달리티별 노름의 가중합)이다.

확인하는 것:
  C1 코드의 '촉각만' 보너스는 선택하지 않은 차원의 불확실성에 영향을 받지 않는다 (의도한 대로).
  C2 모달리티 차원 수가 다르면(잠재 50, 영상 64, 촉각 64) 식 (4)의 노름 합과 코드의 차원 평균은 모달리티 비중이 다르다.
  C3 코드는 ω_ι = 5, ω_v = 1 같은 비율을 표현할 수 없다 (선택 마스크뿐).
  C4 표준화 때문에 보너스의 절반 가까이가 음수가 된다 (표준화 이후 보상의 부호).

실행: python3 analysis/toy_bonus_weights.py   → analysis/toy_bonus_weights_results.json
"""
import json
from pathlib import Path

import numpy as np

rng = np.random.default_rng(0)
OUT = Path(__file__).resolve().parent / "toy_bonus_weights_results.json"
DIMS = {"latent": 50, "image": 64, "tactile": 64}
D_REWARD = 1
AL_STD = 1e-3


def slices():
    out, s = {}, 0
    for k, d in DIMS.items():
        out[k] = (s, s + d)
        s += d
    out["reward"] = (s, s + D_REWARD)
    return out, s + D_REWARD


SL, D_OUT = slices()


def code_weights(selected):
    """ensemble_utils.disagreement_weights 와 같은 계산."""
    sel_dim = sum(SL[k][1] - SL[k][0] for k in selected)
    w = np.zeros(D_OUT)
    for k in selected:
        a, b = SL[k]
        w[a:b] = D_OUT / sel_dim
    return w


def code_bonus(ep_std, weights):
    """get_info_gain: mean_dims(log(1 + (ep_std/al_std)^2 * w)). 표준화 전."""
    ratio = (ep_std / AL_STD) ** 2 * weights
    return np.log1p(ratio).mean(axis=-1)


def eq4_bonus(ep_std, omega):
    """논문 식 (4)의 괄호 안: sum_k omega_k * ||sigma_k||_2."""
    tot = 0.0
    for k, w in omega.items():
        a, b = SL[k]
        tot = tot + w * np.linalg.norm(ep_std[..., a:b], axis=-1)
    return tot


def make_std(n, scale_by_modality):
    s = np.zeros((n, D_OUT))
    for k, sc in scale_by_modality.items():
        a, b = SL[k]
        s[:, a:b] = np.abs(rng.normal(size=(n, b - a))) * sc
    return s


res = {}
n = 20000

# C1: 촉각만 선택하면 다른 차원의 불확실성이 바뀌어도 보너스가 같다
base = make_std(n, {"latent": 1e-3, "image": 1e-3, "tactile": 1e-3, "reward": 1e-3})
noisy = base.copy()
noisy[:, SL["latent"][0]:SL["image"][1]] *= 50.0  # 잠재와 영상의 불확실성을 50배로
w_t = code_weights(["tactile"])
b0, b1 = code_bonus(base, w_t), code_bonus(noisy, w_t)
res["C1_tactile_only_invariance"] = {"max_abs_change": float(np.abs(b0 - b1).max()), "mean_bonus": float(b0.mean())}

# C2: 세 모달리티의 요소별 불확실성이 같을 때 식 (4)와 코드의 모달리티 비중
eq = {"latent": 1.0, "image": 1.0, "tactile": 1.0}
equal = make_std(n, {"latent": 1e-3, "image": 1e-3, "tactile": 1e-3})
res["C2_modality_share_equal_per_dim_std"] = {}
for k in DIMS:
    only = np.zeros_like(equal)
    a, b = SL[k]
    only[:, a:b] = equal[:, a:b]
    # 비중: 이 모달리티만 있을 때의 기여 / 세 모달리티 기여 합
    eq_share = eq4_bonus(only, {k: 1.0}).mean()
    code_share = code_bonus(only, code_weights(list(DIMS))).mean()
    res["C2_modality_share_equal_per_dim_std"][k] = {"eq4_norm_term": float(eq_share), "code_mean_log_term": float(code_share)}
tot_eq = sum(v["eq4_norm_term"] for v in res["C2_modality_share_equal_per_dim_std"].values())
tot_code = sum(v["code_mean_log_term"] for v in res["C2_modality_share_equal_per_dim_std"].values())
for k, v in res["C2_modality_share_equal_per_dim_std"].items():
    v["eq4_share"] = round(v["eq4_norm_term"] / tot_eq, 3)
    v["code_share"] = round(v["code_mean_log_term"] / tot_code, 3)

# C3: 코드가 낼 수 있는 가중치 모양
res["C3_expressible_weight_patterns"] = {}
for sel in (["tactile"], ["image", "tactile"], ["latent", "tactile"], ["latent", "image", "tactile"]):
    w = code_weights(sel)
    res["C3_expressible_weight_patterns"]["+".join(sel)] = {k: round(float(w[SL[k][0]]), 3) for k in DIMS}
res["C3_note"] = "선택한 모달리티는 모두 같은 값(전체 차원/선택 차원)이다. omega_tactile = 5, omega_vision = 1 같은 비율은 표현되지 않는다."

# C4: 표준화 후 부호
raw = code_bonus(make_std(n, {"latent": 2e-3, "image": 2e-3, "tactile": 3e-3, "reward": 1e-3}) *
                 np.abs(rng.lognormal(0, 0.5, size=(n, 1))), code_weights(["tactile"]))
z = (raw - raw.mean()) / raw.std()
res["C4_standardized_bonus"] = {"fraction_negative": float((z < 0).mean()), "raw_mean": float(raw.mean()),
                                "note": "실행 평균과 표준편차로 표준화하면 보너스의 평균이 0이 되어 상당수가 음수 보상이 된다(코드의 info_gain_normalizer)."}

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1))
