"""FlashSAC 유도 검산: 목표 엔트로피(식 7), Zeta 잡음 반복, 재생 비율, 파라미터 수, 학습률 스케줄 step 수.

실행: python3 analysis/derivation_checks.py  → analysis/derivation_checks_results.json
공식 코드를 실행하지 않는다. 논문 식과 공개 설정 값(부록 Table 9–10, 스크립트의 인자)으로 손으로 계산한 값을 수치로 확인한다.
파라미터 수는 공식 저장소의 층 정의(flash_rl/agents/flashSAC/layer.py, network.py)를 읽고 세어 적은 식이다.
"""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "derivation_checks_results.json"
rng = np.random.default_rng(0)
out = {}

# 1. 목표 엔트로피 (식 7): H̄ = ½|A| log(2πe σ²), σ = 0.15
sigma = 0.15
per_dim_gauss = 0.5 * math.log(2 * math.pi * math.e * sigma ** 2)
out["target_entropy_per_dim_gaussian"] = per_dim_gauss
out["target_entropy"] = {str(a): a * per_dim_gauss for a in (6, 19, 20, 29, 37)}
# tanh로 눌린 행동의 엔트로피는 Gaussian보다 낮다. 코드의 엔트로피는 tanh 야코비안 보정을 포함한 log π이므로 비교해 본다.
u = rng.normal(0.0, sigma, size=2_000_000)
log_jac = np.log(1 - np.tanh(u) ** 2)
h_squashed = -(np.mean(-0.5 * (u / sigma) ** 2 - np.log(sigma) - 0.5 * np.log(2 * np.pi)) - np.mean(log_jac))
out["per_dim_entropy_tanh_squashed_mc"] = float(h_squashed)
out["per_dim_entropy_tanh_squashed_minus_target"] = float(h_squashed - per_dim_gauss)

# 2. Zeta 잡음 반복: P(k) ∝ k^(-s), k = 1..16, s = 2 (부록 Table 9, 코드 mu=2, max=16)
ks = np.arange(1, 17)
pk = ks ** -2.0
pk = pk / pk.sum()
ek = float((ks * pk).sum())
out["zeta_repeat"] = {"E_k": ek, "P_k1": float(pk[0]), "P_k_ge_8": float(pk[7:].sum()),
                      "fraction_of_steps_resampling_noise": 1 / ek,
                      "note": "반복 길이 k의 갱신 과정에서 새 잡음이 뽑히는 step 비율은 1/E[k] (코드: count==0 또는 count>=n일 때 재추출)"}

# 3. 업데이트와 데이터 비율
configs = {
    "gpu_sim (IsaacLab 등, 1024 env)": {"envs": 1024, "updates_per_iter": 2, "batch": 2048},
    "cpu_sim (MuJoCo 등, 1 env)": {"envs": 1, "updates_per_iter": 1, "batch": 512},
    "vision (DMC 픽셀, 1 env, 코드 설정 UTD 0.5, 논문 Table 11)": {"envs": 1, "updates_per_iter": 0.5, "batch": 256},
}
ratios = {}
for name, c in configs.items():
    upd_per_transition = c["updates_per_iter"] / c["envs"]
    draws_per_transition = upd_per_transition * c["batch"]
    ratios[name] = {"updates_per_transition": upd_per_transition, "batch_draws_per_transition": draws_per_transition}
out["replay_ratios"] = ratios

# 4. 학습률 스케줄과 업데이트 수 (공개 스크립트 run_isaaclab.sh 값)
num_env_steps = 50_000_896
num_train_envs = 1024
interaction_steps = num_env_steps // num_train_envs  # 48,828
updates = interaction_steps * 2
warmup_steps = int(1e-6 * interaction_steps * 2)
out["gpu_schedule"] = {"interaction_steps": interaction_steps, "gradient_updates": updates,
                       "lr_warmup_steps_after_int": warmup_steps,
                       "actor_updates_period2": updates // 2,
                       "note": "warmup = int(1e-6 × 97,656) = 0 이므로 사실상 워밍업이 없다. 코사인 감쇠는 3e-4 → 1.5e-4"}

# 5. 파라미터 수 (층 정의를 손으로 센 식). 액터: 임베더 BN(2·in) + Linear(in×128, bias 없음), 블록 2개(128→512→128, BN 2개),
#    RMSNorm(128), 예측기 mean/std(각 128·A + A). 비판자: 앙상블 2개, 은닉 256, 블록 2개(256→1024→256), 출력 101 bins.
def actor_params(obs, act, h=128, blocks=2):
    emb = 2 * obs + obs * h
    blk = (h * 4 * h) * 2 + 2 * (4 * h) + 2 * h
    rms = h
    pred = 2 * (h * act + act)
    return emb + blocks * blk + rms + pred


def critic_params(obs, act, h=256, blocks=2, bins=101, members=2):
    inp = obs + act
    per = (2 * inp + inp * h) + blocks * ((h * 4 * h) * 2 + 2 * (4 * h) + 2 * h) + h + (h * bins + bins)
    return members * per


param_table = {}
for name, (o, a) in {"IsaacLab Shadow Repose (obs 157, act 20)": (157, 20),
                     "MuJoCo Playground G1 flat (obs 123, act 37)": (123, 37),
                     "MuJoCo HalfCheetah-v4 (obs 17, act 6)": (17, 6)}.items():
    ap, cp = actor_params(o, a), critic_params(o, a)
    param_table[name] = {"actor": ap, "critic_ensemble_2": cp, "actor_plus_critic": ap + cp}
out["parameter_counts"] = param_table
out["parameter_note"] = ("논문 §4.1의 '2.5M-parameter'는 액터+비판자 앙상블(온라인)에 해당한다. 액터만은 약 0.29M(Shadow 기준). "
                         "공개 코드의 SAC 기준선 Small MLP(0.2–0.5M)와 비교할 때 비판자 앙상블을 포함해야 같은 기준이 된다.")

OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
print(json.dumps({k: out[k] for k in ["target_entropy_per_dim_gaussian", "zeta_repeat", "replay_ratios", "gpu_schedule", "parameter_counts"]},
                 indent=1, ensure_ascii=False, default=float)[:3000])
print("wrote", OUT)
