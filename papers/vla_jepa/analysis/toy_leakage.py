"""Toy (numpy, CPU): does the predictor's input history carry future-frame information?

Motivation (04 D4): the public code feeds the predictor the frozen encoder's features of the
first T-1 time steps, but the encoder is run on the whole clip. If the encoder mixes frames
bidirectionally, the input features already contain information about the future state,
which is the leakage the paper's 'future frames are never input' claim is about.

Toy: latent state s_t follows a linear AR(1) process; each frame is observed as
x_t = M s_t + nuisance. Two frozen 'encoders' map the clip (x_0..x_T) to per-step features:
  * causal: feat_t = mix over x_0..x_t only
  * bidi:   feat_t = mix over x_0..x_T (whole clip)
We measure how well a linear readout from the INPUT history features (steps 0..T-1) predicts
the final state s_T, on held-out clips. Mechanism check on a toy, not a replication.
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)
d, obs_dim, T, K = 4, 16, 8, 12
A = np.array([[0.95, 0.10, 0, 0], [0, 0.9, 0.1, 0], [0, 0, 0.85, 0.05], [0.05, 0, 0, 0.9]])
M = rng.normal(size=(obs_dim, d)) / np.sqrt(d)

def sample_clips(n, r):
    S = np.zeros((n, T + 1, d))
    S[:, 0] = r.normal(size=(n, d))
    for t in range(T):
        S[:, t + 1] = S[:, t] @ A.T + 0.2 * r.normal(size=(n, d))
    X = S @ M.T + 0.3 * r.normal(size=(n, T + 1, obs_dim))
    return S, X

def make_encoder(seed):
    r = np.random.default_rng(seed)
    W = r.normal(size=(obs_dim, K)) / np.sqrt(obs_dim)
    causal_mix = np.tril(np.ones((T + 1, T + 1)))                 # row t: frames 0..t
    bidi_mix = r.normal(size=(T + 1, T + 1))                      # row t: all frames (full attention)
    return W, causal_mix, bidi_mix

def encode(X, mode, enc):
    W, causal_mix, bidi_mix = enc
    mix = causal_mix if mode == "causal" else bidi_mix
    # feat_t = (sum_s mix[t,s] * x_s) @ W  (each step pooled over the mixed frames)
    pooled = np.einsum("ts,nsd->ntd", mix, X)
    return pooled @ W

def readout_r2(F_in, y_tr, F_te, y_te):
    Xf = np.concatenate([F_in, np.ones((len(F_in), 1))], axis=1)
    Xt = np.concatenate([F_te, np.ones((len(F_te), 1))], axis=1)
    coef, *_ = np.linalg.lstsq(Xf, y_tr, rcond=None)
    pred = Xt @ coef
    return 1 - ((y_te - pred) ** 2).sum() / ((y_te - y_te.mean(0)) ** 2).sum()

r_tr, r_te = np.random.default_rng(1), np.random.default_rng(2)
S_tr, X_tr = sample_clips(3000, r_tr)
S_te, X_te = sample_clips(1000, r_te)
results = {}
enc = make_encoder(7)
for mode in ["causal", "bidi"]:
    F_tr = encode(X_tr, mode, enc)[:, :T].reshape(3000, -1)     # input: steps 0..T-1 only
    F_te = encode(X_te, mode, enc)[:, :T].reshape(1000, -1)
    results[f"{mode}_history_features"] = round(float(readout_r2(F_tr, S_tr[:, T], F_te, S_te[:, T])), 3)
results["current_frame_only"] = round(float(readout_r2(X_tr[:, 0], S_tr[:, T], X_te[:, 0], S_te[:, T])), 3)

out = {"setup": {"latent_dim": d, "obs_dim": obs_dim, "clip_steps": T, "feat_dim": K,
                 "n_train": 3000, "n_test": 1000, "seeds": {"data": [1, 2], "encoder": 7},
                 "target": "final latent state s_T, linear readout (OLS) from history features"},
       "r2_future_from_input": results}
with open(os.path.join(HERE, "toy_leakage_results.json"), "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
