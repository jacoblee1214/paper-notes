"""추가 검증 (beyond_implicit_force): 공개 ALOHA 데이터에서 δ = action − qpos 와 전류 대리값(effort)의 관계.

실행: .venv/bin/python -I analysis/extra_aloha_effort.py   (결과: analysis/extra_aloha_effort_results.json)

- 데이터: LeRobot 공개 데이터셋 lerobot/aloha_static_screw_driver의 parquet 한 개(data/chunk-000/file-000.parquet).
  에피소드 50개 × 400프레임, 50 Hz, 관절 14개(좌우 팔 각 6관절 + 그리퍼). 영상은 쓰지 않는다.
- δ = action − observation.state, e = observation.effort (단위는 데이터셋에 명시되지 않음, 원시값).
- 준정적 프레임: 관절 속도 |q_t − q_{t−1}|가 해당 관절의 중앙값 이하인 프레임.
- 중력 모델: 같은 팔의 관절 6개에 대해 e ≈ c0 + Σ(a_k sin q_k + b_k cos q_k)를 준정적 프레임에서 최소제곱.
  에피소드 0–34로 학습하고 35–49에서 검증한다. 그리퍼는 중력 항이 없다고 보고 모델을 적용하지 않는다.
- 잔차 r = e − ĝ(q). 접촉 라벨이 없으므로 "접촉 후보"는 잔차가 준정적 잔차 분포의 k·σ(MAD 기준)를 넘는 프레임으로 정의한다.
  k = 2, 3, 4로 민감도를 본다.
- 불확실성: 에피소드 단위 부트스트랩 200회(시드 0).
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "4"

import hashlib
import json
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow
import pyarrow.parquet as pq
import scipy
from scipy.stats import rankdata, spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data" / "aloha_static_screw_driver_file-000.parquet"
INFO = ROOT / "data" / "aloha_static_screw_driver_info.json"
RESULTS = HERE / "extra_aloha_effort_results.json"
FIG = ROOT / "report" / "figures" / "extra_aloha_effort.png"

JOINTS = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate", "gripper"]
NAMES = [f"{arm}_{j}" for arm in ("left", "right") for j in JOINTS]
GRIP = {6, 13}
N_BOOT = 200
SEED = 0
K_LIST = (2.0, 3.0, 4.0)
TRAIN_EP_MAX = 34  # 에피소드 0–34 학습, 35–49 검증

t0 = time.time()
tab = pq.read_table(DATA)
ep = tab["episode_index"].to_numpy()
st = np.stack(tab["observation.state"].to_numpy(zero_copy_only=False)).astype(float)
ef = np.stack(tab["observation.effort"].to_numpy(zero_copy_only=False)).astype(float)
ac = np.stack(tab["action"].to_numpy(zero_copy_only=False)).astype(float)
delta = ac - st

# 속도 (같은 에피소드 안의 후향 차분). 에피소드 첫 프레임은 제외한다.
first = np.r_[True, ep[1:] != ep[:-1]]
vel = np.zeros_like(st)
vel[1:] = np.abs(st[1:] - st[:-1])
valid = ~first

qs = np.zeros_like(st, dtype=bool)
for j in range(st.shape[1]):
    thr = np.median(vel[valid, j])
    qs[:, j] = valid & (vel[:, j] <= thr)

train = ep <= TRAIN_EP_MAX
test = ~train


def design(q_arm):
    cols = [np.ones(q_arm.shape[0])]
    for k in range(q_arm.shape[1]):
        cols += [np.sin(q_arm[:, k]), np.cos(q_arm[:, k])]
    return np.stack(cols, axis=1)


def r2(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


def auc(score_pos, score_neg):
    allv = np.r_[score_pos, score_neg]
    ranks = rankdata(allv)
    n1, n0 = len(score_pos), len(score_neg)
    u = ranks[:n1].sum() - n1 * (n1 + 1) / 2
    return float(u / (n1 * n0))


rng = np.random.default_rng(SEED)
eps = np.unique(ep)

# 1) 관절별 중력 모델과 잔차
ghat = np.zeros_like(ef)
gravity = {}
resid = np.zeros_like(ef)
for arm, off in (("left", 0), ("right", 7)):
    idx = list(range(off, off + 6))
    X = design(st[:, idx])
    for j in idx:
        if j in GRIP:
            continue
        m_tr = train & qs[:, j]
        m_te = test & qs[:, j]
        coef, *_ = np.linalg.lstsq(X[m_tr], ef[m_tr, j], rcond=None)
        ghat[:, j] = X @ coef
        gravity[NAMES[j]] = {
            "R2_train_qs": r2(ef[m_tr, j], ghat[m_tr, j]),
            "R2_test_qs": r2(ef[m_te, j], ghat[m_te, j]),
        }
for j in range(14):
    if j in GRIP:
        ghat[:, j] = np.median(ef[qs[:, j] & train, j])
    resid[:, j] = ef[:, j] - ghat[:, j]

# 2) δ와 e의 관계: Spearman(전체, 준정적), 준정적 최소제곱 기울기, 에피소드 부트스트랩 CI
def boot_spearman(x, y, mask):
    base = spearmanr(x[mask], y[mask]).statistic
    eps_m = [e for e in eps if mask[ep == e].any()]
    vals = []
    for _ in range(N_BOOT):
        pick = rng.choice(eps_m, size=len(eps_m), replace=True)
        sel = np.concatenate([np.where((ep == e) & mask)[0] for e in pick])
        vals.append(spearmanr(x[sel], y[sel]).statistic)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(base), float(lo), float(hi)


per_joint = {}
for j, name in enumerate(NAMES):
    d = delta[:, j]
    e = ef[:, j]
    rho_all = float(spearmanr(d, e).statistic)
    rho_qs, lo, hi = boot_spearman(d, e, qs[:, j])
    x = d[qs[:, j]]
    y = e[qs[:, j]]
    slope = float(np.polyfit(x, y, 1)[0]) if x.size > 2 else float("nan")
    # 잔차 r과 δ의 관계 (중력 항을 뺀 뒤)
    rho_resid_qs = float(spearmanr(d[qs[:, j]], resid[qs[:, j], j]).statistic)
    per_joint[name] = {
        "delta_std_rad": float(delta[:, j].std()),
        "effort_std": float(e.std()),
        "spearman_all": rho_all,
        "spearman_qs": rho_qs,
        "spearman_qs_ci95": [lo, hi],
        "slope_effort_per_rad_qs": slope,
        "spearman_delta_vs_resid_qs": rho_resid_qs,
    }

# 3) 접촉 후보 검출 민감도: 잔차의 준정적 MAD 기준 k σ 초과 프레임과 |δ|의 AUC (전체 프레임, 준정적 프레임)
contact = {}
for name_j, j in [(NAMES[k], k) for k in range(14) if k not in GRIP]:
    med = np.median(resid[qs[:, j], j])
    sig = 1.4826 * np.median(np.abs(resid[qs[:, j], j] - med))
    per = {}
    for k in K_LIST:
        flag = np.abs(resid[:, j] - med) > k * sig
        entry = {}
        for tag, m in (("all", valid), ("qs", qs[:, j])):
            pos = np.abs(delta[m & flag, j])
            neg = np.abs(delta[m & ~flag, j])
            entry[f"flag_frac_{tag}"] = float(flag[m].mean())
            entry[f"AUC_abs_delta_{tag}"] = (
                auc(pos, neg) if pos.size > 5 and neg.size > 5 else float("nan")
            )
        per[f"k{int(k)}"] = entry
    contact[name_j] = per

wall = time.time() - t0
results = {
    "data_source": {
        "repo": "lerobot/aloha_static_screw_driver",
        "url": "https://huggingface.co/datasets/lerobot/aloha_static_screw_driver/resolve/main/data/chunk-000/file-000.parquet",
        "file": "data/aloha_static_screw_driver_file-000.parquet",
        "bytes": DATA.stat().st_size,
        "sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "license": "MIT (meta/info.json 및 HF 데이터셋 카드 기준 확인)",
        "gated": False,
        "info_json_bytes": INFO.stat().st_size,
        "episodes": int(eps.size),
        "frames": int(ep.size),
        "fps": 50,
        "note": "영상(videos/) 미사용. effort 단위는 데이터셋에 명시되지 않음",
    },
    "versions": {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pyarrow": pyarrow.__version__,
        "matplotlib": matplotlib.__version__,
    },
    "settings": {
        "qs_rule": "관절 속도 ≤ 해당 관절 중앙값",
        "gravity_model": "같은 팔 6관절 sin/cos 선형 최소제곱 (그리퍼 제외)",
        "train_episodes": "0-34",
        "test_episodes": "35-49",
        "n_boot": N_BOOT,
        "seed": SEED,
        "k_sigma": list(K_LIST),
    },
    "frames_qs_fraction_per_joint": {NAMES[j]: float(qs[valid, j].mean()) for j in range(14)},
    "gravity_fit": gravity,
    "per_joint": per_joint,
    "contact_candidates": contact,
    "wall_time_s": round(wall, 2),
}
RESULTS.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

# 그림 한 장: 왼쪽 팔 팔꿈치 δ 대 e(준정적), 관절별 준정적 Spearman과 부트스트랩 CI
FIG.parent.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
j = NAMES.index("left_elbow")
m = qs[:, j]
ax[0].scatter(delta[m, j], ef[m, j], s=3, alpha=0.3)
ax[0].set_xlabel("δ = action − qpos (rad)")
ax[0].set_ylabel("effort (raw)")
ax[0].set_title("left_elbow, quasi-static frames")
names = [n for n in NAMES if n not in ("left_gripper", "right_gripper")]
rhos = [per_joint[n]["spearman_qs"] for n in names]
lo = [per_joint[n]["spearman_qs"] - per_joint[n]["spearman_qs_ci95"][0] for n in names]
hi = [per_joint[n]["spearman_qs_ci95"][1] - per_joint[n]["spearman_qs"] for n in names]
ax[1].bar(range(len(names)), rhos, yerr=[lo, hi], capsize=3)
ax[1].set_xticks(range(len(names)))
ax[1].set_xticklabels(names, rotation=60, ha="right", fontsize=7)
ax[1].set_ylabel("Spearman (δ, effort), quasi-static")
ax[1].axhline(0, color="k", lw=0.5)
ax[1].set_title("per-joint, episode bootstrap 95% CI")
fig.tight_layout()
fig.savefig(FIG, dpi=150)
print(f"done in {wall:.1f}s -> {RESULTS.name}")
