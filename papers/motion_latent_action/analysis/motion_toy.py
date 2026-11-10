"""잠재 행동 토이: 프레임 차이에 에이전트 움직임과 분포 이동(카메라 이동, 밝기)이 함께 있을 때
(1) 전체 차이를 쓰면 코드가 분포 이동을 잡는가, (2) 전경 마스크로 제한하면 어떻게 되는가,
(3) 코드 수 K에 따라 혼잡도(perplexity)와 행동 판독력이 어떻게 변하는가,
(4) 코드 실행에서 보인 "이미지만 자르고 마스크는 안 자르는" 불일치가 가짜 움직임을 만드는가를 본다.

k-means가 VQ 코드북을 대신한다(그래디언트 학습 없음, 단일 스레드 numpy). 목적은 메커니즘 확인이다.
실행: python3 analysis/motion_toy.py → analysis/motion_toy_results.json (수십 초)
"""
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

OUT = Path(__file__).resolve().parent / "motion_toy_results.json"
SIZE, BLOB, STEP = 32, 4, 2
MOVES = np.array([[0, 2], [0, -2], [-2, 0], [2, 0]])          # 행동 0..3: 오른쪽, 왼쪽, 위, 아래
CAM = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]])     # 카메라 이동 라벨 0..4
BRIGHT = np.array([-0.2, 0.0, 0.2])                             # 밝기 변화 라벨 0..2


def make_data(n, rng, shift_mode="none", tex_amp=0.05, fixed_pos=False):
    """shift_mode: 'none' | 'shared' (두 프레임 같은 자르기: 마스크만 어긋남) | 'indep' (프레임 t+1만 자름: 가짜 움직임)"""
    X_full, X_fg, X_bg, X_fgn, M_fg = [], [], [], [], []
    a_lab, c_lab, b_lab, s_lab = [], [], [], []
    for _ in range(n):
        tex = np.kron(rng.uniform(-tex_amp, tex_amp, (8, 8)), np.ones((4, 4)))   # 배경 질감 진폭(기본 0.05: 거의 정적인 배경)
        a = rng.integers(4)
        c = rng.integers(5)
        bl = rng.integers(3)
        p0 = np.array([14, 14]) if fixed_pos else rng.integers(2, SIZE - BLOB - 2, size=2)   # fixed_pos: 시작 위치 교란 제거
        p1 = np.clip(p0 + MOVES[a] * STEP // 2, 0, SIZE - BLOB)
        s0 = np.zeros(2, int)
        s1 = np.zeros(2, int)
        s = rng.integers(-2, 3, size=2) if shift_mode != "none" else np.zeros(2, int)
        if shift_mode == "shared":
            s0 = s
            s1 = s
        elif shift_mode == "indep":
            s1 = s
        scene0 = tex.copy()
        scene0[p0[0]:p0[0] + BLOB, p0[1]:p0[1] + BLOB] = 1.0
        scene1 = tex.copy()
        scene1[p1[0]:p1[0] + BLOB, p1[1]:p1[1] + BLOB] = 1.0
        blob1 = np.zeros_like(tex)
        blob1[p1[0]:p1[0] + BLOB, p1[1]:p1[1] + BLOB] = 1.0
        # 카메라 이동은 프레임 t에서 t+1로 장면 전체를 옮긴다. 자르기 불일치는 위 s로 따로 넣는다.
        d = CAM[c]
        f0 = np.roll(scene0, tuple(s0), axis=(0, 1))
        f1 = np.roll(scene1, tuple(d + s1), axis=(0, 1)) + BRIGHT[bl]
        m1 = np.roll(blob1, tuple(d + s1 if shift_mode == "none" else d), axis=(0, 1))
        if shift_mode == "shared":
            m1 = np.roll(blob1, tuple(d), axis=(0, 1))   # 마스크는 자르기 전 좌표 그대로
        diff = f1 - f0
        # 전경 마스크 = 에이전트가 t와 t+1 어디에든 있는 영역을 1px 팽창 (논문은 SAM2 손 마스크; 여기서는 블롭 기준 근사)
        mu = np.maximum(m1, np.roll(blob0_of(p0), tuple(s0), axis=(0, 1)))
        mdil = mu.copy()
        for ax in (0, 1):
            mdil = np.maximum(mdil, np.roll(mu, 1, axis=ax))
            mdil = np.maximum(mdil, np.roll(mu, -1, axis=ax))
        X_full.append(diff.ravel())
        X_fg.append((diff * mdil).ravel())
        X_bg.append((diff * (1 - mdil)).ravel())
        fgv = diff[mdil > 0]
        X_fgn.append((diff - fgv.mean() * mdil).ravel() * mdil.ravel())   # 가설적 보정: 전경 평균 제거(논문에 없음)
        M_fg.append(mdil.ravel())
        a_lab.append(a); c_lab.append(c); b_lab.append(bl); s_lab.append(int((s[0] + 2) * 5 + (s[1] + 2)))
    return {"full": np.array(X_full), "fg": np.array(X_fg), "bg": np.array(X_bg), "fg_norm": np.array(X_fgn),
            "action": np.array(a_lab), "cam": np.array(c_lab), "bright": np.array(b_lab),
            "shift": np.array(s_lab)}


def blob0_of(p0):
    m = np.zeros((SIZE, SIZE))
    m[p0[0]:p0[0] + BLOB, p0[1]:p0[1] + BLOB] = 1.0
    return m


def pool2(X):
    """32x32 차이를 2x2 평균풀링해 16x16(256차원)으로 줄인다."""
    n = X.shape[0]
    return X.reshape(n, SIZE, SIZE).reshape(n, SIZE // 2, 2, SIZE // 2, 2).mean(axis=(2, 4)).reshape(n, -1)


def kmeans(X, K, rng, iters=25):
    idx = rng.choice(len(X), K, replace=False)
    C = X[idx].copy()
    for _ in range(iters):
        d = (X ** 2).sum(1)[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None]
        lab = d.argmin(1)
        for k in range(K):
            if np.any(lab == k):
                C[k] = X[lab == k].mean(0)
    d = (X ** 2).sum(1)[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None]
    return d.argmin(1)


def entropy_bits(p):
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def metrics(code, labels, K):
    n = len(code)
    usage = np.bincount(code, minlength=K) / n
    ppl = 2 ** entropy_bits(usage)
    joint = np.zeros((K, labels.max() + 1))
    np.add.at(joint, (code, labels), 1)
    purity = joint.max(1).sum() / n
    pl = joint.sum(0) / n
    pk = joint.sum(1) / n
    pj = joint / n
    mi = 0.0
    for k in range(K):
        for l in range(joint.shape[1]):
            if pj[k, l] > 0:
                mi += pj[k, l] * np.log2(pj[k, l] / (pk[k] * pl[l]))
    chance = np.bincount(labels).max() / n
    return {"perplexity": round(float(ppl), 2), "purity": round(float(purity), 3),
            "chance": round(float(chance), 3), "mi_bits": round(float(mi), 3),
            "h_label_bits": round(entropy_bits(pl), 3)}


def run_mech(data, K, rng, feat):
    X = pool2(data[feat])
    X = X - X.mean(0)
    code = kmeans(X, K, rng)
    return {"action": metrics(code, data["action"], K),
            "cam": metrics(code, data["cam"], K),
            "bright": metrics(code, data["bright"], K)}


def run_all(tex_amp, fixed_pos=False):
    rng = np.random.default_rng(0)
    N = 2000
    out = {}
    data = make_data(N, rng, "none", tex_amp, fixed_pos)
    out["mechanism_K16"] = {f: run_mech(data, 16, np.random.default_rng(1), f) for f in ["full", "fg", "bg", "fg_norm"]}
    out["K_sweep"] = {K: {f: run_mech(data, K, np.random.default_rng(2), f) for f in ["full", "fg"]}
                      for K in [4, 8, 16, 32, 64]}
    d_shared = make_data(N, np.random.default_rng(3), "shared", tex_amp, fixed_pos)
    d_indep = make_data(N, np.random.default_rng(4), "indep", tex_amp, fixed_pos)
    cm = {"no_shift_correct_mask": run_mech(data, 16, np.random.default_rng(5), "fg"),
          "shared_crop_mask_not_cropped": run_mech(d_shared, 16, np.random.default_rng(5), "fg"),
          "indep_crop_frame_t1_only": run_mech(d_indep, 16, np.random.default_rng(5), "fg")}
    cm["indep_shift_label"] = metrics(kmeans(pool2(d_indep["fg"]) - pool2(d_indep["fg"]).mean(0), 16,
                                             np.random.default_rng(6)), d_indep["shift"], 16)
    out["crop_mismatch_K16_fg"] = cm
    return out


def main():
    res = {"setup": {"frame": SIZE, "pooled": 16, "N": 2000, "K_default": 16, "blob": BLOB, "step": STEP,
                     "note": "k-means를 VQ 코드북 대용으로 씀(그래디언트 학습 없음). 분포 이동 라벨은 카메라 5종과 밝기 3종. "
                             "fg_norm은 논문에 없는 가설적 보정(전경 평균 제거). 기본 배경 진폭 0.05, 민감도로 0.25"}}
    for amp in [0.05, 0.25]:
        res[f"tex{amp}"] = run_all(amp)
    res["fixed_start_pos_tex0.05"] = run_all(0.05, fixed_pos=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print("wrote", OUT)
    for key in ["tex0.05", "tex0.25", "fixed_start_pos_tex0.05"]:
        r = res[key]
        print("==", key)
        for f in ["full", "fg", "bg", "fg_norm"]:
            m = r["mechanism_K16"][f]
            print(f, "act", m["action"]["purity"], "(chance", m["action"]["chance"], ") cam", m["cam"]["purity"],
                  "(chance", m["cam"]["chance"], ") bright", m["bright"]["purity"], "(chance", m["bright"]["chance"], ") ppl", m["action"]["perplexity"])
        for K in r["K_sweep"]:
            print("K", K, "full ppl", r["K_sweep"][K]["full"]["action"]["perplexity"], "act", r["K_sweep"][K]["full"]["action"]["purity"],
                  "| fg ppl", r["K_sweep"][K]["fg"]["action"]["perplexity"], "act", r["K_sweep"][K]["fg"]["action"]["purity"])
        for k, v in r["crop_mismatch_K16_fg"].items():
            if k == "indep_shift_label":
                print(k, v["purity"], "chance", v["chance"])
            else:
                print(k, "act", v["action"]["purity"], "cam", v["cam"]["purity"], "bright", v["bright"]["purity"])


if __name__ == "__main__":
    main()
