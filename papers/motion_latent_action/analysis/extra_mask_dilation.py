"""추가 검증 1: 전경 마스크 팽창 크기(0–4 px)와 카메라 이동 크기(1, 2 px)에 따른 마스크 안 카메라 판독 (CPU numpy 1스레드).

motion_toy.py와 같은 장면 생성(32×32, 블롭 4×4, 배경 질감 진폭 0.05 또는 0.25, 행동 4종·카메라 5종·밝기 3종),
같은 2×2 풀링, 같은 k-means(K=16, VQ 대용)를 쓴다. 바꾸는 것은 두 가지뿐이다.
- 마스크 팽창: 블롭 합집합을 4방향으로 dil번 팽창 (기존 toy는 dil=1)
- 카메라 이동 크기 mag: 라벨 CAM × mag (기존 toy는 mag=1)
측정: 마스크 안 차이(fg)로 만든 코드의 카메라 순도와 기회 수준(chance)의 차이(누수 = purity − chance),
행동 순도(에이전트 정보가 남는지), 밝기 순도. 시드는 2개(데이터와 k-means).
실행: python3 analysis/extra_mask_dilation.py  (수 분 이내)
출력: analysis/extra_mask_dilation_results.json
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import motion_toy as mt  # noqa: E402

OUT = HERE / "extra_mask_dilation_results.json"
SIZE, BLOB = mt.SIZE, mt.BLOB
N = 2000
DILS = [0, 1, 2, 3, 4]
MAGS = [1, 2]
TEXS = [0.05, 0.25]
POSES = [True, False]     # True: 시작 위치 고정(03의 표 A와 같은 설정), False: 무작위(교란 있음)
SEEDS = [0, 1, 2, 3, 4]


def dilate(mask, k):
    m = mask.copy()
    for _ in range(k):
        n = m.copy()
        for ax in (0, 1):
            n = np.maximum(n, np.roll(m, 1, axis=ax))
            n = np.maximum(n, np.roll(m, -1, axis=ax))
        m = n
    return m


def make(n, rng, dil, mag, tex_amp, fixed_pos):
    cam_tab = mt.CAM * mag
    X_fg, a_lab, c_lab, b_lab = [], [], [], []
    for _ in range(n):
        tex = np.kron(rng.uniform(-tex_amp, tex_amp, (8, 8)), np.ones((4, 4)))
        a = rng.integers(4)
        c = rng.integers(5)
        bl = rng.integers(3)
        p0 = np.array([14, 14]) if fixed_pos else rng.integers(2, SIZE - BLOB - 2, size=2)
        p1 = np.clip(p0 + mt.MOVES[a] * mt.STEP // 2, 0, SIZE - BLOB)
        scene0 = tex.copy()
        scene0[p0[0]:p0[0] + BLOB, p0[1]:p0[1] + BLOB] = 1.0
        scene1 = tex.copy()
        scene1[p1[0]:p1[0] + BLOB, p1[1]:p1[1] + BLOB] = 1.0
        f1 = np.roll(scene1, tuple(cam_tab[c]), axis=(0, 1)) + mt.BRIGHT[bl]
        diff = f1 - scene0
        mu = np.zeros((SIZE, SIZE))
        mu[p0[0]:p0[0] + BLOB, p0[1]:p0[1] + BLOB] = 1.0
        b1 = np.zeros((SIZE, SIZE))
        b1[p1[0]:p1[0] + BLOB, p1[1]:p1[1] + BLOB] = 1.0
        mu = np.maximum(mu, np.roll(b1, tuple(cam_tab[c]), axis=(0, 1)))
        mdil = dilate(mu, dil)
        X_fg.append((diff * mdil).ravel())
        a_lab.append(a)
        c_lab.append(c)
        b_lab.append(bl)
    return np.array(X_fg), np.array(a_lab), np.array(c_lab), np.array(b_lab)


def probe(X, labels_dict, rng):
    Xp = mt.pool2(X)
    Xp = Xp - Xp.mean(0)
    code = mt.kmeans(Xp, 16, rng)
    out = {}
    for name, lab in labels_dict.items():
        m = mt.metrics(code, lab, 16)
        # 귀무: 라벨을 무작위로 섞은 순도. 유한 표본(K=16)에서 순도는 기회 수준보다 높게 나온다.
        null = mt.metrics(code, rng.permutation(lab), 16)["purity"]
        out[name] = {"purity": m["purity"], "chance": m["chance"], "null_purity": null,
                     "excess_vs_null": round(m["purity"] - null, 3), "mi_bits": m["mi_bits"]}
    return out


def main():
    t0 = time.time()
    res = {"setup": {"N": N, "K": 16, "pool": "2x2 mean", "seeds": SEEDS, "dil_px": DILS, "cam_mag_px": MAGS,
                     "tex_amp": TEXS, "start_pos": ["fixed (14,14)", "random"], "note": "마스크 안 차이만 사용. excess_vs_null = 실제 순도 − 라벨 무작위 섞기 순도(귀무)"},
           "grid": {}}
    for fixed in POSES:
        for tex in TEXS:
            for mag in MAGS:
                for dil in DILS:
                    runs = []
                    for sd in SEEDS:
                        X, a, c, b = make(N, np.random.default_rng(10 + sd), dil, mag, tex, fixed)
                        runs.append(probe(X, {"action": a, "cam": c, "bright": b}, np.random.default_rng(20 + sd)))
                    agg = {}
                    for name in ["action", "cam", "bright"]:
                        agg[name] = {k: round(float(np.mean([r[name][k] for r in runs])), 3)
                                     for k in ["purity", "chance", "null_purity", "excess_vs_null", "mi_bits"]}
                    pos = "fixpos" if fixed else "randpos"
                    res["grid"][f"{pos}_tex{tex}_cam{mag}px_dil{dil}"] = agg
                    print(f"{pos} tex {tex} mag {mag} dil {dil} | cam pur {agg['cam']['purity']} null {agg['cam']['null_purity']} "
                          f"| bright pur {agg['bright']['purity']} null {agg['bright']['null_purity']} "
                          f"| act pur {agg['action']['purity']} | {time.time() - t0:.0f}s", flush=True)
    res["runtime_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("wrote", OUT.name, res["runtime_s"], "s")


if __name__ == "__main__":
    main()
