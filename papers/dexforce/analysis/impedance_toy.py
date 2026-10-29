"""DexForce 식 (1)–(3)의 toy 검증 (CPU, numpy, 단일 스레드). 실행: python3 analysis/impedance_toy.py

T1 힘 배율: 손끝(점 질량)을 데카르트 임피던스(k_p = 310, k_v = 2, 논문 값)로 x_f = x_o + k_f f를 추종하게 하고,
   접촉 강성 k_e인 면을 눌렀을 때 재생 힘 / 시연 힘을 측정한다. 해석식 G = (1/k_e + k_f) / (1/k_p + 1/k_e)와 비교.
T2 위에서 밀어 미는 slide cube: 손끝이 위에서 눌러 마찰로 블록을 레일 위로 끈다. 필요 조건
   μ_f N > μ_r (N + m g). 시연(사람이 직접 눌러 끈 궤적)을 x_f로 재생하며 k_f를 훑어 블록이 목표까지 가는지 본다.
   k_e(손끝 패드·물체 강성)에 따라 k_f 하한이 어떻게 바뀌는지, 관측 위치(k_f = 0)가 왜 실패하는지 확인.
T3 Fig. 9 해석: 과제가 허용하는 힘 구간 [N_lo, N_hi]와 시연자가 누른 힘의 변동(시연마다 다름)에서 k_f 허용 범위의
   평균과 표준편차가 어떻게 나오는지 표본으로 확인.
결과: analysis/impedance_toy_results.json
"""
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
KP, KV, KF_USED = 310.0, 2.0, 0.0045  # 논문 v2 Sec. III
MF, DT = 0.02, 2e-4  # 손끝 유효 질량(kg, 가정), 적분 간격(s)


def gain_formula(kf, ke):
    return (1 / ke + kf) / (1 / KP + 1 / ke)


def t1_press(kfs, ke, f_demo=3.0, T=1.5):
    """1차원 누르기. 시연: 사람이 힘 f_demo로 누름 → 정지 시 침투 x_o = -f/k_e. 재생: x_d = x_o - k_f f."""
    kfs = np.asarray(kfs, float)
    xo = -f_demo / ke
    xd = xo - kfs * f_demo  # 면 안쪽이 -x
    x = np.zeros_like(kfs)
    v = np.zeros_like(kfs)
    ce = 2 * np.sqrt(ke * MF) * 0.5
    for _ in range(int(T / DT)):
        pen = np.maximum(0.0, -x)
        fc = ke * pen + ce * np.where(pen > 0, np.maximum(0.0, -v), 0.0)  # 면이 손끝을 미는 힘(+)
        u = KP * (xd - x) - KV * v
        a = (u + fc) / MF
        v += a * DT
        x += v * DT
    pen = np.maximum(0.0, -x)
    return ke * pen / f_demo


def t2_slide(kfs, ke, mu_f=0.8, mu_r=0.3, m=0.05, n_demo=2.0, dist=0.04, T=3.0, tol=0.005):
    """위에서 눌러 끄는 slide cube. 시연: 사람이 손끝을 z로 n_demo 힘에 해당하는 깊이로 누르고 x로 dist만큼 끈다.
    재생: x_f = x_o + k_f f를 임피던스로 추종. 블록 최종 위치가 시연 최종 위치 ± tol 안이면 성공."""
    g = 9.81
    t = np.arange(0, T, DT)
    s = np.clip((t - 0.3) / 2.0, 0, 1)
    s = s * s * (3 - 2 * s)
    xo_x = dist * s  # 시연에서 손끝과 블록이 함께 움직인 x
    vx_demo = np.gradient(xo_x, DT)
    # 시연 접촉력(손끝 → 물체): 수직 n_demo(아래), 접선 = 레일 마찰(움직일 때) 근사
    fn_demo = np.full_like(t, n_demo)
    ft_demo = mu_r * (n_demo + m * g) * np.tanh(vx_demo / 1e-3)
    xo_z = -n_demo / ke  # 시연 침투
    kfs = np.asarray(kfs, float)[:, None]
    xd_x = xo_x[None, :] + kfs * ft_demo[None, :]
    xd_z = xo_z - kfs * fn_demo[None, :]
    nk = kfs.shape[0]
    fx = np.zeros(nk); fz = np.zeros(nk); vfx = np.zeros(nk); vfz = np.zeros(nk)
    bx = np.zeros(nk); vb = np.zeros(nk)
    ce = 2 * np.sqrt(ke * MF) * 0.5
    inv = 1 / MF + 1 / m
    for i in range(len(t)):
        pen = np.maximum(0.0, -fz)
        N = ke * pen + ce * np.where(pen > 0, np.maximum(0.0, -vfz), 0.0)
        ux = KP * (xd_x[:, i] - fx) - KV * vfx
        uz = KP * (xd_z[:, i] - fz) - KV * vfz
        vfx = vfx + ux / MF * DT
        vfz = vfz + (uz + N) / MF * DT
        for _ in range(2):  # Coulomb 마찰을 충격량으로 (stick/slip을 수치적으로 안정하게)
            j = np.clip((vfx - vb) / inv, -mu_f * N * DT, mu_f * N * DT)  # 손끝 → 블록
            vfx = vfx - j / MF
            vb = vb + j / m
            jr = np.clip(vb * m, -mu_r * (N + m * g) * DT, mu_r * (N + m * g) * DT)  # 레일 → 블록
            vb = vb - jr / m
        fx += vfx * DT
        fz += vfz * DT
        bx += vb * DT
    err = bx - dist
    return bx, np.abs(err) <= tol


def t3_window(n_lo, n_hi, f_need, cv, n_demos=5, trials=2000, seed=0):
    """강체 근사(G = k_p k_f). 시연 힘 f = f_need · exp(σZ)이면 허용 k_f = [N_lo, N_hi] / (k_p f)."""
    rng = np.random.default_rng(seed)
    sig = np.sqrt(np.log(1 + cv * cv))
    f = f_need * np.exp(sig * rng.standard_normal((trials, n_demos)) - sig * sig / 2)
    lo, hi = n_lo / (KP * f), n_hi / (KP * f)
    return {
        "kf_min_mean": round(float(lo.mean()), 5), "kf_min_sd_within5": round(float(lo.std(axis=1, ddof=1).mean()), 5),
        "kf_max_mean": round(float(hi.mean()), 5), "kf_max_sd_within5": round(float(hi.std(axis=1, ddof=1).mean()), 5),
        "ratio_hi_over_lo": round(float(n_hi / n_lo), 2), "cv_kf_max": round(float((hi.std(axis=1, ddof=1) / hi.mean(axis=1)).mean()), 3),
    }


def main() -> None:
    out: dict = {}
    kfs = [0.0, 1 / KP, KF_USED, 0.01]
    t1 = {}
    for ke in (1e3, 3e3, 1e4, 1e5):
        sim = t1_press(kfs, ke)
        t1[f"ke={int(ke)}"] = {f"kf={k:.5f}": {"sim": round(float(s), 3), "formula": round(float(gain_formula(k, ke)), 3)}
                               for k, s in zip(kfs, sim)}
    out["T1_force_gain"] = t1

    grid = np.round(np.arange(0.0, 0.0201, 0.0005), 5)
    t2 = {"condition_N_min": round(0.3 * 0.05 * 9.81 / (0.8 - 0.3), 3), "N_demo": 2.0, "target_mm": 40}
    for ke in (1e3, 3e3, 3e4):
        bx, _ = t2_slide(grid, ke)
        row = {"kf0_block_disp_mm": round(float(bx[0] * 1000), 2),
               "kf_used_block_disp_mm": round(float(np.interp(KF_USED, grid, bx) * 1000), 2)}
        for tol in (0.005, 0.010, 0.015):
            okk = grid[np.abs(bx - 0.04) <= tol]
            row[f"window_tol{int(tol * 1000)}mm"] = [float(okk.min()), float(okk.max())] if okk.size else None
        t2[f"ke={int(ke)}"] = row
    out["T2_slide_cube"] = t2
    out["T2_curve_ke3000"] = {"kf": grid.tolist(), "block_disp_mm": [round(float(v * 1000), 2) for v in t2_slide(grid, 3e3)[0]]}

    out["T3_fig9_window"] = {
        "cv0.3_ratio4": t3_window(1.0, 4.0, 1.0 / (KP * 0.0018), 0.3),
        "cv0.3_ratio20": t3_window(1.0, 20.0, 1.0 / (KP * 0.0010), 0.3),
        "note": "f_need는 k_f 하한이 Fig. 9 근처가 되도록 고른 값. 비율(N_hi/N_lo)과 시연 힘 CV만 바꿔 평균·SD 모양을 본다",
    }
    path = ROOT / "analysis" / "impedance_toy_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items() if k != "T2_curve_ke3000"}, indent=1, ensure_ascii=False))
    print("wrote", path)


if __name__ == "__main__":
    main()
