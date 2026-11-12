"""물리 신뢰도 지표의 임계값·프레임레이트 민감도 장난감 (numpy, CPU, 수 초).

PHUMA 논문의 지표(본문 4.3절 Evaluation, Table 1)를 합성 걸음 궤적에 적용한다.
  - Non-Floating: 접촉 프레임에서 발 높이 <= +1 cm
  - Non-Penetration: 접촉 프레임에서 발 높이 >= -1 cm
  - Non-Skating: 접촉 프레임에서 수평 발 속도 < 0.10 m/s
접촉 라벨은 생성기의 입각 구간을 그대로 쓴다(이상화). 실제 데이터에서는 접촉 비율로 정한다.
논문 모델이나 PHUMA 데이터가 아니다. 임계값과 프레임레이트 변화에 지표가 얼마나 흔들리는지만 본다.

실행: python3 analysis/sensitivity_toy.py  -> analysis/sensitivity_toy_results.json
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

OUT = Path(__file__).with_name("sensitivity_toy_results.json")
RNG = np.random.default_rng(0)
STEP_PERIOD = 0.8  # 초. 한 발 입각 0.4 s, 두 발이 반대 위상
SWING_H = 0.08  # m
STEP_LEN = 0.30  # m, 스윙 중 전진량
DURATION = 8.0
N_TRIALS = 20  # 잡음 실현 반복 (잡음이 0이면 1회면 충분)


def foot_traj(t, phase0, v_skate):
    """연속시간 발 위치 (x, z)와 접촉 라벨. 입각 중 x는 v_skate로 흘러간다."""
    ph = ((t - phase0) / STEP_PERIOD) % 1.0
    stance = ph < 0.5
    # 입각: x가 v_skate로 흐름 (발이 미끄러짐). 스윙: 전진
    t_st = np.where(stance, t - phase0, 0.0)
    x = np.where(stance, v_skate * t_st, STEP_LEN * (ph - 0.5) / 0.5)
    z = np.where(stance, 0.0, SWING_H * np.sin(np.pi * (ph - 0.5) / 0.5))
    return x, z, stance


def sample(fps, v_skate, z_offset_stance, noise_sd):
    t = np.arange(0.0, DURATION, 1.0 / fps)
    xl, zl, cl = foot_traj(t, 0.0, v_skate)
    xr, zr, cr = foot_traj(t, STEP_PERIOD / 2, v_skate)
    zl = zl + np.where(cl, z_offset_stance, 0.0)
    zr = zr + np.where(cr, z_offset_stance, 0.0)
    xs = np.stack([xl, xr], 1)
    zs = np.stack([zl, zr], 1)
    cs = np.stack([cl, cr], 1)
    if noise_sd > 0:
        xs = xs + RNG.normal(0, noise_sd, xs.shape)
        zs = zs + RNG.normal(0, noise_sd, zs.shape)
    return t, xs, zs, cs


def metrics(fps, v_skate, z_offset, noise_sd, thr_height=0.01, thr_skate=0.10):
    """접촉 프레임 기준 세 지표의 통과율(%) 평균."""
    res = {"nf": [], "np": [], "ns": []}
    for _ in range(N_TRIALS if noise_sd > 0 else 1):
        _, xs, zs, cs = sample(fps, v_skate, z_offset, noise_sd)
        dt = 1.0 / fps
        v = np.diff(xs, axis=0) / dt  # 수평 속도 (x만, 1D 가정 충분)
        c0 = cs[:-1]
        n_contact = c0.sum(0)
        z0 = zs[:-1]
        # 접촉 프레임만 센다 (발 두 개 평균)
        nf = np.array([((z0[c0[:, j], j] <= thr_height)).mean() for j in range(2)])
        np_ = np.array([((z0[c0[:, j], j] >= -thr_height)).mean() for j in range(2)])
        ns = np.array([((np.abs(v[c0[:, j], j]) < thr_skate)).mean() for j in range(2)])
        res["nf"].append(nf.mean() * 100)
        res["np"].append(np_.mean() * 100)
        res["ns"].append(ns.mean() * 100)
    return {k: round(float(np.mean(v)), 1) for k, v in res.items()}


def run():
    out = {}

    # E1. 입각 중 상수 오프셋(부유 +, 관통 -)과 임계 높이 1 cm, 0.5 cm, 2 cm, 잡음 0 및 5 mm
    offsets = [-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02]
    thrs = [0.005, 0.01, 0.02]
    e1 = {}
    for noise in (0.0, 0.005):
        for thr in thrs:
            row = {}
            for off in offsets:
                m = metrics(30, 0.0, off, noise, thr_height=thr)
                row[f"{off:+.3f}"] = {"nf": m["nf"], "np": m["np"]}
            e1[f"noise{noise*1000:.0f}mm_thr{thr*100:.1f}cm"] = row
    out["E1_height_threshold_vs_offset"] = e1

    # E2. 수평 미끄럼 속도와 임계 (0.05, 0.10, 0.15 m/s), 프레임레이트 15/30/60, 잡음 0 및 5 mm
    speeds = [0.0, 0.03, 0.05, 0.08, 0.12, 0.20]
    e2 = {}
    for fps in (15, 30, 60):
        for noise in (0.0, 0.005):
            for thr in (0.05, 0.10, 0.15):
                row = {}
                for vs in speeds:
                    m = metrics(fps, vs, 0.0, noise, thr_skate=thr)
                    row[f"{vs:.2f}"] = m["ns"]
                e2[f"fps{fps}_noise{noise*1000:.0f}mm_thr{thr:.2f}"] = row
    out["E2_skate_threshold_fps_noise"] = e2

    # E3. 잡음이 있는 발 속도 표준편차 (거짓 미끄럼 기준). 정지 발(v=0)이 임계 0.10 m/s를 넘는 비율
    e3 = {}
    for fps in (15, 30, 60):
        for noise in (0.002, 0.005, 0.01):
            dt = 1.0 / fps
            # 정지 발의 x 차분 잡음 sd = sqrt(2) * sigma / dt
            sd = np.sqrt(2) * noise / dt
            # 속도 0.10 m/s를 넘을 확률 (양측)
            p_over = 2 * (1 - norm.cdf(0.10 / sd))
            e3[f"fps{fps}_sigma{noise*1000:.0f}mm"] = {"vel_noise_sd_m_s": round(float(sd), 3),
                                                       "false_skate_pct_at_0.10": round(float(p_over) * 100, 1)}
    out["E3_false_skate_from_position_noise"] = e3

    # E4. 접촉 점수 = 프레임 평균 max(4개 영역 접촉 비율) (논문 식 12, 코드 preprocess_smplx.py와 같은 정의)
    #     발 하나가 공중에 떠 있어도 다른 발이 붙어 있으면 점수가 1이 되는지 확인
    verts = np.linspace(-0.01, 0.01, 22)  # 한 영역 22개 꼭짓점의 높이 분포 (±1 cm)

    def contact_ratio(height):
        h = height + verts
        return np.mean(np.abs(h) <= 0.025)

    scen = {}
    T = 240
    for name, (zl, zr) in {
        "walk_ok": (np.zeros(T), np.zeros(T)),
        "left_on_right_floats_6cm": (np.zeros(T), np.full(T, 0.06)),
        "both_float_5cm": (np.full(T, 0.05), np.full(T, 0.05)),
        "right_penetrates_2cm": (np.zeros(T), np.full(T, -0.02)),
    }.items():
        scores = []
        for k in range(T):
            cl = contact_ratio(zl[k])
            cr = contact_ratio(zr[k])
            # 영역 4개 = 왼발 힐·토, 오른발 힐·토. 힐·토 높이 차는 무시하고 발당 값을 두 번 넣는다
            scores.append(max(cl, cl, cr, cr))
        score = float(np.mean(scores))
        scen[name] = {"contact_score": round(score, 3), "passes_0.6": score > 0.6}
    out["E4_contact_score_max_over_regions"] = scen

    return out


if __name__ == "__main__":
    res = run()
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps(res["E4_contact_score_max_over_regions"], indent=2))
    e1 = res["E1_height_threshold_vs_offset"]
    print("E1 noise0 thr1cm NF/NP:", {k: v for k, v in e1["noise0mm_thr1.0cm"].items()})
    e2 = res["E2_skate_threshold_fps_noise"]
    for key in ("fps15_noise5mm_thr0.10", "fps30_noise5mm_thr0.10", "fps60_noise5mm_thr0.10"):
        print(key, e2[key])
    print(json.dumps(res["E3_false_skate_from_position_noise"], indent=1))
