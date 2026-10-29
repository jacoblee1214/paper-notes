"""스펀지 강성 E를 영상만으로 맞출 수 있는가: 시각 관측 vs 힘 관측의 식별성 toy (CPU, numpy). 실행: python3 analysis/identifiability_toy.py

논문 Sec. 4.1–4.3은 사람이나 코딩 에이전트가 영상과 시뮬레이션을 눈으로 비교해 물성 값을 고친다고 쓴다.
여기서는 가장 단순한 모델로, 그 비교가 E를 어디까지 정할 수 있는지 잰다.

- 모델: 스펀지 면(50 × 75 mm, 두께 30 mm, 공식 데모 bar.py의 치수)을 힘 F로 누르면 선형 탄성 압축 δ = F h / (E A)
- 시각 관측: 영상에서 압축량 δ를 픽셀 단위로 읽는다. 잡음 σ_px(가장자리 검출 0.5 px, 흐림 포함 2 px), 화면 폭 0.6 m
- 힘 관측: 로봇이 알려진 압입 δ_c를 주고 힘 F를 잡음 σ_F로 읽는다(힘 피드백·센서의 이상화)
- 표준오차 SE(E)를 해석식으로 구하고 Monte Carlo로 확인한다
"""
import json
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(0)

A = 0.050 * 0.075  # 스펀지 면적 [m²] (bar.py Layout.sponge_size)
H = 0.030          # 두께 [m]
F_PRESS = 4.0      # 닦기 누름 힘 [N] (wipe_the_bar README: stripe reads 4 N)
FOV = 0.60         # 화면 폭 [m] (가정: 양팔 작업대를 한 화면에)
CAMS = {"1080p": 1920, "4K": 3840}
SIGMA_PX = (0.5, 2.0)
SIGMA_F = 0.1      # 힘 잡음 [N] (가정)
DELTA_CMD = 0.002  # 힘 관측에서 주는 압입 [m] (가정)
E_GRID = np.geomspace(2e3, 300e3, 200)
E_MARKS = (6e3, 20e3, 80e3, 100e3)  # Fig. 7e 세 값 + 공식 twin의 80 kPa


def delta(E):
    return F_PRESS * H / (E * A)


def se_vision(E, sigma_m):
    """|∂δ/∂E| = δ/E 이므로 SE(E) = σ_δ E / δ = σ_δ E² A / (F h)."""
    return sigma_m * E ** 2 * A / (F_PRESS * H)


def se_force(E):
    """F = E A δ_c / h 이므로 SE(E) = σ_F h / (A δ_c), E와 무관."""
    return np.full_like(np.asarray(E, float), SIGMA_F * H / (A * DELTA_CMD))


def mc_vision(E, sigma_m, n=20000):
    d = delta(E) + rng.normal(0, sigma_m, n)
    d = d[d > 0]
    est = F_PRESS * H / (A * d)
    # δ 잡음이 δ에 비해 크면 1/δ 꼴이라 꼬리가 길다: 표준편차 대신 IQR/1.349(정규 환산)로 비교
    q1, q3 = np.percentile(est, [25, 75])
    return float((q3 - q1) / 1.349), float(np.median(est))


def main() -> None:
    out = {"assumptions": {"A_m2": A, "h_m": H, "F_N": F_PRESS, "fov_m": FOV, "sigma_F_N": SIGMA_F,
                           "delta_cmd_m": DELTA_CMD}}
    marks = {}
    for E in E_MARKS:
        row = {"delta_mm": round(1e3 * delta(E), 3)}
        for cam, px in CAMS.items():
            mm_per_px = 1e3 * FOV / px
            row[f"delta_px_{cam}"] = round(1e3 * delta(E) / mm_per_px, 2)
        marks[f"{int(E / 1e3)}kPa"] = row
    out["deflection_at_marks"] = marks

    curves = {}
    thresholds = {}
    for cam, px in CAMS.items():
        for s in SIGMA_PX:
            sig_m = s * FOV / px
            rel = se_vision(E_GRID, sig_m) / E_GRID
            key = f"{cam}_sigma{s}px"
            curves[key] = rel.tolist()
            # 상대 SE가 50%를 넘는 E(이 위로는 사실상 식별 불가)
            above = E_GRID[rel > 0.5]
            thresholds[key] = round(float(above[0]) / 1e3, 1) if len(above) else None
            # 20 kPa와 100 kPa를 2σ로 구분할 수 있는가
            sep = abs(delta(20e3) - delta(100e3)) / (sig_m * np.sqrt(2))
            out.setdefault("separation_20_vs_100kPa_in_sigma", {})[key] = round(float(sep), 1)
            sep2 = abs(delta(80e3) - delta(100e3)) / (sig_m * np.sqrt(2))
            out.setdefault("separation_80_vs_100kPa_in_sigma", {})[key] = round(float(sep2), 2)
    curves["force"] = (se_force(E_GRID) / E_GRID).tolist()
    out["E_where_vision_rel_SE_exceeds_50pct_kPa"] = thresholds
    out["force_SE_kPa"] = round(float(se_force(1.0)) / 1e3, 3)
    out["force_rel_SE_at_100kPa"] = round(float(se_force(1.0)) / 100e3, 4)

    # Monte Carlo로 해석식 확인 (1080p, 0.5 px)
    sig_m = 0.5 * FOV / CAMS["1080p"]
    mc = {}
    for E in (20e3, 80e3):
        sd, med = mc_vision(E, sig_m)
        mc[f"{int(E / 1e3)}kPa"] = {"analytic_SE_kPa": round(float(se_vision(E, sig_m)) / 1e3, 2),
                                    "mc_robust_SD_kPa": round(sd / 1e3, 2), "mc_median_kPa": round(med / 1e3, 1)}
    out["monte_carlo_check_1080p_0.5px"] = mc

    out["_curves"] = {"E_kPa": (E_GRID / 1e3).tolist(), **curves}
    path = ROOT / "analysis" / "identifiability_toy_results.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for k, v in out.items():
        if k != "_curves":
            print(k, json.dumps(v, ensure_ascii=False))
    print("wrote", path)


if __name__ == "__main__":
    main()
