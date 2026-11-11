"""추가 검증 (TactSpace R3): penalty 방식 촉각 시뮬의 침투 깊이 분포가 논문 값(중앙 62.63 µm, 범위 0–1330.98 µm)과 맞는가.

가정 모델 (논문 구현이 아니다. 공식 TaxSimIsaacLab은 접근 불가, 04 D10):
  - 센서: 12 x 12 taxel (논문 Sec. III-D, ntaxel = 12). taxel 간격 p는 논문에 없다 (가정: 1, 2, 3 mm 비교).
  - 각 taxel에서 법선(+z) 방향 광선을 쏘고, 광선과 압입체 표면의 교차로 침투 깊이를 잰다.
    압입체는 구, 원기둥(y축 방향), 평평한 펀치 셋 중 하나. 크기 반지름(또는 반폭)은 0.3, 0.7, 1.2 mm
    (논문 Fig. 4의 d = 600, 1400, 2400 µm를 지름으로 가정).
  - 압입 변위 d ~ U(0, D_max). 압입 중심 위치 ~ U(-8, 8) mm (xy).
    D_max = 1.33 mm (논문 최댓값에 맞춘 경우) 와 3.4 mm (설정 파일 max_indentation_length) 두 가지.
  - 구 침투 깊이는 해석 교차: p_i = max(0, d - (R - sqrt(R^2 - r_i^2))), r_i = taxel과 압입 중심의 거리.
    원기둥은 x 방향 거리만, 펀치는 내부에서 p_i = d.
  - 탄성 변형, 히스테리시스, 센서 전달 함수는 모델링하지 않는다 (명시적 한계).
  - 표본 20000개, 시드 0.

측정 정의 (논문의 "침투 깊이" 통계가 어떤 단위인지 본문에 없어서 세 가지를 모두 낸다):
  (A) 표본별 최댓값: 표본마다 taxel 중 최대 깊이 (접촉이 없으면 0).
  (B) 접촉 taxel별 깊이: 깊이 > 0인 taxel 전체.
  (C) 전 taxel 깊이: 0 포함 모든 taxel.
  각각 중앙값, 90분위, 최댓값, 접촉 없는 표본 비율을 보고한다.

실행: python3 analysis/extra_penalty_depth.py -> analysis/extra_penalty_depth_results.json
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "extra_penalty_depth_results.json"
N_TAXEL = 12
PITCHES = [1.0, 2.0, 3.0]
SIZES = [0.3, 0.7, 1.2]
D_MAXES = [1.33, 3.4]
N_SAMPLES = 20000
PAPER = {"median_um": 62.63, "max_um": 1330.98}
SHAPES = ["sphere", "cylinder", "punch"]


def taxel_grid(pitch):
    c = (np.arange(N_TAXEL) - (N_TAXEL - 1) / 2) * pitch
    xx, yy = np.meshgrid(c, c, indexing="ij")
    return xx.ravel(), yy.ravel()


def depth_field(shape, R, d, xc, yc, tx, ty):
    """taxel 위치에서 법선 광선이 만나는 침투 깊이 (mm). 해석적 광선-압입체 교차."""
    if shape == "sphere":
        r2 = (tx - xc) ** 2 + (ty - yc) ** 2
        inside = r2 < R * R
        sag = R - np.sqrt(np.clip(R * R - r2, 0, None))
        return np.where(inside, np.clip(d - sag, 0, None), 0.0)
    if shape == "cylinder":
        dx = (tx - xc) ** 2
        inside = dx < R * R
        sag = R - np.sqrt(np.clip(R * R - dx, 0, None))
        return np.where(inside, np.clip(d - sag, 0, None), 0.0)
    if shape == "punch":
        inside = (np.abs(tx - xc) <= R) & (np.abs(ty - yc) <= R)
        return np.where(inside, d, 0.0)
    raise ValueError(shape)


def simulate(pitch, d_max, rng):
    tx, ty = taxel_grid(pitch)
    per_sample_max = np.zeros(N_SAMPLES)
    contacted = []
    all_taxels = []
    for s in range(N_SAMPLES):
        shape = SHAPES[rng.integers(len(SHAPES))]
        R = SIZES[rng.integers(len(SIZES))]
        d = rng.uniform(0.0, d_max)
        xc, yc = rng.uniform(-8.0, 8.0, 2)
        p = depth_field(shape, R, d, xc, yc, tx, ty)
        per_sample_max[s] = p.max()
        all_taxels.append(p)
        contacted.append(p[p > 0])
    all_taxels = np.concatenate(all_taxels)
    contacted = np.concatenate(contacted) if len(contacted) and any(len(c) for c in contacted) else np.array([0.0])
    um = 1000.0  # mm -> µm
    return {
        "A_sample_max": summarize(per_sample_max * um),
        "B_contacted_taxels": summarize(contacted * um),
        "C_all_taxels": summarize(all_taxels * um),
        "frac_samples_no_contact": round(float(np.mean(per_sample_max <= 0)), 4),
    }


def summarize(x):
    x = np.asarray(x, dtype=float)
    return {"median_um": round(float(np.median(x)), 2), "p90_um": round(float(np.percentile(x, 90)), 2),
            "max_um": round(float(x.max()), 2), "min_um": round(float(x.min()), 2)}


def main():
    out = {"settings": {"n_taxel": N_TAXEL, "pitches_mm": PITCHES, "sizes_mm": SIZES, "d_max_mm": D_MAXES,
                        "samples": N_SAMPLES, "seed": 0, "shapes": SHAPES,
                        "center_range_mm": [-8, 8],
                        "note": "가정 모델. 논문 구현과 다르다. 탄성, 히스테리시스 없음. taxel 간격과 압입체 크기는 가정"},
           "paper_reference": PAPER, "grid": {}}
    for d_max in D_MAXES:
        for pitch in PITCHES:
            rng = np.random.default_rng(0)
            r = simulate(pitch, d_max, rng)
            out["grid"][f"d_max={d_max:g}mm,pitch={pitch:g}mm"] = r
            print(f"d_max={d_max:g} pitch={pitch:g}", "A", r["A_sample_max"], "B", r["B_contacted_taxels"],
                  "no_contact", r["frac_samples_no_contact"])
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
