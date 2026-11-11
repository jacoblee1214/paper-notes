"""추가 검증 1: 같은 합성 궤적에 세 가지 성공 라벨 규칙을 적용해 일치율과 불일치 유형을 본다.

규칙 (04 D1·D2에 적힌 코드의 세 가지)
  C3 (논문 본문): 물체가 5 cm 이상 떠서 3 s 연속 유지  (측정 높이 기준, 추적 잡음 σ = 5 mm 가정)
  Cc (실행 중 자동 라벨, charuco): 들어 올린 뒤 어떤 프레임에서든 보드 코너가 모두 보임
       (보드 가림 = 손이 가림, 기울기 > 60° 이면 보이지 않는 것으로 가정. 지속 시간 조건 없음)
  Cz (데이터셋 success 필드): 파지 이후 높이 최댓값이 50 mm 이상 (지속 시간 조건 없음, 측정 잡음 포함)
참 값(GT, 시뮬레이션의 실제 상태): 실제 높이 ≥ 5 cm가 연속 3 s 이상 (잡음 없음, 가림 없음)

합성 궤적 (10 Hz, 8 s, 수직 높이 h(t), 기울기 θ(t)). 행동 유형과 확률(가정):
  hold_ok    (0.40): 들어 올려 D ∈ [1, 6] s 유지 후 내려놓기, 미끄럼 없음
  hold_slip  (0.15): 유지 중 중간에 미끄러져 탁자에 떨어짐, 튕김(최대 4 cm) 확률 0.3
  toss       (0.15): 짧게 5–12 cm 들었다 떨어뜨림 (0.3–1.5 s)
  occ_hold   (0.15): hold_ok와 같지만 손이 보드를 가림 (보드 불가시)
  tilt_hold  (0.05): 유지 중 기울기 65–90°로 들림 (보드 불가시)
  nolift     (0.10): 거의 들리지 않음 (0–3 cm)
결과: analysis/extra_label_rules_results.json
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"

import json
import time
from pathlib import Path

import numpy as np

OUT_JSON = Path(__file__).with_name("extra_label_rules_results.json")
FS, T_END = 10.0, 8.0
N = 20000
SIG_MEAS = 0.005
THRESH_H = 0.05
TYPES = ["hold_ok", "hold_slip", "toss", "occ_hold", "tilt_hold", "nolift"]
PROBS = [0.40, 0.15, 0.15, 0.15, 0.05, 0.10]


def make_trial(rng, kind):
    n = int(T_END * FS)
    t = np.arange(n) / FS
    h = np.zeros(n)
    tilt = np.zeros(n)
    occl = np.zeros(n, bool)
    ramp = lambda x, t0, dur: np.clip((x - t0) / dur, 0, 1)  # noqa: E731
    if kind in ("hold_ok", "hold_slip", "occ_hold", "tilt_hold"):
        # 들어 올린 뒤 D초 유지, 내리는 램프 0.5 s. 가림(occ_hold)은 파지 구간 동안만, 보드가 손에 가려짐
        H = rng.uniform(0.05, 0.15)
        t_lift = rng.uniform(1.0, 2.5)
        D = rng.uniform(1.0, 6.0)
        t_down = t_lift + D
        rise = ramp(t, t_lift, 0.5) - ramp(t, t_down, 0.5)
        h = H * rise
        if kind == "hold_slip":
            t_slip = rng.uniform(t_lift + 0.3, t_down)
            # 미끄러져 탁자에 떨어짐 (t_slip 이후 높이 0), 튕김은 확률 0.3으로 최대 4 cm
            h = np.where(t >= t_slip, 0.0, h)
            if rng.random() < 0.3:
                bump = rng.uniform(0.01, 0.04)
                h = np.where((t >= t_slip + 0.3) & (t < t_slip + 0.6), bump * np.sin(np.pi * (t - t_slip - 0.3) / 0.3), h)
        if kind == "occ_hold":
            occl = (t >= t_lift) & (t <= t_down)
        if kind == "tilt_hold":
            tilt = np.where((t >= t_lift + 0.5) & (t <= t_down), rng.uniform(65, 90), 0.0)
    elif kind == "toss":
        H = rng.uniform(0.05, 0.12)
        t_lift = rng.uniform(1.0, 3.0)
        dur = rng.uniform(0.3, 1.5)
        h = H * (ramp(t, t_lift, 0.25) - ramp(t, t_lift + dur, 0.25))
    else:  # nolift
        h = rng.uniform(0.0, 0.03) * (ramp(t, rng.uniform(1, 3), 0.5))
    tilt = tilt + rng.normal(0, 2.0, n)
    return t, h, tilt, occl


def runs_ge(mask, min_len):
    """연속 True 구간 중 길이 ≥ min_len 이 있는지."""
    if not mask.any():
        return False
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return bool(np.any(ends - starts >= min_len))


def label_rules(h_true, tilt, occl, rng):
    h_meas = h_true + rng.normal(0, SIG_MEAS, len(h_true))
    dt = 1 / FS
    # C3: 측정 높이 ≥ 5 cm 가 3 s 이상 연속
    c3 = runs_ge(h_meas >= THRESH_H, int(round(3.0 / dt)))
    # Cc: 들어 올린 뒤(높이 ≥ 3 cm) 보드가 보이는 프레임이 하나라도 있으면 성공 (가림·기울기 기준)
    lifted = h_true >= 0.03
    visible = (~occl) & (tilt < 60)
    cc = bool(np.any(lifted & visible))
    # Cz: 파지 이후 측정 높이 최댓값 ≥ 50 mm (지속 조건 없음)
    cz = bool(h_meas.max() >= 0.05)
    gt = runs_ge(h_true >= THRESH_H, int(round(3.0 / dt)))
    return c3, cc, cz, gt


def main():
    t0 = time.time()
    rng = np.random.default_rng(20261111)
    kinds = rng.choice(len(TYPES), size=N, p=PROBS)
    rows = []
    for i in range(N):
        t, h, tilt, occl = make_trial(rng, TYPES[kinds[i]])
        c3, cc, cz, gt = label_rules(h, tilt, occl, rng)
        rows.append((kinds[i], c3, cc, cz, gt))
    arr = np.array(rows, dtype=int)
    k, c3, cc, cz, gt = arr.T
    out = {"setup": {"n_trials": N, "fs_hz": FS, "T_s": T_END, "type_probs": dict(zip(TYPES, PROBS)),
                     "meas_sigma_m": SIG_MEAS, "thresh_m": THRESH_H, "charuco_tilt_limit_deg": 60,
                     "seed": 20261111}}
    out["rates"] = {"GT_hold_3s_5cm": round(float(gt.mean()), 4), "C3_paper_5cm_3s": round(float(c3.mean()), 4),
                    "Cc_charuco_visible_after_lift": round(float(cc.mean()), 4),
                    "Cz_max_height_50mm": round(float(cz.mean()), 4)}
    def agree(a, b):
        return round(float(np.mean(a == b)), 4)
    out["agreement"] = {"C3_vs_Cc": agree(c3, cc), "C3_vs_Cz": agree(c3, cz), "Cc_vs_Cz": agree(cc, cz),
                        "C3_vs_GT": agree(c3, gt), "Cc_vs_GT": agree(cc, gt), "Cz_vs_GT": agree(cz, gt)}
    out["vs_GT_confusion"] = {}
    for name, lab in (("C3", c3), ("Cc", cc), ("Cz", cz)):
        out["vs_GT_confusion"][name] = {
            "false_pos_rate_among_GT_fail": round(float(np.mean(lab[gt == 0])), 4),
            "false_neg_rate_among_GT_success": round(float(1 - np.mean(lab[gt == 1])), 4)}
    # 불일치 유형 (C3 기준): 유형별로 규칙 라벨의 평균
    typ = {}
    for j, name in enumerate(TYPES):
        m = k == j
        typ[name] = {"n": int(m.sum()), "GT": round(float(gt[m].mean()), 3), "C3": round(float(c3[m].mean()), 3),
                     "Cc": round(float(cc[m].mean()), 3), "Cz": round(float(cz[m].mean()), 3)}
    out["by_behavior_type_success_rate"] = typ
    # 불일치 패턴 (세 규칙 조합별 개수, 상위 8개)
    combo = {}
    for c in set(map(tuple, arr[:, 1:4])):
        m = np.all(arr[:, 1:4] == np.array(c), axis=1)
        key = f"C3={c[0]},Cc={c[1]},Cz={c[2]}"
        combo[key] = {"n": int(m.sum()), "GT_mean": round(float(gt[m].mean()), 3),
                      "dominant_types": {TYPES[j]: int(np.sum(k[m] == j)) for j in np.argsort(-np.bincount(k[m], minlength=6))[:3]}}
    out["disagreement_patterns_top"] = dict(sorted(combo.items(), key=lambda kv: -kv[1]["n"])[:8])
    out["runtime_s"] = round(time.time() - t0, 2)
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
