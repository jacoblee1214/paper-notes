#!/usr/bin/env python3
"""논문 벡터 그림(막대·마커)의 좌표에서 값을 읽어 figure_values.json으로 저장한다.

입력: paper/paper.pdf (arXiv:2610.18991v1, 다시 받는 명령은 README 참조)
방법: PyMuPDF로 각 쪽의 벡터 도형(막대 사각형, 마커, 오차 막대)과 축 눈금 숫자의
      위치를 읽어 y 좌표를 값으로 바꾼다. 눈금 숫자 중심은 눈금선보다 약 0.5 pt 아래라서
      막대 바닥(값 0)을 기준점으로 쓴다. 판독 오차는 대략 눈금 간격의 1% 안쪽이다.
실행: python3 analysis/digitize_figures.py
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import fitz  # PyMuPDF

HERE = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.join(HERE, "..", "paper", "paper.pdf")
OUT = os.path.join(HERE, "figure_values.json")

doc = fitz.open(PDF)


def words(pi, ymin, ymax, xmin=0, xmax=700):
    out = []
    for w in doc[pi].get_text("words"):
        x0, y0, x1, y1, t = w[:5]
        if ymin < y0 < ymax and xmin < x0 < xmax:
            out.append(((x0 + x1) / 2, (y0 + y1) / 2, t))
    return out


def draws(pi):
    return doc[pi].get_drawings()


def rects(d):
    """도형 d 안의 사각형 항목을 (x0,y0,x1,y1) 목록으로."""
    return [tuple(it[1]) for it in d["items"] if it[0] == "re"]


def slope_from_ticks(ticks):
    """[(값, y중심)] -> pt per unit (최소제곱)."""
    n = len(ticks)
    mv = sum(v for v, _ in ticks) / n
    my = sum(y for _, y in ticks) / n
    num = sum((v - mv) * (y - my) for v, y in ticks)
    den = sum((v - mv) ** 2 for v, _ in ticks)
    return -num / den  # y는 아래로 증가


res = {}

# ---------------- Fig. 5 (p.5) ----------------
D = draws(4)
# A: 접촉 높이 RMSE (mm), 눈금 0/20/40
sA = slope_from_ticks([(0, 178.19), (20, 146.14), (40, 114.10)])
base = D[2]["rect"].y1
res["fig5A_contact_height_rmse_mm"] = {
    "direct": (base - D[2]["rect"].y0) / sA,
    "stance": (D[7]["rect"].y1 - D[7]["rect"].y0) / sA,
}
# B: 24-step 하드웨어 관절 위치 MSE (1e-3 rad^2), 눈금 0/1/2/3
sB = slope_from_ticks([(0, 178.19), (1, 153.62), (2, 129.05), (3, 104.48)])
res["fig5B_hw_q_mse_1e-3_rad2"] = {
    "direct": (D[15]["rect"].y1 - D[15]["rect"].y0) / sB,
    "stance": (D[20]["rect"].y1 - D[20]["rect"].y0) / sB,
    "contact_uan": (D[25]["rect"].y1 - D[25]["rect"].y0) / sB,
}
# C: sim-to-sim 24-step q RMSE (rad), 눈금 0/0.04/0.08
sC = slope_from_ticks([(0, 178.21), (0.04, 145.45), (0.08, 112.70)])
y0C = 177.71
res["fig5C_q_rmse_rad_estimated"] = {
    "without_residual": (y0C - D[38]["rect"].y0 - 2.0) / sC,
    "with_residual": (y0C - D[39]["rect"].y0 - 2.0) / sC,
}

# ---------------- Fig. 6C (p.6) ----------------
D = draws(5)
sC6 = slope_from_ticks([(0, 164.25), (0.10, 139.03), (0.20, 113.81)])
b0 = D[23]["rect"].y1
no_uan = (b0 - D[23]["rect"].y0) / sC6
uan = (D[28]["rect"].y1 - D[28]["rect"].y0) / sC6
res["fig6C_torque_usage_rel_error"] = {
    "no_uan": no_uan,
    "uan": uan,
    "reduction_pct": 100 * (1 - uan / no_uan),
}

# ---------------- Fig. 7 (p.7) ----------------
D = draws(6)
# A: 달성 속도 (m/s). 눈금 0/1/2/3, 프레임 바닥 153.90
sA7 = slope_from_ticks([(0, 154.40), (1, 133.94), (2, 113.48), (3, 93.02)])
y0A = 153.90
ctr = lambda r: (r.y0 + r.y1) / 2  # noqa: E731
cmds = [1.0, 2.0, 2.5, 3.0]
base_pts = [D[i]["rect"] for i in (44, 45, 46)]
uan_pts = [D[i]["rect"] for i in (47, 48, 49, 50)]
res["fig7A_achieved_speed_mps"] = {
    "command": cmds,
    "baseline": [(y0A - ctr(r)) / sA7 for r in base_pts] + [0.0],
    "uan": [(y0A - ctr(r)) / sA7 for r in uan_pts],
    "uan_3.0_errorbar_halfwidth": (D[38]["rect"].y1 - D[38]["rect"].y0 + D[40]["rect"].y1 - D[40]["rect"].y0) / 2 / sA7 * 1.0,
}
# B: 토크 감소율 (%). 눈금 0/10/20/30, 영선 142.74
sB7 = slope_from_ticks([(0, 143.25), (10, 127.30), (20, 111.36), (30, 95.42)])
y0B = 142.74
eb = {  # 오차 막대 (위끝 y, 아래끝 y)
    1.0: (103.54, 118.80),
    2.0: (101.73, 113.11),
    2.5: (112.13, 128.51),
}
res["fig7B_torque_reduction_pct"] = {
    str(k): {
        "mean": (y0B - (a + b) / 2) / sB7,
        "halfwidth": (b - a) / 2 / sB7,
    }
    for k, (a, b) in eb.items()
}
# C: 측정 힘 (N). 눈금 0/5/10/15/20, 영선 154.26
sC7 = slope_from_ticks([(0, 154.76), (5, 139.18), (10, 123.60), (15, 108.02), (20, 92.44)])
y0C7 = 154.26
bpts = [D[i]["rect"] for i in (108, 109, 110, 111)]
upts = [D[i]["rect"] for i in (112, 113, 114, 115)]
res["fig7C_measured_force_N"] = {
    "command": [5, 10, 15, 20],
    "baseline": [(y0C7 - ctr(r)) / sC7 for r in bpts],
    "uan": [(y0C7 - ctr(r)) / sC7 for r in upts],
}

# ---------------- Fig. 8 (p.8) ----------------
D = draws(7)
# A: 20 s 생존율 (%). 눈금 0/50/100, 바닥 155.17
sA8 = slope_from_ticks([(0, 155.32), (50, 137.28), (100, 119.25)])
y0 = 155.17
names = ["direct", "estimated", "base_oracle", "contact_oracle", "pose_oracle", "all_oracle"]
res["fig8A_survival_pct"] = {
    n: (y0 - D[i]["rect"].y0) / sA8 for n, i in zip(names, range(20, 26))
}
# B: 속도 추종 XY RMSE (m/s). 눈금 0/0.25/0.5/0.75
sB8 = slope_from_ticks([(0, 156.46), (0.25, 142.52), (0.5, 128.58), (0.75, 114.65)])
y0 = 155.95
conds = ["baseline", "wtw_uan", "wtw_vel_uan", "expert"]
tgt = ["easy", "medium", "hard"]
res["fig8B_xy_rmse_mps"] = {
    c: {t: (y0 - r[1]) / sB8 for t, r in zip(tgt, rects(D[i]))}
    for c, i in zip(conds, (47, 60, 73, 86))
}
# C: 제곱 토크 (N^2 m^2). 눈금 1000/2000/3000, 바닥 156.37
sC8 = slope_from_ticks([(1000, 141.48), (2000, 126.77), (3000, 112.06)])
y0 = 156.37
res["fig8C_torque_sq"] = {
    c: {t: (y0 - r[1]) / sC8 for t, r in zip(tgt, rects(D[i]))}
    for c, i in zip(conds, (103, 116, 129, 142))
}

# ---------------- Fig. 11 (p.22) ----------------
D = draws(21)
s11 = slope_from_ticks([(0, 372.95), (25, 353.12), (50, 333.29), (75, 313.46), (100, 293.62)])
y0 = 372.45
labels = ["expert", "base", "wm_33h", "wm_ft_33h", "uan_5min"]
res["fig11_target_env_norm_len_pct"] = {
    n: (y0 - r[1]) / s11 for n, r in zip(labels, rects(D[26]))
}
# 오차 막대 (표준편차, 캡션 "with standard deviation"): 주황 막대 위치의 세로선 반길이
eb11 = {"base": (326.12, 367.06), "wm_33h": (366.26, 367.85), "wm_ft_33h": (317.08, 367.21)}
res["fig11_target_env_sd_pct"] = {k: (b - a) / 2 / s11 for k, (a, b) in eb11.items()}
res["fig11_train_env_norm_len_pct"] = {
    n: (y0 - r[1]) / s11 for n, r in zip(labels, rects(D[5]))
}

# ---------------- Fig. 13 (p.25) ----------------
D = draws(24)
s13 = slope_from_ticks([(0, 210.68), (50, 177.18), (100, 143.68)])
y0 = 210.17
res["fig13_survival_pct"] = {
    "clean": (y0 - D[2]["rect"].y0) / s13,
    "imu_bias_deg": {k: (y0 - r[1]) / s13 for k, r in zip(("0.5", "1", "2"), rects(D[7]))},
    "vel_lpf_hz": {k: (y0 - r[1]) / s13 for k, r in zip(("20", "10", "5"), rects(D[20]))},
}

with open(OUT, "w") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)
json.dump(res, sys.stdout, indent=1, ensure_ascii=False)
print()
