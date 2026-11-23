"""토이: 논문 3.2절 촉각 시뮬레이터(식 3-7)를 numpy로 다시 구현해 '특징이 센서 배치와 힘에 얼마나 덜 의존하는가'를 본다.
실행: python3 analysis/toy_tactile_sim.py   (numpy, 수 초) → analysis/toy_tactile_sim_results.json

논문이 주지 않은 값은 가정으로 둔다(아래 ASSUMED). 결과는 '이 가정에서 식 3-7이 이렇게 움직인다'는 뜻이며 논문 수치의 재현이 아니다.
  - 발 크기 0.22 x 0.09 m, 타셀 격자 10 x 6 = 60개(논문 M = 60, 배치는 미기재)
  - 총 수직력 F = 343 N (G1 질량 약 35 kg 가정, 논문 미기재)
  - eta_n = 0.8, beta = 2, gap_min = 0, gap_max = 0.05 m, alpha = 0.3, k = 4, eps_f = 1 N (모두 미기재)
검사
  M1 접촉 면적비: 발이 계단 모서리 밖으로 나온 비율(참 접촉 비율 = 1 - 돌출)에 대해 식 7의 A_tac이 얼마나 맞는가
  M2 배치 의존: 같은 접촉을 타셀 30/60/120개 배치로 재면 A_tac이 어떻게 달라지는가 (절대 임계 eps_f 고정)
  M3 힘 의존: 같은 접촉에서 총 힘 F를 바꾸면 A_tac이 달라지는가
  M4 표 10의 소음(타셀별 ±10% 배율, XY 오프셋 sigma 1.5 mm, 상한 5 mm)이 특징을 얼마나 흔드는가
  M5 접촉 힘 펄스를 25 Hz와 100 Hz로 표본화했을 때 최댓값이 얼마나 낮게 잡히는가 (부록 C.3의 검증 논리)
  (CoP 여유는 경계 정의(발 윤곽 대 접촉 영역)가 논문에 없어 발 윤곽 기준으로만 참고용 기록한다)
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_tactile_sim_results.json"
ASSUMED = dict(foot_len=0.22, foot_wid=0.09, F=343.0, eta_n=0.8, beta=2.0, gap_min=0.0, gap_max=0.05,
               alpha=0.3, k=4, eps_f=1.0, eps=1e-6, step_h=0.15)
rng = np.random.default_rng(0)


def layout(nx, ny, L=ASSUMED["foot_len"], W=ASSUMED["foot_wid"], jitter=None):
    xs = (np.arange(nx) + 0.5) / nx * L - L / 2
    ys = (np.arange(ny) + 0.5) / ny * W - W / 2
    X, Y = np.meshgrid(xs, ys)
    return np.stack([X.ravel(), Y.ravel()], 1)


def knn_idx(P, k):
    d = np.linalg.norm(P[:, None] - P[None], axis=-1)
    return np.argsort(d, 1)[:, 1:k + 1]


def features(P, gaps, cos_align, F, p=ASSUMED, scale=None, nn=None):
    """식 3-7. gaps: 타셀별 지면까지 거리(m). cos_align: <n_foot, n_mesh> (균일 가정)."""
    g = np.clip(gaps, p["gap_min"], p["gap_max"])
    tau = (p["gap_max"] - p["gap_min"]) / np.log(50.0)
    w = np.clip((cos_align - p["eta_n"]) / (1 - p["eta_n"]), 0, 1) ** p["beta"] * np.exp(-(g - p["gap_min"]) / tau)
    f = w / w.sum() * F                                   # 식 4
    if scale is not None:
        f = f * scale
    nn = knn_idx(P, p["k"]) if nn is None else nn
    fp = (1 - p["alpha"]) * f + p["alpha"] * f[nn].mean(1)  # 식 5
    Ftac = fp.sum()
    A = float((fp > p["eps_f"]).mean())                   # 식 7
    cop = (fp[:, None] * P).sum(0) / (fp.sum() + p["eps"])
    return Ftac, A, cop


def edge_gaps(P, overhang_frac, p=ASSUMED):
    """발 앞쪽 overhang_frac 만큼이 계단 모서리 밖(높이 step_h 아래)에 있다."""
    L = p["foot_len"]
    x_edge = L / 2 - overhang_frac * L
    return np.where(P[:, 0] > x_edge, p["step_h"], 0.0)


def cop_margin(cop, p=ASSUMED, support_x_max=None):
    """직사각형 발 윤곽 안에서 CoP가 경계까지 가진 여유(m). support_x_max: 접촉 가능한 앞쪽 경계."""
    L, W = p["foot_len"], p["foot_wid"]
    xm = L / 2 if support_x_max is None else support_x_max
    return float(min(cop[0] + L / 2, xm - cop[0], cop[1] + W / 2, W / 2 - cop[1]))


res = {"assumed": ASSUMED}

# ------------------------------------------------ M1: 모서리 돌출에 대한 접촉 면적비
xy60 = layout(10, 6)
nn60 = knn_idx(xy60, ASSUMED["k"])
e1 = []
for over in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
    gaps = edge_gaps(xy60, over)
    true_frac = float((gaps == 0).mean())
    row = {"overhang": over, "true_contact_frac": round(true_frac, 3)}
    for eps_f in (0.2, 1.0, 3.0):
        p = dict(ASSUMED, eps_f=eps_f)
        Ft, A, cop = features(xy60, gaps, 1.0, ASSUMED["F"], p, nn=nn60)
        row[f"A_tac_eps{eps_f}"] = round(A, 3)
        if eps_f == 1.0:
            row["force_on_void_taxels_frac"] = round(float(
                (np.clip(gaps, 0, p["gap_max"]) > 0).sum() * np.exp(-np.log(50.0)) /
                ((gaps == 0).sum() + (gaps > 0).sum() * np.exp(-np.log(50.0)))), 4)
            row["cop_x_mm"] = round(float(cop[0] * 1000), 2)
            row["cop_margin_mm"] = round(cop_margin(cop, support_x_max=ASSUMED["foot_len"] / 2) * 1000, 2)
    e1.append(row)
res["M1_edge_overhang"] = e1

# ------------------------------------------------ M2: 타셀 배치 의존 (같은 접촉, 같은 절대 eps_f)
e2 = {}
for nx, ny in [(6, 5), (10, 6), (15, 8)]:
    P = layout(nx, ny)
    nn = knn_idx(P, ASSUMED["k"])
    out = []
    for over in [0.0, 0.3]:
        gaps = edge_gaps(P, over)
        _, A, cop = features(P, gaps, 1.0, ASSUMED["F"], nn=nn)
        out.append({"overhang": over, "M": nx * ny, "true_contact_frac": round(float((gaps == 0).mean()), 3),
                    "A_tac": round(A, 3)})
    e2[f"M={nx * ny}"] = out
res["M2_layout"] = e2

# ------------------------------------------------ M3: 총 힘 의존 (고정 배치, 부분 접촉 = 돌출 0.3)
e3 = []
gaps = edge_gaps(xy60, 0.3)
for F in (50, 100, 200, 343, 600):
    _, A, _ = features(xy60, gaps, 1.0, float(F), nn=nn60)
    e3.append({"F_N": F, "A_tac": round(A, 3), "true_contact_frac": round(float((gaps == 0).mean()), 3)})
res["M3_force_dependence"] = e3

# ------------------------------------------------ M4: 표 10 소음이 특징에 주는 흔들림
e4 = {}
for name, over in (("full_contact", 0.0), ("partial_edge_0.3", 0.3)):
    gaps = edge_gaps(xy60, over)
    Fs, As, margins, cx = [], [], [], []
    for _ in range(2000):
        scale = rng.uniform(0.9, 1.1, size=60)
        off = np.clip(rng.normal(0, 0.0015, size=(1, 2)), -0.005, 0.005)   # 에피소드 단위 오프셋(전 타셀 공통)
        P = xy60 + off
        Ft, A, cop = features(P, gaps, 1.0, ASSUMED["F"], scale=scale, nn=nn60)
        Fs.append(Ft); As.append(A); margins.append(cop_margin(cop - off[0], support_x_max=ASSUMED["foot_len"] / 2 if over else None) * 1000)
        cx.append(cop[0] * 1000)
    e4[name] = {"F_mean": round(float(np.mean(Fs)), 1), "F_sd": round(float(np.std(Fs)), 2),
                "A_mean": round(float(np.mean(As)), 4), "A_sd": round(float(np.std(As)), 4),
                "margin_mm_mean": round(float(np.mean(margins)), 2), "margin_mm_sd": round(float(np.std(margins)), 3),
                "cop_x_mm_sd": round(float(np.std(cx)), 3)}
res["M4_table10_noise"] = e4
# ------------------------------------------------ M5: 힘 펄스 표본화 (반사인 펄스, 폭 w, 위상 무작위)
def sampled_peak_ratio(width_s, rate_hz, n=4000):
    ratios = []
    for _ in range(n):
        phase = rng.uniform(0, 1.0 / rate_hz)
        t = phase + np.arange(0, 0.3, 1.0 / rate_hz)
        t0 = 0.1
        f = np.where((t >= t0) & (t <= t0 + width_s), np.sin(np.pi * (t - t0) / width_s), 0.0)
        ratios.append(f.max())
    return float(np.mean(ratios)), float(np.percentile(ratios, 5))


e5 = []
for w in (0.03, 0.05, 0.08, 0.12, 0.2):
    m25, p25 = sampled_peak_ratio(w, 25)
    m100, p100 = sampled_peak_ratio(w, 100)
    e5.append({"pulse_width_ms": int(w * 1000), "mean_peak_ratio_25Hz": round(m25, 3), "p5_25Hz": round(p25, 3),
               "mean_peak_ratio_100Hz": round(m100, 3), "expected_100_over_25": round(m100 / m25, 3)})
res["M5_sampling"] = e5
res["note"] = ("A_tac의 최소 눈금은 1/M (M=60이면 0.0167). 표 3의 실로봇 A_c 표준편차(0.011-0.044)는 인솔 타셀 수가 "
               "미기재라 눈금과 비교하지 못한다.")

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", OUT)
for r in e1:
    print(r)
print(e2)
print(e3)
print(e4)
