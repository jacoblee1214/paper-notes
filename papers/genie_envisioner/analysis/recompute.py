"""논문에 보고된 수치를 다시 계산한다 (CPU, 수 초). 결과는 recompute_results.json.

V 항목(내부 검사 라벨): V1 EWMBench 합, V2 GPU 시간, V3 데이터 규모, V4 프레임 산술,
V5 Table 1 검정 (시도 횟수 가정), V6 그림 8 근사값 검정 (시도 횟수 가정), V7 Table 2.
"""
import json
import math
from pathlib import Path

from scipy.stats import fisher_exact, binomtest

OUT = Path(__file__).with_name("recompute_results.json")
res = {}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


# V1. Figure 17 EWMBench 세 단계 합과 보고된 Score
fig17 = {  # Scene, Motion, Semantics, Score(보고)
    "GE-Base": (0.9427, 1.6676, 2.0907, 4.7010),
    "Kling": (0.8888, 0.9440, 2.0370, 3.8698),
    "Hailuo": (0.8577, 0.5362, 2.0186, 3.4125),
    "COSMOS": (0.7963, 0.7085, 1.7824, 3.2872),
    "OpenSora": (0.9210, 0.3442, 1.8739, 3.1392),
    "LTX": (0.9156, 0.4002, 1.6518, 2.9676),
}
res["V1_ewmbench_sum"] = {k: {"sum": round(a + b + c, 4), "reported": d, "diff": round(a + b + c - d, 4)}
                          for k, (a, b, c, d) in fig17.items()}

# V2. 학습 GPU 시간 (A100 기준, 논문 §2.2, §3.2 보고값)
gpu_h = {
    "GE-Base-MR (7일 x 32)": 7 * 24 * 32,
    "GE-Base-LF (3일 x 32)": 3 * 24 * 32,
    "GE-Act 행동 사전학습 (3일 x 16)": 3 * 24 * 16,
    "GE-Act 영상 적응 (12시간 x 8)": 12 * 8,
    "GE-Act 행동 특화 (36시간 x 8)": 36 * 8,
}
res["V2_gpu_hours"] = gpu_h
res["V2_gpu_hours_total_base"] = gpu_h["GE-Base-MR (7일 x 32)"] + gpu_h["GE-Base-LF (3일 x 32)"]
res["V2_gpu_hours_total_act_stages"] = gpu_h["GE-Act 행동 사전학습 (3일 x 16)"] + gpu_h["GE-Act 영상 적응 (12시간 x 8)"] + gpu_h["GE-Act 행동 특화 (36시간 x 8)"]

# V3. 데이터 규모: 2,967시간 / 약 100만 에피소드
hours, episodes = 2967, 1_000_000
res["V3_sec_per_episode"] = round(hours * 3600 / episodes, 2)
res["V3_250_demo_hours_implied_sec_each"] = round(3600 / 250, 2)  # 1시간 / 250 에피소드

# V4. 프레임 산술 (AgiBot-World-Beta 30 Hz 행동, 5 Hz 영상, VAE 시간 압축 8)
action_steps, action_hz, video_hz = 54, 30, 5
video_frames = 9
res["V4_action_per_video_frame"] = action_steps / video_frames          # 6 (1:6 비율)
res["V4_action_duration_s"] = round(action_steps / action_hz, 2)         # 1.8 s
res["V4_latent_frames_LF"] = (9 - 1) // 8 + 1                            # 2
res["V4_latent_frames_MR"] = (57 - 1) // 8 + 1                           # 8
res["V4_video_frames_at_5hz_for_54_actions"] = round(action_steps / 6, 2)  # 9

# V5. Table 1 (E2E/SR, w/ S 대 w/o S). 시도 횟수는 보고되지 않음 -> n 가정 민감도
t1 = {  # (VidAW, VidAda): (E2E wS, E2E woS, SR wS, SR woS)
    "scratch (✗,✗)": (0.15, 0.30, 0.05, 0.11),
    "LTX adapt (✗,✓)": (0.00, 0.05, 0.00, 0.00),
    "GE-Base (✓,✗)": (0.81, 0.49, 0.64, 0.26),
    "GE-Base+adapt (✓,✓)": (0.89, 0.37, 0.76, 0.37),
}
res["V5_table1_parsed"] = t1
v5 = {}
for n in (10, 20, 50):
    row = {}
    # 가정: 각 셀 n회 (논문 미보고). 비율을 가장 가까운 정수 횟수로 바꾼다.
    k = lambda p: round(p * n)
    a = (k(0.81), n - k(0.81)); b = (k(0.49), n - k(0.49))
    row["E2E GE-Base wS vs woS"] = {"p_fisher": round(fisher_exact([a, b])[1], 4)}
    a = (k(0.89), n - k(0.89)); b = (k(0.37), n - k(0.37))
    row["E2E GE-Base+adapt wS vs woS"] = {"p_fisher": round(fisher_exact([a, b])[1], 4)}
    a = (k(0.64), n - k(0.64)); b = (k(0.26), n - k(0.26))
    row["SR GE-Base wS vs woS"] = {"p_fisher": round(fisher_exact([a, b])[1], 4)}
    row["Wilson 95% (E2E 0.81)"] = [round(x, 3) for x in wilson(k(0.81), n)]
    v5[f"n={n}"] = row
res["V5_table1_sensitivity"] = v5

# V6. 그림 8 (AgiBot G1) 막대 근사값. 눈으로 읽은 값(±0.03), 논문 본문 수치 아님.
fig8_approx = {
    "Packing SR: GE-Act Fast 0.72 vs GE-Act Slow 0.25": (0.72, 0.25),
    "Packing SR: GE-Act Slow 0.25 vs pi0 0.50": (0.25, 0.50),
    "Packing E2E: GE-Act Slow 0.45 vs pi0 0.80": (0.45, 0.80),
    "Make sandwich SR: GE-Act Fast 0.62 vs pi0 0.50": (0.62, 0.50),
}
v6 = {}
for name, (p1, p2) in fig8_approx.items():
    row = {}
    for n in (10, 20):
        k1, k2 = round(p1 * n), round(p2 * n)
        row[f"n={n}"] = round(fisher_exact([[k1, n - k1], [k2, n - k2]])[1], 4)
    v6[name] = row
res["V6_fig8_approx_fisher"] = v6

# V7. Table 2 (GE-Sim, EWMBench 4지표): LTX 대 COSMOS 차이
res["V7_table2_diff"] = {
    "SA (LTX-COSMOS)": round(0.94 - 0.87, 2),
    "TA (LTX-COSMOS)": round(0.98 - 0.97, 2),
    "Scene (LTX-COSMOS)": round(0.90 - 0.91, 2),
    "Logic (LTX-COSMOS)": 0.0,
    "note": "샘플 수와 반복 횟수 미보고. 차이는 검정 불가",
}


OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1)[:3500])
