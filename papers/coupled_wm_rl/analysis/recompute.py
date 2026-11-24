"""논문(arXiv:2602.06219 v3)의 보고 수치 재계산과 일관성 검사.
실행: python3 analysis/recompute.py   (numpy, scipy 필요, 1초 미만)
출력: analysis/recompute_results.json
시행 수가 논문에 없는 곳은 가정을 키 이름에 적었다. 그림에서 읽은 값은 digitize_fig8_results.json(그림 판독 값)이다.
"""
import json, math, os
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
R = {}

def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]

def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])

# ---- V1 표 1 (Push Cube, N = 10): 성공률 정수성, 성공 에피소드 수, Welch ----
t1 = {  # 방법: (성공 수 k, steps m/sd, straight m/sd, curv m/sd)
    "DMO": (9, (177.2, 52.2), (0.520, 0.266), (0.692, 0.206)),
    "BC": (6, (371.0, 94.0), (0.330, 0.147), (0.761, 0.151)),
    "PPO": (4, (173.5, 20.3), (0.233, 0.182), (1.222, 0.306)),
}
v1 = {"성공 수(성공률 x 10)": {m: v[0] for m, v in t1.items()},
      "성공 구간(Wilson 95%)": {m: wilson(v[0], 10) for m, v in t1.items()}}
def welch(a, na, b, nb):
    t, p = stats.ttest_ind_from_stats(a[0], a[1], na, b[0], b[1], nb, equal_var=False)
    return {"차": round(a[0] - b[0], 3), "t": round(float(t), 2), "p": round(float(p), 4)}
for i, name in [(1, "steps"), (2, "straightness"), (3, "curvature")]:
    v1["Welch_" + name + "_DMO대PPO(n=9 대 4, 표준편차는 표본 SD 가정)"] = welch(t1["DMO"][i], 9, t1["PPO"][i], 4)
    v1["Welch_" + name + "_DMO대BC(n=9 대 6)"] = welch(t1["DMO"][i], 9, t1["BC"][i], 6)
    v1["Welch_" + name + "_BC대PPO(n=6 대 4)"] = welch(t1["BC"][i], 6, t1["PPO"][i], 4)
v1["Fisher_성공_DMO대PPO(9/10 대 4/10)"] = round(fisher(9, 10, 4, 10), 4)
v1["Fisher_성공_DMO대BC(9/10 대 6/10)"] = round(fisher(9, 10, 6, 10), 4)
v1["Fisher_성공_BC대PPO(6/10 대 4/10)"] = round(fisher(6, 10, 4, 10), 4)
v1["steps_PPO_평균이_DMO보다_낮음"] = 173.5 < 177.2
v1["PPO 학습 에피소드 길이(128) 대비 steps 평균(173.5): 배포 에피소드가 학습 길이보다 긺"] = 173.5 > 128
R["V1_표1"] = v1

# ---- V2 표 2, 3 (실로봇 10회) ----
v2 = {}
for tag, rows in {
    "표2_Push-T": {"DMO": 9, "PPO": 1, "PPO_완화기준": 4, "NoDiffusion": 0},
    "표3_G1": {"DMO": 8, "BC": 7, "PPO": 1, "NoDiffusion": 0},
}.items():
    v2[tag] = {"Wilson": {k: wilson(v, 10) for k, v in rows.items()}}
v2["표2_Fisher_DMO대PPO(9 대 1)"] = round(fisher(9, 10, 1, 10), 5)
v2["표2_Fisher_DMO대PPO완화(9 대 4)"] = round(fisher(9, 10, 4, 10), 4)
v2["표3_Fisher_DMO대PPO(8 대 1)"] = round(fisher(8, 10, 1, 10), 4)
v2["표3_Fisher_DMO대BC(8 대 7)"] = round(fisher(8, 10, 7, 10), 4)
v2["표3_Fisher_BC대PPO(7 대 1)"] = round(fisher(7, 10, 1, 10), 4)
# 10회로 구분 가능한 최소 차이: 80% 검정력 근사
def power_fisher(p1, p2, n=10, sims=20000, seed=0):
    rng = np.random.default_rng(seed)
    k1 = rng.binomial(n, p1, sims); k2 = rng.binomial(n, p2, sims)
    cache = {}
    cnt = 0
    for a, b in zip(k1, k2):
        key = (a, b)
        if key not in cache:
            cache[key] = fisher(int(a), n, int(b), n) < 0.05
        cnt += cache[key]
    return cnt / sims
v2["검정력_Fisher_n10_진짜성공률 0.9 대 0.6"] = round(power_fisher(0.9, 0.6), 3)
v2["검정력_Fisher_n10_진짜성공률 0.8 대 0.7"] = round(power_fisher(0.8, 0.7), 3)
v2["검정력_Fisher_n10_진짜성공률 0.9 대 0.4"] = round(power_fisher(0.9, 0.4), 3)
R["V2_표2_표3"] = v2

# ---- V3 부록 A.3 계산 비용 산술 ----
v3 = {}
v3["시간비(1.5 s / 59 ms)"] = round(1.5 / 0.059, 2)  # 논문 '25x'
v3["메모리비(45.6 GB / 3.6 GB)"] = round(45.6 / 3.6, 2)
v3["128 액터 x 3.6 GB(한 GPU에 동시에 올릴 때)"] = round(128 * 3.6, 1)
v3["80 GB H100 필요 장수(올림)"] = math.ceil(128 * 3.6 / 80)
v3["128 액터 순차 처리 시간(59 ms x 128)[s]"] = round(0.059 * 128, 2)
v3["전역 모델 128 GPU 병렬 시간[s]"] = 1.5
v3["GPU-초 비(128 x 1.5 / (1 x 128 x 0.059))"] = round(128 * 1.5 / (128 * 0.059), 2)
v3["논문 '128x fewer GPUs'를 메모리 기준으로 환산하면(128/6)"] = round(128 / 6, 1)
v3["기울기 정렬 비 0.52/0.35"] = round(0.52 / 0.35, 2)
v3["기울기 정렬 비 0.52/0.08"] = round(0.52 / 0.08, 2)
v3["LPIPS 개선율(0.425→0.295)"] = round(1 - 0.295 / 0.425, 3)
v3["spread-to-error 비 0.665/0.275"] = round(0.665 / 0.275, 2)
R["V3_A3_연산비용"] = v3

# ---- V4 차원, 시간, DoF, 데이터 합계 ----
v4 = {}
v4["96 x 256 x 16"] = 96 * 256 * 16
v4["(96 x 256 x 16) / 68"] = round(96 * 256 * 16 / 68, 1)
v4["컨텍스트 96프레임 @5 Hz [s]"] = 96 / 5
v4["Push Cube 초기 컨텍스트 32 + 에피소드 64"] = 32 + 64
v4["G1 DoF: 3 허리 + 14 팔 + 2 그립 + 3 하체"] = 3 + 14 + 2 + 3
v4["G1 정책 출력 17 + 2"] = 17 + 2
v4["G1 배포 11 DoF = 허리 3 + 오른팔 7 + 오른손 1"] = 3 + 7 + 1
v4["패치 라운드 합(h)"] = round(sum([0.19, 0.55, 0.74, 0.80, 1.00, 0.72, 0.43]), 2)
v4["G1 데이터 합 16.5 + 4.4"] = 16.5 + 4.4
v4["G1 BC 프레임/시연 (25960/231)"] = round(25960 / 231, 1)
v4["G1 BC 시연당 길이[s] @10 Hz"] = round(25960 / 231 / 10, 1)
v4["G1 BC 에폭 수 근사(200k 스텝 x 256 / 25,960 프레임)"] = round(200000 * 256 / 25960)
v4["Push Cube BC 프레임/에피소드(37971/1200)"] = round(37971 / 1200, 1)
v4["Push Cube BC 에폭 수 근사(700k x 128 / 37,971)"] = round(700000 * 128 / 37971)
v4["Push Cube 12시간 @5 Hz 프레임 수"] = 12 * 3600 * 5
v4["Push-T 4시간 @5 Hz 프레임 수"] = 4 * 3600 * 5
v4["G1 20.9시간 @10 Hz 프레임 수"] = round(20.9 * 3600 * 10)
v4["PPO 1 반복 표본(256 x 64)"] = 256 * 64
v4["DMO 1 반복 표본(128 x 16)"] = 128 * 16
v4["DMO 4M 표본의 반복 수"] = round(4e6 / (128 * 16))
v4["PPO 48M 표본의 반복 수(Push-T)"] = round(48e6 / (256 * 64))
v4["PPO 25M 표본의 반복 수"] = round(25e6 / (256 * 64))
v4["DMO 모델 갱신 수(10000 + 64 x 반복 수 근사)"] = 10000 + 64 * round(4e6 / (128 * 16))
v4["리플레이 버퍼 10^6 대비 DMO 4M 표본(보유 비율)"] = 1e6 / 4e6
# 트랜스포머 파라미터 대략: 블록당 12 d^2 (어텐션 4 d^2 + MLP 8 d^2)
def tf(d, L): return 12 * d * d * L
pc = tf(768, 12) * 2 + tf(1536, 24)
g1 = tf(1024, 16) * 2 + tf(1536, 24)
v4["파라미터 근사(12 d^2 L) Push Cube[B]"] = round(pc / 1e9, 2)
v4["파라미터 근사(12 d^2 L) G1[B]"] = round(g1 / 1e9, 2)
v4["논문 보고: Push Cube 1.0B(v3), 1.4B(v2); G1 1.4B(v3)"] = "근사는 하한 성격(축 어텐션·잠재 토큰 층 미포함)"
R["V4_산술"] = v4

# ---- V5 그림 판독: 그림 8 과제별 비용과 그림 5 집계 끝점 ----
dg = json.load(open(os.path.join(HERE, "digitize_fig8_results.json")))
gpus = {"Push-T": (1, 1), "Push Cube": (4, 4), "G1 Grab Box": (8, 4)}  # (DMO GPU 수, PPO GPU 수): 표 4 "Compute (per Seed)" 셀을 과제 순서로 읽음
v5 = {}
for task in ["Push-T", "Push Cube", "G1 Grab Box"]:
    d = dg[task]
    S = d["samples_e7"]; T = d["time_s"]
    dmo_s, ppo_s = S["DMO"]["x_end"] * 1e7, S["PPO"]["x_end"] * 1e7
    dmo_t, ppo_t = T["DMO"]["x_end"], T["PPO"]["x_end"]
    v5[task] = {
        "DMO 표본(M)": round(dmo_s / 1e6, 2), "PPO 표본(M)": round(ppo_s / 1e6, 1),
        "표본 비(PPO/DMO)": round(ppo_s / dmo_s, 1),
        "DMO 시간(h)": round(dmo_t / 3600, 1), "PPO 시간(h)": round(ppo_t / 3600, 1),
        "벽시계 비(PPO/DMO)": round(ppo_t / dmo_t, 2),
        "표본당 시간 DMO(ms)": round(dmo_t / dmo_s * 1e3, 2), "표본당 시간 PPO(ms)": round(ppo_t / ppo_s * 1e3, 2),
        "표본당 비용 비(DMO/PPO)": round((dmo_t / dmo_s) / (ppo_t / ppo_s), 2),
        "GPU 수(DMO, PPO; 표 4 셀 위치 가정)": list(gpus[task]),
        "GPU-초 비(PPO/DMO, GPU 종류 무시)": round(ppo_t * gpus[task][1] / (dmo_t * gpus[task][0]), 2),
        "최종 보상 DMO": S["DMO"]["y_end"], "최종 보상 PPO": S["PPO"]["y_end"], "IQL 선": d["iql_line"],
        "DMO가 IQL보다 높음": S["DMO"]["y_end"] > d["iql_line"],
        "PPO가 IQL보다 높음": S["PPO"]["y_end"] > d["iql_line"],
        "SAC 끝 정규화 표본(SAC/PPO)": round(S["SAC"]["x_end"] / S["PPO"]["x_end"], 3),
        "SAC 끝 정규화 시간(SAC/PPO)": round(T["SAC"]["x_end"] / T["PPO"]["x_end"], 3),
        "DMO 정규화 표본(DMO/PPO)": round(dmo_s / ppo_s, 3), "DMO 정규화 시간(DMO/PPO)": round(dmo_t / ppo_t, 3),
        "NoRSSM 정규화 시간": round(T["NoRSSM"]["x_end"] / ppo_t, 3),
    }
R["V5_그림판독_과제별"] = v5
# V5b: 문턱 도달(그림 판독)과 NoRSSM 비용
cr = dg["threshold_crossings"]
v5b = {}
for task in ["Push-T", "Push Cube", "G1 Grab Box"]:
    c = cr[task]
    d = dg[task]
    row = {}
    ds_, dt_ = c["samples_e7"]["PPO최종값 도달_DMO"], c["time_s"]["PPO최종값 도달_DMO"]
    ps_, pt_ = c["samples_e7"]["PPO최종값 도달_PPO"], c["time_s"]["PPO최종값 도달_PPO"]
    row["PPO 최종값 도달 표본: DMO(M), PPO(M)"] = [None if ds_ is None else round(ds_ * 10, 2), None if ps_ is None else round(ps_ * 10, 1)]
    row["PPO 최종값 도달 시간: DMO(h), PPO(h)"] = [None if dt_ is None else round(dt_ / 3600, 2), None if pt_ is None else round(pt_ / 3600, 1)]
    if ds_ and ps_:
        row["표본 비(PPO/DMO)"] = round(ps_ / ds_, 1)
    if dt_ and pt_:
        row["시간 비(PPO/DMO)"] = round(pt_ / dt_, 1)
        row["GPU 가중 시간 비(GPU 종류 무시)"] = round(pt_ * gpus[task][1] / (dt_ * gpus[task][0]), 1)
    row["NoRSSM 끝 시간 / DMO 끝 시간"] = round(d["time_s"]["NoRSSM"]["x_end"] / d["time_s"]["DMO"]["x_end"], 3)
    row["NoRSSM 표본당 시간(ms)"] = round(d["time_s"]["NoRSSM"]["x_end"] / (d["samples_e7"]["NoRSSM"]["x_end"] * 1e7) * 1e3, 2)
    row["NoRSSM 최종 보상 대 DMO"] = [d["samples_e7"]["NoRSSM"]["y_end"], d["samples_e7"]["DMO"]["y_end"]]
    row["NoDiffusion 최종 보상"] = d["samples_e7"]["NoDiffusion"]["y_end"]
    row["참고: 초기 급상승 구간은 곡선이 겹쳐 DMO 도달 지점의 판독 오차가 큼"] = True
    v5b[task] = row
R["V5b_문턱도달_그림판독"] = v5b
ends = {k: [] for k in ["DMO_s", "DMO_t", "SAC_s", "SAC_t", "NoRSSM_t"]}
for task in v5:
    d = dg[task]
    ends["DMO_s"].append(d["samples_e7"]["DMO"]["x_end"] / d["samples_e7"]["PPO"]["x_end"])
    ends["DMO_t"].append(d["time_s"]["DMO"]["x_end"] / d["time_s"]["PPO"]["x_end"])
    ends["SAC_s"].append(d["samples_e7"]["SAC"]["x_end"] / d["samples_e7"]["PPO"]["x_end"])
    ends["SAC_t"].append(d["time_s"]["SAC"]["x_end"] / d["time_s"]["PPO"]["x_end"])
    ends["NoRSSM_t"].append(d["time_s"]["NoRSSM"]["x_end"] / d["time_s"]["PPO"]["x_end"])
f5 = dg["fig5_aggregate"]
R["V6_그림5_끝점"] = {
    "과제별 정규화 끝점 목록": {k: [round(x, 3) for x in v] for k, v in ends.items()},
    "과제 평균": {k: round(float(np.mean(v)), 3) for k, v in ends.items()},
    "과제 최솟값": {k: round(float(np.min(v)), 3) for k, v in ends.items()},
    "그림 5 집계 곡선 끝점(판독)": {"DMO_s": f5["samples"]["DMO"]["x_end"], "DMO_t": f5["time"]["DMO"]["x_end"],
                            "SAC_s": f5["samples"]["SAC"]["x_end"], "SAC_t": f5["time"]["SAC"]["x_end"],
                            "NoRSSM_t": f5["time"]["NoRSSM"]["x_end"]},
    "해석": "그림 5의 곡선 끝점이 과제별 끝점의 평균이 아니라 최솟값과 일치한다(판독 오차 안). 곡선은 세 과제가 모두 데이터를 가진 구간까지만 그려진 것으로 보인다(가능성).",
    "그림 5 최종 보상(판독)": {"DMO": f5["samples"]["DMO"]["y_end"], "PPO": f5["samples"]["PPO"]["y_end"], "IQL": f5["iql_line"]},
}
# 정규화 IQL 선: (IQL - lo)/(hi - lo), lo는 0-2, hi는 곡선 최댓값 근처로 가정
iql_norm = {}
for task, (lo_r, hi_r) in {"Push-T": ((0, 2), (38.0, 39.0)), "Push Cube": ((0, 2), (68.0, 70.0)), "G1 Grab Box": ((0, 1), (25.5, 27.0))}.items():
    q = dg[task]["iql_line"]
    vals = [(q - lo) / (hi - lo) for lo in lo_r for hi in hi_r]
    iql_norm[task] = [round(min(vals), 3), round(max(vals), 3)]
R["V7_IQL_정규화"] = {"과제별 범위(lo, hi 가정 구간)": iql_norm,
                    "평균 범위": [round(float(np.mean([v[0] for v in iql_norm.values()])), 3), round(float(np.mean([v[1] for v in iql_norm.values()])), 3)],
                    "그림 5 판독": f5["iql_line"]}

# ---- V8 v1 -> v3 대조 (Push Cube 표) ----
R["V8_v1대v3"] = {
    "v1 표 2 (N=3): DMO steps 91.3, straightness 0.881, curvature 0.565; PPO 134.3/0.597/0.775; BC steps 없음, 0.447/0.588": "v1 본문",
    "v3 표 1 (N=10, 성공 에피소드만): DMO 177.2/0.520/0.692; PPO 173.5/0.233/1.222; BC 371.0/0.330/0.761": "v3 표 1",
    "DMO steps v1→v3 배율": round(177.2 / 91.3, 2),
    "DMO straightness v1→v3": [0.881, 0.520],
    "PPO steps v1→v3 배율": round(173.5 / 134.3, 2),
    "BC 성공률 v1(본문 '0% success')→v3(60%, 문턱 근처를 성공으로 정의)": [0, 60],
    "Push-T 표본 캡션 v1/v2 → v3": {"DMO": ["8M", "4M"], "PPO": ["40M", "48M"]},
}

with open(os.path.join(HERE, "recompute_results.json"), "w") as f:
    json.dump(R, f, ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))
