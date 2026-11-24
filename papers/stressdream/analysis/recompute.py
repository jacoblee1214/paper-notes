"""StressDream (arXiv:2606.00267 v2) 보고 수치의 재계산과 일관성 검사.

논문 본문·부록 표의 숫자와 그림에서 읽은 값을 다시 계산한다. 그림에서 읽은 값은 "그림 판독 값"이며
오차(약 ±1 %p)가 있다. 시행 수를 논문이 적지 않은 경우 가정한 값을 결과 JSON의 assumptions에 적는다.

실행: python3 analysis/recompute.py   → analysis/recompute_results.json
의존: numpy, scipy (수 초)
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.special import digamma, gammaln, polygamma

OUT = Path(__file__).with_name("recompute_results.json")
R = {"assumptions": {}}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


# ---------------------------------------------------------------- V1 차원 산술
def v1_dims():
    out = {}
    # Vista: 25 프레임 x 4 채널 x (576/8) x (1024/8)
    d_vista = 25 * 4 * (576 // 8) * (1024 // 8)
    # Ctrl-World: 5 프레임 x 4 채널 x (3 시점 x 192/8) x (320/8)
    d_ctrl = 5 * 4 * (3 * (192 // 8)) * (320 // 8)
    d_dubins = 4 * (128 // 8) * (128 // 8)
    cfg = {"Vista": (d_vista, 921600, 192), "Ctrl-World": (d_ctrl, 57600, 240), "Dubins": (d_dubins, 1024, 16)}
    for name, (d, paper_d, k) in cfg.items():
        out[name] = {"D_계산": d, "D_논문": paper_d, "일치": d == paper_d, "k": k, "D_나머지": d % k, "m=D/k": d / k}
    out["서론_근사"] = {"Vista_1M_대비": 921600 / 1e6, "Ctrl-World_50K_대비": 57600 / 5e4}
    R["V1_dims"] = out


# ---------------------------------------------------------------- V2 껍질 폭과 투영 문턱
def v2_shell():
    out = {}
    for name, D in (("Dubins", 1024), ("Ctrl-World", 57600), ("Vista", 921600)):
        # R ~ chi_D : E[R], Var[R]
        mean_r = math.exp(gammaln((D + 1) / 2) - gammaln(D / 2)) * math.sqrt(2)
        var_r = D - mean_r ** 2
        sd = math.sqrt(var_r)
        # |R - sqrt(D)| > 3 의 정확 확률 (chi 분포)
        lo, hi = (math.sqrt(D) - 3) ** 2, (math.sqrt(D) + 3) ** 2
        p_out = stats.chi2.cdf(lo, D) + stats.chi2.sf(hi, D)
        out[name] = {"sqrtD": math.sqrt(D), "sd_R": sd, "문턱_3.0은_sd의_배수": 3.0 / sd,
                     "문턱_밖_확률": p_out, "문턱_상대폭_%": 100 * 3.0 / math.sqrt(D)}
    R["V2_shell"] = out


# ---------------------------------------------------------------- V3 보조정리 2 기댓값
def v3_isotropy():
    rng = np.random.default_rng(0)
    out = {}
    for name, D, k in (("Dubins", 1024, 16), ("Ctrl-World", 57600, 240), ("Vista", 921600, 192)):
        m = D // k
        analytic = (k + 1) / m
        # Monte Carlo (Vista는 m이 커서 블록 수를 줄여 확인)
        mm = min(m, 4800)
        vals = []
        for _ in range(40):
            e = rng.standard_normal((mm, k))
            S = e.T @ e / mm
            vals.append(np.linalg.norm(S - np.eye(k)) ** 2 / k)
        out[name] = {"k": k, "m": m, "E_C_iso_식": analytic, "몬테카를로(m<=4800)": float(np.mean(vals)),
                     "몬테카를로_m": mm, "해당_m의_식": (k + 1) / mm}
    # Table 4 의 Dubins 기준선 0.02 를 내려면 어떤 k 인가 (D = 1024)
    ks = [k for k in (1, 2, 4, 8, 16, 32) if 1024 % k == 0]
    out["Dubins_k별_E_C_iso"] = {str(k): (k + 1) / (1024 // k) for k in ks}
    R["V3_isotropy"] = out


# ---------------------------------------------------------------- V4 이동 예산
def v4_drift_budget():
    out = {}
    # 알고리즘 1: 반복당 ||g|| <= 1.0, lr 1.0, 반복 m. 코드: 노름 클립 20(Vista) / 20(Ctrl-World 최종) / 5(Dubins)
    cfg = {"Vista": (921600, 20, 20.0), "Ctrl-World": (57600, 10, 20.0), "Dubins": (1024, 10, 5.0)}
    for name, (D, m, code_clip) in cfg.items():
        indep = math.sqrt(2 * D)           # 독립 두 표본 사이 거리
        alg = m * 1.0 * 1.0                # 알고리즘 1 상한
        code = m * 1.0 * code_clip         # 코드 상한(lr 1.0, 노름 클립)
        out[name] = {"반복": m, "독립표본_거리_sqrt(2D)": indep, "알고리즘1_이동_상한": alg,
                     "알고리즘1_상한/거리_%": 100 * alg / indep, "코드_노름클립": code_clip,
                     "코드_이동_상한": code, "코드_상한/거리_%": 100 * code / indep,
                     "코드_상한/sqrtD_%": 100 * code / math.sqrt(D)}
        # 한 방향으로 B만큼 이동할 때 노름 변화(표준편차 단위)와 로그밀도 변화
        sd_r = 0.7071
        for B in (alg, code):
            dn = (math.sqrt(D + B * B) - math.sqrt(D))
            out[name][f"B={B:g}_노름변화_sd배"] = dn / sd_r
            out[name][f"B={B:g}_로그밀도감소_nats"] = B * B / 2
        out[name]["로그밀도_sd_nats(sqrt(D/2))"] = math.sqrt(D / 2)
    R["V4_drift"] = out


# ---------------------------------------------------------------- V5 표 4 노름 로그확률
def v5_table4_norm():
    rng = np.random.default_rng(1)
    D = 1024
    out = {"peak_logpdf": float(stats.chi2.logpdf(D - 2, D))}
    # 공칭 표본에서 궤적당 최소 로그확률이 -7.6 이 되려면 길이 L 이 얼마인가
    res = {}
    for L in (10, 20, 40, 50, 60, 80, 100, 200):
        x = rng.chisquare(D, size=(4000, L))
        mn = stats.chi2.logpdf(x, D).min(axis=1)
        res[str(L)] = float(mn.mean())
    out["공칭_궤적당_최소_평균_logpdf(L별)"] = res
    out["표4_공칭"] = -7.6
    out["표4_StressDream"] = -6.3
    # 정규 근사: 최소 -6.3 은 평균적으로 몇 sd 인가
    pk = out["peak_logpdf"]
    out["z_공칭(-7.6)"] = math.sqrt(2 * (pk + 7.6))
    out["z_SD(-6.3)"] = math.sqrt(2 * (pk + 6.3))
    out["z_정규화없음(-717.6)"] = math.sqrt(2 * (pk + 717.6))
    R["V5_table4_norm"] = out


# ---------------------------------------------------------------- V6 회수율 (그림 5, 본문)
def v6_recall():
    n = 100
    out = {}
    for name, k in (("Nominal(N=1)", 54), ("Best-of-N", 71), ("StressDream", 94)):
        lo, hi = wilson(k, n)
        out[name] = {"k": k, "n": n, "Wilson95_%": [round(lo, 1), round(hi, 1)]}
    # 비쌍대 Fisher
    for a, b, label in ((54, 94, "Nominal 대 StressDream"), (71, 94, "BoN 대 StressDream"), (54, 71, "Nominal 대 BoN")):
        _, p = stats.fisher_exact([[a, n - a], [b, n - b]])
        out[f"Fisher_{label}"] = p
    # 같은 100개 궤적이라면 McNemar 의 가능한 범위: 54 -> 94 에서 불일치쌍 b + c
    rows = []
    for c in range(0, 7):                       # Nominal 만 맞힌 수 c <= 6 (StressDream 이 놓친 6개 안)
        b = 40 + c                              # StressDream 만 맞힌 수
        p = stats.binomtest(min(b, c), b + c, 0.5).pvalue
        rows.append({"c": c, "b": b, "McNemar_정확_p": p})
    out["McNemar_가능범위"] = rows
    R["V6_recall"] = out
    R["assumptions"]["V6"] = "세 방법이 같은 100개 실패 궤적에서 평가되었다고 가정(본문 6.1절). 쌍 구조(불일치 수)는 논문에 없어 가능 범위만 계산."


# ---------------------------------------------------------------- V7 정책 성공률 (그림 8, 26)
def v7_policy():
    out = {}
    n_trial = 6 * 20    # 6 과제 x 20 롤아웃 (그림 8 캡션)
    R["assumptions"]["V7"] = "그림 8 캡션 '20 rollouts per task' 로 팔당 120회, 과제별 20회 가정."
    feasible = {}
    for target in (39, 71):
        feasible[str(target)] = [k for k in range(0, n_trial + 1) if round(100 * k / n_trial) == target]
        # 반올림 대신 내림도 확인
    feasible_floor = {str(t): [k for k in range(0, n_trial + 1) if math.floor(100 * k / n_trial) == t] for t in (39, 71)}
    out["정수_가능_k_반올림"] = feasible
    out["정수_가능_k_내림"] = feasible_floor
    k1, k2 = 47, 85
    out["Nominal"] = {"k": k1, "n": n_trial, "%": 100 * k1 / n_trial, "Wilson95_%": [round(x, 1) for x in wilson(k1, n_trial)]}
    out["Robust"] = {"k": k2, "n": n_trial, "%": 100 * k2 / n_trial, "Wilson95_%": [round(x, 1) for x in wilson(k2, n_trial)]}
    _, p = stats.fisher_exact([[k2, n_trial - k2], [k1, n_trial - k1]])
    out["Fisher_Robust_대_Nominal"] = p
    # 그림 26 판독 값 (400 dpi 픽셀 측정, +-0.5 %p)
    nom = [30.4, 65.7, 46.4, 15.4, 30.1, 50.4]
    rob = [51.6, 87.8, 80.5, 59.9, 74.3, 74.3]
    tasks = ["Block Stack", "Utensil Pick", "Knife Put", "Coffee Bean", "Coffee Bag", "Candy Bag"]
    rows = []
    for t, a, b in zip(tasks, nom, rob):
        ga = round(a / 5) * 5
        gb = round(b / 5) * 5
        rows.append({"과제": t, "Nominal_판독": a, "5%격자_차": round(a - ga, 1), "Robust_판독": b, "5%격자_차(Robust)": round(b - gb, 1)})
    out["그림26_판독"] = rows
    out["그림26_판독_평균"] = {"Nominal": float(np.mean(nom)), "Robust": float(np.mean(rob))}
    off = [abs(r["5%격자_차"]) for r in rows] + [abs(r["5%격자_차(Robust)"]) for r in rows]
    out["격자_1.4_이상_벗어난_막대"] = int(sum(1 for x in off if x >= 1.4))
    # 5% 격자로 반올림하면 합이 47, 85 인가
    out["격자_반올림_합_Nominal"] = int(sum(round(a / 5) for a in nom))
    out["격자_반올림_합_Robust"] = int(sum(round(b / 5) for b in rob))
    # 과제 6개 모두 Robust > Nominal : 부호 검정
    out["부호검정_6/6_양측"] = stats.binomtest(6, 6, 0.5).pvalue
    # 과제별 Fisher (격자 반올림 k/20 가정)
    per = []
    for t, a, b in zip(tasks, nom, rob):
        ka, kb = round(a / 5), round(b / 5)
        _, pp = stats.fisher_exact([[kb, 20 - kb], [ka, 20 - ka]])
        per.append({"과제": t, "k_nominal": ka, "k_robust": kb, "Fisher_p": pp})
    out["과제별_Fisher(k/20 가정)"] = per
    R["V7_policy"] = out


# ---------------------------------------------------------------- V8 그림 22 (과제별 회수율, 거짓 양성 유사율)
def v8_fig22():
    tasks = ["Block Stack", "Utensil Pick", "Knife Put", "Coffee Bean", "Coffee Bag", "Candy Bag"]
    sc = 100 / 99.2
    nom = [45.1, 60.8, 79.4, 79.4, 40.6, 28.5]
    bon = [58.8, 66.1, 92.7, 99.2, 70.1, 49.9]
    sd = [94.7, 93.9, 99.2, 99.2, 99.2, 70.9]
    data = [[x * sc if x > 99 else x for x in v] for v in (nom, bon, sd)]
    out = {"그림22L_판독": {"nominal": [round(x, 1) for x in data[0]], "bon": [round(x, 1) for x in data[1]],
                             "stressdream": [round(x, 1) for x in data[2]]}}
    out["단순평균(과제 6개)"] = [round(float(np.mean(v)), 1) for v in data]
    # 본문 54/71/94 가 100개 궤적의 풀링이라면 과제별 개수 n_i 의 가능한 분해
    tol = 1.3
    cand = []
    for i in range(6):
        L = []
        for n in range(4, 60):
            ks = []
            ok = True
            for m in range(3):
                v = data[m][i]
                k = round(v * n / 100)
                if abs(100 * k / n - v) > tol:
                    ok = False
                    break
                ks.append(k)
            if ok:
                L.append((n, tuple(ks)))
        cand.append(L)
    sols = []

    def rec(i, tot, cur):
        if i == 6:
            if tot == 100:
                sols.append(list(cur))
            return
        for n, ks in cand[i]:
            if tot + n <= 100:
                cur.append((n, ks))
                rec(i + 1, tot + n, cur)
                cur.pop()

    rec(0, 0, [])
    good = []
    for s in sols:
        pooled = [round(sum(x[1][m] for x in s)) for m in range(3)]
        if pooled == [54, 71, 94]:
            good.append({"n_i": [x[0] for x in s], "k_nominal": [x[1][0] for x in s], "k_bon": [x[1][1] for x in s],
                         "k_sd": [x[1][2] for x in s]})
    out["허용오차_%p"] = tol
    out["n_i_합100_분해_수"] = len(sols)
    out["풀링_54_71_94_일치_분해"] = good
    R["assumptions"]["V8"] = "100개 실패 궤적이 6과제에 나뉘어 있고 그림 22 막대는 k_i/n_i 라고 가정(n_i 는 논문에 없음). 판독 허용오차 1.3 %p."
    # 그림 22R (정답이 실패가 아닌 궤적 중 WM이 실패를 상상한 비율, 그림 판독 값 약 +-2 %p)
    fp_nom = [0, 12, 12, 0, 1, 1]
    fp_bon = [0, 16, 24, 9, 1, 1]
    fp_sd = [16, 33, 49, 46, 13, 1]
    out["그림22R_판독"] = {"nominal": fp_nom, "bon": fp_bon, "stressdream": fp_sd}
    out["그림22R_단순평균"] = [round(float(np.mean(v)), 1) for v in (fp_nom, fp_bon, fp_sd)]
    # 과제별 Youden J = 회수율 - 거짓양성유사율
    J = []
    for i, t in enumerate(tasks):
        J.append({"과제": t,
                  "J_nominal": round((data[0][i] - fp_nom[i]) / 100, 2),
                  "J_bon": round((data[1][i] - fp_bon[i]) / 100, 2),
                  "J_sd": round((data[2][i] - fp_sd[i]) / 100, 2)})
    out["과제별_J(회수율-유사FP율)"] = J
    out["J_평균"] = {"nominal": round(float(np.mean([j["J_nominal"] for j in J])), 3),
                     "bon": round(float(np.mean([j["J_bon"] for j in J])), 3),
                     "sd": round(float(np.mean([j["J_sd"] for j in J])), 3)}
    R["V8_fig22"] = out


# ---------------------------------------------------------------- V9 Dubins 그림 2a, 그림 16
def v9_dubins():
    out = {}
    # 그림 2a 판독 (TPR, TNR), +-0.01
    pts = {"Nominal(N=1)": (0.64, 0.975), "Best-of-N(N=10)": (0.765, 0.96), "StressDream": (0.89, 0.925),
           "Classifier Guidance": (0.685, 0.84), "StressDream w/o C_pla": (0.905, 0.735)}
    out["그림2a_판독"] = {k: {"TPR": a, "TNR": b, "J": round(a + b - 1, 3), "균형정확도": round((a + b) / 2, 3)} for k, (a, b) in pts.items()}
    # 정답 라벨 문턱: 10,000회 몬테카를로에서 한 번이라도 걸리는 확률
    floor = {}
    for q in (1e-5, 1e-4, 3e-4, 1e-3):
        floor[f"q={q:g}"] = 1 - (1 - q) ** 10000
    out["MC10000_탐지확률(실패확률 q)"] = floor
    out["95%_탐지_q"] = 1 - 0.05 ** (1 / 10000)
    # 그림 16 (낙관 조종, 최대 안전 점수)
    out["그림16"] = {"GT_optimistic": 0.55, "Nominal": 0.49, "BoN": 0.50, "CG": 0.63, "NoiseOptim_noReg": 1.66,
                     "NoiseOptim_Reg": 0.53}
    out["그림16_GT_대비"] = {k: round(v - 0.55, 2) for k, v in out["그림16"].items() if k != "GT_optimistic"}
    R["V9_dubins"] = out


# ---------------------------------------------------------------- V10 표 7, 시간, Best-of-N 예산
def v10_misc():
    out = {}
    tab7 = {"Block Stack": 250, "Knife Put": 170, "Stacked Utensil Pick": 200, "Coffee Bean Pour": 100,
            "Open Coffee Bag": 170, "Open Candy Bag": 150}
    out["표7_합"] = sum(tab7.values())
    out["표7_평균"] = sum(tab7.values()) / 6
    out["표7_범위"] = [min(tab7.values()), max(tab7.values())]
    out["본문_약150_대비_평균_%"] = 100 * (sum(tab7.values()) / 6 - 150) / 150
    # 정책 파인튜닝 데이터: 과제당 40 성공 시연
    out["정책_시연_합"] = 6 * 40
    # 시간 산술: 1-2분 x 20회
    out["Vista_20회_시간_분"] = [20 * 1, 20 * 2]
    # 기대 최댓값: N개 표준정규 표본의 기대 최댓값 (BoN 이 한 방향 보상에서 얻는 이득의 목안)
    rng = np.random.default_rng(2)
    emax = {}
    for N in (1, 5, 10, 20, 40):
        emax[str(N)] = float(rng.standard_normal((200000, N)).max(axis=1).mean())
    out["N개_표준정규_기대최댓값"] = emax
    # 논문 표 1 비용: 근사 기울기 StressDream = N(K T + 2 Tv), 전체 기울기 = N(2K T + 2 Tv), BoN = N(K T + Tv)
    # K T 를 1 로 두고 Tv/KT = r 일 때 비율
    ratios = {}
    for r in (0.05, 0.2, 0.5, 1.0):
        ratios[f"Tv/KT={r}"] = {"SD(근사)/BoN": (1 + 2 * r) / (1 + r), "SD(전체기울기)/BoN": (2 + 2 * r) / (1 + r)}
    out["표1_비용비"] = ratios
    out["동일_예산_BoN_N(Dubins 전체 기울기, r->0)"] = 2 * 10
    R["V10_misc"] = out


# ---------------------------------------------------------------- V11 코드 상수 대 가우시안 기준값
def v11_code_constants():
    """코드의 hinge 문턱(0.0098, 0.055, 0.992/0.96)이 i.i.d. 가우시안 기준값에 가까운지 확인한다.
    코드를 import 하지 않고 정적으로 읽은 식을 numpy로 다시 구현했다."""
    rng = np.random.default_rng(3)
    out = {}

    def spec_loss(x):  # x: (B,T,C,H,W); 위치별 평균 파워의 log 제곱 평균 (코드의 spectral_whiteness_loss 와 같은 식)
        B, T, C, H, W = x.shape
        f = np.fft.fftn(x, axes=(-2, -1))
        p = np.abs(f) ** 2
        pm = p.mean(axis=(0, 1, 2)) / (H * W)
        return float((np.log(pm + 1e-8) ** 2).mean())

    # Vista: (1,25,4,72,128), Ctrl-World: (1,5,4,72,40), Dubins: (1,1,4,16,16)
    for name, shape, thr in (("Vista", (1, 25, 4, 72, 128), 0.0098), ("Ctrl-World", (1, 5, 4, 72, 40), 0.055),
                             ("Dubins", (1, 1, 4, 16, 16), None)):
        vals = [spec_loss(rng.standard_normal(shape)) for _ in range(12 if name != "Vista" else 3)]
        n_slices = shape[0] * shape[1] * shape[2]
        approx = float(polygamma(1, n_slices) + (digamma(n_slices) - math.log(n_slices)) ** 2)
        out[f"spec_{name}"] = {"가우시안_평균": float(np.mean(vals)), "가우시안_sd": float(np.std(vals)),
                               "슬라이스_수": n_slices, "근사(trigamma+bias^2)": approx, "코드_문턱": thr}

    def gram_value(x_flat, k, normalize):
        # flat_iid_gram_regularizer (Dubins 형): 행 그룹 Gram 의 비대각 + 대각 편차
        D = x_flat.size
        n = D // k
        Z = x_flat[:n * k].reshape(n, k)
        Zc = Z - Z.mean(axis=1, keepdims=True)
        if normalize:
            Zc = Zc / (np.linalg.norm(Zc, axis=1, keepdims=True) + 1e-8)
            target = 1.0
        else:
            target = float(k)
        diag = (Zc * Zc).sum(axis=1)
        S = Zc.T @ Zc
        off2 = (S * S).sum() - (diag * diag).sum()
        dev2 = ((diag - target) ** 2).sum()
        return float((off2 + dev2) / (n * n) / k)

    d = {}
    for name, D, k, norm in (("Dubins(k=16, 정규화 없음)", 1024, 16, False), ("Vista thw(k=240, 정규화 없음)", 921600, 240, False),
                             ("Ctrl-World thw(k=240, 정규화)", 57600, 240, True)):
        vals = [gram_value(rng.standard_normal(D), k, norm) for _ in range(6)]
        d[name] = {"가우시안_평균": float(np.mean(vals)), "가우시안_sd": float(np.std(vals))}
    out["gram"] = d
    out["코드_문턱"] = {"Dubins relu(perm-0.96)": 0.96, "Vista relu(perm-0.992)": 0.992, "Ctrl-World square(활성화 없음)": None}
    R["V11_code_constants"] = out


def main():
    for f in (v1_dims, v2_shell, v3_isotropy, v4_drift_budget, v5_table4_norm, v6_recall, v7_policy, v8_fig22,
              v9_dubins, v10_misc, v11_code_constants):
        f()
    OUT.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", OUT)
    print(json.dumps({k: R[k] for k in ("V1_dims", "V2_shell", "V3_isotropy")}, ensure_ascii=False, indent=1)[:3000])


if __name__ == "__main__":
    main()
