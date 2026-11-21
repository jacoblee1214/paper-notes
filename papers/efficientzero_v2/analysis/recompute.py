"""EfficientZero V2 (arXiv:2403.00564 v2, ICML 2024) 논문 표 수치를 다시 계산하고 일관성을 검사한다.

입력: 논문 Table 1(Atari 100k), Table 2(DMC Proprio/Vision), Table 4–6(ablation), Table 7(계산량)의 값을
아래에 그대로 옮겨 적었다(출처는 주석). 외부 데이터는 쓰지 않는다. 학습이나 재현은 하지 않는다.
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)

가정과 한계
- 논문은 시드별 점수, 표준편차, 신뢰구간을 제공하지 않는다(Table 2 캡션만 "3 seeds"라고 말한다).
  그래서 검정과 구간은 "과제(task) 사이" 변동으로만 계산한다. 시드 간 잡음은 계산할 수 없다.
- Table 1의 DreamerV3 Up N Down 값은 "-"(없음)이라 승패 계산에서 뺀다.
- 정규화 점수는 (점수 - Random) / (Human - Random)으로 계산한다(Atari 100k 표준 정의).
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
RES = {}

# ---- Table 1: Atari 100k (v2). 열: Random, Human, SimPLe, CURL, DrQ, SPR, MuZero, EfficientZero, DreamerV3, BBF, EZ-V2
ATARI_COLS = ["Random", "Human", "SimPLe", "CURL", "DrQ", "SPR", "MuZero", "EfficientZero", "DreamerV3", "BBF", "EZ-V2"]
ATARI = {
    "Alien": [227.8, 7127.7, 616.9, 558.2, 771.2, 801.5, 530.0, 808.5, 959, 1173.2, 1557.7],
    "Amidar": [5.8, 1719.5, 88.0, 142.1, 102.8, 176.3, 38.8, 148.6, 139, 244.6, 184.9],
    "Assault": [222.4, 742.0, 527.2, 600.6, 452.4, 571.0, 500.1, 1263.1, 706, 2098.5, 1757.5],
    "Asterix": [210.0, 8503.3, 1128.3, 734.5, 603.5, 977.8, 1734.0, 25557.8, 932, 3946.1, 61810.0],
    "Bank Heist": [14.2, 753.1, 34.2, 131.6, 168.9, 380.9, 192.5, 351.0, 649, 732.9, 1316.7],
    "BattleZone": [2360.0, 37187.5, 5184.4, 14870.0, 12954.0, 16651.0, 7687.5, 13871.2, 12250, 24459.8, 14433.3],
    "Boxing": [0.1, 12.1, 9.1, 1.2, 6.0, 35.8, 15.1, 52.7, 52.7, 85.8, 75.0],
    "Breakout": [1.7, 30.5, 16.4, 4.9, 16.1, 17.1, 48.0, 414.1, 31, 370.6, 400.1],
    "ChopperCommand": [811.0, 7387.8, 1246.9, 1058.5, 780.3, 974.8, 1350.0, 1117.3, 420, 7549.3, 1196.6],
    "Crazy Climber": [10780.5, 35829.4, 62583.6, 12146.5, 20516.5, 42923.6, 56937.0, 83940.2, 97190, 58431.8, 112363.3],
    "Demon Attack": [152.1, 1971.0, 208.1, 817.6, 1113.4, 545.2, 3527.0, 13003.9, 303, 13341.4, 22773.5],
    "Freeway": [0.0, 29.6, 20.3, 26.7, 9.8, 24.4, 21.8, 21.8, 0, 25.5, 0.0],
    "Frostbite": [65.2, 4334.7, 254.7, 1181.3, 331.1, 1821.5, 255.0, 296.3, 909, 2384.8, 1136.3],
    "Gopher": [257.6, 2412.5, 771.0, 669.3, 636.3, 715.2, 1256.0, 3260.3, 3730, 1331.2, 3868.7],
    "Hero": [1027.0, 30826.4, 2656.6, 6279.3, 3736.3, 7019.2, 3095.0, 9315.9, 11161, 7818.6, 9705.0],
    "Jamesbond": [29.0, 302.8, 125.3, 471.0, 236.0, 365.4, 87.5, 517.0, 445, 1129.6, 468.3],
    "Kangaroo": [52.0, 3035.0, 323.1, 872.5, 940.6, 3276.4, 62.5, 724.1, 4098, 6614.7, 1886.7],
    "Krull": [1598.0, 2665.5, 4539.9, 4229.6, 4018.1, 3688.9, 4890.8, 5663.3, 7782, 8223.4, 9080.0],
    "Kung Fu Master": [258.5, 22736.3, 17257.2, 14307.8, 9111.0, 13192.7, 18813.0, 30944.8, 21420, 18991.7, 28883.3],
    "Ms Pacman": [307.3, 6951.6, 1480.0, 1465.5, 960.5, 1313.2, 1265.6, 1281.2, 1327, 2008.3, 2251.0],
    "Pong": [-20.7, 14.6, 12.8, -16.5, -8.5, -5.9, -6.7, 20.1, 18, 16.7, 20.8],
    "Private Eye": [24.9, 69571.3, 58.3, 218.4, -13.6, 124.0, 56.3, 96.7, 882, 40.5, 99.8],
    "Qbert": [163.9, 13455.0, 1288.8, 1042.4, 854.4, 669.1, 3952.0, 13781.9, 3405, 4447.1, 16058.3],
    "Road Runner": [11.5, 7845.0, 5640.6, 5661.0, 8895.1, 14220.5, 2500.0, 17751.3, 15565, 33426.8, 27516.7],
    "Seaquest": [68.4, 42054.7, 683.3, 384.5, 301.2, 583.1, 208.0, 1100.2, 618, 1232.5, 1974.0],
    "Up N Down": [533.4, 11693.2, 3350.3, 2955.2, 3180.8, 28138.5, 2896.9, 17264.2, None, 12101.7, 15224.3],
}
REPORTED_ATARI_NORM = {"mean": {"EfficientZero": 1.945, "DreamerV3": 1.120, "BBF": 2.247, "EZ-V2": 2.428},
                       "median": {"EfficientZero": 1.090, "DreamerV3": 0.490, "BBF": 0.917, "EZ-V2": 1.286}}

# ---- Table 2: DMC. 행 = 과제, 열 = [Proprio50k: SAC, TD-MPC2, DreamerV3, EZ-V2 | Vision100k: CURL, DrQ-v2, DreamerV3, EZ-V2]
#      (easy 10개). hard 10개는 [Proprio100k: SAC, TD-MPC2, DreamerV3, EZ-V2 | Vision200k: CURL, DrQ-v2, DreamerV3, EZ-V2]
DMC_EASY = {
    "Cartpole Balance": [997.6, 962.8, 839.6, 947.3, 963.3, 965.5, 956.4, 911.7],
    "Cartpole Balance Sparse": [993.1, 942.8, 559.0, 999.2, 999.4, 1000.0, 813.0, 951.5],
    "Cartpole Swingup": [861.6, 826.7, 527.7, 805.4, 765.4, 756.0, 374.8, 747.8],
    "Cup Catch": [949.9, 976.0, 729.6, 969.8, 932.3, 468.0, 947.7, 954.7],
    "Finger Spin": [900.0, 965.8, 765.8, 837.1, 850.2, 459.4, 633.2, 927.6],
    "Pendulum Swingup": [158.9, 520.1, 830.4, 825.4, 144.1, 233.3, 619.3, 726.7],
    "Reacher Easy": [744.0, 903.5, 693.4, 940.3, 467.9, 722.1, 441.4, 946.3],
    "Reacher Hard": [646.5, 580.4, 768.0, 795.4, 112.7, 202.9, 120.4, 961.5],
    "Walker Stand": [870.0, 973.9, 767.3, 953.6, 733.8, 426.1, 939.5, 944.9],
    "Walker Walk": [813.2, 965.5, 475.2, 944.0, 538.5, 681.5, 771.2, 888.8],
}
DMC_HARD = {
    "Acrobot Swingup": [44.1, 303.2, 62.8, 297.7, 6.8, 15.1, 67.4, 231.8],
    "Cartpole Swingup Sparse": [256.6, 421.4, 172.7, 795.4, 8.8, 81.2, 392.4, 763.6],
    "Cheetah Run": [680.9, 614.4, 400.8, 677.8, 405.1, 418.4, 587.3, 631.6],
    "Finger Turn Easy": [630.8, 793.3, 560.5, 310.7, 371.5, 286.8, 366.6, 799.2],
    "Finger Turn Hard": [414.0, 604.8, 474.2, 374.1, 236.3, 268.4, 258.5, 794.6],
    "Hopper Hop": [0.1, 84.5, 9.7, 186.5, 84.5, 26.3, 76.3, 206.4],
    "Hopper Stand": [3.8, 807.9, 296.1, 795.4, 627.7, 290.2, 652.5, 805.7],
    "Quadruped Run": [139.7, 742.1, 289.0, 510.6, 170.9, 339.4, 168.0, 384.8],
    "Quadruped Walk": [237.5, 853.7, 256.2, 925.8, 131.8, 311.6, 122.6, 433.3],
    "Walker Run": [635.4, 780.5, 478.9, 657.2, 274.7, 359.9, 618.2, 475.3],
}
REPORTED_DMC_MEAN = {"Proprio50k": None, "Vision100k": None,
                     "Proprio100k": [552.0, 740.9, 517.1, 723.2], "Vision200k": [437.3, 410.3, 498.5, 726.1]}
REPORTED_DMC_MEDIAN = {"Proprio100k": [633.3, 806.4, 543.4, 800.4], "Vision200k": [324.9, 330.6, 484.5, 788.1]}

# ---- Table 4–6: 구성요소 제거 ablation (EZ-V2 원본 vs 변형). 열 순서 = 아래 과제 목록
ABL_TASKS_PROPRIO = ["Acrobot", "Cartpole swingup sparse", "Pendulum swingup", "Reacher hard", "Walker walk", "Walker run", "Quadruped walk"]
ABL_PROPRIO = {
    "Original": [297.7, 795.4, 825.4, 795.4, 944.0, 657.2, 925.8],
    "Multi-step TD": [256.7, 473.6, 836, 590.7, 839.3, 521.6, 649.7],
    "GAE": [275.1, 766.7, 336, 601.6, 757.3, 512.7, 719.7],
    "Sample MCTS": [248.1, 789.3, 357.4, 754.7, 691.7, 381.1, 254.7],
}
ABL_VISION = {
    "Original": [231.8, 763.6, 726.7, 961.5, 888.8, 475.3, 433.3],
    "Multi-step TD": [186.3, 631.7, 411.4, 878, 880.5, 496.7, 410.0],
    "GAE": [122.0, 796.1, 718.0, 934.9, 765.1, 491.5, 299.3],
    "Sample MCTS": [73.9, 786.1, 372.9, 938.8, 619.9, 264.2, 141.4],
}
ABL_ATARI = {  # Table 6: Asterix, UpNDown, Qbert
    "Original": [61810.0, 15224.3, 16058.3],
    "Multi-step TD": [26023.1, 13725.7, 9463.3],
    "GAE": [22816.7, 16255.3, 7807.3],
}


def normed(score, rnd, hum):
    return (score - rnd) / (hum - rnd)


def iqm(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return float(x[lo:hi].mean())


def boot_ci(x, stat, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    vals = np.array([stat(x[i]) for i in idx])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    # 1) Atari 정규화 평균·중앙값
    games = list(ATARI.keys())
    col = {c: i for i, c in enumerate(ATARI_COLS)}
    norm_mean, norm_med = {}, {}
    for c in ["EfficientZero", "DreamerV3", "BBF", "EZ-V2"]:
        vals = []
        for g in games:
            v = ATARI[g][col[c]]
            if v is None:
                continue  # Up N Down DreamerV3
            vals.append(normed(v, ATARI[g][0], ATARI[g][1]))
        norm_mean[c] = float(np.mean(vals))
        norm_med[c] = float(np.median(vals))
    RES["atari_normed"] = {
        "recomputed_mean": {k: round(v, 3) for k, v in norm_mean.items()},
        "recomputed_median": {k: round(v, 3) for k, v in norm_med.items()},
        "reported_mean": REPORTED_ATARI_NORM["mean"],
        "reported_median": REPORTED_ATARI_NORM["median"],
        "note": "DreamerV3 평균·중앙값은 Up N Down을 뺀 25개 게임 기준으로 계산했다(논문 값과 차이가 나면 이 때문일 수 있다).",
    }

    # 1-2) EZ-V2 정규화 점수 기여도 상위 게임과, 보고값 2.428과 맞는지 보는 민감도
    contrib = sorted([(normed(ATARI[g][10], ATARI[g][0], ATARI[g][1]), g) for g in games], reverse=True)
    top = [(g, round(v, 3)) for v, g in contrib[:4]]
    rest_mean = float(np.mean([v for v, g in contrib[2:]]))
    RES["atari_ez_contrib"] = {"top4": top, "mean_without_top2": round(float(np.mean([v for v, g in contrib[2:]])), 3),
                               "mean_without_top1": round(float(np.mean([v for v, g in contrib[1:]])), 3),
                               "n_games": len(contrib)}
    # 1-1) EZ-V2 대 DreamerV3 게임별 승패와 대응 검정
    ez = np.array([ATARI[g][10] for g in games if ATARI[g][8] is not None])
    dv3 = np.array([ATARI[g][8] for g in games if ATARI[g][8] is not None])
    names = [g for g in games if ATARI[g][8] is not None]
    wins = int(np.sum(ez > dv3)); ties = int(np.sum(ez == dv3)); losses = int(np.sum(ez < dv3))
    w = stats.wilcoxon(normed_arr := np.array([normed(ez[i], ATARI[names[i]][0], ATARI[names[i]][1]) for i in range(len(names))]),
                       np.array([normed(dv3[i], ATARI[names[i]][0], ATARI[names[i]][1]) for i in range(len(names))]))
    RES["atari_vs_dreamerv3"] = {"n_games": len(names), "wins": wins, "ties": ties, "losses": losses,
                                 "wilcoxon_p_normed": round(float(w.pvalue), 6),
                                 "losing_games": [names[i] for i in range(len(names)) if ez[i] < dv3[i]]}

    # 2) DMC 평균·중앙값 (블록별 10개 과제)
    def block(d, sl):
        return {k: v[sl] for k, v in d.items()}
    tasks_easy = list(DMC_EASY.keys()); tasks_hard = list(DMC_HARD.keys())
    E = np.array([DMC_EASY[t] for t in tasks_easy]); H = np.array([DMC_HARD[t] for t in tasks_hard])
    # 열: 0-3 Proprio(SAC,TD,Dv3,EZ), 4-7 Vision(CURL,DrQ,Dv3,EZ)
    dmc_stats = {}
    for label, M, cols in [("Proprio50k", E, slice(0, 4)), ("Vision100k", E, slice(4, 8)),
                           ("Proprio100k", H, slice(0, 4)), ("Vision200k", H, slice(4, 8))]:
        sub = M[:, cols]
        dmc_stats[label] = {"mean": np.round(sub.mean(axis=0), 1).tolist(),
                            "median": np.round(np.median(sub, axis=0), 1).tolist()}
    # 전체 20개 과제 평균(Proprio: 50k+100k, Vision: 100k+200k)
    dmc_stats["Proprio_all20"] = {"mean": np.round(np.vstack([E[:, 0:4], H[:, 0:4]]).mean(axis=0), 1).tolist()}
    dmc_stats["Vision_all20"] = {"mean": np.round(np.vstack([E[:, 4:8], H[:, 4:8]]).mean(axis=0), 1).tolist()}
    RES["dmc_block_stats"] = {
        "recomputed": dmc_stats,
        "reported_mean_row_100k_200k": {"Proprio100k": REPORTED_DMC_MEAN["Proprio100k"], "Vision200k": REPORTED_DMC_MEAN["Vision200k"]},
        "reported_median_row_100k_200k": REPORTED_DMC_MEDIAN,
    }
    # 2-1) 블록 평균 서로 다른 집계와 본문 주장 대조
    ez_p100 = H[:, 3]; ez_v200 = H[:, 7]
    RES["dmc_text_checks"] = {
        "text_723.2_across_20_tasks": {
            "mean_hard10_proprio": round(float(ez_p100.mean()), 1),
            "mean_all20_proprio": round(float(np.vstack([E[:, 3], H[:, 3]]).mean()), 1),
        },
        "text_726.1_vision": {
            "mean_hard10_vision": round(float(ez_v200.mean()), 1),
            "mean_all20_vision": round(float(np.vstack([E[:, 7], H[:, 7]]).mean()), 1),
            "vs_DreamerV3_hard10_pct": round(100 * (ez_v200.mean() / H[:, 6].mean() - 1), 1),
            "vs_DreamerV3_all20_pct": round(100 * (np.vstack([E[:, 7], H[:, 7]]).mean() / np.vstack([E[:, 6], H[:, 6]]).mean() - 1), 1),
        },
    }
    # 2-2) "16 of 20 tasks 신기록" 주장: Vision 100k+200k에서 EZ-V2가 CURL, DrQ-v2, DreamerV3보다 높은 과제 수
    vis_all = np.vstack([E[:, 4:8], H[:, 4:8]])
    best_base = vis_all[:, :3].max(axis=1)
    RES["dmc_new_record_claim"] = {
        "vision_tasks_ez_best": int(np.sum(vis_all[:, 3] > best_base)),
        "n_tasks": int(vis_all.shape[0]),
        "claim": "16 of 20 (Vision Control, 본문 5.2)",
        "note": "기준선은 Table 2의 CURL, DrQ-v2, DreamerV3만 본다(SAC, TD-MPC2는 Vision 표에 없음)",
    }


    # 2-3) Table 2 평균·중앙값 행을 10개 블록과 20개 전체로 각각 맞춰 본다
    allmed = {"Proprio_all20_median": np.round(np.median(np.vstack([E[:, 0:4], H[:, 0:4]]), axis=0), 1).tolist(),
              "Vision_all20_median": np.round(np.median(np.vstack([E[:, 4:8], H[:, 4:8]]), axis=0), 1).tolist()}
    RES["dmc_mean_row_check"] = {"reported_mean_Proprio100k_row": REPORTED_DMC_MEAN["Proprio100k"],
                                 "recomputed_hard10_mean_Proprio": dmc_stats["Proprio100k"]["mean"],
                                 "recomputed_all20_mean_Proprio": dmc_stats["Proprio_all20"]["mean"],
                                 "reported_mean_Vision200k_row": REPORTED_DMC_MEAN["Vision200k"],
                                 "recomputed_hard10_mean_Vision": dmc_stats["Vision200k"]["mean"],
                                 "recomputed_all20_mean_Vision": dmc_stats["Vision_all20"]["mean"],
                                 "reported_median_row_Proprio": REPORTED_DMC_MEDIAN["Proprio100k"],
                                 "reported_median_row_Vision": REPORTED_DMC_MEDIAN["Vision200k"],
                                 "recomputed_all20_median": allmed}

    # 3) "50 of 66 과제에서 DreamerV3보다 우수" 주장 재계산
    wins_all = {"Atari": (wins, len(names))}
    pw = {}
    for label, M, cols in [("Proprio50k", E, (0, 2, 3)), ("Vision100k", E, (4, 6, 7)),
                           ("Proprio100k", H, (0, 2, 3)), ("Vision200k", H, (4, 6, 7))]:
        pass
    dv_w = 0; dv_n = 0
    for M, dv, ezc in [(E, 6, 7), (H, 6, 7)]:
        pass
    dmc_dv_wins = int(np.sum(E[:, 0 + 3] > E[:, 0 + 2])) + int(np.sum(E[:, 7] > E[:, 6])) \
        + int(np.sum(H[:, 3] > H[:, 2])) + int(np.sum(H[:, 7] > H[:, 6]))
    dmc_dv_n = 4 * 10
    total_wins = wins + dmc_dv_wins
    total_n = len(names) + dmc_dv_n
    RES["win_count_vs_dreamerv3"] = {
        "atari": wins_all["Atari"], "dmc_wins": dmc_dv_wins, "dmc_n": dmc_dv_n,
        "total_wins": total_wins, "total_n": total_n,
        "reported": "50 of 66 (Abstract, 서론)",
        "note": "Up N Down(DreamerV3 값 없음)은 분모에서 뺐다. 논문의 66개는 Atari 26 + DMC 40 = 66이다.",
    }
    # Wilcoxon / sign 검정 (과제 단위, 정규화 없음: DMC는 점수 자체, 짝지은 과제)
    dmc_pairs = np.concatenate([E[:, 3] - E[:, 2], E[:, 7] - E[:, 6], H[:, 3] - H[:, 2], H[:, 7] - H[:, 6]])
    w_dmc = stats.wilcoxon(dmc_pairs)
    RES["dmc_vs_dreamerv3_wilcoxon"] = {"n": int(len(dmc_pairs)), "median_diff": round(float(np.median(dmc_pairs)), 1),
                                       "p_two_sided": round(float(w_dmc.pvalue), 6),
                                       "note": "블록(Proprio/Vision, 50k/100k/200k)을 합쳐 짝지은 40개 과제. 서로 다른 벤치마크 점수 차이를 합쳐 쓰는 것은 근사다."}

    # 4) 과제 사이 부트스트랩 CI(시드 잡음 아님)
    RES["bootstrap_task_ci"] = {}
    for label, arr in [("Atari_normed_EZV2", np.array([normed(ATARI[g][10], ATARI[g][0], ATARI[g][1]) for g in games]))]:
        RES["bootstrap_task_ci"][label] = {"mean": round(float(arr.mean()), 3),
                                          "ci95_mean": [round(v, 3) for v in boot_ci(arr, np.mean)],
                                          "iqm": round(iqm(arr), 3), "ci95_iqm": [round(v, 3) for v in boot_ci(arr, iqm)],
                                          "n": int(len(arr))}
    for label, arr in [("Proprio100k_EZV2", H[:, 3]), ("Vision200k_EZV2", H[:, 7]),
                       ("Proprio100k_DreamerV3", H[:, 2]), ("Vision200k_DreamerV3", H[:, 6])]:
        RES["bootstrap_task_ci"][label] = {"mean": round(float(arr.mean()), 1),
                                          "ci95_mean": [round(v, 1) for v in boot_ci(arr, np.mean)], "n": int(len(arr))}
    # 차이 (EZ-V2 - DreamerV3)를 과제 단위로 부트스트랩
    for label, a, b in [("Proprio100k_diff", H[:, 3], H[:, 2]), ("Vision200k_diff", H[:, 7], H[:, 6])]:
        d = a - b
        RES["bootstrap_task_ci"][label] = {"mean_diff": round(float(d.mean()), 1),
                                          "ci95_mean_diff": [round(v, 1) for v in boot_ci(d, np.mean)],
                                          "wilcoxon_p": round(float(stats.wilcoxon(d).pvalue), 5), "n": int(len(d))}

    # 5) 구성요소 제거 ablation: 과제별 원본 대 변형 승패 (Wilcoxon, 짝지은 과제)
    abl = {}
    for label, tab in [("proprio", ABL_PROPRIO), ("vision", ABL_VISION)]:
        for var in ["Multi-step TD", "GAE", "Sample MCTS"]:
            a = np.array(tab["Original"]); b = np.array(tab[var])
            d = a - b
            abl[f"{label}:{var}"] = {"orig_wins": int(np.sum(d > 0)), "ties": int(np.sum(d == 0)),
                                     "var_wins": int(np.sum(d < 0)), "n": int(len(d)),
                                     "wilcoxon_p": round(float(stats.wilcoxon(d).pvalue), 4) if np.any(d != 0) else None,
                                     "mean_orig": round(float(a.mean()), 1), "mean_var": round(float(b.mean()), 1)}
    a = np.array(ABL_ATARI["Original"])
    for var in ["Multi-step TD", "GAE"]:
        b = np.array(ABL_ATARI[var])
        abl[f"atari3:{var}"] = {"orig_wins": int(np.sum(a > b)), "var_wins": int(np.sum(a < b)), "n": 3}
    RES["ablation_tables_4_6"] = abl
    # Table 4-5 원본 값과 Table 2 값 일치 확인
    RES["ablation_consistency_with_table2"] = {
        "proprio": all(abs(ABL_PROPRIO["Original"][i] - DMC_HARD.get(t, [0]*8)[3]) < 1e-6 or
                       abs(ABL_PROPRIO["Original"][i] - DMC_EASY.get(t, [0]*8)[3]) < 1e-6
                       for i, t in enumerate(["Acrobot Swingup", "Cartpole Swingup Sparse", "Pendulum Swingup", "Reacher Hard", "Walker Walk", "Walker Run", "Quadruped Walk"])),
        "vision": all(abs(ABL_VISION["Original"][i] - DMC_HARD.get(t, [0]*8)[7]) < 1e-6 or
                      abs(ABL_VISION["Original"][i] - DMC_EASY.get(t, [0]*8)[7]) < 1e-6
                      for i, t in enumerate(["Acrobot Swingup", "Cartpole Swingup Sparse", "Pendulum Swingup", "Reacher Hard", "Walker Walk", "Walker Run", "Quadruped Walk"])),
    }

    # 6) Table 7 계산량 비율 (본문: "1000배 적은 FLOPs", "약 4배 적은 파라미터")
    ez_flops, tdm_flops = 4.7e7, 3.6e10
    ez_par, tdm_par = 1.3e6, 4.9e6
    RES["compute_ratio"] = {"flops_ratio_tdmpc2_over_ez": round(tdm_flops / ez_flops, 0),
                            "params_ratio": round(tdm_par / ez_par, 2),
                            "time_ratio_100k": round(3.3 / 2.7, 2),
                            "claim": "본문 J.3: 1000배 적은 FLOPs, 약 4배 적은 파라미터"}

    OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(RES, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
