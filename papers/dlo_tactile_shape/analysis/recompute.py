"""논문 표 수치 재계산과 일관성 검사 (CPU, 수 초).

입력: 논문 Table I, Table II, 본문 수치(§III.B, §III.E, §III.F, §III.G)와 저장소 문서(docs/data.md, docs/training.md).
출력: analysis/recompute_results.json

가정(논문에 명시가 없어 직접 세운 것):
  - 표의 "±"는 표본 표준편차(SD, 모집단 ddof=0)이다. 저장소 engine/metrics.py의 std와 같은 정의다.
  - 표의 단위는 mm이다(논문 표 1, 2 캡션과 본문).
  - 시험 표본 수는 21,853(논문 §III.E)이다. 그러나 시험 표본은 4개 에피소드에서 나온 프레임이라
    서로 독립이 아니다. 그래서 표본 수 기준 검정은 과대평가이고, 에피소드 수준 상한을 따로 계산한다.
  - 모델 간 짝짓기 정보가 없어 두 표본 Welch 검정을 쓴다(순서 독립 근사).
"""
import json
import math
from pathlib import Path

from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
N_TEST = 21853          # 논문 §III.E, 시험 프레임 수
N_TEST_EPISODES = 4     # 논문 §III.E, 독립 에피소드 수
N_TRAIN_EPISODES = 36   # 논문 §III.B
N_TRAIN = 196678        # 논문 §III.E

# 논문 Table I (mean, SD)
TABLE1 = {
    "Pose-Only (MLP)":                 {"params_tr": 3.04, "params_tot": 3.04, "rmse": (55.00, 22.00), "cd": (44.00, 19.00), "ms": 0.9},
    "Pose + Direction":                {"params_tr": 2.70, "params_tot": 2.70, "rmse": (50.63, 24.25), "cd": (40.70, 17.06), "ms": None},
    "ResNet-18 + Cross-Attn":          {"params_tr": 3.49, "params_tot": 3.49, "rmse": (36.00, 20.00), "cd": (30.00, 16.00), "ms": 17.6},
    "Proposed w/o B-spline":           {"params_tr": 5.33, "params_tot": 23.74, "rmse": (35.01, 16.62), "cd": (30.06, 13.64), "ms": 61.5},
    "Proposed w/o L_weighted":         {"params_tr": 6.38, "params_tot": 24.80, "rmse": (37.00, 20.90), "cd": (30.82, 16.42), "ms": 61.5},
    "Proposed":                        {"params_tr": 6.38, "params_tot": 24.80, "rmse": (30.52, 15.22), "cd": (27.29, 12.30), "ms": 61.5},
}
# 논문 Table I의 Freq 열 (Hz). 표에 적힌 값과 1000/ms를 비교한다.
TABLE1_FREQ = {"Pose-Only (MLP)": 1.11e3, "ResNet-18 + Cross-Attn": 56.82, "Proposed w/o B-spline": 16.26,
               "Proposed w/o L_weighted": 16.26, "Proposed": 16.26}

# 논문 Table II (mean, SD)
TABLE2 = {
    "new_1_removed": (34.2, 18.0), "new_2_removed": (37.8, 21.2), "new_3_removed": (46.3, 23.3),
    "fault_1": (45.3, 23.3), "fault_2": (70.9, 30.8), "fault_3": (106.2, 34.8),
}


def welch(m1, s1, n1, m2, s2, n2):
    se = math.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
    t = (m1 - m2) / se
    df = (s1 ** 2 / n1 + s2 ** 2 / n2) ** 2 / ((s1 ** 2 / n1) ** 2 / (n1 - 1) + (s2 ** 2 / n2) ** 2 / (n2 - 1))
    p = 2 * stats.t.sf(abs(t), df)
    return {"diff": m1 - m2, "se": se, "t": t, "df": df, "p": p}


def cohen_d(m1, s1, m2, s2):
    sp = math.sqrt((s1 ** 2 + s2 ** 2) / 2)
    return (m1 - m2) / sp


def episode_threshold(diff):
    """에피소드 4개, 짝지은 t검정(df=3)에서 p<0.05가 되려면 에피소드별 차이의 SD가 이 값보다 작아야 한다.
    mean_d / (s_d / sqrt(4)) > t_crit(3)  ->  s_d < 2 * mean_d / t_crit."""
    tcrit = stats.t.ppf(0.975, N_TEST_EPISODES - 1)
    return 2 * abs(diff) / tcrit


def main():
    res = {"assumptions": {"sd_ddof": 0, "unit": "mm", "n_test_frames": N_TEST,
                           "n_test_episodes": N_TEST_EPISODES, "n_train_episodes": N_TRAIN_EPISODES}}

    # 1. 표 1 내부 검사: 주파수 = 1000 / 지연 시간
    freq_check = {}
    for k, f in TABLE1_FREQ.items():
        ms = TABLE1[k]["ms"]
        freq_check[k] = {"table_hz": f, "from_ms_hz": round(1000 / ms, 2), "diff": round(f - 1000 / ms, 2)}
    res["t1_freq_check"] = freq_check

    # 2. 파라미터 수 검사: 저장소 docs/training.md 예시 (K=32 추정)와 README 릴리스 K=16 수치
    K16_tr, K16_tot = 5327367, 23747607     # README (릴리스 K=16 체크포인트)
    K32_tr, K32_tot = 6377991, 24796167     # docs/training.md 예시 로그 (K 값은 표시 안 됨, 기본 설정 K=32로 추정)
    latent, hidden = 128, 512
    pose_last_extra = (32 - 16) * latent * hidden + (32 - 16) * latent   # pose MLP 마지막 Linear: out = latent*K
    res["t2_param_check"] = {
        "trainable_diff_docs_minus_release": K32_tr - K16_tr,
        "total_diff_docs_minus_release": K32_tot - K16_tot,
        "pose_last_layer_extra_for_K32_vs_K16": pose_last_extra,
        "trainable_diff_explained_by_pose_layer": (K32_tr - K16_tr) == pose_last_extra,
        "total_diff_unexplained": (K32_tot - K16_tot) - pose_last_extra,
        "paper_proposed_row_matches_K32": (round(K32_tr / 1e6, 2), round(K32_tot / 1e6, 2)),
        "paper_woBspline_row_matches_K16": (round(K16_tr / 1e6, 2), round(K16_tot / 1e6, 2)),
    }

    # 3. 표 1 쌍별 검정: (a) 표본 수 기준 Welch, (b) 에피소드 수준 상한
    pairs = [("Proposed", "Pose + Direction"), ("Proposed", "Pose-Only (MLP)"), ("Proposed", "ResNet-18 + Cross-Attn"),
             ("Proposed", "Proposed w/o B-spline"), ("Proposed", "Proposed w/o L_weighted")]
    t1 = []
    for a, b in pairs:
        for metric, key in (("RMSE(mean L2)", "rmse"), ("Chamfer", "cd")):
            ma, sa = TABLE1[a][key]
            mb, sb = TABLE1[b][key]
            w = welch(ma, sa, N_TEST, mb, sb, N_TEST)
            t1.append({"a": a, "b": b, "metric": metric, "mean_a": ma, "mean_b": mb,
                       "diff_a_minus_b": round(ma - mb, 2), "cohen_d": round(cohen_d(ma, sa, mb, sb), 3),
                       "welch_p_frames": float("%.3g" % w["p"]),
                       "episode_sd_threshold_mm": round(episode_threshold(ma - mb), 2)})
    res["t1_pairwise"] = t1

    # 4. 95% 구간 (표본 수 기준, 참고용): 평균 ± 1.96 SE
    ci = {}
    for k, (m, s) in {**{f"T1 {k}": v["rmse"] for k, v in TABLE1.items()}}.items():
        se = s / math.sqrt(N_TEST)
        ci[k] = [round(m - 1.96 * se, 3), round(m + 1.96 * se, 3)]
    res["t1_ci95_frame_based"] = ci

    # 5. 표 2: 센서 제거 (재학습 대 그대로 사용), 전체 모델과 비교
    t2 = []
    full = TABLE1["Proposed"]["rmse"]
    for k, (m, s) in TABLE2.items():
        w = welch(m, s, N_TEST, full[0], full[1], N_TEST)
        t2.append({"case": k, "mean": m, "sd": s, "diff_vs_full": round(m - full[0], 2),
                   "rel_vs_full_pct": round(100 * (m / full[0] - 1), 1),
                   "cohen_d": round(cohen_d(m, s, full[0], full[1]), 3),
                   "welch_p_frames": float("%.3g" % w["p"]),
                   "episode_sd_threshold_mm": round(episode_threshold(m - full[0]), 2)})
    res["t2_vs_full"] = t2

    # 6. 본문 수치 검사
    res["text_checks"] = {
        "train_sessions_sum_docs": 196968, "test_sessions_sum_docs": 21563,
        "paper_train": N_TRAIN, "paper_test": N_TEST,
        "shift_train_to_test_docs": 196968 - N_TRAIN,
        "test_docs_plus_shift": 21563 + (196968 - N_TRAIN),
        "frames_per_episode_train_paper": round(N_TRAIN / N_TRAIN_EPISODES, 1),
        "frames_per_episode_test_paper": round(N_TEST / N_TEST_EPISODES, 1),
        "episodes_15_of_36_pct": round(100 * 15 / 36, 1),
        "lr_backbone_stated_x005": 1e-4 * 0.05,
        "lr_backbone_stated_1e5": 1e-5,
        "lr_backbone_release_k16": 7.147008552233424e-05 * 0.5,
        "scale_500mm_err_vs_full_mm": round(36 - full[0], 2),
        "scale_580mm_err_vs_full_mm": round(40 - full[0], 2),
        "w_wo_bspline_rel_pct": round(100 * (TABLE1["Proposed w/o B-spline"]["rmse"][0] / full[0] - 1), 1),
        "w_wo_lw_rel_pct": round(100 * (TABLE1["Proposed w/o L_weighted"]["rmse"][0] / full[0] - 1), 1),
        "pose_dir_rel_pct": round(100 * (TABLE1["Pose + Direction"]["rmse"][0] / TABLE1["Pose-Only (MLP)"]["rmse"][0] - 1), 1),
        "chamfer_best_check": min(TABLE1, key=lambda k: TABLE1[k]["cd"][0]),
        "rmse_best_check": min(TABLE1, key=lambda k: TABLE1[k]["rmse"][0]),
    }
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
