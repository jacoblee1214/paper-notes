"""T3 (arXiv:2406.13640 v3) 보고 수치 재계산과 검정. 수 초 안에 끝나며 numpy와 scipy만 쓴다.

  1. FoTa 데이터셋 크기 합계 (표 2, 3,083,452 확인)
  2. 모델 크기: 논문 표 (Table 1, 12M/45M/174M/308M)와 공식 체크포인트 파일 크기에서 얻은 파라미터 수 대조
     (checkpoint 크기 = fp32 4바이트 x 파라미터 수, 피클 여분은 0.1% 안팎)
  3. 실로봇 삽입 (그림 4c 막대 판독값, 과제당 15회): Wilson 구간, 과제별 Fisher, 합산 Fisher
     판독값은 픽셀 측정(±0.01)이며 시도 수는 판독을 가장 가까운 정수로 맞춘 근사다.
  4. 마스킹 비율 (그림 3b, 85% 대 89%): 검증 크기가 보고되지 않아 n을 가정한 민감도
  5. 그림 3a 중앙값 24% 점검 (판독 근사): 절대 %p 대 상대 증가

실행: python3 analysis/recompute.py   → analysis/recompute_results.json
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).with_name("recompute_results.json")
RES = {}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(float(c - h), 3), round(float(c + h), 3)]


# 1. FoTa 표 2 (논문 v3 p.14). 센서 행별 크기
FOTA = {
    "VisGel-small": 726740, "TVL": 82463, "Touch-and-Go": 262082, "Calandra17": 24118,
    "Yuan18": 494655, "YCB-Sight real": 480, "YCB-Sight sim": 1800, "ObjectFolder-real": 1417600,
    "Tippur23 GelSight360": 13341, "Tippur23 Wedge": 10000, "Tippur23 Mini": 24000, "Tippur23 DIGIT": 4000,
    "new Wedge": 3516, "new Finray": 6432, "new Svelte": 5335, "new DenseTact2.0": 3531, "new Mini": 3359,
}
total = sum(FOTA.values())
RES["fota_total"] = {"sum": total, "paper": 3083452, "match": total == 3083452}
# 가장 큰 두 센서 묶음의 비율 (논문 6절 "두 센서가 50% 이상")
gs17 = sum(FOTA[k] for k in ["VisGel-small", "Touch-and-Go", "Calandra17", "Yuan18"])
RES["fota_share"] = {"GelSight17 family": round(gs17 / total, 3),
                     "ObjectFolder GelSight var2": round(1417600 / total, 3)}

# 2. 체크포인트 크기 (바이트), HF tree API에서 2026-11-22 조회 (models/t3_*)
CKPT = {
    "tiny": {"enc": [6093786, 6093830, 6093830, 6093874, 6093874, 6093874, 6096522, 6096566],
             "trunk": 16051250, "pose": [5823282, 5825092, 5825134, 5825134], "total": 91100160},
    "small": {"enc": [22791706, 22791750, 22791750, 22791794, 22791794, 22791794, 22794442, 22794486],
              "trunk": 63918578, "pose": [22624434, 22626308, 22626350, 22626350], "total": 341328512},
    "medium": {"enc": [88037914, 88037958, 88037958, 88038002, 88038002, 88038002, 88040650, 88040694],
               "trunk": 255204338, "pose": [89406212, 89406254, 89406254, 89406254], "total": 1324851196},
    "large": {"enc": [155128390, 155128434, 155128434, 155131126, 306294338, 306294418, 306294498, 306299522],
              "trunk": 453507058, "pose": [158500338, 158502148, 158502190, 158502190], "total": 2943022940},
}
PAPER_M = {"tiny": 12, "small": 45, "medium": 174, "large": 308}
params = {}
for size, c in CKPT.items():
    p = lambda b: b / 4 / 1e6
    all_total = p(c["total"])
    h2 = p(sum(sorted(c["enc"])[:4]) + c["trunk"]) + p(c["pose"][0])  # 4 encoders + trunk + 1 pose decoder
    params[size] = {
        "paper_M": PAPER_M[size],
        "ckpt_all_fp32_M": round(all_total, 1),
        "ckpt_8enc_trunk_M": round(p(sum(c["enc"]) + c["trunk"]), 1),
        "ckpt_4enc_trunk_1pose_M": round(h2, 1),
        "enc_sizes_M": sorted({round(p(x), 2) for x in c["enc"]}),
    }
# 이론값 점검: ViT 블록 한 개 = 12 D^2 (attention 4D^2 + MLP 8D^2, 편향 제외)
for size, D in {"tiny": 192, "small": 384, "medium": 768, "large": 1024}.items():
    params[size]["block_12D2_M"] = round(12 * D * D / 1e6, 3)
    params[size]["trunk9_theory_M"] = round(9 * 12 * D * D / 1e6, 1)
RES["params"] = params
RES["params_note"] = ("논문 표 1의 파라미터 수는 '모든 인코더와 디코더' 포함이라고 적혀 있으나, "
                      "체크포인트의 인코더 8개와 포즈 디코더 4개를 모두 넣으면 medium 331M으로 맞지 않는다. "
                      "4개 인코더 + 트렁크 + 포즈 디코더 1개 조합이 네 크기 모두에서 논문 값과 1M 안에서 맞는다(가설).")

# 3. 그림 4c 실로봇 삽입 (과제당 15회). 판독값과 근사 정수 시도 결과
N = 15
READ_SUCC = {  # (판독 성공률)
    "none": {"USB": 0.007, "Switch": 0.130, "VGA": 0.000},
    "scratch": {"USB": 0.307, "Switch": 0.331, "VGA": 0.396},
    "T3": {"USB": 0.518, "Switch": 0.665, "VGA": 0.598},
}
COUNTS = {a: {k: int(round(v * N)) for k, v in d.items()} for a, d in READ_SUCC.items()}
RES["insertion_counts_assumed"] = COUNTS
RES["insertion_wilson"] = {a: {k: wilson(x, N) for k, x in d.items()} for a, d in COUNTS.items()}
fisher = {}
for part in ["USB", "Switch", "VGA"]:
    for ref in ["scratch", "none"]:
        t = [[COUNTS["T3"][part], N - COUNTS["T3"][part]], [COUNTS[ref][part], N - COUNTS[ref][part]]]
        fisher[f"{part} T3 vs {ref}"] = round(float(stats.fisher_exact(t)[1]), 4)
RES["insertion_fisher_per_part"] = fisher
for ref in ["scratch", "none"]:
    k1, k2, n = sum(COUNTS["T3"].values()), sum(COUNTS[ref].values()), 3 * N
    t = [[k1, n - k1], [k2, n - k2]]
    RES[f"insertion_pooled_T3_vs_{ref}"] = {"k_T3": k1, "k_ref": k2, "n_per_arm": n,
                                           "fisher_p": round(float(stats.fisher_exact(t)[1]), 4)}
# 평균 판독 차이: 초록의 "25% higher", "53% higher"가 %p인지 상대 증가인지
m = {a: np.mean(list(d.values())) for a, d in READ_SUCC.items()}
RES["insertion_means"] = {a: round(float(v), 3) for a, v in m.items()}
RES["insertion_pp_diff"] = {"T3_minus_scratch": round(float(m["T3"] - m["scratch"]), 3),
                            "T3_minus_none": round(float(m["T3"] - m["none"]), 3)}
RES["insertion_rel_gain"] = {"T3_over_scratch": round(float(m["T3"] / m["scratch"] - 1), 3),
                             "T3_over_none": round(float(m["T3"] / m["none"] - 1), 1)}
# 평균 스텝 (성공 에피소드만, 판독값)
STEPS = {"none_Switch": 23.5, "scratch": {"USB": 14.4, "Switch": 14.3, "VGA": 11.4},
         "T3": {"USB": 15.5, "Switch": 3.6, "VGA": 8.7}}
RES["steps_read"] = STEPS
RES["steps_T3_minus_scratch"] = {k: round(STEPS["T3"][k] - STEPS["scratch"][k], 1) for k in STEPS["T3"]}
RES["caption_lowest_steps_check"] = ("USB에서 T3 15.5 대 scratch 14.4: 캡션 '가장 낮은 평균 에피소드 길이'는 USB에서 성립하지 않음")

# 4. 마스킹 비율 (그림 3b). 검증 크기 미보고 -> 가정 n
mask = {}
for nval in [330, 660, 1000]:
    p1, p2 = 0.85, 0.89
    se = np.sqrt(p1 * (1 - p1) / nval + p2 * (1 - p2) / nval)
    z = (p2 - p1) / se
    mask[f"n={nval}"] = {"z": round(float(z), 2), "p": round(float(2 * (1 - stats.norm.cdf(z))), 3),
                         "diff_95CI": [round(float((p2 - p1) - 1.96 * se), 3), round(float((p2 - p1) + 1.96 * se), 3)]}
RES["masking_sensitivity"] = mask

# 5. 그림 3a 판독 (tiny..large, 짝: scratch -> pretrain1&2, 반 데이터와 전체 데이터, 판독 근사 ±0.03)
pairs = [(0.30, 0.45), (0.63, 0.75), (0.33, 0.63), (0.53, 0.82), (0.63, 0.68), (0.68, 0.90), (0.30, 0.77), (0.30, 0.93)]
d_abs = [b - a for a, b in pairs]
d_rel = [b / a for a, b in pairs]
RES["fig3a_reading_check"] = {"median_abs_pp": round(float(np.median(d_abs)) * 100, 1),
                              "median_rel_gain": round(float(np.median(d_rel)) - 1, 2),
                              "note": "본문 '중앙값 24%'는 상대 증가가 아니라 %p 차이와 맞음 (판독 근사)"}

OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(RES, ensure_ascii=False, indent=2))
