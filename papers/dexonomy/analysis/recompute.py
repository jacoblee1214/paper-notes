"""Dexonomy (arXiv:2504.18829 v2) 보고 수치 재계산과 일관성 검사.

논문 표의 숫자를 그대로 가져와 계산한다. 학습·시뮬레이션은 돌리지 않는다.
표준 라이브러리만 쓴다(math, json). 실행: python3 analysis/recompute.py
결과: analysis/recompute_results.json

분모(시도 수, 객체 수)는 논문에 명시되지 않은 것이 많다. 가정한 값은 결과의 "assumption" 필드에 적는다.
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
Z = 1.959964


def wilson(k, n):
    """Wilson 95% 구간 (비율, 하한, 상한). 단위는 %."""
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return round(100 * p, 2), round(100 * (c - h), 2), round(100 * (c + h), 2)


def two_prop_z(p1, p2, n1, n2):
    """두 비율 차이의 pooled z와 양측 p (독립 가정; 객체 군집을 반영하지 않음)."""
    pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    p = math.erfc(abs(z) / math.sqrt(2))
    return round(z, 2), float(f"{p:.3g}")


R = {}

# V1. 표 II의 Ours 행과 표 V의 Ours 행이 같은지
t2_ours = {"GSR": 60.50, "OSR": 96.53, "CDC": 0.21, "PD": 0.00, "SPD": 0.00}
t5_ours = {"GSR": 60.5, "OSR": 96.5, "CDC": 0.21, "PD": 0.00}
R["V1_table2_vs_table5_ours"] = {
    "match": all(abs(t2_ours[k] - t5_ours[k]) < 0.051 for k in t5_ours),
    "note": "표 II와 표 V의 Ours 행은 소수 첫째 자리까지 같다",
}

# V2. 속도 차이: 본문 "slightly lower"
s_ours, s_bodex = 323.4, 403.9
R["V2_speed_ratio"] = {
    "ours_vs_bodex_pct": round(100 * (s_ours / s_bodex - 1), 1),
    "note": "본문은 '약간 낮다(slightly lower)'고 쓰지만 약 20% 낮다",
}

# V3. 표 III: 시도 20 -> 100에서 GSR과 OSR 변화
t3 = {
    "BODex_DGN": {"20": (14.79, 71.30), "100": (14.80, 89.84)},
    "BODex_Obj": {"20": (6.92, 43.48), "100": (6.91, 73.53)},
    "Ours_DGN": {"20": (27.16, 91.28), "100": (27.18, 95.13)},
    "Ours_Obj": {"20": (18.25, 84.17), "100": (18.34, 94.63)},
}
R["V3_table3_attempts"] = {
    k: {"dGSR_pp": round(v["100"][0] - v["20"][0], 2), "dOSR_pp": round(v["100"][1] - v["20"][1], 2)}
    for k, v in t3.items()
}
R["V3_table3_attempts"]["note"] = (
    "BODex의 GSR은 거의 같지만 OSR은 30 pp(Obj)와 18.5 pp(DGN) 오른다. "
    "본문 '변화 없음(unchanged)'은 GSR에만 맞는다. Ours의 GSR 증가는 0.02와 0.09 pp뿐이다."
)

# V4. 표 III: BODex의 100회 실패율 (본문 "about 27%")
R["V4_bodex_fail_obj_100"] = {"fail_pct": round(100 - 73.53, 2)}

# V5. 표 V 절제 효과 (Ours 60.5 기준, GSR pp)
t5 = {
    "global_wo_opt": 45.7, "global_wo_filter": 18.6, "local_wo_opt": 17.8,
    "local_wo_filter": 62.8, "template_wo_update": 41.5, "ours": 60.5,
}
R["V5_table5_ablation_pp"] = {
    k: round(t5["ours"] - v, 1) for k, v in t5.items() if k != "ours"
}
R["V5_table5_ablation_pp"]["note"] = (
    "global_wo_opt 감소는 14.8 pp로 본문의 '약간의 개선(slight improvement)'과 맞지 않는다. "
    "local_wo_filter는 Ours보다 2.3 pp 높다(본문의 '성공률을 낮춘다'와 맞음)."
)

# V6. 표 IV: normal -> hard 감소 (GSR)
t4 = {"Power": (24.2, 12.8), "Intermediate": (23.0, 6.6), "Precision": (36.0, 11.4)}
R["V6_table4_friction_drop"] = {
    k: {"abs_pp": round(a - b, 1), "rel_pct": round(100 * (a - b) / a, 1)} for k, (a, b) in t4.items()
}
R["V6_table4_friction_drop"]["note"] = (
    "절대 감소는 정밀(24.6 pp)이 가장 크다. 상대 감소는 중간(71.3%)이 정밀(68.3%)보다 크다. "
    "본문 '정밀이 더 빨리 떨어진다'는 힘 절대치와 동력 비교에서 맞다."
)

# V7. 표 VI의 Ours 열과 표 IV의 normal GSR (같은 설정이어야 함)
R["V7_table6_vs_table4_normal"] = {
    "Power": {"t6": 24.23, "t4": 24.2}, "Intermediate": {"t6": 23.03, "t4": 23.0},
    "Precision": {"t6": 36.00, "t4": 36.0},
    "match": True,
}

# V8. 표 VI: 제어 전략 비교 (Ours - w/ pre-grasp, pp)
t6 = {
    "Shadow_Power": (17.64, 25.85, 24.23), "Shadow_Intermediate": (9.81, 19.93, 23.03),
    "Shadow_Precision": (13.79, 34.56, 36.00), "Allegro_Fingertip": (24.60, 45.08, 60.50),
}
R["V8_table6_control_pp"] = {
    k: {"ours_minus_prepose": round(o - p, 2), "ours_minus_grasponly": round(o - g, 2)}
    for k, (g, p, o) in t6.items()
}

# V9. 실제 로봇 성공률. 분모는 본문 "3회씩, 약 200회"에서 역산: 3x12x3 + 10x3x3 = 198
n_real = 3 * 12 * 3 + 10 * 3 * 3
k_real = [k for k in range(n_real + 1) if round(100 * k / n_real, 1) == 82.3]
p_real, lo, hi = wilson(163, n_real)
R["V9_realworld"] = {
    "assumption": "분모 n = 3 시물체 x 12 유형 x 3회 + 10 시물체 x 3 유형 x 3회 = 198",
    "successes_matching_82.3pct": k_real,
    "wilson95_at_163_of_198": [p_real, lo, hi],
    "wilson95_pct_range_text": f"{lo}–{hi}%",
}

# V10. 표 VIII: 학습 기반 비교. 분모 미기재 -> 객체 수준 가정으로 민감도
gap_main = round(63.9 - 54.0, 1)
gap_type1 = round(55.5 - 54.0, 1)
R["V10_table8"] = {
    "typecond_minus_bodex_pp": gap_main,
    "type1_minus_bodex_pp": gap_type1,
    "assumption": "테스트 객체 수 약 2140 (10.7k의 1/5) 또는 객체x크기 약 12840로 가정, 독립 표본 근사",
    "sensitivity": {
        "n_2140": {"type1_vs_bodex": two_prop_z(0.555, 0.540, 2140, 2140)},
        "n_12840": {"type1_vs_bodex": two_prop_z(0.555, 0.540, 12840, 12840)},
    },
    "note": "Ours-type1 대 BODex 1.5 pp 차이는 객체 수준 가정에서 유의하지 않을 수 있다",
}

# V11. 표 VII: 1000 객체 x 100 시도 가정 (상한)
t7 = [("6.Prismatic4F_T1", 45.6, 57.1), ("6.Prismatic4F_T2", 12.0, 56.7), ("6.Prismatic4F_T3", 1.7, 46.2),
      ("1.LargeDiam_T1", 22.9, 30.1), ("1.LargeDiam_T2", 22.5, 34.7), ("1.LargeDiam_T3", 7.2, 26.8)]
rows = []
for name, wo, w in t7:
    z_att, p_att = two_prop_z(w / 100, wo / 100, 100000, 100000)
    z_obj, p_obj = two_prop_z(w / 100, wo / 100, 1000, 1000)
    rows.append({"cell": name, "delta_pp": round(w - wo, 1), "z_attempt_level": z_att,
                 "z_object_level_n1000": z_obj, "p_object_level": p_obj})
R["V11_table7"] = {
    "assumption": "1000 객체 x 100 시도 = 100000 (시도 단위 상한), 객체 수준 n=1000 민감도",
    "rows": rows,
    "note": "모든 차이가 객체 수준 근사에서도 유의하다. 분산은 시드 한 번으로 보고되지 않음",
}

# V12. 표 VIII·X·IX·XI의 공유 값
R["V12_shared_values"] = {
    "table10_type1_eq_table8_type1": 55.5 == 55.5,
    "table9_flow_mlp_eq_table8_type1": 55.5 == 55.5,
    "table11_classifierB_eq_table8_typecond": 63.9 == 63.9,
    "table11_top1_ge_classifierB": 78.2 >= 63.9,
    "note": "표 XI의 Classifier B(63.9)는 표 VIII 본 모델 값과 같다. 본문은 어느 설정을 쓴 것인지 명시하지 않는다",
}

# V13. 데이터셋 규모
R["V13_dataset"] = {
    "objects_sum": 5697 + 5000,
    "grasps_per_object_avg": round(9.5e6 / 10.7e3, 0),
    "note": "5697 + 5000 = 10697 ≈ 10.7k. 객체당 평균 약 888 파지 (유효 기준 통과분 기준)",
}

# V14. 표 II 시도 수 가정: 20 시도 x 5697 x 6 크기
n_attempts = 20 * 5697 * 6
R["V14_table2_attempts"] = {
    "assumption": "시도 수 = 20 x 5697 객체 x 6 크기 = 683640 (상한, 유효 결과 기준이면 더 적음)",
    "gsr_ours_vs_bodex_pp": round(60.50 - 49.23, 2),
    "wilson_ours_gsr": wilson(round(0.6050 * n_attempts), n_attempts),
    "note": "시도 단위 구간은 좁지만 객체 간 군집을 무시한다. 객체 수준 민감도는 V11과 같은 방식으로 본다",
}

# V15. 인트로 주장 "power 파지에서 10개 넘는 링크가 2 mm 이내" vs 표 IV CLN
R["V15_intro_links"] = {
    "table4_power_cln_mean": 9.1,
    "claim": "> 10 hand links within 2 mm (power grasps)",
    "note": "평균 9.1이라 '10개 초과'는 개별 예시이거나 다른 집계일 가능성이 있다",
}

# V16. 부록 B: 10 epochs x 10 attempts per object = 100 시도 (본문 1 epoch x 100과 비교)
R["V16_appendixB"] = {"epochs": 10, "attempts_per_epoch": 10, "total_per_object": 100,
                      "note": "본문 설명과 맞다. 공식 설정 기본값(epoch 5)과는 다르다 (04 D1)"}

# V17. 표 II 대 OSR: BODex 96.56 > Ours 96.53 (논문은 OSR 최고를 주장하지 않음)
R["V17_osr_tie"] = {"ours": 96.53, "bodex": 96.56, "diff_pp": round(96.53 - 96.56, 2)}

# V18. 표 III: 본문 "our method's object success rate exceeding 94%" (100 시도)
R["V18_objaverse_osr100"] = {"ours_osr_pct": 94.63, "claim_ok": 94.63 > 94}

out = HERE / "recompute_results.json"
out.write_text(json.dumps(R, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote", out)
for key in ["V2_speed_ratio", "V5_table5_ablation_pp", "V9_realworld", "V10_table8"]:
    print(key, R[key])
