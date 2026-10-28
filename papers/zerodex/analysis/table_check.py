"""Table I–V·본문 수치 재계산과 v1 ↔ v3 비교 (CPU). 실행: python3 analysis/table_check.py

- C1: 표 내부 일관성 (All = 두 조건 평균, Table II 실패 합 = Table I 실패 수)
- C2: 시행 수가 작은 비교의 Wilson 95% 구간과 Fisher 정확검정 (모든 시행 결과가 없어 비대응 근사)
- C3: Table III의 합산 비율이 정수 개수와 맞는 조합이 있는지 (반올림 역산)
- C4: Table IV 비율과, P_avg·Coll%·P_max가 함께 성립하려면 필요한 최악 장면 침투 깊이(하한)
- C5: Table V 단계별 연쇄가 End-to-End와 맞는지
- C6: 실행 시간 합과 VLM 호출 수 추정
- C7: arXiv v1(2026-06-17) ↔ v3(2026-10-20) 실험·방법 변경 목록
"""
import json
import math
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]


def wilson(k: int, n: int, z: float = 1.959964) -> list[float]:
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 3), round(c + h, 3)]


def fisher(k1: int, n1: int, k2: int, n2: int) -> float:
    return float(f"{stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]:.3g}")


# Table I (Franka, zero-shot grounding): (standard n=10, cluttered&precise n=10, all n=20)
T1 = {"MOKA": (0.60, 0.20, 0.40), "ReKep": (0.50, 0.60, 0.55), "Ours": (0.90, 0.80, 0.85)}
# Table I (xArm, Throw Away Trash n=20): (approach, grasp, success)
T1_VLA = {"GR00T": (0.95, 0.35, 0.30), "Being-H0": (1.00, 0.05, 0.00), "Ours": (1.00, 0.95, 0.95)}
# Table II: success, ground, solver, grasp, exec failures (n=20)
T2 = {"MOKA": (0.40, 6, 0, 3, 3), "ReKep": (0.55, 7, 2, 0, 0), "Ours": (0.85, 0, 0, 3, 0)}
# Table III: (#G mean/scene, Clear%, IK%, succ k/10), scenes
T3 = {
    "tool_off": (22, 16.7, 16.7, 2, 5), "tool_on": (16, 96.2, 35.0, 7, 5),
    "pp_off": (69, 23.4, 5.6, 9, 10), "pp_on": (27, 96.0, 37.2, 9, 10),
}
# Table IV: repeatability (sigma, delta_max) and (P_max, P_avg, Coll%, IK%), scenes
T4_REP = {"MOKA": (8.1, 64.9), "Ours": (2.5, 3.9)}
T4 = {
    "tool/MOKA": (20.4, 20.4, 20, 30.3, 5), "tool/Ours": (20.3, 4.7, 5, 35.0, 5),
    "pp/MOKA": (37.2, 36.3, 100, 24.2, 10), "pp/Ours": (32.6, 16.3, 17, 37.2, 10),
}
# Table V: with retry, (without retry) as (k, n) per step and end-to-end
T5 = {
    "Organize Objects": {"retry": [(10, 10), (10, 10), (8, 10), (8, 10)],
                         "no_retry": [(10, 10), (7, 10), (6, 7), (6, 10)]},
    "Cooking": {"retry": [(9, 10), (9, 9), (9, 9), (9, 10)],
                "no_retry": [(8, 10), (7, 8), (6, 7), (6, 10)]},
}


def consistent_counts(mean_g: float, pct: float, scenes: int, dec: int = 1) -> list[tuple[int, int]]:
    """mean_g가 반올림된 장면당 평균 후보 수, pct가 반올림된 합산 비율일 때 가능한 (N, k)."""
    out = []
    lo, hi = math.ceil((mean_g - 0.5) * scenes), math.floor((mean_g + 0.5) * scenes - 1e-9)
    for N in range(max(1, lo), hi + 1):
        for k in range(N + 1):
            if abs(round(100 * k / N, dec) - pct) < 1e-9 or abs(100 * k / N - pct) <= 0.05 + 1e-9:
                out.append((N, k))
    return out


def main() -> None:
    out: dict = {}

    # C1 내부 일관성
    c1 = {"all_is_mean_of_conditions": {m: abs((s + c) / 2 - a) < 1e-9 for m, (s, c, a) in T1.items()},
          "table2_failures_match": {}}
    for m, (succ, g, so, gr, ex) in T2.items():
        c1["table2_failures_match"][m] = {"failures_from_T1": round((1 - succ) * 20), "sum_T2": g + so + gr + ex,
                                          "ok": round((1 - succ) * 20) == g + so + gr + ex}
    c1["table3_ik_le_clear"] = {k: v[2] <= v[1] for k, v in T3.items()}
    c1["table4_ik_equals_table3_on"] = {"tool": T4["tool/Ours"][3] == T3["tool_on"][2],
                                        "pp": T4["pp/Ours"][3] == T3["pp_on"][2]}
    out["C1_internal_consistency"] = c1

    # C2 신뢰구간과 검정
    c2 = {}
    for i, (cond, n) in enumerate((("standard", 10), ("cluttered_precise", 10), ("all", 20))):
        ko = round(T1["Ours"][i] * n)
        for b in ("MOKA", "ReKep"):
            kb = round(T1[b][i] * n)
            c2[f"{cond}/Ours_vs_{b}"] = {"ours": [ko, n, wilson(ko, n)], b: [kb, n, wilson(kb, n)],
                                         "fisher_p": fisher(ko, n, kb, n)}
    for j, stage in enumerate(("approach", "grasp", "success")):
        ko = round(T1_VLA["Ours"][j] * 20)
        for b in ("GR00T", "Being-H0"):
            kb = round(T1_VLA[b][j] * 20)
            c2[f"vla_{stage}/Ours_vs_{b}"] = {"ours": [ko, 20], b: [kb, 20], "fisher_p": fisher(ko, 20, kb, 20)}
    # 조건부: 잡은 뒤 성공 (grasp → success)
    c2["table3_tool_affordance_on_vs_off"] = {"on": "7/10", "off": "2/10", "fisher_p": fisher(7, 10, 2, 10)}
    c2["table5_retry_vs_no_retry_organize"] = {"retry": "8/10", "no_retry": "6/10", "fisher_p": fisher(8, 10, 6, 10)}
    c2["vla_success_given_grasp"] = {m: f"{round(v[2] * 20)}/{round(v[1] * 20)}" for m, v in T1_VLA.items()}
    out["C2_ci_and_tests"] = c2

    # C3 Table III 반올림 역산
    c3 = {}
    for name, (g, clear, ik, succ, scenes) in T3.items():
        cl = consistent_counts(g, clear, scenes)
        ikc = consistent_counts(g, ik, scenes)
        Ns = sorted({N for N, _ in cl} & {N for N, _ in ikc})
        c3[name] = {"possible_total_candidates": Ns,
                    "feasible_per_scene": round([k for n_, k in ikc if n_ == Ns[0]][0] / scenes, 1) if Ns else None,
                    "examples": [(N, [k for n_, k in cl if n_ == N], [k for n_, k in ikc if n_ == N]) for N in Ns[:3]],
                    "consistent": bool(Ns)}
    out["C3_table3_rounding"] = c3

    # C4 Table IV
    c4 = {"repeatability_ratio_sigma": round(T4_REP["MOKA"][0] / T4_REP["Ours"][0], 2),
          "repeatability_ratio_delta_max": round(T4_REP["MOKA"][1] / T4_REP["Ours"][1], 1),
          "worst_scene_lower_bound_mm": {}}
    # 장면 s마다 P_avg,s <= c_s * P_max,s + (1 - c_s) * 1mm (1 mm 이하 waypoint는 '비충돌')
    # → Σ c_s P_max,s >= n (P_avg - (1 - Coll))  이고  Σ c_s P_max,s <= max_s P_max,s · n·Coll
    for key, (pmax, pavg, coll, _, n) in T4.items():
        rows = []
        for coll_r in (coll - 0.5, coll, coll + 0.5):  # Coll%는 정수 반올림
            c = min(1.0, max(1e-9, coll_r / 100))
            rows.append(round((pavg - 0.05 - (1 - c) * 1.0) / c, 1))
        c4["worst_scene_lower_bound_mm"][key] = {
            "mean_P_max": pmax, "lower_bound_on_max_scene_P_max": min(rows),
            "share_of_total_P_max_mass": round(min(rows) / (pmax * n), 2),
        }
    # C4b σ 평균이 한 장면에 얼마나 좌우되는가 (10 장면 × 8회, δ_max 정의가 본문에 모호해 두 해석으로 계산)
    n_sc, runs = 10, 8
    rep = {}
    for m, (sig, dmax) in T4_REP.items():
        # 해석 A: δ_max = 장면별 σ_s의 최댓값 → 나머지 9개 장면의 σ 평균
        a = (n_sc * sig - dmax) / (n_sc - 1)
        # 해석 B: δ_max = 한 실행의 중심 이탈 최댓값 → 그 장면의 σ_s ≥ δ_max/√(runs-1) (편차 합 0 제약)
        sb = dmax / math.sqrt(runs - 1)
        b = (n_sc * sig - sb) / (n_sc - 1)
        rep[m] = {"A_other9_mean_sigma_mm": round(a, 2), "B_worst_scene_sigma_ge_mm": round(sb, 1),
                  "B_other9_mean_sigma_le_mm": round(b, 2)}
    c4["sigma_outlier_dependence"] = rep
    out["C4_table4"] = c4

    # C5 Table V 연쇄
    c5 = {}
    for task, d in T5.items():
        row = {}
        for mode, steps in d.items():
            chain_ok = all(steps[i + 1][1] == steps[i][0] for i in range(2))  # 다음 단계 분모 = 이전 단계 성공
            e2e_ok = steps[3][0] == steps[2][0] and steps[3][1] == steps[0][1]
            row[mode] = {"chain_ok": chain_ok, "e2e_equals_last_step": e2e_ok,
                         "e2e": f"{steps[3][0]}/{steps[3][1]}", "wilson": wilson(*steps[3])}
        c5[task] = row
    out["C5_table5"] = c5

    # C6 실행 시간과 VLM 호출 수 (Sec. IV-D Runtime, Sec. III)
    M = 4
    out["C6_runtime_and_calls"] = {
        "grounding_sum_s": round(6.7 + 9.6 + 3.1, 1), "reported_grounding_s": 19.4,
        "motion_s": round(4.8 + 9.4, 1),
        "per_subtask_without_grasp_sim_s": round(19.4 + 4.8 + 9.4, 1),
        "vlm_calls_per_subtask_estimate": {
            # 기준 시점 선택(Eq. 3) 1 + 계획(Eq. 4) 1 + 키포인트 × 시점 + 어포던스 박스 M + 검증 1
            "pick_and_place (3 keypoints)": 1 + 1 + 3 * M + M + 1,
            "tool_use (grasp 2 + apply 1 + terminal 1 keypoints)": 1 + 1 + 4 * M + M + 1,
        },
        "note": "9.6 s가 키포인트 하나의 4개 시점 질의인지, 전체인지 본문이 명시하지 않음",
    }

    # C7 v1 ↔ v3
    out["C7_version_changes_v1_to_v3"] = {
        "method": "v1 Eq. 5–6의 reference-view ray voting(삼각측량 합의가 τ_tri 미만이면 대체)이 v3에서 사라지고 삼각측량 합의만 남음. 프로젝트 페이지 초록은 아직 ray voting을 설명",
        "robots": "v1 xArm만 → v3 Franka(zero-shot 기준선 비교) + xArm(VLA 비교)",
        "zero_shot_baselines": "v1 단일 시점 RGB-D 기준선 1개 → v3 MOKA, ReKep",
        "table1_trials": "v1 과제별 n=5 (Ours 5/5, 4/5, 4/5 vs RGB-D 4/5, 4/5, 2/5) → v3 조건별 n=10, 합계 n=20",
        "vla_demos": "v1 과제당 30개 시연, GR00T·Being-H0 모두 0/5(두 과제) → v3 100개 시연, GR00T 0.30, Being-H0 0.00 (한 과제, n=20)",
        "grounding_metric": "v1 정확도(L_grasp: RGB-D 16.43 cm vs Ours 4.58 cm)와 침투 오차(9.91 vs 9.81) → v3 반복성 σ·δ_max와 P_max·P_avg·Coll%",
        "views": "v1 2·3·5 시점 ablation(시점 수 효과가 작음) → v3 '모든 실험 4시점', 시점 수 ablation 없음",
        "long_horizon": "v1 Organize 4/6, Cooking 1/3 (retry 포함) → v3 8/10, 9/10 (retry 1회), 6/10, 6/10 (retry 없음)",
        "motion_generator": "v1 [54] 하나 → v3 cuRobo v2 [52] + Natural functional gradients [53] (Korea Univ. 공저자 추가)",
        "v1_collision_finding": "v1에서는 침투 오차가 RGB-D와 거의 같았음(9.91 vs 9.81) — v3 Table IV의 큰 차이(Coll% 100 vs 17)와 지표·기준선이 다름",
    }

    path = ROOT / "analysis" / "table_check_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for k, v in out.items():
        print(k, json.dumps(v, ensure_ascii=False)[:400])
    print("wrote", path)


if __name__ == "__main__":
    main()
