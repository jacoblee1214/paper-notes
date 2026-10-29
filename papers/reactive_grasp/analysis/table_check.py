"""그림·표·본문 수치 재계산 (CPU). 실행: python3 analysis/table_check.py

- C1: Fig. 3 막대값이 각 구간 n에서 정수 k/n으로 나올 수 있는지, 790개 가중 평균, 본문 서술(3–5 구간, 1/3)
- C2: Fig. 4 판독값(fig_digitize_results.json)이 50 grasp·500 시행의 격자에 맞는지, 각도 평균과 Fig. 7 비교
- C3: Fig. 6, Fig. 7, 정밀 파지 38.5%→42.1% 값이 명시된 시행 수에서 나올 수 있는지
- C4: Table I 합계와 본문(78회, 8회 상실, 6회 복구), Wilson 구간
- C5: Fig. 7 계산 속도 ↔ QP 접점 수 선형 적합, 프로젝트 페이지의 "약 84 Hz"와 대조
- C6: 하중 시험 크기 대 마찰 용량, Fig. 5 잔차 0.428 N 대 하중
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import math
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
G = 9.81


def wilson(k: int, n: int, z: float = 1.959964) -> list:
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 3), round(c + h, 3)]


def attainable(pct: float, n: int, decimals: int = 0) -> list:
    """반올림해서 pct(%)가 되는 정수 k (0..n)."""
    return [k for k in range(n + 1) if round(100 * k / n, decimals) == round(pct, decimals)]


def main() -> None:
    out: dict = {}

    # C1 Fig. 3 (막대 위 숫자, 정수 %)
    bins = ["2", "3", "4", "5", "6", "7", "8", "9", ">=10"]
    n = [51, 45, 78, 98, 131, 116, 116, 100, 55]
    fig3 = {
        "feedforward": [6, 33, 32, 43, 76, 76, 83, 88, 100],
        "impedance": [12, 36, 47, 56, 82, 78, 83, 91, 100],
        "ours": [37, 82, 83, 95, 99, 98, 100, 100, 100],
    }
    c1 = {"n_total": sum(n), "share_le5_contacts": round(sum(n[:4]) / sum(n), 3)}
    ks = {}
    for m, vals in fig3.items():
        kk = []
        for v, ni in zip(vals, n):
            cand = attainable(v, ni)
            kk.append(cand[0] if len(cand) == 1 else (cand if cand else None))
        ks[m] = kk
        flat = [k if isinstance(k, int) else k[0] for k in kk]
        c1[f"{m}_overall"] = {"k": sum(flat), "rate": round(sum(flat) / sum(n), 3),
                              "wilson": wilson(sum(flat), sum(n))}
    c1["counts_per_bin"] = ks
    c1["all_values_attainable"] = all(k is not None for kk in ks.values() for k in kk)
    c1["text_claim_3to5"] = {"baselines_range": [min(fig3["feedforward"][1:4] + fig3["impedance"][1:4]),
                                                 max(fig3["feedforward"][1:4] + fig3["impedance"][1:4])],
                             "ours_range": [min(fig3["ours"][1:4]), max(fig3["ours"][1:4])],
                             "claim": "32–56% versus 82–95% (Sec. IV-A.2)"}
    # 3–5 구간: Ours vs 더 나은 기준선(impedance) Fisher
    k_o = sum(ks["ours"][i] for i in (1, 2, 3))
    k_i = sum(ks["impedance"][i] for i in (1, 2, 3))
    nn = sum(n[1:4])
    c1["bins3to5_ours_vs_impedance"] = {"ours": [k_o, nn], "impedance": [k_i, nn],
                                        "fisher_p": float(f"{stats.fisher_exact([[k_o, nn - k_o], [k_i, nn - k_i]])[1]:.2e}")}
    out["C1_fig3"] = c1

    # C2 Fig. 4 판독값
    dig = json.loads((ROOT / "analysis" / "fig_digitize_results.json").read_text())["values_pct"]
    c2 = {}
    for m, d in dig.items():
        c2[m] = {
            "0deg_on_2pct_grid(50 grasps)": abs(d["0"] / 2 - round(d["0"] / 2)) < 0.15,
            "30to90_nearest_0.2_grid(500 trials)": [round(round(d[a] / 0.2) * 0.2, 1) for a in ("30", "60", "90")],
            "mean_30_60_90": round((d["30"] + d["60"] + d["90"]) / 3, 2),
        }
    c2["fig7_K5_success_reported"] = 93.2
    c2["ours_90deg_vs_fingertip_fisher_unpaired"] = float(
        f"{stats.fisher_exact([[443, 57], [362, 138]])[1]:.2e}")
    c2["note"] = "Fig. 7(K=5)은 같은 50 grasp·세 각도·5회 반복. Fig. 4 Ours 세 각도 평균과 0.5%p 안에서 일치"
    out["C2_fig4"] = c2

    # C3 다른 그림의 시행 수 정합
    c3 = {}
    for name, v, nn in (("fig7_K1", 90.9, 750), ("fig7_K5", 93.2, 750), ("fig7_K7", 91.9, 750)):
        c3[name] = attainable(v, nn, 1)
    # 정밀 파지: 68 grasp라면 38.5%, 42.1%가 나올 수 없다 → 파지당 시행 수 m 탐색
    prec = {}
    for m in range(1, 11):
        a, b = attainable(38.5, 68 * m, 1), attainable(42.1, 68 * m, 1)
        if a and b:
            prec[m] = {"k_without": a, "k_with": b}
    c3["precision_68_grasps_single_trial_possible"] = bool(attainable(38.5, 68, 1) and attainable(42.1, 68, 1))
    c3["precision_trials_per_grasp_m_that_work"] = prec
    m5 = prec.get(5)
    if m5:
        a, b = m5["k_without"][0], m5["k_with"][0]
        c3["precision_m5_fisher_p_unpaired"] = round(stats.fisher_exact([[b, 340 - b], [a, 340 - a]])[1], 3)
    # Fig. 7: K=1 vs K=5 비대응 근사
    c3["fig7_K1_vs_K5_fisher_p_unpaired"] = round(stats.fisher_exact([[699, 51], [682, 68]])[1], 3)
    out["C3_trial_grids"] = c3

    # C4 Table I
    table = {"Pringles": (59, 16, 3, 3), "Small box": (48, 31, 2, 1), "Large box": (94, 31, 3, 2)}
    dur = sum(v[0] for v in table.values())
    pert = sum(v[1] for v in table.values())
    loss = sum(v[2] for v in table.values())
    rec = sum(v[3] for v in table.values())
    out["C4_table1"] = {
        "duration_s": dur, "perturbations": pert, "losses": loss, "recovered": rec,
        "text": "78 perturbations over about 200 s; eight losses; six recovered (Sec. IV-C.3)",
        "loss_per_perturbation": [round(loss / pert, 3), wilson(loss, pert)],
        "recovery_all_losses": [round(rec / loss, 3), wilson(rec, loss)],
        "recovery_excluding_workspace_exits": [rec, loss - 2],
        "per_object_loss_rate": {k: round(v[2] / v[1], 3) for k, v in table.items()},
        "perturbations_per_minute": round(pert / (dur / 60), 1),
    }

    # C5 계산 속도 적합: t = a + b * N_contacts
    n1, f1, n7, f7 = 14, 227.0, 91, 66.0
    t1, t7 = 1000 / f1, 1000 / f7
    b = (t7 - t1) / (n7 - n1)
    a = t1 - b * n1
    n5_lin = n1 + (n7 - n1) * (5 - 1) / (7 - 1)
    out["C5_compute"] = {
        "ms_at_K1": round(t1, 2), "ms_at_K7": round(t7, 2),
        "fit_ms": {"intercept": round(a, 2), "per_contact": round(b, 4)},
        "K5_contacts_if_linear_in_K": round(n5_lin, 1),
        "K5_rate_pred_Hz": round(1000 / (a + b * n5_lin), 1),
        "project_page_rate_Hz": 84,
        "contacts_implied_by_84Hz": round((1000 / 84 - a) / b, 1),
        "qp_vars_at_K5": int(round(3 * n5_lin)),
        "hardware_loop_Hz": 30,
        "note": "실물에서는 추적 자세가 올 때마다(30 Hz) 접점 추정과 QP를 돈다(Sec. IV-C.1)",
    }

    # C6 하중 대 마찰 용량
    mu, fmin = 0.7, 10.0
    cap = mu * fmin
    out["C6_load_vs_capacity"] = {
        "friction_capacity_N(mu*fn_min)": cap,
        "mass_where_0.1mg_equals_capacity_kg": round(cap / (0.1 * G), 2),
        "pyramid_vs_cone_min_ratio": round(1 / math.sqrt(2), 3),
        "fig5_residual_N": 0.428,
        "mass_where_residual_equals_0.1mg_kg": round(0.428 / (0.1 * G), 3),
        "mass_where_residual_equals_weight_kg": round(0.428 / G, 3),
        "hand_surface_area_cm2_implied(3000 pts / 2 per cm2)": 1500,
        "note": "중력이 꺼진 시험 하중은 마찰 용량의 수십분의 일 이하. Fig. 5의 자기 잔차가 0.44 kg 미만 물체에서는 시험 하중보다 크다",
    }

    path = ROOT / "analysis" / "table_check_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for k, v in out.items():
        print(k, json.dumps(v, ensure_ascii=False)[:400])
    print("wrote", path)


if __name__ == "__main__":
    main()
