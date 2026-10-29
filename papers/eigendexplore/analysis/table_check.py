"""본문·그림·프로젝트 페이지 수치 재계산 (CPU). 실행: python3 analysis/table_check.py

- C1–C4: 헤드라인 비율(34%, 40%, 33→82%, 41%, 28%)
- C5: DexMachina "28%"를 Fig. 5 판독값(fig_digitize_results.json)으로 재계산
- C6: SPIDER 손별 Δcost 평균(16%)과 DoF·k·정렬 이득과의 순위 상관
- C7: Fig. 7b 수치(0.066, 0.089, 0.126)
- C8: 실물 SimToolReal: Fig. 8의 과제별 평균 4개로 33%·82% 재계산
- C9: 계산 규모(프레임 수, 실행 수) 환산
"""
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "analysis"


def pct(a, b):
    return round(100 * (b / a - 1), 1)


def main():
    out = {}
    out["C1_simtoolreal_9B"] = {"joint": 7.9, "ours": 10.6, "gain_pct": pct(7.9, 10.6), "claim": "34% more consecutive successes (Sec. 1)",
                                "n_seeds": 5}
    out["C2_dextreme_2B"] = {"joint": 8.8, "ours": 12.3, "gain_pct": pct(8.8, 12.3), "claim": "40% more in DeXtreme (Sec. 1)", "n_seeds": 5}
    out["C3_extended_60B_single_run"] = {"joint": 4.9, "eigen_residual": 12.0, "ours": 14.1,
                                         "ours_over_joint": round(14.1 / 4.9, 2), "ours_over_residual_pct": pct(12.0, 14.1),
                                         "note": "방법마다 실행 1개(9B 시점 중앙값 seed를 이어 학습). v1은 Sec. 1에서 'larger gains in extended runs'로 내세웠고 v2에서 뺌"}
    out["C4_real_dextreme"] = {"mean": [8.45, 11.90], "gain_pct": pct(8.45, 11.90), "longest": [32, 41], "longest_gain_pct": pct(32, 41),
                               "trials_per_method": 40, "claim": "41%, 28% (Sec. 4.5.2)"}
    # C5: DexMachina
    fd = json.loads((A / "fig_digitize_results.json").read_text())
    out["C5_dexmachina_from_fig5"] = {"task_only": fd["summary"]["task_only"], "curriculum": fd["summary"]["curriculum"],
                                      "claim": "improves average object-tracking accuracy by 28% without contact reward shaping or curricula (Sec. 1)",
                                      "digitize_check": fd["text_check"]}
    # C6: SPIDER
    hands = ["sharpa", "wuji", "allegro", "xhand", "schunk", "inspire"]
    dof = np.array([22, 20, 16, 12, 9, 6])
    dcost = np.array([25.4, 31.2, 21.5, 14.3, 3.5, 0.4])            # Fig. 6
    k = np.array([9, 9, 10, 6, 5, 3])
    pcr = json.loads((A / "pca_check_results.json").read_text())
    align = np.array([pcr["P4_alignment_gain_all_hands"][h]["human_error_dirs"] for h in hands])
    rows = {}
    for name, x in (("DoF", dof), ("k90", k), ("d_minus_k", dof - k), ("alignment_gain", align)):
        r, p = stats.spearmanr(x, dcost)
        rows[name] = {"spearman": round(float(r), 3), "p": round(float(p), 3), "values": [round(float(v), 3) for v in x]}
    out["C6_spider"] = {"delta_cost_pct": dcost.tolist(), "mean_delta_pct": round(float(dcost.mean()), 2),
                        "claim": "reduces retargeting cost by 16% on average across six hands (Sec. 1)", "rank_corr_with_delta": rows,
                        "text_rounding": {"Schunk 3.5 → '4%'": True, "Allegro 21.5 → '22%'": True}}
    # C7: Fig. 7b
    out["C7_fig7b_sharpa_spider"] = {"joint": 0.089, "eigen_only_rho0.5": 0.126, "combined_rho0.9": 0.066,
                                     "combined_vs_joint_pct": pct(0.089, 0.066), "eigen_only_rho0.5_vs_joint_pct": pct(0.089, 0.126),
                                     "fig6_sharpa_pct": -25.4, "claim": "25% below Joint-Space (Sec. 4.4)"}
    # C8: 실물 SimToolReal (Fig. 8 박스 옆에 적힌 과제별 평균)
    fig8 = {"hammering": (0.42, 0.71), "wiping": (0.34, 0.92), "scooping": (0.20, 0.76), "brushing": (0.36, 0.90)}
    j = np.mean([v[0] for v in fig8.values()])
    o = np.mean([v[1] for v in fig8.values()])
    out["C8_real_simtoolreal"] = {"per_task_fig8": fig8, "mean_of_tasks": [round(float(j), 4), round(float(o), 4)],
                                  "text_claim": "averaging 82% across tasks compared with 33% (Sec. 4.5.1)",
                                  "project_page_hammer": [0.42, 0.71],
                                  "trials": "물체-궤적 조합당 10회, 정책은 방법마다 1개. 표준편차·검정 없음",
                                  "note": "Joint-Space 정책은 원 SimToolReal 공개 체크포인트가 아니고, 120B의 절반인 60B 학습(Sec. 4.5.1)"}
    # C9: 계산 규모
    sim_frames = {"SimToolReal fixed (4 methods × 5 seeds × 9B)": 4 * 5 * 9e9,
                  "SimToolReal extended (4 runs × 51B more)": 4 * 51e9,
                  "DeXtreme (4 methods × 5 seeds × 2B)": 4 * 5 * 2e9,
                  "DexMachina (4 hands × 2 tasks × 2 settings × 2 methods × 5 seeds × ~2B)": 4 * 2 * 2 * 2 * 5 * 2e9,
                  "Appendix A delta variants (2 × 5 seeds × (9B + 2B))": 2 * 5 * (9e9 + 2e9),
                  "Dataset ablation (2 extra priors × 5 seeds × 9B)": 2 * 5 * 9e9}
    out["C9_compute"] = {"frames": {k_: f"{v:.3g}" for k_, v in sim_frames.items()},
                         "total_rl_frames": f"{sum(sim_frames.values()):.3g}",
                         "extended_run_frames_per_s": round(60e9 / (7 * 86400)),
                         "spider_runs_main": 6 * 10 * 10 * 2,
                         "spider_runs_sweep_sharpa": 6 * 2 * 10 * 10 * 3,
                         "note": "SimToolReal 연장 실행 1주(Sec. 4.1.1)로 처리량 환산. SPIDER sweep은 ρ 6개 × (eigen만/결합) × 과제 10 × seed 10 × 데이터셋 3으로 가정"}
    path = A / "table_check_results.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
