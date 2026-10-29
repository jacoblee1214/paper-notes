"""논문 수치·표와 공식 저장소 문서의 수치 재계산 (CPU). 실행: python3 analysis/table_check.py

- C1: Table 1 기호 집계, RoboSimStudio만 ✓인 행
- C2: GaP 곡선(Fig. 1, 7f)의 0·29·71·100%가 어떤 분모 n에서 k/n 반올림으로 나오는지 (30개 배치와 맞는지)
- C3: 에이전트 구축 시간표(Fig. 7c)
- C4: 공식 저장소 wipe_the_bar 데모의 측정값(README, 커밋 81f57c9): latch 유무 비교
- C5: Preston 마모 법칙 수치 검산(데모의 k_p, p_min, 스트로크 조건)
- C6: 스펀지 강성과 집게 압착: 데모 README 값으로 선형 탄성 추정
- C7: v1 ↔ v2 텍스트 비교 수치(pdftotext + difflib로 계산해 둔 값)
"""
import json
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

# Table 1 (논문 3쪽). ✓ = Y, ∼ = P(partial), – = N, Custom = C
FRAMEWORKS = ("MuJoCo", "IsaacLab", "ManiSkill3", "Genesis", "Newton", "RoboSimStudio")
TABLE1 = {
    "Rigid/articulated bodies": "YYYYYY",
    "Deformable objects": "YYPYYY",
    "Coupled multi-physics solvers": "NPNYPY",
    "Interactive scene inspection": "YYYYYY",
    "Physics parameter tuning": "PYPPPY",
    "Structured feedback for coding agents": "CCCCCY",
    "Shared human-agent refinement": "CCCCCY",
    "Video-driven agentic Real2Sim": "NNNNNY",
    "Force-feedback Real2Sim": "NPNNNY",
    "Failure-driven simulation refinement": "NNNNNY",
}

# Fig. 1 ④ / Fig. 7f: GaP 반복별 값
GAP_CURVE = {0: 0, 9: 29, 11: 71, 14: 100}
N_LAYOUTS = 30  # Sec. 4.3, Fig. 7e

# Fig. 7c: 에이전트 구축 시간표 (분)
TIMELINE = {"arms, bar": 14, "first wipe": 42, "scenery": 51, "verified": 56}

# 공식 저장소 examples/08_demos/wipe_the_bar/README.md (커밋 81f57c9) "Measured" 절과 "Why a latch" 절
DEMO = {
    "latch": {"cleaned": [0.955, 0.959, 0.959], "peak_press_N": [4.93, 4.91, 4.91], "peak_pressure_kPa": 84},
    "latch_strokes": [0.91, 0.94, 0.95, 0.955],  # 네 번 스트로크 뒤 누적 cleaned (마지막은 0.95–0.96)
    "no_latch": {"cleaned": [0.761, 0.724, 0.687], "slide_mm": [69, 226, 218]},
    "extra_no_latch_takes": 30,  # "Over 30 more takes, none kept the slide under 10 mm"
    "gpu": "RTX PRO 6000 Blackwell",
    "sim_seconds_per_take": 59.5,
}
# bar.py StripeSpec / Layout
PRESTON = {"k_p": 1.2e-2, "p_min": 300.0, "cleared_below": 0.1, "sponge_m": (0.050, 0.075, 0.030),
           "stroke_speed": 0.060, "press_N": 4.0, "claim_speed": 0.030, "claim_seconds": 2.0}
# bar.py FOAM 주석: 80 kPa 폼은 50 mm → 45.5 mm, 패드당 8.6–8.9 N
SQUEEZE = {"E_twin": 80e3, "w0_mm": 50.0, "w_mm": 45.5, "F_pad": 8.75, "E_clip": 20e3,
           "E_fig7e": (6e3, 20e3, 100e3)}


def main() -> None:
    out: dict = {}

    # C1 Table 1 집계
    tally = {fw: {"Y": 0, "P": 0, "N": 0, "C": 0} for fw in FRAMEWORKS}
    for row in TABLE1.values():
        for fw, c in zip(FRAMEWORKS, row):
            tally[fw][c] += 1
    only_rss = [k for k, row in TABLE1.items() if row[-1] == "Y" and "Y" not in row[:-1]]
    out["C1_table1_tally"] = {
        "tally": tally,
        "rows_only_rss_full_support": only_rss,
        "n_rows": len(TABLE1),
        "note": "RSS만 ✓인 5행은 모두 이 논문이 정의한 워크플로(구조화 피드백, 공동 정제, 영상·힘·실패 기반 Real2Sim)",
    }

    # C2 GaP 곡선 분모
    def consistent(n: int) -> bool:
        return all(any(round(100 * k / n) == v for k in range(n + 1)) for v in GAP_CURVE.values())

    ns = [n for n in range(1, 61) if consistent(n)]
    out["C2_gap_curve_denominators"] = {
        "values": GAP_CURVE,
        "consistent_n_upto_60": ns,
        "n_layouts_30_consistent": consistent(N_LAYOUTS),
        "k_over_30_near_29_71": [round(100 * k / 30, 1) for k in (8, 9, 21, 22)],
        "axis_labels": {"Fig. 1": "completion rate", "Sec. 4.3 / Fig. 7f": "simulated wiped area"},
        "note": "30개 배치의 성공 비율이라면 29%·71%가 나올 수 없음 → 닦인 면적 같은 연속량이거나 다른 분모",
    }

    # C3 시간표
    out["C3_agent_timeline_min"] = {
        "timeline": TIMELINE,
        "first_wipe_share_of_total": round(TIMELINE["first wipe"] / TIMELINE["verified"], 3),
        "claim": "approximately one hour (Sec. 4.3, 두 에이전트 모두), 56분은 한 에이전트만 보고",
    }

    # C4 latch 유무
    a, b = DEMO["latch"]["cleaned"], DEMO["no_latch"]["cleaned"]
    t = stats.ttest_ind(a, b, equal_var=False)
    out["C4_demo_latch_effect"] = {
        "latch_mean": round(sum(a) / 3, 4), "no_latch_mean": round(sum(b) / 3, 4),
        "diff_pp": round(100 * (sum(a) - sum(b)) / 3, 1), "welch_p": float(f"{t.pvalue:.2e}"),
        "no_latch_slide_mm": DEMO["no_latch"]["slide_mm"],
        "extra_takes_none_under_10mm": DEMO["extra_no_latch_takes"],
        "note": "공식 twin은 friction만으로 스펀지를 잡지 못해 패드 주변 폼 노드 106개를 손에 고정(latch)한다",
    }

    # C5 Preston 마모 검산: ink' = -k_p p v, 0.9를 지우면 clean(≤ 0.1)
    kp = PRESTON["k_p"]
    sx, sy, sz = PRESTON["sponge_m"]
    p_face = PRESTON["press_N"] / (sx * sy)
    t_claim = 0.9 / (kp * p_face * PRESTON["claim_speed"])
    wear_per_stroke_face = kp * p_face * sy  # 셀 하나가 스펀지 긴 변(75 mm)을 지나는 동안의 마모량 = k_p p L (속도 무관)
    p_for_one_stroke = 0.9 / (kp * sy)
    out["C5_preston_check"] = {
        "face_mean_pressure_Pa": round(p_face, 1),
        "time_to_clear_at_30mm_s_face_mean_s": round(t_claim, 2),
        "claim": "a 4 N, 30 mm/s stroke clears a cell in ~2 s (bar.py)",
        "time_to_clear_at_peak_84kPa_s": round(0.9 / (kp * 84e3 * PRESTON["claim_speed"]), 3),
        "wear_per_stroke_at_face_mean": round(wear_per_stroke_face, 3),
        "pressure_needed_one_stroke_clean_Pa": round(p_for_one_stroke, 0),
        "p_min_Pa": PRESTON["p_min"],
        "note": "Preston 법칙에서 한 스트로크의 마모 = k_p·p·L이라 스트로크 속도와 무관. 측정 첫 스트로크 0.91과 같은 자릿수",
    }

    # C6 강성과 압착: 선형 탄성 ε = F/(E A)
    eps_twin = (SQUEEZE["w0_mm"] - SQUEEZE["w_mm"]) / SQUEEZE["w0_mm"]
    a_eff = SQUEEZE["F_pad"] / (SQUEEZE["E_twin"] * eps_twin)
    pred = {f"{int(E / 1e3)}kPa": round(SQUEEZE["F_pad"] / (E * a_eff), 3)
            for E in (*SQUEEZE["E_fig7e"], SQUEEZE["E_twin"])}
    out["C6_sponge_squeeze_linear"] = {
        "twin_strain": round(eps_twin, 3),
        "implied_pad_area_cm2": round(a_eff * 1e4, 2),
        "predicted_strain": pred,
        "observed": "20 kPa(영상의 부엌 스펀지)는 패드 사이로 흘러나가 집게가 맞닿음; 80 kPa로 바꿔서 잡힘 (bar.py FOAM 주석)",
        "note": "Fig. 7e의 6·20·100 kPa 중 6·20 kPa는 공개 twin의 집게 조건에서 잡히지 않는 범위",
    }

    # C7 판본 비교 (v1·v2 PDF를 pdftotext + difflib로 비교해 얻은 값을 기록)
    out["C7_v1_v2"] = {
        "pages": {"v1": 13, "v2": 14},
        "word_count_pdftotext": {"v1": 5994, "v2": 6382},
        "word_sequence_ratio": 0.8722,
        "changes": ["Fig. 1 라벨 교체(Simulation Variation → Agentic Parameter Tuning 등)",
                    "관련 연구에 MAESTRO·ASPIRE·ENPIRE·DexAgent·RPG·Push-T 추가 (참고문헌 51 → 58개)",
                    "Real2Sim 절: frontier 모델(Claude Fable 5.1, GPT-6 Astra, Gemini 4 Argon)의 시각 재구성 한계 문단 추가",
                    "v1 Fig. 1 캡션의 'real rollout can be further provided' 문장 삭제"],
    }

    path = ROOT / "analysis" / "table_check_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for k, v in out.items():
        print(k, json.dumps(v, ensure_ascii=False)[:260])
    print("wrote", path)


if __name__ == "__main__":
    main()
