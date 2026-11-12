"""튜토리얼 페이지·저장소 README·챌린지 페이지의 숫자와 공식 채점 코드 상수를 점검한다 (CPU, 수 초).

실행: python3 -I analysis/page_checks.py [kit_dir] [leaderboard_json]
  kit_dir: 제출 키트를 푼 폴더(선택). 주면 채점 코드의 상수를 정규식으로 대조한다.
  leaderboard_json: 챌린지 리더보드 JSON(선택). 주면 트랙별 항목 수와 지표별 최선·중앙값을 요약한다.
  팀 이름은 출력하지 않는다.
결과: analysis/page_checks_results.json
"""
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "page_checks_results.json"


def prize_check():
    """페이지 상금 $54K가 나오는 구성: 트랙 1·3 각 9K, 트랙 2 3개 티어 각 9K, 산업 혁신상 9K."""
    per_track = 6_000 + 3_000
    tracks_plain = 3 * per_track + 9_000          # 트랙을 한 번씩만 셀 때
    tracks_tier = 2 * per_track + 3 * per_track + 9_000  # 트랙 2를 티어 3개로 셀 때
    return {"per_track_winner_plus_runner_up_usd": per_track,
            "sum_if_track2_counted_once_usd": tracks_plain,
            "sum_if_track2_counted_per_tier_usd": tracks_tier,
            "page_pool_usd": 54_000,
            "matches_page_only_if_tiers_counted": tracks_tier == 54_000}


def fisher_check():
    """Sim200→Real10 4/5 대 Raw→Real10 0/5, 두 꼬리 정확 검정. scipy 없이 초기하분포로 계산."""
    def hyp(a, b, c, d):
        n1, n2, k = a + b, c + d, a + c
        N = n1 + n2
        def pmf(x):
            return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(N, k)
        p_obs = pmf(a)
        lo, hi = max(0, k - n2), min(n1, k)
        return sum(pmf(x) for x in range(lo, hi + 1) if pmf(x) <= p_obs + 1e-12)
    return {"4of5_vs_0of5_two_sided_p": hyp(4, 1, 0, 5),
            "4of5_vs_0of5_note": "n=5 vs 5 실제 로봇 시행, 효과 크기 추정에는 신뢰구간이 필요함"}


def formula_check():
    """tools/e2e 코드와 문서의 식을 그대로 옮긴 값. 예: 목표 10개, 파일럿 성공률 0.4."""
    target, rate, safety = 10, 0.4, 1.125
    attempts = math.ceil(target / rate * safety)
    E, N, F, H, B = 5, 10, 600, 16, 32     # 에폭, 에피소드, 프레임, 액션 지평, 배치
    steps = math.ceil(E * N * (F - H + 1) / B)
    # 시간 상한: source_frames 600, source_fps 30, motion_speed 0.5, control_fps 20, 여유 15%
    ceiling = math.ceil(600 / 30 / 0.5 * 20 * 1.15)
    return {"collection_attempts_example": attempts, "finetune_steps_example": steps,
            "collection_max_steps_example_with_15pct_margin": ceiling}


def lift_hold_check():
    """repo 기본 프로파일 값에서 hold 임계와 lift 임계의 제약을 확인한다."""
    lift, hold, min_hold = 0.10, 0.05, 20
    return {"hold_le_lift": hold <= lift, "min_hold_steps_seconds_at_20hz": min_hold / 20.0,
            "placement_window_seconds_at_20hz": 20 / 20.0}


def kit_constants(kit_dir):
    """키트 채점 코드의 상수를 정규식으로 찾는다. 파일이 없으면 건너뛴다."""
    base = Path(kit_dir) / "metric_code"
    checks = {
        "track_2/SP-SR.py: SPIDER position threshold 0.1 m": (r"_SPIDER_POSITION_THRESHOLD_M = 0\.1\b", "track_2/SP-SR.py"),
        "track_2/SP-SR.py: SPIDER orientation threshold 0.5 rad": (r"_SPIDER_ORIENTATION_THRESHOLD_RAD = 0\.5\b", "track_2/SP-SR.py"),
        "track_2/MP-SR.py: ManipTrans position threshold 0.03 m": (r"_MANIPTRANS_POSITION_THRESHOLD_M = 0\.03\b", "track_2/MP-SR.py"),
        "track_2/MP-SR.py: ManipTrans orientation threshold 30 deg": (r"_MANIPTRANS_ORIENTATION_THRESHOLD_DEG = 30\.0\b", "track_2/MP-SR.py"),
        "track_2/AUC.py: ADD thresholds arange(0.01, 0.10, 0.01)": (r"_ADD_THRESHOLDS_M = np\.arange\(0\.01, 0\.10, 0\.01\)", "track_2/AUC.py"),
        "track_2/MPPE.py: keypoint radius 0.05 m": (r"_MPPE_KEYPOINT_RADIUS_M = 0\.05\b", "track_2/MPPE.py"),
        "track_1/PEN.py: penetration = relu(-sdf) mean x100 (cm)": (r"np\.maximum\(-distances, 0\.0\)\.mean\(axis=1\) \* 100\.0", "track_1/PEN.py"),
        "track_1/PEN.py: contact accuracy threshold 0.015 m": (r"distance_threshold_m: float = 0\.015", "track_1/PEN.py"),
        "track_3/RPE.py: 10000 metric samples": (r"_METRIC_SAMPLES = 10_000", "track_3/RPE.py"),
    }
    out = {}
    for label, (pat, rel) in checks.items():
        f = base / rel
        if not f.exists():
            out[label] = "파일 없음"
            continue
        out[label] = bool(re.search(pat, f.read_text(errors="replace")))
    return out


def leaderboard_summary(path):
    """트랙별 항목 수와 지표별 최선·중앙값. 팀 이름은 넣지 않는다."""
    d = json.loads(Path(path).read_text())
    out = {"generated_at": d.get("generated_at")}
    for t in d["tracks"]:
        row = {"entries": len(t["rows"]), "incomplete": len(t.get("incomplete", []))}
        for m in t["metrics"]:
            vals = sorted(r["scores"][m["key"]] for r in t["rows"] if r["scores"].get(m["key"]) is not None)
            if vals:
                best = max(vals) if m["higher_is_better"] else min(vals)
                row[m["key"]] = {"n": len(vals), "best": round(best, 4), "median": round(vals[len(vals) // 2], 4),
                                 "higher_is_better": m["higher_is_better"]}
        out[t["key"]] = row
    return out


def main():
    res = {"prize": prize_check(), "fisher": fisher_check(), "formulas": formula_check(),
           "lift_hold": lift_hold_check()}
    if len(sys.argv) > 1 and sys.argv[1] != "-":
        res["kit_constants"] = kit_constants(sys.argv[1])
    if len(sys.argv) > 2:
        res["leaderboard"] = leaderboard_summary(sys.argv[2])
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1)[:3000])


if __name__ == "__main__":
    main()
