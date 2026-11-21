"""논문 수치 재계산과 내부 일관성 검사 (표준 라이브러리만 사용, 수 초).

입력: analysis/inputs/ 의 저자 공개 결과 파일 (작은 메타데이터 JSON 네 개, 원 저장소 results/ 에서 복사)
  - d_ladder.json : 적응 사다리(논문 §9.5, 부록 C)의 정책별 걷기 점수, 낙상률, 판정
  - d_stats.json  : 같은 연구의 등록 통계 (논문 §9.5 표)
  - c_authors.json: 저자 정책의 걷기 점수 (논문 §3.5, §9.1)
  - timeline.json : 등록 커밋 시각 (논문 부록 B)

출력: analysis/recompute_results.json (검사 이름, 논문 값, 재계산 값, 판정)
실행: python3 analysis/recompute.py
"""
import json
import math
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
INP = HERE / "inputs"
OUT = HERE / "recompute_results.json"
FAIL = 0.15  # 논문 §9.5: 정책 실패 기준 (걷기 점수 < 0.15 또는 학습 붕괴 0.000)

ladder = json.loads((INP / "d_ladder.json").read_text(encoding="utf-8"))
stats = json.loads((INP / "d_stats.json").read_text(encoding="utf-8"))
authors = json.loads((INP / "c_authors.json").read_text(encoding="utf-8"))
timeline = json.loads((INP / "timeline.json").read_text(encoding="utf-8"))

checks = []


def check(key, what, paper, recomputed, ok, note=""):
    checks.append({"key": key, "what": what, "paper": paper, "recomputed": recomputed, "ok": bool(ok), "note": note})


# ---------- 1. 부호 검정: 10개 에피소드 모두 같은 부호일 때의 정확 양측 p ----------
def signflip_two_sided_all_same(n):
    # 모든 부호가 같을 때 가능한 부호 배치 2^n 중 극단 2개
    return 2 / 2 ** n


p10 = signflip_two_sided_all_same(10)
check("signflip_10", "10/10 에피소드 부호 검정 양측 p (§5.1, 'p = 0.002')", 0.002, round(p10, 5), abs(p10 - 0.002) < 0.0005)
p4 = signflip_two_sided_all_same(4)
check("signflip_4", "4 궤적 전부 같은 부호일 때 최소 양측 p (§3.6, '1/8 아래로 갈 수 없음')", 0.125, p4, abs(p4 - 0.125) < 1e-9)

# ---------- 2. 데이터 창 수 (§4.3) ----------
n_rows = 10000
win = n_rows - 32 - 8 + 1  # 40행 창 전체 수 (코드 train.py: range(shape[1]-H-F+1))
glued = 352
check("windows_total", "창 전체 수 9,961 = 10,000 - 40 + 1 (§4.3)", 9961, win, win == 9961)
check("windows_kept", "에피소드 경계 보정 뒤 창 수 9,609 = 9,961 - 352 (§4.3)", 9609, win - glued, win - glued == 9609)
check("glued_share", "접합 창 비율 352/9,961 = 3.5% (§4.3)", "3.5%", f"{100 * glued / win:.2f}%", round(100 * glued / win, 1) == 3.5)

# ---------- 3. 데이터 구조 (§3.2, §3.4) ----------
check("rows_seconds", "10,000행 = 50 Hz 200 s (§3.2)", 200, n_rows / 50, n_rows / 50 == 200)
check("stride", "보행 주기 27 스텝 = 0.54 s, 1.85 Hz (§3.2)", "0.54 s / 1.85 Hz", f"{27 / 50:.2f} s / {50 / 27:.2f} Hz", round(50 / 27, 2) == 1.85)
check("traj_per_episode", "에피소드 20 s(1,000행)당 400스텝 궤적 2개 (§3.4)", 2, 1000 // 400, 1000 // 400 == 2)
check("pool_counts", "학습 시뮬레이터 수 = 57 + 55 - 11 중복 = 101 (§3.4)", 101, 57 + 55 - 11, 57 + 55 - 11 == 101)
check("arch_counts", "다른 구조 38 = 13 + 12 + 13 (§3.4)", 38, 13 + 12 + 13, 13 + 12 + 13 == 38)
check("compressed_sum", "압축 복사본: 초록 '200'은 풀 합 40 + 200 = 240과 다름 (초록 vs §3.4)", 200, 40 + 200,
      False, "초록의 '200 compressed copies'는 §3.4 표의 두 풀 합(240)과 맞지 않음. 중복 여부 설명 없음")

# ---------- 4. 저자 정책 (§3.5, §9.1) ----------
check("authors_score", "저자 정책 걷기 점수 0.942 (§3.5)", 0.942, round(authors["score"], 3), round(authors["score"], 3) == 0.942)
check("standing_score", "가만히 서 있기 0.069 (§3.5)", 0.069, round(authors["standing_reference"], 3),
      round(authors["standing_reference"], 3) == 0.069)
check("episodes", "걷기 점수 평가 96 에피소드 (§3.5)", 96, authors["n_episodes"], authors["n_episodes"] == 96)

# ---------- 5. 적응 사다리 (§9.5, 부록 C) ----------
tests = ladder["tested"]
n_tests = len(tests)
pols = [p for t in tests for p in t["policies"]]
n_pol = len(pols)
n_third = sum(1 for t in tests if t.get("third_policy"))
zero = [p for p in pols if p["score"] == 0.0]
failed_pol = [p for p in pols if p["score"] < FAIL or p["score"] == 0.0]
more_half = [p for p in failed_pol if p["fall_rate"] > 0.5]
check("ladder_tests", "사다리 시험 시뮬레이터 수 18 (§9.5)", 18, n_tests, n_tests == 18)
check("ladder_policies", "정책 수 40 (§9.5)", 40, n_pol, n_pol == 40)
check("ladder_third", "세 번째 정책이 필요한 시험 4개 (§9.5)", 4, n_third, n_third == 4)
check("ladder_zero", "학습 붕괴(0.000) 정책 3개 (§9.5)", 3, len(zero), len(zero) == 3)
check("ladder_failed_policies", "실패 정책 27개 (§9.5)", 27, len(failed_pol), len(failed_pol) == 27)
check("ladder_failed_flag", "저장된 policy 'failed' 플래그와 점수 기준 실패 수 일치", len(failed_pol), sum(1 for p in pols if p["failed"]),
      sum(1 for p in pols if p["failed"]) == len(failed_pol))
check("ladder_fall_majority", "실패 정책 중 낙상률 > 0.5인 것 21개 (§9.5)", 21, len(more_half), len(more_half) == 21)

verdict_mismatch = []
did_not = []
for t in tests:
    nfail = sum(1 for p in t["policies"] if p["score"] < FAIL or p["score"] == 0.0)
    rec = "failed" if nfail >= 2 else "did not fail"
    if rec != t["verdict"]:
        verdict_mismatch.append((t["sim"], t["verdict"], rec))
    if rec == "did not fail":
        did_not.append(t["sim"])
check("ladder_verdicts", "시뮬레이터 판정 (정책 2개 이상 실패 = 실패)과 저장된 판정 일치 (18개)", 0, len(verdict_mismatch),
      len(verdict_mismatch) == 0)
check("ladder_not_failed", "실패하지 않은 시뮬레이터 5개 (§9.5 표 'did not fail')", 5, len(did_not), len(did_not) == 5, ", ".join(did_not))
check("ladder_failed_count", "실패 시뮬레이터 13개 (§9.5 '13 of the 18 failed')", 13, n_tests - len(did_not), n_tests - len(did_not) == 13)

# 학습률별 판정 (§9.5 '원래 학습률 8개 실패', '1/10 학습률 4개 실패' 등)
def run_group(name):
    return [t for t in tests if t["run"] == name]

orig_lr = [t for t in tests if t["run"].startswith(("W_", "M_", "Wtf_")) and not t["run"].startswith(("Wg_", "Mg_"))]
orig_ft_20_80 = [t for t in orig_lr if t["steps"] in (20, 80)]
check("orig_lr_20_80_failed", "원래 학습률, 20·80 스텝 미세조정 8개 모두 실패 (§9.5)", 8,
      sum(1 for t in orig_ft_20_80 if t["verdict"] == "failed"), len(orig_ft_20_80) == 8 and all(t["verdict"] == "failed" for t in orig_ft_20_80))
five = [t for t in orig_lr if t["steps"] == 5]
check("orig_lr_5_ok", "원래 학습률, 5 스텝 4개 중 3개 실패 안 함 (§9.5)", 3,
      sum(1 for t in five if t["verdict"] == "did not fail"), len(five) == 4 and sum(1 for t in five if t["verdict"] == "did not fail") == 3)
tenth = [t for t in tests if t["run"].startswith(("Wg_", "Mg_"))]
t80 = [t for t in tenth if t["steps"] == 80]
t160 = [t for t in tenth if t["steps"] in (160, 320)]
check("tenth_lr_80_ok", "1/10 학습률, 80 스텝 2개 모두 실패 안 함 (§9.5)", 2,
      sum(1 for t in t80 if t["verdict"] == "did not fail"), len(t80) == 2 and all(t["verdict"] == "did not fail" for t in t80))
check("tenth_lr_160_320_failed", "1/10 학습률, 160·320 스텝 4개 모두 실패 (§9.5)", 4,
      sum(1 for t in t160 if t["verdict"] == "failed"), len(t160) == 4 and all(t["verdict"] == "failed" for t in t160))

# 상위 3개 시뮬레이터 평균 (§9.5 '0.60')
rel_stored = stats["secondary"]["best_three"]
best3_stored = rel_stored["best_three_mean"]
check("best_three_mean", "상위 3개 평균 0.60 (§9.5)", 0.60, round(best3_stored, 3), abs(best3_stored - 0.60) < 0.005,
      "저장값 기준. 저장된 best_three 목록은 " + ", ".join(rel_stored["best_three"]))

for key, label, paper in [("E1", "1-step 오차가 고른 3개와 최상위 평균의 차이", 0.53),
                          ("E8", "8-step 오차가 고른 3개와 최상위 평균의 차이", 0.54)]:
    sf = rel_stored["checks"][key]["shortfall"]
    check("shortfall_" + key, label + f" (§9.5 '{paper}')", paper, round(sf, 3), abs(sf - paper) < 0.006)

# ---------- 6. 등록 통계 (§9.5 표)와 저장 파일 비교 ----------
prem = list(stats["premise"].values())[0]
check("premise_rho", "전제: 100스텝 오차 vs 걷기 순위 Spearman +0.16 (§9.5)", 0.16, round(prem["rho"], 2), abs(prem["rho"] - 0.16) < 0.006)
r1, r2 = list(stats["primary"].values())[:2]
check("primary1_delta", "주 검정 1 (성장률 - 1스텝 오차) +0.49 (§9.5, 97.5% [-0.42, +1.32])", 0.49, round(r1["delta"], 2),
      abs(r1["delta"] - 0.49) < 0.006 and round(r1["ci97.5"][0], 2) == -0.42 and round(r1["ci97.5"][1], 2) == 1.32)
check("primary2_delta", "주 검정 2 (잔차 - 8스텝 오차) +0.75 (§9.5, 97.5% [-0.17, +1.42])", 0.75, round(r2["delta"], 2),
      abs(r2["delta"] - 0.75) < 0.006 and round(r2["ci97.5"][0], 2) == -0.17 and round(r2["ci97.5"][1], 2) == 1.42)
wt = stats["secondary"]["walk_test"]["tracking"]["minus_E8"]
check("walk_tracking", "보조: 걷기 시험 추적 보상 - 8스텝 +1.22 [+0.71, +1.58] (§9.5)", 1.22, round(wt["delta"], 2),
      abs(wt["delta"] - 1.22) < 0.006 and round(wt["ci95"][0], 2) == 0.71 and round(wt["ci95"][1], 2) == 1.58)

# ---------- 7. 본문 내부 불일치 (텍스트 vs 표) ----------
check("six_million_falls", "§9.4 본문: 600만 전이 시뮬레이터 2차 시험 낙상률 15.5%, 두 배 학습 26.6%",
      "15.5% / 26.6%", "표 9.4: 0.23 / 0.39", False,
      "같은 시뮬레이터 두 번째 시험의 낙상률이 본문(26.6%)과 표(0.39)에서 다름. §9.6은 39%를 씀")
check("slowest_form", "§8.4 본문: 가장 느린 형태 '8비트 앙상블' 4 ms 미만 vs 표 8.4 (8비트 앙상블 행 없음)",
      "< 4 ms (8-bit ensemble)", "표의 가장 느린 행 = 32비트 앙상블 1,816 µs", False,
      "표에 8비트 앙상블 측정이 없음")
check("factor_68", "§6.3 본문 '68% 덮기 위한 배수 6.6 / 31 / 42' vs 표 6.3 평균 오차/평균 SD 8.3 / 33.4 / 34.4",
      "6.6, 31, 42", "8.3, 33.4, 34.4 (다른 지표)", False,
      "두 지표가 같은 말로 불림. 본문 배수가 어떤 표에도 없음")

# ---------- 8. 등록 시각: 저장소의 timeline.json (부록 B와 대조) ----------
check("timeline_first_prereg", "첫 사전등록 커밋 시각 2026-11-01 08:17 UTC (부록 B '1 Nov')", "2026-11-01",
      timeline["prereg_commit"]["time"][:10], timeline["prereg_commit"]["time"][:10] == "2026-11-01")

OUT.write_text(json.dumps({"checks": checks}, ensure_ascii=False, indent=2), encoding="utf-8")
n_ok = sum(c["ok"] for c in checks)
for c in checks:
    print(("OK  " if c["ok"] else "DIFF"), c["key"], "|", c["what"], "| paper:", c["paper"], "| recomputed:", c["recomputed"])
print(f"{n_ok}/{len(checks)} checks agree with the paper text")
