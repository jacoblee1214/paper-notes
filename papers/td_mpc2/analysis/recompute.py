"""TD-MPC2 (arXiv:2310.16828 v2, ICLR 2024) 논문 수치와 공개 결과 파일을 다시 계산하고 통계 검정을 한다.

입력 두 가지
1. 공식 저장소의 학습 곡선 파일 `repo/results/{tdmpc2,tdmpc,sac,dreamerv3,tdmpc2-pixels}/*.csv`
   (환경 단계별 평가 값, 시드별 3개). 정적 읽기만 한다. 코드는 실행하지 않는다.
2. 논문 본문, 표, 부록의 값을 아래에 옮겨 적었다. 출처는 주석에 쓴다.
   논문 텍스트는 `paper/paper.pdf`를 `pdftotext`로 다시 읽어 표 2–5의 과제 목록만 확인한다.

출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 -I analysis/recompute.py   (numpy, scipy, 수 초)

가정(논문에 없는 것은 표시한다)
- 신뢰구간은 시드 3개의 t 분포 구간(t = 4.303, df = 2)으로 계산한다. 논문은 "95% CI"만 쓰고 방법을 밝히지 않는다.
- 과제 단계의 최종 값은 네 알고리즘이 모두 기록한 공통 마지막 단계(100k 단위 내림)에서 잡는다.
  기록이 없으면 그 이전의 가장 가까운 값을 쓴다.
- 성공률 과제(Meta-World, ManiSkill2, MyoSuite)는 0–1 값을 100배 한다. DMControl은 에피소드 반환(0–1000)이다.
- 논문 Figure 4는 과제 평균 곡선이다. 여기서 만든 도메인 평균은 그 그림을 다시 그리는 근사이지 그림 값의 복제가 아니다.
"""
import csv
import collections
import json
import math
import subprocess
from pathlib import Path

import warnings

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore", message="Sample size too small")

HERE = Path(__file__).resolve().parent
TD = HERE.parent
RES = TD / "repo" / "results"
PAPER_PDF = TD / "paper" / "paper.pdf"
OUT = HERE / "recompute_results.json"

ALGS = ["tdmpc2", "tdmpc", "sac", "dreamerv3"]
BASE = ["tdmpc", "sac", "dreamerv3"]
DOMAINS = ["DMControl", "Meta-World", "ManiSkill2", "MyoSuite"]
T_CRIT = 4.302652729911275  # t(0.975, df=2), 시드 3개
GRID = 100_000
RES_OUT = {}


# ---------- 데이터 읽기 ----------

def domain_of(task):
    if task.startswith("mw-"):
        return "Meta-World"
    if task.startswith("myo-"):
        return "MyoSuite"
    if task in ("pick-cube", "pick-ycb", "stack-cube", "lift-cube", "turn-faucet"):
        return "ManiSkill2"
    return "DMControl"


def scale_of(task):
    return 1.0 if domain_of(task) == "DMControl" else 100.0


def load(alg, task):
    """시드별 {step: value}. 성공률은 100배 한다."""
    path = RES / alg / f"{task}.csv"
    out = collections.defaultdict(dict)
    with open(path) as f:
        rd = csv.DictReader(f)
        vcol = [c for c in rd.fieldnames if c not in ("step", "seed")][0]
        for r in rd:
            v = float(r[vcol]) * (scale_of(task) if vcol == "success" else 1.0)
            out[int(r["seed"])][int(round(float(r["step"])))] = v
    return dict(out)


def value_at(seed_d, s):
    """단계 s 이하에서 가장 가까운 기록 값."""
    ks = [k for k in seed_d if k <= s]
    if not ks:
        return None
    return seed_d[max(ks)]


def tasks_list():
    return sorted(p.stem for p in (RES / "tdmpc2").glob("*.csv"))


def common_end(task):
    ends = []
    for a in ALGS:
        d = load(a, task)
        m = max(max(v.keys()) for v in d.values())
        ends.append((m // GRID) * GRID)
    return min(ends)


def final_stats(alg, task, end):
    d = load(alg, task)
    vals = [value_at(d[s], end) for s in sorted(d)]
    vals = np.array(vals, dtype=float)
    return vals


def ci_t(vals):
    m = vals.mean()
    se = vals.std(ddof=1) / math.sqrt(len(vals))
    return float(m), float(m - T_CRIT * se), float(m + T_CRIT * se)


# ---------- 0. 데이터 무결성과 논문 과제 목록 ----------

def check_integrity():
    tasks = tasks_list()
    for a in ALGS:
        names = sorted(p.stem for p in (RES / a).glob("*.csv"))
        assert names == tasks, a
    dom_count = collections.Counter(domain_of(t) for t in tasks)
    seeds = collections.Counter()
    for a in ALGS:
        for t in tasks:
            seeds[len(load(a, t))] += 1
    budgets = collections.Counter()
    for t in tasks:
        budgets[(domain_of(t), common_end(t))] += 1
    # 논문 표 2–5의 과제 이름을 PDF 텍스트에서 읽어 저장소 파일 이름과 맞춘다.
    txt = subprocess.run(["pdftotext", str(PAPER_PDF), "-"], capture_output=True, text=True).stdout
    lines = [ln.strip() for ln in txt.splitlines()]
    start = next(i for i, ln in enumerate(lines) if ln.startswith("Table 2. DMControl"))
    end = next(i for i, ln in enumerate(lines) if ln.startswith("Table 3. Meta-World"))
    names_dmc = []
    cap = False
    for ln in lines[start:end]:
        if ln == "Task":
            cap = True
            continue
        if cap and ln and not ln.isdigit() and ln not in ("Published as a conference paper at ICLR 2024",):
            if ln.startswith("Figure") or ln.startswith("Published"):
                break
            names_dmc.append(ln)
    names_dmc = [n for n in names_dmc if n[0].isupper()]
    file_dmc = [t for t in tasks if domain_of(t) == "DMControl"]
    slug = lambda s: s.lower().replace(" ", "-")
    paper_slugs = set(slug(n) for n in names_dmc)
    res = {
        "tasks_total": len(tasks),
        "domain_counts": dict(dom_count),
        "paper_text_dmc_entries": len(names_dmc),
        "files_dmc": len(file_dmc),
        "dmc_files_missing_from_table2_text": sorted(set(file_dmc) - paper_slugs),
        "table2_text_tokens_not_in_files": len(paper_slugs - set(file_dmc)),
        "seeds_per_file_histogram": dict(seeds),
        "budgets_domain_end_step_counts": {f"{d}|{e}": c for (d, e), c in sorted(budgets.items())},
        "pixel_files": len(list((RES / "tdmpc2-pixels").glob("*.csv"))),
    }
    RES_OUT["0_integrity"] = res
    print("0. 무결성:", dom_count, "시드 분포", dict(seeds))
    print("   Table 2 항목 수:", len(names_dmc), "저장소 DMC 파일:", len(file_dmc))
    print("   저장소 DMC 파일 중 Table 2 텍스트에 없는 것:", res["dmc_files_missing_from_table2_text"],
          "| Table 2 텍스트의 비과제 토큰 수:", res["table2_text_tokens_not_in_files"])
    print("   단계 예산(도메인, 끝 단계):", res["budgets_domain_end_step_counts"])


# ---------- 1. 도메인 평균과 최종 값 ----------

def per_task_table():
    tasks = tasks_list()
    table = {}
    for t in tasks:
        end = common_end(t)
        row = {"domain": domain_of(t), "end": end}
        for a in ALGS:
            v = final_stats(a, t, end)
            m, lo, hi = ci_t(v)
            d = load(a, t)
            # 데이터 효율: 1M 단계 값, 면적(0부터 끝까지 100k 간격 평균)
            v1m = np.array([value_at(d[s], 1_000_000) for s in sorted(d)], dtype=float)
            grid = np.arange(GRID, end + 1, GRID)
            area = np.array([[value_at(d[s], g) for g in grid] for s in sorted(d)], dtype=float).mean(axis=1)
            row[a] = {"final_mean": m, "ci_lo": lo, "ci_hi": hi, "seeds": v.tolist(),
                      "at_1M_mean": float(v1m.mean()), "auc_mean": float(area.mean())}
        table[t] = row
    return table


def domain_summary(table):
    out = {}
    for dom in DOMAINS:
        ts = [t for t in table if table[t]["domain"] == dom]
        out[dom] = {"n_tasks": len(ts)}
        for a in ALGS:
            fm = np.array([table[t][a]["final_mean"] for t in ts])
            e1 = np.array([table[t][a]["at_1M_mean"] for t in ts])
            auc = np.array([table[t][a]["auc_mean"] for t in ts])
            out[dom][a] = {"final_task_mean": round(float(fm.mean()), 2),
                           "final_task_sd_across_tasks": round(float(fm.std(ddof=1)), 2),
                           "at_1M_task_mean": round(float(e1.mean()), 2),
                           "auc_task_mean": round(float(auc.mean()), 2)}
    all_ts = list(table)
    out["ALL"] = {a: {"final_task_mean": round(float(np.mean([table[t][a]["final_mean"] for t in all_ts])), 2)}
                  for a in ALGS}
    RES_OUT["1_domain_summary"] = out
    print("1. 도메인 평균(과제 평균의 과제 평균, 최종 단계):")
    for dom in DOMAINS:
        s = out[dom]
        print(f"   {dom:10s} n={s['n_tasks']:2d} " + "  ".join(f"{a}={s[a]['final_task_mean']:7.2f}" for a in ALGS))
    print("   1M 단계 값:")
    for dom in DOMAINS:
        s = out[dom]
        print(f"   {dom:10s} " + "  ".join(f"{a}={s[a]['at_1M_task_mean']:7.2f}" for a in ALGS))


# ---------- 2. 짝 비교 검정 (과제 단위) ----------

def paired_tests(table):
    res = {}
    for base in BASE:
        for scope in ["ALL"] + DOMAINS:
            ts = [t for t in table if scope == "ALL" or table[t]["domain"] == scope]
            d = np.array([table[t]["tdmpc2"]["final_mean"] - table[t][base]["final_mean"] for t in ts])
            wins = int((d > 0).sum())
            losses = int((d < 0).sum())
            ties = int((d == 0).sum())
            if len(d) > 1 and np.any(d != 0):
                w = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
                wp = float(w.pvalue)
            else:
                wp = float("nan")
            sign_p = float(stats.binomtest(wins, wins + losses, 0.5).pvalue) if wins + losses > 0 else float("nan")
            rng = np.random.default_rng(0)
            boots = np.array([rng.choice(d, size=len(d), replace=True).mean() for _ in range(10000)])
            res[f"{base}|{scope}"] = {
                "n": len(ts), "mean_diff": round(float(d.mean()), 2),
                "boot95": [round(float(np.percentile(boots, 2.5)), 2), round(float(np.percentile(boots, 97.5)), 2)],
                "wins": wins, "losses": losses, "ties": ties,
                "wilcoxon_p": round(wp, 5), "sign_test_p": round(sign_p, 5),
            }
    RES_OUT["2_paired_tests_task_level"] = res
    print("2. TD-MPC2 - 기준선 (과제 단위 짝 비교, 최종 단계):")
    for k, v in res.items():
        if k.endswith("|ALL") or k.split("|")[1] in DOMAINS:
            print(f"   {k:22s} n={v['n']:3d} 평균차={v['mean_diff']:7.2f} 부트스트랩95={v['boot95']} "
                  f"승/패/무={v['wins']}/{v['losses']}/{v['ties']} Wilcoxon p={v['wilcoxon_p']} 부호 p={v['sign_test_p']}")


# ---------- 3. 붕괴(발산) 검사: 논문 주장 확인 ----------

def collapse_events(alg, tasks):
    """피크 대비 최종 값이 절반 이하로 떨어진 시드 수. 피크가 너무 낮으면 제외한다."""
    events = {}
    for t in tasks:
        d = load(alg, t)
        end = common_end(t)
        grid = np.arange(GRID, end + 1, GRID)
        sc = scale_of(t)
        thr = 100.0 if domain_of(t) == "DMControl" else 20.0
        n = 0
        for s in sorted(d):
            series = np.array([value_at(d[s], g) for g in grid], dtype=float)
            peak = series.max()
            final = series[-1]
            if peak >= thr and final < 0.5 * peak:
                n += 1
        if n:
            events[t] = n
    return events


def check_claims(table):
    tasks = tasks_list()
    out = {}
    for a in ALGS:
        ev = collapse_events(a, tasks)
        out[a] = {"tasks_with_collapse_seed": len(ev), "collapse_seed_events": int(sum(ev.values())),
                  "examples": dict(list(sorted(ev.items()))[:12])}
    RES_OUT["3_collapse_check"] = out
    print("3. 붕괴 검사 (피크 대비 최종 < 50%, 시드 단위):")
    for a in ALGS:
        print(f"   {a:10s} 과제 {out[a]['tasks_with_collapse_seed']}개, 시드 사건 {out[a]['collapse_seed_events']}개")
    print("   tdmpc 예:", out["tdmpc"]["examples"])

    # 논문 본문 주장과 값
    dog = {a: {t: table[t][a]["final_mean"] for t in ["dog-run", "dog-stand", "dog-trot", "dog-walk"]} for a in ALGS}
    hum = {a: {t: table[t][a]["at_1M_mean"] for t in ["humanoid-run", "humanoid-stand", "humanoid-walk"]} for a in ALGS}
    RES_OUT["4_locomotion_snapshot"] = {"dog_final_mean": dog, "humanoid_at_1M": hum}
    hum_f = {a: {t: round(table[t][a]["final_mean"], 1) for t in ["humanoid-run", "humanoid-stand", "humanoid-walk"]} for a in ALGS}
    RES_OUT["4_locomotion_snapshot"]["humanoid_final"] = hum_f
    print("   Humanoid 최종(14M) 과제별:", hum_f)
    print("4. Dog 최종 평균:", {a: round(float(np.mean(list(dog[a].values()))), 1) for a in ALGS})
    print("   Humanoid 1M 평균:", {a: round(float(np.mean(list(hum[a].values()))), 1) for a in ALGS})
    # DreamerV3 Dog-Run 시드별 최종
    print("   dreamerv3 dog-run 시드별 최종:", [round(x, 1) for x in table["dog-run"]["dreamerv3"]["seeds"]])
    print("   tdmpc walker-stand 시드별 최종:", [round(x, 1) for x in table["walker-stand"]["tdmpc"]["seeds"]])


# ---------- 4. 픽셀 결과 (논문 Figure 10: 10개 과제) ----------

def pixel_check():
    files = sorted(p.stem for p in (RES / "tdmpc2-pixels").glob("*.csv"))
    res = {}
    for t in files:
        d = load("tdmpc2-pixels", t) if (RES / "tdmpc2-pixels" / f"{t}.csv").exists() else {}
        end = max(max(v.keys()) for v in d.values())
        v = np.array([value_at(d[s], end) for s in sorted(d)], dtype=float)
        res[t] = {"end": end, "seeds": len(d), "final_mean": round(float(v.mean()), 1)}
    RES_OUT["5_pixel_files"] = {"n_files": len(files), "tasks": res}
    print("5. 픽셀 파일:", len(files), "개 (논문 본문과 Figure 10은 10개 과제)")
    print("   추가 파일:", sorted(set(files) - {"acrobot-swingup", "cheetah-run", "finger-spin", "finger-turn-easy",
                                              "finger-turn-hard", "quadruped-walk", "reacher-easy", "reacher-hard",
                                              "walker-run", "walker-walk"}))


# ---------- 5. 논문 본문, 표, 부록 값의 일관성 ----------

def paper_numbers():
    # Table 1: 학습 비용 (GPU일, RTX 3090), 점수 (80과제 데이터셋). Figure 7 왼쪽과 같은 값.
    params = np.array([1, 5, 19, 48, 317], dtype=float)
    gpu_days = np.array([3.7, 4.2, 5.3, 12, 33], dtype=float)
    score = np.array([16.0, 49.5, 57.1, 68.0, 70.6])
    x = np.log10(params)
    b1, b0 = np.polyfit(x, score, 1)
    pred = b0 + b1 * x
    r2 = 1 - ((score - pred) ** 2).sum() / ((score - score.mean()) ** 2).sum()
    b1_no317, b0_no317 = np.polyfit(x[:4], score[:4], 1)
    pred_no317 = b0_no317 + b1_no317 * x[:4]
    r2_no317 = 1 - ((score[:4] - pred_no317) ** 2).sum() / ((score[:4] - score[:4].mean()) ** 2).sum()

    # 부록 H: 5M 구조의 파라미터 수를 세어 표시된 값과 맞춘다.
    def count(enc_dim, mlp, latent, n_enc, n_q, S=39, D=96, A=6, T=80, n_bins=101):
        def nl(i, o):  # NormedLinear: Linear + LayerNorm
            return i * o + o + 2 * o

        def lin(i, o):
            return i * o + o
        enc_dims = [S + D] + [enc_dim] * (n_enc - 1) + [latent]
        enc_dims = [S + D, enc_dim, latent] if n_enc == 2 else [S + D] + [enc_dim] * (n_enc - 2) + [enc_dim, latent]
        enc = sum(nl(enc_dims[i], enc_dims[i + 1]) for i in range(n_enc))
        dyn = nl(latent + D + A, mlp) + nl(mlp, mlp) + nl(mlp, latent)
        rew = nl(latent + D + A, mlp) + nl(mlp, mlp) + lin(mlp, n_bins)
        pi = nl(latent + D, mlp) + nl(mlp, mlp) + lin(mlp, 2 * A)
        q = n_q * (nl(latent + D + A, mlp) + nl(mlp, mlp) + lin(mlp, n_bins))
        task = T * D
        return {"encoder": enc, "dynamics": dyn, "reward": rew, "policy": pi, "Q": q, "task": task,
                "total": enc + dyn + rew + pi + q + task}

    c5 = count(256, 512, 512, 2, 5)
    configs = {"1M": (256, 384, 128, 2, 2), "5M": (256, 512, 512, 2, 5), "19M": (1024, 1024, 768, 3, 5),
               "48M": (1792, 1792, 768, 4, 5), "317M": (4096, 4096, 1376, 5, 8)}
    pc = {k: count(*v)["total"] for k, v in configs.items()}
    # 부록 H 표의 값: 인코더 167,936 / 동역학 843,264 / 보상 631,397 / 정책 582,668 / Q 3,156,985 / 과제 7,680 / 합 5,389,930
    paper_c5 = {"encoder": 167936, "dynamics": 843264, "reward": 631397, "policy": 582668,
                "Q": 3156985, "task": 7680, "total": 5389930}
    # Table 8 Q 학습률 0.99 등은 코드 대조 문서에서 다룬다.

    # 예: 미세조정 2배 (Figure 8): 47.0 대 24.0
    ft = 47.0 / 24.0
    # Table 10: 테스트 시 정규화 (19M, 80과제)
    t10 = {"no_reg": 56.54, "c=0.001": 58.14, "c=0.01": 62.01, "c=0.1": 44.13}
    # 같은 19M 80과제 기준 점수가 여러 곳에서 다르게 나온다.
    same_19m = {"Table 1, Figure 7 (80과제, 19M)": 57.1, "Table 10 (no reg)": 56.54,
                "Figure 9 (기본 설정, 19M 80과제)": 54.2, "Figure 18 (정규화 임베딩, 19M 80과제)": 54.2}
    RES_OUT["6_paper_numbers"] = {
        "loglinear_fit_all5": {"slope_per_decade": round(float(b1), 2), "intercept": round(float(b0), 2), "r2": round(float(r2), 4),
                               "pred_317M": round(float(b0 + b1 * np.log10(317)), 2), "obs_317M": 70.6},
        "loglinear_fit_1_to_48M": {"slope_per_decade": round(float(b1_no317), 2), "r2": round(float(r2_no317), 4),
                                   "pred_317M": round(float(b0_no317 + b1_no317 * np.log10(317)), 2)},
        "gpu_days_ratio_317_over_1M": round(float(33 / 3.7), 2),
        "params_ratio_317_over_1M": round(float(317 / 1), 1),
        "param_count_5M_recomputed": c5,
        "param_count_5M_paper": paper_c5,
        "param_count_recomputed_by_size": pc,
        "finetune_ratio_47_over_24": round(ft, 3),
        "test_time_reg_table10": t10,
        "test_time_reg_gain_c001_pts": round(t10["c=0.01"] - t10["no_reg"], 2),
        "test_time_reg_gain_c001_rel_pct": round(100 * (t10["c=0.01"] - t10["no_reg"]) / t10["no_reg"], 1),
        "same_19M_80task_values": same_19m,
    }
    print("6. 점수 대 log10(파라미터) 직선 적합 (5점): 기울기/10배=", round(b1, 2), "R²=", round(r2, 4),
          "| 317M 예측", round(b0 + b1 * np.log10(317), 1), "관측 70.6")
    print("   1M–48M 4점 적합: R²=", round(r2_no317, 4), " 317M 예측", round(b0_no317 + b1_no317 * np.log10(317), 1))
    print("   5M 파라미터 수 재계산:", c5["total"], "논문 5,389,930", "| 인코더", c5["encoder"], "(논문 167,936)")
    print("   크기별 재계산 총합:", {k: round(v / 1e6, 2) for k, v in pc.items()}, "M (표 9 명목 크기 1,5,19,48,317)")
    print("   미세조정 비율:", round(ft, 3), "| 테스트 시 정규화 c=0.01 이득:", round(t10["c=0.01"] - t10["no_reg"], 2),
          f"({round(100 * (t10['c=0.01'] - t10['no_reg']) / t10['no_reg'], 1)}%)")


def main():
    check_integrity()
    table = per_task_table()
    RES_OUT["per_task"] = {t: {"domain": v["domain"], "end": v["end"],
                               **{a: {"final_mean": round(v[a]["final_mean"], 2),
                                      "ci95": [round(v[a]["ci_lo"], 2), round(v[a]["ci_hi"], 2)],
                                      "at_1M_mean": round(v[a]["at_1M_mean"], 2),
                                      "seeds": [round(x, 2) for x in v[a]["seeds"]]} for a in ALGS}}
                           for t, v in table.items()}
    domain_summary(table)
    paired_tests(table)
    check_claims(table)
    pixel_check()
    paper_numbers()
    with open(OUT, "w") as f:
        json.dump(RES_OUT, f, ensure_ascii=False, indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
