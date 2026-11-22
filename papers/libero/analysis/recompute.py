"""LIBERO (arXiv:2306.03310 v2, NeurIPS 2023 D&B) 논문 표 수치를 다시 읽고 검정한다.

입력: paper/paper.txt (pdftotext -layout 결과). Table 1과 Table 8은 본문 텍스트에서 파싱하고,
      Table 3은 캡션 위 본문 값을 그대로 옮겨 적었다(출처 주석).
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)

가정 (논문에 없는 것은 표시한다)
- "±" 뒤 값은 표준오차(SE)로 본다. 논문은 "mean and standard error"라고 쓴다(Table 1 캡션).
- 시드는 3개다(부록 D: {100, 200, 300}). 따라서 SD = SE * sqrt(3), 자유도 2 ~ 4 범위의 Welch 검정이다.
- 소수 둘째 자리로 반올림된 SE가 0.00이면 실제 값은 [0, 0.005)다. 검정은 SE를 0.005 ± 0.005 범위에서 바꿔 p 범위를 보고한다.
- 논문은 Table 1·2에 유의성 표시(보라색)를 넣었지만 그림에서 읽을 수 없어, 텍스트 주장만 검정한다.
"""
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
PAPER_TXT = HERE.parent / "paper" / "paper.txt"
OUT = HERE / "recompute_results.json"
N_SEED = 3
T_CRIT_DF2 = stats.t.ppf(0.975, df=2)  # 4.303

RES = {}
PAIR = re.compile(r"(\d\.\d{2}) ± (\d\.\d{2})")
SUITES = {"LIBERO-LONG": "LONG", "LIBERO-SPATIAL": "SPATIAL", "LIBERO-OBJECT": "OBJECT", "LIBERO-GOAL": "GOAL"}


def norm(s):
    return s.replace(" ", "")


def arch_of(prefix):
    p = norm(prefix)
    for key, name in [("RESNET-RNN", "RNN"), ("RESNET-T", "T"), ("VIT-T", "ViT")]:
        if key in p:
            return name
    return None


def algo_of(prefix):
    p = norm(prefix)
    for key, name in [("SEQL", "SeqL"), ("PACKNET", "PackNet"), ("EWC", "EWC"), ("MTL", "MTL"), ("ER", "ER")]:
        if p.startswith(key) or (key in p and key != "ER"):
            return name
    return None


def parse_tables(lines):
    lines = [l.rstrip("\n") for l in lines]
    # ---- Table 1: 한 suite당 헤더 1개, 행에 ER(FWT,NBT,AUC)과 PackNet(FWT,NBT,AUC)
    t1_start = next(i for i, l in enumerate(lines) if re.match(r"^\s*ER\s+PACK N ET\s*$", l))
    t1_end = next(i for i, l in enumerate(lines) if "Table 1: Performance" in l)
    t1 = {}
    suite = None
    for l in lines[t1_start:t1_end]:
        m = re.search(r"LIBERO-(L ONG|S PATIAL|O BJECT|G OAL)", l)
        if m:
            suite = SUITES["LIBERO-" + norm(m.group(1))]
            continue
        pairs = PAIR.findall(l)
        arch = arch_of(l[: l.find(pairs[0][0])]) if pairs else None
        if len(pairs) == 6 and arch and suite:
            vals = [float(v) for v, _ in pairs]
            ses = [float(s) for _, s in pairs]
            for j, (algo, metric) in enumerate([(a, m_) for a in ["ER", "PackNet"] for m_ in ["FWT", "NBT", "AUC"]]):
                t1[(algo, arch, suite, metric)] = (vals[j], ses[j])
    # ---- Table 8: 블록마다 suite 두 개(좌·우), 행 6쌍 = 좌 3 + 우 3, MTL 행은 AUC 2쌍
    t8_start = next(i for i, l in enumerate(lines) if l.strip().startswith("Algo.") and "Policy Arch." in l)
    t8_end = next(i for i, l in enumerate(lines) if "Table 8: We present" in l)
    # 알고리즘 이름은 아키텍처 3행 묶음(RNN, T, ViT)의 가운데(T) 행에만 붙어 있다.
    # 그래서 RNN 행이 나올 때마다 새 묶음을 열고, 묶음 안에서 이름을 찾는다.
    groups = []
    left = right = None
    cur = None
    for l in lines[t8_start:t8_end]:
        names = re.findall(r"LIBERO-(L ONG|S PATIAL|O BJECT|G OAL)", l)
        if names:
            ss = [SUITES["LIBERO-" + norm(n)] for n in names]
            left, right = (ss[0], ss[1]) if len(ss) == 2 else (ss[0], None)
            continue
        pairs = PAIR.findall(l)
        if not pairs:
            continue
        prefix = l[: l.find(pairs[0][0])]
        arch = arch_of(prefix)
        if arch is None:
            continue
        if arch == "RNN":
            cur = {"left": left, "right": right, "algo": None, "rows": []}
            groups.append(cur)
        if cur is None:
            continue
        lab = algo_of(prefix)
        if lab:
            cur["algo"] = lab
        cur["rows"].append((arch, pairs))
    t8 = {}
    for g in groups:
        algo = g["algo"]
        for arch, pairs in g["rows"]:
            if algo == "MTL" or len(pairs) == 2:
                if g["left"]:
                    t8[("MTL", arch, g["left"], "AUC")] = (float(pairs[0][0]), float(pairs[0][1]))
                if g["right"] and len(pairs) == 2:
                    t8[("MTL", arch, g["right"], "AUC")] = (float(pairs[1][0]), float(pairs[1][1]))
                continue
            if len(pairs) == 6:
                for side_suite, chunk in [(g["left"], pairs[:3]), (g["right"], pairs[3:])]:
                    for metric, (v, s) in zip(["FWT", "NBT", "AUC"], chunk):
                        t8[(algo, arch, side_suite, metric)] = (float(v), float(s))
    return t1, t8


def parse_multitask(lines):
    """부록 E.3: 과제별 다중과제 성공률 (세 아키텍처)."""
    s = next(i for i, l in enumerate(lines) if l.strip().startswith("E.3") and "Multitask" in l)
    e = next(i for i, l in enumerate(lines) if l.strip().startswith("E.4") and "Attention" in l)
    rows = []
    for l in lines[s:e]:
        m = re.match(r"\s*((?:KITCHEN|LIVING ROOM|STUDY) SCENE\d+)\s+(.*?)\s{2,}([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", l)
        if m:
            rows.append((m.group(1), m.group(2).strip(), [float(m.group(i)) for i in (3, 4, 5)]))
    return rows


def welch(m1, se1, m2, se2):
    """평균 ± SE 두 개와 n = 3을 받아 Welch 검정을 한다 (논문: 등 표본 크기, 불등 분산 양측 t 검정)."""
    # SE가 0.00으로 반올림된 경우를 위해 아주 작은 하한을 둔다(p가 사실상 0 또는 1이 되는 것을 막는다).
    v1 = (max(se1, 1e-4) * math.sqrt(N_SEED)) ** 2
    v2 = (max(se2, 1e-4) * math.sqrt(N_SEED)) ** 2
    t = (m1 - m2) / math.sqrt(v1 / N_SEED + v2 / N_SEED)
    df = (v1 / N_SEED + v2 / N_SEED) ** 2 / ((v1 / N_SEED) ** 2 / (N_SEED - 1) + (v2 / N_SEED) ** 2 / (N_SEED - 1))
    p = 2 * stats.t.sf(abs(t), df)
    return float(t), float(p)


def welch_range(m1, se1, m2, se2):
    """반올림 SE 불확실성(±0.005)을 반영한 p 범위."""
    ps = []
    for s1 in {max(se1 - 0.005, 1e-4), se1, se1 + 0.005}:
        for s2 in {max(se2 - 0.005, 1e-4), se2, se2 + 0.005}:
            ps.append(welch(m1, s1, m2, s2)[1])
    return min(ps), max(ps)


def main():
    lines = PAPER_TXT.read_text(encoding="utf-8").split("\n")
    t1, t8 = parse_tables(lines)
    RES["parsed_counts"] = {"table1_cells": len(t1), "table8_cells": len(t8)}
    print(f"[parse] Table 1 셀 {len(t1)}개 (기대 72), Table 8 셀 {len(t8)}개 (기대 156 = 4 suite x 4 algo x 3 arch x 3 metric + MTL 12)")

    # 1) 표 1 ↔ 표 8 일치
    mismatch = []
    for key, val in t1.items():
        algo, arch, suite, metric = key
        if key not in t8:
            mismatch.append((key, "absent in table 8"))
        elif t8[key] != val:
            mismatch.append((key, val, t8[key]))
    RES["table1_vs_table8_mismatches"] = mismatch
    print(f"[check] 표 1 ↔ 표 8 불일치: {len(mismatch)}개 (대상 {len(t1)}개)")

    # 2) 표 2 (ResNet-T 행) ↔ 표 8: 본문 표 2 값을 옮겨 적어 대조
    t2_reported = {
        ("SeqL", "LONG"): (0.54, 0.63, 0.15), ("ER", "LONG"): (0.48, 0.32, 0.32), ("EWC", "LONG"): (0.13, 0.22, 0.02), ("PackNet", "LONG"): (0.22, 0.08, 0.25),
        ("SeqL", "SPATIAL"): (0.72, 0.81, 0.20), ("ER", "SPATIAL"): (0.65, 0.27, 0.56), ("EWC", "SPATIAL"): (0.23, 0.33, 0.06), ("PackNet", "SPATIAL"): (0.55, 0.07, 0.63),
        ("SeqL", "OBJECT"): (0.78, 0.76, 0.26), ("ER", "OBJECT"): (0.67, 0.43, 0.44), ("EWC", "OBJECT"): (0.56, 0.69, 0.16), ("PackNet", "OBJECT"): (0.60, 0.17, 0.60),
        ("SeqL", "GOAL"): (0.77, 0.82, 0.22), ("ER", "GOAL"): (0.64, 0.34, 0.49), ("EWC", "GOAL"): (0.32, 0.48, 0.06), ("PackNet", "GOAL"): (0.63, 0.06, 0.75),
    }
    t2_mis = []
    for (algo, suite), (f, n, a) in t2_reported.items():
        for metric, v in zip(["FWT", "NBT", "AUC"], (f, n, a)):
            got = t8[(algo, "T", suite, metric)][0]
            if abs(got - v) > 1e-9:
                t2_mis.append((algo, suite, metric, v, got))
    RES["table2_vs_table8_mismatches"] = t2_mis
    print(f"[check] 표 2 (ResNet-T) ↔ 표 8: 불일치 {len(t2_mis)}개 (대상 {len(t2_reported)*3}개)")

    # 3) 표 3 (언어 임베딩, 본문 값) ↔ 표 1 ER·ResNet-T·LONG
    t3 = {"BERT": (0.48, 0.02, 0.32, 0.04, 0.32, 0.01), "CLIP": (0.52, 0.00, 0.34, 0.01, 0.35, 0.01),
          "GPT-2": (0.46, 0.01, 0.34, 0.02, 0.30, 0.01), "Task-ID": (0.50, 0.01, 0.37, 0.01, 0.33, 0.01)}
    bert_t1 = t1[("ER", "T", "LONG", "FWT")], t1[("ER", "T", "LONG", "NBT")], t1[("ER", "T", "LONG", "AUC")]
    RES["table3_bert_vs_table1"] = bool(
        abs(bert_t1[0][0] - 0.48) < 1e-9 and abs(bert_t1[2][0] - 0.32) < 1e-9 and abs(bert_t1[1][0] - 0.32) < 1e-9)

    # 4) 부록 E.3 과제별 값: 0.05 배수 여부 (20회 롤아웃이면 성공률은 k/20)
    mt = parse_multitask(lines)
    grid_bad = [r for r in mt for v in r[2] if abs(v * 20 - round(v * 20)) > 1e-9]
    per_arch = {a: [] for a in range(3)}
    for _, _, vals in mt:
        for i, v in enumerate(vals):
            per_arch[i].append(v)
    RES["multitask_e3"] = {
        "n_tasks_rows": len(mt),
        "not_multiple_of_0.05": len(grid_bad),
        "mean_success_by_arch": {"RNN": round(float(np.mean(per_arch[0])), 3),
                                  "ResNet-T": round(float(np.mean(per_arch[1])), 3),
                                  "ViT-T": round(float(np.mean(per_arch[2])), 3)},
    }
    print(f"[check] E.3 행 {len(mt)}개, 0.05(=1/20) 배수가 아닌 값 {len(grid_bad)}개")

    # 5) 본문 주장 검정 (Welch, n = 3, SE 기반)
    claims = []

    def add(label, a, b, metric_name):
        (m1, s1), (m2, s2) = a, b
        t, p = welch(m1, s1, m2, s2)
        plo, phi = welch_range(m1, s1, m2, s2)
        claims.append({"주장": label, "지표": metric_name, "평균차": round(m1 - m2, 3),
                       "t": round(t, 2), "p": round(p, 4), "p_범위": [round(plo, 4), round(phi, 4)],
                       "p<0.05": bool(p < 0.05)})

    # 5-1 Table 1, 본문 "ViT가 ER에서 OBJECT만 ResNet-T보다 좋다" 및 "ResNet-T가 나머지 세 suite에서 낫다"
    for suite in ["LONG", "SPATIAL", "OBJECT", "GOAL"]:
        add(f"ER: ViT-T vs ResNet-T ({suite})", t8[("ER", "ViT", suite, "AUC")], t8[("ER", "T", suite, "AUC")], "AUC")
    # 5-2 본문 "PackNet에서는 ResNet-T와 ViT-T 차이가 없고 LONG에서만 ViT가 크다"
    for suite in ["LONG", "SPATIAL", "OBJECT", "GOAL"]:
        add(f"PackNet: ViT-T vs ResNet-T ({suite})", t8[("PackNet", "ViT", suite, "AUC")], t8[("PackNet", "T", suite, "AUC")], "AUC")
    # 5-3 Table 2 "SeqL이 모든 suite에서 FWT가 가장 높다"
    for suite in ["LONG", "SPATIAL", "OBJECT", "GOAL"]:
        for other in ["ER", "EWC", "PackNet"]:
            add(f"FWT: SeqL vs {other} ({suite})", t8[("SeqL", "T", suite, "FWT")], t8[(other, "T", suite, "FWT")], "FWT")
    # 5-4 본문 "PackNet이 LIBERO-X에서 ER보다 낫고, LONG에서는 ER이 유의하게 낫다"
    for suite in ["LONG", "SPATIAL", "OBJECT", "GOAL"]:
        add(f"AUC: PackNet vs ER ({suite})", t8[("PackNet", "T", suite, "AUC")], t8[("ER", "T", suite, "AUC")], "AUC")
    # 5-5 본문 "EWC가 SeqL보다 나쁘다"
    for suite in ["LONG", "SPATIAL", "OBJECT", "GOAL"]:
        add(f"AUC: EWC vs SeqL ({suite})", t8[("EWC", "T", suite, "AUC")], t8[("SeqL", "T", suite, "AUC")], "AUC")
    # 5-6 표 3 "언어 임베딩 사이에 유의한 차이 없음" (BERT 기준)
    for emb in ["CLIP", "GPT-2", "Task-ID"]:
        for metric, idx in [("FWT", 0), ("NBT", 2), ("AUC", 4)]:
            se_b = t3["BERT"][idx + 1]
            se_e = t3[emb][idx + 1]
            add(f"표 3: {emb} vs BERT", (t3[emb][idx], se_e), (t3["BERT"][idx], se_b), metric)

    RES["claim_tests"] = claims
    n_sig = sum(1 for c in claims if c["p<0.05"])
    print(f"[test] 주장 검정 {len(claims)}개, 유의(p<0.05) {n_sig}개")
    for c in claims:
        print(f"  {c['주장']:<42} {c['지표']:<4} 차 {c['평균차']:+.3f}  p={c['p']:.4f}  범위 [{c['p_범위'][0]:.4f}, {c['p_범위'][1]:.4f}]  {'유의' if c['p<0.05'] else '비유의'}")

    # 6) 신뢰구간: n = 3, 95% t 구간 = 평균 ± 4.303 * SE
    ci_examples = {}
    for key in [("ER", "T", "OBJECT", "AUC"), ("PackNet", "T", "GOAL", "AUC"), ("SeqL", "T", "LONG", "FWT")]:
        m, s = t8[key]
        ci_examples["/".join(key)] = [round(m - T_CRIT_DF2 * s, 3), round(m + T_CRIT_DF2 * s, 3)]
    RES["t_ci_n3"] = ci_examples

    # 7) 20회 롤아웃의 이항 불확실성 (코드의 Wald 식 1.96*sqrt(p(1-p)/20))
    binom = {}
    for p in [0.5, 0.8, 0.9]:
        binom[str(p)] = round(1.96 * math.sqrt(p * (1 - p) / 20), 3)
    RES["binomial_halfwidth_n20"] = binom

    OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {OUT}")


if __name__ == "__main__":
    main()
