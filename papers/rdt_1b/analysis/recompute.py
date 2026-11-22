"""RDT-1B 논문 표 재계산과 검정 (CPU, 수 초). 학습과 재현은 하지 않는다.

실행: python3 analysis/recompute.py   → analysis/recompute_results.json
입력: paper/paper.txt (pdftotext -layout), repo/configs/pretrain_sample_weights.json (있으면 대조)

1. 표 3 셀별 성공 수 (시도 수 8 또는 25)와 Wilson 95% 구간, 주요 대비의 Fisher 정확 검정
2. 표 2 절제 (시도 8)와 대비 검정
3. 그림 1 종합 점수를 부록 H 계산 규칙으로 재계산하고 셀 단위 부트스트랩 구간
4. 일관성 검사: 정수 성공 수, 로봇 둘 열 중복, 추론 주파수, 사전학습 가중치(표 5 대 공식 설정)
"""
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact, norm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "recompute_results.json"
rng = np.random.default_rng(0)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def fisher(k1, n1, k2, n2):
    return float(fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def pct_to_count(pct, n):
    k = round(pct * n / 100)
    assert abs(k * 100 / n - pct) < 1e-6, (pct, n)  # 정수 성공 수인지 확인
    return k


# ---------- 1. 표 3 (논문 p.10). 값은 PDF 페이지를 눈으로 확인한 값 ----------
# 셀: (과제, 하위 열) -> {시스템: 백분율}. 시도 수 n은 셀 옆에 적는다.
T3 = {
    # Wash Cup Total 열: seen cup 1, unseen cup 1, unseen cup 2 (n=8 each)
    "wash_seen": {"n": 8, "ACT": 0, "OpenVLA": 0, "Octo": 0, "RDT_scratch": 0, "RDT": 50.0},
    "wash_unseen1": {"n": 8, "ACT": 0, "OpenVLA": 0, "Octo": 0, "RDT_scratch": 0, "RDT": 75.0},
    "wash_unseen2": {"n": 8, "ACT": 0, "OpenVLA": 0, "Octo": 0, "RDT_scratch": 0, "RDT": 50.0},
    # Pour Water unseen rooms (n=8 each), Total 열
    "room1_total": {"n": 8, "ACT": 0, "OpenVLA": 0, "Octo": 12.5, "RDT_scratch": 25.0, "RDT": 62.5},
    "room2_total": {"n": 8, "ACT": 37.5, "OpenVLA": 0, "Octo": 0, "RDT_scratch": 75.0, "RDT": 100.0},
    "room3_total": {"n": 8, "ACT": 12.5, "OpenVLA": 0, "Octo": 0, "RDT_scratch": 25.0, "RDT": 62.5},
    # Pour Water-L-1/3 (왼손), Pour Water-R-2/3 (오른손). 각 n=8
    "L_total": {"n": 8, "OpenVLA": 0, "Octo": 0, "RDT_scratch": 62.5, "RDT": 100.0},
    "R_total": {"n": 8, "OpenVLA": 0, "Octo": 0, "RDT_scratch": 25.0, "RDT": 87.5},
    "L_hand": {"n": 8, "OpenVLA": 50.0, "Octo": 0, "RDT_scratch": 100.0, "RDT": 100.0},
    "R_hand": {"n": 8, "OpenVLA": 0, "Octo": 0, "RDT_scratch": 75.0, "RDT": 87.5},
    "L_amount": {"n": 8, "OpenVLA": 0, "Octo": 0, "RDT_scratch": 62.5, "RDT": 100.0},
    "R_amount": {"n": 8, "OpenVLA": 0, "Octo": 0, "RDT_scratch": 12.5, "RDT": 75.0},
    # Handover 5-shot (n=25), Total 열
    "handover_total": {"n": 25, "ACT": 0, "OpenVLA": 0, "Octo": 0, "RDT_scratch": 16.0, "RDT": 40.0},
    # Fold Shorts 1-shot (n=25), Total 열
    "fold_total": {"n": 25, "ACT": 0, "OpenVLA": 0, "Octo": 4.0, "RDT_scratch": 40.0, "RDT": 68.0},
    # Robot Dog (n=25): Total(밀기) 열과 Walk Straight 열
    "dog_push_total": {"n": 25, "ACT": 32.0, "OpenVLA": 0, "Octo": 4.0, "RDT_scratch": 64.0, "RDT": 76.0},
    "dog_walk": {"n": 25, "ACT": 32.0, "OpenVLA": 0, "Octo": 0, "RDT_scratch": 32.0, "RDT": 48.0},
}

# 과제별 점수 규칙 (부록 H, 그림 1 설명): 과제 안에서 Total과 그 오른쪽 열을 평균
TASKS = {
    "Unseen Object (Wash Cup)": ("obj", ["wash_seen", "wash_unseen1", "wash_unseen2"]),
    "Unseen Scene (Pour Water)": ("scene", ["room1_total", "room2_total", "room3_total"]),
    # 지시 따르기: 왼손과 오른손 과제를 각각 (Total, Correct Hand, Correct Amount)의 평균으로 만든 뒤 평균
    "Instruction L (1/3)": ("instr", ["L_total", "L_hand", "L_amount"]),
    "Instruction R (2/3)": ("instr", ["R_total", "R_hand", "R_amount"]),
    "Handover (5-shot)": ("fewshot", ["handover_total"]),
    "Fold Shorts (1-shot)": ("fewshot", ["fold_total"]),
    "Robot Dog (Dexterity)": ("dex", ["dog_push_total", "dog_walk"]),
}
SYSTEMS = ["ACT", "OpenVLA", "Octo", "RDT_scratch", "RDT"]


def task_score(system, cells):
    vals = [T3[c][system] for c in cells]
    return float(np.mean(vals))


def overall(system, perturb=None, dims_ignore=()):
    """그림 1 방식: 과제 -> 차원 평균 -> 전체 평균. ACT는 지시 따르기 차원이 없어 제외한다."""
    dim_scores = {}
    for name, (dim, cells) in TASKS.items():
        if dim in dims_ignore:
            continue
        if perturb is None:
            s = task_score(system, cells)
        else:
            s = float(np.mean([perturb[c] for c in cells]))
        dim_scores.setdefault(dim, []).append(s)
    dims = {d: float(np.mean(v)) for d, v in dim_scores.items()}
    return float(np.mean(list(dims.values()))), dims


# ---------- 3. 종합 점수 재계산과 셀 단위 부트스트랩 ----------
overall_res = {}
for sys_ in SYSTEMS:
    ignore = ("instr",) if sys_ == "ACT" else ()
    tot, dims = overall(sys_, dims_ignore=ignore)
    overall_res[sys_] = {"overall": round(tot, 2), "dims": {k: round(v, 2) for k, v in dims.items()},
                         "n_dims": len(dims)}

# 셀 단위 부트스트랩: 각 셀의 성공 수를 Binomial(n, p̂)로 다시 뽑아 점수를 다시 계산
B = 10000
boot = {s: [] for s in SYSTEMS}
for _ in range(B):
    for sys_ in SYSTEMS:
        ignore = ("instr",) if sys_ == "ACT" else ()
        pert = {}
        for c, row in T3.items():
            if sys_ not in row:
                continue
            p = row[sys_] / 100
            pert[c] = rng.binomial(row["n"], p) / row["n"] * 100
        tot, _ = overall(sys_, perturb=pert, dims_ignore=ignore)
        boot[sys_].append(tot)
ci = {s: [round(float(np.percentile(boot[s], 2.5)), 1), round(float(np.percentile(boot[s], 97.5)), 1)]
      for s in SYSTEMS}
gap = np.array(boot["RDT"]) - np.array(boot["ACT"])
gap_ci = [round(float(np.percentile(gap, 2.5)), 1), round(float(np.percentile(gap, 97.5)), 1)]
gap_s = np.array(boot["RDT"]) - np.array(boot["RDT_scratch"])
gap_scratch_ci = [round(float(np.percentile(gap_s, 2.5)), 1), round(float(np.percentile(gap_s, 97.5)), 1)]

# 논문 본문의 "56%"는 68.2 - 12.2 (%p, 대 ACT)로 맞는지, 그리고 계산 규칙에 따른 대안 값
ract4 = overall_res["ACT"]["overall"]
rdt4 = overall("RDT", dims_ignore=("instr",))[0]  # RDT도 지시 차원을 뺀 4차원 평균
dim_vals = {}
for name, (dim, cells) in TASKS.items():
    v = 0.0 if dim == "instr" else task_score("ACT", cells)
    dim_vals.setdefault(dim, []).append(v)
act5 = float(np.mean([np.mean(v) for v in dim_vals.values()]))
gap_alt = {
    "gap_as_paper_5dim_vs_ACT_4dim": round(overall_res["RDT"]["overall"] - ract4, 1),
    "gap_4dim_both": round(rdt4 - ract4, 1),
    "gap_5dim_ACT_instr_zero": round(overall_res["RDT"]["overall"] - act5, 1),
}

# ---------- 2. 검정 ----------
def pooled(cells, system):
    k = sum(round(T3[c][system] * T3[c]["n"] / 100) for c in cells)
    n = sum(T3[c]["n"] for c in cells)
    return k, n


contrasts = [
    ("Handover 전체 (RDT 대 RDT scratch)", ["handover_total"], "RDT", "RDT_scratch"),
    ("Handover 전체 (RDT 대 ACT)", ["handover_total"], "RDT", "ACT"),
    ("Fold Shorts (RDT 대 RDT scratch)", ["fold_total"], "RDT", "RDT_scratch"),
    ("Robot Dog 밀기 (RDT 대 ACT)", ["dog_push_total"], "RDT", "ACT"),
    ("Robot Dog 밀기 (RDT 대 RDT scratch)", ["dog_push_total"], "RDT", "RDT_scratch"),
    ("Robot Dog 직진 (RDT 대 RDT scratch)", ["dog_walk"], "RDT", "RDT_scratch"),
    ("Robot Dog 직진 (RDT 대 ACT)", ["dog_walk"], "RDT", "ACT"),
    ("Wash Cup 세 컵 합산 (RDT 대 RDT scratch)", ["wash_seen", "wash_unseen1", "wash_unseen2"], "RDT", "RDT_scratch"),
    ("Pour Water 세 방 합산 (RDT 대 RDT scratch)", ["room1_total", "room2_total", "room3_total"], "RDT", "RDT_scratch"),
    ("Pour Water 세 방 합산 (RDT 대 ACT)", ["room1_total", "room2_total", "room3_total"], "RDT", "ACT"),
    ("지시 L 정확한 양 (RDT 대 RDT scratch)", ["L_amount"], "RDT", "RDT_scratch"),
    ("지시 R 정확한 양 (RDT 대 RDT scratch)", ["R_amount"], "RDT", "RDT_scratch"),
]
fisher_res = []
for label, cells, a, b in contrasts:
    ka, na = pooled(cells, a)
    kb, nb = pooled(cells, b)
    fisher_res.append({"label": label, "a": f"{a} {ka}/{na}", "b": f"{b} {kb}/{nb}",
                       "a_pct": round(100 * ka / na, 1), "b_pct": round(100 * kb / nb, 1),
                       "wilson_a": wilson(ka, na), "wilson_b": wilson(kb, nb),
                       "p_fisher": round(fisher(ka, na, kb, nb), 4)})

# 표 2 절제 (시도 8). 열: 미지 물체 2 총합, 미지 방 3 총합, 지시 L 정확한 양
T2 = {
    "RDT (regress)": (12.5, 50.0, 12.5),
    "RDT (small)": (37.5, 62.5, 25.0),
    "RDT (scratch)": (0.0, 25.0, 62.5),
    "RDT (ours)": (50.0, 62.5, 100.0),
}
ablation_res = []
for j, col in enumerate(["unseen_object_cup2", "unseen_scene_room3", "instruction_L_amount"]):
    for variant, vals in T2.items():
        if variant == "RDT (ours)":
            continue
        k_o = round(T2["RDT (ours)"][j] * 8 / 100)
        k_v = round(vals[j] * 8 / 100)
        ablation_res.append({"column": col, "variant": variant,
                             "ours": f"{k_o}/8", "variant_count": f"{k_v}/8",
                             "p_fisher": round(fisher(k_o, 8, k_v, 8), 3)})

# 검정력: 두 비율 차이를 탐지하는 표본 크기 (정규 근사, 양측 0.05, 검정력 0.8)
def n_per_arm(p1, p2, alpha=0.05, power=0.8):
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    pbar = (p1 + p2) / 2
    num = (za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(num / (p1 - p2) ** 2)


power = {
    "25%p 차이 (50% 대 25%)": n_per_arm(0.5, 0.25),
    "25%p 차이 (40% 대 16%)": n_per_arm(0.40, 0.16),
    "12.5%p 차이 (62.5% 대 50%)": n_per_arm(0.625, 0.5),
}

# ---------- 4. 일관성 검사 ----------
checks = {}
# 셀 수가 정수인지 (위에서 pct_to_count가 확인)
for c, row in T3.items():
    for s in SYSTEMS:
        if s in row:
            pct_to_count(row[s], row["n"])
checks["integer_counts_all_cells"] = True
# 표 3과 표 2 사이: RDT 방 3 합계 = 표 2 미지 장면, RDT 컵 2 = 표 2 미지 물체, RDT scratch 대응
checks["table2_vs_table3_consistency"] = {
    "unseen_object_ours_T2_vs_T3": (T2["RDT (ours)"][0], T3["wash_unseen2"]["RDT"]),
    "unseen_scene_ours_T2_vs_T3": (T2["RDT (ours)"][1], T3["room3_total"]["RDT"]),
    "instr_L_amount_ours_T2_vs_T3": (T2["RDT (ours)"][2], T3["L_amount"]["RDT"]),
    "unseen_object_scratch_T2_vs_T3": (T2["RDT (scratch)"][0], T3["wash_unseen2"]["RDT_scratch"]),
    "unseen_scene_scratch_T2_vs_T3": (T2["RDT (scratch)"][1], T3["room3_total"]["RDT_scratch"]),
    "instr_L_amount_scratch_T2_vs_T3": (T2["RDT (scratch)"][2], T3["L_amount"]["RDT_scratch"]),
}
# 추론 주파수: 6 Hz 청크 대 381 Hz 행동 (청크 길이 64)
checks["inference_rate"] = {"chunk_hz_claim": 6, "action_hz_claim": 381, "chunk_len": 64,
                            "implied_chunk_hz": round(381 / 64, 2)}

# 사전학습 데이터 가중치: 논문 표 5(46행) 대 공식 저장소 설정 (있을 때만)
repo_w = ROOT / "repo" / "configs" / "pretrain_sample_weights.json"
if repo_w.exists():
    w = json.loads(repo_w.read_text(encoding="utf-8"))
    tot = sum(w.values())
    code_pct = sorted(round(100 * v / tot, 2) for v in w.values())
    lines = (ROOT / "paper" / "paper.txt").read_text(encoding="utf-8").splitlines()
    paper_pct = []
    for line in lines:
        m = re.match(r"^(.*\S)\s{2,}([0-9]+\.[0-9]{2})$", line.strip())
        if m and "(" in m.group(1) and len(paper_pct) < 46:
            paper_pct.append(float(m.group(2)))
    mism = sum(1 for a, b in zip(sorted(paper_pct), code_pct) if abs(a - b) > 0.011)
    checks["pretrain_weights_vs_table5"] = {"paper_rows": len(paper_pct), "code_rows": len(w),
                                            "sum_paper_pct": round(sum(paper_pct), 2),
                                            "sorted_mismatches_over_0.01": mism}
else:
    checks["pretrain_weights_vs_table5"] = "repo 없음 (건너뜀)"

result = {
    "table3_overall_recompute": overall_res,
    "overall_bootstrap_95ci": ci,
    "gap_rdt_minus_act_bootstrap_95ci": gap_ci,
    "gap_rdt_minus_scratch_bootstrap_95ci": gap_scratch_ci,
    "gap_definitions": gap_alt,
    "fisher_contrasts": fisher_res,
    "table2_ablation_vs_ours": ablation_res,
    "power_n_per_arm": power,
    "consistency": checks,
}
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    print(json.dumps(result, ensure_ascii=False, indent=2))
