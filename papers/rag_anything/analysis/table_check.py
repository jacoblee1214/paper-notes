"""RAG-Anything 표·그림 수치 재계산 (CPU). 실행: python3 analysis/table_check.py

- C1: Table 2·3·4의 Overall이 영역별 정확도를 문항 수(Table 5·6)로 가중 평균한 값과 맞는지
- C2: Table 2 유형별(Txt·Mm·Una) 정확도로 Overall을 재현하는 유형별 문항 수 추정
- C3: Overall 차이의 표본 오차 (Wilson 구간, 비대응 2표본 검정 근사)
- C4: Table 1 평균 쪽수 vs Table 5·6에서 다시 계산한 평균 쪽수
- C5: Fig. 2 (그림 픽셀 판독값)의 구간별 문항 수·정확도, 작은 구간의 정수 해석, 본문 수치와 대조
- C6: GPT-4o-mini 기준선이 볼 수 있는 쪽 비율 (최대 50쪽, Sec. 3.1)
- C7: 영역별 차이를 MMGraphRAG 대비 분해
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

# Table 5 (DocBench): 영역별 문서 수, 문항 수, 평균 쪽수
DOC_DOMAINS = ["Aca", "Fin", "Gov", "Law", "News"]
DOC_DOCS = [49, 40, 44, 46, 50]
DOC_Q = [303, 288, 148, 191, 172]
DOC_PAGES = [11, 192, 69, 58, 1]
# Table 6 (MMLongBench)
MML_DOMAINS = ["Res", "Tut", "Acad", "Guid", "Broch", "Admin", "Fin"]
MML_DOCS = [34, 17, 26, 22, 15, 10, 11]
MML_Q = [292, 138, 199, 155, 100, 81, 117]
MML_PAGES = [39, 58, 35, 78, 30, 17, 87]

# Table 2: 영역 5개, 유형 3개(Txt, Mm, Una), Overall
T2 = {
    "GPT-4o-mini": ([40.3, 46.9, 60.3, 59.2, 61.0], [61.0, 43.8, 49.6], 51.2),
    "LightRAG": ([53.8, 56.2, 59.5, 61.8, 65.7], [85.0, 59.7, 46.8], 58.4),
    "MMGraphRAG": ([64.3, 52.8, 64.9, 40.0, 61.5], [67.6, 66.0, 60.5], 61.0),
    "RAGAnything": ([61.4, 67.0, 61.5, 60.2, 66.3], [85.0, 76.3, 46.0], 63.4),
}
# Table 3: 영역 7개, Overall
T3 = {
    "GPT-4o-mini": ([35.5, 44.0, 24.6, 33.1, 29.5, 46.8, 31.1], 33.5),
    "LightRAG": ([40.8, 34.1, 36.2, 39.4, 41.0, 44.4, 38.3], 38.9),
    "MMGraphRAG": ([40.8, 36.5, 35.7, 35.8, 28.2, 46.9, 38.5], 37.7),
    "RAGAnything": ([46.6, 43.5, 38.7, 43.9, 34.0, 45.7, 43.6], 42.8),
}
# Table 4: ablation (DocBench)
T4 = {
    "Chunk-only": ([55.8, 61.5, 60.1, 60.7, 64.0], [81.6, 66.2, 43.5], 60.0),
    "w/o Reranker": ([60.9, 63.5, 58.8, 60.2, 68.6], [81.7, 74.7, 45.4], 62.4),
    "RAGAnything": ([61.4, 67.0, 61.5, 60.2, 66.3], [85.0, 76.3, 46.0], 63.4),
}
# Table 1
T1 = {"DocBench": {"docs": 229, "pages": 66, "q": 1102}, "MMLongBench": {"docs": 135, "pages": 47.5, "q": 1082}}

# Fig. 2: 300 dpi로 다시 렌더링한 그림에서 막대 높이와 표식 중심을 픽셀로 읽은 값(±1 px).
# 정확도 축은 20%p당 64 px(0.31%p/px), 문항 수 축은 DocBench 0.874 px/문항, MMLongBench 0.431 px/문항
BINS = ["1-10", "11-50", "51-100", "101-200", "200+"]
FIG2 = {
    "DocBench": {
        "count": [339, 348, 160, 153, 79],
        "RAGAnything": [61.9, 61.6, 61.9, 68.1, 68.7],   # 11-50은 MMGraphRAG 표식에 가려 선으로 읽음
        "MMGraphRAG": [64.8, 62.0, 59.5, 54.5, 54.9],
    },
    "MMLongBench": {
        "count": [14, 705, 148, 100, 4],                   # 1-10, 200+ 막대는 높이가 6 px, 2 px 이하
        "RAGAnything": [18.6, 43.7, 39.6, 41.5, 59.9],
        "MMGraphRAG": [12.3, 40.3, 30.4, 33.6, 39.9],
    },
}
# 본문 Sec. 3.2의 길이별 수치
TEXT_LEN = {
    "DocBench 101-200": (68.2, 54.6), "DocBench 200+": (68.8, 55.0),
    "MMLongBench gain 11-50": 3.4, "MMLongBench gain 51-100": 9.3, "MMLongBench gain 101-200": 7.9,
}


def wavg(vals, w):
    return float(np.dot(vals, w) / np.sum(w))


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(100 * (c - h), 1), round(100 * (c + h), 1)


def two_prop_p(p1, p2, n):
    """같은 n에서 두 비율의 비대응 z 검정 p값 (문항별 결과가 없어 대응 검정은 불가)."""
    pp = (p1 + p2) / 2
    se = math.sqrt(2 * pp * (1 - pp) / n)
    return float(2 * stats.norm.sf(abs(p1 - p2) / se))


def main() -> None:
    out: dict = {}

    # C1 가중 평균
    c1 = {}
    for name, (dom, _, ov) in T2.items():
        c1[f"T2/{name}"] = {"reported": ov, "domain_weighted": round(wavg(dom, DOC_Q), 2),
                            "domain_unweighted": round(float(np.mean(dom)), 2)}
    for name, (dom, ov) in T3.items():
        c1[f"T3/{name}"] = {"reported": ov, "domain_weighted": round(wavg(dom, MML_Q), 2),
                            "domain_unweighted": round(float(np.mean(dom)), 2)}
    for name, (dom, _, ov) in T4.items():
        c1[f"T4/{name}"] = {"reported": ov, "domain_weighted": round(wavg(dom, DOC_Q), 2)}
    out["C1_overall_vs_weighted_domains"] = c1
    out["C1_question_totals"] = {"DocBench": sum(DOC_Q), "MMLongBench": sum(MML_Q),
                                 "DocBench_docs": sum(DOC_DOCS), "MMLongBench_docs": sum(MML_DOCS)}

    # C2 유형별 문항 수 추정: Overall = Σ n_t·acc_t / N, Σ n_t = N (Table 2, 4의 7개 행)
    rows = list(T2.values()) + list(T4.values())
    A = np.array([r[1] for r in rows]) / 100
    b = np.array([r[2] for r in rows]) / 100
    N = sum(DOC_Q)
    # 비음수 최소제곱으로 비율 w(합 1)를 구함: 큰 가중치의 합 제약 행 추가
    lam = 50.0
    A2 = np.vstack([A, lam * np.ones(3)])
    b2 = np.concatenate([b, [lam]])
    from scipy.optimize import nnls
    w, _ = nnls(A2, b2)
    resid = A @ w - b
    out["C2_type_counts_fit"] = {
        "share_txt_mm_una": [round(float(x), 3) for x in w],
        "counts_txt_mm_una": [round(float(x) * N) for x in w],
        "residual_pp_per_row": {k: round(100 * float(r), 2) for k, r in zip(list(T2) + [f"abl/{k}" for k in T4], resid)},
        "max_abs_residual_pp": round(100 * float(np.max(np.abs(resid))), 2),
        "note": "세 유형만으로 7개 행의 Overall을 동시에 맞출 수 있는지 확인. 잔차가 크면 표에 없는 유형이 Overall에 섞여 있을 가능성",
    }

    # C3 표본 오차
    c3 = {}
    for bench, table, n in (("DocBench", T2, sum(DOC_Q)), ("MMLongBench", T3, sum(MML_Q))):
        ra = table["RAGAnything"][-1] / 100
        k = round(ra * n)
        c3[bench] = {"RAGAnything_ci95": wilson(k, n)}
        for base in ("GPT-4o-mini", "LightRAG", "MMGraphRAG"):
            pb = table[base][-1] / 100
            c3[bench][f"vs_{base}"] = {"diff_pp": round(100 * (ra - pb), 1), "p_unpaired": float(f"{two_prop_p(ra, pb, n):.3g}")}
    abl_n = sum(DOC_Q)
    c3["DocBench_ablation"] = {
        k: {"diff_pp_vs_full": round(T4["RAGAnything"][-1] - v[-1], 1),
            "p_unpaired": float(f"{two_prop_p(T4['RAGAnything'][-1] / 100, v[-1] / 100, abl_n):.3g}")}
        for k, v in T4.items() if k != "RAGAnything"
    }
    out["C3_sampling_error"] = c3

    # C4 평균 쪽수
    out["C4_avg_pages"] = {
        "DocBench": {"table1": T1["DocBench"]["pages"], "by_docs_table5": round(wavg(DOC_PAGES, DOC_DOCS), 1),
                     "by_questions_table5": round(wavg(DOC_PAGES, DOC_Q), 1),
                     "rounding_band": [round(wavg(DOC_PAGES, DOC_DOCS) - 0.5, 1), round(wavg(DOC_PAGES, DOC_DOCS) + 0.5, 1)]},
        "MMLongBench": {"table1": T1["MMLongBench"]["pages"], "by_docs_table6": round(wavg(MML_PAGES, MML_DOCS), 1),
                        "by_questions_table6": round(wavg(MML_PAGES, MML_Q), 1),
                        "rounding_band": [round(wavg(MML_PAGES, MML_DOCS) - 0.5, 1), round(wavg(MML_PAGES, MML_DOCS) + 0.5, 1)]},
    }

    # C5 Fig. 2
    c5 = {}
    for bench, d in FIG2.items():
        cnt = np.array(d["count"], dtype=float)
        row = {"count_sum_read": int(cnt.sum()), "table_total": T1[bench]["q"]}
        for m in ("RAGAnything", "MMGraphRAG"):
            row[f"{m}_weighted_by_fig_counts"] = round(wavg(d[m], cnt), 2)
        row["reported_overall"] = {"RAGAnything": (T2 if bench == "DocBench" else T3)["RAGAnything"][-1],
                                   "MMGraphRAG": (T2 if bench == "DocBench" else T3)["MMGraphRAG"][-1]}
        # 작은 구간: 정확도가 k/n이 되는 가장 작은 n 찾기
        small = {}
        for i, b in enumerate(BINS):
            if cnt[i] > 30:
                continue
            cands = []
            for n in range(1, 41):
                ok = all(abs(round(d[m][i] / 100 * n) / n * 100 - d[m][i]) <= 0.7 for m in ("RAGAnything", "MMGraphRAG"))
                if ok:
                    cands.append(n)
            n0 = next((n for n in cands if abs(n - cnt[i]) <= max(3, 0.3 * cnt[i])), cands[0] if cands else None)
            if n0:
                kr = round(d["RAGAnything"][i] / 100 * n0)
                km = round(d["MMGraphRAG"][i] / 100 * n0)
                small[b] = {"n_candidates": cands[:6], "n_used": n0, "RAGAnything_correct": kr, "MMGraphRAG_correct": km,
                            "RAGAnything_ci95": wilson(kr, n0), "MMGraphRAG_ci95": wilson(km, n0),
                            "fisher_p": float(f"{stats.fisher_exact([[kr, n0 - kr], [km, n0 - km]])[1]:.3g}")}
        row["small_bins"] = small
        row["gaps_pp"] = [round(a - b, 1) for a, b in zip(d["RAGAnything"], d["MMGraphRAG"])]
        c5[bench] = row
    c5["text_claims"] = TEXT_LEN
    out["C5_fig2_pixel_read"] = c5

    # C6 GPT-4o-mini 기준선이 볼 수 있는 쪽 비율 (영역 평균 쪽수 기준의 근사)
    def visible(pages):
        return [round(min(1.0, 50 / p), 2) for p in pages]
    out["C6_gpt4omini_visible_fraction"] = {
        "DocBench": dict(zip(DOC_DOMAINS, visible(DOC_PAGES))),
        "MMLongBench": dict(zip(MML_DOMAINS, visible(MML_PAGES))),
        "note": "영역 평균 쪽수로 계산한 근사. 50쪽을 넘는 문서는 앞 50쪽만 이미지로 들어간다는 Sec. 3.1 설명을 따름",
    }

    # C7 MMGraphRAG 대비 차이 분해 (DocBench)
    ra, mm = T2["RAGAnything"], T2["MMGraphRAG"]
    out["C7_vs_MMGraphRAG_docbench"] = {
        "domain_gap_pp": dict(zip(DOC_DOMAINS, [round(a - b, 1) for a, b in zip(ra[0], mm[0])])),
        "type_gap_pp": dict(zip(["Txt", "Mm", "Una"], [round(a - b, 1) for a, b in zip(ra[1], mm[1])])),
        "domain_contrib_pp": dict(zip(DOC_DOMAINS, [round((a - b) * q / sum(DOC_Q), 2) for a, b, q in zip(ra[0], mm[0], DOC_Q)])),
        "vs_LightRAG_type_gap_pp": dict(zip(["Txt", "Mm", "Una"], [round(a - b, 1) for a, b in zip(ra[1], T2["LightRAG"][1])])),
    }

    # C8 API 비용의 자릿수 추정 (가정은 모두 명시. 가격은 내 사전 지식이라 확인 필요)
    price = {"gpt4omini_in_per_M": 0.15, "gpt4omini_out_per_M": 0.60, "emb3large_per_M": 0.13}
    est = {}
    for bench, docs, tok, nq in (("DocBench", 229, 46377, 1102), ("MMLongBench", 135, 21214, 1082)):
        doc_tok = docs * tok / 1e6                        # M tokens
        chunks = docs * tok / 1200                        # LightRAG 기본 chunk 1,200 토큰
        extract_in = chunks * 2 * (1200 + 1500) / 1e6     # 추출 1회 + gleaning 1회(DEFAULT_MAX_GLEANING=1), 프롬프트 약 1,500 토큰 가정
        extract_out = chunks * 2 * 600 / 1e6              # 호출당 출력 약 600 토큰 가정
        query_in = nq * (32000 + 1500) / 1e6              # MAX_TOTAL_TOKENS 32,000 + 지시문
        query_kw = nq * 2000 / 1e6                        # 키워드 추출 호출
        judge = nq * 1500 / 1e6
        emb = (doc_tok * 1.5)                             # 청크 + 개체·관계 설명 임베딩
        cost = ((extract_in + query_in + query_kw + judge) * price["gpt4omini_in_per_M"]
                + extract_out * price["gpt4omini_out_per_M"] + emb * price["emb3large_per_M"])
        est[bench] = {"doc_tokens_M": round(doc_tok, 1), "llm_in_M": round(extract_in + query_in + query_kw + judge, 1),
                      "llm_out_M": round(extract_out, 1), "emb_M": round(emb, 1), "usd_text_only": round(cost, 1)}
    out["C8_cost_order_of_magnitude"] = {
        "prices_usd_assumed": price, "estimate": est,
        "excluded": "그림·표·수식마다 부르는 VLM 호출과 이미지 토큰, 재시도, reranker 비용. 문서당 그림 수가 논문에 없어 추정하지 않음",
    }

    path = ROOT / "analysis" / "table_check_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for k, v in out.items():
        print(k, json.dumps(v, ensure_ascii=False)[:400])
    print("wrote", path)


if __name__ == "__main__":
    main()
