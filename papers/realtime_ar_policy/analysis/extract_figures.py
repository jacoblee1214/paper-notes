"""원문 PDF(paper/paper.pdf)의 벡터 그림에서 막대·점 값을 읽어 figure_values.json으로 저장한다.

대상: 그림 3(지연 선, Kinetix), 그림 4(LIBERO 절제), 그림 5(DROID 평균), 그림 7a(지연 대 N), 부록 그림 C(과제별 성공률).
읽는 방법: pymupdf로 쪽의 벡터 도형(막대 사각형, 오차 막대 선)을 꺼내고, 눈금 글자의 y 좌표로 값을 환산한다.
값은 그림에서 읽은 값이며 논문 본문이 준 숫자가 아니다. 실행: python3 analysis/extract_figures.py (pymupdf 필요. paper/paper.pdf 는 arXiv에서 받아 둘 것)
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
doc = pymupdf.open(ROOT / "paper" / "paper.pdf")


def words(page):
    return [(w[4], (w[1] + w[3]) / 2, (w[0] + w[2]) / 2) for w in page.get_text("words")]


def calib(ticks):
    """(값, y중심) 목록에서 선형 환산 y = a*value + b 를 최소제곱으로 구한다."""
    n = len(ticks)
    sx = sum(v for v, _ in ticks); sy = sum(y for _, y in ticks)
    sxx = sum(v * v for v, _ in ticks); sxy = sum(v * y for v, y in ticks)
    a = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    b = (sy - a * sx) / n
    return a, b


def val(y, cal):
    a, b = cal
    return (y - b) / a


def fig5():
    p = doc[7]
    out = {}
    ws = words(p)
    left = [(float(t), y) for t, y, x in ws if 112 < x < 126 and t in ("0.4", "0.6", "0.8")]
    right = [(float(t), y) for t, y, x in ws if 320 < x < 335 and t in ("0.3", "0.4", "0.5", "0.6", "0.7")]
    cl, cr = calib(left), calib(right)
    bars = []
    for dr in p.get_drawings():
        r = dr["rect"]; f = dr.get("fill")
        if dr["type"] == "f" and f and 12 < r.width < 20 and r.height > 5 and 60 < r.y0 < 170 and abs(r.y1 - 167.2) < 0.3:
            bars.append((r.x0, r.y0, dr.get("fill_opacity")))
    outlines = [(dr["rect"].x0, dr["rect"].y0) for dr in p.get_drawings()
                if dr["type"] == "s" and dr.get("color") and dr["color"][0] < 0.5 and dr["rect"].width > 10 and 60 < dr["rect"].y0 < 175]
    out["bars_left_x_value_opacity"] = [(round(x, 1), round(val(y, cl), 4), o) for x, y, o in sorted(bars) if x < 300]
    out["bars_right_x_value_opacity"] = [(round(x, 1), round(val(y, cr), 4), o) for x, y, o in sorted(bars) if x > 300]
    out["hatched_left"] = [(round(x, 1), round(val(y, cl), 4)) for x, y in outlines if x < 300]
    out["hatched_right"] = [(round(x, 1), round(val(y, cr), 4)) for x, y in outlines if x > 300]
    return out


def figC():
    """부록 그림 C의 성공률 패널 3개(π0, π0.5, π0-FAST). 각 과제 위치에 불투명 막대(동기)와 반투명 막대(실시간)가 겹쳐 있다.
    겹친 영역은 짧은 막대 스타일, 더 긴 막대의 위쪽 연장부는 긴 막대 스타일로 칠해져 있다.
    읽는 규칙: 바닥에 닿는 사각형이 둘이면 높이가 작은 쪽이 짧은 막대, 하나이고 연장부가 있으면 바닥 사각형이 짧은 막대,
    사각형이 하나뿐이면 두 막대 높이가 같다. 불투명=동기 추론, 반투명=실시간 실행(RTC 또는 π0-REALFAST).
    처리율 패널은 읽지 않았다(그림 5의 평균 값만 사용)."""
    p = doc[18]
    ws = words(p)
    names = ["pi0", "pi05", "pi0fast"]
    bases = [204.9, 312.7, 420.45]
    res = {}
    for i, nm in enumerate(names):
        ts = [(float(t), y) for t, y, x in ws if 112 < x < 126 and re.fullmatch(r"\d\.\d", t)]
        tick = [(t, y) for t, y in ts if bases[i] - 85 < y < bases[i] + 5]
        a, _ = calib(tick)
        scale = abs(a)
        rects = defaultdict(list)
        for dr in p.get_drawings():
            r = dr["rect"]; f = dr.get("fill")
            if dr["type"] in ("f", "fs") and f and 12 < r.width < 14 and r.x0 < 300 and bases[i] - 85 < r.y0 <= bases[i] + 0.2 and r.y1 <= bases[i] + 0.2:
                if f == (1.0, 1.0, 1.0) or abs(f[0] - 0.9176) < 0.01:
                    continue
                rects[round(r.x0, 1)].append((r.y0, r.y1, dr.get("fill_opacity")))
        tasks = []
        for x in sorted(rects):
            rs = rects[x]
            base = max(r[1] for r in rs)
            touch = sorted([r for r in rs if abs(r[1] - base) < 0.3], key=lambda r: -r[0])   # 짧은 것부터
            ext = [r for r in rs if r not in touch]
            short = touch[0]
            if len(touch) > 1:
                tall = touch[-1]; tall_top = tall[0]; tall_op = tall[2]
            elif ext:
                et = min(ext, key=lambda r: r[0]); tall_top, tall_op = et[0], et[2]
            else:
                tall_top, tall_op = short[0], None
            sv = round((base - short[0]) / scale, 3); tv = round((base - tall_top) / scale, 3)
            # 불투명(1.0) = 동기, 반투명(0.5) = 실시간
            if tall_op is None:
                sync = rt = sv
            elif short[2] == 1.0:
                sync, rt = sv, tv
            else:
                rt, sync = sv, tv
            tasks.append({"task": len(tasks) + 1, "sync": round(sync, 1), "realtime": round(rt, 1)})
        res[nm] = tasks
    return res


def polylines(page, box):
    """box=(x0,y0,x1,y1) 안의 꺾은선(선분 4개 이상)을 색별로 꼭짓점 목록으로 돌려준다."""
    out = []
    for dr in page.get_drawings():
        r = dr["rect"]
        if dr["type"] == "s" and len(dr["items"]) >= 3 and r.width > 30 and box[0] <= r.x0 and r.x1 <= box[2] and box[1] <= r.y0 and r.y1 <= box[3]:
            pts = []
            for it in dr["items"]:
                if it[0] == "l":
                    for q in (it[1], it[2]):
                        if not pts or pts[-1] != (q.x, q.y):
                            pts.append((q.x, q.y))
            out.append({"color": tuple(round(c, 2) for c in dr["color"]), "dashed": bool(dr.get("dashes") and dr["dashes"].strip("[] 0") != ""), "pts": pts})
    return out


def fig4():
    """그림 4a(학습 스텝별 평균 성공률, τ=0)와 4b(τ별, 6k 스텝 모델). 색: 진한 파랑 N=1, 청록 N=2, 연두 N=4, 회색 점선은 제약 디코딩 없음(N=4)."""
    p = doc[6]
    ws = words(p)
    ta = [(float(t), y) for t, y, x in ws if 372 < y < 450 and 118 < x < 125 and t in ("0.86", "0.88", "0.90", "0.92", "0.94", "0.96")]
    tb = [(float(t), y) for t, y, x in ws if 372 < y < 450 and 325 < x < 333 and t in ("0.945", "0.950", "0.955", "0.960", "0.965")]
    ca, cb = calib(ta), calib(tb)
    names = {(0.23, 0.32, 0.55): "N=1", (0.13, 0.57, 0.55): "N=2", (0.37, 0.79, 0.38): "N=4", (0.5, 0.5, 0.5): "no_constrained_N=4"}
    steps = ["3k", "6k", "10k", "15k", "30k"]
    out = {"a": {}, "b": {}}
    for ln in polylines(p, (100, 372, 300, 455)):
        out["a"][names[ln["color"]]] = {st: round(val(y, ca), 4) for st, (x, y) in zip(steps, ln["pts"])}
    taus = [round(0.1 * i, 1) for i in range(11)]
    for ln in polylines(p, (300, 372, 520, 455)):
        out["b"][names[ln["color"]]] = {str(t): round(val(y, cb), 4) for t, (x, y) in zip(taus, ln["pts"])}
    return out


def fig3():
    """그림 3a(토큰 길이 분포의 평균·99% 지연 선)와 3b(Kinetix, d_m과 m별 평균 성공률)."""
    p = doc[3]
    out = {}
    # 3a: 점선 두 개의 x 위치를 토큰 수로 환산 (LIBERO 눈금 4@138.4, 14@217.4 ; DROID 5@259.1, 25@335.2)
    lib = lambda x: 4 + (x - 138.4) / (217.4 - 138.4) * 10
    dro = lambda x: 5 + (x - 259.1) / (335.2 - 259.1) * 20
    out["token_len_at_latency_lines"] = {"LIBERO_129ms": round(lib(162.1), 2), "LIBERO_183ms": round(lib(193.7), 2),
                                         "DROID_183ms": round(dro(282.0), 2), "DROID_302ms": round(dro(316.2), 2)}
    # 3b: 오차 막대(90% 구간)의 중점을 평균으로 읽는다. y 눈금 0.9@87.3, 0.5@143.8
    cal = calib([(0.9, 87.3), (0.5, 143.8)])
    xs = {400.8: 1, 432.9: 2, 465.0: 3, 497.1: 4}
    cols = {(0.25, 0.27, 0.53): 1, (0.16, 0.47, 0.56): 2, (0.13, 0.66, 0.52): 3, (0.48, 0.82, 0.32): 4}
    pts = {}
    for dr in p.get_drawings():
        r = dr["rect"]
        if dr["type"] == "s" and r.x0 > 400 and 70 < r.y0 < 150 and r.width < 1 and r.height > 1 and dr.get("color") and tuple(round(c, 2) for c in dr["color"]) in cols:
            dm = cols[tuple(round(c, 2) for c in dr["color"])]
            m = xs[round(r.x0, 1)]
            pts[f"dm={dm},m={m}"] = {"mean": round(val((r.y0 + r.y1) / 2, cal), 3), "lo90": round(val(r.y1, cal), 3), "hi90": round(val(r.y0, cal), 3)}
    out["kinetix"] = pts
    # 3a 히스토그램: 막대 높이 -> 확률. LIBERO 눈금 0.00@142.8, 0.40@79.4 ; DROID 0.00@142.8, 0.16@83.2. 막대 폭 하나가 토큰 1개.
    bars = []
    for dr in p.get_drawings():
        r = dr["rect"]; f = dr.get("fill")
        if dr["type"] == "f" and f and 140 < r.y1 < 145 and r.height > 0.05 and r.width < 30 and r.x1 < 352:
            bars.append((r.x0, r.x1, r.y0))
    bars.sort()
    lib = [(round(lib((x0 + x1) / 2)), (142.7 - y0) / ((142.8 - 79.4) / 0.40)) for x0, x1, y0 in bars if x1 < 230]
    dro_bars = [(round(dro((x0 + x1) / 2)), (142.7 - y0) / ((142.8 - 83.2) / 0.16)) for x0, x1, y0 in bars if x0 > 250]
    out["hist_LIBERO"] = {int(t): round(v, 4) for t, v in lib}
    out["hist_DROID"] = {int(t): round(v, 4) for t, v in dro_bars}
    return out


def fig7a():
    """그림 7a: 지연 증가율 (δ(N) - δ(1)) / δ(1) 대 N. 눈금 0.0@125.9, 2.5@77.1 ; N 0@135.7, 250@219.6.
    색: 진한 파랑 Jetson Orin, 청록 RTX 4090, 연두 A100."""
    p = doc[8]
    cal = calib([(0.0, 125.9), (2.5, 77.1)])
    names = {(0.23, 0.32, 0.55): "Jetson Orin", (0.13, 0.57, 0.55): "RTX 4090", (0.37, 0.79, 0.38): "A100"}
    out = {}
    for ln in polylines(p, (100, 60, 235, 140)):
        nm = names.get(ln["color"])
        if nm:
            out[nm] = [{"N": round((x - 135.7) / (219.6 - 135.7) * 250), "rel_overhead": round(val(y, cal), 3)} for x, y in ln["pts"]]
    return out


if __name__ == "__main__":
    res = {"fig3": fig3(), "fig4": fig4(), "fig5": fig5(), "fig7a": fig7a(), "figC_success": figC()}
    (ROOT / "analysis" / "figure_values.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    for k, v in res["figC_success"].items():
        print(k, "동기", [t["sync"] for t in v], "합 %.1f" % sum(t["sync"] for t in v),
              "실시간", [t["realtime"] for t in v], "합 %.1f" % sum(t["realtime"] for t in v))
    print("그림 5 성공률(좌 패널):", res["fig5"]["bars_left_x_value_opacity"], res["fig5"]["hatched_left"])
