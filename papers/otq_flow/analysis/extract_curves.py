"""논문 PDF의 벡터 그래프에서 Fig. 2, 3, 5 곡선 점을 읽어 curves_extracted.json으로 저장한다.
실행: python3 extract_curves.py ../paper/paper.pdf
PyMuPDF가 필요하다. 격자선(0.0~1.0)으로 y좌표를 보정하므로 눈으로 읽는 것보다 정확하다."""
import sys, json
import fitz

def rc(t): return tuple(round(c, 2) for c in t)

def grid_lines(page, col):
    g = []
    for p in page.get_drawings():
        if p['type'] == 's' and p['color'] and rc(p['color']) == col and len(p['items']) == 1:
            a, b = p['items'][0][1], p['items'][0][2]
            if abs(a.y - b.y) < 0.01:
                g.append((min(a.x, b.x), max(a.x, b.x), a.y))
    return g

def polylines(page, colmap, minseg):
    out = []
    for p in page.get_drawings():
        if p['type'] == 's' and p['color'] and rc(p['color']) in colmap and len(p['items']) >= minseg and all(i[0] == 'l' for i in p['items']):
            pts = [p['items'][0][1]] + [i[2] for i in p['items']]
            out.append((colmap[rc(p['color'])], [(q.x, q.y) for q in pts]))
    return out

def to_val(pts, grid, ymax=1.0):
    xm = sum(p[0] for p in pts) / len(pts)
    ys = sorted({round(y, 2) for (x0, x1, y) in grid if x0 <= xm <= x1}, reverse=True)
    y0, y1 = ys[0], ys[-1]          # 0.0과 가장 윗 격자선
    top = ymax
    return [round((y0 - p[1]) / (y0 - y1) * top, 3) for p in pts], len(ys)

if __name__ == '__main__':
    doc = fitz.open(sys.argv[1])
    res = {}
    p7 = doc[6]
    g7 = grid_lines(p7, (0.73, 0.73, 0.73))
    names2 = {(0.16, 0.36, 0.67): 'OTQL', (0.35, 0.35, 0.35): 'DSRL', (0.5, 0.5, 0.5): 'QAM-F', (0.65, 0.65, 0.65): 'FQL'}
    names3 = {(0.16, 0.36, 0.67): 'OTQL', (0.35, 0.35, 0.35): 'FQL', (0.5, 0.5, 0.5): 'QC', (0.65, 0.65, 0.65): 'EWFM'}
    for name, nm, panels in [('fig2', names2, {'cube-double-task4': (100, 200), 'humanoidmaze-task1': (200, 300)}),
                             ('fig3', names3, {'OpenMicrowave-v0': (300, 410), 'ArrangeBoxes-v0': (410, 520)})]:
        res[name] = {}
        for lab, pts in polylines(p7, nm, 7):
            x0 = pts[0][0]
            for pn, (a, b) in panels.items():
                if a <= x0 < b:
                    v, ng = to_val(pts, g7)
                    res[name].setdefault(pn, {})[lab] = {'n_points': len(pts), 'values': v, 'n_gridlines': ng}
    p8 = doc[7]
    g8 = grid_lines(p8, (0.73, 0.73, 0.73))
    res['fig5'] = {}
    tasks = ['stack cups', 'tape in penholder', 'close door']
    pl = sorted(polylines(p8, {(0.16, 0.36, 0.67): 'OTQL'}, 5), key=lambda t: t[1][0][0])
    base = []
    for p in p8.get_drawings():
        if p['type'] == 's' and p['color'] and rc(p['color']) == (0.69, 0.2, 0.12) and len(p['items']) == 1:
            a = p['items'][0][1]; base.append((a.x, a.y))
    base.sort()
    for t, (lab, pts) in zip(tasks, pl):
        v, ng = to_val(pts, g8)
        xm = sum(p[0] for p in pts) / len(pts)
        # 기준선: 같은 패널에 있는 빨간 점선의 y
        xs_ = [p[0] for p in pts]
        cand = [b for b in base if min(xs_) - 5 <= b[0] <= max(xs_) + 5]
        ys = sorted({round(y, 2) for (x0, x1, y) in g8 if x0 <= xm <= x1}, reverse=True)
        bv = round((ys[0] - cand[0][1]) / (ys[0] - ys[-1]), 3) if cand else None
        res['fig5'][t] = {'otql': v, 'base': bv, 'n_gridlines': ng}
    json.dump(res, open('curves_extracted.json', 'w'), indent=1, ensure_ascii=False)
    print(json.dumps(res, ensure_ascii=False)[:6000])
