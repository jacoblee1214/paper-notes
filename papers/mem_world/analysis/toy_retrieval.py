"""토이 T1: 손목 카메라 재방문 상황에서 기억 검색 규칙 네 가지의 커버리지 비교 (2차원 평면).

목적: Mem-World(arXiv:2606.18960)의 W-VMem 검색(식 2의 가시성 점수 + NMS)이
  (a) 가장 최근 K프레임(short-term), (b) 같은 간격 K프레임(stride), (c) 카메라 자세가 가장 가까운 K프레임
보다 "목표 시점에서 보일 장면 조각"을 더 많이 덮는지, 그리고 (d) 식 2의 시간 감쇠가 이 지표에 어떤 영향을 주는지 본다.
이것은 논문 수치의 재현이 아니라 설계 의도가 단순한 설정에서 성립하는지의 확인이다.

가정(모두 토이의 가정이며 논문 값이 아니다)
- 장면은 정적이다(움직이는 물체 없음). 따라서 최근성의 이득은 이 토이에 없다.
- 장면 조각(surfel) 300개를 단위 정사각형에 무작위로 뿌리고, 카메라는 시야각 70도, 시야 거리 0.45의 2차원 쐐기다.
- 궤적은 T = 20 스텝의 폐곡선에 가까운 이동이고 마지막에 처음 구역으로 돌아온다. 궤적 모양과 시작 위치는 시드마다 무작위다.
- 그리퍼 가림: 무작위로 정한 스텝의 시야 중앙 40%가 가려져 그 부분의 조각은 기록되지 않는다(손목 시점 영상에서 조각이 생성되지 않는 효과).
- 기억은 손목 카메라로 본 조각만 갱신한다(논문 3.2절). 검색 점수는 가시 조각마다 cos/(1+깊이)를 더한다.
- 식 2: λ_min = 0.1, 반감기 H = 0.3 × T_max. m_s(조작 물체 표지)는 토이에 없다(모두 0).
- "자세 유사도"는 위치와 방향이 가까운 순이다. Ctrl-World의 실제 검색(간격 표본 + 자세 조건 교차주의)과는 다르다.

실행: python3 analysis/toy_retrieval.py   (numpy, 수 초)
출력: analysis/toy_retrieval_results.json
"""
import json
import math
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_retrieval_results.json"
T, NS, FOV, RNG = 20, 300, math.radians(70), 0.45
LAM, HFRAC = 0.1, 0.3


def visible(pts, pose, occ=None):
    """pose = (x, y, theta). 시야 안(각도, 거리)에 있는 조각 마스크와 cos 가중, 깊이."""
    d = pts - pose[:2]
    dist = np.linalg.norm(d, axis=1)
    ang = np.arctan2(d[:, 1], d[:, 0]) - pose[2]
    ang = (ang + math.pi) % (2 * math.pi) - math.pi
    ok = (dist < RNG) & (np.abs(ang) < FOV / 2)
    if occ:
        ok &= np.abs(ang) > occ * FOV / 2  # 중앙 occ 비율이 가려짐
    return ok, np.cos(ang), dist


def make_trajectory(rng):
    c = rng.uniform(0.35, 0.65, 2)
    r = rng.uniform(0.15, 0.3)
    ph = rng.uniform(0, 2 * math.pi)
    poses = []
    for t in range(T + 1):
        a = ph + 2 * math.pi * t / T
        pos = c + r * np.array([math.cos(a), math.sin(a)])
        look = c + 0.3 * r * np.array([math.cos(ph), math.sin(ph)]) * (1 + 0.5 * math.sin(a * 2))  # 안쪽 한 구역을 자주 봄
        d = look - pos
        poses.append((pos[0], pos[1], math.atan2(d[1], d[0]) + rng.normal(0, 0.15)))
    return poses


def nms_topk(score, sets, k, thr=0.7):
    order = np.argsort(-score)
    sel = []
    for i in order:
        if score[i] <= 0:
            break
        if all(len(sets[i] & sets[j]) / max(1, len(sets[i] | sets[j])) <= thr for j in sel):
            sel.append(int(i))
        if len(sel) == k:
            break
    for i in order:  # 부족하면 점수 순으로 채움
        if len(sel) == k:
            break
        if int(i) not in sel:
            sel.append(int(i))
    return sel


def run(seed, K):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, 1, (NS, 2))
    poses = make_trajectory(rng)
    occ = [0.4 if rng.random() < 0.4 else 0.0 for _ in range(T)]
    obs = []  # t별로 기록된 조각 집합
    for t in range(T):
        ok, _, _ = visible(pts, poses[t], occ[t])
        obs.append(set(np.where(ok)[0].tolist()))
    # 목표: 마지막 스텝에서 곧 보게 될 시점(재방문)
    tgt = poses[T]
    tgt_ok, cosv, dist = visible(pts, tgt)
    need = set(np.where(tgt_ok)[0].tolist())
    if len(need) < 5:
        return None
    # 시점마다 목표 시점에서 보이는 조각 중 그 시점에 기록된 것의 가중 합(식 2의 기하항, 시간항 분리)
    geo = np.zeros(T)
    for t in range(T):
        geo[t] = sum(max(cosv[i], 0) / (1 + dist[i]) for i in (obs[t] & need))
    age = (T - np.arange(T)) / T
    decay = LAM + (1 - LAM) * 2 ** (-(age) / HFRAC)
    sel = {
        "short": list(range(T - K, T)),
        "stride": [int(round(x)) for x in np.linspace(0, T - 1, K)],
        "pose": list(np.argsort([math.hypot(p[0] - tgt[0], p[1] - tgt[1]) + 0.3 * abs(((p[2] - tgt[2]) + math.pi) % (2 * math.pi) - math.pi)
                                 for p in poses[:T]])[:K]),
        "wvmem": nms_topk(geo * decay, obs, K),
        "wvmem_nodecay": nms_topk(geo, obs, K),
    }
    cov = {}
    for k, idx in sel.items():
        got = set().union(*[obs[i] for i in idx]) & need
        cov[k] = len(got) / len(need)
    cov["oracle_all_history"] = len(set().union(*obs) & need) / len(need)
    return cov


def summarize(K):
    res = [c for c in (run(s, K) for s in range(400)) if c]
    keys = list(res[0])
    out = {"n_runs": len(res), "K": K, "T": T, "mean_coverage": {}, "ci95": {}, "paired_vs_wvmem": {}}
    arr = {k: np.array([r[k] for r in res]) for k in keys}
    for k in keys:
        m = arr[k].mean()
        se = arr[k].std(ddof=1) / math.sqrt(len(res))
        out["mean_coverage"][k] = round(float(m), 3)
        out["ci95"][k] = [round(float(m - 1.96 * se), 3), round(float(m + 1.96 * se), 3)]
    for k in ("short", "stride", "pose", "wvmem_nodecay"):
        d = arr["wvmem"] - arr[k]
        se = d.std(ddof=1) / math.sqrt(len(d))
        out["paired_vs_wvmem"][k] = dict(wvmem_minus=round(float(d.mean()), 3),
                                         ci95=[round(float(d.mean() - 1.96 * se), 3), round(float(d.mean() + 1.96 * se), 3)])
    return out


def main():
    out = {f"K{K}": summarize(K) for K in (2, 3, 4)}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
