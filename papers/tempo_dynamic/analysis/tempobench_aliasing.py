"""공개 데이터 TempoBench의 관절 상태 열로 "같은 자세를 다른 방향으로 지나가는 비율"을 센다.

논문 3절과 부록 A는 "접근과 후퇴가 같은 관측을 만든다"고 말한다. 이 스크립트는 이미지가 아니라 14차원 관절 상태(observation.state)만으로
그 재방문이 얼마나 자주, 얼마나 반대 방향인지, 그리고 과거 이동 방향이 이를 가르는지를 센다.
정책은 이미지와 함께 상태도 받으므로 관절 상태의 별칭은 "상태가 같으면 이미지도 같다"는 가정 아래의 근사다. 한계는 03 문서에 적었다.

정의(에피소드 안에서만 비교)
- s_t: 14차원 관절 상태를 전체 표준편차로 나눈 값. 미래 이동 f_t = s_{t+10} - s_t, 과거 이동 p_t = s_t - s_{t-10} (0.33초, 30 fps).
- 재방문: |t - t'| >= 60 프레임(2초)이면서 ||s_t - s_t'|| < eps 인 가장 가까운 t'. 두 시점 모두 움직이는 중(||f|| > thr)이어야 한다.
- 반대 방향: cos(f_t, f_t') < -0.5.
- 과거 이동이 가른다: 반대 방향 쌍에서 cos(p_t, p_t') < 0.

eps는 움직이는 프레임의 10스텝 이동 크기 중앙값의 0.25, 0.5, 1.0배 세 가지로 본다(임의 선택이라 민감도를 같이 적는다).
실행: python3 analysis/tempobench_aliasing.py <TempoBench의 data 폴더>   (numpy, pyarrow, 수 초)
입력 데이터는 외부에서 받은 것이므로 읽기만 한다. 출력: analysis/tempobench_aliasing_results.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

OUT = Path(__file__).resolve().parent / "tempobench_aliasing_results.json"
STEP = 10
MIN_GAP = 60
TASKS = ["dynamic_handover", "ball_drop", "spartan", "dynamic_pour", "catch_ball"]


def load_task(root: Path, task: str):
    eps = []
    for f in sorted((root / task / "data" / "chunk-000").glob("episode_*.parquet")):
        t = pq.read_table(f, columns=["observation.state"])
        eps.append(np.array(t["observation.state"].to_pylist(), dtype=np.float64))
    return eps


def cos(a, b):
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    return (a * b).sum(-1) / np.maximum(na * nb, 1e-12)


def analyze(eps, scale, mults):
    # 이동 크기 중앙값으로 eps 기준을 정한다
    mags = []
    for s in eps:
        z = s / scale
        mags.append(np.linalg.norm(z[STEP:] - z[:-STEP], axis=1))
    mags = np.concatenate(mags)
    thr = 0.5 * np.median(mags)
    moving_med = float(np.median(mags[mags > thr]))
    out = {"중앙 10스텝 이동(표준화)": round(moving_med, 4), "움직임 임계": round(float(thr), 4), "배수별": {}}
    for m in mults:
        eps_d = m * moving_med
        n_mov = n_rev = n_opp = n_sep = 0
        for s in eps:
            z = s / scale
            T = len(z)
            if T < 2 * STEP + MIN_GAP + 1:
                continue
            f = np.full_like(z, np.nan)
            p = np.full_like(z, np.nan)
            f[:-STEP] = z[STEP:] - z[:-STEP]
            p[STEP:] = z[STEP:] - z[:-STEP]
            ok = np.where(np.isfinite(f[:, 0]) & np.isfinite(p[:, 0]) & (np.linalg.norm(np.nan_to_num(f), axis=1) > thr))[0]
            if len(ok) < 2:
                continue
            Z = z[ok]
            D = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=2)
            gap = np.abs(ok[:, None] - ok[None, :]) < MIN_GAP
            D[gap] = np.inf
            j = D.argmin(axis=1)
            dmin = D[np.arange(len(ok)), j]
            n_mov += len(ok)
            hit = dmin < eps_d
            n_rev += int(hit.sum())
            a, b = ok[hit], ok[j[hit]]
            c_f = cos(f[a], f[b])
            c_p = cos(p[a], p[b])
            opp = c_f < -0.5
            n_opp += int(opp.sum())
            n_sep += int((opp & (c_p < 0)).sum())
        out["배수별"][str(m)] = {
            "eps(표준화)": round(eps_d, 4), "움직이는 프레임": n_mov, "재방문 프레임": n_rev,
            "재방문 비율": round(n_rev / max(n_mov, 1), 4), "그중 반대 방향": n_opp,
            "반대 방향 비율(재방문 중)": round(n_opp / max(n_rev, 1), 4),
            "반대 방향 중 과거 이동도 반대": n_sep,
            "과거 이동이 가르는 비율": round(n_sep / max(n_opp, 1), 4),
            "전체 움직이는 프레임 중 반대 방향 재방문 비율": round(n_opp / max(n_mov, 1), 4),
        }
    return out


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    root = Path(sys.argv[1])
    res = {}
    for task in TASKS:
        eps = load_task(root, task)
        allz = np.concatenate(eps)
        scale = allz.std(axis=0)
        scale[scale < 1e-6] = 1.0
        r = analyze(eps, scale, (0.25, 0.5, 1.0))
        r["에피소드"] = len(eps)
        r["프레임"] = int(sum(len(e) for e in eps))
        res[task] = r
        print(task, r["에피소드"], r["프레임"])
        for k, v in r["배수별"].items():
            print("  eps x", k, "재방문", v["재방문 비율"], "반대방향(재방문 중)", v["반대 방향 비율(재방문 중)"],
                  "과거가 가름", v["과거 이동이 가르는 비율"], "전체 중 반대 재방문", v["전체 움직이는 프레임 중 반대 방향 재방문 비율"])
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
