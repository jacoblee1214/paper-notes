"""추가 검증 1의 이어서: 1차원 손가락 toy에서 접촉 뒤 되돌리기(back-off)와 재전진 시의 힘 반응
(numpy, CPU 1스레드, 수 초).

이 toy가 시험하는 것:
  - 앞으로 갈 때는 정책을 다시 질의하고(닫힌 루프), 뒤로 갈 때는 질의하지 않고 저장된 명령을 되짚는다
    (열린 루프). 이 비대칭이 접촉을 푸는 시간과 재접촉 때 힘 최댓값을 어떻게 바꾸는가.
  - 되돌리기 깊이(0.4 → 빈 공간 쪽으로 얼마나 물러나는가)와 chunk 길이 H(1, 30)를 바꾼다.

모형(가정 모두 작성자가 임의로 둠. 논문 모형이 아니다):
  - 주기 0.01 s, T_max = 300. 시연 명령 q_demo(k) = 0.9 · smoothstep(k / (T_max − 1)).
  - 손가락 x ← x + α(q − x), α = 0.3. 물체 표면 x_obj = 0.6에서 막힌다. 막혔을 때 힘 f = K (q − x_obj)+, K = 50.
  - 클러치 c(t): 0–3 s에 0 → 1, 3–4 s 유지, 4–5 s에 1 → c_back(되돌리기), 5–6 s 유지, 6–7 s에 c_back → 1(재전진).
  - 실행 index r은 틱당 최대 한 칸으로 floor(c(t)(T_max − 1))를 따른다.
  - 정책(모사): 앞으로 움직일 때(r이 늘 때) 버퍼가 비면 H개의 명령 q_demo(r + j)를 질의한다. 접촉이 관측되면
    그 chunk에 "꽉 쥐기" 보정 +0.15를 더한다(≤ 1.0). 뒤로 움직일 때(r이 줄 때)는 질의하지 않고 q_demo(r)을
    그대로 재생하며 보정을 더하지 않는다. 버퍼는 뒤로 갈 때 비운다.
측정: (1) 첫 접촉 뒤 0.5 s 힘 평균, (2) 되돌리기 시작 뒤 힘이 0이 되는 시간(접촉 해제 시간),
(3) 되돌리기 구간의 힘 최댓값, (4) 재전진 뒤 재접촉 시점의 0.5 s 힘 평균·최댓값.
한계: 1차원, 손가락 하나, 힘 모형 단순, 보정 +0.15는 가정, 뒤로 갈 때 재생은 "현재 index의 시연 명령"이라는
단순화(실제 버퍼 되짚기 순서는 모형화하지 않음).
실행: python3 analysis/extra_backoff_toy.py -> analysis/extra_backoff_toy_results.json
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DT = 0.01
TMAX = 300
ALPHA = 0.3
K = 50.0
SQUEEZE = 0.15
X_OBJ = 0.6
T_END = 8.0


def smoothstep(s):
    s = np.clip(s, 0.0, 1.0)
    return s * s * (3 - 2 * s)


def q_demo(k):
    k = np.minimum(k, TMAX - 1)
    return 0.9 * smoothstep(k / (TMAX - 1))


def clutch_fn(c_back):
    def c(t):
        if t < 3.0:
            return t / 3.0
        if t < 4.0:
            return 1.0
        if t < 5.0:
            return 1.0 - (1.0 - c_back) * (t - 4.0)
        if t < 6.0:
            return c_back
        if t < 7.0:
            return c_back + (1.0 - c_back) * (t - 6.0)
        return 1.0
    return c


def run(H, c_back, squeeze=SQUEEZE):
    clutch = clutch_fn(c_back)
    n = int(round(T_END / DT))
    r, x = 0, 0.0
    buf = []
    f_hist = np.zeros(n)
    r_hist = np.zeros(n, dtype=int)
    q_hist = np.zeros(n)
    for i in range(n):
        t = i * DT
        s = int(np.floor(clutch(t) * (TMAX - 1)))
        r_prev = r
        if r < s:
            r += 1
        elif r > s:
            r -= 1
        if r < r_prev:                      # backing off: replay, no query, no squeeze
            buf = []
            q = float(q_demo(r))
        else:
            if not buf:                     # forward: query a chunk (closed loop)
                in_contact = x >= X_OBJ - 1e-3
                chunk = q_demo(np.arange(r, r + H))
                if in_contact:
                    chunk = np.minimum(chunk + squeeze, 1.0)
                buf = list(chunk)
            q = buf.pop(0)
        x_free = x + ALPHA * (q - x)
        if x_free > X_OBJ:
            x = X_OBJ
            f = K * max(q - X_OBJ, 0.0)
        else:
            x = x_free
            f = 0.0
        f_hist[i], r_hist[i], q_hist[i] = f, r, q
    return dict(f=f_hist, r=r_hist, q=q_hist)


def metrics(res):
    f, r, q = res["f"], res["r"], res["q"]
    t = np.arange(len(f)) * DT
    contact = np.where(f > 0)[0]
    i1 = contact[0]
    first = f[i1:i1 + int(round(0.5 / DT))]
    # back-off window: from t = 4 s to the end of the back-off hold (t = 6 s)
    b0, b1 = int(round(4.0 / DT)), int(round(6.0 / DT))
    back = f[b0:b1]
    released = np.where(f[b0:b1] == 0.0)[0]
    t_release = float(released[0] * DT) if len(released) else None
    # re-advance: first contact after the back-off start
    after = np.where(f[b1:] > 0)[0]
    if len(after):
        j = b1 + after[0]
        re = f[j:j + int(round(0.5 / DT))]
        re_mean, re_max = float(re.mean()), float(re.max())
        t_recontact = float(t[j])
    else:
        re_mean = re_max = 0.0
        t_recontact = None
    return dict(first_contact_s=float(t[i1]), first_0_0p5s_mean=float(first.mean()),
                first_0_0p5s_max=float(first.max()),
                backoff_release_after_4s_s=t_release,
                backoff_force_max=float(back.max()),
                r_min_in_backoff=int(r[b0:b1].min()),
                recontact_s=t_recontact,
                recontact_0_0p5s_mean=re_mean, recontact_0_0p5s_max=re_max)


def main():
    out = {"setup": dict(dt=DT, T_max=TMAX, alpha=ALPHA, K=K, squeeze_bias=SQUEEZE, x_obj=X_OBJ,
                         clutch="0-3 s ramp to 1, hold to 4 s, 4-5 s down to c_back, hold to 6 s, "
                                "6-7 s up to 1, hold to 8 s",
                         simplification="backward motion replays q_demo(r) without query or squeeze"),
           "backoff": {}}
    for c_back in (0.4, 0.2):
        for H in (1, 30):
            out["backoff"][f"c_back={c_back}_H={H}"] = metrics(run(H, c_back))
        out["backoff"][f"c_back={c_back}_H=30_no_squeeze"] = metrics(run(30, c_back, squeeze=0.0))
    (HERE / "extra_backoff_toy_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for k, v in out["backoff"].items():
        print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})


if __name__ == "__main__":
    main()
