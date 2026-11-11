"""추가 검증 1: 1차원 손가락 닫기 toy에서 클러치(Eq. 6–8) 속도 상한과 chunk 재질의 주기가 접촉 반응에 주는 영향
(numpy, CPU 1스레드, 수 초).

논문 대응 (작성자 해석, 03 문서 §2): 실행 index r는 제어 주기(100 Hz)마다 최대 한 칸만 목표 s를 따라간다.
앞으로 갈 때는 정책을 다시 질의하고(닫힌 루프), chunk H 스텝 동안은 버퍼의 명령을 연다(열린 루프).
이 toy가 시험하는 것:
  (a) 속도 상한: 클러치를 0 → 끝까지 한 번에 당겨도 r은 T_max − 1 틱(= (T_max−1)/100 s)보다 빨리 가지 않는다.
  (b) 접촉 반응: 물체 위치 x_obj(0.5, 0.6, 0.7)와 chunk 길이 H(1, 10, 30 스텝)에 따라
      접촉 뒤 힘 최댓값과 명령 초과량(q_max − x_obj)이 어떻게 변하는가.
      H = 1은 매 틱 질의(사실상 닫힌 루프), H = 30은 0.3 s마다 질의.
모형(가정 모두 작성자가 임의로 둠. 논문 모형이 아니다):
  - 시간 주기 0.01 s, T_max = 300(3 s). 시연 명령 q_demo(k) = 0.9 · smoothstep(k / (T_max − 1)).
  - 손가락: x ← x + α(q − x), α = 0.3. 물체 표면 x_obj에서 막힌다. 막혔을 때 힘 f = K (q − x_obj)+, K = 50.
  - 정책(모사): 질의 시점의 실행 index r에서 H개의 명령 q_demo(r + j) 를 낸다. 관측된 접촉(손가락이 막힘)이 있으면
    그 chunk에 "꽉 쥐기" 보정 +0.15를 더한다(≤ 1.0). 접촉을 보지 못하면 보정은 없다.
  - 클러치: 0–3 s 동안 0 → 1, 이후 1 유지. 실행 index는 s_t = floor(c_t (T_max − 1))를 따라 틱당 최대 1칸.
측정: 접촉 후 힘 최댓값, 명령 최댓값 − x_obj, 접촉 유지 시간. 속도 상한 시험은 클러치를 0 → 1로 한 틱에 바꾼다.
한계: 1차원, 손가락 하나, 힘 모형 단순, 정책 보정 +0.15는 가정. 역방향(되돌리기)과 외삽 구간은 여기서 시험하지 않는다.
실행: python3 analysis/extra_clutch_toy.py -> analysis/extra_clutch_toy_results.json
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


def smoothstep(s):
    s = np.clip(s, 0.0, 1.0)
    return s * s * (3 - 2 * s)


def q_demo(k):
    k = np.minimum(k, TMAX - 1)
    return 0.9 * smoothstep(k / (TMAX - 1))


def clutch(t):
    """0–3 s 동안 0 → 1, 이후 1."""
    return np.clip(t / 3.0, 0.0, 1.0)


def run(x_obj, H, t_end=4.5, clutch_fn=clutch, squeeze=SQUEEZE):
    n = int(round(t_end / DT))
    r = 0
    x = 0.0
    buf = []          # 현재 chunk의 남은 명령
    buf_start = None
    f_hist = np.zeros(n)
    q_hist = np.zeros(n)
    x_hist = np.zeros(n)
    r_hist = np.zeros(n, dtype=int)
    for i in range(n):
        t = i * DT
        s = int(np.floor(clutch_fn(t) * (TMAX - 1)))
        # 실행 index: 틱당 최대 한 칸
        if r < s:
            r += 1
        elif r > s:
            r -= 1
        if r_hist is not None:
            r_hist[i] = r
        # 명령 생성: 버퍼가 비었거나 chunk를 다 쓰면 질의
        if not buf:
            in_contact = x >= x_obj - 1e-3
            idx = np.arange(r, r + H)
            chunk = q_demo(idx)
            if in_contact:
                chunk = np.minimum(chunk + squeeze, 1.0)
            buf = list(chunk)
        q = buf.pop(0)
        # 손가락 동역학
        x_free = x + ALPHA * (q - x)
        if x_free > x_obj:
            x = x_obj
            f = K * max(q - x_obj, 0.0)
        else:
            x = x_free
            f = 0.0
        q_hist[i], x_hist[i], f_hist[i] = q, x, f
    return dict(t=np.arange(n) * DT, q=q_hist, x=x_hist, f=f_hist, r=r_hist)


def metrics(res, x_obj):
    """접촉 직후 0.5 s 창의 힘(평균·최댓값)과 전체 최댓값을 따로 본다.
    전체 최댓값은 시연의 마지막 명령(1.0에서 잘림)이 정하므로 chunk 길이에 둔감하다."""
    f, q = res["f"], res["q"]
    dt = DT
    contact = np.where(f > 0)[0]
    if len(contact) == 0:
        return dict(first_contact_s=None, force_0_0p5s_mean=0.0, force_0_0p5s_max=0.0, force_peak_all=0.0)
    i0 = contact[0]
    win = f[i0:i0 + int(round(0.5 / dt))]
    return dict(first_contact_s=float(i0 * dt), force_0_0p5s_mean=float(win.mean()), force_0_0p5s_max=float(win.max()),
                force_peak_all=float(f.max()), cmd_max_minus_obj=float(q.max() - x_obj))


def speed_limit_check():
    """클러치를 t = 0.5 s에 0 → 1로 한 틱에 바꾼 뒤 r이 T_max − 1에 도달하는 데 걸린 시간."""
    def step_clutch(t):
        return np.where(t < 0.5, 0.0, 1.0)
    res = run(x_obj=2.0, H=30, t_end=8.0, clutch_fn=step_clutch)
    r = res["r"]
    reach = np.where(r >= TMAX - 1)[0]
    t_reach = float(res["t"][reach[0]] - 0.5) if len(reach) else None
    return dict(T_max=TMAX, min_time_by_rate_limit_s=(TMAX - 1) * DT, measured_time_after_step_s=t_reach,
                note="x_obj=2.0(접촉 없음). r이 틱당 1칸만 움직이므로 측정값은 (T_max−1)·0.01 s와 같아야 한다.")


def main():
    out = {"setup": dict(dt=DT, T_max=TMAX, alpha=ALPHA, K=K, squeeze_bias=SQUEEZE,
                         clutch="0–3 s ramp to 1", chunk_H=[1, 10, 30], x_obj=[0.5, 0.6, 0.7]),
           "speed_limit": speed_limit_check(), "contact_response": {}}
    for x_obj in (0.5, 0.6, 0.7):
        for H in (1, 10, 30):
            res = run(x_obj, H)
            m = metrics(res, x_obj)
            out["contact_response"][f"x_obj={x_obj}_H={H}"] = m
    # 같은 조건에서 보정 없음(SQUEEZE = 0)의 대조는 한 설정만 (x_obj = 0.6, H = 30)
    out["no_squeeze_control_x_obj=0.6_H=30"] = metrics(run(0.6, 30, squeeze=0.0), 0.6)
    out["squeeze_H=1_x_obj=0.6"] = out["contact_response"]["x_obj=0.6_H=1"]
    (HERE / "extra_clutch_toy_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("speed limit:", out["speed_limit"])
    for k, v in out["contact_response"].items():
        print(f"{k:>24}  F(0-0.5s) mean={v['force_0_0p5s_mean']:.2f} max={v['force_0_0p5s_max']:.2f} peak_all={v['force_peak_all']:.2f}")
    print("no-squeeze control:", out["no_squeeze_control_x_obj=0.6_H=30"])


if __name__ == "__main__":
    main()
