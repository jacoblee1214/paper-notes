"""토이: 관측 지연 아래에서 움직이는 물체를 잡는 1차원 점 로봇.

F2F-AP의 핵심 주장(지연만큼 앞선 물체 위치를 흐름으로 예측해 행동 계획에 쓰면 낡은 관측 문제가 줄어든다)에서
'물체 움직임 보상' 부분만 가장 단순한 형태로 시험한다. 학습 없음, numpy만 쓴다(수 초).

설정
  - 물체는 x(t) = x0 + v t 로 움직이고, 시각 tc에서 속도가 dv만큼 바뀐다(비선형 구간). v ~ U(0.1, 0.4) m/s.
  - 5 Hz(0.2 s)마다 영상 한 장을 쓴다. 영상으로 만든 행동 덩어리는 총 지연 L = 0.375 s
    (Δo 0.125 + Δi 0.2 + Δc 0.05) 뒤에 로봇에 반영되고, 덩어리의 각 행동은 절대 시각표를 가진다.
  - 정책은 지난 6프레임(0.1 s 간격, 관측 잡음 sigma)에서 물체 위치를 읽는다.
      stale  : 마지막 관측 위치를 계속 목표로 둔다 (낡은 관측을 그대로 쓰는 정책)
      flow   : 6프레임 선형 적합 속도로 시각 t의 위치를 x_last + slope (t - t_img)로 예측한다
               (예측 흐름을 쓰는 정책의 가장 단순한 모형. 등속 가정)
      oracle : 시각 t의 실제 물체 위치 (예측 한계 상한)
  - 로봇은 반영된 가장 최신 덩어리의 목표를 최대 속도 제한 P 제어로 쫓는다. T_g = 1.2 s에 그리퍼가 닫히고,
    그때 로봇과 물체의 거리가 r = 3 cm 이하이면 성공이다.

한계: 로봇 상태는 정확히 안다고 가정해 VLASH식 상태 굴림의 효과는 stale과 구분되지 않는다. stale은 속도를 추정하지
않는 정책이라서, 다중 프레임에서 속도를 스스로 읽는 일반 정책보다 약한 기준이다. 행동에 절대 시각표가 있어
H와 실제 지연이 어긋나도 이 토이에서는 영향이 없고(훈련과 추론 분포 이동은 모형화하지 않음), 오차는 영상 시각에서
T_g까지 외삽 거리에서 생긴다. 속도 변화 분포, 잡음, 파지 반경은 가정이며 논문의 실험을 재현하지 않는다.
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("toy_latency_intercept_results.json")
DT = 0.005
TG = 1.2
R = 0.03
VMAX = 1.0
GAIN = 60.0
CYCLE = 0.2
L_NOM = 0.375


def simulate(policy, n=6000, seed=0, sigma=0.01, dv_std=0.0, l_jitter=0.0, v_lo=0.1, v_hi=0.4, l_true=L_NOM):
    rng = np.random.default_rng(seed)
    v = rng.uniform(v_lo, v_hi, n)
    x0 = 0.3
    tc = rng.uniform(0.4, 1.0, n)
    dv = rng.normal(0.0, dv_std, n)

    def pos(t):
        return x0 + v * t + dv * np.maximum(t - tc, 0.0)

    n_cycles = int(TG / CYCLE) + 4
    chunks = []  # (반영 시각 t_eff[n], 영상 시각 t_img, 마지막 관측 x_last[n], 속도 slope[n])
    for k in range(n_cycles):
        t_img = k * CYCLE
        lat = np.maximum(l_true + rng.normal(0.0, l_jitter, n), 0.05) if l_jitter > 0 else np.full(n, l_true)
        frames_t = t_img - 0.1 * np.arange(5, -1, -1)
        obs = np.stack([pos(max(ft, -1.0)) + rng.normal(0.0, sigma, n) for ft in frames_t], axis=1)
        tt = frames_t - frames_t.mean()
        slope = (obs * tt).sum(axis=1) / (tt ** 2).sum()
        chunks.append((t_img + lat, t_img, obs[:, -1], slope))
    xr = np.zeros(n)
    cur_img = np.full(n, -1e9)  # 현재 쓰는 덩어리의 영상 시각
    cur_x = np.zeros(n)
    cur_s = np.zeros(n)
    for s in range(int(TG / DT) + 1):
        t = s * DT
        for t_eff, t_img, x_last, slope in chunks:
            use = (t_eff <= t) & (t_img > cur_img)
            cur_img = np.where(use, t_img, cur_img)
            cur_x = np.where(use, x_last, cur_x)
            cur_s = np.where(use, slope, cur_s)
        if policy == "stale":
            target = cur_x
        elif policy == "flow":
            target = cur_x + cur_s * (t - cur_img)
        elif policy == "oracle":
            target = pos(np.full(n, t))
        else:
            raise ValueError(policy)
        xr = xr + np.clip(GAIN * (target - xr) * DT, -VMAX * DT, VMAX * DT)
    err = np.abs(xr - pos(np.full(n, TG)))
    return float((err <= R).mean()), float(np.median(err))


def main():
    pols = ("stale", "flow", "oracle")
    res = {"settings": {"L": L_NOM, "sigma_m": 0.01, "grasp_radius_m": R, "T_g": TG, "n_per_cell": 6000}}
    res["A_speed"] = {name: {p: simulate(p, v_lo=lo, v_hi=hi)[0] for p in pols}
                      for name, (lo, hi) in {"v 0.1-0.2": (0.1, 0.2), "v 0.2-0.3": (0.2, 0.3), "v 0.3-0.4": (0.3, 0.4)}.items()}
    res["B_speed_change_std"] = {str(dv): {p: simulate(p, dv_std=dv)[0] for p in pols} for dv in (0.0, 0.05, 0.1, 0.2, 0.3)}
    res["C_true_latency_s"] = {str(lt): {p: simulate(p, l_true=lt)[0] for p in pols} for lt in (0.2, 0.3, 0.375, 0.5, 0.6, 0.8)}
    res["D_latency_jitter_std_s"] = {str(j): {p: simulate(p, l_jitter=j)[0] for p in pols} for j in (0.0, 0.05, 0.1, 0.2)}
    res["E_obs_noise_m"] = {str(sg): {p: simulate(p, sigma=sg)[0] for p in pols} for sg in (0.005, 0.01, 0.02, 0.04)}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    for k, v in res.items():
        if k != "settings":
            print(k, json.dumps(v, ensure_ascii=False))


if __name__ == "__main__":
    main()
