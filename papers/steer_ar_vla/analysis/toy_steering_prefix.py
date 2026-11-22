"""Token Steering 토이: 조향 토큰 앞부분(prefix)이 실제로 실어 나르는 정보.

논문 4.1·4.2는 "주입 창 크기 w"와 "시작 인덱스 b"를 훑고, b가 커지면 고주파 토큰을 편집하는 것이라고 해석한다.
FAST 평탄화 순서는 "주파수별로 모든 차원을 먼저"이다(FAST 논문 V-B). 조향 청크는 모든 시간 단계에
같은 관절 속도 벡터를 복사한 것이라(공식 코드 transforms.py) DCT에서 k=0(직류) 계수만 0이 아니다.
그러면 앞쪽 D=8개 위치는 모두 서로 다른 관절의 직류 계수이고, b ∈ [0,3], w ≤ 2는 그 안에 들어간다.

이 스크립트가 보이는 것(모두 가정이 붙은 토이이며 실제 토크나이저는 돌리지 않았다)
  T1  상수 속도 청크의 DCT 계수 배치와 평탄화 위치별 값 (H=10, D=8, 스케일 γ=10, 정규직교 DCT)
  T2  Franka Panda 대략 기구학: 말단 속도 1 m/s(앞·뒤·왼·오)의 관절 속도 q̇ = J⁺v 와
      "위치 b..b+w-1의 관절 직류 토큰만 알고 나머지는 모른다"고 할 때 prefix가 복원하는 말단 방향
가정: 토큰 하나가 계수 하나(BPE 병합 무시), 위치 i는 관절 i+1, DROID 홈 자세, 공구 끝 0.21 m.
공식 코드 로그(scripts/arrow_log.txt)에서는 7개 비영 관절 속도 입력이 9개 토큰이 되어 이 가정과 대략 맞는다.

실행: python3 analysis/toy_steering_prefix.py   (numpy, 1초 미만)
결과: analysis/toy_steering_prefix_results.json
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_steering_prefix_results.json"

# ---------------------------------------------------------------- T1
H, D, GAMMA = 10, 8, 10


def dct_ortho(x):
    """시간축 DCT-II (정규직교)."""
    n = x.shape[0]
    k = np.arange(n)[:, None]
    t = np.arange(n)[None, :]
    c = np.cos(np.pi / n * (t + 0.5) * k)
    c[0] *= np.sqrt(1 / n)
    c[1:] *= np.sqrt(2 / n)
    return c @ x


def t1():
    # 논문 4.5의 한 방향 입력에 해당하는 임의 관절 속도 예시(정규화 단위, 마지막은 그리퍼 0)
    qd = np.array([-0.30, -0.03, -0.53, -0.02, -0.28, -0.01, -0.65, 0.0])
    chunk = np.tile(qd, (H, 1))               # 시간 단계마다 같은 벡터 ("패딩")
    coef = dct_ortho(chunk)                   # (H, D) : 행 k = 주파수
    q = np.round(GAMMA * coef).astype(int)    # 양자화
    flat = q.reshape(-1)                      # 주파수 우선(행 우선) 평탄화: 저주파의 모든 차원이 먼저
    nz = np.flatnonzero(flat)
    return {
        "chunk_shape": [H, D], "gamma": GAMMA,
        "quantized_dc_row": q[0].tolist(),
        "nonzero_in_k_ge_1": int(np.count_nonzero(q[1:])),
        "flat_len": int(flat.size), "flat_nonzero_positions": nz.tolist(),
        "first_position_of_k1_block": D,
        "positions_touched_by_b0to3_w2": [0, 4],
        "conclusion": "b=0..3, w=2는 평탄화 위치 0–4 = 관절 1–5의 직류 계수. 고주파(k>=1)는 위치 8부터라 건드리지 않는다.",
    }


# ---------------------------------------------------------------- T2
A = [0, 0, 0, 0.0825, -0.0825, 0, 0.088]
DD = [0.333, 0, 0.316, 0, 0.384, 0, 0]
AL = [0, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2]
TOOL = 0.107 + 0.103   # 플랜지 + 그리퍼 대략


def mdh(a, d, al, th):
    ca, sa, ct, st = np.cos(al), np.sin(al), np.cos(th), np.sin(th)
    return np.array([[ct, -st, 0, a],
                     [st * ca, ct * ca, -sa, -sa * d],
                     [st * sa, ct * sa, ca, ca * d],
                     [0, 0, 0, 1]])


def jacobian(q):
    T = np.eye(4)
    zs, ps = [], []
    for i in range(7):
        T = T @ mdh(A[i], DD[i], AL[i], q[i])
        zs.append(T[:3, 2].copy())
        ps.append(T[:3, 3].copy())
    p_ee = T[:3, 3] + T[:3, 2] * TOOL
    J = np.zeros((6, 7))
    for i in range(7):
        J[:3, i] = np.cross(zs[i], p_ee - ps[i])
        J[3:, i] = zs[i]
    return J, p_ee


def t2():
    q0 = np.array([0, -np.pi / 5, 0, -4 * np.pi / 5, 0, 3 * np.pi / 5, 0.0])
    J, p = jacobian(q0)
    dirs = {"forward": [1, 0, 0], "back": [-1, 0, 0], "left": [0, 1, 0], "right": [0, -1, 0]}
    names = list(dirs)
    U = np.array([dirs[n] for n in names], float)
    qdots = {}
    for n in names:
        v6 = np.r_[np.array(dirs[n], float) * 1.0, 0, 0, 0]
        qdots[n] = np.linalg.pinv(J) @ v6
    res = {"ee_pos_m": np.round(p, 3).tolist(),
           "qdot_per_direction_rad_s": {n: np.round(qdots[n], 3).tolist() for n in names}}

    def decode(b, w):
        acc, cos_rows = 0, {}
        for n in names:
            qm = np.zeros(7)
            idx = [i for i in range(b, b + w) if i < 7]
            qm[idx] = qdots[n][idx]
            v = J[:3] @ qm
            nv = np.linalg.norm(v)
            cos = U @ v / nv if nv > 1e-9 else np.zeros(4)
            cos_rows[n] = round(float(cos[names.index(n)]), 3)
            if nv > 1e-9 and names[int(np.argmax(cos))] == n:
                acc += 1
        return acc / 4, cos_rows

    res["window_sweep_b0"] = {}
    for w in (1, 2, 4, 6):
        a, c = decode(0, w)
        res["window_sweep_b0"][f"w{w}"] = {"direction_recovered_fraction": a, "cos_with_intent": c}
    res["start_sweep_w2"] = {}
    for b in (0, 1, 2, 3):
        a, c = decode(b, 2)
        res["start_sweep_w2"][f"b{b}"] = {"direction_recovered_fraction": a, "cos_with_intent": c}
    res["paper_reference"] = {"Table1_SIR": {"w1": 0.38, "w2": 1.00, "w4": 1.00, "w6": 0.94},
                              "Fig3b_SIR_w2": {"b0": 1.00, "b1": 0.88, "b2": 0.88, "b3": "0.14 (프로젝트 페이지 0.12)"}}
    res["caveat"] = ("정책이 나머지 토큰을 채우므로 prefix가 방향을 '완전히 복원'해야 조향이 되는 것은 아니다. "
                     "이 값은 prefix 자체의 방향 정보량이며 SIR의 예측값이 아니다.")
    return res


if __name__ == "__main__":
    out = {"T1_dct_layout": t1(), "T2_prefix_information": t2()}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
