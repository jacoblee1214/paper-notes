"""표 1 표현의 좌표계 불변성 점검 (CPU, numpy+scipy, 1초 미만). 재현이 아니라 정의 수준의 검산이다.

질문: 로봇 베이스 프레임을 강체변환 G로 옮겼을 때(= 다른 로봇의 베이스 정의), 표 1의 네 표현이 바뀌는가?
  - Abs. EEF 상태   V(T_b,et)           : 바뀜 (베이스 정의에 의존)
  - EEF-Delta 상태  V(T_et^-1 T_b,et-i) : 불변
  - World-Delta 행동 V(T_b,et+1) - V(T_b,et) (오일러각 차): 바뀜 (회전 G에 대해)
  - EEF-Delta 행동  V(T_b,et^-1 T_b,et+1)  : 불변
실행: python3 -I analysis/toy_frame_invariance.py → analysis/toy_frame_invariance_results.json
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

rng = np.random.default_rng(0)
ROOT = Path(__file__).resolve().parents[1]


def mat(R, p):
    T = np.eye(4)
    T[:3, :3] = R.as_matrix()
    T[:3, 3] = p
    return T


def V(T):
    """V(T) = [p, Euler(R)] (논문 표 1; 오일러 순서는 논문에 명시 없음 -> xyz 가정)."""
    return np.concatenate([T[:3, 3], Rot.from_matrix(T[:3, :3]).as_euler("xyz")])


def inv(T):
    return np.linalg.inv(T)


N, H = 200, 4
res = {k: [] for k in ("abs_state", "eef_state", "world_action", "eef_action")}
for _ in range(N):
    # 4-스텝 이력 + 다음 스텝 (EEF 자세는 베이스 기준)
    Ts = [mat(Rot.from_rotvec(rng.normal(size=3) * 0.4), rng.normal(size=3) * 0.2) for _ in range(H + 1)]
    G = mat(Rot.from_euler("z", rng.uniform(-np.pi, np.pi)) * Rot.from_rotvec(rng.normal(size=3) * 0.1), rng.normal(size=3) * 0.3)
    Ts2 = [G @ T for T in Ts]  # 베이스 정의가 G만큼 다른 로봇에서의 같은 물리 동작
    res["abs_state"].append(max(np.abs(V(Ts[i]) - V(Ts2[i])).max() for i in range(H)))
    res["eef_state"].append(max(np.abs(V(inv(Ts[0]) @ Ts[i]) - V(inv(Ts2[0]) @ Ts2[i])).max() for i in range(H)))
    res["world_action"].append(np.abs((V(Ts[H]) - V(Ts[0])) - (V(Ts2[H]) - V(Ts2[0]))).max())
    res["eef_action"].append(np.abs(V(inv(Ts[0]) @ Ts[H]) - V(inv(Ts2[0]) @ Ts2[H])).max())

out = {k: dict(median_max_abs_change=float(np.median(v)), max=float(np.max(v))) for k, v in res.items()}
# 표 1의 h=0 '상태 없음' 해석: 이력 {V(T_et^-1 T_et-i)}_{i=0}^{h-1} 은 h=0 이면 빈 집합, h=1 이면 항등 변환 하나
out["history_len_h0"] = len(range(0))
out["history_len_h1"] = len(range(1))
out["identity_state_vector"] = V(np.eye(4)).tolist()
(ROOT / "analysis" / "toy_frame_invariance_results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
if __name__ == "__main__":
    print(json.dumps(out, indent=1))
