"""공개된 XGen 예제 npz(3개)로 XGen 합성 데이터의 두 가지 성질을 점검한다.

(1) 접촉 전 구간 물체 궤적이 중력 가속도(-9.80665 m/s^2)의 포물선인지 (논문 §3: 접촉 전 궤적은 시뮬레이터로 역방향 합성)
(2) 접촉 구간에서 물체와 손목 중점 사이 거리가 일정한지 (논문 §3: 앵커-물체 상대 변환 ϕ 보존)

pickle 은 읽지 않고 npz(allow_pickle=False)만 읽는다. 실행: python3 xgen_example_check.py <repo/xgen/data/example 경로>
"""
import json
import sys
import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "../repo/xgen/data/example"
out = {}

# (1) 포물선 점검: basketball_catch_hoi.npz, 접촉 시작 프레임 이전
d = np.load(f"{root}/basketball_catch_hoi.npz", allow_pickle=False)
fps = float(d["fps"][0])
cs = int(d["object_contact_start_frame"][0])
pos = d["object_pos_w"][: cs + 1].astype(np.float64)
t = np.arange(len(pos)) / fps
acc = {}
for ax, name in enumerate("xyz"):
    c = np.polyfit(t, pos[:, ax], 2)
    resid = pos[:, ax] - np.polyval(c, t)
    acc[name] = {"fit_accel": 2 * c[0], "rms_resid_mm": 1000 * float(np.sqrt((resid ** 2).mean()))}
# 속도 필드와 위치 차분 일관성
v_fd = np.gradient(pos, 1 / fps, axis=0)
v_rec = d["object_lin_vel_w"][: cs + 1]
out["parabola"] = {
    "file": "basketball_catch_hoi.npz", "fps": fps, "pre_contact_frames": cs + 1,
    "fit_accel_xyz": {k: round(v["fit_accel"], 4) for k, v in acc.items()},
    "rms_resid_mm": {k: round(v["rms_resid_mm"], 4) for k, v in acc.items()},
    "gravity_field": d["object_gravity_w"].tolist(),
    "max_abs_diff_recorded_vel_vs_finite_diff_mps": float(np.abs(v_fd[1:-1] - v_rec[1:-1]).max()),
    "pre_contact_velocity_w": d["object_pre_contact_velocity_w"].tolist(),
}

# (2) 접촉 구간 거리 일정성: box_pick_robot_only_synth_hoi.npz
d = np.load(f"{root}/box_pick_robot_only_synth_hoi.npz", allow_pickle=False)
s, e = int(d["object_contact_start_frame"][0]), int(d["object_release_frame"][0])
ee = d["ee_pos_w"].astype(np.float64)       # (T,5,3); README_details: 인덱스 2,3 = 발목. 0,1 을 손목으로 가정
mid = 0.5 * (ee[:, 0] + ee[:, 1])
obj = d["object_pos_w"].astype(np.float64)
dist = np.linalg.norm(obj - mid, axis=1)
seg = dist[s: e + 1]
pre = dist[:s]
out["anchor_distance"] = {
    "file": "box_pick_robot_only_synth_hoi.npz", "contact_frames": [s, e],
    "mean_dist_m": float(seg.mean()), "std_dist_mm": 1000 * float(seg.std()),
    "range_mm": 1000 * float(seg.max() - seg.min()),
    "pre_contact_dist_range_m": [float(pre.min()), float(pre.max())],
    "object_size_m": float(d["object_size"][0]),
    "contact_optimized": bool(d["object_contact_optimized"][0]),
}
# 접촉 전 'hold' 모드: 물체가 정지해 있는지
out["pre_hold_motion_mm"] = 1000 * float(np.ptp(obj[:s], axis=0).max())
out["series"] = {
    "t_pre_s": t.tolist(), "z_pre_m": pos[:, 2].tolist(), "z_fit_m": np.polyval(np.polyfit(t, pos[:, 2], 2), t).tolist(),
    "dist_m": dist.tolist(), "contact_start": s, "release": e,
}
json.dump(out, open("xgen_example_check_results.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
