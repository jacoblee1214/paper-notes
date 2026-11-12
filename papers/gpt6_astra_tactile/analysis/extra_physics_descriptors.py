"""UniVTAC Insert Hole 공개 데모에서 물리량 descriptor를 계산해 진행 방향 판별력을 잰다.

흐름 (에피소드마다, 원자료는 남기지 않는다):
  1. HF 데이터셋 byml/UniVTAC의 isaac45/insert_hole/hdf5/N.hdf5를 스트리밍으로 받는다
     (.scratch/stream/, 이어받기 지원, 서버가 준 sha256과 대조).
  2. 촉각 배열과 end-effector 자세만 읽어 descriptor를 계산한다.
  3. 작은 특징값만 data/derived/ep_NNN.npz에 저장한다.
  4. 원자료 HDF5를 지운 뒤 다음 에피소드로 간다.

descriptor:
  - 로봇 좌표계 전단: 마커 변위(픽셀)를 센서 자세 쿼터니언(wxyz)으로 로봇 기준 축에 올린다.
    센서 image v축 = 센서 local Y 가정은 같은 데모의 자세 대조로 얻은 값이다(부호 확인은 analysis/07 참고).
  - 압입 δ: 첫 프레임 깊이를 기준으로 한 깊이 감소량 (깊이 맵은 센서에서 젤까지 거리다).
  - Hertz 역산(비례 근사): F_n ∝ δ^{3/2}. E*, R 값이 없어서 절대 힘은 쓰지 않는다.
  - slip margin 근사: 전단 크기 / δ^{3/2}. μ와 픽셀-힘 비례상수가 없어서 무차원 비율만 본다.

판정 대상: 막힘 시점 이후 end-effector x가 가장 크게 움직인 방향(+1/−1).
"""
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import requests
import scipy
from scipy import stats
from scipy.spatial.transform import Rotation as Rot

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STREAM = ROOT / ".scratch" / "stream"
DERIVED = ROOT / "data" / "derived"
RESULTS = HERE / "extra_physics_descriptors_results.json"

REPO = "byml/UniVTAC"
PREFIX = "isaac45/insert_hole/hdf5"
EPISODES = list(range(30))
SIDES = ["left_gsmini", "right_gsmini"]
UA = "research-roadmap-analysis/1.0"
MIN_EXCURSION_M = 0.01  # 막힘 이후 x 이동이 1 cm보다 작으면 방향 라벨을 쓰지 않는다
STALL_WIN = 5


def url_of(n: int) -> str:
    return f"https://huggingface.co/datasets/{REPO}/resolve/main/{PREFIX}/{n}.hdf5"


def head(url: str) -> dict:
    for attempt in range(5):
        try:
            r = requests.head(url, headers={"User-Agent": UA}, allow_redirects=False, timeout=60)  # 302 응답에 크기·sha256이 있다
            if r.status_code not in (200, 302):
                r.raise_for_status()
            return {
                "size": int(r.headers["x-linked-size"]),
                "sha256": r.headers["x-linked-etag"].strip('"'),
                "commit": r.headers.get("x-repo-commit", ""),
            }
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"HEAD failed: {url}")


def download(url: str, dst: Path, size: int) -> None:
    """이어받기 가능한 스트리밍 다운로드. dst.part에 쓰고 완료되면 이름을 바꾼다."""
    part = dst.with_suffix(".part")
    for attempt in range(10):
        have = part.stat().st_size if part.exists() else 0
        if have >= size:
            break
        headers = {"User-Agent": UA, "Range": f"bytes={have}-"}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as r:
                r.raise_for_status()
                with open(part, "ab") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
    if part.stat().st_size != size:
        raise RuntimeError(f"incomplete download: {part.stat().st_size} of {size}")
    os.replace(part, dst)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def decision_frame(z: np.ndarray) -> int | None:
    """z가 처음으로 멈추는 프레임(하강 10 mm 이상 뒤, 5프레임 동안 0.5 mm 미만 하강)."""
    for t in range(len(z) - STALL_WIN):
        if z[0] - z[t] >= 0.01 and z[t] - z[t + STALL_WIN] < 0.0005:
            return t
    return None


def side_features(f: h5py.File, side: str, t_q: int, ref_depth: np.ndarray) -> dict:
    g = f[f"tactile/{side}"]
    pose = g["pose"][t_q]
    rot = Rot.from_quat(np.r_[pose[4:7], pose[3]])  # 파일은 wxyz
    x_axis = rot.as_matrix()[:, 0]  # 센서 local X의 로봇 기준 방향
    y_axis = rot.as_matrix()[:, 1]  # 센서 local Y의 로봇 기준 방향 (image v)

    win = slice(t_q - STALL_WIN + 1, t_q + 1)
    marker = g["marker"][win]  # (W, 2, P, 2): k=0 기준 위치, k=1 현재 위치
    disp = (marker[:, 1] - marker[:, 0]).mean(axis=1)  # (W, 2) 점 평균 변위(px)
    # 첫 프레임에도 접촉 전 정적 오프셋(약 2 px)이 있으므로 첫 프레임 변위를 뺀다
    base = (g["marker"][0, 1] - g["marker"][0, 0]).mean(axis=0)
    du, dv = disp.mean(axis=0) - base
    disp = disp - base
    world = du * x_axis + dv * y_axis  # 로봇 기준 전단 벡터(px 단위, 부호만 의미)

    depth = g["depth"][t_q]
    delta = ref_depth - depth  # 양수 = 젤이 눌려 들어감 (mm)
    delta_p99 = float(np.percentile(delta, 99))
    contact_frac = float((delta > 0.05).mean())
    shear_mag = float(np.hypot(du, dv))
    hertz = max(delta_p99, 1e-6) ** 1.5
    return {
        "du_px": float(du),
        "du0_px": float(base[0]),
        "dv0_px": float(base[1]),
        "dv_px": float(dv),
        "shear_world_x_px": float(world[0]),
        "shear_world_z_px": float(world[2]),
        "delta_p99_mm": delta_p99,
        "contact_frac": contact_frac,
        "shear_mag_px": shear_mag,
        "hertz_Fn_rel": float(hertz),
        "slip_ratio_rel": float(shear_mag / hertz),
    }


def episode_features(path: Path) -> dict | None:
    with h5py.File(path, "r") as f:
        ee = f["embodiment/ee"][:, :3]
        t_q = decision_frame(ee[:, 2].astype(float))
        if t_q is None or t_q + STALL_WIN >= len(ee):
            return {"ok": False, "reason": "no decision frame"}
        excursion = ee[t_q:, 0] - ee[t_q, 0]
        k = int(np.argmax(np.abs(excursion)))
        if abs(excursion[k]) < MIN_EXCURSION_M:
            return {"ok": False, "reason": "lateral move < 1 cm", "t_q": t_q}
        label = int(np.sign(excursion[k]))
        out = {"ok": True, "t_q": t_q, "n_frames": int(len(ee)), "label": label,
               "excursion_mm": float(excursion[k] * 1000)}
        ref = {}
        for side in SIDES:
            ref[side] = f[f"tactile/{side}/depth"][0]
        feats = {side: side_features(f, side, t_q, ref[side]) for side in SIDES}
    # 두 손가락 평균 (각 손가락의 x 성분은 같은 부호여야 한다)
    agg = {}
    for key in feats[SIDES[0]]:
        agg[key] = float(np.mean([feats[s][key] for s in SIDES]))
    agg["shear_world_x_px_left"] = feats[SIDES[0]]["shear_world_x_px"]
    agg["shear_world_x_px_right"] = feats[SIDES[1]]["shear_world_x_px"]
    out["features"] = agg
    out["per_side"] = feats
    return out


def process_episode(n: int) -> dict:
    dst = DERIVED / f"ep_{n:03d}.npz"
    meta_path = DERIVED / f"ep_{n:03d}.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    STREAM.mkdir(parents=True, exist_ok=True)
    DERIVED.mkdir(parents=True, exist_ok=True)
    url = url_of(n)
    info = head(url)
    local = STREAM / f"{n}.hdf5"
    t0 = time.time()
    download(url, local, info["size"])
    dl_s = time.time() - t0
    digest = sha256_of(local)
    if digest != info["sha256"]:
        local.unlink()
        raise RuntimeError(f"sha256 mismatch for episode {n}")
    try:
        feat = episode_features(local)
    finally:
        local.unlink()  # 원자료는 항상 지운다
    record = {"episode": n, "url": url, "bytes": info["size"], "sha256": digest,
              "commit": info["commit"], "download_s": round(dl_s, 1), **feat}
    if feat.get("ok"):
        np.savez(dst, **{k: np.float64(v) for k, v in feat["features"].items()},
                 label=np.int64(feat["label"]), t_q=np.int64(feat["t_q"]))
    meta_path.write_text(json.dumps(record, indent=1, ensure_ascii=False))
    return record


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(c - h), float(c + h))


def auc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = score[label > 0], score[label < 0]
    return float(stats.mannwhitneyu(pos, neg, alternative="two-sided").statistic / (len(pos) * len(neg)))


def boot_auc_ci(score, label, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    idx = np.arange(len(label))
    while len(vals) < reps:
        b = rng.choice(idx, size=len(idx), replace=True)
        if (label[b] > 0).sum() == 0 or (label[b] < 0).sum() == 0:
            continue
        vals.append(auc(score[b], label[b]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def summarize(records: list[dict]) -> dict:
    ok = [r for r in records if r.get("ok")]
    label = np.array([r["label"] for r in ok])
    feats = {k: np.array([r["features"][k] for r in ok]) for k in ok[0]["features"]}
    summary = {"n_episodes_requested": len(records), "n_usable": len(ok),
               "n_excluded": len(records) - len(ok),
               "label_counts": {"+x": int((label > 0).sum()), "-x": int((label < 0).sum())},
               "majority_rate": float(max((label > 0).mean(), (label < 0).mean())),
               "descriptors": {}}
    signed = {"shear_world_x_px": "로봇 x 전단(signed, 주 descriptor)",
              "shear_world_x_px_left": "왼쪽 손가락 로봇 x 전단",
              "shear_world_x_px_right": "오른쪽 손가락 로봇 x 전단",
              "dv_px": "image v 변위(좌표 변환 없음, 손가락 평균)",
              "du_px": "image u 변위(좌표 변환 없음, 손가락 평균)",
              "shear_world_z_px": "로봇 z 전단(signed)"}
    for k, desc in signed.items():
        s = feats[k]
        sign_hit = int(((np.sign(s) == label) & (s != 0)).sum())
        n = len(label)
        summary["descriptors"][k] = {
            "meaning": desc, "auc": auc(s, label), "auc_ci95_boot": boot_auc_ci(s, label),
            "sign_accuracy": sign_hit / n, "sign_accuracy_ci95": wilson(sign_hit, n),
            "sign_binom_p_two_sided": float(stats.binomtest(sign_hit, n, 0.5).pvalue),
        }
    # 크기·법선 계열: 방향 정보가 없어야 하는 대조군
    for k in ["delta_p99_mm", "contact_frac", "shear_mag_px", "hertz_Fn_rel", "slip_ratio_rel"]:
        s = feats[k]
        summary["descriptors"][k] = {
            "meaning": "크기/법선 계열 (방향 대조군)", "auc": auc(s, label),
            "auc_ci95_boot": boot_auc_ci(s, label), "abs_auc_minus_half": abs(auc(s, label) - 0.5),
        }
    return summary


def main() -> None:
    t_start = time.time()
    records = [process_episode(n) for n in EPISODES]
    for r in records:
        r.setdefault("ok", False)
    summary = summarize(records)
    ok = [r for r in records if r.get("ok")]
    peak_bytes = max(r["bytes"] for r in records)
    results = {
        "script": "analysis/extra_physics_descriptors.py",
        "dataset": {"repo": REPO, "files": f"{PREFIX}/N.hdf5", "episodes": EPISODES,
                    "total_bytes_transferred": int(sum(r["bytes"] for r in records)),
                    "largest_single_file_bytes": int(peak_bytes),
                    "license_note": "HF 데이터셋 byml/UniVTAC, 공개(게이트 없음). 원자료는 저장하지 않음"},
        "decision_frame_rule": {"descent_min_m": 0.01, "stall_window_frames": STALL_WIN,
                                "stall_threshold_m": 0.0005},
        "label_rule": {"label": "막힘 프레임 이후 ee x 변위가 가장 큰 방향",
                       "min_excursion_m": MIN_EXCURSION_M},
        "sensor_frame_assumption": "image u = local X, image v = local Y, 쿼터니언은 wxyz (analysis/07 §1.1 자세 대조와 일치)",
        "units_note": "변위는 픽셀, 깊이는 mm. 부호와 순위만 쓰고 절대 힘·slip margin은 계산하지 않음",
        "summary": summary,
        "episodes": records,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "h5py": h5py.__version__, "scipy": scipy.__version__,
                     "requests": requests.__version__},
        "wall_time_s": round(time.time() - t_start, 1),
    }
    RESULTS.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(f"usable {len(ok)}/{len(records)}; signed x AUC "
          f"{summary['descriptors']['shear_world_x_px']['auc']:.3f}; "
          f"sign acc {summary['descriptors']['shear_world_x_px']['sign_accuracy']:.3f}")


if __name__ == "__main__":
    main()
