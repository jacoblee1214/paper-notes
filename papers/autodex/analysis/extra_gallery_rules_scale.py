"""공개 갤러리 데이터를 넓혀 높이 최댓값 규칙과 3 s 유지 규칙의 일치율을 다시 잰다.

기존 40개(extra_gallery_rules.py, data/allegro/)에 더해, 허브 목록의 allegro 궤적 중 기존 40개를 뺀 나머지에서
seed 1로 120개를 무작위로 받는다. 받는 파일은 두 가지다.
  - object_tracking/gotrack_output/world_pose_records.json (추적 높이)
  - executed_grasp/meta.json (실행 상태 시각)
받은 파일은 data/gallery_extra/ 아래에 두고, 허브 목록의 크기와 git blob sha1로 대조한다.
규칙 정의(5 cm, 3 s, 부호 두 가지)는 extra_gallery_rules.py와 같다.

실행: python3 -I analysis/extra_gallery_rules_scale.py
결과: analysis/extra_gallery_rules_scale_results.json
"""
import glob
import hashlib
import json
import math
import os
import platform
import sys
import time
import urllib.parse
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
DATA_OLD = os.path.join(ROOT, "data")
DATA_NEW = os.path.join(ROOT, "data", "gallery_extra")
MANIFEST = os.path.join(DATA_NEW, "manifest.json")
RESULTS = os.path.join(HERE, "extra_gallery_rules_scale_results.json")
HF_REPO = "willi19/autodex-gallery"
UA = "research-roadmap-analysis/1.0"
N_NEW = 120
SEED_NEW = 1
BUDGET_NEW_BYTES = 700 * 1024 * 1024
THR_MM = 50.0
HOLD_S = 3.0
HOLD_SENS = [1.0, 2.0, 3.0, 4.0]
SIGNS = {"as_recorded": 1.0, "flipped": -1.0}


def http_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r), r.headers.get("Link", "")
        except Exception:
            time.sleep(4 * (attempt + 1))
    raise RuntimeError(f"listing failed: {url}")


def build_manifest():
    """허브 목록에서 allegro 아래 world_pose_records.json을 모은다. 목록은 경로순이므로 allegro를 벗어나면 멈춘다."""
    url = f"https://huggingface.co/api/datasets/{HF_REPO}/tree/main?recursive=true&expand=false&limit=1000"
    found, seen_allegro = [], False
    while url:
        items, link = http_json(url)
        done = False
        for x in items:
            if x.get("type") != "file":
                continue
            p = x["path"]
            if p.startswith("allegro/"):
                seen_allegro = True
                if p.endswith("object_tracking/gotrack_output/world_pose_records.json"):
                    found.append({"path": p, "size": x["size"], "git_oid": x["oid"]})
            elif seen_allegro:
                done = True
                break
        if done:
            break
        url = None
        for part in link.split(","):
            if 'rel="next"' in part:
                url = part[part.index("<") + 1:part.index(">")]
    os.makedirs(DATA_NEW, exist_ok=True)
    with open(MANIFEST, "w") as f:
        json.dump({"repo": HF_REPO, "population_world_pose_files": len(found), "files": found}, f, indent=1)
    return found


def git_blob_sha1(path):
    h = hashlib.sha1()
    size = os.path.getsize(path)
    h.update(f"blob {size}\0".encode())
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(rel_path, size, oid):
    """ROOT 기준이 아니라 DATA_NEW 아래에 받는다. 이미 있으면 크기와 oid만 확인한다."""
    dst = os.path.join(DATA_NEW, rel_path)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{urllib.parse.quote(rel_path)}"
        part = dst + ".part"
        for attempt in range(6):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=180) as r, open(part, "wb") as fh:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
                break
            except Exception:
                time.sleep(4 * (attempt + 1))
        os.replace(part, dst)
    if os.path.getsize(dst) != size:
        raise RuntimeError(f"size mismatch {rel_path}")
    if git_blob_sha1(dst) != oid:
        raise RuntimeError(f"git blob sha1 mismatch {rel_path}")
    return dst


def fetch_plain(rel_path):
    """작은 메타 파일: 크기는 응답 길이로, 식별은 git blob sha1 기록으로 남긴다."""
    dst = os.path.join(DATA_NEW, rel_path)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{urllib.parse.quote(rel_path)}"
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=120) as r:
            body = r.read()
        with open(dst, "wb") as fh:
            fh.write(body)
    return dst


def runs_above(t, h, thr):
    out, start = [], None
    for i in range(len(h)):
        if h[i] >= thr:
            if start is None:
                start = i
        elif start is not None:
            out.append((t[start], t[i - 1], False))
            start = None
    if start is not None:
        out.append((t[start], t[-1], True))
    return out


def hold_status(t, h, thr, hold_s):
    undet = False
    for t0, t1, open_end in runs_above(t, h, thr):
        if t1 - t0 >= hold_s:
            return "yes"
        if open_end:
            undet = True
    return "undetermined" if undet else "no"


def load_trial(trial_dir):
    rec = json.load(open(os.path.join(trial_dir, "object_tracking", "gotrack_output", "world_pose_records.json")))
    meta = json.load(open(os.path.join(trial_dir, "executed_grasp", "meta.json")))
    frames = [r for r in rec if r.get("status") == "ok" and r.get("translation_world_mm") is not None]
    t = np.array([r["time_sec"] for r in frames], float)
    z = np.array([r["translation_world_mm"][2] for r in frames], float)
    st = {s["state"]: s["t_rel_s"] for s in meta.get("execution_states", [])}
    return t, z, st


def analyse_trial(t, z, st, sign):
    if "grasp" not in st:
        return None
    tg = st["grasp"]
    pre = (t >= tg - 0.5) & (t <= tg)
    if pre.sum() < 3:
        return None
    z_ref = float(np.median(z[pre]))
    post = t >= tg
    tp, hp = t[post], sign * (z[post] - z_ref)
    if tp.size == 0:
        return None  # 파지 시각 뒤에 기록이 없는 시도는 분석에서 뺀다
    onset = next((float(tp[i]) for i in range(len(hp)) if hp[i] >= THR_MM), None)
    return {
        "record_end_s": float(t[-1]), "grasp_t_s": tg, "lift_t_s": st.get("lift"),
        "zmax_mm": float(hp.max()), "zmax_rule": bool(hp.max() >= THR_MM),
        "hold_3s": hold_status(tp, hp, THR_MM, HOLD_S),
        "hold_by_duration": {str(s): hold_status(tp, hp, THR_MM, s) for s in HOLD_SENS},
        "onset_s": onset,
    }


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def summarise(rows):
    det = [r for r in rows if r["hold_3s"] != "undetermined"]
    agree = sum(1 for r in det if r["zmax_rule"] == (r["hold_3s"] == "yes"))
    cnt = {"zmax_yes_hold_yes": 0, "zmax_no_hold_no": 0, "zmax_yes_hold_no": 0, "zmax_no_hold_yes": 0}
    for r in det:
        cnt[f"zmax_{'yes' if r['zmax_rule'] else 'no'}_hold_{'yes' if r['hold_3s'] == 'yes' else 'no'}"] += 1
    sens = {}
    for s in HOLD_SENS:
        key = str(s)
        dd = [r for r in rows if r["hold_by_duration"][key] != "undetermined"]
        a = sum(1 for r in dd if r["zmax_rule"] == (r["hold_by_duration"][key] == "yes"))
        sens[key] = {"n_determined": len(dd), "agree": a, "agree_rate": round(a / len(dd), 4) if dd else None}
    return {
        "n_analysed": len(rows),
        "n_zmax_yes": sum(1 for r in rows if r["zmax_rule"]),
        "n_hold_yes": sum(1 for r in rows if r["hold_3s"] == "yes"),
        "n_hold_undetermined": sum(1 for r in rows if r["hold_3s"] == "undetermined"),
        "n_determined": len(det),
        "agreement": {"agree": agree, "rate": round(agree / len(det), 4) if det else None,
                      "wilson95": wilson(agree, len(det))},
        "contingency_determined": cnt,
        "zmax_yes_but_hold_no_trials": [r["trial"] for r in det if r["zmax_rule"] and r["hold_3s"] == "no"],
        "sensitivity_hold_duration": sens,
    }


def main():
    t_start = time.time()
    if os.path.exists(MANIFEST):
        man = json.load(open(MANIFEST))
    else:
        found = build_manifest()
        man = {"population_world_pose_files": len(found), "files": found}
    pop = man["files"]
    old_rel = sorted(os.path.relpath(p, DATA_OLD) for p in glob.glob(
        os.path.join(DATA_OLD, "*", "*", "*", "object_tracking", "gotrack_output", "world_pose_records.json")))
    pool = [x for x in pop if x["path"] not in set(old_rel)]
    rng = np.random.default_rng(SEED_NEW)
    pick = [pool[i] for i in sorted(rng.choice(len(pool), size=N_NEW, replace=False))]
    planned = sum(x["size"] for x in pick)
    assert planned <= BUDGET_NEW_BYTES, planned

    trial_dirs = []  # (이름, 궤적 디렉터리, 출처)
    for p in old_rel:
        d = os.path.join(DATA_OLD, os.path.dirname(os.path.dirname(os.path.dirname(p))))
        trial_dirs.append((os.path.relpath(d, DATA_OLD), d, "first40"))
    new_bytes = 0
    missing_meta = []
    for x in pick:
        rel_wp = x["path"]
        rel_trial = os.path.dirname(os.path.dirname(os.path.dirname(rel_wp)))
        rel_meta = rel_trial + "/executed_grasp/meta.json"
        fetch(rel_wp, x["size"], x["git_oid"])
        new_bytes += x["size"]
        try:
            dst = fetch_plain(rel_meta)
        except Exception:
            missing_meta.append(rel_meta)
            continue
        new_bytes += os.path.getsize(dst)
        d = os.path.join(DATA_NEW, rel_trial)
        trial_dirs.append((rel_trial, d, "new120"))

    trials = []
    for name, d, origin in trial_dirs:
        t, z, st = load_trial(d)
        trials.append((name, t, z, st, origin))

    rows = {k: [] for k in SIGNS}
    skipped = {k: [] for k in SIGNS}
    for name, t, z, st, origin in trials:
        for k, s in SIGNS.items():
            r = analyse_trial(t, z, st, s)
            if r is None:
                skipped[k].append(name)
                continue
            r["trial"] = name
            r["origin"] = origin
            rows[k].append(r)

    by_origin = {}
    for k in SIGNS:
        by_origin[k] = {o: summarise([r for r in rows[k] if r["origin"] == o]) for o in ("first40", "new120")}
    results = {
        "script": "analysis/extra_gallery_rules_scale.py",
        "source": {"repo": HF_REPO, "commit_sha_seen": "4d9212eeaf3892e260289078c2190076b9a6ae3a",
                   "repo_last_modified": "2026-09-30T16:55:20.000Z", "gated": False,
                   "license": "데이터셋 카드에 license 필드 없음(미표기)",
                   "population_allegro_world_pose_files": man["population_world_pose_files"],
                   "population_note": "기존 analysis/extra_gallery_rules.py 주석은 639개였다. 이번 목록은 allegro 978개이며, 차이 원인은 확인하지 않았다.",
                   "verification": "크기와 git blob sha1(목록의 oid)로 대조. 목록 oid는 LFS sha256이 아니다."},
        "selection": {"first40": "기존 40개(data/allegro/, extra_gallery_rules.py와 같은 파일)",
                      "new120": f"population에서 기존 40개를 뺀 나머지 중 seed {SEED_NEW}로 {N_NEW}개"},
        "bytes_new_this_run": int(new_bytes),
        "missing_meta_trials": missing_meta,
        "n_trials_total": len(trials),
        "skipped_no_grasp_state": skipped,
        "threshold_mm": THR_MM, "hold_s": HOLD_S,
        "conventions_all": {k: summarise(rows[k]) for k in SIGNS},
        "conventions_by_origin": by_origin,
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform()},
        "wall_time_s": round(time.time() - t_start, 2),
    }
    with open(RESULTS, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=float)
    for k in SIGNS:
        a = results["conventions_all"][k]
        print(k, "analysed", a["n_analysed"], "zmax_yes", a["n_zmax_yes"], "hold_yes", a["n_hold_yes"],
              "undet", a["n_hold_undetermined"], "agree", a["agreement"], flush=True)
    print("bytes new", new_bytes, "missing meta", len(missing_meta), flush=True)


if __name__ == "__main__":
    main()
