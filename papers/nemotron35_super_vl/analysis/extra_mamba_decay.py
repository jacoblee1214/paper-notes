"""Mamba 층의 A_log·dt_bias(BF16, 층당 128개)를 공개 safetensors 샤드에서 HTTP Range로 받아
헤드별 감쇠 α의 실제 분포를 만들고, 그 분포로 FP16 상태 반올림 비교(RN, SR, 체크포인트 L=16)를 다시 돌린다.

받는 양: 샤드 40개의 헤더(JSON)와 Mamba 텐서 80개(각 256 B)만. 가중치 전체는 받지 않는다.
원자료 바이트는 data/mamba_decay_raw/에 저장한다(.gitignore). 파싱은 그 파일을 읽는다.
이미 받은 경우(manifest.json이 있으면) 다시 받지 않는다.

감쇠 식(표준 Mamba-2 형태): A = -exp(A_log), dt = softplus(dt_raw + dt_bias), α = exp(A·dt).
dt_raw는 입력에 따라 달라지므로 가중치에 없다. 본선은 dt_raw = 0(편향만)이고,
dt_raw = ±2 민감도를 함께 본다. 모델 코드 파일은 로컬에 없어서 식은 확인하지 않았다.

실행: python3 -I analysis/extra_mamba_decay.py
결과: analysis/extra_mamba_decay_results.json
"""
import hashlib
import json
import os
import platform
import struct
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import requests
import scipy

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = ROOT / "data" / "mamba_decay_raw"
RESULTS = HERE / "extra_mamba_decay_results.json"
INDEX = ROOT / "paper" / "hf" / "bf16_index.json"

REPO = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"
REV = "fb0ee0bc6b3f70342afd8d84cb80dbadbaa8a1fe"
UA = "research-roadmap-analysis/1.0"
N_HEAD = 128
N_TOK, L_CKPT, N_HEADS_TOY = 4096, 16, 64
STATE_N, STATE_P = 128, 64


def get_range(url: str, start: int, end: int) -> bytes:
    last = None
    for attempt in range(8):
        try:
            r = requests.get(url, headers={"User-Agent": UA, "Range": f"bytes={start}-{end}"}, timeout=(30, 180))
            r.raise_for_status()
            if len(r.content) != end - start + 1:
                raise requests.RequestException(f"short read {len(r.content)}")
            return r.content
        except requests.RequestException as e:
            last = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"range {start}-{end} failed: {last}")


def fetch() -> dict:
    """샤드 헤더와 Mamba 텐서 80개를 받아 RAW에 저장한다. 받은 바이트 수를 반환한다."""
    weight_map = json.loads(INDEX.read_text())["weight_map"]
    names = sorted(k for k in weight_map if k.endswith("mixer.A_log") or k.endswith("mixer.dt_bias"))
    by_shard: dict[str, list[str]] = {}
    for k in names:
        by_shard.setdefault(weight_map[k], []).append(k)

    RAW.mkdir(parents=True, exist_ok=True)
    assert not any(RAW.iterdir()), "raw dir must be empty before fetching"
    total = 0
    tensors = []
    shards = {}
    for shard in sorted(by_shard):
        url = f"https://huggingface.co/{REPO}/resolve/{REV}/{shard}"
        h = requests.head(url, headers={"User-Agent": UA}, allow_redirects=False, timeout=60)
        commit = h.headers.get("x-repo-commit", "")
        assert commit == REV, (shard, commit)
        size = int(h.headers["x-linked-size"])
        n = struct.unpack("<Q", get_range(url, 0, 7))[0]
        hdr_raw = get_range(url, 8, 8 + n - 1)
        total += 8 + n
        header = json.loads(hdr_raw)
        header.pop("__metadata__", None)
        base = 8 + n
        shards[shard] = {"file_bytes": size, "header_bytes": n, "header_sha256": hashlib.sha256(hdr_raw).hexdigest()}
        for k in by_shard[shard]:
            info = header[k]
            assert info["dtype"] == "BF16" and info["shape"] == [N_HEAD], (k, info)
            s, e = base + info["data_offsets"][0], base + info["data_offsets"][1] - 1
            raw = get_range(url, s, e)
            total += len(raw)
            fname = k.replace("language_model.backbone.", "").replace(".", "_") + ".bin"
            (RAW / fname).write_bytes(raw)
            tensors.append({"name": k, "shard": shard, "dtype": info["dtype"], "shape": info["shape"],
                            "file": fname, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
            print(f"fetched {k} ({len(raw)} B)", flush=True)
    manifest = {"repo": REPO, "revision": REV, "bytes_transferred": total, "shards": shards, "tensors": tensors}
    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    return manifest


def bf16_to_f32(raw: bytes) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
    return (u16 << 16).view(np.float32).astype(np.float64)


def load_params() -> dict:
    man = json.loads((RAW / "manifest.json").read_text())
    A, D, layer_of = [], [], []
    for t in man["tensors"]:
        raw = (RAW / t["file"]).read_bytes()
        assert len(raw) == t["bytes"] == N_HEAD * 2
        assert hashlib.sha256(raw).hexdigest() == t["sha256"]
        v = bf16_to_f32(raw)
        layer = int(t["name"].split("layers.")[1].split(".")[0])
        if t["name"].endswith("A_log"):
            A.append((layer, v))
        else:
            D.append((layer, v))
    A.sort(key=lambda x: x[0])
    D.sort(key=lambda x: x[0])
    assert [a[0] for a in A] == [d[0] for d in D]
    return {"layers": [a[0] for a in A], "A_log": np.stack([a[1] for a in A]), "dt_bias": np.stack([d[1] for d in D]),
            "manifest": man}


def softplus(x):
    return np.logaddexp(0.0, x)


def neglog_alpha(A_log, dt_bias, dt_raw=0.0):
    """-log α = exp(A_log) · softplus(dt_raw + dt_bias)."""
    return np.exp(A_log) * softplus(dt_raw + dt_bias)


# ---- FP16 반올림 비교 (mechanism_toy.py와 같은 정의) ----
def to_fp16(x, stochastic, rng):
    x16 = x.astype(np.float16)
    if not stochastic:
        return x16.astype(np.float64)
    up = x16.astype(np.float64) < x
    lo = np.where(up, x16, np.nextafter(x16, np.float16(-np.inf)))
    hi = np.where(up, np.nextafter(x16, np.float16(np.inf)), x16)
    exact = x16.astype(np.float64) == x
    lo, hi = lo.astype(np.float64), hi.astype(np.float64)
    p = np.where(exact, 0.0, (x - lo) / np.where(hi > lo, hi - lo, 1.0))
    return np.where(exact, x, np.where(rng.random(x.shape) < p, hi, lo))


VARIANTS = ["FP16 RN 매 토큰", "FP16 SR 매 토큰", f"Replay L={L_CKPT} + FP16 RN", f"Replay L={L_CKPT} + FP16 SR"]


def toy_one_head(neglog: float, seed: int, n=N_TOK, L=L_CKPT):
    """헤드 하나: α = exp(-neglog) 고정, 4096 토큰. 반환: 최종 상태 상대 오차, 출력 평균 상대 오차."""
    rng = np.random.default_rng(seed)
    a = np.full(n, np.exp(-neglog))
    k = rng.normal(size=(n, STATE_N)) / np.sqrt(STATE_N)
    v = rng.normal(size=(n, STATE_P))
    q = rng.normal(size=(n, STATE_N)) / np.sqrt(STATE_N)
    ref = np.zeros((STATE_N, STATE_P))
    st = {kk: np.zeros((STATE_N, STATE_P)) for kk in VARIANTS}
    errs = {kk: [] for kk in VARIANTS}
    for t in range(n):
        kv = np.outer(k[t], v[t])
        ref = a[t] * ref + kv
        for kk in VARIANTS:
            if kk.startswith("FP16"):
                st[kk] = to_fp16(a[t] * st[kk] + kv, "SR" in kk, rng)
            else:
                st[kk] = a[t] * st[kk] + kv
                if (t + 1) % L == 0:
                    st[kk] = to_fp16(st[kk], "SR" in kk, rng)
            o_ref = q[t] @ ref
            errs[kk].append(np.linalg.norm(q[t] @ st[kk] - o_ref) / (np.linalg.norm(o_ref) + 1e-12))
    final = {kk: float(np.linalg.norm(st[kk] - ref) / np.linalg.norm(ref)) for kk in VARIANTS}
    outerr = {kk: float(np.mean(errs[kk])) for kk in VARIANTS}
    return final, outerr


def run_toy(neglogs: np.ndarray, seed0: int) -> dict:
    fin = {kk: [] for kk in VARIANTS}
    out = {kk: [] for kk in VARIANTS}
    for i, nl in enumerate(neglogs):
        f, o = toy_one_head(float(nl), seed0 + i)
        for kk in VARIANTS:
            fin[kk].append(f[kk])
            out[kk].append(o[kk])
    summ = {}
    for kk in VARIANTS:
        summ[kk] = {"final_state_rel_err_mean": float(np.mean(fin[kk])), "output_rel_err_mean": float(np.mean(out[kk])),
                    "output_rel_err_median": float(np.median(out[kk]))}
    sr_gt_rn = float(np.mean(np.array(out[VARIANTS[1]]) > np.array(out[VARIANTS[0]])))
    srck_gt_rnck = float(np.mean(np.array(out[VARIANTS[3]]) > np.array(out[VARIANTS[2]])))
    return {"n_heads": len(neglogs), "neglog_alpha": {"min": float(neglogs.min()), "median": float(np.median(neglogs)),
            "max": float(neglogs.max())}, "summary": summ,
            "frac_heads_SR_worse_than_RN_per_token": sr_gt_rn,
            "frac_heads_replaySR_worse_than_replayRN": srck_gt_rnck}


def main() -> None:
    t0 = time.time()
    if (RAW / "manifest.json").exists():
        bytes_fetched = json.loads((RAW / "manifest.json").read_text())["bytes_transferred"]
        fetched_now = False
    else:
        bytes_fetched = fetch()["bytes_transferred"]
        fetched_now = True
    P = load_params()
    man = P["manifest"]

    A_log, dt_bias = P["A_log"], P["dt_bias"]  # (층 40, 헤드 128)
    stats_all = {}
    for name, dt_raw in [("dt_raw_0_bias_only", 0.0), ("dt_raw_minus2", -2.0), ("dt_raw_plus2", 2.0)]:
        nl = neglog_alpha(A_log, dt_bias, dt_raw)
        flat = nl.ravel()
        stats_all[name] = {
            "quantiles_neglog": {str(q): float(np.quantile(flat, q)) for q in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]},
            "frac_neglog_below_2.4e-4": float(np.mean(flat < 2.4e-4)),
            "frac_neglog_below_1e-2": float(np.mean(flat < 1e-2)),
            "frac_neglog_above_1e-1": float(np.mean(flat > 1e-1)),
            "per_layer_median_neglog": {str(l): float(np.median(nl[i])) for i, l in enumerate(P["layers"])},
            "per_layer_min_neglog": {str(l): float(np.min(nl[i])) for i, l in enumerate(P["layers"])},
        }
    nl0 = neglog_alpha(A_log, dt_bias, 0.0).ravel()
    head_stats = {"A_log_mean": float(A_log.mean()), "A_log_std": float(A_log.std()),
                  "A_log_min": float(A_log.min()), "A_log_max": float(A_log.max()),
                  "dt_bias_mean": float(dt_bias.mean()), "dt_bias_std": float(dt_bias.std()),
                  "dt_bias_min": float(dt_bias.min()), "dt_bias_max": float(dt_bias.max())}

    # 실제 분포에서 헤드 64개를 -log α 순위로 고르게 뽑는다 (분위수 간격)
    order = np.argsort(nl0)
    pick = order[np.linspace(0, len(order) - 1, N_HEADS_TOY).round().astype(int)]
    real_nl = nl0[pick]
    real = run_toy(real_nl, seed0=100000)
    # 민감도: dt_raw = -2로 옮기면 느린 헤드가 늘어난다. 같은 방식으로 헤드 64개를 다시 뽑는다
    nl_m2 = neglog_alpha(A_log, dt_bias, -2.0).ravel()
    pick_m2 = np.argsort(nl_m2)[np.linspace(0, nl_m2.size - 1, N_HEADS_TOY).round().astype(int)]
    real_m2 = run_toy(nl_m2[pick_m2], seed0=150000)

    # 비교 대상: 이전 toy와 같은 방식의 가정 분포(log-uniform, 빠른 헤드 / 느린 헤드 두 구간)를 같은 헤드 수로
    rng_assume = np.random.default_rng(0)
    assumed_fast = np.exp(rng_assume.uniform(np.log(1e-2), np.log(5e-2), N_HEADS_TOY))
    assumed_slow = np.exp(rng_assume.uniform(np.log(2e-5), np.log(2e-4), N_HEADS_TOY))
    assumed = {"fast_heads": run_toy(assumed_fast, seed0=200000), "slow_heads": run_toy(assumed_slow, seed0=300000)}

    # 실제 분포를 빠른/느린 구간으로 나눠 따로 본다
    bins = {"neglog_lt_2.4e-4": nl0 < 2.4e-4, "neglog_2.4e-4_to_1e-2": (nl0 >= 2.4e-4) & (nl0 < 1e-2),
            "neglog_ge_1e-2": nl0 >= 1e-2}
    by_bin = {}
    for name, m in bins.items():
        idx = np.where(m)[0]
        if len(idx) == 0:
            by_bin[name] = {"n_heads_total": 0}
            continue
        sel = idx[np.linspace(0, len(idx) - 1, min(N_HEADS_TOY // 2, len(idx))).round().astype(int)]
        by_bin[name] = {"n_heads_total": int(len(idx)), "n_heads_toy": int(len(sel)),
                        **run_toy(nl0[sel], seed0=400000 + int(idx[0]))}

    results = {
        "script": "analysis/extra_mamba_decay.py",
        "source": {"repo": REPO, "revision": REV, "shards": sorted(man["shards"]),
                   "n_shards": len(man["shards"]), "tensor_names_sample": [t["name"] for t in man["tensors"][:4]],
                   "tensors": [{"name": t["name"], "shape": t["shape"], "dtype": t["dtype"], "bytes": t["bytes"],
                                "sha256": t["sha256"]} for t in man["tensors"]],
                   "bytes_transferred_total": int(bytes_fetched), "fetched_in_this_run": fetched_now,
                   "disk_bytes_raw_saved": int(sum(t["bytes"] for t in man["tensors"])),
                   "note": "샤드 40개의 헤더와 Mamba 텐서 80개만 받음. 전체 샤드는 받지 않음. sha256은 받은 바이트 기준(서버 LFS 해시 대조 안 함)."},
        "layers": P["layers"], "heads_per_layer": N_HEAD,
        "formula": {"A": "-exp(A_log)", "dt": "softplus(dt_raw + dt_bias)", "alpha": "exp(A * dt)",
                    "dt_raw": "가중치에 없음. 0(본선), ±2(민감도)", "limits": "time_step_limit [0, inf]: softplus>0이라 영향 없음",
                    "verified_against_model_code": False},
        "param_stats": head_stats,
        "decay_distribution": stats_all,
        "toy": {"settings": {"tokens": N_TOK, "L": L_CKPT, "heads": N_HEADS_TOY, "state": [STATE_N, STATE_P],
                             "input": "iid gaussian k, v, q (as mechanism_toy.py)", "reference": "FP64"},
                "real_heads_quantile_sample": real,
                "real_heads_quantile_sample_dt_raw_minus2": real_m2,
                "assumed_log_uniform": assumed,
                "real_heads_by_bin": by_bin},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "wall_time_s": round(time.time() - t0, 1),
    }
    RESULTS.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(json.dumps({"bytes_transferred": bytes_fetched, "real": real["summary"], "assumed_fast": assumed["fast_heads"]["summary"],
                      "assumed_slow": assumed["slow_heads"]["summary"], "by_bin": by_bin}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
