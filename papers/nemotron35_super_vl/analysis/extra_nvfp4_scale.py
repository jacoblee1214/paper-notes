"""BF16 체크포인트 shard 하나의 expert 가중치를 NVFP4로 에뮬레이션해 블록 scale 선택 방식을 비교한다.

원자료는 디스크에 쓰지 않는다. HTTP Range로 safetensors 헤더와 고른 expert 텐서만 메모리로 받고,
BF16을 numpy로 FP32로 풀어 NVFP4(E2M1 값, 16원소 블록의 FP8 E4M3 scale, 텐서당 FP32 scale)로 양자화한다.

두 가지 scale 선택:
  - Max: 블록의 절댓값 최댓값으로 scale을 정한다.
  - MSE: Max scale 근처의 E4M3 값들을 훑어 블록 오차 제곱합이 가장 작은 scale을 고른다.

결과 파일에는 통계만 남긴다(텐서·가중치 값 없음).
"""
import hashlib
import json
import platform
import struct
import sys
import time
from pathlib import Path

import numpy as np
import requests
import scipy
from scipy import stats

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "extra_nvfp4_scale_results.json"
FIGURE = HERE.parent / "report" / "figures" / "extra_nvfp4_scale.png"

REPO = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"
SHARD = "model-00009-of-00063.safetensors"
URL = f"https://huggingface.co/{REPO}/resolve/main/{SHARD}"
UA = "research-roadmap-analysis/1.0"
BLOCK = 16
N_EXPERTS_PICK = 12  # 같은 층(layer 12)의 expert 12개, up_proj와 down_proj 둘 다
MSE_GRID = np.unique(np.r_[np.linspace(0.5, 1.25, 31), 1.0])  # 1.0 = Max scale 포함

E2M1_POS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1_MID = (E2M1_POS[:-1] + E2M1_POS[1:]) / 2


def _e4m3_positive_values() -> np.ndarray:
    vals = []
    for e in range(16):
        for m in range(8):
            if e == 15 and m == 7:  # E4M3(fn)의 NaN 코드
                continue
            vals.append(m / 8 * 2.0**-6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7))
    return np.unique(np.array(vals))


E4M3 = _e4m3_positive_values()  # 0 .. 448
E4M3_MID = (E4M3[:-1] + E4M3[1:]) / 2
assert E4M3[-1] == 448.0


def round_e4m3(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 0.0, 448.0)
    return E4M3[np.searchsorted(E4M3_MID, v)]


def round_e2m1(a: np.ndarray) -> np.ndarray:
    """절댓값을 E2M1 값으로 반올림(가장 가까운 값, 동점은 작은 쪽)."""
    idx = np.searchsorted(E2M1_MID, np.minimum(a, 6.0))
    return E2M1_POS[idx]


def quant_blocks(X: np.ndarray, d: np.ndarray) -> np.ndarray:
    """X: (B, 16) FP32, d: (B,) 디코딩된 블록 scale. 반환: 양자화 후 값."""
    safe = np.where(d > 0, d, 1.0)[:, None]
    y = X / safe
    q = np.sign(y) * round_e2m1(np.abs(y)) * safe
    return np.where(d[:, None] > 0, q, 0.0)


def quantize_max(X: np.ndarray, g: float):
    amax = np.abs(X).max(axis=1)
    s_raw = amax / (6.0 * g)
    code = round_e4m3(s_raw)
    return quant_blocks(X, code * g), code, s_raw


def quantize_mse(X: np.ndarray, g: float, s_raw: np.ndarray, code_max: np.ndarray):
    best_err = np.full(X.shape[0], np.inf)
    best_code = code_max.copy()
    for r in MSE_GRID:
        code = round_e4m3(s_raw * r)
        xq = quant_blocks(X, code * g)
        err = ((X - xq) ** 2).sum(axis=1)
        better = err < best_err
        best_err = np.where(better, err, best_err)
        best_code = np.where(better, code, best_code)
    return quant_blocks(X, best_code * g), best_code, best_err


def bf16_to_f32(raw: bytes, shape) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
    return (u16 << 16).view(np.float32).reshape(shape)


def get_range(start: int, end: int) -> bytes:
    """[start, end] 바이트를 받는다. 네트워크 오류는 재시도한다."""
    last = None
    for attempt in range(8):
        try:
            r = requests.get(URL, headers={"User-Agent": UA, "Range": f"bytes={start}-{end}"}, timeout=(30, 180))
            r.raise_for_status()
            if len(r.content) != end - start + 1:
                raise requests.RequestException(f"short read {len(r.content)}")
            return r.content
        except requests.RequestException as e:
            last = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"range {start}-{end} failed: {last}")


def server_meta() -> dict:
    r = requests.head(URL, headers={"User-Agent": UA}, allow_redirects=False, timeout=60)
    return {
        "size": int(r.headers["x-linked-size"]),
        "sha256_server": r.headers["x-linked-etag"].strip('"'),
        "commit": r.headers.get("x-repo-commit", ""),
    }


def stats_for(W: np.ndarray, What: np.ndarray, Xb: np.ndarray, Yb: np.ndarray) -> dict:
    """W, What: 원본/양자화 텐서. Xb: (B,16) 원본 블록, Yb: 양자화 블록."""
    err = W - What
    aw = np.abs(W)
    thr = np.quantile(aw, 0.999)
    out = aw >= thr
    # 블록 안에서 가장 큰 원소(블록 outlier)의 상대 오차
    j = np.argmax(np.abs(Xb), axis=1)
    rows = np.arange(Xb.shape[0])
    blk_max_rel = np.abs(Xb[rows, j] - Yb[rows, j]) / np.maximum(np.abs(Xb[rows, j]), 1e-30)
    return {
        "rel_fro": float(np.linalg.norm(err) / np.linalg.norm(W)),
        "sqnr_db": float(10 * np.log10(np.sum(W**2) / np.sum(err**2))),
        "outlier_top01_mean_rel_err": float(np.mean(np.abs(err[out]) / aw[out])),
        "outlier_top01_shrunk_frac": float(np.mean(np.abs(What[out]) < 0.9 * aw[out])),
        "block_max_mean_rel_err": float(blk_max_rel.mean()),
        "block_max_p99_rel_err": float(np.quantile(blk_max_rel, 0.99)),
    }


def main() -> None:
    t_start = time.time()
    meta = server_meta()

    n = struct.unpack("<Q", get_range(0, 7))[0]
    header = json.loads(get_range(8, 8 + n - 1))
    data_base = 8 + n
    header.pop("__metadata__", None)
    up = {int(k.split("experts.")[1].split(".")[0]): k for k in header if "layers.12.mixer.experts." in k and k.endswith("up_proj.weight")}
    dn = {int(k.split("experts.")[1].split(".")[0]): k for k in header if "layers.12.mixer.experts." in k and k.endswith("down_proj.weight")}
    both = sorted(set(up) & set(dn))
    pick = [both[i] for i in np.linspace(0, len(both) - 1, N_EXPERTS_PICK).round().astype(int)]
    names = []
    for e in pick:
        names.append((e, "up_proj", up[e]))
        names.append((e, "down_proj", dn[e]))

    global_stats = {"header_bytes": n, "tensors_in_shard": len(header),
                    "expert_tensors_layer12_in_shard": {"up_proj": len(up), "down_proj": len(dn)},
                    "experts_picked": pick}

    rows = []
    total_bytes = 8 + n
    for e, proj, name in names:
        info = header[name]
        assert info["dtype"] == "BF16"
        s, t = data_base + info["data_offsets"][0], data_base + info["data_offsets"][1] - 1
        raw = get_range(s, t)
        total_bytes += len(raw)
        W = bf16_to_f32(raw, info["shape"])
        sha = hashlib.sha256(raw).hexdigest()
        del raw
        X = W.reshape(-1, BLOCK)
        g = float(np.abs(W).max() / (6.0 * 448.0))  # 텐서당 FP32 scale

        Ymax, code_max, s_raw = quantize_max(X, g)
        Ymse, code_mse, _ = quantize_mse(X, g, s_raw, code_max)
        # MSE 선택은 Max 후보를 포함하므로 블록 오차는 Max보다 클 수 없다
        e_max = ((X - Ymax) ** 2).sum(1)
        e_mse = ((X - Ymse) ** 2).sum(1)
        assert np.all(e_mse <= e_max + 1e-12 * np.maximum(e_max, 1.0))

        Wmax = Ymax.reshape(W.shape)
        Wmse = Ymse.reshape(W.shape)
        blk_amax = np.abs(X).max(1)
        blk_rms = np.sqrt((X**2).mean(1)) + 1e-30
        rec = {
            "expert": e, "proj": proj, "tensor": name, "shape": info["shape"], "bytes": int(t - s + 1),
            "sha256_bytes": sha, "fp32_absmax": float(np.abs(W).max()), "fp32_std": float(W.std()),
            "tensor_scale_fp32": g,
            "block_amax_over_rms_median": float(np.median(blk_amax / blk_rms)),
            "block_amax_over_rms_p99": float(np.quantile(blk_amax / blk_rms, 0.99)),
            "mse_code_below_max_frac": float(np.mean(code_mse < code_max)),
            "mse_code_ratio_median": float(np.median(code_mse / np.maximum(code_max, 1e-30))),
            "block_err_reduction_frac_blocks": float(np.mean(e_mse < e_max * (1 - 1e-9))),
            "max": stats_for(W, Wmax, X, Ymax),
            "mse": stats_for(W, Wmse, X, Ymse),
        }
        rec["total_err_ratio_mse_over_max"] = float(e_mse.sum() / e_max.sum())
        rows.append(rec)
        print(f"{name}: max rel {rec['max']['rel_fro']:.4f} mse rel {rec['mse']['rel_fro']:.4f}", flush=True)

    def col(key, method, proj=None):
        return np.array([r[method][key] for r in rows if proj is None or r["proj"] == proj])

    summary = {}
    for proj in [None, "up_proj", "down_proj"]:
        rmax = col("rel_fro", "max", proj)
        rmse = col("rel_fro", "mse", proj)
        key = proj or "all"
        summary[key] = {
            "n_tensors": int(len(rmax)),
            "rel_fro_max_mean": float(rmax.mean()), "rel_fro_mse_mean": float(rmse.mean()),
            "rel_fro_max_std_across_experts": float(rmax.std(ddof=1)),
            "rel_fro_mse_std_across_experts": float(rmse.std(ddof=1)),
            "mean_reduction_pct": float(100 * np.mean(1 - rmse / rmax)),
            "mse_better_tensors": int(np.sum(rmse < rmax)),
            "sqnr_db_max_mean": float(col("sqnr_db", "max", proj).mean()),
            "sqnr_db_mse_mean": float(col("sqnr_db", "mse", proj).mean()),
            "outlier_top01_mean_rel_err_max": float(col("outlier_top01_mean_rel_err", "max", proj).mean()),
            "outlier_top01_mean_rel_err_mse": float(col("outlier_top01_mean_rel_err", "mse", proj).mean()),
            "outlier_top01_shrunk_frac_max": float(col("outlier_top01_shrunk_frac", "max", proj).mean()),
            "outlier_top01_shrunk_frac_mse": float(col("outlier_top01_shrunk_frac", "mse", proj).mean()),
            "block_max_mean_rel_err_max": float(col("block_max_mean_rel_err", "max", proj).mean()),
            "block_max_mean_rel_err_mse": float(col("block_max_mean_rel_err", "mse", proj).mean()),
            "block_max_p99_rel_err_max": float(col("block_max_p99_rel_err", "max", proj).mean()),
            "block_max_p99_rel_err_mse": float(col("block_max_p99_rel_err", "mse", proj).mean()),
            "mse_code_below_max_frac_mean": float(np.mean([r["mse_code_below_max_frac"] for r in rows if proj is None or r["proj"] == proj])),
            "amax_over_rms_median": float(np.median([r["block_amax_over_rms_median"] for r in rows if proj is None or r["proj"] == proj])),
        }
    paired = stats.wilcoxon(col("rel_fro", "mse"), col("rel_fro", "max"), alternative="less")
    summary["all"]["wilcoxon_mse_lt_max_p"] = float(paired.pvalue)

    results = {
        "script": "analysis/extra_nvfp4_scale.py",
        "source": {"repo": REPO, "file": SHARD, "url": URL, "commit": meta["commit"],
                   "file_bytes": meta["size"], "sha256_server_linked_etag": meta["sha256_server"],
                   "sha256_verified_locally": False,
                   "note": "전체 파일은 받지 않았다. 서버가 준 X-Linked-ETag(LFS sha256)만 기록한다. 받은 range마다 sha256을 rows[].sha256_bytes에 기록했다."},
        "method": {"format": "NVFP4 emulation (E2M1 values, 16-element blocks, FP8 E4M3 block scale, FP32 per-tensor scale)",
                   "tensor_scale": "g = absmax(tensor) / (6 * 448)",
                   "max_scale": "code = E4M3(absmax_block / (6 g))",
                   "mse_scale": f"argmin over candidates E4M3(s_raw * r), r in {len(MSE_GRID)} values from 0.5 to 1.25 (1.0 included)",
                   "outlier_definition": "|w| >= 99.9th percentile of the tensor",
                   "rounding": "E2M1 nearest, ties to lower magnitude; E4M3 nearest table value"},
        "bytes_transferred": int(total_bytes),
        "disk_bytes_written_for_weights": 0,
        "shard_info": global_stats,
        "summary": summary,
        "rows": rows,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "requests": requests.__version__},
        "wall_time_s": round(time.time() - t_start, 1),
    }
    RESULTS.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    make_figure(rows)
    s = summary["all"]
    print(f"bytes {total_bytes}; mean rel_fro max {s['rel_fro_max_mean']:.5f} mse {s['rel_fro_mse_mean']:.5f}; "
          f"mse better {s['mse_better_tensors']}/{s['n_tensors']}")


def make_figure(rows: list[dict]) -> None:
    """왼쪽: 텐서별 상대 오차(Max vs MSE). 오른쪽: 블록 최댓값 원소의 오차와 상위 0.1% 원소의 scale 축소 비율."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1, 1.1]})
    mx = np.array([r["max"]["rel_fro"] for r in rows]) * 100
    mm = np.array([r["mse"]["rel_fro"] for r in rows]) * 100
    up = np.array([r["proj"] == "up_proj" for r in rows])
    a.scatter(mx[up], mm[up], s=28, color="#2b6cb0", label="up_proj")
    a.scatter(mx[~up], mm[~up], s=28, color="#dd6b20", label="down_proj")
    lim = [min(mx.min(), mm.min()) * 0.98, max(mx.max(), mm.max()) * 1.02]
    a.plot(lim, lim, color="#888", lw=1, ls="--")
    a.set_xlim(lim)
    a.set_ylim(lim)
    a.set_xlabel("Max scale: relative Frobenius error (%)")
    a.set_ylabel("MSE scale: relative Frobenius error (%)")
    a.set_title("Per tensor (24 expert weights)")
    a.legend(frameon=False, fontsize=9)

    labels = ["block-max\nmean rel. err", "block-max\np99 rel. err", "top 0.1% elements\nscale shrunk"]
    keys = ["block_max_mean_rel_err", "block_max_p99_rel_err", "outlier_top01_shrunk_frac"]
    mean_max = [np.mean([r["max"][k] for r in rows]) * 100 for k in keys]
    mean_mse = [np.mean([r["mse"][k] for r in rows]) * 100 for k in keys]
    x = np.arange(len(keys))
    b.bar(x - 0.18, mean_max, 0.36, color="#4a5568", label="Max")
    b.bar(x + 0.18, mean_mse, 0.36, color="#38a169", label="MSE")
    for i, (u, v) in enumerate(zip(mean_max, mean_mse)):
        b.text(i - 0.18, u, f"{u:.1f}", ha="center", va="bottom", fontsize=8)
        b.text(i + 0.18, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    b.set_xticks(x)
    b.set_xticklabels(labels, fontsize=9)
    b.set_ylabel("%")
    b.set_title("Outlier behaviour (mean over 24 tensors)")
    b.legend(frameon=False, fontsize=9)
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("NVFP4 emulation of one BF16 shard (layer 12 experts): Max vs MSE block scale", fontsize=10)
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=160)
    plt.close(fig)


def plot_only() -> None:
    """저장된 결과 JSON에서 그림만 다시 그린다(가중치는 받지 않는다)."""
    make_figure(json.loads(RESULTS.read_text())["rows"])


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot_only()
    else:
        main()
