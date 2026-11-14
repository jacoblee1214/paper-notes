"""여러 층의 expert 가중치에서 NVFP4 블록 scale 선택(Max 대 MSE)이 같은 방향인지 본다.

extra_nvfp4_scale.py의 양자화 함수를 가져와 쓴다. 층마다 expert 8개(up_proj, down_proj)를 HTTP Range로만 받는다.
원자료는 디스크에 쓰지 않는다. 층 12의 결과는 기존 결과 파일에서 가져온다(같은 방법, 같은 expert 12개).

결과 파일에는 통계만 남긴다(텐서·가중치 값 없음).
실행: python3 -I analysis/extra_nvfp4_scale_layers.py
"""
import hashlib
import json
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np
import requests
import scipy
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # python3 -I는 스크립트 폴더를 경로에 넣지 않으므로 직접 넣는다
from extra_nvfp4_scale import (BLOCK, UA, bf16_to_f32, quantize_max,  # noqa: E402
                               quantize_mse, stats_for)

RESULTS = HERE / "extra_nvfp4_scale_layers_results.json"
PREV = HERE / "extra_nvfp4_scale_results.json"
BF16_INDEX = HERE.parent / "paper" / "hf" / "bf16_index.json"
REPO = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"
PREFIX = "language_model.backbone.layers.{L}.mixer.experts.{e}.{proj}.weight"
NEW_LAYERS = [3, 30, 48, 65, 79, 87]
N_EXPERTS = 8
BUDGET_BYTES = 800 * 1024 * 1024  # 이번 실행의 새 다운로드 상한 (기존 132 MB와 합쳐 1 GB 미만)


def url_of(shard: str) -> str:
    return f"https://huggingface.co/{REPO}/resolve/main/{shard}"


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


def read_header(url: str) -> tuple[int, dict]:
    n = int(np.frombuffer(get_range(url, 0, 7), dtype="<u8")[0])
    hdr = json.loads(get_range(url, 8, 8 + n - 1))
    hdr.pop("__metadata__", None)
    return 8 + n, hdr


def pick_experts(index: dict, layer: int) -> list[int]:
    pat = re.compile(rf"language_model\.backbone\.layers\.{layer}\.mixer\.experts\.(\d+)\.up_proj\.weight")
    ids = sorted({int(m.group(1)) for k in index if (m := pat.fullmatch(k))})
    pick = [ids[i] for i in np.linspace(0, len(ids) - 1, N_EXPERTS).round().astype(int)]
    return pick


def main() -> None:
    t_start = time.time()
    index = json.loads(BF16_INDEX.read_text())["weight_map"]
    headers: dict[str, tuple[int, dict]] = {}
    plan = []
    planned_bytes = 0
    for layer in NEW_LAYERS:
        for e in pick_experts(index, layer):
            for proj in ("up_proj", "down_proj"):
                key = PREFIX.format(L=layer, e=e, proj=proj)
                shard = index[key]
                plan.append((layer, e, proj, key, shard))
    for layer, e, proj, key, shard in plan:
        if shard not in headers:
            headers[shard] = read_header(url_of(shard))
        info = headers[shard][1][key]
        assert info["dtype"] == "BF16", key
        planned_bytes += info["data_offsets"][1] - info["data_offsets"][0]
    print(f"planned tensors {len(plan)}, planned bytes {planned_bytes} (budget {BUDGET_BYTES})", flush=True)
    assert planned_bytes <= BUDGET_BYTES, "planned download exceeds the budget"

    rows = []
    total = sum(h[0] for h in headers.values())  # 헤더 크기(범위 요청 양)
    for layer, e, proj, key, shard in plan:
        url = url_of(shard)
        base, hdr = headers[shard]
        info = hdr[key]
        s, t = base + info["data_offsets"][0], base + info["data_offsets"][1] - 1
        raw = get_range(url, s, t)
        total += len(raw)
        W = bf16_to_f32(raw, info["shape"])
        sha = hashlib.sha256(raw).hexdigest()
        del raw
        X = W.reshape(-1, BLOCK)
        g = float(np.abs(W).max() / (6.0 * 448.0))
        Ymax, code_max, s_raw = quantize_max(X, g)
        Ymse, code_mse, _ = quantize_mse(X, g, s_raw, code_max)
        e_max = ((X - Ymax) ** 2).sum(1)
        e_mse = ((X - Ymse) ** 2).sum(1)
        assert np.all(e_mse <= e_max + 1e-12 * np.maximum(e_max, 1.0))
        blk_amax = np.abs(X).max(1)
        blk_rms = np.sqrt((X**2).mean(1)) + 1e-30
        rec = {
            "layer": layer, "expert": e, "proj": proj, "tensor": key, "shard": shard,
            "shape": info["shape"], "bytes": int(t - s + 1), "sha256_bytes": sha,
            "tensor_scale_fp32": g,
            "block_amax_over_rms_median": float(np.median(blk_amax / blk_rms)),
            "mse_code_below_max_frac": float(np.mean(code_mse < code_max)),
            "max": stats_for(W, Ymax.reshape(W.shape), X, Ymax),
            "mse": stats_for(W, Ymse.reshape(W.shape), X, Ymse),
        }
        rows.append(rec)
        print(f"L{layer} e{e} {proj}: max {rec['max']['rel_fro']:.4f} mse {rec['mse']['rel_fro']:.4f}", flush=True)

    # 층 12: 기존 결과(같은 방법으로 받은 12개 expert)를 가져온다
    prev = json.loads(PREV.read_text())
    prev_rows = [dict(r, layer=12) for r in prev["rows"]]
    all_rows = prev_rows + rows

    def arr(sel, method, key):
        return np.array([r[method][key] for r in sel])

    def block(sel):
        rmax = arr(sel, "max", "rel_fro")
        rmse = arr(sel, "mse", "rel_fro")
        return {
            "n_tensors": len(sel),
            "rel_fro_max_mean": float(rmax.mean()),
            "rel_fro_mse_mean": float(rmse.mean()),
            "mean_reduction_pct": float(100 * np.mean(1 - rmse / rmax)),
            "min_reduction_pct": float(100 * np.min(1 - rmse / rmax)),
            "mse_better_tensors": int(np.sum(rmse < rmax)),
            "sqnr_db_max_mean": float(arr(sel, "max", "sqnr_db").mean()),
            "sqnr_db_mse_mean": float(arr(sel, "mse", "sqnr_db").mean()),
            "block_max_mean_rel_err_max": float(arr(sel, "max", "block_max_mean_rel_err").mean()),
            "block_max_mean_rel_err_mse": float(arr(sel, "mse", "block_max_mean_rel_err").mean()),
            "outlier_top01_mean_rel_err_max": float(arr(sel, "max", "outlier_top01_mean_rel_err").mean()),
            "outlier_top01_mean_rel_err_mse": float(arr(sel, "mse", "outlier_top01_mean_rel_err").mean()),
            "outlier_top01_shrunk_frac_max": float(arr(sel, "max", "outlier_top01_shrunk_frac").mean()),
            "outlier_top01_shrunk_frac_mse": float(arr(sel, "mse", "outlier_top01_shrunk_frac").mean()),
        }

    per_layer = {}
    for L in sorted({r["layer"] for r in all_rows}):
        per_layer[str(L)] = block([r for r in all_rows if r["layer"] == L])
        per_layer[str(L)]["up_proj_mse_better"] = int(sum(1 for r in all_rows if r["layer"] == L and r["proj"] == "up_proj" and r["mse"]["rel_fro"] < r["max"]["rel_fro"]))
    per_proj = {p: block([r for r in all_rows if r["proj"] == p]) for p in ("up_proj", "down_proj")}
    overall = block(all_rows)
    overall["wilcoxon_mse_lt_max_p"] = float(stats.wilcoxon(arr(all_rows, "mse", "rel_fro"), arr(all_rows, "max", "rel_fro"), alternative="less").pvalue)
    layer_means = np.array([per_layer[str(L)]["mean_reduction_pct"] for L in sorted({r["layer"] for r in all_rows})])
    layer_mean_stats = {"n_layers": int(len(layer_means)), "min_layer_mean_reduction_pct": float(layer_means.min()),
                        "max_layer_mean_reduction_pct": float(layer_means.max()),
                        "layers_with_mse_better_all_tensors": int(sum(per_layer[str(L)]["mse_better_tensors"] == per_layer[str(L)]["n_tensors"] for L in sorted({r["layer"] for r in all_rows})))}

    results = {
        "script": "analysis/extra_nvfp4_scale_layers.py",
        "source": {"repo": REPO, "index": "paper/hf/bf16_index.json (local, from the same repo)",
                   "shards_used": sorted({r["shard"] for r in rows}),
                   "layer12_rows_from": "analysis/extra_nvfp4_scale_results.json (same method, shard 9)",
                   "note": "전체 shard는 받지 않았다. 각 텐서의 Range마다 sha256을 rows[].sha256_bytes에 기록했다."},
        "layers_new": NEW_LAYERS,
        "experts_per_layer": N_EXPERTS,
        "expert_ids_per_layer": {str(L): pick_experts(index, L) for L in NEW_LAYERS},
        "bytes_transferred_this_run": int(total),
        "bytes_transferred_previous_run": int(prev["bytes_transferred"]),
        "disk_bytes_written_for_weights": 0,
        "overall": overall,
        "per_proj": per_proj,
        "per_layer": per_layer,
        "layer_mean_stats": layer_mean_stats,
        "rows": [{k: v for k, v in r.items()} for r in all_rows],
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "requests": requests.__version__},
        "wall_time_s": round(time.time() - t_start, 1),
    }
    RESULTS.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(f"bytes this run {total}; overall mse better {overall['mse_better_tensors']}/{overall['n_tensors']}; "
          f"mean reduction {overall['mean_reduction_pct']:.2f}%", flush=True)


if __name__ == "__main__":
    main()
