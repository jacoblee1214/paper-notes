"""공개 GGUF 파일의 헤더(메타데이터와 텐서 목록)만 읽어 텐서별 저장 형식과 크기를 집계한다 (CPU, 가중치는 받지 않음).

실행: python3 analysis/gguf_header_check.py
- 헤더가 없으면 Hugging Face에서 HTTP Range로 앞부분만 받는다(본 모델 24 MB, mmproj 8 MB). 받은 파일은 .scratch/(git 제외)에 둔다.
- 텐서 크기는 데이터 영역 안의 오프셋 차이로 구한다. 형식 ID를 몰라도 크기는 정확하다.
- 결과: analysis/gguf_header_results.json (Table 2 구성 요소별 크기와 대조)
"""
import json
import os
import struct
import subprocess
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / ".scratch"
BASE = "https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF/resolve/main/"
FILES = {
    "main": ("NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf", 25165824, 22018326528),
    "mmproj": ("mmproj-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf", 8388608, 1733689408),
}
# 24 GiB 맥 판단용: 더 작은 커뮤니티 양자화와 E4B(전체 URL, 헤더 크기, 파일 크기). 2026-10-30 HF API 기준 크기
EXTRA = {
    "unsloth_UD-IQ1_S": ("https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-GGUF/resolve/main/"
                         "NVIDIA-Nemotron-3.5-Bolt-74B-A9B-UD-IQ1_S.gguf", 25165824, 18507811840),
    "e4b_Q4_K_M": ("https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB/resolve/main/"
                   "Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf", 16777216, 6180446208),
    # 초안 모델: 아키텍처 이름과 LiLiCorr 헤드 텐서가 실제로 들어 있는지 (04 D4, D8)
    "drafter": ("https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr-GGUF/resolve/main/"
                "NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr.gguf", 16777216, 998242560),
}
# ggml_type 번호 (llama.cpp ggml.h). 모르는 번호는 그대로 적는다.
GGML = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K",
        11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
        19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32",
        27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4"}


class Reader:
    def __init__(self, buf: bytes):
        self.b, self.p = buf, 0

    def take(self, fmt):
        v = struct.unpack_from("<" + fmt, self.b, self.p)
        self.p += struct.calcsize("<" + fmt)
        return v[0]

    def string(self):
        n = self.take("Q")
        s = self.b[self.p:self.p + n].decode("utf-8", "replace")
        self.p += n
        return s

    def value(self, t):
        scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
        if t in scalar:
            return self.take(scalar[t])
        if t == 8:
            return self.string()
        if t == 9:
            et, n = self.take("I"), self.take("Q")
            if et == 8:
                return [self.string() for _ in range(n)] if n <= 64 else (self.skip_strings(n), f"<{n} strings>")[1]
            if et in scalar:
                sz = struct.calcsize("<" + scalar[et])
                vals = struct.unpack_from("<" + scalar[et] * n, self.b, self.p)
                self.p += sz * n
                return list(vals) if n <= 128 else f"<{n} values>"
            raise ValueError(f"array type {et}")
        raise ValueError(f"kv type {t}")

    def skip_strings(self, n):
        for _ in range(n):
            length = self.take("Q")  # take()가 p를 먼저 옮기므로 따로 받아서 더한다
            self.p += length


def fetch(name, nbytes, url=None):
    SCRATCH.mkdir(exist_ok=True)
    out = SCRATCH / f"{name}_head.bin"
    if not out.exists() or out.stat().st_size < nbytes:
        url = url or BASE + FILES[name][0]
        subprocess.run(["curl", "-sL", "--max-time", "180", "-r", f"0-{nbytes - 1}", url, "-o", str(out)], check=True)
    return out.read_bytes()


def parse(buf, file_size):
    r = Reader(buf)
    assert r.take("4s") == b"GGUF"
    version, n_t, n_kv = r.take("I"), r.take("Q"), r.take("Q")
    kv = {}
    for _ in range(n_kv):
        k = r.string()
        kv[k] = r.value(r.take("I"))
    tensors = []
    for _ in range(n_t):
        name = r.string()
        nd = r.take("I")
        dims = [r.take("Q") for _ in range(nd)]
        t = r.take("I")
        off = r.take("Q")
        tensors.append({"name": name, "dims": dims, "type_id": t, "type": GGML.get(t, f"type{t}"), "offset": off})
    align = kv.get("general.alignment", 32)
    data_start = (r.p + align - 1) // align * align
    order = sorted(range(n_t), key=lambda i: tensors[i]["offset"])
    data_len = file_size - data_start
    for j, i in enumerate(order):
        nxt = tensors[order[j + 1]]["offset"] if j + 1 < n_t else data_len
        tensors[i]["bytes"] = nxt - tensors[i]["offset"]
        ne = 1
        for d in tensors[i]["dims"]:
            ne *= d
        tensors[i]["n"] = ne
        tensors[i]["bpw"] = 8 * tensors[i]["bytes"] / ne
    return version, kv, tensors, data_start


def component(name):
    n = name
    if n.startswith("token_embd"):
        return "Token embedding"
    if n.startswith("output.") or n == "output.weight":
        return "LM head"
    if "output_norm" in n or n.endswith("_norm.weight") or ".norm" in n:
        return "norms"
    if "ffn_gate_inp" in n or "exp_probs_b" in n:
        return "MoE router weights"
    if "_shexp" in n:
        return "Shared-expert projections"
    if "_exps" in n:
        return "Routed-expert projections"
    if "latent" in n:
        return "Latent MoE projections"
    if "ssm_conv1d" in n:
        return "Mamba convolution weights"
    if "ssm_in" in n or "ssm_out" in n:
        return "Mamba input/output projections"
    if n.split(".")[-2] in ("attn_q", "attn_k", "attn_v", "attn_output"):
        return "Attention projections"
    if "ssm_" in n:
        return "Mamba small params (A, D, dt, norm)"
    return "other:" + ".".join(n.split(".")[2:4]) if n.startswith("blk.") else "other:" + n


TABLE2_GB = {"Attention projections": 0.30, "Routed-expert projections": 17.05, "Shared-expert projections": 0.45,
             "Latent MoE projections": 0.36, "Mamba input/output projections": 2.47, "Token embedding": 0.57,
             "LM head": 0.57, "MoE router weights": 0.23, "Mamba convolution weights": 0.01}


def main():
    out = {}
    for key, (fname, head, size) in FILES.items():
        version, kv, tensors, data_start = parse(fetch(key, head), size)
        by_comp = defaultdict(lambda: {"bytes": 0, "n": 0, "types": defaultdict(int)})
        for t in tensors:
            c = component(t["name"]) if key == "main" else ("vision/projector")
            by_comp[c]["bytes"] += t["bytes"]
            by_comp[c]["n"] += t["n"]
            by_comp[c]["types"][t["type"]] += 1
        comp = {c: {"GB": round(v["bytes"] / 1e9, 4), "params_B": round(v["n"] / 1e9, 4),
                    "bpw": round(8 * v["bytes"] / v["n"], 4) if v["n"] else None,
                    "types": dict(v["types"]), "table2_GB": TABLE2_GB.get(c)} for c, v in sorted(by_comp.items())}
        small_kv = {k: v for k, v in kv.items() if not (isinstance(v, str) and len(v) > 200)}
        out[key] = {
            "file": fname, "file_bytes": size, "gguf_version": version, "tensor_count": len(tensors),
            "header_bytes": data_start,
            "total_params": sum(t["n"] for t in tensors),
            "data_GB": round(sum(t["bytes"] for t in tensors) / 1e9, 4),
            "components": comp,
            "type_counts": dict(sorted(defaultdict(int, {}).items())),
            "kv_selected": {k: small_kv[k] for k in sorted(small_kv) if any(s in k for s in (
                "architecture", "block_count", "context_length", "embedding_length", "feed_forward", "expert",
                "head_count", "ssm", "rope", "vocab_size", "file_type", "quantization", "attention", "clip",
                "vision", "projector", "image_size", "patch"))},
        }
        tc = defaultdict(int)
        for t in tensors:
            tc[t["type"]] += 1
        out[key]["type_counts"] = dict(tc)
        out[key]["unknown_type_examples"] = sorted({(t["type"], round(t["bpw"], 4)) for t in tensors if t["type"].startswith("type")})
    # 맥 Metal 판단: llama.cpp master(889f2ec, 2026-10-30)의 ggml_metal_supports_mul_mat_op은 NVFP4(type 40)를 거부한다
    for key, (url, head, size) in EXTRA.items():
        version, kv, tensors, data_start = parse(fetch(key, head, url), size)
        tc = defaultdict(int)
        nv = defaultdict(int)
        for t in tensors:
            tc[t["type"]] += 1
            if t["type"] == "type40":
                nv[component(t["name"])] += t["bytes"]
        out[key] = {"file_bytes": size, "tensor_count": len(tensors), "total_params": sum(t["n"] for t in tensors),
                    "architecture": kv.get("general.architecture"), "type_counts": dict(tc),
                    "nvfp4_GB_by_component": {k: round(v / 1e9, 4) for k, v in nv.items()}}
        if key == "drafter":
            out[key]["lilicorr_tensors"] = sum("lilicorr" in t["name"] for t in tensors)
            out[key]["kv_selected"] = {k: v for k, v in sorted(kv.items()) if k.startswith(("dflash.", "lilicorr."))
                                       and not (isinstance(v, str) and len(v) > 200)}
    (ROOT / "analysis" / "gguf_header_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    m = out["main"]
    print("main: tensors", m["tensor_count"], "params", m["total_params"], "data GB", m["data_GB"], "types", m["type_counts"])
    for c, v in m["components"].items():
        print(f"  {c:38s} {v['GB']:8.4f} GB  {v['params_B']:8.4f} B  bpw {v['bpw']}  table2 {v['table2_GB']}  {v['types']}")
    print("mmproj:", out["mmproj"]["tensor_count"], out["mmproj"]["total_params"], out["mmproj"]["data_GB"], out["mmproj"]["type_counts"])
    print("unknown:", m["unknown_type_examples"])
    for key in EXTRA:
        e = out[key]
        print(key, e["architecture"], e["tensor_count"], e["total_params"], e["type_counts"], "NVFP4:", e["nvfp4_GB_by_component"])


if __name__ == "__main__":
    main()
