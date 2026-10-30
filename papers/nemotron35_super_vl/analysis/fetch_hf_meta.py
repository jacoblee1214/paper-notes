"""Hugging Face 메타데이터를 다시 받는다 (가중치는 받지 않음). 실행: python3 analysis/fetch_hf_meta.py [출력 폴더]

- 두 저장소(BF16, NVFP4)의 config, generation_config, 색인, hf_quant_config, Hub API 응답
- BF16 shard 64개의 safetensors 헤더만 HTTP range 요청으로 읽어 텐서 dtype·모양을 bf16_shapes.json에 저장
기본 출력은 paper/hf/ (큰 파일은 .gitignore). 리비전을 고정해 분석 당시와 같은 파일을 받는다.
"""
import json
import struct
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "paper" / "hf"
REPOS = {"bf16": ("nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16", "fb0ee0bc6b3f70342afd8d84cb80dbadbaa8a1fe"),
         "nvfp4": ("nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4", "da5b2650a835292c678994a1fa3fb653250fb508")}
FILES = {"bf16": ["config.json", "generation_config.json", "preprocessor_config.json", "model.safetensors.index.json"],
         "nvfp4": ["config.json", "generation_config.json", "hf_quant_config.json", "model.safetensors.index.json"]}


def get(url, rng=None):
    req = urllib.request.Request(url, headers={"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for key, (repo, rev) in REPOS.items():
        api = get(f"https://huggingface.co/api/models/{repo}/revision/{rev}")
        (OUT / f"{repo.split('/')[1]}_api.json").write_bytes(api)
        for f in FILES[key]:
            name = f"{key}_" + ("index.json" if f.endswith("index.json") else f)
            (OUT / name).write_bytes(get(f"https://huggingface.co/{repo}/resolve/{rev}/{f}"))
            print("got", name)
    repo, rev = REPOS["bf16"]
    shards = sorted(set(json.load(open(OUT / "bf16_index.json"))["weight_map"].values()))
    shapes = {}
    for i, sh in enumerate(shards):
        url = f"https://huggingface.co/{repo}/resolve/{rev}/{sh}"
        n = struct.unpack("<Q", get(url, (0, 7)))[0]
        header = json.loads(get(url, (8, 8 + n - 1)))
        header.pop("__metadata__", None)
        shapes.update({k: [v["dtype"], v["shape"]] for k, v in header.items()})
        print(f"header {i + 1}/{len(shards)} {sh}: {len(header)} tensors")
    (OUT / "bf16_shapes.json").write_text(json.dumps(shapes))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
