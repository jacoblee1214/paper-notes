"""공개 저장소·모델·데이터셋의 메타데이터만 받는다 (가중치와 코드 실행 없음).

실행: python3 analysis/fetch_meta.py  →  analysis/meta/*.json, analysis/meta/gr00t_n1d7_config.json
받는 것: GitHub REST 요약, Hugging Face API 요약, N1.7 config.json(2 KB), Human-1M 데이터셋 카드, arXiv 검색 결과 수.
"""
import json
import urllib.parse
import urllib.request
from pathlib import Path

META = Path(__file__).resolve().parent / "meta"
META.mkdir(exist_ok=True)
UA = {"User-Agent": "research-roadmap-metadata-check/1.0"}


def get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=40) as r:
        return r.read()


def pick_github(repo):
    d = json.loads(get(f"https://api.github.com/repos/{repo}"))
    lic = (d.get("license") or {}).get("spdx_id")
    rel = json.loads(get(f"https://api.github.com/repos/{repo}/releases?per_page=5"))
    com = json.loads(get(f"https://api.github.com/repos/{repo}/commits?per_page=1"))[0]
    return {
        "repo": repo, "stars": d["stargazers_count"], "forks": d["forks_count"], "license": lic,
        "pushed_at": d["pushed_at"], "description": d.get("description"),
        "latest_commit": com["sha"][:7], "latest_commit_date": com["commit"]["committer"]["date"],
        "releases": [(r["tag_name"], r["published_at"][:10]) for r in rel],
    }


def pick_hf(kind, rid):
    d = json.loads(get(f"https://huggingface.co/api/{kind}/{rid}"))
    card = d.get("cardData") or {}
    return {"id": rid, "sha": d.get("sha", "")[:7], "lastModified": d.get("lastModified", "")[:10],
            "likes": d.get("likes"), "downloads": d.get("downloads"), "license": card.get("license"),
            "license_name": card.get("license_name"), "gated": d.get("gated")}


def search_hf(q):
    rows = json.loads(get(f"https://huggingface.co/api/models?search={urllib.parse.quote(q)}&limit=50"))
    return sorted(r["id"] for r in rows if "gr00t" in r["id"].lower() and "2" in r["id"].lower())


out = {}
out["github"] = [pick_github("NVIDIA/Isaac-GR00T")]
out["hf"] = [pick_hf("models", "nvidia/GR00T-N1.7-3B"), pick_hf("models", "nvidia/GR00T-N1.6-3B"),
             pick_hf("datasets", "nvidia/PhysicalAI-Human-1M-100K")]
out["hf_search_gr00t2_like"] = {q: search_hf(q) for q in ("GR00T-2", "GR00T-N2")}
arx = get("https://export.arxiv.org/api/query?search_query=ti:%22GR00T%202%22&max_results=5").decode()
out["arxiv_title_hits"] = arx.count("<entry>")
(META / "gr00t2_meta.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
get_cfg = get("https://huggingface.co/nvidia/GR00T-N1.7-3B/raw/main/config.json")
(META / "gr00t_n1d7_config.json").write_bytes(get_cfg)
(META / "human1m_README.md").write_bytes(
    get("https://huggingface.co/datasets/nvidia/PhysicalAI-Human-1M-100K/raw/main/README.md"))
print(json.dumps(out, indent=1, ensure_ascii=False))
