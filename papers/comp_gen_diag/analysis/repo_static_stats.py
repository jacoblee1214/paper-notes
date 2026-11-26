"""공식 저장소 `projects/corl2026_cg/configs/*.json` 프리셋의 설정값을 표로 모은다 (정적 읽기, 저장소 코드는 실행하거나 import하지 않는다).

필요한 것: repo/ = 공식 저장소의 얕은 복제.
  git clone --depth 1 https://github.com/YixiaoWang7/Diagnosing_Compositional_Generalization_Robot repo
  git -C repo fetch --depth 1 origin e29c44474495b341d19636bbf90f3cf0fcb1d2d2 && git -C repo checkout e29c44474495b341d19636bbf90f3cf0fcb1d2d2
출력: analysis/repo_static_stats_results.json
실행: python3 analysis/repo_static_stats.py   (표준 라이브러리만)
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONF = ROOT / "repo" / "projects" / "corl2026_cg" / "configs"
OUT = Path(__file__).resolve().parent / "repo_static_stats_results.json"


def main():
    res = {}
    for p in sorted(CONF.glob("*.json")):
        c = json.loads(p.read_text(encoding="utf-8"))
        run = c.get("run", {})
        sel = run.get("dataset", {}).get("selection", {})
        ev = c.get("evaluation", {})
        head = run.get("policy", {}).get("head", {})
        res[p.stem] = {
            "family": c.get("family"),
            "instruction": c.get("instruction"),
            "seeds": c.get("seeds"),
            "sets": c.get("sets") if len(json.dumps(c.get("sets"))) < 300 else "(긴 목록)",
            "max_updates": run.get("training", {}).get("max_updates"),
            "batch_size": run.get("loader", {}).get("batch_size"),
            "max_demos": sel.get("max_demos"),
            "max_demos_per_task": sel.get("max_demos_per_task"),
            "balance_demos_across_tasks": sel.get("balance_demos_across_tasks"),
            "n_trials_per_task": ev.get("n_trials_per_task", 25),
            "horizon": ev.get("horizon", 400),
            "head_layers": head.get("layers"),
            "matrix": c.get("matrix"),
            "finetune": bool(c.get("finetune_from")),
        }
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(res)} presets ->", OUT)


if __name__ == "__main__":
    main()
