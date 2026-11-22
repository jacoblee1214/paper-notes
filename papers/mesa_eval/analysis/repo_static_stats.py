"""공식 저장소(repo/)를 정적으로만 읽어 평가 BDDL의 구조 통계를 센다. 저장소 코드는 import하거나 실행하지 않는다.

읽는 것: repo/mesa/task_suites/task_sets.py (ast로 파싱, 실행 안 함), repo/mesa/task_suites/bddl_files/*/<과제>/eval/*.json
출력: analysis/repo_static_stats_results.json
실행: python3 analysis/repo_static_stats.py   (repo/ 폴더가 있어야 함. 표준 라이브러리만 사용)
"""
import ast
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "repo" / "mesa" / "task_suites"


def lit(n):
    try:
        return ast.literal_eval(n)
    except Exception:
        if isinstance(n, ast.BinOp):
            a, b = lit(n.left), lit(n.right)
            return a + b if isinstance(n.op, ast.Add) else a * b
        raise


def load_sets():
    tree = ast.parse((REPO / "task_sets.py").read_text())
    out = {}
    for n in tree.body:
        if isinstance(n, ast.Assign):
            d = {}
            for k, v in zip(n.value.keys, n.value.values):
                d[k.value] = {kk.value: lit(vv) for kk, vv in zip(v.keys, v.values)}
            out[n.targets[0].id] = d
    return out


def stats_for(suite_dir, tasks):
    n_inst, n_spec = [], []
    tgt_obj_specs, self_specs, dest_as_obj = [], [], []
    n_objects, n_fix = [], []
    missing = 0
    n_files = 0
    for t in tasks:
        d = REPO / "bddl_files" / suite_dir / t / "eval"
        files = sorted(d.glob("*.json")) if d.exists() else []
        if not files:
            missing += 1
        n_inst.append(len(files))
        for f in files:
            n_files += 1
            j = json.loads(f.read_text())
            specs = j.get("distractor_goal_specs", [])
            goal_objs = {s[1] for s in j["goal_state"][1:] if len(s) == 3}
            n_spec.append(len(specs))
            tgt_obj_specs.append(sum(1 for s in specs if s[1] in goal_objs))
            self_specs.append(sum(1 for s in specs if len(s) == 3 and s[1] == s[2]))
            n_objects.append(sum(len(v) for v in j["objects"].values()))
            n_fix.append(len(j["fixtures"]))
    mean = lambda x: round(sum(x) / len(x), 2) if x else None
    return {"tasks": len(tasks), "tasks_without_eval_dir": missing, "eval_files": n_files,
            "instances_per_task": dict(Counter(n_inst)),
            "mean_distractor_goal_specs": mean(n_spec), "min": min(n_spec) if n_spec else None, "max": max(n_spec) if n_spec else None,
            "mean_specs_moving_target_object": mean(tgt_obj_specs), "mean_self_referential_specs": mean(self_specs),
            "mean_objects": mean(n_objects), "mean_fixtures": mean(n_fix),
            "chance_lfr_if_uniform_over_target_plus_specs_pct": round(100 / (1 + (sum(n_spec) / len(n_spec))), 1) if n_spec else None}


def main():
    sets = load_sets()
    ev, tr = sets["EVAL_SETS"], sets["TRAIN_SETS"]
    res = {"eval_set_task_counts": {k: len(v["tasks"]) for k, v in ev.items()},
           "train_set_task_counts": {k: len(v["tasks"]) for k, v in tr.items()},
           "train_mesa70_demos_sum": sum(tr["mesa-70"]["demos_per_task"]),
           "train_mesa70_tasks_not_in_eval70": sorted(set(tr["mesa-70"]["tasks"]) - set(ev["mesa-70"]["tasks"])),
           "aux_in_eval_sets": {k: len(set(tr["mesa-aux"]["tasks"]) & set(v["tasks"])) for k, v in ev.items()},
           "mesa_all_in_eval_sets": {k: len(set(tr["mesa-all"]["tasks"]) & set(v["tasks"])) for k, v in ev.items()},
           "per_suite": {}}
    for name, v in ev.items():
        if name == "lime-overfit":
            continue
        res["per_suite"][name] = stats_for(v["task_suite_name"], v["tasks"])
    (ROOT / "analysis" / "repo_static_stats_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in res["per_suite"].items():
        print(k, v["tasks"], "specs", v["mean_distractor_goal_specs"], "target-moving", v["mean_specs_moving_target_object"],
              "chance LFR", v["chance_lfr_if_uniform_over_target_plus_specs_pct"], v["instances_per_task"])
    print("train mesa-70 demos", res["train_mesa70_demos_sum"], "extra", res["train_mesa70_tasks_not_in_eval70"])
    print("aux in eval", res["aux_in_eval_sets"])
    print("all in eval", res["mesa_all_in_eval_sets"])


if __name__ == "__main__":
    main()
