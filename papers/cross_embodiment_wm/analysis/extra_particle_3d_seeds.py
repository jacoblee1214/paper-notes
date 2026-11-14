"""3차원 손 기하 집계 검사를 시드 3개로 반복: extra_particle_3d.py의 함수를 그대로 쓰고 시드만 바꾼다.

앞선 3차원 검사는 시드 1개였다. 여기서는 시드 0, 1, 2에 대해 같은 8개 셀(참 세계 2 × 학습 자료 2 × 집계 2)을
다시 돌리고, 시험 셀(손 C × 밀도 3)의 상대 MSE를 시드 평균 ± 표준편차로 보고한다.
시드는 에피소드 생성(rng_tr, rng_te)과 학습기 초기화·미니배치 순서를 함께 바꾼다. 손 기하와 참 세계 정의는 그대로다.

실행: python3 analysis/extra_particle_3d_seeds.py -> analysis/extra_particle_3d_seeds_results.json
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "4"

import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extra_particle_3d as base  # noqa: E402

HERE = Path(__file__).resolve().parent
SEEDS = [0, 1, 2]
TEST_HANDS = ["A_long2", "B_arm3", "C_short2"]
TRAINS = {"mixed_density": base.DENSITIES, "single_density_0.025": (0.025,)}


def main():
    t_all = time.perf_counter()
    models_data = {}
    for name, xml in base.HANDS.items():
        model = mujoco.MjModel.from_xml_string(xml)
        models_data[name] = (model, mujoco.MjData(model))

    per_seed = []
    for seed in SEEDS:
        cells = {}
        for world in ("S", "M"):
            for tname, tdens in TRAINS.items():
                rng_tr = np.random.default_rng(seed + 1)
                train_data = base.build_set(rng_tr, ["A_long2", "B_arm3"], tdens, world, models_data)
                key = f"world_{world}__train_{tname}"
                cells[key] = {}
                for agg in ("sum", "mean"):
                    net, wall, last = base.train_model(agg, train_data, seed)
                    test = {}
                    for hname in TEST_HANDS:
                        for h in base.DENSITIES:
                            rng_te = np.random.default_rng(seed + 7)
                            te = base.build_set(rng_te, [hname], (h,), world, models_data)
                            test[f"{hname}@{h}"] = base.rel_mse(net, te)
                    cells[key][agg] = {"train_wall_s": round(wall, 2), "final_train_loss_rel": last, "test": test}
                    print(f"seed {seed} world {world} train {tname:22s} agg {agg:4s} wall {wall:5.1f}s "
                          f"C_short2 {[round(test[f'C_short2@{h}'], 3) for h in base.DENSITIES]}", flush=True)
        per_seed.append(cells)

    summary = {}
    for key in per_seed[0]:
        summary[key] = {}
        for agg in ("sum", "mean"):
            summary[key][agg] = {}
            for hname in TEST_HANDS:
                for h in base.DENSITIES:
                    k = f"{hname}@{h}"
                    vals = [s[key][agg]["test"][k] for s in per_seed]
                    summary[key][agg][k] = {"mean": round(float(np.mean(vals)), 4),
                                            "sd": round(float(np.std(vals, ddof=1)), 4),
                                            "per_seed": [round(v, 4) for v in vals]}

    results = {"seeds": SEEDS, "summary_rel_mse": summary, "per_seed_cells": per_seed,
               "setup": dict(densities=list(base.DENSITIES), n_obj=base.N_OBJ, r_contact=base.R_CONTACT,
                             n_ep_per_cell=base.N_EP, train_steps=base.TRAIN_STEPS, batch=base.BATCH, lr=base.LR,
                             train_hands=["A_long2", "B_arm3"], test_hand="C_short2", learner="pair kernel MLP"),
               "versions": dict(torch=torch.__version__, mujoco=mujoco.__version__, numpy=np.__version__,
                                python=sys.version.split()[0]),
               "device": "cpu", "threads": torch.get_num_threads(),
               "total_wall_s": round(time.perf_counter() - t_all, 2)}
    (HERE / "extra_particle_3d_seeds_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("total wall", results["total_wall_s"], "s")


if __name__ == "__main__":
    main()
