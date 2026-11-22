"""Toy discrete-event model of Armory's closed loop (paper Sec. 3, Appendix A.1/A.4/A.9).

N robots tick at f_c, each executes one queued action per tick or starves. One GPU serves one batch at a
time with the profiled pi0.5 latency table (paper Table 1 / the code's profile table). A chunk covers
Hmax action indices starting at the robot's next index at dispatch time and replaces the queued indices
>= its start when it arrives (stale indices are skipped). Schedulers: RR, EDF (fill to cap), LA (L=1,
one-step score = weighted executed steps credited to the candidate chunk inside a 1 s window divided by
the batch latency; candidate batches = EDF prefixes per weight tier, as in the public code).

This is an independent re-implementation of the paper's description, not the authors' code.
It does not model the policy or task success: outputs are starvation rates and executed-step rates.

Usage: python3 toy_mirror_sim.py [out.json]   (single process, about a minute on CPU)
"""
import json
import math
import sys
import time

import numpy as np

from paper_tables import T4, T4_SCHED

L40S = {1: .0730, 2: .1098, 3: .1424, 4: .1775, 5: .2111, 6: .2422, 7: .2753, 8: .3169, 9: .3453,
        10: .3752, 11: .4047, 12: .4342, 13: .4665, 14: .4904, 15: .5347, 16: .5763, 17: .6142,
        18: .6405, 19: .6847, 20: .7191}
FC = 20
US = 1_000_000


class Robot:
    def __init__(self, hmax, hmin, w, phase_us):
        self.hmax, self.hmin, self.w = hmax, hmin, w
        self.n = 0            # next action index to execute
        self.hi = -1          # last available (arrived) index
        self.pend_hi = -1     # last index of an in-flight chunk
        self.last_start = -10**9
        self.next_tick = phase_us
        self.executed = 0
        self.starved = 0
        self.fast = False


def compositions(n, k, caps):
    if k == 1:
        if n <= caps[0]:
            yield (n,)
        return
    for i in range(min(n, caps[0]) + 1):
        for rest in compositions(n - i, k - 1, caps[1:]):
            yield (i,) + rest


def la_candidates(order, robots, cap):
    """Candidate batches in the order of the public code (_candidate_batches)."""
    tiers = {}
    for j in order:
        tiers.setdefault(robots[j].w, []).append(j)
    tl = [tiers[p] for p in sorted(tiers, reverse=True)]
    sizes = [len(t) for t in tl]
    out, seen = [], set()
    for size in range(1, min(len(order), cap) + 1):
        for counts in compositions(size, len(tl), sizes):
            b = tuple(j for t, c in zip(tl, counts) for j in t[:c])
            if b not in seen:
                seen.add(b)
                out.append(b)
    return out


def simulate(n_rob, fast_mask, sched, cap, seed, dur_s=120.0, fast_h=6, slow_h=10, hmin=1,
             w_fast=1.0, net_s=0.005, budget=None, lat=L40S, return_batches=False, reading="A"):
    rng = np.random.default_rng(seed)
    per = US // FC
    rob = []
    for j in range(n_rob):
        f = fast_mask[j]
        r = Robot(fast_h if f else slow_h, hmin, w_fast if f else 1.0, int(rng.integers(0, per)))
        r.fast = f
        rob.append(r)
    gpu_free = 0
    arrivals = []        # (arr_us, robot, start, hmax)
    rr_ptr = 0
    end = int(dur_s * US)
    warm = int(5 * US)
    meas = {"ex": [0] * n_rob, "st": [0] * n_rob}
    t = 0
    bsizes = []
    net = int(net_s * US)
    while t < end:
        # next event time
        cand = [min(r.next_tick for r in rob)]
        if arrivals:
            cand.append(min(a[0] for a in arrivals))
        if gpu_free > t:
            cand.append(gpu_free)
        t = min(cand)
        if t >= end:
            break
        # arrivals
        keep = []
        for a in arrivals:
            if a[0] <= t:
                _, j, s, hm = a
                r = rob[j]
                new_hi = s + hm - 1
                r.hi = new_hi if new_hi >= r.n else r.n - 1
            else:
                keep.append(a)
        arrivals = keep
        # ticks
        for j, r in enumerate(rob):
            if r.next_tick <= t:
                if r.n <= r.hi:
                    r.n += 1
                    if t >= warm:
                        meas["ex"][j] += 1
                else:
                    if t >= warm:
                        meas["st"][j] += 1
                r.next_tick += per
        # dispatch
        if gpu_free <= t:
            cands = [j for j, r in enumerate(rob) if r.n >= r.last_start + r.hmin]
            if not cands:
                continue

            def deadline(j):
                r = rob[j]
                eff = max(r.hi, r.pend_hi)
                return t + max(0, eff - r.n + 1) * per

            order = sorted(cands, key=deadline)
            if sched == "edf":
                batch = order[:cap]
            elif sched == "rr":
                batch = []
                cs = set(cands)
                idx = rr_ptr
                for _ in range(n_rob):
                    if idx % n_rob in cs:
                        batch.append(idx % n_rob)
                    idx += 1
                    if len(batch) == cap:
                        break
                rr_ptr = idx % n_rob
            else:  # la
                cl = la_candidates(order, rob, cap)
                if budget is not None:
                    cl = cl[:budget]
                best, bo = None, -1.0
                for b in cl:
                    inf = lat[len(b)]
                    tot = 0.0
                    for j in b:
                        r = rob[j]
                        arr = t + int((inf + net_s) * US)
                        ticks = (arr - t) / per
                        eff = max(r.hi, r.pend_hi)
                        n_arr = min(r.n + ticks, eff + 1)
                        rem = max(0.0, r.n + r.hmax - max(n_arr, r.n))
                        win = max(0.0, (t + US - arr) / per)
                        gain = min(rem, math.floor(win))
                        if reading == "B":
                            # marginal reading: executed steps in the 1 s window with the new chunk
                            # minus executed steps if the robot is not served now
                            rem_old = max(0.0, eff - r.n + 1)
                            x1 = min(ticks, rem_old)
                            e_new = math.floor(x1) + gain
                            e_old = min(rem_old, FC * 1.0)
                            gain = e_new - e_old
                        tot += r.w * gain / FC
                    o = tot / inf
                    if best is None or o > bo:
                        bo, best = o, b
                batch = list(best)
            inf_us = int(lat[len(batch)] * US)
            gpu_free = t + inf_us
            bsizes.append(len(batch))
            for j in batch:
                r = rob[j]
                s = r.n
                r.last_start = s
                r.pend_hi = s + r.hmax - 1
                arrivals.append((gpu_free + net, j, s, r.hmax))
    ex, st = np.array(meas["ex"], float), np.array(meas["st"], float)
    tot = ex + st
    fm = np.array(fast_mask)
    res = {
        "starv": float(st.sum() / tot.sum() * 100),
        "exec_per_s": float(ex.sum() / (dur_s - 5)),
        "starv_fast": float(st[fm].sum() / tot[fm].sum() * 100) if fm.any() else None,
        "starv_slow": float(st[~fm].sum() / tot[~fm].sum() * 100) if (~fm).any() else None,
        "exec_fast_per_s": float(ex[fm].sum() / (dur_s - 5)) if fm.any() else None,
        "exec_slow_per_s": float(ex[~fm].sum() / (dur_s - 5)) if (~fm).any() else None,
        "mean_batch": float(np.mean(bsizes)) if bsizes else None,
        "median_batch": float(np.median(bsizes)) if bsizes else None,
    }
    if return_batches:
        res["batches"] = bsizes
    return res


def masks(shape, n):
    if shape == "all_fast":
        return [True] * n
    if shape == "one_fast":
        return [True] + [False] * (n - 1)
    return [True] * (n // 2) + [False] * (n - n // 2)


def avg(rs, keys):
    return {k: (float(np.mean([r[k] for r in rs])) if rs[0][k] is not None else None) for k in keys}


KEYS = ["starv", "starv_fast", "starv_slow", "exec_per_s", "exec_fast_per_s", "exec_slow_per_s", "mean_batch"]


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "toy_mirror_sim_results.json"
    t0 = time.time()
    seeds = [0, 1, 2]
    results = {"setup": {"fc": FC, "fast_h": 6, "slow_h": 10, "hmin": 1, "cap": 3, "net_one_way_s": 0.005,
                         "dur_s": 120, "seeds": seeds, "latency_table": "pi0.5 on L40S (paper Table 1)",
                         "reading_A": "LA score credits only steps executed from the candidate chunks (literal code reading)",
                         "reading_B": "LA score = marginal gain in executed steps inside the 1 s window"}}
    variants = [("rr", dict(sched="rr")), ("edf", dict(sched="edf"))]
    for rd in ["A", "B"]:
        for w in [1, 3, 5]:
            variants.append((f"la{rd}_w{w}", dict(sched="la", w_fast=float(w), reading=rd)))
    rows = []
    for shape in ["all_fast", "one_fast", "half_fast"]:
        for n in [2, 4, 6, 8, 10]:
            row = {"shape": shape, "N": n}
            for name, kw in variants:
                rs = [simulate(n, masks(shape, n), cap=3, seed=s, **kw) for s in seeds]
                row[name] = avg(rs, KEYS)
            rows.append(row)
        print(shape, round(time.time() - t0, 1), "s", flush=True)
    results["table4_like"] = rows
    # mean absolute error of average starvation vs paper Table 4
    pair = {"rr": "rr", "edf": "edf", "laA_w1": "la1", "laA_w3": "la3", "laA_w5": "la5",
            "laB_w1": "la1", "laB_w3": "la3", "laB_w5": "la5"}
    mae = {}
    for name, p in pair.items():
        errs = []
        for row in rows:
            ref = T4[row["shape"]][row["N"]][T4_SCHED.index(p)][1]
            errs.append(abs(row[name]["starv"] - ref))
        mae[name] = {"all": float(np.mean(errs))}
        for shape in ["all_fast", "one_fast", "half_fast"]:
            e = [abs(r[name]["starv"] - T4[r["shape"]][r["N"]][T4_SCHED.index(p)][1]) for r in rows if r["shape"] == shape]
            mae[name][shape] = float(np.mean(e))
    results["mae_starvation_vs_table4_pp"] = mae
    # batch size distribution, all-fast N=10 cap=10 (paper Fig. 7 left)
    bs = {}
    for name, kw in [("rr", dict(sched="rr")), ("edf", dict(sched="edf")), ("la", dict(sched="la"))]:
        allb = []
        for s in seeds:
            allb += simulate(10, masks("all_fast", 10), cap=10, seed=s, return_batches=True, **kw)["batches"]
        bs[name] = {"mean": float(np.mean(allb)), "median": float(np.median(allb))}
    results["fig7_batch_size_cap10"] = bs
    caps = {}
    for cap in [1, 2, 3, 4, 5, 6, 8, 10]:
        caps[cap] = {}
        for name, kw in [("rr", dict(sched="rr")), ("edf", dict(sched="edf")), ("la", dict(sched="la"))]:
            rs = [simulate(10, masks("all_fast", 10), cap=cap, seed=s, **kw) for s in seeds]
            caps[cap][name] = {"starv": float(np.mean([r["starv"] for r in rs])),
                               "exec_per_s": float(np.mean([r["exec_per_s"] for r in rs]))}
    results["cap_sweep_allfast_N10"] = caps
    # noise probe: the same algorithm with different seeds (all-fast LA is weight-invariant)
    noise = [simulate(10, masks("all_fast", 10), cap=3, seed=s, sched="la")["exec_per_s"] for s in range(10)]
    results["seed_noise_allfast_N10_exec_per_s"] = {"mean": float(np.mean(noise)), "sd": float(np.std(noise, ddof=1))}
    # candidate-truncation probe: only the first 8 candidate batches evaluated (step_budget_nodes=8)
    trunc = {}
    for shape in ["one_fast", "half_fast"]:
        for rd in ["A", "B"]:
            for label, bud in [("full", None), ("first8", 8)]:
                rs = [simulate(10, masks(shape, 10), cap=5, seed=s, sched="la", w_fast=5.0, budget=bud, reading=rd)
                      for s in seeds]
                trunc[f"{shape}_{rd}_{label}"] = avg(rs, KEYS)
    results["truncation_probe_cap5_w5_N10"] = trunc
    results["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(results, open(out, "w"), indent=1)
    print("done", results["elapsed_s"], "s")


if __name__ == "__main__":
    main()
