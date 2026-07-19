"""
Offline-oracle probe promised in Sec. IV-F: on small instances extracted
from the real Seoul V2X trace, compare FACE's realized P2 objective with a
hindsight-optimal plan computed by an ILP that sees the whole contact
sequence in advance.

Instance construction (quasi-static): at a window start w0 the predicted
rewards v-hat_{j,x} (the exact FACE-full formula), the holdings, and the
encoder sizes are frozen; the instance consists of the matched
sender-receiver pairs FACE actually formed during [w0, w0+W) restricted to
a small vehicle subset. Both the oracle and FACE are scored on the same
frozen table and the same contacts:

    P2 objective = sum_{first useful deliveries (j,x)} v-hat_{j,x}
                   - lambda * sum_{transfers} S_x .

The oracle chooses which encoders to transfer and cache (relay copies
enable later transfers, cache capacity enforced) via CBC; a no-caching
oracle (relay holdings frozen at w0) isolates the value of relaying.

Run:  python3 -m sim.face_oracle
Output: results/face_oracle.npz + printed per-window table.
"""

import numpy as np
import torch
import pulp

from .config import Config
from .mobility import RoadNetwork, MobilitySim
from .mfl import MultimodalFL
from .simulator import make_modality_availability, make_arch_assignment
from .v2x_trace import build_v2x_trace
from .face import FACE

W = 30                                 # scored window length (rounds)
WARM = 15                              # estimator warm-up before scoring
N_SMALL = 18                           # instance size (vehicles)
N_INST = 20                            # number of instances
MIN_CONTACTS = 6                       # discard degenerate instances
SEED = 2026


def _small_world(inst_seed):
    """Run FACE on a small world: a random vehicle subset on a random
    window of the real Seoul trace. Returns (cfg, alg) with oracle logs
    and the frozen value table snapshotted at the end of warm-up."""
    cfg = Config()
    cfg.num_vehicles = N_SMALL
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    full = build_v2x_trace(Config())
    rng = np.random.default_rng(inst_seed)
    Kfull = full["veh_xy"].shape[0]
    idx = np.sort(rng.choice(full["veh_xy"].shape[1], N_SMALL,
                             replace=False))
    w0 = int(rng.integers(0, Kfull - 1))
    rounds = WARM + W
    sel = [(w0 + t) % Kfull for t in range(rounds)]
    trace = dict(full)
    for kk in ("veh_seg", "veh_xy", "veh_speed"):
        trace[kk] = full[kk][sel][:, idx]
    road = RoadNetwork(trace)
    mob = MobilitySim(cfg, road, trace)
    avail = make_modality_availability(cfg,
                                       np.random.default_rng(inst_seed + 7))
    arch = make_arch_assignment(cfg, np.random.default_rng(inst_seed + 11),
                                avail)
    mfl = MultimodalFL(cfg, np.random.default_rng(inst_seed), avail)
    mfl.arch = arch
    alg = FACE(cfg, mfl, mob, seed=inst_seed)
    alg.oracle_log = True
    alg.value_table_rounds = {WARM}
    for k in range(rounds):
        mob.k = k % mob.Krounds
        mfl.local_train()
        alg.run_round(k)
    return cfg, alg


def _oracle(contacts, own, held0, sizes, table, lam, cache_mb, moda,
            allow_caching=True, time_limit=90):
    """Hindsight ILP on one instance; returns the optimal P2 objective."""
    T = max(t for (t, _, _, _) in contacts) + 1 if contacts else 0
    X = sorted({x for xs in own.values() for x in xs}
               | {x for xs in held0.values() for x in xs})
    V = sorted(own.keys())
    prob = pulp.LpProblem("oracle", pulp.LpMaximize)
    # relay holdings R[i][x][t]
    R = {(i, x, t): pulp.LpVariable(f"R_{i}_{x}_{t}", cat="Binary")
         for i in V for x in X for t in range(T + 1)}
    A = {}                              # transfers per contact
    for ci, (t, i, j, cap) in enumerate(contacts):
        for x in X:
            A[(ci, x)] = pulp.LpVariable(f"a_{ci}_{x}", cat="Binary")
    D = {(j, x): pulp.LpVariable(f"d_{j}_{x}", cat="Binary")
         for (j, x) in table if j in own and x in X}
    # objective
    prob += (pulp.lpSum(table[jx] * D[jx] for jx in D)
             - lam * pulp.lpSum(A[(ci, x)] * sizes[x]
                                for (ci, x) in A))
    # initial relay holdings
    for i in V:
        for x in X:
            prob += R[(i, x, 0)] == (1 if x in held0[i] else 0)
    # contact feasibility + budget
    for ci, (t, i, j, cap) in enumerate(contacts):
        for x in X:
            if x in own[i]:
                continue                      # own versions always available
            prob += A[(ci, x)] <= R[(i, x, t)]
        prob += pulp.lpSum(A[(ci, x)] * sizes[x] for x in X) <= cap
    # holding evolution + cache capacity
    for i in V:
        for x in X:
            for t in range(T):
                recv = pulp.lpSum(
                    A[(ci, x)] for ci, (tt, si, sj, _c)
                    in enumerate(contacts) if tt == t and sj == i)
                if allow_caching:
                    prob += R[(i, x, t + 1)] <= R[(i, x, t)] + recv
                else:
                    prob += R[(i, x, t + 1)] <= R[(i, x, t)]
        for t in range(T + 1):
            prob += pulp.lpSum(R[(i, x, t)] * sizes[x]
                               for x in X) <= cache_mb
    # deliveries
    for (j, x) in D:
        prob += D[(j, x)] <= pulp.lpSum(
            A[(ci, x)] for ci, (tt, si, sj, _c) in enumerate(contacts)
            if sj == j)
    # a served demand collapses: at most one counted delivery per
    # (receiver, modality), matching the adopt-best-per-modality protocol
    for j in V:
        for r in {moda[x] for (jj, x) in D if jj == j}:
            prob += pulp.lpSum(D[(jj, x)] for (jj, x) in D
                               if jj == j and moda[x] == r) <= 1
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))
    return pulp.value(prob.objective) or 0.0


def main():
    rows = []
    for n in range(N_INST):
        cfg, alg = _small_world(3000 + n)
        lam = cfg.face_lam
        table = alg.value_tables.get(WARM, {})
        snap = alg.state_snapshots.get(WARM)
        if not table or snap is None:
            continue
        sub = set(range(cfg.num_vehicles))
        contacts = [(t - WARM, i, j, c) for (t, i, j, c) in alg.pair_log
                    if WARM <= t < WARM + W]
        own = {i: set(snap["own"].get(i, [])) for i in sub}
        held0 = {i: set(snap["held"].get(i, [])) for i in sub}
        sizes = snap["sizes"]
        moda = snap["moda"]
        X = ({x for xs in own.values() for x in xs}
             | {x for xs in held0.values() for x in xs})
        tab = {(j, x): v for (j, x), v in table.items()
               if x in X and x not in own[j] and x not in held0[j]}
        if len(contacts) < MIN_CONTACTS or not tab:
            print(f"  inst={n} skipped (contacts={len(contacts)}, "
                  f"pairs={len(tab)})")
            continue
        # cap the ILP encoder universe to the most valuable versions
        if len(X) > 40:
            best_of = {}
            for (j, x), v in tab.items():
                best_of[x] = max(best_of.get(x, 0.0), v)
            keep = set(sorted(best_of, key=best_of.get)[-40:])
            keep |= {x for xs in own.values() for x in xs}
            X = X & keep if X & keep else X
            tab = {jx: v for jx, v in tab.items() if jx[1] in X}
        opt = _oracle(contacts, own, held0, sizes, tab, lam,
                      cfg.cache_capacity_mb, moda, allow_caching=True)
        nocache = _oracle(contacts, own, held0, sizes, tab, lam,
                          cfg.cache_capacity_mb, moda,
                          allow_caching=False)
        seen, face_val, face_cost = set(), 0.0, 0.0
        for (t, i, j, x, _vh, S) in alg.deliv_log:
            if not (WARM <= t < WARM + W and x in X):
                continue
            face_cost += lam * S
            if (j, x) in tab and (j, moda[x]) not in seen:
                seen.add((j, moda[x]))
                face_val += tab[(j, x)]
        face_obj = face_val - face_cost
        rows.append((n, cfg.num_vehicles, len(contacts), len(tab),
                     face_obj, opt, nocache))
        print(f"  inst={n} contacts={len(contacts):3d} "
              f"pairs={len(tab):3d}  FACE={face_obj:7.3f}  "
              f"oracle={opt:7.3f}  no-cache={nocache:7.3f}  "
              f"ratio={face_obj / opt if opt > 1e-9 else float('nan'):.3f}")
    arr = np.array(rows)
    ratios = arr[:, 4] / np.maximum(arr[:, 5], 1e-9)
    nc = arr[:, 6] / np.maximum(arr[:, 5], 1e-9)
    print(f"\nFACE/oracle ratio: {ratios.mean():.3f} +- {ratios.std():.3f}"
          f"   no-cache/oracle: {nc.mean():.3f} +- {nc.std():.3f}")
    np.savez("results/face_oracle.npz", rows=arr)
    print("saved results/face_oracle.npz")


if __name__ == "__main__":
    main()
