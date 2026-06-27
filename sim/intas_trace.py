"""
Real Ingolstadt (InTAS) mobility trace via SUMO.

We load the genuine InTAS road network (ingolstadt.net.xml: 3,342 nodes,
7,968 edges, 98 traffic lights) and run the SUMO microsimulation over a
dense-hour demand file (InTAS_008.rou.xml, ~09:30-10:30) with libsumo. At a
fixed round interval we snapshot the position and current road edge of every
running vehicle, then pick a cohort of N vehicles that are present for the whole
window. This produces the road graph G^road and the per-round vehicle states /
V2V graph G^com(k) used by the rest of the pipeline (Sec. III-B).

Reference scenario: S. Lobo et al., "InTAS - The Ingolstadt Traffic Scenario
for SUMO," arXiv:2011.11995.
"""

import os
import numpy as np
import sumolib
import libsumo as traci

INTAS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "InTAS", "scenario")
NET_FILE = os.path.join(INTAS_DIR, "ingolstadt.net.xml")
ROUTE_FILE = os.path.join(INTAS_DIR, "routes", "InTAS_008.rou.xml")


def build_road_graph_from_net(net):
    """
    Build the directed road-segment graph from a sumolib net.
    Returns a dict of arrays compatible with sim.mobility.RoadNetwork.
    """
    edges = [e for e in net.getEdges() if not e.isSpecial()]  # drop internal :junction edges
    eid = {e.getID(): k for k, e in enumerate(edges)}
    V = len(edges)

    L = np.array([max(e.getLength(), 1.0) for e in edges], dtype=np.float64)
    mid = np.zeros((V, 2))
    head = np.zeros((V, 2))   # downstream end coordinate (for turn geometry)
    tail = np.zeros((V, 2))
    for k, e in enumerate(edges):
        shape = np.array(e.getShape())
        mid[k] = shape.mean(axis=0)
        tail[k] = shape[0]
        head[k] = shape[-1]

    # successors via SUMO connections (turn topology O(e))
    successors = [[] for _ in range(V)]
    for k, e in enumerate(edges):
        for out in e.getOutgoing():       # dict: outgoing edge -> connections
            oid = out.getID()
            if oid in eid:
                successors[k].append(eid[oid])
        successors[k] = sorted(set(successors[k]))

    # edge index (2, E_road) of directed road links incl. self-loops (sparse GAT)
    src, dst = [], []
    for k in range(V):
        src.append(k); dst.append(k)          # self-loop
        for k2 in successors[k]:
            src.append(k); dst.append(k2)
    edge_index = np.array([src, dst], dtype=np.int64)

    # turn-direction label delta(e,e') in {straight,left,right,uturn}
    turn = {}
    for k in range(V):
        inc = head[k] - tail[k]
        a0 = np.arctan2(inc[1], inc[0])
        for k2 in successors[k]:
            out = head[k2] - tail[k2]
            a1 = np.arctan2(out[1], out[0])
            dth = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
            if abs(dth) < np.deg2rad(30):
                lab = 0
            elif dth > np.deg2rad(150) or dth < -np.deg2rad(150):
                lab = 3
            elif dth > 0:
                lab = 1
            else:
                lab = 2
            turn[(k, k2)] = lab

    return {
        "edges": [e.getID() for e in edges],
        "eid": eid,
        "V": V,
        "L": L,
        "mid": mid,
        "edge_index": edge_index,
        "successors": successors,
        "turn": turn,
    }


def run_trace(cfg, begin=34050.0, dt=2.0, warmup_s=480.0, pool_rounds=None, verbose=True):
    """
    Run SUMO over a window and extract a vehicle trace.

    A warm-up phase (warmup_s seconds) populates the network before the first
    snapshot. Vehicles momentarily on internal junction edges keep their last
    known road edge so they are not dropped from the persistent cohort.

    Returns dict with:
      road graph arrays (see build_road_graph_from_net)
      veh_seg   : [K, N] int   edge index of each cohort vehicle per round
      veh_xy    : [K, N, 2]    position per round (network-centred)
      veh_speed : [K, N]       speed per round
    """
    K = cfg.K if pool_rounds is None else pool_rounds
    net = sumolib.net.readNet(NET_FILE)
    road = build_road_graph_from_net(net)
    eid = road["eid"]

    sumo_cmd = [
        "sumo",
        "-n", NET_FILE,
        "-r", ROUTE_FILE,
        "--begin", str(begin),
        "--step-length", "1.0",
        "--no-warnings", "true",
        "--no-step-log", "true",
        "--time-to-teleport", "120",
        "--max-depart-delay", "60",
        "--ignore-route-errors", "true",
        "--xml-validation", "never",
    ]
    if verbose:
        print(f"    [SUMO] starting InTAS at t={begin}s, warm-up {warmup_s}s, "
              f"{K} rounds x {dt}s ...")
    traci.start(sumo_cmd)

    sim_t = begin
    # warm-up to populate the network
    while sim_t < begin + warmup_s:
        traci.simulationStep()
        sim_t += 1.0

    last_edge = {}   # vid -> last known normal-edge index (carry through junctions)
    # records[t_index] -> dict vid -> (edge_idx, x, y, speed)
    snaps = []
    V = road["V"]
    edge_count = np.zeros((K, V), dtype=np.float32)   # all running vehicles per edge
    edge_spd = np.zeros((K, V), dtype=np.float32)
    target = sim_t
    for k in range(K):
        while sim_t < target + 1e-6:
            traci.simulationStep()
            sim_t += 1.0
        rec = {}
        for vid in traci.vehicle.getIDList():
            road_id = traci.vehicle.getRoadID(vid)
            if road_id in eid:
                e = eid[road_id]
                last_edge[vid] = e
            elif vid in last_edge:
                e = last_edge[vid]            # on internal junction edge -> carry last
            else:
                continue                      # never seen on a normal edge yet
            x, y = traci.vehicle.getPosition(vid)
            spd = traci.vehicle.getSpeed(vid)
            rec[vid] = (e, x, y, spd)
            edge_count[k, e] += 1.0
            edge_spd[k, e] += spd
        snaps.append(rec)
        target += dt
        if verbose and (k % 25 == 0):
            print(f"      round {k:3d}: {len(rec)} running vehicles")
    traci.close()

    # choose cohort: vehicles present in every snapshot
    present_all = set(snaps[0].keys())
    for rec in snaps[1:]:
        present_all &= set(rec.keys())
    cohort = sorted(present_all)
    if verbose:
        print(f"    [SUMO] {len(cohort)} vehicles present across all {K} rounds")
    if len(cohort) < cfg.num_vehicles:
        raise RuntimeError(
            f"only {len(cohort)} persistent vehicles; reduce num_vehicles, dt, or K")

    # deterministic subsample to N
    rng = np.random.default_rng(cfg.seed)
    cohort = list(rng.choice(cohort, size=cfg.num_vehicles, replace=False))
    N = len(cohort)

    veh_seg = np.zeros((K, N), dtype=np.int64)
    veh_xy = np.zeros((K, N, 2), dtype=np.float64)
    veh_speed = np.zeros((K, N), dtype=np.float64)
    for k, rec in enumerate(snaps):
        for j, vid in enumerate(cohort):
            e, x, y, spd = rec[vid]
            veh_seg[k, j] = e
            veh_xy[k, j] = (x, y)
            veh_speed[k, j] = spd

    # centre coordinates for numerical stability
    ctr = veh_xy.reshape(-1, 2).mean(axis=0)
    veh_xy -= ctr
    road["mid"] = road["mid"] - ctr

    return {
        **road,
        "veh_seg": veh_seg,
        "veh_xy": veh_xy,
        "veh_speed": veh_speed,
        "edge_count": edge_count,        # [K, V] all running vehicles per edge
        "edge_spd": edge_spd,            # [K, V] summed speed per edge
        "begin": begin,
        "dt": dt,
    }


def get_or_build_trace(cfg, cache_path, **kw):
    if os.path.exists(cache_path):
        print(f"    [trace] loading cached InTAS trace {cache_path}")
        d = np.load(cache_path, allow_pickle=True)
        return {k: d[k] for k in d.files}
    tr = run_trace(cfg, **kw)
    save = {}
    for k, v in tr.items():
        if isinstance(v, (list, dict)):
            save[k] = np.array(v, dtype=object)
        else:
            save[k] = v
    np.savez(cache_path, **save)
    print(f"    [trace] saved InTAS trace -> {cache_path}")
    return tr
