"""
Hierarchical road-vehicle graph and vehicle mobility (Sec. III-B),
backed by the real Ingolstadt (InTAS) SUMO trace produced in sim.intas_trace.

  RoadNetwork : directed road-segment graph G^road = (V, A^road) from
                ingolstadt.net.xml (segments, lengths L_e, turn topology O(e),
                turn-direction labels delta(e,e')).
  MobilitySim : replays the SUMO trace round by round, exposing per-round
                vehicle states, the dynamic V2V graph G^com(k), and the realized
                road transitions used to (self-)supervise the mobility predictor.
"""

import numpy as np


class RoadNetwork:
    """Directed road-segment graph G^road = (V, A^road) from the InTAS net."""

    def __init__(self, trace):
        self.V = int(trace["V"])
        self.L = np.asarray(trace["L"], dtype=np.float64)
        self.mid = np.asarray(trace["mid"], dtype=np.float64)
        self.edge_index = np.asarray(trace["edge_index"], dtype=np.int64)  # [2, E_road]
        succ = trace["successors"]
        self.successors = [list(s) for s in (succ.tolist() if hasattr(succ, "tolist") else succ)]
        turn = trace["turn"]
        self.turn = turn.item() if hasattr(turn, "item") and turn.dtype == object else dict(turn)
        self._reach_cache = {}

    def reachable(self, e, hops=2):
        key = (e, hops)
        if key in self._reach_cache:
            return self._reach_cache[key]
        seen = {e}
        frontier = {e}
        for _ in range(hops):
            nxt = set()
            for x in frontier:
                nxt |= set(self.successors[x])
            frontier = nxt - seen
            seen |= nxt
        out = sorted(seen)
        self._reach_cache[key] = out
        return out


class MobilitySim:
    """Replays the InTAS trace; same interface as the live SUMO feed."""

    def __init__(self, cfg, road: RoadNetwork, trace):
        self.cfg = cfg
        self.road = road
        self.veh_seg = np.asarray(trace["veh_seg"], dtype=np.int64)        # [K,N]
        self.veh_xy = np.asarray(trace["veh_xy"], dtype=np.float64)        # [K,N,2]
        self.veh_speed = np.asarray(trace["veh_speed"], dtype=np.float64)  # [K,N]
        self.edge_count = np.asarray(trace["edge_count"], dtype=np.float64)  # [K,V]
        self.edge_spd = np.asarray(trace["edge_spd"], dtype=np.float64)      # [K,V]
        self.dt = float(trace["dt"])
        self.Krounds, self.N = self.veh_seg.shape
        self.k = 0
        self._precompute_realized_transitions()

    # ---- round pointer ----
    @property
    def seg(self):
        return self.veh_seg[self.k]

    @property
    def speed(self):
        return self.veh_speed[self.k]

    @property
    def prog(self):
        # progress proxy along current edge (not provided by snapshot); use 0.5
        return np.full(self.N, 0.5)

    def step(self):
        self.k = min(self.k + 1, self.Krounds - 1)

    def reset(self):
        self.k = 0

    # ---- traffic state z_e(k) = [L_e, density rho_e, avg speed, flow] ----
    def traffic_state(self):
        road = self.road
        counts = self.edge_count[self.k]
        density = counts / (road.L + 1.0) * 1000.0
        avg_speed = np.where(counts > 0, self.edge_spd[self.k] / np.maximum(counts, 1), 8.0)
        flow = density * avg_speed
        self._density = density
        z = np.stack([road.L, density, avg_speed, flow], axis=1).astype(np.float32)
        # standardise columns for the GAT
        zmu, zsd = z.mean(0), z.std(0) + 1e-6
        return ((z - zmu) / zsd).astype(np.float32)

    def density(self):
        if not hasattr(self, "_density"):
            self.traffic_state()
        return self.edge_count[self.k] / (self.road.L + 1.0) * 1000.0

    # ---- V2V graph G^com(k) ----
    def vehicle_xy(self):
        return self.veh_xy[self.k]

    def v2v_graph(self):
        xy = self.vehicle_xy()
        d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
        A = (d <= self.cfg.comm_range).astype(np.float32)
        np.fill_diagonal(A, 0.0)
        self._dist = d
        return A

    def neighbors(self, A, i):
        return np.where(A[i] > 0)[0]

    def link_quality(self, i, j):
        """Successful tx probability P^tx from V2V distance."""
        if not hasattr(self, "_dist"):
            self.v2v_graph()
        d = self._dist[i, j]
        return float(np.clip(1.0 - (d / self.cfg.comm_range) ** 2, 0.05, 0.99))

    # ---- realized road transitions (labels for the mobility predictor) ----
    def _precompute_realized_transitions(self):
        """
        For each (round k, vehicle i) find the realized next *direct successor*
        edge from the trace. Stored as index into successors[e], or -1 if none
        identifiable within the look-ahead window.
        """
        road = self.road
        K, N = self.Krounds, self.N
        self.realized_idx = -np.ones((K, N), dtype=np.int64)
        look = 8
        for i in range(N):
            traj = self.veh_seg[:, i]
            for k in range(K):
                e = traj[k]
                succ = road.successors[e]
                if len(succ) < 2:
                    continue
                # first future edge that is a direct successor of e
                for kk in range(k + 1, min(k + 1 + look, K)):
                    ne = traj[kk]
                    if ne == e:
                        continue
                    if ne in succ:
                        self.realized_idx[k, i] = succ.index(ne)
                    break

    def transition_probs_true(self):
        """
        Return per-vehicle (e, succ, w) at the current round, where w is the
        one-hot realized successor distribution (skipped later if len(succ)<2 or
        no realized successor was identified).
        """
        road = self.road
        out = []
        for i in range(self.N):
            e = int(self.seg[i])
            succ = road.successors[e]
            idx = self.realized_idx[self.k, i]
            if len(succ) < 2 or idx < 0:
                w = np.ones(max(len(succ), 1)) / max(len(succ), 1)
            else:
                w = np.zeros(len(succ))
                w[idx] = 1.0
            out.append((e, succ, w))
        return out
