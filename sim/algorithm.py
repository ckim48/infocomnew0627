"""
Online encoder caching and forwarding (Sec. IV).

Implements the queue-weighted submodular maximization of problem P (Eq. 19)
solved by the marginal-gain greedy algorithm (Sec. IV-D, Eq. 24-27) with the
(1-1/e) guarantee, plus the Psi-based cache update (Eq. 28-30) and the Lyapunov
virtual-queue update (Eq. 20).

A single configurable routine `run_round` realizes the proposed scheme and all
baselines through feature flags:
    use_link      : weight contributions by V2V link quality P^tx (else ideal)
    use_dis       : include the road-aware dissemination utility F^dis (Gamma)
    use_queue     : Lyapunov queue weighting (Q_{i,r}+V); else plain V
    demand_aware  : score deliveries by receiver learning need / loss gain;
                    if False (caching-assisted), forward cached encoders blindly
    carry         : allow store-carry-forward of *other* vehicles' encoders
    cache_policy  : 'psi' (Eq. 30) | 'lru' | 'own'
"""

import numpy as np
from .utility import modality_needs, mean_modality_data


class CachingForwarding:
    def __init__(self, cfg, mfl, mob, scheme, seed=0):
        self.cfg = cfg
        self.mfl = mfl
        self.mob = mob
        self.scheme = scheme
        self.flags = SCHEME_FLAGS[scheme]
        self.rng = np.random.default_rng(seed + 101)     # transmission-success RNG
        # virtual queues Q_{i,r}(k) (Eq. 20)
        self.Q = {(i, r): 0.0 for (i, r) in mfl.theta}
        # cache: vehicle i -> dict (owner m, modality r) -> theta snapshot
        # initialised with each vehicle's own encoders
        self.cache = {i: {} for i in range(mfl.N)}
        self.lru_clock = {i: {} for i in range(mfl.N)}   # (m,r) -> last-use round
        self._clock = 0

    # ---------- per-round entry point ----------
    def run_round(self, k, gamma):
        cfg, mfl, mob, fl = self.cfg, self.mfl, self.mob, self.flags
        self._clock = k
        A = mob.v2v_graph()
        need = modality_needs(cfg, mfl)
        Dr = mean_modality_data(mfl)

        # refresh own encoders in cache (always available to forward)
        for (i, r) in mfl.pairs:
            self.cache[i][(i, r)] = mfl.theta[(i, r)]

        # ----- build candidate forwarding set E^cand(k) (Eq. 25) -----
        # candidate e = (i sender, j receiver, m owner, r modality)
        cands = []
        for i in range(mfl.N):
            nbrs = mob.neighbors(A, i)
            if len(nbrs) == 0:
                continue
            held = list(self.cache[i].keys())
            for (m, r) in held:
                for j in nbrs:
                    if r not in mfl.avail[j]:
                        continue
                    if m == j:                      # receiver already owns it
                        continue
                    if (m, r) in self.cache[j]:     # receiver already has this encoder
                        continue
                    cands.append((i, int(j), m, r))

        # ----- precompute per-candidate beta_learn, beta_dis, tx time -----
        info = {}
        for e in cands:
            i, j, m, r = e
            ptx_real = mob.link_quality(i, j)               # true (physical) link quality
            ptx_dec = ptx_real if fl["use_link"] else 1.0   # quality the scheme accounts for
            S = cfg.encoder_size[r]
            # physical airtime is consumed regardless of success (real link);
            # a link-blind scheme cannot avoid wasting it on poor links.
            t_tx = S / (cfg.tx_rate_mbps * max(ptx_real, 0.05))
            # learning contribution beta^learn (Eq. 12-13)
            if fl["demand_aware"]:
                s_m = self.cache[i][(m, r)]
                g, _, _ = mfl.gain_single(j, r, m, s_m)
                beta_learn = ptx_dec * (mfl.Dmr(m, r) / (Dr[r] + cfg.eps0)) * g
            else:
                beta_learn = ptx_dec * 0.05         # demand-agnostic placeholder weight
            beta_learn = float(np.clip(beta_learn, 0.0, 0.999))
            # dissemination contribution beta^dis (Eq. 14)
            beta_dis = ptx_dec * (1.0 - np.exp(-gamma[j])) if fl["use_dis"] else 0.0
            info[e] = dict(ptx_real=ptx_real, S=S, t_tx=t_tx,
                           beta_learn=beta_learn, beta_dis=float(beta_dis))

        # ----- greedy marginal-gain selection (Eq. 24-27) -----
        selected = self._greedy(cands, info, need)

        # ----- apply forwarding: receivers aggregate (Eq. 2), update cache & queue -----
        self._apply(selected, info, need)
        self._last_need = need
        # ----- cache update for next round (Eq. 28-30) -----
        self._update_caches(gamma, need, Dr)
        return selected

    # ---------- greedy submodular selection ----------
    def _greedy(self, cands, info, need):
        cfg, fl = self.cfg, self.flags
        V, nu = cfg.V, cfg.nu
        learn_prod = {}   # (j,r) -> running prod (1-beta_learn)
        dis_prod = {}     # (m,r) -> running prod (1-beta_dis)
        used_contact = {} # (i,j) -> used tx time  (C1)
        received = set()  # (j,m,r) delivered  (C4: at most one sender per (j,m,r))
        recv_mod = {}     # (j,r) -> count: receiver integrates <= 1 encoder per
                          # modality per round (radio/compute reception limit)

        selected = []
        remaining = set(cands)

        # caching-assisted: no marginal-gain ranking; fill by LRU recency
        if not fl["demand_aware"]:
            order = sorted(cands, key=lambda e: -self.lru_clock[e[0]].get((e[2], e[3]), -1))
            for e in order:
                i, j, m, r = e
                d = info[e]
                if (j, m, r) in received or recv_mod.get((j, r), 0) >= 1:
                    continue
                if used_contact.get((i, j), 0.0) + d["t_tx"] > cfg.contact_time_per_round:
                    continue
                selected.append(e)
                used_contact[(i, j)] = used_contact.get((i, j), 0.0) + d["t_tx"]
                received.add((j, m, r))
                recv_mod[(j, r)] = recv_mod.get((j, r), 0) + 1
            return selected

        while remaining:
            best_e, best_eta, best_delta = None, 0.0, 0.0
            for e in remaining:
                i, j, m, r = e
                d = info[e]
                if (j, m, r) in received or recv_mod.get((j, r), 0) >= 1:
                    continue
                if used_contact.get((i, j), 0.0) + d["t_tx"] > cfg.contact_time_per_round:
                    continue
                w_learn = (self.Q[(j, r)] + V) if fl["use_queue"] else V
                lp = learn_prod.get((j, r), 1.0)
                delta_learn = w_learn * need.get((j, r), 0.0) * lp * d["beta_learn"]
                if fl["use_dis"]:
                    dp = dis_prod.get((m, r), 1.0)
                    delta_dis = V * nu * dp * d["beta_dis"]
                else:
                    delta_dis = 0.0
                delta = delta_learn + delta_dis
                eta = delta / (d["t_tx"] + cfg.lam * d["S"])
                if eta > best_eta:
                    best_eta, best_e, best_delta = eta, e, delta
            if best_e is None or best_delta <= 1e-12:
                break
            i, j, m, r = best_e
            d = info[best_e]
            selected.append(best_e)
            remaining.discard(best_e)
            used_contact[(i, j)] = used_contact.get((i, j), 0.0) + d["t_tx"]
            received.add((j, m, r))
            recv_mod[(j, r)] = recv_mod.get((j, r), 0) + 1
            learn_prod[(j, r)] = learn_prod.get((j, r), 1.0) * (1.0 - d["beta_learn"])
            if fl["use_dis"]:
                dis_prod[(m, r)] = dis_prod.get((m, r), 1.0) * (1.0 - d["beta_dis"])
        return selected

    # ---------- apply forwarding ----------
    def _apply(self, selected, info, need):
        cfg, mfl, fl = self.cfg, self.mfl, self.flags
        # group received encoders per receiver-modality; track achieved coverage
        recv = {}            # (j,r) -> list of (owner m, theta)
        learn_prod = {}      # (j,r) -> running prod (1-beta_learn)
        for e in selected:
            i, j, m, r = e
            # stochastic transmission success over the lossy V2V link (P^tx)
            if self.rng.random() > info[e]["ptx_real"]:
                continue                              # transmission failed: nothing delivered
            s_m = self.cache[i][(m, r)]
            recv.setdefault((j, r), []).append((m, s_m))
            learn_prod[(j, r)] = learn_prod.get((j, r), 1.0) * (1.0 - info[e]["beta_learn"])
            # store-carry-forward: receiver caches the encoder for future rounds
            if fl["carry"]:
                self.cache[j][(m, r)] = s_m
                self.lru_clock[j][(m, r)] = self._clock

        # aggregation (Eq. 2) and commit new local encoders
        for (j, r), lst in recv.items():
            mfl.commit(j, r, lst)

        # queue update Q_{i,r}(k+1) = [Q + alpha^need - F^learn]^+  (Eq. 20).
        # The arrival is the *residual* unmet learning need, weighted by demand and
        # scaled by the current quality gap (1 - Q^eff), so it vanishes once a
        # vehicle is well served and the virtual queue can stabilize.
        for (i, r) in mfl.pairs:
            gap = 1.0 - mfl.q_eff(i, r)
            arrival = need.get((i, r), 0.0) * gap
            ach = need.get((i, r), 0.0) * (1.0 - learn_prod.get((i, r), 1.0))
            self.Q[(i, r)] = max(self.Q[(i, r)] + arrival - ach, 0.0)

    # ---------- cache update (Eq. 28-30) ----------
    def _update_caches(self, gamma, need, Dr):
        cfg, mfl, mob, fl = self.cfg, self.mfl, self.mob, self.flags
        A = mob.v2v_graph()
        for i in range(mfl.N):
            cap = cfg.cache_capacity_mb
            own = [(i, r) for r in mfl.avail[i]]                  # always keep own encoders
            others = [key for key in self.cache[i] if key[0] != i]

            if fl["cache_policy"] == "own":
                keep = set(own)
            elif fl["cache_policy"] == "lru":
                # keep own + most-recently-used others within capacity
                size_own = sum(cfg.encoder_size[r] for (_, r) in own)
                budget = cap - size_own
                others_sorted = sorted(others, key=lambda kr: -self.lru_clock[i].get(kr, -1))
                keep = set(own)
                for kr in others_sorted:
                    s = cfg.encoder_size[kr[1]]
                    if budget - s >= 0:
                        keep.add(kr); budget -= s
            else:  # 'psi': knapsack by future utility Psi (Eq. 29-30)
                size_own = sum(cfg.encoder_size[r] for (_, r) in own)
                budget = cap - size_own
                scored = []
                nbrs = mob.neighbors(A, i)
                for (m, r) in others:
                    s_m = self.cache[i][(m, r)]
                    # future dissemination term
                    dis = 0.0
                    for j in nbrs:
                        if r in mfl.avail[j]:
                            ptx = mob.link_quality(i, j)
                            dis += ptx * gamma[j]
                    # future learning term over modalities (use own modality r)
                    learn = 0.0
                    for j in nbrs:
                        if r in mfl.avail[j] and (m, r) not in self.cache[j]:
                            g, _, _ = mfl.gain_single(j, r, m, s_m)
                            learn += self.Q[(j, r)] * need.get((j, r), 0.0) * g
                    psi = cfg.nu * dis + learn
                    scored.append(((m, r), psi))
                scored.sort(key=lambda x: -x[1] / cfg.encoder_size[x[0][1]])
                keep = set(own)
                for (kr, psi) in scored:
                    s = cfg.encoder_size[kr[1]]
                    if budget - s >= 0:
                        keep.add(kr); budget -= s

            self.cache[i] = {kr: self.cache[i][kr] for kr in self.cache[i] if kr in keep}
            self.lru_clock[i] = {kr: self.lru_clock[i].get(kr, self._clock) for kr in keep}


# scheme feature flags (proposed + three baselines, Sec. V-A)
SCHEME_FLAGS = {
    "Proposed":         dict(use_link=True,  use_dis=False, use_queue=True,
                             demand_aware=True,  carry=True,  cache_policy="psi"),
    "Caching-assisted": dict(use_link=True,  use_dis=False, use_queue=False,
                             demand_aware=False, carry=True,  cache_policy="lru"),
    "V2V-aware":        dict(use_link=True,  use_dis=False, use_queue=False,
                             demand_aware=True,  carry=False, cache_policy="own"),
    "Learning-aware":   dict(use_link=False, use_dis=False, use_queue=False,
                             demand_aware=True,  carry=False, cache_policy="own"),
}
