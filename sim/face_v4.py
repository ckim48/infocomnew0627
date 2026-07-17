"""
FACE: future-contact-aware joint encoder forwarding and caching (paper v4).

Implements the v4 system model and algorithm (Sec. III-IV):
  * immutable encoder VERSIONS x with architecture-family compatibility chi
    (same modality AND same family; incompatible vehicles may still relay)
  * kinematic sojourn-time contact budget B-hat = R * tau-hat shared by both
    directions of an activated contact (eq:sojourn_time, eq:contact_budget)
  * half-duplex contact activation via a distributed mutual-proposal pairing
    protocol (eq:matching_constraint, Sec. IV pairing protocol)
  * TWO transfer modes (eq:transfer_advantage): consume-only (requester
    aggregates and discards; no relay storage, no tokens) and retain
    (creates a relay copy; requires replication tokens and cache space)
  * replication tokens in binary spray-and-wait style enforcing the copy
    cap K_x exactly (sender needs k>=2, passes floor(k/2))
  * FedAvg aggregation over the aggregation set with a validation
    acceptance test (never hurts, Prop. stability) and leave-one-out
    attribution v_{i,x} (eq:loo_attribution) feeding a sliding-window
    ridge reward predictor (eq:predicted_gain)
  * causal zone estimators: context-free decayed transitions P-hat,
    ACTIVATED-exchange complete-exchange probability kappa-hat, requester
    fraction q and conditional mean reward v-bar with horizon forecasts
  * first-delivery survival recursion (eq:first_contact_survival, with
    rho^(1) = e_a P-hat), residual coverage Omega (eq:residual_coverage),
    continuation value F (eq:forwarding_potential) with version-age
    staleness discounting of the frozen-metadata gain prior
  * joint bidirectional bundle selection with per-item mode choice and
    receiver-side eviction, solved exactly by layered DP over MB-quantized
    (sent, retained) volumes (eq:bundle_knapsack)
  * coverage-aware cache refresh knapsack (eq:cache_refresh)

Ablation flags (FACE_FLAGS) switch each mechanism off independently so the
component ablation can attribute the end-to-end gain.
"""

import numpy as np
from .utility import modality_needs, mean_modality_data


# ---------------------------------------------------------------- versions
class Version:
    __slots__ = ("vid", "src", "r", "nu", "payload", "s_meta", "arch",
                 "t_gen", "t_exp", "S", "K", "compat", "resolved")

    def __init__(self, vid, src, r, nu, payload, s_meta, t_gen, t_exp, S, K,
                 N, avail, arch=None):
        self.vid, self.src, self.r, self.nu = vid, src, r, nu
        self.payload = payload            # immutable snapshot (strength / sd)
        self.s_meta = float(s_meta)       # source-quality metadata at t_gen
        self.t_gen, self.t_exp, self.S, self.K = t_gen, t_exp, float(S), K
        self.arch = None if arch is None else arch.get((src, r), 0)
        # chi_{i,x}: same modality AND same architecture family
        if arch is None:
            self.compat = np.array([r in avail[i] for i in range(N)])
        else:
            self.compat = np.array(
                [r in avail[i] and arch.get((i, r), 0) == self.arch
                 for i in range(N)])
        self.resolved = np.zeros(N, dtype=bool)     # incorporated / rejected


# ---------------------------------------------------------------- road zones
class Zones:
    """Compact grid road-zones over the trace extent (occupied cells only)."""

    def __init__(self, cfg, mob):
        cell = cfg.face_zone_cell
        xy = mob.veh_xy.reshape(-1, 2)
        self.x0, self.y0, self.cell = xy[:, 0].min(), xy[:, 1].min(), cell
        gx = ((mob.veh_xy[..., 0] - self.x0) // cell).astype(np.int64)
        gy = ((mob.veh_xy[..., 1] - self.y0) // cell).astype(np.int64)
        self.nx = int(gx.max()) + 1
        raw = gx * 10_000 + gy                              # [K,N] raw cell key
        keys = np.unique(raw)
        lut = {int(kk): z for z, kk in enumerate(keys)}
        self.Z = len(keys)
        self.zone = np.vectorize(lambda v: lut[int(v)])(raw)   # [K,N] zone ids
        # 8-neighbour adjacency (+self) among occupied cells
        A = np.eye(self.Z)
        cx, cy = keys // 10_000, keys % 10_000
        for a in range(self.Z):
            near = (np.abs(cx - cx[a]) <= 1) & (np.abs(cy - cy[a]) <= 1)
            A[a, near] = 1.0
        self.adj = A


# ---------------------------------------------------------------- ridge gain
class RidgeGain:
    """Per-modality sliding-window ridge on realized LOO attributions
    (eq:predicted_gain)."""
    DIM = 6

    def __init__(self, cfg):
        self.cfg = cfg
        self.A = {}   # r -> [d,d]
        self.b = {}   # r -> [d]

    def _get(self, r):
        if r not in self.A:
            self.A[r] = np.eye(self.DIM) * self.cfg.face_ridge_lam
            # weak prior mean of 0.3 on the intercept: bootstraps forwarding
            # before any attributions are observed, washed out by real samples
            self.b[r] = np.zeros(self.DIM)
            self.b[r][0] = self.cfg.face_ridge_lam * 0.3
        return self.A[r], self.b[r]

    def update(self, r, psi, y):
        A, b = self._get(r)
        g = self.cfg.face_ridge_decay
        self.A[r] = g * A + np.outer(psi, psi) \
            + (1 - g) * np.eye(self.DIM) * self.cfg.face_ridge_lam
        self.b[r] = g * b + psi * y

    def predict(self, r, Psi):
        """Psi: [n, d] -> predicted rewards in [0,1] (posterior mean)."""
        A, b = self._get(r)
        mu = Psi @ (np.linalg.inv(A) @ b)
        return np.clip(mu, 0.0, 1.0)


FACE_FLAGS = dict(use_future=True,     # continuation value F in retain mode
                  use_coverage=True,   # residual coverage Omega
                  use_tickets=True,    # replication-token copy cap K_x
                  use_ridge=True,      # ridge reward prediction (else bandit)
                  use_relay=True,      # retain-mode relaying of others' encoders
                  use_demand=True,     # requester-demand-aware immediate value
                  use_consume=True,    # consume-only transfer mode
                  link_blind=False,    # schedule ignoring the V2V link quality
                  select=None,         # candidate rule: None|'mmfedmc'|'autofed'
                  refresh="knapsack")  # cache refresh: 'knapsack' | 'lru'

# All schemes run under the SAME system-model protocol (encoder versions,
# copy cap, acceptance-gated aggregation, publication cadence, Sec. III);
# they differ only in the forwarding POLICY, exactly as in the paper.
SCHEME_FACE_FLAGS = {
    "Proposed":         {},
    # direct V2V exchange of own encoders, demand + link aware, no carrying
    "V2V-aware":        dict(use_relay=False, use_future=False,
                             use_coverage=False, refresh="lru"),
    # like V2V-aware but blind to the V2V link condition when scheduling
    "Learning-aware":   dict(use_relay=False, use_future=False,
                             use_coverage=False, refresh="lru",
                             link_blind=True),
    # relays cached encoders blindly (epidemic-style), LRU, demand-blind
    "Caching-assisted": dict(use_demand=False, use_future=False,
                             use_coverage=False, refresh="lru"),
    # mmFedMC: sender offers only its top own modality by contribution/cost
    "mmFedMC":          dict(use_relay=False, use_future=False,
                             use_coverage=False, refresh="lru",
                             select="mmfedmc"),
    # AutoFed: prioritize high-quality (clean, data-rich) sources
    "AutoFed":          dict(use_relay=False, use_future=False,
                             use_coverage=False, refresh="lru",
                             select="autofed"),
}

# retain-mode bonus for blind-caching policies (use_relay without a future
# value): a flat caching propensity replacing the continuation value F
BLIND_CACHE_BONUS = 0.02


class FACE:
    """Drop-in engine implementing the v4 system model for every scheme."""

    def __init__(self, cfg, mfl, mob, scheme="FACE", seed=0, flags=None):
        self.cfg, self.mfl, self.mob, self.scheme = cfg, mfl, mob, scheme
        self.flags = dict(FACE_FLAGS)
        self.flags.update(SCHEME_FACE_FLAGS.get(scheme, {}))
        self.flags.update(flags or {})
        self.rng = np.random.default_rng(seed + 101)
        self.zones = Zones(cfg, mob)
        Z = self.zones.Z
        self.ridge = RidgeGain(cfg)
        self.gstat = {}                     # (src,r) -> (n, mean) mean-bandit
        # --- causal estimator state ---
        self.Ntrans = np.zeros((Z, Z))      # decayed transition counts
        self.Ez = np.zeros(Z)               # decayed vehicle-round counts
        self.size_bins = sorted(set(cfg.encoder_size.values()))
        self.Czb = np.zeros((Z, len(self.size_bins)))
        R = list(cfg.modalities)
        self.r_idx = {r: a for a, r in enumerate(R)}
        self.q_hist = np.zeros((Z, len(R)))   # missing history -> zero value
        self.v_hist = np.zeros((Z, len(R)))
        # --- version / token state ---
        self.versions = {}                  # vid -> Version
        self.tickets = {i: {} for i in range(mfl.N)}   # i -> vid -> tokens
        self.own_pub = {}                   # (i,r) -> vid (latest own)
        self.lru = {i: {} for i in range(mfl.N)}
        self.arch = getattr(mfl, "arch", None)   # (i,r) -> family
        self._vid = 0
        self._clock = 0
        self._evermet = set()
        self._need_ok = {}                  # r -> [N] bool, mu_{i,r} >= delta
        self._publish_all(t=0)

    # ------------------------------------------------------------ publication
    def _snapshot(self, i, r):
        snap = getattr(self.mfl, "snapshot_encoder", None)
        return snap(i, r) if snap else float(self.mfl.theta[(i, r)])

    def _publish_all(self, t):
        cfg = self.cfg
        Kt = cfg.face_K_tickets if self.flags["use_tickets"] else 10 ** 9
        for (i, r) in self.mfl.pairs:
            old = self.own_pub.get((i, r))
            nu = 0 if old is None or old not in self.versions \
                else self.versions[old].nu + 1
            v = Version(self._vid, i, r, nu, self._snapshot(i, r),
                        self.mfl.strength[(i, r)], t, t + cfg.face_ttl,
                        cfg.encoder_size[r], Kt, self.mfl.N, self.mfl.avail,
                        self.arch)
            v.resolved[i] = True            # source never re-adopts its own
            self.versions[self._vid] = v
            self.own_pub[(i, r)] = self._vid
            self.tickets[i][self._vid] = Kt
            # retire the superseded version at the source (distributed relay
            # copies elsewhere remain valid until expiry)
            if old is not None:
                self.tickets[i].pop(old, None)
            self._vid += 1

    # ------------------------------------------------------------ helpers
    def _zone_now(self):
        return self.zones.zone[min(self.mob.k, self.zones.zone.shape[0] - 1)]

    def _is_own_pub(self, i, x):
        v = self.versions[x]
        return v.src == i and self.own_pub.get((i, v.r)) == x

    def _cache_used(self, i):
        return sum(self.versions[x].S for x, m in self.tickets[i].items()
                   if m > 0 and x in self.versions
                   and not self._is_own_pub(i, x))

    def _psi(self, x, s_own, need, logD, t):
        v = self.versions[x]
        age = min((t - v.t_gen) / max(self.cfg.face_ttl, 1), 1.0)
        return np.array([1.0, v.s_meta, s_own, need, logD, age])

    def _sojourn_frac(self, i, j):
        """Kinematic sojourn fraction tau-hat / Delta_rd (eq:sojourn_time):
        the share of the round the contact is expected to survive, from the
        relative position and (per-round) velocity of the two vehicles."""
        xy = self.mob.veh_xy
        kk = min(self.mob.k, xy.shape[0] - 1)
        kp = max(kk - 1, 0)
        p = xy[kk, j] - xy[kk, i]
        vel = (xy[kk, j] - xy[kp, j]) - (xy[kk, i] - xy[kp, i])
        v2 = float(vel @ vel)
        if v2 < 1e-9:
            return 1.0                          # parallel / both stationary
        r2 = float(self.cfg.comm_range) ** 2
        d2 = float(p @ p)
        rho = float(p @ vel)
        disc = rho * rho + v2 * max(r2 - d2, 0.0)
        s = (-rho + np.sqrt(disc)) / v2         # rounds until range exit
        return float(np.clip(s, 0.05, 1.0))

    # ------------------------------------------------------------ estimators
    def _update_estimators(self, k, zn):
        d = self.cfg.face_decay
        self.Ntrans *= d
        self.Ez *= d
        self.Czb *= d
        if k > 0:
            zp = self.zones.zone[min(k - 1, self.zones.zone.shape[0] - 1)]
            np.add.at(self.Ntrans, (zp, zn), 1.0)
        np.add.at(self.Ez, zn, 1.0)

    def _credit_activated(self, zn, activated):
        """kappa-hat numerator (Sec. IV estimation): count vehicle-rounds
        that participated in an ACTIVATED exchange whose shared budget could
        carry each size bin -- consistent with the half-duplex feasibility."""
        for (i, j, B_mb) in activated:
            for e in (i, j):
                for b, s in enumerate(self.size_bins):
                    if s <= B_mb + 1e-9:
                        self.Czb[zn[e], b] += 1.0

    def _P_hat(self):
        """Smoothed one-step zone transition matrix."""
        cfg = self.cfg
        M = self.Ntrans + cfg.face_alpha_P * self.zones.adj
        return M / np.maximum(M.sum(1, keepdims=True), 1e-12)

    def _kappa(self, S):
        b = next(a for a, s in enumerate(self.size_bins) if S <= s + 1e-9)
        cfg = self.cfg
        return (self.Czb[:, b] + cfg.face_alpha_C) / \
               (self.Ez + cfg.face_alpha_C + cfg.face_beta_C)

    # ------------------------------------------------------------ value packs
    def _pack(self, x, zn, P, zstats):
        """Per-version arrays: per-round useful-delivery prob D [H,Z]
        (eq:useful_contact_probability), reward W [H,Z], and lazily filled
        first-delivery rows f_a [H,Z]."""
        cfg, v = self.cfg, self.versions[x]
        H, Z = cfg.face_H, self.zones.Z
        ri = self.r_idx[v.r]
        # requester indicator (Sec. III-C): compatible, unresolved, demand
        # above the threshold delta_mu
        req = v.compat & ~v.resolved & self._need_ok[v.r]
        cnt_all = np.bincount(zn, minlength=Z).astype(float)
        cnt_req = np.bincount(zn[req], minlength=Z).astype(float)
        q_cur = np.divide(cnt_req, cnt_all, out=np.zeros(Z),
                          where=cnt_all > 0)
        # conditional mean predicted reward per zone
        s_mean, need_mean, logD_mean = zstats[v.r]
        Psi = np.stack([np.ones(Z),
                        np.full(Z, v.s_meta),
                        s_mean, need_mean, logD_mean,
                        np.full(Z, min((self._clock - v.t_gen)
                                       / max(cfg.face_ttl, 1), 1.0))], axis=1)
        g_z = self._gain_zone(v, Psi)
        v_cur = need_mean * g_z
        mu = cfg.face_mu
        kap = self._kappa(v.S)
        # horizon staleness: a copy delivered at offset h is h rounds older
        # than now (and worthless past expiry), so the per-offset reward is
        # rescaled from the current-age discount embedded in g_z to the
        # delivery-time discount; implements H_x = min(H, t_exp - t - 1)
        ttl = max(cfg.face_ttl, 1)
        age0 = min((self._clock - v.t_gen) / ttl, 1.0)
        D = np.empty((H, Z))
        W = np.empty((H, Z))
        for h in range(H):
            w = mu ** (h + 1)
            qh = w * q_cur + (1 - w) * self.q_hist[:, ri]
            vh = w * v_cur + (1 - w) * self.v_hist[:, ri]
            if cfg.face_gain_prior:
                fresh_h = max(1.0 - age0 - (h + 1) / ttl, 0.0)
                vh = vh * (fresh_h / max(1.0 - age0, 1e-6))
            D[h] = kap * qh
            W[h] = np.maximum(vh - cfg.face_lam * v.S, 0.0)
        return dict(D=D, W=W, f={})

    def _f_rows(self, pk, P, a):
        """First-delivery distribution f_{az}^{(h)} of a copy starting in
        zone a (eq:first_contact_survival, rho^(1) = e_a P-hat)."""
        f = pk["f"].get(a)
        if f is None:
            H, Z = pk["D"].shape
            f = np.empty((H, Z))
            rho = P[a].copy()
            for h in range(H):
                f[h] = rho * pk["D"][h]
                rho = (rho * (1.0 - pk["D"][h])) @ P
            pk["f"][a] = f
        return f

    def _gain_zone(self, v, Psi):
        if self.flags["use_ridge"]:
            g = self.ridge.predict(v.r, Psi)
        else:
            n, mu = self.gstat.get((v.src, v.r), (0, 1.0))
            g = np.full(Psi.shape[0], mu if n else 1.0)
        if self.cfg.face_gain_prior:
            # causal prior at zone-mean strength, discounted by version age
            # (the frozen s_meta overstates stale snapshots)
            prior = np.clip((v.s_meta - Psi[:, 2])
                            / np.maximum(1.0 - Psi[:, 2], 1e-6), 0.0, 1.0)
            age = min((self._clock - v.t_gen) / max(self.cfg.face_ttl, 1), 1.0)
            g = np.maximum(g, prior * (1.0 - age))
        return g

    def _omega(self, pk, P, nvec):
        """Residual coverage Omega [H,Z] (eq:residual_coverage)."""
        H, Z = pk["D"].shape
        Om = np.ones((H, Z))
        if not self.flags["use_coverage"]:
            return Om
        for zeta in np.nonzero(nvec)[0]:
            f_z = self._f_rows(pk, P, int(zeta))
            Om *= np.clip(1.0 - f_z, 0.0, 1.0) ** nvec[zeta]
        return Om

    def _F(self, a, pk, P, nvec):
        """Continuation value of one copy in zone a
        (eq:forwarding_potential)."""
        if not self.flags["use_future"]:
            return 0.0
        Om = self._omega(pk, P, nvec)
        f_a = self._f_rows(pk, P, int(a))
        H = pk["D"].shape[0]
        beta = self.cfg.face_beta
        return float(sum((beta ** h) * (f_a[h] @ (Om[h] * pk["W"][h]))
                         for h in range(H)))

    # ------------------------------------------------------------ aggregation
    def _aggregate(self, k, need, recv_agg):
        """Round phase (ii): FedAvg over the aggregation set with the
        acceptance test, LOO attribution (eq:loo_attribution), predictor
        updates, and resolution bookkeeping."""
        mfl, cfg = self.mfl, self.cfg
        self._n_adopt = 0
        self._round_gain = 0.0
        for i in range(mfl.N):
            by_mod = {}
            seen = set()
            # newly delivered requested encoders (consume or retain mode)
            for (x, vhat) in recv_agg.get(i, []):
                v = self.versions.get(x)
                if v is None or x in seen:
                    continue
                seen.add(x)
                by_mod.setdefault(v.r, []).append((vhat, x))
            # cached relay copies that became requestable (deferred demand)
            for x, m in self.tickets[i].items():
                v = self.versions.get(x)
                if v is None or m <= 0 or x in seen:
                    continue
                if v.compat[i] and not v.resolved[i] \
                        and self._need_ok[v.r][i]:
                    seen.add(x)
                    by_mod.setdefault(v.r, []).append((0.0, x))
            if not by_mod:
                continue
            # examine modalities in decreasing order of best predicted reward
            order = sorted(by_mod.items(),
                           key=lambda kv: -max(w for w, _ in kv[1]))
            for r, lst in order:
                lst.sort(reverse=True)
                # aggregation-set size: how many encoders a vehicle REQUESTS
                # per modality-round (a selective-request policy; averaging
                # many candidates dilutes fresh encoders with stale relays)
                lst = lst[:max(getattr(cfg, "face_agg_max", 1), 1)]
                cands = [(self.versions[x].src, self.versions[x].payload)
                         for (_, x) in lst]
                accepted, attr, base, full = mfl.aggregate_test(i, r, cands)
                s_own = float(mfl.strength.get((i, r), 0.0))
                logD = np.log1p(mfl.Dmr(i, r)) / np.log1p(cfg.data_max)
                got_gain = False
                for a, (_, x) in enumerate(lst):
                    v = self.versions[x]
                    v.resolved[i] = True
                    vx = float(np.clip(attr.get(a, 0.0), 0.0, 1.0))
                    psi = self._psi(x, s_own, need.get((i, r), 0.0), logD, k)
                    self.ridge.update(r, psi, vx)
                    n, mu = self.gstat.get((v.src, r), (0, 0.0))
                    self.gstat[(v.src, r)] = \
                        (n + 1, mu + (vx - mu) / (n + 1))
                    if accepted and vx > 0.0:
                        got_gain = True
                        self._round_gain += vx
                        if (v.src, i) not in self._evermet and v.src != i:
                            self._n_beyond_adopt += 1
                if accepted and got_gain:
                    self._n_adopt += 1

    # ------------------------------------------------------------ round entry
    def run_round(self, k, gamma=None, gamma_eval=None):
        cfg, mfl, mob = self.cfg, self.mfl, self.mob
        self._clock = k
        self._n_beyond_adopt = 0
        A = mob.v2v_graph()
        zn = self._zone_now()
        # expiry (a vehicle's own latest publication lives in model storage
        # and never expires from the relay system)
        latest = set(self.own_pub.values())
        for x in [x for x, v in self.versions.items()
                  if k > v.t_exp and x not in latest]:
            for i in range(mfl.N):
                self.tickets[i].pop(x, None)
            self.versions.pop(x)
        self._update_estimators(k, zn)
        need = modality_needs(cfg, mfl)
        # requester demand gate (Sec. III-C): mu_{i,r} >= delta_mu
        self._need_ok = {r: np.array([need.get((i, r), 0.0) >= cfg.face_delta_d
                                      for i in range(mfl.N)])
                         for r in cfg.modalities}
        # per-vehicle contact registration
        for i in range(mfl.N):
            for j in mob.neighbors(A, i):
                self._evermet.add((i, int(j)))
                self._evermet.add((int(j), i))

        P = self._P_hat()
        # zone-mean receiver features per modality (for v-bar)
        Z = self.zones.Z
        zstats = {}
        for r in cfg.modalities:
            s = np.zeros(Z)
            nd = np.zeros(Z)
            ld = np.zeros(Z)
            c = np.zeros(Z)
            for i in range(mfl.N):
                if r in mfl.avail[i]:
                    z = zn[i]
                    c[z] += 1
                    s[z] += mfl.strength[(i, r)]
                    nd[z] += need.get((i, r), 0.0)
                    ld[z] += np.log1p(mfl.Dmr(i, r)) / np.log1p(cfg.data_max)
            occ = c > 0
            for arr in (s, nd, ld):
                arr[occ] /= c[occ]
            zstats[r] = (s, nd, ld)

        # historical zone profiles feeding the forecast blend
        cnt_all = np.bincount(zn, minlength=Z).astype(float)
        occ_z = cnt_all > 0
        for r in cfg.modalities:
            ri = self.r_idx[r]
            vs = [v for v in self.versions.values() if v.r == r]
            if not vs:
                continue
            req_frac = np.mean([(v.compat & ~v.resolved & self._need_ok[r])
                                for v in vs], axis=0)
            q_cur = np.divide(np.bincount(zn, weights=req_frac, minlength=Z),
                              cnt_all, out=np.zeros(Z), where=occ_z)
            s_mean, need_mean, logD_mean = zstats[r]
            s_meta = float(np.mean([v.s_meta for v in vs]))
            Psi = np.stack([np.ones(Z), np.full(Z, s_meta), s_mean,
                            need_mean, logD_mean, np.zeros(Z)], axis=1)
            if self.flags["use_ridge"]:
                g_z = self.ridge.predict(r, Psi)
            else:
                stats = [m for (src, rr), (n, m) in self.gstat.items()
                         if rr == r and n > 0]
                g_z = np.full(Z, float(np.mean(stats)) if stats else 1.0)
            v_cur = need_mean * g_z
            self.q_hist[occ_z, ri] = (0.9 * self.q_hist[occ_z, ri]
                                      + 0.1 * q_cur[occ_z])
            self.v_hist[occ_z, ri] = (0.9 * self.v_hist[occ_z, ri]
                                      + 0.1 * v_cur[occ_z])

        packs = {}

        def pack(x):
            if x not in packs:
                packs[x] = self._pack(x, zn, P, zstats)
            return packs[x]

        # copy vectors n_x over zones (round-frozen background)
        nvec = {}
        for x in self.versions:
            holders = [i for i in range(mfl.N) if self.tickets[i].get(x, 0) > 0]
            if holders:
                nvec[x] = np.bincount(zn[holders], minlength=Z).astype(float)

        rate, T = cfg.tx_rate_mbps, cfg.contact_time_per_round

        # mmFedMC: each sender offers only its top own modality encoder,
        # ranked by contribution per communication cost
        top_own = {}
        if self.flags["select"] == "mmfedmc":
            for i in range(mfl.N):
                own = [(mfl.strength[(i, r)] / cfg.encoder_size[r],
                        self.own_pub[(i, r)]) for r in mfl.avail[i]
                       if (i, r) in self.own_pub]
                if own:
                    top_own[i] = max(own)[1]

        # retention values of cached relay copies (eq:retention_value)
        keepv_cache = {}

        def keep_value(j, y):
            kv = keepv_cache.get((j, y))
            if kv is None:
                if self.flags["refresh"] == "lru" or \
                        not self.flags["use_future"]:
                    kv = 0.0                    # LRU baselines: drop-oldest
                else:
                    ex = nvec.get(y, np.zeros(Z)).copy()
                    ex[zn[j]] = max(ex[zn[j]] - 1, 0)
                    kv = self._F(zn[j], pack(y), P, ex)
                keepv_cache[(j, y)] = kv
            return kv

        def vhat_of(u, v_veh, x):
            """Predicted reward of delivering x to v_veh (eq:predicted_gain),
            requester-gated, with the age-discounted causal prior."""
            v = self.versions[x]
            if self.flags["select"] == "autofed":
                return (0.05 + 0.5 * v.s_meta) \
                    if (v.compat[v_veh] and not v.resolved[v_veh]) else 0.0
            if not self.flags["use_demand"]:
                return 0.05 if (v.compat[v_veh]
                                and not v.resolved[v_veh]) else 0.0
            if not (v.compat[v_veh] and not v.resolved[v_veh]
                    and self._need_ok[v.r][v_veh]):
                return 0.0
            s_own = float(mfl.strength.get((v_veh, v.r), 0.0))
            psi = self._psi(x, s_own, need.get((v_veh, v.r), 0.0),
                            np.log1p(mfl.Dmr(v_veh, v.r))
                            / np.log1p(cfg.data_max), k)
            if self.flags["use_ridge"]:
                g = float(self.ridge.predict(v.r, psi[None])[0])
            else:
                n, mu = self.gstat.get((v.src, v.r), (0, 1.0))
                g = mu if n else 1.0
            if cfg.face_gain_prior:
                # causal prior (s_meta - s_own)/(1 - s_own), age-discounted:
                # s_meta is frozen at publication while receivers keep training
                prior = max(v.s_meta - s_own, 0.0) / max(1.0 - s_own, 1e-6)
                age = min((k - v.t_gen) / max(cfg.face_ttl, 1), 1.0)
                g = max(g, min(prior * (1.0 - age), 1.0))
            else:
                g = max(g, cfg.face_g_floor)
            return need.get((v_veh, v.r), 0.0) * g

        def dir_items(u, v_veh):
            """Candidate transfers u -> v_veh with both mode advantages
            (eq:transfer_advantage, eq:candidate_set)."""
            out = []
            for x, m in self.tickets[u].items():
                if m <= 0 or x not in self.versions \
                        or self.tickets[v_veh].get(x, 0) > 0:
                    continue
                if not self.flags["use_relay"] and not self._is_own_pub(u, x):
                    continue                # no ferrying of others' encoders
                if self.flags["select"] == "mmfedmc" and x != top_own.get(u):
                    continue
                if x not in nvec:
                    continue
                v = self.versions[x]
                vh = vhat_of(u, v_veh, x)
                lamS = cfg.face_lam * v.S
                a_con = max(vh - lamS, 0.0) if self.flags["use_consume"] \
                    else 0.0
                a_ret = 0.0
                can_ret = (m >= 2 or not self.flags["use_tickets"]) \
                    and self.flags["use_relay"]
                if can_ret:
                    if self.flags["use_future"]:
                        Fv = self._F(zn[v_veh], pack(x), P, nvec[x])
                    else:
                        Fv = BLIND_CACHE_BONUS      # blind caching policies
                    a_ret = max(vh + Fv - lamS, 0.0)
                if a_con <= 0.0 and a_ret <= 0.0:
                    continue
                out.append(dict(u=u, v=v_veh, x=x,
                                w=max(int(np.ceil(v.S)), 1), S=v.S,
                                con=a_con, ret=a_ret, vhat=vh))
            return out

        def evict_profile(v_veh):
            """g_v(f): minimum retention loss freeing >= f MB at v_veh, with
            exact reconstruction (part of eq:bundle_knapsack)."""
            evs = [(keep_value(v_veh, y), self.versions[y].S, y)
                   for y, my in self.tickets[v_veh].items()
                   if my > 0 and y in self.versions
                   and not self._is_own_pub(v_veh, y)]
            if self.flags["refresh"] == "lru":
                evs.sort(key=lambda e: self.lru[v_veh].get(e[2], -1))
            capf = int(np.ceil(cfg.cache_capacity_mb)) + 1
            nev = len(evs)
            EV = np.full((nev + 1, capf), np.inf)
            EV[0, 0] = 0.0
            for a in range(1, nev + 1):
                kv, Sy, _ = evs[a - 1]
                w = max(int(np.ceil(Sy)), 1)
                for f in range(capf):
                    take = EV[a - 1, max(f - w, 0)] + kv
                    EV[a, f] = min(EV[a - 1, f], take)

            def evict_set(f):
                sel, a, ff = [], nev, f
                while a > 0 and ff > 0:
                    kv, Sy, y = evs[a - 1]
                    w = max(int(np.ceil(Sy)), 1)
                    if EV[a, ff] == EV[a - 1, ff]:
                        a -= 1
                    else:
                        sel.append(y)
                        ff = max(ff - w, 0)
                        a -= 1
                return sel
            return EV[nev], evict_set, capf

        def dir_profile(items, v_veh, B):
            """Layered DP over (sent MB, retained MB) with per-item mode
            choice, folded with the receiver eviction profile: returns
            val(b) = best direction value using exactly <= b MB, plus a
            reconstructor of (mode assignments, evictions)."""
            layers = [np.full((B + 1, B + 1), -np.inf)]
            layers[0][0, 0] = 0.0
            for it in items:
                prev = layers[-1]
                cur = prev.copy()
                w = it["w"]
                if w <= B:
                    if it["con"] > 0:
                        cand = prev[:B + 1 - w, :] + it["con"]
                        np.maximum(cur[w:, :], cand, out=cur[w:, :])
                    if it["ret"] > 0:
                        cand = prev[:B + 1 - w, :B + 1 - w] + it["ret"]
                        np.maximum(cur[w:, w:], cand, out=cur[w:, w:])
                layers.append(cur)
            gv, evict_set, capf = evict_profile(v_veh)
            cfree = cfg.cache_capacity_mb - self._cache_used(v_veh)
            loss = np.full(B + 1, np.inf)
            for s in range(B + 1):
                fneed = int(np.ceil(max(s - cfree, 0.0) - 1e-9))
                loss[s] = gv[fneed] if fneed < capf else np.inf
            net = layers[-1] - loss[None, :]
            val = np.max(net, axis=1)
            s_star = np.argmax(net, axis=1)

            def recon(b):
                s = int(s_star[b])
                fneed = int(np.ceil(max(s - cfree, 0.0) - 1e-9))
                ev = evict_set(fneed) if fneed > 0 else []
                sel, bb, ss = [], b, s
                for a in range(len(items) - 1, -1, -1):
                    it, w = items[a], items[a]["w"]
                    cur, prev = layers[a + 1], layers[a]
                    if cur[bb, ss] == prev[bb, ss]:
                        continue
                    if ss >= w and it["ret"] > 0 and \
                            np.isclose(cur[bb, ss],
                                       prev[bb - w, ss - w] + it["ret"]):
                        sel.append((it, "ret"))
                        bb, ss = bb - w, ss - w
                    elif it["con"] > 0 and \
                            np.isclose(cur[bb, ss],
                                       prev[bb - w, ss] + it["con"]):
                        sel.append((it, "con"))
                        bb = bb - w
                return sel, ev
            return val, recon

        dbg = getattr(self, "dbg", None)

        def bundle_directed(i, j):
            """Directed-activation variant (cfg.face_directed): an activated
            contact carries transfers in ONE direction only, using the full
            sojourn budget; each vehicle joins at most one activation."""
            ptx = mob.link_quality(i, j)
            ptx_s = 1.0 if self.flags["link_blind"] else max(ptx, 0.05)
            frac = self._sojourn_frac(i, j)
            B_mb = rate * ptx_s * T * frac
            B = int(B_mb)
            if B < 1:
                return 0.0, [], {}, B_mb
            items = dir_items(i, j)
            if not items:
                return 0.0, [], {}, B_mb
            val, recon = dir_profile(items, j, B)
            b = int(np.argmax(val))
            if val[b] <= 1e-12:
                return 0.0, [], {}, B_mb
            sel, ev = recon(b)
            return float(val[b]), sel, {j: ev}, B_mb

        def bundle(i, j):
            """Joint bidirectional bundle value Phi_ij (eq:bundle_knapsack):
            both directions share the sojourn contact budget; per-item mode
            choice and receiver-side evictions are selected jointly."""
            ptx = mob.link_quality(i, j)
            ptx_s = 1.0 if self.flags["link_blind"] else max(ptx, 0.05)
            frac = self._sojourn_frac(i, j)
            B_mb = rate * ptx_s * T * frac
            B = int(B_mb)
            if dbg is not None:
                dbg["pairs"] = dbg.get("pairs", 0) + 1
                dbg.setdefault("B", []).append(B_mb)
            if B < 1:
                return 0.0, [], {}, B_mb
            it_ij = dir_items(i, j)
            it_ji = dir_items(j, i)
            if dbg is not None:
                dbg["items"] = dbg.get("items", 0) + len(it_ij) + len(it_ji)
                dbg["fit"] = dbg.get("fit", 0) + sum(
                    1 for it in it_ij + it_ji if it["w"] <= B)
            if not it_ij and not it_ji:
                return 0.0, [], {}, B_mb
            val_ij, rec_ij = dir_profile(it_ij, j, B)
            val_ji, rec_ji = dir_profile(it_ji, i, B)
            # best split of the shared budget across the two directions
            best, b1s = -np.inf, 0
            run_ji = np.maximum.accumulate(val_ji)
            for b1 in range(B + 1):
                tot = val_ij[b1] + run_ji[B - b1]
                if tot > best:
                    best, b1s = tot, b1
            if best <= 1e-12:
                return 0.0, [], {}, B_mb
            b2s = int(np.argmax(val_ji[:B - b1s + 1]))
            sel1, ev1 = rec_ij(b1s)
            sel2, ev2 = rec_ji(b2s)
            if dbg is not None:
                dbg["phi_pos"] = dbg.get("phi_pos", 0) + 1
                dbg["recon"] = dbg.get("recon", 0) + len(sel1) + len(sel2)
            return float(best), sel1 + sel2, {j: ev1, i: ev2}, B_mb

        # ---------- pairing protocol (eq:matching_constraint) ----------
        # each vehicle proposes to the neighbor with the largest positive
        # bundle value; a contact activates on mutual proposal; repeat among
        # the unmatched. The globally best remaining pair is always mutual,
        # so the protocol terminates.
        committed = []      # (i, j, actions, evicts, B_mb)
        if getattr(cfg, "face_directed", False):
            # directed activations: greedy over ordered pairs; a commitment
            # removes BOTH endpoints (half-duplex single-peer)
            dpairs = {}
            for i in range(mfl.N):
                for j in mob.neighbors(A, i):
                    dpairs[(i, int(j))] = None
            while True:
                best, bkey = 0.0, None
                for key in dpairs:
                    if dpairs[key] is None:
                        dpairs[key] = bundle_directed(*key)
                    if dpairs[key][0] > best:
                        best, bkey = dpairs[key][0], key
                if bkey is None or best <= 1e-12:
                    break
                i, j = bkey
                _, sel, evicts, B_mb = dpairs[bkey]
                committed.append((i, j, sel, evicts, B_mb))
                for key in [p for p in dpairs if i in p or j in p]:
                    dpairs.pop(key)
        cand_pairs = {}
        if not getattr(cfg, "face_directed", False):
            for i in range(mfl.N):
                for j in mob.neighbors(A, i):
                    j = int(j)
                    if i < j:
                        cand_pairs[(i, j)] = None
        unmatched = set(range(mfl.N))
        while True:
            best_of = {}
            for (i, j) in cand_pairs:
                if i not in unmatched or j not in unmatched:
                    continue
                if cand_pairs[(i, j)] is None:
                    cand_pairs[(i, j)] = bundle(i, j)
                phi = cand_pairs[(i, j)][0]
                if phi <= 1e-12:
                    continue
                if phi > best_of.get(i, (0.0, -1))[0]:
                    best_of[i] = (phi, j)
                if phi > best_of.get(j, (0.0, -1))[0]:
                    best_of[j] = (phi, i)
            acts = [(i, j) for (i, j) in cand_pairs
                    if best_of.get(i, (0, -1))[1] == j
                    and best_of.get(j, (0, -1))[1] == i]
            if not acts:
                break
            for (i, j) in acts:
                _, sel, evicts, B_mb = cand_pairs[(i, j)]
                committed.append((i, j, sel, evicts, B_mb))
                unmatched.discard(i)
                unmatched.discard(j)
        if getattr(self, "dbg", None) is not None:
            self.dbg["committed"] = len(committed)
            self.dbg["acted"] = sum(len(s) for (_, _, s, _, _) in committed)

        # ---------- execute transfers (Bernoulli link success) ----------
        self._n_tx = self._n_deliv = self._n_relay = self._n_beyond = 0
        self.last_tx_mb = 0.0
        sched, recv = [], {}
        recv_agg = {}
        activated = []
        for (i, j, sel, evicts, B_mb) in committed:
            activated.append((i, j, B_mb))
            ptx = mob.link_quality(i, j)
            evq = {e: list(lst) for e, lst in evicts.items()}
            for (it, mode) in sel:
                u, v_veh, x = it["u"], it["v"], it["x"]
                v = self.versions.get(x)
                if v is None:
                    continue
                sched.append((u, v_veh, v))
                self._n_tx += 1
                self.last_tx_mb += it["S"]
                if it["S"] / (rate * max(ptx, 0.05)) \
                        > T * self._sojourn_frac(u, v_veh):
                    continue            # link-blind overrun: airtime lost
                if self.rng.random() > ptx:
                    continue            # terminated transfer: airtime lost
                self._n_deliv += 1
                recv.setdefault((v_veh, v.r), []).append(v.s_meta)
                if it["vhat"] > 0.0:    # delivered to a requester
                    recv_agg.setdefault(v_veh, []).append((x, it["vhat"]))
                if mode == "ret":
                    ku = self.tickets[u].get(x, 0)
                    if ku >= 2 or not self.flags["use_tickets"]:
                        # lazy eviction on acknowledged arrival
                        while evq.get(v_veh) and \
                                self._cache_used(v_veh) + it["S"] \
                                > cfg.cache_capacity_mb + 1e-9:
                            y = evq[v_veh].pop(0)
                            self.tickets[v_veh].pop(y, None)
                            self.lru[v_veh].pop(y, None)
                        # binary spray: pass floor(k/2) tokens, keep the rest
                        g = max(ku // 2, 1)
                        self.tickets[u][x] = ku - g
                        if self.tickets[u][x] <= 0:
                            self.tickets[u][x] = 1      # sender keeps a copy
                        self.tickets[v_veh][x] = g
                        self.lru[v_veh][x] = k
                if v.src != u:
                    self._n_relay += 1
                if (v.src, v_veh) not in self._evermet and v.src != v_veh:
                    self._n_beyond += 1
        self._credit_activated(zn, activated)
        self._score_round(sched, recv, need, gamma_eval, zn)

        # ---------- aggregation phase (ii): FedAvg + acceptance + LOO ----
        self._aggregate(k, need, recv_agg)

        # ---------- coverage-aware cache refresh (eq:cache_refresh) ----------
        for i in range(mfl.N):
            relay = [x for x, m in self.tickets[i].items()
                     if m > 0 and x in self.versions
                     and not self._is_own_pub(i, x)]
            if not relay:
                continue
            cap = cfg.cache_capacity_mb
            if self.flags["refresh"] == "lru" or not self.flags["use_future"]:
                ranked = sorted(relay, key=lambda x: -self.lru[i].get(x, -1))
            else:
                scored = []
                for x in relay:
                    v = self.versions[x]
                    ex = nvec.get(x, np.zeros(Z)).copy()
                    ex[zn[i]] = max(ex[zn[i]] - 1, 0)
                    keepv = self._F(zn[i], pack(x), P, ex)
                    scored.append((keepv / v.S, x))
                # a copy with no positive continuation value is dropped even
                # if it fits, freeing relay space for future receptions
                ranked = [x for d, x in sorted(scored, reverse=True)
                          if d > 1e-9]
            used, keep = 0.0, set()
            for x in ranked:
                if used + self.versions[x].S <= cap + 1e-9:
                    keep.add(x)
                    used += self.versions[x].S
            for x in relay:
                if x not in keep:       # eviction destroys the copy's tokens
                    self.tickets[i].pop(x, None)
                    self.lru[i].pop(x, None)

        # ---------- publication (real backend; no-op with static encoders) ---
        if hasattr(self.mfl, "snapshot_encoder") and \
                cfg.face_Qpub > 0 and (k + 1) % cfg.face_Qpub == 0:
            self._publish_all(t=k + 1)

        # prune dead versions (no tokens anywhere, not a latest publication)
        alive = set(self.own_pub.values())
        for i in range(mfl.N):
            alive.update(x for x, m in self.tickets[i].items() if m > 0)
        for x in [x for x in self.versions if x not in alive]:
            self.versions.pop(x)

        # scheduled transfers of the round (Version refs captured at
        # execution time, before publication/pruning retires old versions)
        self.last_selected = [(u, vv, v.src, v.r) for (u, vv, v) in sched]
        return self.last_selected

    # ---------- scheme-independent round scoring (same formulas the old
    # engine reports, so Table I utilities are comparable across schemes) ----
    def _score_round(self, sched, recv, need, gamma_eval, zn):
        cfg, mfl, mob = self.cfg, self.mfl, self.mob
        Dr = mean_modality_data(mfl)
        learn_prod, fwd_prod = {}, {}
        for (i, j, v) in sched:
            ptx = mob.link_quality(i, j)
            g = max(v.s_meta - float(mfl.strength.get((j, v.r), 0.0)), 0.0)
            be = float(np.clip(ptx * (mfl.Dmr(v.src, v.r)
                                      / (Dr[v.r] + cfg.eps0)) * g, 0.0, 0.999))
            learn_prod[(j, v.r)] = learn_prod.get((j, v.r), 1.0) * (1.0 - be)
            if gamma_eval is not None:
                bf = ptx * (1.0 - np.exp(-float(gamma_eval[j]) * v.s_meta))
                fwd_prod[(i, v.src, v.r)] = \
                    fwd_prod.get((i, v.src, v.r), 1.0) * (1.0 - bf)
        u_learn = sum(need.get(jr, 0.0) * (1.0 - p)
                      for jr, p in learn_prod.items())
        u_fwd = sum(1.0 - p for p in fwd_prod.values())
        self.last_utility_learn = float(u_learn)
        self.last_utility_fwd = float(cfg.nu * u_fwd)
        self.last_utility_txcost = float(cfg.lam_tx * len(sched))
        self.last_utility = (self.last_utility_learn + self.last_utility_fwd
                             - self.last_utility_txcost)
        # demand-weighted satisfaction of successful deliveries
        tot_need = sum(need.values()) + 1e-9
        self.last_satisfaction = sum(need.get(jr, 0.0) for jr in recv) / tot_need
        useful = [jr for jr, lst in recv.items()
                  if any(s > float(mfl.strength.get(jr, 0.0)) + 0.02
                         for s in lst)]
        self.last_useful_sat = sum(need.get(jr, 0.0) for jr in useful) / tot_need
        # high-demand road-segment availability (same metric as the old engine)
        seg_need, seg_have = {}, {}
        segs = mob.seg
        for i in range(mfl.N):
            e = int(segs[i])
            held = [self.versions[x] for x, m in self.tickets[i].items()
                    if m > 0 and x in self.versions]
            for r in mfl.avail[i]:
                a = need.get((i, r), 0.0)
                seg_need[(e, r)] = 1.0 - (1.0 - seg_need.get((e, r), 0.0)) \
                    * (1.0 - a)
                if not seg_have.get((e, r), False):
                    if any(v.r == r and v.s_meta >= 0.6 for v in held):
                        seg_have[(e, r)] = True
        if seg_need:
            lam = np.array(list(seg_need.values()))
            thr = np.quantile(lam, 0.75)
            top = [kk for kk, vv in seg_need.items() if vv >= thr]
            self.last_avail = (sum(seg_have.get(kk, False) for kk in top)
                               / max(len(top), 1))
        else:
            self.last_avail = 0.0
