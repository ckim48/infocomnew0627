"""
FACE: future-contact-aware encoder caching and forwarding (new system model).

Implements the revised paper model (Sec. III-IV):
  * immutable encoder VERSIONS x = (src, r, nu, payload, t_gen, t_exp, S, K)
    with copy tickets m_{i,x} bounded by K_x (replication / custody transfer)
  * relay caches separate from the vehicle's own latest published encoders
  * causal zone estimators: decayed transition counts -> P-hat (Eq. 11),
    Beta-smoothed size-binned complete-contact prob kappa-hat (Eq. 12),
    ESV fraction q_{z,x} and conditional mean reward v-bar (Eq. 13)
  * sliding-window ridge gain prediction with optimism (Eq. 10)
  * first-contact survival Gamma (Eq. 14), residual coverage Omega (Eq. 15),
    coverage-aware continuation value F (Eq. 16)
  * marginal transfer advantage A_ijx with reciprocal priority factor
    (Eq. transfer_advantage), JOINT admission-eviction bundle selection
    (Eq. bundle_knapsack: two 1-D DPs, h(s) admitted value / g(f) min
    retention loss), greedy max-weight matching under the half-duplex
    single-peer constraint (Eq. matching_constraint, Prop. 2)
  * coverage-aware cache refresh knapsack (Eq. cache_refresh)
  * reputation and reciprocal cooperation (Sec. III-E): delivery credit
    u v + mu_f (1-u) v-hat (Eq. rep_delivery), storage credit mu_s S c
    (Eq. rep_storage), decayed state Psi (Eq. rep_update), zone-normalized
    priority pi (Eq. rep_priority)
  * ESV demand threshold delta_d (Eq. esv_indicator); evaluation/adoption
    happens AFTER the contact phase (round steps S2 -> S3)

Ablation flags (FACE_FLAGS) switch each mechanism off independently so the
component ablation can attribute the end-to-end gain.
"""

import numpy as np
from .utility import modality_needs, mean_modality_data


# ---------------------------------------------------------------- versions
class Version:
    __slots__ = ("vid", "src", "r", "nu", "payload", "s_meta", "arch",
                 "t_gen", "t_exp", "S", "K", "compat", "resolved")

    def __init__(self, vid, src, r, nu, payload, s_meta, t_gen, t_exp, S, K, N,
                 avail, arch=None):
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
        self.resolved = np.zeros(N, dtype=bool)                     # rho_{i,x}


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
    """Per-modality sliding-window ridge with optimism (Eq. 10)."""
    DIM = 6

    def __init__(self, cfg):
        self.cfg = cfg
        self.A = {}   # r -> [d,d]
        self.b = {}   # r -> [d]

    def _get(self, r):
        if r not in self.A:
            self.A[r] = np.eye(self.DIM) * self.cfg.face_ridge_lam
            # weak prior mean of 0.3 on the intercept: bootstraps forwarding
            # before any gains are observed, washed out by real samples
            self.b[r] = np.zeros(self.DIM)
            self.b[r][0] = self.cfg.face_ridge_lam * 0.3
        return self.A[r], self.b[r]

    def update(self, r, psi, y):
        A, b = self._get(r)
        g = self.cfg.face_ridge_decay
        self.A[r] = g * A + np.outer(psi, psi) \
            + (1 - g) * np.eye(self.DIM) * self.cfg.face_ridge_lam
        self.b[r] = g * b + psi * y

    def predict(self, r, Psi, optimistic=False):
        """Psi: [n, d] -> gains in [0,1]. Posterior mean by default (used for
        forwarding and zone summaries, Eq. 10); the optimistic UCB variant is
        used only to rank candidates for local evaluation under N_ev."""
        A, b = self._get(r)
        Ainv = np.linalg.inv(A)
        mu = Psi @ (Ainv @ b)
        if optimistic:
            mu = mu + self.cfg.face_alpha_g * np.sqrt(
                np.maximum(np.einsum("nd,dk,nk->n", Psi, Ainv, Psi), 0.0))
        return np.clip(mu, 0.0, 1.0)


FACE_FLAGS = dict(use_future=True,     # F continuation value (Eq. 16) in A_ijx
                  use_coverage=True,   # residual coverage Omega (Eq. 15)
                  use_tickets=True,    # replication bound K_x / custody transfer
                  use_split=True,      # value-weighted ticket splitting (Eq. 7)
                  use_ridge=True,      # ridge gain prediction (else mean bandit)
                  use_relay=True,      # forward cached copies of OTHERS' versions
                  use_demand=True,     # receiver-demand-aware immediate value
                  use_recip=True,      # reciprocal priority (Sec. III-E):
                                       # measured +0.7pp on real KITTI --
                                       # keep the reputation section in the
                                       # paper (v3 draft has the full text)
                  link_blind=False,    # schedule ignoring the V2V link quality
                  select=None,         # candidate rule: None|'mmfedmc'|'autofed'
                  refresh="knapsack")  # cache refresh: 'knapsack' | 'lru'

# All schemes run under the SAME system-model protocol (encoder versions,
# copy tickets, evaluation-gated adoption, publication cadence, Sec. III);
# they differ only in the forwarding POLICY, exactly as in the paper.
SCHEME_FACE_FLAGS = {
    "Proposed":         {},
    # pure V2V opportunism: picks link-quality-favorable neighbors and hands
    # over own encoders RANDOMLY (demand-blind), no carrying
    "V2V-aware":        dict(use_relay=False, use_future=False,
                             use_coverage=False, refresh="lru",
                             use_demand=False, use_recip=False),
    # like V2V-aware but blind to the V2V link condition when scheduling
    "Learning-aware":   dict(use_relay=False, use_future=False,
                             use_coverage=False, refresh="lru",
                             link_blind=True, use_recip=False),
    # relays cached encoders, LRU, demand-blind
    "Caching-assisted": dict(use_demand=False, use_future=False,
                             use_coverage=False, refresh="lru",
                             use_recip=False),
    # mmFedMC: sender offers only its top own modality by contribution/cost
    "mmFedMC":          dict(use_relay=False, use_future=False,
                             use_coverage=False, refresh="lru",
                             select="mmfedmc", use_recip=False),
    # AutoFed: prioritize high-quality (clean, data-rich) sources
    "AutoFed":          dict(use_relay=False, use_future=False,
                             use_coverage=False, refresh="lru",
                             select="autofed", use_recip=False),
}


class FACE:
    """Drop-in replacement for CachingForwarding implementing the new model."""

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
        # --- version / ticket state ---
        self.versions = {}                  # vid -> Version
        self.tickets = {i: {} for i in range(mfl.N)}   # i -> vid -> m_{i,x}
        self.own_pub = {}                   # (i,r) -> vid (latest, ell=1)
        self.lru = {i: {} for i in range(mfl.N)}
        self._vid = 0
        self._clock = 0
        self._evermet = set()
        # --- reputation / reciprocal cooperation state (Sec. III-E) ---
        self.Psi = np.zeros(mfl.N)          # reputation Psi_i (Eq. rep_update)
        self._dPsi = np.zeros(mfl.N)        # this-round increment dPsi_i
        self._pi = np.zeros(mfl.N)          # reciprocal priority pi_i (Eq. rep_priority)
        self._deliv_credit = {}             # (receiver, vid) -> (sender, vhat at delivery)
        self._need_ok = {}                  # r -> [N] bool, d_{i,r} >= delta_d
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
                        getattr(self.mfl, "arch", None))
            v.resolved[i] = True            # source never re-adopts its own
            self.versions[self._vid] = v
            self.own_pub[(i, r)] = self._vid
            self.tickets[i][self._vid] = Kt
            # retire the superseded version at the source: its unsent tickets
            # are burned so the stale snapshot does not occupy the source's
            # relay cache (distributed copies elsewhere remain valid)
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
                   if m > 0 and not self._is_own_pub(i, x))

    def _psi(self, x, s_own, need, logD, t):
        v = self.versions[x]
        age = min((t - v.t_gen) / max(self.cfg.face_ttl, 1), 1.0)
        return np.array([1.0, v.s_meta, s_own, need, logD, age])

    # ------------------------------------------------------------ estimators
    def _update_estimators(self, k, A, zn):
        cfg, d = self.cfg, self.cfg.face_decay
        self.Ntrans *= d
        self.Ez *= d
        self.Czb *= d
        if k > 0:
            zp = self.zones.zone[min(k - 1, self.zones.zone.shape[0] - 1)]
            np.add.at(self.Ntrans, (zp, zn), 1.0)
        np.add.at(self.Ez, zn, 1.0)
        # size-binned complete-contact capability per vehicle-round (Eq. 12)
        rate, T = cfg.tx_rate_mbps, cfg.contact_time_per_round
        for i in range(self.mfl.N):
            nbrs = self.mob.neighbors(A, i)
            if len(nbrs) == 0:
                continue
            best = max(self.mob.link_quality(i, int(j)) for j in nbrs)
            cap_mb = rate * max(best, 0.05) * T
            for b, s in enumerate(self.size_bins):
                if s <= cap_mb:
                    self.Czb[zn[i], b] += 1.0

    def _P_hat(self):
        """Smoothed one-step zone transition matrix (Eq. 11)."""
        cfg = self.cfg
        M = self.Ntrans + cfg.face_alpha_P * self.zones.adj
        return M / np.maximum(M.sum(1, keepdims=True), 1e-12)

    def _kappa(self, S):
        b = next(a for a, s in enumerate(self.size_bins) if S <= s + 1e-9)
        cfg = self.cfg
        return (self.Czb[:, b] + cfg.face_alpha_C) / \
               (self.Ez + cfg.face_alpha_C + cfg.face_beta_C)

    # ------------------------------------------------------------ value packs
    def _pack(self, x, zn, need_vec, P, zstats):
        """Per-version arrays: per-round useful-delivery prob D [H,Z]
        (Eq. useful_contact_probability), reward W [H,Z], and lazily filled
        first-contact rows f_a [H,Z] (Eq. first_contact_dist)."""
        cfg, v = self.cfg, self.versions[x]
        H, Z = cfg.face_H, self.zones.Z
        ri = self.r_idx[v.r]
        # ESV indicator (Eq. esv_indicator): compatible, unevaluated, and
        # with modality demand above the threshold delta_d
        esv = v.compat & ~v.resolved & self._need_ok[v.r]
        cnt_all = np.bincount(zn, minlength=Z).astype(float)
        cnt_esv = np.bincount(zn[esv], minlength=Z).astype(float)
        q_cur = np.divide(cnt_esv, cnt_all, out=np.zeros(Z),
                          where=cnt_all > 0)
        # conditional mean predicted reward per zone (Eq. 13, posterior mean)
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
        """First-useful-contact distribution f_{az}^{(h)} of a copy starting
        in zone a, via the absorbing recursion (Eq. first_contact_survival)."""
        f = pk["f"].get(a)
        if f is None:
            H, Z = pk["D"].shape
            f = np.empty((H, Z))
            rho = np.zeros(Z)
            rho[a] = 1.0
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
            # causal prior at zone-mean strength, age-discounted (frozen
            # s_meta overstates stale snapshots)
            prior = np.clip((v.s_meta - Psi[:, 2])
                            / np.maximum(1.0 - Psi[:, 2], 1e-6), 0.0, 1.0)
            age = min((self._clock - v.t_gen) / max(self.cfg.face_ttl, 1), 1.0)
            g = np.maximum(g, prior * (1.0 - age))
        return g

    def _omega(self, pk, P, nvec):
        """Residual coverage Omega [H,Z] (Eq. residual_coverage) for the
        background copy vector nvec."""
        H, Z = pk["D"].shape
        Om = np.ones((H, Z))
        if not self.flags["use_coverage"]:
            return Om
        for zeta in np.nonzero(nvec)[0]:
            f_z = self._f_rows(pk, P, int(zeta))
            Om *= np.clip(1.0 - f_z, 0.0, 1.0) ** nvec[zeta]
        return Om

    def _F(self, a, pk, P, nvec):
        """First-contact continuation value of one copy in zone a
        (Eq. forwarding_potential)."""
        if not self.flags["use_future"]:
            return 0.0
        Om = self._omega(pk, P, nvec)
        f_a = self._f_rows(pk, P, int(a))
        H = pk["D"].shape[0]
        beta = self.cfg.face_beta
        return float(sum((beta ** h) * (f_a[h] @ (Om[h] * pk["W"][h]))
                         for h in range(H)))

    # ------------------------------------------------------------ adoption
    def _adopt(self, k, need, zn):
        mfl, cfg = self.mfl, self.cfg
        self._n_adopt = 0
        for i in range(mfl.N):
            cand = []
            for x, m in self.tickets[i].items():
                v = self.versions.get(x)
                if v is None or m <= 0 or v.resolved[i] or not v.compat[i]:
                    continue
                cand.append(x)
            # evaluation budget N_ev: rank candidates by the OPTIMISTIC
            # predicted reward (exploration lives here, Eq. 10) and evaluate
            # only the top N_ev this round; the rest stay unresolved
            if len(cand) > cfg.face_Nev:
                scores = []
                for x in cand:
                    v = self.versions[x]
                    s_own = float(mfl.strength.get((i, v.r), 0.0))
                    logD = np.log1p(mfl.Dmr(i, v.r)) / np.log1p(cfg.data_max)
                    psi = self._psi(x, s_own, need.get((i, v.r), 0.0), logD, k)
                    if self.flags["use_ridge"]:
                        g = float(self.ridge.predict(v.r, psi[None],
                                                     optimistic=True)[0])
                    else:
                        n, mu = self.gstat.get((v.src, v.r), (0, 1.0))
                        g = mu if n else 1.0
                    scores.append((need.get((i, v.r), 0.0) * g, x))
                scores.sort(reverse=True)
                cand = [x for _, x in scores[:cfg.face_Nev]]
            by_mod = {}
            for x in cand:
                by_mod.setdefault(self.versions[x].r, []).append(x)
            for r, xs in by_mod.items():
                s_own = float(mfl.strength.get((i, r), 0.0))
                logD = np.log1p(mfl.Dmr(i, r)) / np.log1p(cfg.data_max)
                evals = []
                for x in xs:
                    v = self.versions[x]
                    delta = mfl.gain_single(i, r, v.src, v.payload)[0]
                    delta = float(np.clip(delta, 0.0, 1.0))
                    v.resolved[i] = True                     # rho update
                    psi = self._psi(x, s_own, need.get((i, r), 0.0), logD, k)
                    self.ridge.update(r, psi, delta)
                    n, mu = self.gstat.get((v.src, r), (0, 0.0))
                    self.gstat[(v.src, r)] = (n + 1, mu + (delta - mu) / (n + 1))
                    evals.append((delta, x))
                delta, x = max(evals)
                adopted = delta >= cfg.face_delta
                # resolve pending delivery-reputation credits (Eq. rep_delivery):
                # the relay that delivered an ADOPTED encoder earns the realized
                # gain v_{j,x}; a delivered-but-not-adopted one earns mu_f * vhat
                for dlt, xx in evals:
                    cred = self._deliv_credit.pop((i, xx), None)
                    if cred is not None:
                        sender, vhat_d = cred
                        u = adopted and xx == x
                        self._dPsi[sender] += dlt if u \
                            else cfg.face_mu_f * vhat_d
                if adopted:                                  # Eq. adoption
                    v = self.versions[x]
                    mfl.commit(i, r, [(v.src, v.payload)])
                    self._n_adopt += 1
                    if (v.src, i) not in self._evermet and v.src != i:
                        self._n_beyond_adopt += 1

    # ------------------------------------------------------------ round entry
    def run_round(self, k, gamma=None, gamma_eval=None):
        cfg, mfl, mob = self.cfg, self.mfl, self.mob
        self._clock = k
        self._n_beyond_adopt = 0
        A = mob.v2v_graph()
        zn = self._zone_now()
        # expiry (a vehicle's own latest publication lives in model storage,
        # ell=1, and never expires from the relay system)
        latest = set(self.own_pub.values())
        for x in [x for x, v in self.versions.items()
                  if k > v.t_exp and x not in latest]:
            for i in range(mfl.N):
                self.tickets[i].pop(x, None)
            self.versions.pop(x)
        self._update_estimators(k, A, zn)
        need = modality_needs(cfg, mfl)
        # ESV demand gate (Eq. esv_indicator): d_{i,r} >= delta_d
        self._need_ok = {r: np.array([need.get((i, r), 0.0) >= cfg.face_delta_d
                                      for i in range(mfl.N)])
                         for r in cfg.modalities}
        # reciprocal priority pi_j from the zone-average reputation
        # (Eq. rep_priority); identity-free: only zone means are gossiped
        Zz = self.zones.Z
        zsum = np.bincount(zn, weights=self.Psi, minlength=Zz)
        zcnt = np.maximum(np.bincount(zn, minlength=Zz), 1)
        self._pi = np.clip(self.Psi / (zsum[zn] / zcnt[zn] + 1e-6),
                           0.0, cfg.face_pi_cap)
        # per-vehicle contact registration
        for i in range(mfl.N):
            for j in mob.neighbors(A, i):
                self._evermet.add((i, int(j)))
                self._evermet.add((int(j), i))

        P = self._P_hat()
        # zone-mean receiver features per modality (for v-bar in Eq. 13)
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

        # historical zone profiles feeding the Eq. 13 forecast blend,
        # updated once per round from causal (current-round) observations
        cnt_all = np.bincount(zn, minlength=Z).astype(float)
        occ_z = cnt_all > 0
        for r in cfg.modalities:
            ri = self.r_idx[r]
            vs = [v for v in self.versions.values() if v.r == r]
            if not vs:
                continue
            esv_frac = np.mean([(v.compat & ~v.resolved & self._need_ok[r])
                                for v in vs], axis=0)
            q_cur = np.divide(np.bincount(zn, weights=esv_frac, minlength=Z),
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
                packs[x] = self._pack(x, zn, need, P, zstats)
            return packs[x]

        # copy vectors n_x over zones (tentative during matching)
        nvec = {}
        for x in self.versions:
            holders = [i for i in range(mfl.N) if self.tickets[i].get(x, 0) > 0]
            if holders:
                nvec[x] = np.bincount(zn[holders], minlength=Z).astype(float)

        # ---------- bundle value per directed contact (Eq. 17-18) ----------
        rate, T = cfg.tx_rate_mbps, cfg.contact_time_per_round

        dbg = getattr(self, "dbg", None)
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

        # retention values of the receiver's relay copies
        # (Eq. retention_value), cached for the round-frozen valuation
        keepv_cache = {}

        def keep_value(j, y):
            kv = keepv_cache.get((j, y))
            if kv is None:
                v = self.versions[y]
                if v.compat[j] and not v.resolved[j]:
                    kv = np.inf                 # pending own evaluation
                elif self.flags["refresh"] == "lru" or \
                        not self.flags["use_future"]:
                    kv = 0.0                    # LRU baselines: drop-oldest
                else:
                    ex = nvec.get(y, np.zeros(Z)).copy()
                    ex[zn[j]] = max(ex[zn[j]] - 1, 0)
                    kv = self._F(zn[j], pack(y), P, ex)
                keepv_cache[(j, y)] = kv
            return kv

        def bundle(i, j, count=False):
            c = dbg if (count and dbg is not None) else None
            items = []
            ptx = mob.link_quality(i, j)
            cfree = cfg.cache_capacity_mb - self._cache_used(j)
            for x, m in self.tickets[i].items():
                if m <= 0 or self.tickets[j].get(x, 0) > 0:
                    continue
                if not self.flags["use_relay"] and not self._is_own_pub(i, x):
                    continue                # no encoder ferrying
                if self.flags["select"] == "mmfedmc" and x != top_own.get(i):
                    continue                # mmFedMC: top own modality only
                v = self.versions[x]
                if x not in nvec:
                    continue
                if c is not None:
                    c["cand"] += 1
                    if not (v.compat[j] and not v.resolved[j]):
                        c["not_esv"] += 1
                    elif v.s_meta <= float(mfl.strength.get((j, v.r), 0.0)):
                        c["gap_le0"] += 1
                pk = pack(x)
                # immediate predicted reward at j (Eq. 10)
                if self.flags["select"] == "autofed":
                    # AutoFed: quality-ranked, demand-blind
                    vhat = (0.05 + 0.5 * v.s_meta) \
                        if (v.compat[j] and not v.resolved[j]) else 0.0
                elif not self.flags["use_demand"]:
                    # demand-blind: uniform value with random jitter, so the
                    # bundle hands over an arbitrary subset of own encoders
                    vhat = 0.05 + 0.01 * self.rng.random() \
                        if (v.compat[j] and not v.resolved[j]) else 0.0
                elif v.compat[j] and not v.resolved[j] \
                        and self._need_ok[v.r][j]:
                    s_own = float(mfl.strength.get((j, v.r), 0.0))
                    psi = self._psi(x, s_own, need.get((j, v.r), 0.0),
                                    np.log1p(mfl.Dmr(j, v.r))
                                    / np.log1p(cfg.data_max), k)
                    if self.flags["use_ridge"]:
                        g = float(self.ridge.predict(v.r, psi[None])[0])
                    else:
                        n, mu = self.gstat.get((v.src, v.r), (0, 1.0))
                        g = mu if n else 1.0
                    if cfg.face_gain_prior:
                        # causal prior (s_meta - s_own)/(1 - s_own), age-
                        # discounted: s_meta is frozen at publication while
                        # receivers keep training (staleness correction)
                        prior = max(v.s_meta - s_own, 0.0) \
                            / max(1.0 - s_own, 1e-6)
                        age = min((k - v.t_gen) / max(cfg.face_ttl, 1), 1.0)
                        g = max(g, min(prior * (1.0 - age), 1.0))
                    else:
                        # blind exploration floor (ablation reference)
                        g = max(g, cfg.face_g_floor)
                    vhat = need.get((j, v.r), 0.0) * g
                else:
                    vhat = 0.0
                # reciprocal priority factor (Eq. transfer_advantage): an
                # above-average contributor's immediate gain is scaled up,
                # so it is admitted first under link and cache contention
                if self.flags["use_recip"] and vhat > 0.0:
                    vhat *= 1.0 + cfg.face_gamma_r \
                        * max(self._pi[j] - 1.0, 0.0)
                # continuation-value change D_ijx (Eq. 17)
                if m > 1:
                    D = self._F(zn[j], pk, P, nvec[x])
                else:
                    ex = nvec[x].copy()
                    ex[zn[i]] = max(ex[zn[i]] - 1, 0)
                    D = self._F(zn[j], pk, P, ex) \
                        - self._F(zn[i], pk, P, ex)
                adv = vhat + D - cfg.face_lam * v.S
                if adv <= 0:
                    if c is not None:
                        c["adv_le0"] += 1
                    continue
                # link-blind schemes schedule assuming an ideal link; the
                # physical airtime is still consumed at execution time
                ptx_s = 1.0 if self.flags["link_blind"] else max(ptx, 0.05)
                t_tx = v.S / (rate * ptx_s)
                if c is not None and t_tx > T:
                    c["airtime"] += 1
                items.append((adv, t_tx, v.S, x, vhat))
            if not items:
                return 0.0, [], []
            # ---- joint admission-eviction (Eq. bundle_knapsack) ----
            # h(s): 0/1 knapsack on airtime (DP, 0.05 s units)
            cap = int(T / 0.05)
            wts = [max(int(np.ceil(t / 0.05)), 1) for (_, t, _, _, _) in items]
            dp = np.zeros(cap + 1)
            keep = np.zeros((len(items), cap + 1), dtype=bool)
            for a, it in enumerate(items):
                w = wts[a]
                if w > cap:
                    continue
                cand = dp[:cap + 1 - w] + it[0]
                upd = cand > dp[w:]
                keep[a, w:][upd] = True
                dp[w:][upd] = cand[upd]

            def recon(c0):
                sel, cc = [], c0
                for a in range(len(items) - 1, -1, -1):
                    if keep[a, cc]:
                        sel.append(a)
                        cc -= wts[a]
                return sel

            # g(f): min retention loss freeing >= f MB from the receiver's
            # evictable relay copies (pending-evaluation copies are kept)
            evs = [(keep_value(j, y), self.versions[y].S, y)
                   for y, my in self.tickets[j].items()
                   if my > 0 and y in self.versions
                   and not self._is_own_pub(j, y)]
            evs = [e for e in evs if np.isfinite(e[0])]
            if self.flags["refresh"] == "lru":
                evs.sort(key=lambda e: self.lru[j].get(e[2], -1))  # oldest first
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
                out, a, ff = [], nev, f
                while a > 0 and ff > 0:
                    kv, Sy, y = evs[a - 1]
                    w = max(int(np.ceil(Sy)), 1)
                    if EV[a, ff] == EV[a - 1, ff]:
                        a -= 1
                    else:
                        out.append(y)
                        ff = max(ff - w, 0)
                        a -= 1
                return out

            # Phi_ij = max_s [ h(s) - g((s - cfree)^+) ] (Eq. bundle_knapsack)
            best_phi, best_sel, best_ev = 0.0, [], []
            for c0 in range(cap + 1):
                if dp[c0] <= best_phi:
                    continue                 # eviction loss only shrinks value
                idxs = recon(c0)
                if not idxs:
                    continue
                byt = sum(items[a][2] for a in idxs)
                needf = int(np.ceil(max(byt - cfree, 0.0) - 1e-9))
                if needf >= capf or not np.isfinite(EV[nev, needf]):
                    continue
                phi = dp[c0] - EV[nev, needf]
                if phi > best_phi:
                    best_phi = phi
                    best_sel = [items[a] for a in idxs]
                    best_ev = evict_set(needf)
            return best_phi, best_sel, best_ev

        # ---------- distributed link contention = greedy matching ----------
        # Bundle weights are frozen at their round-start values (A3); when a
        # directed contact commits, BOTH endpoints leave contention
        # (half-duplex single-peer, Eq. matching_constraint), so the committed
        # exchanges form a matching on the contact graph and the greedy
        # 1/2-approximation guarantee of Prop. 2 applies.
        pairs = {}
        for i in range(mfl.N):
            for j in mob.neighbors(A, i):
                pairs[(i, int(j))] = None
        committed = []
        while True:
            best, bkey = 0.0, None
            for key in pairs:
                if pairs[key] is None:
                    pairs[key] = bundle(*key, count=True)
                if pairs[key][0] > best:
                    best, bkey = pairs[key][0], key
            if bkey is None or best <= 1e-12:
                break
            i, j = bkey
            _, sel, evicts = pairs[bkey]
            sel_g = []
            for (_, t_tx, S, x, vhat) in sel:
                m_avail = self.tickets[i].get(x, 0)
                pk = pack(x)
                Fi = self._F(zn[i], pk, P, nvec[x])
                Fj = self._F(zn[j], pk, P, nvec[x])
                g = self._split_tickets(m_avail, Fi, Fj)
                sel_g.append((x, S, g, vhat))
            committed.append((i, j, sel_g, evicts))
            # matching: both endpoints leave contention this round
            for key in [p for p in pairs if i in p or j in p]:
                pairs.pop(key)

        # ---------- execute transfers (Bernoulli link success) ----------
        self._n_tx = self._n_deliv = self._n_relay = self._n_beyond = 0
        self.last_tx_mb = 0.0
        sched, recv = [], {}
        for (i, j, sel_g, evicts) in committed:
            ptx = mob.link_quality(i, j)
            T_budget = cfg.contact_time_per_round
            evq = list(evicts)
            for (x, S, g, vhat) in sel_g:
                v = self.versions[x]
                sched.append((i, j, v))
                self._n_tx += 1
                self.last_tx_mb += S
                if S / (rate * max(ptx, 0.05)) > T_budget:
                    continue                # link-blind overrun: airtime lost
                if self.rng.random() > ptx:
                    continue                # terminated transfer: airtime lost
                # lazy eviction on acknowledged arrival (Eq. bundle_knapsack):
                # displaced relay copies burn their tickets (Eq. ticket_update)
                while evq and self._cache_used(j) + S \
                        > cfg.cache_capacity_mb + 1e-9:
                    y = evq.pop(0)
                    self.tickets[j].pop(y, None)
                    self.lru[j].pop(y, None)
                self.tickets[i][x] -= g     # value-weighted split (Eq. 7)
                if self.tickets[i][x] <= 0:      # custody transferred away
                    del self.tickets[i][x]
                self.tickets[j][x] = g
                self.lru[j][x] = k
                self._n_deliv += 1
                recv.setdefault((j, v.r), []).append(v.s_meta)
                # delivery reputation (Eq. rep_delivery): relayed versions
                # (1 - ell = 1) earn credit, resolved at the receiver's
                # evaluation once the adoption outcome u_{j,x} is known
                if not self._is_own_pub(i, x):
                    self._deliv_credit[(j, x)] = (i, vhat)
                if v.src != i:
                    self._n_relay += 1
                if (v.src, j) not in self._evermet and v.src != j:
                    self._n_beyond += 1
        self._score_round(sched, recv, need, gamma_eval, zn)
        # ---------- evaluate + adopt (step S3: after the contact phase, so
        # newly received encoders are evaluated in the same round) ----------
        self._adopt(k, need, zn)

        # ---------- coverage-aware cache refresh (Eq. 19) ----------
        for i in range(mfl.N):
            relay = [x for x, m in self.tickets[i].items()
                     if m > 0 and not self._is_own_pub(i, x)]
            if not relay:
                continue
            cap = cfg.cache_capacity_mb
            if self.flags["refresh"] == "lru" or not self.flags["use_future"]:
                ranked = sorted(relay, key=lambda x: -self.lru[i].get(x, -1))
            else:
                scored = []
                for x in relay:
                    v = self.versions[x]
                    if v.compat[i] and not v.resolved[i]:
                        scored.append((np.inf, x))   # pending own evaluation
                        continue
                    ex = nvec.get(x, np.zeros(Z)).copy()
                    ex[zn[i]] = max(ex[zn[i]] - 1, 0)
                    keepv = self._F(zn[i], pack(x), P, ex)
                    scored.append((keepv / v.S, x))
                # Eq. 19 tie-breaking toward eviction: a copy with no positive
                # continuation value is dropped even if it fits, freeing relay
                # space for future receptions
                ranked = [x for d, x in sorted(scored, reverse=True)
                          if d > 1e-9]
            used, keep = 0.0, set()
            for x in ranked:
                if used + self.versions[x].S <= cap + 1e-9:
                    keep.add(x)
                    used += self.versions[x].S
            for x in relay:
                if x not in keep:           # eviction burns the copy tickets
                    self.tickets[i].pop(x, None)
                    self.lru[i].pop(x, None)

        # ---------- reputation update (Sec. III-E) ----------
        # storage contribution (Eq. rep_storage): caching others' versions is
        # rewarded in proportion to the occupied relay storage
        for i in range(mfl.N):
            self._dPsi[i] += cfg.face_mu_s * self._cache_used(i)
        self.Psi = cfg.face_gamma_psi * self.Psi + self._dPsi   # Eq. rep_update
        self._dPsi = np.zeros(mfl.N)

        # ---------- publication (real backend; no-op with static encoders) ---
        if hasattr(self.mfl, "snapshot_encoder") and \
                cfg.face_Qpub > 0 and (k + 1) % cfg.face_Qpub == 0:
            self._publish_all(t=k + 1)

        # prune dead versions (no tickets anywhere, not a latest publication)
        alive = set(self.own_pub.values())
        for i in range(mfl.N):
            alive.update(x for x, m in self.tickets[i].items() if m > 0)
        for x in [x for x in self.versions if x not in alive]:
            self.versions.pop(x)
        # drop pending delivery credits of retired versions
        self._deliv_credit = {kx: cr for kx, cr in self._deliv_credit.items()
                              if kx[1] in self.versions}

        self.last_selected = [(i, j, self.versions[x].src, self.versions[x].r)
                              for (i, j, sel_g, _) in committed
                              for (x, _, _, _) in sel_g
                              if x in self.versions]
        return self.last_selected

    def _split_tickets(self, m, Fi, Fj):
        """Value-weighted ticket split g_ijx (Eq. ticket_split): tickets move
        in proportion to the receiver's share of continuation value; a
        replicating sender always retains at least one ticket."""
        if m <= 1:
            return max(m, 1)                # custody transfer
        if not self.flags["use_split"]:
            return 1
        frac = Fj / (Fi + Fj + 1e-12)
        return int(np.clip(np.ceil(m * frac), 1, m - 1))

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
