"""
Multimodal federated learning model (Sec. III-C, Eq. (1)-(2)).

Dataset-free realisation of the learning dynamics the encoder exchange optimises,
in the probabilistic-coverage form assumed by the paper's utilities (Eq. 13-17):

  * Each modality-r encoder owned by vehicle m has a *strength* s_{m,r} in (0,1)
    set by the owner's local sensing quality Q_{m,r} and data size D_{m,r}
    (good, plentiful data -> strong encoder; occluded / scarce data -> weak).
  * A vehicle's achieved modality-r model quality after aggregating a set A of
    encoders (its own + received, Eq. 2) is dominated by the single strongest
    encoder adopted:
        Q^eff_{i,r} = max_{m in A} s_{m,r}.
    One strong encoder saturates the quality; many weak encoders do not
    substitute for a strong one (a better modality encoder is plugged in).
  * Validation loss   L^val_{i,r} = (1 - Q^eff_{i,r})^2.

Consequence (the paper's thesis): a vehicle with poor local modality-r data only
improves by *receiving a strong modality-r encoder*. Strong encoders are owned by
few vehicles, so they must be propagated -- via store-carry-forward and future
contacts -- to the needy vehicles that are not in direct contact with the owners.
"""

import numpy as np


class MultimodalFL:
    def __init__(self, cfg, rng, modality_avail):
        self.cfg = cfg
        self.rng = rng
        self.R = cfg.modalities
        self.N = cfg.num_vehicles

        self.avail = modality_avail
        self.D = {}          # (i,r) -> D_{i,r}
        self.Q = {}          # (i,r) -> Q_{i,r}^loc sensing quality
        self.strength = {}   # (m,r) -> encoder strength s_{m,r}
        self.theta = {}      # (i,r) -> own encoder strength (forwardable handle)
        self.acquired = {}   # (i,r) -> set of owner ids aggregated into i's model
        self.pairs = []

        Dmax = cfg.data_max
        # bimodal sensing quality: a minority of vehicles hold strong (clean-data)
        # encoders per modality; the majority hold weak (occluded/scarce) encoders.
        for i in range(self.N):
            for r in self.avail[i]:
                D = int(rng.integers(cfg.data_min, cfg.data_max))
                if rng.random() < cfg.frac_good:
                    q = rng.uniform(0.80, 1.00)        # good: clear conditions
                else:
                    q = rng.uniform(0.10, 0.35)        # poor: occluded / congested
                self.D[(i, r)] = D
                self.Q[(i, r)] = q
                # strength rises with quality and (sub-linearly) with data size
                s = q * (0.70 + 0.30 * (D / Dmax))
                s = float(np.clip(s, 0.05, 0.97))
                self.strength[(i, r)] = s
                self.theta[(i, r)] = s
                self.acquired[(i, r)] = {i}
                self.pairs.append((i, r))

    # ---- encoder strength accessor (cache stores these handles) ----
    def enc_strength(self, m, r):
        return self.strength[(m, r)]

    def Dmr(self, m, r):
        return self.D.get((m, r), 1)

    # ---- local training: own encoder converges to the local-data quality ----
    def local_train(self):
        # A vehicle's own encoder strength is bounded by its local sensing quality
        # and data: poor-data vehicles cannot lift their own encoder by training
        # more, so improvement must come from received (stronger) encoders.
        # Own strengths are therefore fixed at their data-determined level.
        return

    # ---- achieved quality / validation loss ----
    def q_eff(self, i, r, extra=None):
        # achieved quality = best encoder adopted (own or received); a vehicle's
        # fused model is dominated by the strongest modality encoder available to
        # it, so quantity of weak encoders does not substitute for a strong one.
        best = max(self.strength[(m, r)] for m in self.acquired[(i, r)])
        if extra is not None:
            for (m, s_m) in extra:
                best = max(best, s_m)
        return best

    def val_loss(self, i, r, q_eff=None):
        qe = self.q_eff(i, r) if q_eff is None else q_eff
        return float((1.0 - qe) ** 2)

    def local_val_loss(self, i, r):
        return self.val_loss(i, r)

    # ---- loss-reduction gain G^loss when i adds encoder m (Eq. loss_reduction_gain) ----
    def gain_single(self, i, r, m, s_m):
        before_q = self.q_eff(i, r)
        before = (1.0 - before_q) ** 2
        after_q = self.q_eff(i, r, extra=[(m, s_m)])
        after = (1.0 - after_q) ** 2
        g = (before - after) / (before + self.cfg.eps0)
        return max(g, 0.0), before, after

    # ---- commit received encoders: aggregate into the vehicle's model (Eq. 2) ----
    def commit(self, i, r, received):
        for (m, s_m) in received:
            self.acquired[(i, r)].add(m)

    # ---- global metrics ----
    def mean_val_loss(self):
        return float(np.mean([self.local_val_loss(i, r) for (i, r) in self.pairs]))

    def mean_accuracy(self):
        return float(np.mean([self.q_eff(i, r) for (i, r) in self.pairs]))

    def tail_accuracy(self, q=0.1):
        qs = np.array([self.q_eff(i, r) for (i, r) in self.pairs])
        thr = np.quantile(qs, q)
        return float(qs[qs <= thr].mean())

    def poor_accuracy(self, thr=0.5):
        """Mean achieved quality among poor-local-data vehicle-modality pairs."""
        qs = [self.q_eff(i, r) for (i, r) in self.pairs if self.strength[(i, r)] < thr]
        return float(np.mean(qs)) if qs else 0.0
