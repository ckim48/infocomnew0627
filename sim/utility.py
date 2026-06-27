"""
Submodular utility model (Sec. IV-A, Eq. (11)-(18)).

Provides the per-round building blocks used by the greedy algorithm:
  alpha^need_{i,r}            modality learning need              (Eq. 11)
  beta^learn_{j->i,m,r}       learning contribution of a delivery (Eq. 12-13)
  beta^dis_{i->j,m,r}         road-aware forwarding contribution  (Eq. 14)
The coverage-form utilities F^learn (Eq. 15) and F^dis (Eq. 17) and the total
objective (Eq. 18 / weighted Eq. 23) are assembled incrementally in algorithm.py.
"""

import numpy as np


def modality_needs(cfg, mfl):
    """alpha^need_{i,r}(k): normalized inverse of (D_{i,r} * Q_{i,r}) over R_i (Eq. 11)."""
    eps = cfg.eps0
    need = {}
    for i in range(mfl.N):
        avail = mfl.avail[i]
        inv = {r: 1.0 / (mfl.D[(i, r)] * mfl.Q[(i, r)] + eps) for r in avail}
        s = sum(inv.values()) + eps
        for r in avail:
            need[(i, r)] = inv[r] / s
    return need


def mean_modality_data(mfl):
    """D_r: mean modality-r data size across owners (denominator in Eq. 12)."""
    Dr = {}
    for r in mfl.R:
        vals = [mfl.D[(i, r)] for i in range(mfl.N) if (i, r) in mfl.D]
        Dr[r] = float(np.mean(vals)) if vals else 1.0
    return Dr
