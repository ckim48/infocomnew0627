"""
Empirical interaction-error probe (Sec. IV bound constant eps_int).

In the deployed protocol the aggregation set is effectively a sequential
singleton (half-duplex single-peer matching + evaluation-gated adoption),
so |A|>=2 sets never arise online.  We therefore estimate eps_int OFFLINE:
while FACE trains on the real KITTI backend, every SAMPLE_EVERY rounds we
draw random (receiver, modality) pairs, form candidate sets A of 2-3
snapshots of other vehicles' current encoders, and measure

    eps_int(A) = | sum_{x in A} v({x}) - v(A) |

with v(.) the realized normalized validation gain (RealMFL.gain_single /
gain_joint).  Output: results/face_eint_probe.npz
(round, nset, solo_sum, joint).

Run:  python3 -m sim.face_eint_probe
"""

import numpy as np
import torch

from .config import Config
from .run_v2x_real import _prepare_v2x
from .real_fl import RealMFL, _prep_data, _device
from .face import FACE
from .simulator import make_modality_availability, make_arch_assignment

ROUNDS = 250
SEED = 2026
SAMPLE_EVERY = 10          # sampling stage interval (training progresses)
RECEIVERS = 20             # receivers sampled per stage
SET_SIZES = (2, 3)


def _sample_eint(mfl, rng, k, out):
    """Offline candidate-set sampling at round k."""
    donors = {}
    for r in mfl.cfg.modalities:
        donors[r] = [m for m in range(mfl.N)
                     if r in mfl.avail[m] and mfl.rich.get(m, False)]
    pairs = [(i, r) for i in range(mfl.N) for r in mfl.avail[i]
             if len([m for m in donors[r] if m != i]) >= max(SET_SIZES)]
    if not pairs:
        return
    for (i, r) in [pairs[j] for j in
                   rng.choice(len(pairs), min(RECEIVERS, len(pairs)),
                              replace=False)]:
        ks = int(rng.choice(SET_SIZES))
        srcs = rng.choice([m for m in donors[r] if m != i], ks,
                          replace=False)
        cands = [(int(m), mfl.snapshot_encoder(int(m), r)) for m in srcs]
        solo = sum(mfl.gain_single(i, r, m, sd)[0] for (m, sd) in cands)
        joint = mfl.gain_joint(i, r, cands)
        out.append((k, ks, float(solo), float(joint)))


def main():
    cfg = Config()
    cfg.num_vehicles = 180
    cfg.face_ttl = 15
    cfg.face_K_tickets = 6
    cfg.face_Qpub = 1
    cfg.face_lam = 0.001
    cfg.modalities = ["camera", "lidar"]
    cfg.modality_prob = {"camera": 1.0, "lidar": 0.85}
    device = _device()
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    road, mob, gammas = _prepare_v2x(cfg, device)
    data = _prep_data(cfg, cfg.seed, dataset="kitti", min_class_count=0)

    avail = make_modality_availability(cfg, np.random.default_rng(SEED + 7))
    arch = make_arch_assignment(cfg, np.random.default_rng(SEED + 11), avail)
    torch.manual_seed(SEED)
    mfl = RealMFL(cfg, np.random.default_rng(SEED), avail, data,
                  device=device)
    mfl.arch = arch
    alg = FACE(cfg, mfl, mob, "Proposed", seed=SEED)
    srng = np.random.default_rng(SEED + 99)
    events = []
    for k in range(ROUNDS):
        kk = k % mob.Krounds
        mob.k = kk
        mfl.local_train()
        mfl.refresh_strengths()
        g = gammas[kk] if alg.flags.get("use_dis") \
            or alg.flags.get("cache_policy") == "psi" else np.zeros(mob.N)
        alg.run_round(k, g, gamma_eval=gammas[kk])
        if (k + 1) % SAMPLE_EVERY == 0:
            _sample_eint(mfl, srng, k, events)
            print(f"  round {k+1}/{ROUNDS}  eint samples {len(events)}",
                  flush=True)
    ev = np.array(events, dtype=np.float64)
    if ev.size == 0:
        ev = np.zeros((0, 4))
    np.savez("results/face_eint_probe.npz",
             round=ev[:, 0], nset=ev[:, 1], solo_sum=ev[:, 2],
             joint=ev[:, 3])
    if len(ev):
        e = np.abs(ev[:, 2] - ev[:, 3])
        print(f"  saved results/face_eint_probe.npz  n={len(ev)}  "
              f"eps_int mean {e.mean():.4f}  p95 {np.percentile(e, 95):.4f}")


if __name__ == "__main__":
    main()
