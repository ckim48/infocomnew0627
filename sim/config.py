"""
Global configuration for the MDFL encoder caching/forwarding simulation.

All symbols follow the notation in the paper:
  - vehicles  i in I
  - rounds    k in K
  - modalities r in R
  - road segments e in V
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ----- reproducibility -----
    seed: int = 2026

    # ----- federated learning rounds -----
    K: int = 150                      # number of global rounds

    # ----- vehicles -----
    num_vehicles: int = 150           # |I|
    comm_range: float = 150.0         # V2V communication range (m)

    # ----- road network (synthetic urban grid, Ingolstadt-scaled) -----
    grid_rows: int = 14
    grid_cols: int = 14
    block_len_min: float = 80.0       # min road segment length (m)
    block_len_max: float = 200.0      # max road segment length (m)

    # ----- modalities R = {camera, lidar, radar, gps} -----
    modalities: List[str] = field(default_factory=lambda: ["camera", "lidar", "radar", "gps"])
    # probability a given vehicle is equipped with a modality (modality heterogeneity)
    modality_prob: dict = field(default_factory=lambda: {
        "camera": 0.95, "lidar": 0.55, "radar": 0.70, "gps": 1.0})
    # storage size S_{m,r} per modality encoder (MB), used for cache capacity C2
    encoder_size: dict = field(default_factory=lambda: {
        "camera": 12.0, "lidar": 18.0, "radar": 6.0, "gps": 1.5})

    # ----- multimodal learning model -----
    enc_dim: int = 16                 # dimension d of an encoder parameter vector theta_{i,r}
    local_lr: float = 0.30            # local SGD step rate toward the local optimum mu_{i,r}
    local_epochs: int = 1             # local updates per round
    data_min: int = 80                # min |D_{i,r}|
    data_max: int = 600               # max |D_{i,r}|
    # local data sensing quality Q_{i,r}^loc in (0,1]; low quality -> weak encoder
    quality_min: float = 0.20
    quality_max: float = 1.00
    frac_good: float = 0.15           # fraction of vehicles with strong (clean-data) encoders

    # ----- caching -----
    cache_capacity_mb: float = 45.0   # C_i^cache (MB) per vehicle

    # ----- algorithm: Lyapunov + submodular -----
    V: float = 2.0                    # Lyapunov V (drift-plus-penalty weight)
    nu: float = 1.0                   # weight nu on forwarding (dissemination) utility
    lam: float = 0.02                 # lambda: cache-cost weight in normalized marginal gain
    # lambda_tx: optional per-transmission cost R(a) - lam_tx*|a| (modular ->
    # submodularity preserved). Default 0: pilots showed pricing tx inside the
    # greedy either prunes forwarding-valuable volume (learn-only selection) or
    # diverts budget from learning (use_dis=True, acc -5pp); the published
    # operating point already wins per-transmission utility U/Tx.
    lam_tx: float = 0.0
    eps0: float = 1e-6                # epsilon_0

    # ----- contact budget (C1) -----
    contact_time_per_round: float = 1.6   # \bar T^con_{i,j} budget (s) per contact
    tx_rate_mbps: float = 12.0            # effective V2V link rate (MB/s) for tx time

    # ----- FACE (new system model): versions, tickets, zones, first-contact value -----
    face_zone_cell: float = 300.0     # road-zone grid cell size (m)
    face_H: int = 6                   # future-contact valuation horizon H (rounds)
    face_beta: float = 0.85           # discount beta in the first-contact value
    face_lam: float = 0.004           # lambda: communication price per MB in P1
    face_delta: float = 0.005         # adoption threshold delta_i on the validation gain
    face_K_tickets: int = 16          # K_x: max distributed copies per encoder version
                                      # (abstract backend; under the matching constraint
                                      # K=16 recovers K=inf accuracy at bounded replication,
                                      # see results/kx_sweep.log. The real backend overrides
                                      # to K=6 in run_v2x_real: stale-version spread there
                                      # makes tighter replication strictly better.)
    face_ttl: int = 1_000_000         # version lifetime t_exp - t_gen (rounds)
    face_Qpub: int = 10               # publication period Q_pub (real backend)
    face_alpha_g: float = 0.6         # optimism bonus alpha_g in the ridge gain predictor
    face_ridge_lam: float = 1.0       # ridge regularization
    face_ridge_decay: float = 0.98    # sliding-window decay of the ridge sufficient stats
    face_g_floor: float = 0.2         # exploration floor on the immediate gain term
    face_gain_prior: bool = True      # causal (s_meta - s_own)/(1 - s_own) prior on gains
    face_decay: float = 0.97          # exponential decay of zone transition/contact counts
    face_alpha_P: float = 0.3         # adjacency smoothing alpha_P in P-hat (Eq. 11)
    face_alpha_C: float = 1.0         # Beta prior alpha_C in kappa-hat (Eq. 12)
    face_beta_C: float = 3.0          # Beta prior beta_C in kappa-hat (Eq. 12)
    face_mu: float = 0.85             # forecast blend mu_q = mu_v (Eq. 13)
    face_Nev: int = 6                 # per-round candidate evaluation budget N_ev
    face_delta_d: float = 0.05        # requester demand threshold delta_mu (Sec. III-C)
    # ----- heterogeneous sensor scenario (Sec. I / II) -----
    vehicle_types: object = None      # typed sensor-suite mixture [(weight, mods), ...]
    use_arch_families: bool = True    # architecture-family compatibility chi
    arch_high_frac: float = 0.5       # fraction of high-compute (large-family) vehicles
    spec_low_prob: float = 0.3        # P(low-spec sensor) per (vehicle, modality)

    # ----- hierarchical GAT mobility prediction -----
    gat_hidden: int = 32
    gat_heads: int = 4
    gat_epochs: int = 120
    gat_lr: float = 5e-3
    H_max: int = 4                    # maximum number of future transitions
    gamma_disc: float = 0.8           # discount factor 0<gamma<1

    # ----- output -----
    results_dir: str = "results"
    figures_dir: str = "Figures"


# baseline scheme identifiers
SCHEMES = ["Proposed", "Caching-assisted", "V2V-aware", "Learning-aware"]
