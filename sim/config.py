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
