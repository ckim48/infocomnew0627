# Encoder Forwarding and Caching for Multimodal Federated Learning in Vehicular Networks — Simulation

Reference implementation of the simulation in the paper. It evaluates the
proposed future-contact-based encoder caching/forwarding framework for
multimodal decentralized FL (MDFL) against three baselines on the **real
Ingolstadt (InTAS) SUMO traffic scenario** (S. Lobo et al., *InTAS – The
Ingolstadt Traffic Scenario for SUMO*, arXiv:2011.11995).

## What it does

1. **Real mobility (`sim/intas_trace.py`).** Loads the genuine InTAS road
   network (`ingolstadt.net.xml`: 3,342 nodes / 7,968 edges / 98 traffic lights)
   and runs the SUMO microsimulation over a dense-hour demand file
   (`InTAS_008.rou.xml`, ~09:30–10:30) with `libsumo`. A cohort of `N` vehicles
   that persist through the window is extracted as the per-round vehicle states
   and the dynamic V2V graph `G^com(k)`.
2. **Hierarchical road–vehicle graph (`sim/mobility.py`).** Directed
   road-segment graph `G^road=(V,A^road)` with lengths `L_e`, turn topology
   `O(e)`, and turn-direction labels; per-round traffic-state features
   `z_e(k)=[L_e, ρ_e, v̄, flow]` from all running vehicles.
3. **Hierarchical GAT mobility predictor (`sim/hgat.py`).** Sparse two-stage
   graph attention (road → vehicle, Eq. 5–6), a transition head (Eq. 8),
   multi-step reachability, and the future-contact score `Γ_j^road(k)` (Eq. 10),
   trained self-supervised on realized InTAS turns.
4. **Multimodal FL (`sim/mfl.py`).** Coverage/best-adoption learning model:
   encoders have a strength set by sensing quality and data size; a vehicle's
   achieved modality quality is dominated by the strongest encoder it has
   adopted. Poor-data vehicles can only improve by *receiving* a strong encoder.
5. **Online caching/forwarding (`sim/algorithm.py`, `sim/utility.py`).**
   Queue-weighted submodular maximization (problem P, Eq. 19) solved by the
   marginal-gain greedy (Eq. 24–27, (1−1/e) guarantee), Lyapunov virtual-queue
   update (Eq. 20), and the Ψ-based cache update (Eq. 28–30).

### Schemes (Sec. V-A)

| Scheme | link-aware | store-carry-forward | demand-aware | future Γ |
|---|---|---|---|---|
| **Proposed** | ✓ | ✓ | ✓ | ✓ |
| Caching-assisted (LRU) | ✓ | ✓ | ✗ | ✗ |
| V2V-aware | ✓ | ✗ | ✓ | ✗ |
| Learning-aware | ✗ | ✗ | ✓ | ✗ |

## Run

```bash
pip install eclipse-sumo libsumo sumolib traci numpy scipy matplotlib networkx torch pillow

# (1) abstract large-scale simulation (150 vehicles) + figures
python -m sim.simulator          # 3 seeds -> results/metrics.npz
python -m sim.plotting           # Figures/fig_*.{png,pdf}

# (2) road-awareness demonstration (mobility prediction: road vs straight-line)
python -m sim.mobility_prediction    # Figures/fig_mobpred_*.{png,pdf}

# (3) map visualizations on the real Ingolstadt network
python -m sim.map_viz                # Figures/fig_map_*.{png,pdf}

# (4) REAL multimodal FL on KITTI (camera + LiDAR)
python -m sim.kitti_dataset          # build the multimodal object dataset
python -m sim.real_fl                # 3 seeds -> results/metrics_real.npz + Figures/fig_real_*
```

The first run invokes SUMO and caches the trace to
`results/intas_trace_N{N}_K{K}.npz`; later runs reuse it.

### Real multimodal FL on KITTI (`sim/real_fl.py`, `sim/kitti_dataset.py`, `sim/multimodal_model.py`)

Real vehicular multimodal data (KITTI object detection benchmark): each labeled
object becomes a multimodal sample of a **camera** RGB patch + the **LiDAR**
points inside its 3D box, classified into Car/Pedestrian/Cyclist. `RealMFL`
holds real CNN/PointNet encoders + a local fusion head per vehicle and plugs
into the *same* caching/forwarding algorithm, performing **real FedAvg of
encoder weights (Eq. 2), real local SGD, and reporting real classification
accuracy**. Data-poor vehicles (few samples) freeze their encoder and adapt only
their local head, so they depend on receiving a strong encoder from others.
Download is automatic from the public KITTI S3 mirror (~40 GB; not committed).

### Road-awareness demonstration (`sim/mobility_prediction.py`)

Predicts each vehicle's future position H seconds ahead two ways — constrained
to the road network (using the GAT turn probabilities) vs straight-line
constant-velocity extrapolation — and compares displacement error against the
realized SUMO trajectory. For turning vehicles the road-aware prediction is
13–16% more accurate at every horizon.

## Outputs (`Figures/`)

- `fig_accuracy` — mean model accuracy vs. round
- `fig_loss` — mean validation loss vs. round
- `fig_poor_accuracy` — accuracy of poor-data vehicles (heterogeneous demand)
- `fig_forwarding` — cumulative successful encoder deliveries
- `fig_queue` — average virtual-queue backlog (Lyapunov stability)
- `fig_final_acc_bar` — final accuracy bar chart

Key parameters are in `sim/config.py`.
