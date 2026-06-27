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
pip install eclipse-sumo libsumo sumolib traci numpy scipy matplotlib networkx torch
python -m sim.simulator          # runs 3 seeds, writes results/metrics.npz
python -m sim.plotting           # writes Figures/fig_*.{png,pdf}
```

The first run invokes SUMO and caches the trace to
`results/intas_trace_N{N}_K{K}.npz`; later runs reuse it.

## Outputs (`Figures/`)

- `fig_accuracy` — mean model accuracy vs. round
- `fig_loss` — mean validation loss vs. round
- `fig_poor_accuracy` — accuracy of poor-data vehicles (heterogeneous demand)
- `fig_forwarding` — cumulative successful encoder deliveries
- `fig_queue` — average virtual-queue backlog (Lyapunov stability)
- `fig_final_acc_bar` — final accuracy bar chart

Key parameters are in `sim/config.py`.
