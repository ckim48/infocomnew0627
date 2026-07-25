# newnewdata — INFOCOM review-risk experiments (2026-07-24)

Queue driver: `overnight_driver.py` (nohup; log: `overnight.log`).
All runs use the real backend (`sim/run_v2x_real.py`), KITTI, 250 rounds,
`merge=True` so `results/` and the paper figures are untouched.

## Review point 1 — single trace / single parameter values
- `metrics_v2x_real_kitti_night.npz` — **second mobility window**: Fri
  late-night Seoul V2X trace (collected 23:36–23:51 KST, ~510 veh/snapshot
  citywide vs ~1040 in the evening-peak window → natural density/contact-
  sparsity contrast). Full 6-scheme comparison, 3 seeds.
  Raw trace: `data/gangnam/seoul_v2x_trace_fri_night.npz`,
  snapped cache: `v2x_seoul_trace_night.npz`.
- `metrics_sweep_*.npz` — FACE-only real-backend sensitivity (1 seed each,
  original window): `noise05/noise10` (Gamma + 0.5/1.0·std Gaussian noise →
  prediction-error robustness), `range100/range200` (comm range),
  `zone150/zone600` (zone cell), `rate6/rate24` (link rate), `n90/n135`
  (density; wall-clock per config in the log → runtime-vs-N datapoint;
  signaling = `txmb`/`tx` keys in each npz).
- Already existed elsewhere: abstract-backend N/Lambda/H sweeps
  (`results/face_sens_probe.npz`, 3 seeds), real-backend K_x sweep
  (`results/face_kx_probe.npz`, `results/kx_sweep.log`), 8-seed run
  (`new_more_seed/metrics_v2x_real_kitti_8seed.npz`).

## Review point 2 — baselines
- Offline hindsight oracle ALREADY EXISTS: `sim/face_oracle.py` →
  `results/face_oracle.npz` (16 windows, ILP): FACE/oracle 0.525 mean /
  0.600 median, no-cache-oracle/oracle 0.963. Figure panel ready in
  `sim/face_figs.py` (~line 221). Fix = include in Sec. Evaluation.
- NEW baselines implemented in `sim/face.py` (same protocol, new scheme
  flags): `PRoPHET`, `SprayWait` (binary spray of K_x tickets),
  `Caching-LFU`, `Caching-rand`, `NoComm` (lower bound), `FullContact`
  (unconstrained-communication upper bound). CPU-smoke-tested; the full
  3-seed×250-round KITTI comparison runs via `newbaselines_driver.py`
  (waits for phase 1; log `newbase.log`) →
  `metrics_v2x_real_kitti_newbase.npz` — same seeds as Table I.
  ("Learning" baseline already = demand-aware direct forwarding.)
- Third mobility window scheduled: cron fires `monday_morning.py` on
  Mon 2026-07-27 08:30 KST (morning peak) →
  `metrics_v2x_real_kitti_morning.npz`. Remove the crontab line after.

## Paper drafts
- `draft_snippets.tex` — 4 drop-in LaTeX blocks: reproducibility addendum
  (NOTE: fixes the wrong "80–600 objects" claim → ~83 rich / 4–11 poor on
  KITTI), oracle-comparison paragraph, surrogate-fidelity reframing with
  measured eps values, baseline-fairness + new-benchmark descriptions.

## Review point 3 — theory
- `theory_stats.txt` — measured surrogate errors from existing runs:
  eps_pred from Table-I calibration pairs
  (`results/metrics_v2x_real_kitti_events.npz`), eps_int from
  `results/face_eint_probe.npz` (n=500), oracle ratios.

## Review point 4 — ML reproducibility
No new runs needed: document splits/partition/skew/fusion/seeds/±std from
`sim/real_fl.py`, `sim/kitti_dataset.py`, `sim/config.py` in the settings
subsection; state the classifier-not-detection scope explicitly.
