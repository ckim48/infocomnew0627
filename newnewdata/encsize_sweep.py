"""Encoder-size sensitivity sweep (reviewer-facing robustness).

The paper anchors per-modality encoder parameter sizes at 12/18/6 MB
(camera MobileNetV2, LiDAR PointPillars, lightweight radar encoder).
This sweep re-runs the Table-I protocol (main Seoul evening trace, N=180,
T=250 replay, 6 schemes x 3 seeds) under two additional size profiles:

  compact (0.5x):  6 /  9 / 3 MB  -- pruned/quantized encoder variants
  heavy   (2.0x): 24 / 36 / 12 MB -- higher-capacity encoder types

Cache capacity (45 MB) and V2V budgets stay fixed, so the heavy profile
stresses the contact/cache constraints (a 36 MB LiDAR encoder almost fills
a cache), while compact relaxes them.

Run:  python3 newnewdata/encsize_sweep.py {heavy|compact} [kitti|nuscenes]
Out:  results/metrics_v2x_real_<ds>_enc<profile>.npz
"""
import os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

PROFILE = sys.argv[1]
DS = sys.argv[2] if len(sys.argv) > 2 else "kitti"
SCALE = {"heavy": 2.0, "compact": 0.5}[PROFILE]
OUT = f"metrics_v2x_real_{DS}_enc{PROFILE}.npz"

if os.path.exists(os.path.join("results", OUT)):
    print(f"[encsize] {OUT} exists, skipping")
    sys.exit(0)

from sim.config import Config
import sim.run_v2x_real as R

cfg = Config()
cfg.encoder_size = {k: v * SCALE for k, v in cfg.encoder_size.items()}
print(f"[encsize] profile={PROFILE} ds={DS} sizes={cfg.encoder_size}")
t0 = time.time()
R.run(cfg=cfg, seeds=[2026, 2027, 2028], dataset=DS, rounds=250,
      num_vehicles=180, merge=True, out_name=OUT)
print(f"[encsize] {PROFILE}/{DS} done in {(time.time() - t0) / 60:.0f} min")
