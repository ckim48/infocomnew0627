#!/bin/bash
# FINAL v3.5 configuration (user decision 2026-07-17):
#   engine  = sim/face.py (v3.5: directed matching, evaluation-gated
#             single-best adoption, value-weighted copy tickets, joint
#             admission-eviction, coverage refresh, age/horizon staleness
#             corrections; reputation dormant since it left the paper)
#   scenario = per-modality availability draws (missing-modality vehicles
#             via modality_prob), no arch families / spec tiers
#   The v4 engine (2-mode transfers, bidirectional budgets, FedAvg-multi
#   aggregation) is preserved in sim/face_v4.py; 15 isolation probes
#   (results/probe_*.npz) showed it removes the ferrying advantage.
#
# Baselines are bit-identical to the 0d7781f v3 run (verified: none of the
# post-0d7781f edits touch baseline code paths), so they are restored from
# git and only FACE (recip-off) is rerun and merged.
set -u
cd /home/wnlab/ckim48_info
export OMP_NUM_THREADS=8
LOG=results/v35_assets.log
: > "$LOG"

echo "[driver5] $(date) restoring baseline npz from 0d7781f" >> "$LOG"
git show 0d7781f:results/metrics_v2x_real_kitti.npz > results/metrics_v2x_real_kitti.npz
git show 0d7781f:results/metrics_v2x_real_nuscenes.npz > results/metrics_v2x_real_nuscenes.npz

echo "[driver5] $(date) launching ablation sweeps (v3.5, K=16)" >> "$LOG"
python3 -m sim.face_ablation 250 > results/face_abl_uniform.log 2>&1 &
ABL1=$!
python3 -m sim.face_ablation 250 part > results/face_abl_part.log 2>&1 &
ABL2=$!

echo "[driver5] $(date) rerunning FACE (recip-off) on both datasets" >> "$LOG"
python3 -c "
from sim.run_v2x_real import run
run(seeds=[2026, 2027, 2028], dataset='kitti', rounds=250,
    schemes=['Proposed'], merge=True)
" > results/face_real_kitti.log 2>&1 &
M1=$!
python3 -c "
from sim.run_v2x_real import run
run(seeds=[2026, 2027, 2028], dataset='nuscenes', rounds=250,
    schemes=['Proposed'], merge=True)
" > results/face_real_nuscenes.log 2>&1 &
M2=$!
wait $M1 $M2
echo "[driver5] $(date) FACE reruns done; replotting" >> "$LOG"
python3 -c "
import numpy as np
from sim.config import Config
from sim.real_fl import REAL_SCHEMES
from sim.run_v2x_real import _plot
cfg = Config()
for ds in ['kitti', 'nuscenes']:
    d = np.load(f'results/metrics_v2x_real_{ds}.npz')
    results = {}
    for s in REAL_SCHEMES:
        keys = [k for k in d.files if k.startswith(s + '__')]
        if keys:
            results[s] = {k.split('__', 1)[1]: d[k] for k in keys}
    _plot(results, cfg, ds)
" >> "$LOG" 2>&1

wait $ABL1 $ABL2
echo "[driver5] $(date) ablations done; regenerating assets" >> "$LOG"
python3 -m sim.face_abl_table >> "$LOG" 2>&1
python3 -m sim.face_main_table >> "$LOG" 2>&1
python3 -m sim.face_figs >> "$LOG" 2>&1
python3 -m sim.make_tables >> "$LOG" 2>&1 || echo "[driver5] make_tables FAILED" >> "$LOG"
python3 seoul_pack/generate.py >> "$LOG" 2>&1 || echo "[driver5] seoul_pack FAILED" >> "$LOG"

echo "[driver5] $(date) committing" >> "$LOG"
git add sim/ scripts/ experiments/ new_result/ Tables/ Figures/ seoul_pack/ \
    results/metrics_v2x_real_kitti.npz results/metrics_v2x_real_nuscenes.npz \
    results/metrics_face_ablation_v2x.npz results/metrics_face_ablation_v2x_part.npz \
    >> "$LOG" 2>&1
git commit -q -m "FINAL: v3.5 engine + staleness corrections; v4 exchange model shelved

Decision backed by 15 isolation probes (results/probe_*.npz):
- the v4 exchange machinery (consume-only mode, bidirectional shared
  sojourn budgets, FedAvg-average aggregation) makes direct V2V
  diffusion sufficient on the real Seoul trace: FACE ties or trails
  in every environment (density, range, arch families, aggregation
  policy all probed). Engine preserved in sim/face_v4.py.
- typed sensor-suite mixtures REMOVE the demand FACE serves (camera-
  only vehicles have no LiDAR demand; camera supply is ubiquitous),
  so missing-modality vehicles are realized via per-modality
  availability draws instead.
- production engine sim/face.py = v3 protocol (directed matching,
  evaluation-gated single-best adoption, value-weighted copy tickets,
  joint admission-eviction) + age-discounted gain prior + horizon
  staleness correction; reputation mechanism dormant (left the paper).
- FACE rerun (recip-off) merged over the 0d7781f baselines (baseline
  code paths untouched since); ablations rerun at K=16.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>" >> "$LOG" 2>&1

echo "[driver5] $(date) ALL DONE" >> "$LOG"
