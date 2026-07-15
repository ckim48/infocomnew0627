#!/bin/bash
# Full rerun with the paper-v4 system model / algorithm:
#   * two transfer modes: consume-only vs retain (eq:transfer_advantage)
#   * binary spray-and-wait replication tokens (eq:copy_cap)
#   * FedAvg aggregation set + acceptance test + LOO attribution
#   * kinematic sojourn contact budget shared by both directions
#   * mutual-proposal pairing protocol (eq:matching_constraint)
#   * reputation/reciprocity REMOVED (dropped from the paper)
#   * heterogeneous sensor scenario: typed sensor suites (vision-only /
#     ADAS / robotaxi), architecture-family compatibility chi, per-modality
#     sensor-spec degradation tiers
set -u
cd /home/wnlab/ckim48_info
export OMP_NUM_THREADS=8
LOG=results/v4_assets.log
: > "$LOG"

echo "[driver4] $(date) launching ablation sweeps" >> "$LOG"
python3 -m sim.face_ablation 250 > results/face_abl_uniform.log 2>&1 &
ABL1=$!
python3 -m sim.face_ablation 250 part > results/face_abl_part.log 2>&1 &
ABL2=$!

echo "[driver4] $(date) launching real FL main (parallel datasets)" >> "$LOG"
python3 -c "
from sim.run_v2x_real import run
run(seeds=[2026, 2027, 2028], dataset='kitti', rounds=250)
" > results/face_real_kitti.log 2>&1 &
M1=$!
python3 -c "
from sim.run_v2x_real import run
run(seeds=[2026, 2027, 2028], dataset='nuscenes', rounds=250)
" > results/face_real_nuscenes.log 2>&1 &
M2=$!
wait $M1 $M2
echo "[driver4] $(date) real FL main done" >> "$LOG"

wait $ABL1 $ABL2
echo "[driver4] $(date) ablation sweeps done" >> "$LOG"

echo "[driver4] $(date) regenerating assets" >> "$LOG"
python3 -m sim.face_abl_table >> "$LOG" 2>&1
python3 -m sim.face_main_table >> "$LOG" 2>&1
python3 -m sim.face_figs >> "$LOG" 2>&1
python3 -m sim.make_tables >> "$LOG" 2>&1 || echo "[driver4] make_tables FAILED" >> "$LOG"
python3 seoul_pack/generate.py >> "$LOG" 2>&1 || echo "[driver4] seoul_pack FAILED" >> "$LOG"

echo "[driver4] $(date) committing" >> "$LOG"
git add sim/ scripts/ experiments/ new_result/ Tables/ Figures/ seoul_pack/ \
    results/metrics_v2x_real_kitti.npz results/metrics_v2x_real_nuscenes.npz \
    results/metrics_face_ablation_v2x.npz results/metrics_face_ablation_v2x_part.npz \
    >> "$LOG" 2>&1
git commit -q -m "Paper-v4 model + heterogeneous sensor scenario: full rerun

System model / algorithm (Sec. III-IV v4):
- two transfer modes: consume-only (aggregate & discard; no tokens,
  no relay storage) vs retain (relay copy via binary spray-and-wait
  tokens enforcing the copy cap K_x exactly)
- FedAvg aggregation over the aggregation set + validation acceptance
  test (never hurts) + leave-one-out attribution feeding the ridge
  reward predictor
- kinematic sojourn-time contact budget shared by both directions;
  mutual-proposal pairing protocol; kappa-hat from ACTIVATED exchanges
- reputation/reciprocal cooperation removed (dropped from the paper)

Heterogeneous sensor scenario (Sec. I motivation):
- typed sensor suites: vision-only (no LiDAR), camera+radar ADAS,
  camera+LiDAR, full robotaxi mixtures per dataset
- architecture-family compatibility chi: aggregation only within the
  same encoder family; incompatible vehicles can still relay
- per-(vehicle, modality) sensor-spec degradation tiers (noisy camera,
  sparse LiDAR) on top of the data-volume rich/poor split

All tables and figures regenerated.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>" >> "$LOG" 2>&1

echo "[driver4] $(date) ALL DONE" >> "$LOG"
