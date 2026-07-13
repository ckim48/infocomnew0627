#!/bin/bash
# Full rerun with the REVISED system model / algorithm (paper v3):
#   * reputation + reciprocal priority (Sec. III-E, eq:rep_* / eq:rep_priority)
#   * greedy max-weight matching under the half-duplex single-peer
#     constraint (eq:matching_constraint, Prop. 2)
#   * joint admission-eviction bundle selection (eq:bundle_knapsack)
#   * ESV demand threshold delta_d (eq:esv_indicator)
#   * evaluation/adoption AFTER the contact phase (round steps S2 -> S3)
#
#   1. real FL main comparison (KITTI + nuScenes, 3 seeds, 6 schemes) [GPU]
#   2. component ablation sweeps (uniform + partitioned ECVs)         [CPU]
#   3. regenerate every paper asset (tables, seoul_pack figs, new figs)
#   4. commit (push is done by the attended session after verification)
set -u
cd /home/wnlab/ckim48_info
export OMP_NUM_THREADS=8
LOG=results/revised_assets.log
: > "$LOG"

echo "[driver3] $(date) launching ablation sweeps" >> "$LOG"
python3 -m sim.face_ablation 250 > results/face_abl_uniform.log 2>&1 &
ABL1=$!
python3 -m sim.face_ablation 250 part > results/face_abl_part.log 2>&1 &
ABL2=$!

echo "[driver3] $(date) launching real FL main (parallel datasets)" >> "$LOG"
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
echo "[driver3] $(date) real FL main done" >> "$LOG"

wait $ABL1 $ABL2
echo "[driver3] $(date) ablation sweeps done" >> "$LOG"

echo "[driver3] $(date) regenerating assets" >> "$LOG"
python3 -m sim.face_abl_table >> "$LOG" 2>&1
python3 -m sim.face_main_table >> "$LOG" 2>&1
python3 -m sim.face_figs >> "$LOG" 2>&1
python3 -m sim.make_tables >> "$LOG" 2>&1 || echo "[driver3] make_tables FAILED" >> "$LOG"
python3 seoul_pack/generate.py >> "$LOG" 2>&1 || echo "[driver3] seoul_pack FAILED" >> "$LOG"

echo "[driver3] $(date) committing" >> "$LOG"
git add sim/ scripts/ new_result/ Tables/ Figures/ seoul_pack/ \
    results/metrics_v2x_real_kitti.npz results/metrics_v2x_real_nuscenes.npz \
    results/metrics_face_ablation_v2x.npz results/metrics_face_ablation_v2x_part.npz \
    >> "$LOG" 2>&1
git commit -q -m "Revised system model rerun: reciprocity, matching, joint admission-eviction

- reputation + reciprocal priority (Sec. III-E): delivery credit
  u*v + mu_f(1-u)*v-hat, storage credit mu_s*S*c, decayed Psi state,
  zone-normalized priority factor in eq:transfer_advantage
- half-duplex single-peer matching (eq:matching_constraint): committed
  exchanges form a matching, greedy 1/2-approx (Prop. 2)
- joint admission-eviction bundle selection (eq:bundle_knapsack):
  h(s)/g(f) two-DP, lazy eviction on acknowledged arrival
- ESV demand threshold delta_d (eq:esv_indicator)
- evaluation/adoption moved AFTER the contact phase (S2 -> S3)
- new ablation variant: w/o reciprocity
- all tables/figures regenerated

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>" >> "$LOG" 2>&1

echo "[driver3] $(date) ALL DONE" >> "$LOG"
