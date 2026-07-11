#!/bin/bash
# Overnight full rerun with the revised FACE system model:
#   1. real FL main comparison (KITTI + nuScenes, 3 seeds, 6 schemes) [GPU]
#   2. component ablation sweeps (uniform + partitioned ECVs)         [CPU]
#   3. regenerate every paper asset (tables, seoul_pack figs, new figs)
#   4. commit + push
set -u
cd /home/wnlab/ckim48_info
export OMP_NUM_THREADS=8
LOG=results/overnight_assets.log
: > "$LOG"

echo "[driver] $(date) launching ablation sweeps" >> "$LOG"
python3 -m sim.face_ablation 250 > results/face_abl_uniform.log 2>&1 &
ABL1=$!
python3 -m sim.face_ablation 250 part > results/face_abl_part.log 2>&1 &
ABL2=$!

echo "[driver] $(date) launching real FL main (parallel datasets)" >> "$LOG"
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
echo "[driver] $(date) real FL main done" >> "$LOG"

wait $ABL1 $ABL2
echo "[driver] $(date) ablation sweeps done" >> "$LOG"

echo "[driver] $(date) regenerating assets" >> "$LOG"
python3 -m sim.face_abl_table >> "$LOG" 2>&1
python3 -m sim.face_main_table >> "$LOG" 2>&1
python3 -m sim.face_figs >> "$LOG" 2>&1
python3 -m sim.make_tables >> "$LOG" 2>&1 || echo "[driver] make_tables FAILED" >> "$LOG"
python3 seoul_pack/generate.py >> "$LOG" 2>&1 || echo "[driver] seoul_pack FAILED" >> "$LOG"

echo "[driver] $(date) committing" >> "$LOG"
git add sim/ scripts/ new_result/ Tables/ Figures/ seoul_pack/ \
    results/metrics_v2x_real_kitti.npz results/metrics_v2x_real_nuscenes.npz \
    results/metrics_face_ablation_v2x.npz results/metrics_face_ablation_v2x_part.npz \
    >> "$LOG" 2>&1
git commit -q -m "Overnight rerun: revised FACE model — all tables/figures regenerated

- revised mechanisms: posterior-mean forwarding + exploration floor,
  value-weighted ticket splitting (eq:ticket_split), absorbing-chain
  first-contact recursion (eq:first_contact_survival), round-frozen
  matching weights, N_ev evaluation budget
- real backend: sample-size aggregation weight n_i=0 for frozen encoders
- new assets: 3-seed ablation table, ablation bars, beyond-encounter,
  traffic-Pareto and tx-rate figures in new_result/

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" >> "$LOG" 2>&1

# push is done by the attended session after result verification
echo "[driver] $(date) ALL DONE" >> "$LOG"
