#!/bin/bash
# Main comparison under the Sec. II motivation scenario: encoder-carrier
# vehicles confined to the west half (partitioned ECVs). Full 250-round runs,
# then regenerate the paper assets.
set -u
cd /home/wnlab/ckim48_info
export OMP_NUM_THREADS=6
LOG=results/partitioned_assets.log
: > "$LOG"

echo "[part] $(date) launching kitti + nuscenes (partitioned)" >> "$LOG"
python3 -c "
from sim.run_v2x_real import run
run(seeds=[2026, 2027, 2028], dataset='kitti', rounds=250, partitioned=True)
" > results/part_kitti.log 2>&1 &
P1=$!
python3 -c "
from sim.run_v2x_real import run
run(seeds=[2026, 2027, 2028], dataset='nuscenes', rounds=250, partitioned=True)
" > results/part_nuscenes.log 2>&1 &
P2=$!
wait $P1 $P2
echo "[part] $(date) main runs done" >> "$LOG"

python3 -m sim.face_main_table >> "$LOG" 2>&1
python3 -m sim.face_figs >> "$LOG" 2>&1

git add sim/ scripts/ new_result/ \
    results/metrics_v2x_real_kitti.npz results/metrics_v2x_real_nuscenes.npz \
    >> "$LOG" 2>&1
git commit -q -m "Main comparison under the Sec. II partitioned-ECV scenario

Data-rich encoder carriers confined to the west half of the region;
strong encoders must be ferried east to reach demand. All schemes share
the unified system-model protocol. Table I + figures regenerated." \
    >> "$LOG" 2>&1

echo "[part] $(date) ALL DONE" >> "$LOG"
