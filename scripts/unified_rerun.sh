#!/bin/bash
# Full main-comparison rerun under the UNIFIED system-model protocol
# (all schemes = FACE-engine policy variants), then regenerate assets.
set -u
cd /home/wnlab/ckim48_info
export OMP_NUM_THREADS=6
LOG=results/unified_assets.log
: > "$LOG"

echo "[driver2] $(date) launching kitti + nuscenes in parallel" >> "$LOG"
python3 -c "
from sim.run_v2x_real import run
run(seeds=[2026, 2027, 2028], dataset='kitti', rounds=250)
" > results/unified_kitti.log 2>&1 &
P1=$!
python3 -c "
from sim.run_v2x_real import run
run(seeds=[2026, 2027, 2028], dataset='nuscenes', rounds=250)
" > results/unified_nuscenes.log 2>&1 &
P2=$!
wait $P1 $P2
echo "[driver2] $(date) main runs done" >> "$LOG"

python3 -m sim.face_abl_table >> "$LOG" 2>&1
python3 -m sim.face_main_table >> "$LOG" 2>&1
python3 -m sim.face_figs >> "$LOG" 2>&1
python3 -m sim.make_tables >> "$LOG" 2>&1 || echo "[driver2] make_tables FAILED" >> "$LOG"
python3 seoul_pack/generate.py >> "$LOG" 2>&1 || echo "[driver2] seoul_pack FAILED" >> "$LOG"

git add sim/ scripts/ new_result/ Tables/ Figures/ seoul_pack/ \
    results/metrics_v2x_real_kitti.npz results/metrics_v2x_real_nuscenes.npz \
    >> "$LOG" 2>&1
git commit -q -m "Unified protocol comparison: all schemes as policy variants of the system model

Baselines ported into the FACE engine (versions, copy tickets,
evaluation-gated adoption identical for every scheme; only the
forwarding policy differs, per Sec. III). Operating point
Q_pub=1, TTL=15, lambda=0.001. All tables/figures regenerated.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" >> "$LOG" 2>&1

echo "[driver2] $(date) ALL DONE" >> "$LOG"
