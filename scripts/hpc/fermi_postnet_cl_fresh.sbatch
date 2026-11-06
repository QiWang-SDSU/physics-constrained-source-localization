#!/bin/bash
#SBATCH -J rls_pnclF
#SBATCH -p wang
#SBATCH -N 1
#SBATCH -c 36
#SBATCH --mem=120G
#SBATCH -t 1-00:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# E57 closed loop on 300 fresh tasks (task seeds 1000-1299 on DNS seeds 6-8, never used before; offline test tasks are
# 0-299, the E26 closed-loop tasks 0-99), planners m1, m3, chase, surgecast.
#   STAGE=final : the 9500-episode networks of E37/E54 (O/ON), E55 (F/N) and E56 (R/RN), training seeds 0-2 -> E57fresh
#   STAGE=lopo  : the E57 leave-one-planner-out networks (7000 episodes, seed 0)                        -> E57lopo
# Usage: sbatch --export=ALL,SET=strong,STAGE=final scripts/hpc/fermi_postnet_cl_fresh.sbatch
# Python environment (the authors used a conda env named torch_AE; adapt to your cluster):
# source /path/to/anaconda3/etc/profile.d/conda.sh
# conda activate torch_AE
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "${RLS_CODE:?set RLS_CODE to the release folder}"
C="python -W ignore scripts/postnet_closedloop.py --set $SET --workers 36 --policies m1 m3 chase surgecast --n 300 --seed0 1000"
if [ "$STAGE" = final ]; then
  X="o0:b0:runs_fp32:full:0 o1:b0:runs_full_reps:full:1 o2:b0:runs_full_reps:full:2"
  for r in 0 1 2; do
    X="$X on$r:b0:runs_nomem:nomem:$r f$r:b0:runs_mix5:full:$r n$r:b0:runs_mix5_nomem:nomem:$r"
    X="$X r$r:b0:runs_real:full:$r rn$r:b0:runs_real_nomem:nomem:$r"
  done
  $C --runs runs_fp32 --extra $X --out $RLS_OUT/E57fresh
else
  X="c:b0:runs_lopo_all:full cn:b0:runs_lopo_all_nomem:nomem"
  for x in 0 5 3 4; do X="$X x$x:b0:runs_lopo_x$x:full xn$x:b0:runs_lopo_x${x}_nomem:nomem"; done
  $C --runs none --n_train 7000 --extra $X --out $RLS_OUT/E57lopo
fi
echo "end $(date)"
