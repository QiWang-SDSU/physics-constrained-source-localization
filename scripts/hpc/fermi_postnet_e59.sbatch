#!/bin/bash
#SBATCH -J rls_pnE59
#SBATCH -p wang
#SBATCH -N 1
#SBATCH -c 36
#SBATCH --mem=120G
#SBATCH -t 1-00:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# E59: the E58 closed loop (100 tasks, the four planners of the paper, posteriors saved at n = 10-80) and posterior maps,
# with all learned posteriors trained on the planners' own paths: runs_real (B, B0) and runs_real_vel (Bvel, B0vel), seed 0.
# Usage: sbatch --export=ALL,SET=strong scripts/hpc/fermi_postnet_e59.sbatch
# Python environment (the authors used a conda env named torch_AE; adapt to your cluster):
# source /path/to/anaconda3/etc/profile.d/conda.sh
# conda activate torch_AE
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "${RLS_CODE:?set RLS_CODE to the release folder}"
python -W ignore scripts/postnet_closedloop.py --set $SET --workers 36 --runs runs_real --runs_vel runs_real_vel \
  --policies m1 m3 chase surgecast --save_q --q_ns 10 20 30 40 80 --out $RLS_OUT/E59
python -W ignore scripts/postnet_maps.py --set $SET --runs runs_real --runs_vel runs_real_vel --map_all --out $RLS_OUT/E59maps
echo "end $(date)"
