#!/bin/bash
#SBATCH -J rls_pncl
#SBATCH -p wang
#SBATCH -N 1
#SBATCH -c 36
#SBATCH --mem=120G
#SBATCH -t 1-00:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# E38 closed loop with the learned posterior B (scripts/postnet_closedloop.py), one forcing per job.
# Usage: sbatch --export=ALL,SET=strong scripts/hpc/fermi_postnet_cl.sbatch
# E40 (posterior maps): sbatch --export=ALL,SET=strong,OUT=$RLS_OUT/E40,EXTRA=--save_q scripts/hpc/fermi_postnet_cl.sbatch
# Python environment (the authors used a conda env named torch_AE; adapt to your cluster):
# source /path/to/anaconda3/etc/profile.d/conda.sh
# conda activate torch_AE
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "${RLS_CODE:?set RLS_CODE to the release folder}"
python -W ignore scripts/postnet_closedloop.py --set $SET --workers 36 --out ${OUT:-$RLS_OUT} $EXTRA
echo "end $(date)"
