#!/bin/bash
#SBATCH -J rls_pndata
#SBATCH -p wang
#SBATCH -N 1
#SBATCH -c 36
#SBATCH --mem=120G
#SBATCH -t 1-00:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# Postnet data (spec section 4): 10000 training + 300 test episodes for one forcing, written to $RLS_DATA/postnet (read by the GPU jobs).
# Usage: sbatch --export=ALL,SET=strong scripts/hpc/fermi_postnet_data.sbatch
# Python environment (the authors used a conda env named torch_AE; adapt to your cluster):
# source /path/to/anaconda3/etc/profile.d/conda.sh
# conda activate torch_AE
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "${RLS_CODE:?set RLS_CODE to the release folder}"
D=$RLS_DATA/postnet/data
python -W ignore scripts/postnet_data.py --set $SET --split test --n 300 --workers 36 --out $D
python -W ignore scripts/postnet_data.py --set $SET --split train --n 10000 --workers 36 --out $D
echo "end $(date)"
