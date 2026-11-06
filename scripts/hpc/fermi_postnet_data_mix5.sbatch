#!/bin/bash
#SBATCH -J rls_pndat5
#SBATCH -p wang
#SBATCH -N 1
#SBATCH -c 36
#SBATCH --mem=120G
#SBATCH -t 1-00:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# E55 postnet data with five path types (physics infotaxis, zigzag, random walk, plume tracing, cast-and-surge):
# 10000 training + 300 test episodes for one forcing, same tasks as the three-type data, written to postnet/data_mix5.
# Usage: sbatch --export=ALL,SET=strong scripts/hpc/fermi_postnet_data_mix5.sbatch
# Python environment (the authors used a conda env named torch_AE; adapt to your cluster):
# source /path/to/anaconda3/etc/profile.d/conda.sh
# conda activate torch_AE
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "${RLS_CODE:?set RLS_CODE to the release folder}"
D=$RLS_DATA/postnet/data_mix5
python -W ignore scripts/postnet_data.py --set $SET --mix 5 --split test --n 300 --workers 36 --out $D
python -W ignore scripts/postnet_data.py --set $SET --mix 5 --split train --n 10000 --workers 36 --out $D
echo "end $(date)"
