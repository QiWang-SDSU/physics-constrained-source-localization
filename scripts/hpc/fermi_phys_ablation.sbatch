#!/bin/bash
#SBATCH -J rls_abl
#SBATCH -p wang
#SBATCH -N 1
#SBATCH -c 36
#SBATCH --mem=120G
#SBATCH -t 0-06:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# E50 physics-posterior collapse ablation (scripts/checks/phys_ablation_replay.py), one forcing per job.
# Usage: sbatch --export=ALL,SET=strong scripts/hpc/fermi_phys_ablation.sbatch   (test: EXTRA="--n 2 --workers 6")
# Python environment (the authors used a conda env named torch_AE; adapt to your cluster):
# source /path/to/anaconda3/etc/profile.d/conda.sh
# conda activate torch_AE
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "${RLS_CODE:?set RLS_CODE to the release folder}"
python -W ignore scripts/checks/phys_ablation_replay.py --set $SET --out ${OUT:-$RLS_OUT/E50} $EXTRA
echo "end $(date)"
