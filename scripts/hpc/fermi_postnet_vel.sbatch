#!/bin/bash
#SBATCH -J rls_pnvel
#SBATCH -p cpu
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 2:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# E48: add the local velocity at the sensor to the postnet data shards (scripts/postnet_add_velocity.py), both forcings.
# Usage: sbatch scripts/hpc/fermi_postnet_vel.sbatch
# Python environment (the authors used a conda env named torch_AE; adapt to your cluster):
# source /path/to/anaconda3/etc/profile.d/conda.sh
# conda activate torch_AE
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "${RLS_CODE:?set RLS_CODE to the release folder}"
for s in strong weak; do
  python -W ignore scripts/postnet_add_velocity.py --set $s --data $RLS_DATA/postnet/data
done
echo "end $(date)"
