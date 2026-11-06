#!/bin/bash
#SBATCH -J rls_pnvelR
#SBATCH -p wang
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 8:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# E59: add the local velocity at the sensor to the policy-path postnet shards (data_real, E56), both forcings.
# Writes <shard>_vel.npz next to each shard; the shards themselves are not changed.
# Usage: sbatch scripts/hpc/fermi_postnet_vel_real.sbatch
# Python environment (the authors used a conda env named torch_AE; adapt to your cluster):
# source /path/to/anaconda3/etc/profile.d/conda.sh
# conda activate torch_AE
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "${RLS_CODE:?set RLS_CODE to the release folder}"
for s in strong weak; do
  python -W ignore scripts/postnet_add_velocity.py --set $s --data $RLS_DATA/postnet/data_real &
done
wait
echo "end $(date)"
