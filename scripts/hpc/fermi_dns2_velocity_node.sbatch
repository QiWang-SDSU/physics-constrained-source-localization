#!/bin/bash
#SBATCH -J rls_dnsvel
#SBATCH -p cpu
#SBATCH -C "Gold6240|Platinum8268"
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# E44: all 8 seeds of one forcing on one node (8 x 2 workers), so the two forcings use at most two nodes.
# Usage: sbatch --export=ALL,SET=strong scripts/hpc/fermi_dns2_velocity_node.sbatch
# Restarts of the original runs (original DNS logs rls_dns2_95280_*.out): weak seeds 3-6 at record 500, 7-8 at 1000.
# Python environment (the authors used a conda env named torch_AE; adapt to your cluster):
# source /path/to/anaconda3/etc/profile.d/conda.sh
# conda activate torch_AE
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "${RLS_CODE:?set RLS_CODE to the release folder}"
echo "host $(hostname) set $SET start $(date)"
OUT=$RLS_DATA/dns/${OUT:-vel}
mkdir -p $OUT/logs
for s in 1 2 3 4 5 6 7 8; do
  R=""
  [ $SET = weak ] && [ $s -ge 3 ] && [ $s -le 6 ] && R="--resume 500"
  [ $SET = weak ] && [ $s -ge 7 ] && R="--resume 1000"
  python scripts/dns2_velocity.py --seed $s --set $SET --out $OUT --workers 2 $R $EXTRA \
    > $OUT/logs/${SET}_seed0${s}.log 2>&1 &
done
wait
echo "end $(date)"
