#!/bin/bash
#SBATCH -J rls_pnmem
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
##SBATCH -x gpu001,gpu002,gpu003   # authors' cluster: excluded K80 nodes
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# E54 memory tests, B0 with 9500 training episodes, three training seeds each:
#   runs_nomem      GRU state fixed at zero (no memory, no step clock), seeds 0-2
#   runs_bn_fixed_r0  no measurement enters the GRU (step clock only), seeds 1-2 (seed 0 from fermi_postnet_bottleneck)
#   runs_full_reps  the unchanged network, seeds 1-2 (seed 0 = runs_fp32)
# Usage: sbatch --export=ALL,SET=strong scripts/hpc/fermi_postnet_mem.sbatch
# Python environment (the authors used a conda env named torch_AE; adapt to your cluster):
# source /path/to/anaconda3/etc/profile.d/conda.sh
# conda activate torch_AE
cd "${RLS_CODE:?set RLS_CODE to the release folder}"
P=$RLS_DATA/postnet
mkdir -p $P/runs_nomem $P/runs_bn_fixed_r0 $P/runs_full_reps
run() {   # out rep extra...
  local O=$1 r=$2; shift 2
  [ -f $O/${SET}_b0_n9500_r$r.json ] && return
  python -W ignore scripts/postnet_train.py --set $SET --model b0 --n_train 9500 --rep $r --data $P/data --out $O "$@"
}
for r in 0 1 2; do run $P/runs_nomem $r --nomem; done
for r in 1 2; do run $P/runs_bn_fixed_r0 $r --proj $P/sens/${SET}_b0_proj_r0.npy; done
for r in 1 2; do run $P/runs_full_reps $r; done
echo "end $(date)"
