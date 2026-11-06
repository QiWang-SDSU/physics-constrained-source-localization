#!/bin/bash
#SBATCH -J rls_pnreal
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
##SBATCH -x gpu001,gpu002,gpu003   # authors' cluster: excluded K80 nodes
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# E56: B0 trained on the paths of the four closed-loop planners (data_real), 9500 episodes, three training seeds, with and without memory.
#   runs_real        the unchanged network
#   runs_real_nomem  GRU state fixed at zero
# Usage: sbatch --export=ALL,SET=strong scripts/hpc/fermi_postnet_real.sbatch
# Python environment (the authors used a conda env named torch_AE; adapt to your cluster):
# source /path/to/anaconda3/etc/profile.d/conda.sh
# conda activate torch_AE
cd "${RLS_CODE:?set RLS_CODE to the release folder}"
P=$RLS_DATA/postnet
mkdir -p $P/runs_real $P/runs_real_nomem
run() {   # out rep extra...
  local O=$1 r=$2; shift 2
  [ -f $O/${SET}_b0_n9500_r$r.json ] && return
  python -W ignore scripts/postnet_train.py --set $SET --model b0 --n_train 9500 --rep $r --data $P/data_real --out $O "$@"
}
for r in 0 1 2; do run $P/runs_real $r; run $P/runs_real_nomem $r --nomem; done
echo "end $(date)"
