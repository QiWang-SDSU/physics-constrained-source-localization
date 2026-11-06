#!/bin/bash
#SBATCH -J rls_pnmix5
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
##SBATCH -x gpu001,gpu002,gpu003   # authors' cluster: excluded K80 nodes
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# E55: B0 trained on the five-path-type data (data_mix5), 9500 episodes, three training seeds, with and without memory.
#   runs_mix5        the unchanged network
#   runs_mix5_nomem  GRU state fixed at zero
# Usage: sbatch --export=ALL,SET=strong scripts/hpc/fermi_postnet_mix5.sbatch
# Python environment (the authors used a conda env named torch_AE; adapt to your cluster):
# source /path/to/anaconda3/etc/profile.d/conda.sh
# conda activate torch_AE
cd "${RLS_CODE:?set RLS_CODE to the release folder}"
P=$RLS_DATA/postnet
mkdir -p $P/runs_mix5 $P/runs_mix5_nomem
run() {   # out rep extra...
  local O=$1 r=$2; shift 2
  [ -f $O/${SET}_b0_n9500_r$r.json ] && return
  python -W ignore scripts/postnet_train.py --set $SET --model b0 --n_train 9500 --rep $r --data $P/data_mix5 --out $O "$@"
}
for r in 0 1 2; do run $P/runs_mix5 $r; run $P/runs_mix5_nomem $r --nomem; done
echo "end $(date)"
