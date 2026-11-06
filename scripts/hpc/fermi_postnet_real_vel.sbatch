#!/bin/bash
#SBATCH -J rls_pnrealV
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
##SBATCH -x gpu001,gpu002,gpu003   # authors' cluster: excluded K80 nodes
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# E59: B and B0 with velocity sensing trained on the paths of the four closed-loop planners (data_real), 9500 episodes,
# three seeds -> runs_real_vel.
# Usage: sbatch --export=ALL,SET=strong scripts/hpc/fermi_postnet_real_vel.sbatch
# Python environment (the authors used a conda env named torch_AE; adapt to your cluster):
# source /path/to/anaconda3/etc/profile.d/conda.sh
# conda activate torch_AE
cd "${RLS_CODE:?set RLS_CODE to the release folder}"
P=$RLS_DATA/postnet
O=$P/runs_real_vel
mkdir -p $O
for r in 0 1 2; do for m in b b0; do
  [ -f $O/${SET}_${m}_n9500_r$r.json ] && continue
  python -W ignore scripts/postnet_train.py --set $SET --model $m --vel --n_train 9500 --rep $r --data $P/data_real --out $O
done; done
echo "end $(date)"
