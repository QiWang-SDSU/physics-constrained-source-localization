#!/bin/bash
#SBATCH -J rls_pnbn
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
##SBATCH -x gpu001,gpu002,gpu003   # authors' cluster: excluded K80 nodes
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# E54: B0 (9500 training episodes) with the token entering the GRU through a rank-r linear map: learned (MODE=learned)
# or fixed to the leading sensitivity directions (MODE=fixed, projections $P/sens/{set}_b0_proj_r{r}.npy).
# Usage: sbatch --export=ALL,MODE=learned scripts/hpc/fermi_postnet_bottleneck.sbatch
# Python environment (the authors used a conda env named torch_AE; adapt to your cluster):
# source /path/to/anaconda3/etc/profile.d/conda.sh
# conda activate torch_AE
cd "${RLS_CODE:?set RLS_CODE to the release folder}"
P=$RLS_DATA/postnet
for s in strong weak; do for r in ${RS:-1 2 3}; do
  O=$P/runs_bn_${MODE}_r$r; mkdir -p $O
  [ -f $O/${s}_b0_n9500_r0.json ] && continue
  if [ "$MODE" = fixed ]; then X="--proj $P/sens/${s}_b0_proj_r$r.npy"; else X="--bottleneck $r"; fi
  python -W ignore scripts/postnet_train.py --set $s --model b0 --n_train 9500 --rep 0 $X --data $P/data --out $O
done; done
echo "end $(date)"
