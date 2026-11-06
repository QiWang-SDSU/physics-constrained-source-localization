#!/bin/bash
#SBATCH -J rls_sens
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
##SBATCH -x gpu001,gpu002,gpu003   # authors' cluster: excluded K80 nodes
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# E54: sensitivity of the learned posterior to the tokens entering the GRU (scripts/checks/gru_sensitivity.py),
# models B0 / B, without and with velocity sensing, 9500 training episodes, both forcings.
# Usage: sbatch scripts/hpc/fermi_gru_sens.sbatch
# Python environment (the authors used a conda env named torch_AE; adapt to your cluster):
# source /path/to/anaconda3/etc/profile.d/conda.sh
# conda activate torch_AE
cd "${RLS_CODE:?set RLS_CODE to the release folder}"
O=$RLS_DATA/postnet/sens
for s in strong weak; do for m in b0 b; do for v in "" "--vel"; do
  python -W ignore scripts/checks/gru_sensitivity.py --set $s --model $m $v --out $O
done; done; done
echo "end $(date)"
