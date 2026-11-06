#!/bin/bash
#SBATCH -J rls_pntrain
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
##SBATCH -x gpu001,gpu002,gpu003   # authors' cluster: excluded K80 nodes
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# B0 / B learning curves for one forcing on one GPU (spec section 5.1). Sizes in episodes (81 readings each);
# 3 random subsets for <= 1000 episodes; K80 nodes excluded (too old); GPU partition limit 12 h, the loop skips finished runs, so resubmit to continue. Usage: sbatch --export=ALL,SET=strong scripts/hpc/fermi_postnet_train.sbatch
# Python environment (the authors used a conda env named torch_AE; adapt to your cluster):
# source /path/to/anaconda3/etc/profile.d/conda.sh
# conda activate torch_AE
cd "${RLS_CODE:?set RLS_CODE to the release folder}"
P=$RLS_DATA/postnet
RUNS=${RUNS:-runs}   # E47: RUNS=runs_fp32 for the rerun with fp32 accumulation (rlscalar/search/postnet.py)
# EXTRA: extra postnet_train.py flags (E48: EXTRA=--vel with RUNS=runs_vel)
mkdir -p $P/$RUNS
for n in 9500 3000 1000 300 100 30 10; do
  reps="0"; [ $n -le 1000 ] && reps="0 1 2"
  for r in $reps; do for m in b b0; do
    [ -f $P/$RUNS/${SET}_${m}_n${n}_r${r}.json ] && continue
    python -W ignore scripts/postnet_train.py --set $SET --model $m --n_train $n --rep $r --data $P/data --out $P/$RUNS $EXTRA
  done; done
done
echo "end $(date)"
