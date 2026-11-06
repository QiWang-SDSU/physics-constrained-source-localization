#!/bin/bash
#SBATCH -J rls_pnlopo
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
##SBATCH -x gpu001,gpu002,gpu003   # authors' cluster: excluded K80 nodes
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# E57 leave one planner out: B0 with and without memory on the planner-path data (data_real, E56), 7000 training
# episodes, training seed 0, with one path type removed from training and validation (policy codes: 0 physics
# infotaxis, 5 corrected infotaxis, 3 plume tracing, 4 cast-and-surge) or none removed (control, same 7000 episodes).
# Output runs_lopo_x{code} / runs_lopo_all, and the same with _nomem.
# Usage: sbatch --export=ALL,SET=strong scripts/hpc/fermi_postnet_lopo.sbatch
# Python environment (the authors used a conda env named torch_AE; adapt to your cluster):
# source /path/to/anaconda3/etc/profile.d/conda.sh
# conda activate torch_AE
cd "${RLS_CODE:?set RLS_CODE to the release folder}"
P=$RLS_DATA/postnet
run() {   # out extra...
  local O=$1; shift
  mkdir -p $O
  [ -f $O/${SET}_b0_n7000_r0.json ] && return
  python -W ignore scripts/postnet_train.py --set $SET --model b0 --n_train 7000 --rep 0 --data $P/data_real --out $O "$@"
}
for x in all 5 3 4 0; do
  if [ $x = all ]; then O=$P/runs_lopo_all; E=""; else O=$P/runs_lopo_x$x; E="--exclude_policy $x"; fi
  run $O $E
  run ${O}_nomem $E --nomem
done
echo "end $(date)"
