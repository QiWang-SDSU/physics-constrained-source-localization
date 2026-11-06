#!/bin/bash
# Bit-for-bit replay of DNS2 strong runs, saving full-resolution fields over the windows of the
# visualised search episodes (viz_dns_strong.npz seeds 3, 7, 12). See scripts/dns2_replay.py.
#   sbatch scripts/hpc/fermi_dns2_replay.sbatch
#SBATCH -J rls_replay
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 03:00:00
#SBATCH -a 6-7
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

# Python environment (the authors used a conda env named torch_AE; adapt to your cluster):
# source /path/to/anaconda3/etc/profile.d/conda.sh
# conda activate torch_AE
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "${RLS_CODE:?set RLS_CODE to the release folder}"
# episode seed -> (dataset seed, start record t0), from search_sweep.dns_factory; window = t0 : t0 + 83
case "$SLURM_ARRAY_TASK_ID" in
  6) WIN="2029:2112 287:370" ;;   # episode seeds 3 and 12
  7) WIN="4797:4880" ;;           # episode seed 7
esac
echo "host $(hostname) dataset seed $SLURM_ARRAY_TASK_ID windows $WIN start $(date)"
python scripts/dns2_replay.py --seed "$SLURM_ARRAY_TASK_ID" --set strong --eps 0.005 --windows $WIN \
    --out $RLS_DATA/dns/replay --workers 2
echo "end $(date)"
