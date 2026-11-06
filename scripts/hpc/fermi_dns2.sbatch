#!/bin/bash
# DNS2 dataset on Fermi (partition cpu). One array task = one seed. It checkpoints every 25 t.u.
# and resumes on resubmission.
#   benchmark : sbatch --array=1 --time=00:30:00 --export=ALL,SET=bench,EPS=0.005,TSPIN=5,TREC=10 scripts/hpc/fermi_dns2.sbatch
#   strong    : sbatch --export=ALL,SET=strong,EPS=0.005 scripts/hpc/fermi_dns2.sbatch
#   weak      : sbatch --export=ALL,SET=weak,EPS=0.0012 scripts/hpc/fermi_dns2.sbatch
#SBATCH -J rls_dns2
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 12:00:00
#SBATCH -a 1-8
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

# Python environment (the authors used a conda env named torch_AE; adapt to your cluster):
# source /path/to/anaconda3/etc/profile.d/conda.sh
# conda activate torch_AE
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "${RLS_CODE:?set RLS_CODE to the release folder}"
echo "host $(hostname) seed $SLURM_ARRAY_TASK_ID set=$SET eps=$EPS start $(date)"
python scripts/dns2_run.py --seed "$SLURM_ARRAY_TASK_ID" --out "$RLS_DATA/dns/$SET" \
    --eps "$EPS" --t_spin "${TSPIN:-100}" --t_rec "${TREC:-500}" --workers "$SLURM_CPUS_PER_TASK"
echo "end $(date)"
