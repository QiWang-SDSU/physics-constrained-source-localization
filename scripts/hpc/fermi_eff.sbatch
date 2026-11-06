#!/bin/bash
#SBATCH -J rls_eff
#SBATCH -p wang
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# E31: physics vs learned observation model vs number of readings. Usage:
#   sbatch --export=ALL,SET=strong scripts/hpc/fermi_eff.sbatch
# Python environment (the authors used a conda env named torch_AE; adapt to your cluster):
# source /path/to/anaconda3/etc/profile.d/conda.sh
# conda activate torch_AE
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "${RLS_CODE:?set RLS_CODE to the release folder}"
python -W ignore scripts/checks/dns2_data_efficiency.py --set $SET --threads 8 --out out/dns2fit/nn/${SET}_eff.json
