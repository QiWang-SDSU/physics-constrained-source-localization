#!/bin/bash
#SBATCH -J rls_fit
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# Python environment (the authors used a conda env named torch_AE; adapt to your cluster):
# source /path/to/anaconda3/etc/profile.d/conda.sh
# conda activate torch_AE
cd "${RLS_CODE:?set RLS_CODE to the release folder}"
python scripts/checks/dns2_fit_and_law.py --set "$SET" --train 1 2 3 4 5 --test 6 7 8 --T "${T:-5.0}" \
    --out "$RLS_DATA/dns/${SET}_fit.json"
