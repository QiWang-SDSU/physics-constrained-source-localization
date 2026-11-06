# Physics-based and learned observation models for mobile source localization in turbulent plumes

Code for the paper *Physics-Based and Learned Observation Models for Mobile Source Localization in Turbulent Plumes*
(Qi Wang and Jun Chen, San Diego State University; L4DC 2027 submission).

A mobile sensor moves through a turbulent scalar plume and has to locate the source from a short history of
concentration measurements (and, optionally, the local velocity). The truth is a two-dimensional forced-turbulence DNS
with four passive scalar plumes, at two forcing levels. The paper compares two source posteriors on a 91 x 61 grid of
candidate cells: a physics posterior, built from an adjoint (Taylor-dispersion) mean field and an intermittent
likelihood whose measurement weights come from a fitted correlation law, and a learned posterior, a GRU over the
measurement tokens with a pairwise decoder, with or without the physics posterior as an input. Four motion policies
choose the paths: infotaxis on the physics model, infotaxis on a corrected observation model (a small MLP on top of the
physics mean field), plume tracing and cast-and-surge. The scripts here produce the DNS, the fits, the episodes, the
trained networks, the closed-loop runs, the posterior maps and every figure of the paper. The experiment numbers
(E37-E59) refer to the authors' internal log and are kept in file names and comments.

## Install

Python 3.11. The compile and import checks of this release were run with numpy 2.4, scipy 1.17, matplotlib 3.10 and
torch 2.11; the HPC runs of the paper used a conda environment with the same packages (versions not recorded here).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The figure scripts render all text with LaTeX (`text.usetex=True` in `scripts/figures/plotstyle.py`), so a LaTeX
installation with `dvipng` and Ghostscript is needed for the figures. Training uses a GPU if one is available
(`torch.cuda`), and the rest runs on CPU.

## Data

Download the data from Zenodo ([10.5281/zenodo.23991624](https://doi.org/10.5281/zenodo.23991624); its `README.md` gives the unpacking commands) and arrange them as follows (details and file list in
[`DATA_MANIFEST.md`](DATA_MANIFEST.md)):

| Zenodo folder | Put it at | Needed for |
|---|---|---|
| `figure_data/` | `docs/research/data/` (inside this folder) | figures only |
| `networks/` and `episodes/` | both merged into `$RLS_DATA/postnet/` | closed loop, maps, checks |
| `dns/` | `$RLS_DATA/dns/` | fits, episodes, closed loop |

To redraw the figures, only `figure_data/` is needed: unpack it into `docs/research/data/` and run the commands of
stage 8 from the top of this folder.

### Paths

The scripts read two environment variables. Command-line flags (`--root`, `--fitdir`, `--pn`, `--data`, `--vel`,
`--vel_dir`, `--out`) override them.

| Variable | Default | Holds |
|---|---|---|
| `RLS_DATA` | `data` | `dns/` (DNS runs `dns/{strong,weak}/seedXX.npz`, `dns/{set}_fit.json`, `dns/fits/`, `dns/vel/`, `dns/replay/`) and `postnet/` (episode folders `data`, `data_real`, ... and network folders `runs_*`) |
| `RLS_OUT` | `out` | outputs of closed-loop runs and checks |
| `RLS_CODE` | (none) | this folder; used only by the sbatch scripts (`cd $RLS_CODE`) |

Forcing levels are selected with `--set strong` (eps = 5e-3) or `--set weak` (eps = 1.2e-3). Every stage below is run
for both.

## Pipeline

Commands are given for the strong forcing and are run from the top of this folder. The sbatch scripts in
`scripts/hpc/` wrap the same commands for a Slurm cluster (see "HPC notes"). Outputs of stages 1-7 are written to
`$RLS_DATA` or `$RLS_OUT`; the files that the figures read were then copied by hand into `docs/research/data/` under the
names listed in `DATA_MANIFEST.md` (the copy step is given with each stage).

1. **DNS** (data of all experiments; appendix on the DNS). Eight runs per forcing, 100 time units of spin-up and 500
   recorded; each run checkpoints every 25 time units and resumes from the checkpoint when resubmitted.
   ```bash
   python scripts/dns2_run.py --seed 1 --out $RLS_DATA/dns/strong --eps 0.005 --t_spin 100 --t_rec 500 --workers 2   # seeds 1-8
   python scripts/dns2_run.py --seed 1 --out $RLS_DATA/dns/weak   --eps 0.0012 --t_spin 100 --t_rec 500 --workers 2
   ```
   (`scripts/hpc/fermi_dns2.sbatch`, array 1-8.) Runs 1-5 are used for fitting and training, runs 6-8 for testing.
   Two DNS replays reproduce stored runs bit for bit (same seed, parameters, worker count and CPU type):
   - full-resolution window for Fig. 2 (`schematic_likelihood_v7`):
     `python scripts/dns2_replay.py --seed 6 --set strong --eps 0.005 --windows 287:370 --out $RLS_DATA/dns/replay`
     (`fermi_dns2_replay.sbatch`); copy `strong_seed06_k287.npz` to `docs/research/data/viz/replay/`.
   - velocity on the sensor sub-grid (E44, velocity sensing):
     `python scripts/dns2_velocity.py --seed 1 --set strong --out $RLS_DATA/dns/vel --workers 2`, with
     `--resume 500` for weak seeds 3-6 and `--resume 1000` for weak seeds 7-8, where the original runs were restarted
     from a checkpoint (`fermi_dns2_velocity_node.sbatch`). The replay is exact only on the CPU family of the original
     runs (Intel Cascade Lake, Gold6240 / Platinum8268); the script stores the worst mismatch of the replayed scalar
     (`sub_maxdiff`), which must be 0.

2. **Physics fits** (physics posterior M1; appendix on the physics posterior, table of fitted parameters).
   ```bash
   python scripts/checks/dns2_fit_and_law.py --set strong --train 1 2 3 4 5 --test 6 7 8 --T 5.0 --out $RLS_DATA/dns/strong_fit.json
   python scripts/checks/dns2_taylor_fit.py --set strong --out $RLS_DATA/dns/fits/strong_fit_taylor.json
   python -c "import json; f=json.load(open('$RLS_DATA/dns/fits/strong_fit_taylor.json')); f['surrogate']['src_r']=0.1; json.dump(f, open('$RLS_DATA/dns/fits/strong_fit_taylor_src.json','w'), indent=1)"
   python scripts/checks/dns2_likelihood_calibration.py --set strong --fit $RLS_DATA/dns/fits/strong_fit_taylor_src.json --out $RLS_DATA/dns/fits/strong_lik_taylor_src.json
   ```
   The first script gives the time-mean plume moments and the correlation scales (tau, L) with the out-of-sample test
   of the correlation law (`fermi_fit.sbatch`); the second the Taylor mean field (K, T_L, s0^2); the third line adds
   the DNS source radius 0.1 (finite-source adjoint, E26; the published `_taylor_src.json` differs from `_taylor.json`
   only in this entry); the fourth fits the likelihood coefficients (key `zi2`).

3. **Corrected observation model** (M3, used by corrected infotaxis; appendix on the corrected model).
   ```bash
   python scripts/checks/dns2_data_efficiency.py --set strong --map --fixmean --sampling traj --out $RLS_DATA/dns/fits/nn/strong_eff_traj.json
   ```
   Fits the MLP correction on readings along zigzag and random-walk paths for N = 1e2 ... 1e6 readings; the paper uses
   the N = 1e6 files `$RLS_DATA/dns/fits/nn/strong_N1000000_map_fix_traj_{fit,lik}.json` and `_hyb.npz` (E36).

4. **Episodes** (training and test histories with the physics log-posterior; appendix on the learned posterior).
   ```bash
   # paths of the four motion policies (E56): used by every learned posterior in the main text
   python scripts/postnet_data.py --set strong --paths eig eig3 chase surgecast --split test  --n 300   --out $RLS_DATA/postnet/data_real
   python scripts/postnet_data.py --set strong --paths eig eig3 chase surgecast --split train --n 10000 --out $RLS_DATA/postnet/data_real
   # generic paths (E37): training-size sweep, sensitivity, memory study
   python scripts/postnet_data.py --set strong --split test  --n 300   --out $RLS_DATA/postnet/data
   python scripts/postnet_data.py --set strong --split train --n 10000 --out $RLS_DATA/postnet/data
   # five path types (E55): only for the E55 networks scored in stage 6
   python scripts/postnet_data.py --set strong --mix 5 --split train --n 10000 --out $RLS_DATA/postnet/data_mix5   # and --split test --n 300
   # velocity at the sensor for every shard (E48 generic, E59 policy paths)
   python scripts/postnet_add_velocity.py --set strong --data $RLS_DATA/postnet/data_real
   python scripts/postnet_add_velocity.py --set strong --data $RLS_DATA/postnet/data
   ```
   (`fermi_postnet_data_real.sbatch`, `fermi_postnet_data.sbatch`, `fermi_postnet_data_mix5.sbatch`,
   `fermi_postnet_vel_real.sbatch`, `fermi_postnet_vel.sbatch`.)

5. **Training the learned posteriors** (`b` = with physics input, `b0` = without; `--vel` adds velocity sensing).
   One run, about 7 min on one GPU for 9500 episodes:
   ```bash
   python scripts/postnet_train.py --set strong --model b --n_train 9500 --rep 0 --data $RLS_DATA/postnet/data_real --out $RLS_DATA/postnet/runs_real
   ```
   Each run writes `{set}_{model}_n{n_train}_r{rep}.pt` and a JSON with the test metrics.

   | Network folder | Command / sbatch | Experiment | Used in |
   |---|---|---|---|
   | `runs_real` (b, b0, seeds 0-2), `runs_real_nomem` | `fermi_postnet_real_b.sbatch`, `fermi_postnet_real.sbatch` | E58, E56 | Figs. 3-7, offline and closed-loop tables |
   | `runs_real_vel` (b, b0 with `--vel`, seeds 0-2) | `fermi_postnet_real_vel.sbatch` | E59 | velocity figures, tables |
   | `runs_fp32` (sweep 10 ... 9500 episodes) | `fermi_postnet_train.sbatch` with `RUNS=runs_fp32` | E47 | Fig. learning curve, Fig. memory (generic), E52, E54 |
   | `runs_vel` (sweep with `--vel`) | `fermi_postnet_train.sbatch` with `RUNS=runs_vel EXTRA=--vel` | E48 | Fig. learning curve, E54 |
   | `runs_nomem`, `runs_full_reps`, `runs_bn_fixed_r0` | `fermi_postnet_mem.sbatch` | E54 | Fig. memory |
   | `runs_bn_learned_r{1,2,3}`, `runs_bn_fixed_r*` | `fermi_postnet_bottleneck.sbatch` (`MODE=learned` / `fixed`) | E54 | Fig. GRU sensitivity (c) |
   | `runs_mix5`, `runs_mix5_nomem` | `fermi_postnet_mix5.sbatch` | E55 | scored in stage 6 (E57), not plotted |
   | `runs_lopo_*` (7000 episodes, one policy held out) | `fermi_postnet_lopo.sbatch` | E57 | Fig. memory |

   Copy the JSONs of `runs_fp32` and `runs_vel` to `docs/research/data/E47/` and `E48/`, and
   `runs_fp32/{set}_b0_n9500_r0.json` to `docs/research/data/E54/{set}_b0_full.json`.

6. **Closed loop** (four policies x all posteriors, 100 held-out tasks; Fig. 5, closed-loop table and appendix figures).
   ```bash
   python scripts/postnet_closedloop.py --set strong --runs runs_real --runs_vel runs_real_vel \
       --policies m1 m3 chase surgecast --save_q --q_ns 10 20 30 40 80 --out $RLS_OUT/E59
   ```
   (`fermi_postnet_e59.sbatch`, 36 cores, a few minutes.) Policy codes: `m1` physics infotaxis, `m3` corrected
   infotaxis, `chase` plume tracing, `surgecast` cast-and-surge. The script first checks that the incremental
   evaluation of the networks matches the batch forward pass (`max |dlog q|` < 5e-3). Copy
   `{set}_closedloop.npz` and `{set}_closedloop_q.npz` to `docs/research/data/E59/`.
   Further closed-loop runs used in the paper:
   - memory study on 300 fresh tasks (E57, Fig. memory): `fermi_postnet_cl_fresh.sbatch` with `STAGE=final` and
     `STAGE=lopo`; copy `$RLS_OUT/E57fresh/{set}_closedloop.json` to `docs/research/data/E57/{set}_fresh.json` and
     `$RLS_OUT/E57lopo/{set}_closedloop.json` to `E57/{set}_lopo.json`.
   - overview figure (E40, Fig. 1 and plume snapshots): `postnet_closedloop.py --set strong --save_q --runs runs --out $RLS_OUT/E40`
     (`fermi_postnet_cl.sbatch` with `OUT=$RLS_OUT/E40 EXTRA=--save_q`), then
     `python scripts/plume_snapshots.py --set strong --task 14 --out $RLS_OUT/E40` (weak: `--task 44`); copy
     `strong_closedloop_q.npz` and `{set}_plume_task{14,44}.npz` to `docs/research/data/E40/`. This run used the first
     (E37) networks `runs`, trained with `fermi_postnet_train.sbatch` (`RUNS=runs`) before the fp32 accumulation fix
     in `rlscalar/search/postnet.py` (see "Known differences").

7. **Posterior maps and checks.**
   ```bash
   # posterior maps on the 300 held-out test episodes (E59; Figs. 3, 4, App. compare and velocity)
   python scripts/postnet_maps.py --set strong --runs runs_real --runs_vel runs_real_vel --map_all --out $RLS_OUT/E59maps
   # replay ablation of the physics posterior along the closed-loop paths (E50; Fig. collapse (a), App. ablation)
   python scripts/checks/phys_ablation_replay.py --set strong --e49 $RLS_OUT/E59 --out $RLS_OUT/E50
   # non-Gaussian measurements and evidence of one detection (E52; Fig. collapse (b), App. non-Gaussian)
   python scripts/checks/nongauss_evidence.py --set strong --out $RLS_OUT/E52
   # sensitivity of the learned posterior to the GRU inputs (E54; App. GRU sensitivity)
   python scripts/checks/gru_sensitivity.py --set strong --model b0 --out $RLS_DATA/postnet/sens          # and --vel
   python scripts/checks/gru_sens_proj.py
   ```
   Copy `$RLS_OUT/E59maps/{set}_maps.npz` to `docs/research/data/E59/`, `$RLS_OUT/E50/{set}_ablation.npz` to
   `docs/research/data/E50_{set}_ablation.npz`, `$RLS_OUT/E52/{set}_nongauss.npz` to
   `docs/research/data/E52_{set}_nongauss.npz`, and the `sens` outputs `{set}_{b0,b0_vel}_sens.npz` to
   `docs/research/data/E54/`. The ablation replays the paths stored by the closed loop; it recomputes M1 and M3 on
   them and stops if they do not reproduce the stored errors. `gru_sens_proj.py` reads `E54/{set}_b0_sens.npz` and the
   learned bottleneck networks (copied as `E54/{set}_b0_bn_learned_r{r}.pt`) and writes the fixed projections
   `E54/{set}_b0_proj_r{r}.npy`, which `fermi_postnet_mem.sbatch` and `fermi_postnet_bottleneck.sbatch` (`MODE=fixed`)
   read from `$RLS_DATA/postnet/sens/`. The bottleneck JSONs are copied to `E54/{set}_b0_bn_{learned,fixed}_r{r}.json`.

8. **Figures** (from the top of this folder; output in `figures/` as `.png` and `.eps`).

   | Paper figure (file in the manuscript) | Command | Data |
   |---|---|---|
   | Fig. 1 `fig_overview_v6` | `python scripts/figures/fig_overview_v6.py` | E40 |
   | Fig. 2 `schematic_likelihood_v7` | `python scripts/figures/schematic_likelihood_v7.py` | DNS replay |
   | Fig. 3 `schematic_posterior_net_v7` | `python scripts/figures/schematic_posterior_net_v7.py` | E59 maps |
   | Fig. 4 `fig_posterior_compare_strong_row_v6` | `python scripts/figures/fig_posterior_compare_v6.py --cols 40` | E59 maps |
   | Fig. 5 `fig_closedloop_bars_v6` | `python scripts/figures/fig_closedloop_bars_v6.py` | E59 |
   | Fig. 6 `fig_collapse_v6` | `python scripts/figures/fig_collapse_v6.py` | E50, E52 |
   | Fig. 7 `fig_memory_v6` | `python scripts/figures/fig_memory_v6.py` | E57 |
   | App. `fig_plume_snapshots` | `python scripts/figures/fig_plume_snapshots.py` | E40 |
   | App. `fig_learning_curve` | `python scripts/figures/fig_learning_curve.py` | E47, E48 |
   | App. `fig_closedloop_row_v6` | `python scripts/figures/fig_closedloop_row_v6.py` | E59 |
   | App. `fig_calib_reach_v6` | `python scripts/figures/fig_calib_reach_v6.py` | E59 |
   | App. `fig_posterior_compare_strong_v6` | `python scripts/figures/fig_posterior_compare_v6.py --cols 20 40` | E59 maps |
   | App. `fig_phys_ablation_v6` | `python scripts/figures/fig_phys_ablation_v6.py` | E50 |
   | App. `fig_nongauss_v6` | `python scripts/figures/fig_nongauss_v6.py` | E52 |
   | App. `fig_gru_sensitivity_v6` | `python scripts/figures/fig_gru_sensitivity_v6.py` | E54 |
   | App. `fig_posterior_planner_velocity_{strong,weak}_m3` | `python scripts/figures/fig_posterior_planner_velocity.py --set strong` (and `--set weak`) | E59 |
   | App. `fig_posterior_velocity_strong` | `python scripts/figures/fig_posterior_velocity.py` | E59 maps |

   The figure numbers follow the order of the figures in the submitted manuscript; the tables were compiled by hand
   from the JSON outputs of stages 5-6 (see `DATA_MANIFEST.md`).

## HPC notes

- The sbatch scripts in `scripts/hpc/` are the ones used on the authors' Slurm cluster, with personal paths replaced by
  `$RLS_CODE`, `$RLS_DATA` and `$RLS_OUT`, and the conda activation commented out (the authors used a conda env named
  `torch_AE`). Partitions (`-p cpu`, `-p wang`, `-p gpu`), memory and time limits are those of that cluster; adapt them.
  Log files go to `logs/` under the submission directory (create it first). Submit with, for example,
  `sbatch --export=ALL,SET=strong,RLS_CODE=$PWD,RLS_DATA=/path/to/data,RLS_OUT=/path/to/out scripts/hpc/fermi_postnet_e59.sbatch`.
- CPU jobs (episodes, closed loop, checks) use 36 worker processes on one node (`--workers`); single-threaded BLAS is
  set with `OMP_NUM_THREADS=1`. GPU jobs (training, GRU sensitivity) use one GPU each; the node exclusion of old K80
  GPUs (`#SBATCH -x`) is commented out.
- Exact DNS replays (`dns2_velocity.py`, `dns2_replay.py`) need the same CPU family as the original runs
  (`#SBATCH -C "Gold6240|Platinum8268"` in `fermi_dns2_velocity_node.sbatch`); on other CPUs the turbulence amplifies
  floating-point differences to O(1) after a few hundred records.
- Wall times recorded for the paper runs: velocity replay about 2 h per forcing on one 16-core node; generic episodes
  51-74 min per forcing on 36 cores; training about 7 min per 9500-episode network on one GPU; closed loop and maps
  3-8 min per forcing on 36 cores. The DNS sbatch requests 12 h per run on 2 cores and resumes from checkpoints.

## Known differences and limits

- The overview figure (E40) uses the first learned posteriors (`runs`, E37), trained before the per-measurement sum in
  `PostNet.forward` was accumulated in fp32 (E46). Retraining `runs` with the current code does not reproduce those
  networks exactly; the E40 figure data are therefore included in `figure_data/`. All other learned posteriors in the
  paper were trained with the current code.
- The step that adds `src_r = 0.1` to the Taylor fit (stage 2, third line) is not a script in the original repository;
  the command shown reproduces the file content that the downstream scripts read (`surrogate.src_r`).
- `scripts/search_sweep.py` is included only for `dns_factory`, the mapping from an episode seed to a DNS run, plume and
  start record; its sweep runner belongs to earlier experiments that are not part of this paper.
- Docstrings refer to internal documents (`docs/research/spec_posterior_networks.md`, the experiment log) that are not
  part of this release.
