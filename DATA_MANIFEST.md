# Data manifest

The data are published on Zenodo ([10.5281/zenodo.23991624](https://doi.org/10.5281/zenodo.23991624)) as tar archives that
unpack into the four folders below; the record's `README.md` gives the commands.

| Zenodo folder | Contents | Where the code expects it |
|---|---|---|
| `figure_data/` | the small processed outputs the figure scripts read (listed below) | unpack into `docs/research/data/` of this repository |
| `networks/` | all trained learned posteriors (`networks.tar`): `runs_real`, `runs_real_vel` (main results), `runs_vel`, `runs_fp32` and the top-level `{set}_{b,b0}_n{N}_r{seed}` files (generic paths, learning curve), `runs_nomem`, `runs_real_nomem`, `runs_full_reps`, `runs_mix5*`, `runs_lopo_*` (memory study), `runs_bn_*` and `sens/` (GRU sensitivity), `runs` (first networks, Fig. 1), and the E51/E53 variants; each run is `{set}_{b,b0}_n{n_train}_r{seed}.{pt,json}` | `$RLS_DATA/postnet/` |
| `episodes/` | `data/` (generic paths: physics infotaxis, zigzag, random walk), `data_real/` (paths of the four motion policies) and `data_mix5/` (five-path mixture, memory study): `{set}_{train,test}_XXX.npz` shards and their `_vel.npz` velocity files | `$RLS_DATA/postnet/` |
| `dns/` | DNS runs `{strong,weak}/seed0[1-8].npz`, the full-resolution replay window `replay/strong_seed06_k287.npz`, the sensor-grid velocity `vel/{set}/seed0X_vel.npz`, the fits `{set}_fit.json` and `fits/` (`{set}_fit_taylor.json`, `{set}_fit_taylor_src.json`, `{set}_lik_taylor_src.json`, `nn/{set}_N1000000_map_fix_traj_{fit,lik}.json`, `_hyb.npz`, `_nn.npz`) | `$RLS_DATA/dns/` |

`networks/` and `episodes/` are merged into one folder, `$RLS_DATA/postnet/`, because the scripts take a single `--pn` /
`--data` root that holds both the run folders and the episode folders.

## Figure data read by the figure scripts

Paths are relative to `docs/research/data/`. The list was produced by expanding every file name pattern in the 16 figure
scripts (loops over forcing, model, training size and training seed) and keeping the files that exist; scripts that skip
missing files (`fig_learning_curve.py`, `fig_gru_sensitivity_v6.py`) only read the files listed.
166 files, 202.2 MiB in total.

| File | Size (bytes) | Read by |
|---|---|---|
| `E40/strong_closedloop_q.npz` | 10,534,181 | fig_overview_v6 |
| `E40/strong_plume_task14.npz` | 342,435 | fig_overview_v6, fig_plume_snapshots |
| `E40/weak_plume_task44.npz` | 279,430 | fig_plume_snapshots |
| `viz/replay/strong_seed06_k287.npz` | 74,375,599 | schematic_likelihood_v7 |
| `E59/strong_maps.npz` | 15,073,330 | schematic_posterior_net_v7, fig_posterior_compare_v6, fig_posterior_velocity |
| `E59/weak_maps.npz` | 8,722,534 | fig_posterior_compare_v6 --set weak, fig_posterior_velocity --set weak (not in the paper) |
| `E59/strong_closedloop.npz` | 8,298,228 | fig_closedloop_bars_v6, fig_closedloop_row_v6, fig_calib_reach_v6 |
| `E59/strong_closedloop_q.npz` | 43,199,978 | fig_posterior_planner_velocity --set strong, fig_calib_reach_v6 |
| `E50_strong_ablation.npz` | 5,884,962 | fig_collapse_v6, fig_phys_ablation_v6 |
| `E52_strong_nongauss.npz` | 1,406,896 | fig_nongauss_v6, fig_collapse_v6 |
| `E57/strong_fresh.json` | 36,898 | fig_memory_v6 |
| `E57/strong_lopo.json` | 20,328 | fig_memory_v6 |
| `E47/strong_b_n10_r0.json` | 5,349 | fig_learning_curve |
| `E47/strong_b_n10_r1.json` | 5,212 | fig_learning_curve |
| `E47/strong_b_n10_r2.json` | 5,111 | fig_learning_curve |
| `E47/strong_b_n30_r0.json` | 5,640 | fig_learning_curve |
| `E47/strong_b_n30_r1.json` | 5,242 | fig_learning_curve |
| `E47/strong_b_n30_r2.json` | 5,891 | fig_learning_curve |
| `E47/strong_b_n100_r0.json` | 7,188 | fig_learning_curve |
| `E47/strong_b_n100_r1.json` | 5,212 | fig_learning_curve |
| `E47/strong_b_n100_r2.json` | 6,769 | fig_learning_curve |
| `E47/strong_b_n300_r0.json` | 7,730 | fig_learning_curve |
| `E47/strong_b_n300_r1.json` | 7,391 | fig_learning_curve |
| `E47/strong_b_n300_r2.json` | 7,586 | fig_learning_curve |
| `E47/strong_b_n1000_r0.json` | 6,661 | fig_learning_curve |
| `E47/strong_b_n1000_r1.json` | 7,548 | fig_learning_curve |
| `E47/strong_b_n1000_r2.json` | 7,269 | fig_learning_curve |
| `E47/strong_b_n3000_r0.json` | 6,861 | fig_learning_curve |
| `E47/strong_b_n9500_r0.json` | 6,679 | fig_learning_curve |
| `E47/strong_b0_n10_r0.json` | 5,080 | fig_learning_curve |
| `E47/strong_b0_n10_r1.json` | 5,356 | fig_learning_curve |
| `E47/strong_b0_n10_r2.json` | 7,443 | fig_learning_curve |
| `E47/strong_b0_n30_r0.json` | 7,471 | fig_learning_curve |
| `E47/strong_b0_n30_r1.json` | 8,495 | fig_learning_curve |
| `E47/strong_b0_n30_r2.json` | 9,048 | fig_learning_curve |
| `E47/strong_b0_n100_r0.json` | 8,304 | fig_learning_curve |
| `E47/strong_b0_n100_r1.json` | 8,313 | fig_learning_curve |
| `E47/strong_b0_n100_r2.json` | 8,458 | fig_learning_curve |
| `E47/strong_b0_n300_r0.json` | 7,350 | fig_learning_curve |
| `E47/strong_b0_n300_r1.json` | 7,403 | fig_learning_curve |
| `E47/strong_b0_n300_r2.json` | 7,836 | fig_learning_curve |
| `E47/strong_b0_n1000_r0.json` | 7,183 | fig_learning_curve |
| `E47/strong_b0_n1000_r1.json` | 6,639 | fig_learning_curve |
| `E47/strong_b0_n1000_r2.json` | 7,235 | fig_learning_curve |
| `E47/strong_b0_n3000_r0.json` | 7,088 | fig_learning_curve |
| `E47/strong_b0_n9500_r0.json` | 6,647 | fig_learning_curve |
| `E48/strong_b_n10_r0.json` | 5,213 | fig_learning_curve |
| `E48/strong_b_n10_r1.json` | 5,202 | fig_learning_curve |
| `E48/strong_b_n10_r2.json` | 5,299 | fig_learning_curve |
| `E48/strong_b_n30_r0.json` | 5,559 | fig_learning_curve |
| `E48/strong_b_n30_r1.json` | 5,184 | fig_learning_curve |
| `E48/strong_b_n30_r2.json` | 5,540 | fig_learning_curve |
| `E48/strong_b_n100_r0.json` | 6,995 | fig_learning_curve |
| `E48/strong_b_n100_r1.json` | 5,240 | fig_learning_curve |
| `E48/strong_b_n100_r2.json` | 6,754 | fig_learning_curve |
| `E48/strong_b_n300_r0.json` | 7,068 | fig_learning_curve |
| `E48/strong_b_n300_r1.json` | 6,796 | fig_learning_curve |
| `E48/strong_b_n300_r2.json` | 6,799 | fig_learning_curve |
| `E48/strong_b_n1000_r0.json` | 6,865 | fig_learning_curve |
| `E48/strong_b_n1000_r1.json` | 6,633 | fig_learning_curve |
| `E48/strong_b_n1000_r2.json` | 7,250 | fig_learning_curve |
| `E48/strong_b_n3000_r0.json` | 6,730 | fig_learning_curve |
| `E48/strong_b_n9500_r0.json` | 7,068 | fig_learning_curve |
| `E48/strong_b0_n10_r0.json` | 6,405 | fig_learning_curve |
| `E48/strong_b0_n10_r1.json` | 5,388 | fig_learning_curve |
| `E48/strong_b0_n10_r2.json` | 5,688 | fig_learning_curve |
| `E48/strong_b0_n30_r0.json` | 8,766 | fig_learning_curve |
| `E48/strong_b0_n30_r1.json` | 7,959 | fig_learning_curve |
| `E48/strong_b0_n30_r2.json` | 9,095 | fig_learning_curve |
| `E48/strong_b0_n100_r0.json` | 8,153 | fig_learning_curve |
| `E48/strong_b0_n100_r1.json` | 7,646 | fig_learning_curve |
| `E48/strong_b0_n100_r2.json` | 8,276 | fig_learning_curve |
| `E48/strong_b0_n300_r0.json` | 7,419 | fig_learning_curve |
| `E48/strong_b0_n300_r1.json` | 7,301 | fig_learning_curve |
| `E48/strong_b0_n300_r2.json` | 7,452 | fig_learning_curve |
| `E48/strong_b0_n1000_r0.json` | 7,308 | fig_learning_curve |
| `E48/strong_b0_n1000_r1.json` | 6,756 | fig_learning_curve |
| `E48/strong_b0_n1000_r2.json` | 7,559 | fig_learning_curve |
| `E48/strong_b0_n3000_r0.json` | 6,901 | fig_learning_curve |
| `E48/strong_b0_n9500_r0.json` | 6,744 | fig_learning_curve |
| `E54/strong_b0_sens.npz` | 719,152 | fig_gru_sensitivity_v6 |
| `E54/strong_b0_vel_sens.npz` | 719,896 | fig_gru_sensitivity_v6 |
| `E54/strong_b0_full.json` | 6,647 | fig_gru_sensitivity_v6 |
| `E54/strong_b0_bn_learned_r1.json` | 6,485 | fig_gru_sensitivity_v6 |
| `E54/strong_b0_bn_learned_r2.json` | 6,446 | fig_gru_sensitivity_v6 |
| `E54/strong_b0_bn_learned_r3.json` | 6,941 | fig_gru_sensitivity_v6 |
| `E59/weak_closedloop.npz` | 8,298,228 | fig_closedloop_bars_v6, fig_closedloop_row_v6, fig_calib_reach_v6 --set weak |
| `E59/weak_closedloop_q.npz` | 24,256,323 | fig_posterior_planner_velocity --set weak, fig_calib_reach_v6 --set weak |
| `E50_weak_ablation.npz` | 5,884,962 | fig_collapse_v6, fig_phys_ablation_v6 |
| `E52_weak_nongauss.npz` | 1,406,896 | fig_nongauss_v6 |
| `E57/weak_fresh.json` | 36,665 | fig_memory_v6 |
| `E57/weak_lopo.json` | 20,175 | fig_memory_v6 |
| `E47/weak_b_n10_r0.json` | 5,480 | fig_learning_curve |
| `E47/weak_b_n10_r1.json` | 6,789 | fig_learning_curve |
| `E47/weak_b_n10_r2.json` | 5,196 | fig_learning_curve |
| `E47/weak_b_n30_r0.json` | 5,789 | fig_learning_curve |
| `E47/weak_b_n30_r1.json` | 5,213 | fig_learning_curve |
| `E47/weak_b_n30_r2.json` | 6,191 | fig_learning_curve |
| `E47/weak_b_n100_r0.json` | 8,738 | fig_learning_curve |
| `E47/weak_b_n100_r1.json` | 7,924 | fig_learning_curve |
| `E47/weak_b_n100_r2.json` | 8,598 | fig_learning_curve |
| `E47/weak_b_n300_r0.json` | 8,272 | fig_learning_curve |
| `E47/weak_b_n300_r1.json` | 7,959 | fig_learning_curve |
| `E47/weak_b_n300_r2.json` | 8,694 | fig_learning_curve |
| `E47/weak_b_n1000_r0.json` | 7,047 | fig_learning_curve |
| `E47/weak_b_n1000_r1.json` | 7,411 | fig_learning_curve |
| `E47/weak_b_n1000_r2.json` | 6,698 | fig_learning_curve |
| `E47/weak_b_n3000_r0.json` | 7,300 | fig_learning_curve |
| `E47/weak_b_n9500_r0.json` | 6,183 | fig_learning_curve |
| `E47/weak_b0_n10_r0.json` | 5,140 | fig_learning_curve |
| `E47/weak_b0_n10_r1.json` | 5,299 | fig_learning_curve |
| `E47/weak_b0_n10_r2.json` | 7,498 | fig_learning_curve |
| `E47/weak_b0_n30_r0.json` | 9,168 | fig_learning_curve |
| `E47/weak_b0_n30_r1.json` | 11,640 | fig_learning_curve |
| `E47/weak_b0_n30_r2.json` | 9,466 | fig_learning_curve |
| `E47/weak_b0_n100_r0.json` | 8,522 | fig_learning_curve |
| `E47/weak_b0_n100_r1.json` | 8,963 | fig_learning_curve |
| `E47/weak_b0_n100_r2.json` | 9,588 | fig_learning_curve |
| `E47/weak_b0_n300_r0.json` | 6,897 | fig_learning_curve |
| `E47/weak_b0_n300_r1.json` | 8,279 | fig_learning_curve |
| `E47/weak_b0_n300_r2.json` | 7,678 | fig_learning_curve |
| `E47/weak_b0_n1000_r0.json` | 7,913 | fig_learning_curve |
| `E47/weak_b0_n1000_r1.json` | 8,676 | fig_learning_curve |
| `E47/weak_b0_n1000_r2.json` | 8,611 | fig_learning_curve |
| `E47/weak_b0_n3000_r0.json` | 7,182 | fig_learning_curve |
| `E47/weak_b0_n9500_r0.json` | 7,201 | fig_learning_curve |
| `E48/weak_b_n10_r0.json` | 5,377 | fig_learning_curve |
| `E48/weak_b_n10_r1.json` | 6,784 | fig_learning_curve |
| `E48/weak_b_n10_r2.json` | 5,427 | fig_learning_curve |
| `E48/weak_b_n30_r0.json` | 5,835 | fig_learning_curve |
| `E48/weak_b_n30_r1.json` | 5,303 | fig_learning_curve |
| `E48/weak_b_n30_r2.json` | 5,778 | fig_learning_curve |
| `E48/weak_b_n100_r0.json` | 7,643 | fig_learning_curve |
| `E48/weak_b_n100_r1.json` | 7,957 | fig_learning_curve |
| `E48/weak_b_n100_r2.json` | 7,196 | fig_learning_curve |
| `E48/weak_b_n300_r0.json` | 7,821 | fig_learning_curve |
| `E48/weak_b_n300_r1.json` | 7,849 | fig_learning_curve |
| `E48/weak_b_n300_r2.json` | 7,642 | fig_learning_curve |
| `E48/weak_b_n1000_r0.json` | 6,928 | fig_learning_curve |
| `E48/weak_b_n1000_r1.json` | 7,530 | fig_learning_curve |
| `E48/weak_b_n1000_r2.json` | 7,055 | fig_learning_curve |
| `E48/weak_b_n3000_r0.json` | 6,880 | fig_learning_curve |
| `E48/weak_b_n9500_r0.json` | 6,517 | fig_learning_curve |
| `E48/weak_b0_n10_r0.json` | 6,448 | fig_learning_curve |
| `E48/weak_b0_n10_r1.json` | 5,841 | fig_learning_curve |
| `E48/weak_b0_n10_r2.json` | 5,703 | fig_learning_curve |
| `E48/weak_b0_n30_r0.json` | 9,908 | fig_learning_curve |
| `E48/weak_b0_n30_r1.json` | 10,232 | fig_learning_curve |
| `E48/weak_b0_n30_r2.json` | 8,670 | fig_learning_curve |
| `E48/weak_b0_n100_r0.json` | 9,306 | fig_learning_curve |
| `E48/weak_b0_n100_r1.json` | 8,291 | fig_learning_curve |
| `E48/weak_b0_n100_r2.json` | 9,246 | fig_learning_curve |
| `E48/weak_b0_n300_r0.json` | 8,724 | fig_learning_curve |
| `E48/weak_b0_n300_r1.json` | 7,763 | fig_learning_curve |
| `E48/weak_b0_n300_r2.json` | 7,461 | fig_learning_curve |
| `E48/weak_b0_n1000_r0.json` | 7,925 | fig_learning_curve |
| `E48/weak_b0_n1000_r1.json` | 7,475 | fig_learning_curve |
| `E48/weak_b0_n1000_r2.json` | 7,371 | fig_learning_curve |
| `E48/weak_b0_n3000_r0.json` | 7,728 | fig_learning_curve |
| `E48/weak_b0_n9500_r0.json` | 7,176 | fig_learning_curve |
| `E54/weak_b0_sens.npz` | 719,152 | fig_gru_sensitivity_v6 |
| `E54/weak_b0_vel_sens.npz` | 719,896 | fig_gru_sensitivity_v6 |
| `E54/weak_b0_full.json` | 7,201 | fig_gru_sensitivity_v6 |
| `E54/weak_b0_bn_learned_r1.json` | 6,629 | fig_gru_sensitivity_v6 |
| `E54/weak_b0_bn_learned_r2.json` | 7,135 | fig_gru_sensitivity_v6 |
| `E54/weak_b0_bn_learned_r3.json` | 7,292 | fig_gru_sensitivity_v6 |

## Table data (not read by any figure script)

The offline table of the paper (Table "Posteriors on 300 held-out test episodes") was compiled from the evaluation JSONs
that `scripts/postnet_train.py` writes next to each trained network (keys `net` and `physics`, `all_n20/40/80`).
They are part of `networks/` (`runs_real`, `runs_real_vel`); they are small and can also be put in `figure_data/`
(at `E58/runs_real/` and `E59/runs_real_vel/`) so that the table can be checked without `networks/`:

| File | Size (bytes) |
|---|---|
| `E58/runs_real/strong_b0_n9500_r0.json` | 9,212 |
| `E58/runs_real/strong_b0_n9500_r1.json` | 9,328 |
| `E58/runs_real/strong_b0_n9500_r2.json` | 8,562 |
| `E58/runs_real/strong_b_n9500_r0.json` | 8,803 |
| `E58/runs_real/strong_b_n9500_r1.json` | 8,965 |
| `E58/runs_real/strong_b_n9500_r2.json` | 8,673 |
| `E58/runs_real/weak_b0_n9500_r0.json` | 9,334 |
| `E58/runs_real/weak_b0_n9500_r1.json` | 9,097 |
| `E58/runs_real/weak_b0_n9500_r2.json` | 8,574 |
| `E58/runs_real/weak_b_n9500_r0.json` | 9,933 |
| `E58/runs_real/weak_b_n9500_r1.json` | 9,130 |
| `E58/runs_real/weak_b_n9500_r2.json` | 8,306 |
| `E59/runs_real_vel/strong_b0_n9500_r0.json` | 9,100 |
| `E59/runs_real_vel/strong_b0_n9500_r1.json` | 9,000 |
| `E59/runs_real_vel/strong_b0_n9500_r2.json` | 8,701 |
| `E59/runs_real_vel/strong_b_n9500_r0.json` | 7,684 |
| `E59/runs_real_vel/strong_b_n9500_r1.json` | 8,586 |
| `E59/runs_real_vel/strong_b_n9500_r2.json` | 7,981 |
| `E59/runs_real_vel/weak_b0_n9500_r0.json` | 9,401 |
| `E59/runs_real_vel/weak_b0_n9500_r1.json` | 9,216 |
| `E59/runs_real_vel/weak_b0_n9500_r2.json` | 8,387 |
| `E59/runs_real_vel/weak_b_n9500_r0.json` | 8,694 |
| `E59/runs_real_vel/weak_b_n9500_r1.json` | 8,329 |
| `E59/runs_real_vel/weak_b_n9500_r2.json` | 8,631 |

The fitted parameters of the physics model (appendix table) are in `dns/{strong,weak}_fit.json` (correlation scales
`tau`, `L`), `dns/fits/{set}_fit_taylor_src.json` (`K`, `T_L`, `s0sq_taylor`) and `dns/fits/{set}_lik_taylor_src.json`
(likelihood coefficients, key `zi2`).
