"""E54: fixed projections for the bottleneck runs from the GRU-path sensitivity (docs/research/data/E54/{set}_b0_sens.npz),
and the learned rank-r projections compared with them.

proj_r{r}: the r leading eigenvectors of the Fisher-weighted sensitivity G (standardised token coordinates), mapped back
to raw token coordinates (columns V / std); proj_r0: a zero column, so the GRU receives no measurement (no memory of
the readings). The detection flag d is not in G (it is binary); it still enters the decoder directly.
Run (from repo/): py -3.11 scripts/checks/gru_sens_proj.py   -> docs/research/data/E54/{set}_b0_proj_r{r}.npy
"""
import numpy as np
import torch

D = "docs/research/data/E54"
for s in ("strong", "weak"):
    z = np.load(f"{D}/{s}_b0_sens.npz")
    names, std = list(z["names"]), z["std"]
    w, V = np.linalg.eigh(z["Gg"][0])
    V = V[:, ::-1]
    np.save(f"{D}/{s}_b0_proj_r0.npy", np.zeros((len(names), 1), np.float32))
    for r in (1, 2, 3):
        np.save(f"{D}/{s}_b0_proj_r{r}.npy", (V[:, :r] / std[:, None]).astype(np.float32))
    print(s, "eigval shares", np.round(w[::-1] / w.sum(), 3))
    for r in (1, 2, 3):
        W = torch.load(f"{D}/{s}_b0_bn_learned_r{r}.pt", map_location="cpu")["proj.weight"].numpy().T   # (F, r) raw
        Ws = W * std[:, None]                                               # standardised coordinates
        Ws[0] = 0                                                           # d is outside G
        Q, _ = np.linalg.qr(Ws)
        cos = np.linalg.svd(Q.T @ V[:, :r], compute_uv=False)              # cosines of the principal angles
        lead = Ws[:, 0] / np.linalg.norm(Ws[:, 0])
        print(f"  learned r={r}: cos principal angles with G's top-{r}", np.round(cos, 3),
              "| learned dir 1:", dict(zip(names, np.round(lead, 2))))
