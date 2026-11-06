"""Add the local velocity at the sensor to the postnet data shards (E48: velocity sensing; velocity from E44 v2).

For every episode in <data>/<set>_<split>_XXX.npz this rebuilds the DNS truth mapping exactly as search_sweep.dns_factory
does (file = seed % n_files, plume and start record from default_rng(50_000 + seed)), translates each stored sensor
position by the plume offset, and interpolates the velocity sub-grid (bilinear in space, periodic in y, nearest record
in time), as DNSWorld.measure does for c. The sensor reads (u - U, v) with no noise; outside the recorded x-range
(upstream of the plume origin or in the sponge) it reads 0 (= the mean wind) and the flag is 0.
Output <data>/<set>_<split>_XXX_vel.npz: uv (E, 81, 2) float32, inrange (E, 81) bool.
Usage (Fermi CPU): python scripts/postnet_add_velocity.py --set strong --data $RLS_DATA/postnet/data
"""
import argparse, glob, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--set", default="strong")
ap.add_argument("--data", required=True)
ap.add_argument("--root", default=os.environ.get("RLS_DATA", "data") + "/dns")
ap.add_argument("--vel", default=os.environ.get("RLS_DATA", "data") + "/dns/vel")
ap.add_argument("--dt", type=float, default=0.1, help="sensor time step P.dt")
ap.add_argument("--n_steps", type=int, default=80)
a = ap.parse_args()

for split, seeds in (("train", (1, 2, 3, 4, 5)), ("test", (6, 7, 8))):
    D = []
    for s in seeds:
        ref = np.load(f"{a.root}/{a.set}/seed{s:02d}.npz")
        P = json.loads(str(ref["params"]))
        v = np.load(f"{a.vel}/{a.set}/seed{s:02d}_vel.npz")
        assert np.allclose(v["times"], ref["times"]) and np.allclose(v["sub_x"], ref["sub_x"]) and v["sub_maxdiff"].max() == 0
        D.append(dict(P=P, U=P["U"], sx=v["sub_x"], sy=v["sub_y"], u=v["u_sub"], v=v["v_sub"], c=ref["c_sub"], nt=ref["c_sub"].shape[0],
                      dt=float(ref["times"][1] - ref["times"][0])))
    rms = []
    for fn in sorted(glob.glob(f"{a.data}/{a.set}_{split}_[0-9][0-9][0-9].npz")):
        z = np.load(fn)
        seed, src, pos = z["seed"], z["src"], z["pos"]
        E, K = pos.shape[:2]
        uv, inr = np.zeros((E, K, 2), np.float32), np.zeros((E, K), bool)
        cc = np.zeros((E, K), np.float32)          # check: c from the same mapping vs the stored noisy reading zraw
        for e in range(E):
            d = D[int(seed[e]) % len(D)]
            r = np.random.default_rng(50_000 + int(seed[e]))
            plume = int(r.integers(len(d["P"]["sources"])))
            nrec = int(np.ceil((a.n_steps + 1) * a.dt / d["dt"])) + 1
            t0 = int(r.integers(0, d["nt"] - nrec))
            xs_d, ys_d = d["P"]["sources"][plume]
            nx, ny = len(d["sx"]), len(d["sy"])
            for k in range(K):
                X, Y = pos[e, k, 0] + xs_d - src[e, 0], (pos[e, k, 1] + ys_d - src[e, 1]) % d["P"]["Ly"]
                kk = t0 + int(round(k * a.dt / d["dt"]))
                fi = (X - d["sx"][0]) / (d["sx"][1] - d["sx"][0])
                if fi < 0 or fi > nx - 1 or kk >= d["nt"]:
                    continue
                fj = (Y - d["sy"][0]) / (d["sy"][1] - d["sy"][0])
                i0, j0 = int(fi), int(fj)
                al, be = fi - i0, fj - j0
                i1, j0, j1 = min(i0 + 1, nx - 1), j0 % ny, (j0 + 1) % ny
                for c, f in enumerate((d["u"][kk], d["v"][kk], d["c"][kk, plume])):
                    f = f.astype(np.float32)
                    val = ((1 - al) * ((1 - be) * f[i0, j0] + be * f[i0, j1])
                                   + al * ((1 - be) * f[i1, j0] + be * f[i1, j1]))
                    if c < 2:
                        uv[e, k, c] = val
                    else:
                        cc[e, k] = val
                uv[e, k, 0] -= d["U"]
                inr[e, k] = True
        np.savez(fn.replace(".npz", "_vel.npz"), uv=uv, inrange=inr)
        rms.append(np.sqrt((uv[inr] ** 2).mean(0)))
        det = np.isfinite(z["zraw"])
        res = z["zraw"][det] - np.log(np.maximum(cc[det], 1e-30))
        print(f"{os.path.basename(fn)}: in range {inr.mean():.3f}, rms (u-U, v) {rms[-1].round(3)}; "
              f"check zraw - log c: mean {res.mean():.3f} sd {res.std():.3f} max |.| {np.abs(res).max():.2f} "
              f"(sensor noise sd, c > 0 at all {det.sum()} detections: {bool((cc[det] > 0).all())})", flush=True)
    print(f"{a.set} {split}: mean rms (u-U, v) {np.mean(rms, 0).round(3)}", flush=True)
