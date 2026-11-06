"""Training / test data for the learned grid posteriors B0 and B (docs/research/spec_posterior_networks.md, section 4).

Each episode: a task with the search geometry (snap_src, start x = 3.5, 80 moves) in the given DNS files, a sensor path
from one of three policies (episode index i: i % 3 == 0 one-step EIG on the physics belief, 1 zigzag survey,
2 persistent random walk), all 81 readings, and the physics (M1) log-posterior after n = 10, 20, ..., 80 moves.
E55 (--mix 5): i % 5, adding 3 plume tracing and 4 cast-and-surge (the model-free planners of E42), so the training
paths include long in-plume runs of detections; corrected infotaxis (m3) stays out of training as the held-out path type.
E56 (--paths eig eig3 chase surgecast): the paths of the four closed-loop planners themselves, with eig3 = one-step EIG
on the corrected-model belief M3 (as postnet_closedloop.py builds it). policy codes: eig 0, zigzag 1, walk 2, chase 3,
surgecast 4, eig3 5.
M1 = the E26 belief: adjoint mean field (Taylor width + finite source) with the zi2 observation model and kappa tempering.
Output: one npz shard per worker block with arrays
  tok (E, 81, 7)     reading tokens: detected flag, (z - log m_min)/5 or 0, x/4, y, t/T, dx/(V dt), dy/(V dt)
  pos (E, 81, 2)     sensor positions; zraw (E, 81) log c or nan (censored); src (E, 2); policy (E,); seed (E,)
  lpost (E, 8, J)    float16 physics log-posterior at n = 10..80 (grid as Belief: xs fastest-varying second)
Usage (Fermi CPU node):
  python scripts/postnet_data.py --set strong --split train --n 10000 --out $RLS_DATA/postnet/data
  python scripts/postnet_data.py --set strong --split test --n 300 --out ...
"""
import argparse, json, os, sys
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataclasses import replace
from rlscalar.search.core import Params, Belief, step_pose
from rlscalar.search.episode import sample_task
from rlscalar.search.policies import Greedy, Zigzag, Chase, SurgeCast

ap = argparse.ArgumentParser()
ap.add_argument("--set", default="strong")
ap.add_argument("--split", default="train", choices=["train", "test"])
ap.add_argument("--n", type=int, default=10000)
ap.add_argument("--root", default=os.environ.get("RLS_DATA", "data") + "/dns")
ap.add_argument("--fitdir", default=os.environ.get("RLS_DATA", "data") + "/dns/fits")
ap.add_argument("--workers", type=int, default=36)
ap.add_argument("--block", type=int, default=250)
ap.add_argument("--mix", type=int, default=3, choices=[3, 5], help="number of path types (E55: 5)")
ap.add_argument("--paths", nargs="*", default=[], help="path types in rotation, overrides --mix (E56)")
ap.add_argument("--out", required=True)
a = ap.parse_args()

SEEDS = (1, 2, 3, 4, 5) if a.split == "train" else (6, 7, 8)
SEED0 = 7_000_000 if a.split == "train" else 0             # test: episode seeds 0..n-1 (E26 tasks are 0..99)
NS = np.arange(10, 81, 10)


def build_params():
    """The E26 physics belief, exactly as search_sweep.py builds it from --dns_fit / --dns_lik / --temper / --snap."""
    fit = json.load(open(f"{a.fitdir}/{a.set}_fit_taylor_src.json"))
    sur = fit["surrogate"]
    z2 = json.load(open(f"{a.fitdir}/{a.set}_lik_taylor_src.json"))["zi2"]
    return Params(U=sur["U"], K=sur["K"], sigma=z2["sigma"], tau=sur["tau"], L=sur["L"],
                  Q=float(np.median([q["mass_flux"] for q in fit["plumes"]])), T_L=sur["T_L"],
                  s0sq=sur.get("s0sq_taylor", 0.0), src_r=sur.get("src_r", 0.0), n_steps=80, snap_src=True,
                  temper=True, obs="zi", **{k: z2[k] for k in ("zi_a0", "zi_a1", "zi_c0", "zi_c1", "leak_g", "leak_mu", "leak_s")})


def m3_params():
    """M3 exactly as postnet_closedloop.py builds it (the E36h1000000 corrected model)."""
    base = f"{a.fitdir}/nn/{a.set}_N1000000_map_fix_traj"
    fit, z2 = json.load(open(f"{base}_fit.json")), json.load(open(f"{base}_lik.json"))["zi2"]
    sur = fit["surrogate"]
    pd = dict(U=sur["U"], K=sur["K"], tau=sur["tau"], L=sur["L"], Q=float(np.median([q["mass_flux"] for q in fit["plumes"]])),
              s0sq=max(0.0, float(np.median([q["s0sq"] for q in fit["plumes"]]))), n_steps=80, snap_src=True, temper=True)
    if sur.get("T_L"):
        pd.update(T_L=sur["T_L"], s0sq=sur.get("s0sq_taylor", 0.0))
    if sur.get("src_r"):
        pd.update(src_r=sur["src_r"])
    pd.update(obs="zi", sigma=z2["sigma"], **{k: z2[k] for k in ("zi_a0", "zi_a1", "zi_c0", "zi_c1", "leak_g", "leak_mu", "leak_s")})
    return replace(Params(**pd, nn_file=f"{base}_hyb.npz"), mean_model="nn", obs="zi")


CODES = dict(eig=0, zigzag=1, walk=2, chase=3, surgecast=4, eig3=5)
PATHS = a.paths or list(CODES)[:a.mix]
P = build_params()
P3 = m3_params() if "eig3" in PATHS else None
FILES = [f"{a.root}/{a.set}/seed{s:02d}.npz" for s in SEEDS]


def episode(i):
    from search_sweep import dns_factory
    seed = SEED0 + i
    make = episode.make = getattr(episode, "make", None) or dns_factory(FILES)
    src, pos = sample_task(P, seed)
    world = make(P, src, np.random.default_rng(10_000 + seed), seed)
    rng = np.random.default_rng(30_000 + seed)
    b = Belief(P, correlated=True)
    pol = CODES[PATHS[i % len(PATHS)]]
    greedy, zz, heading = Greedy(), Zigzag(), int(rng.integers(8))
    zz.reset()
    free = {3: Chase, 4: SurgeCast}.get(pol)
    if free:
        free = free()
        free.reset()
    b3 = Belief(P3, correlated=False) if pol == 5 else None
    T, thr = P.n_steps, np.log(P.m_min)
    tok = np.zeros((T + 1, 7), np.float32); posa = np.zeros((T + 1, 2), np.float32); zr = np.full(T + 1, np.nan, np.float32)
    lpost = np.zeros((len(NS), b.cx.size), np.float16)
    t, z, prev = 0.0, world.measure(*pos, 0.0), pos.copy()
    for k in range(T + 1):
        if k > 0:
            if pol == 0:
                act = greedy.act(dict(belief=b, pos=pos, t=t, p=P))
            elif pol == 1:
                act = zz.act(dict(pos=pos))
            elif free:
                act = free.act(dict(pos=pos, t=t, p=P, last_z=z))
            elif pol == 5:
                act = greedy.act(dict(belief=b3, pos=pos, t=t, p=P3))
            else:
                if rng.uniform() < 0.2:
                    heading = int(rng.integers(8))
                act = heading
            prev, pos, t = pos, step_pose(pos, act, P), t + P.dt
            z = world.measure(*pos, t)
        b.update(*pos, t, z)
        if b3 is not None:
            b3.update(*pos, t, z)
        d = pos - prev
        tok[k] = [0.0 if z is None else 1.0, 0.0 if z is None else (z - thr) / 5, pos[0] / 4, pos[1], k / T,
                  d[0] / (P.V * P.dt), d[1] / (P.V * P.dt)]
        posa[k] = pos; zr[k] = np.nan if z is None else z
        if k in NS:
            lpost[list(NS).index(k)] = b.logw.astype(np.float16)
    return dict(tok=tok, pos=posa, zraw=zr, src=src.astype(np.float32), policy=pol, seed=seed, lpost=lpost)


if __name__ == "__main__":
    os.makedirs(a.out, exist_ok=True)
    blocks = [list(range(s, min(s + a.block, a.n))) for s in range(0, a.n, a.block)]
    with Pool(a.workers, maxtasksperchild=50) as pool:
        for bi, blk in enumerate(blocks):
            fn = f"{a.out}/{a.set}_{a.split}_{bi:03d}.npz"
            if os.path.exists(fn):
                continue
            eps = pool.map(episode, blk, chunksize=1)
            np.savez(fn + ".tmp.npz", **{k: np.array([e[k] for e in eps]) for k in eps[0]},
                     grid_x=np.unique(Belief(P).cx), grid_y=np.unique(Belief(P).cy))
            os.replace(fn + ".tmp.npz", fn)
            print(f"block {bi + 1}/{len(blocks)} done", flush=True)
