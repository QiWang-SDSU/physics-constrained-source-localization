"""E50: why does the physics posterior M1 collapse along corrected-infotaxis (m3) paths? Replay ablation.

The sensor paths of E49 ({e49}/{set}_closedloop.npz) are replayed in the same DNS worlds (same seeds, so the same
measurements), and M1 is recomputed with one ingredient changed at a time:
  base     M1 as in the paper (must reproduce the stored M1 errors)
  k1       measurement weights kappa_k = 1                       } are the measurements weighted too heavily?
  k05/k025 kappa_k halved / quartered                            }
  nobg     pairs (k, j) with the sensor upstream of x_j ignored  } is the background-detection term misspecified?
  bgM3     those pairs scored with the corrected model M3        }
  plumeM3  the other pairs (sensor downstream of x_j) scored with M3: is the in-plume model misspecified?
  nond     non-detections ignored
  M3       corrected-model posterior (must reproduce the stored M3 errors)
Upstream = M1's mean field is zero (M1 uses the background constants there). Diagnostic (base M1): per-measurement
contribution kappa_k [log p_k(j*) - log p_k(jhat)] of the true cell j* against the final MAP cell jhat, with the
detection flag and the sensor offset from the true source, to see which measurements pull the posterior away.
Output: {out}/{set}_ablation.npz and {set}_ablation.json.
  python scripts/checks/phys_ablation_replay.py --set strong --out $RLS_OUT/E50
"""
import argparse, json, os, sys
from dataclasses import replace
import numpy as np
from multiprocessing import Pool
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from rlscalar.search.core import Params, Belief, kappa
from rlscalar.search.episode import sample_task

ap = argparse.ArgumentParser()
ap.add_argument("--set", default="strong")
ap.add_argument("--policies", nargs="+", default=["m1", "m3", "chase"])
ap.add_argument("--root", default=os.environ.get("RLS_DATA", "data") + "/dns")
ap.add_argument("--fitdir", default=os.environ.get("RLS_DATA", "data") + "/dns/fits")
ap.add_argument("--e49", default=os.environ.get("RLS_OUT", "out") + "/E49")
ap.add_argument("--n", type=int, default=0, help="first n episodes only (test); 0 = all")
ap.add_argument("--workers", type=int, default=36)
ap.add_argument("--out", required=True)
a = ap.parse_args()
FILES = [f"{a.root}/{a.set}/seed{s:02d}.npz" for s in (6, 7, 8)]
ZI = ("zi_a0", "zi_a1", "zi_c0", "zi_c1", "leak_g", "leak_mu", "leak_s")
VAR = ["base", "k1", "k05", "k025", "nobg", "bgM3", "plumeM3", "nond", "M3"]
NS = (30, 80)


def m1_params():
    """As postnet_closedloop.py."""
    fit = json.load(open(f"{a.fitdir}/{a.set}_fit_taylor_src.json"))
    sur = fit["surrogate"]
    z2 = json.load(open(f"{a.fitdir}/{a.set}_lik_taylor_src.json"))["zi2"]
    return Params(U=sur["U"], K=sur["K"], sigma=z2["sigma"], tau=sur["tau"], L=sur["L"],
                  Q=float(np.median([q["mass_flux"] for q in fit["plumes"]])), T_L=sur["T_L"],
                  s0sq=sur.get("s0sq_taylor", 0.0), src_r=sur.get("src_r", 0.0), n_steps=80, snap_src=True,
                  temper=True, obs="zi", **{k: z2[k] for k in ZI})


def m3_params():
    """As postnet_closedloop.py."""
    base = f"{a.fitdir}/nn/{a.set}_N1000000_map_fix_traj"
    fit, z2 = json.load(open(f"{base}_fit.json")), json.load(open(f"{base}_lik.json"))["zi2"]
    sur = fit["surrogate"]
    pd = dict(U=sur["U"], K=sur["K"], tau=sur["tau"], L=sur["L"], Q=float(np.median([q["mass_flux"] for q in fit["plumes"]])),
              s0sq=max(0.0, float(np.median([q["s0sq"] for q in fit["plumes"]]))), n_steps=80, snap_src=True, temper=True)
    if sur.get("T_L"):
        pd.update(T_L=sur["T_L"], s0sq=sur.get("s0sq_taylor", 0.0))
    if sur.get("src_r"):
        pd.update(src_r=sur["src_r"])
    pd.update(obs="zi", sigma=z2["sigma"], **{k: z2[k] for k in ZI})
    return replace(Params(**pd, nn_file=f"{base}_hyb.npz"), mean_model="nn", obs="zi")


P1, P3 = m1_params(), m3_params()
Z = dict(np.load(f"{a.e49}/{a.set}_closedloop.npz"))       # eager: a lazy NpzFile breaks across forked workers
POLS = [str(p) for p in Z["policies"]]
EST = [str(e) for e in Z["estimators"]]


def scores(logw, j, cx, cy):
    w = np.exp(logw - np.logaddexp.reduce(logw))
    return np.array([w @ cx, w @ cy]), float(np.log(max(w[j], 1e-300))), float(w[w > w[j]].sum() < 0.9)


def episode(args):
    from search_sweep import dns_factory
    pol, seed = args
    if not hasattr(episode, "make"):
        episode.make = dns_factory(FILES)
    path = Z["path"][POLS.index(pol), seed]
    src, pos = sample_task(P1, seed)
    assert np.allclose(pos, path[0]) and np.allclose(src, Z["src"][seed]), (pol, seed)
    world = episode.make(P1, src, np.random.default_rng(10_000 + seed), seed)
    b1, b3 = Belief(P1, correlated=False), Belief(P3, correlated=False)
    cx, cy = b1.cx, b1.cy
    js = int(np.argmin((cx - src[0]) ** 2 + (cy - src[1]) ** 2))
    T = P1.n_steps
    logw = {v: b1.logw.copy() for v in VAR}
    est, ls, in90 = np.zeros((T + 1, len(VAR), 2)), np.zeros((T + 1, len(VAR))), np.zeros((T + 1, len(VAR)))
    LL, K1 = np.zeros((T + 1, cx.size)), np.zeros(T + 1)
    det, bg_true = np.zeros(T + 1, bool), np.zeros(T + 1, bool)
    t, prev = 0.0, None
    for k in range(T + 1):
        x, y = path[k]
        if k > 0:
            t += P1.dt
        z = world.measure(x, y, t)
        ll1, ll3 = b1.loglik(x, y, t, z), b3.loglik(x, y, t, z)
        bg = ~np.isfinite(b1.mu(x, y))                     # M1 uses the background-detection constants here
        k1, k3 = kappa(prev, (x, y), P1), kappa(prev, (x, y), P3)
        prev = (x, y)
        upd = dict(base=(ll1, k1), k1=(ll1, 1.0), k05=(ll1, 0.5 * k1), k025=(ll1, 0.25 * k1),
                   nobg=(np.where(bg, 0.0, ll1), k1), bgM3=(np.where(bg, ll3, ll1), k1),
                   plumeM3=(np.where(bg, ll1, ll3), k1), nond=(ll1 * (z is not None), k1), M3=(ll3, k3))
        for i, v in enumerate(VAR):
            ll, kk = upd[v]
            logw[v] = Belief.transition(logw[v], ll, kk)
            est[k, i], ls[k, i], in90[k, i] = scores(logw[v], js, cx, cy)
        LL[k], K1[k], det[k], bg_true[k] = ll1, k1, z is not None, bg[js]
    jhat = int(np.argmax(logw["base"]))
    contrib = K1 * (LL[:, js] - LL[:, jhat])               # > 0 favours the true cell over the final MAP cell
    off = path - src[None]                                 # sensor offset from the true source
    return pol, seed, est, ls, in90, contrib, det, bg_true, off, src


if __name__ == "__main__":
    os.makedirs(a.out, exist_ok=True)
    E = a.n or Z["src"].shape[0]
    jobs = [(p, s) for p in a.policies for s in range(E)]
    with Pool(a.workers, maxtasksperchild=20) as pool:
        res = pool.map(episode, jobs, chunksize=1)
    P, NV = len(a.policies), len(VAR)
    est, ls, in90 = np.zeros((P, E, 81, NV, 2)), np.zeros((P, E, 81, NV)), np.zeros((P, E, 81, NV))
    contrib, det, bgt, off = np.zeros((P, E, 81)), np.zeros((P, E, 81), bool), np.zeros((P, E, 81), bool), np.zeros((P, E, 81, 2))
    src = Z["src"][:E]
    for pol, s, e_, l_, i_, c_, d_, b_, o_, _ in res:
        pi = a.policies.index(pol)
        est[pi, s], ls[pi, s], in90[pi, s], contrib[pi, s], det[pi, s], bgt[pi, s], off[pi, s] = e_, l_, i_, c_, d_, b_, o_
    err = np.hypot(*(est - src[None, :, None, None, :]).transpose(4, 0, 1, 2, 3))
    np.savez(f"{a.out}/{a.set}_ablation.npz", err=err, logscore=ls, in90=in90, contrib=contrib, det=det, bg_true=bgt,
             off=off, src=src, policies=np.array(a.policies), variants=np.array(VAR))
    summ = dict(set=a.set, e49=a.e49, check={})
    for pi, pol in enumerate(a.policies):          # reproduction of the stored M1 / M3 errors along the same paths
        zi = POLS.index(pol)
        for v, e in (("base", "M1"), ("M3", "M3")):
            summ["check"][f"{pol}_{v}"] = float(np.abs(err[pi, :, :, VAR.index(v)] - Z["err"][zi, :E, :, EST.index(e)]).max())
        for vi, v in enumerate(VAR):
            for n in NS:
                summ[f"{pol}_{v}_n{n}"] = dict(P01=float(np.mean(err[pi, :, n, vi] < 0.1)), med=float(np.median(err[pi, :, n, vi])),
                                               logscore=float(ls[pi, :, n, vi].mean()), cov90=float(in90[pi, :, n, vi].mean()))
            r = summ[f"{pol}_{v}_n80"]
            print(f"{a.set} path {pol:6s} {v:8s}: n80 P01 {r['P01']:.2f} med {r['med']:.3f} logscore {r['logscore']:.2f} cov90 {r['cov90']:.2f}", flush=True)
        # where the evidence against the true cell comes from (base M1): sum of contributions by measurement class
        up = off[pi, ..., 0] <= 0
        cls = dict(det_down=det[pi] & ~up, det_up=det[pi] & up, nondet_down=~det[pi] & ~up, nondet_up=~det[pi] & up)
        summ[f"{pol}_contrib"] = {c: dict(mean_sum=float((contrib[pi] * m).sum(1).mean()), frac_steps=float(m.mean()))
                                  for c, m in cls.items()}
        summ[f"{pol}_contrib"]["total"] = float(contrib[pi].sum(1).mean())
        print(f"{a.set} path {pol:6s} contrib (true vs MAP):", json.dumps(summ[f"{pol}_contrib"]), flush=True)
    print("check (max |err - stored|):", summ["check"], flush=True)
    json.dump(summ, open(f"{a.out}/{a.set}_ablation.json", "w"), indent=1)
