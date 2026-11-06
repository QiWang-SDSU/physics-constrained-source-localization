"""Track B experiment runner: policies x regimes x episodes, multiprocessing.

Usage: python scripts/search_sweep.py --tag E2 --taus 0.1 1 10 --policies greedy greedy_indep zigzag surgecast \
           --episodes 100 --steps 80 [--sigma 1.0 --L 0.4 --V 1.0 --kernel gauss]
Output: docs/research/data/<tag>_tau<tau>_<policy>.npz  (per-step arrays over episodes)
"""
import argparse, os, sys, time, json
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rlscalar.search.core import Params
from rlscalar.search.episode import run_episode


OUTDIR = "docs/research/data"


def ep_file(tag, tau, pol, seed):
    return f"{OUTDIR}/{tag}_parts/tau{tau:g}_{pol}_s{seed:04d}.npz"


_DNS = {}


def dns_factory(files):
    """World factory for DNS truth. Episode seed -> (test file, plume, start record), deterministic."""
    from rlscalar.search.dnsworld import DNSData, DNSWorld

    def make(p, src, rng, seed):
        f = files[seed % len(files)]
        if f not in _DNS:
            _DNS[f] = DNSData(f)
        d = _DNS[f]
        r = np.random.default_rng(50_000 + seed)
        plume = int(r.integers(len(d.P["sources"])))
        nrec = int(np.ceil((p.n_steps + 1) * p.dt / d.dt)) + 1
        t0 = int(r.integers(0, d.c.shape[0] - nrec))
        return DNSWorld(p, src, rng, d, plume, t0)
    return make


def job(args):
    """Runs one episode and saves it immediately (crash-safe; the sweep resumes by skipping
    existing files)."""
    pdict, pol, seed, tag, outdir = args
    global OUTDIR
    OUTDIR = outdir
    pdict = dict(pdict)
    files = pdict.pop("_dns_files", None)
    true_over = pdict.pop("_true", None)
    ziw = pdict.pop("_ziworld", None)
    label = pdict.pop("_label_tau", pdict["tau"])
    wf = dns_factory(files) if files else None
    if true_over:                      # controlled misspecification: truth params differ from the belief's
        from rlscalar.search.core import World
        pt = Params(**{**pdict, **true_over})
        wf = lambda p, src, rng, sd: World(pt, src, rng)
    if ziw is not None:                # E12: intermittent truth, switch correlated (1) or i.i.d. (0)
        from rlscalar.search.core import ZIWorld
        wf = lambda p, src, rng, sd: ZIWorld(p, src, rng, corr_switch=bool(ziw))
    out = run_episode(Params(**pdict), pol, seed, world_factory=wf)
    fn = ep_file(tag, label, pol, seed)
    np.savez(fn + ".tmp.npz", **out)
    os.replace(fn + ".tmp.npz", fn)
    return pol, label, seed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--taus", type=float, nargs="+", default=[1.0])
    ap.add_argument("--policies", nargs="+", default=["greedy", "greedy_indep", "zigzag", "surgecast"])
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--L", type=float, default=0.4)
    ap.add_argument("--V", type=float, default=1.0)
    ap.add_argument("--K", type=float, default=0.03)
    ap.add_argument("--start_x", type=float, default=3.5)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--kernel", default="gauss")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--outdir", default="docs/research/data")
    ap.add_argument("--snap", action="store_true")
    ap.add_argument("--no_ep", action="store_true", help="disable EP pseudo-observations for censored readings (pre-E4 behaviour)")
    ap.add_argument("--m_min", type=float, default=1e-2)
    ap.add_argument("--dns_fit", default=None, help="JSON from dns2_fit_and_law.py: surrogate U,K,sigma,tau,L (+Q,s0sq)")
    ap.add_argument("--dns_files", nargs="+", default=None, help="held-out DNS2 seed files used as truth")
    ap.add_argument("--zi_from", default=None, help="E12: take zi2 observation params from this *_lik.json (model world)")
    ap.add_argument("--ziworld", type=int, default=None, help="E12: truth = ZIWorld with correlated (1) or iid (0) switch")
    ap.add_argument("--true_sigma", type=float, default=None, help="truth sigma (belief uses --sigma)")
    ap.add_argument("--true_tau", type=float, default=None, help="truth tau (belief uses --taus)")
    ap.add_argument("--temper", action="store_true", help="effective-sample-size tempered composite likelihood (E8d)")
    ap.add_argument("--temper_const", action="store_true", help="E22 ablation: constant kappa (stationary-sensor rate), no motion dependence")
    ap.add_argument("--dns_lik", default=None, help="JSON from dns2_likelihood_calibration.py: use the fitted zi2 observation model")
    ap.add_argument("--nn_file", default=None, help="E30: learned observation model (npz from dns2_nn_likelihood.py)")
    ap.add_argument("--iso_lik", default=None, help="E27: JSON from dns2_likelihood_calibration.py --mean iso (no-physics baseline belief)")
    a = ap.parse_args()
    OUTDIR = a.outdir
    jobs = []
    for tau in a.taus:
        pd = dict(tau=tau, sigma=a.sigma, L=a.L, V=a.V, K=a.K, start_x=a.start_x, dt=a.dt, kernel=a.kernel, n_steps=a.steps,
                  snap_src=a.snap, m_min=a.m_min, censor_ep=not a.no_ep, temper=a.temper, temper_const=a.temper_const)
        tr = {k: v for k, v in (("sigma", a.true_sigma), ("tau", a.true_tau)) if v is not None}
        if tr:
            pd["_true"] = tr
        if a.zi_from:
            z2 = json.load(open(a.zi_from))["zi2"]
            pd.update(obs="zi", sigma=z2["sigma"], **{k: z2[k] for k in
                      ("zi_a0", "zi_a1", "zi_c0", "zi_c1", "leak_g", "leak_mu", "leak_s")})
        if a.ziworld is not None:
            pd["_ziworld"] = a.ziworld
        if a.dns_fit:
            fit = json.load(open(a.dns_fit))
            sur = fit["surrogate"]
            pd.update(U=sur["U"], K=sur["K"], sigma=sur["sigma"], tau=sur["tau"], L=sur["L"],
                      Q=float(np.median([q["mass_flux"] for q in fit["plumes"]])),
                      s0sq=max(0.0, float(np.median([q["s0sq"] for q in fit["plumes"]]))),  # fitted s0^2 < 0 (near-source ballistic growth): clamp
                      _dns_files=a.dns_files, _label_tau=tau)
            if sur.get("T_L"):
                pd.update(T_L=sur["T_L"], s0sq=sur.get("s0sq_taylor", 0.0))   # E24: Taylor width law + source-size variance
            if sur.get("src_r"):
                pd.update(src_r=sur["src_r"])                             # E26: finite (Gaussian) source
        if a.dns_lik:
            z2 = json.load(open(a.dns_lik))["zi2"]
            pd.update(obs="zi", sigma=z2["sigma"], **{k: z2[k] for k in
                      ("zi_a0", "zi_a1", "zi_c0", "zi_c1", "leak_g", "leak_mu", "leak_s")})
        if a.nn_file:
            pd["nn_file"] = a.nn_file
        if a.iso_lik:
            iz = json.load(open(a.iso_lik))
            pd.update(iso_l=iz["iso_l"], iso_zi={k: iz["zi2"][k] for k in
                      ("sigma", "zi_a0", "zi_a1", "zi_c0", "zi_c1", "leak_g", "leak_mu", "leak_s")})
        for pol in a.policies:
            jobs += [(pd, pol, a.seed0 + e, a.tag, a.outdir) for e in range(a.episodes)
                     if not os.path.exists(ep_file(a.tag, tau, pol, a.seed0 + e))]
    os.makedirs(f"{a.outdir}/{a.tag}_parts", exist_ok=True)
    t0 = time.time()
    print(f"{len(jobs)} episodes to run", flush=True)
    done = 0
    with Pool(a.workers, maxtasksperchild=20) as pool:
        for pol, tau, seed in pool.imap_unordered(job, jobs, chunksize=1):
            done += 1
            if done % 50 == 0:
                print(f"{done}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    # collate everything on disk (including episodes from earlier, interrupted runs)
    for tau in a.taus:
        for pol in a.policies:
            seeds = [a.seed0 + e for e in range(a.episodes)
                     if os.path.exists(ep_file(a.tag, tau, pol, a.seed0 + e))]
            if not seeds:
                continue
            outs = [dict(np.load(ep_file(a.tag, tau, pol, sd))) for sd in seeds]
            np.savez(f"{a.outdir}/{a.tag}_tau{tau:g}_{pol}.npz", seeds=seeds,
                     args=json.dumps(vars(a)), **{k: np.array([o[k] for o in outs]) for k in outs[0]})
    print(f"done in {time.time()-t0:.0f}s", flush=True)
