"""Closed-loop localisation with the learned grid posterior B (E38; spec docs/research/spec_posterior_networks.md, section 5.2).

On the 100 held-out DNS tasks of E26 / E36 (episode seeds 0..99, test seeds 6-8, snapped source, 80 moves) three planners
choose the moves by one-step EIG:
  m1    on the physics belief M1 (E26: adjoint mean field + zi2 + kappa tempering),
  m3    on the hybrid belief M3 (E36, N = 1e6 path readings),
  m3qB  with M3's predictive observation model but B's posterior as the weights ("plan with M3, update with B"),
  chase / surgecast  model-free baselines (E42): plume-centre follower and moth-like surge-cast (policies.py).
Along every path four posteriors are scored after each reading: M1, M3, B and B0 (both trained on 9500 episodes, E37).
B / B0 are evaluated incrementally (one GRU step and one pairwise term per reading); the script first checks this against
the batch forward pass on stored test histories.
E48 (velocity sensing): --runs_vel runs_vel also scores Bvel / B0vel (sensor reads (u - U, v) from the E44 v2 velocity
replays, DNSWorld.velocity), and planner m3qBvel uses Bvel's weights; --runs selects the no-velocity models (runs_fp32).
With --save_q also {out}/{set}_closedloop_q.npz (q (P, E, 3, 4, J) float16 at n = 20, 40, 80; E40 maps).
Output: {out}/{set}_closedloop.npz (est (P, E, 81, 4, 2), logscore / in90 (P, E, 81, 4)) and {set}_closedloop.json.
  python scripts/postnet_closedloop.py --set strong --out $RLS_OUT
"""
import argparse, json, os, sys
from dataclasses import replace
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rlscalar.search.core import Params, Belief, step_pose
from rlscalar.search.episode import sample_task
from rlscalar.search.policies import Greedy, Chase, SurgeCast

ap = argparse.ArgumentParser()
ap.add_argument("--set", default="strong")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--seed0", type=int, default=0, help="first task seed (E57 fresh tasks: 1000; offline test tasks are 0-299)")
ap.add_argument("--policies", nargs="+", default=["m1", "m3", "m3qB"])
ap.add_argument("--root", default=os.environ.get("RLS_DATA", "data") + "/dns")
ap.add_argument("--fitdir", default=os.environ.get("RLS_DATA", "data") + "/dns/fits")
ap.add_argument("--pn", default=os.environ.get("RLS_DATA", "data") + "/postnet")
ap.add_argument("--n_train", type=int, default=9500)
ap.add_argument("--runs", default="runs", help="run folder of B / B0 (E48: runs_fp32)")
ap.add_argument("--runs_vel", default="", help="also score the velocity-sensing models from this run folder (E48)")
ap.add_argument("--vel_dir", default=os.environ.get("RLS_DATA", "data") + "/dns/vel")
ap.add_argument("--workers", type=int, default=36)
ap.add_argument("--out", required=True)
ap.add_argument("--save_q", action="store_true", help="also store the full posteriors at n = --q_ns (float16)")
ap.add_argument("--q_ns", type=int, nargs="+", default=[20, 40, 80])
ap.add_argument("--extra", nargs="*", default=[],
                help="E54: more learned posteriors to score, name:model:runs:kind[:rep] with kind full | r0 | nomem "
                     "(r0: fixed zero projection {pn}/sens/{set}_{model}_proj_r0.npy; nomem: GRU state fixed at zero); "
                     "rep = training seed (default 0)")
a = ap.parse_args()
FILES = [f"{a.root}/{a.set}/seed{s:02d}.npz" for s in (6, 7, 8)]
NETS = [] if a.runs == "none" else [("b", "b", a.runs, False), ("b0", "b0", a.runs, False)]   # E57: --runs none = extras only
if a.runs_vel:
    NETS += [("bvel", "b", a.runs_vel, True), ("b0vel", "b0", a.runs_vel, True)]
KIND, REP = {}, {}
for x in a.extra:
    nm, m, r, kind, *rep = x.split(":")
    NETS.append((nm, m, r, False))
    KIND[nm], REP[nm] = kind, int(rep[0]) if rep else 0
EST = ["M1", "M3"] + [n[0].upper().replace("VEL", "vel") for n in NETS]
VSCALE = {"strong": 0.18, "weak": 0.09}                   # as in postnet_train.py
QN = a.q_ns
ZI = ("zi_a0", "zi_a1", "zi_c0", "zi_c1", "leak_g", "leak_mu", "leak_s")


def m1_params():
    """M1 exactly as postnet_data.py builds it (the input of B)."""
    fit = json.load(open(f"{a.fitdir}/{a.set}_fit_taylor_src.json"))
    sur = fit["surrogate"]
    z2 = json.load(open(f"{a.fitdir}/{a.set}_lik_taylor_src.json"))["zi2"]
    return Params(U=sur["U"], K=sur["K"], sigma=z2["sigma"], tau=sur["tau"], L=sur["L"],
                  Q=float(np.median([q["mass_flux"] for q in fit["plumes"]])), T_L=sur["T_L"],
                  s0sq=sur.get("s0sq_taylor", 0.0), src_r=sur.get("src_r", 0.0), n_steps=80, snap_src=True,
                  temper=True, obs="zi", **{k: z2[k] for k in ZI})


def m3_params():
    """M3 exactly as search_sweep.py builds the E36h1000000 agent (greedy_nn): --dns_fit/--dns_lik/--temper/--snap/--nn_file."""
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


class IncNet:
    """B / B0 of rlscalar.search.postnet evaluated one reading at a time: l_jk needs only g_{k-1} and reading k."""

    def __init__(self, model, runs="runs", vel=False, kind="full", rep=0):
        import torch
        from rlscalar.search.postnet import PostNet
        self.torch = torch
        b = Belief(P1)
        proj = np.load(f"{a.pn}/sens/{a.set}_{model}_proj_r0.npy") if kind == "r0" else None
        self.net = PostNet(b.cx, b.cy, physics=(model == "b"), vel=vel, proj=proj, nomem=(kind == "nomem"))
        self.net.load_state_dict(torch.load(f"{a.pn}/{runs}/{a.set}_{model}_n{a.n_train}_r{rep}.pt", map_location="cpu"))
        self.net.eval()
        self.reset()

    def reset(self):
        H = self.net.gru.hidden_size
        self.h = self.torch.zeros(1, 1, H)
        self.S = self.torch.zeros(self.net.cells.shape[0])

    def step(self, tok, pos):
        t, n = self.torch, self.net
        with t.no_grad():
            tok, pos = t.as_tensor(tok, dtype=t.float32), t.as_tensor(pos, dtype=t.float32)
            rd = n.f_rd(t.cat([tok[n.rd_cols], self.h[0, 0]]))
            self.S += n.f_rest(n.f_off((pos[None] - n.cells) / n.scale) + rd)[:, 0]
            if not n.nomem:                                       # E54: nomem keeps h = 0
                x = tok[None, None]
                _, self.h = n.gru(n.enc(x if n.proj is None else n.proj(x)), self.h)

    def logq(self, lphys):
        t = self.torch
        with t.no_grad():
            lg = self.S if not self.net.physics else t.as_tensor(lphys.astype(np.float16).astype(np.float32)) + self.net.alpha * self.S
            return t.log_softmax(lg.double(), -1).numpy()


def check_incremental():
    """Incremental vs batch forward on stored test histories (max |log q| difference at n = 80)."""
    import glob, torch
    fn = sorted(glob.glob(f"{a.pn}/data/{a.set}_test_[0-9][0-9][0-9].npz"))[0]
    z = np.load(fn)
    out = {}
    for name, m, runs, vel in NETS:
        inc = IncNet(m, runs, vel, KIND.get(name, "full"), REP.get(name, 0))
        tk = z["tok"]
        if vel:
            tk = np.concatenate([tk, np.load(fn.replace(".npz", "_vel.npz"))["uv"] / VSCALE[a.set]], -1).astype(np.float32)
        dif = []
        for e in range(3):
            inc.reset()
            for k in range(81):
                inc.step(tk[e, k], z["pos"][e, k])
            lp = z["lpost"][e, -1].astype(np.float32)
            with torch.no_grad():
                ref = inc.net(torch.tensor(tk[e:e + 1]), torch.tensor(z["pos"][e:e + 1]), [80],
                              torch.tensor(lp[None, None]))[0, 0].double().numpy()
            dif.append(np.abs(inc.logq(z["lpost"][e, -1]) - ref).max())
        out[name] = float(max(dif))
    return out


def scores(logw, src, cx, cy):
    w = np.exp(logw - np.logaddexp.reduce(logw))
    j = int(np.argmin((cx - src[0]) ** 2 + (cy - src[1]) ** 2))
    return np.array([w @ cx, w @ cy]), float(np.log(max(w[j], 1e-300))), float(w[w > w[j]].sum() < 0.9)


def episode(args):
    import torch
    torch.set_num_threads(1)
    from search_sweep import dns_factory
    pol, seed = args
    if not hasattr(episode, "make"):
        episode.make, episode.nets = dns_factory(FILES), {nm: IncNet(m, r, v, KIND.get(nm, "full"), REP.get(nm, 0))
                                                         for nm, m, r, v in NETS}
    nets = episode.nets
    for n in nets.values():
        n.reset()
    src, pos = sample_task(P1, seed)
    world = episode.make(P1, src, np.random.default_rng(10_000 + seed), seed)
    if a.runs_vel and not hasattr(world.d, "u"):
        world.d.add_velocity(f"{a.vel_dir}/{a.set}/" + os.path.basename(FILES[seed % len(FILES)]).replace(".npz", "_vel.npz"))
    b1, b3 = Belief(P1, correlated=True), Belief(P3, correlated=False)
    planner = {"m1": b1, "m3": b3, "m3qB": b3, "m3qBvel": b3}.get(pol)
    greedy = Greedy()
    free = {"chase": Chase, "surgecast": SurgeCast}.get(pol)       # model-free baselines (E42): no belief
    if free:
        free = free()
        free.reset()
    T, thr = P1.n_steps, np.log(P1.m_min)
    NE = len(EST)
    est, ls, in90 = np.zeros((T + 1, NE, 2)), np.zeros((T + 1, NE)), np.zeros((T + 1, NE))
    path = np.zeros((T + 1, 2))
    qs = np.zeros((len(QN), NE, b1.cx.size), np.float16) if a.save_q else None
    t, z, prev, lq = 0.0, world.measure(*pos, 0.0), pos.copy(), None
    for k in range(T + 1):
        if k > 0:
            if pol in ("m3qB", "m3qBvel"):             # M3's predictive, B's (Bvel's) weights
                keep, b3.logw = b3.logw, lq["b" if pol == "m3qB" else "bvel"]
                act = greedy.act(dict(belief=b3, pos=pos, t=t, p=P3))
                b3.logw = keep
            elif free:
                act = free.act(dict(pos=pos, t=t, p=P1, last_z=z))
            else:
                act = greedy.act(dict(belief=planner, pos=pos, t=t, p=planner.p))
            prev, pos, t = pos, step_pose(pos, act, P1), t + P1.dt
            z = world.measure(*pos, t)
        b1.update(*pos, t, z)
        b3.update(*pos, t, z)
        d = pos - prev
        tok = [0.0 if z is None else 1.0, 0.0 if z is None else (z - thr) / 5, pos[0] / 4, pos[1], k / T,
               d[0] / (P1.V * P1.dt), d[1] / (P1.V * P1.dt)]
        tokv = np.concatenate([tok, world.velocity(*pos, t) / VSCALE[a.set]]) if a.runs_vel else None
        for n in nets.values():
            n.step(np.array(tokv if n.net.rd_cols[-1] > 1 else tok, np.float32), np.array(pos, np.float32))
        lq = {m: n.logq(b1.logw) for m, n in nets.items()}
        path[k] = pos
        for i, lw in enumerate([b1.logw, b3.logw] + [lq[nm] for nm, *_ in NETS]):
            est[k, i], ls[k, i], in90[k, i] = scores(lw, src, b1.cx, b1.cy)
            if a.save_q and k in QN:
                qs[QN.index(k), i] = np.exp(lw - np.logaddexp.reduce(lw))
    return pol, seed, est, ls, in90, path, src, qs


if __name__ == "__main__":
    os.makedirs(a.out, exist_ok=True)
    chk = check_incremental()
    print("incremental vs batch max |dlog q|:", chk, flush=True)
    assert max(chk.values()) < 5e-3, chk
    jobs = [(p, a.seed0 + s) for p in a.policies for s in range(a.n)]
    with Pool(a.workers, maxtasksperchild=20) as pool:
        res = pool.map(episode, jobs, chunksize=1)
    P, E = len(a.policies), a.n
    NE = len(EST)
    est, ls, in90 = np.zeros((P, E, 81, NE, 2)), np.zeros((P, E, 81, NE)), np.zeros((P, E, 81, NE))
    path, src = np.zeros((P, E, 81, 2)), np.zeros((E, 2))
    q = np.zeros((P, E, len(QN), NE, res[0][-1].shape[-1]), np.float16) if a.save_q else None
    for pol, s, e_, l_, i_, pa, sr, q_ in res:
        pi, s = a.policies.index(pol), s - a.seed0
        est[pi, s], ls[pi, s], in90[pi, s], path[pi, s], src[s] = e_, l_, i_, pa, sr
        if a.save_q:
            q[pi, s] = q_
    err = np.hypot(*(est - src[None, :, None, None, :]).transpose(4, 0, 1, 2, 3))
    np.savez(f"{a.out}/{a.set}_closedloop.npz", est=est, logscore=ls, in90=in90, path=path, src=src, err=err,
             policies=np.array(a.policies), estimators=np.array(EST))
    if a.save_q:
        b = Belief(P1)
        np.savez_compressed(f"{a.out}/{a.set}_closedloop_q.npz", q=q, q_ns=QN, path=path, src=src, err=err,
                            grid_x=np.unique(b.cx), grid_y=np.unique(b.cy), policies=np.array(a.policies),
                            estimators=np.array(EST))
    summ = dict(set=a.set, check=chk, n=a.n, seed0=a.seed0, n_train=a.n_train, runs=a.runs, runs_vel=a.runs_vel)
    for pi, pol in enumerate(a.policies):
        for ei, nm in enumerate(EST):
            for n in (20, 40, 80):
                summ[f"{pol}_{nm}_n{n}"] = dict(P01=float(np.mean(err[pi, :, n, ei] < 0.1)), med=float(np.median(err[pi, :, n, ei])),
                                                logscore=float(ls[pi, :, n, ei].mean()), cov90=float(in90[pi, :, n, ei].mean()))
            r = summ[f"{pol}_{nm}_n80"]
            print(f"{a.set} plan {pol:7s} est {nm:5s}: n80 P01 {r['P01']:.2f} med {r['med']:.3f} logscore {r['logscore']:.2f} cov90 {r['cov90']:.2f}", flush=True)
    json.dump(summ, open(f"{a.out}/{a.set}_closedloop.json", "w"), indent=1)
