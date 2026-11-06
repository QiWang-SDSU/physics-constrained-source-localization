"""Run one search episode. The evaluation belief is always the correctly correlated one.
The agent's own belief is either the same object (correlated agents) or an
independence-assuming belief (greedy_indep)."""
from dataclasses import replace
import numpy as np
from .core import Params, World, Belief, HitBelief, step_pose
from .policies import make_policy


def sample_task(p: Params, seed):
    """True source drawn from the belief's prior exactly: a uniform grid cell if p.snap_src,
    else uniform over the full prior box. Pre-E4b runs (E2, E3, E4) used a box shrunk by 0.1 on
    each side, a prior/truth mismatch that biases calibration while the posterior is broad."""
    rng = np.random.default_rng(seed)
    if p.snap_src:
        nx = int(round((p.xs_range[1] - p.xs_range[0]) / p.grid_h)) + 1
        ny = int(round((p.ys_range[1] - p.ys_range[0]) / p.grid_h)) + 1
        src = np.array([p.xs_range[0] + rng.integers(nx) * p.grid_h,
                        p.ys_range[0] + rng.integers(ny) * p.grid_h])
    else:
        src = np.array([rng.uniform(*p.xs_range), rng.uniform(*p.ys_range)])
    start = np.array([getattr(p, "start_x", 3.5), rng.uniform(-0.4, 0.4)])
    return src, start


def credible_level(b, src, rng=None):
    """Posterior mass of cells more probable than the truth's cell, plus a uniformly random
    share of the tied mass (randomised tie-breaking, so u ~ U(0,1) exactly under calibration even
    with flat, never-informed regions; E4 lesson). Uniform across episodes iff calibrated."""
    w = b.w()
    j = np.argmin((b.cx - src[0]) ** 2 + (b.cy - src[1]) ** 2)
    tie = np.isclose(w, w[j], rtol=1e-9, atol=0.0)
    r = (rng or np.random.default_rng(j)).uniform()
    return float(w[(w > w[j]) & ~tie].sum() + r * w[tie].sum())


def task_diagnostics(b, src):
    """MAP error; posterior mass of the true cell; its rank (cells strictly more probable); area of the
    smallest 90% credible set; whether the true cell lies inside that set."""
    w = b.w()
    j = int(np.argmin((b.cx - src[0]) ** 2 + (b.cy - src[1]) ** 2))
    jm = int(np.argmax(w))
    o = np.sort(w)[::-1]
    k90 = int(np.searchsorted(np.cumsum(o), 0.9)) + 1
    return (float(np.hypot(b.cx[jm] - src[0], b.cy[jm] - src[1])), float(w[j]), float(np.sum(w > w[j])),
            k90 * b.p.grid_h ** 2, float(w[j] >= o[k90 - 1]))


def run_episode(p: Params, policy_name, seed, record_path=False, world_factory=None, snap_steps=()):
    """world_factory(p, src, rng, seed) -> world with .measure(x, y, t). Default: model World (GP truth)."""
    src, pos = sample_task(p, seed)
    rng_world = np.random.default_rng(10_000 + seed)     # same noise stream across policies
    rng_pol = np.random.default_rng(20_000 + seed)
    world = World(p, src, rng_world) if world_factory is None else world_factory(p, src, rng_world, seed)
    b_eval = Belief(p, correlated=True)
    # agent belief from the name suffix: "_hits" -> classical binary-hit belief, "_indep" -> classical continuous
    # independent belief (both untempered, E21/E22); "greedy_indep" keeps its pre-E21 meaning (independent, p as is)
    if policy_name == "greedy_indep":
        b_agent = Belief(p, correlated=False)
    elif policy_name.endswith(("_nn", "_nnindep")):
        # E30 learned observation model (no physics); "_nn" keeps the tempered inference, "_nnindep" is untempered
        pn = replace(p, mean_model="nn", obs="zi")
        b_agent = Belief(pn if policy_name.endswith("_nn") else replace(pn, temper=False), correlated=False)
    elif policy_name.endswith(("_iso", "_isoindep", "_isohits")):
        # E27 no-physics baselines: isotropic mean field with its own zi2 fit. "_iso" keeps the tempered
        # inference of the physics belief (only the mean is swapped); "_isoindep" / "_isohits" are fully classical
        pi = replace(p, mean_model="iso", **p.iso_zi)
        if policy_name.endswith("_iso"):
            b_agent = Belief(pi, correlated=True)
        elif policy_name.endswith("_isoindep"):
            b_agent = Belief(replace(pi, temper=False), correlated=False)
        else:
            b_agent = HitBelief(pi)
    elif policy_name.endswith("_indep"):
        b_agent = Belief(replace(p, temper=False), correlated=False)
    elif policy_name.endswith("_hits"):
        b_agent = HitBelief(p)
    else:
        b_agent = b_eval
    pol = make_policy(policy_name)
    pol.reset()
    n = p.n_steps + 1
    H, H_agent, err, det, dist, u_e, u_a = (np.zeros(n) for _ in range(7))
    err_agent, in90_agent = np.zeros(n), np.zeros(n)          # E21: scored on the agent's own belief
    map_err, p_true, rank_true, a90, in90 = (np.zeros(n) for _ in range(5))   # E18b calibration diagnostics
    g_max, g_spread = np.full(n, np.nan), np.full(n, np.nan)
    acts, greedy_acts = np.full(n, -1), np.full(n, -1)
    g_all = np.full((n, 9), np.nan)
    pred1, predm = np.full(n, np.nan), np.full(n, np.nan)   # E16: predicted one-step / m-step entropy reduction
    extras = {}
    audit = None                                             # E17: MC one-step values on the same belief states
    path = np.zeros((n, 2))
    est = np.zeros((n, 2))                                   # posterior mean of the evaluation belief
    snaps = {}
    t = 0.0
    z = world.measure(*pos, t)
    for k in range(n):
        if k > 0:
            a = pol.act(dict(belief=b_agent, pos=pos, t=t, p=p, last_z=z, rng=rng_pol))
            if getattr(pol, "last_g", None) is not None:
                g_max[k], g_spread[k] = pol.last_g.max(), pol.last_g.max() - pol.last_g.min()
                g_all[k] = pol.last_g
                pred1[k] = pol.last_g[a]
            if getattr(pol, "last_pred", None) is not None:
                predm[k] = pol.last_pred
            for nm, v in (getattr(pol, "last_extra", None) or {}).items():   # E18: per-step policy extras
                extras.setdefault(nm, np.full((n,) + np.shape(v), np.nan))[k] = v
            if getattr(pol, "last_audit", None) is not None:
                if audit is None:
                    audit = np.full((n,) + pol.last_audit.shape, np.nan)
                audit[k] = pol.last_audit
            acts[k] = a
            greedy_acts[k] = getattr(pol, "last_greedy", a if pol.name == "greedy" else -1)
            pos, t = step_pose(pos, a, p), t + p.dt
            z = world.measure(*pos, t)
        b_eval.update(*pos, t, z)
        if b_agent is not b_eval:
            b_agent.update(*pos, t, z)
        H[k], H_agent[k] = b_eval.entropy(), b_agent.entropy()
        est[k] = b_eval.mean()
        err[k] = np.linalg.norm(est[k] - src)
        err_agent[k] = np.linalg.norm(b_agent.mean() - src)
        in90_agent[k] = task_diagnostics(b_agent, src)[4]
        det[k] = z is not None
        dist[k] = np.linalg.norm(pos - src)
        path[k] = pos
        u_e[k], u_a[k] = credible_level(b_eval, src), credible_level(b_agent, src)
        map_err[k], p_true[k], rank_true[k], a90[k], in90[k] = task_diagnostics(b_eval, src)
        if k in snap_steps:
            snaps[k] = b_eval.w().reshape(b_eval.shape).astype(np.float32)
    out = dict(H=H, H_agent=H_agent, err=err, det=det, dist=dist, src=src,
               u_eval=u_e, u_agent=u_a, eig_max=g_max, eig_spread=g_spread,
               actions=acts, greedy_actions=greedy_acts, eig_all=g_all,
               pred1=pred1, predm=predm, map_err=map_err, p_true=p_true, rank_true=rank_true,
               a90=a90, in90=in90, err_agent=err_agent, in90_agent=in90_agent, est=est, pos=path)
    if audit is not None:
        out["audit"] = audit
    for nm, v in extras.items():
        out["x_" + nm] = v
    if record_path:
        out["path"] = path
        out["w_final"] = b_eval.w().reshape(b_eval.shape)
        out["snap_steps"] = np.array(sorted(snaps))
        out["w_snaps"] = np.array([snaps[k] for k in sorted(snaps)]) if snaps else np.zeros((0,) + b_eval.shape)
        out["grid_x"] = np.unique(b_eval.cx); out["grid_y"] = np.unique(b_eval.cy)
    return out
