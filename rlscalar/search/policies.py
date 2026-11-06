"""Search policies. Each is a callable class: act(state) -> action index into ACTIONS.

state: dict with keys belief (the belief the AGENT uses), pos, t, p, last_z, rng.
"""
import numpy as np
from .core import ACTIONS, step_pose, Belief

UPWIND = 4          # ACTIONS[4] = (-1, 0)


class Greedy:
    """One-step EIG maximiser (infotaxis with the continuous censored likelihood).
    Uses whatever belief it is given: correlated -> 'greedy', independent -> 'greedy_indep'."""
    name = "greedy"

    def __init__(self, tie_eps=1e-6):
        self.tie_eps = tie_eps

    def reset(self):
        pass

    def act(self, s):
        b, pos, t, p = s["belief"], s["pos"], s["t"], s["p"]
        idx = b.active()
        g = np.array([b.eig(*step_pose(pos, a, p), t + p.dt, idx) for a in range(len(ACTIONS))])
        self.last_g = g                    # recorded by run_episode (E13c mechanism diagnostics)
        if g.max() < self.tie_eps:
            return UPWIND
        return int(np.argmax(g))


class Seek:
    """E28 source seeking with a belief: one-step EIG while the posterior is broad, then move to the posterior
    mean once its standard deviation sqrt(var_x + var_y) falls below `commit` (the belief is chosen by the
    policy-name suffix in run_episode, as for the infotaxis baselines)."""
    name = "seek"

    def __init__(self, commit=0.1):
        self.commit, self.base = commit, Greedy()

    def reset(self):
        pass

    def act(self, s):
        b, pos, p = s["belief"], s["pos"], s["p"]
        w, m = b.w(), b.mean()
        sd = np.sqrt(w @ (b.cx - m[0]) ** 2 + w @ (b.cy - m[1]) ** 2)
        self.last_extra = dict(sd=sd)
        if sd < self.commit:
            self.last_g = None
            return int(np.argmin([np.hypot(*(step_pose(pos, a, p) - m)) for a in range(len(ACTIONS))]))
        a = self.base.act(s)
        self.last_g = self.base.last_g
        return a


class Zigzag:
    """Open-loop survey: diagonal upwind sweeps between y = +-A."""
    name = "zigzag"

    def __init__(self, A=0.6):
        self.A = A

    def reset(self):
        self.up = True

    def act(self, s):
        y = s["pos"][1]
        if y > self.A:
            self.up = False
        elif y < -self.A:
            self.up = True
        return 3 if self.up else 5          # (-.71, +.71) or (-.71, -.71)


class SurgeCast:
    """Model-free moth-like heuristic: surge upwind on detection, otherwise cast crosswind
    with growing amplitude (Kennedy & Marsh-type baseline)."""
    name = "surgecast"

    def reset(self):
        self.k, self.leg, self.dir = 0, 2, 1

    def act(self, s):
        if s["last_z"] is not None:
            self.k, self.leg = 0, 2
            return UPWIND
        self.k += 1
        if self.k >= self.leg:
            self.k, self.leg, self.dir = 0, self.leg + 2, -self.dir
        return 2 if self.dir > 0 else 6


class Chase:
    """Model-free plume-centre follower (chemotaxis with upwind bias, E42): on a detection, fit the crosswind
    gradient of log c over the last `k` detections and move diagonally upwind toward higher concentration
    (straight upwind when the gradient is unresolved, i.e. on the centreline); when the plume is lost, cast
    crosswind with growing legs, starting toward the side of the last increase."""
    name = "chase"

    def __init__(self, k=6, gmin=0.5):
        self.k, self.gmin = k, gmin

    def reset(self):
        self.hist, self.dir, self.n, self.leg = [], 1, 0, 2

    def act(self, s):
        z, pos = s["last_z"], s["pos"]
        if z is None:
            self.n += 1
            if self.n >= self.leg:
                self.n, self.leg, self.dir = 0, self.leg + 2, -self.dir
            return 2 if self.dir > 0 else 6
        self.n, self.leg = 0, 2
        self.hist = (self.hist + [(pos[1], z)])[-self.k:]
        y, c = np.array(self.hist).T
        gy = np.polyfit(y, c, 1)[0] if len(y) >= 3 and np.ptp(y) > 1e-6 else 0.0
        if abs(gy) < self.gmin:
            return UPWIND
        self.dir = 1 if gy > 0 else -1
        return 3 if gy > 0 else 5


class Rollout:
    """Non-myopic one-step lookahead with greedy base-policy rollouts (Bertsekas-style).
    Score(a) = E[ H(p_{n+1+H}) ] estimated by M sampled futures; hypothesised truth drawn from
    the belief, observations drawn from the belief's own predictive (so the planner is
    self-consistent with its model). Chooses argmin expected terminal entropy."""
    name = "rollout"

    def __init__(self, M=6, H=4, cand_actions=None):
        self.M, self.H = M, H
        self.base = Greedy()

    def reset(self):
        pass

    def _sim_obs(self, b, x, y, t, j, rng):
        return b.sample_obs(x, y, t, j, rng)

    def act(self, s):
        import copy
        b, pos, t, p, rng = s["belief"], s["pos"], s["t"], s["p"], s["rng"]
        # prune to the greedy top-4 actions to control cost
        idx = b.active()
        g = np.array([b.eig(*step_pose(pos, a, p), t + p.dt, idx) for a in range(len(ACTIONS))])
        if g.max() < 1e-6:
            return UPWIND
        cands = np.argsort(g)[::-1][:4]
        scores = []
        w = b.w()
        for a in cands:
            tot = 0.0
            for m in range(self.M):
                bb = copy.deepcopy(b)
                j = rng.choice(w.size, p=w)
                ps, tt = step_pose(pos, a, p), t + p.dt
                bb.update(*ps, tt, self._sim_obs(bb, *ps, tt, j, rng))
                for h in range(self.H):
                    a2 = self.base.act(dict(belief=bb, pos=ps, t=tt, p=p))
                    ps, tt = step_pose(ps, a2, p), tt + p.dt
                    bb.update(*ps, tt, self._sim_obs(bb, *ps, tt, j, rng))
                tot += bb.entropy()
            scores.append(tot / self.M)
        return int(cands[int(np.argmin(scores))])


class Macro:
    """Non-myopic receding-horizon planner over straight-line macro-actions.
    For each heading a, simulate M futures in which the sensor moves along a for m steps.
    The hypothesised truth j is drawn from the belief, and observations are drawn from the
    belief's own correlated predictive (so the planner is consistent with its model).
    Score = expected entropy after m steps. Execute the first step of the best heading.
    Rationale (theory_notes T1): under strong correlation, information comes from moving
    relative to the flow over several steps, which a one-step EIG cannot see."""
    name = "macro"

    def __init__(self, m=10, M=8):
        self.m, self.M = m, M

    def reset(self):
        pass

    def act(self, s):
        import copy
        b, pos, t, p, rng = s["belief"], s["pos"], s["t"], s["p"], s["rng"]
        w = b.w()
        js = rng.choice(w.size, size=self.M, p=w)
        # E13d diagnostic: greedy's one-step choice on the same belief
        idx = b.active()
        g = np.array([b.eig(*step_pose(pos, a, p), t + p.dt, idx) for a in range(len(ACTIONS))])
        self.last_greedy = int(np.argmax(g)) if g.max() >= 1e-6 else UPWIND
        self.last_g = g
        scores = np.zeros(8)
        for a in range(8):
            for j in js:
                bb = copy.deepcopy(b)
                ps, tt = pos, t
                for h in range(self.m):
                    ps, tt = step_pose(ps, a, p), tt + p.dt
                    bb.update(*ps, tt, bb.sample_obs(*ps, tt, j, rng))
                scores[a] += bb.entropy() / self.M
        a_best = int(np.argmin(scores))
        # E16 diagnostic: the planner's own predicted m-step entropy reduction for the chosen heading
        self.last_pred = float(b.entropy() - scores[a_best])
        return a_best


class GreedyMasked(Greedy):
    """E14a causal test: greedy restricted to non-downwind actions (excludes +x, +x+y, +x-y)."""
    name = "greedy_nodown"
    ALLOWED = [2, 3, 4, 5, 6, 8]

    def act(self, s):
        b, pos, t, p = s["belief"], s["pos"], s["t"], s["p"]
        idx = b.active()
        g = np.full(len(ACTIONS), -np.inf)
        for a in self.ALLOWED:
            g[a] = b.eig(*step_pose(pos, a, p), t + p.dt, idx)
        self.last_g = np.where(np.isfinite(g), g, np.nan)
        if np.nanmax(self.last_g) < self.tie_eps:
            return UPWIND
        return int(np.nanargmax(self.last_g))


class GreedyNoStay(GreedyMasked):
    """E17 control: exact greedy over the 8 headings only (the planner's action set; no 'stay')."""
    name = "greedy_nostay"
    ALLOWED = list(range(8))


class GreedyAudit(Greedy):
    """E17 estimator audit: acts exactly like greedy, and on the same belief state also computes the
    Monte-Carlo one-step value used by Macro(m=1, M): I_M(a) = H[p] - mean_j H[p | simulated z_a^(j)],
    for each M in Ms and the 8 headings. Stored in last_audit, shape (len(Ms), 8)."""
    name = "greedy_audit"
    Ms = (8, 32, 128)

    def act(self, s):
        a = super().act(s)
        self.last_audit = mc_one_step(s, self.Ms)
        return a


def mc_one_step(s, Ms):
    """Macro(m=1, M) Monte-Carlo one-step values on belief s['belief'] for the 8 headings and each M:
    H[p] - mean_i H[p after update(simulated z_i)], hypotheses j_i ~ p, z_i ~ p(z | x_j, a)."""
    import copy
    b, pos, t, p, rng = s["belief"], s["pos"], s["t"], s["p"], s["rng"]
    w, H0 = b.w(), b.entropy()
    out = np.zeros((len(Ms), 8))
    for i, M in enumerate(Ms):
        js = rng.choice(w.size, size=M, p=w)
        for h in range(8):
            ps = step_pose(pos, h, p)
            acc = 0.0
            for j in js:
                bb = copy.deepcopy(b)
                bb.update(*ps, t + p.dt, bb.sample_obs(*ps, t + p.dt, j, rng))
                acc += bb.entropy()
            out[i, h] = H0 - acc / M
    return out


class GreedyV(Greedy):
    """E18 consistent greedy: argmax_a V_k(a) = H[p] - E_{z~p(z|a)} H[T_k(p, z, a)], the expected entropy
    reduction of the tempered transition the agent actually applies (Belief.vkappa). Identical to greedy
    when k = 1. Under tempering, plain 'greedy' maximises k * I instead (alias 'scaled_mi_greedy')."""
    name = "greedy_v"

    def act(self, s):
        b, pos, t, p = s["belief"], s["pos"], s["t"], s["p"]
        idx = b.active()
        g = np.array([b.vkappa(*step_pose(pos, a, p), t + p.dt, idx) for a in range(len(ACTIONS))])
        self.last_g = g
        if g.max() < self.tie_eps:
            return UPWIND
        return int(np.argmax(g))


class GreedyVAudit(GreedyV):
    """E18 (ii): consistent greedy, plus the Macro(m=1, M) Monte-Carlo values on the same belief states."""
    name = "greedy_v_audit"
    Ms = (8, 32, 128, 512)
    ref_every = 4        # 16000-point reference V_k on every 4th step (quadrature-error control)

    def act(self, s):
        a = super().act(s)
        self.last_audit = mc_one_step(s, self.Ms)
        b, pos, t, p = s["belief"], s["pos"], s["t"], s["p"]
        idx = b.active()
        k = int(round(t / p.dt))
        ki = np.array([b.eig(*step_pose(pos, h, p), t + p.dt, idx) for h in range(len(ACTIONS))])   # kappa * I
        ref = np.full(8, np.nan)
        if k % self.ref_every == 0:
            ref = np.array([b.vkappa(*step_pose(pos, h, p), t + p.dt, idx, nq=16000, nmax=16000) for h in range(8)])
        self.last_extra = dict(ki=ki, vref16=ref)
        return a


class Tree2:
    """E14b: two-step lookahead infotaxis. score(a) = EIG(a) + E_{z1}[max_{a2} EIG(a2 | z1)],
    with z1 sampled M times from the belief's predictive (source hypothesis drawn from the belief)."""
    name = "tree2"

    def __init__(self, M=6):
        self.M = M

    def reset(self):
        pass

    def act(self, s):
        import copy
        b, pos, t, p, rng = s["belief"], s["pos"], s["t"], s["p"], s["rng"]
        idx = b.active()
        w = b.w()
        score = np.zeros(len(ACTIONS))
        for a in range(len(ACTIONS)):
            p1, t1 = step_pose(pos, a, p), t + p.dt
            score[a] = b.eig(*p1, t1, idx)
            fut = 0.0
            for j in rng.choice(w.size, size=self.M, p=w):
                bb = copy.deepcopy(b)
                bb.update(*p1, t1, bb.sample_obs(*p1, t1, j, rng))
                i2 = bb.active()
                fut += max(bb.eig(*step_pose(p1, a2, p), t1 + p.dt, i2) for a2 in range(len(ACTIONS)))
            score[a] += fut / self.M
        self.last_g = score
        return int(np.argmax(score))


class GreedyEnd:
    """E14b: one-step EIG of a reading taken at the END of an m-step straight move (no intermediate
    updates). This separates 'looking further' from 'accumulating correlated information'."""
    name = "gend"

    def __init__(self, m=5):
        self.m = m

    def reset(self):
        pass

    def act(self, s):
        b, pos, t, p = s["belief"], s["pos"], s["t"], s["p"]
        idx = b.active()
        g = np.zeros(8)
        for a in range(8):
            ps = pos
            for _ in range(self.m):
                ps = step_pose(ps, a, p)
            g[a] = b.eig(*ps, t + self.m * p.dt, idx)
        self.last_g = np.append(g, np.nan)
        if g.max() < 1e-6:
            return UPWIND
        return int(np.argmax(g))


def make_policy(name, **kw):
    for suf in ("_isohits", "_isoindep", "_iso", "_nnindep", "_nn", "_hits", "_indep"):        # E22/E27: "<planner>_hits" / "<planner>_indep" = same planner on a classical belief
        if name.startswith("macro") and name.endswith(suf):
            name = name[:-len(suf)]
            break
    if name == "greedy_nodown":
        return GreedyMasked()
    if name == "tree2":
        return Tree2(**kw)
    if name.startswith("gend"):
        return GreedyEnd(m=int(name[4:] or 5))
    if name == "greedy_nostay":
        return GreedyNoStay()
    if name == "greedy_audit":
        return GreedyAudit()
    if name == "greedy_v":
        return GreedyV()
    if name == "greedy_v_audit":
        return GreedyVAudit()
    if name == "tempered_entropy":       # E18 naming: argmax V_k (same as greedy_v)
        return GreedyV()
    if name == "scaled_mi_greedy":       # E18 naming: under tempering, greedy maximises k * I
        return Greedy()
    if name.startswith("seek"):             # E28: "seek", "seek_indep", "seek_hits", "seek_iso", "seek_isohits"
        return Seek(**kw)
    if name.startswith("gomean"):           # E28b: always head for the posterior mean (pure exploitation of the belief)
        return Seek(commit=np.inf, **kw)
    if name.startswith("macro"):            # "macro10" or "macro1M32" (m = 1, M = 32 samples)
        m, _, M = name[5:].partition("M")
        return Macro(m=int(m or 10), M=int(M) if M else 8, **kw)
    # E21 classical infotaxis baselines: one-step EIG on an independent belief used for BOTH inference and
    # decisions (episode.py gives them their own belief; errors are also scored on it, err_agent)
    return {"greedy": Greedy, "greedy_indep": Greedy, "infotaxis_hits": Greedy, "infotaxis_indep": Greedy,
            "greedy_iso": Greedy, "greedy_nn": Greedy, "infotaxis_nnindep": Greedy, "infotaxis_isoindep": Greedy, "infotaxis_isohits": Greedy,
            "zigzag": Zigzag,
            "surgecast": SurgeCast, "chase": Chase, "rollout": Rollout}[name](**kw)
