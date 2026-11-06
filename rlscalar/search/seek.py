"""Source seeking on DNS2 (E28): model-free RL baselines vs the physics-constrained belief.

Task (shared by every agent): the sensor starts at x = start_x and must come within r_succ of the true source
within n_steps moves; reward -0.01 per move, the episode ends on success. Tasks are generated exactly as in the
search sweeps (sample_task + search_sweep.dns_factory: source, start, DNS file, plume and time window from the
episode seed), so episode seed s on the held-out files is the same task for the RL agents and for the belief
planners run by search_sweep.py with the same --steps.

Model-free agents see only their own readings (detected log c or censored) and their moves; no flow model.
  tabular: SARSA(lambda) or Watkins Q-learning on a finite memory state
           (reading level, steps since the last detection, cross-wind side and along-wind offset from the last
           detection, previous action), epsilon-greedy;
  DRQN:    double DQN with a GRU over the raw history (reading, detection flag, position, previous action, time).
"""
import numpy as np
from .core import Params, ACTIONS, step_pose
from .episode import sample_task

NA = len(ACTIONS)
R_STEP = -0.01


class SeekTasks:
    """Episode seed -> (DNSWorld, src, start); the DNS files are loaded once per process."""

    def __init__(self, p: Params, files):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
        from search_sweep import dns_factory
        self.p, self.make = p, dns_factory(files)

    def world(self, seed):
        src, start = sample_task(self.p, seed)
        w = self.make(self.p, src, np.random.default_rng(10_000 + seed), seed)
        return w, src, start


class Episode:
    """One seeking episode; step(a) -> (z or None, done, success)."""

    def __init__(self, tasks: SeekTasks, seed, r_succ=0.2):
        self.p, self.r = tasks.p, r_succ
        self.w, self.src, self.pos = tasks.world(seed)
        self.k, self.t = 0, 0.0
        self.z = self.w.measure(*self.pos, self.t)
        self.success = bool(np.hypot(*(self.pos - self.src)) < r_succ)

    def step(self, a):
        self.pos = step_pose(self.pos, a, self.p)
        self.k, self.t = self.k + 1, self.t + self.p.dt
        self.z = self.w.measure(*self.pos, self.t)
        self.success = bool(np.hypot(*(self.pos - self.src)) < self.r)
        return self.z, self.success or self.k >= self.p.n_steps, self.success


# ---------------------------------------------------------------------------------------------- tabular
TSL_EDGES = np.array([1, 2, 3, 5, 9, 17, 33])          # steps since last detection -> 8 bins (+1 "never")
Z_HIGH = np.log(0.1)


class Memory:
    """Finite memory state of the tabular agents."""
    n = 3 * 9 * 3 * 3 * NA

    def reset(self, pos, z):
        self.last = None if z is None else np.array(pos, float)
        self.tsl, self.prev = 0, NA - 1
        return self.index(pos, z)

    def update(self, pos, z, a):
        self.prev = a
        if z is not None:
            self.last, self.tsl = np.array(pos, float), 0
        else:
            self.tsl += 1
        return self.index(pos, z)

    def index(self, pos, z):
        lev = 0 if z is None else (1 if z < Z_HIGH else 2)
        if self.last is None:
            tb, side, along = 8, 1, 1
        else:
            tb = int(np.searchsorted(TSL_EDGES, self.tsl, side="right"))
            dy, dx = pos[1] - self.last[1], pos[0] - self.last[0]
            side = 0 if dy < -0.05 else (2 if dy > 0.05 else 1)
            along = 0 if dx < -0.3 else (2 if dx > 0.05 else 1)
        return (((lev * 9 + tb) * 3 + side) * 3 + along) * NA + self.prev


class Tabular:
    def __init__(self, algo="sarsa", alpha=0.1, gamma=0.99, lam=0.9, seed=0):
        self.algo, self.alpha, self.gamma, self.lam = algo, alpha, gamma, (lam if algo == "sarsa" else 0.0)
        self.Q = np.zeros((Memory.n, NA))
        self.rng = np.random.default_rng(seed)
        self.visits = np.zeros(Memory.n, np.int64)

    def act(self, s, eps):
        if self.rng.uniform() < eps:
            return int(self.rng.integers(NA))
        q = self.Q[s]
        return int(self.rng.choice(np.flatnonzero(q == q.max())))

    def train_episode(self, tasks, seed, eps):
        ep, mem = Episode(tasks, seed), Memory()
        s = mem.reset(ep.pos, ep.z)
        a = self.act(s, eps)
        E = {}                                            # sparse replacing traces
        done = ep.success
        while not done:
            z, done, succ = ep.step(a)
            s2 = mem.update(ep.pos, z, a)
            self.visits[s] += 1
            a2 = self.act(s2, eps)
            if self.algo == "sarsa":
                target = R_STEP + (0.0 if succ else (self.gamma * self.Q[s2, a2] if not done else 0.0))
            else:
                target = R_STEP + (0.0 if done else self.gamma * self.Q[s2].max())
            delta = target - self.Q[s, a]
            if self.lam == 0.0:
                self.Q[s, a] += self.alpha * delta
            else:
                E[(s, a)] = 1.0
                for key in list(E):
                    self.Q[key] += self.alpha * delta * E[key]
                    E[key] *= self.gamma * self.lam
                    if E[key] < 1e-3:
                        del E[key]
            s, a = s2, a2
        return ep.success, ep.k

    def policy(self):
        rng = np.random.default_rng(0)
        mem = Memory()
        Q = self.Q

        def act(pos, z, prev_a, first):
            s = mem.reset(pos, z) if first else mem.update(pos, z, prev_a)
            q = Q[s]
            return int(rng.choice(np.flatnonzero(q == q.max())))
        return act


# ---------------------------------------------------------------------------------------------- DRQN
def obs_vec(p, pos, z, prev_a, k):
    v = np.zeros(5 + NA, np.float32)
    v[0] = 0.0 if z is None else 1.0
    v[1] = 0.0 if z is None else (z - np.log(p.m_min)) / 5
    v[2], v[3], v[4] = pos[0] / 4, pos[1], k / p.n_steps
    v[5 + prev_a] = 1.0
    return v


def make_drqn(hidden=128):
    import torch.nn as nn

    class DRQN(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(5 + NA, hidden), nn.ReLU())
            self.gru = nn.GRU(hidden, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, NA))

        def forward(self, x, h=None):
            y, h = self.gru(self.inp(x), h)
            return self.head(y), h
    return DRQN()


def run_drqn_batch(net, tasks, seeds, eps, rng):
    """Run len(seeds) episodes in lock-step with one batched GRU forward per move.
    Returns obs (B, T+1, d), actions (B, T), rewards (B, T), done mask (B, T), success, length."""
    import torch
    p, B, T = tasks.p, len(seeds), tasks.p.n_steps
    eps_ = [Episode(tasks, s) for s in seeds]
    O = np.zeros((B, T + 1, 5 + NA), np.float32)
    A = np.zeros((B, T), np.int64); R = np.zeros((B, T), np.float32); M = np.zeros((B, T), np.float32)
    D = np.zeros((B, T), np.float32)
    for b, e in enumerate(eps_):
        O[b, 0] = obs_vec(p, e.pos, e.z, NA - 1, 0)
    alive = np.array([not e.success for e in eps_])
    h = None
    with torch.no_grad():
        for k in range(T):
            if not alive.any():
                break
            q, h = net(torch.from_numpy(O[:, k:k + 1]), h)
            q = q[:, 0].numpy()
            for b in np.flatnonzero(alive):
                a = int(rng.integers(NA)) if rng.uniform() < eps else int(np.argmax(q[b]))
                z, done, succ = eps_[b].step(a)
                A[b, k], R[b, k], M[b, k], D[b, k] = a, R_STEP, 1.0, float(succ or done)
                O[b, k + 1] = obs_vec(p, eps_[b].pos, z, a, k + 1)
                if done:
                    alive[b] = False
    return O, A, R, M, D, np.array([e.success for e in eps_]), np.array([e.k for e in eps_])


# ---------------------------------------------------------------------------------------------- E29 localisation
def make_locnet(hidden=128):
    """GRU over the raw history with two heads: Q values (9 moves) and a Gaussian estimate of the source
    (mean 2, log sd 2), read out after every reading."""
    import torch.nn as nn

    class LocNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(5 + NA, hidden), nn.ReLU())
            self.gru = nn.GRU(hidden, hidden, batch_first=True)
            self.q = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, NA))
            self.est = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 4))

        def forward(self, x, h=None):
            y, h = self.gru(self.inp(x), h)
            e = self.est(y)
            return self.q(y), e[..., :2], e[..., 2:].clamp(-5, 2), h
    return LocNet()


def run_loc_batch(net, tasks, seeds, eps, rng):
    """Fixed-length localisation episodes (n_steps moves, no early stop) in lock-step.
    Returns obs (B, T+1, d), actions (B, T), true sources (B, 2) and sensor paths (B, T+1, 2)."""
    import torch
    p, B, T = tasks.p, len(seeds), tasks.p.n_steps
    eps_ = [Episode(tasks, s, r_succ=-1.0) for s in seeds]
    O = np.zeros((B, T + 1, 5 + NA), np.float32)
    A = np.zeros((B, T), np.int64)
    P = np.zeros((B, T + 1, 2), np.float32)
    for b, e in enumerate(eps_):
        O[b, 0] = obs_vec(p, e.pos, e.z, NA - 1, 0); P[b, 0] = e.pos
    h = None
    with torch.no_grad():
        for k in range(T):
            q, _, _, h = net(torch.from_numpy(O[:, k:k + 1]), h)
            q = q[:, 0].numpy()
            for b, e in enumerate(eps_):
                a = int(rng.integers(NA)) if rng.uniform() < eps else int(np.argmax(q[b]))
                z, _, _ = e.step(a)
                A[b, k] = a
                O[b, k + 1] = obs_vec(p, e.pos, z, a, k + 1); P[b, k + 1] = e.pos
    return O, A, np.array([e.src for e in eps_], np.float32), P
