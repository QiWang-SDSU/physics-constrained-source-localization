"""Search harness: truth world, correlated censored belief, exact 1-D EIG.

Model (manuscript notation):
  mean plume          cbar(x; x_s) = Q c_dag(x_s; x)   (2-D slender plume, eq. adjoint_plume)
  log reading         z = log Q + log c_dag - sigma^2/2 + sigma*phi + eta,  eta ~ N(0, nug)
  fluctuation         phi ~ GP,  R(dx, dt) = K(|dx - U dt| / L) * exp(-|dt|/tau)
  detector            reading reported only if z >= log m_min, else "censored" (below threshold)

Belief: posterior on a grid of candidate sources. For detections, the likelihood uses exact Gaussian
conditioning on all previous DETECTED residuals (manuscript eqs. 3.16-3.17 plus nugget).
Censored readings use the conditional probability Phi((log m_min - zbar_j)/nu). This is an
approximation: censored latent values are not propagated into later conditioning. It is recorded
in decision_log D7.

EIG of a candidate pose: exact mutual information between the source index and the mixed
discrete/continuous observation. The mixture entropy is computed by 1-D quadrature; the
conditional entropies are closed-form (truncated Gaussian plus censoring mass).
"""
from dataclasses import dataclass, replace, field
import numpy as np
from scipy.special import ndtr, log_ndtr

LOG2PI = np.log(2 * np.pi)


@dataclass
class Params:
    U: float = 1.0            # mean wind (+x)
    K: float = 0.03           # crosswind eddy diffusivity (DNS strong: 0.030)
    Q: float = 1.0
    sigma: float = 1.0        # log-amplitude of the fluctuation
    tau: float = 1.0          # Lagrangian correlation time
    L: float = 0.4            # spatial correlation length
    kernel: str = "gauss"     # "gauss" or "exp"
    nug: float = 0.05 ** 2    # sensor noise variance in log
    m_min: float = 1e-2       # detection threshold (concentration units)
    dt: float = 0.1           # control step
    V: float = 1.0            # sensor speed
    n_steps: int = 80
    xs_range: tuple = (0.2, 2.0)
    ys_range: tuple = (-0.6, 0.6)
    grid_h: float = 0.02
    box: tuple = (0.0, 4.0, -1.0, 1.0)   # sensor bounds
    snap_src: bool = False    # place the true source on a grid-cell centre (calibration tests, E4)
    s0sq: float = 0.0         # virtual-origin crosswind variance of the mean plume (finite source size; DNS fit)
    T_L: float = 0.0          # Lagrangian time of Taylor dispersion (E24); 0 = Fickian width 2 K xi / U
    src_r: float = 0.0        # Gaussian source radius (E26): mean reading = adjoint integrated over the source; 0 = point source
    # E27 no-physics baseline: "iso" replaces Q c_dag by an isotropic kernel exp(-r^2 / 2 iso_l^2) (no wind, no
    # transport); its amplitude is absorbed by the fitted zi2 coefficients. iso_zi holds that belief's zi2 fit.
    mean_model: str = "adjoint"
    iso_l: float = 0.5
    iso_zi: dict = field(default_factory=dict)
    # E30 learned observation model: mean_model "nn" takes (gamma, log-mean, sd) of every candidate from an MLP of
    # the sensor-minus-candidate offset trained on DNS readings (scripts/checks/dns2_nn_likelihood.py), no physics
    nn_file: str = ""
    start_x: float = 3.5      # sensor start x (E10 geometry test)
    censor_ep: bool = True    # censored readings enter conditioning as EP pseudo-observations (E4 fix)
    # observation model: "lognormal" (manuscript) or "zi" (intermittent plume, E8c):
    #   gamma = sigmoid(zi_a0 + zi_a1 * lc)       intermittency factor, lc = log(Q c_dag)
    #   in-puff log c ~ N(zi_c0 + zi_c1 * lc, s^2), s = sigma, correlated as before
    #   outside the model support (c_dag = 0): gamma = leak_g, log c ~ N(leak_mu, leak_s^2), independent
    obs: str = "lognormal"
    # effective-sample-size tempering (E8d): per-step log-likelihood x kappa, kappa = min(1, dt / (2 T_int(w))),
    # T_int = tau * g(w tau / L), w = |sensor velocity - U e_x| (theory_notes T1). Used with correlated=False.
    temper: bool = False
    temper_const: bool = False   # E22 ablation: kappa at the stationary-sensor rate w = U for every move (ignores the law's motion dependence)
    zi_a0: float = 0.0
    zi_a1: float = 1.0
    zi_c0: float = 0.0
    zi_c1: float = 1.0
    leak_g: float = 0.0
    leak_mu: float = -4.0
    leak_s: float = 0.5


def plume_var(xi, p: Params):
    """Cross-wind variance of the mean plume at upstream distance xi > 0. With T_L > 0 this is Taylor's (1921)
    dispersion law, i.e. the slender adjoint solution for a developing eddy diffusivity K (1 - exp(-xi / U T_L))."""
    T_L = getattr(p, "T_L", 0.0)
    if T_L > 0:
        r = xi / (p.U * T_L)
        return 2 * p.K * T_L * (r - 1 + np.exp(-r)) + getattr(p, "s0sq", 0.0)
    return 2 * p.K * xi / p.U + getattr(p, "s0sq", 0.0)


_GH_T = np.linspace(-4.0, 4.0, 17)                 # uniform rule: the integrand jumps at xi = 0, so Gauss-Hermite
_GH_W = np.exp(-0.5 * _GH_T ** 2)                  # nodes would resolve the upstream tail poorly
_GH_W = _GH_W / _GH_W.sum()


def c_dag(xs, ys, x, y, p: Params):
    """Adjoint plume rooted at sensor (x, y) evaluated at candidate sources (xs, ys). With p.src_r > 0 the field is
    integrated over the streamwise extent of a Gaussian source of radius src_r (adjoint identity c = int s c_dag);
    the cross-wind extent is part of the fitted s0^2."""
    if getattr(p, "src_r", 0.0) > 0:
        return sum(w * _c_dag_point(xs + p.src_r * t, ys, x, y, p) for t, w in zip(_GH_T, _GH_W))
    return _c_dag_point(xs, ys, x, y, p)


_NN = {}


def nn_features(xs, ys, x, y, W):
    """Inputs of the learned observation model: the scaled offset, plus for the hybrid model (E31, W has "phys" =
    [U, K, T_L, s0sq, src_r, Q]) the adjoint log-mean lb/10 (clipped at -30) and a support flag."""
    dx, dy = np.broadcast_arrays((x - xs) / W["scale"][0], (y - ys) / W["scale"][1])
    if "phys" not in W:
        return np.stack([dx, dy], -1)
    U, K, T_L, s0sq, src_r, Q = W["phys"]
    lb = log_cbar(xs, ys, x, y, Params(U=U, K=K, T_L=T_L, s0sq=s0sq, src_r=src_r, Q=Q))
    fin = np.isfinite(lb)
    return np.stack([dx, dy, np.broadcast_to(np.maximum(np.where(fin, lb, -30.0), -30.0) / 10, dx.shape),
                     np.broadcast_to(fin, dx.shape).astype(float)], -1)


def nn_components(xs, ys, x, y, p: Params):
    """Learned single-point observation model (E30/E31): gamma, in-plume mean and sd of log c at offset
    (x-xs, y-ys); the hybrid variant also sees the adjoint log-mean."""
    if p.nn_file not in _NN:
        _NN[p.nn_file] = dict(np.load(p.nn_file))
    W = _NN[p.nn_file]
    h = nn_features(xs, ys, x, y, W)
    for i in range(int(W["n_layers"])):
        h = h @ W[f"W{i}"] + W[f"b{i}"]
        if i < int(W["n_layers"]) - 1:
            h = np.tanh(h)
    with np.errstate(over="ignore"):
        g = 1 / (1 + np.exp(-h[..., 0]))
    return g, h[..., 1], np.exp(np.clip(h[..., 2], -5, 3))


def log_cbar(xs, ys, x, y, p: Params):
    """log of the mean reading of the belief's observation model: log(Q c_dag) (physics; -inf outside the support)
    or the isotropic kernel (E27), evaluated in log space so it never underflows to the leak branch."""
    if getattr(p, "mean_model", "adjoint") == "iso":
        return -0.5 * ((x - xs) ** 2 + (y - ys) ** 2) / p.iso_l ** 2
    if getattr(p, "mean_model", "adjoint") == "nn":       # no mean field: components come from the MLP
        return np.zeros(np.broadcast(xs, x).shape)
    with np.errstate(divide="ignore"):
        return np.log(p.Q * c_dag(xs, ys, x, y, p))


def _c_dag_point(xs, ys, x, y, p: Params):
    xi = x - xs
    out = np.zeros(np.broadcast(xs, x).shape)
    m = xi > 1e-9
    s2 = plume_var(np.where(m, xi, 1.0), p)
    val = np.exp(-0.5 * (y - ys) ** 2 / s2) / (p.U * np.sqrt(2 * np.pi * s2))
    out[m] = np.broadcast_to(val, out.shape)[m]
    return out


def tint(w, p: Params):
    """Integral time of the fluctuation seen along a path at relative speed w (Gaussian kernel, T1)."""
    from scipy.special import erfcx
    b = max(w * p.tau / p.L, 1e-9)
    return p.tau * np.sqrt(np.pi / 2) / b * erfcx(1 / (np.sqrt(2) * b))


def kappa(prev, cur, p: Params):
    """Effective independent samples contributed by one step from prev to cur (<= 1)."""
    if prev is None or getattr(p, "temper_const", False):
        w = p.U
    else:
        v = (np.asarray(cur) - np.asarray(prev)) / p.dt
        w = np.hypot(v[0] - p.U, v[1])
    return min(1.0, p.dt / (2 * tint(w, p)))


def corr(dx, dy, dtt, p: Params):
    rx = dx - p.U * dtt
    r = np.sqrt(rx ** 2 + dy ** 2) / p.L
    k = np.exp(-0.5 * r ** 2) if p.kernel == "gauss" else np.exp(-r)
    return k * np.exp(-np.abs(dtt) / p.tau)


def cov_block(A, B, p: Params):
    """A, B: arrays (n,3) of (x, y, t). Returns sigma^2 R."""
    return p.sigma ** 2 * corr(A[:, None, 0] - B[None, :, 0], A[:, None, 1] - B[None, :, 1],
                               A[:, None, 2] - B[None, :, 2], p)


class World:
    """Truth: source location plus an exact GP fluctuation sampled lazily along the path."""

    def __init__(self, p: Params, src, rng):
        self.p, self.src, self.rng = p, np.asarray(src, float), rng
        self.pts = np.zeros((0, 3))
        self.phi = np.zeros(0)
        self.Lchol = None

    def measure(self, x, y, t):
        p = self.p
        new = np.array([[x, y, t]])
        if len(self.pts) == 0:
            mu, var = 0.0, p.sigma ** 2
        else:
            k = cov_block(self.pts, new, p)[:, 0]
            S = cov_block(self.pts, self.pts, p) + 1e-10 * np.eye(len(self.pts))
            a = np.linalg.solve(S, np.column_stack([self.phi, k]))
            mu, var = k @ a[:, 0], max(p.sigma ** 2 - k @ a[:, 1], 1e-12)
        f = mu + np.sqrt(var) * self.rng.standard_normal()     # this is sigma*phi
        self.pts = np.vstack([self.pts, new])
        self.phi = np.append(self.phi, f)
        cd = c_dag(self.src[0], self.src[1], x, y, p)
        if cd <= 0:
            return None                                          # upstream of source: zero reading
        z = np.log(p.Q * cd) - 0.5 * p.sigma ** 2 + f + np.sqrt(p.nug) * self.rng.standard_normal()
        return z if z >= np.log(p.m_min) else None


class ZIWorld:
    """Model-world truth with the intermittent (zi) observation structure, E12.
    in-puff log c = zi_c0 + zi_c1 lc + phi,  phi ~ GP(sigma^2, advected kernel)      (as the belief)
    switch: the reading is in-puff iff psi < Phi^{-1}(gamma(lc)), psi ~ GP(1, same kernel), independent
            of phi. So the switch has the belief's marginal gamma but is CORRELATED in space-time
            (corr_switch=True), or i.i.d. per reading (corr_switch=False, which matches the belief exactly).
    leak (c_dag = 0): in-puff with probability leak_g (i.i.d.), log c ~ N(leak_mu, leak_s^2)."""

    def __init__(self, p: Params, src, rng, corr_switch=True):
        self.p, self.src, self.rng, self.corr_switch = p, np.asarray(src, float), rng, corr_switch
        self.pts = np.zeros((0, 3))
        self.f = np.zeros((0, 2))          # sampled (phi, psi) along the path

    def _sample(self, new):
        p = self.p
        if len(self.pts) == 0:
            mu, var = np.zeros(2), np.array([p.sigma ** 2, 1.0])
        else:
            k = corr(self.pts[:, 0] - new[0], self.pts[:, 1] - new[1], self.pts[:, 2] - new[2], p)
            R = corr(self.pts[:, None, 0] - self.pts[None, :, 0], self.pts[:, None, 1] - self.pts[None, :, 1],
                     self.pts[:, None, 2] - self.pts[None, :, 2], p) + 1e-9 * np.eye(len(self.pts))
            a = np.linalg.solve(R, np.column_stack([self.f[:, 0] / p.sigma, self.f[:, 1], k]))
            mu = np.array([p.sigma * (k @ a[:, 0]), k @ a[:, 1]])
            var = np.maximum(1 - k @ a[:, 2], 1e-12) * np.array([p.sigma ** 2, 1.0])
        return mu + np.sqrt(var) * self.rng.standard_normal(2)

    def measure(self, x, y, t):
        from scipy.special import ndtri
        p = self.p
        new = np.array([x, y, t])
        phi, psi = self._sample(new)
        self.pts = np.vstack([self.pts, new]); self.f = np.vstack([self.f, [phi, psi]])
        cd = c_dag(self.src[0], self.src[1], x, y, p)
        u = psi if self.corr_switch else self.rng.standard_normal()
        if cd <= 0:
            if self.rng.uniform() >= p.leak_g:
                return None
            z = p.leak_mu + p.leak_s * self.rng.standard_normal()
        else:
            lc = np.log(p.Q * cd)
            g = 1 / (1 + np.exp(-(p.zi_a0 + p.zi_a1 * lc)))
            if u >= ndtri(min(max(g, 1e-12), 1 - 1e-12)):
                return None
            z = p.zi_c0 + p.zi_c1 * lc + phi
        z = z + np.sqrt(p.nug) * self.rng.standard_normal()
        return z if z >= np.log(p.m_min) else None


class Belief:
    """Grid posterior with correlated conditioning on detected residuals."""

    def __init__(self, p: Params, correlated=True):
        self.p, self.correlated = p, correlated
        xs = np.arange(p.xs_range[0], p.xs_range[1] + 1e-9, p.grid_h)
        ys = np.arange(p.ys_range[0], p.ys_range[1] + 1e-9, p.grid_h)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        self.shape = X.shape
        self.cx, self.cy = X.ravel(), Y.ravel()
        self.logw = np.full(self.cx.size, -np.log(self.cx.size))
        self.D = np.zeros((0, 3))              # detected poses
        self.Rres = np.zeros((0, self.cx.size))  # residuals z - mu_j at detected poses
        self.SinvR = None
        self.Sinv = None
        self.thr = np.log(p.m_min)
        self.prev = None
        if p.temper:
            self.correlated = False          # tempering replaces explicit correlated conditioning

    # -------------------------------------------------------------- prediction
    def lc(self, x, y):
        """log cbar_j = log(Q c_dag) at the pose for every candidate (-inf outside the support)."""
        return log_cbar(self.cx, self.cy, x, y, self.p)

    def mu(self, x, y):
        """Unconditional mean of the (in-puff) log reading; -inf where c_dag = 0."""
        lc = self.lc(x, y)
        if self.p.obs == "zi":
            return self.p.zi_c0 + self.p.zi_c1 * lc
        return lc - 0.5 * self.p.sigma ** 2

    def components(self, x, y, t, idx=None):
        """Per-candidate observation triple (gamma, zbar, sd): the reading is detected-branch with
        probability gamma, then log c ~ N(zbar, sd^2) (censored below thr); otherwise it is below
        threshold. lognormal: gamma = 1 inside the support, 0 outside. zi: logistic gamma(lc) inside,
        leak branch outside."""
        if getattr(self.p, "mean_model", "adjoint") == "nn":
            cx, cy = (self.cx, self.cy) if idx is None else (self.cx[idx], self.cy[idx])
            g, zb, s = nn_components(cx, cy, x, y, self.p)
            return g, zb, np.sqrt(s ** 2 + self.p.nug)
        zbar, nu = self.predict(x, y, t, idx)
        fin = np.isfinite(zbar)
        sd = np.full(zbar.shape, nu)
        if self.p.obs == "zi":
            lc = self.lc(x, y) if idx is None else self.lc(x, y)[idx]
            with np.errstate(over="ignore", invalid="ignore"):
                g = np.where(fin, 1 / (1 + np.exp(-(self.p.zi_a0 + self.p.zi_a1 * np.where(fin, lc, 0.0)))),
                             self.p.leak_g)
            zbar = np.where(fin, zbar, self.p.leak_mu)
            sd = np.where(fin, sd, self.p.leak_s)
        else:
            g = np.where(fin, 1.0, 0.0)
            zbar = np.where(fin, zbar, 0.0)
        return g, zbar, sd

    def sample_obs(self, x, y, t, j, rng):
        g, zb, sd = self.components(x, y, t, np.array([j]))
        if rng.uniform() >= g[0]:
            return None
        z = zb[0] + sd[0] * rng.standard_normal()
        return z if z >= self.thr else None

    def predict(self, x, y, t, idx=None):
        """zbar_j and nu for candidates idx at pose (x, y, t)."""
        p = self.p
        mu = self.mu(x, y) if idx is None else self.mu(x, y)[idx]
        nu2 = p.sigma ** 2 + p.nug
        if self.correlated and len(self.D):
            k = cov_block(self.D, np.array([[x, y, t]]), p)[:, 0]
            A = self.SinvR if idx is None else self.SinvR[:, idx]
            mu = mu + k @ A
            nu2 = nu2 - k @ self.Sinv @ k
        return mu, np.sqrt(max(nu2, 1e-8))

    # -------------------------------------------------------------- update
    def _append_row(self, x, y, t, res, noise):
        self.D = np.vstack([self.D, [x, y, t]])
        self.Rres = np.vstack([self.Rres, res])
        self.noise = np.append(getattr(self, "noise", np.zeros(0)), noise)
        S = cov_block(self.D, self.D, self.p) + np.diag(self.noise)
        self.Sinv = np.linalg.inv(S)
        self.SinvR = self.Sinv @ self.Rres

    def update(self, x, y, t, z):
        """Bayes update. Detected readings enter the correlated conditioning exactly.
        Censored readings (z is None) update the weights with the exact conditional probability
        Phi((thr - zbar_j)/nu). If censor_ep is set (default), they also enter later conditioning
        as a Gaussian pseudo-observation (assumed-density / EP site). Its per-candidate mean
        matches the first moment of the truncated predictive residual; its noise variance s^2 is
        shared across candidates (posterior-weighted mean site precision). This replaces the
        E2/E3 approximation that ignored censored readings in later conditioning, which E4 showed
        to be overconfident under correlation."""
        zbar, nu = self.predict(x, y, t)
        mu = self.mu(x, y)
        fin = np.isfinite(zbar)
        ll = self.loglik(x, y, t, z)
        if z is None:
            if self.correlated and getattr(self.p, "censor_ep", True) and self.p.obs == "lognormal":
                mres = np.where(fin, zbar - np.where(np.isfinite(mu), mu, 0.0), 0.0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    alpha = np.where(fin, (self.thr - zbar) / nu, np.inf)
                    lam = np.where(fin, np.exp(-0.5 * alpha ** 2 - 0.5 * LOG2PI - log_ndtr(alpha)), 0.0)
                    vt = np.where(fin, nu ** 2 * np.clip(1 - alpha * lam - lam ** 2, 1e-6, 1.0), nu ** 2)
                    mt = mres - nu * lam
                    prec = np.clip(1 / vt - 1 / nu ** 2, 0.0, None)          # site precision
                w = np.exp(self.logw + ll - np.logaddexp.reduce(self.logw + ll))
                pbar = float(w @ prec)
                if pbar > 1e-3 / nu ** 2:
                    s2 = 1.0 / pbar
                    res = mres + (mt - mres) * (nu ** 2 + s2) / nu ** 2
                    self._append_row(x, y, t, np.where(fin, res, mres), self.p.nug + s2)
        else:
            if self.correlated:
                self._append_row(x, y, t, np.where(np.isfinite(mu), z - mu, 0.0), self.p.nug)
        self.logw = self.transition(self.logw, ll, self.kappa_at(x, y))
        self.prev = (x, y)

    # -------------------------------------------------------------- the two primitives (E18)
    def loglik(self, x, y, t, z, idx=None):
        """Untempered log p(z | x_j, pose) for candidates idx (up to a z-only constant).
        z: None (censored), a float, or a 1-D array of detected values -> shape (nz, J).
        This is the predictive observation model p(z | x, a); predictive_obs is its mixture over w."""
        zbar, nu = self.predict(x, y, t, idx)
        fin = np.isfinite(zbar)
        g, zb, sd = self.components(x, y, t, idx)
        with np.errstate(divide="ignore", invalid="ignore"):
            if z is None:
                if self.p.obs == "lognormal":           # exact log-space form (avoids underflow)
                    return np.where(fin, log_ndtr((self.thr - zbar) / nu), 0.0)
                return np.log(np.clip((1 - g) + g * ndtr((self.thr - zb) / sd), 1e-300, None))
            z = np.asarray(z, float)
            zz = z[..., None] if z.ndim else z
            return np.where(g > 0, np.log(np.where(g > 0, g, 1.0)) - 0.5 * ((zz - zb) / sd) ** 2 - np.log(sd), -np.inf)

    def kappa_at(self, x, y):
        """Tempering exponent of a move from the last pose to (x, y); 1 without tempering."""
        return kappa(self.prev, (x, y), self.p) if self.p.temper else 1.0

    @staticmethod
    def transition(logw, ll, k):
        """THE posterior transition T_k[p, z, a]: log p+ = log p + k * ll, normalised. Broadcasts over
        leading axes of ll (one row per hypothetical reading). Used by update (real and imagined) and
        by vkappa; no other code applies tempering to the weights."""
        ll = np.where(np.isfinite(ll), ll * k, ll)
        lw = logw + ll
        return lw - np.logaddexp.reduce(lw, axis=-1, keepdims=lw.ndim > 1)

    # -------------------------------------------------------------- summaries
    def w(self):
        return np.exp(self.logw)

    def entropy(self):
        w = self.w()
        nz = w > 0
        return -np.sum(w[nz] * np.log(w[nz])) + np.log(self.p.grid_h ** 2)   # differential, nats

    def mean(self):
        w = self.w()
        return np.array([w @ self.cx, w @ self.cy])

    def active(self, mass=1 - 1e-6):
        w = self.w()
        o = np.argsort(w)[::-1]
        k = np.searchsorted(np.cumsum(w[o]), mass) + 1
        return o[:k]

    def eig(self, x, y, t, idx=None, nq=400, nmax=4000):
        """Exact mutual information I(source; reading) at pose, over candidates idx, for the mixed
        observation: detected-branch with probability gamma_j, log c ~ N(zbar_j, sd_j^2) censored
        below thr. H(Y|j) is closed-form; H(Y) uses 1-D quadrature of the mixture density."""
        if idx is None:
            idx = self.active()
        w = self.w()[idx]
        w = w / w.sum()
        g, zb, sd = self.components(x, y, t, idx)
        a = (self.thr - zb) / sd
        Phi = ndtr(a)
        pd = g * (1 - Phi)                                           # P(detected | j)
        Pc = 1 - pd
        with np.errstate(divide="ignore", invalid="ignore"):
            phi_a = np.exp(-0.5 * a ** 2) / np.sqrt(2 * np.pi)
            Hunit = (1 - Phi) * 0.5 * (LOG2PI + 2 * np.log(sd)) + 0.5 * ((1 - Phi) + a * phi_a)
            Hcont = np.where(g > 0, g * Hunit - np.where(pd > 0, pd * np.log(np.where(g > 0, g, 1.0)), 0.0), 0.0)
            Hc = -np.where(Pc > 0, Pc * np.log(np.where(Pc > 0, Pc, 1.0)), 0.0)
        HYj = w @ (Hc + Hcont)
        Pc_m = w @ Pc
        H = -Pc_m * np.log(Pc_m) if Pc_m > 0 else 0.0
        act = (g * w) > 0
        if act.any() and Pc_m < 1 - 1e-12:
            zf, wf, sf = zb[act], (w * g)[act], sd[act]
            hi = np.max(zf + 6 * sf)
            lo = self.thr
            if hi > lo:
                n = int(min(max(nq, (hi - lo) / (sf.min() / 4)), nmax))
                yq = np.linspace(lo, hi, n)
                dens = (wf[:, None] / sf[:, None] * np.exp(-0.5 * ((yq[None, :] - zf[:, None]) / sf[:, None]) ** 2)).sum(0)                     / np.sqrt(2 * np.pi)
                with np.errstate(divide="ignore", invalid="ignore"):
                    integrand = np.where(dens > 0, -dens * np.log(dens), 0.0)
                H += np.trapezoid(integrand, yq)
        mi = max(H - HYj, 0.0)
        if self.p.temper:                 # information scales with the effective samples of this move
            mi *= kappa(self.prev, (x, y), self.p)
        return mi


    def vkappa(self, x, y, t, idx=None, nq=400, nmax=4000):
        """Consistent one-step value (E18): V_k(a) = H[p] - E_{z ~ p(z|a)} H[T_k(p, z, a)], with the
        UNtempered predictive p(z|a) and the tempered transition actually used by the agent. Equals the
        mutual information when k = 1. Not clamped at 0 (a tempered update is not a Bayes posterior).
        Entropies are over the candidates idx (default: active set), renormalised."""
        if idx is None:
            idx = self.active()
        lw0 = self.logw[idx] - np.logaddexp.reduce(self.logw[idx])
        w = np.exp(lw0)
        k = self.kappa_at(x, y)
        ent = lambda lw: -np.sum(np.where(np.isfinite(lw), np.exp(lw) * lw, 0.0), axis=-1)
        H0 = ent(lw0)
        g, zb, sd = self.components(x, y, t, idx)
        Pc_j = 1 - g * (1 - ndtr((self.thr - zb) / sd))
        Pc = float(w @ Pc_j)
        EH = Pc * ent(self.transition(lw0, self.loglik(x, y, t, None, idx), k)) if Pc > 0 else 0.0
        act = (g * w) > 0
        if act.any() and Pc < 1 - 1e-12:
            zf, wf, sf = zb[act], (w * g)[act], sd[act]
            hi, lo = np.max(zf + 6 * sf), self.thr
            if hi > lo:
                n = int(min(max(nq, (hi - lo) / (sf.min() / 4)), nmax))
                yq = np.linspace(lo, hi, n)
                dens = (wf[:, None] / sf[:, None] * np.exp(-0.5 * ((yq[None, :] - zf[:, None]) / sf[:, None]) ** 2)).sum(0)                     / np.sqrt(2 * np.pi)
                Hz = np.empty(n)
                for c in range(0, n, 256):                       # chunk to bound memory (n x J)
                    Hz[c:c + 256] = ent(self.transition(lw0, self.loglik(x, y, t, yq[c:c + 256], idx), k))
                EH += np.trapezoid(dens * Hz, yq)
        return float(H0 - EH)


class HitBelief(Belief):
    """Classical infotaxis belief (Vergassola et al. 2007; E21 baseline): the reading is reduced to a binary
    detection h in {0, 1}, and hits are conditionally independent in space and time given the source.
    P(h = 1 | x_j, pose) = p_j, the true MARGINAL detection probability of the model,
    p_j = 1 - Phi((thr - mu_j) / sqrt(sigma^2 + nug)) inside the plume support and 0 outside; equivalently a
    Poisson hit count with mean -log(1 - p_j), of which only 'any hit' is observed. This is the most favourable
    calibration of the classical model: each hit probability is exact, only the correlation is ignored."""

    def __init__(self, p: Params):
        super().__init__(replace(p, temper=False), correlated=False)     # classical: untempered

    def phit(self, x, y, idx=None):
        # marginal detection probability gamma_j (1 - Phi((thr - zbar_j) / sd_j)); for the lognormal model this is
        # 1 - Phi((thr - mu_j) / sqrt(sigma^2 + nug)) inside the support and 0 outside; for zi it includes the leak (E22)
        g, zb, sd = self.components(x, y, t=0.0, idx=idx)
        return g * (1 - ndtr((self.thr - zb) / sd))

    def loglik(self, x, y, t, z, idx=None):
        ph = self.phit(x, y, idx)
        with np.errstate(divide="ignore"):
            if z is None:
                return np.log1p(-ph)
            z = np.asarray(z, float)
            return np.broadcast_to(np.log(ph), z.shape + ph.shape) if z.ndim else np.log(ph)

    def eig(self, x, y, t, idx=None, nq=None, nmax=None):
        """Mutual information between the source and the binary detection."""
        if idx is None:
            idx = self.active()
        w = self.w()[idx]
        w = w / w.sum()
        ph = self.phit(x, y, idx)
        hb = lambda q: -np.where(q > 0, q * np.log(np.where(q > 0, q, 1.0)), 0.0) \
            - np.where(q < 1, (1 - q) * np.log(np.where(q < 1, 1 - q, 1.0)), 0.0)
        return max(float(hb(np.array(w @ ph)) - w @ hb(ph)), 0.0)


ACTIONS = np.array([[np.cos(a), np.sin(a)] for a in np.arange(8) * np.pi / 4] + [[0.0, 0.0]])


def step_pose(pos, a, p: Params):
    x = np.clip(pos[0] + p.V * p.dt * ACTIONS[a, 0], p.box[0], p.box[1])
    y = np.clip(pos[1] + p.V * p.dt * ACTIONS[a, 1], p.box[2], p.box[3])
    return np.array([x, y])
