"""2-D forced turbulence with a uniform mean wind and several passive-scalar plumes (v2).

Replaces rlscalar/dns.py for all paper results. Changes are recorded in decision_log D8.

Equations (x streamwise, periodic box Lx x Ly):
    d_t w + (U e_x + u) . grad w = nu lap w - mu w + f,         u = curl(psi e_z), -lap psi = w
    d_t c_i + (U e_x + u) . grad c_i = kappa lap c_i + s_i(x) - lambda(x) c_i,   i = 1..n_src
- f: Gaussian, white in time, isotropic in the band k_f in [kmin, kmax]. Added as
  sqrt(dt) * F once per step (Euler-Maruyama), so the injection rate eps = <|F|^2>/2 does not
  depend on dt.
- mu: linear (Ekman) drag. It arrests the inverse cascade, which the v1 solver did not.
- lambda(x): scalar sponge at the downstream end, so plumes never wrap. The velocity is
  x-periodic, with recurrence Lx/U (4*pi in the default), much longer than the sampling
  windows used.
- Time stepping: integrating-factor RK2 (Heun) for the deterministic part, exact on the linear
  terms. The scalar is advanced with the same stage velocities (consistent coupling).
- Smooth exponential filter applied ONCE per step to the scalars only; no clipping (the
  negative fraction is monitored).
"""
from dataclasses import dataclass, field, asdict
import numpy as np
from scipy import fft as sfft


@dataclass
class DNS2Params:
    Lx: float = 4 * np.pi
    Ly: float = 2 * np.pi
    Nx: int = 384
    Ny: int = 192
    nu: float = 2e-3
    mu: float = 0.05              # linear drag
    kappa: float = 2e-3           # Sc = 1
    U: float = 1.0
    dt: float = 0.005
    kf_min: float = 3.0
    kf_max: float = 6.0
    eps: float = 0.005            # injection rate; E_ss ~ eps/(2 mu) -> u_rms ~ 0.3 U (strong). Weak: 0.0012
    sources: tuple = ((0.8, np.pi), (0.8, np.pi - 1.3), (0.8, np.pi + 1.3), (2.0, np.pi + 0.6))
    src_radius: float = 0.1
    sponge_start: float = 0.75    # fraction of Lx
    sponge_amp: float = 5.0
    filt_alpha: float = 36.0
    filt_p: int = 36


class DNS2:
    def __init__(self, p: DNS2Params, seed=0, workers=1):
        self.p, self.rng, self.workers = p, np.random.default_rng(seed), workers
        dx, dy = p.Lx / p.Nx, p.Ly / p.Ny
        self.dx, self.dy = dx, dy
        self.x = np.arange(p.Nx) * dx
        self.y = np.arange(p.Ny) * dy
        X, Y = np.meshgrid(self.x, self.y, indexing="ij")
        kx = 2 * np.pi * np.fft.fftfreq(p.Nx, d=dx)
        ky = 2 * np.pi * np.fft.rfftfreq(p.Ny, d=dy)
        self.KX, self.KY = np.meshgrid(kx, ky, indexing="ij")
        self.K2 = self.KX ** 2 + self.KY ** 2
        self.K2inv = np.where(self.K2 > 0, 1 / np.where(self.K2 > 0, self.K2, 1), 0.0)
        K = np.sqrt(self.K2)
        self.dealias = (np.abs(self.KX) <= (2 / 3) * np.abs(kx).max()) & (self.KY <= (2 / 3) * ky.max())
        kn = np.sqrt((self.KX / np.abs(kx).max()) ** 2 + (self.KY / ky.max()) ** 2)
        self.filt = np.exp(-p.filt_alpha * kn ** p.filt_p) * self.dealias
        self.band = (K >= p.kf_min) & (K <= p.kf_max)
        # forcing normalisation: injection eps = 0.5 * sum over modes of |F_k|^2 / (Nx Ny)^2
        # (Parseval with rfft weights). Computed numerically below.
        self._fscale = 1.0
        F = self._raw_forcing()
        self._fscale = np.sqrt(p.eps / self._injection(F))
        self.E_w = np.exp(-(p.nu * self.K2 + p.mu) * p.dt)
        self.E_c = np.exp(-p.kappa * self.K2 * p.dt)
        self.src_hat = []
        for (sx, sy) in p.sources:
            s = np.exp(-((X - sx) ** 2 + (Y - sy) ** 2) / (2 * p.src_radius ** 2)) / (2 * np.pi * p.src_radius ** 2)
            self.src_hat.append(self._fft(s) * self.filt)
        self.src_hat = np.array(self.src_hat)
        xs = np.clip((X - p.sponge_start * p.Lx) / ((1 - p.sponge_start) * p.Lx), 0, 1)
        self.sponge = p.sponge_amp * xs ** 2
        self.w = np.zeros(self.KX.shape, complex)
        self.c = np.zeros((len(p.sources),) + self.KX.shape, complex)
        self.t = 0.0

    # ---------------------------------------------------------------- transforms
    def _fft(self, f):
        return sfft.rfft2(f, workers=self.workers)

    def _ifft(self, f):
        return sfft.irfft2(f, s=(self.p.Nx, self.p.Ny), workers=self.workers)

    def _raw_forcing(self):
        F = self._fft(self.rng.standard_normal((self.p.Nx, self.p.Ny))) * self.band
        return F * self._fscale

    def _injection(self, F):
        # energy injection rate for white-in-time vorticity forcing: eps = 0.5 <|F_hat/k|^2>
        wts = np.where(self.KY == 0, 1.0, 2.0)
        return 0.5 * np.sum(wts * np.abs(F) ** 2 * self.K2inv) / (self.p.Nx * self.p.Ny) ** 2

    # ---------------------------------------------------------------- dynamics
    def _vel(self, w):
        psi = w * self.K2inv
        return self._ifft(1j * self.KY * psi), self._ifft(-1j * self.KX * psi)

    def _adv(self, fh, ut, v):
        return self._fft(ut * self._ifft(1j * self.KX * fh) + v * self._ifft(1j * self.KY * fh)) * self.dealias

    def _rhs(self, w, c):
        u, v = self._vel(w)
        ut = u + self.p.U
        Nw = -self._adv(w, ut, v)
        Nc = np.empty_like(c)
        for i in range(len(c)):
            ci = self._ifft(c[i])
            Nc[i] = -self._adv(c[i], ut, v) + self.src_hat[i] - self._fft(self.sponge * ci) * self.dealias
        return Nw, Nc

    def step(self):
        dt = self.p.dt
        k1w, k1c = self._rhs(self.w, self.c)
        w1 = self.E_w * (self.w + dt * k1w)
        c1 = self.E_c * (self.c + dt * k1c)
        k2w, k2c = self._rhs(w1, c1)
        self.w = self.E_w * self.w + 0.5 * dt * (self.E_w * k1w + k2w)
        self.c = self.E_c * self.c + 0.5 * dt * (self.E_c * k1c + k2c)
        self.w = (self.w + np.sqrt(dt) * self._raw_forcing()) * self.dealias
        self.c = self.c * self.filt
        self.t += dt

    # ---------------------------------------------------------------- output
    def scalars(self):
        return np.array([self._ifft(ci) for ci in self.c])

    def velocity(self):
        u, v = self._vel(self.w)
        return u + self.p.U, v

    def energy(self):
        u, v = self._vel(self.w)
        return 0.5 * np.mean(u ** 2 + v ** 2)
