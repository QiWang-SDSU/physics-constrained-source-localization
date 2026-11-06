"""Truth world backed by a DNS2 dataset (E8: planners built on the surrogate, evaluated against DNS).

The DNS plume i (source at (xs_d, ys_d)) is translated so its source sits at the task's true source
location in harness coordinates. That is exact for a statistically homogeneous flow with uniform mean
wind, which DNS2 is (up to the near-source region). Readings are bilinear interpolation of the
recorded sub-grid field c_sub[t, i] in space and nearest-record in time. The same log-normal sensor
noise and detection threshold as the model world are applied. Outside the recorded x-range
(upstream of the source or in the sponge) the reading is 0, i.e. censored.
"""
import numpy as np
from scipy.ndimage import map_coordinates


class DNSData:
    def __init__(self, path):
        z = np.load(path)
        import json
        self.P = json.loads(str(z["params"]))
        self.c = z["c_sub"]                       # (Nt, n_src, nx, ny) float16
        self.sx, self.sy = z["sub_x"], z["sub_y"]
        self.dt = float(z["times"][1] - z["times"][0])
        self.Ly = self.P["Ly"]

    def add_velocity(self, path):
        """Velocity sub-grid of the same run (E44 v2 replay, scripts/dns2_velocity.py), for DNSWorld.velocity."""
        v = np.load(path)
        assert np.allclose(v["sub_x"], self.sx) and v["u_sub"].shape[0] == self.c.shape[0]
        self.u, self.v = v["u_sub"], v["v_sub"]


class DNSWorld:
    def __init__(self, p, src, rng, data: DNSData, plume, t0_idx):
        self.p, self.src, self.rng = p, np.asarray(src, float), rng
        self.d, self.i, self.t0 = data, plume, t0_idx
        xs_d, ys_d = data.P["sources"][plume]
        self.off = np.array([xs_d - src[0], ys_d - src[1]])

    def measure(self, x, y, t):
        """Bilinear interpolation on the four neighbouring sub-grid nodes (periodic in y). Same value as
        measure_ref (map_coordinates on the whole slice) up to float32 rounding, about 100x faster (E28)."""
        d = self.d
        X, Y = x + self.off[0], (y + self.off[1]) % d.Ly
        k = self.t0 + int(round(t / d.dt))
        nx, ny = len(d.sx), len(d.sy)
        fi = (X - d.sx[0]) / (d.sx[1] - d.sx[0])
        if fi < 0 or fi > nx - 1 or k >= d.c.shape[0]:
            c = 0.0
        else:
            fj = (Y - d.sy[0]) / (d.sy[1] - d.sy[0])
            i0, j0 = int(fi), int(fj)
            a, b = fi - i0, fj - j0
            i1, j0, j1 = min(i0 + 1, nx - 1), j0 % ny, (j0 + 1) % ny
            f = d.c[k, self.i]
            c = float((1 - a) * ((1 - b) * float(f[i0, j0]) + b * float(f[i0, j1]))
                      + a * ((1 - b) * float(f[i1, j0]) + b * float(f[i1, j1])))
        if c <= 0:
            return None
        z = np.log(c) + np.sqrt(self.p.nug) * self.rng.standard_normal()
        return z if z >= np.log(self.p.m_min) else None

    def velocity(self, x, y, t):
        """Local (u - U, v) at the sensor, noise-free (E48), from data.u / data.v (DNSData.add_velocity); same bilinear
        interpolation and record as measure. Outside the recorded x-range it is 0 (= the mean wind), as in
        scripts/postnet_add_velocity.py."""
        d = self.d
        X, Y = x + self.off[0], (y + self.off[1]) % d.Ly
        k = self.t0 + int(round(t / d.dt))
        nx, ny = len(d.sx), len(d.sy)
        fi = (X - d.sx[0]) / (d.sx[1] - d.sx[0])
        if fi < 0 or fi > nx - 1 or k >= d.c.shape[0]:
            return np.zeros(2, np.float32)
        fj = (Y - d.sy[0]) / (d.sy[1] - d.sy[0])
        i0, j0 = int(fi), int(fj)
        a, b = fi - i0, fj - j0
        i1, j0, j1 = min(i0 + 1, nx - 1), j0 % ny, (j0 + 1) % ny
        out = []
        for f in (d.u[k], d.v[k]):
            f = f.astype(np.float32)
            out.append((1 - a) * ((1 - b) * f[i0, j0] + b * f[i0, j1]) + a * ((1 - b) * f[i1, j0] + b * f[i1, j1]))
        return np.array([out[0] - d.P["U"], out[1]], np.float32)

    def measure_ref(self, x, y, t):
        d = self.d
        X, Y = x + self.off[0], (y + self.off[1]) % d.Ly
        k = self.t0 + int(round(t / d.dt))
        fi = (X - d.sx[0]) / (d.sx[1] - d.sx[0])
        if fi < 0 or fi > len(d.sx) - 1 or k >= d.c.shape[0]:
            c = 0.0
        else:
            fj = (Y - d.sy[0]) / (d.sy[1] - d.sy[0])
            fld = d.c[k, self.i].astype(np.float32)
            fld = np.concatenate([fld, fld[:, :1]], axis=1)       # periodic wrap in y
            c = float(map_coordinates(fld, [[fi], [fj]], order=1)[0])
        if c <= 0:
            return None
        z = np.log(c) + np.sqrt(self.p.nug) * self.rng.standard_normal()
        return z if z >= np.log(self.p.m_min) else None
