"""Learned grid posteriors B0 and B (docs/research/spec_posterior_networks.md, section 3; schematic
figures/schematic_posterior_net.png).

  encoder:  token (7) -> MLP 7-64-64 -> GRU 64 ;  g_{k-1} = GRU state before reading k (g_{-1} = 0)
  decoder:  l_jk = f((p_k - x_j)/s, d_k, z_k, g_{k-1}),  f: 68 -> 64 -> 64 -> 1 (ReLU), shared over (j, k)
  B0:       log q_n(j) = log_softmax_j( sum_{k<=n} l_jk )                      (uniform prior)
  B:        log q_n(j) = log_softmax_j( log b^phys_n(j) + alpha * sum_{k<=n} l_jk ),  alpha learned, init 0.1
Velocity sensing (E48, vel=True): the token gains the scaled local velocity (u - U, v) at the sensor (9 numbers), and
the reading part of f takes (d_k, z_k, u_k, v_k, g_{k-1}).
The first layer of f is split into an offset part (per j, k) and a reading/state part (per k), so the (j, k) tensor
is formed only after one linear map.
E51 (cnn=C > 0, for comparison only): a residual CNN on the 91 x 61 logit map, 5 dilated 5x5 conv layers of C channels
(dilations 1, 2, 4, 8, 1) and a zero-initialised 3x3 output conv, input (log b^phys, alpha S) for B or (S) for B0.
E53 (attn=True, for comparison only): f has a second output e_jk, and each cell weights its readings by attention,
  S_jn = (n+1) sum_{k<=n} softmax_{k<=n}(e_jk) l_jk,
i.e. learned per-cell measurement weights with mean 1 (the counterpart of kappa_k). e is bounded by 10 tanh(e/10) and
its output row is zero-initialised, so the network starts as the plain sum. Offline evaluation only (the incremental
update in postnet_closedloop.py assumes the plain sum).
E54 (bottleneck=r > 0): the token enters the GRU encoder only through a rank-r linear map, learned, or fixed to the
leading sensitivity directions of scripts/checks/gru_sensitivity.py when proj (a (F, r) array) is given. The decoder
still receives (d_k, z_k[, u_k]) and the offsets directly, so the bottleneck restricts only what the GRU can remember.
E54 (nomem=True): g_{k-1} = 0 for every k, so l_jk = f(offset, d_k, z_k[, u_k], 0): independent per-reading terms with
no memory and no step clock (bottleneck r = 0 still leaves the GRU a clock, g depends on k only).
"""
import torch
import torch.nn as nn

TOK = 7
OFF_SCALE = (1.0, 0.5)


class PostNet(nn.Module):
    def __init__(self, cx, cy, physics=False, hidden=64, vel=False, cnn=0, attn=False, bottleneck=0, proj=None,
                 nomem=False):
        super().__init__()
        self.nomem = nomem
        self.rd_cols = [0, 1, 7, 8] if vel else [0, 1]                      # token columns that enter f directly
        self.register_buffer("cells", torch.stack([torch.as_tensor(cx, dtype=torch.float32),
                                                   torch.as_tensor(cy, dtype=torch.float32)], -1))   # (J, 2)
        self.register_buffer("scale", torch.tensor(OFF_SCALE))
        self.physics = physics
        F = TOK + 2 * vel
        self.proj = None
        if proj is not None:                              # E54: fixed projection onto the sensitivity directions
            self.proj = nn.Linear(F, proj.shape[1], bias=False)
            self.proj.weight.data.copy_(torch.as_tensor(proj, dtype=torch.float32).T)
            self.proj.weight.requires_grad_(False)
            F = proj.shape[1]
        elif bottleneck:                                  # E54: learned rank-r projection
            self.proj = nn.Linear(F, bottleneck, bias=False)
            F = bottleneck
        self.enc = nn.Sequential(nn.Linear(F, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.f_off = nn.Linear(2, hidden)                 # first layer of f, offset part
        self.f_rd = nn.Linear(len(self.rd_cols) + hidden, hidden, bias=False)   # first layer of f, (d_k, z_k, g_{k-1}) part
        self.attn = attn
        self.f_rest = nn.Sequential(nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1 + attn))
        if attn:                                          # E53: attention score row starts at zero (uniform weights)
            with torch.no_grad():
                self.f_rest[-1].weight[1].zero_(); self.f_rest[-1].bias[1].zero_()
        if physics:
            self.alpha = nn.Parameter(torch.tensor(0.1))
        self.cnn = None
        if cnn:                                           # E51: residual CNN on the logit map (cell-to-cell mixing)
            self.nx, self.ny = len(torch.unique(self.cells[:, 0])), len(torch.unique(self.cells[:, 1]))
            cin, layers = 1 + physics, []
            for i, dil in enumerate((1, 2, 4, 8, 1)):
                layers += [nn.Conv2d(cin if i == 0 else cnn, cnn, 5, padding=2 * dil, dilation=dil), nn.GELU()]
            last = nn.Conv2d(cnn, 1, 3, padding=1)
            nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)    # starts as the network without the CNN
            self.cnn = nn.Sequential(*layers, last)

    def refine(self, logits, maps):
        """logits + CNN(maps); maps: list of (B, N, J) log maps, each shifted to max 0, clipped at -30, scaled by 1/10."""
        B, N, J = logits.shape
        x = torch.stack([(m - m.amax(-1, keepdim=True)).clamp(min=-30) / 10 for m in maps], 2)
        x = x.reshape(B * N, len(maps), self.nx, self.ny)
        return logits + self.cnn(x).reshape(B, N, J)

    def terms(self, tok, pos):
        """l_jk for a batch: tok (B, K, 7 or 9), pos (B, K, 2) -> (B, K, J); with attn, (B, K, J, 2) = (l_jk, e_jk)."""
        B, K, _ = tok.shape
        if self.nomem:                                                       # E54: no memory, no clock
            g_prev = tok.new_zeros(B, K, self.gru.hidden_size)
        else:
            g, _ = self.gru(self.enc(tok if self.proj is None else self.proj(tok)))   # (B, K, H): state after reading k
            g_prev = torch.cat([torch.zeros_like(g[:, :1]), g[:, :-1]], 1)   # state before reading k
        rd = self.f_rd(torch.cat([tok[..., self.rd_cols], g_prev], -1))                # (B, K, H)
        off = (pos[:, :, None, :] - self.cells[None, None]) / self.scale     # (B, K, J, 2)
        h = self.f_off(off) + rd[:, :, None, :]                              # (B, K, J, H)
        out = self.f_rest(h)
        return out if self.attn else out[..., 0]                             # (B, K, J)

    def forward(self, tok, pos, ns, lphys=None):
        """log q at the steps ns (list of move indices; reading k = 0..n enter step n): (B, len(ns), J)."""
        if self.attn:                                                        # E53: causal per-cell softmax over k
            le = self.terms(tok, pos).float()
            l, e = le[..., 0], 10 * torch.tanh(le[..., 1] / 10)
            w = torch.exp(e - e.amax(1, keepdim=True))                       # >= exp(-20): no underflow in fp32
            cnt = torch.arange(1, l.shape[1] + 1, device=l.device, dtype=l.dtype)[None, :, None]
            S = (cnt * torch.cumsum(w * l, 1) / torch.cumsum(w, 1))[:, ns]
        else:
            S = torch.cumsum(self.terms(tok, pos).float(), 1)[:, ns]         # (B, N, J); fp32 sum (bf16 diverges, E45)
        logits = S if not self.physics else lphys + self.alpha * S
        if self.cnn is not None:
            logits = self.refine(logits.float(), [S] if not self.physics else [lphys, self.alpha * S])
        return torch.log_softmax(logits, -1)
