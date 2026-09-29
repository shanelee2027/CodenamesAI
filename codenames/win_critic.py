"""The critic for win_actor_critic: V(position) = P(the side to move wins),
against the learned listener, from the board's actual words (docs/log.md,
"win_actor_critic: design").

**Inputs.** Nothing from the listener. Roles are relative to the side to
move, plus a flag saying whether that side is our agent (the two sides are
different players, so a position is worth different amounts to each).

- Per unrevealed word: its role and the word features the policy uses.
- Per pair of unrevealed words, for each similarity source in `SOURCES`:
  take the legal clue that best covers both, meaning the highest weaker
  similarity of the two (`level`). Then record that clue's margin over the
  nearest OTHER unrevealed word of each role, and how many other words of
  each role sit within `NEAR` sd below or above it (`crowd`).

These keep magnitudes and gaps, not just order. Two words far above
everything else give a large level and large margins. Two words barely
above a crowd give a small level, small margins and big crowd counts, even
when the order is the same. Keeping each source separate keeps
disagreement between the embeddings and the association data visible.

**Network.** Attention over the unrevealed words. The pair features enter
as per-head attention biases and as part of each message. The readout
pools per role (mean and max), adds role counts and the agent flag, and
gives one logit.

**Targets.** TD(lambda) along real games, in `lambda_returns`. The next
position belongs to the other side, so its value from the mover's side is
1 - V(next). The last position's target is the real result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from codenames.board import Role
from codenames.clue_policy import BOARD_SLOTS, ROLE_ID, ROLE_ORDER, DeviceFeatures, PolicyFeatures

SOURCES = ("z_glove", "z_numberbatch", "z_wiki2vec", "z_glove840", "z_fasttext",
           "swow_fwd1", "swow_rev1", "pmi", "wn_wup")
NEAR = 0.5
CLIP = 8.0
PER_SOURCE = 1 + 2 * len(ROLE_ORDER)          # level, margin per role, crowd per role
N_EDGE = PER_SOURCE * len(SOURCES)
IU = np.triu_indices(BOARD_SLOTS, 1)          # the 300 unordered pairs


class CriticFeatures:
    """Pair features for positions, computed on the GPU from the policy's
    feature table (`DeviceFeatures`), which must already be loaded."""

    def __init__(self, feats: PolicyFeatures, gf: DeviceFeatures):
        import torch

        self.gf = gf
        self.src = torch.as_tensor([feats.pair_names.index(s) for s in SOURCES], device=gf.device)
        self.iu = torch.as_tensor(np.stack(IU), device=gf.device)

    def edges(self, words, roles, present, chunk: int = 4):
        """(B, 300, N_EDGE) float16 for the unordered pairs (i < j); zero for a
        pair with a revealed word. `words`, `roles` (relative to the side to
        move) and `present` are (B, 25) arrays."""
        import torch

        gf = self.gf
        w = torch.as_tensor(np.asarray(words), device=gf.device)
        rol = torch.as_tensor(np.asarray(roles), device=gf.device)
        pres = torch.as_tensor(np.asarray(present), device=gf.device)
        out = []
        for s in range(0, len(w), chunk):
            out.append(self._edges(w[s: s + chunk], rol[s: s + chunk], pres[s: s + chunk]))
        return torch.cat(out)

    def _edges(self, w, rol, pres):
        import torch

        B, N = w.shape
        legal = ~self.gf.illegal[w].any(1)                                   # (B, P)
        X = self.gf.pair[w][..., self.src].float().permute(0, 3, 1, 2)      # (B, S, N, P)
        X = X.masked_fill(~legal[:, None, None, :], -1e4)
        S = X.shape[1]
        level = torch.empty(B, S, N, N, device=w.device)
        cstar = torch.empty(B, S, N, N, dtype=torch.long, device=w.device)
        for i in range(N):
            level[:, :, i], cstar[:, :, i] = torch.minimum(X[:, :, i: i + 1], X).max(-1)
        # Every word's similarity to each pair's best clue: (B, S, i, j, w).
        at = X.gather(-1, cstar.view(B, S, 1, N * N).expand(-1, -1, N, -1))  # (B, S, w, N*N)
        at = at.view(B, S, N, N, N).permute(0, 1, 3, 4, 2)
        eye = torch.eye(N, dtype=torch.bool, device=w.device)
        other = pres[:, None, None, None, :] & ~eye[None, None, :, None, :] & ~eye[None, None, None, :, :]
        feats = [level.clamp(-CLIP, CLIP)]
        crowd = []
        for r in range(len(ROLE_ORDER)):
            m = other & (rol == r)[:, None, None, None, :]
            nearest = at.masked_fill(~m, -1e4).amax(-1)
            margin = torch.where(m.any(-1), level - nearest, torch.full_like(level, CLIP))
            feats.append(margin.clamp(-CLIP, CLIP))
            crowd.append(((at >= level[..., None] - NEAR) & m).sum(-1).float() / 4.0)
        f = torch.stack(feats + crowd, dim=-1)                                # (B, S, N, N, 9)
        f = f.permute(0, 2, 3, 1, 4).reshape(B, N, N, S * PER_SOURCE)
        both = pres[:, :, None] & pres[:, None, :]
        f = f * both[..., None]
        return f[:, self.iu[0], self.iu[1]].half()


def full_edges(e300):
    """(B, 300, F) unordered pairs -> (B, 25, 25, F) symmetric, zero diagonal."""
    import torch

    B, _, F = e300.shape
    out = torch.zeros(B, BOARD_SLOTS, BOARD_SLOTS, F, dtype=e300.dtype, device=e300.device)
    i, j = IU
    out[:, i, j] = e300
    out[:, j, i] = e300
    return out


def build_critic(n_word: int, n_edge: int = N_EDGE, d: int = 64, heads: int = 4, layers: int = 3):
    import torch
    from torch import nn

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.qkv = nn.Linear(d, 3 * d)
            self.bias = nn.Linear(d, heads)
            self.ev = nn.Linear(d, d)
            self.out = nn.Linear(d, d)
            self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

        def forward(self, h, e, mask):
            B, N, _ = h.shape
            dh = d // heads
            q, k, v = self.qkv(self.norm1(h)).view(B, N, 3, heads, dh).unbind(2)
            att = torch.einsum("bihc,bjhc->bhij", q, k) / dh ** 0.5 + self.bias(e).permute(0, 3, 1, 2)
            att = att.masked_fill(~mask[:, None, None, :], float("-inf")).softmax(-1)
            msg = torch.einsum("bhij,bjhc->bihc", att, v).reshape(B, N, d)
            msg = msg + torch.einsum("bhij,bijhc->bihc", att, self.ev(e).view(B, N, N, heads, dh)).reshape(B, N, d)
            h = h + self.out(msg)
            return h + self.ff(self.norm2(h))

    class WinCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = dict(n_word=n_word, n_edge=n_edge, d=d, heads=heads, layers=layers)
            self.node = nn.Sequential(nn.Linear(n_word + len(ROLE_ORDER), d), nn.GELU(), nn.Linear(d, d))
            self.edge = nn.Sequential(nn.Linear(n_edge, d), nn.GELU(), nn.Linear(d, d))
            self.layers = nn.ModuleList(Layer() for _ in range(layers))
            n_read = 2 * len(ROLE_ORDER) * d + 2 * len(ROLE_ORDER) + 1
            self.read = nn.Sequential(nn.LayerNorm(n_read), nn.Linear(n_read, 128), nn.GELU(), nn.Linear(128, 1))

        def forward(self, word, roles, present, edges, agent):
            """word (B, 25, Fw), roles (B, 25) relative to the side to move,
            present (B, 25) bool, edges (B, 300, F), agent (B,) bool -> logit
            of P(side to move wins), (B,)."""
            onehot = nn.functional.one_hot(roles, len(ROLE_ORDER)).float()
            h = self.node(torch.cat([word.float(), onehot], -1))
            e = self.edge(full_edges(edges).float())
            for layer in self.layers:
                h = layer(h, e, present)
            parts, counts, has = [], [], []
            for r in range(len(ROLE_ORDER)):
                m = (roles == r) & present
                n = m.sum(1, keepdim=True)
                parts.append((h * m[..., None]).sum(1) / n.clamp(min=1))
                parts.append(h.masked_fill(~m[..., None], -1e4).amax(1) * (n > 0))
                counts.append(n.float() / 9.0)
                has.append((n > 0).float())
            x = torch.cat(parts + counts + has + [agent.float()[:, None]], -1)
            return self.read(x).squeeze(-1)

    return WinCritic()


def save_critic(net, path: Path, meta: dict) -> None:
    import torch

    torch.save({"config": net.config, "state": net.state_dict(), "meta": meta}, path)


def load_critic(path: Path, device: str = "cpu"):
    import torch

    blob = torch.load(path, map_location=device, weights_only=False)
    net = build_critic(**blob["config"]).to(device)
    net.load_state_dict(blob["state"])
    net.eval()
    return net, blob.get("meta", {})


def lambda_returns(v_next_mover: np.ndarray, last_won: float, lam: float) -> np.ndarray:
    """TD(lambda) targets for one game's positions s_0..s_{T-1}, each from its
    own mover's side. `v_next_mover[t]` is V(s_{t+1}) as s_{t+1}'s mover sees
    it (t < T-1). `last_won` is 1 if s_{T-1}'s mover won. lam = 1 is the
    Monte-Carlo result; lam = 0 is one-step TD."""
    T = len(v_next_mover) + 1
    g = np.empty(T)
    g[-1] = last_won
    for t in range(T - 2, -1, -1):
        g[t] = (1 - lam) * (1 - v_next_mover[t]) + lam * (1 - g[t + 1])
    return g


def relative_roles(team_a_roles: np.ndarray, mover: str) -> np.ndarray:
    """Role ids from team A's view -> from `mover`'s view (OWN and OPPONENT
    swap for team B)."""
    if mover == "A":
        return team_a_roles
    own, opp = ROLE_ID[Role.OWN], ROLE_ID[Role.OPPONENT]
    out = team_a_roles.copy()
    out[team_a_roles == own] = opp
    out[team_a_roles == opp] = own
    return out
