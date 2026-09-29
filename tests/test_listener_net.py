import numpy as np
import torch

from codenames.listener_net import ListenerNet


def _inputs(B=3, N=7, F=5, W=40, S=5, D=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    vecs = torch.nn.functional.normalize(torch.randn(W, S, D, generator=g), dim=-1).numpy().astype(np.float16)
    x = torch.randn(B, N, F, generator=g)
    clue = torch.randint(0, W, (B,), generator=g)
    cand = torch.randint(0, W, (B, N), generator=g)
    avail = torch.ones(B, N, dtype=torch.bool)
    avail[:, -2:] = False
    base = torch.randn(B, N, generator=g)
    return vecs, x, clue, cand, avail, base


def test_scores_follow_the_words_when_candidates_are_shuffled():
    vecs, x, clue, cand, avail, base = _inputs()
    net = ListenerNet(vecs, x.shape[-1], d=16, heads=2, rank=4, dropout=0.0).eval()
    perm = torch.randperm(x.shape[1])
    with torch.no_grad():
        a = net(x, clue, cand, avail)
        b = net(x[:, perm], clue, cand[:, perm], avail[:, perm])
    assert torch.allclose(a[:, perm], b, atol=1e-5, equal_nan=True)


def test_unavailable_words_score_minus_infinity_and_do_not_affect_the_rest():
    vecs, x, clue, cand, avail, base = _inputs()
    net = ListenerNet(vecs, x.shape[-1], d=16, heads=2, rank=4, dropout=0.0).eval()
    with torch.no_grad():
        a = net(x, clue, cand, avail)
        x2 = x.clone()
        x2[:, -2:] = 100.0  # change only the unavailable rows
        b = net(x2, clue, cand, avail)
    assert torch.isinf(a[:, -2:]).all()
    assert torch.allclose(a[:, :-2], b[:, :-2], atol=1e-5)


def test_residual_starts_exactly_at_the_base():
    vecs, x, clue, cand, avail, base = _inputs()
    net = ListenerNet(vecs, x.shape[-1], d=16, heads=2, rank=4, residual=True).eval()
    with torch.no_grad():
        s = net(x, clue, cand, avail, base)
    assert torch.equal(s[avail], base[avail])


def test_features_only_variant_ignores_the_vectors():
    vecs, x, clue, cand, avail, base = _inputs()
    net = ListenerNet(vecs, x.shape[-1], d=16, heads=2, rank=4, dropout=0.0, use_vectors=False).eval()
    with torch.no_grad():
        a = net(x, clue, cand, avail)
        b = net(x, (clue + 1) % vecs.shape[0], (cand + 1) % vecs.shape[0], avail)
    assert torch.equal(a, b)
