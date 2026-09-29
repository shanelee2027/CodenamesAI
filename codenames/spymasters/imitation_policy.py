"""A clue policy trained to imitate the incumbent: the control for
`gptoss_reward_policy`.

The network (codenames/clue_policy.py) reads raw clue-word evidence and each
word's role and outputs pi(clue | board) over the incumbent's clue pool, plus
an outcome head whose expected reward per k picks the number. It was trained
(scripts/pipeline/train_imitation_policy.py) on the incumbent's picks and its
expected rewards for its shortlisted clues on simulated training boards. At
play time nothing of the listener or expected_words runs: one forward pass,
the most probable legal clue, and the k with the highest expected reward.

It is registered for one reason: `gptoss_reward_policy` starts from these
weights, so the gap between the two is what the RL stage added, and the gap
between this and the incumbent is what the policy form costs.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from codenames.clue_policy import (
    AUX_FILE,
    FEATURES_FILE,
    PolicyFeatures,
    expected_rewards,
    load_policy,
    masked_log_k,
    masked_log_policy,
    outcome_log_probs,
    stack_inputs,
)
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor
from codenames.spymasters.base import Spymaster, TurnContext


class ImitationPolicySpymaster(Spymaster):
    MODEL = "policy_imitation.pt"

    def __init__(self, torch_threads: int = 2, *, cache_dir: Path = DEFAULT_CACHE_DIR,
                 model_path: Path | None = None):
        """`torch_threads` caps intra-op threads for this process: the arena
        runs several workers, each playing many games at once."""
        import torch

        torch.set_num_threads(torch_threads)
        self.feats = PolicyFeatures.load(cache_dir)
        self.net, self.meta = load_policy(Path(model_path or Path(cache_dir) / self.MODEL))
        # The arena's worker threads share one instance; torch inference is
        # thread-safe but running a dozen 10k-clue passes at once only
        # thrashes the cores.
        self._lock = threading.Lock()

    @classmethod
    def model_files(cls, params: dict) -> list[Path]:
        cache_dir = Path(params.get("cache_dir", DEFAULT_CACHE_DIR))
        return [Path(params.get("model_path") or cache_dir / cls.MODEL),
                cache_dir / AUX_FILE, cache_dir / FEATURES_FILE]

    def scores(self, board) -> tuple[np.ndarray, np.ndarray]:
        """(log pi over the pool, -inf off the legal clues; per clue and k,
        (n_pool, max_number), -inf beyond K_max: the expected reward for an
        outcome-head policy, log pi(k | clue) for a k-head one. Either way
        the number played is its argmax.)"""
        import torch

        b = self.feats.encode(board)
        pair, word, roles, present, legal = stack_inputs(self.feats, [b])
        roles_t, present_t = torch.as_tensor(roles), torch.as_tensor(present)
        kmax = ((roles_t == 0) & present_t).sum(1).clamp(max=self.net.max_number)
        with self._lock, torch.no_grad():
            logits, outcome, _ = self.net(torch.as_tensor(pair, dtype=torch.float32),
                                          torch.as_tensor(word), roles_t, present_t)
            logp = masked_log_policy(logits, torch.as_tensor(legal))[0]
            if self.net.head == "k":
                er = masked_log_k(outcome, kmax)[0]
            else:
                er = expected_rewards(outcome_log_probs(outcome, kmax), kmax)[0]
        return logp.numpy(), er.numpy()

    def top_clues(self, ctx: TurnContext, sims: SimilarityTensor, k: int) -> list[tuple[str, int, float]]:
        logp, er = self.scores(ctx.board)
        finite = np.flatnonzero(np.isfinite(logp))
        order = finite[np.argsort(-logp[finite], kind="stable")][:k]
        return [(self.feats.clue_words[i], int(np.argmax(er[i])) + 1, float(logp[i])) for i in order]
