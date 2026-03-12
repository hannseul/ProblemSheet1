from __future__ import annotations

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass(frozen=True)
class UCBConfig:
    delta: float = 0.1
    seed: Optional[int] = None


class UCB:
    """
    Upper Confidence Bound (UCB) algorithm for stochastic bandits.
    """

    def __init__(self, bandit, config: UCBConfig):

        if not hasattr(bandit, "pull"):
            raise TypeError("bandit must implement pull(arm)")

        if not hasattr(bandit, "cfg") or not hasattr(bandit.cfg, "n_arms"):
            raise TypeError("bandit must expose bandit.cfg.n_arms")

        self.bandit = bandit
        self.cfg = config
        self.n_arms = bandit.cfg.n_arms

        self.rng = np.random.default_rng(config.seed)

        # statistics
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.sums = np.zeros(self.n_arms)

        self.t = 0

    def _empirical_means(self):

        means = np.zeros(self.n_arms)
        played = self.counts > 0

        means[played] = self.sums[played] / self.counts[played]

        return means

    def _ucb_values(self):

        means = self._empirical_means()

        ucb = np.zeros(self.n_arms)

        for a in range(self.n_arms):

            if self.counts[a] == 0:
                return a   # play each arm once first

            bonus = np.sqrt((2 * np.log(self.t)) / self.counts[a])

            ucb[a] = means[a] + bonus

        return int(np.argmax(ucb))
    
    def step(self) -> Tuple[int, float]:

        self.t += 1

        arm = self._ucb_values()

        reward = float(self.bandit.pull(arm))

        # update statistics
        self.counts[arm] += 1
        self.sums[arm] += reward

        return arm, reward
    