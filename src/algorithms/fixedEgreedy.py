from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

PullFn = Callable[[int], float]     # takes arm index and returns reward (bandit environment)

def _update_running_mean(q_hat: np.ndarray, counts: np.ndarray, a: int, x: float) -> None:
    """
    Incremental mean update:
      Q_n = Q_{n-1} + (1/n) (R_n - Q_{n-1})
    """
    counts[a] += 1
    n = counts[a]       #counts how many times arm a has been played
    q_hat[a] = q_hat[a] + (x - q_hat[a]) / n        #q_hat current estimation of mean of arm a

def _random_argmax(values: np.ndarray, rng: np.random.Generator) -> int:
    """Argmax with uniform tie-breaking.""" # if multiple arms have the same max value, choose among them uniformly at random
    m = np.max(values)
    candidates = np.flatnonzero(values == m)
    return int(rng.choice(candidates))

def run_epsilon_greedy(
    pull: PullFn,
    K: int,
    n_steps: int,
    *,
    epsilon: float,
    q0: Optional[Sequence[float]] = None,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    ε-greedy with fixed epsilon
      - With prob ε: explore uniformly random arm
      - With prob 1-ε: exploit argmax Q_hat
      - Update Q_hat by incremental mean

    Returns dict with:
      actions, rewards, q_hat, counts
    """
    if not (0.0 <= epsilon <= 1.0):
        raise ValueError("epsilon must be in [0,1].")

    rng = np.random.default_rng(seed)

    q_hat = np.zeros(K, dtype=float) if q0 is None else np.array(q0, dtype=float).copy()
    if q_hat.shape != (K,):
        raise ValueError(f"q0 must have shape ({K},), got {q_hat.shape}.")

    counts = np.zeros(K, dtype=int)
    actions = np.zeros(n_steps, dtype=int)
    rewards = np.zeros(n_steps, dtype=float)

    for t in range(1, n_steps + 1):
        u = rng.random() #ziehe u aus uniform(0,1)
        if u < epsilon: #dann wähle einen arm uniform zufällig
            a = int(rng.integers(low=0, high=K))  # uniform exploration
        else:
            a = _random_argmax(q_hat, rng)        # greedy exploitation

        x = float(pull(a))

        actions[t - 1] = a
        rewards[t - 1] = x
        _update_running_mean(q_hat, counts, a, x)

    return {"actions": actions, "rewards": rewards, "q_hat": q_hat, "counts": counts}
