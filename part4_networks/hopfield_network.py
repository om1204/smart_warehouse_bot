# part4_networks/hopfield.py
# Hopfield Network implementation per §6.1.
# NumPy only, no NN library.

import numpy as np


def generate_pattern_1() -> np.ndarray:
    """Pattern 1: Cross"""
    p = np.full((5, 5), -1)
    p[:, 2] = 1
    p[2, :] = 1
    return p.flatten()


def generate_pattern_2() -> np.ndarray:
    """Pattern 2: Square"""
    p = np.full((5, 5), -1)
    p[1:4, 1:4] = 1
    p[2, 2] = -1
    return p.flatten()


def generate_pattern_3() -> np.ndarray:
    """Pattern 3: Diagonal"""
    p = np.full((5, 5), -1)
    for i in range(5):
        p[i, i] = 1
    return p.flatten()


def generate_pattern_4() -> np.ndarray:
    """Pattern 4: Horizontal Stripes"""
    p = np.full((5, 5), -1)
    p[0, :] = 1
    p[2, :] = 1
    p[4, :] = 1
    return p.flatten()


def generate_pattern_5() -> np.ndarray:
    """Pattern 5: Vertical Stripes"""
    p = np.full((5, 5), -1)
    p[:, 0] = 1
    p[:, 2] = 1
    p[:, 4] = 1
    return p.flatten()


def generate_pattern_6() -> np.ndarray:
    """Pattern 6: Four Corners"""
    p = np.full((5, 5), -1)
    p[0, 0] = 1
    p[0, 4] = 1
    p[4, 0] = 1
    p[4, 4] = 1
    return p.flatten()


def train(patterns: list[np.ndarray]) -> np.ndarray:
    """
    Hebbian learning rule to construct the weight matrix.
    Diagonal elements are set to zero.
    """
    if not patterns:
        raise ValueError("No patterns provided")

    n_features = len(patterns[0])
    W = np.zeros((n_features, n_features))

    for p in patterns:
        p = p.reshape(-1, 1)
        W += np.dot(p, p.T)

    np.fill_diagonal(W, 0)
    # Average by number of neurons
    W /= n_features
    return W


def recall(W: np.ndarray, pattern: np.ndarray, max_iters: int = 100) -> np.ndarray:
    """
    Asynchronous update rule for recall.
    """
    n_features = len(pattern)
    state = pattern.copy()

    for _ in range(max_iters):
        changed = False
        # Asynchronous update: pick random order
        indices = np.random.permutation(n_features)
        for i in indices:
            old_val = state[i]
            net_input = np.dot(W[i, :], state)
            new_val = 1 if net_input >= 0 else -1
            if new_val != old_val:
                state[i] = new_val
                changed = True

        if not changed:
            break

    return state


def energy(W: np.ndarray, pattern: np.ndarray) -> float:
    """
    Calculate the energy of a given pattern in the Hopfield network.
    E = -0.5 * sum_i(sum_j(W_ij * S_i * S_j))
    """
    return -0.5 * np.dot(pattern.T, np.dot(W, pattern))


def corrupt(pattern: np.ndarray, flip_fraction: float) -> np.ndarray:
    """
    Corrupt a pattern by randomly flipping a fraction of its bits.
    """
    corrupted = pattern.copy()
    n_features = len(pattern)
    n_flips = int(n_features * flip_fraction)

    indices = np.random.choice(n_features, n_flips, replace=False)
    corrupted[indices] *= -1
    return corrupted
