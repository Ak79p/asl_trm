import numpy as np

def uniform_sample(sequence, T=48):
    N = len(sequence)
    if N == 0:
        raise ValueError("Empty sequence")

    idxs = np.linspace(0, N - 1, T).astype(int)
    return sequence[idxs]
