# features/build_tensor.py

import numpy as np
# from temporal_sampling import uniform_sample
# from compute_motion import compute_velocity, compute_acceleration

from features.temporal_sampling import uniform_sample
from features.compute_motion import compute_velocity, compute_acceleration

EXPECTED_T = 48
EXPECTED_J = 48

def _sanity_check(X):
    """
    Hard fail if tensor is malformed.
    """
    if X.ndim != 3:
        raise ValueError(f"Tensor must be 3D (T,J,D), got {X.ndim}D")

    T, J, D = X.shape

    if T != EXPECTED_T:
        raise ValueError(f"Expected T={EXPECTED_T}, got {T}")

    if J != EXPECTED_J:
        raise ValueError(f"Expected J={EXPECTED_J}, got {J}")

    if not np.isfinite(X).all():
        raise ValueError("Tensor contains NaN or Inf")

    # motion must exist (non-static video)
    motion_energy = np.mean(np.abs(X[1:] - X[:-1]))
    if motion_energy < 1e-6:
        raise ValueError("Tensor has near-zero motion (static or failed extraction)")

def build_feature_tensor(kps, T=EXPECTED_T):
    """
    Build (T, J, D) tensor with automatic sanity checks.
    """
    # (T_raw, J, 2)
    if kps.ndim != 3 or kps.shape[1:] != (EXPECTED_J, 2):
        raise ValueError(f"Invalid keypoint shape: {kps.shape}")

    kps = uniform_sample(kps, T)

    vel = compute_velocity(kps)
    acc = compute_acceleration(vel)

    # Hand-relative normalization
    lh_center = kps[:, 0:21].mean(axis=1, keepdims=True)
    rh_center = kps[:, 21:42].mean(axis=1, keepdims=True)

    rel = kps.copy()
    rel[:, 0:21] -= lh_center
    rel[:, 21:42] -= rh_center

    # Concatenate: pos | vel | acc | rel
    X = np.concatenate([kps, vel, acc, rel], axis=-1).astype(np.float32)

    _sanity_check(X)
    return X

def build_feature_tensor_continuous(kps):
    """
    Build full-length (T_full, J, D) tensor for continuous videos.
    No temporal resampling.
    No fixed-length sanity enforcement.
    """

    if kps.ndim != 3 or kps.shape[1:] != (EXPECTED_J, 2):
        raise ValueError(f"Invalid keypoint shape: {kps.shape}")

    # Do NOT resample
    kps_full = kps

    vel = compute_velocity(kps_full)
    acc = compute_acceleration(vel)

    # Hand-relative normalization
    lh_center = kps_full[:, 0:21].mean(axis=1, keepdims=True)
    rh_center = kps_full[:, 21:42].mean(axis=1, keepdims=True)

    rel = kps_full.copy()
    rel[:, 0:21] -= lh_center
    rel[:, 21:42] -= rh_center

    X = np.concatenate([kps_full, vel, acc, rel], axis=-1).astype(np.float32)

    # ---- Light sanity check (no fixed T constraint) ----
    if X.ndim != 3:
        raise ValueError(f"Tensor must be 3D (T,J,D), got {X.ndim}D")

    if not np.isfinite(X).all():
        raise ValueError("Tensor contains NaN or Inf")

    motion_energy = np.mean(np.abs(X[1:] - X[:-1]))
    if motion_energy < 1e-6:
        raise ValueError("Tensor has near-zero motion")

    return X