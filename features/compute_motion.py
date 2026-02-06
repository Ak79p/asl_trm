import numpy as np

def compute_velocity(x):
    v = np.zeros_like(x)
    v[1:] = x[1:] - x[:-1]
    return v

def compute_acceleration(v):
    a = np.zeros_like(v)
    a[1:] = v[1:] - v[:-1]
    return a
