import numpy as np

def norm_pts(x):
    x -= np.mean(x, axis=0)
    dists = np.linalg.norm(x, axis=1)
    return x / np.max(dists)
