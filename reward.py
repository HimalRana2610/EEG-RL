import numpy as np

def compute_reward(feature, prototypes):
    dists = np.linalg.norm(prototypes - feature, axis=1)
    nearest = np.argmin(dists)
    intra = dists[nearest]
    other_dists = np.delete(dists, nearest)
    inter = np.min(other_dists)
    center_reward = 1 / (1 + intra)
    inter_intra_reward = np.exp(-(intra / inter))
    reward = center_reward + inter_intra_reward
    return reward