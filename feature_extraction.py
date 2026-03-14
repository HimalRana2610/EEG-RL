import numpy as np

def differential_entropy(segment):
    var = np.var(segment, axis=1)
    de = 0.5 * np.log(2 * np.pi * np.e * var)
    return de


def extract_features(segments):
    features = []
    for seg in segments:
        features.append(differential_entropy(seg))

    return np.array(features)