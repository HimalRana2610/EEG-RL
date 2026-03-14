import numpy as np

def segment_eeg(eeg, fs=128, window=1):
    segment_length = fs * window
    segments = []

    for i in range(0, eeg.shape[1] - segment_length, segment_length):

        seg = eeg[:, i:i+segment_length]
        segments.append(seg)

    return np.array(segments)