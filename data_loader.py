import pickle
import numpy as np

def load_deap(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    eeg = data['data'][:, :32, :]
    labels = data['labels']

    return eeg, labels