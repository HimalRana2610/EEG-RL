from sklearn.preprocessing import StandardScaler
from data_loader import load_deap
from segmentation import segment_eeg
from feature_extraction import extract_features
from prototype_learning import learn_prototypes
from train_agent import train

def main():
    print("Loading dataset...")
    eeg, labels = load_deap("data/s01.dat")
    trial = eeg[0]

    print("Segmenting EEG...")
    segments = segment_eeg(trial)

    print("Extracting features...")
    features = extract_features(segments)

    print("Normalizing features...")
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    print("Learning emotion prototypes...")
    kmeans, prototypes = learn_prototypes(features)

    print("Training RL agent...")
    train(features, prototypes)
    print("Training finished!")

if __name__ == "__main__":
    main()