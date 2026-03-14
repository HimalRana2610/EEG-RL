from sklearn.cluster import KMeans

def learn_prototypes(features, k=5):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features)
    prototypes = kmeans.cluster_centers_
    return kmeans, prototypes