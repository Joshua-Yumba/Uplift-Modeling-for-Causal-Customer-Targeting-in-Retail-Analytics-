from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

def perform_clustering(rfm, n_clusters):
    """
    Perform K-Means clustering on RFM data.
    Returns cluster labels.
    """
    rfm_features = rfm[['recency', 'frequency', 'monetary_value']]
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_features)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    return kmeans.fit_predict(rfm_scaled)

def get_elbow_data(rfm):
    """
    Compute inertia for elbow method.
    Returns list of inertia values for 1 to 7 clusters.
    """
    rfm_features = rfm[['recency', 'frequency', 'monetary_value']]
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_features)
    inertia = []
    for n in range(1, 8):
        kmeans = KMeans(n_clusters=n, n_init=10, random_state=42)
        kmeans.fit(rfm_scaled)
        inertia.append(kmeans.inertia_)
    return inertia

def perform_auto_gmm_segmentation(rfm, max_components: int = 7):
    """
    Select number of clusters automatically using BIC with Gaussian Mixture Models.
    Returns cluster labels (ints).
    """
    rfm_features = rfm[['recency', 'frequency', 'monetary_value']]
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_features)
    best_gmm = None
    best_bic = float('inf')
    for n in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
        gmm.fit(rfm_scaled)
        bic = gmm.bic(rfm_scaled)
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
    labels = best_gmm.predict(rfm_scaled)
    return labels