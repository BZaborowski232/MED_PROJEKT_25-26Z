from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

def segment_customers(rfm: pd.DataFrame, n_clusters: int = 3):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    segments = kmeans.fit_predict(rfm_scaled)

    rfm["Segment"] = segments
    return rfm, rfm_scaled
