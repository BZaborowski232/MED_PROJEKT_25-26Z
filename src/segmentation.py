import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


def segment_customers(df: pd.DataFrame, n_clusters: int = 3) -> tuple[pd.DataFrame, np.ndarray]:
    
    """
        Dokonuje segmentacji klientów metodą K-Means.
        Zabezpiecza przed wartościami odstającymi poprzez logarytmizację cech skośnych.
        Zwraca: (df z kolumną Segment, macierz X_scaled użyta do modelu)
    """

    # Kopia, aby nie modyfikować oryginału w funkcji
    data = df.copy()

    # Wybieramy tylko kolumny numeryczne do segmentacji
    # (Pomijamy ewentualne kolumny tekstowe, jeśli by się pojawiły)
    features_col = data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Kopia do przekształceń
    X = data[features_col].copy()

    # --- 1. Logarytmizacja (Obsługa wartości odstających) ---
    # Obsługa problemu "klientów o bardzo wysokich wartościach".
    # Logarytmizacja (np. log(x+1)) niweluje wpływ skrajnych wartości (outliers).
    # Zamiast 100 vs 1 000 000, mamy np. 4.6 vs 13.8.
    
    skewed_features = [
        "Recency", "Frequency", "Monetary", 
        "TotalQuantity", "AvgTicketValue", "AvgDaysBetweenPurchases",
        "MaxTicketValue", "ReturnCount"
    ]
    
    for col in skewed_features:
        if col in X.columns:
            # Zabezpieczenie: logarytm nie przyjmuje wartości ujemnych.
            # Jeśli mamy ujemne (np. specyficzny zwrot), przesuwamy rozkład.
            min_val = X[col].min()
            if min_val < 0:
                X[col] = X[col] - min_val  # przesunięcie, by min było 0
            
            # log1p to log(x + 1), radzi sobie z zerami
            X[col] = np.log1p(X[col])

    # --- 2. Skalowanie (StandardScaler) ---
    # K-Means wymaga, by każda cecha miała ten sam wpływ na odległość.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- 3. K-Means ---
    # Ustawiamy n_init=10 dla stabilności wyników
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    # Przypisanie segmentu do oryginalnej ramki danych
    df["Segment"] = kmeans.labels_

    # Zwracamy df (z segmentami) oraz X_scaled (gotowe wejście dla klasyfikatorów)
    return df, X_scaled



    ## --- DBSCAN ---

def suggest_dbscan_eps(X_scaled, k=4):
 
    """
        Pomocnicza funkcja do pracy magisterskiej: Rysuje wykres k-distance,
        aby znaleźć optymalne 'eps' metodą łokcia.
    """

    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(X_scaled)
    distances, indices = neighbors_fit.kneighbors(X_scaled)
    distances = np.sort(distances[:, k-1], axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.title("Wykres k-distance (Metoda łokcia dla DBSCAN)")
    plt.xlabel("Punkty posortowane wg odległości")
    plt.ylabel(f"Odległość do {k}-tego sąsiada")
    plt.grid(True)
    plt.savefig("Visualizations/MGR/DBSCAN_Eps_Estimation.png")
    print("Wykres pomocniczy dla DBSCAN zapisano w Visualizations/")


def segment_customers_dbscan(df: pd.DataFrame, X_scaled, eps=0.5, min_samples=5):

    """
        Implementacja DBSCAN.
        Zwraca DataFrame z kolumną 'Segment_DBSCAN'.
        Uwaga: DBSCAN generuje etykietę -1 dla szumu (outliers).
    """

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)
    
    df["Segment_DBSCAN"] = labels
    
    # Informacja badawcza
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    
    print(f"\n[DBSCAN] Szacowana liczba klastrów: {n_clusters_}")
    print(f"[DBSCAN] Liczba punktów szumu (outliers): {n_noise_}")
    
    return df