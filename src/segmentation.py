import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

"""
    Dokonuje segmentacji klientów metodą K-Means.
    Zabezpiecza przed wartościami odstającymi poprzez logarytmizację cech skośnych.
    Zwraca: (df z kolumną Segment, macierz X_scaled użyta do modelu)
"""


def segment_customers(df: pd.DataFrame, n_clusters: int = 3) -> tuple[pd.DataFrame, np.ndarray]:
    
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