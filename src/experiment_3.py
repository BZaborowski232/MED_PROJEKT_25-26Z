import os
import numpy as np
import pandas as pd
from sklearn_extra.cluster import KMedoids

# Importy modułów
from data_loader import load_data
from preprocessing import preprocess
from feature_engineering import engineer_features
from segmentation import segment_customers
from visualization import visualize_pca


# PAM Przypisał etykiety klastrów następująco:
# Segment 0: Standardowi / Lojalni
# Segment 1: Uśpieni / Odchodzący
# Segment 2: VIP / HURT


def run_experiment_3():
    print("=== EKSPERYMENT E3: Porównanie K-Means vs PAM (K-Medoids) ===")
    
    # 1. Wczytanie i przygotowanie danych
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "OnlineRetail.csv")
    
    if not os.path.exists(data_path):
        print(f"BŁĄD: Nie znaleziono pliku: {data_path}")
        return

    print("[1/3] Ładowanie danych i przygotowanie przestrzeni cech...")
    raw_data = load_data(data_path)
    clean_data = preprocess(raw_data)
    features_df = engineer_features(clean_data)
    
    # Używamy tej samej funkcji segment_customers, żeby mieć zmapowany K-Means do porównania
    df_kmeans, X_scaled = segment_customers(features_df, n_clusters=3)
    
    print("\n[2/3] Uruchamianie algorytmu PAM (K-Medoids)...")
    # Inicjalizacja PAM z taką samą liczbą klastrów (k=3)
    pam = KMedoids(n_clusters=3, random_state=42, method='pam')
    pam_labels = pam.fit_predict(X_scaled)
    
    # Dodanie etykiet PAM do ramki danych
    features_df["Segment_PAM"] = pam_labels
    
    # Porównanie liczebności klastrów
    print("\n--- Porównanie liczebności segmentów ---")
    print("K-Means liczebność klastrów:")
    print(df_kmeans["Segment"].value_counts().sort_index())
    
    print("\nPAM (K-Medoids) liczebność klastrów:")
    print(features_df["Segment_PAM"].value_counts().sort_index())
    
    # 3. Wizualizacja PCA dla PAM
    print("\n[3/3] Generowanie wizualizacji PCA dla algorytmu PAM...")
    visualize_pca(
        X_scaled=X_scaled, 
        labels=features_df["Segment_PAM"], 
        title="Wizualizacja Segmentów - Algorytm PAM (K-Medoids)", 
        filename="PCA_PAM_Comparison"
    )
    
    # Analiza średnich wartości dla klastrów PAM (celach weryfikacji biznesowej)
    print("\n--- Charakterystyka segmentów PAM (Średnie wartości cech) ---")
    pam_stats = features_df.groupby("Segment_PAM")[["Recency", "Frequency", "Monetary", "TotalQuantity"]].mean()
    print(pam_stats)
    
    print("\n=== ZAKOŃCZONO EKSPERYMENT E3 ===")

if __name__ == "__main__":
    run_experiment_3()