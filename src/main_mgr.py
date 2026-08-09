import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os


# Importy modułów przetwarzania
from data_loader import load_data
from preprocessing import preprocess
from feature_engineering import engineer_features
from segmentation import segment_customers, segment_customers_dbscan, suggest_dbscan_eps
from evaluation import evaluate_model
from visualization import visualize_pca, plot_clusters_pca, plot_confusion_matrix_heatmap

# --- Importujemy TYLKO nowe klasyfikatory ---
from classifiers.random_forest import RandomForestModel
from classifiers.xgboost_model import XGBoostModel

def main():
    print("=== ROZPOCZYNAM EKSPERYMENT MAGISTERSKI (RF + XGBoost) ===")
    
    # 1. Wczytanie danych (ścieżka dynamiczna)
    print("\n[1/6] Wczytywanie danych...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "OnlineRetail.csv")
    
    if not os.path.exists(data_path):
        print(f"BŁĄD: Nie znaleziono pliku: {data_path}")
        return

    raw_data = load_data(data_path)
    
    # 2. Preprocessing
    print("\n[2/6] Wstępne przetwarzanie...")
    clean_data = preprocess(raw_data)
    
    # 3. Feature Engineering
    print("\n[3/6] Inżynieria cech...")
    features_df = engineer_features(clean_data)
    print(f"Dane gotowe: {features_df.shape}")

    # 4. Segmentacja
    print("\n[4/6] Segmentacja klientów...")
    
    # K-MEANS
    print("   -> K-Means...")
    df_kmeans, X_scaled = segment_customers(features_df, n_clusters=3)
    visualize_pca(X_scaled, df_kmeans["Segment"], "K-Means Segmentation", "PCA_K-Means_MGR")

# DBSCAN
    print("   -> DBSCAN...")
    suggest_dbscan_eps(X_scaled, k=4)
    df_dbscan = segment_customers_dbscan(df_kmeans, X_scaled, eps=3.0, min_samples=10)

    # --- NOWY KOD DO WIZUALIZACJI PCA DBSCAN ---
    print("\nGenerowanie wizualizacji PCA dla DBSCAN...")
    
    labels_for_pca_dbscan = df_dbscan["Segment_DBSCAN"].apply(
        lambda x: "Outliers" if x == -1 else "Główny klaster"
    )
    
    plot_clusters_pca(X_scaled, labels_for_pca_dbscan)

    # 5. Klasyfikacja
    print("\n[5/6] Przygotowanie danych treningowych...")
    
    # --- POPRAWKA (Fix Data Leakage) ---
    # features_df zawiera teraz kolumny 'Segment' i 'Segment_DBSCAN', 
    # ponieważ funkcje segmentacji dodały je do ramki danych.
    # Musimy je usunąć ze zbioru X, aby model nie uczył się z odpowiedzi.
    columns_to_drop = ["Segment", "Segment_DBSCAN"]
    X = features_df.drop(columns=columns_to_drop, errors='ignore')
    
    # Y to nasze etykiety (Target)
    y = df_kmeans["Segment"]
    
    # Weryfikacja (opcjonalnie wypisz kolumny, by mieć pewność)
    print(f"Cechy treningowe: {list(X.columns)}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
# 6. Modele - TYLKO Random Forest i XGBoost
    print("\n[6/6] Trenowanie modeli (Random Forest & XGBoost)...")

    # Mapowanie nazw segmentów dla macierzy konfuzji
    segment_map = {
        0: "Uśpieni / Odchodzący",     
        1: "VIP / Hurt",      
        2: "Standardowi / Lojalni"          
    }

    models = [
        ("Random Forest", RandomForestModel(n_estimators=100)),
        ("XGBoost", XGBoostModel(n_estimators=100))
    ]

    for name, model in models:
        print(f"\n--- Model: {name} ---")
        model.train(X_train, y_train)
        preds = model.predict(X_test)
        
        evaluate_model(y_test, preds, model_name=name)
        
        # --- NOWY KOD: Generowanie macierzy konfuzji ---
        file_name = f"Matrix_{name.replace(' ', '_')}"
        plot_confusion_matrix_heatmap(
            y_true=y_test, 
            y_pred=preds, 
            labels_map=segment_map, 
            title=f"Macierz - {name}", 
            filename=file_name
        )
        
        # Wykresy ważności cech
        if name == "Random Forest":
            model.get_feature_importance(X.columns)
        elif name == "XGBoost":
            model.get_feature_importance()

if __name__ == "__main__":
    main()