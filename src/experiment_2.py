import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Importy modułów
from data_loader import load_data
from preprocessing import preprocess
from feature_engineering import engineer_features
from segmentation import segment_customers
from evaluation import evaluate_model
from visualization import plot_confusion_matrix_heatmap
from classifiers.xgboost_model import XGBoostModel

def run_experiment_2():
    print("=== EKSPERYMENT E2: Symulacja degradacji danych (Awarie systemu) ===")
    
    # 1. Wczytanie i przygotowanie danych
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "OnlineRetail.csv")
    
    if not os.path.exists(data_path):
        print(f"BŁĄD: Nie znaleziono pliku: {data_path}")
        return

    print("[1/3] Ładowanie danych i generowanie pełnego wektora cech...")
    raw_data = load_data(data_path)
    clean_data = preprocess(raw_data)
    features_df = engineer_features(clean_data)
    
    # Segmentacja K-Means
    df_kmeans, _ = segment_customers(features_df, n_clusters=3)
    
    # Przygotowanie zmiennych
    columns_to_drop = ["Segment", "Segment_DBSCAN"]
    X = features_df.drop(columns=[col for col in columns_to_drop if col in features_df.columns])
    y = df_kmeans["Segment"]
    
    # Podział danych (70% trening - czyste dane, 30% test - zostaną uszkodzone)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # 2. Trenowanie modelu na danych IDEALNYCH
    print("\n[2/3] Trenowanie modelu XGBoost na idealnych danych treningowych...")
    model_xgb = XGBoostModel(n_estimators=100)
    model_xgb.train(X_train, y_train)
    
    # 3. SYMULACJA DEGRADACJI DANYCH (TESTOWYCH)
    print("\n[3/3] Wprowadzanie uszkodzeń do zbioru testowego...")
    X_test_corrupted = X_test.copy()
    
    # Uszkodzenie A: Awaria systemu zwrotów (brak danych o zwrotach)
    X_test_corrupted["ReturnCount"] = 0
    X_test_corrupted["ReturnedUniqueProducts"] = 0
    print(" -> Wyzerowano atrybuty zwrotów: 'ReturnCount', 'ReturnedUniqueProducts'")
    
    # Uszkodzenie B: Utrata danych behawioralnych (NaN) dla 10% klientów
    np.random.seed(42)
    mask = np.random.rand(len(X_test_corrupted)) < 0.1
    X_test_corrupted.loc[mask, "AvgDaysBetweenPurchases"] = np.nan
    print(f" -> Zastąpiono 'AvgDaysBetweenPurchases' wartością NaN dla {mask.sum()} rekordów.")
    
    # 4. Ewaluacja na uszkodzonym zbiorze
    print("\n--- Rozpoczynam predykcję na zdegradowanych danych ---")
    preds = model_xgb.predict(X_test_corrupted)
    
    evaluate_model(y_test, preds, model_name="XGBoost (E2 - Zdegradowane Dane)")
    
    # Zapis macierzy konfuzji
    segment_map = {
        0: "Uśpieni / Odchodzący",     
        1: "VIP / Hurt",      
        2: "Standardowi / Lojalni"          
    }
    plot_confusion_matrix_heatmap(
        y_true=y_test, 
        y_pred=preds, 
        labels_map=segment_map, 
        title="Macierz - E2 (Degradacja Danych)", 
        filename="Matrix_E2_Data_Degradation"
    )
    
    print("\n=== ZAKOŃCZONO EKSPERYMENT E2 ===")

if __name__ == "__main__":
    run_experiment_2()