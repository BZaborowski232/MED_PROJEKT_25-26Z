import os
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

def run_experiment_1():
    print("=== EKSPERYMENT E1: Redukcja wektora cech do surowych danych ===")
    
    # 1. Wczytanie i przygotowanie danych (identycznie jak w main_mgr.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "OnlineRetail.csv")
    
    if not os.path.exists(data_path):
        print(f"BŁĄD: Nie znaleziono pliku: {data_path}")
        return

    print("[1/4] Ładowanie i inżynieria cech...")
    raw_data = load_data(data_path)
    clean_data = preprocess(raw_data)
    features_df = engineer_features(clean_data)
    
    # 2. Segmentacja K-Means (aby uzyskać etykiety 'y')
    print("[2/4] Generowanie etykiet K-Means (y)...")
    df_kmeans, _ = segment_customers(features_df, n_clusters=3)
    y = df_kmeans["Segment"]
    
    # 3. REDUKCJA WEKTORA CECH (Cel eksperymentu)
    print("[3/4] Redukcja wektora cech (Feature Selection)...")
    
    # Wybieramy tylko absolutnie "surowe" atrybuty, łatwe do policzenia w SQL
    raw_features = [
        "Recency", 
        "Frequency", 
        "Monetary", 
        "TotalQuantity", 
        "UniqueProducts"
    ]
    
    X = features_df[raw_features].copy()
    print(f"-> Zredukowano wektor z {features_df.shape[1]} do {X.shape[1]} cech.")
    print(f"-> Użyte cechy: {list(X.columns)}")
    
    # 4. Klasyfikacja XGBoost na zredukowanym zbiorze
    print("\n[4/4] Trenowanie i ewaluacja XGBoost...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    model_xgb = XGBoostModel(n_estimators=100)
    model_xgb.train(X_train, y_train)
    preds = model_xgb.predict(X_test)
    
    # Ewaluacja i wyniki
    evaluate_model(y_test, preds, model_name="XGBoost (E1 - Zredukowane Cechy)")
    
    # Opcjonalnie: Zapis macierzy konfuzji dla eksperymentu
    segment_map = {
        0: "Uśpieni / Odchodzący",     
        1: "VIP / Hurt",      
        2: "Standardowi / Lojalni"          
    }
    plot_confusion_matrix_heatmap(
        y_true=y_test, 
        y_pred=preds, 
        labels_map=segment_map, 
        title="Macierz - E1 (Zredukowane Cechy)", 
        filename="Matrix_E1_Reduced_Features"
    )
    
    print("\n=== ZAKOŃCZONO EKSPERYMENT E1 ===")

if __name__ == "__main__":
    run_experiment_1()