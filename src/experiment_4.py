import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb

# Importy modułów
from data_loader import load_data
from preprocessing import preprocess
from feature_engineering import engineer_features
from segmentation import segment_customers

def main():
    print("=== EKSPERYMENT E4: Cost-Sensitive Learning (XGBoost) ===")
    
    # 1. Wczytanie i przygotowanie danych
    print("\n[1/4] Ładowanie danych i segmentacja bazowa (K-Means)...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "OnlineRetail.csv")
    
    if not os.path.exists(data_path):
        print(f"BŁĄD: Nie znaleziono pliku: {data_path}")
        return

    raw_data = load_data(data_path)
    clean_data = preprocess(raw_data)
    features_df = engineer_features(clean_data)
    
    # K-MEANS - Tworzymy etykiety (Ground Truth)
    df_kmeans, _ = segment_customers(features_df, n_clusters=3)
    
    # Identyfikacja, który numer dostał segment VIP (na bazie największych wydatków)
    vip_segment = df_kmeans.groupby('Segment')['Monetary'].mean().idxmax()
    print(f"-> Zidentyfikowano Segment VIP jako: {vip_segment}")
    
    # --- FIX DATA LEAKAGE (wzorowane na main_mgr.py) ---
    columns_to_drop = ["Segment", "Segment_DBSCAN", "Segment_PAM"]
    X = features_df.drop(columns=columns_to_drop, errors='ignore')
    y = df_kmeans["Segment"]
    
    # Weryfikacja
    print(f"-> Cechy użyte do treningu ({len(X.columns)}): {list(X.columns)}")
    
    # 2. Podział na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # 3. Model 1: Standardowy XGBoost (Z Twoimi hiperparametrami z XGBoostModel!)
    print("\n[2/4] Trening standardowego modelu XGBoost (Baseline)...")
    xgb_standard = xgb.XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=5, 
        random_state=42, 
        eval_metric='mlogloss', 
        n_jobs=-1
    )
    xgb_standard.fit(X_train, y_train)
    y_pred_std = xgb_standard.predict(X_test)
    
    # 4. Model 2: XGBoost z Cost-Sensitive Learning
    print("[3/4] Trening XGBoost z modyfikacją wag (Cost-Sensitive)...")
    
    # Zwykły klient = waga 1. Klient VIP = waga 50
    sample_weights = np.ones(len(y_train))
    sample_weights[y_train == vip_segment] = 50.0
    
    xgb_cost_sensitive = xgb.XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=5, 
        random_state=42, 
        eval_metric='mlogloss', 
        n_jobs=-1
    )
    xgb_cost_sensitive.fit(X_train, y_train, sample_weight=sample_weights)
    y_pred_cs = xgb_cost_sensitive.predict(X_test)
    
    # 5. Generowanie wyników i macierzy konfuzji
    print("\n[4/4] Generowanie i zapisywanie macierzy konfuzji...")
    os.makedirs("Visualizations/MGR", exist_ok=True)
    
    # Definiujemy nazwy dla osi (kolejność musi odpowiadać klasom 0, 1, 2)
    segment_names = ["Uśpieni / Odchodzący", "VIP / Hurt", "Standardowi / Lojalni"]
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7)) # Lekko poszerzamy figurę, żeby napisy się zmieściły
    
    # Macierz standardowa
    cm_std = confusion_matrix(y_test, y_pred_std)
    sns.heatmap(cm_std, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False,
                xticklabels=segment_names, yticklabels=segment_names)
    axes[0].set_title('XGBoost - Wersja Standardowa')
    axes[0].set_xlabel('Przewidywany Segment')
    axes[0].set_ylabel('Prawdziwy Segment')
    # Obrót etykiet dla czytelności
    axes[0].tick_params(axis='x', rotation=0)
    axes[0].tick_params(axis='y', rotation=90)
    
    # Macierz Cost-Sensitive
    cm_cs = confusion_matrix(y_test, y_pred_cs)
    sns.heatmap(cm_cs, annot=True, fmt='d', cmap='Oranges', ax=axes[1], cbar=False,
                xticklabels=segment_names, yticklabels=segment_names)
    axes[1].set_title('XGBoost - Cost-Sensitive (Wysoka kara za pominięcie VIP)')
    axes[1].set_xlabel('Przewidywany Segment')
    axes[1].set_ylabel('Prawdziwy Segment')
    # Obrót etykiet dla czytelności
    axes[1].tick_params(axis='x', rotation=0)
    axes[1].tick_params(axis='y', rotation=90)
    
    plt.tight_layout()
    plt.savefig("Visualizations/MGR/Matrix_E4_Cost_Sensitive_Comparison.png", dpi=300)
    
    print("\n--- ZAPISANO WYKRES: Visualizations/MGR/Matrix_E4_Cost_Sensitive_Comparison.png ---")
    
    # Raporty tekstowe
    print("\n--- Raport: Wersja Standardowa ---")
    print(classification_report(y_test, y_pred_std))
    
    print("\n--- Raport: Cost-Sensitive (Wagi x50 dla VIP) ---")
    print(classification_report(y_test, y_pred_cs))
    
    print("=== ZAKOŃCZONO EKSPERYMENT E4 ===")

if __name__ == "__main__":
    main()