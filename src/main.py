# Import funkcji do wczytywania i przetwarzania danych
from data_loader import load_data
from preprocessing import preprocess
from feature_engineering import engineer_features 
from segmentation import segment_customers

# Import modeli klasyfikacyjnych
from classifiers.decision_tree import DecisionTreeModel
from classifiers.naive_bayes import NaiveBayesModel
from evaluation import evaluate_model

# Import narzędzia do podziału danych
from sklearn.model_selection import train_test_split
import pandas as pd


def main():
    print(">>> KROK 1: Wczytywanie danych")
    df = load_data("../data/OnlineRetail.csv")
    print(f"Wczytano {len(df)} wierszy.")

    print("\n>>> KROK 2: Preprocessing")
    # Preprocess nie usuwa zwrotów, tylko czyści braki i typy danych
    df = preprocess(df)
    print("Dane wyczyszczone.")

    print("\n>>> KROK 3: Feature Engineering (Nowe Cechy)")
    # Obliczamy bogaty zestaw cech (RFM + Zwroty + Czasowe + Produktowe)
    customers_df = engineer_features(df)
    print(f"Utworzono profil dla {len(customers_df)} klientów.")
    print("Przykładowe nowe cechy:", list(customers_df.columns[:5]), "...")

    print("\n>>> KROK 4: Segmentacja K-Means")
    # Funkcja zajmuje się logarytmizacją i skalowaniem
    customers_df, X_scaled = segment_customers(customers_df, n_clusters=3)
    
    print("Liczebność segmentów:")
    print(customers_df["Segment"].value_counts().sort_index())

    # --- Przygotowanie do klasyfikacji ---
    # X - macierz cech (używamy przeskalowanej wersji X_scaled dla lepszych wyników modeli)
    # y - etykiety (segmenty z K-Means)
    X = X_scaled
    y = customers_df["Segment"]

    # Podział na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print("\n>>> KROK 5: Klasyfikacja i Ewaluacja")
    
    # Decision Tree
    print("\n--- Decision Tree ---")
    dt = DecisionTreeModel()
    dt.train(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    evaluate_model(y_test, y_pred_dt, "Decision Tree")

    # Naive Bayes
    print("\n--- Naive Bayes ---")
    nb = NaiveBayesModel()
    nb.train(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    evaluate_model(y_test, y_pred_nb, "Naive Bayes")

if __name__ == "__main__":
    main()