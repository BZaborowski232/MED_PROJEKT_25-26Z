from data_loader import load_data
from preprocessing import preprocess
from feature_engineering import engineer_features 
from segmentation import segment_customers
from classifiers.decision_tree import DecisionTreeModel
from classifiers.naive_bayes import NaiveBayesModel
from evaluation import evaluate_model
from visualization import plot_clusters_pca, plot_feature_importance, plot_confusion_matrix_heatmap
from sklearn.model_selection import train_test_split
import pandas as pd

def main():
    print(">>> KROK 1: Wczytywanie danych")
    df = load_data("../data/OnlineRetail.csv")
    print(f"Wczytano {len(df)} wierszy.")

    print("\n>>> KROK 2: Preprocessing")
    df = preprocess(df)

    print("\n>>> KROK 3: Feature Engineering")
    customers_df = engineer_features(df)
    print(f"Liczba klientów: {len(customers_df)}")

    print("\n>>> KROK 4: Segmentacja K-Means")
    customers_df, X_scaled = segment_customers(customers_df, n_clusters=3)
    
    # --- ANALIZA SEGMENTÓW (Żeby nadać im nazwy) ---
    print("\n--- Charakterystyka Segmentów (Średnie wartości) ---")
    # Wybieramy kluczowe cechy do podglądu
    stats = customers_df.groupby("Segment")[["Recency", "Frequency", "Monetary", "TotalQuantity"]].mean()
    print(stats)
    
    # MAPOWANIE NAZW SEGMENTÓW
    # Przykład logiczny (do weryfikacji po uruchomieniu):
    # Często K-means sortuje lub losuje, więc spójrz na output konsoli:
    # - Segment z dużym Monetary/Freq -> "VIP / Lojalni"
    # - Segment z dużym Recency -> "Uśpieni / Odchodzący"
    # - Segment środkowy -> "Standardowi"
    
    segment_map = {
        0: "Uśpieni / Odchodzący",     
        1: "VIP / Hurt",      
        2: "Standardowi / Lojalni"          
    }
    
    print(f"\nPrzypisane nazwy: {segment_map}")

    # --- Przygotowanie danych ---
    X = X_scaled
    y = customers_df["Segment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print("\n>>> KROK 5: Klasyfikacja i Wizualizacja")
    
    # --- Decision Tree ---
    print("\n--- Decision Tree ---")
    dt = DecisionTreeModel()
    dt.train(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    evaluate_model(y_test, y_pred_dt, "Decision Tree")
    
    # Macierz z nazwami
    plot_confusion_matrix_heatmap(y_test, y_pred_dt, labels_map=segment_map, title="Macierz - Decision Tree")
    
    # Ważność cech (z mniejszą czcionką)
    feature_names = customers_df.drop(columns=["Segment"]).select_dtypes(include=['number']).columns.tolist()
    plot_feature_importance(dt.model, feature_names)

    # --- Naive Bayes ---
    print("\n--- Naive Bayes ---")
    nb = NaiveBayesModel()
    nb.train(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    evaluate_model(y_test, y_pred_nb, "Naive Bayes")
    
    # Macierz z nazwami
    plot_confusion_matrix_heatmap(y_test, y_pred_nb, labels_map=segment_map, title="Macierz - Naive Bayes")

    # --- Wizualizacja PCA (z nazwami) ---
    print("\nGenerowanie wizualizacji PCA...")
    # Mapujemy kolumnę Segment na nazwy dla wykresu PCA
    labels_for_pca = customers_df["Segment"].map(segment_map)
    plot_clusters_pca(X_scaled, labels_for_pca)
    
if __name__ == "__main__":
    main()