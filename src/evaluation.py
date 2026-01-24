from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def evaluate_model(y_true, y_pred, model_name="Model"):
    
    """
        Wyświetla tekstowy raport klasyfikacji.
        Nie generuje wykresu, aby uniknąć duplikacji w main.py.
    """

    print(f"\n--- Ocena modelu: {model_name} ---")
    
    # Podstawowa celność
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")
    
    # Pełny raport tekstowy (Precision, Recall, F1)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    # Macierz tekstowa (dla szybkiego podglądu w konsoli)
    print("Confusion Matrix (text):")
    print(confusion_matrix(y_true, y_pred))