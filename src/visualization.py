import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import numpy as np
import os

# ==========================================
# SEKCJA 1: FUNKCJE STARE (Dla main.py / MED)
# Używają plt.show()
# ==========================================

def plot_clusters_pca(X, clusters, title="Wizualizacja Segmentów (PCA)"):
    """
    Stara funkcja dla main.py - wyświetla okno z wykresem.
    """
    pca = PCA(n_components=2)
    # Zabezpieczenie: jeśli X to DataFrame, bierzemy values
    if hasattr(X, "values"):
        X_pca = pca.fit_transform(X)
    else:
        X_pca = pca.fit_transform(X)
    
    df_pca = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
    
    # Obsługa formatu klastrów (używamy .values)
    if hasattr(clusters, "values"):
        df_pca['Segment'] = clusters.values
    else:
        df_pca['Segment'] = clusters
    
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x='PCA1', y='PCA2', 
        hue='Segment', 
        data=df_pca, 
        palette='viridis',
        s=60, alpha=0.7
    )
    plt.title(title)
    plt.xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0]:.2%} variancji)")
    plt.ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1]:.2%} variancji)")
    plt.legend(title='Segment')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_feature_importance(model, feature_names):
    """
    Stara funkcja dla main.py - wyświetla okno.
    """
    if not hasattr(model, "feature_importances_"):
        print("Model nie posiada atrybutu feature_importances_")
        return

    importances = model.feature_importances_
    indices = importances.argsort()[::-1]
    
    sorted_names = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=sorted_importances, 
        y=sorted_names, 
        hue=sorted_names, 
        legend=False, 
        palette="magma"
    )
    plt.title("Wpływ cech na decyzję modelu (Feature Importance)")
    plt.xlabel("Waga cechy")
    plt.yticks(fontsize=9) 
    plt.show()

# ==========================================
# SEKCJA 2: FUNKCJE NOWE (Dla main_mgr.py / MGR)
# Zapisują pliki do folderu Visualizations/MGR
# ==========================================

def visualize_pca(X_scaled, labels, title, filename):
    """
    Nowa funkcja dla main_mgr.py - zapisuje plik PNG.
    """
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    
    df_pca = pd.DataFrame(data=components, columns=['PCA1', 'PCA2'])
    
    # --- POPRAWKA: Używamy .values, aby uniknąć problemów z indeksami pandas ---
    # To naprawia błąd "Ignoring palette because no hue variable has been assigned"
    if hasattr(labels, "values"):
        df_pca['Segment'] = labels.values
    else:
        df_pca['Segment'] = labels
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='PCA1', y='PCA2', 
        hue='Segment', 
        palette='viridis', 
        data=df_pca, 
        s=100, 
        alpha=0.8
    )
    plt.title(f"{title}\n(Wyjaśniona wariancja: {pca.explained_variance_ratio_.sum():.2%})")
    plt.grid(True, alpha=0.3)
    
    save_path = f"Visualizations/MGR/{filename}.png"
    # Zabezpieczenie: tworzenie folderu
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path)
    plt.close()
    print(f"Zapisano wykres PCA: {save_path}")

def visualize_feature_importance(importances, feature_names, title, filename):
    """
    Nowa funkcja pomocnicza - zapisuje plik PNG.
    Przyjmuje czyste dane (importances), a nie obiekt modelu.
    """
    feature_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=feature_imp,
        y=feature_imp.index,
        palette="magma",
        hue=feature_imp.index,  
        legend=False
        )
    plt.title(title)
    plt.xlabel("Waga cechy")
    plt.tight_layout()
    
    save_path = f"Visualizations/MGR/{filename}.png"
    # Zabezpieczenie: tworzenie folderu
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path)
    plt.close()
    print(f"Zapisano wykres ważności cech: {save_path}")

def plot_confusion_matrix_heatmap(y_true, y_pred, labels_map=None, title="Macierz Konfuzji", filename=None):
    """
    Funkcja uniwersalna - obsługuje i wyświetlanie (stary main) i zapisywanie (nowy main).
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if labels_map:
        labels = [labels_map[i] for i in sorted(labels_map.keys())]
    else:
        labels = sorted(list(set(y_true) | set(y_pred)))

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('Prawdziwa klasa')
    plt.xlabel('Przewidziana klasa')
    
    if filename:
        save_path = f"Visualizations/MGR/{filename}.png"
        # Zabezpieczenie: tworzenie folderu
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.savefig(save_path)
        plt.close()
        print(f"Zapisano macierz konfuzji: {save_path}")
    else:
        plt.show()