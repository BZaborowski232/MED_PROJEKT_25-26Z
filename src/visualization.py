import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix


def plot_clusters_pca(X, clusters, title="Wizualizacja Segmentów (PCA)"):

    """
        Rzutuje wielowymiarowe dane na 2D za pomocą PCA i rysuje klastry.
    """

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    df_pca = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
    
    # Używamy .values, aby zignorować indeksy i uniknąć NaN
    df_pca['Segment'] = clusters.values
    
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
        Rysuje wykres słupkowy ważności cech.
    """

    if not hasattr(model, "feature_importances_"):
        print("Model nie posiada atrybutu feature_importances_")
        return

    importances = model.feature_importances_
    indices = importances.argsort()[::-1]
    
    sorted_names = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    plt.figure(figsize=(10, 6))
    
    # Rysowanie wykresu
    plot = sns.barplot(
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


def plot_confusion_matrix_heatmap(y_true, y_pred, labels_map=None, title="Macierz Konfuzji"):

    """
        Rysuje macierz konfuzji. Obsługuje zamianę numerów na nazwy (labels_map).
    """

    cm = confusion_matrix(y_true, y_pred)
    
    # Przygotowanie etykiet osi
    if labels_map:
        # Sortujemy klucze (0, 1, 2), żeby pobrać nazwy w dobrej kolejności
        labels = [labels_map[i] for i in sorted(labels_map.keys())]
    else:
        labels = "auto"

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=False,
        xticklabels=labels, yticklabels=labels
    )
    plt.xlabel('Przewidziany Segment')
    plt.ylabel('Prawdziwy Segment')
    plt.title(title)
    plt.show()