import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

class RandomForestModel:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1  # -1 to użycie wszystkich dostępnych rdzeni CPU, jakbym ustawił liczbę to ta konkretna liczba rdzeni bedzie dzialac
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def get_feature_importance(self, feature_names):
        
        importances = self.model.feature_importances_
        feature_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=feature_imp, y=feature_imp.index)
        plt.title("Ważność cech - Random Forest")
        plt.xlabel("Waga")
        plt.ylabel("Cechy")
        
        save_path = "Visualizations/MGR/Feature_Importance_RF.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.tight_layout()
        plt.savefig("Visualizations/MGR/Feature_Importance_RF.png")
        print("Zapisano wykres ważności cech dla Random Forest.")
        
        return feature_imp