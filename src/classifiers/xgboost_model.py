import xgboost as xgb
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score


class XGBoostModel:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42):
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            eval_metric='mlogloss',
            n_jobs=-1   # -1 to użycie wszystkich dostępnych rdzeni CPU, jakbym ustawił liczbę to ta konkretna liczba rdzeni bedzie dzialac
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def get_feature_importance(self):
        plt.figure(figsize=(10, 8))
        xgb.plot_importance(self.model, max_num_features=15, height=0.5)
        plt.title("Ważność cech - XGBoost")
        plt.tight_layout()
        
        # Zapis do MGR z tworzeniem folderu
        save_path = "Visualizations/MGR/Feature_Importance_XGBoost.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.savefig(save_path)
        plt.close() # Zamknięcie figury (chociaż xgb.plot może tworzyć własną, to close() jest bezpieczne)
        print(f"Zapisano wykres ważności cech: {save_path}")