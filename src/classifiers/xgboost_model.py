import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class XGBoostModel:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42):
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric='mlogloss',
            n_jobs=-1   # -1 to użycie wszystkich dostępnych rdzeni CPU, jakbym ustawił liczbę to ta konkretna liczba rdzeni bedzie dzialac
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def get_feature_importance(self):
        # XGBoost ma wbudowaną metodę do plotowania
        plt.figure(figsize=(10, 8))
        xgb.plot_importance(self.model, max_num_features=15)
        plt.title("Ważność cech - XGBoost")
        plt.tight_layout()
        plt.savefig("Visualizations/Feature_Importance_XGBoost.png")
        print("Zapisano wykres ważności cech dla XGBoost.")