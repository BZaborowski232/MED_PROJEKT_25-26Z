from sklearn.naive_bayes import GaussianNB

class NaiveBayesModel:
    def __init__(self):
        self.model = GaussianNB()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
