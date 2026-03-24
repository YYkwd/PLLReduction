"""KNN classifier wrapper (thin layer over sklearn)."""

from sklearn.neighbors import KNeighborsClassifier


class KNNClassifier:
    def __init__(self, params=None):
        params = params or {}
        self.n_neighbors = params.get('n_neighbors', 5)
        self._clf = KNeighborsClassifier(n_neighbors=self.n_neighbors)

    def fit(self, X, y):
        self._clf.fit(X, y)
        return self

    def predict(self, X):
        return self._clf.predict(X)
