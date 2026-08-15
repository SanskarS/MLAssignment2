from sklearn.neighbors import KNeighborsClassifier

from models.common import evaluate, split_and_scale


def run(X, y, **params):
    n_neighbors = params.get('n_neighbors', 5)
    X_train, X_test, y_train, y_test = split_and_scale(X, y, scaler='standard')
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return evaluate(model, X_test, y_test)
