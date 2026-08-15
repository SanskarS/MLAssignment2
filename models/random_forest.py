from sklearn.ensemble import RandomForestClassifier

from models.common import evaluate, split_and_scale


def run(X, y, **params):
    n_estimators = params.get('n_estimators', 100)
    random_state = params.get('random_state', 7)
    X_train, X_test, y_train, y_test, sc = split_and_scale(X, y, scaler=None)
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    result = evaluate(model, X_test, y_test)
    result['scaler'] = sc
    return result
