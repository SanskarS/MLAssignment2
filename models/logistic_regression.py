from sklearn.linear_model import LogisticRegression

from models.common import evaluate, split_and_scale


def run(X, y, **params):
    max_iter = params.get('max_iter', 1000)
    random_state = params.get('random_state', 7)
    X_train, X_test, y_train, y_test, sc = split_and_scale(X, y, scaler='standard')
    model = LogisticRegression(random_state=random_state, max_iter=max_iter)
    model.fit(X_train, y_train)
    result = evaluate(model, X_test, y_test)
    result['scaler'] = sc
    return result
