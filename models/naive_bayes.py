from sklearn.naive_bayes import GaussianNB, MultinomialNB

from models.common import evaluate, split_and_scale


def run(X, y, **params):
    variant = params.get('variant', 'gaussian')
    if variant == 'multinomial':
        X_train, X_test, y_train, y_test, sc = split_and_scale(X, y, scaler='minmax')
        model = MultinomialNB()
    else:
        X_train, X_test, y_train, y_test, sc = split_and_scale(X, y, scaler='standard')
        model = GaussianNB()
    model.fit(X_train, y_train)
    result = evaluate(model, X_test, y_test)
    result['scaler'] = sc
    return result
