from typing import Optional

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

RANDOM_STATE = 0

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'model': model,
        'accuracy': accuracy_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr'),
        'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'mcc': matthews_corrcoef(y_test, y_pred),
        'report': classification_report(y_true=y_test, y_pred=y_pred, zero_division=0),
        'confusion': confusion_matrix(y_test, y_pred),
    }


def split_and_scale(X, y, scaler: Optional[str] = 'standard', test_size=0.25, random_state=RANDOM_STATE):
    y = np.ravel(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    if scaler == 'standard':
        sc = StandardScaler()
    elif scaler == 'minmax':
        sc = MinMaxScaler()
    else:
        sc = None
    if sc is not None:
        # Scale based on train feature data only
        X_train = sc.fit_transform(X_train)
        # Only transform the test feature data
        X_test = sc.transform(X_test)
    return X_train, X_test, y_train, y_test
