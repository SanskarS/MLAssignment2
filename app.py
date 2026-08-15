import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo

from models import RUNNERS
from models.persistence import list_saved_models, load_model, model_path, save_model

st.title('ML Assignment 2')

st.write('Classifier Models — Dry Bean (UCI id=602)')

models = {
    '1. Logistic Regression': 'logistic_regression',
    '2. Decision Tree Classifier': 'decision_tree',
    '3. K-Nearest Neighbor Classifier': 'knn',
    '4. Naive Bayes Classifier - Gaussian or Multinomial': 'naive_bayes',
    '5. Ensemble Model - Random Forest': 'random_forest',
}

option = st.selectbox(label="Select Model", options=list(models.keys()))
identifier = models[option]

nb_variant = None
if identifier == 'naive_bayes':
    nb_variant = st.selectbox(
        label="Naive Bayes Variant",
        options=['gaussian', 'multinomial'],
        format_func=lambda v: v.capitalize(),
    )


@st.cache_data
def load_training_dataset():
    fetched_dataset = fetch_ucirepo(id=602)
    X = fetched_dataset.data.features
    le = LabelEncoder()
    y = le.fit_transform(np.ravel(fetched_dataset.data.targets))
    return X, y, [str(c) for c in le.classes_]


X, y, CLASSES = load_training_dataset()


def display_results(result):
    metrics = {
        'Accuracy': result['accuracy'],
        'AUC Score': result['auc'],
        'Precision': result['precision'],
        'Recall': result['recall'],
        'F1 Score': result['f1'],
        'MCC Score': result['mcc'],
    }
    cols = st.columns(3)
    for i, (label, value) in enumerate(metrics.items()):
        cols[i % 3].metric(label=label, value=f"{value:.4f}")
    st.code(result['report'])
    disp = ConfusionMatrixDisplay(confusion_matrix=result['confusion'])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap='Blues')
    st.pyplot(fig)
    plt.close(fig)


def build_predictions(df, payload, scaler, model):
    feats = payload['feature_columns']
    X_new = df[feats]
    if scaler is not None:
        X_new = scaler.transform(X_new)
    pred = model.predict(X_new)
    out = df.copy()
    out['Prediction'] = [payload['classes'][int(p)] for p in pred]
    return out


tab_train, tab_predict = st.tabs(["Train & Save", "Load & Predict"])

with tab_train:
    if st.button("Train & Save Model"):
        params = {'variant': nb_variant} if nb_variant else {}
        result = RUNNERS[identifier](X, y, **params)
        display_results(result)
        path = model_path(identifier, nb_variant)
        save_model(
            path,
            result['model'],
            result['scaler'],
            {
                'model_id': identifier,
                'variant': nb_variant,
                'metrics': {k: result[k] for k in ('accuracy', 'auc', 'precision', 'recall', 'f1', 'mcc')},
                'classes': CLASSES,
                'feature_columns': list(X.columns),
            },
        )
        st.success(f"Model saved to `{path}`")

    saved = list_saved_models()
    if saved:
        st.write("**Saved models:**")
        for name in saved:
            st.write(f"- `{name}`")
    else:
        st.info("No saved models yet. Train & Save a model first.")

with tab_predict:
    saved = list_saved_models()
    if not saved:
        st.info("No saved models found. Go to **Train & Save** and train a model first.")
    else:
        files = list(saved)
        expected = os.path.basename(model_path(identifier, nb_variant))
        index = files.index(expected) if expected in files else 0
        choice = st.selectbox("Saved model file", options=files, index=index, disabled=True)
        payload, scaler, model = load_model(saved[choice])

        if payload.get('sklearn_version') != sklearn.__version__:
            st.warning(
                f"Model was saved with sklearn {payload['sklearn_version']} "
                f"but you are running {sklearn.__version__}. Predictions may be affected."
            )
        st.caption(
            f"Saved: {payload['saved_at']} | model: {payload['model_id']}"
            f"{' (' + payload['variant'] + ')' if payload.get('variant') else ''}"
        )

        uploaded = st.file_uploader("Upload test data CSV", type='csv')
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            missing = [c for c in payload['feature_columns'] if c not in df.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                out = build_predictions(df, payload, scaler, model)
                st.dataframe(out)
