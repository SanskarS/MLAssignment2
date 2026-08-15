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


def display_saved_results(payload):
    st.subheader("Saved Evaluation Metrics")
    metrics = payload.get('metrics', {})
    metric_labels = {
        'accuracy': 'Accuracy',
        'auc': 'AUC Score',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1': 'F1 Score',
        'mcc': 'MCC Score',
    }
    cols = st.columns(3)
    for i, (key, label) in enumerate(metric_labels.items()):
        cols[i % 3].metric(label=label, value=f"{metrics.get(key, 0):.4f}")

    st.subheader("Classification Report")
    report = payload.get('report')
    if report:
        st.code(report)
    else:
        st.info("Saved before the report was persisted — retrain the model to include it.")

    st.subheader("Confusion Matrix")
    confusion = payload.get('confusion')
    if confusion:
        disp = ConfusionMatrixDisplay(confusion_matrix=np.array(confusion))
        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(ax=ax, cmap='Blues')
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Saved before the confusion matrix was persisted — retrain the model to include it.")


def build_predictions(df, payload, scaler, model):
    feats = payload['feature_columns']
    X_new = df[feats]
    if scaler is not None:
        X_new = scaler.transform(X_new)
    pred = model.predict(X_new)
    out = df.copy()
    values = [payload['classes'][int(p)] for p in pred]
    # out['Prediction']
    out.insert(1, 'Prediction', values)
    out = out.iloc[:, :2]
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
                'report': result['report'],
                'confusion': result['confusion'].tolist(),
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

    expected = f"{identifier}{'_' + nb_variant if nb_variant else ''}.json"
    if expected in saved:
        payload, _, _ = load_model(saved[expected])
        st.caption(
            f"Showing saved evaluation for `{expected}` — click **Train & Save Model** to retrain and update."
        )
        display_saved_results(payload)
    elif saved:
        st.info(f"No saved model for `{expected}` yet. Click **Train & Save Model** to train and save it.")

with (tab_predict):
    saved = list_saved_models()
    if not saved:
        st.info("No saved models found. Go to **Train & Save** and train a model first.")
    else:
        files = list(saved)
        expected = os.path.basename(model_path(identifier, nb_variant))
        if expected in files:
            index = files.index(expected)
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

            display_saved_results(payload)

            uploaded = st.file_uploader("Upload test data CSV", type='csv')
            if uploaded is not None:
                df = pd.read_csv(uploaded)
                missing = [c for c in payload['feature_columns'] if c not in df.columns]
                if missing:
                    st.error(f"Missing required columns: {missing}")
                else:
                    out = build_predictions(df, payload, scaler, model)
                    st.dataframe(out)
                    st.download_button(
                        label="Download predictions",
                        data=out.to_csv(index=False).encode('utf-8'),
                        file_name='predictions.csv',
                        mime='text/csv',
                    )
        else:
            st.info("This model not trained yet. Train & Save this model first.")

