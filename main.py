import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay
from ucimlrepo import fetch_ucirepo

from models import RUNNERS

st.title('ML Assignment 2')

st.write('Classifier Models')

models = {
    '1. Logistic Regression': 'logistic_regression',
    '2. Decision Tree Classifier': 'decision_tree',
    '3. K-Nearest Neighbor Classifier': 'knn',
    '4. Naive Bayes Classifier - Gaussian or Multinomial': 'naive_bayes',
    '5. Ensemble Model - Random Forest': 'random_forest',
}

option = st.selectbox(label="Select Model", options=list(models.keys()))
identifier = models[option]


@st.cache_data
def load_dataset():
    fetched_dataset = fetch_ucirepo(id=936)
    return fetched_dataset.data.features, fetched_dataset.data.targets


X, y = load_dataset()


def run_selected(identifier, **params):
    result = RUNNERS[identifier](X, y, **params)
    st.metric(label="Test Accuracy", value=f"{result['accuracy']:.4f}")
    st.code(result['report'])
    disp = ConfusionMatrixDisplay(confusion_matrix=result['confusion'])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap='Blues')
    st.pyplot(fig)
    plt.close(fig)


nb_variant = None
if identifier == 'naive_bayes':
    nb_variant = st.selectbox(
        label="Naive Bayes Variant",
        options=['gaussian', 'multinomial'],
        format_func=lambda v: v.capitalize(),
    )

if st.button("Train & Predict"):
    params = {'variant': nb_variant} if nb_variant else {}
    run_selected(identifier, **params)
