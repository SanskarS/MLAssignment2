import base64
import io
import json
import os
from datetime import datetime

import joblib
import sklearn

SAVED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'saved_models')


def model_path(identifier, variant=None):
    os.makedirs(SAVED_DIR, exist_ok=True)
    stem = identifier if variant is None else f"{identifier}_{variant}"
    return os.path.join(SAVED_DIR, f"{stem}.json")


def save_model(path, model, scaler, metadata):
    buf = io.BytesIO()
    joblib.dump((scaler, model), buf)
    blob = base64.b64encode(buf.getvalue()).decode('ascii')
    payload = {
        **metadata,
        'sklearn_version': sklearn.__version__,
        'saved_at': datetime.now().isoformat(timespec='seconds'),
        'blob': blob,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(payload, f)
    return payload


def load_model(path):
    with open(path) as f:
        payload = json.load(f)
    buf = io.BytesIO(base64.b64decode(payload['blob']))
    scaler, model = joblib.load(buf)
    return payload, scaler, model


def list_saved_models():
    if not os.path.isdir(SAVED_DIR):
        return {}
    files = {}
    for name in sorted(os.listdir(SAVED_DIR)):
        if name.endswith('.json'):
            files[name] = os.path.join(SAVED_DIR, name)
    return files
