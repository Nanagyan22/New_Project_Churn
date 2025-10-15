import joblib
import json
from sklearn.pipeline import Pipeline

def load_model_and_features():
    model = joblib.load("model.pkl")
    pipeline = joblib.load("pipeline.pkl")
    with open("feature_names.json") as f:
        feature_names = json.load(f)
    return model, pipeline, feature_names