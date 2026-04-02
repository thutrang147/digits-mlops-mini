from pathlib import Path
import joblib

MODEL_PATH = Path("artifacts/model.pkl")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Please run `python src/train.py` first."
        )
    return joblib.load(MODEL_PATH)


def predict_digit(features):
    if len(features) != 64:
        raise ValueError("Input must contain exactly 64 features.")
    model = load_model()
    pred = model.predict([features])[0]
    return int(pred)