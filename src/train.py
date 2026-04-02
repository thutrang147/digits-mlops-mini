from pathlib import Path
import joblib
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "model.pkl"


def train_and_save():
    digits = load_digits()
    X, y = digits.data, digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=3000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    return {
        "model_path": str(MODEL_PATH),
        "accuracy": float(acc),
    }


if __name__ == "__main__":
    result = train_and_save()
    print(result)