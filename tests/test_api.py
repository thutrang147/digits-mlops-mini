from fastapi.testclient import TestClient
from sklearn.datasets import load_digits

from app.main import app
from src.train import train_and_save

client = TestClient(app)


def setup_module():
    train_and_save()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict():
    digits = load_digits()
    sample = digits.data[0].tolist()

    response = client.post("/predict", json={"features": sample})
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], int)