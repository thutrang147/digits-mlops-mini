from pathlib import Path
from src.train import train_and_save


def test_train_and_save():
    result = train_and_save()
    assert Path(result["model_path"]).exists()
    assert result["accuracy"] > 0.8