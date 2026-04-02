from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.predict import predict_digit

app = FastAPI(title="Digits MLOps Mini API")


class DigitRequest(BaseModel):
    features: list[float] = Field(..., min_length=64, max_length=64)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_api(data: DigitRequest):
    try:
        pred = predict_digit(data.features)
        return {"prediction": pred}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))